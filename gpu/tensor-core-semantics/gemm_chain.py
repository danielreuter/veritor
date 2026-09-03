"""Fixed-order GEMM chain: GPU kernel vs CPU references vs cuBLAS, at GPT-2 Small shapes.

Runs on the GPU host.  For each (dtype, M, K, N):
  * random operands -> gemm_chain kernel -> compared bit-for-bit with the C++
    chain reference (all elements), the Python `tc_dot_chain` (random subset)
    and, for BF16, the Hawkeye Ampere simulator (whose k-loop *is* the chain);
  * kernel time vs torch.matmul (cuBLAS) / torch._scaled_mm (cuBLASLt FP8);
  * with --gpt2, the same on real GPT-2 Small weights and activations.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import tcs_gpu as G
import torch

sys.path.insert(0, G.HERE)
from veritor_core import silicon as S

_gemm = ctypes.CDLL(os.path.join(G.HERE, "libgemm_chain.so"))
_ref = ctypes.CDLL(os.path.join(G.HERE, "libref_chain.so"))
for name in ("gemm_chain_bf16", "gemm_chain_e4m3"):
    fn = getattr(_gemm, name)
    fn.restype = ctypes.c_int
    fn.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_float)]
    )
for name in ("ref_chain_bf16", "ref_chain_e4m3"):
    fn = getattr(_ref, name)
    fn.restype = None
    fn.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_int] * 3
        + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    )

PIPE = {"bf16": S.ADA_BF16_M16N8K16, "e4m3": S.ADA_E4M3_M16N8K32}


def gpu_chain(dtype, A, Bt, C, bench_iters=0):
    M, K = A.shape
    N = Bt.shape[0]
    A = np.ascontiguousarray(A)
    Bt = np.ascontiguousarray(Bt)
    D = np.zeros((M, N), np.uint32)
    ms = ctypes.c_float(0)
    fn = _gemm.gemm_chain_bf16 if dtype == "bf16" else _gemm.gemm_chain_e4m3
    cptr = None if C is None else np.ascontiguousarray(C).ctypes.data
    rc = fn(
        A.ctypes.data,
        Bt.ctypes.data,
        cptr,
        D.ctypes.data,
        M,
        N,
        K,
        bench_iters,
        ctypes.byref(ms),
    )
    if rc != 0:
        raise RuntimeError(f"gemm_chain error {rc}")
    return D, ms.value


def cpu_chain(dtype, A, Bt, C):
    p = PIPE[dtype]
    M, K = A.shape
    N = Bt.shape[0]
    A = np.ascontiguousarray(A)
    Bt = np.ascontiguousarray(Bt)
    D = np.zeros((M, N), np.uint32)
    groups = (ctypes.c_int * len(p.groups))(*p.groups)
    fn = _ref.ref_chain_bf16 if dtype == "bf16" else _ref.ref_chain_e4m3
    cptr = None if C is None else np.ascontiguousarray(C).ctypes.data
    fn(
        A.ctypes.data,
        Bt.ctypes.data,
        cptr,
        D.ctypes.data,
        M,
        N,
        K,
        groups,
        len(p.groups),
        p.width,
        p.zero_exponent,
    )
    return D


def python_subset(dtype, A, Bt, C, D, rng, n=400):
    p = PIPE[dtype]
    M, N = D.shape
    idx = rng.integers(0, M * N, size=min(n, M * N))
    mism = 0
    for flat in idx:
        i, j = divmod(int(flat), N)
        acc = 0 if C is None else int(C[i, j])
        if S.tc_dot_chain(p, acc, A[i].tolist(), Bt[j].tolist()) != int(D[i, j]):
            mism += 1
    return {"checked": len(idx), "mismatches": mism}


def hawkeye_bf16(A, Bt):
    sys.path.insert(0, "/workspace/hawkeye")
    import gpu_simulator_py

    sim = gpu_simulator_py.Ampere_simulator()
    a = torch.from_numpy(A.astype(np.int16)).view(torch.bfloat16)
    b = (
        torch.from_numpy(np.ascontiguousarray(Bt).astype(np.int16))
        .view(torch.bfloat16)
        .t()
    )  # col-major [K, N]
    return sim.matmul(a, b).contiguous().view(torch.int32).numpy().astype(np.uint32)


def cublas_ms(dtype, A, Bt, iters=20):
    """cuBLAS time for the same product with the vendor path (bf16 out for bf16; fp32 out for fp8)."""

    dev = "cuda"
    if dtype == "bf16":
        a = torch.from_numpy(A.astype(np.int16)).view(torch.bfloat16).to(dev)
        b = (
            torch.from_numpy(np.ascontiguousarray(Bt).astype(np.int16))
            .view(torch.bfloat16)
            .to(dev)
            .t()
        )
        run = lambda: torch.matmul(a, b)
        label = "torch.matmul(bf16,bf16)->bf16"
    else:
        a = torch.from_numpy(A).view(torch.float8_e4m3fn).to(dev)
        b = (
            torch.from_numpy(np.ascontiguousarray(Bt))
            .view(torch.float8_e4m3fn)
            .to(dev)
            .t()
        )
        one = torch.ones((), device=dev, dtype=torch.float32)
        try:
            torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
            run = lambda: torch._scaled_mm(
                a, b, scale_a=one, scale_b=one, out_dtype=torch.float32
            )
            label = "torch._scaled_mm(e4m3,e4m3)->f32"
        except Exception as e:  # noqa: BLE001 - depends on the torch build
            return None, f"unavailable: {type(e).__name__}: {str(e)[:120]}"
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        run()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters, label


def run_case(dtype, M, K, N, A, Bt, C, rng, label, hawkeye_ok):
    D, ms = gpu_chain(dtype, A, Bt, C, bench_iters=20)
    t0 = time.time()
    R = cpu_chain(dtype, A, Bt, C)
    t_ref = time.time() - t0
    out = {
        "label": label,
        "dtype": dtype,
        "M": M,
        "K": K,
        "N": N,
        "elements": M * N,
        "mismatches_vs_cpp_chain": int(np.sum(D != R)),
        "cpp_ref_seconds": t_ref,
        "python_subset": python_subset(dtype, A, Bt, C, D, rng),
        "kernel_ms": ms,
        "kernel_tflops": 2.0 * M * N * K / (ms * 1e-3) / 1e12,
    }
    cms, clabel = cublas_ms(dtype, A, Bt)
    out["cublas_ms"] = cms
    out["cublas_label"] = clabel
    out["kernel_over_cublas_time"] = (ms / cms) if cms else None
    if (
        dtype == "bf16"
        and C is None
        and hawkeye_ok
        and M * N * K <= 1024 * 3072 * 768 * 2
    ):
        t0 = time.time()
        H = hawkeye_bf16(A, Bt)
        out["hawkeye_ampere"] = {
            "mismatches_vs_gpu": int(np.sum(D != H)),
            "seconds": time.time() - t0,
        }
    if out["mismatches_vs_cpp_chain"]:
        i, j = np.argwhere(D != R)[0]
        out["first_mismatch"] = {
            "i": int(i),
            "j": int(j),
            "gpu": f"{int(D[i, j]):08x}",
            "cpp": f"{int(R[i, j]):08x}",
        }
    print(json.dumps(out), flush=True)
    return out


SHAPES = [(768, 768), (768, 2304), (768, 3072), (3072, 768), (768, 50304)]


def random_operands(dtype, M, K, N, rng):
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((N, K)).astype(np.float32)
    if dtype == "e4m3":
        a *= 32.0
        b *= 32.0
    return G.f32_to_operand(dtype, a), G.f32_to_operand(dtype, b)


def e4m3_scaled(x: np.ndarray):
    """Per-tensor absmax scaling to the E4M3 range: q = round(x * 448/amax) (documented in the report)."""

    amax = float(np.max(np.abs(x))) or 1.0
    scale = 448.0 / amax
    return G.to_e4m3_bits((x * scale).astype(np.float32)), scale


def gpt2_cases(rng):
    """Real GPT-2 Small weights and activations for the five GEMM shapes."""

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with open("/workspace/hawkeye/README.md") as f:
        text = f.read() * 20
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        h0 = out.hidden_states[0][0]  # [1024, 768] embeddings (input to block 0)
        blk = model.transformer.h[0]
        x_attn = blk.ln_1(h0)  # input to c_attn
        x_mlp = blk.ln_2(h0)  # stand-in input to c_fc
        hidden_fc = blk.mlp.act(blk.mlp.c_fc(x_mlp))  # [1024, 3072] input to mlp c_proj
        h_last = model.transformer.ln_f(out.hidden_states[-1][0])  # input to lm_head
    W = {
        "c_attn": blk.attn.c_attn.weight.detach().T.contiguous(),  # [2304, 768] as Bt (N, K)
        "attn_c_proj": blk.attn.c_proj.weight.detach().T.contiguous(),  # [768, 768]
        "c_fc": blk.mlp.c_fc.weight.detach().T.contiguous(),  # [3072, 768]
        "mlp_c_proj": blk.mlp.c_proj.weight.detach().T.contiguous(),  # [768, 3072]
        "lm_head": model.lm_head.weight.detach().contiguous(),  # [50257, 768]
    }
    X = {
        "c_attn": x_attn,
        "attn_c_proj": x_attn,
        "c_fc": x_mlp,
        "mlp_c_proj": hidden_fc,
        "lm_head": h_last,
    }
    for name, weight in W.items():
        x = X[name].numpy().astype(np.float32)
        w = weight.numpy().astype(np.float32)
        if name == "lm_head":
            w = np.concatenate([w, np.zeros((50304 - 50257, 768), np.float32)])
        yield name, x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpt2", action="store_true")
    ap.add_argument("--no-hawkeye", action="store_true")
    ap.add_argument("--ms", default="1,32,1024")
    ap.add_argument("--skip-random", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(99)
    results = {"gpu": torch.cuda.get_device_name(0), "cases": []}
    Ms = [int(x) for x in args.ms.split(",")]
    for dtype in ("bf16", "e4m3"):
        for K, N in SHAPES:
            for M in Ms:
                if args.skip_random:
                    break
                A, Bt = random_operands(dtype, M, K, N, rng)
                C = None
                if (
                    M == 32
                ):  # one shape per dtype also with a nonzero initial accumulator
                    C = G.f32_bits((rng.standard_normal((M, N)) * 4).astype(np.float32))
                results["cases"].append(
                    run_case(
                        dtype,
                        M,
                        K,
                        N,
                        A,
                        Bt,
                        C,
                        rng,
                        f"random{'+acc' if C is not None else ''}",
                        not args.no_hawkeye,
                    )
                )
    if args.gpt2:
        for name, x, w in gpt2_cases(rng):
            for dtype in ("bf16", "e4m3"):
                if dtype == "bf16":
                    A, Bt, scales = G.to_bf16_bits(x), G.to_bf16_bits(w), None
                else:
                    (A, sa), (Bt, sb) = e4m3_scaled(x), e4m3_scaled(w)
                    scales = {"x": sa, "w": sb}
                for M in Ms:
                    res = run_case(
                        dtype,
                        M,
                        x.shape[1],
                        w.shape[0],
                        A[:M],
                        Bt,
                        None,
                        rng,
                        f"gpt2:{name}",
                        not args.no_hawkeye,
                    )
                    res["scales"] = scales
                    results["cases"].append(res)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
