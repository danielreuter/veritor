"""GPT-2 Small through the pinned kernels: GPU capture, CPU re-execution, match table, goldens, kernel cost.

Runs in a numpy-only interpreter (Python 3.12, ``PYTHONPATH=src``) next to
the built ``libgemm_chain.so`` / ``libref_chain.so`` / ``libpinned_ops.so``.

    python run_gpt2.py capture --hf /workspace/gpt2 --out /workspace/gpt2   # GPU forward -> capture_gpu.npz
    python run_gpt2.py cpu     --hf /workspace/gpt2 --out /workspace/gpt2   # CPU forward -> capture_cpu.npz, match.json
    python run_gpt2.py golden  --out /workspace/gpt2                        # GPU golden vectors for the elementwise gates
    python run_gpt2.py bench   --hf /workspace/gpt2 --out /workspace/gpt2   # our GEMM chain kernel times at the run's shapes
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import cast

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_ops

from veritor.constructors.gpt2 import GPT2Shape
from veritor.constructors.gpt2_reference import (
    GPT2Weights,
    LayerWeights,
    NumpyOps,
    as_bf16_words,
    forward,
)


def load_weights(hf_dir: str) -> GPT2Weights:
    """HF fp32 -> BF16 words, rounded to nearest even once (the model's single rounding)."""

    z = np.load(os.path.join(hf_dir, "hf_gpt2_fp32.npz"))
    shape = GPT2Shape.small()
    layers = []
    for i in range(shape.layers):
        fields = {
            name: as_bf16_words(z[f"layer{i}.{name}"])
            for name in LayerWeights.__dataclass_fields__
        }
        layers.append(LayerWeights(**fields))
    return GPT2Weights(
        shape,
        as_bf16_words(z["wte"]),
        as_bf16_words(z["wpe"]),
        tuple(layers),
        as_bf16_words(z["lnf_g"]),
        as_bf16_words(z["lnf_b"]),
    )


def meta(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "hf_meta.json")) as f:
        return json.load(f)


def gpu_identity() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,uuid,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        out = f"unavailable: {error}"
    return {
        "nvidia_smi": out,
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def save_capture(path: str, run, extra: dict) -> None:
    arrays = dict(run.capture)
    arrays["_tokens_out"] = np.array(run.tokens, dtype=np.uint16)
    arrays["_prompt"] = np.array(run.prompt, dtype=np.uint16)
    np.savez_compressed(path, **arrays)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(extra, f, indent=2)


def cmd_capture(args) -> None:
    weights = load_weights(args.hf)
    m = meta(args.hf)
    ops = cuda_ops.CudaOps()
    t0 = time.perf_counter()
    run = forward(weights, m["prompt"], m["max_new"], ops)
    seconds = time.perf_counter() - t0
    info = {
        "backend": "cuda",
        "prompt": m["prompt"],
        "max_new": m["max_new"],
        "tokens": list(run.tokens),
        "hf_fp32_greedy_tokens": m["hf_fp32_greedy_tokens"],
        "forward_seconds_wall": seconds,
        "kernel_calls": ops.ops.calls,
        "identity": gpu_identity(),
        "captured_tensors": len(run.capture),
        "captured_words": int(sum(a.size for a in run.capture.values())),
    }
    save_capture(os.path.join(args.out, "capture_gpu.npz"), run, info)
    print(json.dumps(info, indent=2))


def compare(a: dict, b: dict) -> dict:
    rows = {}
    total = mism = 0
    for name, x in a.items():
        y = b[name]
        xw = x.view(np.uint32) if x.dtype == np.float32 else x
        yw = y.view(np.uint32) if y.dtype == np.float32 else y
        bad = xw != yw
        n_bad = int(bad.sum())
        row: dict[str, object] = {"elements": int(x.size), "mismatches": n_bad}
        if n_bad:
            flat = int(np.flatnonzero(bad)[0])
            row["first"] = {
                "index": list(map(int, np.unravel_index(flat, x.shape))),
                "gpu": f"{int(xw.reshape(-1)[flat]):#x}",
                "cpu": f"{int(yw.reshape(-1)[flat]):#x}",
            }
        rows[name] = row
        total += x.size
        mism += n_bad
    return {"tensors": rows, "elements": int(total), "mismatches": int(mism)}


def cmd_cpu(args) -> None:
    weights = load_weights(args.hf)
    m = meta(args.hf)
    ops = NumpyOps(cuda_ops.CppGemm())
    t0 = time.perf_counter()
    run = forward(weights, m["prompt"], m["max_new"], ops)
    seconds = time.perf_counter() - t0
    info = {
        "backend": "numpy+ref_chain",
        "tokens": list(run.tokens),
        "forward_seconds_wall": seconds,
        "identity": gpu_identity(),
    }
    save_capture(os.path.join(args.out, "capture_cpu.npz"), run, info)
    gpu = np.load(os.path.join(args.out, "capture_gpu.npz"))
    gpu_capture = {k: gpu[k] for k in gpu.files if not k.startswith("_")}
    report = compare(gpu_capture, run.capture)
    report["tokens_gpu"] = [int(t) for t in gpu["_tokens_out"]]
    report["tokens_cpu"] = list(run.tokens)
    report["cpu_forward_seconds"] = seconds
    with open(os.path.join(args.out, "match.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}, indent=2))
    for name, row in report["tensors"].items():
        if row["mismatches"]:
            print("MISMATCH", name, row)


def _hex32(a: np.ndarray) -> list[str]:
    return [
        f"{int(w):08x}"
        for w in np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).reshape(-1)
    ]


def _hex16(a: np.ndarray) -> list[str]:
    return [
        f"{int(w):04x}" for w in np.ascontiguousarray(a, dtype=np.uint16).reshape(-1)
    ]


def cmd_golden(args) -> None:
    """A few hundred GPU records per pinned elementwise gate, over inputs that exercise every branch."""

    rng = np.random.default_rng(20260903)
    lib = cuda_ops.PinnedOpsLib()
    n = args.records

    def mixed(scale: float, n: int = n) -> np.ndarray:
        parts = [
            rng.normal(0, scale, n // 2).astype(np.float32),
            (rng.normal(0, 1, n // 4) * np.exp2(rng.uniform(-30, 30, n // 4))).astype(
                np.float32
            ),
            rng.uniform(-scale * 4, scale * 4, n - n // 2 - n // 4).astype(np.float32),
        ]
        x = np.concatenate(parts)
        specials = np.array(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                88.0,
                88.5,
                -86.5,
                -87.0,
                9.0,
                -9.0,
                1e-40,
                -1e-40,
                np.inf,
                -np.inf,
            ],
            dtype=np.float32,
        )
        k = min(len(specials), n)
        x[:k] = specials[:k]
        rng.shuffle(x)
        return x

    records: dict[str, dict[str, object]] = {}
    xs = {
        "f32_exp": mixed(20.0),
        "f32_tanh": mixed(4.0),
        "gelu_tanh": mixed(4.0),
        "ln_rstd": np.abs(mixed(2.0)),
    }
    ops = {
        "f32_exp": cuda_ops.U_EXP,
        "f32_tanh": cuda_ops.U_TANH,
        "gelu_tanh": cuda_ops.U_GELU,
        "ln_rstd": cuda_ops.U_RSTD,
    }
    for name, x in xs.items():
        y = lib.unary(ops[name], x)
        records[name] = {
            "arg_widths": [32],
            "width": 32,
            "x": _hex32(x),
            "y": _hex32(y),
        }
    a, b = mixed(3.0), mixed(3.0)
    for name, op in (
        ("f32_add", cuda_ops.B_ADD),
        ("f32_sub", cuda_ops.B_SUB),
        ("f32_mul", cuda_ops.B_MUL),
        ("f32_div", cuda_ops.B_DIV),
        ("f32_max", cuda_ops.B_MAX),
    ):
        y = lib.binary(op, a, b)
        records[name] = {
            "arg_widths": [32, 32],
            "width": 32,
            "a": _hex32(a),
            "b": _hex32(b),
            "y": _hex32(y),
        }
    x = mixed(3.0)
    records["f32_to_bf16"] = {
        "arg_widths": [32],
        "width": 16,
        "x": _hex32(x),
        "y": _hex16(lib.round(x)),
    }
    la, lb = mixed(3.0), mixed(3.0)
    lb[: n // 8] = la[: n // 8]  # ties keep the earlier index
    ia, ib = (
        rng.integers(0, 50257, n).astype(np.uint16),
        rng.integers(0, 50257, n).astype(np.uint16),
    )
    records["argmax_select"] = {
        "arg_widths": [32, 32, 16, 16],
        "width": 16,
        "la": _hex32(la),
        "lb": _hex32(lb),
        "ia": _hex16(ia),
        "ib": _hex16(ib),
        "y": _hex16(lib.select(la, lb, ia, ib)),
    }
    t = rng.integers(0, 50257, n).astype(np.uint16)
    j = t.copy()
    j[n // 3 :] = rng.integers(0, 50257, n - n // 3).astype(np.uint16)
    records["token_eq"] = {
        "arg_widths": [16, 16],
        "width": 16,
        "t": _hex16(t),
        "j": _hex16(j),
        "y": _hex16(lib.token_eq(t, j)),
    }
    # the reductions: one row each of the lengths GPT-2 uses (d = 768, c = 1..24) as ln_stats / row reductions
    rows = []
    for c in list(range(1, 25)) + [64, 768]:
        u = mixed(3.0, c)
        rows.append(
            {
                "c": c,
                "u": _hex32(u),
                "tree_sum": _hex32(lib.row_reduce(0, u[None, :])),
                "tree_max": _hex32(lib.row_reduce(1, u[None, :])),
            }
        )
    x = mixed(1.0, 768)[None, :]
    mean, center, rstd = lib.ln_stats(x, 768.0)
    golden = {
        "pipeline": "ada_bf16_m16n8k16",
        "gate_set": "veritor.pinned.ada_bf16_m16n8k16",
        "compiler_flags": "nvcc -O3 -arch=sm_89 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false",
        "identity": gpu_identity(),
        "gates": records,
        "reductions": rows,
        "ln_stats": {
            "x": _hex32(x),
            "n": _hex32(np.array([768.0], dtype=np.float32)),
            "mean": _hex32(mean),
            "center": _hex32(center),
            "rstd": _hex32(rstd),
        },
    }
    path = os.path.join(args.out, "ada_pinned_f32.json")
    with open(path, "w") as f:
        json.dump(golden, f, separators=(",", ":"))
    print(
        "wrote",
        path,
        os.path.getsize(path),
        "bytes",
        {k: len(cast(list, v["y"])) for k, v in records.items()},
    )


def cmd_bench(args) -> None:
    m = meta(args.hf)
    chain = cuda_ops.GemmChainLib()
    rng = np.random.default_rng(0)
    results = {}
    m_prefill = len(m["prompt"])
    shapes = []
    for mm in sorted({m_prefill, 1}):
        shapes += [(mm, 768, 768), (mm, 768, 3072), (mm, 3072, 768), (mm, 50272, 768)]
    shapes.append((1, 768, 50257))
    for mm, k, n in shapes:
        a = as_bf16_words(rng.normal(0, 1, (mm, k)).astype(np.float32))
        bt = as_bf16_words(rng.normal(0, 1, (n, k)).astype(np.float32))
        chain.kernel_ms = 0.0
        chain.gemm(a, bt, None, bench_iters=50)
        results[f"{mm}x{k}x{n}"] = {
            "chain_ms": chain.kernel_ms,
            "cublas_ms": m["cublas_bf16_ms"].get(f"{mm}x{k}x{n}"),
        }
    with open(os.path.join(args.out, "bench.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["capture", "cpu", "golden", "bench"])
    parser.add_argument("--hf", default="/workspace/gpt2")
    parser.add_argument("--out", default="/workspace/gpt2")
    parser.add_argument("--records", type=int, default=400)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"capture": cmd_capture, "cpu": cmd_cpu, "golden": cmd_golden, "bench": cmd_bench}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
