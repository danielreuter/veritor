"""Bit-exact validation of the Python tensor-core model against the GPU.

For each operand type, runs >= 100,000 random tiles (one `mma.sync` each) plus
targeted edge-case families through the hardware, evaluates the same
(acc, a[k], b[k]) triples with ``veritor.core.silicon.tc_dot`` and, where the
Hawkeye C++ simulator can express the case (zero accumulator), with the
Hawkeye simulator too.  Writes a results JSON and a small golden-vector JSON.

Usage: python validate_tiles.py bf16|e4m3 --pipeline NAME --out results/... --golden golden/...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import tcs_gpu as G

sys.path.insert(0, G.HERE)
from veritor_core import silicon as S

FINITE_MAX_F32 = 0x7F7FFFFF


def random_finite_fp32_bits(rng, n, exp_lo=1, exp_hi=254):
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, size=n).astype(np.uint32)
    s = rng.integers(0, 2, size=n).astype(np.uint32)
    return (s << 31) | (e << 23) | m


def random_finite_operand_bits(dtype, rng, shape, exp_lo=None, exp_hi=None):
    n = int(np.prod(shape))
    if dtype == "bf16":
        lo, hi = (1 if exp_lo is None else exp_lo), (254 if exp_hi is None else exp_hi)
        e = rng.integers(lo, hi + 1, size=n).astype(np.uint16)
        m = rng.integers(0, 128, size=n).astype(np.uint16)
        s = rng.integers(0, 2, size=n).astype(np.uint16)
        return ((s << 15) | (e << 7) | m).reshape(shape)
    lo, hi = (1 if exp_lo is None else exp_lo), (15 if exp_hi is None else exp_hi)
    e = rng.integers(lo, hi + 1, size=n).astype(np.uint8)
    m = rng.integers(0, 8, size=n).astype(np.uint8)
    s = rng.integers(0, 2, size=n).astype(np.uint8)
    w = (s << 7) | (e << 3) | m
    w[(w & 0x7F) == 0x7F] &= 0xFE  # avoid NaN encodings
    return w.reshape(shape)


def families(dtype, rng, n_random):
    """Yield (name, A_bits (n,16,k), Bt_bits (n,8,k), C_bits (n,16,8))."""

    k = G.K_OF[dtype]
    cast = lambda x: G.f32_to_operand(dtype, x.astype(np.float32))
    zeros_acc = lambda n: np.zeros((n, 16, 8), np.uint32)

    # 1. Hawkeye-style: standard normal operands, zero accumulator.
    n = n_random
    yield (
        "randn_zero_acc",
        cast(rng.standard_normal((n, 16, k))),
        cast(rng.standard_normal((n, 8, k))),
        zeros_acc(n),
    )

    # 2. standard normal operands, random nonzero accumulators over a wide range.
    n = n_random // 4
    acc = rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-20, 21, (n, 16, 8)))
    yield (
        "randn_random_acc",
        cast(rng.standard_normal((n, 16, k))),
        cast(rng.standard_normal((n, 8, k))),
        G.f32_bits(acc.astype(np.float32)),
    )

    # 3. uniformly random finite encodings for everything (all exponents).
    n = n_random // 4
    yield (
        "uniform_bits",
        random_finite_operand_bits(dtype, rng, (n, 16, k)),
        random_finite_operand_bits(dtype, rng, (n, 8, k)),
        random_finite_fp32_bits(rng, n * 128).reshape(n, 16, 8),
    )

    # 4. subnormal-heavy operands, tiny/subnormal accumulators.
    n = n_random // 8
    if dtype == "bf16":
        a = random_finite_operand_bits(dtype, rng, (n, 16, k), 0, 3)
        b = random_finite_operand_bits(dtype, rng, (n, 8, k), 0, 3)
    else:
        a = random_finite_operand_bits(dtype, rng, (n, 16, k), 0, 1)
        b = random_finite_operand_bits(dtype, rng, (n, 8, k), 0, 1)
    acc = random_finite_fp32_bits(rng, n * 128, 0, 3).reshape(n, 16, 8)
    yield "subnormal", a, b, acc

    # 5. exact cancellation: pairs of equal-magnitude opposite products, small residuals, random acc.
    n = n_random // 8
    a = rng.standard_normal((n, 16, k)) * 2.0**8
    b = rng.standard_normal((n, 8, k)) * 2.0**8
    a[:, :, 1::2] = a[:, :, 0::2]
    b[:, :, 1::2] = -b[:, :, 0::2]
    # perturb a few residual slots per row to small magnitudes
    small = rng.standard_normal((n, 16, k)) * 2.0**-6
    mask = rng.random((n, 16, k)) < 0.15
    a = np.where(mask, small, a)
    acc = rng.standard_normal((n, 16, 8)) * 2.0**-4
    yield "cancellation", cast(a), cast(b), G.f32_bits(acc.astype(np.float32))

    # 6. mixed magnitudes: per-element random exponents over the whole operand range.
    n = n_random // 8
    yield (
        "mixed_magnitude",
        random_finite_operand_bits(dtype, rng, (n, 16, k)),
        random_finite_operand_bits(dtype, rng, (n, 8, k)),
        G.f32_bits(
            (
                rng.standard_normal((n, 16, 8))
                * np.exp2(rng.integers(-60, 61, (n, 16, 8)))
            ).astype(np.float32)
        ),
    )

    # 7. near FP32 overflow: accumulators near max, large operands.
    n = n_random // 16
    acc = random_finite_fp32_bits(rng, n * 128, 250, 254).reshape(n, 16, 8)
    if dtype == "bf16":
        a = random_finite_operand_bits(dtype, rng, (n, 16, k), 245, 254)
        b = random_finite_operand_bits(dtype, rng, (n, 8, k), 245, 254)
    else:
        a = random_finite_operand_bits(dtype, rng, (n, 16, k), 12, 15)
        b = random_finite_operand_bits(dtype, rng, (n, 8, k), 12, 15)
    yield "near_overflow", a, b, acc

    # 8. zero handling: -0 accumulators, all-zero products, cancellations to zero.
    n = 256
    a = np.zeros((n, 16, k), np.float32)
    b = np.zeros((n, 8, k), np.float32)
    a[n // 2 :, :, 0] = 1.0
    a[n // 2 :, :, 1] = 1.0
    b[n // 2 :, :, 0] = 3.0
    b[n // 2 :, :, 1] = -3.0
    acc = np.where(
        rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)
    ).astype(np.uint32)
    yield "signed_zero", cast(a), cast(b), acc

    # 9. tiny accumulators with zero products (exponent floor probe).
    n = 512
    acc = random_finite_fp32_bits(rng, n * 128, 0, 30).reshape(n, 16, 8)
    yield (
        "tiny_acc_zero_products",
        np.zeros((n, 16, k), G.WORD_OF[dtype]),
        np.zeros((n, 8, k), G.WORD_OF[dtype]),
        acc,
    )


def records_of(A, Bt, C, D):
    """Flatten tiles into per-output-element (acc, a[k], b[k], d) records (numpy)."""

    n, _, k = A.shape
    a = np.broadcast_to(A[:, :, None, :], (n, 16, 8, k)).reshape(-1, k)
    b = np.broadcast_to(Bt[:, None, :, :], (n, 16, 8, k)).reshape(-1, k)
    return C.reshape(-1), a, b, D.reshape(-1)


_PIPE = None


def _init(pipeline_name):
    global _PIPE
    _PIPE = S.PIPELINES[pipeline_name]


def _eval_chunk(args):
    acc, a, b = args
    out = np.zeros(len(acc), np.uint32)
    err = np.zeros(len(acc), np.bool_)
    for i in range(len(acc)):
        try:
            out[i] = S.tc_dot(_PIPE, int(acc[i]), a[i].tolist(), b[i].tolist())
        except Exception:  # noqa: BLE001 - domain errors are counted, not raised
            err[i] = True
    return out, err


def model_eval(pipeline_name, acc, a, b, procs):
    n = len(acc)
    chunks = max(1, min(n // 2000 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    with mp.Pool(procs, initializer=_init, initargs=(pipeline_name,)) as pool:
        parts = pool.map(_eval_chunk, [(acc[i], a[i], b[i]) for i in idx])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def hawkeye_eval(dtype, A, Bt, sim):
    """Hawkeye C++ simulator on zero-accumulator tiles: D = A @ B per tile."""

    import torch

    n = A.shape[0]
    out = np.zeros((n, 16, 8), np.uint32)
    for t in range(n):
        if dtype == "bf16":
            a = torch.from_numpy(A[t].astype(np.int16)).view(torch.bfloat16)
            b = (
                torch.from_numpy(Bt[t].astype(np.int16)).view(torch.bfloat16).t()
            )  # column-major [k, 8]
            d = sim.matmul(a, b)
        else:
            a = torch.from_numpy(A[t]).view(torch.float8_e4m3fn)
            b = torch.from_numpy(Bt[t]).view(torch.float8_e4m3fn)
            d = sim.matmul(a, b)
        out[t] = d.contiguous().view(torch.int32).numpy().astype(np.uint32)
    return out


def hexrec(acc, a, b, d, bits):
    width = bits // 4
    return {
        "acc": f"{int(acc):08x}",
        "a": "".join(f"{int(x):0{width}x}" for x in a),
        "b": "".join(f"{int(x):0{width}x}" for x in b),
        "d": f"{int(d):08x}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dtype", choices=["bf16", "e4m3"])
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--n-random", type=int, default=100_000)
    ap.add_argument(
        "--hawkeye",
        default=None,
        help="Hawkeye simulator class name (Ampere_simulator, Hopper_simulator, Hopper_fp8_simulator)",
    )
    ap.add_argument("--hawkeye-tiles", type=int, default=100_000)
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument(
        "--families",
        default=None,
        help="comma list of family names to run (default all)",
    )
    args = ap.parse_args()
    only = set(args.families.split(",")) if args.families else None

    rng = np.random.default_rng(20260902)
    pipe = S.PIPELINES[args.pipeline]
    bits = pipe.operand_bits
    results = {"dtype": args.dtype, "pipeline": args.pipeline, "families": {}}
    golden = {
        "pipeline": args.pipeline,
        "dtype": args.dtype,
        "k": pipe.k,
        "operand_bits": bits,
        "note": "acc/d are FP32 bit patterns (hex); a/b are k operand words (hex, k*bits/4 chars) as measured on the GPU named in results",
        "records": [],
    }
    sim = None
    if args.hawkeye:
        sys.path.insert(0, "/workspace/hawkeye")
        import gpu_simulator_py

        sim = getattr(gpu_simulator_py, args.hawkeye)()

    t_all = time.time()
    for name, A, Bt, C in families(args.dtype, rng, args.n_random):
        if only is not None and name not in only:
            continue
        t0 = time.time()
        D = G.mma_tiles(args.dtype, A, Bt, C)
        t_gpu = time.time() - t0
        acc, a, b, d = records_of(A, Bt, C, D)
        t0 = time.time()
        model, err = model_eval(args.pipeline, acc, a, b, args.procs)
        t_model = time.time() - t0
        mism = (model != d) | err
        n_mism = int(mism.sum())
        fam = {
            "tiles": int(A.shape[0]),
            "elements": len(acc),
            "mismatches_vs_model": n_mism,
            "model_domain_errors": int(err.sum()),
            "gpu_seconds": t_gpu,
            "model_seconds": t_model,
            "first_mismatches": [],
        }
        for i in np.flatnonzero(mism)[:5]:
            r = hexrec(acc[i], a[i], b[i], d[i], bits)
            r["model"] = "error" if err[i] else f"{int(model[i]):08x}"
            fam["first_mismatches"].append(r)
        if sim is not None and name == "randn_zero_acc":
            nt = min(args.hawkeye_tiles, A.shape[0])
            t0 = time.time()
            H = hawkeye_eval(args.dtype, A[:nt], Bt[:nt], sim)
            fam["hawkeye"] = {
                "simulator": args.hawkeye,
                "tiles": nt,
                "elements": nt * 128,
                "mismatches_vs_gpu": int(np.sum(H != D[:nt])),
                "seconds": time.time() - t0,
            }
            hm = np.flatnonzero((H != D[:nt]).reshape(-1))[:3]
            fam["hawkeye"]["first_mismatches"] = [
                dict(
                    hexrec(acc[i], a[i], b[i], d[i], bits),
                    hawkeye=f"{int(H.reshape(-1)[i]):08x}",
                )
                for i in hm
            ]
        results["families"][name] = fam
        print(
            name,
            {k_: v for k_, v in fam.items() if k_ != "first_mismatches"},
            flush=True,
        )
        # golden sample
        pick = rng.choice(
            len(acc), size=min(args.golden_per_family, len(acc)), replace=False
        )
        for i in pick:
            golden["records"].append(
                dict(hexrec(acc[i], a[i], b[i], d[i], bits), family=name)
            )
    results["seconds"] = time.time() - t_all
    results["total_elements"] = int(
        sum(f["elements"] for f in results["families"].values())
    )
    results["total_mismatches_vs_model"] = int(
        sum(f["mismatches_vs_model"] for f in results["families"].values())
    )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    with open(args.golden, "w") as f:
        json.dump(golden, f, separators=(",", ":"))
    print(
        "TOTAL",
        results["total_elements"],
        "mismatches",
        results["total_mismatches_vs_model"],
        "in",
        round(results["seconds"], 1),
        "s",
    )


if __name__ == "__main__":
    main()
