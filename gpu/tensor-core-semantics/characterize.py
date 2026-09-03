"""Characterise the tensor-core accumulation pipeline of the local GPU.

Reproduces the Hawkeye methodology (experiments/GPU_reproduction*.py in
https://github.com/badasherez/gpu-simulator) with batched `mma.sync` probes:

1. fragment-layout check with exactly representable small integers;
2. computationally neutral subgroup search over contiguous k-ranges, with
   and without the accumulator, which exposes the grouping and chain order;
3. internal precision (adder width) from a cancellation + tiny-residual probe;
4. exponent floor and width confirmed by fitting the Python model
   (``veritor.core.silicon``) to random tiny-valued tiles.

Usage:  python characterize.py bf16|e4m3 --out results/characterize_<dtype>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import tcs_gpu as G

sys.path.insert(0, G.HERE)
from veritor_core import (
    silicon as S,
)


def layout_test(dtype: str, rng: np.random.Generator, n: int = 512) -> dict:
    """Small integers: the FP32 result is exact, so any layout error shows up."""

    k = G.K_OF[dtype]
    a = rng.integers(-3, 4, size=(n, 16, k)).astype(np.float32)
    b = rng.integers(-3, 4, size=(n, 8, k)).astype(np.float32)
    c = rng.integers(-8, 9, size=(n, 16, 8)).astype(np.float32)
    want = c + np.einsum("nik,njk->nij", a, b)
    got = G.bits_f32(
        G.mma_tiles(
            dtype, G.f32_to_operand(dtype, a), G.f32_to_operand(dtype, b), G.f32_bits(c)
        )
    )
    mism = int(np.sum(got != want))
    return {"tiles": n, "elements": n * 128, "mismatches": mism}


def _model_of(dtype: str, groups, width, zero_exponent) -> S.Pipeline:
    return S.Pipeline(
        name="probe",
        arch="probe",
        dtype=dtype,
        operand_bits=16 if dtype == "bf16" else 8,
        groups=tuple(groups),
        width=width,
        zero_exponent=zero_exponent,
        validated="probe",
    )


def neutral_subgroups(dtype: str) -> dict:
    """Hawkeye's neutral-subgroup test, batched: all contiguous k-ranges, +-acc, acc small / zero."""

    k = G.K_OF[dtype]
    if dtype == "bf16":
        v_large, v_small = 2.0**20, 2.0**-20
    else:
        v_large, v_small = (
            2.0**14,
            2.0**-16,
        )  # sqrt: 2^7 (normal), 2^-8 (subnormal) in E4M3
    sl, ss = math.sqrt(v_large), math.sqrt(v_small)

    cases = []  # (label, acc_mode, S) ; acc_mode: "small" or "zero"
    for acc_mode in ("small", "zero"):
        for start in range(k):
            for end in range(start + 1, k + 1):
                prods = list(range(start, end))
                if len(prods) >= 2:
                    cases.append((acc_mode, prods))
                cases.append((acc_mode, [-1] + prods))
    n = len(cases)
    A = np.zeros((2 * n, 16, k), np.float32)
    B = np.zeros((2 * n, 8, k), np.float32)
    C = np.zeros((2 * n, 16, 8), np.float32)
    for idx, (acc_mode, Sset) in enumerate(cases):
        Sl = set(Sset)
        acc_in = -1 in Sl
        # cancellation scenario (row 0, col 0)
        for kk in range(k):
            v = sl if kk in Sl else ss
            A[idx, 0, kk] = v
            B[idx, 0, kk] = v
        acc_small = 0.0 if acc_mode == "zero" else v_small
        C[idx, 0, 0] = v_large if acc_in else acc_small
        Slist = sorted(Sset)
        m = len(Slist)
        neg = (m + 1) // 2
        for i in range(neg):
            j = Slist[i]
            if j == -1:
                C[idx, 0, 0] *= -1.0
            else:
                A[idx, 0, j] = -sl
        if m % 2 == 1:
            j = Slist[neg]
            if j == -1:
                C[idx, 0, 0] *= 2.0
            else:
                A[idx, 0, j] = 2.0 * sl
        # zeroed scenario
        for kk in range(k):
            v = 0.0 if kk in Sl else ss
            A[n + idx, 0, kk] = v
            B[n + idx, 0, kk] = v
        C[n + idx, 0, 0] = 0.0 if acc_in else acc_small
    D = G.mma_tiles(
        dtype, G.f32_to_operand(dtype, A), G.f32_to_operand(dtype, B), G.f32_bits(C)
    )
    r_cancel, r_zero = D[:n, 0, 0], D[n:, 0, 0]
    neutral = {"small": [], "zero": []}
    for idx, (acc_mode, Sset) in enumerate(cases):
        if r_cancel[idx] == r_zero[idx]:
            neutral[acc_mode].append(Sset)
    out = {"k": k, "cases": n}
    for mode, sets in neutral.items():
        out[f"neutral_acc_{mode}"] = [
            [("acc" if x == -1 else x) for x in s] for s in sets
        ]
        out[f"neutral_acc_{mode}_count"] = len(sets)
    return out


def internal_precision(dtype: str) -> dict:
    """D = 1*1 + 1*(-1) + 2^-c, largest exact c; also the same with the residual placed in each k slot."""

    k = G.K_OF[dtype]
    min_exp = -133 if dtype == "bf16" else -9  # smallest power of two representable
    results = {}
    for slot in range(2, k):
        cs = list(range(1, 60))
        A = np.zeros((len(cs), 16, k), np.float32)
        B = np.zeros((len(cs), 8, k), np.float32)
        C = np.zeros((len(cs), 16, 8), np.float32)
        ok = []
        for i, c in enumerate(cs):
            up, fl = math.ceil(-c / 2), math.floor(-c / 2)
            if fl < min_exp:
                ok.append(None)
                continue
            A[i, 0, 0], B[i, 0, 0] = 1.0, 1.0
            A[i, 0, 1], B[i, 0, 1] = 1.0, -1.0
            A[i, 0, slot], B[i, 0, slot] = 2.0**up, 2.0**fl
            ok.append(c)
        D = G.bits_f32(
            G.mma_tiles(
                dtype,
                G.f32_to_operand(dtype, A),
                G.f32_to_operand(dtype, B),
                G.f32_bits(C),
            )
        )
        prec = None
        for i, c in enumerate(cs):
            if ok[i] is None:
                break
            if D[i, 0, 0] != 2.0**-c:
                prec = c - 1
                break
        results[slot] = prec
    return {"precision_bits_by_residual_slot": results}


def fit_parameters(
    dtype: str, groups, rng: np.random.Generator, ntiles: int = 400
) -> dict:
    """Fit (width, zero_exponent) of the model on tiny-valued and mixed random tiles."""

    k = G.K_OF[dtype]
    # tiny-valued tiles: exponents near the floor, many subnormal products
    if dtype == "bf16":
        exps = rng.integers(-133, -100, size=(ntiles, 16, k))
        exps_b = rng.integers(-133, -100, size=(ntiles, 8, k))
        exps = np.where(
            rng.random(exps.shape) < 0.5, exps, rng.integers(-126, 5, size=exps.shape)
        )
    else:
        exps = rng.integers(-9, -3, size=(ntiles, 16, k))
        exps_b = rng.integers(-9, -3, size=(ntiles, 8, k))
        exps = np.where(
            rng.random(exps.shape) < 0.5, exps, rng.integers(-9, 4, size=exps.shape)
        )
    mant = rng.random((ntiles, 16, k)) + 1.0
    mant_b = rng.random((ntiles, 8, k)) + 1.0
    sgn = rng.choice([-1.0, 1.0], size=(ntiles, 16, k))
    sgn_b = rng.choice([-1.0, 1.0], size=(ntiles, 8, k))
    A = G.f32_to_operand(
        dtype, (sgn * mant * np.exp2(exps.astype(np.float64))).astype(np.float32)
    )
    B = G.f32_to_operand(
        dtype, (sgn_b * mant_b * np.exp2(exps_b.astype(np.float64))).astype(np.float32)
    )
    accs = rng.random((ntiles, 16, 8)).astype(np.float32) * 2.0**-120
    accs[rng.random(accs.shape) < 0.5] = 0.0
    C = G.f32_bits(accs)
    D = G.mma_tiles(dtype, A, B, C)
    sub = rng.integers(0, ntiles * 128, size=min(ntiles * 128, 6000))
    recs = []
    for flat in sub:
        t, rem = divmod(int(flat), 128)
        i, j = divmod(rem, 8)
        recs.append(
            (
                int(C[t, i, j]),
                [int(x) for x in A[t, i]],
                [int(x) for x in B[t, j]],
                int(D[t, i, j]),
            )
        )
    table = {}
    for width in range(22, 29):
        for ze in range(-150, -119):
            p = _model_of(dtype, groups, width, ze)
            mism = 0
            for acc, a, b, d in recs:
                try:
                    if S.tc_dot(p, acc, a, b) != d:
                        mism += 1
                except Exception:  # noqa: BLE001 - any model failure counts as a mismatch
                    mism += 1
            table[f"w{width}_z{ze}"] = mism
    best = sorted(table.items(), key=lambda kv: kv[1])[:8]
    return {
        "records": len(recs),
        "best": best,
        "zero_mismatch": [k for k, v in table.items() if v == 0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dtype", choices=["bf16", "e4m3"])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--groups",
        default=None,
        help="comma list, e.g. 8,8; default: infer from neutral search",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(1234)
    t0 = time.time()
    out = {"dtype": args.dtype, "k": G.K_OF[args.dtype]}
    out["layout"] = layout_test(args.dtype, rng)
    print("layout:", out["layout"], flush=True)
    out["neutral"] = neutral_subgroups(args.dtype)
    print("neutral (acc small):", out["neutral"]["neutral_acc_small"], flush=True)
    print("neutral (acc zero):", out["neutral"]["neutral_acc_zero"], flush=True)
    out["precision"] = internal_precision(args.dtype)
    print("precision:", out["precision"], flush=True)
    if args.groups:
        groups = tuple(int(x) for x in args.groups.split(","))
    else:
        # infer: the smallest neutral set containing acc gives the first group size;
        # assume equal-sized groups covering k.
        small = [s for s in out["neutral"]["neutral_acc_small"] if s[0] == "acc"]
        g1 = min(len(s) - 1 for s in small) if small else G.K_OF[args.dtype]
        groups = tuple([g1] * (G.K_OF[args.dtype] // g1))
    out["groups_assumed"] = list(groups)
    out["fit"] = fit_parameters(args.dtype, groups, rng)
    print("fit:", out["fit"], flush=True)
    out["seconds"] = time.time() - t0
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
