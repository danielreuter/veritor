"""Hawkeye C++ simulators vs the Python port, on random zero-accumulator tiles (CPU only).

Checks that ``veritor.core.silicon`` reproduces ``gpu_simulator_py``'s
Ampere, Hopper and Hopper-FP8 simulators bit-for-bit, independent of any GPU.
Usage: python hawkeye_vs_port.py --tiles 2000 --out results/hawkeye_vs_port.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import tcs_gpu as G
import torch

sys.path.insert(0, G.HERE)
sys.path.insert(0, "/workspace/hawkeye")
import gpu_simulator_py as H
from veritor_core import silicon as S

CASES = [
    ("Ampere_simulator", S.AMPERE_BF16_M16N8K16),
    ("Hopper_simulator", S.HOPPER_BF16_M16N8K16),
    ("Hopper_fp8_simulator", S.HOPPER_E4M3_K32),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    rng = np.random.default_rng(5)
    for name, pipe in CASES:
        sim = getattr(H, name)()
        k = pipe.k
        mism = 0
        mism_randn = 0  # mismatches on the standard-normal tiles only
        overflow_range = 0  # elements of the wide-encoding tiles whose exact result leaves FP32 range
        t0 = time.time()
        for t in range(args.tiles):
            if pipe.dtype == "bf16":
                # half standard-normal tiles, half uniformly random finite encodings
                if t % 2 == 0:
                    a_w = G.to_bf16_bits(
                        rng.standard_normal((16, k)).astype(np.float32)
                    )
                    b_w = G.to_bf16_bits(rng.standard_normal((8, k)).astype(np.float32))
                else:
                    e = rng.integers(1, 200, size=(24, k)).astype(np.uint16)
                    m = rng.integers(0, 128, size=(24, k)).astype(np.uint16)
                    s = rng.integers(0, 2, size=(24, k)).astype(np.uint16)
                    w = (s << 15) | (e << 7) | m
                    a_w, b_w = w[:16], w[16:]
                a = torch.from_numpy(a_w.astype(np.int16)).view(torch.bfloat16)
                bt = torch.from_numpy(b_w.astype(np.int16)).view(torch.bfloat16)
                d = sim.matmul(a, bt.t())
            else:
                if t % 2 == 0:
                    a_w = G.to_e4m3_bits(
                        rng.standard_normal((16, k)).astype(np.float32) * 8
                    )
                    b_w = G.to_e4m3_bits(
                        rng.standard_normal((8, k)).astype(np.float32) * 8
                    )
                else:
                    w = rng.integers(0, 256, size=(24, k)).astype(np.uint8)
                    w[(w & 0x7F) == 0x7F] &= 0xFE
                    a_w, b_w = w[:16], w[16:]
                d = sim.matmul(
                    torch.from_numpy(a_w).view(torch.float8_e4m3fn),
                    torch.from_numpy(b_w).view(torch.float8_e4m3fn),
                )
            d_bits = d.contiguous().view(torch.int32).numpy().astype(np.uint32)
            for i in range(16):
                for j in range(8):
                    got = S.tc_dot(pipe, 0, a_w[i].tolist(), b_w[j].tolist())
                    if (got & 0x7F800000) == 0x7F800000:
                        overflow_range += 1
                    if got != int(d_bits[i, j]):
                        mism += 1
                        if t % 2 == 0:
                            mism_randn += 1
        out[name] = {
            "pipeline": pipe.name,
            "tiles": args.tiles,
            "elements": args.tiles * 128,
            "mismatches": mism,
            "mismatches_on_randn_tiles": mism_randn,
            "randn_elements": (args.tiles + 1) // 2 * 128,
            "port_results_outside_fp32_range": overflow_range,
            "seconds": time.time() - t0,
        }
        print(name, out[name], flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
