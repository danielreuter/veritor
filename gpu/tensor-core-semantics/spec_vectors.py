"""Produce FP8 vectors for the tc-dot-spec (hawkeye_ampere_groupsum_fp8e4m3_v0) comparison.

Runs on the GPU host.  Writes a text file of GPU-measured E4M3 records, one
per line: ``acc_hex a_hex b_hex d_hex`` with a/b as 32 E4M3 bytes (64 hex
chars).  Two families:

* ``half``: products 16..31 are zero, so the k32 instruction reduces to the
  hardware's 16-element relation (group 2 is an identity on group 1's
  output) and can be compared directly with ``tc_dot_spec::tile``;
* ``full``: all 32 products random, to be compared with two chained
  ``tile`` calls (how a K=32 dot would be compiled into v0 gates).

Also runs the ten ``GOLDEN_TILES`` of tc-dot-spec through the hardware.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tcs_gpu as G
from validate_tiles import random_finite_fp32_bits, random_finite_operand_bits

SPEC_GOLDEN = [  # name, a[16], b[16], c_bits, expected d_bits (from tc-dot-spec golden.rs)
    ("sixteen_ones", [0x38] * 16, [0x38] * 16, 0x00000000, 0x41800000),
    (
        "mixed_sign_nonzero_accumulator",
        [0x3C, 0x40] + [0] * 14,
        [0x38, 0xBC] + [0] * 14,
        0x3E800000,
        0xBFA00000,
    ),
    (
        "product_remains_unnormalized",
        [0x3C] + [0] * 15,
        [0x3C] + [0] * 15,
        0x00000000,
        0x40100000,
    ),
    (
        "exact_sign_cancellation",
        [0x38, 0x38] + [0] * 14,
        [0x38, 0xB8] + [0] * 14,
        0x00000000,
        0x00000000,
    ),
    (
        "unnormalized_subnormal_product",
        [0x03, 0x48] + [0] * 14,
        [0x38, 0x38] + [0] * 14,
        0x00000000,
        0x40803000,
    ),
    (
        "alignment_truncates_small_product",
        [0x98] + [0] * 15,
        [0x18] + [0] * 15,
        0x48000000,
        0x48000000,
    ),
    (
        "one_bit_closer_product_survives",
        [0xA0] + [0] * 15,
        [0x18] + [0] * 15,
        0x48000000,
        0x47FFFFFF,
    ),
    (
        "normalization_is_not_round_to_nearest",
        [0x3C] + [0] * 15,
        [0x38] + [0] * 15,
        0x4B000000,
        0x4B000001,
    ),
    (
        "two_stage_cancellation_preserves_second_half",
        [0x51, 0x50, 0, 0, 0, 0, 0, 0, 0x04, 0, 0, 0, 0, 0, 0, 0],
        [0x50, 0xD0, 0, 0, 0, 0, 0, 0, 0x04, 0, 0, 0, 0, 0, 0, 0],
        0x00000000,
        0x41000040,
    ),
    ("negative_zero_canonicalizes", [0] * 16, [0] * 16, 0x80000000, 0x00000000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--tiles", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(31337)
    n = args.tiles
    for fam in ("half", "full"):
        # mix of standard-normal-derived operands and uniformly random encodings
        A = np.concatenate(
            [
                G.to_e4m3_bits(
                    rng.standard_normal((n // 2, 16, 32)).astype(np.float32) * 8
                ),
                random_finite_operand_bits("e4m3", rng, (n - n // 2, 16, 32)),
            ]
        )
        B = np.concatenate(
            [
                G.to_e4m3_bits(
                    rng.standard_normal((n // 2, 8, 32)).astype(np.float32) * 8
                ),
                random_finite_operand_bits("e4m3", rng, (n - n // 2, 8, 32)),
            ]
        )
        C = np.where(
            rng.random((n, 16, 8)) < 0.5,
            random_finite_fp32_bits(rng, n * 128, 90, 160).reshape(n, 16, 8),
            np.uint32(0),
        ).astype(np.uint32)
        if fam == "half":
            A[:, :, 16:] = 0
            B[:, :, 16:] = 0
        D = G.mma_tiles("e4m3", A, B, C)
        with open(f"{args.out_prefix}_{fam}.txt", "w") as f:
            for t in range(n):
                for i in range(16):
                    f.writelines(
                        f"{int(C[t, i, j]):08x} {A[t, i].tobytes().hex()} {B[t, j].tobytes().hex()} {int(D[t, i, j]):08x}\n"
                        for j in range(8)
                    )
    # tc-dot-spec golden tiles on hardware
    A = np.zeros((len(SPEC_GOLDEN), 16, 32), np.uint8)
    B = np.zeros((len(SPEC_GOLDEN), 8, 32), np.uint8)
    C = np.zeros((len(SPEC_GOLDEN), 16, 8), np.uint32)
    for t, (_, a, b, c, _) in enumerate(SPEC_GOLDEN):
        A[t, 0, :16] = a
        B[t, 0, :16] = b
        C[t, 0, 0] = c
    D = G.mma_tiles("e4m3", A, B, C)
    rows = []
    for t, (name, a, b, c, want) in enumerate(SPEC_GOLDEN):
        got = int(D[t, 0, 0])
        rows.append(
            {
                "name": name,
                "spec_d": f"{want:08x}",
                "gpu_d": f"{got:08x}",
                "match": got == want,
            }
        )
        print(
            f"{'OK  ' if got == want else 'DIFF'} {name:48s} spec={want:08x} gpu={got:08x}"
        )
    with open(f"{args.out_prefix}_golden.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
