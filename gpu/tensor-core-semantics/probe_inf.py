import numpy as np
import tcs_gpu as G


def run(dtype, cases):
    K = G.K_OF[dtype]
    n = len(cases)
    A = np.zeros((n, 16, K), np.float32)
    B = np.zeros((n, 8, K), np.float32)
    C = np.zeros((n, 16, 8), np.uint32)
    for i, (a, b, c) in enumerate(cases):
        A[i, 0, : len(a)] = a
        B[i, 0, : len(b)] = b
        C[i, 0, 0] = c
    return [
        f"{x:08x}"
        for x in G.mma_tiles(
            dtype, G.f32_to_operand(dtype, A), G.f32_to_operand(dtype, B), C
        )[:, 0, 0]
    ]


INF = 0x7F800000
NINF = 0xFF800000
NAN = 0x7FC00000
MAXF = 0x7F7FFFFF
for dtype in ("bf16", "e4m3"):
    big = 2.0**60 if dtype == "bf16" else 256.0
    print(dtype, "acc=+inf, zero products        ->", run(dtype, [([0], [0], INF)]))
    print(dtype, "acc=+inf, products +big*big    ->", run(dtype, [([big], [big], INF)]))
    print(
        dtype, "acc=+inf, products -big*big    ->", run(dtype, [([-big], [big], INF)])
    )
    print(dtype, "acc=NaN(7fc00000), zero prods  ->", run(dtype, [([0], [0], NAN)]))
    print(
        dtype, "acc=NaN(7fc12345), zero prods  ->", run(dtype, [([0], [0], 0x7FC12345)])
    )
    print(dtype, "acc=maxfloat + 1*1             ->", run(dtype, [([1], [1], MAXF)]))
    print(
        dtype, "acc=maxfloat + big*big         ->", run(dtype, [([big], [big], MAXF)])
    )
    print(
        dtype,
        "acc=maxfloat, g1 +big*big, g2 -(big*big) ->",
        run(
            dtype,
            [
                (
                    [big] + [0] * (G.K_OF[dtype] // 2 - 1) + [-big],
                    [big] + [0] * (G.K_OF[dtype] // 2 - 1) + [big],
                    MAXF,
                )
            ],
        ),
    )
    if dtype == "bf16":
        inf16 = float("inf")
        nan16 = float("nan")
        print(
            dtype,
            "operand inf*1 acc 0            ->",
            run(dtype, [([inf16], [1.0], 0)]),
        )
        print(
            dtype,
            "operand inf*0 acc 0            ->",
            run(dtype, [([inf16], [0.0], 0)]),
        )
        print(
            dtype,
            "operand inf*1 + (-inf)*1       ->",
            run(dtype, [([inf16, -inf16], [1.0, 1.0], 0)]),
        )
        print(
            dtype,
            "operand nan*1                  ->",
            run(dtype, [([nan16], [1.0], 0)]),
        )
        print(
            dtype,
            "products 2^200 - 2^200 (exact cancel above fp32 max) ->",
            run(dtype, [([2.0**100, -(2.0**100)], [2.0**100, 2.0**100], 0)]),
        )
        print(
            dtype,
            "products 2^200 - 2^200 + 1     ->",
            run(dtype, [([2.0**100, -(2.0**100), 1.0], [2.0**100, 2.0**100, 1.0], 0)]),
        )
        print(
            dtype,
            "g1: 2^128 (+inf?) ; g2: -2^128 ->",
            run(
                dtype,
                [
                    (
                        [2.0**64] + [0] * 7 + [-(2.0**64)],
                        [2.0**64] + [0] * 7 + [2.0**64],
                        0,
                    )
                ],
            ),
        )
    else:
        print(
            dtype,
            "operand NaN(0x7f) * 1 acc 0    ->",
            [
                f"{x:08x}"
                for x in G.mma_tiles(
                    "e4m3",
                    np.array([[[0x7F] + [0] * 31] * 16], np.uint8),
                    np.array([[[0x38] + [0] * 31] * 8], np.uint8),
                    np.zeros((1, 16, 8), np.uint32),
                )[:, 0, 0]
            ],
        )
        print(
            dtype,
            "operand NaN(0x7f) * 0 acc 0    ->",
            [
                f"{x:08x}"
                for x in G.mma_tiles(
                    "e4m3",
                    np.array([[[0x7F] + [0] * 31] * 16], np.uint8),
                    np.array([[[0x00] + [0] * 31] * 8], np.uint8),
                    np.zeros((1, 16, 8), np.uint32),
                )[:, 0, 0]
            ],
        )
