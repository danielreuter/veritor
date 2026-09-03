"""Smoke test on the GPU host: every CUDA op of ``CudaOps`` against the numpy op of ``NumpyOps``, bit for bit."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_ops

from veritor.constructors.gpt2_reference import NumpyOps, as_bf16_words


def words(a: np.ndarray) -> np.ndarray:
    return a.view(np.uint32) if a.dtype == np.float32 else a


def report(name: str, gpu: np.ndarray, cpu: np.ndarray) -> int:
    bad = int((words(gpu) != words(cpu)).sum())
    print(f"{name:14s} {gpu.size:8d} elements, {bad} mismatches")
    if bad:
        i = int(np.flatnonzero(words(gpu) != words(cpu))[0])
        print(
            f"   first: gpu {int(words(gpu).reshape(-1)[i]):#x} cpu {int(words(cpu).reshape(-1)[i]):#x}"
        )
    return bad


def main() -> None:
    rng = np.random.default_rng(1)
    gpu = cuda_ops.CudaOps()
    cpu = NumpyOps(cuda_ops.CppGemm())
    total = 0

    def mixed(scale: float, *shape: int) -> np.ndarray:
        x = rng.normal(0, scale, shape).astype(np.float32)
        mask = rng.random(shape) < 0.2
        x[mask] = (x[mask] * np.exp2(rng.uniform(-40, 40, int(mask.sum())))).astype(
            np.float32
        )
        return x

    a, b = mixed(3.0, 5000), mixed(3.0, 5000)
    total += report("add", gpu.add(a, b), cpu.add(a, b))
    total += report(
        "scale", gpu.scale(a, np.float32(0.125)), cpu.scale(a, np.float32(0.125))
    )
    total += report("round", gpu.round(a), cpu.round(a))
    x = mixed(4.0, 20000)
    total += report("gelu", gpu.gelu(x), cpu.gelu(x))
    x = mixed(1.0, 40, 768)
    gm, gc, gr = gpu.ln_stats(x, np.float32(768.0))
    cm, cc, cr = cpu.ln_stats(x, np.float32(768.0))
    total += (
        report("ln_mean", gm, cm)
        + report("ln_center", gc, cc)
        + report("ln_rstd", gr, cr)
    )
    g, bb = (
        as_bf16_words(rng.normal(1, 0.1, 768).astype(np.float32)),
        as_bf16_words(rng.normal(0, 0.1, 768).astype(np.float32)),
    )
    total += report("ln_out", gpu.ln_out(gc, gr, g, bb), cpu.ln_out(cc, cr, g, bb))
    for width in (1, 2, 3, 7, 16, 24, 1000):
        u = mixed(3.0, 9, width)
        total += report(f"row_max[{width}]", gpu.row_max(u), cpu.row_max(u))
        total += report(f"row_sum[{width}]", gpu.row_sum(u), cpu.row_sum(u))
        mx = gpu.row_max(u)
        e_g, e_c = gpu.exp_shift(u, mx), cpu.exp_shift(u, mx)
        total += report(f"exp_shift[{width}]", e_g, e_c)
        s = gpu.row_sum(e_g)
        total += report(
            f"div_round[{width}]", gpu.div_round(e_g, s), cpu.div_round(e_g, s)
        )
    logits = mixed(3.0, 50257)
    logits[100] = logits[7000] = np.float32(
        logits.max() + 1
    )  # a tie: the earlier index wins
    tokens = np.arange(50257, dtype=np.uint16)
    blocks = [64] * 785 + [17]
    bg, ig, tg = gpu.argmax(logits, tokens, blocks)
    bc, ic, tc = cpu.argmax(logits, tokens, blocks)
    total += report("argmax_best", bg, bc) + report("argmax_idx", ig, ic)
    print("argmax token", tg, tc, "mismatch" if tg != tc else "ok")
    total += int(tg != tc)
    # the GPU chain against the C++ chain at the run's shapes
    for m, k, n_out in (
        (16, 768, 768),
        (16, 768, 3072),
        (3, 3072, 768),
        (1, 768, 50257),
        (16, 64, 16),
        (16, 32, 64),
    ):
        aa = as_bf16_words(rng.normal(0, 1, (m, k)).astype(np.float32))
        bt = as_bf16_words(rng.normal(0, 1, (n_out, k)).astype(np.float32))
        acc = mixed(1.0, m, n_out) if m % 2 else None
        total += report(
            f"gemm{m}x{k}x{n_out}", gpu.gemm(aa, bt, acc), cpu.gemm(aa, bt, acc)
        )
    print("TOTAL MISMATCHES", total)


if __name__ == "__main__":
    main()
