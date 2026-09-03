"""ctypes driver for the single-instruction tensor-core probes in ``mma_tiles.cu``.

Runs on the GPU host.  All tensors are raw bit patterns in numpy arrays:
BF16 words are ``uint16``, E4M3 bytes ``uint8``, FP32 accumulators/results
``uint32``.  ``mma_tiles`` executes exactly one ``mma.sync`` per tile.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, "libmma_tiles.so"))
for name in ("mma_bf16_tiles", "mma_e4m3_tiles"):
    fn = getattr(_lib, name)
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]

K_OF = {"bf16": 16, "e4m3": 32}
WORD_OF = {"bf16": np.uint16, "e4m3": np.uint8}
MAX_TILES_PER_CALL = 1 << 16


def mma_tiles(dtype: str, A: np.ndarray, Bt: np.ndarray, C: np.ndarray) -> np.ndarray:
    """One ``mma.sync`` per tile.  ``A``: (n,16,K), ``Bt``: (n,8,K), ``C``: (n,16,8) -> ``D`` (n,16,8)."""

    k = K_OF[dtype]
    n = A.shape[0]
    A = np.ascontiguousarray(A, dtype=WORD_OF[dtype]).reshape(n, 16, k)
    Bt = np.ascontiguousarray(Bt, dtype=WORD_OF[dtype]).reshape(n, 8, k)
    C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
    D = np.zeros((n, 16, 8), dtype=np.uint32)
    fn = _lib.mma_bf16_tiles if dtype == "bf16" else _lib.mma_e4m3_tiles
    for start in range(0, n, MAX_TILES_PER_CALL):
        stop = min(n, start + MAX_TILES_PER_CALL)
        a, b, c, d = A[start:stop], Bt[start:stop], C[start:stop], D[start:stop]
        d = np.ascontiguousarray(d)
        rc = fn(
            a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, stop - start
        )
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in {dtype} tile kernel")
        D[start:stop] = d
    return D


# --- encodings ----------------------------------------------------------------


def f32_bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def bits_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.uint32).view(np.float32)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 (same as torch's cast)."""

    u = f32_bits(x).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    return rounded.astype(np.uint16)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_e4m3_bits(x: np.ndarray) -> np.ndarray:
    """FP32 -> OCP E4M3 (finite, saturating to +-448), round to nearest even.

    Uses torch's float8_e4m3fn cast, then clamps NaN encodings that torch
    emits for overflow back to the maximum finite value.
    """

    import torch

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t = torch.clamp(t, -448.0, 448.0)
    b = t.to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    return b


def e4m3_to_f32(x: np.ndarray) -> np.ndarray:
    import torch

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8)).view(
        torch.float8_e4m3fn
    )
    return t.to(torch.float32).numpy()


def operand_to_f32(dtype: str, x: np.ndarray) -> np.ndarray:
    return bf16_to_f32(x) if dtype == "bf16" else e4m3_to_f32(x)


def f32_to_operand(dtype: str, x: np.ndarray) -> np.ndarray:
    return to_bf16_bits(x) if dtype == "bf16" else to_e4m3_bits(x)
