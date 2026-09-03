"""The GPU backend of the pinned forward pass: every operation is one of our CUDA kernels.

``CudaOps`` implements :class:`veritor.constructors.gpt2_reference.Ops`
over ``libpinned_ops.so`` (this directory) and ``libgemm_chain.so``
(``gpu/tensor-core-semantics/gemm_chain.cu``), through ``ctypes``; numpy
arrays are the containers, no framework does any arithmetic.  ``CppGemm``
is the CPU chain reference ``ref_chain.cpp`` behind the same interface.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence

import numpy as np

from veritor.core.silicon import ADA_BF16_M16N8K16

HERE = os.path.dirname(os.path.abspath(__file__))
TCS = os.path.join(os.path.dirname(HERE), "tensor-core-semantics")

_f32p = ctypes.POINTER(ctypes.c_float)
_u16p = ctypes.POINTER(ctypes.c_uint16)
_i32p = ctypes.POINTER(ctypes.c_int)


def _f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _u16(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.uint16)


def _fp(a: np.ndarray):
    return a.ctypes.data_as(_f32p)


def _up(a: np.ndarray):
    return a.ctypes.data_as(_u16p)


class GemmChainLib:
    """``gemm_chain_bf16`` from ``libgemm_chain.so``: the fixed-order tensor-core GEMM on the GPU."""

    def __init__(self, path: str | None = None) -> None:
        self.lib = ctypes.CDLL(path or os.path.join(TCS, "libgemm_chain.so"))
        fn = self.lib.gemm_chain_bf16
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [_f32p]
        self.kernel_ms = 0.0  # accumulated timed kernel time when bench_iters > 0

    def gemm(
        self, a: np.ndarray, bt: np.ndarray, c: np.ndarray | None, bench_iters: int = 0
    ) -> np.ndarray:
        a, bt = _u16(a), _u16(bt)
        m, k = a.shape
        n, k2 = bt.shape
        assert k == k2 and k % 16 == 0
        d = np.zeros((m, n), dtype=np.float32)
        ms = ctypes.c_float(0)
        cc = None if c is None else _f32(c)
        rc = self.lib.gemm_chain_bf16(
            a.ctypes.data,
            bt.ctypes.data,
            None if cc is None else cc.ctypes.data,
            d.ctypes.data,
            m,
            n,
            k,
            bench_iters,
            ctypes.byref(ms),
        )
        if rc != 0:
            raise RuntimeError(f"gemm_chain_bf16 failed: {rc}")
        self.kernel_ms += ms.value
        return d


class CppGemm:
    """``ref_chain_bf16`` from ``libref_chain.so``: the CPU chain reference (OpenMP over outputs)."""

    def __init__(self, path: str | None = None) -> None:
        self.lib = ctypes.CDLL(path or os.path.join(TCS, "libref_chain.so"))
        fn = self.lib.ref_chain_bf16
        fn.restype = None
        fn.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        )
        p = ADA_BF16_M16N8K16
        self._groups = (ctypes.c_int * len(p.groups))(*p.groups)
        self._ngroups, self._width, self._zero = len(p.groups), p.width, p.zero_exponent

    def gemm(self, a: np.ndarray, bt: np.ndarray, c: np.ndarray | None) -> np.ndarray:
        a, bt = _u16(a), _u16(bt)
        m, k = a.shape
        n, k2 = bt.shape
        assert k == k2 and k % 16 == 0
        d = np.zeros((m, n), dtype=np.uint32)
        cc = None if c is None else _f32(c)
        self.lib.ref_chain_bf16(
            a.ctypes.data,
            bt.ctypes.data,
            None if cc is None else cc.ctypes.data,
            d.ctypes.data,
            m,
            n,
            k,
            self._groups,
            self._ngroups,
            self._width,
            self._zero,
        )
        return d.view(np.float32)


U_EXP, U_TANH, U_GELU, U_RSTD = 0, 1, 2, 3
B_ADD, B_SUB, B_MUL, B_DIV, B_MAX = 0, 1, 2, 3, 4


class PinnedOpsLib:
    """The elementwise / reduction kernels of ``pinned_ops.cu`` as numpy functions."""

    def __init__(self, path: str | None = None) -> None:
        lib = ctypes.CDLL(path or os.path.join(HERE, "libpinned_ops.so"))
        self.lib = lib
        lib.pinned_unary.argtypes = [ctypes.c_int, _f32p, _f32p, ctypes.c_int]
        lib.pinned_binary.argtypes = [ctypes.c_int, _f32p, _f32p, _f32p, ctypes.c_int]
        lib.pinned_scale.argtypes = [_f32p, ctypes.c_float, _f32p, ctypes.c_int]
        lib.pinned_round.argtypes = [_f32p, _u16p, ctypes.c_int]
        lib.pinned_select.argtypes = [_f32p, _f32p, _u16p, _u16p, _u16p, ctypes.c_int]
        lib.pinned_token_eq.argtypes = [_u16p, _u16p, _u16p, ctypes.c_int]
        lib.pinned_ln_stats.argtypes = [
            _f32p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            _f32p,
            _f32p,
            _f32p,
        ]
        lib.pinned_ln_out.argtypes = [
            _f32p,
            _f32p,
            _u16p,
            _u16p,
            _u16p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.pinned_row_reduce.argtypes = [
            ctypes.c_int,
            _f32p,
            ctypes.c_int,
            ctypes.c_int,
            _f32p,
        ]
        lib.pinned_exp_shift.argtypes = [
            _f32p,
            _f32p,
            _f32p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.pinned_div_round.argtypes = [
            _f32p,
            _f32p,
            _u16p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.pinned_tournament.argtypes = [
            _f32p,
            _u16p,
            ctypes.c_int,
            _i32p,
            _i32p,
            ctypes.c_int,
            _f32p,
            _u16p,
        ]
        for name in (
            "pinned_unary",
            "pinned_binary",
            "pinned_scale",
            "pinned_round",
            "pinned_select",
            "pinned_token_eq",
            "pinned_ln_stats",
            "pinned_ln_out",
            "pinned_row_reduce",
            "pinned_exp_shift",
            "pinned_div_round",
            "pinned_tournament",
        ):
            getattr(lib, name).restype = ctypes.c_int
        self.calls = 0

    def _check(self, rc: int, name: str) -> None:
        self.calls += 1
        if rc != 0:
            raise RuntimeError(f"{name} failed with CUDA error {rc}")

    def unary(self, op: int, a: np.ndarray) -> np.ndarray:
        a = _f32(a)
        out = np.empty_like(a)
        self._check(self.lib.pinned_unary(op, _fp(a), _fp(out), a.size), "pinned_unary")
        return out

    def binary(self, op: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _f32(a), _f32(b)
        assert a.shape == b.shape
        out = np.empty_like(a)
        self._check(
            self.lib.pinned_binary(op, _fp(a), _fp(b), _fp(out), a.size),
            "pinned_binary",
        )
        return out

    def scale(self, a: np.ndarray, s: float) -> np.ndarray:
        a = _f32(a)
        out = np.empty_like(a)
        self._check(
            self.lib.pinned_scale(_fp(a), ctypes.c_float(float(s)), _fp(out), a.size),
            "pinned_scale",
        )
        return out

    def round(self, a: np.ndarray) -> np.ndarray:
        a = _f32(a)
        out = np.empty(a.shape, dtype=np.uint16)
        self._check(self.lib.pinned_round(_fp(a), _up(out), a.size), "pinned_round")
        return out

    def select(
        self, la: np.ndarray, lb: np.ndarray, ia: np.ndarray, ib: np.ndarray
    ) -> np.ndarray:
        la, lb, ia, ib = _f32(la), _f32(lb), _u16(ia), _u16(ib)
        out = np.empty(la.shape, dtype=np.uint16)
        self._check(
            self.lib.pinned_select(
                _fp(la), _fp(lb), _up(ia), _up(ib), _up(out), la.size
            ),
            "pinned_select",
        )
        return out

    def token_eq(self, t: np.ndarray, j: np.ndarray) -> np.ndarray:
        t, j = _u16(t), _u16(j)
        out = np.empty(t.shape, dtype=np.uint16)
        self._check(
            self.lib.pinned_token_eq(_up(t), _up(j), _up(out), t.size),
            "pinned_token_eq",
        )
        return out

    def ln_stats(
        self, x: np.ndarray, n: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = _f32(x)
        rows, d = x.shape
        mean, rstd = np.empty(rows, dtype=np.float32), np.empty(rows, dtype=np.float32)
        center = np.empty_like(x)
        self._check(
            self.lib.pinned_ln_stats(
                _fp(x),
                rows,
                d,
                ctypes.c_float(float(n)),
                _fp(mean),
                _fp(center),
                _fp(rstd),
            ),
            "pinned_ln_stats",
        )
        return mean, center, rstd

    def ln_out(
        self, center: np.ndarray, rstd: np.ndarray, g: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        center, rstd, g, b = _f32(center), _f32(rstd), _u16(g), _u16(b)
        rows, d = center.shape
        out = np.empty((rows, d), dtype=np.uint16)
        self._check(
            self.lib.pinned_ln_out(
                _fp(center), _fp(rstd), _up(g), _up(b), _up(out), rows, d
            ),
            "pinned_ln_out",
        )
        return out

    def row_reduce(self, op: int, u: np.ndarray) -> np.ndarray:
        u = _f32(u)
        rows, c = u.shape
        out = np.empty(rows, dtype=np.float32)
        self._check(
            self.lib.pinned_row_reduce(op, _fp(u), rows, c, _fp(out)),
            "pinned_row_reduce",
        )
        return out

    def exp_shift(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        u, m = _f32(u), _f32(m)
        rows, c = u.shape
        out = np.empty_like(u)
        self._check(
            self.lib.pinned_exp_shift(_fp(u), _fp(m), _fp(out), rows, c),
            "pinned_exp_shift",
        )
        return out

    def div_round(self, e: np.ndarray, s: np.ndarray) -> np.ndarray:
        e, s = _f32(e), _f32(s)
        rows, c = e.shape
        out = np.empty((rows, c), dtype=np.uint16)
        self._check(
            self.lib.pinned_div_round(_fp(e), _fp(s), _up(out), rows, c),
            "pinned_div_round",
        )
        return out

    def tournament(
        self, logits: np.ndarray, tokens: np.ndarray, blocks: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        logits, tokens = _f32(logits), _u16(tokens)
        sizes = np.array(list(blocks), dtype=np.int32)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
        best = np.empty(len(sizes), dtype=np.float32)
        index = np.empty(len(sizes), dtype=np.uint16)
        self._check(
            self.lib.pinned_tournament(
                _fp(logits),
                _up(tokens),
                logits.size,
                starts.ctypes.data_as(_i32p),
                sizes.ctypes.data_as(_i32p),
                len(sizes),
                _fp(best),
                _up(index),
            ),
            "pinned_tournament",
        )
        return best, index


class CudaOps:
    """:class:`veritor.constructors.gpt2_reference.Ops` on the GPU: chains and CUDA-core ops are our kernels."""

    def __init__(
        self, gemm: GemmChainLib | None = None, ops: PinnedOpsLib | None = None
    ) -> None:
        self.chain = GemmChainLib() if gemm is None else gemm
        self.ops = PinnedOpsLib() if ops is None else ops

    def gemm(self, a: np.ndarray, bt: np.ndarray, c: np.ndarray | None) -> np.ndarray:
        return self.chain.gemm(a, bt, c)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.ops.binary(B_ADD, a, b)

    def scale(self, a: np.ndarray, s: np.float32) -> np.ndarray:
        return self.ops.scale(a, float(s))

    def round(self, a: np.ndarray) -> np.ndarray:
        return self.ops.round(a)

    def ln_stats(
        self, x: np.ndarray, n: np.float32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.ops.ln_stats(x, float(n))

    def ln_out(
        self, center: np.ndarray, rstd: np.ndarray, g: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        return self.ops.ln_out(center, rstd, g, b)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return self.ops.unary(U_GELU, x)

    def row_max(self, u: np.ndarray) -> np.ndarray:
        return self.ops.row_reduce(1, u)

    def exp_shift(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        return self.ops.exp_shift(u, m)

    def row_sum(self, e: np.ndarray) -> np.ndarray:
        return self.ops.row_reduce(0, e)

    def div_round(self, e: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self.ops.div_round(e, s)

    def argmax(
        self, logits: np.ndarray, tokens: np.ndarray, blocks: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, int]:
        best, index = self.ops.tournament(logits, tokens, blocks)
        if len(blocks) == 1:
            return best, index, int(index[0])
        _, top = self.ops.tournament(best, index, [len(blocks)])
        return best, index, int(top[0])
