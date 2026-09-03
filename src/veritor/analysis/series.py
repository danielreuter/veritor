"""Log2-domain series with upward rounding: the arithmetic behind the fold.

Two kinds of series appear in :mod:`veritor.analysis.bound`.  An
:class:`ErrorSeries` is indexed by the number of erroneous verification
units ``l`` inside one copy of a kind and holds ``log2`` of the summed
capacity weight of the covers those ``l``-subsets use; everything beyond a
truncation ``limit`` is lumped into one *tail* term.  A *cost series* is a
dense array indexed by discretised survival cost, truncated at the budget.

Every entry is ``log2`` of a nonnegative quantity (``-inf`` for zero).  Each
operation adds a slack dominating the float64 rounding it performs, so
every computed entry is an upper bound on the exact value of the same
expression -- the property the bound needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import as_strided

NEG_INF = -math.inf
_CHUNK = 256


def _up(values: np.ndarray | float, terms: int = 1) -> np.ndarray:
    """Round ``values`` up by more than ``terms`` float64 operations can err."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    slack = (terms + 4) * 2.0**-50
    result = values.copy()
    result[finite] = values[finite] + np.abs(values[finite]) * slack + slack
    return result


def log2_sum(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """``log2`` of the sum of ``2**values`` along ``axis``, rounded up."""

    values = np.asarray(values, dtype=float)
    terms = values.size if axis is None else values.shape[axis]
    if terms == 0:
        shape = () if axis is None else tuple(np.delete(values.shape, axis))
        return np.full(shape, NEG_INF)
    peak = np.max(values, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.sum(np.exp2(values - safe), axis=axis, keepdims=True)
        result = safe + np.log2(total)
    result = np.where(np.isfinite(peak), result, NEG_INF)
    return _up(
        np.squeeze(result, axis=axis) if axis is not None else result.reshape(()), terms
    )


def log2_binomials(n: int, upto: int) -> np.ndarray:
    """``log2 C(n, l)`` for ``l = 0..upto`` (``upto <= n``), rounded up."""

    if upto > n:
        raise ValueError("upto must not exceed n")
    if upto == 0:
        return np.zeros(1)
    steps = np.arange(upto, dtype=float)
    increments = np.log2(n - steps) - np.log2(steps + 1)
    values = np.concatenate([[0.0], np.cumsum(increments)])
    return _up(values, 3 * upto)


def _skew(grid: np.ndarray) -> np.ndarray:
    """View ``grid[i, j - i]`` as ``skew[i, j]`` (``-inf`` outside), no copy."""

    rows, cols = grid.shape
    padded = np.full((rows, cols + rows), NEG_INF)
    padded[:, :cols] = grid
    stride_row, stride_col = padded.strides
    return as_strided(
        padded,
        shape=(rows, rows + cols - 1),
        strides=(stride_row - stride_col, stride_col),
    )


def convolve(a: np.ndarray, b: np.ndarray, limit: int) -> np.ndarray:
    """``log2`` of the convolution of ``2**a`` and ``2**b`` up to index ``limit``."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = min(len(a) + len(b) - 1, limit + 1)
    if len(a) == 0 or len(b) == 0 or length <= 0:
        return np.full(max(length, 0), NEG_INF)
    a = a[: min(len(a), length)]
    b = b[: min(len(b), length)]
    partials: list[np.ndarray] = []
    for start in range(0, len(a), _CHUNK):
        rows = a[start : start + _CHUNK]
        skew = _skew(rows[:, None] + b[None, :])
        partial = np.full(length, NEG_INF)
        span = min(skew.shape[1], length - start)
        partial[start : start + span] = log2_sum(skew[:, :span], axis=0)
        partials.append(partial)
    if len(partials) == 1:
        return partials[0]
    return log2_sum(np.stack(partials), axis=0)


def prefix_sums(values: np.ndarray) -> np.ndarray:
    """``log2`` of the running sums of ``2**values``, rounded up."""

    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        running = np.logaddexp2.accumulate(values)
    return _up(running, len(values))


# -- error series ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorSeries:
    """Cover weights of the ``l``-subsets of one copy's verification units.

    ``head[l]`` is ``log2`` of the summed ``2**kappa`` over the distinct
    covers used by the subsets of exactly ``l`` units, for ``l`` up to the
    truncation; ``tail`` is the same for every subset with more units.  The
    empty subset always contributes ``head[0] == 0`` (one cover of zero bits).
    """

    head: np.ndarray
    tail: float

    @property
    def total(self) -> float:
        return float(log2_sum(np.append(self.head, self.tail)))

    @property
    def head_total(self) -> float:
        return float(log2_sum(self.head))


def empty_series() -> ErrorSeries:
    """A kind with no verification units inside."""

    return ErrorSeries(np.zeros(1), NEG_INF)


def unit_series(out_bits: int) -> ErrorSeries:
    """A verification unit: either correct or covered by its own interface."""

    return ErrorSeries(np.array([0.0, float(out_bits)]), NEG_INF)


def multiply(a: ErrorSeries, b: ErrorSeries, limit: int) -> ErrorSeries:
    """Cover weights of the disjoint union of two collections of units."""

    full = convolve(a.head, b.head, len(a.head) + len(b.head) - 2)
    head = full[: limit + 1]
    overflow = full[limit + 1 :]
    tail = log2_sum(
        np.concatenate([[a.tail + b.total, b.tail + a.head_total], overflow])
    )
    return ErrorSeries(head, float(tail))


def _binomial_power(weight: float, copies: int, limit: int) -> ErrorSeries:
    """``(1 + 2**weight x)**copies``: ``copies`` independent units of one cover each."""

    upto = min(copies, limit)
    head = _up(log2_binomials(copies, upto) + weight * np.arange(upto + 1), 1)
    if copies <= limit:
        return ErrorSeries(head, NEG_INF)
    if copies - limit <= 64:
        rest = log2_binomials(copies, copies)[limit + 1 :] + weight * np.arange(
            limit + 1, copies + 1
        )
        return ErrorSeries(head, float(log2_sum(_up(rest, 1))))
    # log2(1 + 2**w) = w + log2(1 + 2**-w): the sum over every subset.
    everything = copies * float(
        _up(weight + math.log1p(math.exp2(-weight)) / math.log(2), 3)
    )
    return ErrorSeries(head, everything)


def power(series: ErrorSeries, copies: int, limit: int) -> ErrorSeries:
    """Cover weights of ``copies`` disjoint copies of ``series``."""

    if copies == 0:
        return empty_series()
    if len(series.head) == 2 and series.tail == NEG_INF and series.head[0] == 0.0:
        return _binomial_power(float(series.head[1]), copies, limit)
    result = empty_series()
    base = series
    while True:
        if copies & 1:
            result = multiply(result, base, limit)
        copies >>= 1
        if not copies:
            return result
        base = multiply(base, base, limit)


def cap(series: ErrorSeries, out_bits: int) -> ErrorSeries:
    """Every nonempty subset may instead be covered by the enclosing node."""

    head = series.head.copy()
    head[1:] = np.minimum(head[1:], float(out_bits))
    return ErrorSeries(head, min(series.tail, float(out_bits)))


# -- cost series -------------------------------------------------------------


def sparse_power(
    exponents: np.ndarray, values: np.ndarray, copies: int, limit: int
) -> np.ndarray:
    """``log2`` coefficients of ``P**copies`` up to degree ``limit``.

    ``P = sum_i 2**values[i] x**exponents[i]`` with distinct exponents
    including ``0``.  Miller's recurrence is used when every coefficient of
    it is positive, which holds once ``(copies + 1) * min_exponent > limit``;
    it costs ``O(limit * terms)`` and never enumerates copies.  Otherwise
    ``copies < limit / min_exponent`` and the power is formed by that many
    sparse multiplications, ``O(copies * limit * terms)``.
    """

    order = np.argsort(exponents)
    exponents = np.asarray(exponents)[order]
    values = np.asarray(values, dtype=float)[order]
    if exponents[0] != 0:
        raise ValueError("the constant term must be present")
    if copies == 0:
        result = np.full(limit + 1, NEG_INF)
        result[0] = 0.0
        return result
    if len(exponents) == 1:
        result = np.full(limit + 1, NEG_INF)
        result[0] = float(_up(copies * values[0], 1))
        return result
    lowest = int(exponents[1])
    if (copies + 1) * lowest > limit:
        return _miller(exponents, values, copies, limit)
    keep = exponents <= limit
    exponents, values = exponents[keep], values[keep]
    result = np.full(limit + 1, NEG_INF)
    result[0] = 0.0
    for _ in range(copies):
        result = sparse_multiply(result, exponents, values)
    return result


def sparse_multiply(
    dense: np.ndarray, exponents: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """``dense * P`` truncated to ``len(dense)``, ``O(len(dense) * terms)``."""

    limit = len(dense)
    stacked = np.full((len(exponents), limit), NEG_INF)
    for row, (shift, value) in enumerate(zip(exponents, values, strict=True)):
        if shift < limit:
            stacked[row, shift:] = dense[: limit - shift] + value
    return log2_sum(stacked, axis=0)


def _miller(
    exponents: np.ndarray, values: np.ndarray, copies: int, limit: int
) -> np.ndarray:
    """``P**copies`` by ``j p_0 f_j = sum_i ((n+1) e_i - j) p_i f_{j-e_i}``."""

    constant = values[0]
    shifts = exponents[1:]
    weights = values[1:]
    block = int(shifts[0])
    result = np.full(limit + 1, NEG_INF)
    result[0] = float(_up(copies * constant, 1))
    for start in range(1, limit + 1, block):
        positions = np.arange(start, min(start + block, limit + 1))
        sources = positions[None, :] - shifts[:, None]
        valid = sources >= 0
        gathered = np.where(valid, result[np.where(valid, sources, 0)], NEG_INF)
        factors = (copies + 1) * shifts[:, None] - positions[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                valid,
                np.log2(np.maximum(factors, 1)) + weights[:, None] + gathered,
                NEG_INF,
            )
        summed = log2_sum(terms, axis=0)
        result[positions] = _up(summed - np.log2(positions) - constant, 2)
    return result
