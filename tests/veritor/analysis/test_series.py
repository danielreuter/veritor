"""The log2 series arithmetic against exact integer polynomials."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from veritor.analysis.series import (
    NEG_INF,
    ErrorSeries,
    cap,
    convolve,
    empty_series,
    log2_binomials,
    log2_sum,
    multiply,
    power,
    prefix_sums,
    sparse_multiply,
    sparse_power,
    unit_series,
)

TOLERANCE = 1e-6
"""Upper bounds must sit above the exact value, and within this many bits of it."""


def poly_multiply(a: list[int], b: list[int]) -> list[int]:
    result = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return result


def poly_power(base: list[int], copies: int) -> list[int]:
    result = [1]
    for _ in range(copies):
        result = poly_multiply(result, base)
    return result


def log2_int(value: int) -> float:
    return math.log2(value) if value else NEG_INF


def assert_upper(computed: np.ndarray | float, exact: list[int] | int) -> None:
    """``computed`` is ``log2`` of something at least ``exact`` and not much more."""

    computed = np.atleast_1d(np.asarray(computed, dtype=float))
    exact = [exact] if isinstance(exact, int) else exact
    assert len(computed) == len(exact)
    for got, want in zip(computed, exact, strict=True):
        if want == 0:
            assert got == NEG_INF, (got, want)
            continue
        assert got >= log2_int(want), (got, want)
        assert got <= log2_int(want) + TOLERANCE, (got, want)


def test_log2_sum_rounds_up_and_handles_empty_and_zero():
    assert log2_sum(np.array([0.0, 1.0, NEG_INF])) >= math.log2(3)
    assert log2_sum(np.array([0.0, 1.0, NEG_INF])) < math.log2(3) + TOLERANCE
    assert log2_sum(np.array([])) == NEG_INF
    assert log2_sum(np.array([NEG_INF, NEG_INF])) == NEG_INF
    rows = log2_sum(np.array([[0.0, 1.0], [NEG_INF, NEG_INF]]), axis=1)
    assert rows[0] == pytest.approx(math.log2(3), abs=TOLERANCE)
    assert rows[1] == NEG_INF
    huge = log2_sum(np.array([1e6, 1e6]))
    assert 1e6 + 1 <= huge < 1e6 + 1 + 1e-6


def test_log2_binomials_match_math_comb():
    for n in (0, 1, 5, 40, 1000):
        assert_upper(log2_binomials(n, n), [math.comb(n, k) for k in range(n + 1)])
    big = log2_binomials(10**8, 3)
    assert_upper(big, [math.comb(10**8, k) for k in range(4)])
    with pytest.raises(ValueError, match="upto"):
        log2_binomials(3, 4)


def test_convolve_matches_integer_convolution_across_chunk_boundaries():
    rng = random.Random(1)
    for size_a, size_b, limit in [(3, 2, 10), (3, 2, 2), (600, 700, 1000), (300, 5, 299), (1, 1, 0)]:
        a = [rng.randint(0, 1 << 20) for _ in range(size_a)]
        b = [rng.randint(0, 1 << 20) for _ in range(size_b)]
        exact = poly_multiply(a, b)[: limit + 1]
        got = convolve(np.array([log2_int(x) for x in a]), np.array([log2_int(x) for x in b]), limit)
        assert_upper(got, exact)
    assert len(convolve(np.array([]), np.array([0.0]), 5)) == 0


def test_prefix_sums_are_running_totals():
    values = [1, 2, 3, 0, 5]
    running = [sum(values[: i + 1]) for i in range(len(values))]
    assert_upper(prefix_sums(np.array([log2_int(v) for v in values])), running)


def exact_unit_power(out_bits: int, copies: int) -> list[int]:
    return poly_power([1, 1 << out_bits], copies)


@pytest.mark.parametrize(("copies", "limit"), [(3, 5), (10, 4), (200, 4), (5, 5), (0, 3)])
def test_power_of_a_unit_is_the_binomial_expansion(copies, limit):
    exact = exact_unit_power(2, copies)
    result = power(unit_series(2), copies, limit)
    assert_upper(result.head, exact[: limit + 1])
    tail = sum(exact[limit + 1 :])
    if tail:
        assert result.tail >= math.log2(tail)
        assert result.tail <= math.log2(tail) + TOLERANCE
    else:
        assert result.tail == NEG_INF


def test_power_by_squaring_tracks_head_and_lumps_the_tail():
    base = cap(power(unit_series(2), 2, 5), 3)  # 1 + 8x + 8x^2 after the cap
    assert_upper(base.head, [1, 8, 8])
    cubed = power(base, 3, 3)
    exact = poly_power([1, 8, 8], 3)
    assert_upper(cubed.head, exact[:4])
    assert cubed.tail >= math.log2(sum(exact[4:]))
    assert cubed.tail <= math.log2(sum(exact[4:])) + TOLERANCE


def test_multiply_accounts_for_every_tail_combination():
    a = ErrorSeries(np.array([0.0, 1.0]), 2.0)  # 1 + 2x + 4 x^(>1)
    b = ErrorSeries(np.array([0.0, 0.0]), 1.0)  # 1 + x + 2 x^(>1)
    product = multiply(a, b, 1)
    assert_upper(product.head, [1, 3])
    # everything of degree > 1: 2*1 (x*x) + tails: 4*(1+1+2) + 2*(1+2) = 2 + 16 + 6
    assert product.tail >= math.log2(24)
    assert product.tail <= math.log2(24) + TOLERANCE
    assert_upper(power(empty_series(), 5, 3).head, [1])


def test_cap_leaves_the_empty_subset_alone():
    series = ErrorSeries(np.array([0.0, 10.0, 20.0]), 30.0)
    capped = cap(series, 12)
    assert capped.head.tolist() == [0.0, 10.0, 12.0]
    assert capped.tail == 12.0


@pytest.mark.parametrize(
    ("exponents", "coefficients", "copies", "limit"),
    [
        ((0, 3, 5), (1, 2, 3), 7, 12),  # Miller: (7 + 1) * 3 > 12
        ((0, 2), (2, 1), 50, 8),  # Miller with a non-unit constant term
        ((0, 1, 2), (1, 5, 7), 6, 40),  # repeated sparse multiplication
        ((0, 3, 5), (1, 2, 3), 2, 12),
        ((0, 4), (1, 9), 0, 6),
        ((0,), (3,), 4, 6),
        ((0, 9), (1, 1), 3, 6),  # terms beyond the limit vanish
    ],
)
def test_sparse_power_matches_exact_polynomial_powers(exponents, coefficients, copies, limit):
    dense = [0] * (max(exponents) + 1)
    for exponent, coefficient in zip(exponents, coefficients, strict=True):
        dense[exponent] = coefficient
    exact = poly_power(dense, copies)
    exact = (exact + [0] * (limit + 1))[: limit + 1]
    got = sparse_power(
        np.array(exponents), np.array([log2_int(c) for c in coefficients]), copies, limit
    )
    assert len(got) == limit + 1
    assert_upper(got, exact)


def test_sparse_power_requires_a_constant_term():
    with pytest.raises(ValueError, match="constant term"):
        sparse_power(np.array([1, 2]), np.array([0.0, 0.0]), 2, 5)


def test_sparse_multiply_is_a_truncated_product():
    dense = [1, 2, 3, 4]
    got = sparse_multiply(np.array([log2_int(v) for v in dense]), np.array([0, 2]), np.array([0.0, 1.0]))
    assert_upper(got, poly_multiply(dense, [1, 0, 2])[:4])
