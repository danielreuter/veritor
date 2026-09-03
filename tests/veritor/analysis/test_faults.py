"""``veritor.analysis.faults``: the price of ``f_max`` declared VUs.

``u(1) = log2(1 + |S| 2**W_V)`` is read off the kind table; ``bound(...,
max_faults=f)`` is the base bound plus ``f u(1)``, capped by the interface;
the adaptive variant lowers the threshold by ``(1 - s)**f`` and is what a
prover choosing its declarations after the q-challenge is actually held to.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from veritor.analysis.bound import bound, cut_bits
from veritor.analysis.faults import (
    adaptive_fault_allowance_bits,
    adaptive_fault_bound,
    fault_allowance_bits,
    unit_fault_bits,
)
from veritor.core import VerificationPolicy
from veritor.core.description import VERIFICATION

from .conftest import build_compiled, paper_example

ETA = Fraction(1, 2**40)
LOOSE = Fraction(1, 4)


def manual_unit_bits(compiled) -> float:
    rows = [row for row in compiled.kind_table().rows if row.role == VERIFICATION]
    units = sum(row.copies for row in rows)
    widest = max(
        cut_bits(row) for row in rows if row.size > row.source_inputs + row.source_weights
    )
    return math.log2(1 + units * 2.0**widest)


@pytest.mark.parametrize("sizes", [(1,), (3, 2), (4, 4, 4, 4)])
def test_unit_fault_bits_is_the_widest_cut_plus_the_log_of_the_unit_count(sizes) -> None:
    compiled = build_compiled(sizes)
    unit = unit_fault_bits(compiled)
    assert unit == pytest.approx(manual_unit_bits(compiled), rel=1e-12)
    # every VU is ``in, add`` with an 8-bit output word: W_V = 8, |S| = the number of VUs
    assert unit == pytest.approx(8 + math.log2(sum(sizes) + 2.0**-8), rel=1e-12)
    assert unit_fault_bits(compiled.kind_table()) == unit


def test_the_paper_example_prices_its_widest_unit() -> None:
    compiled = paper_example(width=2)
    unit = unit_fault_bits(compiled)
    assert unit == pytest.approx(manual_unit_bits(compiled), rel=1e-12)
    assert unit > 0


def test_fault_allowance_is_linear_and_validated() -> None:
    compiled = build_compiled((3, 2))
    unit = unit_fault_bits(compiled)
    assert fault_allowance_bits(compiled, 0) == 0.0
    for f in (1, 2, 7):
        assert fault_allowance_bits(compiled, f) == f * unit
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError):
            fault_allowance_bits(compiled, bad)  # type: ignore[arg-type]


def test_bound_grows_by_the_allowance_until_the_interface_caps_it() -> None:
    compiled = build_compiled((4, 4, 4, 4))
    unit = unit_fault_bits(compiled)
    full = VerificationPolicy(1, 1)
    base = bound(compiled, full, ETA)
    assert base.bits == 0.0 and not base.capped
    for f in range(1, 4):
        widened = bound(compiled, full, ETA, max_faults=f)
        expected = min(f * unit, float(base.out_bits))
        assert widened.bits == pytest.approx(expected, abs=1e-9)
        assert widened.capped == (expected >= base.out_bits)
    saturated = bound(compiled, full, ETA, max_faults=1000)
    assert saturated.bits == float(base.out_bits) and saturated.capped
    assert bound(compiled, full, ETA, max_faults=0) == base

    partial = VerificationPolicy(1, Fraction(1, 2))
    loose = bound(compiled, partial, LOOSE)
    assert 0 < loose.bits < loose.out_bits
    assert bound(compiled, partial, LOOSE, max_faults=1).bits == pytest.approx(
        min(loose.bits + unit, float(loose.out_bits)), abs=1e-9
    )


def test_adaptive_bound_lowers_the_threshold_when_the_q_challenge_is_informative() -> None:
    compiled = build_compiled((4, 4, 4, 4))
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
    for f in (1, 2):
        adaptive = adaptive_fault_bound(compiled, policy, LOOSE, f)
        lowered = LOOSE * Fraction(1, 2) ** f
        assert adaptive == bound(compiled, policy, lowered)
        assert adaptive.eta == lowered
        assert adaptive.bits >= bound(compiled, policy, LOOSE).bits
    assert adaptive_fault_bound(compiled, policy, LOOSE, 0) == bound(compiled, policy, LOOSE)
    assert adaptive_fault_allowance_bits(compiled, policy, LOOSE, 1) == (
        adaptive_fault_bound(compiled, policy, LOOSE, 1).bits - bound(compiled, policy, LOOSE).bits
    )


def test_adaptive_bound_is_the_specified_charge_or_better_when_every_ru_is_replayed() -> None:
    compiled = build_compiled((4, 4, 4, 4))
    everything = VerificationPolicy(1, 1)
    assert adaptive_fault_bound(compiled, everything, ETA, 2) == bound(
        compiled, everything, ETA, max_faults=2
    )
    half = VerificationPolicy(1, Fraction(1, 2))
    adaptive = adaptive_fault_bound(compiled, half, LOOSE, 1)
    specified = bound(compiled, half, LOOSE, max_faults=1)
    lowered = bound(compiled, half, LOOSE * Fraction(1, 2))
    assert adaptive.bits == min(specified.bits, lowered.bits)


def test_adaptive_bound_is_the_trivial_cap_when_every_sampled_unit_is_checked() -> None:
    compiled = build_compiled((4, 4, 4, 4))
    policy = VerificationPolicy(Fraction(1, 2), 1)
    result = adaptive_fault_bound(compiled, policy, ETA, 1)
    assert result.bits == float(result.out_bits) and result.capped
    with pytest.raises(ValueError):
        adaptive_fault_bound(compiled, policy, ETA, -1)
    with pytest.raises(TypeError):
        adaptive_fault_bound(compiled, (1, 1), ETA, 1)  # type: ignore[arg-type]
