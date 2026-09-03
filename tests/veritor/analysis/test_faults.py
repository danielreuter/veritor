"""``veritor.analysis.faults``: the price of ``f_max`` declared VUs.

``u(1) = log2(1 + n 2**W_V)`` is read off the kind table.  ``bound(...,
max_faults=f)`` charges the smallest of the rigorous bounds for a prover
that declares after the q-challenge: the fold at threshold ``eta (1 - s)**f``
(``s < 1``), the fold at ``eta / (1 + n)**f`` plus ``f u(1)`` (every ``s``),
and, at ``q = 1`` where the q-challenge reveals nothing, the base plus
``f u(1)``; all capped by the interface.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from veritor.analysis.bound import BoundOptions, bound, cut_bits
from veritor.analysis.faults import declared_bits, fault_allowance_bits, unit_fault_bits
from veritor.core import VerificationPolicy
from veritor.core.description import VERIFICATION

from .conftest import build_compiled, paper_example

ETA = Fraction(1, 2**40)
LOOSE = Fraction(1, 4)


def manual_unit_bits(compiled) -> float:
    rows = [row for row in compiled.kind_table().rows if row.role == VERIFICATION]
    units = sum(row.copies for row in rows)
    widest = max(
        cut_bits(row)
        for row in rows
        if row.size > row.source_inputs + row.source_weights
    )
    return math.log2(1 + units * 2.0**widest)


def vu_count(compiled) -> int:
    return sum(
        row.copies for row in compiled.kind_table().rows if row.role == VERIFICATION
    )


def rigorous_candidates(compiled, policy, eta, f) -> list[float]:
    """The bounds ``declared_bits`` takes the minimum of, written out."""

    n = vu_count(compiled)
    allowance = f * unit_fault_bits(compiled)
    candidates = [bound(compiled, policy, eta / (1 + n) ** f).bits + allowance]
    if policy.s < 1:
        candidates.append(bound(compiled, policy, eta * (1 - policy.s) ** f).bits)
    if policy.q == 1:
        candidates.append(bound(compiled, policy, eta).bits + allowance)
    return candidates


@pytest.mark.parametrize("sizes", [(1,), (3, 2), (4, 4, 4, 4)])
def test_unit_fault_bits_is_the_widest_cut_plus_the_log_of_the_unit_count(
    sizes,
) -> None:
    compiled = build_compiled(sizes)
    unit = unit_fault_bits(compiled)
    assert unit == pytest.approx(manual_unit_bits(compiled), rel=1e-12)
    # every VU is ``in, add`` with an 8-bit output word: W_V = 8, n = the number of VUs
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
    with pytest.raises(ValueError):
        declared_bits(
            compiled.kind_table(),
            VerificationPolicy(1, 1),
            ETA,
            BoundOptions(),
            -1,
            0.0,
        )


def test_full_verification_pays_exactly_the_allowance_until_the_interface_caps_it() -> (
    None
):
    compiled = build_compiled((4, 4, 4, 4))
    unit = unit_fault_bits(compiled)
    full = VerificationPolicy(1, 1)
    base = bound(compiled, full, ETA)
    assert base.bits == 0.0 and not base.capped
    for f in range(
        1, 4
    ):  # q = 1: the q-challenge reveals nothing, so a declaration is one VU
        widened = bound(compiled, full, ETA, max_faults=f)
        expected = min(f * unit, float(base.out_bits))
        assert widened.bits == pytest.approx(expected, abs=1e-9)
        assert widened.capped == (expected >= base.out_bits)
    saturated = bound(compiled, full, ETA, max_faults=1000)
    assert saturated.bits == float(base.out_bits) and saturated.capped
    assert bound(compiled, full, ETA, max_faults=0) == base


@pytest.mark.parametrize(
    "policy",
    [
        VerificationPolicy(1, Fraction(1, 2)),  # every RU replayed, half the VUs proved
        VerificationPolicy(
            Fraction(1, 2), Fraction(1, 2)
        ),  # the q-challenge is informative
        VerificationPolicy(Fraction(1, 2), 1),  # ... and every sampled VU is checked
    ],
)
def test_declared_capacity_is_the_smallest_rigorous_bound(
    policy: VerificationPolicy,
) -> None:
    compiled = build_compiled((4, 4, 4, 4))
    base = bound(compiled, policy, LOOSE)
    assert 0 < base.bits < base.out_bits
    previous = base.bits
    for f in (1, 2):
        result = bound(compiled, policy, LOOSE, max_faults=f)
        expected = min(
            min(rigorous_candidates(compiled, policy, LOOSE, f)), float(base.out_bits)
        )
        assert result.bits == pytest.approx(expected, abs=1e-9)
        assert result.eta == LOOSE and result.policy == policy
        assert result.bits >= previous  # more declarations never certify less
        previous = result.bits
    assert bound(compiled, policy, LOOSE, max_faults=0) == base


def test_declaring_after_the_q_challenge_costs_more_than_a_fixed_declaration() -> None:
    """At ``q < 1`` the adaptive prover pardons whichever opened error it likes:
    the charge exceeds ``f u(1)`` (the fixed-declaration price) until the cap."""

    compiled = build_compiled((4, 4, 4, 4))
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
    base = bound(compiled, policy, LOOSE).bits
    unit = unit_fault_bits(compiled)
    charged = bound(compiled, policy, LOOSE, max_faults=1)
    assert charged.bits > base + unit or charged.capped
    assert (
        declared_bits(compiled.kind_table(), policy, LOOSE, BoundOptions(), 0, 2.5)
        == 2.5
    )
