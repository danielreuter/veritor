"""Component 5: soundness of the acceptance bound and of ``Bound`` (analysis/).

Claim A (P[accept | error set E] <= sigma(E)) is the sampling law, tested in
``test_sampling.py``; here Claim B: ``Bound(C, I, theta)`` at ``eta`` upper
bounds ``log2 |Y_eta|`` under the UNION definition, on random small circuits
under random markings (the client's choice of partition), with whole units
corrupted, with the integer-count tightening, and with the source-only rule.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from veritor.analysis import bound
from veritor.analysis.bound import _integer_count
from veritor.analysis.probability import survival
from veritor.analysis.reference import (
    accepted_outputs,
    admissible_sets,
    cover_bits,
    error_counts,
    out_bits,
    subset_sum_bits,
    transcript_outputs,
)
from veritor.core import VerificationPolicy

TOLERANCE = 1e-6
POLICIES = [  # (theta, eta): sampling, whole-unit corruption (s = 1), nothing, everything
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
    (VerificationPolicy(Fraction(1, 3), Fraction(1, 5)), Fraction(1, 100)),
    (VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 4)),
    (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    (VerificationPolicy(0, 1), Fraction(1, 2)),
    (VerificationPolicy(1, 1), Fraction(0)),
]


def relaxed(eta: Fraction, result, replay_units: int) -> Fraction:
    """The threshold the cost grid actually enforces (one step per replay unit)."""

    if eta == 0 or math.isinf(result.cost_step):
        return eta
    return eta * Fraction(math.exp(-replay_units * result.cost_step))


@pytest.mark.parametrize("seed", range(4))
def test_union_over_random_markings_is_below_the_fold(sec, seed):
    """For one gate graph under several client-chosen markings, log2 |Y_eta| <= bits."""

    for marking in range(3):
        compiled = sec.random_marked_compiled(seed, marking)
        inputs = list(range(1, compiled.index.input_count + 1))
        outputs = transcript_outputs(compiled, inputs)
        for policy, eta in POLICIES:
            union = len(accepted_outputs(outputs, policy, eta))
            result = bound(compiled, policy, eta)
            assert math.log2(union) <= result.bits + TOLERANCE, (seed, marking, policy, eta)
            assert result.bits <= result.out_bits
            # bits == 0.0 certifies a single output; the honest output is always reachable
            if result.bits == 0.0:
                assert union == 1
            assert union >= 1
            if policy.q == policy.s == 1 and eta == 0:
                assert result.bits == 0.0 and union == 1
            if policy.q == 0:  # nothing is checked: every output, and the fold says so
                assert union == 1 << result.out_bits and result.capped
            # every relaxation rounds toward admitting more: the fold never drops below
            # the exact union and never exceeds the per-set sum at the relaxed threshold
            per_set = subset_sum_bits(
                compiled, policy, relaxed(eta, result, compiled.index.replay_units.count)
            )
            assert result.bits <= per_set + TOLERANCE


def test_whole_unit_corruption_is_covered_by_the_unit_interface(sec):
    """With s = 1 every subset of a unit costs the same; the union stays below the fold."""

    policy, eta = VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 4)
    for seed in range(4):
        compiled = sec.random_marked_compiled(seed, 1)
        outputs = transcript_outputs(compiled, list(range(1, compiled.index.input_count + 1)))
        union = len(accepted_outputs(outputs, policy, eta))
        result = bound(compiled, policy, eta)
        assert result.errors_limit == 1  # one error already saturates the unit's cost
        assert math.log2(union) <= result.bits + TOLERANCE


def test_integer_count_never_undercounts_and_never_exceeds_its_input():
    """``floor(2**bits)`` is taken with the power scaled up; the result stays <= bits."""

    for count in range(1, 1 << 14):
        bits = math.log2(count)
        tightened = _integer_count(bits)
        assert tightened >= bits, count  # never below the log2 of an integer it was given
        assert tightened <= bits
    for count in range(1, 1 << 10):
        # ... nor below the exact log2 of the count when the input carries slack
        bits = math.log2(count) + 1e-9
        slack = _integer_count(bits)
        exact = Decimal(count).ln() / Decimal(2).ln()
        assert Decimal(slack) >= exact and slack <= bits, count
    for bits in (0.0, 1e-14, 0.5, 0.999999):
        assert _integer_count(bits) == 0.0  # at most one output
    assert _integer_count(1.0) == 1.0 and _integer_count(48.0) == 48.0
    assert _integer_count(53.0) == 53.0 and _integer_count(1000.0) == 1000.0  # unchanged past 2**53
    # the fold's slack is removed where visible: a fully checked run is exactly zero bits
    assert _integer_count(7.460698725481157e-14) == 0.0
    # a count between two integers rounds down to the integer below (never up); a power of two
    # is exact, any other count is rounded up by one ulp
    assert _integer_count(math.log2(1024) + 1e-9) == 10.0
    assert _integer_count(math.log2(1000) + 1e-9) == math.nextafter(math.log2(1000), math.inf)


def test_fully_checked_run_has_exactly_zero_capacity(model, sec):
    result = bound(model.compiled, sec.CHECK_EVERYTHING, 0)
    assert result.bits == 0.0 and not result.capped
    assert result.knapsack_bits > 0.0  # the raw fold carries upward-rounding slack


def unit_cover_sum(compiled, policy, eta, *, source_only_units: bool) -> float:
    """``log2 sum_E 2**(sum of the interfaces of the units of E)`` over admissible ``E``.

    This is the fold's own per-kind cover (each unit by its interface; the
    chain's stage interface equals the sum of its cells', so no parent cover
    is cheaper) before the final cap by the circuit's interface.  Without
    ``source_only_units`` the sum ranges over error sets naming only units
    holding a non-source gate.
    """

    circuit, index = compiled.circuit, compiled.index
    corruptible = {
        unit
        for unit in range(index.verification_unit_count)
        if any(not circuit[a].is_source for a in index.verification_unit(unit).interval)
    }
    total = 0
    for errors in admissible_sets(compiled, policy, eta):
        if not source_only_units and not errors <= corruptible:
            continue
        total += 1 << sum(out_bits(circuit, index.verification_unit(unit)) for unit in errors)
    return math.log2(total)


def test_source_only_units_contribute_no_error_terms(sec):
    """Kinds with no non-source gate enter the knapsack with ``l = 0`` only.

    The chain's source unit holds four one-gate source cells.  Counting their
    subsets would multiply every term by the number of admissible subsets of
    four units that can never be in error (up to 16).  The chain with whole
    cells: there a stage's interface is the sum of its cells', so the fold's
    per-kind cover is exactly ``unit_cover_sum`` (with split cells the stage
    covers a cell's two units more cheaply than they cover themselves).
    """

    model = sec.Model(2, 2, split_cells=False)
    compiled = model.compiled
    policy, eta = sec.HALVES, Fraction(1, 4)
    result = bound(compiled, policy, eta)
    assert result.errors_limit >= 2  # two cells per stage: nothing is lumped
    exact = unit_cover_sum(compiled, policy, eta, source_only_units=False)
    admitted = unit_cover_sum(
        compiled,
        policy,
        relaxed(eta, result, compiled.index.replay_units.count),
        source_only_units=False,
    )
    assert exact - TOLERANCE <= result.knapsack_bits <= admitted + TOLERANCE
    with_sources = unit_cover_sum(compiled, policy, eta, source_only_units=True)
    assert result.knapsack_bits < with_sources - 0.5  # the source cells' subsets are gone
    assert with_sources > exact + 1.0  # (the previous fold reproduced ``with_sources``)
    # an error set naming a source cell does survive sampling ...
    source_cell = frozenset({0})
    assert survival(policy, error_counts(compiled.index, source_cell)) > eta
    # ... but a transcript with a wrong input is rejected at the boundary with certainty and a
    # weight has no value but kappa_W's (test_local_checks), so no such transcript is in Y_eta
    assert cover_bits(compiled, source_cell) == 0
    assert result.bits <= result.out_bits
    # the per-set reference sum, which covers by the root as well, stays above ``bits``
    assert result.bits <= subset_sum_bits(compiled, policy, eta) + TOLERANCE


def test_source_only_rule_is_exact_against_the_enumerated_union(sec):
    """A 2-bit chain small enough to enumerate: the union is below the fold with the rule."""

    compiled = sec.chain_compiled(stages=1, cells=2, width=2)
    outputs = transcript_outputs(compiled, [1, 2], [3, 1])
    for policy, eta in POLICIES:
        union = len(accepted_outputs(outputs, policy, eta))
        result = bound(compiled, policy, eta)
        assert math.log2(union) <= result.bits + TOLERANCE
        if policy.q == policy.s == 1 and eta == 0:
            assert union == 1 and result.bits == 0.0
