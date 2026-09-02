from __future__ import annotations

import random
from fractions import Fraction

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis import (
    BoundClaimStrength,
    CapacityEvidence,
    CircuitCapacityOracle,
    TerminationStatus,
    VerificationUnitCapacityOracle,
    branch_and_bound_finite_bound,
    exhaustive_finite_bound,
)
from veritor.compile import Compiler, Tracer
from veritor.core import VerificationPolicy, make_word_gate_set


def test_branch_and_bound_matches_reference_on_random_tiny_games(
    make_index,
    exact_oracle_type,
):
    rng = random.Random(20260831)
    for _case in range(30):
        replay_sizes = tuple(rng.randint(1, 3) for _ in range(rng.randint(1, 3)))
        index = make_index(replay_sizes)
        weights = tuple(rng.randint(1, 5) for _ in range(index.verification_unit_count))
        policy = VerificationPolicy(
            Fraction(rng.randint(0, 4), 4),
            Fraction(rng.randint(0, 4), 4),
            Fraction(rng.randint(0, 3), 4),
        )
        oracle = exact_oracle_type(weights, frontier=12)

        exhaustive = exhaustive_finite_bound(index, policy, oracle)
        branch = branch_and_bound_finite_bound(index, policy, oracle)

        assert branch.lower_bound == exhaustive.lower_bound
        assert branch.upper_bound == exhaustive.upper_bound
        assert branch.witness.capacity_lower_bound == branch.lower_bound
        assert branch.claim_strength is BoundClaimStrength.EXACT
        assert branch.termination_status is TerminationStatus.COMPLETE


def test_resource_limit_returns_bracket_containing_reference_optimum(
    make_index,
    exact_oracle_type,
):
    index = make_index((3, 3))
    oracle = exact_oracle_type((1, 2, 3, 4, 5, 6), frontier=20)
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 4))
    exact = exhaustive_finite_bound(index, policy, oracle)

    limited = branch_and_bound_finite_bound(
        index,
        policy,
        oracle,
        max_states=1,
        max_capacity_queries=1,
    )

    assert limited.termination_status is TerminationStatus.RESOURCE_LIMIT
    assert limited.claim_strength is BoundClaimStrength.CERTIFIED_BRACKET
    assert limited.lower_bound <= exact.lower_bound <= limited.upper_bound
    assert limited.capacity_query_count == 1
    assert limited.state_count == 1


def test_strict_eta_equality_excludes_attack_from_finite_bound(
    make_index,
    exact_oracle_type,
):
    index = make_index((1,))
    oracle = exact_oracle_type((7,))
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 2))

    result = exhaustive_finite_bound(index, policy, oracle)

    assert result.exact_capacity == LogCardinality.zero()
    assert result.feasible_state_count == 1
    assert result.witness.error_units == ()


def test_all_feasible_and_capacity_dominated_subtrees_are_pruned(
    make_index,
    exact_oracle_type,
):
    index = make_index((2, 2))
    oracle = exact_oracle_type((1, 1, 1, 1), frontier=2)

    all_feasible = branch_and_bound_finite_bound(
        index, VerificationPolicy(0, 1, Fraction(9, 10)), oracle
    )
    dominated = branch_and_bound_finite_bound(
        index, VerificationPolicy(1, Fraction(1, 2), Fraction(1, 4)), oracle
    )

    assert all_feasible.pruned_all_feasible_count == 1
    assert all_feasible.state_count == 1
    assert dominated.pruned_capacity_dominated_count > 0


def test_bound_binds_policy_tuple_and_propagates_oracle_assumptions(
    make_index,
    exact_oracle_type,
):
    index = make_index((1, 1))
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 3), Fraction(1, 4))
    oracle = exact_oracle_type((2, 3), assumptions=("declared structural model",))

    result = exhaustive_finite_bound(
        index, policy, oracle, assumptions=("fixed public execution",)
    )

    assert result.policy_identity == policy.digest
    assert result.identities.index_identity == index.digest
    assert result.identities.index_identity != make_index((2, 1)).digest
    assert result.assumptions == (
        "fixed public execution",
        "declared structural model",
    )
    assert result.claim_strength is BoundClaimStrength.CONDITIONAL


def test_circuit_capacity_oracle_composes_with_verification_units() -> None:
    """A chain of two one-gate verification units inside one replay unit.

    Attacking both gates still only reaches the output through the second
    gate, so the exact capacity is one 8-bit value.
    """

    gate_set = make_word_gate_set(8)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key="double", role="verification")
    def double(v):
        return add(v[0], v[0])

    @tracer.definition(input_count=1, key="chain", role="replay")
    def chain(v):
        return double(double(v[0]))

    @tracer.definition(input_count=1, key="root")
    def root(v):
        return chain(v[0])

    compiled = Compiler(gate_set).compile(tracer.serialize(root), [1])
    unit_oracle = VerificationUnitCapacityOracle(
        CircuitCapacityOracle(compiled.circuit), compiled.index
    )

    result = branch_and_bound_finite_bound(
        compiled.index, VerificationPolicy(0, 1, 0), unit_oracle
    )

    assert compiled.index.verification_unit_count == 2
    assert result.exact_capacity == LogCardinality.bits(8)
    assert result.upper_witness is not None
    assert result.upper_witness.error_units == (0, 1)


def test_certified_interval_oracle_produces_bracket_not_exact(make_index):
    index = make_index((1,))

    class IntervalOracle:
        def evaluate(self, support):
            return CapacityEvidence(
                lower_bound=LogCardinality.bits(2),
                upper_bound=LogCardinality.bits(5),
                requested_support=support,
                evaluated_support=support,
                method="test-interval",
            )

    result = exhaustive_finite_bound(index, VerificationPolicy(0, 1, 0), IntervalOracle())

    assert result.lower_bound == LogCardinality.bits(2)
    assert result.upper_bound == LogCardinality.bits(5)
    assert not result.is_exact
    assert result.claim_strength is BoundClaimStrength.CERTIFIED_BRACKET
