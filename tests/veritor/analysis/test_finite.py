from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis import (
    BoundClaimStrength,
    CapacityEvidence,
    StructuralCircuitCapacityOracle,
    TerminationStatus,
    VerificationUnitCapacityOracle,
    branch_and_bound_finite_bound,
    exhaustive_finite_bound,
)
from veritor.core import Port, StructuralGate, VerificationPolicy


def test_branch_and_bound_matches_reference_on_random_tiny_games(
    make_partitions,
    exact_oracle_type,
):
    rng = random.Random(20260831)
    for case in range(30):
        replay_sizes = tuple(rng.randint(1, 3) for _ in range(rng.randint(1, 3)))
        replay, verification = make_partitions(
            replay_sizes,
            label=f"random-{case}",
        )
        weights = tuple(
            rng.randint(1, 5) for _ in range(verification.unit_count)
        )
        policy = VerificationPolicy(
            Fraction(rng.randint(0, 4), 4),
            Fraction(rng.randint(0, 4), 4),
            Fraction(rng.randint(0, 3), 4),
        )
        oracle = exact_oracle_type(weights, frontier=12)

        exhaustive = exhaustive_finite_bound(
            replay,
            verification,
            policy,
            oracle,
        )
        branch = branch_and_bound_finite_bound(
            replay,
            verification,
            policy,
            oracle,
        )

        assert branch.lower_bound == exhaustive.lower_bound
        assert branch.upper_bound == exhaustive.upper_bound
        assert branch.witness.capacity_lower_bound == branch.lower_bound
        assert branch.claim_strength is BoundClaimStrength.EXACT
        assert branch.termination_status is TerminationStatus.COMPLETE


def test_resource_limit_returns_bracket_containing_reference_optimum(
    make_partitions,
    exact_oracle_type,
):
    replay, verification = make_partitions((3, 3), label="resource")
    oracle = exact_oracle_type((1, 2, 3, 4, 5, 6), frontier=20)
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 4))
    exact = exhaustive_finite_bound(replay, verification, policy, oracle)

    limited = branch_and_bound_finite_bound(
        replay,
        verification,
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
    make_partitions,
    exact_oracle_type,
):
    replay, verification = make_partitions((1,), label="strict")
    oracle = exact_oracle_type((7,))
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 2))

    result = exhaustive_finite_bound(replay, verification, policy, oracle)

    assert result.exact_capacity == LogCardinality.zero()
    assert result.feasible_state_count == 1
    assert result.witness.error_units == ()


def test_all_feasible_and_capacity_dominated_subtrees_are_pruned(
    make_partitions,
    exact_oracle_type,
):
    replay, verification = make_partitions((2, 2), label="pruning")
    oracle = exact_oracle_type((1, 1, 1, 1), frontier=2)

    all_feasible = branch_and_bound_finite_bound(
        replay,
        verification,
        VerificationPolicy(0, 1, Fraction(9, 10)),
        oracle,
    )
    dominated = branch_and_bound_finite_bound(
        replay,
        verification,
        VerificationPolicy(1, Fraction(1, 2), Fraction(1, 4)),
        oracle,
    )

    assert all_feasible.pruned_all_feasible_count == 1
    assert all_feasible.state_count == 1
    assert dominated.pruned_capacity_dominated_count > 0


def test_bound_binds_policy_tuple_and_propagates_oracle_assumptions(
    make_partitions,
    exact_oracle_type,
):
    replay, verification = make_partitions((1, 1), label="identity")
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 3), Fraction(1, 4))
    oracle = exact_oracle_type(
        (2, 3),
        assumptions=("declared structural model",),
    )

    result = exhaustive_finite_bound(
        replay,
        verification,
        policy,
        oracle,
        assumptions=("fixed public execution",),
    )

    assert result.policy_identity == policy.digest
    assert result.identities.replay_partition_identity == replay.identity.digest
    assert (
        result.identities.verification_partition_identity
        == verification.identity.digest
    )
    assert result.assumptions == (
        "fixed public execution",
        "declared structural model",
    )
    assert result.claim_strength is BoundClaimStrength.CONDITIONAL


@dataclass(frozen=True, slots=True)
class TinyStructuralCircuit:
    identity: object
    computed_positions: object
    input_ports: tuple[Port, ...]
    output_ports: tuple[Port, ...]
    gates: tuple[StructuralGate, ...]

    def gate_at(self, position):
        return self.gates[self.computed_positions.rank(position)]


def test_core_structural_circuit_adapter_keeps_partitions_separate(
    make_partitions,
):
    replay, verification = make_partitions((2,), label="adapter")
    circuit = TinyStructuralCircuit(
        identity=replay.structure_identity,
        computed_positions=replay.eligible_positions,
        input_ports=(Port("x", 0, "u8"),),
        output_ports=(Port("y", 11, "u8"),),
        gates=(
            StructuralGate(10, "copy", (0,), 256),
            StructuralGate(11, "copy", (10,), 256),
        ),
    )
    position_oracle = StructuralCircuitCapacityOracle(circuit)
    unit_oracle = VerificationUnitCapacityOracle(
        position_oracle,
        verification,
    )
    policy = VerificationPolicy(0, 1, 0)

    result = branch_and_bound_finite_bound(
        replay,
        verification,
        policy,
        unit_oracle,
    )

    assert result.exact_capacity == LogCardinality.bits(8)
    assert result.upper_witness is not None
    assert result.upper_witness.error_units == (0, 1)


def test_certified_interval_oracle_produces_bracket_not_exact(
    make_partitions,
):
    replay, verification = make_partitions((1,), label="interval")

    class IntervalOracle:
        def evaluate(self, support):
            return CapacityEvidence(
                lower_bound=LogCardinality.bits(2),
                upper_bound=LogCardinality.bits(5),
                requested_support=support,
                evaluated_support=support,
                method="test-interval",
            )

    result = exhaustive_finite_bound(
        replay,
        verification,
        VerificationPolicy(0, 1, 0),
        IntervalOracle(),
    )

    assert result.lower_bound == LogCardinality.bits(2)
    assert result.upper_bound == LogCardinality.bits(5)
    assert not result.is_exact
    assert result.claim_strength is BoundClaimStrength.CERTIFIED_BRACKET
