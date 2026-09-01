from __future__ import annotations

from fractions import Fraction

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.capacity_oracle import (
    ExplicitCircuitCapacityOracle,
    StructuralCapacityEvaluation,
)
from circuit_cut_analysis.circuit import CircuitDAG, Gate, GateId
from circuit_cut_analysis.sampling import (
    SamplingBudgetMode,
    SamplingOptimizationStatus,
    SamplingOutcome,
    SamplingStrategy,
    VerificationPartition,
    VerificationUnit,
    adversarial_best_response,
    independent_unit_sampling,
    optimize_sampling_strategy_finite,
    optimize_sampling_strategy_robust,
    uniform_fixed_count_sampling,
)


def _two_branch_circuit() -> CircuitDAG:
    return CircuitDAG(
        gates=(
            Gate("x", 1, op="input"),
            Gate("a", 1),
            Gate("y1", 1),
            Gate("b", 3),
            Gate("y2", 3),
        ),
        edges=(
            ("x", "a"),
            ("a", "y1"),
            ("x", "b"),
            ("b", "y2"),
        ),
        outputs=("y1", "y2"),
    )


def test_verification_partition_requires_an_exact_disjoint_cover() -> None:
    with pytest.raises(ValueError, match="both"):
        VerificationPartition(
            (
                VerificationUnit("left", ("a",)),
                VerificationUnit("right", ("a", "b")),
            ),
            ("a", "b"),
        )

    with pytest.raises(ValueError, match="missing"):
        VerificationPartition((VerificationUnit("left", ("a",)),), ("a", "b"))


def test_independent_sampling_has_exact_detection_and_expected_cost() -> None:
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = independent_unit_sampling(
        {"a": Fraction(1, 2), "b": Fraction(1, 4)},
        partition,
    )

    assert len(strategy.outcomes) == 4
    assert strategy.detection_probability(("a",), partition) == Fraction(1, 2)
    assert strategy.detection_probability(("b",), partition) == Fraction(1, 4)
    assert strategy.detection_probability(("a", "b"), partition) == Fraction(5, 8)
    assert strategy.expected_checked_gates(partition) == Fraction(3, 4)
    assert strategy.expected_checked_fraction(partition) == Fraction(3, 8)


def test_exact_adversary_uses_joint_structural_cut() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = independent_unit_sampling(
        {"a": Fraction(1, 2), "b": Fraction(1, 4)},
        partition,
    )

    attack = adversarial_best_response(circuit, partition, strategy)

    assert attack is not None
    assert attack.attacked_units == frozenset({"a", "b"})
    assert attack.detection_probability == Fraction(5, 8)
    assert attack.log2_reachable_bound.integral_width_bits == 4
    assert attack.cut_result.cut == frozenset({"y1", "y2"})


def test_detection_threshold_can_exclude_the_strongest_joint_attack() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = independent_unit_sampling(
        {"a": Fraction(1, 2), "b": Fraction(1, 4)},
        partition,
    )

    attack = adversarial_best_response(
        circuit,
        partition,
        strategy,
        detection_threshold=Fraction(3, 5),
    )

    assert attack is not None
    assert attack.attacked_units == frozenset({"b"})
    assert attack.log2_reachable_bound.integral_width_bits == 3


def test_uniform_fixed_count_sampling_models_correlated_checks() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = uniform_fixed_count_sampling(partition, 1)

    assert strategy.expected_checked_gates(partition) == 1
    assert strategy.detection_probability(("a", "b"), partition) == 1
    attack = adversarial_best_response(circuit, partition, strategy)
    assert attack is not None
    assert attack.attacked_units == frozenset({"b"})
    assert attack.log2_reachable_bound.integral_width_bits == 3


def test_same_marginals_can_have_different_joint_detection() -> None:
    partition = VerificationPartition.singleton_gates(("a", "b"))
    independent = independent_unit_sampling(
        {"a": Fraction(1, 2), "b": Fraction(1, 2)},
        partition,
    )
    fixed_count = uniform_fixed_count_sampling(partition, 1)

    for unit_id in ("a", "b"):
        assert independent.detection_probability((unit_id,), partition) == Fraction(
            1, 2
        )
        assert fixed_count.detection_probability((unit_id,), partition) == Fraction(
            1, 2
        )
    assert independent.detection_probability(("a", "b"), partition) == Fraction(3, 4)
    assert fixed_count.detection_probability(("a", "b"), partition) == 1


def test_attack_at_detection_threshold_is_not_admissible() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = independent_unit_sampling(
        {"a": Fraction(3, 4), "b": Fraction(1, 4)},
        partition,
    )

    attack = adversarial_best_response(
        circuit,
        partition,
        strategy,
        detection_threshold=Fraction(3, 4),
    )

    assert attack is not None
    assert attack.attacked_units == frozenset({"b"})
    assert attack.detection_probability == Fraction(1, 4)


def test_coarse_unit_attack_corrupts_every_gate_in_the_unit() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition(
        (VerificationUnit("both", ("a", "b")),),
        ("a", "b"),
    )
    strategy = independent_unit_sampling({"both": Fraction(1, 2)}, partition)

    attack = adversarial_best_response(circuit, partition, strategy)

    assert attack is not None
    assert attack.error_gates == frozenset({"a", "b"})
    assert attack.log2_reachable_bound.integral_width_bits == 4


def test_always_checking_every_unit_leaves_no_subthreshold_attack() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = SamplingStrategy((SamplingOutcome(("a", "b"), 1),))

    assert adversarial_best_response(circuit, partition, strategy) is None


def test_attackable_and_checkable_gate_sets_are_separate() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a",))
    strategy = SamplingStrategy((SamplingOutcome(("a",), 1),))

    attack = adversarial_best_response(
        circuit,
        partition,
        strategy,
        attackable_gates=("a", "b"),
    )

    assert attack is not None
    assert attack.error_gates == frozenset({"b"})
    assert attack.detection_probability == 0
    assert attack.log2_reachable_bound.integral_width_bits == 3


def test_trusted_checkable_gate_is_excluded_from_attack_support() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    strategy = SamplingStrategy((SamplingOutcome((), 1),))

    attack = adversarial_best_response(
        circuit,
        partition,
        strategy,
        attackable_gates=("a",),
    )

    assert attack is not None
    assert attack.error_gates == frozenset({"a"})
    assert attack.log2_reachable_bound.integral_width_bits == 1


def test_finite_optimizer_finds_nonuniform_correlated_strategy() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))

    result = optimize_sampling_strategy_finite(
        circuit,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert result.replay_certified_log2_reachable_bound.integral_width_bits == 1
    assert result.status is SamplingOptimizationStatus.NUMERICAL_LP_REPLAY_CERTIFIED
    assert not result.global_optimality_certified
    assert result.strategy.expected_checked_gates(partition) <= 1
    assert result.worst_attack is not None
    assert result.worst_attack.attacked_units == frozenset({"a"})
    assert result.worst_attack.log2_reachable_bound.integral_width_bits == 1
    assert result.candidate_sampling_action_count == 4
    assert result.adversarial_action_count == 3


def test_zero_budget_leaves_the_full_joint_attack() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))

    result = optimize_sampling_strategy_finite(
        circuit,
        partition,
        checked_gate_budget=0,
        detection_threshold=Fraction(3, 4),
    )

    assert result.strategy.expected_checked_gates(partition) == 0
    assert result.worst_attack is not None
    assert result.worst_attack.attacked_units == frozenset({"a", "b"})
    assert result.replay_certified_log2_reachable_bound.integral_width_bits == 4


def test_finite_optimizer_accounts_for_never_checkable_attack_gates() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a",))

    result = optimize_sampling_strategy_finite(
        circuit,
        partition,
        attackable_gates=("a", "b"),
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert result.worst_attack is not None
    assert result.worst_attack.error_gates == frozenset({"b"})
    assert result.worst_attack.detection_probability == 0
    assert result.replay_certified_log2_reachable_bound.integral_width_bits == 3
    assert result.adversarial_action_count == 3


def test_fine_partition_strictly_dominates_a_coarse_unit_at_same_budget() -> None:
    circuit = _two_branch_circuit()
    fine = VerificationPartition.singleton_gates(("a", "b"))
    coarse = VerificationPartition(
        (VerificationUnit("both", ("a", "b")),),
        ("a", "b"),
    )

    fine_result = optimize_sampling_strategy_finite(
        circuit,
        fine,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )
    coarse_result = optimize_sampling_strategy_finite(
        circuit,
        coarse,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert fine_result.replay_certified_log2_reachable_bound.integral_width_bits == 1
    assert coarse_result.replay_certified_log2_reachable_bound.integral_width_bits == 4


def test_robust_optimizer_matches_exact_optimizer_with_exact_evaluations() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    oracle = ExplicitCircuitCapacityOracle(circuit)

    exact = optimize_sampling_strategy_finite(
        circuit,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )
    robust = optimize_sampling_strategy_robust(
        oracle.evaluate,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert robust.all_scenarios_exact
    assert (
        robust.certified_upper_log2_reachable_bound
        == exact.replay_certified_log2_reachable_bound
    )
    assert robust.residual_exact_lower_bound == (
        robust.certified_upper_log2_reachable_bound
    )
    assert robust.worst_attack is not None
    assert robust.worst_attack.attacked_units == frozenset({"a"})


def test_robust_optimizer_is_sound_but_conservative_under_inflated_uppers() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))
    oracle = ExplicitCircuitCapacityOracle(circuit)

    def inflated(
        gates: frozenset[GateId],
    ) -> StructuralCapacityEvaluation[frozenset[GateId]]:
        evaluation = oracle.evaluate(gates)
        return StructuralCapacityEvaluation(
            lower_bound=evaluation.lower_bound,
            upper_bound=evaluation.upper_bound + LogCardinality.bits(2),
            requested_support=evaluation.requested_support,
            evaluated_support=evaluation.evaluated_support,
            cut_gate_ids=None,
            method="inflated-interval",
        )

    robust = optimize_sampling_strategy_robust(
        inflated,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert robust.bounded_scenario_count == 3
    assert robust.worst_attack is not None
    true_worst = robust.worst_attack.evaluation.lower_bound
    assert robust.certified_upper_log2_reachable_bound >= true_worst
    assert robust.residual_exact_lower_bound <= (
        robust.certified_upper_log2_reachable_bound
    )


def test_robust_optimizer_accounts_for_unchecked_attackable_gates() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a",))
    oracle = ExplicitCircuitCapacityOracle(circuit)

    robust = optimize_sampling_strategy_robust(
        oracle.evaluate,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
        unchecked_attackable_gates=("b",),
    )

    assert robust.worst_attack is not None
    assert "b" in robust.worst_attack.error_gates
    assert robust.worst_attack.detection_probability == 0
    assert robust.certified_upper_log2_reachable_bound.integral_width_bits == 3


def test_hard_budget_removes_over_budget_sampling_actions() -> None:
    circuit = _two_branch_circuit()
    partition = VerificationPartition.singleton_gates(("a", "b"))

    result = optimize_sampling_strategy_finite(
        circuit,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
        budget_mode=SamplingBudgetMode.HARD,
    )

    assert result.candidate_sampling_action_count == 3
    assert all(len(outcome.sampled_units) <= 1 for outcome in result.strategy.outcomes)
