from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

from circuit_cut_analysis.circuit import CircuitDAG, Gate
from circuit_cut_analysis.class_sampling import (
    ClassQuotaOutcome,
    ClassSymmetricStrategy,
    GateClassPartition,
    GateProbabilityClass,
    enumerate_class_quotas,
    optimize_class_sampling_finite,
    quota_miss_probability,
)
from circuit_cut_analysis.sampling import (
    SamplingBudgetMode,
    VerificationPartition,
    optimize_sampling_strategy_finite,
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


def test_quota_detection_matches_concrete_subset_enumeration() -> None:
    class_sizes = (3, 2)
    checked = (1, 1)
    attacked = (2, 1)

    miss = quota_miss_probability(class_sizes, checked, attacked)

    class_zero = tuple(range(3))
    class_one = tuple(range(3, 5))
    attacked_gates = {0, 1, 3}
    concrete_samples = [
        set(left).union(right)
        for left in itertools.combinations(class_zero, checked[0])
        for right in itertools.combinations(class_one, checked[1])
    ]
    brute_miss = Fraction(
        sum(not sample.intersection(attacked_gates) for sample in concrete_samples),
        len(concrete_samples),
    )
    assert miss == brute_miss == Fraction(1, 6)


def test_mixture_over_quotas_has_exact_joint_detection() -> None:
    partition = GateClassPartition(
        (
            GateProbabilityClass("left", ("a0", "a1")),
            GateProbabilityClass("right", ("b0", "b1")),
        ),
        ("a0", "a1", "b0", "b1"),
    )
    strategy = ClassSymmetricStrategy(
        partition,
        2,
        (
            ClassQuotaOutcome((2, 0), Fraction(1, 2)),
            ClassQuotaOutcome((0, 2), Fraction(1, 2)),
        ),
    )

    assert strategy.class_inclusion_probabilities == {
        "left": Fraction(1, 2),
        "right": Fraction(1, 2),
    }
    assert strategy.detection_probability(("a0",), partition) == Fraction(1, 2)
    assert strategy.detection_probability(("b0",), partition) == Fraction(1, 2)
    assert strategy.detection_probability(("a0", "b0"), partition) == 1


def test_quota_enumeration_is_bounded_and_exact_budget() -> None:
    quotas = enumerate_class_quotas((2, 3, 1), 3)
    expected = {
        counts
        for counts in itertools.product(range(3), range(4), range(2))
        if sum(counts) == 3
    }
    assert set(quotas) == expected
    assert all(sum(quota) == 3 for quota in quotas)

    with pytest.raises(ValueError, match="more than 2"):
        enumerate_class_quotas((2, 3, 1), 3, max_actions=2)


def test_finite_optimizer_weights_scalar_classes_nonuniformly() -> None:
    circuit = _two_branch_circuit()
    partition = GateClassPartition.singleton_gates(("a", "b"))

    result = optimize_class_sampling_finite(
        circuit,
        partition,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert result.replay_certified_log2_reachable_bound.integral_width_bits == 1
    assert result.worst_attack is not None
    assert result.worst_attack.error_gates == frozenset({"a"})
    probabilities = result.strategy.class_inclusion_probabilities
    assert probabilities["b"] >= Fraction(3, 4)
    assert probabilities["a"] <= Fraction(1, 4)
    assert all(sum(outcome.checked_counts) == 1 for outcome in result.strategy.outcomes)


def test_actual_gate_locations_remain_adversarial_inside_one_class() -> None:
    circuit = _two_branch_circuit()
    coarse_class = GateClassPartition(
        (GateProbabilityClass("both", ("a", "b")),),
        ("a", "b"),
    )

    result = optimize_class_sampling_finite(
        circuit,
        coarse_class,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert result.worst_attack is not None
    assert result.worst_attack.error_gates == frozenset({"b"})
    assert result.worst_attack.detection_probability == Fraction(1, 2)
    assert result.replay_certified_log2_reachable_bound.integral_width_bits == 3


def test_refining_a_probability_class_can_strictly_improve_the_game() -> None:
    circuit = _two_branch_circuit()
    coarse = GateClassPartition(
        (GateProbabilityClass("both", ("a", "b")),),
        ("a", "b"),
    )
    fine = GateClassPartition.singleton_gates(("a", "b"))

    coarse_result = optimize_class_sampling_finite(
        circuit,
        coarse,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )
    fine_result = optimize_class_sampling_finite(
        circuit,
        fine,
        checked_gate_budget=1,
        detection_threshold=Fraction(3, 4),
    )

    assert coarse_result.replay_certified_log2_reachable_bound.integral_width_bits == 3
    assert fine_result.replay_certified_log2_reachable_bound.integral_width_bits == 1


def test_checking_every_gate_leaves_no_subthreshold_attack() -> None:
    circuit = _two_branch_circuit()
    partition = GateClassPartition(
        (GateProbabilityClass("both", ("a", "b")),),
        ("a", "b"),
    )

    result = optimize_class_sampling_finite(
        circuit,
        partition,
        checked_gate_budget=2,
        detection_threshold=Fraction(99, 100),
    )

    assert result.worst_attack is None
    assert result.replay_certified_log2_reachable_bound.is_zero


def test_singleton_class_game_matches_explicit_hard_budget_games() -> None:
    """Singleton classes represent every exactly-B scalar sampling strategy."""

    rng = random.Random(20260829)
    for _ in range(24):
        gate_count = rng.randint(2, 5)
        computed = tuple(f"g{index}" for index in range(gate_count))
        gates = [Gate("input", 1, op="input")]
        gates.extend(Gate(gate_id, rng.randint(1, 4)) for gate_id in computed)
        edges = [("input", computed[0])]
        edges.extend(
            (computed[index - 1], computed[index]) for index in range(1, gate_count)
        )
        edges.extend(
            (computed[source], computed[target])
            for source in range(gate_count)
            for target in range(source + 2, gate_count)
            if rng.random() < 0.35
        )
        circuit = CircuitDAG(gates, edges, (computed[-1],))
        budget = rng.randint(0, gate_count)
        threshold = rng.choice((Fraction(1, 2), Fraction(3, 4)))

        class_result = optimize_class_sampling_finite(
            circuit,
            GateClassPartition.singleton_gates(computed),
            checked_gate_budget=budget,
            detection_threshold=threshold,
        )
        explicit_result = optimize_sampling_strategy_finite(
            circuit,
            VerificationPartition.singleton_gates(computed),
            checked_gate_budget=budget,
            detection_threshold=threshold,
            budget_mode=SamplingBudgetMode.HARD,
        )

        assert (
            class_result.replay_certified_log2_reachable_bound
            == explicit_result.replay_certified_log2_reachable_bound
        )
