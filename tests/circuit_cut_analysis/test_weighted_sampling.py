from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
from scipy.optimize import linprog

from circuit_cut_analysis.capacity import GateCapacity, LogCardinality
from circuit_cut_analysis.class_sampling import (
    enumerate_class_quotas,
    quota_miss_probability,
)
from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.weighted_sampling import (
    WeightedGateClass,
    WeightedGateClassPartition,
    _capacity_from_width,
    _scalable_miss_probability,
    _upper_capacity_from_width,
    capacity_upper_bound_for_counts,
    coalesce_equal_capacity_classes,
    coalesce_frontier_equivalent_classes,
    equalized_log_quota_strategy,
    fixed_quota_strategy,
    optimize_fixed_quota_strategy,
    pure_class_moment_lower_bound,
    stratified_quota_best_response,
    uniform_quota_strategy,
    universal_minimax_lower_bound,
    weighted_partition_from_capacity_profile,
)


def _partition(
    classes: tuple[WeightedGateClass, ...],
    *,
    frontier_bits: int = 20,
) -> WeightedGateClassPartition:
    return WeightedGateClassPartition(
        model_id="toy",
        classes=classes,
        output_frontier=LogCardinality.bits(frontier_bits),
        certificate="test linear certificate",
    )


def _exact_abstract_minimax(
    partition: WeightedGateClassPartition,
    *,
    budget: int,
    threshold: Fraction,
) -> LogCardinality:
    quotas = enumerate_class_quotas(partition.class_sizes, budget)
    attacks = tuple(
        counts
        for counts in itertools.product(
            *(range(size + 1) for size in partition.class_sizes)
        )
        if any(counts)
    )
    capacities = {
        counts: capacity_upper_bound_for_counts(partition, counts) for counts in attacks
    }
    for candidate in sorted({LogCardinality.zero(), *capacities.values()}):
        bad = tuple(counts for counts in attacks if capacities[counts] > candidate)
        result = linprog(
            c=[0.0] * len(quotas),
            A_ub=[
                [
                    float(
                        quota_miss_probability(
                            partition.class_sizes,
                            quota,
                            counts,
                        )
                    )
                    for quota in quotas
                ]
                for counts in bad
            ]
            or None,
            b_ub=[float(1 - threshold)] * len(bad) or None,
            A_eq=[[1.0] * len(quotas)],
            b_eq=[1.0],
            bounds=(0.0, None),
            method="highs",
        )
        if result.success:
            return candidate
    raise AssertionError("checking every gate must make the game feasible")


def test_scalable_miss_probability_matches_binomial_ratio() -> None:
    for population in range(1, 15):
        for checked in range(population + 1):
            for attacked in range(population + 1):
                expected = (
                    Fraction(
                        math.comb(population - attacked, checked),
                        math.comb(population, checked),
                    )
                    if checked <= population - attacked
                    else Fraction()
                )
                assert (
                    _scalable_miss_probability(population, checked, attacked)
                    == expected
                )


def test_float_capacity_recovery_never_snaps_downward() -> None:
    just_above_power = math.log2(2**34 + 1)
    singleton = _capacity_from_width(
        just_above_power,
        logical_vocabulary_size=2,
    )
    aggregate = _upper_capacity_from_width(
        just_above_power,
        logical_vocabulary_size=2,
    )
    assert singleton == LogCardinality.bits(35)
    assert aggregate == LogCardinality.bits(35)

    token_width = math.log2(3)
    just_above_token = math.nextafter(token_width, math.inf)
    assert _capacity_from_width(
        just_above_token,
        logical_vocabulary_size=3,
    ) == LogCardinality.bits(2)


def test_single_class_best_response_matches_explicit_attack_count() -> None:
    partition = _partition(
        (WeightedGateClass("all", 100, LogCardinality.bits(2)),),
        frontier_bits=20,
    )
    strategy = fixed_quota_strategy(partition, (10,))
    attack = stratified_quota_best_response(
        partition,
        strategy,
        detection_threshold=Fraction(99, 100),
    )

    assert attack is not None
    expected_count = max(
        attacked
        for attacked in range(1, 11)
        if _scalable_miss_probability(100, 10, attacked) > Fraction(1, 100)
    )
    assert attack.attacked_counts == (expected_count,)
    assert attack.capacity_upper_bits == min(20, 2 * expected_count)


def test_multiclass_best_response_matches_brute_force_counts() -> None:
    partition = _partition(
        (
            WeightedGateClass("narrow", 5, LogCardinality.bits(1)),
            WeightedGateClass("wide", 4, LogCardinality.bits(3)),
        ),
        frontier_bits=12,
    )
    strategy = fixed_quota_strategy(partition, (2, 1))
    threshold = Fraction(3, 4)
    attack = stratified_quota_best_response(
        partition,
        strategy,
        detection_threshold=threshold,
    )

    brute: list[tuple[int, Fraction, tuple[int, int]]] = []
    for narrow, wide in itertools.product(range(6), range(5)):
        if narrow + wide == 0:
            continue
        miss = _scalable_miss_probability(5, 2, narrow)
        miss *= _scalable_miss_probability(4, 1, wide)
        if miss > 1 - threshold:
            brute.append((min(12, narrow + 3 * wide), miss, (narrow, wide)))
    expected = max(brute)
    assert attack is not None
    assert attack.capacity_upper_bits == expected[0]
    assert attack.detection_probability == 1 - expected[1]
    assert attack.attacked_counts == expected[2]


def test_equalized_strategy_uses_exact_budget_and_weights_wide_gates_more() -> None:
    partition = _partition(
        (
            WeightedGateClass("dead", 1_000, LogCardinality.zero()),
            WeightedGateClass("bf16", 10_000, LogCardinality.bits(16)),
            WeightedGateClass("fp32", 10_000, LogCardinality.bits(32)),
        ),
        frontier_bits=100,
    )
    strategy = equalized_log_quota_strategy(partition, 2_000)
    probabilities = strategy.inclusion_probabilities

    assert strategy.checked_gate_budget == 2_000
    assert strategy.checked_counts[0] == 0
    assert probabilities[2] > probabilities[1] > 0


def test_equalized_strategy_checks_dead_gates_only_after_live_gates() -> None:
    partition = _partition(
        (
            WeightedGateClass("dead", 10, LogCardinality.zero()),
            WeightedGateClass("live", 10, LogCardinality.bits(1)),
        )
    )
    strategy = equalized_log_quota_strategy(partition, 15)
    assert strategy.checked_counts == (5, 10)


def test_capacity_profile_conversion_preserves_exact_token_capacity() -> None:
    profile = ModelCapacityProfile(
        model_id="toy",
        prompt_tokens=2,
        generated_tokens=3,
        logical_vocabulary_size=11,
        numerical_profile_id="toy",
        regions=(
            CapacityRegion("work", "work", 100, 32.0),
            CapacityRegion("argmax", "tokens", 3, math.log2(11)),
            CapacityRegion(
                "selector",
                "seven-way index",
                4,
                math.log2(7),
                value_cardinality_upper_bound=7,
            ),
        ),
        assumptions=(),
    )
    partition = weighted_partition_from_capacity_profile(profile)

    assert partition.classes[0].singleton_capacity == LogCardinality.bits(32)
    assert partition.classes[1].singleton_capacity == GateCapacity.values(11).log_value
    assert partition.classes[2].singleton_capacity == GateCapacity.values(7).log_value
    assert partition.output_frontier == GateCapacity.values(11).log_value.scale(3)
    assert partition.total_gate_count == profile.total_gate_count


def test_equal_capacity_classes_coalesce_without_changing_accounting() -> None:
    partition = _partition(
        (
            WeightedGateClass("a", 10, LogCardinality.bits(16)),
            WeightedGateClass("b", 20, LogCardinality.bits(32)),
            WeightedGateClass("c", 30, LogCardinality.bits(16)),
        )
    )
    coalesced = coalesce_equal_capacity_classes(partition)

    assert len(coalesced.classes) == 2
    assert coalesced.total_gate_count == partition.total_gate_count
    by_capacity = {
        gate_class.singleton_capacity: gate_class.gate_count
        for gate_class in coalesced.classes
    }
    assert by_capacity[LogCardinality.bits(16)] == 40
    assert by_capacity[LogCardinality.bits(32)] == 20


def test_aggregate_capacity_caps_arbitrary_count_attacks() -> None:
    partition = _partition(
        (
            WeightedGateClass(
                "shared-funnel",
                100,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(4),
            ),
        ),
        frontier_bits=20,
    )
    strategy = fixed_quota_strategy(partition, (0,))
    attack = stratified_quota_best_response(partition, strategy)

    assert attack is not None
    assert attack.capacity_upper_bound == LogCardinality.bits(4)


def test_coalescing_sums_equal_width_aggregate_caps() -> None:
    partition = _partition(
        (
            WeightedGateClass(
                "left",
                100,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(4),
            ),
            WeightedGateClass(
                "right",
                100,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(6),
            ),
        ),
        frontier_bits=20,
    )
    merged = coalesce_equal_capacity_classes(partition)

    assert len(merged.classes) == 1
    assert merged.classes[0].full_class_capacity == LogCardinality.bits(10)


def test_coalescing_is_a_sound_capacity_relaxation_when_caps_differ() -> None:
    partition = _partition(
        (
            WeightedGateClass(
                "small-cap",
                4,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(2),
            ),
            WeightedGateClass(
                "large-cap",
                4,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(8),
            ),
        ),
        frontier_bits=10,
    )
    merged = coalesce_equal_capacity_classes(partition)

    for left in range(5):
        for right in range(5):
            original = capacity_upper_bound_for_counts(
                partition,
                (left, right),
            )
            relaxed = capacity_upper_bound_for_counts(
                merged,
                (left + right,),
            )
            assert relaxed >= original


def test_frontier_equivalent_coalescing_preserves_capacity_exactly() -> None:
    partition = _partition(
        (
            WeightedGateClass(
                "frontier-a",
                3,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(10),
            ),
            WeightedGateClass(
                "frontier-b",
                4,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(12),
            ),
            WeightedGateClass(
                "local-cap",
                4,
                LogCardinality.bits(2),
                aggregate_capacity=LogCardinality.bits(3),
            ),
        ),
        frontier_bits=10,
    )
    merged = coalesce_frontier_equivalent_classes(partition)
    assert len(merged.classes) == 2

    for left in range(4):
        for right in range(5):
            for local in range(5):
                original = capacity_upper_bound_for_counts(
                    partition,
                    (left, right, local),
                )
                merged_counts = tuple(
                    left + right
                    if set(gate_class.source_class_ids) == {"frontier-a", "frontier-b"}
                    else local
                    for gate_class in merged.classes
                )
                coalesced = capacity_upper_bound_for_counts(
                    merged,
                    merged_counts,
                )
                assert coalesced == original

    for budget in range(partition.total_gate_count + 1):
        assert _exact_abstract_minimax(
            partition,
            budget=budget,
            threshold=Fraction(3, 4),
        ) == _exact_abstract_minimax(
            merged,
            budget=budget,
            threshold=Fraction(3, 4),
        )


def test_frontier_equivalent_coalescing_avoids_generated_id_collisions() -> None:
    partition = _partition(
        (
            WeightedGateClass("zero", 2, LogCardinality.zero()),
            WeightedGateClass(
                "minimax-equivalent/0/0-bit",
                1,
                LogCardinality.bits(1),
                aggregate_capacity=LogCardinality.bits(1),
            ),
        )
    )

    reduced = coalesce_frontier_equivalent_classes(partition)

    assert len({gate_class.id for gate_class in reduced.classes}) == 2


def test_weighted_partition_rejects_negative_frontier() -> None:
    with pytest.raises(ValueError, match="frontier"):
        WeightedGateClassPartition(
            model_id="negative",
            classes=(WeightedGateClass("gate", 1, LogCardinality.bits(1)),),
            output_frontier=LogCardinality(Fraction(1, 2)),
            certificate="invalid test",
        )


def test_universal_lower_bound_uses_exact_strict_miss_threshold() -> None:
    partition = _partition(
        (WeightedGateClass("all", 100, LogCardinality.bits(2)),),
        frontier_bits=20,
    )
    certificate = universal_minimax_lower_bound(
        partition,
        checked_gate_budget=10,
        detection_threshold=Fraction(99, 100),
    )
    expected_count = max(
        attacked
        for attacked in range(1, 11)
        if _scalable_miss_probability(100, 10, attacked) > Fraction(1, 100)
    )
    assert certificate.attack_size == expected_count
    assert certificate.capacity_lower_bits == min(20, 2 * expected_count)
    assert certificate.miss_probability > Fraction(1, 100)


def test_universal_lower_bound_remains_valid_when_budget_targets_pool() -> None:
    partition = _partition(
        (WeightedGateClass("wide", 10, LogCardinality.bits(4)),),
        frontier_bits=20,
    )
    certificate = universal_minimax_lower_bound(
        partition,
        checked_gate_budget=10,
        detection_threshold=Fraction(3, 4),
    )

    # The proof pessimistically allows all checks to target the candidate pool.
    assert certificate.capacity_lower_bound.is_zero
    assert certificate.pool_class_ids == ()


def test_lower_certificate_never_exceeds_fixed_quota_upper() -> None:
    partition = _partition(
        (
            WeightedGateClass("narrow", 1_000, LogCardinality.bits(1)),
            WeightedGateClass("wide", 1_000, LogCardinality.bits(3)),
        ),
        frontier_bits=20,
    )
    budget = 200
    strategy = equalized_log_quota_strategy(partition, budget)
    attack = stratified_quota_best_response(partition, strategy)
    certificate = universal_minimax_lower_bound(
        partition,
        checked_gate_budget=budget,
    )

    assert attack is not None
    assert certificate.capacity_lower_bound <= attack.capacity_upper_bound


def test_uniform_quota_uses_exact_budget_and_nearly_equal_marginals() -> None:
    partition = _partition(
        (
            WeightedGateClass("a", 17, LogCardinality.bits(1)),
            WeightedGateClass("b", 23, LogCardinality.bits(2)),
            WeightedGateClass("c", 60, LogCardinality.bits(3)),
        )
    )
    strategy = uniform_quota_strategy(partition, 33)

    assert strategy.checked_gate_budget == 33
    probabilities = [float(value) for value in strategy.inclusion_probabilities]
    assert max(probabilities) - min(probabilities) < 1 / 17


def test_fixed_quota_search_never_loses_to_its_seed_strategies() -> None:
    partition = _partition(
        (
            WeightedGateClass("narrow", 1_000, LogCardinality.bits(1)),
            WeightedGateClass("wide", 200, LogCardinality.bits(4)),
        ),
        frontier_bits=20,
    )
    budget = 120
    threshold = Fraction(3, 4)
    seeds = (
        uniform_quota_strategy(partition, budget),
        equalized_log_quota_strategy(partition, budget),
    )
    seed_bounds = [
        stratified_quota_best_response(
            partition,
            seed,
            detection_threshold=threshold,
        )
        for seed in seeds
    ]
    result = optimize_fixed_quota_strategy(
        partition,
        checked_gate_budget=budget,
        detection_threshold=threshold,
        max_evaluations=300,
    )

    assert result.strategy.checked_gate_budget == budget
    assert result.evaluated_quota_count <= 300
    assert result.worst_attack is not None
    assert all(seed is not None for seed in seed_bounds)
    assert result.certified_upper_bound <= min(
        seed.capacity_upper_bound for seed in seed_bounds if seed is not None
    )
    assert not result.globally_optimal


def test_fixed_quota_search_respects_tiny_evaluation_limit() -> None:
    partition = _partition(
        tuple(
            WeightedGateClass(
                f"class-{index}",
                10,
                LogCardinality.bits(index % 3 + 1),
            )
            for index in range(20)
        ),
        frontier_bits=10,
    )
    result = optimize_fixed_quota_strategy(
        partition,
        checked_gate_budget=50,
        max_evaluations=1,
    )

    assert result.evaluated_quota_count == 1
    assert result.strategy.checked_gate_budget == 50


def test_class_moment_inverse_interpolates_exact_hypergeometric_curve() -> None:
    partition = _partition(
        (WeightedGateClass("all", 100, LogCardinality.bits(1)),),
        frontier_bits=2,
    )
    certificate = pure_class_moment_lower_bound(
        partition,
        checked_gate_budget=89,
        detection_threshold=Fraction(99, 100),
    )
    requirement = certificate.requirements[0]
    miss_89 = _scalable_miss_probability(100, 89, 2)
    miss_90 = _scalable_miss_probability(100, 90, 2)
    expected = 89 + Fraction(
        miss_89 - Fraction(1, 100),
        miss_89 - miss_90,
    )

    assert certificate.capacity_lower_bits == 2
    assert requirement.attack_size == 2
    assert requirement.required_expected_checks == expected
    assert requirement.required_expected_checks > 89


def test_class_moment_bound_captures_cross_class_budget_tradeoff() -> None:
    partition = _partition(
        (
            WeightedGateClass("fp16", 990, LogCardinality.bits(16)),
            WeightedGateClass("fp32", 10, LogCardinality.bits(32)),
        ),
        frontier_bits=32,
    )
    certificate = pure_class_moment_lower_bound(
        partition,
        checked_gate_budget=900,
    )

    assert certificate.capacity_lower_bits == 32
    assert {item.class_id: item.attack_size for item in certificate.requirements} == {
        "fp16": 2,
        "fp32": 1,
    }
    assert certificate.required_expected_checks > 900


def test_class_moment_lower_bound_does_not_exceed_replayed_upper() -> None:
    partition = _partition(
        (
            WeightedGateClass("narrow", 1_000, LogCardinality.bits(1)),
            WeightedGateClass("wide", 200, LogCardinality.bits(4)),
        ),
        frontier_bits=20,
    )
    budget = 120
    strategy = equalized_log_quota_strategy(partition, budget)
    attack = stratified_quota_best_response(partition, strategy)
    certificate = pure_class_moment_lower_bound(
        partition,
        checked_gate_budget=budget,
    )

    assert attack is not None
    assert certificate.capacity_lower_bound <= attack.capacity_upper_bound


def test_class_moment_bound_skips_unreachable_loose_aggregate_cap() -> None:
    partition = _partition(
        (
            WeightedGateClass(
                "loose",
                1,
                LogCardinality.bits(1),
                aggregate_capacity=LogCardinality.bits(2),
            ),
            WeightedGateClass("reachable", 1, LogCardinality.bits(2)),
        ),
        frontier_bits=8,
    )

    certificate = pure_class_moment_lower_bound(
        partition,
        checked_gate_budget=0,
    )

    assert certificate.capacity_lower_bound == LogCardinality.bits(2)


def test_global_lower_certificates_do_not_exceed_exact_tiny_minimax() -> None:
    partitions = (
        _partition(
            (
                WeightedGateClass("a", 3, LogCardinality.bits(1)),
                WeightedGateClass("b", 4, LogCardinality.bits(3)),
            ),
            frontier_bits=7,
        ),
        _partition(
            (
                WeightedGateClass(
                    "funnel",
                    4,
                    LogCardinality.bits(2),
                    aggregate_capacity=LogCardinality.bits(3),
                ),
                WeightedGateClass("linear", 3, LogCardinality.bits(2)),
            ),
            frontier_bits=6,
        ),
    )
    threshold = Fraction(3, 4)
    for partition in partitions:
        for budget in range(partition.total_gate_count + 1):
            exact = _exact_abstract_minimax(
                partition,
                budget=budget,
                threshold=threshold,
            )
            pool = universal_minimax_lower_bound(
                partition,
                checked_gate_budget=budget,
                detection_threshold=threshold,
            )
            moment = pure_class_moment_lower_bound(
                partition,
                checked_gate_budget=budget,
                detection_threshold=threshold,
            )
            assert pool.capacity_lower_bound <= exact
            assert moment.capacity_lower_bound <= exact
