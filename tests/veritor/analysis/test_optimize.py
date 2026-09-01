from fractions import Fraction

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis import (
    AdditiveExpectedCost,
    CountedCapacityClass,
    CountedCapacitySchema,
    CountedCapacitySemantics,
    GridOptimizationStatus,
    RationalPolicyGrid,
    mega_unit_relaxation_bound,
    optimize_finite_policy_grid,
    optimize_policy_grid,
)
from veritor.core import VerificationPolicy


def test_additive_expected_cost_is_exact():
    cost = AdditiveExpectedCost(1, 2, 3)
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 4), 0)

    assert cost.evaluate(policy) == Fraction(19, 8)


def test_finite_grid_optimizer_reports_exact_on_grid_and_cost(
    make_partitions,
    exact_oracle_type,
):
    replay, verification = make_partitions((1,), label="optimizer")
    oracle = exact_oracle_type((5,))
    grid = RationalPolicyGrid((0, 1), (0, 1), 0)
    cost = AdditiveExpectedCost(1, 2, 3)

    result = optimize_finite_policy_grid(
        replay,
        verification,
        oracle,
        grid,
        cost,
        solver="exhaustive",
        capacity_limit=LogCardinality.zero(),
    )

    assert result.status is GridOptimizationStatus.EXACT_ON_GRID
    assert result.exact_on_grid
    assert result.chosen_policy == VerificationPolicy(1, 1, 0)
    assert result.chosen[0].expected_cost == Fraction(6)
    assert result.chosen[0].bound.is_exact
    assert result.chosen[0].bound.upper_bound == LogCardinality.zero()


def test_bracketed_or_relaxed_grid_is_labeled_heuristic():
    singleton = LogCardinality.bits(1)
    schema = CountedCapacitySchema(
        model_id="optimizer-counted",
        classes=(
            CountedCapacityClass(
                "scalar",
                2,
                singleton,
                singleton_lower_bound=singleton,
            ),
        ),
        output_frontier=LogCardinality.bits(2),
        semantics=CountedCapacitySemantics.EXACT_CAPPED_LINEAR,
        certificate="exact toy capacity",
    )
    grid = RationalPolicyGrid((0, 1), (1,), 0)
    cost = AdditiveExpectedCost(0, 1, 1)

    result = optimize_policy_grid(
        grid,
        cost,
        lambda policy: mega_unit_relaxation_bound(schema, policy),
    )

    assert result.status is GridOptimizationStatus.HEURISTIC
    assert not result.exact_on_grid
    assert result.chosen
    assert all(choice.bound.certified_upper_bound is not None for choice in result.chosen)
    assert all(not choice.bound.is_exact for choice in result.chosen)


def test_grid_cost_constraint_can_exclude_every_policy(
    make_partitions,
    exact_oracle_type,
):
    replay, verification = make_partitions((1,), label="optimizer-empty")
    grid = RationalPolicyGrid((0, 1), (0, 1), 0)
    result = optimize_finite_policy_grid(
        replay,
        verification,
        exact_oracle_type((1,)),
        grid,
        AdditiveExpectedCost(5, 1, 1),
        solver="exhaustive",
        maximum_expected_cost=4,
    )

    assert result.status is GridOptimizationStatus.EXACT_ON_GRID
    assert result.chosen == ()
    assert "no grid policy" in result.reason
