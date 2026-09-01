from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.weighted_sampling import (
    WeightedGateClass,
    WeightedGateClassPartition,
)
from veritor.analysis import (
    BoundClaimStrength,
    CountedCapacityClass,
    CountedCapacitySchema,
    CountedCapacitySemantics,
    CountedReplayLayout,
    CountedReplayType,
    CountedSolverLimits,
    TerminationStatus,
    actual_counted_layout_bound,
    capped_linear_allocation,
    counted_capacity_upper_bound,
    counted_fixed_policy_bound,
    counted_schema_from_capacity_profile,
    mega_unit_relaxation_bound,
    mega_unit_threshold,
    reconcile_counted_layout,
)
from veritor.core import VerificationPolicy


def exact_schema(
    *,
    count: int = 4,
    frontier_bits: int = 10,
) -> CountedCapacitySchema:
    capacity = LogCardinality.bits(1)
    return CountedCapacitySchema(
        model_id="tiny-counted",
        classes=(
            CountedCapacityClass(
                "scalar",
                count,
                capacity,
                singleton_lower_bound=capacity,
            ),
        ),
        output_frontier=LogCardinality.bits(frontier_bits),
        semantics=CountedCapacitySemantics.EXACT_CAPPED_LINEAR,
        certificate="exact additive toy model",
    )


def test_capacity_classes_and_replay_incidence_reconcile_separately():
    schema = exact_schema(count=4)
    layout = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(CountedReplayType("pairs", 2, (2,)),),
    )
    bad_order = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=("other",),
        replay_types=(CountedReplayType("pairs", 2, (2,)),),
    )
    bad_total = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(CountedReplayType("single", 3, (1,)),),
    )

    reconcile_counted_layout(schema, layout)
    with pytest.raises(ValueError, match="ordered class ids"):
        reconcile_counted_layout(schema, bad_order)
    with pytest.raises(ValueError, match="totals"):
        reconcile_counted_layout(schema, bad_total)


def test_counted_schema_identity_binds_external_artifact_provenance():
    schema = exact_schema()

    first = replace(schema, provenance_identity="11" * 32)
    second = replace(schema, provenance_identity="22" * 32)

    assert first.identity != schema.identity
    assert second.identity != schema.identity
    assert first.identity != second.identity


def test_actual_layout_is_exact_for_declared_exact_counted_model():
    schema = exact_schema(count=4)
    layout = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(CountedReplayType("spread", 4, (1,)),),
    )
    policy = VerificationPolicy(
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
    )

    result = actual_counted_layout_bound(schema, layout, policy)

    assert result.is_exact
    assert result.method == "actual-counted-layout-exhaustive"
    assert result.identities.capacity_schema_identity == schema.identity
    assert result.identities.replay_layout_identity == layout.identity


def test_mega_unit_upper_dominates_tiny_actual_replay_layouts():
    narrow = LogCardinality.bits(1)
    wide = LogCardinality.bits(3)
    schema = CountedCapacitySchema(
        model_id="tiny-counted",
        classes=(
            CountedCapacityClass(
                "narrow",
                2,
                narrow,
                singleton_lower_bound=narrow,
            ),
            CountedCapacityClass(
                "wide",
                2,
                wide,
                singleton_lower_bound=wide,
            ),
        ),
        output_frontier=LogCardinality.bits(7),
        semantics=CountedCapacitySemantics.EXACT_CAPPED_LINEAR,
        certificate="exact additive toy model",
    )
    clustered = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(CountedReplayType("clustered", 1, (2, 2)),),
    )
    spread = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(
            CountedReplayType("narrow", 2, (1, 0)),
            CountedReplayType("wide", 2, (0, 1)),
        ),
    )
    paired = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(CountedReplayType("paired", 2, (1, 1)),),
    )
    policy = VerificationPolicy(
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
    )
    mega = mega_unit_relaxation_bound(schema, policy)

    for layout in (clustered, spread, paired):
        actual = actual_counted_layout_bound(schema, layout, policy)
        assert actual.upper_bound <= mega.upper_bound
    assert mega.claim_strength is BoundClaimStrength.CERTIFIED_UPPER
    assert not mega.is_exact
    assert "mega-unit" in mega.method


def test_mega_threshold_respects_strict_equality():
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 2))

    threshold = mega_unit_threshold(100, policy)

    assert threshold.max_attack_count == 0
    assert threshold.exact


def test_mega_threshold_matches_exact_small_rational_grid():
    values = (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1))
    etas = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    for total, q, s, eta in itertools.product(range(7), values, values, etas):
        policy = VerificationPolicy(q, s, eta)
        expected = max(
            attacked
            for attacked in range(total + 1)
            if 1 - q + q * (1 - s) ** attacked > eta
        )
        threshold = mega_unit_threshold(total, policy)
        assert threshold.exact
        assert threshold.max_attack_count == expected


def test_capped_linear_allocation_prefers_capacity_and_obeys_frontier():
    schema = CountedCapacitySchema(
        model_id="allocation",
        classes=(
            CountedCapacityClass(
                "narrow",
                10,
                LogCardinality.bits(1),
                singleton_lower_bound=LogCardinality.bits(1),
            ),
            CountedCapacityClass(
                "wide",
                10,
                LogCardinality.bits(4),
                aggregate_upper_bound=LogCardinality.bits(6),
                singleton_lower_bound=LogCardinality.bits(4),
            ),
        ),
        output_frontier=LogCardinality.bits(7),
        semantics=CountedCapacitySemantics.EXACT_CAPPED_LINEAR,
        certificate="test exact capped allocation",
    )

    allocation = capped_linear_allocation(schema, 3)

    assert allocation.attacked_counts == (1, 2)
    assert allocation.capacity_upper_bound == schema.output_frontier
    assert not allocation.output_frontier_fallback


def test_capped_linear_allocation_matches_brute_force_tiny_counts():
    schema = CountedCapacitySchema(
        model_id="allocation-brute",
        classes=(
            CountedCapacityClass(
                "a",
                3,
                LogCardinality.bits(2),
                aggregate_upper_bound=LogCardinality.bits(5),
            ),
            CountedCapacityClass(
                "b",
                4,
                LogCardinality.bits(3),
                aggregate_upper_bound=LogCardinality.bits(7),
            ),
        ),
        output_frontier=LogCardinality.bits(9),
        semantics=CountedCapacitySemantics.CERTIFIED_CAPPED_LINEAR_UPPER,
        certificate="tiny brute certificate",
    )

    for budget in range(schema.total_verification_units + 1):
        result = capped_linear_allocation(schema, budget)
        brute = max(
            counted_capacity_upper_bound(schema, counts)
            for counts in itertools.product(range(4), range(5))
            if sum(counts) <= budget
        )
        assert result.capacity_upper_bound == brute


def test_counted_upper_is_always_capped_by_output_frontier():
    partition = WeightedGateClassPartition(
        model_id="frontier",
        classes=(
            WeightedGateClass("a", 100, LogCardinality.bits(8)),
            WeightedGateClass("b", 100, LogCardinality.bits(16)),
        ),
        output_frontier=LogCardinality.bits(5),
        certificate="self-cut upper",
    )

    result = counted_fixed_policy_bound(
        partition,
        VerificationPolicy(0, 1, 0),
    )

    assert result.upper_bound == partition.output_frontier
    assert result.claim_strength is BoundClaimStrength.CERTIFIED_UPPER


def test_huge_exponents_use_conservative_inclusion_without_underflow():
    schema = exact_schema(count=10**12, frontier_bits=10)
    limits = CountedSolverLimits(
        max_actual_verification_units=0,
        max_exact_exponent=2,
        max_exact_power_bits=256,
    )
    policy = VerificationPolicy(1, Fraction(1, 10**12), Fraction(1, 2))

    result = mega_unit_relaxation_bound(
        schema,
        policy,
        limits=limits,
    )

    assert result.upper_bound == schema.output_frontier
    assert result.numerically_conservative
    assert (
        result.termination_status
        is TerminationStatus.NUMERICALLY_CONSERVATIVE
    )
    assert result.upper_witness is not None
    assert result.upper_witness.survival_probability is None


def test_nonintegral_giant_allocation_has_output_frontier_fallback():
    schema = CountedCapacitySchema(
        model_id="fallback",
        classes=(
            CountedCapacityClass(
                "ternary",
                100,
                LogCardinality.cardinality(3),
                singleton_lower_bound=LogCardinality.cardinality(3),
            ),
        ),
        output_frontier=LogCardinality.cardinality(5),
        semantics=CountedCapacitySemantics.EXACT_CAPPED_LINEAR,
        certificate="test nonintegral model",
    )
    limits = CountedSolverLimits(
        max_actual_verification_units=0,
        max_exact_exponent=0,
        max_exact_power_bits=0,
    )

    result = mega_unit_relaxation_bound(
        schema,
        VerificationPolicy(0, 1, 0),
        limits=limits,
    )

    assert result.upper_bound == schema.output_frontier
    assert result.numerically_conservative
    assert any("output-frontier fallback" in step for step in result.relaxation_chain)


def test_profile_assumptions_survive_counted_conversion_and_bound():
    profile = ModelCapacityProfile(
        model_id="aggregate",
        prompt_tokens=1,
        generated_tokens=1,
        logical_vocabulary_size=8,
        numerical_profile_id="fp-test",
        regions=(CapacityRegion("work", "work gates", 4, 2.0),),
        assumptions=("aggregate wiring omitted", "fixed token count"),
    )
    schema = counted_schema_from_capacity_profile(profile)

    result = mega_unit_relaxation_bound(
        schema,
        VerificationPolicy(Fraction(1, 2), Fraction(1, 2), 0),
        assumptions=("public numerical profile",),
    )

    assert result.assumptions == (
        "aggregate wiring omitted",
        "fixed token count",
        "public numerical profile",
    )


def test_large_supplied_layout_falls_back_with_certified_resource_bracket():
    schema = exact_schema(count=8)
    layout = CountedReplayLayout(
        model_id=schema.model_id,
        class_ids=schema.class_ids,
        replay_types=(CountedReplayType("spread", 8, (1,)),),
    )
    result = counted_fixed_policy_bound(
        schema,
        VerificationPolicy(Fraction(1, 2), Fraction(1, 2), 0),
        replay_layout=layout,
        limits=CountedSolverLimits(max_actual_verification_units=4),
    )

    assert result.termination_status is TerminationStatus.RESOURCE_LIMIT
    assert result.claim_strength is BoundClaimStrength.CERTIFIED_UPPER
    assert result.identities.replay_layout_identity == layout.identity
