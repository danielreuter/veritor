from __future__ import annotations

from fractions import Fraction

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.profiles import VLLM_FP16_REFERENCE, ServingProfile
from veritor import (
    AdditiveExpectedCost,
    ArchitectureId,
    Bound,
    BoundOptions,
    Compile,
    DeepSeekV4ProCompileRequest,
    GPT2CompileRequest,
    GreedyTextExecutionShape,
    MatmulCompileRequest,
    OptimizationResult,
    Optimize,
    RationalPolicyGrid,
    Unsupported,
    VerificationPolicy,
)
from veritor.analysis import (
    BoundClaimStrength,
    GridOptimizationStatus,
    PolicyGridOptimizationResult,
    TerminationStatus,
)
from veritor.core import Capability, identity_digest


def _unsupported_gpt2_request() -> GPT2CompileRequest:
    return GPT2CompileRequest(
        execution_shape=GreedyTextExecutionShape(
            prompt_tokens=2,
            generated_tokens=2,
        ),
        config=GPT2Config(
            model_id="research-api-gpt2-unsupported",
            layers=1,
            hidden_size=2,
            heads=1,
            intermediate_size=3,
            vocabulary_size=5,
            max_context=4,
        ),
        profile=VLLM_FP16_REFERENCE,
    )


def _catalog_gpt2_request() -> GPT2CompileRequest:
    profile = ServingProfile(
        id="research-api-three-bit",
        description="Uniform tiny width profile for lifted certificates.",
        weight_bits=3,
        activation_boundary_bits=3,
        kv_cache_bits=3,
        accumulator_bits=3,
        reduction_bits=3,
        nonlinear_internal_bits=3,
        probability_boundary_bits=3,
        residual_bits=3,
        logit_bits=3,
        assumptions=("Every represented structural boundary has width 3.",),
    )
    return GPT2CompileRequest(
        execution_shape=GreedyTextExecutionShape(
            prompt_tokens=2,
            generated_tokens=2,
        ),
        config=GPT2Config(
            model_id="research-api-gpt2-catalog",
            layers=3,
            hidden_size=9,
            heads=3,
            intermediate_size=9,
            vocabulary_size=4,
            max_context=4,
        ),
        profile=profile,
    )


def test_demo_bound_uses_literal_finite_partitions_exactly() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 4))

    exhaustive = Bound(
        artifact,
        policy,
        solver="exhaustive",
        max_verification_units=20,
    )
    branch = Bound(
        artifact,
        policy,
        options=BoundOptions(solver="branch-and-bound"),
    )

    assert not isinstance(exhaustive, Unsupported)
    assert not isinstance(branch, Unsupported)
    assert exhaustive.is_exact
    assert exhaustive.lower_bound == branch.lower_bound == branch.upper_bound
    assert exhaustive.identities.structure_identity == artifact.circuit.identity.digest
    assert (
        exhaustive.identities.replay_partition_identity
        == artifact.replay_partition.identity.digest
    )
    assert (
        exhaustive.identities.verification_partition_identity
        == artifact.verification_partition.identity.digest
    )


def test_matmul_bound_and_optimize_use_literal_finite_tuple() -> None:
    artifact = Compile(
        ArchitectureId.MATMUL,
        MatmulCompileRequest(((3,),), (((5,),),)),
    )
    bound = Bound(
        artifact,
        VerificationPolicy(1, 1, 0),
        solver="exhaustive",
    )
    optimized = Optimize(
        artifact,
        RationalPolicyGrid((0, 1), (0, 1), 0),
        AdditiveExpectedCost(0, 1, 1),
        bound_options={"solver": "exhaustive"},
        capacity_limit=LogCardinality.zero(),
    )

    assert not isinstance(bound, Unsupported)
    assert bound.is_exact
    assert bound.upper_bound == LogCardinality.zero()
    assert isinstance(optimized, PolicyGridOptimizationResult)
    assert optimized.status is GridOptimizationStatus.EXACT_ON_GRID
    assert optimized.chosen_policy == VerificationPolicy(1, 1, 0)


def test_gpt2_counted_bound_requires_certified_catalog() -> None:
    artifact = Compile(ArchitectureId.GPT2, _unsupported_gpt2_request())

    outcome = Bound(artifact, VerificationPolicy(1, 1, 0))

    assert isinstance(outcome, Unsupported)
    assert outcome.capability is Capability.STATIC_BOUND
    assert outcome.reason_code == "LIFTED_CERTIFICATE_CONDITIONS_UNMET"


def test_gpt2_certified_catalog_dispatches_to_counted_mega_unit() -> None:
    artifact = Compile(ArchitectureId.GPT2, _catalog_gpt2_request())
    outcome = Bound(
        artifact,
        VerificationPolicy(Fraction(1, 2), Fraction(1, 2), 0),
        granularity="row-layer-band",
        position_bands=2,
    )

    assert not isinstance(outcome, Unsupported)
    assert outcome.method == "counted-adversarial-mega-unit"
    assert outcome.claim_strength is BoundClaimStrength.CERTIFIED_UPPER
    assert not outcome.is_exact
    assert outcome.identities.capacity_schema_identity is not None
    assert "adversarial mega-unit replay relaxation" in outcome.relaxation_chain


@pytest.mark.parametrize(
    "architecture_id",
    (
        ArchitectureId.KIMI_K3,
        ArchitectureId.DEEPSEEK_V4_PRO,
        ArchitectureId.INKLING,
    ),
)
def test_aggregate_profiles_dispatch_to_assumption_scoped_counted_bound(
    architecture_id: ArchitectureId,
) -> None:
    artifact = Compile(architecture_id)
    outcome = Bound(
        artifact,
        VerificationPolicy(Fraction(1, 2), Fraction(1, 2), 0),
    )

    assert not isinstance(outcome, Unsupported)
    assert outcome.claim_strength is BoundClaimStrength.CERTIFIED_UPPER
    assert outcome.assumptions == artifact.assumption_texts
    assert outcome.identities.capacity_schema_identity is not None


def test_deepseek_counted_provenance_binds_distinct_trace_digests() -> None:
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2), 0)
    outcomes = []
    for marker in ("first", "second"):
        trace_digest = identity_digest(
            "tests/veritor/research-api/deepseek-trace/v1",
            {"trace": marker},
        )
        artifact = Compile(
            ArchitectureId.DEEPSEEK_V4_PRO,
            DeepSeekV4ProCompileRequest(trace_digest=trace_digest),
        )
        outcome = Bound(artifact, policy)
        assert not isinstance(outcome, Unsupported)
        assert any(
            "trace_conditional" in assumption.lower()
            for assumption in outcome.assumptions
        )
        outcomes.append(outcome)

    first, second = outcomes
    assert (
        first.identities.capacity_schema_identity
        != second.identities.capacity_schema_identity
    )
    assert first.tuple_identity != second.tuple_identity


def test_optimize_uses_bound_and_preserves_exact_grid_status() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    grid = RationalPolicyGrid((0, 1), (0, 1), 0)
    result: OptimizationResult = Optimize(
        artifact,
        grid,
        AdditiveExpectedCost(1, 2, 3),
        bound_options={"solver": "exhaustive", "max_verification_units": 20},
        capacity_limit=LogCardinality.zero(),
    )

    assert isinstance(result, PolicyGridOptimizationResult)
    assert result.status is GridOptimizationStatus.EXACT_ON_GRID
    assert result.chosen_policy == VerificationPolicy(1, 1, 0)
    assert result.chosen_bound is not None
    assert result.chosen_bound.is_exact


def test_optimize_propagates_unsupported_bound() -> None:
    artifact = Compile(ArchitectureId.GPT2, _unsupported_gpt2_request())

    result = Optimize(
        artifact,
        RationalPolicyGrid((1,), (1,), 0),
        AdditiveExpectedCost(0, 1, 1),
    )

    assert isinstance(result, Unsupported)
    assert result.capability is Capability.STATIC_BOUND


def test_resource_limited_demo_search_keeps_certified_status() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    outcome = Bound(
        artifact,
        VerificationPolicy(1, Fraction(1, 2), Fraction(1, 4)),
        solver="branch-and-bound",
        max_states=1,
        max_capacity_queries=1,
    )

    assert not isinstance(outcome, Unsupported)
    assert outcome.termination_status is TerminationStatus.RESOURCE_LIMIT
    assert outcome.claim_strength is BoundClaimStrength.CERTIFIED_BRACKET
    assert outcome.lower_bound <= outcome.upper_bound
