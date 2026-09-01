from __future__ import annotations

import math

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.models.deepseek_v4_pro import (
    DEEPSEEK_V4_PRO,
    build_deepseek_v4_pro_capacity_profile,
)
from circuit_cut_analysis.models.inkling import (
    INKLING,
    build_inkling_capacity_profile,
)
from circuit_cut_analysis.models.kimi_k3 import (
    KIMI_K3,
    build_kimi_k3_capacity_profile,
)
from circuit_cut_analysis.weighted_sampling import (
    weighted_partition_from_capacity_profile,
)
from veritor.core import Capability, SupportState, Unsupported, identity_digest
from veritor.plugins import (
    DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID,
    INKLING_NUMERICAL_PROFILE_ID,
    KIMI_K3_NUMERICAL_PROFILE_ID,
    AggregateBoundArtifact,
    CapacityClaimKind,
    DeepSeekV4ProCompileRequest,
    TraceBinding,
    compile_deepseek_v4_pro,
    compile_inkling,
    compile_kimi_k3,
)


@pytest.mark.parametrize(
    ("compile_plugin", "direct_builder", "config", "numerical_profile_id"),
    (
        (
            compile_kimi_k3,
            build_kimi_k3_capacity_profile,
            KIMI_K3,
            KIMI_K3_NUMERICAL_PROFILE_ID,
        ),
        (
            compile_deepseek_v4_pro,
            build_deepseek_v4_pro_capacity_profile,
            DEEPSEEK_V4_PRO,
            DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID,
        ),
        (
            compile_inkling,
            build_inkling_capacity_profile,
            INKLING,
            INKLING_NUMERICAL_PROFILE_ID,
        ),
    ),
)
def test_aggregate_plugins_delegate_to_direct_profile_builders(
    compile_plugin,
    direct_builder,
    config,
    numerical_profile_id: str,
) -> None:
    artifact = compile_plugin()
    direct = direct_builder(
        100,
        100,
        config=config,
        numerical_profile_id=numerical_profile_id,
    )
    assert isinstance(artifact, AggregateBoundArtifact)
    assert artifact.profile == direct
    assert artifact.regions == direct.regions
    assert artifact.total_gate_count == direct.total_gate_count
    assert artifact.assumption_texts == direct.assumptions
    assert artifact.weighted_partition == (
        weighted_partition_from_capacity_profile(direct)
    )
    assert artifact.output_frontier == LogCardinality.cardinality(
        direct.logical_vocabulary_size
    ).scale(direct.generated_tokens)
    assert artifact.bound_provider.partition is artifact.weighted_partition
    assert (
        artifact.bound_provider.claim_kind
        is CapacityClaimKind.PROFILE_SELF_CUT_RELAXATION
    )


@pytest.mark.parametrize(
    "compile_plugin",
    (compile_kimi_k3, compile_deepseek_v4_pro, compile_inkling),
)
def test_aggregate_artifacts_do_not_fabricate_circuit_surfaces(
    compile_plugin,
) -> None:
    artifact = compile_plugin()
    for name in (
        "circuit",
        "structural_circuit",
        "indexed",
        "gate_at",
        "gate_domain",
        "output_refs",
    ):
        assert not hasattr(artifact, name)

    replay = artifact.replay()
    execution = artifact.execution_access()
    verification = artifact.verify()
    assert isinstance(replay, Unsupported)
    assert isinstance(execution, Unsupported)
    assert isinstance(verification, Unsupported)
    assert replay.capability is Capability.STATIC_PARTITION
    assert execution.capability is Capability.EXECUTE
    assert verification.capability is Capability.VERIFY


@pytest.mark.parametrize(
    ("compile_plugin", "region_id", "expected_cardinality"),
    (
        (compile_kimi_k3, "moe-top-k-selectors", 896),
        (compile_deepseek_v4_pro, "csa-index-selector", 49),
        (compile_deepseek_v4_pro, "moe-top6-selector", 384),
        (compile_inkling, "moe-top6-selectors", 256),
    ),
)
def test_selector_cardinalities_are_preserved_exactly(
    compile_plugin,
    region_id: str,
    expected_cardinality: int,
) -> None:
    artifact = compile_plugin()
    region = next(item for item in artifact.regions if item.id == region_id)
    assert region.value_cardinality_upper_bound == expected_cardinality
    assert math.isclose(
        region.self_cut_bits_per_gate,
        math.log2(expected_cardinality),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_profile_self_cut_provider_preserves_certificate_scope() -> None:
    artifact = compile_kimi_k3()
    counts = [0] * len(artifact.weighted_partition.classes)
    counts[0] = 1
    result = artifact.bound_provider.evaluate(counts)
    assert result.claim_kind is CapacityClaimKind.PROFILE_SELF_CUT_RELAXATION
    assert result.lower_bound == LogCardinality.zero()
    assert result.upper_bound <= artifact.output_frontier
    assert result.certificate == artifact.weighted_partition.certificate
    assert result.assumptions == artifact.profile.assumptions


@pytest.mark.parametrize(
    ("compile_plugin", "expected_caveats"),
    (
        (
            compile_kimi_k3,
            ("trace-conditional", "mxfp4", "tie-breaking"),
        ),
        (
            compile_inkling,
            ("text-only", "nvfp4", "physical unembedding"),
        ),
    ),
)
def test_model_specific_caveats_are_not_dropped(
    compile_plugin,
    expected_caveats: tuple[str, ...],
) -> None:
    assumptions = "\n".join(compile_plugin().assumption_texts).lower()
    for caveat in expected_caveats:
        assert caveat in assumptions


def test_deepseek_is_trace_conditional_and_unbound_by_default() -> None:
    artifact = compile_deepseek_v4_pro()
    assert artifact.trace_binding is TraceBinding.UNBOUND_TRACE_CONDITIONAL
    assert artifact.trace_digest is None
    bound = artifact.capabilities.status_for(Capability.STATIC_BOUND)
    assert bound.state is SupportState.CONDITIONAL
    assert bound.reason_code == "TRACE_DIGEST_NOT_BOUND"
    assert any(
        "trace_conditional" in assumption.lower()
        and "observed greedy trace" in assumption.lower()
        for assumption in artifact.assumption_texts
    )


def test_deepseek_trace_digest_binds_profile_but_not_execution() -> None:
    trace_digest = identity_digest(
        "tests/veritor/plugins/deepseek-trace/v1",
        {"trace": "fixture"},
    )
    unbound = compile_deepseek_v4_pro()
    bound = compile_deepseek_v4_pro(
        DeepSeekV4ProCompileRequest(trace_digest=trace_digest)
    )
    assert bound.trace_binding is TraceBinding.TRACE_BOUND
    assert bound.trace_digest == trace_digest
    assert bound.identity.digest != unbound.identity.digest
    assert bound.capabilities.supports(Capability.STATIC_BOUND)
    assert not bound.capabilities.supports(Capability.EXECUTE)
    assert not bound.capabilities.supports(Capability.VERIFY)
    assert isinstance(bound.execute(), Unsupported)
    assert not bound.runtime_validated
