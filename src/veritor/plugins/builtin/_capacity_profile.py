"""Shared construction for aggregate capacity-profile plug-ins."""

from __future__ import annotations

from circuit_cut_analysis.models.capacity_profile import ModelCapacityProfile
from circuit_cut_analysis.weighted_sampling import (
    weighted_partition_from_capacity_profile,
)
from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    ClaimStatus,
    Digest,
    EvidenceStatus,
    JSONValue,
    SupportState,
)

from .._common import (
    assumption_records,
    capability,
    capability_report,
    manifest_value,
    profile_manifest,
)
from ..api import (
    AggregateBoundArtifact,
    AggregateProfileBoundProvider,
    ArchitectureArtifactIdentity,
    ArchitectureId,
    EvidenceRecord,
    GreedyTextExecutionShape,
    TraceBinding,
)


def _capabilities(
    plugin_id: str,
    *,
    trace_conditional: bool,
    trace_digest: Digest | None,
) -> CapabilityReport:
    kind = ArtifactKind.CAPACITY_PROFILE
    if trace_conditional and trace_digest is None:
        bound_state = SupportState.CONDITIONAL
        bound_reason = "TRACE_DIGEST_NOT_BOUND"
        bound_detail = (
            "the profile is trace-conditional but the request does not identify "
            "the observed sparse-execution trace"
        )
    else:
        bound_state = SupportState.SUPPORTED
        bound_reason = None
        bound_detail = ""
    return capability_report(
        plugin_id,
        kind,
        (
            capability(
                Capability.STATIC_COMPILE,
                SupportState.SUPPORTED,
                kind,
                guarantee="assumption-scoped exact aggregate region counts",
                evidence=EvidenceStatus.ASSUMPTION_SCOPED,
            ),
            capability(
                Capability.STATIC_PARTITION,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_PROTOCOL_REPLAY_PARTITION",
                detail=(
                    "counted capacity classes have no concrete replay or "
                    "verification-unit members"
                ),
            ),
            capability(
                Capability.STATIC_BOUND,
                bound_state,
                kind,
                guarantee="certified profile self-cut relaxation",
                evidence=EvidenceStatus.CERTIFIED,
                reason_code=bound_reason,
                detail=bound_detail,
            ),
            capability(
                Capability.EXECUTE,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_INDEXED_WIRING_OR_EXECUTABLE_RELATIONS",
                detail="the profile has no scalar gates, wiring, or value evaluator",
            ),
            capability(
                Capability.VERIFY,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_INDEXED_WIRING_OR_EXECUTABLE_RELATIONS",
                detail="aggregate regions cannot be used as protocol gates",
            ),
            capability(
                Capability.HIDDEN_STRUCTURE,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_CONCRETE_STRUCTURE",
                detail="the profile describes aggregate counts rather than a circuit",
            ),
        ),
    )


def build_aggregate_artifact(
    *,
    architecture_id: ArchitectureId,
    plugin_id: str,
    plugin_version: str,
    execution_shape: GreedyTextExecutionShape,
    configuration: object,
    numerical_profile_id: str,
    profile: ModelCapacityProfile,
    source: str,
    trace_conditional: bool = False,
    trace_digest: Digest | None = None,
) -> AggregateBoundArtifact:
    """Build a discriminated aggregate result without fabricating structure."""

    if profile.prompt_tokens != execution_shape.prompt_tokens:
        raise ValueError("profile prompt length disagrees with execution shape")
    if profile.generated_tokens != execution_shape.generated_tokens:
        raise ValueError("profile generation length disagrees with execution shape")
    weighted = weighted_partition_from_capacity_profile(profile)
    provider = AggregateProfileBoundProvider(profile, weighted)
    trace_binding = (
        TraceBinding.TRACE_BOUND
        if trace_conditional and trace_digest is not None
        else (
            TraceBinding.UNBOUND_TRACE_CONDITIONAL
            if trace_conditional
            else TraceBinding.NOT_APPLICABLE
        )
    )
    request_manifest: dict[str, JSONValue] = {
        "configuration": manifest_value(configuration),
        "execution_shape": execution_shape.manifest,
        "numerical_profile_id": numerical_profile_id,
        "trace_digest": trace_digest,
    }
    representation_manifest: dict[str, JSONValue] = {
        "profile": profile_manifest(profile),
        "self_cut_partition": {
            "certificate": weighted.certificate,
            "classes": [
                {
                    "aggregate_capacity": (
                        None
                        if gate_class.aggregate_capacity is None
                        else {
                            "denominator": (
                                gate_class.aggregate_capacity.multiplier.denominator
                            ),
                            "numerator": (
                                gate_class.aggregate_capacity.multiplier.numerator
                            ),
                        }
                    ),
                    "description": gate_class.description,
                    "gate_count": gate_class.gate_count,
                    "id": gate_class.id,
                    "singleton_capacity": {
                        "denominator": (
                            gate_class.singleton_capacity.multiplier.denominator
                        ),
                        "numerator": (
                            gate_class.singleton_capacity.multiplier.numerator
                        ),
                    },
                    "source_class_ids": list(gate_class.source_class_ids),
                }
                for gate_class in weighted.classes
            ],
            "output_frontier": {
                "denominator": weighted.output_frontier.multiplier.denominator,
                "numerator": weighted.output_frontier.multiplier.numerator,
            },
        },
        "trace_binding": trace_binding.value,
    }
    identity = ArchitectureArtifactIdentity.build(
        architecture_id=architecture_id,
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        artifact_kind=ArtifactKind.CAPACITY_PROFILE,
        request_manifest=request_manifest,
        representation_manifest=representation_manifest,
    )
    evidence = [
        EvidenceRecord(
            code="aggregate-region-counts",
            claim=ClaimStatus.CONDITIONAL,
            evidence=EvidenceStatus.ASSUMPTION_SCOPED,
            detail=(
                "region gate counts are exact integers under the builder's "
                "declared execution semantics"
            ),
            source=source,
        ),
        EvidenceRecord(
            code="profile-self-cut",
            claim=ClaimStatus.CERTIFIED_UPPER,
            evidence=EvidenceStatus.CERTIFIED,
            detail=(
                "attacked gates form a valid self-cut and the designated "
                "output frontier is an independent valid cap"
            ),
            source="weighted_partition_from_capacity_profile",
        ),
    ]
    if trace_conditional:
        evidence.append(
            EvidenceRecord(
                code="trace-binding",
                claim=ClaimStatus.CONDITIONAL,
                evidence=(
                    EvidenceStatus.NONE
                    if trace_digest is None
                    else EvidenceStatus.ASSUMPTION_SCOPED
                ),
                detail=(
                    "no observed sparse-execution trace digest is bound"
                    if trace_digest is None
                    else (
                        "the digest binds profile provenance to a named trace; "
                        "it does not validate or execute that trace"
                    )
                ),
                source=source,
            )
        )
    return AggregateBoundArtifact(
        architecture_id=architecture_id,
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        identity=identity,
        capabilities=_capabilities(
            plugin_id,
            trace_conditional=trace_conditional,
            trace_digest=trace_digest,
        ),
        execution_shape=execution_shape,
        profile=profile,
        weighted_partition=weighted,
        bound_provider=provider,
        trace_binding=trace_binding,
        trace_digest=trace_digest,
        assumptions=assumption_records(
            profile.assumptions,
            source=source,
            prefix=architecture_id.value,
        ),
        evidence=tuple(evidence),
    )
