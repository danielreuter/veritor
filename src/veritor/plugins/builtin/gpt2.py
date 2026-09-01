"""Exact structural GPT-2 plug-in.

This adapter deliberately exposes the existing indexed computed-source
projection, not a protocol circuit.  In particular it does not add ordered
operands, executable relations, replay boundaries, or runtime validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from circuit_cut_analysis.capacity import LogCardinality, sum_capacities
from circuit_cut_analysis.indexed import GateRef
from circuit_cut_analysis.models.gpt2 import GPT2_SMALL, GPT2Config
from circuit_cut_analysis.models.gpt2_capacity_oracle import (
    GPT2StructuralCapacityOracle,
)
from circuit_cut_analysis.models.gpt2_circuit import (
    GPT2IndexedCircuit,
    build_gpt2_indexed_circuit,
)
from circuit_cut_analysis.models.gpt2_gate_classes import (
    GPT2ClassGranularity,
    GPT2GateClassCatalog,
    build_gpt2_gate_class_catalog,
)
from circuit_cut_analysis.models.gpt2_partition import (
    lifted_certificate_reasons,
)
from circuit_cut_analysis.profiles import (
    VLLM_FP16_REFERENCE,
    ServingProfile,
)
from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    ClaimStatus,
    EvidenceStatus,
    JSONValue,
    SupportState,
    Unsupported,
)

from .._common import (
    assumption_records,
    capability,
    capability_report,
    manifest_value,
)
from ..api import (
    ArchitectureArtifactIdentity,
    ArchitectureId,
    CapacityBoundEvidence,
    CapacityClaimKind,
    EvidenceRecord,
    GreedyTextExecutionShape,
    IndexedStructureArtifact,
)
from ..ranked import GateRefIndexedDomain

PLUGIN_ID = "veritor.plugins.builtin.gpt2"
PLUGIN_VERSION = "1"
GPT2_ARCHITECTURE_ID = ArchitectureId.GPT2


@dataclass(frozen=True, slots=True)
class GPT2CompileRequest:
    """Shape, architecture dimensions, and numerical boundary profile."""

    execution_shape: GreedyTextExecutionShape = field(
        default_factory=GreedyTextExecutionShape
    )
    config: GPT2Config = GPT2_SMALL
    profile: ServingProfile = VLLM_FP16_REFERENCE
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.GPT2,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        if not isinstance(self.profile, ServingProfile):
            raise TypeError("profile must be ServingProfile")

    @property
    def shape(self) -> GreedyTextExecutionShape:
        return self.execution_shape


@dataclass(frozen=True, slots=True)
class GPT2CapacityBoundProvider:
    """Structural oracle and probability-class factories for GPT-2."""

    indexed: GPT2IndexedCircuit
    lifted_reasons: tuple[str, ...]
    claim_kind: CapacityClaimKind = field(
        init=False,
        default=CapacityClaimKind.CERTIFIED_INTERVAL,
    )

    @property
    def supported_claim_kinds(self) -> tuple[CapacityClaimKind, ...]:
        return (
            CapacityClaimKind.EXACT,
            CapacityClaimKind.CERTIFIED_INTERVAL,
        )

    @property
    def output_frontier(self) -> LogCardinality:
        circuit = self.indexed.circuit
        return sum_capacities(
            circuit.families[ref.family].capacity for ref in self.indexed.output_refs
        )

    @property
    def catalog_available(self) -> bool:
        return not self.lifted_reasons

    @property
    def gate_class_catalog_conditions(self) -> tuple[str, ...]:
        return self.lifted_reasons

    def structural_oracle(
        self,
        *,
        max_exact_gates: int = 200_000,
        max_exact_edges: int = 2_000_000,
    ) -> GPT2StructuralCapacityOracle:
        return GPT2StructuralCapacityOracle(
            self.indexed,
            max_exact_gates=max_exact_gates,
            max_exact_edges=max_exact_edges,
        )

    structural_oracle_factory = structural_oracle

    def gate_class_catalog(
        self,
        *,
        granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
        position_bands: int = 8,
    ) -> GPT2GateClassCatalog | Unsupported:
        if self.lifted_reasons:
            return Unsupported(
                capability=Capability.STATIC_PARTITION,
                plugin_id=PLUGIN_ID,
                reason_code="LIFTED_CERTIFICATE_CONDITIONS_UNMET",
                detail="; ".join(self.lifted_reasons),
                artifact_kind=ArtifactKind.STRUCTURAL_CIRCUIT,
            )
        return build_gpt2_gate_class_catalog(
            self.indexed,
            granularity=granularity,
            position_bands=position_bands,
        )

    gate_class_catalog_factory = gate_class_catalog

    def require_gate_class_catalog(
        self,
        *,
        granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
        position_bands: int = 8,
    ) -> GPT2GateClassCatalog:
        result = self.gate_class_catalog(
            granularity=granularity,
            position_bands=position_bands,
        )
        if isinstance(result, Unsupported):
            raise result.as_error()
        return result

    def evaluate(
        self,
        attack: Iterable[GateRef],
        *,
        max_exact_gates: int = 200_000,
        max_exact_edges: int = 2_000_000,
    ) -> CapacityBoundEvidence:
        evaluation = self.structural_oracle(
            max_exact_gates=max_exact_gates,
            max_exact_edges=max_exact_edges,
        ).evaluate(attack)
        claim_kind = (
            CapacityClaimKind.EXACT
            if evaluation.is_exact
            else CapacityClaimKind.CERTIFIED_INTERVAL
        )
        return CapacityBoundEvidence(
            lower_bound=evaluation.lower_bound,
            upper_bound=evaluation.upper_bound,
            claim_kind=claim_kind,
            method=evaluation.method,
            certificate=(
                "exact indexed corridor cut"
                if evaluation.is_exact
                else "certified lower witness and downstream-cut upper bound"
            ),
            assumptions=self.indexed.profile.assumptions,
            cut_gate_ids=evaluation.cut_gate_ids,
        )


def _request_manifest(request: GPT2CompileRequest) -> dict[str, JSONValue]:
    return {
        "architecture": manifest_value(request.config),
        "execution_shape": request.execution_shape.manifest,
        "numerical_profile": manifest_value(request.profile),
    }


def _representation_manifest(
    indexed: GPT2IndexedCircuit,
    gate_domain: GateRefIndexedDomain,
    computed_gate_domain: GateRefIndexedDomain,
) -> dict[str, JSONValue]:
    circuit = indexed.circuit
    return {
        "computed_gate_domain_digest": computed_gate_domain.identity_digest,
        "edge_rules": [
            {
                "name": rule.name,
                "source_family": rule.source_family,
                "target_family": rule.target_family,
            }
            for rule in circuit.edge_rules
        ],
        "families": [
            {
                "capacity_cardinality": family.capacity.cardinality,
                "count": family.count,
                "index_names": list(family.index_names),
                "name": family.name,
                "operation": family.op,
                "primitive": family.primitive,
                "tags": list(family.tags),
            }
            for family in circuit.families.values()
        ],
        "gate_domain_digest": gate_domain.identity_digest,
        "ordered_outputs": [
            {"family": ref.family, "index": list(ref.index)}
            for ref in indexed.output_refs
        ],
        "primitive_counts": dict(circuit.primitive_counts),
    }


def _capabilities(
    lifted_reasons: tuple[str, ...],
) -> CapabilityReport:
    kind = ArtifactKind.STRUCTURAL_CIRCUIT
    if lifted_reasons:
        bound_state = SupportState.CONDITIONAL
        bound_reason_code = "LIFTED_CERTIFICATE_CONDITIONS_UNMET"
        bound_detail = "; ".join(lifted_reasons)
    else:
        bound_state = SupportState.SUPPORTED
        bound_reason_code = None
        bound_detail = ""
    return capability_report(
        PLUGIN_ID,
        kind,
        (
            capability(
                Capability.STATIC_COMPILE,
                SupportState.SUPPORTED,
                kind,
                guarantee="exact indexed computed-source projection",
                evidence=EvidenceStatus.BY_CONSTRUCTION,
            ),
            capability(
                Capability.STATIC_PARTITION,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_PROTOCOL_REPLAY_PARTITION",
                detail=(
                    "gate-class catalogs are capacity-analysis summaries, not "
                    "replay or verification partitions"
                ),
            ),
            capability(
                Capability.STATIC_BOUND,
                bound_state,
                kind,
                guarantee=(
                    "exact indexed corridor cut or certified structural interval"
                ),
                evidence=EvidenceStatus.CERTIFIED,
                reason_code=bound_reason_code,
                detail=bound_detail,
            ),
            capability(
                Capability.EXECUTE,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_EXECUTABLE_RELATIONS",
                detail=(
                    "the projection has no trusted weights, value codec, rounding "
                    "semantics, or ordered operand relation"
                ),
            ),
            capability(
                Capability.VERIFY,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_EXECUTABLE_RELATIONS",
                detail="structural metadata cannot produce a protocol proof",
            ),
            capability(
                Capability.HIDDEN_STRUCTURE,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="STRUCTURE_IS_PUBLIC",
                detail="the indexed family and edge-rule descriptors are public",
            ),
        ),
    )


def compile_gpt2(
    request: GPT2CompileRequest | None = None,
) -> IndexedStructureArtifact:
    """Build the exact indexed descriptor without global partition reports."""

    selected = GPT2CompileRequest() if request is None else request
    if not isinstance(selected, GPT2CompileRequest):
        raise TypeError("GPT-2 requires GPT2CompileRequest")
    shape = selected.execution_shape
    indexed = build_gpt2_indexed_circuit(
        shape.prompt_tokens,
        shape.generated_tokens,
        config=selected.config,
        profile=selected.profile,
    )
    gate_domain = GateRefIndexedDomain(indexed.circuit)
    computed_gate_domain = GateRefIndexedDomain(
        indexed.circuit,
        computed_only=True,
    )
    reasons = lifted_certificate_reasons(indexed)
    provider = GPT2CapacityBoundProvider(indexed, reasons)
    identity = ArchitectureArtifactIdentity.build(
        architecture_id=ArchitectureId.GPT2,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        artifact_kind=ArtifactKind.STRUCTURAL_CIRCUIT,
        request_manifest=_request_manifest(selected),
        representation_manifest=_representation_manifest(
            indexed,
            gate_domain,
            computed_gate_domain,
        ),
    )
    assumptions = (
        "The artifact is an edge-complete computed-source dependency projection.",
        "Fixed weights, constants, prompt token IDs, and position embeddings are literals or inputs, not computed vertices.",
        "Batch-one fixed-horizon greedy generation produces all requested token IDs.",
        "The final generated token is designated as an output and is not forwarded.",
        "Embedding lookup mux internals, casts, loads, stores, indexing, and control are excluded.",
        "Structural primitive names and adjacency do not define executable local semantics.",
        "No checkpoint weights are loaded and no runtime implementation is validated.",
        *selected.profile.assumptions,
    )
    circuit = indexed.circuit
    return IndexedStructureArtifact(
        architecture_id=ArchitectureId.GPT2,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        identity=identity,
        capabilities=_capabilities(reasons),
        execution_shape=shape,
        indexed=indexed,
        ordered_output_refs=indexed.output_refs,
        gate_domain=gate_domain,
        computed_gate_domain=computed_gate_domain,
        bound_provider=provider,
        gate_count=circuit.gate_count,
        computed_gate_count=circuit.computed_gate_count,
        primitive_gate_count=circuit.primitive_gate_count,
        gate_family_count=len(circuit.families),
        edge_rule_count=len(circuit.edge_rules),
        assumptions=assumption_records(
            assumptions,
            source="circuit_cut_analysis.models.gpt2_circuit",
            prefix="gpt2",
        ),
        evidence=(
            EvidenceRecord(
                code="indexed-structure",
                claim=ClaimStatus.EXACT,
                evidence=EvidenceStatus.BY_CONSTRUCTION,
                detail=(
                    "gate families and bidirectional index relations exactly "
                    "represent the declared computed-source projection"
                ),
                source="build_gpt2_indexed_circuit",
            ),
            EvidenceRecord(
                code="capacity-oracle",
                claim=ClaimStatus.CONDITIONAL,
                evidence=EvidenceStatus.CERTIFIED,
                detail=(
                    "capacity is exact when corridor expansion succeeds and is "
                    "otherwise a certified interval when lifted conditions hold"
                ),
                source="GPT2StructuralCapacityOracle",
            ),
        ),
    )


compile_gpt2_structure = compile_gpt2


@dataclass(frozen=True, slots=True)
class GPT2Plugin:
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.GPT2,
    )
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> GPT2CompileRequest:
        return GPT2CompileRequest()

    def compile(self, request: object | None = None) -> IndexedStructureArtifact:
        if request is not None and not isinstance(request, GPT2CompileRequest):
            raise TypeError("GPT-2 requires GPT2CompileRequest")
        return compile_gpt2(request)


GPT2_PLUGIN = GPT2Plugin()
