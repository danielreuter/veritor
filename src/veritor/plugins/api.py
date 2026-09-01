"""Stable contracts shared by architecture plug-ins.

The three artifact classes in this module are intentionally disjoint:

* :class:`ProtocolCircuitArtifact` is a genuine executable ``(C, R, V)``;
* :class:`IndexedStructureArtifact` is exact structural metadata only; and
* :class:`AggregateBoundArtifact` is a counted capacity model with no circuit.

Keeping those representations separate prevents aggregate accounting from
accidentally acquiring protocol-circuit semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.indexed import GateRef
from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.models.gpt2_capacity_oracle import (
    GPT2StructuralCapacityOracle,
)
from circuit_cut_analysis.models.gpt2_circuit import GPT2IndexedCircuit
from circuit_cut_analysis.models.gpt2_gate_classes import (
    GPT2ClassGranularity,
    GPT2GateClassCatalog,
)
from circuit_cut_analysis.weighted_sampling import (
    WeightedGateClassPartition,
    capacity_upper_bound_for_counts,
)
from veritor.compile import CallDagCircuit
from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    ClaimStatus,
    CompiledArtifact,
    CompiledResultIdentity,
    Digest,
    EvidenceStatus,
    IndexedDomain,
    JSONValue,
    ReplayPartition,
    Unsupported,
    VerificationPartition,
    identity_digest,
)


class ArchitectureId(StrEnum):
    """Stable identifiers for the built-in architecture families."""

    DEMO_G = "demo-g"
    MATMUL = "matmul"
    GPT2 = "gpt2"
    KIMI_K3 = "kimi-k3"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"
    INKLING = "inkling"


class DecodingMode(StrEnum):
    """The only text-generation topology currently represented."""

    GREEDY_ARGMAX = "greedy-argmax"


@dataclass(frozen=True, slots=True)
class GreedyTextExecutionShape:
    """Fixed batch-one, fixed-horizon greedy text execution shape."""

    prompt_tokens: int = 100
    generated_tokens: int = 100
    batch_size: int = 1
    decoding_mode: DecodingMode = DecodingMode.GREEDY_ARGMAX
    fixed_horizon: bool = True
    final_generated_forward: bool = False
    eos_termination: bool = False
    text_only: bool = True

    def __post_init__(self) -> None:
        if type(self.prompt_tokens) is not int or self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer")
        if type(self.generated_tokens) is not int or self.generated_tokens <= 0:
            raise ValueError("generated_tokens must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("architecture profiles support batch_size=1 only")
        if self.decoding_mode is not DecodingMode.GREEDY_ARGMAX:
            raise ValueError("architecture profiles support greedy argmax only")
        for field_name in (
            "fixed_horizon",
            "final_generated_forward",
            "eos_termination",
            "text_only",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if not self.fixed_horizon:
            raise ValueError("architecture profiles require a fixed output horizon")
        if self.final_generated_forward:
            raise ValueError("the final generated token is not forwarded")
        if self.eos_termination:
            raise ValueError("EOS termination must be disabled in the fixed topology")
        if not self.text_only:
            raise ValueError("architecture profiles currently cover text only")

    @property
    def processed_positions(self) -> int:
        return self.prompt_tokens + self.generated_tokens - 1

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        first = self.prompt_tokens - 1
        return tuple(range(first, first + self.generated_tokens))

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "batch_size": self.batch_size,
            "decoding_mode": self.decoding_mode.value,
            "eos_termination": self.eos_termination,
            "final_generated_forward": self.final_generated_forward,
            "fixed_horizon": self.fixed_horizon,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens": self.prompt_tokens,
            "text_only": self.text_only,
        }


FixedGreedyTextShape = GreedyTextExecutionShape


@dataclass(frozen=True, slots=True)
class AssumptionRecord:
    """One stable, auditable assumption attached to an artifact."""

    code: str
    statement: str
    source: str

    def __post_init__(self) -> None:
        for field_name in ("code", "statement", "source"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Evidence strength and claim strength for one artifact assertion."""

    code: str
    claim: ClaimStatus
    evidence: EvidenceStatus
    detail: str
    source: str

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code.strip():
            raise ValueError("evidence code must be a nonempty string")
        if type(self.detail) is not str or not self.detail.strip():
            raise ValueError("evidence detail must be a nonempty string")
        if type(self.source) is not str or not self.source.strip():
            raise ValueError("evidence source must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ArchitectureArtifactIdentity:
    """Deterministic identity of one plug-in request and representation."""

    architecture_id: ArchitectureId
    plugin_id: str
    plugin_version: str
    artifact_kind: ArtifactKind
    request_digest: Digest
    representation_digest: Digest
    digest: Digest

    @classmethod
    def build(
        cls,
        *,
        architecture_id: ArchitectureId,
        plugin_id: str,
        plugin_version: str,
        artifact_kind: ArtifactKind,
        request_manifest: JSONValue,
        representation_manifest: JSONValue,
    ) -> ArchitectureArtifactIdentity:
        if type(plugin_id) is not str or not plugin_id.strip():
            raise ValueError("plugin_id must be a nonempty string")
        if type(plugin_version) is not str or not plugin_version.strip():
            raise ValueError("plugin_version must be a nonempty string")
        request_digest = identity_digest(
            "veritor/plugins/compile-request/v1",
            request_manifest,
        )
        representation_digest = identity_digest(
            "veritor/plugins/representation/v1",
            representation_manifest,
        )
        digest = identity_digest(
            "veritor/plugins/artifact/v1",
            {
                "architecture_id": architecture_id.value,
                "artifact_kind": artifact_kind.value,
                "plugin_id": plugin_id,
                "plugin_version": plugin_version,
                "representation_digest": representation_digest,
                "request_digest": request_digest,
            },
        )
        return cls(
            architecture_id=architecture_id,
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            artifact_kind=artifact_kind,
            request_digest=request_digest,
            representation_digest=representation_digest,
            digest=digest,
        )


class CapacityClaimKind(StrEnum):
    """Strength of the capacity provider attached to an artifact."""

    EXACT = "exact"
    CERTIFIED_INTERVAL = "certified-interval"
    PROFILE_SELF_CUT_RELAXATION = "profile-self-cut-relaxation"

    EXACT_CIRCUIT = "exact"
    CERTIFIED_CIRCUIT_INTERVAL = "certified-interval"


@dataclass(frozen=True, slots=True)
class CapacityBoundEvidence:
    """Guarantee-carrying result from a plug-in bound provider."""

    lower_bound: LogCardinality
    upper_bound: LogCardinality
    claim_kind: CapacityClaimKind
    method: str
    certificate: str
    assumptions: tuple[str, ...] = ()
    cut_gate_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.lower_bound > self.upper_bound:
            raise ValueError("capacity lower bound exceeds upper bound")
        if type(self.method) is not str or not self.method.strip():
            raise ValueError("capacity method must be nonempty")
        if type(self.certificate) is not str or not self.certificate.strip():
            raise ValueError("capacity certificate must be nonempty")

    @property
    def is_exact(self) -> bool:
        return self.lower_bound == self.upper_bound


@runtime_checkable
class CapacityBoundProvider[AttackT](Protocol):
    """Common surface consumed by a later analysis facade."""

    @property
    def claim_kind(self) -> CapacityClaimKind: ...

    @property
    def output_frontier(self) -> LogCardinality: ...

    def evaluate(self, attack: AttackT) -> CapacityBoundEvidence: ...


@dataclass(frozen=True, slots=True)
class AggregateProfileBoundProvider:
    """Certified class-count self-cut relaxation for aggregate profiles."""

    profile: ModelCapacityProfile
    partition: WeightedGateClassPartition
    claim_kind: CapacityClaimKind = field(
        init=False,
        default=CapacityClaimKind.PROFILE_SELF_CUT_RELAXATION,
    )

    def __post_init__(self) -> None:
        if self.partition.model_id != self.profile.model_id:
            raise ValueError("profile and weighted partition model IDs differ")
        if self.partition.total_gate_count != self.profile.total_gate_count:
            raise ValueError("profile and weighted partition gate counts differ")

    @property
    def output_frontier(self) -> LogCardinality:
        return self.partition.output_frontier

    @property
    def weighted_partition(self) -> WeightedGateClassPartition:
        return self.partition

    def evaluate(self, attack: Sequence[int]) -> CapacityBoundEvidence:
        counts = tuple(attack)
        upper = capacity_upper_bound_for_counts(self.partition, counts)
        return CapacityBoundEvidence(
            lower_bound=LogCardinality.zero(),
            upper_bound=upper,
            claim_kind=self.claim_kind,
            method="profile-class-count-self-cut",
            certificate=self.partition.certificate,
            assumptions=self.profile.assumptions,
        )


class TraceBinding(StrEnum):
    """Whether a trace-conditional profile names a concrete trace."""

    NOT_APPLICABLE = "not-applicable"
    UNBOUND_TRACE_CONDITIONAL = "unbound-trace-conditional"
    TRACE_BOUND = "trace-bound"


@runtime_checkable
class GPT2CapacityProviderSurface(Protocol):
    """Factories exposed by the GPT-2 structural plug-in."""

    @property
    def claim_kind(self) -> CapacityClaimKind: ...

    @property
    def output_frontier(self) -> LogCardinality: ...

    @property
    def catalog_available(self) -> bool: ...

    def structural_oracle(
        self,
        *,
        max_exact_gates: int = 200_000,
        max_exact_edges: int = 2_000_000,
    ) -> GPT2StructuralCapacityOracle: ...

    def gate_class_catalog(
        self,
        *,
        granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
        position_bands: int = 8,
    ) -> GPT2GateClassCatalog | Unsupported: ...

    def evaluate(
        self,
        attack: Iterable[GateRef],
        *,
        max_exact_gates: int = 200_000,
        max_exact_edges: int = 2_000_000,
    ) -> CapacityBoundEvidence: ...


@runtime_checkable
class ExecutableCapacityProviderSurface(Protocol):
    """Exact finite-DAG bound surface exposed by executable artifacts."""

    @property
    def claim_kind(self) -> CapacityClaimKind: ...

    @property
    def output_frontier(self) -> LogCardinality: ...

    def evaluate(self, attack: Sequence[int]) -> CapacityBoundEvidence: ...


DemoGCapacityProviderSurface = ExecutableCapacityProviderSurface


def _unsupported(
    *,
    capability: Capability,
    plugin_id: str,
    artifact_kind: ArtifactKind,
    reason_code: str,
    detail: str,
) -> Unsupported:
    return Unsupported(
        capability=capability,
        plugin_id=plugin_id,
        reason_code=reason_code,
        detail=detail,
        artifact_kind=artifact_kind,
    )


@dataclass(frozen=True, slots=True)
class ProtocolCircuitArtifact:
    """Genuine executable protocol compile holding a :class:`CompiledArtifact`."""

    architecture_id: ArchitectureId
    plugin_id: str
    plugin_version: str
    identity: ArchitectureArtifactIdentity
    capabilities: CapabilityReport
    compiled: CompiledArtifact
    public_inputs: tuple[int, ...]
    expected_outputs: tuple[int, ...]
    bound_provider: ExecutableCapacityProviderSurface
    assumptions: tuple[AssumptionRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    runtime_validated: bool = False
    artifact_kind: ArtifactKind = field(
        init=False,
        default=ArtifactKind.EXECUTABLE_CIRCUIT,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.compiled, CompiledArtifact):
            raise TypeError("compiled must be a CompiledArtifact")
        if not isinstance(self.compiled.circuit, CallDagCircuit):
            raise TypeError("executable plug-in artifacts require a CallDagCircuit")

    @property
    def kind(self) -> ArtifactKind:
        return self.artifact_kind

    @property
    def compiled_identity(self) -> CompiledResultIdentity:
        return self.compiled.identity

    @property
    def circuit(self) -> CallDagCircuit:
        return self.compiled.circuit  # type: ignore[return-value]

    @property
    def replay_partition(self) -> ReplayPartition:
        return self.compiled.replay

    @property
    def verification_partition(self) -> VerificationPartition:
        return self.compiled.verification

    @property
    def assumption_texts(self) -> tuple[str, ...]:
        return tuple(record.statement for record in self.assumptions)

    def replay(self) -> tuple[ReplayPartition, VerificationPartition]:
        return self.replay_partition, self.verification_partition

    def execution_access(self) -> CallDagCircuit:
        return self.circuit

    def execute(self, inputs: Sequence[int] | None = None) -> tuple[int, ...]:
        selected = self.public_inputs if inputs is None else tuple(inputs)
        return self.circuit.evaluate(selected)

    def verification_access(self) -> CompiledArtifact:
        """Return the trusted compiled artifact consumed by the verifier."""

        return self.compiled

    def verify(self) -> CompiledArtifact:
        """Compatibility spelling for capability access, not orchestration."""

        return self.verification_access()


@dataclass(frozen=True, slots=True)
class IndexedStructureArtifact:
    """Exact GPT-2 indexed structure without protocol gate semantics."""

    architecture_id: ArchitectureId
    plugin_id: str
    plugin_version: str
    identity: ArchitectureArtifactIdentity
    capabilities: CapabilityReport
    execution_shape: GreedyTextExecutionShape
    indexed: GPT2IndexedCircuit
    ordered_output_refs: tuple[GateRef, ...]
    gate_domain: IndexedDomain[GateRef]
    computed_gate_domain: IndexedDomain[GateRef]
    bound_provider: GPT2CapacityProviderSurface
    gate_count: int
    computed_gate_count: int
    primitive_gate_count: int
    gate_family_count: int
    edge_rule_count: int
    assumptions: tuple[AssumptionRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    runtime_validated: bool = False
    artifact_kind: ArtifactKind = field(
        init=False,
        default=ArtifactKind.STRUCTURAL_CIRCUIT,
    )

    @property
    def kind(self) -> ArtifactKind:
        return self.artifact_kind

    @property
    def shape(self) -> GreedyTextExecutionShape:
        return self.execution_shape

    @property
    def structure(self) -> GPT2IndexedCircuit:
        return self.indexed

    @property
    def structural_circuit(self) -> object:
        return self.indexed.circuit

    @property
    def output_refs(self) -> tuple[GateRef, ...]:
        return self.ordered_output_refs

    @property
    def assumption_texts(self) -> tuple[str, ...]:
        return tuple(record.statement for record in self.assumptions)

    def count_expanded_edges(
        self,
        *,
        max_gates: int,
        max_edges: int,
    ) -> int:
        count = 0
        for _edge in self.indexed.circuit.iter_edges(
            max_gates=max_gates,
            max_edges=max_edges,
        ):
            count += 1
        return count

    def structural_oracle(
        self,
        *,
        max_exact_gates: int = 200_000,
        max_exact_edges: int = 2_000_000,
    ) -> GPT2StructuralCapacityOracle:
        return self.bound_provider.structural_oracle(
            max_exact_gates=max_exact_gates,
            max_exact_edges=max_exact_edges,
        )

    def gate_class_catalog(
        self,
        *,
        granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
        position_bands: int = 8,
    ) -> GPT2GateClassCatalog | Unsupported:
        return self.bound_provider.gate_class_catalog(
            granularity=granularity,
            position_bands=position_bands,
        )

    def replay(self) -> Unsupported:
        return _unsupported(
            capability=Capability.STATIC_PARTITION,
            plugin_id=self.plugin_id,
            artifact_kind=self.artifact_kind,
            reason_code="NO_PROTOCOL_REPLAY_PARTITION",
            detail=(
                "GPT-2 probability classes are analysis classes, not replay "
                "or verification partitions"
            ),
        )

    replay_access = replay

    def execution_access(self) -> Unsupported:
        return _unsupported(
            capability=Capability.EXECUTE,
            plugin_id=self.plugin_id,
            artifact_kind=self.artifact_kind,
            reason_code="NO_EXECUTABLE_RELATIONS",
            detail=(
                "the indexed projection has structural adjacency but no ordered "
                "operands, trusted value codec, or executable local relations"
            ),
        )

    def execute(self, *_args: object, **_kwargs: object) -> Unsupported:
        return self.execution_access()

    def verification_access(self) -> Unsupported:
        return _unsupported(
            capability=Capability.VERIFY,
            plugin_id=self.plugin_id,
            artifact_kind=self.artifact_kind,
            reason_code="NO_EXECUTABLE_RELATIONS",
            detail="structural GPT-2 metadata cannot produce a protocol transcript",
        )

    def verify(self, *_args: object, **_kwargs: object) -> Unsupported:
        return self.verification_access()


@dataclass(frozen=True, slots=True)
class AggregateBoundArtifact:
    """Counted profile and self-cut provider; deliberately not a circuit."""

    architecture_id: ArchitectureId
    plugin_id: str
    plugin_version: str
    identity: ArchitectureArtifactIdentity
    capabilities: CapabilityReport
    execution_shape: GreedyTextExecutionShape
    profile: ModelCapacityProfile
    weighted_partition: WeightedGateClassPartition
    bound_provider: AggregateProfileBoundProvider
    trace_binding: TraceBinding
    trace_digest: Digest | None
    assumptions: tuple[AssumptionRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    runtime_validated: bool = False
    artifact_kind: ArtifactKind = field(
        init=False,
        default=ArtifactKind.CAPACITY_PROFILE,
    )

    @property
    def kind(self) -> ArtifactKind:
        return self.artifact_kind

    @property
    def shape(self) -> GreedyTextExecutionShape:
        return self.execution_shape

    @property
    def regions(self) -> tuple[CapacityRegion, ...]:
        return self.profile.regions

    @property
    def total_gate_count(self) -> int:
        return self.profile.total_gate_count

    @property
    def gate_count(self) -> int:
        return self.profile.total_gate_count

    @property
    def output_frontier(self) -> LogCardinality:
        return self.weighted_partition.output_frontier

    @property
    def assumption_texts(self) -> tuple[str, ...]:
        return tuple(record.statement for record in self.assumptions)

    def replay(self) -> Unsupported:
        return _unsupported(
            capability=Capability.STATIC_PARTITION,
            plugin_id=self.plugin_id,
            artifact_kind=self.artifact_kind,
            reason_code="NO_PROTOCOL_REPLAY_PARTITION",
            detail=(
                "counted probability classes have no concrete gate members, "
                "interiors, cross-unit reads, or replay boundary"
            ),
        )

    replay_access = replay

    def execution_access(self) -> Unsupported:
        return _unsupported(
            capability=Capability.EXECUTE,
            plugin_id=self.plugin_id,
            artifact_kind=self.artifact_kind,
            reason_code="NO_INDEXED_WIRING_OR_EXECUTABLE_RELATIONS",
            detail="the aggregate profile has neither scalar wiring nor local relations",
        )

    def execute(self, *_args: object, **_kwargs: object) -> Unsupported:
        return self.execution_access()

    def verification_access(self) -> Unsupported:
        return _unsupported(
            capability=Capability.VERIFY,
            plugin_id=self.plugin_id,
            artifact_kind=self.artifact_kind,
            reason_code="NO_INDEXED_WIRING_OR_EXECUTABLE_RELATIONS",
            detail="an aggregate capacity profile cannot be verified as a circuit",
        )

    def verify(self, *_args: object, **_kwargs: object) -> Unsupported:
        return self.verification_access()


type CompileResult = (
    ProtocolCircuitArtifact | IndexedStructureArtifact | AggregateBoundArtifact
)
ArchitectureCompileResult = CompileResult


@runtime_checkable
class ArchitecturePlugin(Protocol):
    """Common registry-facing plug-in protocol."""

    @property
    def architecture_id(self) -> ArchitectureId: ...

    @property
    def plugin_id(self) -> str: ...

    @property
    def plugin_version(self) -> str: ...

    def default_request(self) -> object: ...

    def compile(self, request: object | None = None) -> CompileResult: ...


type ArchitectureRegistry = Mapping[ArchitectureId, ArchitecturePlugin]
