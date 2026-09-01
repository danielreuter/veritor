"""Executable built-in plug-in for the production DemoG call-DAG compiler."""

from __future__ import annotations

from dataclasses import dataclass, field

from veritor.compile import (
    DEFAULT_REPLAY_POLICY,
    DEFAULT_VERIFICATION_POLICY,
    BatchInput,
    CompilationLimits,
    DemoG,
    PartitionPolicy,
    compile_call_dag,
    expected_dot_outputs,
    make_demo_request,
    make_word_kernel,
)
from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    ClaimStatus,
    EvidenceStatus,
    JSONValue,
    SupportState,
    validate_compiled_result,
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
    EvidenceRecord,
    ProtocolCircuitArtifact,
)
from ._call_dag_capacity import CallDagCapacityBoundProvider

PLUGIN_ID = "veritor.plugins.builtin.demo-g"
PLUGIN_VERSION = "1"
DEMO_G_ARCHITECTURE_ID = ArchitectureId.DEMO_G


def _default_batch() -> BatchInput:
    return BatchInput(
        (
            make_demo_request(2, 1, 8),
            make_demo_request(3, 2, 8),
        )
    )


@dataclass(frozen=True, slots=True)
class DemoGCompileRequest:
    """Inputs and identity-bound partition policy for DemoG."""

    batch: BatchInput = field(default_factory=_default_batch)
    advice: bytes = b""
    cell_bits: int = 8
    advice_bound_bits: int = 0
    replay_policy: PartitionPolicy = DEFAULT_REPLAY_POLICY
    verification_policy: PartitionPolicy = DEFAULT_VERIFICATION_POLICY
    replay_configuration: JSONValue | None = None
    verification_configuration: JSONValue | None = None
    limits: CompilationLimits | None = None
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.DEMO_G,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BatchInput):
            raise TypeError("batch must be a BatchInput")
        if type(self.advice) is not bytes:
            raise TypeError("advice must be bytes")
        if type(self.cell_bits) is not int or self.cell_bits <= 0:
            raise ValueError("cell_bits must be a positive integer")
        if type(self.advice_bound_bits) is not int or self.advice_bound_bits < 0:
            raise ValueError("advice_bound_bits must be a nonnegative integer")
        if len(self.advice) * 8 > self.advice_bound_bits:
            raise ValueError("advice exceeds advice_bound_bits")
        object.__setattr__(self, "replay_policy", PartitionPolicy(self.replay_policy))
        object.__setattr__(
            self,
            "verification_policy",
            PartitionPolicy(self.verification_policy),
        )
        if self.limits is not None and not isinstance(
            self.limits,
            CompilationLimits,
        ):
            raise TypeError("limits must be CompilationLimits or None")

    @property
    def public_inputs(self) -> tuple[int, ...]:
        return self.batch.cells()

    @property
    def expected_outputs(self) -> tuple[int, ...]:
        return expected_dot_outputs(self.batch, self.cell_bits)


def demo_public_inputs(
    value: DemoGCompileRequest | BatchInput,
) -> tuple[int, ...]:
    """Return the ordered public input cells used by DemoG."""

    return (
        value.public_inputs if isinstance(value, DemoGCompileRequest) else value.cells()
    )


def demo_expected_outputs(
    value: DemoGCompileRequest | BatchInput,
    cell_bits: int | None = None,
) -> tuple[int, ...]:
    """Return the expected modular dot-product outputs."""

    if isinstance(value, DemoGCompileRequest):
        if cell_bits is not None and cell_bits != value.cell_bits:
            raise ValueError("cell_bits disagrees with the compile request")
        return value.expected_outputs
    return expected_dot_outputs(value, 8 if cell_bits is None else cell_bits)


DemoGCapacityBoundProvider = CallDagCapacityBoundProvider


def _request_manifest(request: DemoGCompileRequest) -> dict[str, JSONValue]:
    return {
        "advice": manifest_value(request.advice),
        "advice_bound_bits": request.advice_bound_bits,
        "batch": manifest_value(request.batch),
        "cell_bits": request.cell_bits,
        "limits": manifest_value(request.limits),
        "replay_configuration": request.replay_configuration,
        "replay_policy": request.replay_policy.value,
        "verification_configuration": request.verification_configuration,
        "verification_policy": request.verification_policy.value,
    }


def _capabilities() -> CapabilityReport:
    kind = ArtifactKind.EXECUTABLE_CIRCUIT
    exact = EvidenceStatus.BY_CONSTRUCTION
    return capability_report(
        PLUGIN_ID,
        kind,
        (
            capability(
                Capability.STATIC_COMPILE,
                SupportState.SUPPORTED,
                kind,
                guarantee="validated executable call-DAG circuit",
                evidence=exact,
            ),
            capability(
                Capability.STATIC_PARTITION,
                SupportState.SUPPORTED,
                kind,
                guarantee="exact replay partition and verification refinement",
                evidence=exact,
            ),
            capability(
                Capability.STATIC_BOUND,
                SupportState.SUPPORTED,
                kind,
                guarantee="exact finite structural capacity oracle",
                evidence=EvidenceStatus.CERTIFIED,
            ),
            capability(
                Capability.EXECUTE,
                SupportState.SUPPORTED,
                kind,
                guarantee="trusted modular-word relation evaluation",
                evidence=exact,
            ),
            capability(
                Capability.VERIFY,
                SupportState.SUPPORTED,
                kind,
                guarantee="verification-ready executable (C, R, V) tuple",
                evidence=exact,
            ),
            capability(
                Capability.HIDDEN_STRUCTURE,
                SupportState.UNSUPPORTED,
                kind,
                guarantee="",
                evidence=EvidenceStatus.NONE,
                reason_code="NO_HIDING_PROOF_BACKEND",
                detail="DemoG exposes transparent structure and local relations",
            ),
        ),
    )


def compile_demo_g(
    request: DemoGCompileRequest | None = None,
) -> ProtocolCircuitArtifact:
    """Compile DemoG through :func:`veritor.compile.compile_call_dag`."""

    selected = DemoGCompileRequest() if request is None else request
    if not isinstance(selected, DemoGCompileRequest):
        raise TypeError("DemoG requires DemoGCompileRequest")
    constructor = DemoG(selected.cell_bits)
    kernel = make_word_kernel(selected.cell_bits, limits=selected.limits)
    compiled = compile_call_dag(
        kernel,
        constructor,
        selected.batch,
        selected.advice,
        input_cells=selected.public_inputs,
        advice_bound_bits=selected.advice_bound_bits,
        replay_policy=selected.replay_policy,
        verification_policy=selected.verification_policy,
        replay_configuration=selected.replay_configuration,
        verification_configuration=selected.verification_configuration,
    )
    compiled_identity = validate_compiled_result(*compiled)
    artifact_identity = ArchitectureArtifactIdentity.build(
        architecture_id=ArchitectureId.DEMO_G,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        artifact_kind=ArtifactKind.EXECUTABLE_CIRCUIT,
        request_manifest=_request_manifest(selected),
        representation_manifest={
            "compiled_result_digest": compiled_identity.digest,
            "expected_outputs": list(selected.expected_outputs),
            "public_inputs": list(selected.public_inputs),
        },
    )
    assumptions = (
        "DemoG uses fixed-width unsigned modular arithmetic.",
        "The trusted relation registry contains only modular add and multiply.",
        "Constructor code is untrusted; only its canonical data output is decoded.",
        "No model weights or external runtime are loaded.",
    )
    return ProtocolCircuitArtifact(
        architecture_id=ArchitectureId.DEMO_G,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        identity=artifact_identity,
        compiled_identity=compiled_identity,
        capabilities=_capabilities(),
        _protocol_tuple=compiled,
        public_inputs=selected.public_inputs,
        expected_outputs=selected.expected_outputs,
        bound_provider=DemoGCapacityBoundProvider(compiled[0]),
        assumptions=assumption_records(
            assumptions,
            source="veritor.compile.call_dag",
            prefix="demo-g",
        ),
        evidence=(
            EvidenceRecord(
                code="validated-call-dag",
                claim=ClaimStatus.EXACT,
                evidence=EvidenceStatus.BY_CONSTRUCTION,
                detail="trusted decoder validated the complete call-DAG closure",
                source="veritor.compile.compile_call_dag",
            ),
            EvidenceRecord(
                code="validated-partitions",
                claim=ClaimStatus.EXACT,
                evidence=EvidenceStatus.BY_CONSTRUCTION,
                detail="verification units exactly refine replay units",
                source="veritor.core.validate_compiled_result",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class DemoGPlugin:
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.DEMO_G,
    )
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> DemoGCompileRequest:
        return DemoGCompileRequest()

    def compile(self, request: object | None = None) -> ProtocolCircuitArtifact:
        if request is not None and not isinstance(request, DemoGCompileRequest):
            raise TypeError("DemoG requires DemoGCompileRequest")
        return compile_demo_g(request)


DEMO_G_PLUGIN = DemoGPlugin()
