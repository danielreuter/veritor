"""Executable built-in plug-in for the DemoG call-DAG constructor.

``DemoG`` is an untrusted memoized constructor whose only trusted output is
the canonical call-DAG document decoded by :mod:`veritor.compile`.  The
constructor, its request types, and the plug-in wrapper all live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from veritor.compile import (
    DEFAULT_REPLAY_POLICY,
    DEFAULT_VERIFICATION_POLICY,
    CompilationLimits,
    PartitionPolicy,
    Producer,
    ProducerDefinition,
    ProducerError,
    Wire,
    compile_call_dag,
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


@dataclass(frozen=True, slots=True)
class DotRequest:
    accumulator: int
    values: tuple[int, ...]
    weights: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.values)

    def cells(self) -> tuple[int, ...]:
        if len(self.values) != len(self.weights):
            raise ProducerError("dot-product values and weights have different lengths")
        return (self.accumulator, *self.values, *self.weights)


@dataclass(frozen=True, slots=True)
class BatchInput:
    requests: tuple[DotRequest, ...]

    def cells(self) -> tuple[int, ...]:
        return tuple(cell for request in self.requests for cell in request.cells())


class DemoG:
    """A memoized demo constructor whose emitted bytes remain pure."""

    def __init__(self, cell_bits: int = 8) -> None:
        self.cell_bits = cell_bits
        self.producer = Producer(cell_bits)

        @self.producer.gate(name="add")
        def add(left: int, right: int) -> int:
            return left + right

        @self.producer.gate(name="mul")
        def mul(left: int, right: int) -> int:
            return left * right

        self.add = add
        self.mul = mul

        @self.producer.circuit(key=("mac",), input_count=3)
        def mac(accumulator: Wire, value: Wire, weight: Wire) -> Wire:
            return add(accumulator, mul(value, weight))

        self.mac = mac

    def dot(self, length: int) -> ProducerDefinition:
        if type(length) is not int or length < 0:
            raise ProducerError("dot length must be a nonnegative integer")
        input_count = 1 + 2 * length

        @self.producer.circuit(key=("dot", length), input_count=input_count)
        def dot_definition(*inputs: Wire) -> Wire:
            accumulator = inputs[0]
            values = inputs[1 : 1 + length]
            weights = inputs[1 + length :]
            for value, weight in zip(values, weights, strict=True):
                accumulator = cast(Wire, self.mac(accumulator, value, weight))
            return accumulator

        return dot_definition

    def batch(self, lengths: tuple[int, ...]) -> ProducerDefinition:
        input_count = sum(1 + 2 * length for length in lengths)

        @self.producer.circuit(key=("batch", lengths), input_count=input_count)
        def batch_definition(*inputs: Wire) -> tuple[Wire, ...]:
            outputs: list[Wire] = []
            offset = 0
            for length in lengths:
                child = self.dot(length)
                child_input_count = 1 + 2 * length
                outputs.append(
                    cast(
                        Wire,
                        child(*inputs[offset : offset + child_input_count]),
                    )
                )
                offset += child_input_count
            return tuple(outputs)

        return batch_definition

    def __call__(self, x: object, a: bytes) -> bytes:
        if not isinstance(x, BatchInput):
            raise ProducerError("DemoG expects BatchInput")
        if type(a) is not bytes:
            raise ProducerError("DemoG advice must be bytes")
        for request in x.requests:
            request.cells()
        root = self.batch(tuple(request.length for request in x.requests))
        return self.producer.serialize(root)


def expected_dot_outputs(batch: BatchInput, cell_bits: int) -> tuple[int, ...]:
    mask = (1 << cell_bits) - 1
    outputs = []
    for request in batch.requests:
        accumulator = request.accumulator
        for value, weight in zip(request.values, request.weights, strict=True):
            accumulator = (accumulator + value * weight) & mask
        outputs.append(accumulator)
    return tuple(outputs)


def make_demo_request(
    length: int,
    seed: int,
    cell_bits: int = 8,
) -> DotRequest:
    mask = (1 << cell_bits) - 1
    values = tuple((seed + 3 * index + 1) & mask for index in range(length))
    weights = tuple((2 * seed + 5 * index + 1) & mask for index in range(length))
    return DotRequest(seed & mask, values, weights)


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
    artifact_identity = ArchitectureArtifactIdentity.build(
        architecture_id=ArchitectureId.DEMO_G,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        artifact_kind=ArtifactKind.EXECUTABLE_CIRCUIT,
        request_manifest=_request_manifest(selected),
        representation_manifest={
            "compiled_result_digest": compiled.identity.digest,
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
        capabilities=_capabilities(),
        compiled=compiled,
        public_inputs=selected.public_inputs,
        expected_outputs=selected.expected_outputs,
        bound_provider=DemoGCapacityBoundProvider(compiled.circuit),
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
                source="veritor.core.CompiledArtifact",
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
