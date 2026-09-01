"""Executable plug-in for shared-weight modular matrix multiplication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from veritor.compile import (
    CompilationLimits,
    MatmulWorkload,
    WordMatrix,
    compile_matmul_workload,
    expected_matmul_outputs,
)
from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    ClaimStatus,
    EvidenceStatus,
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

PLUGIN_ID = "veritor.plugins.builtin.matmul"
PLUGIN_VERSION = "1"
MATMUL_ARCHITECTURE_ID = ArchitectureId.MATMUL

_DEFAULT_WEIGHTS = (
    (1, 2),
    (3, 4),
    (5, 6),
)
_DEFAULT_ACTIVATIONS = (
    (
        (1, 2, 3),
        (4, 5, 6),
    ),
    ((7, 8, 9),),
)


@dataclass(frozen=True, slots=True, init=False)
class MatmulCompileRequest:
    """Public shared-weight modular matrix workload."""

    weights: WordMatrix
    activations: tuple[WordMatrix, ...]
    cell_bits: int
    limits: CompilationLimits | None
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.MATMUL,
    )
    _workload: MatmulWorkload = field(init=False, repr=False)

    def __init__(
        self,
        weights: Sequence[Sequence[int]] = _DEFAULT_WEIGHTS,
        activations: Sequence[Sequence[Sequence[int]]] = _DEFAULT_ACTIVATIONS,
        *,
        cell_bits: int = 8,
        limits: CompilationLimits | None = None,
    ) -> None:
        workload = MatmulWorkload(
            weights,
            activations,
            cell_bits=cell_bits,
        )
        if limits is not None and not isinstance(limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits or None")
        object.__setattr__(self, "weights", workload.weights)
        object.__setattr__(self, "activations", workload.activations)
        object.__setattr__(self, "cell_bits", workload.cell_bits)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "_workload", workload)
        object.__setattr__(self, "architecture_id", ArchitectureId.MATMUL)

    @property
    def workload(self) -> MatmulWorkload:
        return self._workload

    @property
    def public_inputs(self) -> tuple[int, ...]:
        return self.workload.public_inputs

    @property
    def expected_outputs(self) -> tuple[int, ...]:
        return expected_matmul_outputs(self.workload)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return self.workload.weight_shape

    @property
    def activation_shapes(self) -> tuple[tuple[int, int], ...]:
        return self.workload.activation_shapes

    @property
    def output_shapes(self) -> tuple[tuple[int, int], ...]:
        return self.workload.output_shapes


def matmul_public_inputs(
    value: MatmulCompileRequest | MatmulWorkload,
) -> tuple[int, ...]:
    """Return shared weights once followed by row-major activations."""

    return value.public_inputs


def matmul_expected_outputs(
    value: MatmulCompileRequest | MatmulWorkload,
) -> tuple[int, ...]:
    """Return activation-ordered, row-major modular outputs."""

    workload = value.workload if isinstance(value, MatmulCompileRequest) else value
    return expected_matmul_outputs(workload)


def matmul_expected_matrices(
    value: MatmulCompileRequest | MatmulWorkload,
) -> tuple[WordMatrix, ...]:
    """Reshape the canonical flat outputs into output matrices."""

    workload = value.workload if isinstance(value, MatmulCompileRequest) else value
    flat = expected_matmul_outputs(workload)
    offset = 0
    matrices: list[WordMatrix] = []
    for rows, columns in workload.output_shapes:
        matrices.append(
            tuple(
                tuple(flat[offset + row * columns : offset + (row + 1) * columns])
                for row in range(rows)
            )
        )
        offset += rows * columns
    return tuple(matrices)


def _request_manifest(request: MatmulCompileRequest) -> dict[str, object]:
    return {
        "limits": manifest_value(request.limits),
        "workload": request.workload.manifest,
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
                guarantee="validated executable shared-weight matmul call DAG",
                evidence=exact,
            ),
            capability(
                Capability.STATIC_PARTITION,
                SupportState.SUPPORTED,
                kind,
                guarantee="one replay unit per matmul and one verification unit per dot",
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
                guarantee="trusted unsigned modular add and multiply relations",
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
                detail="matmul exposes transparent structure and local relations",
            ),
        ),
    )


def compile_matmul(
    request: MatmulCompileRequest | None = None,
) -> ProtocolCircuitArtifact:
    """Compile one shared-weight matrix workload into a literal protocol tuple."""

    selected = MatmulCompileRequest() if request is None else request
    if not isinstance(selected, MatmulCompileRequest):
        raise TypeError("matmul requires MatmulCompileRequest")
    compiled = compile_matmul_workload(
        selected.workload,
        limits=selected.limits,
    )
    artifact_identity = ArchitectureArtifactIdentity.build(
        architecture_id=ArchitectureId.MATMUL,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        artifact_kind=ArtifactKind.EXECUTABLE_CIRCUIT,
        request_manifest=manifest_value(_request_manifest(selected)),
        representation_manifest={
            "compiled_result_digest": compiled.identity.digest,
            "expected_outputs": list(selected.expected_outputs),
            "output_shapes": [list(shape) for shape in selected.output_shapes],
            "public_inputs": list(selected.public_inputs),
        },
    )
    assumptions = (
        "All weights and activations are public unsigned modular words.",
        "Every output is an unbiased inner product evaluated modulo 2^B.",
        "Shared weights occur once in the public input view and fan out structurally.",
        "Replay units are complete matrix multiplications.",
        "Verification units are complete output inner products.",
        "Constructor code is untrusted; only its canonical data output is decoded.",
    )
    return ProtocolCircuitArtifact(
        architecture_id=ArchitectureId.MATMUL,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        identity=artifact_identity,
        capabilities=_capabilities(),
        compiled=compiled,
        public_inputs=selected.public_inputs,
        expected_outputs=selected.expected_outputs,
        bound_provider=CallDagCapacityBoundProvider(compiled.circuit),
        assumptions=assumption_records(
            assumptions,
            source="veritor.compile.matmul",
            prefix="matmul",
        ),
        evidence=(
            EvidenceRecord(
                code="validated-matmul-call-dag",
                claim=ClaimStatus.EXACT,
                evidence=EvidenceStatus.BY_CONSTRUCTION,
                detail="trusted decoder validated the complete matmul call DAG",
                source="veritor.compile.compile_matmul_workload",
            ),
            EvidenceRecord(
                code="validated-matmul-partitions",
                claim=ClaimStatus.EXACT,
                evidence=EvidenceStatus.BY_CONSTRUCTION,
                detail="inner-product verification units exactly refine matmul replay units",
                source="veritor.core.CompiledArtifact",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class MatmulPlugin:
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.MATMUL,
    )
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> MatmulCompileRequest:
        return MatmulCompileRequest()

    def compile(self, request: object | None = None) -> ProtocolCircuitArtifact:
        if request is not None and not isinstance(request, MatmulCompileRequest):
            raise TypeError("matmul requires MatmulCompileRequest")
        return compile_matmul(request)


MATMUL_PLUGIN = MatmulPlugin()


__all__ = [
    "MATMUL_ARCHITECTURE_ID",
    "MATMUL_PLUGIN",
    "MatmulCompileRequest",
    "MatmulPlugin",
    "compile_matmul",
    "matmul_expected_matrices",
    "matmul_expected_outputs",
    "matmul_public_inputs",
]
