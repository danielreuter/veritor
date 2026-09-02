"""Executable plug-in for shared-weight modular matrix multiplication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from veritor.compile import (
    Compiler,
    MatmulG,
    MatmulWorkload,
    WordMatrix,
    expected_matmul_outputs,
)
from veritor.core import CompilationLimits, Compiled, make_word_gate_set

from ..api import ArchitectureId

PLUGIN_ID = "veritor.plugins.builtin.matmul"
PLUGIN_VERSION = "2"
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

    workload: MatmulWorkload
    limits: CompilationLimits | None
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.MATMUL)

    def __init__(
        self,
        weights: Sequence[Sequence[int]] = _DEFAULT_WEIGHTS,
        activations: Sequence[Sequence[Sequence[int]]] = _DEFAULT_ACTIVATIONS,
        *,
        width: int = 8,
        limits: CompilationLimits | None = None,
    ) -> None:
        if limits is not None and not isinstance(limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits or None")
        object.__setattr__(self, "workload", MatmulWorkload(weights, activations, width=width))
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "architecture_id", ArchitectureId.MATMUL)

    @property
    def weights(self) -> WordMatrix:
        return self.workload.weights

    @property
    def activations(self) -> tuple[WordMatrix, ...]:
        return self.workload.activations

    @property
    def width(self) -> int:
        return self.workload.width

    @property
    def public_inputs(self) -> tuple[int, ...]:
        """Every input value: the weights first, then the activations."""

        return self.workload.public_inputs

    @property
    def weight_addresses(self) -> range:
        """The input addresses holding the shared weights, row-major."""

        rows, columns = self.workload.weight_shape
        return range(rows * columns)

    @property
    def activation_inputs(self) -> tuple[int, ...]:
        """The input values outside :attr:`weight_addresses`, in address order."""

        return self.workload.public_inputs[self.weight_addresses.stop :]

    @property
    def expected_outputs(self) -> tuple[int, ...]:
        return expected_matmul_outputs(self.workload)

    @property
    def output_shapes(self) -> tuple[tuple[int, int], ...]:
        return self.workload.output_shapes


def matmul_expected_matrices(request: MatmulCompileRequest) -> tuple[WordMatrix, ...]:
    """Reshape the canonical flat outputs into output matrices."""

    flat = request.expected_outputs
    offset = 0
    matrices: list[WordMatrix] = []
    for rows, columns in request.output_shapes:
        matrices.append(
            tuple(
                tuple(flat[offset + row * columns : offset + (row + 1) * columns])
                for row in range(rows)
            )
        )
        offset += rows * columns
    return tuple(matrices)


def compile_matmul(request: MatmulCompileRequest | None = None) -> Compiled:
    """Trace the workload with :class:`MatmulG` and compile the description."""

    selected = MatmulCompileRequest() if request is None else request
    if not isinstance(selected, MatmulCompileRequest):
        raise TypeError("matmul requires MatmulCompileRequest")
    workload = selected.workload
    description = MatmulG(workload.width)(workload, b"")
    compiler = Compiler(make_word_gate_set(workload.width), selected.limits)
    return compiler.compile(description, workload.public_inputs)


@dataclass(frozen=True, slots=True)
class MatmulPlugin:
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.MATMUL)
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> MatmulCompileRequest:
        return MatmulCompileRequest()

    def compile(self, request: object | None = None) -> Compiled:
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
]
