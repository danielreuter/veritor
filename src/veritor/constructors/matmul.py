"""Shared-weight matrix multiplication over modular values: workload, ``G``, request."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from veritor.compile import Compiler
from veritor.core import CompilationLimits, Compiled, JSONValue, make_word_gate_set

from .tracer import TracedDefinition, Tracer, TracerError, Wire, Wires

WordMatrix: TypeAlias = tuple[tuple[int, ...], ...]  # noqa: UP040


def _wires(value: Wire | Wires) -> Wires:
    """A call's result as a range (a one-output call returns a single wire)."""

    if isinstance(value, Wire):
        return Wires(value.trace, value.space, value.index, 1, 0)
    return value


def _canonical_matrix(
    value: Sequence[Sequence[int]],
    *,
    name: str,
    width: int,
) -> WordMatrix:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    rows: list[tuple[int, ...]] = []
    columns: int | None = None
    limit = 1 << width
    for row_index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise ValueError(f"{name} row {row_index} must be nonempty")
        checked = tuple(row)
        if columns is None:
            columns = len(checked)
        elif len(checked) != columns:
            raise ValueError(f"{name} must be rectangular")
        for column_index, item in enumerate(checked):
            if type(item) is not int or not 0 <= item < limit:
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be an unsigned "
                    f"{width}-bit value"
                )
        rows.append(checked)
    return tuple(rows)


@dataclass(frozen=True, slots=True, init=False)
class MatmulWorkload:
    """Canonical public inputs for repeated ``X_i @ W`` computations."""

    weights: WordMatrix
    activations: tuple[WordMatrix, ...]
    width: int

    def __init__(
        self,
        weights: Sequence[Sequence[int]],
        activations: Sequence[Sequence[Sequence[int]]],
        *,
        width: int = 8,
    ) -> None:
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        if (
            isinstance(activations, (str, bytes))
            or not isinstance(activations, Sequence)
            or not activations
        ):
            raise ValueError("activations must be a nonempty sequence of matrices")
        checked_weights = _canonical_matrix(weights, name="weights", width=width)
        contraction = len(checked_weights)
        checked_activations = tuple(
            _canonical_matrix(activation, name=f"activations[{index}]", width=width)
            for index, activation in enumerate(activations)
        )
        for index, activation in enumerate(checked_activations):
            if len(activation[0]) != contraction:
                raise ValueError(
                    f"activations[{index}] has contraction width "
                    f"{len(activation[0])}, expected {contraction}"
                )
        object.__setattr__(self, "weights", checked_weights)
        object.__setattr__(self, "activations", checked_activations)
        object.__setattr__(self, "width", width)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return len(self.weights), len(self.weights[0])

    @property
    def activation_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (len(activation), len(activation[0])) for activation in self.activations
        )

    @property
    def output_shapes(self) -> tuple[tuple[int, int], ...]:
        columns = self.weight_shape[1]
        return tuple((rows, columns) for rows, _inner in self.activation_shapes)

    @property
    def public_inputs(self) -> tuple[int, ...]:
        """The activations, row-major in activation order: the ``in`` gates by rank."""

        return tuple(
            item for activation in self.activations for row in activation for item in row
        )

    @property
    def weight_values(self) -> tuple[int, ...]:
        """The shared weights, row-major: the ``weight`` gates by rank (under κ_W)."""

        return tuple(item for row in self.weights for item in row)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "activation_shapes": [list(shape) for shape in self.activation_shapes],
            "activations": [
                [list(row) for row in activation] for activation in self.activations
            ],
            "input_order": "activations-row-major",
            "output_order": "activation-order-then-row-major",
            "weight_order": "row-major",
            "weight_shape": list(self.weight_shape),
            "weights": [list(row) for row in self.weights],
            "width": self.width,
        }


def expected_matmul_outputs(workload: MatmulWorkload) -> tuple[int, ...]:
    """Return flattened row-major modular products for every activation."""

    if not isinstance(workload, MatmulWorkload):
        raise TypeError("workload must be MatmulWorkload")
    mask = (1 << workload.width) - 1
    inner, columns = workload.weight_shape
    return tuple(
        sum(row[index] * workload.weights[index][column] for index in range(inner)) & mask
        for activation in workload.activations
        for row in activation
        for column in range(columns)
    )


class MatmulG:
    """The constructor for repeated matrix products with shared weights.

    The root ``batch`` has no ports.  The activations are a replay unit of
    ``in`` gates and the weights a replay unit of ``weight`` gates (each gate
    its own verification unit, through the tracer's one-gate cells), both
    declaring every gate they hold; every row ``x_i W`` is a replay unit of
    dot products (each a verification unit) that reads its activation row and
    ``W`` through ports from those two units' slots.  Rows, columns and the
    products inside a dot are ``repeat`` steps and the dot's sum is a tree of
    ``repeat`` steps, so the description is ``O(log k)`` in the contraction
    length and independent of the number of rows and columns.

    The activations live in their own unit rather than inside the rows so
    that a row is nothing but dots: its ``Out`` then tiles the row, the
    batch's ``rows x columns`` outputs are a single run, and admission stays
    flat in the number of rows.  (A row holding its ``k`` activations first
    would put the batch's outputs on a two-dimensional grid of
    ``min(rows, columns)`` runs.)
    """

    def __init__(self, width: int = 8) -> None:
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.tracer = Tracer(make_word_gate_set(width))
        mul, add = self.tracer.gate("mul"), self.tracer.gate("add")
        self.add = add
        self.mul = self.tracer.definition(input_count=2, key="mul")(
            lambda v: mul(v[0], v[1])
        )
        self.add_pair = self.tracer.definition(input_count=2, key="add")(
            lambda v: add(v[0], v[1])
        )

    def dot(self, k: int) -> TracedDefinition:
        """``x . w`` for ``k``-vectors: ``k`` products, then a sum tree."""

        if type(k) is not int or k <= 0:
            raise TracerError("dot length must be positive")

        @self.tracer.definition(input_count=2 * k, key=("dot", k), role="verification")
        def dot(v: Wires) -> object:
            x, w = v[:k], v[k:]
            level = self.tracer.repeat(k, self.mul, x[0].by(1), w[0].by(1))
            carried = []
            while len(level) > 1:
                if len(level) % 2:
                    carried.append(level[-1])
                level = self.tracer.repeat(len(level) // 2, self.add_pair, level[0:2].by(2))
            result = level[0]
            for carry in carried:
                result = self.add(result, carry)
            return result

        return dot

    def row(self, k: int, columns: int) -> TracedDefinition:
        """``x W`` for one ``k``-row ``x`` and a ``k x columns`` matrix ``W``."""

        @self.tracer.definition(
            input_count=k + k * columns, key=("row", k, columns), role="replay"
        )
        def row(v: Wires) -> object:
            x, w = v[:k], v[k:]
            return self.tracer.repeat(columns, self.dot(k), x, w[0 : k * columns : columns].by(1))

        return row

    def activations_unit(self, count: int) -> TracedDefinition:
        """The replay unit holding ``count`` input gates, all declared."""

        return self.tracer.definition(
            input_count=0, key=("activations", count), role="replay"
        )(lambda _v: self.tracer.inputs(count))

    def weights_unit(self, count: int) -> TracedDefinition:
        """The replay unit holding ``count`` weight gates, all declared."""

        return self.tracer.definition(input_count=0, key=("weights", count), role="replay")(
            lambda _v: self.tracer.weights(count)
        )

    def batch(
        self,
        activation_shapes: tuple[tuple[int, int], ...],
        weight_shape: tuple[int, int],
    ) -> TracedDefinition:
        inner, columns = weight_shape
        weight_cells = inner * columns
        activation_cells = sum(rows * k for rows, k in activation_shapes)
        activations = self.activations_unit(activation_cells)
        weights = self.weights_unit(weight_cells)

        @self.tracer.definition(input_count=0, key=("batch", activation_shapes, weight_shape))
        def batch(_v: Wires) -> object:
            x_all = _wires(activations())
            w = weights()
            outputs = []
            offset = 0
            for rows, k in activation_shapes:
                x = x_all[offset : offset + rows * k]
                offset += rows * k
                outputs.append(self.tracer.repeat(rows, self.row(k, columns), x[0:k].by(k), w))
            return outputs

        return batch

    def __call__(self, x: object, a: bytes) -> bytes:
        if not isinstance(x, MatmulWorkload):
            raise TracerError("MatmulG expects MatmulWorkload")
        if x.width != self.width:
            raise TracerError("workload width differs from MatmulG")
        if a != b"":
            raise TracerError("MatmulG does not accept constructor advice")
        return self.tracer.serialize(self.batch(x.activation_shapes, x.weight_shape))


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
    """A matmul workload plus the compilation limits to compile it under."""

    workload: MatmulWorkload
    limits: CompilationLimits | None

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
        """The activations: the values of the ``in`` gates by rank."""

        return self.workload.public_inputs

    @property
    def weight_values(self) -> tuple[int, ...]:
        """The shared weights, row-major: the values of the ``weight`` gates by rank."""

        return self.workload.weight_values

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
    """Run ``MatmulG`` on the workload and compile its description."""

    selected = MatmulCompileRequest() if request is None else request
    if not isinstance(selected, MatmulCompileRequest):
        raise TypeError("compile_matmul requires a MatmulCompileRequest")
    workload = selected.workload
    description = MatmulG(workload.width)(workload, b"")
    compiler = Compiler(make_word_gate_set(workload.width), selected.limits)
    return compiler.compile(description, workload.public_inputs)


__all__ = [
    "MatmulCompileRequest",
    "MatmulG",
    "MatmulWorkload",
    "WordMatrix",
    "compile_matmul",
    "expected_matmul_outputs",
    "matmul_expected_matrices",
]
