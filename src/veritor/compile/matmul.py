"""Executable shared-weight matrix multiplication over modular words."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from veritor.core import JSONValue, validate_compiled_result

from .call_dag import (
    CallDagCircuit,
    CompilationLimits,
    OccurrencePath,
    Producer,
    ProducerDefinition,
    ProducerError,
    Wire,
    construct,
    make_word_kernel,
)
from .compiler import CompiledCallDag
from .partitions import (
    PARTITION_POLICY_VERSION,
    derive_replay_partition_from_occurrences,
    derive_verification_partition_from_occurrences,
)

MATMUL_REPLAY_PARTITION_ALGORITHM_ID = (
    "veritor.compile.matmul.replay-per-matrix-multiplication"
)
MATMUL_VERIFICATION_PARTITION_ALGORITHM_ID = (
    "veritor.compile.matmul.verification-per-inner-product"
)

WordMatrix: TypeAlias = tuple[tuple[int, ...], ...]  # noqa: UP040


def _canonical_matrix(
    value: Sequence[Sequence[int]],
    *,
    name: str,
    cell_bits: int,
) -> WordMatrix:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    limit = 1 << cell_bits
    for row_index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise ValueError(f"{name} row {row_index} must be nonempty")
        checked = tuple(row)
        if width is None:
            width = len(checked)
        elif len(checked) != width:
            raise ValueError(f"{name} must be rectangular")
        for column_index, item in enumerate(checked):
            if type(item) is not int or not 0 <= item < limit:
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be an unsigned "
                    f"{cell_bits}-bit word"
                )
        rows.append(checked)
    return tuple(rows)


@dataclass(frozen=True, slots=True, init=False)
class MatmulWorkload:
    """Canonical public inputs for repeated ``X_i @ W`` computations."""

    weights: WordMatrix
    activations: tuple[WordMatrix, ...]
    cell_bits: int

    def __init__(
        self,
        weights: Sequence[Sequence[int]],
        activations: Sequence[Sequence[Sequence[int]]],
        *,
        cell_bits: int = 8,
    ) -> None:
        if type(cell_bits) is not int or cell_bits <= 0:
            raise ValueError("cell_bits must be a positive integer")
        if (
            isinstance(activations, (str, bytes))
            or not isinstance(activations, Sequence)
            or not activations
        ):
            raise ValueError("activations must be a nonempty sequence of matrices")
        checked_weights = _canonical_matrix(
            weights,
            name="weights",
            cell_bits=cell_bits,
        )
        contraction = len(checked_weights)
        checked_activations = tuple(
            _canonical_matrix(
                activation,
                name=f"activations[{index}]",
                cell_bits=cell_bits,
            )
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
        object.__setattr__(self, "cell_bits", cell_bits)

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
        """Flatten shared weights once, then activations, all row-major."""

        return (
            *(item for row in self.weights for item in row),
            *(
                item
                for activation in self.activations
                for row in activation
                for item in row
            ),
        )

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "activation_shapes": [list(shape) for shape in self.activation_shapes],
            "activations": [
                [list(row) for row in activation] for activation in self.activations
            ],
            "cell_bits": self.cell_bits,
            "input_order": "weights-row-major-then-activations-row-major",
            "output_order": "activation-order-then-row-major",
            "weights": [list(row) for row in self.weights],
            "weight_shape": list(self.weight_shape),
        }


def expected_matmul_outputs(workload: MatmulWorkload) -> tuple[int, ...]:
    """Return flattened row-major modular products for every activation."""

    if not isinstance(workload, MatmulWorkload):
        raise TypeError("workload must be MatmulWorkload")
    mask = (1 << workload.cell_bits) - 1
    inner, columns = workload.weight_shape
    outputs: list[int] = []
    for activation in workload.activations:
        for row in activation:
            for column in range(columns):
                accumulator = (row[0] * workload.weights[0][column]) & mask
                for index in range(1, inner):
                    product = (row[index] * workload.weights[index][column]) & mask
                    accumulator = (accumulator + product) & mask
                outputs.append(accumulator)
    return tuple(outputs)


class MatmulG:
    """Memoized constructor for repeated matrix products with shared weights."""

    def __init__(self, cell_bits: int = 8) -> None:
        if type(cell_bits) is not int or cell_bits <= 0:
            raise ValueError("cell_bits must be a positive integer")
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

    def inner_product(self, length: int) -> ProducerDefinition:
        if type(length) is not int or length <= 0:
            raise ProducerError("inner-product length must be positive")

        @self.producer.circuit(
            key=("matmul-inner-product", length),
            input_count=2 * length,
        )
        def inner_product_definition(*inputs: Wire) -> Wire:
            left = inputs[:length]
            right = inputs[length:]
            products = [
                self.mul(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            ]
            accumulator = products[0]
            for product in products[1:]:
                accumulator = self.add(accumulator, product)
            return accumulator

        return inner_product_definition

    def matrix_multiplication(
        self,
        rows: int,
        inner: int,
        columns: int,
    ) -> ProducerDefinition:
        if any(type(item) is not int or item <= 0 for item in (rows, inner, columns)):
            raise ProducerError("matrix dimensions must be positive integers")

        @self.producer.circuit(
            key=("matrix-multiplication", rows, inner, columns),
            input_count=rows * inner + inner * columns,
        )
        def matmul_definition(*inputs: Wire) -> tuple[Wire, ...]:
            activation = inputs[: rows * inner]
            weights = inputs[rows * inner :]
            dot = self.inner_product(inner)
            outputs: list[Wire] = []
            for row_index in range(rows):
                left = activation[row_index * inner : (row_index + 1) * inner]
                for column_index in range(columns):
                    right = tuple(
                        weights[inner_index * columns + column_index]
                        for inner_index in range(inner)
                    )
                    outputs.append(cast(Wire, dot(*left, *right)))
            return tuple(outputs)

        return matmul_definition

    def batch(
        self,
        activation_shapes: tuple[tuple[int, int], ...],
        weight_shape: tuple[int, int],
    ) -> ProducerDefinition:
        inner, columns = weight_shape
        weight_cells = inner * columns
        input_count = weight_cells + sum(
            rows * activation_inner for rows, activation_inner in activation_shapes
        )

        @self.producer.circuit(
            key=("shared-weight-matmul-batch", activation_shapes, weight_shape),
            input_count=input_count,
        )
        def batch_definition(*inputs: Wire) -> tuple[Wire, ...]:
            weights = inputs[:weight_cells]
            outputs: list[Wire] = []
            offset = weight_cells
            for rows, activation_inner in activation_shapes:
                activation_count = rows * activation_inner
                activation = inputs[offset : offset + activation_count]
                offset += activation_count
                matmul = self.matrix_multiplication(
                    rows,
                    activation_inner,
                    columns,
                )
                result = matmul(*activation, *weights)
                if isinstance(result, Wire):
                    outputs.append(result)
                else:
                    outputs.extend(result)
            return tuple(outputs)

        return batch_definition

    def __call__(self, x: object, a: bytes) -> bytes:
        if not isinstance(x, MatmulWorkload):
            raise ProducerError("MatmulG expects MatmulWorkload")
        if x.cell_bits != self.cell_bits:
            raise ProducerError("workload cell_bits differs from MatmulG")
        if a != b"":
            raise ProducerError("MatmulG does not accept constructor advice")
        root = self.batch(x.activation_shapes, x.weight_shape)
        return self.producer.serialize(root)


def matmul_replay_occurrence_paths(
    workload: MatmulWorkload,
) -> tuple[OccurrencePath, ...]:
    """Select each top-level matrix multiplication as one replay unit."""

    if not isinstance(workload, MatmulWorkload):
        raise TypeError("workload must be MatmulWorkload")
    return tuple((index,) for index in range(len(workload.activations)))


def matmul_verification_occurrence_paths(
    workload: MatmulWorkload,
) -> tuple[OccurrencePath, ...]:
    """Select each nested output inner product as one verification unit."""

    if not isinstance(workload, MatmulWorkload):
        raise TypeError("workload must be MatmulWorkload")
    return tuple(
        (activation_index, inner_product_index)
        for activation_index, (rows, _inner) in enumerate(workload.activation_shapes)
        for inner_product_index in range(rows * workload.weight_shape[1])
    )


def compile_matmul_workload(
    workload: MatmulWorkload,
    *,
    limits: CompilationLimits | None = None,
) -> CompiledCallDag:
    """Compile the concrete matmul circuit and its two fixed partitions."""

    if not isinstance(workload, MatmulWorkload):
        raise TypeError("workload must be MatmulWorkload")
    kernel = make_word_kernel(workload.cell_bits, limits=limits)
    constructor = MatmulG(workload.cell_bits)
    construction = construct(
        kernel,
        constructor,
        workload,
        b"",
        input_cells=workload.public_inputs,
        advice_bound_bits=0,
    )
    circuit = CallDagCircuit(kernel, construction.load.root)
    partition_manifest: dict[str, JSONValue] = {
        "activation_shapes": [list(shape) for shape in workload.activation_shapes],
        "cell_bits": workload.cell_bits,
        "weight_shape": list(workload.weight_shape),
    }
    replay = derive_replay_partition_from_occurrences(
        circuit,
        matmul_replay_occurrence_paths(workload),
        algorithm_id=MATMUL_REPLAY_PARTITION_ALGORITHM_ID,
        algorithm_version=PARTITION_POLICY_VERSION,
        configuration={
            **partition_manifest,
            "granularity": "one-matmul-per-replay-unit",
        },
    )
    verification = derive_verification_partition_from_occurrences(
        circuit,
        replay,
        matmul_verification_occurrence_paths(workload),
        algorithm_id=MATMUL_VERIFICATION_PARTITION_ALGORITHM_ID,
        algorithm_version=PARTITION_POLICY_VERSION,
        configuration={
            **partition_manifest,
            "granularity": "one-inner-product-per-verification-unit",
        },
    )
    validate_compiled_result(circuit, replay, verification)
    return circuit, replay, verification


__all__ = [
    "MATMUL_REPLAY_PARTITION_ALGORITHM_ID",
    "MATMUL_VERIFICATION_PARTITION_ALGORITHM_ID",
    "MatmulG",
    "MatmulWorkload",
    "WordMatrix",
    "compile_matmul_workload",
    "expected_matmul_outputs",
    "matmul_replay_occurrence_paths",
    "matmul_verification_occurrence_paths",
]
