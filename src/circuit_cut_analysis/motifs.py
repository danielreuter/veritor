"""Small scalar circuit motifs and their exact symbolic gate counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from circuit_cut_analysis.circuit import CircuitDAG, Edge, Gate, GateId


class Primitive(StrEnum):
    ADD = "add"
    MUL = "mul"
    MAX = "max"
    EXP = "exp"
    RECIPROCAL = "reciprocal"
    RSQRT = "rsqrt"
    TANH = "tanh"


@dataclass(frozen=True, slots=True)
class ExpandedMotif:
    """A toy-expandable motif with fixed inputs separated from computed gates."""

    name: str
    circuit: CircuitDAG
    computed_gates: frozenset[GateId]
    bottlenecks: Mapping[str, frozenset[GateId]]

    @property
    def primitive_counts(self) -> Counter[str]:
        return Counter(
            self.circuit.gates[gate_id].op for gate_id in self.computed_gates
        )

    @property
    def unit_gate_count(self) -> int:
        return len(self.computed_gates)


class _MotifBuilder:
    def __init__(self) -> None:
        self.gates: list[Gate] = []
        self.edges: list[Edge] = []
        self.computed: set[GateId] = set()

    def input(self, gate_id: GateId, width_bits: int) -> GateId:
        self.gates.append(Gate(gate_id, width_bits, "input"))
        return gate_id

    def compute(
        self,
        gate_id: GateId,
        width_bits: int,
        op: Primitive,
        arguments: Iterable[GateId],
    ) -> GateId:
        self.gates.append(Gate(gate_id, width_bits, op.value))
        self.computed.add(gate_id)
        self.edges.extend((argument, gate_id) for argument in arguments)
        return gate_id

    def finish(
        self,
        name: str,
        outputs: Iterable[GateId],
        bottlenecks: Mapping[str, Iterable[GateId]],
    ) -> ExpandedMotif:
        output_set = frozenset(outputs)
        return ExpandedMotif(
            name=name,
            circuit=CircuitDAG(self.gates, self.edges, output_set),
            computed_gates=frozenset(self.computed),
            bottlenecks={
                label: frozenset(gates) for label, gates in bottlenecks.items()
            },
        )


def dot_product_counts(
    length: int,
    *,
    bias: bool,
    scale: bool = False,
) -> Counter[str]:
    if length <= 0:
        raise ValueError("dot-product length must be positive")
    return Counter(
        {
            Primitive.MUL.value: length + int(scale),
            Primitive.ADD.value: length if bias else length - 1,
        }
    )


def expand_dot_product(
    length: int,
    *,
    bias: bool,
    scale: bool = False,
    operand_width: int = 16,
    accumulator_width: int = 32,
    output_width: int = 16,
    name: str = "dot_product",
) -> ExpandedMotif:
    """Expand one scalar contraction.

    Every product is an explicit gate.  A biased contraction starts from the
    bias and performs one addition per product; an unbiased contraction uses
    the first product as its initial accumulator.  The final arithmetic gate
    writes directly at ``output_width``, so no uncounted cast gate is needed.
    """

    if length <= 0:
        raise ValueError("dot-product length must be positive")
    builder = _MotifBuilder()
    products: list[GateId] = []
    for index in range(length):
        left = builder.input(f"a:{index}", operand_width)
        right = builder.input(f"b:{index}", operand_width)
        product_width = (
            output_width
            if length == 1 and not bias and not scale
            else accumulator_width
        )
        products.append(
            builder.compute(
                f"mul:{index}",
                product_width,
                Primitive.MUL,
                (left, right),
            )
        )

    if bias:
        previous = builder.input("bias", operand_width)
        start = 0
    else:
        previous = products[0]
        start = 1

    for index in range(start, length):
        is_final = index == length - 1 and not scale
        previous = builder.compute(
            f"acc:{index}",
            output_width if is_final else accumulator_width,
            Primitive.ADD,
            (previous, products[index]),
        )

    if scale:
        scale_value = builder.input("scale", operand_width)
        previous = builder.compute(
            "scaled",
            output_width,
            Primitive.MUL,
            (previous, scale_value),
        )

    return builder.finish(
        name,
        {previous},
        {
            "output": {previous},
        },
    )


def gelu_new_counts() -> Counter[str]:
    return Counter(
        {
            Primitive.MUL.value: 6,
            Primitive.ADD.value: 2,
            Primitive.TANH.value: 1,
        }
    )


def expand_gelu_new(
    *,
    input_width: int = 16,
    internal_width: int = 32,
    output_width: int = 16,
) -> ExpandedMotif:
    """Expand GPT-2's tanh-approximation GELU for one scalar."""

    builder = _MotifBuilder()
    x = builder.input("x", input_width)
    cubic_coefficient = builder.input("cubic_coefficient", internal_width)
    tanh_coefficient = builder.input("tanh_coefficient", internal_width)
    one = builder.input("one", internal_width)
    half = builder.input("half", internal_width)

    x2 = builder.compute("x2", internal_width, Primitive.MUL, (x, x))
    x3 = builder.compute("x3", internal_width, Primitive.MUL, (x2, x))
    cubic = builder.compute(
        "cubic",
        internal_width,
        Primitive.MUL,
        (x3, cubic_coefficient),
    )
    inner = builder.compute("inner", internal_width, Primitive.ADD, (x, cubic))
    scaled = builder.compute(
        "tanh_input",
        internal_width,
        Primitive.MUL,
        (inner, tanh_coefficient),
    )
    activated = builder.compute("tanh", internal_width, Primitive.TANH, (scaled,))
    shifted = builder.compute(
        "one_plus_tanh",
        internal_width,
        Primitive.ADD,
        (one, activated),
    )
    gated = builder.compute("gated", internal_width, Primitive.MUL, (x, shifted))
    output = builder.compute(
        "output",
        output_width,
        Primitive.MUL,
        (half, gated),
    )
    return builder.finish(
        "gelu_new",
        {output},
        {
            "output": {output},
        },
    )


def layer_norm_counts(size: int) -> Counter[str]:
    if size <= 0:
        raise ValueError("LayerNorm size must be positive")
    return Counter(
        {
            Primitive.ADD.value: 4 * size - 1,
            Primitive.MUL.value: 3 * size + 2,
            Primitive.RSQRT.value: 1,
        }
    )


def _sum_chain(
    builder: _MotifBuilder,
    values: list[GateId],
    *,
    prefix: str,
    width_bits: int,
) -> GateId:
    if not values:
        raise ValueError("cannot reduce an empty value list")
    previous = values[0]
    for index, value in enumerate(values[1:], start=1):
        previous = builder.compute(
            f"{prefix}:{index}",
            width_bits,
            Primitive.ADD,
            (previous, value),
        )
    return previous


def expand_layer_norm(
    size: int,
    *,
    input_width: int = 16,
    internal_width: int = 32,
    output_width: int = 16,
) -> ExpandedMotif:
    """Expand two-pass LayerNorm over one vector."""

    if size <= 0:
        raise ValueError("LayerNorm size must be positive")
    builder = _MotifBuilder()
    inputs = [builder.input(f"x:{index}", input_width) for index in range(size)]
    mean_scale = builder.input("mean_scale", internal_width)
    variance_scale = builder.input("variance_scale", internal_width)
    epsilon = builder.input("epsilon", internal_width)

    mean_sum = _sum_chain(
        builder,
        inputs,
        prefix="mean_sum",
        width_bits=internal_width,
    )
    mean = builder.compute(
        "mean",
        internal_width,
        Primitive.MUL,
        (mean_sum, mean_scale),
    )
    centered = [
        builder.compute(
            f"centered:{index}",
            internal_width,
            Primitive.ADD,
            (value, mean),
        )
        for index, value in enumerate(inputs)
    ]
    squares = [
        builder.compute(
            f"square:{index}",
            internal_width,
            Primitive.MUL,
            (value, value),
        )
        for index, value in enumerate(centered)
    ]
    variance_sum = _sum_chain(
        builder,
        squares,
        prefix="variance_sum",
        width_bits=internal_width,
    )
    variance = builder.compute(
        "variance",
        internal_width,
        Primitive.MUL,
        (variance_sum, variance_scale),
    )
    stabilized = builder.compute(
        "stabilized_variance",
        internal_width,
        Primitive.ADD,
        (variance, epsilon),
    )
    inverse_std = builder.compute(
        "inverse_std",
        internal_width,
        Primitive.RSQRT,
        (stabilized,),
    )

    outputs: list[GateId] = []
    for index, value in enumerate(centered):
        gamma = builder.input(f"gamma:{index}", input_width)
        beta = builder.input(f"beta:{index}", input_width)
        normalized = builder.compute(
            f"normalized:{index}",
            internal_width,
            Primitive.MUL,
            (value, inverse_std),
        )
        scaled = builder.compute(
            f"affine_scale:{index}",
            internal_width,
            Primitive.MUL,
            (normalized, gamma),
        )
        outputs.append(
            builder.compute(
                f"output:{index}",
                output_width,
                Primitive.ADD,
                (scaled, beta),
            )
        )

    return builder.finish(
        "layer_norm",
        outputs,
        {
            "mean": {mean},
            "inverse_std": {inverse_std},
            "outputs": outputs,
            **{f"output:{index}": {output} for index, output in enumerate(outputs)},
        },
    )


def softmax_counts(size: int) -> Counter[str]:
    if size <= 0:
        raise ValueError("softmax size must be positive")
    return Counter(
        {
            Primitive.MAX.value: size - 1,
            Primitive.ADD.value: 2 * size - 1,
            Primitive.EXP.value: size,
            Primitive.MUL.value: size,
            Primitive.RECIPROCAL.value: 1,
        }
    )


def expand_softmax(
    size: int,
    *,
    input_width: int = 16,
    internal_width: int = 32,
    output_width: int = 16,
) -> ExpandedMotif:
    """Expand a stable two-pass softmax over one vector.

    ``MAX`` is one value-producing compare/select gate of ``internal_width``;
    no separate Boolean wire is implied by this decomposition.
    """

    if size <= 0:
        raise ValueError("softmax size must be positive")
    builder = _MotifBuilder()
    scores = [builder.input(f"score:{index}", input_width) for index in range(size)]

    maximum = scores[0]
    for index, score in enumerate(scores[1:], start=1):
        maximum = builder.compute(
            f"max:{index}",
            internal_width,
            Primitive.MAX,
            (maximum, score),
        )

    shifted = [
        builder.compute(
            f"shifted:{index}",
            internal_width,
            Primitive.ADD,
            (score, maximum),
        )
        for index, score in enumerate(scores)
    ]
    exponentials = [
        builder.compute(
            f"exp:{index}",
            internal_width,
            Primitive.EXP,
            (value,),
        )
        for index, value in enumerate(shifted)
    ]
    denominator = _sum_chain(
        builder,
        exponentials,
        prefix="denominator",
        width_bits=internal_width,
    )
    reciprocal = builder.compute(
        "reciprocal",
        internal_width,
        Primitive.RECIPROCAL,
        (denominator,),
    )
    probabilities = [
        builder.compute(
            f"probability:{index}",
            output_width,
            Primitive.MUL,
            (value, reciprocal),
        )
        for index, value in enumerate(exponentials)
    ]
    return builder.finish(
        "softmax",
        probabilities,
        {
            "maximum": {maximum},
            "denominator": {denominator},
            "reciprocal": {reciprocal},
            "outputs": probabilities,
            **{
                f"output:{index}": {output}
                for index, output in enumerate(probabilities)
            },
        },
    )


def expand_elementwise_add(width_bits: int = 16) -> ExpandedMotif:
    builder = _MotifBuilder()
    left = builder.input("left", width_bits)
    right = builder.input("right", width_bits)
    output = builder.compute("output", width_bits, Primitive.ADD, (left, right))
    return builder.finish(
        "elementwise_add",
        {output},
        {
            "output": {output},
        },
    )
