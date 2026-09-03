"""DemoG: a small memoized constructor of chained multiply-accumulates.

``DemoG`` is untrusted; the only outputs the compiler reads are the canonical
description and the flat inputs.  The constructor, its input types and
``compile_demo_g`` live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

from veritor.compile import Compilation, constructor_digest
from veritor.core import CompilationLimits, make_word_gate_set
from veritor.research import Compile

from .tracer import TracedDefinition, Tracer, TracerError, Wire, Wires


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
            raise TracerError("dot-product values and weights have different lengths")
        return (self.accumulator, *self.values, *self.weights)


@dataclass(frozen=True, slots=True)
class BatchInput:
    requests: tuple[DotRequest, ...]

    def cells(self) -> tuple[int, ...]:
        return tuple(cell for request in self.requests for cell in request.cells())


class DemoG:
    """A memoized demo constructor: chained multiply-accumulates.

    Every multiply-accumulate is a verification unit and every dot product (a
    chain of them) is a replay unit that holds its own cells (accumulator,
    values, weights) as ``in`` gates, so the root has no ports and the public
    inputs are the requests' cells in request order.  Consecutive requests of
    equal length are one ``repeat`` step, so the description is ``O(distinct
    lengths + runs)``.

    ``DemoG`` accepts advice of any length and ignores it: the batch's
    lengths fix the circuit.  The protocol charges the advice by length all
    the same.
    """

    VERSION = "1"

    def __init__(self, width: int = 8) -> None:
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.digest = constructor_digest(
            type(self).__name__, self.VERSION, {"width": width}
        )
        self.tracer = Tracer(make_word_gate_set(width))
        add, mul = self.tracer.gate("add"), self.tracer.gate("mul")

        @self.tracer.definition(input_count=3, key="mac", role="verification")
        def mac(v: Wires) -> object:
            return add(v[0], mul(v[1], v[2]))

        self.mac = mac

    def dot(self, length: int) -> TracedDefinition:
        if type(length) is not int or length < 0:
            raise TracerError("dot length must be a nonnegative integer")

        @self.tracer.definition(input_count=0, key=("dot", length), role="replay")
        def dot(_v: Wires) -> object:
            cells = self.tracer.inputs(1 + 2 * length)  # accumulator, values, weights
            accumulator: Wire | Wires = cells[0]
            for index in range(length):
                accumulator = self.mac(
                    accumulator, cells[1 + index], cells[1 + length + index]
                )
            return accumulator

        return dot

    def batch(self, lengths: tuple[int, ...]) -> TracedDefinition:
        @self.tracer.definition(input_count=0, key=("batch", lengths))
        def batch(_v: Wires) -> object:
            return [
                self.tracer.repeat(len(list(run)), self.dot(length))
                for length, run in groupby(lengths)
            ]

        return batch

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        """The description for the batch's lengths and its cells as the ``in`` gates' values."""

        if not isinstance(x, BatchInput):
            raise TracerError("DemoG expects BatchInput")
        if type(a) is not bytes:
            raise TracerError("DemoG advice must be bytes")
        for request in x.requests:
            request.cells()
        if not x.requests:
            raise TracerError("DemoG needs at least one dot request")
        if any(request.length == 0 for request in x.requests):
            raise TracerError("DemoG dot requests must be nonempty")
        description = self.tracer.serialize(
            self.batch(tuple(r.length for r in x.requests))
        )
        return description, x.cells()


def expected_dot_outputs(batch: BatchInput, width: int) -> tuple[int, ...]:
    mask = (1 << width) - 1
    outputs = []
    for request in batch.requests:
        accumulator = request.accumulator
        for value, weight in zip(request.values, request.weights, strict=True):
            accumulator = (accumulator + value * weight) & mask
        outputs.append(accumulator)
    return tuple(outputs)


def make_demo_request(length: int, seed: int, width: int = 8) -> DotRequest:
    mask = (1 << width) - 1
    values = tuple((seed + 3 * index + 1) & mask for index in range(length))
    weights = tuple((2 * seed + 5 * index + 1) & mask for index in range(length))
    return DotRequest(seed & mask, values, weights)


def _default_batch() -> BatchInput:
    return BatchInput((make_demo_request(2, 1, 8), make_demo_request(3, 2, 8)))


@dataclass(frozen=True, slots=True)
class DemoGCompileRequest:
    """Inputs and advice for DemoG, with the verifier's advice bound and compilation limits."""

    batch: BatchInput = field(default_factory=_default_batch)
    advice: bytes = b""
    width: int = 8
    max_advice_bits: int = 0
    limits: CompilationLimits | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BatchInput):
            raise TypeError("batch must be a BatchInput")
        if type(self.advice) is not bytes:
            raise TypeError("advice must be bytes")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if type(self.max_advice_bits) is not int or self.max_advice_bits < 0:
            raise ValueError("max_advice_bits must be a nonnegative integer")
        if self.limits is not None and not isinstance(self.limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits or None")

    @property
    def public_inputs(self) -> tuple[int, ...]:
        return self.batch.cells()

    @property
    def expected_outputs(self) -> tuple[int, ...]:
        return expected_dot_outputs(self.batch, self.width)


def compile_demo_g(request: DemoGCompileRequest | None = None) -> Compilation:
    """``Compile(DemoG, batch, advice)``: what the verifier records for the request."""

    selected = DemoGCompileRequest() if request is None else request
    if not isinstance(selected, DemoGCompileRequest):
        raise TypeError("compile_demo_g requires a DemoGCompileRequest")
    return Compile(
        DemoG(selected.width),
        selected.batch,
        selected.advice,
        make_word_gate_set(selected.width),
        limits=selected.limits,
        max_advice_bits=selected.max_advice_bits,
    )


__all__ = [
    "BatchInput",
    "DemoG",
    "DemoGCompileRequest",
    "DotRequest",
    "compile_demo_g",
    "expected_dot_outputs",
    "make_demo_request",
]
