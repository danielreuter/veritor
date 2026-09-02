"""Executable built-in plug-in for the DemoG constructor.

``DemoG`` is an untrusted memoized constructor whose only trusted output is
the canonical description decoded by :mod:`veritor.compile`.  The
constructor, its request types, and the plug-in wrapper all live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

from veritor.compile import (
    Compiler,
    TracedDefinition,
    Tracer,
    TracerError,
    Wires,
)
from veritor.core import CompilationLimits, Compiled, make_word_gate_set

from ..api import ArchitectureId

PLUGIN_ID = "veritor.plugins.builtin.demo-g"
PLUGIN_VERSION = "2"
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
    chain of them) is a replay unit.  Consecutive requests of equal length are
    one ``repeat`` step, so the description is ``O(distinct lengths + runs)``.
    """

    def __init__(self, width: int = 8) -> None:
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.tracer = Tracer(make_word_gate_set(width))
        add, mul = self.tracer.gate("add"), self.tracer.gate("mul")

        @self.tracer.definition(input_count=3, key="mac", role="verification")
        def mac(v: Wires) -> object:
            return add(v[0], mul(v[1], v[2]))

        self.mac = mac

    def dot(self, length: int) -> TracedDefinition:
        if type(length) is not int or length < 0:
            raise TracerError("dot length must be a nonnegative integer")

        @self.tracer.definition(
            input_count=1 + 2 * length, key=("dot", length), role="replay"
        )
        def dot(v: Wires) -> object:
            accumulator = v[0]
            for index in range(length):
                accumulator = self.mac(accumulator, v[1 + index], v[1 + length + index])
            return accumulator

        return dot

    def batch(self, lengths: tuple[int, ...]) -> TracedDefinition:
        @self.tracer.definition(
            input_count=sum(1 + 2 * length for length in lengths), key=("batch", lengths)
        )
        def batch(v: Wires) -> object:
            outputs = []
            offset = 0
            for length, run in groupby(lengths):
                count = len(list(run))
                stride = 1 + 2 * length
                block = v[offset : offset + stride]
                outputs.append(self.tracer.repeat(count, self.dot(length), block.by(stride)))
                offset += count * stride
            return outputs

        return batch

    def __call__(self, x: object, a: bytes) -> bytes:
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
        return self.tracer.serialize(self.batch(tuple(r.length for r in x.requests)))


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
    """Inputs, advice and limits for DemoG."""

    batch: BatchInput = field(default_factory=_default_batch)
    advice: bytes = b""
    width: int = 8
    advice_bound_bits: int = 0
    limits: CompilationLimits | None = None
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.DEMO_G)

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BatchInput):
            raise TypeError("batch must be a BatchInput")
        if type(self.advice) is not bytes:
            raise TypeError("advice must be bytes")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if type(self.advice_bound_bits) is not int or self.advice_bound_bits < 0:
            raise ValueError("advice_bound_bits must be a nonnegative integer")
        if len(self.advice) * 8 > self.advice_bound_bits:
            raise ValueError("advice exceeds advice_bound_bits")
        if self.limits is not None and not isinstance(self.limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits or None")

    @property
    def public_inputs(self) -> tuple[int, ...]:
        return self.batch.cells()

    @property
    def expected_outputs(self) -> tuple[int, ...]:
        return expected_dot_outputs(self.batch, self.width)


def compile_demo_g(request: DemoGCompileRequest | None = None) -> Compiled:
    """Trace the batch with :class:`DemoG` and compile the description."""

    selected = DemoGCompileRequest() if request is None else request
    if not isinstance(selected, DemoGCompileRequest):
        raise TypeError("DemoG requires DemoGCompileRequest")
    description = DemoG(selected.width)(selected.batch, selected.advice)
    compiler = Compiler(make_word_gate_set(selected.width), selected.limits)
    return compiler.compile(
        description,
        selected.public_inputs,
        selected.advice,
        advice_bound_bits=selected.advice_bound_bits,
    )


@dataclass(frozen=True, slots=True)
class DemoGPlugin:
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.DEMO_G)
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> DemoGCompileRequest:
        return DemoGCompileRequest()

    def compile(self, request: object | None = None) -> Compiled:
        if request is not None and not isinstance(request, DemoGCompileRequest):
            raise TypeError("DemoG requires DemoGCompileRequest")
        return compile_demo_g(request)


DEMO_G_PLUGIN = DemoGPlugin()
