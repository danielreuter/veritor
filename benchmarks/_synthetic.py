"""Synthetic descriptions whose one size parameter is under the benchmark's control.

Every builder returns canonical description bytes over ``GATE_SET`` (16-bit
words) whose one ``in`` gate sits in a one-gate replay unit, so
``compile_description`` needs the single input ``(3,)``.  The role marks tile
as the compiler requires: every gate lies in exactly one replay unit (RU) and
one verification unit (VU).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from veritor.compile import Compiler
from veritor.constructors import (
    ClusterG,
    LMShape,
    Request,
    RequestsG,
    Tracer,
    random_parameters,
    schedule_fcfs,
)
from veritor.core import Compiled, make_isa_gate_set, make_word_gate_set

GATE_SET = make_word_gate_set(16)
ISA = make_isa_gate_set(16)
INPUT = (3,)


def compile_description(description: bytes) -> Compiled:
    return Compiler(GATE_SET).compile(description, INPUT)


def _cell(tracer: Tracer, gates: int, role: str | None = "verification"):
    """A VU of ``gates`` chained ``add`` gates over one port."""

    add = tracer.gate("add")

    @tracer.definition(input_count=1, key=("cell", gates, role), role=role)
    def cell(v):
        value = add(v[0], v[0])
        for _ in range(gates - 1):
            value = add(value, v[0])
        return value

    return cell


def _source(tracer: Tracer):
    """The circuit's one input gate in a replay unit of its own."""

    @tracer.definition(input_count=0, key="source", role="replay")
    def source(_v):
        return tracer.inputs(1)

    return source


def deep_repeat(branching: tuple[int, ...], cell_gates: int = 3) -> bytes:
    """``branching[0]`` cells per RU, then one ``repeat`` level per further factor.

    ``n = cell_gates * prod(branching) + 1`` gates from a description of
    ``len(branching) + 3`` definitions; every level passes the input through
    and returns its last output, so ``|Out|`` of every unit is one address and
    the boundary is ``#RU + 1``.
    """

    if not branching or any(b < 1 for b in branching):
        raise ValueError("branching factors must be positive")
    tracer = Tracer(GATE_SET)
    cell = _cell(tracer, cell_gates)
    source = _source(tracer)

    @tracer.definition(input_count=1, key=("block", branching[0]), role="replay")
    def block(v):
        return tracer.repeat(branching[0], cell, v[0])[-1]

    level = block
    for depth, factor in enumerate(branching[1:], start=1):

        @tracer.definition(input_count=1, key=("level", depth, factor))
        def level(v, _child=level, _factor=factor):
            return tracer.repeat(_factor, _child, v[0])[-1]

    @tracer.definition(input_count=0, key=("root", branching))
    def root(_v):
        return level(source())

    return tracer.serialize(root)


def deep_repeat_gates(branching: tuple[int, ...], cell_gates: int = 3) -> int:
    return cell_gates * prod(branching) + 1


def many_definitions(count: int, cell_gates: int = 3) -> bytes:
    """``count`` distinct RU kinds (``repeat(i + 1, cell)``), each called once by the root.

    The root's one output is the last unit's (declaring every unit's output
    would exceed the compiler's 256-run limit on one definition's ``Out``).
    """

    tracer = Tracer(GATE_SET)
    cell = _cell(tracer, cell_gates)
    source = _source(tracer)
    units = []
    for i in range(count):

        @tracer.definition(input_count=1, key=("unit", i), role="replay")
        def unit(v, _i=i):
            return tracer.repeat(_i + 1, cell, v[0])[-1]

        units.append(unit)

    @tracer.definition(input_count=0, key=("root", count))
    def root(_v):
        x = source()
        return [unit(x) for unit in units][-1]

    return tracer.serialize(root)


def chain_steps(steps: int, cell_gates: int = 3) -> bytes:
    """A root of ``steps`` sequential calls of one RU, each reading the previous one's output.

    The root's output is the last unit's: declaring one output per call step
    would exceed the compiler's 256-piece limit on one definition's ``Out``
    (pieces are counted before adjacent ones are merged into a run).
    """

    tracer = Tracer(GATE_SET)
    cell = _cell(tracer, cell_gates)
    source = _source(tracer)

    @tracer.definition(input_count=1, key="unit", role="replay")
    def unit(v):
        return cell(v[0])

    @tracer.definition(input_count=0, key=("chain", steps))
    def root(_v):
        value = source()
        for _ in range(steps):
            value = unit(value)
        return value

    return tracer.serialize(root)


def unrolled_units(count: int, cell_gates: int = 3) -> bytes:
    """A root of ``count`` independent ``call`` steps of one RU (no ``repeat``): descent bisects the step list."""

    tracer = Tracer(GATE_SET)
    cell = _cell(tracer, cell_gates)
    source = _source(tracer)

    @tracer.definition(input_count=1, key="unit", role="replay")
    def unit(v):
        return cell(v[0])

    @tracer.definition(input_count=0, key=("unrolled", count))
    def root(_v):
        x = source()
        return [unit(x) for _ in range(count)][-1]

    return tracer.serialize(root)


def runs_unit(runs: int, copies: int) -> bytes:
    """``copies`` copies of one RU whose ``Out`` resolves to about ``runs`` runs.

    The unit is a chain of ``add`` gates; its declared outputs are single
    gates at gaps cycling through ``1, 2, 3`` so no two adjacent outputs
    continue one progression and every output is its own run (up to the
    tracer's pairing of consecutive wires).  The compiler admits at most 256
    runs per definition.
    """

    tracer = Tracer(GATE_SET)
    source = _source(tracer)
    cell = _cell(tracer, 1)

    @tracer.definition(input_count=1, key=("runs", runs), role="replay")
    def unit(v):
        outputs = []
        value = cell(v[0])
        gap = 0
        while len(outputs) < runs:
            outputs.append(value)
            gap = gap % 3 + 1
            for _ in range(gap):
                value = cell(value)
        for _ in range(
            max(0, 64 - 2 * runs)
        ):  # a body of at least ~64 gates behind the outputs
            value = cell(value)
        return outputs

    @tracer.definition(input_count=0, key=("root", runs, copies))
    def root(_v):
        return tracer.repeat(copies, unit, source())[-1]

    return tracer.serialize(root)


# -- the toy decoder ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecoderCase:
    """One serving workload for ``ClusterG`` or ``RequestsG`` with everything the protocol needs."""

    label: str
    shape: LMShape
    requests: tuple[Request, ...]
    pods: int
    slots: int
    steps: int

    @property
    def cluster(self) -> ClusterG:
        return ClusterG(self.shape, self.pods, self.slots, self.steps)

    @property
    def per_request(self) -> RequestsG:
        return RequestsG(self.shape)

    @property
    def advice(self) -> bytes:
        return schedule_fcfs(self.requests, self.pods, self.slots, self.steps).encode()

    def weights(self, seed: int = 0) -> tuple[int, ...]:
        return random_parameters(self.shape, seed=seed).flatten()


def decoder_case(
    label: str,
    *,
    d_model: int,
    layers: int,
    prompt: int,
    max_new: int,
    requests: int,
    slots: int,
    heads: int = 2,
    vocab: int = 8,
    pods: int = 1,
    width: int = 16,
) -> DecoderCase:
    """A batch of identical requests served first-come-first-served on ``slots`` slots."""

    waves = -(-requests // (pods * slots))
    steps = waves * max_new
    shape = LMShape(
        vocab=vocab,
        d_model=d_model,
        heads=heads,
        layers=layers,
        context=prompt + max_new,
        width=width,
    )
    prompt_tokens = tuple((i * 3 + 1) % vocab for i in range(prompt))
    return DecoderCase(
        label,
        shape,
        tuple(Request(prompt_tokens, max_new) for _ in range(requests)),
        pods,
        slots,
        steps,
    )


CLUSTER_LADDER: tuple[DecoderCase, ...] = (
    decoder_case(
        "d4-L1", d_model=4, layers=1, prompt=3, max_new=3, requests=2, slots=2
    ),
    decoder_case(
        "d8-L1", d_model=8, layers=1, prompt=3, max_new=3, requests=4, slots=2
    ),
    decoder_case(
        "d16-L2", d_model=16, layers=2, prompt=4, max_new=4, requests=8, slots=4
    ),
    decoder_case(
        "d32-L2", d_model=32, layers=2, prompt=4, max_new=4, requests=8, slots=4
    ),
    decoder_case(
        "d64-L2", d_model=64, layers=2, prompt=4, max_new=4, requests=8, slots=4
    ),
    decoder_case(
        "d128-L4", d_model=128, layers=4, prompt=4, max_new=4, requests=8, slots=4
    ),
)
"""Decoder shapes whose compiled circuits grow from a few thousand to tens of millions of gates."""
