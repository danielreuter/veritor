"""Tiny compiled circuits the analysis tests enumerate exhaustively."""

from __future__ import annotations

import random
from collections.abc import Iterable

import pytest

from veritor.compile import Compiler
from veritor.constructors import Tracer
from veritor.core import Compiled, make_word_gate_set

GATE_SET = make_word_gate_set(8)


def build_compiled(replay_sizes: Iterable[int], width: int = 8) -> Compiled:
    """Replay unit ``r`` holds ``replay_sizes[r]`` one-gate verification units.

    Every verification unit doubles one of its own inputs, so the units are
    independent and the circuit's outputs are exactly the unit outputs.
    """

    gate_set = make_word_gate_set(width)
    sizes = tuple(replay_sizes)
    total = sum(sizes)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key="double", role="verification")
    def double(v):
        return add(v[0], v[0])

    def replay(size: int):
        @tracer.definition(input_count=size, key=("replay", size), role="replay")
        def unit(v):
            return tracer.repeat(size, double, v[0].by(1))

        return unit

    @tracer.definition(input_count=total, key="root")
    def root(v):
        outputs = []
        offset = 0
        for size in sizes:
            outputs.append(replay(size)(v[offset : offset + size]))
            offset += size
        return outputs

    return Compiler(gate_set).compile(tracer.serialize(root), [1] * total)


def paper_example(width: int = 2, split: bool = False) -> Compiled:
    """The paper's 8-gate fan-in circuit with ``h`` as the verification unit.

    ``h(u, v, w) = (u+v, v+w, u+w)`` is applied twice, then two more gates
    combine the second application's outputs: eight gates, three declared
    outputs per ``h``.  With ``split`` the first ``h`` is its own replay
    unit and the rest form a second one; otherwise the root is the single
    replay unit.
    """

    gate_set = make_word_gate_set(width)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")

    @tracer.definition(input_count=3, key="h", role="verification")
    def h(v):
        return add(v[0], v[1]), add(v[1], v[2]), add(v[0], v[2])

    @tracer.definition(input_count=3, key="tail", role="verification")
    def tail(v):
        t = add(v[0], v[1])
        return add(t, v[2])

    if not split:

        @tracer.definition(input_count=3, key="fanin", role="replay")
        def fanin(v):
            a, b, c = h(v[0], v[1], v[2])
            p, q, r = h(a, b, c)
            return tail(p, q, r), b

        return Compiler(gate_set).compile(tracer.serialize(fanin), [1, 2, 3])

    @tracer.definition(input_count=3, key="first", role="replay")
    def first(v):
        return h(v[0], v[1], v[2])

    @tracer.definition(input_count=3, key="rest", role="replay")
    def rest(v):
        p, q, r = h(v[0], v[1], v[2])
        return tail(p, q, r)

    @tracer.definition(input_count=3, key="fanin-split")
    def root(v):
        a, b, c = first(v[0], v[1], v[2])
        return rest(a, b, c), b

    return Compiler(gate_set).compile(tracer.serialize(root), [1, 2, 3])


def random_compiled(seed: int, width: int = 2, max_gates: int = 8) -> Compiled:
    """A random small circuit: 1-2 replay units of 1-3 multi-gate verification units.

    Gates read earlier values (inputs or previous gates of the same unit or
    of earlier units, through declared interfaces), so units are chained
    and their interfaces are real cuts.
    """

    rng = random.Random(seed)
    gate_set = make_word_gate_set(width)
    tracer = Tracer(gate_set)
    ops = [tracer.gate("add"), tracer.gate("mul")]
    layout = [[rng.randint(1, 2) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(1, 2))]
    while sum(map(sum, layout)) > max_gates:
        layout[-1].pop()
        if not layout[-1]:
            layout.pop()
    input_count = rng.randint(2, 3)

    def unit_definition(r: int, u: int, gates: int, inputs: int):
        @tracer.definition(input_count=inputs, key=("unit", r, u, gates, inputs), role="verification")
        def unit(v):
            values = list(v)
            outputs = []
            for _ in range(gates):
                a, b = rng.choice(values), rng.choice(values)
                value = rng.choice(ops)(a, b)
                values.append(value)
                outputs.append(value)
            return outputs

        return unit

    def replay_definition(r: int, units: list[int], inputs: int):
        @tracer.definition(input_count=inputs, key=("replay", r, tuple(units), inputs), role="replay")
        def unit(v):
            values = list(v)
            outputs = []
            for u, gates in enumerate(units):
                args = [rng.choice(values) for _ in range(inputs)]
                produced = unit_definition(r, u, gates, inputs)(*args)
                produced = list(produced) if gates > 1 else [produced]
                values.extend(produced)
                outputs.extend(produced)
            return outputs

        return unit

    @tracer.definition(input_count=input_count, key=("root", seed))
    def root(v):
        values = list(v)
        outputs = []
        for r, units in enumerate(layout):
            args = [rng.choice(values) for _ in range(input_count)]
            produced = replay_definition(r, units, input_count)(*args)
            produced = list(produced) if sum(units) > 1 else [produced]
            values.extend(produced)
            outputs = produced
        return outputs

    return Compiler(gate_set).compile(tracer.serialize(root), list(range(1, input_count + 1)))


@pytest.fixture(scope="session")
def make_compiled():
    return build_compiled


@pytest.fixture(scope="session")
def make_paper_example():
    return paper_example


@pytest.fixture(scope="session")
def make_random_compiled():
    return random_compiled
