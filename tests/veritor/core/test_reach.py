"""``KindSummary.reach_bits``: the circuit outputs a copy of a kind can reach.

The exact value is a forward search over the flat circuit from every copy
(:func:`veritor.analysis.reference.reach_bits`); :meth:`Index.kinds` bounds
its maximum over the copies at step granularity, without enumerating them.
Every table is checked for soundness against the brute force, and the toy
layouts and hand-built descriptions for the values the construction is
meant to give: a request reaches its own tokens, the steps of one wave
reach the tokens from themselves to the end of the wave, independent
iterations of a ``repeat`` keep their own outputs and a chain of steps
reaches from the first one.  A replay unit (RU) and a verification unit
(VU) inside it are charged ``min(out_bits, reach_bits)`` by ``Bound``.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest

from veritor import compile_matmul
from veritor.analysis.reference import reach_bits
from veritor.compile import Compiler
from veritor.constructors import (
    ClusterG,
    LMShape,
    Request,
    RequestsG,
    Tracer,
    schedule_fcfs,
)
from veritor.core import (
    Compiled,
    IndexNode,
    KindSummary,
    make_isa_gate_set,
    make_word_gate_set,
)
from veritor.core import index as index_module
from veritor.core.description import (
    INPUT,
    LOCAL,
    REPLAY,
    VERIFICATION,
    CallStep,
    Definition,
    GateStep,
    Range,
    Step,
    _split,
)
from veritor.core.index import _argument_grid, _Coverage, _segment_bits, _step_reach

from ..analysis.conftest import build_compiled, paper_example, random_compiled

WORDS = make_word_gate_set(8)
ISA = make_isa_gate_set(16)
LM = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)


def nodes(node: IndexNode) -> Iterator[IndexNode]:
    yield node
    for child in node.children():
        yield from nodes(child)


def exact_reach(compiled: Compiled) -> dict[str, int]:
    """Per kind, the widest reach of any of its copies, by brute force over the flat circuit.

    The outputs forward of every gate are found once, last gate first, as
    bitmasks; a node's reach is the union over its non-source gates.  On
    small circuits this is checked against the per-node search of
    :func:`veritor.analysis.reference.reach_bits`.
    """

    circuit = compiled.circuit
    n = circuit.n
    gates = [circuit[address] for address in range(n)]
    outputs = {address: gates[address].width for address in circuit.outputs}
    readers: list[list[int]] = [[] for _ in range(n)]
    for address, gate in enumerate(gates):
        for arg in gate.args:
            readers[arg].append(address)
    forward = [0] * n
    for address in reversed(range(n)):
        mask = 1 << address if address in outputs else 0
        for reader in readers[address]:
            mask |= forward[reader]
        forward[address] = mask
    best: dict[str, int] = {}
    for node in nodes(compiled.index.root):
        mask = 0
        for address in node.interval:
            if not gates[address].is_source:
                mask |= forward[address]
        bits = 0
        while mask:
            low = mask & -mask
            bits += outputs[low.bit_length() - 1]
            mask ^= low
        if n <= 256:
            assert bits == reach_bits(circuit, node)
        best[node.kind] = max(best.get(node.kind, 0), bits)
    return best


def rows_by_kind(compiled: Compiled) -> dict[str, KindSummary]:
    return {row.kind: row for row in compiled.index.kinds()}


def assert_sound(compiled: Compiled) -> dict[str, KindSummary]:
    """Every kind's ``reach_bits`` is at least the brute force and at most the whole output."""

    rows = rows_by_kind(compiled)
    exact = exact_reach(compiled)
    root = rows[compiled.index.root.kind]
    assert root.reach_bits == root.out_bits == exact[root.kind]
    assert root.out_bits == sum(compiled.circuit[address].width for address in compiled.circuit.outputs)
    for row in rows.values():
        assert exact[row.kind] <= row.reach_bits <= root.out_bits, row.kind
    return rows


def by_role(rows: dict[str, KindSummary], role: str | None) -> list[KindSummary]:
    return [row for row in rows.values() if row.role == role]


@pytest.mark.parametrize(
    "compiled",
    [
        build_compiled((3, 2)),
        build_compiled((2, 2, 2)),
        paper_example(split=False),
        paper_example(split=True),
    ]
    + [random_compiled(seed) for seed in range(6)]
    + [compile_matmul().compiled],
)
def test_reach_is_sound_and_below_the_whole_output(compiled: Compiled) -> None:
    assert_sound(compiled)


def test_the_fixtures_whose_replay_units_write_the_outputs_reach_exactly_their_interface() -> None:
    """Every RU's outputs are circuit outputs: its reach is its interface.

    A VU inside one is charged its word all the same: below the root the
    pieces of an RU's interface are not told apart, so a cell is given the
    whole RU's reach (three words, here) and keeps the narrower of that and
    its own word.
    """

    rows = assert_sound(build_compiled((3, 2)))
    exact = exact_reach(build_compiled((3, 2)))
    for unit in by_role(rows, REPLAY):
        assert unit.reach_bits == unit.out_bits == exact[unit.kind]
    for cell in by_role(rows, VERIFICATION):
        assert exact[cell.kind] == cell.out_bits == min(cell.out_bits, cell.reach_bits) == 8
        assert cell.reach_bits == max(unit.out_bits for unit in by_role(rows, REPLAY))


def test_the_paper_fanin_units_reach_less_than_their_interface() -> None:
    """``h`` writes three cells but only ``b`` and the tail's value leave the circuit."""

    rows = assert_sound(paper_example(split=False))
    exact = exact_reach(paper_example(split=False))
    h = next(row for row in by_role(rows, VERIFICATION) if row.out_bits == 6)
    tail = next(row for row in by_role(rows, VERIFICATION) if row.size == 2)
    assert h.reach_bits == exact[h.kind] == 4 < h.out_bits  # the first ``h`` reaches both outputs
    assert tail.reach_bits == exact[tail.kind] == 2 == tail.out_bits


def compile_requests(requests: tuple[Request, ...]) -> Compiled:
    description, inputs = RequestsG(LM)(requests, b"")
    return Compiler(ISA).compile(description, inputs)


def compile_cluster(requests: tuple[Request, ...], slots: int, steps: int) -> Compiled:
    constructor = ClusterG(LM, pods=1, slots=slots, steps=steps)
    schedule = schedule_fcfs(requests, 1, slots, steps)
    description, inputs = constructor(requests, schedule.encode())
    return Compiler(ISA).compile(description, inputs)


def test_a_request_and_everything_inside_it_reach_the_requests_own_tokens() -> None:
    prompt, generated, count = 2, 3, 4
    compiled = compile_requests(tuple(Request(tuple(range(prompt)), generated) for _ in range(count)))
    rows = assert_sound(compiled)
    exact = exact_reach(compiled)
    root = rows[compiled.index.root.kind]
    tokens = generated * LM.width

    assert root.out_bits == count * tokens
    weights, request = (row for row in by_role(rows, REPLAY) if row.copies in (1, count))
    if weights.copies != 1:
        weights, request = request, weights
    assert weights.out_bits == 0 and weights.reach_bits == root.out_bits  # read by every request
    assert request.copies == count and request.out_bits == request.reach_bits == exact[request.kind] == tokens
    # every kind below a request -- prefill, decode step, layer, matvec, row, dot, cell -- reaches
    # the request's tokens, however wide its interface
    inside = [row for row in rows.values() if row is not root and row is not weights and row.copies % count == 0]
    assert len(inside) > 10 and any(row.out_bits > tokens for row in inside)
    assert all(row.reach_bits == tokens for row in inside)
    # the profile-level charge: a cell keeps its word, a wide step drops to the tokens
    assert all(min(row.out_bits, row.reach_bits) == row.out_bits for row in inside if row.out_bits <= 16)
    assert all(min(row.out_bits, row.reach_bits) == tokens for row in inside if row.out_bits > tokens)


def test_the_steps_of_a_wave_are_chained_and_reach_to_the_end_of_their_wave() -> None:
    """Under FCFS on identical requests the run is ``waves`` of ``batch`` requests, ``generated`` steps each."""

    prompt, generated, batch, waves = 2, 3, 2, 2
    requests = tuple(Request(tuple(range(prompt)), generated) for _ in range(batch * waves))
    compiled = compile_cluster(requests, batch, generated * waves)
    rows = assert_sound(compiled)
    exact = exact_reach(compiled)
    root = rows[compiled.index.root.kind]
    wave_tokens = batch * generated * LM.width

    assert root.out_bits == waves * wave_tokens
    steps = sorted(
        (row for row in by_role(rows, REPLAY) if row.copies == waves),
        key=lambda row: -row.reach_bits,
    )
    assert len(steps) == generated
    # a step reads the previous step's tokens and cache: it reaches the tokens of itself and the
    # steps after it in its wave, exactly, and nothing of the next wave, whose prefill reads only
    # the weights and its own prompts
    assert [row.reach_bits for row in steps] == [wave_tokens - k * batch * LM.width for k in range(generated)]
    assert all(row.reach_bits == exact[row.kind] < row.out_bits for row in steps)
    # a single wave: the chain makes the first step reach the whole output
    single = compile_cluster(requests[:batch], batch, generated)
    single_rows = assert_sound(single)
    first = max(
        (row for row in by_role(single_rows, REPLAY) if row.out_bits),
        key=lambda row: row.reach_bits,
    )
    assert first.reach_bits == single_rows[single.index.root.kind].out_bits == wave_tokens


def test_a_staggered_schedule_chains_the_whole_run() -> None:
    """Requests of different lengths free their slots at different steps: the chain never breaks."""

    requests = (
        Request((1, 2), 3),
        Request((3,), 2),
        Request((4, 5), 2),
        Request((6,), 3),
        Request((7,), 1),
    )
    compiled = compile_cluster(requests, 2, 7)
    rows = assert_sound(compiled)
    root = rows[compiled.index.root.kind]
    steps = [row for row in by_role(rows, REPLAY) if row.out_bits]
    assert max(row.reach_bits for row in steps) == root.out_bits
    assert min(row.reach_bits for row in steps) < root.out_bits  # the last step reaches only its own tokens


def cells(tracer: Tracer):
    add, mul = tracer.gate("add"), tracer.gate("mul")
    plus = tracer.definition(input_count=2, key="plus", role="verification")(lambda v: add(v[0], v[1]))
    times = tracer.definition(input_count=2, key="times", role="verification")(lambda v: mul(v[0], v[1]))
    return plus, times


def compile_root(tracer: Tracer, root, input_count: int) -> Compiled:
    return Compiler(WORDS).compile(tracer.serialize(root), list(range(1, input_count + 1)))


def test_independent_repeat_iterations_keep_their_own_outputs() -> None:
    """Copy ``j`` of the repeat writes output ``j``: its reach is one word, not the whole output."""

    tracer = Tracer(WORDS)
    plus, times = cells(tracer)
    n = 5

    @tracer.definition(input_count=2, key="pair", role="replay")
    def pair(v):
        return plus(times(v[0], v[1]), v[0])

    @tracer.definition(input_count=0, key="sources", role="replay")
    def sources(_v):
        return tracer.inputs(n + 1)

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        x = sources()
        return tracer.repeat(n, pair, x[0].by(1), x[1].by(1))

    compiled = compile_root(tracer, root, n + 1)
    rows = assert_sound(compiled)
    exact = exact_reach(compiled)
    row = next(r for r in by_role(rows, REPLAY) if r.copies == n)
    assert rows[compiled.index.root.kind].out_bits == 8 * n
    assert row.reach_bits == exact[row.kind] == 8 == row.out_bits
    for cell in by_role(rows, VERIFICATION):
        if cell.out_bits:  # ``times`` feeds ``plus`` inside the same copy: both reach that copy's word
            assert cell.reach_bits == exact[cell.kind] == 8


def test_a_chain_of_steps_reaches_from_its_head() -> None:
    """Step ``k`` reads step ``k - 1``: the first step reaches every output, the last only its own."""

    tracer = Tracer(WORDS)
    plus, times = cells(tracer)
    n = 4

    def link(k: int):
        @tracer.definition(input_count=2, key=("link", k), role="replay")
        def unit(v):
            a = times(v[0], v[1])
            b = a
            for _ in range(k):  # ``k`` more cells, so the links are distinct kinds
                b = plus(b, v[1])
            return plus(b, v[1]), a  # 16-bit interface, both values read by the next link

        return unit

    @tracer.definition(input_count=0, key="sources", role="replay")
    def sources(_v):
        return tracer.inputs(2)

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        a, b = sources()
        outputs = []
        for k in range(n):
            a, b = link(k)(a, b)
            outputs.append(a)
        return outputs

    compiled = compile_root(tracer, root, 2)
    rows = assert_sound(compiled)
    exact = exact_reach(compiled)
    links = sorted((row for row in by_role(rows, REPLAY) if row.out_bits), key=lambda row: row.size)
    assert len(links) == n and all(row.out_bits == 16 for row in links)
    assert [row.reach_bits for row in links] == [8 * (n - k) for k in range(n)]
    assert all(row.reach_bits == exact[row.kind] for row in links)
    # the cells inside a link inherit the link's reach: ``times`` at the head reaches everything
    assert max(row.reach_bits for row in by_role(rows, VERIFICATION)) == 8 * n


def test_a_wide_interface_with_a_narrow_influence_is_charged_the_influence() -> None:
    """A VU writing three words of which one word's worth leaves the circuit."""

    tracer = Tracer(WORDS)
    add, mul = tracer.gate("add"), tracer.gate("mul")

    @tracer.definition(input_count=2, key="wide", role="verification")
    def wide(v):
        return add(v[0], v[1]), mul(v[0], v[1]), add(v[0], v[0])

    @tracer.definition(input_count=3, key="narrow", role="verification")
    def narrow(v):
        return add(add(v[0], v[1]), v[2])

    @tracer.definition(input_count=0, key="request", role="replay")
    def request(_v):
        x = tracer.inputs(2)
        return narrow(*wide(x[0], x[1]))

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        return [request() for _ in range(3)]

    compiled = compile_root(tracer, root, 6)
    rows = assert_sound(compiled)
    exact = exact_reach(compiled)
    wide_row = next(row for row in by_role(rows, VERIFICATION) if row.out_bits == 24)
    assert wide_row.reach_bits == exact[wide_row.kind] == 8 < wide_row.out_bits
    assert min(wide_row.out_bits, wide_row.reach_bits) == 8


def test_a_repeat_read_copy_by_copy_is_charged_the_whole_reader() -> None:
    """Copy ``j`` of ``plus`` reads copy ``j`` of ``times``: exact reach one word, step granularity all ``n``."""

    tracer = Tracer(WORDS)
    plus, times = cells(tracer)
    n = 4

    @tracer.definition(input_count=0, key="run", role="replay")
    def run(_v):
        w = tracer.inputs(n + 1)
        products = tracer.repeat(n, times, w[0].by(1), w[1].by(1))
        return tracer.repeat(n, plus, products[0].by(1), w[0].by(1))

    compiled = compile_root(tracer, run, n + 1)
    rows = assert_sound(compiled)
    exact = exact_reach(compiled)
    times_row = next(row for row in by_role(rows, VERIFICATION) if row.copies == n and row.size == 1)
    plus_row = next(row for row in by_role(rows, VERIFICATION) if row.copies == n and row.kind != times_row.kind)
    assert plus_row.reach_bits == exact[plus_row.kind] == 8  # an output, read by nothing
    assert exact[times_row.kind] == 8 and times_row.reach_bits == 8 * n  # sound, not tight: one step


# -- the interval sweep against the bitmask closure --------------------------------------
#
# ``_step_reach`` keeps ``Down`` as intervals of steps and sweeps them with a
# segment tree; the closure it replaced kept ``Down`` as a bitmask over the
# steps, Θ(S³ / w) on a chain of ``S`` steps.  The bitmask version is the
# oracle here: the sweep must agree with it exactly wherever it records the
# reads exactly (every argument run spanning at most ``_EXACT_READ_STEPS``
# steps and every closure of at most ``_MAX_DOWN_INTERVALS`` intervals, which
# any definition of at most 64 steps satisfies) and may only *exceed* it where
# it falls back to hulls.


def bitmask_step_reach(definition: Definition, total: int, exact: bool) -> list[int]:
    """The closure as bitmasks over the steps: ``Down(j)`` is ``j`` with the ``Down`` of its readers."""

    steps = definition.steps
    count = len(steps)
    readers = [0] * count
    for k, step in enumerate(steps):
        copies = step.count if isinstance(step, CallStep) else 1
        for item in step.args:
            if item.space != LOCAL:
                continue
            for start, run, stride, _, _ in _argument_grid(item, copies):
                for index, _, _ in _split(definition.step_slot, start, run, stride):
                    readers[index] |= 1 << k
    out = [0] * count
    share = [0] * count
    for item in definition.outputs:
        if item.space != LOCAL:
            continue
        for index, first, taken in _split(definition.step_slot, item.start, item.count, item.stride):
            if exact:
                width, single = _segment_bits(definition, index, item.element(first), taken, item.stride)
            else:
                width = single = total
            out[index] += width
            share[index] += single
    down = [0] * count
    reach = [0] * count
    for j in reversed(range(count)):
        mask = 1 << j
        rest = readers[j]
        while rest:
            low = rest & -rest
            mask |= down[low.bit_length() - 1]
            rest ^= low
        down[j] = mask
        bits = share[j]
        rest = mask ^ (1 << j)
        while rest and bits < total:
            low = rest & -rest
            bits += out[low.bit_length() - 1]
            rest ^= low
        reach[j] = min(bits, total)
    return reach


GATES = make_word_gate_set(8)
ADD, MUL, IN = GATES["add"], GATES["mul"], GATES["in"]


def _definition(key: str, input_count: int, steps: tuple[Step, ...], outputs: tuple[Range, ...]) -> Definition:
    return Definition(f"test-reach/{key}", input_count, steps, outputs, None)


def _port(k: int) -> Range:
    return Range(INPUT, k, 1, 0)


# the children a random definition calls: one gate, two gates, a port passed through beside a gate
# (a declared output carrying nothing) and a pinned source (an output carrying nothing either)
ONE = _definition("one", 1, (GateStep(ADD, (_port(0), _port(0))),), (Range(LOCAL, 0, 1, 0),))
TWO = _definition(
    "two",
    2,
    (GateStep(ADD, (Range(INPUT, 0, 2, 1),)), GateStep(MUL, (Range(INPUT, 0, 2, 1),))),
    (Range(LOCAL, 0, 2, 1),),
)
THROUGH = _definition("through", 1, (GateStep(ADD, (_port(0), _port(0))),), (_port(0), Range(LOCAL, 0, 1, 0)))
PINNED = _definition("pinned", 0, (GateStep(IN, ()),), (Range(LOCAL, 0, 1, 0),))


def local_range(rng: random.Random, slots: int, count: int, copies: int = 1) -> Range:
    """A ``LOCAL`` range of ``count`` elements over ``copies`` copies inside the ``slots`` so far."""

    for _ in range(64):
        stride = rng.choice((0, 1, 1, 1, 2, 3, 5, 7)) if count > 1 else 0
        jstride = rng.choice((0, 1, 1, 2, 3, 5)) if copies > 1 else 0
        span = (count - 1) * stride + (copies - 1) * jstride
        if span < slots:
            return Range(LOCAL, rng.randrange(slots - span), count, stride, jstride)
    return Range(LOCAL, rng.randrange(slots), count, 0, 0)  # one slot, repeated


def random_definition(rng: random.Random, count: int) -> Definition:
    """A root of ``count`` steps: sources, gates over earlier slots, calls and repeats of the children.

    Every argument and every declared output is a random progression over
    the slots of the earlier steps, so reads are dense or strided, span one
    step or many, and the copies of a ``repeat`` shift over the steps.
    """

    steps: list[Step] = []
    slots = 0
    for _ in range(count):
        roll = rng.random()
        step: Step
        if slots == 0 or roll < 0.1:
            step = GateStep(IN, ()) if rng.random() < 0.5 else CallStep.make(PINNED, (), rng.choice((1, 1, 3)))
        elif roll < 0.45:
            if rng.random() < 0.5:
                args: tuple[Range, ...] = (local_range(rng, slots, 2),)
            else:
                args = (local_range(rng, slots, 1), local_range(rng, slots, 1))
            step = GateStep(rng.choice((ADD, MUL)), args)
        else:
            child = rng.choice((ONE, TWO, TWO, THROUGH))
            copies = rng.choice((1, 1, 1, 2, 3, 4))
            pieces: list[Range] = []
            remaining = child.input_count
            while remaining:
                taken = rng.randint(1, remaining)
                pieces.append(local_range(rng, slots, taken, copies))
                remaining -= taken
            step = CallStep.make(child, tuple(pieces), copies)
        steps.append(step)
        slots += step.slots
    outputs = tuple(local_range(rng, slots, rng.randint(1, min(4, slots))) for _ in range(rng.randint(1, 3)))
    return _definition(f"random/{rng.random()}", 0, tuple(steps), outputs)


def both_reaches(definition: Definition, rng: random.Random) -> Iterator[tuple[list[int], list[int]]]:
    """``(sweep, bitmask)`` per step, as the root (exact segments) and as a child (``total`` or nothing)."""

    for exact, total in ((True, definition.out_bits), (False, 1 + rng.randrange(64))):
        yield _step_reach(definition, total, exact), bitmask_step_reach(definition, total, exact)


@pytest.mark.parametrize("seed", range(150))
def test_the_sweep_equals_the_bitmask_closure_on_definitions_of_at_most_64_steps(seed: int) -> None:
    rng = random.Random(seed)
    definition = random_definition(rng, rng.randint(1, 64))
    for sweep, bitmask in both_reaches(definition, rng):
        assert sweep == bitmask


@pytest.mark.parametrize("seed", range(60))
def test_the_sweep_never_falls_below_the_bitmask_closure_on_long_definitions(seed: int) -> None:
    """65 to 200 steps: strided runs may span more than 64 steps and closures more than 64 intervals."""

    rng = random.Random(1000 + seed)
    definition = random_definition(rng, rng.randint(65, 200))
    for sweep, bitmask in both_reaches(definition, rng):
        assert all(s >= b for s, b in zip(sweep, bitmask, strict=True))


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize(("read_steps", "intervals"), [(2, 1), (3, 2), (64, 1), (2, 64)])
def test_hulls_only_enlarge_the_reach(
    seed: int, read_steps: int, intervals: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the thresholds forced low, small definitions take the hull paths and stay sound."""

    monkeypatch.setattr(index_module, "_EXACT_READ_STEPS", read_steps)
    monkeypatch.setattr(index_module, "_MAX_DOWN_INTERVALS", intervals)
    rng = random.Random(2000 + seed)
    definition = random_definition(rng, rng.randint(2, 40))
    for sweep, bitmask in both_reaches(definition, rng):
        assert all(s >= b for s, b in zip(sweep, bitmask, strict=True))


def chain_definition(n: int) -> Definition:
    """A source, then ``n`` calls of ``ONE`` each reading the previous step; every call's output is declared."""

    steps: list[Step] = [GateStep(IN, ())]
    for k in range(n):
        steps.append(CallStep.make(ONE, (Range(LOCAL, k, 1, 0),)))
    return _definition(f"chain/{n}", 0, tuple(steps), (Range(LOCAL, 1, n, 1),))


def broadcast_definition(n: int) -> Definition:
    """A source read by ``n`` independent calls of ``ONE``; every call's output is declared."""

    steps: list[Step] = [GateStep(IN, ())]
    steps.extend(CallStep.make(ONE, (Range(LOCAL, 0, 1, 0),)) for _ in range(n))
    return _definition(f"broadcast/{n}", 0, tuple(steps), (Range(LOCAL, 1, n, 1),))


def kv_chain_definition(n: int) -> Definition:
    """A source, then ``n`` calls each reading *every* earlier slot (a KV cache), into cells of growing arity."""

    steps: list[Step] = [GateStep(IN, ())]
    for slots in range(1, n + 1):
        cell = _definition(
            f"kv-cell/{slots}", slots, (GateStep(ADD, (Range(INPUT, 0, 2, slots - 1),)),), (Range(LOCAL, 0, 1, 0),)
        )
        steps.append(CallStep.make(cell, (Range(LOCAL, 0, slots, 1),)))
    return _definition(f"kv/{n}", 0, tuple(steps), (Range(LOCAL, 1, n, 1),))


@pytest.mark.parametrize("build", [chain_definition, broadcast_definition, kv_chain_definition])
def test_the_flagship_shapes_are_exact_far_beyond_the_thresholds(build) -> None:
    """A chain, a broadcast and a KV chain of 300 steps: single-interval closures, so no hull is ever taken."""

    definition = build(300)
    assert len(definition.steps) > 64
    rng = random.Random(0)
    for sweep, bitmask in both_reaches(definition, rng):
        assert sweep == bitmask
    root = _step_reach(definition, definition.out_bits, True)
    assert root[0] == definition.out_bits == 300 * 8  # the source reaches everything in all three shapes
    assert root[-1] == 8  # the last step reaches its own word


def test_a_strided_read_over_many_steps_is_recorded_as_its_hull() -> None:
    """A ``repeat`` reading every other one of 200 independent steps: the sweep charges the ones it skips too."""

    n = 100
    steps: list[Step] = [GateStep(IN, ()), GateStep(IN, ())]
    for _ in range(n):  # ``a`` cells at even steps read source 0, ``b`` cells at odd steps read source 1
        steps.append(CallStep.make(ONE, (Range(LOCAL, 0, 1, 0),)))
        steps.append(CallStep.make(ONE, (Range(LOCAL, 1, 1, 0),)))
    steps.append(CallStep.make(ONE, (Range(LOCAL, 2, 1, 0, 2),), n))  # copy ``j`` reads cell ``a_j``
    definition = _definition("skipping", 0, tuple(steps), (Range(LOCAL, 2, 3 * n, 1),))
    sweep = _step_reach(definition, definition.out_bits, True)
    bitmask = bitmask_step_reach(definition, definition.out_bits, True)
    assert all(s >= b for s, b in zip(sweep, bitmask, strict=True))
    word, block = 8, n * 8
    assert bitmask[2] == sweep[2] == word + block  # ``a_0``: its word and the repeat, both ways
    assert bitmask[3] == word < sweep[3] == word + block  # ``b_0``: the repeat only through the hull
    assert bitmask[0] == sweep[0] == 2 * block  # source 0 reaches the ``a`` cells and the repeat
    assert bitmask[1] == block < sweep[1] == 2 * block  # source 1 is charged the repeat too


def test_a_closure_of_many_intervals_is_kept_as_its_hull() -> None:
    """Two interleaved chains of 100 links: the sweep charges the head of one chain the other's links too."""

    n = 100
    steps: list[Step] = [GateStep(IN, ()), GateStep(IN, ())]
    for k in range(n):  # chain ``a`` at even steps and chain ``b`` at odd ones, each link reading the previous
        steps.append(CallStep.make(ONE, (Range(LOCAL, 2 * k, 1, 0),)))
        steps.append(CallStep.make(ONE, (Range(LOCAL, 2 * k + 1, 1, 0),)))
    definition = _definition("interleaved", 0, tuple(steps), (Range(LOCAL, 2, 2 * n, 1),))
    sweep = _step_reach(definition, definition.out_bits, True)
    bitmask = bitmask_step_reach(definition, definition.out_bits, True)
    assert all(s >= b for s, b in zip(sweep, bitmask, strict=True))
    assert bitmask[0] == bitmask[2] == n * 8  # a head reaches its own chain
    assert sweep[0] > bitmask[0] and sweep[2] > bitmask[2]  # ... and, past 64 intervals, the hull
    assert sweep[-1] == bitmask[-1] == 8 and sweep[-2] == bitmask[-2] == 8  # the tails are exact
    assert sweep[2 * n - 2 * 30] == bitmask[2 * n - 2 * 30] == 31 * 8  # so are the last 64 links of a chain


def test_coverage_counts_match_a_plain_array() -> None:
    """Range adds of ``±1``, the weight of the covered positions and the covered intervals."""

    rng = random.Random(7)
    for count in (1, 2, 3, 5, 8, 13, 64, 100):
        out = [rng.randrange(4) for _ in range(count)]
        cover = _Coverage(out)
        plain = [0] * count
        active: list[tuple[int, int]] = []
        for _ in range(200):
            dirty: list[int] = []
            for _ in range(rng.choice((1, 1, 1, 2, 5))):  # several additions settle together
                if active and rng.random() < 0.4:
                    low, high = active.pop(rng.randrange(len(active)))
                    delta = -1
                else:
                    low = rng.randrange(count)
                    high = rng.randint(low + 1, count)
                    active.append((low, high))
                    delta = 1
                cover.add(low, high, delta, dirty)
                for p in range(low, high):
                    plain[p] += delta
            cover.settle(dirty)
            assert cover.covered_out() == sum(w for w, c in zip(out, plain, strict=True) if c > 0)
            first = min((p for p, c in enumerate(plain) if c > 0), default=count)
            if first == 0:
                continue  # ``intervals`` is only asked below every covered position
            expected = [(first - 1, first)]
            for p in range(first, count):
                if plain[p] > 0:
                    if expected[-1][1] == p:
                        expected[-1] = (expected[-1][0], p + 1)
                    else:
                        expected.append((p, p + 1))
            assert cover.intervals(first - 1, 1 << 30) == expected
            hull = [(first - 1, max(p for p, c in enumerate(plain) if c > 0) + 1)] if first < count else expected
            assert cover.intervals(first - 1, 1) in (expected, hull)
            if len(expected) > 1:
                assert cover.intervals(first - 1, 1) == hull
