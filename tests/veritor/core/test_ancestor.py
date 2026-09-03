"""``KindSummary.ancestor_bits``: the narrowest interface enclosing a copy of a kind.

The exact value is, per copy, the minimum of the declared interfaces of its
proper ancestors (:func:`veritor.analysis.reference.ancestor_bits`), the
root being given its own interface; :meth:`Index.kinds` computes its
maximum over the copies of a kind top-down over the definition DAG without
enumerating them.  The recursion is exact -- the maximum over paths of the
minimum along a path -- so every table is checked for equality with the
brute force, and the layouts the construction is meant for are checked by
hand: everything inside a request is enclosed by the request's word, a kind
called from two places keeps the wider of its two enclosures, and a wide
replay unit (RU) inside a narrow unmarked stage is charged the stage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest

from veritor import compile_matmul
from veritor.analysis.reference import ancestor_bits, out_bits
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
    KindTable,
    make_isa_gate_set,
    make_word_gate_set,
)
from veritor.core.description import REPLAY, VERIFICATION

from ..analysis.conftest import (
    bottlenecked,
    build_compiled,
    paper_example,
    random_compiled,
)

WORDS = make_word_gate_set(8)
ISA = make_isa_gate_set(16)
LM = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)


def chains(
    node: IndexNode, enclosing: list[int]
) -> Iterator[tuple[IndexNode, list[int]]]:
    """Every node with the interfaces of its proper ancestors, root first."""

    yield node, enclosing
    below = [*enclosing, node]
    for child in node.children():
        yield from chains(child, below)


def exact_ancestors(compiled: Compiled) -> dict[str, int]:
    """Per kind, the widest over its copies of the narrowest enclosing interface, by brute force."""

    circuit = compiled.circuit
    root = compiled.index.root
    widths = {}
    best: dict[str, int] = {}
    for node, enclosing in chains(root, []):
        if not enclosing:
            value = out_bits(circuit, node)
        else:
            value = min(widths[id(ancestor)] for ancestor in enclosing)
        widths[id(node)] = out_bits(circuit, node)
        if circuit.n <= 4096:
            assert value == ancestor_bits(circuit, node)
        best[node.kind] = max(best.get(node.kind, 0), value)
    return best


def rows_by_kind(compiled: Compiled) -> dict[str, KindSummary]:
    return {row.kind: row for row in compiled.index.kinds()}


def assert_exact(compiled: Compiled) -> dict[str, KindSummary]:
    """Every kind's ``ancestor_bits`` equals the brute force and is at most the whole output."""

    rows = rows_by_kind(compiled)
    exact = exact_ancestors(compiled)
    root = rows[compiled.index.root.kind]
    assert root.ancestor_bits == root.out_bits == exact[root.kind]
    for row in rows.values():
        assert row.ancestor_bits == exact[row.kind], row.kind
        assert 0 <= row.ancestor_bits <= root.out_bits
        assert row.cut_bits == min(row.out_bits, row.reach_bits, row.ancestor_bits)
    return rows


def by_role(rows: dict[str, KindSummary], role: str | None) -> list[KindSummary]:
    return [row for row in rows.values() if row.role == role]


def compile_requests(requests: tuple[Request, ...]) -> Compiled:
    description, inputs = RequestsG(LM)(requests, b"")
    return Compiler(ISA).compile(description, inputs)


def compile_cluster(requests: tuple[Request, ...], slots: int, steps: int) -> Compiled:
    constructor = ClusterG(LM, pods=1, slots=slots, steps=steps)
    schedule = schedule_fcfs(requests, 1, slots, steps)
    description, inputs = constructor(requests, schedule.encode())
    return Compiler(ISA).compile(description, inputs)


def compile_root(tracer: Tracer, root, input_count: int, gate_set=WORDS) -> Compiled:
    return Compiler(gate_set).compile(
        tracer.serialize(root), list(range(1, input_count + 1))
    )


@pytest.mark.parametrize(
    "compiled",
    [
        build_compiled((3, 2)),
        build_compiled((2, 2, 2)),
        paper_example(split=False),
        paper_example(split=True),
    ]
    + [random_compiled(seed) for seed in range(6)]
    + [compile_matmul().compiled]
    + [bottlenecked(3), bottlenecked(6, width=1)]
    + [
        compile_requests(tuple(Request((0, 1), 3) for _ in range(4))),
        compile_cluster(tuple(Request((0, 1), 3) for _ in range(4)), 2, 6),
        compile_cluster(
            (
                Request((1, 2), 3),
                Request((3,), 2),
                Request((4, 5), 2),
                Request((6,), 3),
            ),
            2,
            7,
        ),
    ],
)
def test_ancestor_bits_equal_the_brute_force_over_copies(compiled: Compiled) -> None:
    assert_exact(compiled)


def test_the_root_and_its_children_carry_the_whole_output() -> None:
    """The root has no ancestor; a kind the root calls is enclosed by the root alone."""

    compiled = build_compiled((3, 2))
    rows = assert_exact(compiled)
    root = rows[compiled.index.root.kind]
    for unit in by_role(rows, REPLAY):
        assert unit.ancestor_bits == root.out_bits == 40
        assert unit.cut_bits == unit.out_bits  # the root never binds below itself
    # a cell inside a unit is enclosed by the unit: its ancestor is the unit's interface
    for cell in by_role(rows, VERIFICATION):
        assert cell.ancestor_bits == max(
            unit.out_bits for unit in by_role(rows, REPLAY)
        )
        assert cell.cut_bits == cell.out_bits == 8


def test_everything_inside_a_request_is_enclosed_by_the_requests_tokens() -> None:
    prompt, generated, count = 2, 3, 4
    compiled = compile_requests(
        tuple(Request(tuple(range(prompt)), generated) for _ in range(count))
    )
    rows = assert_exact(compiled)
    root = rows[compiled.index.root.kind]
    tokens = generated * LM.width
    request = next(row for row in by_role(rows, REPLAY) if row.copies == count)
    assert request.out_bits == tokens and request.ancestor_bits == root.out_bits
    inside = [
        row
        for row in rows.values()
        if row is not root and row.copies % count == 0 and row is not request
    ]
    assert len(inside) > 10
    # a request is the narrowest interface above anything in it (the decode steps are wider), so the
    # ancestor cut is the request's word: the same cut the reach already gave
    assert all(row.ancestor_bits <= tokens for row in inside)
    assert all(
        row.ancestor_bits == row.reach_bits == tokens
        for row in inside
        if row.out_bits > tokens
    )


def test_a_kind_called_from_two_places_keeps_the_wider_enclosure() -> None:
    """``one`` is called inside a one-word ``stage`` and directly under the root: sound for both."""

    tracer = Tracer(WORDS)
    add, mul = tracer.gate("add"), tracer.gate("mul")

    @tracer.definition(input_count=2, key="one", role="verification")
    def one(v):
        return add(v[0], v[1])

    @tracer.definition(input_count=2, key="two", role="verification")
    def two(v):
        return add(v[0], v[1]), mul(v[0], v[1])

    @tracer.definition(input_count=2, key="unit", role="replay")
    def unit(v):
        return one(v[0], v[1])

    @tracer.definition(input_count=2, key="wide", role="replay")
    def wide(v):
        return two(v[0], v[1])

    @tracer.definition(input_count=2, key="stage")
    def stage(v):
        a, b = wide(v[0], v[1])
        return unit(a, b)

    @tracer.definition(input_count=0, key="sources", role="replay")
    def sources(_v):
        return tracer.inputs(4)

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        x = sources()
        return stage(x[0], x[1]), unit(x[2], x[3])

    compiled = compile_root(tracer, root, 4)
    rows = assert_exact(compiled)
    total = rows[compiled.index.root.kind].out_bits
    assert total == 16
    unit_row = next(row for row in by_role(rows, REPLAY) if row.copies == 2)
    wide_row = next(row for row in by_role(rows, REPLAY) if row.out_bits == 16)
    one_row = next(row for row in by_role(rows, VERIFICATION) if row.copies == 2)
    two_row = next(row for row in by_role(rows, VERIFICATION) if row.out_bits == 16)
    # ``unit`` inside the stage is enclosed by 8 bits, under the root by 16: the kind keeps 16
    assert unit_row.ancestor_bits == total and one_row.ancestor_bits == 8
    # ``wide`` only ever sits inside the stage: its 16-bit interface is cut down to the stage's word
    assert wide_row.ancestor_bits == 8 and wide_row.cut_bits == 8 < wide_row.out_bits
    assert two_row.ancestor_bits == 8 and two_row.cut_bits == 8 < two_row.out_bits


def test_a_wide_unit_in_a_narrow_stage_that_reaches_everything_is_charged_the_stage() -> (
    None
):
    """Neither the interface nor the reach binds a ``wide`` RU inside a one-word stage; the stage does."""

    compiled = bottlenecked(3)
    rows = assert_exact(compiled)
    total = rows[compiled.index.root.kind].out_bits
    wide = next(row for row in by_role(rows, REPLAY) if row.out_bits == 16)
    pairs = next(row for row in by_role(rows, VERIFICATION) if row.out_bits == 16)
    assert total == 24
    assert (
        wide.reach_bits == pairs.reach_bits == total
    )  # the stage's word feeds every output
    assert wide.ancestor_bits == pairs.ancestor_bits == 8
    assert wide.cut_bits == pairs.cut_bits == 8 < min(wide.out_bits, wide.reach_bits)


def test_the_table_checks_the_ancestor_bounds() -> None:
    """The root carries the whole output; a child is never narrower than any call site says."""

    table = bottlenecked(3).kind_table()
    rows = {row.kind: row for row in table.rows}
    root = rows[table.root]
    wide = next(row for row in table.rows if row.role == REPLAY and row.out_bits == 16)
    stage = next(
        row
        for row in table.rows
        if any(child == wide.kind for child, _ in row.children)
    )

    def rebuild(rows: tuple[KindSummary, ...]) -> KindTable:
        return KindTable(
            rows,
            table.root,
            table.n,
            table.input_count,
            table.weight_count,
            table.replay_unit_count,
            table.digest,
        )

    with pytest.raises(ValueError, match="has no ancestor"):
        rebuild(
            tuple(
                replace(row, ancestor_bits=8) if row is root else row
                for row in table.rows
            )
        )
    for bad in (root.out_bits + 1, -1):
        with pytest.raises(ValueError, match="claims ancestors"):
            rebuild(
                tuple(
                    replace(row, ancestor_bits=bad) if row is wide else row
                    for row in table.rows
                )
            )
    # narrower than what the stage encloses it by: unsound, rejected; wider is sound, but then the
    # kinds inside must be widened too (they are enclosed by ``wide`` through its own interface)
    assert min(stage.ancestor_bits, stage.out_bits) == 8 == wide.ancestor_bits
    with pytest.raises(ValueError, match="claims ancestors of 7 bits but is called by"):
        rebuild(
            tuple(
                replace(row, ancestor_bits=7) if row is wide else row
                for row in table.rows
            )
        )
    with pytest.raises(ValueError, match="claims ancestors of 8 bits but is called by"):
        rebuild(
            tuple(
                replace(row, ancestor_bits=root.out_bits) if row is wide else row
                for row in table.rows
            )
        )
    pairs = next(
        row for row in table.rows if row.role == VERIFICATION and row.out_bits == 16
    )
    rebuild(
        tuple(
            replace(row, ancestor_bits=root.out_bits) if row is pairs else row
            for row in table.rows
        )
    )
    rebuild(tuple(replace(row, ancestor_bits=root.out_bits) for row in table.rows))
