from __future__ import annotations

import json

import pytest

from veritor.compile.description import (
    FORMAT_VERSION,
    CompileError,
    canonical_description,
    definition_digest,
    description_digest,
    parse_description,
)
from veritor.core import CompilationLimits, make_word_gate_set
from veritor.core.description import CallStep, Frame, GateStep, PieceKind, Range, Run

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def parse(payload: bytes, limits: CompilationLimits | None = None):
    return parse_description(payload, GATES, limits)


def assert_matches_expansion(helpers, root) -> Frame:
    """Every ``C[i]`` by descent equals the O(n) reference expansion."""

    gates, outputs = helpers.expand(root)
    frame = Frame.root(root)
    assert root.size == len(gates)
    for address, (name, args) in enumerate(gates):
        gate, lazy_args = frame.gate(address)
        assert (gate.name, lazy_args) == (name, args), address
    assert [frame.output_address(k) for k in range(root.output_count)] == outputs
    return frame


# -- grammar and descent -----------------------------------------------------


def test_gate_call_and_passthrough_outputs(helpers):
    h = helpers
    doc = h.Document()
    square = doc.add(h.body(1, [h.gate("mul", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0)]))
    # identity: no gates at all, output is an input passthrough
    identity = doc.add(h.body(1, [], [h.rng(IN, 0)]))
    target = doc.add(
        h.body(
            2,
            [
                h.call(identity, h.rng(IN, 1)),  # slot 0 = input 1
                h.call(square, h.rng(LOC, 0)),  # slot 1 = mul(in1, in1)
                h.gate("add", h.rng(IN, 0), h.rng(LOC, 1)),  # slot 2
                h.gate("add", h.rng(LOC, 2), h.rng(LOC, 0)),  # slot 3
            ],
            [h.rng(LOC, 2, 2, 1), h.rng(LOC, 0), h.rng(IN, 0)],
        )
    )
    root = h.wrap(doc, target, 2, 4)  # the two inputs are `in` gates at 0 and 1
    description = parse(doc.serialize(root))

    assert description.digest == description_digest(doc.serialize(root))
    assert [d.digest for d in description.definitions][:3] == [square, identity, target]
    assert description.definitions[-1].digest == root
    r = description.root
    t = r.steps[1].child
    assert (t.input_count, t.size, t.slot_count, t.output_count) == (2, 3, 4, 4)
    assert t.depth == 1 and t.replay_cost == 4 and t.proof_cost == 4
    assert (r.input_count, r.size, r.slot_count, r.output_count) == (0, 5, 6, 4)
    assert (
        r.depth == 2 and r.replay_cost == 4 and r.proof_cost == 4 + 2
    )  # `in` gates: 0 and 1
    assert (r.input_total, r.weight_total) == (2, 0)
    frame = assert_matches_expansion(h, r)
    # gate addresses: 2 = mul(1, 1), 3 = add(0, 2), 4 = add(3, 1)
    assert frame.gate(0)[0].source == "input" and frame.gate(0)[1] == ()
    assert frame.gate(2)[1] == (1, 1)
    assert frame.gate(3)[1] == (0, 2)
    assert frame.gate(4)[1] == (3, 1)
    assert [frame.output_address(k) for k in range(4)] == [3, 4, 1, 0]
    assert h.evaluate(r, (3, 5)) == ((3 + 25) & 255, (28 + 5) & 255, 5, 3)
    # Out of the target: the two adds, one run; the pass-through outputs (slot 0, port 0) are dropped
    assert t.out_runs == (Run(1, 2, 1, 8),)
    assert (t.out_count, t.out_bits) == (2, 16)
    assert t.reads == (0, 1)
    # in the root the pass-through outputs resolve to `in` gates: pinned pieces, still not in Out
    assert r.resolved_outputs == (
        (PieceKind.GATE, Run(3, 2, 1, 8)),
        (PieceKind.PINNED, Run(1, 1, 0, 8)),
        (PieceKind.PINNED, Run(0, 1, 0, 8)),
    )
    assert r.out_runs == (Run(3, 2, 1, 8),) and r.reads == ()
    assert r.input_runs == (Run(0, 2, 1, 8),) and r.weight_runs == ()


def test_repeat_shifts_arguments_per_copy_and_descends_by_division(helpers):
    h = helpers
    doc = h.Document()
    mul = doc.add(h.body(2, [h.gate("mul", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    # dot(x[0:4], w[0:4]) = sum_j x_j * w_j as a tree
    dot = doc.add(
        h.body(
            8,
            [
                h.repeat(4, mul, h.jrng(IN, 0, 1, 0, 1), h.jrng(IN, 4, 1, 0, 1)),
                h.repeat(2, add, h.jrng(LOC, 0, 2, 1, 2)),
                h.gate("add", h.rng(LOC, 4, 2, 1)),
            ],
            [h.rng(LOC, 6)],
        )
    )
    description = parse(doc.serialize(h.wrap(doc, dot, 8, 1)))
    r = description.root.steps[1].child

    assert r.size == 7 and r.slot_count == 7
    steps = r.steps
    assert isinstance(steps[0], CallStep) and steps[0].count == 4
    assert steps[0].args[0] == Range(IN, 0, 1, 0, 1)
    assert r.step_address == (0, 4, 6, 7)
    frame = assert_matches_expansion(h, description.root)
    # copy j of the multiply reads x_j and w_j
    for j in range(4):
        assert frame.gate(8 + j)[1] == (j, 4 + j)
    assert frame.gate(12)[1] == (8, 9)
    assert frame.gate(13)[1] == (10, 11)
    assert frame.gate(14)[1] == (12, 13)
    x, w = (1, 2, 3, 4), (5, 6, 7, 8)
    assert h.evaluate(description.root, x + w) == (
        sum(a * b for a, b in zip(x, w)) & 255,
    )
    located, step = frame.locate(11)
    assert located.j == 3 and located.depth == 2 and located.interval == range(11, 12)
    assert isinstance(step, GateStep) and step.gate.name == "mul"


def test_strided_ranges_describe_matrix_columns_in_constant_size(helpers):
    h = helpers

    def matmul(k: int, cols: int) -> tuple[bytes, object]:
        doc = h.Document()
        mul = doc.add(h.body(2, [h.gate("mul", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
        add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
        # dot over k = 4 with a tree, then a row-times-matrix via strided columns
        dot = doc.add(
            h.body(
                2 * k,
                [
                    h.repeat(k, mul, h.jrng(IN, 0, 1, 0, 1), h.jrng(IN, k, 1, 0, 1)),
                    h.repeat(k // 2, add, h.jrng(LOC, 0, 2, 1, 2)),
                    h.gate("add", h.rng(LOC, k, 2, 1)),
                ],
                [h.rng(LOC, k + k // 2)],
            )
        )
        row = doc.add(
            h.body(
                k + k * cols,
                # copy j reads x (all k of it) and column j of W: start k + j, stride cols
                [
                    h.repeat(
                        cols, dot, h.jrng(IN, 0, k, 1, 0), h.jrng(IN, k, k, cols, 1)
                    )
                ],
                [h.rng(LOC, 0, cols, 1)],
            )
        )
        return doc.serialize(h.wrap(doc, row, k + k * cols, cols)), doc

    payload_2, _ = matmul(4, 2)
    payload_16, _ = matmul(4, 16)
    # the description does not grow with the number of columns (only its digits do)
    assert abs(len(payload_2) - len(payload_16)) <= 8

    root = parse(payload_16).root
    r = root.steps[1].child
    assert r.size == 16 * 7 and r.output_count == 16 and root.size == 68 + 16 * 7
    frame = assert_matches_expansion(helpers, root)
    x = tuple(range(1, 5))
    w_rows = [tuple((i * 3 + j) % 7 for j in range(16)) for i in range(4)]
    flat_w = tuple(v for row_ in w_rows for v in row_)
    expected = tuple(
        sum(x[i] * w_rows[i][j] for i in range(4)) & 255 for j in range(16)
    )
    assert h.evaluate(root, x + flat_w) == expected
    # the last multiply of column 15 reads x_3 and W[3][15]
    assert frame.gate(r.input_count + 15 * 7 + 3)[1] == (3, 4 + 3 * 16 + 15)
    assert r.reads == tuple(range(r.input_count))
    assert r.steps[0].child.reads == tuple(range(8))


def test_nested_repeat_frames_carry_copy_indices(helpers):
    h = helpers
    doc = h.Document()
    inc = doc.add(h.body(1, [h.gate("add", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0)]))
    pair = doc.add(
        h.body(2, [h.repeat(2, inc, h.jrng(IN, 0, 1, 0, 1))], [h.rng(LOC, 0, 2, 1)])
    )
    triple = doc.add(
        h.body(6, [h.repeat(3, pair, h.jrng(IN, 0, 2, 1, 2))], [h.rng(LOC, 0, 6, 1)])
    )
    r = parse(doc.serialize(h.wrap(doc, triple, 6, 6))).root

    frame = assert_matches_expansion(h, r)
    for address in range(6, 12):
        located, _ = frame.locate(address)
        assert located.depth == 3
        assert (located.parent.j, located.j) == divmod(address - 6, 2)
        assert located.parent.parent.parent is frame
    assert h.evaluate(r, (1, 2, 3, 4, 5, 6)) == (2, 4, 6, 8, 10, 12)


# -- source gates -------------------------------------------------------------


def test_source_gates_are_pinned_pieces_with_prefix_summed_ranks(helpers):
    """``in``/``weight`` steps count into the source totals, resolve to pinned pieces
    that ``Out`` drops, and are runs through repeats; a huge weight block is one run."""

    h = helpers
    doc = h.Document()
    weight_cell = h.source_cell(doc, "weight")
    in_cell = h.source_cell(doc, "in")
    weights = doc.add(
        h.body(
            0, [h.repeat(10**9, weight_cell)], [h.rng(LOC, 0, 10**9, 1)], role="replay"
        )
    )
    unit = doc.add(
        h.body(
            2,
            [
                h.call(in_cell),
                h.gate("mul", h.rng(LOC, 0), h.rng(IN, 0)),
                h.call(in_cell),
                h.gate("weight"),
                h.gate("add", h.rng(LOC, 1, 2, 2)),  # mul + weight
            ],
            [h.rng(LOC, 0, 5, 1), h.rng(IN, 1)],
            role="verification",
        )
    )
    root = doc.add(
        h.body(
            0,
            [h.call(weights), h.repeat(3, unit, h.jrng(LOC, 0, 2, 1, 2))],
            [h.rng(LOC, 10**9, 18, 1)],
        )
    )
    description = parse(doc.serialize(root))
    r = description.root
    w, u = r.steps[0].child, r.steps[1].child

    assert (
        weight_cell_def := w.steps[0].child
    ).size == 1 and weight_cell_def.weight_total == 1
    assert weight_cell_def.resolved_outputs == ((PieceKind.PINNED, Run(0, 1, 0, 8)),)
    assert weight_cell_def.out_runs == () and weight_cell_def.weight_runs == (
        Run(0, 1, 0, 8),
    )
    assert w.size == w.weight_total == 10**9 and w.input_total == 0
    assert (
        w.weight_runs == (Run(0, 10**9, 1, 8),)
        and w.out_runs == ()
        and w.out_count == 0
    )
    assert w.resolved_outputs == ((PieceKind.PINNED, Run(0, 10**9, 1, 8)),)
    assert (u.size, u.input_total, u.weight_total) == (5, 2, 1)
    assert u.step_input == (0, 1, 1, 2, 2, 2) and u.step_weight == (0, 0, 0, 0, 1, 1)
    assert u.input_runs == (Run(0, 2, 2, 8),) and u.weight_runs == (Run(3, 1, 0, 8),)
    assert u.out_runs == (Run(1, 2, 3, 8),) and u.out_count == 2  # the mul and the add
    assert u.resolved_outputs == (
        (PieceKind.PINNED, Run(0, 1, 0, 8)),
        (PieceKind.GATE, Run(1, 1, 0, 8)),
        (PieceKind.PINNED, Run(2, 2, 1, 8)),  # the second input and the weight, merged
        (PieceKind.GATE, Run(4, 1, 0, 8)),
        (PieceKind.PORT, Run(1, 1, 0, 0)),
    )
    assert (r.input_total, r.weight_total, r.size) == (6, 10**9 + 3, 10**9 + 15)
    assert r.step_input == (0, 0, 6) and r.step_weight == (0, 10**9, 10**9 + 3)
    # copies at stride 5 lift the unit's runs as one progression per member: two interleaved runs
    assert r.input_runs == (Run(10**9, 3, 5, 8), Run(10**9 + 2, 3, 5, 8))
    assert r.weight_runs == (Run(0, 10**9, 1, 8), Run(10**9 + 3, 3, 5, 8))
    # whole copies declared through the repeat lift the unit's two gate pieces once each
    assert r.out_runs == (Run(10**9 + 1, 3, 5, 8), Run(10**9 + 4, 3, 5, 8))
    # the unit's pass-through port resolves to a weight of the block: pinned in the root, not Out
    assert all(kind is not PieceKind.PORT for kind, _ in r.resolved_outputs)
    pinned = {
        run.element(k)
        for kind, run in r.resolved_outputs
        if kind is PieceKind.PINNED
        for k in range(run.count)
    }
    assert pinned == {1, 3, 5} | {
        10**9 + 5 * j + o for j in range(3) for o in (0, 2, 3)
    }

    frame = Frame.root(r)
    for rank in range(6):
        address = frame.source_address("input", rank)
        assert address == 10**9 + 5 * (rank // 2) + 2 * (rank % 2)
        assert frame.source_rank("input", address) == rank
        assert frame.source_rank("weight", address) is None
    assert (
        frame.source_address("weight", 0) == 0
        and frame.source_address("weight", 10**9 - 1) == 10**9 - 1
    )
    assert frame.source_address("weight", 10**9 + 2) == 10**9 + 13
    assert frame.source_rank("weight", 10**9 + 8) == 10**9 + 1
    assert frame.source_rank("weight", 10**9 + 1) is None
    assert frame.source_rank("weight", 5) == 5 and frame.source_rank("input", 5) is None


def test_root_ports_and_repeated_source_gates_are_rejected(helpers):
    h = helpers
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(
        CompileError, match="the root has no ports; inputs are `in` gates"
    ):
        parse(doc.serialize(add))
    wrapped = h.wrap(doc, add, 2, 1)
    assert parse(doc.serialize(wrapped)).root.input_count == 0

    doc = h.Document()
    twice = doc.add(
        h.body(0, [h.gate("in")], [h.rng(LOC, 0), h.rng(LOC, 0)], role="verification")
    )
    with pytest.raises(
        CompileError, match="the gate at offset 0 as an output more than once"
    ):
        parse(doc.serialize(twice))
    doc = h.Document()
    cell = h.source_cell(doc, "weight")
    block = doc.add(
        h.body(0, [h.repeat(4, cell)], [h.rng(LOC, 0, 4, 1), h.rng(LOC, 2)])
    )
    with pytest.raises(
        CompileError, match="the gate at offset 2 as an output more than once"
    ):
        parse(doc.serialize(block))


def test_source_gate_runs_are_bounded_per_definition(helpers):
    """A definition whose weight gates fall into many runs is rejected without the work."""

    h = helpers
    doc = h.Document()
    cell = h.source_cell(doc, "weight")
    # a weight beside an add: two gates per copy, so `repeat n` of it is one run of stride 2 ...
    pair = doc.add(
        h.body(
            1, [h.call(cell), h.gate("add", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0, 2, 1)]
        )
    )
    block = doc.add(
        h.body(1, [h.repeat(1000, pair, h.jrng(IN, 0))], [h.rng(LOC, 0, 2000, 1)])
    )
    parsed = parse(doc.serialize(h.wrap(doc, block, 1, 2000)))
    assert next(d for d in parsed.definitions if d.digest == block).weight_runs == (
        Run(0, 1000, 2, 8),
    )
    # ... while 300 copies of a 300-weight block with 301 gates make 300 interleaved runs
    doc = h.Document()
    cell = h.source_cell(doc, "weight")
    inner = doc.add(
        h.body(
            1,
            [h.repeat(300, cell), h.gate("add", h.rng(IN, 0, 2, 0))],
            [h.rng(LOC, 300)],
        )
    )
    outer = doc.add(
        h.body(1, [h.repeat(300, inner, h.jrng(IN, 0))], [h.rng(LOC, 0, 300, 1)])
    )
    payload = doc.serialize(h.wrap(doc, outer, 1, 300))
    with pytest.raises(
        CompileError, match="weight gates in more than max_output_runs = 256 runs"
    ):
        parse(payload)
    relaxed = parse(payload, CompilationLimits(max_output_runs=300))
    outer_definition = next(d for d in relaxed.definitions if d.digest == outer)
    assert len(outer_definition.weight_runs) == 300
    assert outer_definition.out_runs == (Run(300, 300, 301, 8),)


# -- rejection ---------------------------------------------------------------


def test_canonical_encoding_is_enforced(helpers):
    h = helpers
    doc = h.Document()
    target = doc.add(h.body(1, [h.gate("add", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0)]))
    root = h.wrap(doc, target, 1, 1)
    payload = doc.serialize(root)
    assert parse(payload).root.digest == root

    pretty = json.dumps(doc.value(root), indent=1).encode()
    with pytest.raises(CompileError, match="canonically serialized"):
        parse(pretty)
    with pytest.raises(CompileError, match="duplicate JSON key"):
        parse(
            b'{"definitions":[],"root":"'
            + b"0" * 64
            + b'","root":"'
            + b"0" * 64
            + b'","version":2}'
        )
    with pytest.raises(CompileError, match="valid UTF-8 JSON"):
        parse(b"\xff")
    with pytest.raises(CompileError, match="must be bytes"):
        parse(payload.decode())  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="unsupported description format"):
        parse(canonical_description({**doc.value(root), "version": FORMAT_VERSION + 1}))
    with pytest.raises(CompileError, match="does not match its digest"):
        tampered = json.loads(payload)
        tampered["definitions"][0]["body"]["input_count"] = 2  # the target's ports
        parse(canonical_description(tampered))
    with pytest.raises(CompileError, match="root names a definition"):
        parse(canonical_description({**doc.value(root), "root": "0" * 64}))
    with pytest.raises(CompileError, match="max_description_bytes"):
        parse(payload, CompilationLimits(max_description_bytes=10))


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"input_count": 1, "steps": [], "outputs": [[IN, 0]]}, "has keys"),
        (
            {"input_count": 1, "role": "x", "steps": [], "outputs": [[IN, 0, 1, 0]]},
            "role must be",
        ),
        (
            {"input_count": 1, "role": None, "steps": [], "outputs": []},
            "declares no outputs",
        ),
        (
            {"input_count": 1, "role": None, "steps": [], "outputs": [[IN, 1, 1, 0]]},
            "reaches input coordinate 1; only 1",
        ),
        (
            {"input_count": 1, "role": None, "steps": [], "outputs": [[LOC, 0, 1, 0]]},
            "reaches local coordinate 0; only 0",
        ),
        (
            {"input_count": 2, "role": None, "steps": [], "outputs": [[IN, 0, 2]]},
            "must have 4 elements",
        ),
        (
            {
                "input_count": 2,
                "role": None,
                "steps": [],
                "outputs": [["slot", 0, 1, 0]],
            },
            "space must be",
        ),
        (
            {
                "input_count": 2,
                "role": None,
                "steps": [{"kind": "leaf"}],
                "outputs": [[IN, 0, 1, 0]],
            },
            "unknown kind 'leaf'",
        ),
        (
            {
                "input_count": 2,
                "role": None,
                "steps": [{"kind": "gate", "gate": "sub", "args": [[IN, 0, 2, 1]]}],
                "outputs": [[LOC, 0, 1, 0]],
            },
            "unknown gate 'sub'",
        ),
        (
            {
                "input_count": 3,
                "role": None,
                "steps": [{"kind": "gate", "gate": "add", "args": [[IN, 0, 3, 1]]}],
                "outputs": [[LOC, 0, 1, 0]],
            },
            "passes 3 arguments to add, which has arity 2",
        ),
        (
            {
                "input_count": 2,
                "role": None,
                "steps": [
                    {"kind": "call", "digest": "0" * 64, "args": [[IN, 0, 2, 1]]}
                ],
                "outputs": [[LOC, 0, 1, 0]],
            },
            "not defined earlier",
        ),
        (
            {
                "input_count": 2,
                "role": None,
                "steps": [
                    {"kind": "repeat", "count": 0, "digest": "0" * 64, "args": []}
                ],
                "outputs": [[IN, 0, 1, 0]],
            },
            "count must be an integer >= 1",
        ),
    ],
)
def test_malformed_definitions_are_rejected(helpers, definition, message):
    with pytest.raises(CompileError, match=message):
        parse(helpers.single(definition))


def test_call_arity_and_repeat_bounds(helpers):
    h = helpers
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))

    with pytest.raises(CompileError, match="passes 1 arguments to a definition with 2"):
        parse(
            doc.serialize(
                doc.add(h.body(2, [h.call(add, h.rng(IN, 0))], [h.rng(LOC, 0)]))
            )
        )
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    # copy 3 would read input 4 of 4
    with pytest.raises(CompileError, match="reaches input coordinate 4; only 4"):
        parse(
            doc.serialize(
                doc.add(
                    h.body(
                        4, [h.repeat(4, add, h.jrng(IN, 0, 2, 1, 1))], [h.rng(LOC, 0)]
                    )
                )
            )
        )
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(CompileError, match="must have 5 elements"):
        parse(
            doc.serialize(
                doc.add(
                    h.body(4, [h.repeat(2, add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)])
                )
            )
        )
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(CompileError, match="appears twice"):
        doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
        parse(doc.serialize(add))


def test_limits_bound_every_summary_without_unrolling(helpers):
    h = helpers
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    big = doc.add(
        h.body(2, [h.repeat(10**12, add, h.jrng(IN, 0, 2, 1, 0))], [h.rng(LOC, 0)])
    )
    payload = doc.serialize(h.wrap(doc, big, 2, 1))

    assert parse(payload).root.size == 10**12 + 2
    with pytest.raises(CompileError, match="gates; the limit is 1000"):
        parse(payload, CompilationLimits(max_addresses=1000))
    with pytest.raises(CompileError, match="replay cost"):
        parse(payload, CompilationLimits(max_cost=1000))
    with pytest.raises(CompileError, match="nesting depth"):
        parse(payload, CompilationLimits(max_depth=0))
    with pytest.raises(CompileError, match="max_steps_per_definition"):
        parse(payload, CompilationLimits(max_steps_per_definition=0))
    with pytest.raises(CompileError, match="max_definitions"):
        parse(payload, CompilationLimits(max_definitions=1))
    # the circuit's `in` gates count against the address space like any gate
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    small = doc.serialize(h.wrap(doc, add, 2, 1))
    assert parse(small, CompilationLimits(max_addresses=3)).root.size == 3
    with pytest.raises(CompileError, match="has 3 gates; the limit is 2"):
        parse(small, CompilationLimits(max_addresses=2))


def test_definition_digest_is_tagged_and_body_sensitive(helpers):
    h = helpers
    a = h.body(1, [], [h.rng(IN, 0)])
    b = h.body(1, [], [h.rng(IN, 0)], role="replay")
    assert definition_digest(a) != definition_digest(b)
    assert len(definition_digest(a)) == 64
