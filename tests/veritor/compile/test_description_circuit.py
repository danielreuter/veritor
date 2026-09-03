from __future__ import annotations

import pytest

from veritor.compile.description import parse_description
from veritor.core import (
    Circuit,
    DescriptionCircuit,
    GateRef,
    InvalidArtifact,
    make_word_gate_set,
)
from veritor.core.description import Frame

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def lazy_and_flat(helpers, payload: bytes) -> tuple[DescriptionCircuit, object]:
    root = parse_description(payload, GATES).root
    return DescriptionCircuit(root, GATES), helpers.flatten(root, GATES)


def assert_same_circuit(helpers, lazy: DescriptionCircuit, flat) -> None:
    """``C[i]``, interfaces, sizes and costs of every copy agree with the flat scan."""

    assert (lazy.n, lazy.input_count, lazy.weight_count) == (
        flat.n,
        flat.input_count,
        flat.weight_count,
    )
    assert (list(lazy.inputs), list(lazy.weights), list(lazy.outputs)) == (
        list(flat.inputs),
        list(flat.weights),
        list(flat.outputs),
    )
    assert lazy.inputs == flat.inputs and lazy.weights == flat.weights
    for address in range(lazy.n):
        assert lazy[address] == flat[address], address
    for rank, address in enumerate(flat.inputs):
        assert lazy.input_rank(address) == flat.input_rank(address) == rank
    for rank, address in enumerate(flat.weights):
        assert lazy.weight_rank(address) == flat.weight_rank(address) == rank
    for frame in helpers.frames(lazy.frame):
        if frame.definition.size == 0:
            continue
        assert lazy.In(frame) == flat.In(frame.interval), frame
        # the declared interface plus the copy's source gates contain everything read from outside
        pinned = {a for a in frame.interval if lazy[a].is_source}
        assert set(lazy.Out(frame)) | pinned >= set(flat.Out(frame.interval)), frame
        assert not pinned & set(lazy.Out(frame))
        assert lazy.Size(frame) == flat.Size(frame.interval)
        assert lazy.Cost(frame) == flat.Cost(frame.interval)
        assert lazy.Cost(frame, "proof") == flat.Cost(frame.interval, "proof")


def test_lazy_circuit_matches_flat_on_matmul(helpers):
    k, cols, rows = 4, 3, 2
    lazy, flat = lazy_and_flat(helpers, helpers.matmul_payload(k, cols, rows))
    layout = helpers.matmul_layout(k, cols, rows)

    assert isinstance(lazy, Circuit)
    assert lazy.n == layout["n"] == rows * k + k * cols + rows * cols * 7
    assert lazy[0] == GateRef("in", (), 8, "input") and lazy[0].is_input
    assert (
        lazy[rows * k] == GateRef("weight", (), 8, "weight")
        and lazy[rows * k].is_weight
    )
    assert list(lazy.inputs) == list(layout["inputs"])
    assert list(lazy.weights) == list(layout["weights"])
    assert (lazy.input_count, lazy.weight_count) == (rows * k, k * cols)
    assert_same_circuit(helpers, lazy, flat)
    x = [(1, 2, 3, 4), (5, 6, 7, 8)]
    w = [tuple((i * 5 + j) % 11 for j in range(3)) for i in range(4)]
    inputs = tuple(v for row in x for v in row)
    weights = tuple(v for row in w for v in row)
    tape = lazy.evaluate(inputs, weights)
    assert tape == flat.evaluate(inputs, weights)
    assert (
        tape[: rows * k] == inputs and tape[rows * k : rows * k + k * cols] == weights
    )
    expected = [
        sum(x[r][i] * w[i][j] for i in range(4)) & 255
        for r in range(2)
        for j in range(3)
    ]
    assert [tape[o] for o in lazy.outputs] == expected


def test_interfaces_of_matmul_units_are_resolved_through_the_frame(helpers):
    k, cols, rows = 4, 3, 2
    lazy, _ = lazy_and_flat(helpers, helpers.matmul_payload(k, cols, rows))
    layout = helpers.matmul_layout(k, cols, rows)
    root = lazy.frame

    activations, weights = root.child(0, 0), root.child(1, 0)
    assert activations.interval == layout["inputs"] == range(rows * k)
    assert weights.interval == layout["weights"] == range(rows * k, rows * k + k * cols)
    for unit in (activations, weights):  # all their outputs are pinned
        assert lazy.In(unit) == () and lazy.Out(unit) == ()
        assert lazy.Size(unit) == len(unit.interval) and lazy.Cost(unit) == 0
        assert lazy.Cost(unit, "proof") == len(unit.interval)

    row1 = root.child(2, 1)
    assert row1.interval == layout["rows"][1]
    # x_1 and all of W, through the row's ports, in address order
    assert lazy.In(row1) == tuple(range(k, 2 * k)) + tuple(layout["weights"])
    # the three dot results of the second row
    assert lazy.Out(row1) == tuple(dot.stop - 1 for dot in layout["dots"][cols:])
    assert lazy.Size(row1) == cols * 7 and lazy.Cost(row1) == cols * (4 * 2 + 3)

    dot = row1.child(0, 2)  # column 2 of row 1
    assert dot.depth == 2 and dot.j == 2
    # x_1 (input gates) and column 2 of W (weight gates)
    assert lazy.In(dot) == tuple(range(k, 2 * k)) + tuple(
        rows * k + i * cols + 2 for i in range(k)
    )
    assert lazy.Out(dot) == (dot.base + 6,)
    assert lazy.Cost(dot, "proof") == 11


def test_declared_interface_may_exceed_what_is_read(helpers):
    h = helpers
    doc = h.Document()
    two = doc.add(
        h.body(
            1,
            [h.gate("add", h.rng(IN, 0, 2, 0)), h.gate("mul", h.rng(IN, 0, 2, 0))],
            [h.rng(LOC, 0, 2, 1)],  # declares both, only the first is used below
        )
    )
    target = doc.add(
        h.body(
            1,
            [h.call(two, h.rng(IN, 0)), h.gate("add", h.rng(LOC, 0, 2, 0))],
            [h.rng(LOC, 2)],
        )
    )
    lazy, flat = lazy_and_flat(helpers, doc.serialize(h.wrap(doc, target, 1, 1)))

    copy = lazy.frame.child(1, 0).child(0, 0)
    assert lazy.Out(copy) == (1, 2)
    assert flat.Out(copy.interval) == (1,)
    assert_same_circuit(helpers, lazy, flat)


def test_lazy_semantics_and_errors(helpers):
    lazy, _ = lazy_and_flat(helpers, helpers.matmul_payload(4, 2))

    assert lazy.evaluate_gate(lazy.n - 1, (200, 100)) == 44
    assert lazy.check_gate(lazy.n - 1, (200, 100), 44)
    assert lazy.decode(3, lazy.encode(3, 77)) == 77
    for source in (0, 4):  # an input gate, a weight gate
        assert lazy[source].is_source
        with pytest.raises(InvalidArtifact, match="is a source gate"):
            lazy.check_gate(source, (), 1)
        with pytest.raises(InvalidArtifact, match="is a source gate"):
            lazy.evaluate_gate(source, ())
    with pytest.raises(IndexError):
        lazy[lazy.n]
    with pytest.raises(TypeError, match="index nodes"):
        lazy.In(range(2))
    with pytest.raises(InvalidArtifact, match="expected 4 inputs, got 1"):
        lazy.evaluate((1,), (0,) * 8)
    with pytest.raises(InvalidArtifact, match="expected 8 weights, got 0"):
        lazy.evaluate((1, 2, 3, 4))
    with pytest.raises(KeyError):
        lazy.input_rank(4)  # a weight
    with pytest.raises(KeyError):
        lazy.weight_rank(lazy.n - 1)  # an add
    with pytest.raises(KeyError):
        lazy.weight_rank(lazy.n)
    frame = Frame.root(lazy.root)
    assert lazy.Size(frame) == lazy.n
