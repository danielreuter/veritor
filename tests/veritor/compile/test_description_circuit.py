from __future__ import annotations

import pytest

from veritor.compile.description import parse_description
from veritor.core import (
    INPUT_OP,
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

    assert (lazy.n, lazy.input_count, lazy.inputs, lazy.outputs) == (
        flat.n,
        flat.input_count,
        flat.inputs,
        flat.outputs,
    )
    for address in range(lazy.n):
        assert lazy[address] == flat[address], address
    for frame in helpers.frames(lazy.frame):
        if frame.definition.size == 0:
            continue
        assert lazy.In(frame) == flat.In(frame.interval), frame
        # the declared interface contains everything actually read from outside
        assert set(lazy.Out(frame)) >= set(flat.Out(frame.interval)), frame
        assert lazy.Size(frame) == flat.Size(frame.interval)
        assert lazy.Cost(frame) == flat.Cost(frame.interval)
        assert lazy.Cost(frame, "proof") == flat.Cost(frame.interval, "proof")


def matmul_payload(helpers, k: int, cols: int, rows: int = 1) -> bytes:
    """``rows`` copies of ``x_i W`` for a ``k x cols`` matrix ``W``."""

    h = helpers
    doc = h.Document()
    mul = doc.add(h.body(2, [h.gate("mul", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    dot = doc.add(
        h.body(
            2 * k,
            [
                h.repeat(k, mul, h.jrng(IN, 0, 1, 0, 1), h.jrng(IN, k, 1, 0, 1)),
                h.repeat(k // 2, add, h.jrng(LOC, 0, 2, 1, 2)),
                h.gate("add", h.rng(LOC, k, 2, 1)),
            ],
            [h.rng(LOC, k + k // 2)],
            role="verification",
        )
    )
    row = doc.add(
        h.body(
            k + k * cols,
            [h.repeat(cols, dot, h.jrng(IN, 0, k, 1, 0), h.jrng(IN, k, k, cols, 1))],
            [h.rng(LOC, 0, cols, 1)],
            role="replay",
        )
    )
    root = doc.add(
        h.body(
            rows * k + k * cols,
            [
                h.repeat(
                    rows,
                    row,
                    h.jrng(IN, 0, k, 1, k),
                    h.jrng(IN, rows * k, k * cols, 1, 0),
                )
            ],
            [h.rng(LOC, 0, rows * cols, 1)],
        )
    )
    return doc.serialize(root)


def test_lazy_circuit_matches_flat_on_matmul(helpers):
    lazy, flat = lazy_and_flat(helpers, matmul_payload(helpers, 4, 3, rows=2))

    assert isinstance(lazy, Circuit)
    assert lazy.n == 2 * 4 + 12 + 2 * 3 * 7
    assert lazy[0] == GateRef(INPUT_OP, (), 8)
    assert_same_circuit(helpers, lazy, flat)
    x = [(1, 2, 3, 4), (5, 6, 7, 8)]
    w = [tuple((i * 5 + j) % 11 for j in range(3)) for i in range(4)]
    inputs = tuple(v for row in x for v in row) + tuple(v for row in w for v in row)
    tape = lazy.evaluate(inputs)
    assert tape == flat.evaluate(inputs)
    expected = [
        sum(x[r][i] * w[i][j] for i in range(4)) & 255
        for r in range(2)
        for j in range(3)
    ]
    assert [tape[o] for o in lazy.outputs] == expected


def test_interfaces_of_matmul_units_are_resolved_through_the_frame(helpers):
    k, cols, rows = 4, 3, 2
    lazy, _ = lazy_and_flat(helpers, matmul_payload(helpers, k, cols, rows))
    root = lazy.frame
    n_in = rows * k + k * cols

    row1 = root.child(0, 1)
    assert row1.interval == range(n_in + cols * 7, n_in + 2 * cols * 7)
    # x_1 and all of W, in address order
    assert lazy.In(row1) == tuple(range(k, 2 * k)) + tuple(range(rows * k, n_in))
    # the three dot results of the second row
    assert lazy.Out(row1) == tuple(n_in + cols * 7 + 7 * j + 6 for j in range(cols))
    assert lazy.Size(row1) == cols * 7 and lazy.Cost(row1) == cols * (4 * 2 + 3)

    dot = row1.child(0, 2)  # column 2 of row 1
    assert dot.depth == 2 and dot.j == 2
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
    root = doc.add(
        h.body(
            1,
            [h.call(two, h.rng(IN, 0)), h.gate("add", h.rng(LOC, 0, 2, 0))],
            [h.rng(LOC, 2)],
        )
    )
    lazy, flat = lazy_and_flat(helpers, doc.serialize(root))

    copy = lazy.frame.child(0, 0)
    assert lazy.Out(copy) == (1, 2)
    assert flat.Out(copy.interval) == (1,)
    assert_same_circuit(helpers, lazy, flat)


def test_lazy_semantics_and_errors(helpers):
    lazy, _ = lazy_and_flat(helpers, matmul_payload(helpers, 4, 2))

    assert lazy.evaluate_gate(lazy.n - 1, (200, 100)) == 44
    assert lazy.check_gate(lazy.n - 1, (200, 100), 44)
    assert lazy.decode(3, lazy.encode(3, 77)) == 77
    with pytest.raises(InvalidArtifact, match="is an input"):
        lazy.check_gate(0, (), 1)
    with pytest.raises(IndexError):
        lazy[lazy.n]
    with pytest.raises(TypeError, match="index nodes"):
        lazy.In(range(2))
    with pytest.raises(InvalidArtifact, match="expected 12 inputs"):
        lazy.evaluate((1,))
    frame = Frame.root(lazy.root)
    assert lazy.Size(frame) == lazy.n - lazy.input_count
