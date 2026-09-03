"""Check outputs: parsing, validation and the bits they take off ``Out``."""

from __future__ import annotations

import pytest

from veritor.compile import Compiler
from veritor.compile.description import CompileError, parse_description
from veritor.core import make_word_gate_set
from veritor.core.description import Check, Run

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def build(
    helpers,
    checks: list[list[int]] | None,
    *,
    copies: int = 1,
    inner_checks: list[list[int]] | None = None,
):
    """A root over ``copies`` replay units of ``(in x; add x x; mul x x)`` declaring ``checks``.

    Each unit declares its input gate (pinned), the sum and the square; the
    root declares, per copy, the same three outputs in the order ``in, add,
    mul``.
    """

    h = helpers
    doc = h.Document()
    in_cell = h.source_cell(doc, "in")
    pair_body = h.body(
        1,
        [h.gate("add", h.rng(IN, 0, 2, 0)), h.gate("mul", h.rng(IN, 0, 2, 0))],
        [h.rng(LOC, 0, 2, 1)],
        role="verification",
    )
    if inner_checks is not None:
        pair_body["checks"] = inner_checks
    pair = doc.add(pair_body)
    unit = doc.add(
        h.body(
            0,
            [h.call(in_cell), h.call(pair, h.rng(LOC, 0))],
            [h.rng(LOC, 0, 3, 1)],
            role="replay",
        )
    )
    if copies == 1:
        root_body = h.body(0, [h.call(unit)], [h.rng(LOC, 0, 3, 1)])
    else:
        root_body = h.body(0, [h.repeat(copies, unit)], [h.rng(LOC, 0, 3 * copies, 1)])
    if checks is not None:
        root_body["checks"] = checks
    return doc.serialize(doc.add(root_body))


def test_checks_are_parsed_and_taken_off_out_bits(helpers) -> None:
    description = parse_description(build(helpers, [[1, 1, 0, 10]]), GATES)
    root = description.root
    assert root.checks == (Check(1, 1, 0, 10),)
    assert root.checked_runs == (Run(1, 1, 0, 8),)  # the add gate, address 1
    assert root.checked_bits == 8 and root.out_bits == 8 and root.out_count == 2
    unit = root.steps[0].child
    assert (
        unit.checks == () and unit.out_bits == 16
    )  # its own interface still carries both words

    compiled = Compiler(GATES).compile(build(helpers, [[1, 1, 0, 10]]), [5])
    assert compiled.checks == (Check(1, 1, 0, 10),)
    assert list(compiled.check_values()) == [(1, 10)]
    rows = {row.kind: row for row in compiled.kind_table().rows}
    assert rows[root.digest].out_bits == 8 and rows[root.digest].reach_bits == 8
    pair = rows[unit.steps[1].child.digest]
    assert (pair.out_bits, pair.reach_bits, pair.ancestor_bits, pair.cut_bits) == (
        16,
        8,
        8,
        8,
    )
    # the check-free description is a different circuit
    plain = Compiler(GATES).compile(build(helpers, None), [5])
    assert plain.digest != compiled.digest and plain.checks == ()
    assert {row.kind: row.cut_bits for row in plain.kind_table().rows}[pair.kind] == 16


def test_a_circuit_whose_outputs_are_all_checks_has_no_out_bits(helpers) -> None:
    compiled = Compiler(GATES).compile(build(helpers, [[1, 2, 1, 0]]), [0])
    root = compiled.index.root.frame.definition
    assert root.out_bits == 0 and root.checked_bits == 16
    for row in compiled.kind_table().rows:
        assert row.reach_bits == 0 and row.ancestor_bits == 0 and row.cut_bits == 0


def test_strided_checks_over_repeat_copies(helpers) -> None:
    compiled = Compiler(GATES).compile(
        build(helpers, [[2, 3, 3, 25]], copies=3), [5, 5, 5]
    )
    root = compiled.index.root.frame.definition
    # the three squares: one gate per copy of the three-gate unit, at addresses 2, 5, 8
    assert root.checked_runs == (Run(2, 3, 3, 8),)
    assert root.out_bits == 3 * 16 - 3 * 8
    assert sorted(compiled.check_values()) == [(2, 25), (5, 25), (8, 25)]
    outputs = compiled.circuit.outputs
    assert [outputs[ordinal] for ordinal, _ in compiled.check_values()] == [2, 5, 8]
    rows = {row.kind: row for row in compiled.kind_table().rows}
    assert rows[root.digest].out_bits == 24 and rows[root.digest].reach_bits == 24
    unit = root.steps[0].child
    # one copy's share of the outputs (three slots, at most 8 bits each) less its checked square
    assert rows[unit.digest].reach_bits == 3 * 8 - 8


@pytest.mark.parametrize(
    ("checks", "detail"),
    [
        ([[3, 1, 0, 0]], "names output ordinal 3; only 3 are declared"),
        ([[1, 2, 2, 0]], "names output ordinal 3; only 3 are declared"),
        ([[1, 1, 0, 256]], "requires the value 256 of a 8-bit gate"),
        ([[1, 1, 0, 10], [1, 1, 0, 10]], "output ordinal 1 as a check more than once"),
        ([[1, 2, 0, 10]], "output ordinal 1 as a check more than once"),
        ([[1, 1, 0, 10], [0, 2, 1, 10]], "output ordinal 1 as a check more than once"),
        (
            [[0, 1, 0, 5]],
            "marks a pinned output; a check output must be a computed gate",
        ),
        ([], "empty checks list; the key must be omitted"),
        ([[1, 1, 0]], "must have 4 elements"),
        ([[1, 0, 0, 0]], "count must be an integer >= 1"),
        ([[1, 1, 0, -1]], "value must be an integer >= 0"),
        ([[1, 1, -1, 0]], "stride must be an integer >= 0"),
        ("checks", "checks must be a list"),
    ],
)
def test_invalid_checks_are_rejected(helpers, checks, detail) -> None:
    with pytest.raises(CompileError, match=detail):
        parse_description(build(helpers, checks), GATES)


def test_checks_below_the_root_are_rejected(helpers) -> None:
    with pytest.raises(CompileError, match="declares checks; only the root may"):
        parse_description(build(helpers, None, inner_checks=[[0, 1, 0, 0]]), GATES)


def test_a_port_passthrough_is_not_a_check_output(helpers) -> None:
    h = helpers
    doc = h.Document()
    passthrough = h.body(1, [], [h.rng(IN, 0)])
    passthrough["checks"] = [[0, 1, 0, 0]]
    with pytest.raises(CompileError, match="marks a port output"):
        parse_description(doc.serialize(doc.add(passthrough)), GATES)
