"""The declared interfaces of a definition are runs, resolved once per definition.

``Out`` is a tuple of ``Run(start, count, stride, width)`` of gate offsets;
``In`` is the declared ``input_count``.  Everything derived from them (the
per-kind table, the boundary, the interiors, ``DescriptionCircuit.Out``) is
checked against brute force on small circuits, the distinctness rule is
exercised both ways, and the whole of admission is timed on descriptions
whose inputs are far larger than anything admission may touch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from fractions import Fraction

import pytest

from veritor.compile import Compiler
from veritor.compile.description import CompileError, parse_description
from veritor.constructors import Tracer
from veritor.core import (
    CompilationLimits,
    DescriptionCircuit,
    Index,
    VerificationPolicy,
    make_word_gate_set,
)
from veritor.core.description import Definition, Run
from veritor.protocol.parameters import expected_work

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def build(payload: bytes) -> tuple[Index, DescriptionCircuit]:
    root = parse_description(payload, GATES).root
    return Index(root), DescriptionCircuit(root, GATES)


# -- equivalence with enumeration --------------------------------------------------


def passthrough_payload(helpers) -> bytes:
    """Outputs that are inputs of the copy are not part of ``Out``.

    ``pair`` emits one gate and passes one input through; a replay unit
    repeats it and declares every slot; the root declares the units' slots
    along with one of its own inputs.
    """

    h = helpers
    doc = h.Document()
    pair = doc.add(
        h.body(
            2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0), h.rng(IN, 1)], role="verification"
        )
    )
    unit = doc.add(
        h.body(4, [h.repeat(2, pair, h.jrng(IN, 0, 2, 1, 2))], [h.rng(LOC, 0, 4, 1)], role="replay")
    )
    root = doc.add(
        h.body(8, [h.repeat(2, unit, h.jrng(IN, 0, 4, 1, 4))], [h.rng(LOC, 0, 8, 1), h.rng(IN, 3)])
    )
    return doc.serialize(root)


def interleaved_payload(helpers, count: int, stride: int) -> bytes:
    """A strided output range over three-output children: the non-affine path.

    ``h3`` has three gates and declares them in the order 2, 0, 1; ``block``
    holds five copies and declares ``count`` slots at ``stride`` from slot 0,
    which land on different child outputs unless ``stride`` is a multiple of
    three.  The root repeats the block twice and declares every slot.
    """

    h = helpers
    doc = h.Document()
    h3 = doc.add(
        h.body(
            1,
            [
                h.gate("add", h.rng(IN, 0, 2, 0)),
                h.gate("mul", h.rng(LOC, 0, 2, 0)),
                h.gate("add", h.rng(LOC, 1), h.rng(IN, 0)),
            ],
            [h.rng(LOC, 2), h.rng(LOC, 0), h.rng(LOC, 1)],
            role="verification",
        )
    )
    block = doc.add(
        h.body(1, [h.repeat(5, h3, h.jrng(IN, 0))], [h.rng(LOC, 0, count, stride)], role="replay")
    )
    root = doc.add(
        h.body(2, [h.repeat(2, block, h.jrng(IN, 0, 1, 0, 1))], [h.rng(LOC, 0, 2 * count, 1)])
    )
    return doc.serialize(root)


CASES: dict[str, Callable[[object], bytes]] = {
    "matmul-2x1x1": lambda h: h.matmul_payload(2, 1, 1),
    "matmul-4x3x2": lambda h: h.matmul_payload(4, 3, 2),
    "matmul-8x2x3": lambda h: h.matmul_payload(8, 2, 3),
    "shared-kinds": lambda h: h.shared_kinds_payload(),
    "passthrough": passthrough_payload,
    "interleaved-7x2": lambda h: interleaved_payload(h, 7, 2),
    "interleaved-4x4": lambda h: interleaved_payload(h, 4, 4),
    "interleaved-5x3": lambda h: interleaved_payload(h, 5, 3),
    "interleaved-3x1": lambda h: interleaved_payload(h, 3, 1),
    "interleaved-2x2": lambda h: interleaved_payload(h, 2, 2),
    "interleaved-2x4": lambda h: interleaved_payload(h, 2, 4),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_run_interfaces_match_enumeration(helpers, check_interfaces, case):
    index, circuit = build(CASES[case](helpers))
    check_interfaces(index, circuit)


def test_out_runs_are_what_the_output_ranges_resolve_to(helpers):
    k, cols, rows = 4, 3, 2
    index, _circuit = build(helpers.matmul_payload(k, cols, rows))
    root = index.root.frame.definition
    row = root.steps[0].child
    dot = row.steps[0].child
    dot_size = 2 * k - 1

    assert dot.out_runs == (Run(dot_size - 1, 1, 0, 8),)
    assert row.out_runs == (Run(dot_size - 1, cols, dot_size, 8),)
    assert root.out_runs == (Run(dot_size - 1, rows * cols, dot_size, 8),)
    assert (root.out_count, root.out_bits) == (rows * cols, 8 * rows * cols)
    assert [root.out_offset(r) for r in range(root.out_count)] == [
        dot_size - 1 + i * dot_size for i in range(rows * cols)
    ]


def test_strided_outputs_over_children_fall_back_to_residue_runs(helpers):
    index, _circuit = build(interleaved_payload(helpers, 7, 2))
    block = index.root.frame.definition.steps[0].child

    # slots 0, 2, ..., 12 of five copies: one progression per residue class of 2 mod 3
    assert block.out_runs == (Run(1, 2, 6, 8), Run(2, 3, 6, 8), Run(3, 2, 6, 8))
    assert block.out_count == 7 and block.out_bits == 56
    members = [1, 7, 2, 8, 14, 3, 9]
    assert [block.out_offset(r) for r in range(7)] == members
    assert [block.out_rank(offset) for offset in members] == list(range(7))
    assert all(block.out_rank(offset) is None for offset in range(15) if offset not in members)


@pytest.mark.parametrize(("count", "stride", "runs"), [(2, 2, (Run(1, 2, 1, 8),)), (2, 4, (Run(2, 2, 1, 8),))])
def test_fewer_elements_than_residue_classes_is_exact_enumeration(helpers, count, stride, runs):
    index, _circuit = build(interleaved_payload(helpers, count, stride))
    block = index.root.frame.definition.steps[0].child

    # slots 0 and `stride` land on different child outputs: one element each, then merged
    assert block.out_runs == runs
    assert block.out_count == count
    index, circuit = build(passthrough_payload(helpers))
    root = index.root.frame.definition
    unit = root.steps[0].child
    pair = unit.steps[0].child

    assert pair.output_count == 2 and pair.out_runs == (Run(0, 1, 0, 8),)
    assert unit.output_count == 4 and unit.out_runs == (Run(0, 2, 1, 8),)
    assert root.output_count == 9 and root.out_runs == (Run(0, 4, 1, 8),)
    assert list(circuit.outputs) == [8, 1, 9, 3, 10, 5, 11, 7, 3]
    assert list(circuit.Out(index.root)) == [8, 9, 10, 11]


def test_input_count_is_the_declared_interface_not_what_is_read(helpers):
    h = helpers
    doc = h.Document()
    cell = doc.add(h.body(3, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    root = doc.add(h.body(3, [h.call(cell, h.rng(IN, 0, 3, 1))], [h.rng(LOC, 0)], role="replay"))
    index, circuit = build(doc.serialize(root))
    node = index.verification_unit(0)

    assert {row.kind: row.input_count for row in index.kinds()} == {cell: 3, root: 3}
    assert node.frame.definition.reads == (0, 1) and list(circuit.In(node)) == [0, 1]
    assert index.root.frame.definition.reads == (0, 1)


# -- distinct declared outputs ---------------------------------------------------------


def test_the_same_slot_declared_twice_is_rejected(helpers):
    h = helpers
    for outputs in ([h.rng(LOC, 0), h.rng(LOC, 0)], [h.rng(LOC, 0, 2, 0)]):
        payload = h.single(h.body(1, [h.gate("add", h.rng(IN, 0, 2, 0))], outputs))
        with pytest.raises(CompileError, match="the gate at offset 0 as an output more than once"):
            parse_description(payload, GATES)


def ten_gates(helpers, outputs: list[list[object]]) -> bytes:
    h = helpers
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    return doc.serialize(doc.add(h.body(2, [h.repeat(10, add, h.jrng(IN, 0, 2, 1))], outputs)))


@pytest.mark.parametrize(
    ("outputs", "repeated"),
    [
        ([("local", 0, 3, 2), ("local", 2, 2, 2)], 2),  # {0, 2, 4} and {2, 4}
        ([("local", 0, 4, 3), ("local", 1, 3, 4)], 9),  # {0, 3, 6, 9} and {1, 5, 9}
        ([("local", 0, 10, 1), ("local", 7)], 7),
        ([("local", 9), ("local", 3, 3, 3)], 9),  # {9} and {3, 6, 9}
    ],
)
def test_overlapping_strided_ranges_are_rejected(helpers, outputs, repeated):
    payload = ten_gates(helpers, [helpers.rng(*item) for item in outputs])
    with pytest.raises(CompileError, match=f"the gate at offset {repeated} as an output more than once"):
        parse_description(payload, GATES)


def test_duplicates_through_passthrough_children_are_rejected(helpers):
    h = helpers
    doc = h.Document()
    identity = doc.add(h.body(1, [], [h.rng(IN, 0)]))
    inner = doc.add(h.body(1, [h.gate("add", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0)]))
    # two identity copies of the one gate: slots 1 and 2 both resolve to gate offset 0
    root = doc.add(
        h.body(
            1,
            [h.call(inner, h.rng(IN, 0)), h.repeat(2, identity, h.jrng(LOC, 0))],
            [h.rng(LOC, 1, 2, 1)],
        )
    )
    with pytest.raises(CompileError, match="the gate at offset 0 as an output more than once"):
        parse_description(doc.serialize(root), GATES)


def test_disjoint_interleaved_strides_are_accepted(helpers):
    h = helpers
    payload = ten_gates(
        helpers, [h.rng(LOC, 1, 3, 2), h.rng(LOC, 0, 3, 2), h.rng(LOC, 6, 2, 3), h.rng(LOC, 7)]
    )
    definition = parse_description(payload, GATES).root

    assert definition.out_runs == (Run(0, 3, 2, 8), Run(1, 3, 2, 8), Run(6, 2, 3, 8), Run(7, 1, 0, 8))
    assert definition.out_count == 9 and definition.out_bits == 72
    assert sorted(definition.out_offset(r) for r in range(9)) == [0, 1, 2, 3, 4, 5, 6, 7, 9]


# -- resolving an interface costs what it produces, and what it may produce is capped ----


def strided_subset_payload(
    helpers, n: int, count: int, stride: int, copies: int = 1, swapped: bool = False
) -> bytes:
    """``count`` slots at ``stride`` over ``copies`` copies of an ``n``-gate block.

    The block declares its gates in order, or as two swapped halves when
    ``swapped`` (so its interface is two runs and the gate offset is no
    longer affine in the slot).
    """

    h = helpers
    doc = h.Document()
    one = doc.add(h.body(1, [h.gate("add", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0)]))
    half = n // 2
    outputs = [h.rng(LOC, half, n - half, 1), h.rng(LOC, 0, half, 1)] if swapped else [h.rng(LOC, 0, n, 1)]
    block = doc.add(h.body(1, [h.repeat(n, one, h.jrng(IN, 0))], outputs))
    step = h.call(block, h.rng(IN, 0)) if copies == 1 else h.repeat(copies, block, h.jrng(IN, 0))
    return doc.serialize(doc.add(h.body(1, [step], [h.rng(LOC, 0, count, stride)])))


@pytest.mark.parametrize(
    ("n", "count", "stride", "copies"),
    [
        (3_000_000, 1_000_000, 3, 1),  # every third slot inside one copy
        (3_000_000, 1_000_000, 3, 3),  # ... crossing copies
        (1_000, 2_140, 1_001, 2_143),  # the diagonal: one element per copy, each a new residue
    ],
)
def test_a_strided_subset_of_a_huge_slot_linear_interface_is_one_run(helpers, n, count, stride, copies):
    start = time.perf_counter()
    root = parse_description(strided_subset_payload(helpers, n, count, stride, copies), GATES).root
    elapsed = time.perf_counter() - start

    assert root.out_runs == (Run(0, count, stride, 8),)
    assert elapsed < 0.1, elapsed


def test_interfaces_resolving_to_too_many_runs_are_rejected_without_doing_the_work(helpers):
    # the diagonal over a block whose interface is two swapped halves: one piece per element
    payload = strided_subset_payload(helpers, 1_000, 2_140, 1_001, 2_143, swapped=True)
    start = time.perf_counter()
    with pytest.raises(CompileError, match="more than max_output_runs = 256 runs"):
        parse_description(payload, GATES)
    assert time.perf_counter() - start < 0.1

    relaxed = CompilationLimits(max_output_runs=4_096)
    root = parse_description(payload, GATES, relaxed).root
    assert root.out_count == 2_140 and len(root.out_runs) > 256


def test_the_total_number_of_runs_over_a_description_is_capped(helpers):
    payload = strided_subset_payload(helpers, 10, 5, 2, swapped=True)
    description = parse_description(payload, GATES)
    total = sum(len(d.resolved_outputs) for d in description.definitions)
    assert total >= 3

    assert parse_description(payload, GATES, CompilationLimits(max_output_runs_total=total))
    with pytest.raises(CompileError, match="max_output_runs_total"):
        parse_description(payload, GATES, CompilationLimits(max_output_runs_total=total - 1))


# -- admission never touches the inputs -------------------------------------------------


def wide_description(copies: int, length: int) -> bytes:
    """One replay layer of ``copies`` dots, each over ``length`` weights and a shared vector.

    The root has ``length + copies * length`` inputs and passes them down as
    ranges; the description is ``O(log length)`` whatever ``copies`` is.
    """

    tracer = Tracer(GATES)
    mul, add = tracer.gate("mul"), tracer.gate("add")
    times = tracer.definition(input_count=2, key="times")(lambda v: mul(v[0], v[1]))
    plus = tracer.definition(input_count=2, key="plus")(lambda v: add(v[0], v[1]))

    @tracer.definition(input_count=2 * length, key=("dot", length), role="verification")
    def dot(v):
        x, w = v[:length], v[length:]
        level = tracer.repeat(length, times, x[0].by(1), w[0].by(1))
        while len(level) > 1:
            level = tracer.repeat(len(level) // 2, plus, level[0:2].by(2))
        return level[0]

    @tracer.definition(
        input_count=length + copies * length, key=("layer", copies, length), role="replay"
    )
    def layer(v):
        x, w = v[:length], v[length:]
        return tracer.repeat(copies, dot, x, w[0:length].by(length))

    return tracer.serialize(layer)


def admission(copies: int, length: int) -> Callable[[], float]:
    """Seconds to parse, index, tabulate the kinds, build the boundary and price one run."""

    description = wide_description(copies, length)
    input_count = length + copies * length
    policy = VerificationPolicy(Fraction(1, 2), 1)

    def timed() -> float:
        start = time.perf_counter()
        compiled = Compiler(GATES).compile(description, range(input_count))
        kinds = compiled.index.kinds()
        boundary = compiled.index.boundary()
        work = expected_work(compiled, policy, 2)
        elapsed = time.perf_counter() - start
        assert compiled.index.root.frame.definition.input_count == input_count
        assert boundary.count == input_count + copies
        assert {row.out_count for row in kinds} == {1, copies}
        assert work > 0
        return elapsed

    return timed


def fastest(timed: Callable[[], float], repetitions: int = 5) -> float:
    return min(timed() for _ in range(repetitions))


SMALL, LARGE, LENGTH = 1024, 16384, 1024


def test_admission_does_not_scale_with_the_input_count(monkeypatch):
    monkeypatch.setattr(
        Definition,
        "reads",
        property(lambda self: pytest.fail("admission must not enumerate what a definition reads")),
    )
    small = fastest(admission(SMALL, LENGTH))
    large = fastest(admission(LARGE, LENGTH))

    assert small < 0.25 and large < 0.25, (small, large)
    assert large < 3 * small + 0.005, (small, large)


def test_the_wide_layer_declares_and_reads_every_input():
    compiled = Compiler(GATES).compile(wide_description(4, 4), range(4 + 4 * 4))
    layer = compiled.index.root.frame.definition
    dot = compiled.index.verification_unit(0).frame.definition

    assert layer.out_runs == (Run(2 * 4 - 2, 4, 2 * 4 - 1, 8),)
    assert len(layer.reads) == layer.input_count == 20
    assert len(dot.reads) == dot.input_count == 8
    assert [row.input_count for row in compiled.index.kinds() if row.role == "replay"] == [20]
