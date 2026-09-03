"""The declared interfaces of a definition are runs, resolved once per definition.

``Out`` is a tuple of ``Run(start, count, stride, width)`` of gate offsets;
``In`` is the declared ``input_count``.  Everything derived from them (the
per-kind table, the boundary, the interiors, ``DescriptionCircuit.Out``) is
checked against brute force on small circuits, the distinctness rule is
exercised both ways, and the whole of admission is timed on descriptions
whose inputs are far larger than anything admission may touch.
"""

from __future__ import annotations

import random
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
from veritor.core.description import CallStep, Definition, Run
from veritor.protocol.parameters import expected_work

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def build(payload: bytes) -> tuple[Index, DescriptionCircuit]:
    root = parse_description(payload, GATES).root
    return Index(root), DescriptionCircuit(root, GATES)


def target_of(index: Index) -> Definition:
    """The ported definition under a ``helpers.wrap`` root (its second step)."""

    step = index.root.frame.definition.steps[1]
    assert isinstance(step, CallStep)
    return step.child


# -- equivalence with enumeration --------------------------------------------------


def passthrough_payload(helpers) -> bytes:
    """Outputs that are ports of the copy are not part of ``Out``.

    ``pair`` emits one gate and passes one port through; a replay unit
    repeats it and declares every slot; the target declares the units' slots
    along with one of its own ports (one no unit passes through: under the
    ``wrap`` root the ports are ``in`` gates, and a pinned gate may be
    declared only once), and sits under a ``wrap`` root whose eight ``in``
    gates feed it.
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
    target = doc.add(
        h.body(8, [h.repeat(2, unit, h.jrng(IN, 0, 4, 1, 4))], [h.rng(LOC, 0, 8, 1), h.rng(IN, 2)])
    )
    return doc.serialize(h.wrap(doc, target, 8, 9))


def interleaved_payload(helpers, count: int, stride: int) -> bytes:
    """A strided output range over three-output children: the non-affine path.

    ``h3`` has three gates and declares them in the order 2, 0, 1; ``block``
    holds five copies and declares ``count`` slots at ``stride`` from slot 0,
    which land on different child outputs unless ``stride`` is a multiple of
    three.  The target repeats the block twice and declares every slot.
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
    target = doc.add(
        h.body(2, [h.repeat(2, block, h.jrng(IN, 0, 1, 0, 1))], [h.rng(LOC, 0, 2 * count, 1)])
    )
    return doc.serialize(h.wrap(doc, target, 2, 2 * count))


def within_copy_payload(helpers) -> bytes:
    """Two slots at stride 2 over a six-output child whose copy holds three such slots.

    ``six`` is two ``h3`` in a row (outputs at gate offsets 2, 0, 1, 5, 3, 4:
    not affine in the slot); a replay ``block`` repeats it twice and declares
    slots 0 and 2 of the first copy only.  The within-copy branch of the
    resolver visits one copy and must stop at the two slots asked for, not at
    the last slot of the copy at that stride (slot 4, gate 3 -- distinct from
    the two declared gates, so the distinctness rule would not notice).
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
        )
    )
    six = doc.add(h.body(1, [h.repeat(2, h3, h.jrng(IN, 0))], [h.rng(LOC, 0, 6, 1)], role="verification"))
    block = doc.add(h.body(1, [h.repeat(2, six, h.jrng(IN, 0))], [h.rng(LOC, 0, 2, 2)], role="replay"))
    target = doc.add(h.body(2, [h.repeat(2, block, h.jrng(IN, 0, 1, 0, 1))], [h.rng(LOC, 0, 4, 1)]))
    return doc.serialize(h.wrap(doc, target, 2, 4))


def random_strided_payload(helpers, seed: int) -> bytes | None:
    """Random progressions of slots declared over a repeated leaf with permuted outputs.

    The leaf has ``m`` gates and declares them in a random order, so a slot's
    gate is not affine in the slot; a replay ``block`` repeats it and declares
    one to three random ``(start, count, stride)`` runs of slots, which visit
    whole copies, partial copies, one copy or several residue classes at
    random.  ``None`` when the runs name a gate twice (rejected, tested
    elsewhere).
    """

    rng = random.Random(seed)
    h = helpers
    doc = h.Document()
    m = rng.randint(1, 4)
    steps = [h.gate("add", h.rng(IN, 0, 2, 0))]
    for g in range(1, m):
        if rng.random() < 0.25:  # a source gate among the leaf's gates: a pinned piece when declared
            steps.append(h.gate(rng.choice(("in", "weight"))))
        else:
            steps.append(h.gate("mul", h.rng(LOC, rng.randrange(g)), h.rng(IN, 0)))
    order = list(range(m))
    rng.shuffle(order)
    leaf = doc.add(h.body(1, steps, [h.rng(LOC, p) for p in order], role="verification"))
    copies = rng.randint(1, 5)
    slots = copies * m
    runs: list[tuple[int, int, int]] = []
    gates: list[int] = []
    for _ in range(rng.randint(1, 3)):
        start = rng.randrange(slots)
        stride = rng.randint(1, max(1, slots // 2))
        count = rng.randint(1, 1 + (slots - 1 - start) // stride)
        runs.append((start, count, stride if count > 1 else 0))
        for k in range(count):
            copy, ordinal = divmod(start + k * stride, m)
            gates.append(copy * m + order[ordinal])
    if len(set(gates)) != len(gates):
        return None
    block = doc.add(
        h.body(
            1,
            [h.repeat(copies, leaf, h.jrng(IN, 0))],
            [h.rng(LOC, start, count, stride) for start, count, stride in runs],
            role="replay",
        )
    )
    reps = rng.randint(1, 2)
    target = doc.add(
        h.body(reps, [h.repeat(reps, block, h.jrng(IN, 0, 1, 0, 1))], [h.rng(LOC, 0, reps * len(gates), 1)])
    )
    return doc.serialize(h.wrap(doc, target, reps, reps * len(gates)))


def sources_payload(helpers) -> bytes:
    """Source gates everywhere they may be: a weight block, a unit mixing inputs,
    weights and gates whose declared outputs include them, and a strided repeat."""

    h = helpers
    doc = h.Document()
    weight_cell = h.source_cell(doc, "weight")
    weights = doc.add(h.body(0, [h.repeat(6, weight_cell)], [h.rng(LOC, 0, 6, 1)], role="replay"))
    unit = doc.add(
        h.body(
            2,
            [
                h.gate("in"),
                h.gate("mul", h.rng(LOC, 0), h.rng(IN, 0)),
                h.gate("in"),
                h.gate("weight"),
                h.gate("add", h.rng(LOC, 1, 2, 2)),
            ],
            [h.rng(LOC, 0, 5, 1), h.rng(IN, 1)],
            role="verification",
        )
    )
    layer = doc.add(
        h.body(6, [h.repeat(3, unit, h.jrng(IN, 0, 2, 1, 2))], [h.rng(LOC, 0, 18, 1)], role="replay")
    )
    root = doc.add(
        h.body(0, [h.call(weights), h.call(layer, h.rng(LOC, 0, 6, 1))], [h.rng(LOC, 6, 18, 1)])
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
    "within-copy": within_copy_payload,
    "sources": sources_payload,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_run_interfaces_match_enumeration(helpers, check_interfaces, case):
    index, circuit = build(CASES[case](helpers))
    check_interfaces(index, circuit)


def test_out_runs_are_what_the_output_ranges_resolve_to(helpers):
    k, cols, rows = 4, 3, 2
    index, _circuit = build(helpers.matmul_payload(k, cols, rows))
    layout = helpers.matmul_layout(k, cols, rows)
    root = index.root.frame.definition
    activations, weights, row = (step.child for step in root.steps)
    dot = row.steps[0].child
    dot_size = 2 * k - 1

    assert dot.out_runs == (Run(dot_size - 1, 1, 0, 8),)
    assert activations.out_runs == () and activations.input_runs == (Run(0, rows * k, 1, 8),)
    assert weights.out_runs == () and weights.weight_runs == (Run(0, k * cols, 1, 8),)
    assert row.out_runs == (Run(dot_size - 1, cols, dot_size, 8),)
    assert row.input_runs == row.weight_runs == ()
    # the rows hold nothing but dots, so their Out tiles the copies: one run over the batch
    assert root.out_runs == (Run(rows * k + k * cols + dot_size - 1, rows * cols, dot_size, 8),)
    assert (root.out_count, root.out_bits) == (rows * cols, 8 * rows * cols)
    assert [root.out_offset(r) for r in range(root.out_count)] == [
        dot.stop - 1 for dot in layout["dots"]
    ]
    assert root.input_runs == (Run(0, rows * k, 1, 8),)
    assert root.weight_runs == (Run(rows * k, k * cols, 1, 8),)


def test_strided_outputs_over_children_fall_back_to_residue_runs(helpers):
    index, _circuit = build(interleaved_payload(helpers, 7, 2))
    block = target_of(index).steps[0].child

    # slots 0, 2, ..., 12 of five copies: one progression per residue class of 2 mod 3
    assert block.out_runs == (Run(1, 2, 6, 8), Run(2, 3, 6, 8), Run(3, 2, 6, 8))
    assert block.out_count == 7 and block.out_bits == 56
    members = [1, 7, 2, 8, 14, 3, 9]
    assert [block.out_offset(r) for r in range(7)] == members
    assert [block.out_rank(offset) for offset in members] == list(range(7))
    assert all(block.out_rank(offset) is None for offset in range(15) if offset not in members)


def test_random_strided_declarations_match_enumeration(helpers, check_interfaces):
    """The run resolver against the per-ordinal resolver on random strided declarations."""

    checked = 0
    for seed in range(400):
        payload = random_strided_payload(helpers, seed)
        if payload is None:
            continue
        index, circuit = build(payload)
        block = target_of(index).steps[0].child
        assert sum(run.count for _, run in block.resolved_outputs) == block.output_count
        check_interfaces(index, circuit)
        checked += 1
    assert checked > 200


def test_a_strided_run_inside_one_copy_stops_at_the_declared_count(helpers):
    index, circuit = build(within_copy_payload(helpers))
    block = target_of(index).steps[0].child
    six = block.steps[0].child

    assert six.output_count == 6 and six.out_runs == (Run(0, 6, 1, 8),)
    # slots 0 and 2 of copy 0 are gates 2 and 1 of `six`; slot 4 (gate 3) is not declared
    assert block.output_count == 2 and block.out_runs == (Run(1, 2, 1, 8),)
    assert block.out_count == 2
    # two `in` gates, then two block copies of twelve gates: gates 2 and 1 of each copy
    assert list(circuit.outputs) == [4, 3, 16, 15]
    assert sorted(circuit.Out(index.root)) == [3, 4, 15, 16]


@pytest.mark.parametrize(("count", "stride", "runs"), [(2, 2, (Run(1, 2, 1, 8),)), (2, 4, (Run(2, 2, 1, 8),))])
def test_fewer_elements_than_residue_classes_is_exact_enumeration(helpers, count, stride, runs):
    index, _circuit = build(interleaved_payload(helpers, count, stride))
    block = target_of(index).steps[0].child

    # slots 0 and `stride` land on different child outputs: one element each, then merged
    assert block.out_runs == runs
    assert block.out_count == count
    index, circuit = build(passthrough_payload(helpers))
    root = index.root.frame.definition
    target = target_of(index)
    unit = target.steps[0].child
    pair = unit.steps[0].child

    assert pair.output_count == 2 and pair.out_runs == (Run(0, 1, 0, 8),)
    assert unit.output_count == 4 and unit.out_runs == (Run(0, 2, 1, 8),)
    assert target.output_count == 9 and target.out_runs == (Run(0, 4, 1, 8),)
    # in the root the ports become `in` gates (pinned) and the target's gates shift past them
    assert root.output_count == 9 and root.out_runs == (Run(8, 4, 1, 8),)
    assert list(circuit.outputs) == [8, 1, 9, 3, 10, 5, 11, 7, 2]
    assert list(circuit.Out(index.root)) == [8, 9, 10, 11]


def test_input_count_is_the_declared_interface_not_what_is_read(helpers):
    h = helpers
    doc = h.Document()
    cell = doc.add(h.body(3, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    target = doc.add(h.body(3, [h.call(cell, h.rng(IN, 0, 3, 1))], [h.rng(LOC, 0)], role="replay"))
    index, circuit = build(doc.serialize(h.wrap(doc, target, 3, 1)))
    node = index.verification_unit(3)  # after the three `in` cells of the input block

    assert node.kind == cell
    assert {row.kind: row.input_count for row in index.kinds()}.items() >= {cell: 3, target: 3}.items()
    assert node.frame.definition.reads == (0, 1) and list(circuit.In(node)) == [0, 1]
    assert target_of(index).reads == (0, 1) and index.root.frame.definition.reads == ()


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
    target = doc.add(h.body(2, [h.repeat(10, add, h.jrng(IN, 0, 2, 1))], outputs))
    return doc.serialize(h.wrap(doc, target, 2, sum(int(item[2]) for item in outputs)))  # type: ignore[call-overload]


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
    root = parse_description(payload, GATES).root
    definition = root.steps[1].child

    # the two interleaved runs at stride 2 are woven into the six gates they cover
    assert definition.out_runs == (Run(0, 6, 1, 8), Run(6, 2, 3, 8), Run(7, 1, 0, 8))
    assert definition.out_count == 9 and definition.out_bits == 72
    assert sorted(definition.out_offset(r) for r in range(9)) == [0, 1, 2, 3, 4, 5, 6, 7, 9]
    # the same interface seen from the wrapping root, shifted past its two `in` gates
    assert root.out_runs == (Run(2, 6, 1, 8), Run(8, 2, 3, 8), Run(9, 1, 0, 8))


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
    target = doc.add(h.body(1, [step], [h.rng(LOC, 0, count, stride)]))
    return doc.serialize(h.wrap(doc, target, 1, count))


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

    assert root.steps[1].child.out_runs == (Run(0, count, stride, 8),)
    assert root.out_runs == (Run(1, count, stride, 8),)  # shifted past the root's one `in` gate
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


def scattered_repeat_payload(helpers, copies: int) -> bytes:
    """``copies`` copies of a six-gate child declaring gates 5, 2 and 0 (three pieces), all slots declared."""

    h = helpers
    doc = h.Document()
    child = doc.add(
        h.body(
            1,
            [h.gate("add", h.rng(IN, 0, 2, 0))] + [h.gate("mul", h.rng(LOC, k, 2, 0)) for k in range(5)],
            [h.rng(LOC, 5), h.rng(LOC, 2), h.rng(LOC, 0)],
            role="replay",
        )
    )
    target = doc.add(h.body(1, [h.repeat(copies, child, h.jrng(IN, 0))], [h.rng(LOC, 0, 3 * copies, 1)]))
    return doc.serialize(h.wrap(doc, target, 1, 3 * copies))


@pytest.mark.parametrize("copies", [1, 2, 3, 50])
def test_whole_copies_of_a_scattered_interface_cost_the_child_pieces_once(helpers, copies):
    root = parse_description(scattered_repeat_payload(helpers, copies), GATES).root
    target = root.steps[1].child

    # three pieces whatever the number of copies: one progression per declared gate of the child
    assert len(target.resolved_outputs) == 3
    assert target.out_count == 3 * copies
    assert sorted(target.out_offset(r) for r in range(3 * copies)) == sorted(
        6 * j + offset for j in range(copies) for offset in (0, 2, 5)
    )
    if copies > 1:
        # gates 2 and 5 of every copy interleave at the copy's pitch and are woven into one run
        assert target.out_runs == (Run(0, copies, 6, 8), Run(2, 2 * copies, 3, 8))
        assert parse_description(scattered_repeat_payload(helpers, copies), GATES, CompilationLimits(max_output_runs=3))
        with pytest.raises(CompileError, match="more than max_output_runs = 2 runs"):
            parse_description(scattered_repeat_payload(helpers, copies), GATES, CompilationLimits(max_output_runs=2))


def brute_force_repeated(runs: list[Run]) -> bool:
    seen: set[int] = set()
    for run in runs:
        for k in range(run.count):
            offset = run.element(k)
            if offset in seen:
                return True
            seen.add(offset)
    return False


def test_the_distinctness_sweep_agrees_with_enumeration_and_stays_near_linear():
    from veritor.compile.description import _repeated_output

    rng = random.Random(11)
    for _ in range(500):
        runs = [
            Run(rng.randrange(40), count := rng.randrange(1, 6), rng.randrange(1, 7) if count > 1 else 0, 8)
            for _ in range(rng.randrange(1, 7))
        ]
        found = _repeated_output(tuple(runs))
        assert (found is not None) == brute_force_repeated(runs), runs
        if found is not None:
            assert sum(run.index(found) is not None for run in runs) >= 2
    assert _repeated_output((Run(3, 2, 0, 8),)) == 3  # the same gate twice

    # runs laid out one after another, then the same with one late collision
    disjoint = tuple(Run(7 * k, 3, 2, 8) for k in range(50_000))
    start = time.perf_counter()
    assert _repeated_output(disjoint) is None
    assert _repeated_output((*disjoint, Run(7 * 49_999 + 4, 1, 0, 8))) == 7 * 49_999 + 4
    assert time.perf_counter() - start < 2.0


def test_interleaved_runs_that_tile_their_stride_are_woven():
    from veritor.core.description import _weave

    assert _weave((Run(0, 3, 2, 8), Run(1, 3, 2, 8))) == (Run(0, 6, 1, 8),)
    assert _weave((Run(10, 4, 6, 8), Run(12, 4, 6, 8), Run(14, 4, 6, 8))) == (Run(10, 12, 2, 8),)
    # too few runs for the stride, a mismatched pitch, a mismatched width: untouched
    for runs in (
        (Run(0, 3, 4, 8), Run(1, 3, 4, 8)),
        (Run(0, 3, 3, 8), Run(1, 3, 3, 8), Run(3, 3, 3, 8)),
        (Run(0, 3, 2, 8), Run(1, 3, 2, 16)),
        (Run(0, 1, 0, 8), Run(1, 1, 0, 8)),
    ):
        assert _weave(runs) == runs
    # a woven group followed by more runs
    assert _weave((Run(0, 2, 2, 8), Run(1, 2, 2, 8), Run(9, 2, 5, 8))) == (Run(0, 4, 1, 8), Run(9, 2, 5, 8))


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

    The layer holds ``length`` input gates and ``copies * length`` weight gates
    (each a repeat of the tracer's one-gate cells) and passes them down as
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

    @tracer.definition(input_count=0, key=("layer", copies, length), role="replay")
    def layer(_v):
        x = tracer.inputs(length)
        w = tracer.weights(copies * length)
        return tracer.repeat(copies, dot, x, w[0:length].by(length))

    return tracer.serialize(layer)


def admission(copies: int, length: int) -> Callable[[], float]:
    """Seconds to parse, index, tabulate the kinds, build the boundary and price one run."""

    description = wide_description(copies, length)
    policy = VerificationPolicy(Fraction(1, 2), 1)

    def timed() -> float:
        start = time.perf_counter()
        compiled = Compiler(GATES).compile(description, range(length))
        kinds = compiled.index.kinds()
        boundary = compiled.index.boundary()
        work = expected_work(compiled, policy, 2)
        elapsed = time.perf_counter() - start
        assert compiled.index.root.frame.definition.input_count == 0
        assert (compiled.index.input_count, compiled.index.weight_count) == (length, copies * length)
        assert boundary.count == length + copies
        assert {row.out_count for row in kinds} == {0, 1, copies}
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
    compiled = Compiler(GATES).compile(wide_description(4, 4), range(4))
    index = compiled.index
    layer = index.root.frame.definition
    dot = index.verification_unit(4 + 16).frame.definition  # after the input and weight cells

    assert index.verification_units(0).count == 4 + 16 + 4
    assert layer.out_runs == (Run(4 + 16 + 2 * 4 - 2, 4, 2 * 4 - 1, 8),)
    assert layer.input_runs == (Run(0, 4, 1, 8),) and layer.weight_runs == (Run(4, 16, 1, 8),)
    assert (layer.input_total, layer.weight_total) == (4, 16)
    assert layer.input_count == 0 and layer.reads == ()
    assert len(dot.reads) == dot.input_count == 8
    assert [row.input_count for row in index.kinds() if row.role == "replay"] == [0]
    assert [index.boundary().unrank(r) for r in range(4)] == [0, 1, 2, 3]
    assert [index.weights().rank(a) for a in range(4, 20)] == list(range(16))
    tape = compiled.circuit.evaluate(range(1, 5), [1] * 16)
    assert [tape[o] for o in compiled.circuit.outputs] == [10, 10, 10, 10]
