from __future__ import annotations

import pytest

from veritor.compile.description import parse_description
from veritor.core import (
    CompilationLimits,
    DescriptionCircuit,
    Index,
    IndexNode,
    InvalidArtifact,
    KindSummary,
    iter_domain,
    make_word_gate_set,
)

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def build(helpers, payload: bytes, limits: CompilationLimits | None = None):
    root = parse_description(payload, GATES).root
    return Index(root, limits), DescriptionCircuit(root, GATES), helpers.flatten(root, GATES)


def reference_boundary(flat, units: list[range]) -> list[int]:
    """O(n) derivation: inputs, outputs, and every address read across units."""

    owner = {address: r for r, unit in enumerate(units) for address in unit}
    boundary = set(flat.inputs) | set(flat.outputs)
    for address in range(flat.n):
        for arg in flat[address].args:
            if owner.get(arg) != owner.get(address):
                boundary.add(arg)
    return sorted(boundary)


def test_units_are_the_marked_copies_with_lazy_lookup(helpers):
    k, cols, rows = 4, 3, 2
    index, lazy, _flat = build(helpers, helpers.matmul_payload(k, cols, rows))
    n_in = rows * k + k * cols
    dot_size = k + (k - 1)

    assert index.root.interval == range(n_in, lazy.n)
    assert index.root.role is None and index.root.depth == 0
    assert index.replay_units.count == rows
    units = list(index.replay_units)
    assert [u.interval for u in units] == [
        range(n_in + r * cols * dot_size, n_in + (r + 1) * cols * dot_size) for r in range(rows)
    ]
    assert all(u.role == "replay" and u.depth == 1 for u in units)
    assert units[0].kind == units[1].kind and units[0] != units[1]
    assert units[0] == index.replay_units.unit(0) and hash(units[0]) == hash(index.replay_units.unit(0))
    for r, unit in enumerate(units):
        assert unit.replay_unit == r
        assert all(index.replay_units.owner(a) == r for a in unit.interval)
        children = list(unit.children())
        vunits = index.verification_units(r)
        assert vunits.count == cols and vunits.first == r * cols
        assert list(vunits) == children
        for c, node in enumerate(children):
            assert node.role == "verification" and node.depth == 2
            assert node.interval == range(unit.interval.start + c * dot_size, unit.interval.start + (c + 1) * dot_size)
            assert node.replay_unit == r
            assert all(vunits.owner(a) == c for a in node.interval)
            assert index.verification_unit(r * cols + c) == node
            assert list(node.children()) == [IndexNode(node.frame.child(0, j)) for j in range(k)] + [
                IndexNode(node.frame.child(s, j))
                for s in range(1, len(node.frame.definition.steps))
                for j in range(node.frame.definition.steps[s].count)
            ]
    assert index.verification_unit_count == rows * cols
    assert index.input_count == n_in and index.n == lazy.n
    with pytest.raises(IndexError):
        index.replay_units.unit(rows)
    with pytest.raises(IndexError):
        index.verification_unit(rows * cols)
    with pytest.raises(KeyError):
        index.replay_units.owner(0)  # inputs belong to no unit
    with pytest.raises(KeyError):
        index.verification_units(0).owner(units[1].interval.start)


def test_boundary_and_interiors_match_the_reference_derivation(helpers):
    for k, cols, rows in ((2, 1, 1), (4, 3, 2), (8, 2, 3)):
        index, lazy, flat = build(helpers, helpers.matmul_payload(k, cols, rows))
        units = [unit.interval for unit in index.replay_units]
        expected = reference_boundary(flat, units)
        boundary = index.boundary()

        assert boundary.count == len(expected)
        assert list(iter_domain(boundary)) == expected
        assert [boundary.rank(a) for a in expected] == list(range(len(expected)))
        assert [a for a in range(lazy.n) if boundary.contains(a)] == expected
        assert not boundary.contains(lazy.n) and not boundary.contains(-1)
        assert all(o in boundary for o in lazy.outputs)
        with pytest.raises(KeyError):
            boundary.rank(next(a for a in range(lazy.n) if a not in expected))
        with pytest.raises(IndexError):
            boundary.unrank(boundary.count)
        for r, unit in enumerate(units):
            interior = index.interior(r)
            members = [a for a in unit if a not in expected]
            assert list(iter_domain(interior)) == members
            assert [interior.rank(a) for a in members] == list(range(len(members)))
            assert lazy.Out(index.replay_units.unit(r)) == tuple(a for a in unit if a in expected)
        assert boundary.identity_digest != index.interior(0).identity_digest
    assert len(boundary) == boundary.count


def test_kinds_table_summarizes_each_definition_once(helpers):
    k, cols, rows = 4, 3, 2
    index, _, _ = build(helpers, helpers.matmul_payload(k, cols, rows))
    table = {row.role: row for row in index.kinds() if row.role}
    kinds = index.kinds()

    assert len(kinds) == 5  # root, row, dot, mul, add
    assert kinds[0].copies == 1 and kinds[0].min_depth == kinds[0].max_depth == 0
    assert table["replay"] == KindSummary(
        kind=index.replay_units.unit(0).kind,
        role="replay",
        copies=rows,
        size=cols * (2 * k - 1),
        replay_cost=cols * (2 * k + k - 1),
        proof_cost=cols * (2 * k + k - 1),
        in_count=k + k * cols,
        out_count=cols,
        min_depth=1,
        max_depth=1,
    )
    assert table["verification"].copies == rows * cols
    assert table["verification"].in_count == 2 * k and table["verification"].out_count == 1
    leaves = [row for row in kinds if row.size == 1]
    assert sorted(row.copies for row in leaves) == sorted(
        [rows * cols * k, rows * cols * (k - 1)]
    )
    assert all(row.min_depth == row.max_depth == 3 for row in leaves)


# -- validity of marks --------------------------------------------------------


def two_gates(helpers):
    h = helpers
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    return doc, add


def test_gate_step_above_the_replay_cut_is_named(helpers):
    h = helpers
    doc, add = two_gates(h)
    unit = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    replay = doc.add(h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay"))
    root = doc.add(
        h.body(
            2,
            [h.call(replay, h.rng(IN, 0, 2, 1)), h.gate("mul", h.rng(LOC, 0, 2, 0))],
            [h.rng(LOC, 1)],
        )
    )
    with pytest.raises(InvalidArtifact, match=f"gate step 1 of definition {root[:12]} is not inside a replay unit"):
        build(h, doc.serialize(root))
    # through an unmarked intermediate definition the offending step is still found
    doc, add = two_gates(h)
    middle = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    root = doc.add(h.body(2, [h.call(middle, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(InvalidArtifact, match=f"gate step 0 of definition {add[:12]} is not inside a replay unit"):
        build(h, doc.serialize(root))


def test_gate_step_inside_a_replay_unit_must_be_in_a_verification_unit(helpers):
    h = helpers
    doc, add = two_gates(h)
    replay = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay"))
    root = doc.add(h.body(2, [h.call(replay, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(InvalidArtifact, match=f"gate step 0 of definition {add[:12]} is not inside a verification unit"):
        build(h, doc.serialize(root))


def test_marks_may_not_nest_and_verification_needs_a_replay_unit(helpers):
    h = helpers
    doc, add = two_gates(h)
    inner = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    outer = doc.add(h.body(2, [h.call(inner, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    with pytest.raises(InvalidArtifact, match="marked verification and contains a verification mark"):
        build(h, doc.serialize(outer))

    doc, add = two_gates(h)
    unit = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    inner = doc.add(h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay"))
    outer = doc.add(h.body(2, [h.call(inner, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay"))
    with pytest.raises(InvalidArtifact, match="marked replay and contains a replay mark"):
        build(h, doc.serialize(outer))

    doc, add = two_gates(h)
    unit = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    root = doc.add(h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(InvalidArtifact, match="calls verification-marked definition .* outside any replay unit"):
        build(h, doc.serialize(root))
    with pytest.raises(InvalidArtifact, match="root is marked verification"):
        build(h, doc.serialize(unit))

    doc, add = two_gates(h)
    unit = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    replay = doc.add(h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay"))
    wrapper = doc.add(h.body(2, [h.call(replay, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    with pytest.raises(InvalidArtifact, match="marked verification and contains a replay mark"):
        build(h, doc.serialize(wrapper))


def test_units_must_have_gates_and_verification_units_respect_the_proof_cap(helpers):
    h = helpers
    doc = h.Document()
    empty = doc.add(h.body(1, [], [h.rng(IN, 0)], role="verification"))
    with pytest.raises(InvalidArtifact, match="marked verification but has no gates"):
        build(h, doc.serialize(empty))

    payload = h.matmul_payload(4, 2, 1)
    build(h, payload, CompilationLimits(max_verification_unit_proof_cost=11))
    with pytest.raises(InvalidArtifact, match="verification unit of proof cost 11; the limit is 10"):
        build(h, payload, CompilationLimits(max_verification_unit_proof_cost=10))


def test_root_may_itself_be_the_only_replay_unit(helpers):
    h = helpers
    doc, add = two_gates(h)
    unit = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    root = doc.add(h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay"))
    index, _lazy, _flat = build(h, doc.serialize(root))

    assert index.replay_units.count == 1
    assert index.replay_units.unit(0) == index.root
    assert index.verification_units(0).count == 1
    assert list(iter_domain(index.boundary())) == [0, 1, 2]
    assert index.interior(0).count == 0
