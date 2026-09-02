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


def test_boundary_excludes_a_range_of_inputs_lazily(helpers):
    """``boundary(exclude=W)`` is ``∂ \\ W`` for any sub-range ``W`` of the inputs."""

    index, lazy, _flat = build(helpers, helpers.matmul_payload(4, 3, 2))
    inputs = index.input_count
    assert inputs == 20  # 8 activations, then the 12 weights at [8, 20)
    full = list(iter_domain(index.boundary()))

    for exclude in (range(8, 20), range(8), range(20), range(5, 11), range(3, 3), range(0)):
        boundary = index.boundary(exclude=exclude)
        expected = [address for address in full if address not in exclude]

        assert boundary.count == len(boundary) == len(expected) == len(full) - len(exclude)
        assert list(iter_domain(boundary)) == expected
        assert [boundary.rank(address) for address in expected] == list(range(len(expected)))
        assert [address for address in range(lazy.n) if boundary.contains(address)] == expected
        assert all(address not in boundary for address in exclude)
        assert all(output in boundary for output in lazy.outputs)
        for address in exclude:
            with pytest.raises(KeyError):
                boundary.rank(address)
        with pytest.raises(IndexError):
            boundary.unrank(boundary.count)
        with pytest.raises(IndexError):
            boundary.unrank(-1)
        same_set = len(exclude) == 0
        assert (boundary.identity_digest == index.boundary().identity_digest) == same_set

    for bad in (range(inputs + 1), range(-1, 3), range(0, 8, 2), range(9, 8), (8, 20), [8, 20]):
        with pytest.raises(InvalidArtifact, match="sub-range of the inputs range\\(0, 20\\)"):
            index.boundary(exclude=bad)  # type: ignore[arg-type]


def test_kinds_table_summarizes_each_definition_once(helpers):
    k, cols, rows = 4, 3, 2
    index, _, _ = build(helpers, helpers.matmul_payload(k, cols, rows))
    table = {row.role: row for row in index.kinds() if row.role}
    kinds = index.kinds()

    assert len(kinds) == 5  # root, row, dot, mul, add
    assert kinds[0].copies == 1 and kinds[0].min_depth == kinds[0].max_depth == 0
    dot = table["verification"].kind
    assert table["replay"] == KindSummary(
        kind=index.replay_units.unit(0).kind,
        role="replay",
        copies=rows,
        size=cols * (2 * k - 1),
        replay_cost=cols * (2 * k + k - 1),
        proof_cost=cols * (2 * k + k - 1),
        in_count=k + k * cols,
        out_count=cols,
        out_bits=cols * 8,
        min_depth=1,
        max_depth=1,
        children=((dot, cols),),
        verification_units=cols,
        verification_kinds=((dot, cols),),
    )
    assert table["verification"].copies == rows * cols
    assert table["verification"].in_count == 2 * k and table["verification"].out_count == 1
    assert table["verification"].verification_kinds == ((dot, 1),)
    assert kinds[0].verification_kinds == ((dot, rows * cols),)
    leaves = [row for row in kinds if row.size == 1]
    assert sorted(row.copies for row in leaves) == sorted(
        [rows * cols * k, rows * cols * (k - 1)]
    )
    assert all(row.min_depth == row.max_depth == 3 for row in leaves)
    assert all(row.children == () and row.verification_kinds == () for row in leaves)
    assert dict(table["verification"].children) == {
        leaves[0].kind: leaves[0].copies // (rows * cols),
        leaves[1].kind: leaves[1].copies // (rows * cols),
    }


def shared_kinds_payload(helpers) -> bytes:
    """Two replay kinds reaching one verification kind through different paths.

    Replay kind ``A`` (two copies) calls an unmarked ``middle`` twice, each
    holding two copies of ``V1``; replay kind ``B`` (one copy) calls ``V1``
    once and ``V2`` twice.  Exercises sharing, an unmarked layer between the
    cuts and a kind at two depths.
    """

    h = helpers
    doc = h.Document()
    v1 = doc.add(
        h.body(
            2,
            [h.gate("mul", h.rng(IN, 0, 2, 1)), h.gate("add", h.rng(LOC, 0), h.rng(IN, 1))],
            [h.rng(LOC, 1)],
            role="verification",
        )
    )
    v2 = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"))
    middle = doc.add(
        h.body(2, [h.repeat(2, v1, h.jrng(IN, 0, 2, 1))], [h.rng(LOC, 0, 2, 1)])
    )
    a = doc.add(
        h.body(
            2,
            [h.call(middle, h.rng(IN, 0, 2, 1)), h.call(middle, h.rng(LOC, 0, 2, 1))],
            [h.rng(LOC, 2, 2, 1)],
            role="replay",
        )
    )
    b = doc.add(
        h.body(
            2,
            [h.call(v1, h.rng(IN, 0, 2, 1)), h.repeat(2, v2, h.jrng(IN, 0), h.jrng(LOC, 0))],
            [h.rng(LOC, 1, 2, 1)],
            role="replay",
        )
    )
    root = doc.add(
        h.body(
            2,
            [h.repeat(2, a, h.jrng(IN, 0, 2, 1)), h.call(b, h.rng(LOC, 0, 2, 1))],
            [h.rng(LOC, 4, 2, 1)],
        )
    )
    return doc.serialize(root)


@pytest.mark.parametrize("which", ["matmul", "shared"])
def test_kinds_table_matches_enumeration_of_every_copy(helpers, which):
    payload = helpers.matmul_payload(4, 3, 2) if which == "matmul" else shared_kinds_payload(helpers)
    index, lazy, _flat = build(helpers, payload)
    nodes = [IndexNode(frame) for frame in helpers.frames(index.root.frame)]
    by_kind: dict[str, list[IndexNode]] = {}
    for node in nodes:
        by_kind.setdefault(node.kind, []).append(node)

    def below(node: IndexNode, role: str) -> list[IndexNode]:
        found = []
        for frame in helpers.frames(node.frame):
            inner = IndexNode(frame)
            if inner != node and inner.role == role:
                found.append(inner)
        return found

    table = index.kinds()
    assert [row.kind for row in table] == list(by_kind)  # first-visit order, each once
    for row in table:
        copies = by_kind[row.kind]
        assert row.copies == len(copies)
        assert row.min_depth == min(c.depth for c in copies)
        assert row.max_depth == max(c.depth for c in copies)
        for node in copies:
            assert row.role == node.role and row.size == node.size
            assert row.out_count == len(lazy.Out(node))
            assert row.out_bits == sum(lazy[a].width for a in lazy.Out(node))
            assert row.in_count == len(lazy.In(node))
            assert row.replay_cost == lazy.Cost(node, "replay")
            assert row.proof_cost == lazy.Cost(node, "proof")
            children: dict[str, int] = {}
            for child in node.children():
                children[child.kind] = children.get(child.kind, 0) + 1
            assert dict(row.children) == children
            units = [node] if node.role == "verification" else below(node, "verification")
            assert row.verification_units == len(units)
            kinds: dict[str, int] = {}
            for unit in units:
                kinds[unit.kind] = kinds.get(unit.kind, 0) + 1
            assert dict(row.verification_kinds) == kinds
    if which == "shared":
        v1 = next(row for row in table if row.role == "verification" and row.size == 2)
        assert v1.copies == 2 * 4 + 1 and (v1.min_depth, v1.max_depth) == (2, 3)
        replay = {row.copies: row for row in table if row.role == "replay"}
        assert replay[2].verification_kinds == ((v1.kind, 4),)
        assert dict(replay[1].verification_kinds)[v1.kind] == 1 and replay[1].verification_units == 3


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
