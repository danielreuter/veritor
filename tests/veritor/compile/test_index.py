from __future__ import annotations

import pytest

from veritor.compile.description import parse_description
from veritor.constructors import compile_demo_g
from veritor.core import (
    CompilationLimits,
    DescriptionCircuit,
    Index,
    IndexNode,
    InvalidArtifact,
    KindSummary,
    iter_domain,
    iter_members,
    make_word_gate_set,
)
from veritor.core.description import Frame

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def build(helpers, payload: bytes, limits: CompilationLimits | None = None):
    root = parse_description(payload, GATES).root
    return (
        Index(root, limits),
        DescriptionCircuit(root, GATES),
        helpers.flatten(root, GATES),
    )


def reference_boundary(flat, units: list[range]) -> list[int]:
    """O(n) derivation: inputs, outputs, and every unpinned address read across units.

    A weight read across units is under the weight commitment, not the boundary.
    """

    owner = {address: r for r, unit in enumerate(units) for address in unit}
    boundary = set(flat.inputs) | set(flat.outputs)
    for address in range(flat.n):
        for arg in flat[address].args:
            if owner.get(arg) != owner.get(address) and not flat[arg].is_weight:
                boundary.add(arg)
    return sorted(boundary)


def test_units_are_the_marked_copies_with_lazy_lookup(helpers):
    k, cols, rows = 4, 3, 2
    index, lazy, _flat = build(helpers, helpers.matmul_payload(k, cols, rows))
    layout = helpers.matmul_layout(k, cols, rows)
    dot_size = k + (k - 1)

    assert index.root.interval == range(lazy.n)
    assert index.root.role is None and index.root.depth == 0
    assert (
        index.replay_units.count == rows + 2
    )  # the activations, the weights, then the rows
    units = list(index.replay_units)
    activations, weights, rows_ = units[0], units[1], units[2:]
    assert (
        activations.interval == layout["inputs"]
        and weights.interval == layout["weights"]
    )
    assert [u.interval for u in rows_] == layout["rows"]
    assert all(u.role == "replay" and u.depth == 1 for u in units)
    assert (
        rows_[0].kind == rows_[1].kind != weights.kind != activations.kind
        and rows_[0] != rows_[1]
    )
    assert units[0] == index.replay_units.unit(0) and hash(units[0]) == hash(
        index.replay_units.unit(0)
    )
    # the source cells are the verification units of the two source units
    for r, (unit, source) in enumerate(((activations, "input"), (weights, "weight"))):
        vunits = index.verification_units(r)
        assert unit.replay_unit == r and vunits.count == len(unit.interval)
        assert vunits.first == unit.interval.start
        assert [node.interval for node in vunits] == [
            range(a, a + 1) for a in unit.interval
        ]
        assert all(lazy[node.interval.start].source == source for node in vunits)
        assert all(
            node.role == "verification" and node.depth == 2 and node.size == 1
            for node in vunits
        )
    for r, unit in enumerate(rows_, start=2):
        assert unit.replay_unit == r
        assert all(index.replay_units.owner(a) == r for a in unit.interval)
        children = list(unit.children())
        vunits = index.verification_units(r)
        assert (
            vunits.count == cols
            and vunits.first == rows * k + k * cols + (r - 2) * cols
        )
        assert list(vunits) == children
        for c, node in enumerate(children):
            assert node.role == "verification" and node.depth == 2
            start = unit.interval.start + c * dot_size
            assert node.interval == range(start, start + dot_size)
            assert node.replay_unit == r
            assert all(vunits.owner(a) == c for a in node.interval)
            assert index.verification_unit(vunits.first + c) == node
            assert list(node.children()) == [
                IndexNode(node.frame.child(0, j)) for j in range(k)
            ] + [
                IndexNode(node.frame.child(s, j))
                for s in range(1, len(node.frame.definition.steps))
                for j in range(node.frame.definition.steps[s].count)
            ]
    assert index.verification_unit_count == rows * k + k * cols + rows * cols
    assert (index.input_count, index.weight_count) == (rows * k, k * cols)
    assert index.n == lazy.n == layout["n"]
    with pytest.raises(IndexError):
        index.replay_units.unit(rows + 2)
    with pytest.raises(IndexError):
        index.verification_unit(index.verification_unit_count)
    with pytest.raises(KeyError):
        index.replay_units.owner(lazy.n)
    with pytest.raises(KeyError):
        index.verification_units(0).owner(rows_[0].interval.start)


def test_boundary_and_interiors_match_the_reference_derivation(helpers):
    for k, cols, rows in ((2, 1, 1), (4, 3, 2), (8, 2, 3)):
        index, lazy, flat = build(helpers, helpers.matmul_payload(k, cols, rows))
        units = [unit.interval for unit in index.replay_units]
        expected = reference_boundary(flat, units)
        boundary = index.boundary()

        assert expected[: rows * k] == list(
            flat.inputs
        )  # the input gates come first ...
        assert boundary.count == len(expected)
        assert (
            list(iter_domain(boundary)) == expected
        )  # ... and Out(R_r) in unit order is address order here
        assert [boundary.rank(a) for a in expected] == list(range(len(expected)))
        assert [a for a in range(lazy.n) if boundary.contains(a)] == expected
        assert not boundary.contains(lazy.n) and not boundary.contains(-1)
        assert all(o in boundary for o in lazy.outputs)
        assert not any(w in boundary for w in flat.weights)
        with pytest.raises(KeyError):
            boundary.rank(next(a for a in range(lazy.n) if a not in expected))
        with pytest.raises(IndexError):
            boundary.unrank(boundary.count)
        for r, unit in enumerate(units):
            interior = index.interior(r)
            # the interior: the declared outputs of the verification units inside, less the boundary
            unit_outputs = {
                a for node in index.verification_units(r) for a in lazy.Out(node)
            }
            members = [a for a in unit if a in unit_outputs and a not in expected]
            assert list(iter_domain(interior)) == members
            assert [interior.rank(a) for a in members] == list(range(len(members)))
            assert [a for a in unit if interior.contains(a)] == members
            assert lazy.Out(index.replay_units.unit(r)) == tuple(
                a for a in unit if a in expected and not flat[a].is_input
            )
            assert (
                set(lazy.Out(index.replay_units.unit(r))) <= unit_outputs
            )  # Out(R) ⊆ ⋃ Out(V)
        # the activations and the weights units are all source gates: nothing to replay; a dot
        # declares only its sum, which is the row's output: the row commits no interior position
        assert index.interior(0).count == 0 and index.interior(1).count == 0
        assert index.interior(2).count == 0
        assert boundary.identity_digest != index.interior(0).identity_digest
    assert len(boundary) == boundary.count


def test_every_domain_iterates_in_rank_order(helpers):
    """``iter_members`` (the commit paths' linear walk) must agree with the ``unrank`` enumeration."""

    indices = [
        build(helpers, helpers.matmul_payload(4, 3, 2))[0],
        compile_demo_g().compiled.index,
    ]
    for index in indices:
        domains = [index.boundary(), index.weights(), index.inputs()]
        domains += [index.interior(r) for r in range(index.replay_units.count)]
        for domain in domains:
            walked = list(iter_members(domain))
            assert walked == list(iter_domain(domain))
            assert walked == list(domain)
            assert [domain.rank(a) for a in walked] == list(range(domain.count))


def test_input_and_weight_domains_are_lazy_and_ranked_in_address_order(helpers):
    k, cols, rows = 4, 3, 2
    index, lazy, flat = build(helpers, helpers.matmul_payload(k, cols, rows))
    layout = helpers.matmul_layout(k, cols, rows)
    inputs, weights = index.inputs(), index.weights()

    assert (
        (inputs.count, weights.count)
        == (rows * k, k * cols)
        == (index.input_count, index.weight_count)
    )
    assert (
        list(iter_domain(inputs))
        == list(layout["inputs"])
        == list(flat.inputs)
        == list(lazy.inputs)
    )
    assert (
        list(iter_domain(weights))
        == list(layout["weights"])
        == list(flat.weights)
        == list(lazy.weights)
    )
    for rank, address in enumerate(layout["inputs"]):
        assert inputs.rank(address) == lazy.input_rank(
            address
        ) == rank and inputs.contains(address)
        assert not weights.contains(address)
    for rank, address in enumerate(layout["weights"]):
        assert weights.rank(address) == lazy.weight_rank(
            address
        ) == rank and weights.contains(address)
        assert not inputs.contains(address)
    dot_output = layout["dots"][0].stop - 1
    assert not inputs.contains(dot_output) and not weights.contains(dot_output)
    for bad in (dot_output, -1, lazy.n, "0"):
        with pytest.raises(KeyError):
            inputs.rank(bad)  # type: ignore[arg-type]
        with pytest.raises(KeyError):
            weights.rank(bad)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        inputs.unrank(inputs.count)
    with pytest.raises(IndexError):
        weights.unrank(-1)
    with pytest.raises(TypeError):
        weights.unrank("0")  # type: ignore[arg-type]
    assert inputs.identity_digest != weights.identity_digest
    assert inputs.identity_digest == index.inputs().identity_digest
    other = build(helpers, helpers.matmul_payload(k, cols, rows + 1))[0]
    assert other.inputs().identity_digest != inputs.identity_digest
    # the boundary is the inputs by rank, then the units' Out
    boundary = index.boundary()
    assert [boundary.unrank(r) for r in range(inputs.count)] == list(layout["inputs"])
    assert boundary.count == inputs.count + rows * cols


def test_kinds_table_summarizes_each_definition_once(helpers):
    k, cols, rows = 4, 3, 2
    index, _, _ = build(helpers, helpers.matmul_payload(k, cols, rows))
    kinds = index.kinds()
    by_kind = {row.kind: row for row in kinds}
    activations_unit, weights_unit = (
        index.replay_units.unit(0),
        index.replay_units.unit(1),
    )
    row_unit = index.replay_units.unit(2)
    activations_row, weights_row = (
        by_kind[activations_unit.kind],
        by_kind[weights_unit.kind],
    )
    row_row = by_kind[row_unit.kind]
    dot = index.verification_units(2).unit(0).kind
    in_cell, weight_cell = (
        index.verification_units(0).unit(0).kind,
        index.verification_units(1).unit(0).kind,
    )

    assert (
        len(kinds) == 9
    )  # root, activations, in cell, weights, weight cell, row, dot, mul, add
    assert kinds[0].copies == 1 and kinds[0].min_depth == kinds[0].max_depth == 0
    assert (kinds[0].source_inputs, kinds[0].source_weights) == (rows * k, k * cols)
    assert (
        row_row
        == KindSummary(
            kind=row_unit.kind,
            role="replay",
            copies=rows,
            size=cols * (2 * k - 1),
            replay_cost=cols * (2 * k + k - 1),
            proof_cost=cols * (2 * k + k - 1),
            input_count=k + k * cols,
            out_count=cols,
            out_bits=cols * 8,
            interior_count=0,  # every dot's output is a row output: committed at the boundary
            reach_bits=cols
            * 8,  # a row's dots are circuit outputs and nothing reads them
            ancestor_bits=rows * cols * 8,  # enclosed by the root alone
            source_inputs=0,
            source_weights=0,
            min_depth=1,
            max_depth=1,
            children=((dot, cols),),
            verification_units=cols,
            verification_kinds=((dot, cols),),
            closed=True,
        )
    )
    assert activations_row == KindSummary(
        kind=activations_unit.kind,
        role="replay",
        copies=1,
        size=rows * k,
        replay_cost=0,
        proof_cost=rows * k,
        input_count=0,
        out_count=0,  # every declared output is an input gate: pinned, not in Out
        out_bits=0,
        interior_count=0,
        reach_bits=rows
        * cols
        * 8,  # every row reads the activations: structurally they reach everything
        ancestor_bits=rows * cols * 8,
        source_inputs=rows * k,
        source_weights=0,
        min_depth=1,
        max_depth=1,
        children=((in_cell, rows * k),),
        verification_units=rows * k,
        verification_kinds=((in_cell, rows * k),),
        closed=True,
    )
    assert weights_row == KindSummary(
        kind=weights_unit.kind,
        role="replay",
        copies=1,
        size=k * cols,
        replay_cost=0,
        proof_cost=k * cols,
        input_count=0,
        out_count=0,  # every declared output is a weight gate: pinned, not in Out
        out_bits=0,
        interior_count=0,
        reach_bits=rows * cols * 8,
        ancestor_bits=rows * cols * 8,
        source_inputs=0,
        source_weights=k * cols,
        min_depth=1,
        max_depth=1,
        children=((weight_cell, k * cols),),
        verification_units=k * cols,
        verification_kinds=((weight_cell, k * cols),),
        closed=True,
    )
    assert by_kind[dot].copies == rows * cols
    assert by_kind[dot].input_count == 2 * k and by_kind[dot].out_count == 1
    assert (by_kind[dot].source_inputs, by_kind[dot].source_weights) == (0, 0)
    assert by_kind[dot].verification_kinds == ((dot, 1),)
    assert kinds[0].verification_kinds == (
        (in_cell, rows * k),
        (weight_cell, k * cols),
        (dot, rows * cols),
    )
    cells = {by_kind[in_cell], by_kind[weight_cell]}
    assert {
        (row.copies, row.source_inputs, row.source_weights, row.out_count)
        for row in cells
    } == {
        (rows * k, 1, 0, 0),
        (k * cols, 0, 1, 0),
    }
    leaves = [row for row in kinds if row.size == 1 and row.role is None]
    # the products read the dot's ports (retained through the row from the source units);
    # the sums read products and partial sums
    assert {row.copies: row.closed for row in leaves} == {
        rows * cols * k: True,
        rows * cols * (k - 1): False,
    }
    assert by_kind[dot].closed
    assert sorted(row.copies for row in leaves) == sorted(
        [rows * cols * k, rows * cols * (k - 1)]
    )
    assert all(row.min_depth == row.max_depth == 3 for row in leaves)
    assert all(row.children == () and row.verification_kinds == () for row in leaves)
    assert dict(by_kind[dot].children) == {
        leaves[0].kind: leaves[0].copies // (rows * cols),
        leaves[1].kind: leaves[1].copies // (rows * cols),
    }


@pytest.mark.parametrize("which", ["matmul", "shared"])
def test_kinds_table_matches_enumeration_of_every_copy(helpers, which):
    payload = (
        helpers.matmul_payload(4, 3, 2)
        if which == "matmul"
        else helpers.shared_kinds_payload()
    )
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
            # the declared inputs bound what the copy actually reads
            assert (
                row.input_count
                == node.frame.definition.input_count
                >= len(lazy.In(node))
            )
            assert row.replay_cost == lazy.Cost(node, "replay")
            assert row.proof_cost == lazy.Cost(node, "proof")
            children: dict[str, int] = {}
            for child in node.children():
                children[child.kind] = children.get(child.kind, 0) + 1
            assert dict(row.children) == children
            units = (
                [node] if node.role == "verification" else below(node, "verification")
            )
            assert row.verification_units == len(units)
            kinds: dict[str, int] = {}
            for unit in units:
                kinds[unit.kind] = kinds.get(unit.kind, 0) + 1
            assert dict(row.verification_kinds) == kinds
    if which == "shared":
        v1 = next(row for row in table if row.role == "verification" and row.size == 2)
        assert v1.copies == 2 * 4 + 1 and (v1.min_depth, v1.max_depth) == (
            3,
            4,
        )  # one level under the wrap root
        replay = {row.verification_units: row for row in table if row.role == "replay"}
        assert replay[4].copies == 2 and replay[4].verification_kinds == ((v1.kind, 4),)
        assert (
            replay[3].copies == 1 and dict(replay[3].verification_kinds)[v1.kind] == 1
        )
        assert replay[2].source_inputs == 2  # the input block under the wrap root


# -- validity of marks --------------------------------------------------------


def two_gates(helpers):
    h = helpers
    doc = h.Document()
    add = doc.add(h.body(2, [h.gate("add", h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    return doc, add


def build_wrapped(h, doc, target: str, input_count: int = 2, output_count: int = 1):
    return build(h, doc.serialize(h.wrap(doc, target, input_count, output_count)))


def test_gate_step_above_the_replay_cut_is_named(helpers):
    h = helpers
    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    replay = doc.add(
        h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    target = doc.add(
        h.body(
            2,
            [h.call(replay, h.rng(IN, 0, 2, 1)), h.gate("mul", h.rng(LOC, 0, 2, 0))],
            [h.rng(LOC, 1)],
        )
    )
    with pytest.raises(
        InvalidArtifact,
        match=f"gate step 1 of definition {target[:12]} is not inside a replay unit",
    ):
        build_wrapped(h, doc, target)
    # through an unmarked intermediate definition the offending step is still found
    doc, add = two_gates(h)
    middle = doc.add(h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    target = doc.add(h.body(2, [h.call(middle, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(
        InvalidArtifact,
        match=f"gate step 0 of definition {add[:12]} is not inside a replay unit",
    ):
        build_wrapped(h, doc, target)


def test_stray_source_gates_are_caught_by_the_tiling_check(helpers):
    """An `in`/`weight` gate is a gate: it must lie in a replay unit and a verification unit."""

    h = helpers
    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    replay = doc.add(
        h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    # an input gate at the root, beside the replay unit it feeds
    root = doc.add(
        h.body(
            0,
            [h.gate("in"), h.gate("weight"), h.call(replay, h.rng(LOC, 0, 2, 1))],
            [h.rng(LOC, 2)],
        )
    )
    with pytest.raises(
        InvalidArtifact,
        match=f"gate step 0 of definition {root[:12]} is not inside a replay unit",
    ):
        build(h, doc.serialize(root))
    # a weight gate directly inside a replay unit, outside its verification units
    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    replay = doc.add(
        h.body(
            0,
            [
                h.call(h.source_cell(doc, "in")),
                h.gate("weight"),
                h.call(unit, h.rng(LOC, 0, 2, 1)),
            ],
            [h.rng(LOC, 2)],
            role="replay",
        )
    )
    with pytest.raises(
        InvalidArtifact,
        match=f"gate step 1 of definition {replay[:12]} is not inside a verification unit",
    ):
        build(h, doc.serialize(replay))
    # inside a verification cell both are fine, and the source gates are units like any other
    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    replay = doc.add(
        h.body(
            0,
            [
                h.call(h.source_cell(doc, "in")),
                h.call(h.source_cell(doc, "weight")),
                h.call(unit, h.rng(LOC, 0, 2, 1)),
            ],
            [h.rng(LOC, 2)],
            role="replay",
        )
    )
    index, lazy, _flat = build(h, doc.serialize(replay))
    assert (
        index.verification_units(0).count == 3
        and lazy[0].is_input
        and lazy[1].is_weight
    )
    assert list(iter_domain(index.boundary())) == [0, 2] and list(
        iter_domain(index.weights())
    ) == [1]
    assert index.interior(0).count == 0


def test_gate_step_inside_a_replay_unit_must_be_in_a_verification_unit(helpers):
    h = helpers
    doc, add = two_gates(h)
    replay = doc.add(
        h.body(2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    target = doc.add(h.body(2, [h.call(replay, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(
        InvalidArtifact,
        match=f"gate step 0 of definition {add[:12]} is not inside a verification unit",
    ):
        build_wrapped(h, doc, target)


def test_marks_may_not_nest_and_verification_needs_a_replay_unit(helpers):
    h = helpers
    doc, add = two_gates(h)
    inner = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    outer = doc.add(
        h.body(
            2, [h.call(inner, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    with pytest.raises(
        InvalidArtifact, match="marked verification and contains a verification mark"
    ):
        build_wrapped(h, doc, outer)

    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    inner = doc.add(
        h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    outer = doc.add(
        h.body(2, [h.call(inner, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    with pytest.raises(
        InvalidArtifact, match="marked replay and contains a replay mark"
    ):
        build_wrapped(h, doc, outer)

    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    target = doc.add(h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)]))
    with pytest.raises(
        InvalidArtifact,
        match="calls verification-marked definition .* outside any replay unit",
    ):
        build_wrapped(h, doc, target)
    doc = h.Document()
    bare = h.source_cell(doc, "in")  # a verification-marked root of one `in` gate
    with pytest.raises(InvalidArtifact, match="root is marked verification"):
        build(h, doc.serialize(bare))

    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    replay = doc.add(
        h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    wrapper = doc.add(
        h.body(
            2,
            [h.call(replay, h.rng(IN, 0, 2, 1))],
            [h.rng(LOC, 0)],
            role="verification",
        )
    )
    with pytest.raises(
        InvalidArtifact, match="marked verification and contains a replay mark"
    ):
        build_wrapped(h, doc, wrapper)


def test_units_must_have_gates_and_verification_units_respect_the_proof_cap(helpers):
    h = helpers
    doc = h.Document()
    empty = doc.add(h.body(1, [], [h.rng(IN, 0)], role="verification"))
    with pytest.raises(InvalidArtifact, match="marked verification but has no gates"):
        build_wrapped(h, doc, empty, 1, 1)

    payload = h.matmul_payload(4, 2, 1)
    build(h, payload, CompilationLimits(max_verification_unit_proof_cost=11))
    with pytest.raises(
        InvalidArtifact, match="verification unit of proof cost 11; the limit is 10"
    ):
        build(h, payload, CompilationLimits(max_verification_unit_proof_cost=10))


def test_root_may_itself_be_the_only_replay_unit(helpers):
    h = helpers
    doc, add = two_gates(h)
    in_cell = h.source_cell(doc, "in")
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    root = doc.add(
        h.body(
            0,
            [h.repeat(2, in_cell), h.call(unit, h.rng(LOC, 0, 2, 1))],
            [h.rng(LOC, 2)],
            role="replay",
        )
    )
    index, _lazy, _flat = build(h, doc.serialize(root))

    assert index.replay_units.count == 1
    assert index.replay_units.unit(0) == index.root
    assert index.verification_units(0).count == 3
    assert list(iter_domain(index.boundary())) == [0, 1, 2]
    assert list(iter_domain(index.inputs())) == [0, 1] and index.weight_count == 0
    assert index.interior(0).count == 0


def test_the_root_has_no_ports(helpers):
    h = helpers
    doc, add = two_gates(h)
    unit = doc.add(
        h.body(
            2, [h.call(add, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="verification"
        )
    )
    ported = doc.add(
        h.body(2, [h.call(unit, h.rng(IN, 0, 2, 1))], [h.rng(LOC, 0)], role="replay")
    )
    with pytest.raises(
        InvalidArtifact, match="the root has no ports; inputs are `in` gates"
    ):
        build(h, doc.serialize(ported))
    parsed = parse_description(doc.serialize(h.wrap(doc, ported, 2, 1)), GATES)
    with pytest.raises(InvalidArtifact, match="the root has no ports"):
        Frame.root(next(d for d in parsed.definitions if d.digest == ported))
    assert Frame.root(parsed.root).interval == range(3)
