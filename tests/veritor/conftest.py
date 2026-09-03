"""The run-based interfaces checked against brute-force enumeration.

``Out`` of a copy, the per-kind table, the input and weight domains, the
boundary and the interiors are all derived from per-definition runs and
prefix sums; on a small circuit every one of them can be compared with the
definition they stand for: the declared outputs of every copy, resolved
address by address through the frame, and the source gates found by scanning
``C[i]``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from veritor.core import DescriptionCircuit, FlatCircuit, Index, IndexNode, iter_domain


def nodes_below(node: IndexNode) -> Iterator[IndexNode]:
    yield node
    for child in node.children():
        yield from nodes_below(child)


def declared_out(node: IndexNode, circuit: DescriptionCircuit) -> list[int]:
    """Brute force: the declared, unpinned output addresses of the copy that lie inside it."""

    frame = node.frame
    interval = frame.interval
    addresses = {frame.output_address(k) for k in range(frame.definition.output_count)}
    return sorted(a for a in addresses if a in interval and not circuit[a].is_source)


def assert_source_domain_matches(domain, expected: list[int], rank_of, n: int) -> None:
    """A lazy source domain (``In`` or ``W``) against the scanned addresses."""

    assert domain.count == len(domain) == len(expected)
    assert [domain.unrank(rank) for rank in range(domain.count)] == expected == list(domain)
    assert [domain.rank(address) for address in expected] == list(range(len(expected)))
    assert [rank_of(address) for address in expected] == list(range(len(expected)))
    assert [a for a in range(-1, n + 1) if domain.contains(a)] == expected
    for address in range(n):
        if address not in expected:
            with pytest.raises(KeyError):
                domain.rank(address)
            with pytest.raises(KeyError):
                rank_of(address)
    with pytest.raises(IndexError):
        domain.unrank(domain.count)
    with pytest.raises(IndexError):
        domain.unrank(-1)


def assert_interfaces_match_enumeration(index: Index, circuit: DescriptionCircuit) -> None:
    flat = FlatCircuit(
        [circuit[address] for address in range(circuit.n)], tuple(circuit.outputs), circuit.gate_set
    )
    rows = {row.kind: row for row in index.kinds()}
    expected_out: dict[IndexNode, list[int]] = {}
    for node in nodes_below(index.root):
        definition = node.frame.definition
        expected = expected_out[node] = declared_out(node, circuit)
        pinned = {a for a in node.interval if circuit[a].is_source}
        out = circuit.Out(node)
        assert len(out) == definition.out_count == rows[node.kind].out_count == len(expected)
        assert sorted(out) == expected
        assert not pinned & set(out)  # Out never holds a source gate ...
        assert set(out) | pinned >= set(flat.Out(node.interval))  # ... yet covers every outside read
        widths = sum(circuit[address].width for address in expected)
        assert rows[node.kind].out_bits == definition.out_bits == widths
        assert rows[node.kind].input_count == definition.input_count >= len(circuit.In(node))
        inputs_inside = [a for a in node.interval if circuit[a].is_input]
        weights_inside = [a for a in node.interval if circuit[a].is_weight]
        assert rows[node.kind].source_inputs == definition.input_total == len(inputs_inside)
        assert rows[node.kind].source_weights == definition.weight_total == len(weights_inside)
        base = node.frame.base
        for runs, inside in ((definition.input_runs, inputs_inside), (definition.weight_runs, weights_inside)):
            members = sorted(base + run.element(k) for run in runs for k in range(run.count))
            assert members == inside
        for rank, address in enumerate(out):
            assert definition.out_rank(address - base) == rank
        for offset in range(definition.size):
            found = definition.out_rank(offset) is not None
            assert found == (base + offset in expected)

    n = circuit.n
    expected_inputs = [a for a in range(n) if circuit[a].is_input]
    expected_weights = [a for a in range(n) if circuit[a].is_weight]
    assert expected_inputs == list(flat.inputs) == list(circuit.inputs)
    assert expected_weights == list(flat.weights) == list(circuit.weights)
    assert (circuit.input_count, circuit.weight_count) == (len(expected_inputs), len(expected_weights))
    assert (index.input_count, index.weight_count) == (len(expected_inputs), len(expected_weights))
    assert_source_domain_matches(index.inputs(), expected_inputs, circuit.input_rank, n)
    assert_source_domain_matches(index.weights(), expected_weights, circuit.weight_rank, n)
    assert index.inputs().identity_digest != index.weights().identity_digest

    units = list(index.replay_units)
    boundary = index.boundary()
    expected_boundary = sorted(set(expected_inputs).union(*(expected_out[unit] for unit in units)))
    assert boundary.count == len(boundary) == len(expected_boundary)
    members = [boundary.unrank(rank) for rank in range(boundary.count)]
    assert sorted(members) == expected_boundary
    assert members[: circuit.input_count] == expected_inputs
    assert [boundary.rank(address) for address in members] == list(range(boundary.count))
    assert [address for address in range(n) if boundary.contains(address)] == expected_boundary
    # every circuit output is committed: in the boundary, or a weight gate under κ_W
    assert all(output in boundary or circuit[output].is_weight for output in circuit.outputs)
    assert not any(boundary.contains(address) for address in expected_weights)
    for address in range(n):
        if address not in expected_boundary:
            with pytest.raises(KeyError):
                boundary.rank(address)

    for r, unit in enumerate(units):
        interior = index.interior(r)
        # the outputs of the verification units inside, less the unit's own (boundary) outputs
        unit_outputs = {a for node in index.verification_units(r) for a in expected_out[node]}
        assert set(expected_out[unit]) <= unit_outputs  # the refinement: Out(R) ⊆ ⋃ Out(V)
        expected_interior = [a for a in unit.interval if a in unit_outputs and a not in expected_out[unit]]
        assert not any(circuit[a].is_source for a in expected_interior)
        assert interior.count == len(interior) == len(expected_interior)
        assert interior.count == rows[unit.kind].interior_count
        assert list(iter_domain(interior)) == expected_interior == list(interior)
        assert [interior.unrank(rank) for rank in range(interior.count)] == expected_interior
        assert [interior.rank(a) for a in expected_interior] == list(range(len(expected_interior)))
        assert [a for a in range(-1, n + 1) if interior.contains(a)] == expected_interior
        for address in unit.interval:
            if address not in expected_interior:
                with pytest.raises(KeyError):
                    interior.rank(address)
        with pytest.raises(IndexError):
            interior.unrank(interior.count)
        with pytest.raises(IndexError):
            interior.unrank(-1)
    for node in nodes_below(index.root):
        if node.role == "verification":
            assert rows[node.kind].interior_count == 0
    assert rows[index.root.kind].interior_count == sum(index.interior(r).count for r in range(len(units)))
    # the owners partition the positions the relations touch: every address is exactly one of
    # input, weight, Out of its unit, interior of its unit, or an internal gate of a verification
    # unit (never committed: recomputed from the unit's opened inputs)
    committed = [set(expected_inputs), set(expected_weights), set(expected_boundary) - set(expected_inputs)]
    committed += [set(iter_domain(index.interior(r))) for r in range(len(units))]
    assert sum(len(part) for part in committed) == len(set().union(*committed))  # pairwise disjoint
    internal = set(range(n)) - set().union(*committed)
    assert not any(circuit[a].is_source for a in internal)
    for r, unit in enumerate(units):
        for node in index.verification_units(r):
            for address in node.interval:
                if address in internal:
                    assert address not in expected_out[node]  # an internal gate declares nothing
            # what a unit's relation touches is committed somewhere: its declared outputs and its reads
            touched = set(expected_out[node]) | set(circuit.In(node))
            assert touched <= set().union(*committed)


@pytest.fixture
def check_interfaces() -> Callable[[Index, DescriptionCircuit], None]:
    return assert_interfaces_match_enumeration
