"""The run-based interfaces checked against brute-force enumeration.

``Out`` of a copy, the per-kind table, the boundary and the interiors are all
derived from per-definition runs; on a small circuit every one of them can be
compared with the definition they stand for: the declared outputs of every
copy, resolved address by address through the frame.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from veritor.core import DescriptionCircuit, FlatCircuit, Index, IndexNode, iter_domain


def nodes_below(node: IndexNode) -> Iterator[IndexNode]:
    yield node
    for child in node.children():
        yield from nodes_below(child)


def declared_out(node: IndexNode) -> list[int]:
    """Brute force: the declared output addresses of the copy that lie inside it."""

    frame = node.frame
    interval = frame.interval
    addresses = {frame.output_address(k) for k in range(frame.definition.output_count)}
    return sorted(address for address in addresses if address in interval)


def assert_interfaces_match_enumeration(index: Index, circuit: DescriptionCircuit) -> None:
    flat = FlatCircuit(
        [circuit[address] for address in range(circuit.n)], tuple(circuit.outputs), circuit.gate_set
    )
    rows = {row.kind: row for row in index.kinds()}
    expected_out: dict[IndexNode, list[int]] = {}
    for node in nodes_below(index.root):
        definition = node.frame.definition
        expected = expected_out[node] = declared_out(node)
        out = circuit.Out(node)
        assert len(out) == definition.out_count == rows[node.kind].out_count == len(expected)
        assert sorted(out) == expected
        assert set(out) >= set(flat.Out(node.interval))  # every address read from outside
        widths = sum(circuit[address].width for address in expected)
        assert rows[node.kind].out_bits == definition.out_bits == widths
        assert rows[node.kind].input_count == definition.input_count >= len(circuit.In(node))
        for rank, address in enumerate(out):
            assert definition.out_rank(address - node.frame.base) == rank
        for offset in range(definition.size):
            found = definition.out_rank(offset) is not None
            assert found == (node.frame.base + offset in expected)

    units = list(index.replay_units)
    boundary = index.boundary()
    expected_boundary = sorted(set(circuit.inputs).union(*(expected_out[unit] for unit in units)))
    assert boundary.count == len(boundary) == len(expected_boundary)
    members = [boundary.unrank(rank) for rank in range(boundary.count)]
    assert sorted(members) == expected_boundary
    assert members[: circuit.input_count] == list(circuit.inputs)
    assert [boundary.rank(address) for address in members] == list(range(boundary.count))
    assert [address for address in range(circuit.n) if boundary.contains(address)] == expected_boundary
    assert all(output in boundary for output in circuit.outputs)
    for address in range(circuit.n):
        if address not in expected_boundary:
            with pytest.raises(KeyError):
                boundary.rank(address)

    for r, unit in enumerate(units):
        interior = index.interior(r)
        expected_interior = [a for a in unit.interval if a not in expected_out[unit]]
        assert interior.count == len(interior) == len(expected_interior)
        assert list(iter_domain(interior)) == expected_interior == list(interior)
        assert [interior.rank(a) for a in expected_interior] == list(range(len(expected_interior)))
        assert [a for a in range(circuit.n) if interior.contains(a)] == expected_interior
        for address in unit.interval:
            if address not in expected_interior:
                with pytest.raises(KeyError):
                    interior.rank(address)


@pytest.fixture
def check_interfaces() -> Callable[[Index, DescriptionCircuit], None]:
    return assert_interfaces_match_enumeration
