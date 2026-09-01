"""Finite, width-annotated circuit DAGs.

The graph contains one vertex per computed or input value and an edge
``u -> v`` when gate ``v`` reads gate ``u``.  Gate widths are logarithms of
alphabet cardinalities (normally physical bit widths); values themselves are
never represented.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from circuit_cut_analysis.capacity import GateCapacity

GateId = str
Edge = tuple[GateId, GateId]
INPUT_OPERATION = "input"


@dataclass(frozen=True, slots=True, init=False)
class Gate:
    """A circuit value with a positive capacity weight."""

    id: GateId
    capacity: GateCapacity
    op: str = "gate"

    def __init__(
        self,
        id: GateId,
        width_bits: int | GateCapacity,
        op: str = "gate",
    ) -> None:
        if not id:
            raise ValueError("gate id must be non-empty")
        if isinstance(width_bits, int):
            if width_bits <= 0:
                raise ValueError(f"gate {id!r} must have positive width")
            capacity = GateCapacity.bits(width_bits)
        elif isinstance(width_bits, GateCapacity):
            capacity = width_bits
        else:
            raise TypeError("gate width must be integer bits or GateCapacity")
        if not op:
            raise ValueError(f"gate {id!r} must have a non-empty op")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "op", op)

    @property
    def width_bits(self) -> int | float:
        """Capacity in bits, exactly integral when the alphabet is a power of two."""

        integral = self.capacity.integral_width_bits
        return self.capacity.width_bits if integral is None else integral

    @property
    def is_input(self) -> bool:
        """Whether this gate is a fixed circuit input."""

        return self.op == INPUT_OPERATION


class CircuitDAG:
    """Validated immutable dependency graph for a finite circuit."""

    __slots__ = ("_edges", "_gates", "_outputs", "_pred", "_succ", "_topological")

    def __init__(
        self,
        gates: Iterable[Gate],
        edges: Iterable[Edge],
        outputs: Iterable[GateId],
    ) -> None:
        gate_map: dict[GateId, Gate] = {}
        for gate in gates:
            if gate.id in gate_map:
                raise ValueError(f"duplicate gate id: {gate.id!r}")
            gate_map[gate.id] = gate
        if not gate_map:
            raise ValueError("a circuit must contain at least one gate")

        edge_set = frozenset(edges)
        for source, target in edge_set:
            if source not in gate_map:
                raise ValueError(f"edge references unknown source gate: {source!r}")
            if target not in gate_map:
                raise ValueError(f"edge references unknown target gate: {target!r}")
            if source == target:
                raise ValueError(f"self-loop at gate {source!r}")

        output_set = frozenset(outputs)
        if not output_set:
            raise ValueError("a circuit must designate at least one output")
        unknown_outputs = output_set.difference(gate_map)
        if unknown_outputs:
            raise ValueError(f"unknown output gates: {sorted(unknown_outputs)!r}")

        pred: dict[GateId, set[GateId]] = {gate_id: set() for gate_id in gate_map}
        succ: dict[GateId, set[GateId]] = {gate_id: set() for gate_id in gate_map}
        for source, target in edge_set:
            succ[source].add(target)
            pred[target].add(source)

        topological = self._topological_sort(gate_map, pred, succ)

        self._gates = gate_map
        self._edges = edge_set
        self._outputs = output_set
        self._pred = {gate_id: frozenset(nodes) for gate_id, nodes in pred.items()}
        self._succ = {gate_id: frozenset(nodes) for gate_id, nodes in succ.items()}
        self._topological = topological

    @staticmethod
    def _topological_sort(
        gates: Mapping[GateId, Gate],
        pred: Mapping[GateId, set[GateId]],
        succ: Mapping[GateId, set[GateId]],
    ) -> tuple[GateId, ...]:
        indegree = {gate_id: len(pred[gate_id]) for gate_id in gates}
        ready = deque(
            sorted(gate_id for gate_id, degree in indegree.items() if degree == 0)
        )
        order: list[GateId] = []
        while ready:
            gate_id = ready.popleft()
            order.append(gate_id)
            for target in sorted(succ[gate_id]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(order) != len(gates):
            raise ValueError("circuit dependencies must form a DAG")
        return tuple(order)

    @property
    def gates(self) -> Mapping[GateId, Gate]:
        return self._gates

    @property
    def edges(self) -> frozenset[Edge]:
        return self._edges

    @property
    def outputs(self) -> frozenset[GateId]:
        return self._outputs

    @property
    def input_gates(self) -> frozenset[GateId]:
        """Gates declared as fixed inputs via ``op="input"``."""

        return frozenset(
            gate_id for gate_id, gate in self._gates.items() if gate.is_input
        )

    @property
    def computed_gates(self) -> frozenset[GateId]:
        """All non-input computation gates."""

        return frozenset(self._gates).difference(self.input_gates)

    @property
    def topological_order(self) -> tuple[GateId, ...]:
        return self._topological

    def predecessors(self, gate_id: GateId) -> frozenset[GateId]:
        self.require_gates({gate_id})
        return self._pred[gate_id]

    def successors(self, gate_id: GateId) -> frozenset[GateId]:
        self.require_gates({gate_id})
        return self._succ[gate_id]

    def require_gates(self, gate_ids: Iterable[GateId]) -> frozenset[GateId]:
        result = frozenset(gate_ids)
        unknown = result.difference(self._gates)
        if unknown:
            raise ValueError(f"unknown gates: {sorted(unknown)!r}")
        return result

    def descendants(self, starts: Iterable[GateId]) -> frozenset[GateId]:
        """Return starts and every gate reachable from them."""

        start_set = self.require_gates(starts)
        seen = set(start_set)
        queue = deque(start_set)
        while queue:
            gate_id = queue.popleft()
            for target in self._succ[gate_id]:
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return frozenset(seen)

    def ancestors(self, targets: Iterable[GateId]) -> frozenset[GateId]:
        """Return targets and every gate that can reach them."""

        target_set = self.require_gates(targets)
        seen = set(target_set)
        queue = deque(target_set)
        while queue:
            gate_id = queue.popleft()
            for source in self._pred[gate_id]:
                if source not in seen:
                    seen.add(source)
                    queue.append(source)
        return frozenset(seen)

    def live_corridor(
        self,
        sources: Iterable[GateId],
        outputs: Iterable[GateId] | None = None,
    ) -> frozenset[GateId]:
        """Gates lying on at least one selected source-to-output path."""

        source_set = self.require_gates(sources)
        output_set = self.outputs if outputs is None else self.require_gates(outputs)
        if not source_set or not output_set:
            return frozenset()
        return self.descendants(source_set).intersection(self.ancestors(output_set))

    def find_path(
        self,
        sources: Iterable[GateId],
        outputs: Iterable[GateId] | None = None,
        *,
        removed: Iterable[GateId] = (),
        allowed: Iterable[GateId] | None = None,
    ) -> tuple[GateId, ...] | None:
        """Find one path, including a possible zero-edge source/output path."""

        source_set = self.require_gates(sources)
        output_set = self.outputs if outputs is None else self.require_gates(outputs)
        removed_set = self.require_gates(removed)
        allowed_set = (
            frozenset(self._gates) if allowed is None else self.require_gates(allowed)
        )
        usable = allowed_set.difference(removed_set)
        starts = sorted(source_set.intersection(usable))
        targets = output_set.intersection(usable)
        if not starts or not targets:
            return None

        parent: dict[GateId, GateId | None] = {gate_id: None for gate_id in starts}
        queue = deque(starts)
        endpoint: GateId | None = None
        while queue:
            gate_id = queue.popleft()
            if gate_id in targets:
                endpoint = gate_id
                break
            for target in sorted(self._succ[gate_id]):
                if target in usable and target not in parent:
                    parent[target] = gate_id
                    queue.append(target)
        if endpoint is None:
            return None

        path: list[GateId] = []
        cursor: GateId | None = endpoint
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        return tuple(path)

    def is_downstream_cut(
        self,
        sources: Iterable[GateId],
        cut: Iterable[GateId],
        outputs: Iterable[GateId] | None = None,
    ) -> bool:
        """Whether deleting ``cut`` destroys every selected source/output path."""

        return self.find_path(sources, outputs, removed=cut) is None
