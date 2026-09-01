"""Exact minimum-width downstream vertex cuts."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import networkx as nx

from circuit_cut_analysis.capacity import LogCardinality, sum_capacities
from circuit_cut_analysis.circuit import CircuitDAG, GateId


class CutStatus(StrEnum):
    FINITE = "finite"
    NO_PATH = "no_path"
    NO_FINITE_CUT = "no_finite_cut"


class CutPolicy(StrEnum):
    """Which path endpoints may themselves serve as cut gates."""

    ALL = "all"
    DOWNSTREAM = "downstream"
    INTERNAL = "internal"


class CanonicalCut(StrEnum):
    """Canonical representative when multiple cuts have minimum width."""

    SOURCE_MOST = "source_most"
    DOWNSTREAM_MOST = "downstream_most"


CuttableSpec: TypeAlias = CutPolicy | Collection[GateId]


@dataclass(frozen=True, slots=True)
class CutResult:
    """Result of one source-set to output-set cut query."""

    status: CutStatus
    width_bits: int | float | None
    exact_capacity: LogCardinality | None
    cut: frozenset[GateId]
    source_most_cut: frozenset[GateId]
    downstream_most_cut: frozenset[GateId]
    canonical: CanonicalCut
    sources: frozenset[GateId]
    reachable_outputs: frozenset[GateId]
    corridor: frozenset[GateId]
    cuttable: frozenset[GateId]
    witness_path: tuple[GateId, ...] | None = None

    @property
    def tied(self) -> bool:
        """Whether the two extremal minimum cuts differ."""

        return self.source_most_cut != self.downstream_most_cut


def _resolve_cuttable(
    circuit: CircuitDAG,
    corridor: frozenset[GateId],
    sources: frozenset[GateId],
    outputs: frozenset[GateId],
    spec: CuttableSpec,
) -> frozenset[GateId]:
    if isinstance(spec, CutPolicy):
        if spec is CutPolicy.ALL:
            return corridor
        if spec is CutPolicy.DOWNSTREAM:
            return corridor.difference(sources)
        if spec is CutPolicy.INTERNAL:
            return corridor.difference(sources).difference(outputs)
        raise AssertionError(f"unhandled cut policy: {spec}")
    return circuit.require_gates(spec).intersection(corridor)


def _positive_residual_graph(
    residual: nx.DiGraph[object],
) -> nx.DiGraph[object]:
    positive: nx.DiGraph[object] = nx.DiGraph()
    positive.add_nodes_from(residual.nodes)
    for source, target, data in residual.edges(data=True):
        capacity = int(data["capacity"])
        flow = int(data["flow"])
        if capacity - flow > 0:
            positive.add_edge(source, target)
    return positive


def _cut_from_source_side(
    source_side: set[object],
    corridor: frozenset[GateId],
) -> frozenset[GateId]:
    return frozenset(
        gate_id
        for gate_id in corridor
        if ("in", gate_id) in source_side and ("out", gate_id) not in source_side
    )


def _cut_capacity(
    circuit: CircuitDAG,
    cut: Iterable[GateId],
) -> LogCardinality:
    return sum_capacities(circuit.gates[gate_id].capacity for gate_id in cut)


@dataclass(slots=True)
class _ExactResidualEdge:
    target: object
    reverse: int
    residual: LogCardinality


class _ExactFlowNetwork:
    """Dinic residual network over exact logarithmic capacities."""

    def __init__(self) -> None:
        self.adjacency: dict[object, list[_ExactResidualEdge]] = {}

    def add_edge(
        self,
        source: object,
        target: object,
        capacity: LogCardinality,
    ) -> None:
        source_edges = self.adjacency.setdefault(source, [])
        target_edges = self.adjacency.setdefault(target, [])
        source_reverse = len(target_edges)
        target_reverse = len(source_edges)
        source_edges.append(
            _ExactResidualEdge(
                target=target,
                reverse=source_reverse,
                residual=capacity,
            )
        )
        target_edges.append(
            _ExactResidualEdge(
                target=source,
                reverse=target_reverse,
                residual=LogCardinality.zero(),
            )
        )

    def _levels(self, source: object, sink: object) -> dict[object, int]:
        zero = LogCardinality.zero()
        levels = {source: 0}
        queue = deque((source,))
        while queue:
            node = queue.popleft()
            next_level = levels[node] + 1
            for edge in self.adjacency[node]:
                if edge.residual <= zero or edge.target in levels:
                    continue
                levels[edge.target] = next_level
                queue.append(edge.target)
        if sink not in levels:
            return {}
        return levels

    def _send(
        self,
        node: object,
        sink: object,
        available: LogCardinality,
        levels: dict[object, int],
        next_edges: dict[object, int],
    ) -> LogCardinality:
        if node == sink:
            return available
        zero = LogCardinality.zero()
        edges = self.adjacency[node]
        while next_edges[node] < len(edges):
            edge_index = next_edges[node]
            edge = edges[edge_index]
            if edge.residual > zero and levels.get(edge.target) == levels[node] + 1:
                pushed = self._send(
                    edge.target,
                    sink,
                    min(available, edge.residual),
                    levels,
                    next_edges,
                )
                if pushed > zero:
                    edge.residual -= pushed
                    reverse = self.adjacency[edge.target][edge.reverse]
                    reverse.residual += pushed
                    return pushed
            next_edges[node] += 1
        return zero

    def maximum_flow(
        self,
        source: object,
        sink: object,
        sentinel: LogCardinality,
    ) -> LogCardinality:
        flow = LogCardinality.zero()
        zero = LogCardinality.zero()
        while levels := self._levels(source, sink):
            next_edges = {node: 0 for node in self.adjacency}
            while True:
                pushed = self._send(
                    source,
                    sink,
                    sentinel,
                    levels,
                    next_edges,
                )
                if pushed == zero:
                    break
                flow += pushed
        return flow

    def source_reachable(self, source: object) -> set[object]:
        zero = LogCardinality.zero()
        seen = {source}
        queue = deque((source,))
        while queue:
            node = queue.popleft()
            for edge in self.adjacency[node]:
                if edge.residual <= zero or edge.target in seen:
                    continue
                seen.add(edge.target)
                queue.append(edge.target)
        return seen

    def can_reach(self, sink: object) -> set[object]:
        zero = LogCardinality.zero()
        predecessors: dict[object, list[object]] = {node: [] for node in self.adjacency}
        for source, edges in self.adjacency.items():
            for edge in edges:
                if edge.residual > zero:
                    predecessors[edge.target].append(source)
        seen = {sink}
        queue = deque((sink,))
        while queue:
            node = queue.popleft()
            for source in predecessors[node]:
                if source in seen:
                    continue
                seen.add(source)
                queue.append(source)
        return seen


def _solve_exact_flow(
    circuit: CircuitDAG,
    corridor: frozenset[GateId],
    eligible: frozenset[GateId],
    source_set: frozenset[GateId],
    reachable_outputs: frozenset[GateId],
    sentinel: LogCardinality,
) -> tuple[LogCardinality, set[object], set[object]]:
    network = _ExactFlowNetwork()
    super_source: tuple[str, str] = ("super", "source")
    super_sink: tuple[str, str] = ("super", "sink")
    for gate_id in corridor:
        capacity = (
            circuit.gates[gate_id].capacity.log_value
            if gate_id in eligible
            else sentinel
        )
        network.add_edge(("in", gate_id), ("out", gate_id), capacity)
    for source, target in circuit.edges:
        if source in corridor and target in corridor:
            network.add_edge(("out", source), ("in", target), sentinel)
    for gate_id in source_set.intersection(corridor):
        network.add_edge(super_source, ("in", gate_id), sentinel)
    for gate_id in reachable_outputs:
        network.add_edge(("out", gate_id), super_sink, sentinel)

    flow = network.maximum_flow(super_source, super_sink, sentinel)
    source_side_min = network.source_reachable(super_source)
    can_reach_sink = network.can_reach(super_sink)
    source_side_max = set(network.adjacency).difference(can_reach_sink)
    return flow, source_side_min, source_side_max


def minimum_vertex_cut(
    circuit: CircuitDAG,
    sources: Iterable[GateId],
    outputs: Iterable[GateId] | None = None,
    *,
    cuttable: CuttableSpec = CutPolicy.ALL,
    canonical: CanonicalCut = CanonicalCut.DOWNSTREAM_MOST,
) -> CutResult:
    """Compute a minimum-total-width downstream gate cut.

    The reduction splits every gate into an input and output vertex joined by
    an edge carrying the gate width.  Circuit dependency edges receive a
    rigorously finite sentinel capacity greater than every eligible cut.
    """

    source_set = circuit.require_gates(sources)
    output_set = circuit.outputs if outputs is None else circuit.require_gates(outputs)
    corridor = circuit.live_corridor(source_set, output_set)
    reachable_outputs = output_set.intersection(corridor)

    if not corridor:
        return CutResult(
            status=CutStatus.NO_PATH,
            width_bits=0,
            exact_capacity=LogCardinality.zero(),
            cut=frozenset(),
            source_most_cut=frozenset(),
            downstream_most_cut=frozenset(),
            canonical=canonical,
            sources=source_set,
            reachable_outputs=frozenset(),
            corridor=frozenset(),
            cuttable=frozenset(),
        )

    eligible = _resolve_cuttable(
        circuit,
        corridor,
        source_set,
        output_set,
        cuttable,
    )
    protected = corridor.difference(eligible)
    protected_path = circuit.find_path(
        source_set,
        output_set,
        allowed=protected,
    )
    if protected_path is not None:
        return CutResult(
            status=CutStatus.NO_FINITE_CUT,
            width_bits=None,
            exact_capacity=None,
            cut=frozenset(),
            source_most_cut=frozenset(),
            downstream_most_cut=frozenset(),
            canonical=canonical,
            sources=source_set,
            reachable_outputs=reachable_outputs,
            corridor=corridor,
            cuttable=eligible,
            witness_path=protected_path,
        )

    finite_capacity = sum_capacities(
        circuit.gates[gate_id].capacity for gate_id in eligible
    )
    if finite_capacity.is_zero:
        raise AssertionError(
            "a live path without a protected path needs a cuttable gate"
        )
    sentinel_capacity = finite_capacity + LogCardinality.bits(1)
    integral_width = finite_capacity.integral_width_bits
    if integral_width is not None:
        sentinel_width = integral_width + 1
        network: nx.DiGraph[object] = nx.DiGraph()
        super_source: tuple[str, str] = ("super", "source")
        super_sink: tuple[str, str] = ("super", "sink")
        for gate_id in corridor:
            gate_width = circuit.gates[gate_id].capacity.integral_width_bits
            if gate_id in eligible and gate_width is None:
                raise AssertionError("integral total contains a non-integral gate")
            capacity = gate_width if gate_id in eligible else sentinel_width
            network.add_edge(("in", gate_id), ("out", gate_id), capacity=capacity)
        for source, target in circuit.edges:
            if source in corridor and target in corridor:
                network.add_edge(
                    ("out", source),
                    ("in", target),
                    capacity=sentinel_width,
                )
        for gate_id in source_set.intersection(corridor):
            network.add_edge(
                super_source,
                ("in", gate_id),
                capacity=sentinel_width,
            )
        for gate_id in reachable_outputs:
            network.add_edge(
                ("out", gate_id),
                super_sink,
                capacity=sentinel_width,
            )

        residual = nx.algorithms.flow.preflow_push(
            network,
            super_source,
            super_sink,
            capacity="capacity",
        )
        flow_width = int(residual.graph["flow_value"])
        flow_capacity = LogCardinality.bits(flow_width)
        positive = _positive_residual_graph(residual)
        source_side_min: set[object] = {
            super_source,
            *nx.descendants(positive, super_source),
        }
        can_reach_sink: set[object] = {
            super_sink,
            *nx.ancestors(positive, super_sink),
        }
        source_side_max = set(positive.nodes).difference(can_reach_sink)
    else:
        flow_capacity, source_side_min, source_side_max = _solve_exact_flow(
            circuit,
            corridor,
            eligible,
            source_set,
            reachable_outputs,
            sentinel_capacity,
        )

    if flow_capacity >= sentinel_capacity:
        raise AssertionError("finite-cut precheck disagrees with max-flow result")

    source_most_cut = _cut_from_source_side(source_side_min, corridor)
    downstream_most_cut = _cut_from_source_side(source_side_max, corridor)
    for name, result_cut in (
        ("source-most", source_most_cut),
        ("downstream-most", downstream_most_cut),
    ):
        if _cut_capacity(circuit, result_cut) != flow_capacity:
            raise AssertionError(f"{name} residual partition has inconsistent width")
        if not circuit.is_downstream_cut(source_set, result_cut, output_set):
            raise AssertionError(f"{name} residual partition is not a valid cut")

    chosen = (
        source_most_cut
        if canonical is CanonicalCut.SOURCE_MOST
        else downstream_most_cut
    )
    return CutResult(
        status=CutStatus.FINITE,
        width_bits=flow_capacity.width_bits,
        exact_capacity=flow_capacity,
        cut=chosen,
        source_most_cut=source_most_cut,
        downstream_most_cut=downstream_most_cut,
        canonical=canonical,
        sources=source_set,
        reachable_outputs=reachable_outputs,
        corridor=corridor,
        cuttable=eligible,
    )


def singleton_source_cuts(
    circuit: CircuitDAG,
    sources: Iterable[GateId] | None = None,
    outputs: Iterable[GateId] | None = None,
    *,
    cuttable: CuttableSpec = CutPolicy.ALL,
    canonical: CanonicalCut = CanonicalCut.DOWNSTREAM_MOST,
) -> dict[GateId, CutResult]:
    """Compute one exact cut query for each selected source gate.

    By default, every gate not declared with ``op="input"`` is selected.
    Passing ``sources`` overrides that default, including when the requested
    set contains fixed inputs.
    """

    selected_set = (
        circuit.computed_gates if sources is None else circuit.require_gates(sources)
    )
    selected = tuple(
        gate_id for gate_id in circuit.topological_order if gate_id in selected_set
    )
    return {
        gate_id: minimum_vertex_cut(
            circuit,
            {gate_id},
            outputs,
            cuttable=cuttable,
            canonical=canonical,
        )
        for gate_id in selected
    }
