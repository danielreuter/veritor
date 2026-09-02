"""Exact finite capacity provider shared by executable call-DAG plug-ins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from circuit_cut_analysis.capacity import GateCapacity, LogCardinality, sum_capacities
from circuit_cut_analysis.capacity_oracle import ExplicitCircuitCapacityOracle
from circuit_cut_analysis.circuit import CircuitDAG, Gate
from veritor.compile import CallDagCircuit

from ..api import (
    CapacityBoundEvidence,
    CapacityClaimKind,
)


@dataclass(frozen=True, slots=True)
class CallDagCapacityBoundProvider:
    """Exact finite structural capacity oracle for a compiled call DAG."""

    circuit: CallDagCircuit
    claim_kind: CapacityClaimKind = field(
        init=False,
        default=CapacityClaimKind.EXACT,
    )

    def _call_dag(self) -> CallDagCircuit:
        if not isinstance(self.circuit, CallDagCircuit):
            raise TypeError("capacity provider needs a CallDagCircuit")
        return self.circuit

    @staticmethod
    def _gate_id(position: int) -> str:
        return f"value/{position}"

    def _explicit(self) -> CircuitDAG:
        circuit = self._call_dag()
        cardinality = 1 << circuit.cell_bits
        capacity = GateCapacity.values(cardinality)
        gates = [
            Gate(self._gate_id(port.position), capacity, "input")
            for port in circuit.input_ports
        ]
        edges: list[tuple[str, str]] = []
        for rank in range(circuit.computed_positions.count):
            position = circuit.computed_positions.unrank(rank)
            gate = circuit.gate_at(position)
            gates.append(
                Gate(
                    self._gate_id(position),
                    capacity,
                    str(gate.operation),
                )
            )
            edges.extend(
                (self._gate_id(source), self._gate_id(position))
                for source in set(gate.predecessors)
            )
        return CircuitDAG(
            gates,
            edges,
            (self._gate_id(port.position) for port in circuit.output_ports),
        )

    @property
    def output_frontier(self) -> LogCardinality:
        circuit = self._call_dag()
        capacity = GateCapacity.values(1 << circuit.cell_bits)
        return sum_capacities(
            capacity for _position in {port.position for port in circuit.output_ports}
        )

    def evaluate(self, attack: Sequence[int]) -> CapacityBoundEvidence:
        positions = frozenset(attack)
        if not positions:
            zero = LogCardinality.zero()
            return CapacityBoundEvidence(
                lower_bound=zero,
                upper_bound=zero,
                claim_kind=self.claim_kind,
                method="empty-support",
                certificate="the empty attack has zero structural capacity",
                cut_gate_ids=frozenset(),
            )
        circuit = self._call_dag()
        for position in positions:
            if not circuit.computed_positions.contains(position):
                raise ValueError(f"attack position {position} is not a computed gate")
        evaluation = ExplicitCircuitCapacityOracle(self._explicit()).evaluate(
            frozenset(self._gate_id(position) for position in positions)
        )
        return CapacityBoundEvidence(
            lower_bound=evaluation.lower_bound,
            upper_bound=evaluation.upper_bound,
            claim_kind=self.claim_kind,
            method=evaluation.method,
            certificate="exact finite-DAG minimum vertex cut",
            cut_gate_ids=evaluation.cut_gate_ids,
        )


__all__ = ["CallDagCapacityBoundProvider"]
