"""Guarantee-carrying interfaces for structural multi-source capacity queries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.circuit import CircuitDAG, GateId
from circuit_cut_analysis.mincut import CutResult, minimum_vertex_cut
from circuit_cut_analysis.partition import GateCutPartition

AttackSupportT = TypeVar("AttackSupportT")


class StructuralCapacityStatus(StrEnum):
    """Strength of one structural capacity answer."""

    EXACT = "exact"
    BOUNDED = "bounded"


@dataclass(frozen=True, slots=True)
class StructuralCapacityEvaluation(Generic[AttackSupportT]):
    """Certified interval for one structural output-capacity query.

    Both bounds are individually certified: the lower bound by a monotone
    witness (for example one selected source's exact minimum cut) and the
    upper bound by an explicitly verified downstream cut.  ``EXACT`` means the
    two certificates coincide.
    """

    lower_bound: LogCardinality
    upper_bound: LogCardinality
    requested_support: AttackSupportT
    evaluated_support: AttackSupportT
    cut_gate_ids: frozenset[GateId] | None
    method: str
    solver_result: CutResult | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lower_bound > self.upper_bound:
            raise ValueError("capacity lower bound exceeds upper bound")
        if not self.method:
            raise ValueError("capacity-evaluation method must be non-empty")

    @property
    def exact_capacity(self) -> LogCardinality | None:
        """Return the capacity only when both certified bounds coincide."""

        if self.lower_bound == self.upper_bound:
            return self.lower_bound
        return None

    @property
    def is_exact(self) -> bool:
        return self.exact_capacity is not None

    @property
    def status(self) -> StructuralCapacityStatus:
        return (
            StructuralCapacityStatus.EXACT
            if self.is_exact
            else StructuralCapacityStatus.BOUNDED
        )


class StructuralCapacityOracle(Protocol[AttackSupportT]):
    """Evaluate ``lambda(E)`` exactly or with an explicit certified interval."""

    def evaluate(
        self,
        attack_support: AttackSupportT,
    ) -> StructuralCapacityEvaluation[AttackSupportT]:
        """Return guarantee-carrying structural capacity evidence."""


@dataclass(frozen=True, slots=True, init=False)
class ExplicitCircuitCapacityOracle:
    """Exact oracle for an explicit circuit, with optional region compression."""

    circuit: CircuitDAG
    outputs: frozenset[GateId]
    canonical_partition: GateCutPartition | None

    def __init__(
        self,
        circuit: CircuitDAG,
        outputs: Iterable[GateId] | None = None,
        *,
        canonical_partition: GateCutPartition | None = None,
    ) -> None:
        resolved_outputs = (
            circuit.outputs if outputs is None else circuit.require_gates(outputs)
        )
        if (
            canonical_partition is not None
            and canonical_partition.outputs != resolved_outputs
        ):
            raise ValueError(
                "canonical partition and capacity oracle must use identical outputs"
            )
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "outputs", resolved_outputs)
        object.__setattr__(self, "canonical_partition", canonical_partition)

    def evaluate(
        self,
        attack_support: frozenset[GateId],
    ) -> StructuralCapacityEvaluation[frozenset[GateId]]:
        requested = self.circuit.require_gates(attack_support)
        evaluated = (
            requested
            if self.canonical_partition is None
            else self.canonical_partition.compress_sources(requested)
        )
        result = minimum_vertex_cut(self.circuit, evaluated, self.outputs)
        capacity = result.exact_capacity
        if capacity is None:
            raise AssertionError("all-gate cut policy must produce an exact capacity")
        method = (
            "explicit-min-cut"
            if evaluated == requested
            else "canonical-region-compressed-explicit-min-cut"
        )
        return StructuralCapacityEvaluation(
            lower_bound=capacity,
            upper_bound=capacity,
            requested_support=requested,
            evaluated_support=evaluated,
            cut_gate_ids=result.cut,
            method=method,
            solver_result=result,
        )
