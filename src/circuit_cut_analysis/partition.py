"""Automatic gate partitions induced by exact canonical downstream cuts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from circuit_cut_analysis.capacity import LogCardinality, sum_capacities
from circuit_cut_analysis.circuit import CircuitDAG, GateId
from circuit_cut_analysis.mincut import (
    CanonicalCut,
    CutPolicy,
    CutResult,
    CutStatus,
    minimum_vertex_cut,
    singleton_source_cuts,
)


@dataclass(frozen=True, slots=True)
class CutGroup:
    """One automatically discovered region ``A_D`` and its cut occurrence."""

    cut: frozenset[GateId]
    source_gates: frozenset[GateId]
    width_bits: int | float
    exact_capacity: LogCardinality
    singleton_status: CutStatus
    joint_result: CutResult
    joint_cut_valid: bool
    joint_minimum_matches_singletons: bool

    @property
    def verified(self) -> bool:
        """Whether both required joint-cut checks succeeded."""

        return self.joint_cut_valid and self.joint_minimum_matches_singletons


@dataclass(frozen=True, slots=True)
class GateCutPartition:
    """All singleton cuts and the exact-cut partition they induce."""

    outputs: frozenset[GateId]
    source_gates: frozenset[GateId]
    singleton_cuts: Mapping[GateId, CutResult]
    groups: tuple[CutGroup, ...]

    def group_for_gate(self, gate_id: GateId) -> CutGroup:
        """Return the unique cut group containing ``gate_id``."""

        try:
            cut = self.singleton_cuts[gate_id].cut
        except KeyError:
            raise KeyError(f"gate {gate_id!r} is not in this partition") from None
        for group in self.groups:
            if group.cut == cut:
                return group
        raise AssertionError(f"missing cut group for gate {gate_id!r}")

    def compress_sources(self, source_gates: Iterable[GateId]) -> frozenset[GateId]:
        """Replace selected gates in each canonical region by one representative.

        This preserves the joint minimum-cut capacity, not the identity of a
        canonical cut.  If ``A_D`` is a region with singleton capacity
        ``lambda({g}) = lambda(A_D)`` for every ``g`` in the region, monotonicity
        and submodularity imply

        ``lambda(F union {g}) = lambda(F union A_D)``

        for every other source set ``F``.  Repeatedly applying that identity
        leaves at most one stable representative per selected region.  The
        result is useful for multi-source capacity queries; it must not be used
        to merge distinct verification units when computing detection.
        """

        selected = frozenset(source_gates)
        unknown = selected.difference(self.source_gates)
        if unknown:
            raise ValueError(
                f"gates are outside this cut partition: {sorted(unknown)!r}"
            )
        selected_cuts = {self.singleton_cuts[gate_id].cut for gate_id in selected}
        return frozenset(
            min(group.source_gates)
            for group in self.groups
            if group.cut in selected_cuts
        )


def partition_gate_cuts(
    circuit: CircuitDAG,
    source_gates: Iterable[GateId] | None = None,
    outputs: Iterable[GateId] | None = None,
) -> GateCutPartition:
    """Partition gates by their exact downstream-most minimum cut.

    With no ``source_gates`` override, every non-input computation gate is
    analyzed independently. Fixed inputs are gates declared with
    ``op="input"``. The identity of a group is the complete set of cut gate
    IDs, so equal-width cuts at different circuit locations stay distinct.

    Every returned group is checked in two ways: its cut must jointly separate
    all group members from the selected outputs, and a fresh joint-source
    minimum-cut query must have the same width as the singleton queries.
    """

    output_set = circuit.outputs if outputs is None else circuit.require_gates(outputs)
    selected_set = (
        circuit.computed_gates
        if source_gates is None
        else circuit.require_gates(source_gates)
    )
    selected = tuple(
        gate_id for gate_id in circuit.topological_order if gate_id in selected_set
    )
    singleton_cuts = singleton_source_cuts(
        circuit,
        selected,
        output_set,
        cuttable=CutPolicy.ALL,
        canonical=CanonicalCut.DOWNSTREAM_MOST,
    )

    grouped_sources: dict[frozenset[GateId], list[GateId]] = {}
    for gate_id in selected:
        result = singleton_cuts[gate_id]
        if result.status is CutStatus.NO_FINITE_CUT or result.width_bits is None:
            raise AssertionError("all-gate cut policy must yield a finite cut")
        if result.canonical is not CanonicalCut.DOWNSTREAM_MOST:
            raise AssertionError("singleton cut used the wrong canonical policy")
        grouped_sources.setdefault(result.cut, []).append(gate_id)

    groups: list[CutGroup] = []
    for cut, gate_ids in grouped_sources.items():
        group_sources = frozenset(gate_ids)
        exact_capacity = sum_capacities(
            circuit.gates[gate_id].capacity for gate_id in cut
        )
        width_bits = exact_capacity.width_bits
        statuses = {singleton_cuts[gate_id].status for gate_id in group_sources}
        singleton_capacities = {
            singleton_cuts[gate_id].exact_capacity for gate_id in group_sources
        }
        if len(statuses) != 1 or singleton_capacities != {exact_capacity}:
            raise AssertionError("equal cut occurrences have inconsistent results")

        joint_cut_valid = circuit.is_downstream_cut(
            group_sources,
            cut,
            output_set,
        )
        joint_result = minimum_vertex_cut(
            circuit,
            group_sources,
            output_set,
            cuttable=CutPolicy.ALL,
            canonical=CanonicalCut.DOWNSTREAM_MOST,
        )
        joint_minimum_matches = joint_result.exact_capacity == exact_capacity
        if not joint_cut_valid:
            raise AssertionError("canonical cut does not jointly cut its group")
        if not joint_minimum_matches:
            raise AssertionError(
                "joint minimum width differs from shared singleton width"
            )

        groups.append(
            CutGroup(
                cut=cut,
                source_gates=group_sources,
                width_bits=width_bits,
                exact_capacity=exact_capacity,
                singleton_status=next(iter(statuses)),
                joint_result=joint_result,
                joint_cut_valid=joint_cut_valid,
                joint_minimum_matches_singletons=joint_minimum_matches,
            )
        )

    readonly_singletons: Mapping[GateId, CutResult] = MappingProxyType(
        dict(singleton_cuts)
    )
    return GateCutPartition(
        outputs=output_set,
        source_gates=frozenset(selected),
        singleton_cuts=readonly_singletons,
        groups=tuple(groups),
    )
