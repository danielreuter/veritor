"""Fixed-boundary checkpoint skeletons that preserve every multi-source cut.

For each gate ``g`` let ``D(g)`` be its exact downstream-most minimum cut.  A
*fixed atom* is a gate that is its own cut, ``D(b) = {b}``.  Because
``lambda`` is monotone and submodular over source sets, and because
``lambda({g} u D(g)) = lambda(D(g)) = lambda({g})``, replacing any source by
its cut preserves every joint capacity:

``lambda(F u {g}) = lambda(F u D(g))``.

Iterating this identity pushes every source down to a set of fixed atoms.  The
same replacement applied to cut gates shows some minimum cut consists of fixed
atoms only.  The skeleton therefore keeps just the atoms, connected exactly
when the original circuit has a path between them with no internal atom, and
answers every multi-source query with the same capacity as the full circuit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
class CheckpointSkeleton:
    """Atoms-only quotient preserving every multi-source minimum-cut capacity."""

    original: CircuitDAG
    skeleton: CircuitDAG
    fixed_atoms: frozenset[GateId]
    atom_frontier: Mapping[GateId, frozenset[GateId]]

    def project_sources(self, sources: Iterable[GateId]) -> frozenset[GateId]:
        """Push each source down to its capacity-preserving atom frontier."""

        selected = self.original.require_gates(sources)
        return frozenset(
            atom for gate_id in selected for atom in self.atom_frontier[gate_id]
        )

    def evaluate(self, sources: Iterable[GateId]) -> CutResult:
        """Solve one exact multi-source cut query on the skeleton."""

        return minimum_vertex_cut(
            self.skeleton,
            self.project_sources(sources),
            cuttable=CutPolicy.ALL,
            canonical=CanonicalCut.DOWNSTREAM_MOST,
        )


def compile_checkpoint_skeleton(
    circuit: CircuitDAG,
    outputs: Iterable[GateId] | None = None,
) -> CheckpointSkeleton:
    """Compile the exact fixed-boundary checkpoint skeleton of a circuit."""

    output_set = circuit.outputs if outputs is None else circuit.require_gates(outputs)
    singleton_cuts = singleton_source_cuts(
        circuit,
        sources=tuple(circuit.gates),
        outputs=output_set,
        cuttable=CutPolicy.ALL,
        canonical=CanonicalCut.DOWNSTREAM_MOST,
    )

    atoms: set[GateId] = set()
    for gate_id, result in singleton_cuts.items():
        if result.status is CutStatus.NO_FINITE_CUT:
            raise AssertionError("all-gate cut policy must yield finite cuts")
        if result.cut == frozenset({gate_id}):
            atoms.add(gate_id)
        elif gate_id in result.cut:
            raise AssertionError(
                f"a minimum cut containing {gate_id!r} must be exactly itself"
            )
    missing_outputs = output_set.difference(atoms)
    if missing_outputs:
        raise AssertionError(
            f"designated outputs must be fixed atoms: {sorted(missing_outputs)!r}"
        )

    frontier: dict[GateId, frozenset[GateId]] = {}
    for gate_id in reversed(circuit.topological_order):
        result = singleton_cuts[gate_id]
        if gate_id in atoms:
            frontier[gate_id] = frozenset((gate_id,))
        elif result.status is CutStatus.NO_PATH:
            frontier[gate_id] = frozenset()
        else:
            frontier[gate_id] = frozenset(
                atom for cut_gate in result.cut for atom in frontier[cut_gate]
            )

    skeleton_edges: list[tuple[GateId, GateId]] = []
    for atom in atoms:
        reached: set[GateId] = set()
        queue = deque(circuit.successors(atom))
        seen = set(queue)
        while queue:
            gate_id = queue.popleft()
            if gate_id in atoms:
                reached.add(gate_id)
                continue
            for successor in circuit.successors(gate_id):
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        skeleton_edges.extend((atom, target) for target in sorted(reached))

    skeleton = CircuitDAG(
        gates=(circuit.gates[atom] for atom in sorted(atoms)),
        edges=skeleton_edges,
        outputs=output_set,
    )
    return CheckpointSkeleton(
        original=circuit,
        skeleton=skeleton,
        fixed_atoms=frozenset(atoms),
        atom_frontier=MappingProxyType(frontier),
    )
