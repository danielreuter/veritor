"""Exact cut queries that lazily expand an indexed circuit corridor."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from circuit_cut_analysis.indexed import GateRef, IndexedCircuit
from circuit_cut_analysis.mincut import (
    CanonicalCut,
    CutPolicy,
    CutResult,
    CuttableSpec,
    minimum_vertex_cut,
)


@dataclass(frozen=True, slots=True)
class IndexedCutResult:
    """An explicit solver result plus its stable indexed gate identities."""

    result: CutResult
    cut: frozenset[GateRef]
    source_most_cut: frozenset[GateRef]
    downstream_most_cut: frozenset[GateRef]
    sources: frozenset[GateRef]
    outputs: frozenset[GateRef]
    expanded_gate_count: int
    expanded_edge_count: int


def _refs_by_id(refs: Iterable[GateRef]) -> dict[str, GateRef]:
    ref_tuple = tuple(refs)
    result = {ref.id: ref for ref in ref_tuple}
    if len(result) != len(ref_tuple):
        raise AssertionError("indexed gate IDs must be injective")
    return result


def minimum_vertex_cut_indexed(
    circuit: IndexedCircuit,
    sources: Iterable[GateRef],
    outputs: Iterable[GateRef] | None = None,
    *,
    max_gates: int,
    max_edges: int,
    cuttable: CutPolicy | Collection[GateRef] = CutPolicy.ALL,
    canonical: CanonicalCut = CanonicalCut.DOWNSTREAM_MOST,
) -> IndexedCutResult:
    """Expand one exact live corridor on demand, then run vertex-split max-flow.

    The safety limits are part of the API: exceeding them raises
    :class:`~circuit_cut_analysis.indexed.ExpansionLimitExceeded` rather than
    silently approximating the graph.
    """

    source_set = frozenset(sources)
    output_set = circuit.outputs if outputs is None else frozenset(outputs)
    expanded = circuit.materialize_corridor(
        source_set,
        output_set,
        max_gates=max_gates,
        max_edges=max_edges,
    )
    all_refs = tuple(circuit.ref_from_id(gate_id) for gate_id in expanded.gates)
    by_id = _refs_by_id(all_refs)

    explicit_cuttable: CuttableSpec
    if isinstance(cuttable, CutPolicy):
        explicit_cuttable = cuttable
    else:
        explicit_cuttable = {ref.id for ref in cuttable if ref.id in expanded.gates}
    result = minimum_vertex_cut(
        expanded,
        {ref.id for ref in source_set},
        {ref.id for ref in output_set if ref.id in expanded.gates},
        cuttable=explicit_cuttable,
        canonical=canonical,
    )

    def convert(gate_ids: Iterable[str]) -> frozenset[GateRef]:
        return frozenset(by_id[gate_id] for gate_id in gate_ids)

    return IndexedCutResult(
        result=result,
        cut=convert(result.cut),
        source_most_cut=convert(result.source_most_cut),
        downstream_most_cut=convert(result.downstream_most_cut),
        sources=source_set,
        outputs=frozenset(by_id[gate_id] for gate_id in expanded.outputs),
        expanded_gate_count=len(expanded.gates),
        expanded_edge_count=len(expanded.edges),
    )
