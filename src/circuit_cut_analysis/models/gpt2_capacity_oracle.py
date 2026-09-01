"""Multi-source structural capacity queries against the indexed GPT-2 circuit.

For an attacked gate set ``E`` the quantity of interest is ``lambda(E)``, the
minimum downstream cut capacity separating ``E`` from the designated token
outputs.  Three certified evaluation routes exist, tried in order:

1. **Corridor max flow.**  When the live corridor from ``E`` to the outputs
   fits inside the expansion limits, the answer is the exact finite-DAG
   solver's, with the cut identity attached.
2. **Single lifted region.**  When every selected gate shares one certified
   canonical cut ``D``, monotonicity plus the cut give
   ``c(D) = lambda({g}) <= lambda(E) <= c(D)``, so the answer is exact.
3. **Certified interval.**  Otherwise the result is bounded below by the
   widest selected singleton capacity (monotonicity) and above by the
   cheapest of three verified downstream cuts: the union of the per-gate
   canonical cuts, the selected gates themselves, and the full output
   frontier.

Routes 2 and 3 require the lifted certificates, which only apply when
:func:`~circuit_cut_analysis.models.gpt2_partition.lifted_certificate_reasons`
is empty.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from circuit_cut_analysis.capacity import LogCardinality, sum_capacities
from circuit_cut_analysis.capacity_oracle import StructuralCapacityEvaluation
from circuit_cut_analysis.indexed import ExpansionLimitExceeded, GateRef
from circuit_cut_analysis.indexed_mincut import minimum_vertex_cut_indexed
from circuit_cut_analysis.models.gpt2_circuit import GPT2IndexedCircuit
from circuit_cut_analysis.models.gpt2_partition import (
    lifted_certificate_reasons,
    lifted_downstream_cut,
)


@dataclass(frozen=True, slots=True, init=False)
class GPT2StructuralCapacityOracle:
    """Exact-when-feasible, interval-certified multi-source ``lambda`` oracle."""

    indexed: GPT2IndexedCircuit
    max_exact_gates: int
    max_exact_edges: int
    lifted_reasons: tuple[str, ...]

    def __init__(
        self,
        indexed: GPT2IndexedCircuit,
        *,
        max_exact_gates: int = 200_000,
        max_exact_edges: int = 2_000_000,
    ) -> None:
        if max_exact_gates <= 0 or max_exact_edges <= 0:
            raise ValueError("expansion limits must be positive")
        object.__setattr__(self, "indexed", indexed)
        object.__setattr__(self, "max_exact_gates", max_exact_gates)
        object.__setattr__(self, "max_exact_edges", max_exact_edges)
        object.__setattr__(
            self,
            "lifted_reasons",
            lifted_certificate_reasons(indexed),
        )

    def _capacity_of(self, refs: Iterable[GateRef]) -> LogCardinality:
        families = self.indexed.circuit.families
        return sum_capacities(families[ref.family].capacity for ref in refs)

    def _require_computed(
        self, attack_support: Iterable[GateRef]
    ) -> frozenset[GateRef]:
        refs = frozenset(attack_support)
        for ref in refs:
            family = self.indexed.circuit.require_ref(ref)
            if family.op == "input":
                raise ValueError(
                    f"attack support must be computed gates, got input {ref.id!r}"
                )
        return refs

    def evaluate(
        self,
        attack_support: Iterable[GateRef],
    ) -> StructuralCapacityEvaluation[frozenset[GateRef]]:
        refs = self._require_computed(attack_support)
        if not refs:
            zero = LogCardinality.zero()
            return StructuralCapacityEvaluation(
                lower_bound=zero,
                upper_bound=zero,
                requested_support=refs,
                evaluated_support=refs,
                cut_gate_ids=frozenset(),
                method="empty-support",
            )

        try:
            indexed_result = minimum_vertex_cut_indexed(
                self.indexed.circuit,
                refs,
                max_gates=self.max_exact_gates,
                max_edges=self.max_exact_edges,
            )
        except ExpansionLimitExceeded as overflow:
            return self._certified_interval(refs, overflow)
        capacity = indexed_result.result.exact_capacity
        if capacity is None:
            raise AssertionError("all-gate corridor cuts must be finite")
        return StructuralCapacityEvaluation(
            lower_bound=capacity,
            upper_bound=capacity,
            requested_support=refs,
            evaluated_support=refs,
            cut_gate_ids=frozenset(ref.id for ref in indexed_result.cut),
            method="indexed-corridor-max-flow",
            solver_result=indexed_result.result,
        )

    def _certified_interval(
        self,
        refs: frozenset[GateRef],
        overflow: ExpansionLimitExceeded,
    ) -> StructuralCapacityEvaluation[frozenset[GateRef]]:
        if self.lifted_reasons:
            raise ExpansionLimitExceeded(
                "the live corridor exceeds the exact expansion limits and the "
                f"lifted certificates do not apply: {self.lifted_reasons!r}"
            ) from overflow

        zero = LogCardinality.zero()
        lower = zero
        union_cut: set[GateRef] = set()
        for ref in refs:
            region_cut = lifted_downstream_cut(self.indexed, ref)
            union_cut.update(region_cut)
            region_capacity = self._capacity_of(region_cut)
            if region_capacity > lower:
                lower = region_capacity

        candidates = {
            "union-of-certified-singleton-cuts": self._capacity_of(union_cut),
            "attacked-gates-self-cut": self._capacity_of(refs),
            "designated-output-frontier": self._capacity_of(
                self.indexed.circuit.outputs
            ),
        }
        upper_method, upper = min(candidates.items(), key=lambda item: item[1])
        cut_ids = (
            frozenset(ref.id for ref in union_cut)
            if upper_method == "union-of-certified-singleton-cuts"
            else None
        )
        return StructuralCapacityEvaluation(
            lower_bound=lower,
            upper_bound=upper,
            requested_support=refs,
            evaluated_support=refs,
            cut_gate_ids=cut_ids,
            method="lifted-region-certified-interval",
            notes=(
                "lower bound: widest selected singleton canonical cut",
                f"upper bound: {upper_method}",
            ),
        )
