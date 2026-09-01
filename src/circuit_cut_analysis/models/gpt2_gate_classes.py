"""Counted GPT-2 probability classes for scalar-gate minimax sampling.

Probability classes tie scalar-gate inclusion probabilities; they are not
all-or-nothing verification units.  The adversary may choose arbitrary gates
inside every class.  Each class carries:

* an exact computed-gate count;
* a certified upper bound on every member's singleton canonical cut; and
* a certified cap for the union of every canonical cut represented by the
  class.

Consequently arbitrary attack counts ``e_i`` have the sound envelope

``min(output_frontier, sum_i min(e_i * singleton_i, aggregate_i))``.

The row, row-layer, and row-layer-band catalogs are increasingly refined
parameterizations of verifier probabilities.  Capacity-equal classes may be
coalesced only after adopting this capped-linear envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from circuit_cut_analysis.capacity import GateCapacity, LogCardinality
from circuit_cut_analysis.indexed import GateRef
from circuit_cut_analysis.models.gpt2_circuit import GPT2IndexedCircuit
from circuit_cut_analysis.models.gpt2_partition import (
    _assignment_row,
    _owner_plan,
    _ref_schedule,
    lifted_certificate_reasons,
)
from circuit_cut_analysis.models.gpt2_sampling_study import build_gpt2_region_units
from circuit_cut_analysis.weighted_sampling import (
    WeightedGateClassPartition,
    capacity_upper_bound_for_counts,
    coalesce_frontier_equivalent_classes,
    weighted_partition_from_region_units,
)


class GPT2ClassGranularity(StrEnum):
    """Supported shared-probability refinements."""

    ROW = "row"
    ROW_LAYER = "row-layer"
    ROW_LAYER_BAND = "row-layer-band"


@dataclass(frozen=True, slots=True)
class GPT2GateClassCatalog:
    """A proof-carrying symbolic probability-class inventory."""

    indexed: GPT2IndexedCircuit
    granularity: GPT2ClassGranularity
    position_bands: int
    partition: WeightedGateClassPartition

    @property
    def class_count(self) -> int:
        return len(self.partition.classes)

    @property
    def computed_gate_count(self) -> int:
        return self.partition.total_gate_count

    def capacity_upper_bound(
        self,
        attacked_counts: tuple[int, ...],
    ) -> LogCardinality:
        """Return the certified structural bound for arbitrary class counts."""

        return capacity_upper_bound_for_counts(self.partition, attacked_counts)

    def coalesced_for_linear_game(self) -> WeightedGateClassPartition:
        """Return the exact frontier-equivalent minimax reduction."""

        return coalesce_frontier_equivalent_classes(self.partition)

    def class_id_for(self, ref: GateRef) -> str:
        """Return the symbolic probability class containing one computed gate."""

        return classify_gpt2_gate(
            self.indexed,
            ref,
            granularity=self.granularity,
            position_bands=self.position_bands,
        )


def classify_gpt2_gate(
    indexed: GPT2IndexedCircuit,
    ref: GateRef,
    *,
    granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
    position_bands: int = 8,
) -> str:
    """Classify one scalar reference exactly as the counted catalog does."""

    resolved = GPT2ClassGranularity(granularity)
    if position_bands <= 0:
        raise ValueError("position_bands must be positive")
    family = indexed.circuit.require_ref(ref)
    if family.op == "input":
        raise ValueError("fixed input gates are not verifier probability classes")
    plan = _owner_plan(family.name)
    layer, position = _ref_schedule(indexed, ref)
    row_id = _assignment_row(indexed, plan, position=position, layer=layer)
    key: tuple[str | int | None, ...]
    if resolved is GPT2ClassGranularity.ROW:
        key = (row_id,)
    elif resolved is GPT2ClassGranularity.ROW_LAYER:
        key = (row_id, layer)
    else:
        band = position * position_bands // max(indexed.processed_positions, 1)
        key = (row_id, layer, band)
    return "/".join(str(part) for part in key)


def build_gpt2_gate_class_catalog(
    indexed: GPT2IndexedCircuit,
    *,
    granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
    position_bands: int = 8,
) -> GPT2GateClassCatalog:
    """Build a counted class catalog without materializing scalar gates."""

    resolved = GPT2ClassGranularity(granularity)
    if position_bands <= 0:
        raise ValueError("position_bands must be positive")
    reasons = lifted_certificate_reasons(indexed)
    if reasons:
        raise ValueError(f"lifted certificates do not apply: {reasons!r}")
    units = build_gpt2_region_units(
        indexed,
        granularity=resolved.value,
        position_bands=position_bands,
    )
    token = GateCapacity.values(indexed.config.vocabulary_size)
    partition = weighted_partition_from_region_units(
        model_id=indexed.config.model_id,
        units=units,
        output_frontier=token.log_value.scale(indexed.generated_tokens),
        token_cardinality=indexed.config.vocabulary_size,
    )
    if partition.total_gate_count != indexed.circuit.computed_gate_count:
        raise AssertionError("GPT-2 probability classes do not cover computed gates")
    return GPT2GateClassCatalog(
        indexed=indexed,
        granularity=resolved,
        position_bands=position_bands,
        partition=partition,
    )
