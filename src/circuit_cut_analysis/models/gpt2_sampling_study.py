"""Certified verification-sampling study over lifted GPT-2 canonical regions.

The full inference circuit is far too large to enumerate, so this study works
on *region units*: disjoint groups of computed gates aggregated from the
certified lifted partition at a configurable granularity.  For each unit the
lifted certificates give an exact checked-gate cost and a certified upper
bound on the structural capacity an adversary gains by corrupting the unit:

* every corrupted gate's canonical cut lies inside the unit's own boundary
  occurrences, so the union-of-cuts capacity is ``occurrences x cut width``;
* the designated output frontier (``G x log2(V)`` bits) caps every attack.

Detection probabilities are exact for the strategy families studied here, so
the reported verifier guarantees are sound; when a bound rounds, it rounds in
the adversary's favor.  Optimality is *not* claimed: the strategies are
well-defined families (uniform independent, greedy protection, coarse LP)
whose certified bounds can be compared across budgets and granularities.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from circuit_cut_analysis.models.gpt2_circuit import GPT2IndexedCircuit
from circuit_cut_analysis.models.gpt2_partition import (
    _assignment_row,
    _capacity_for_plan,
    _count_family,
    _iter_family_chunks,
    _owner_plan,
    lifted_certificate_reasons,
)
from circuit_cut_analysis.sampling_study import (
    GranularitySummary,
    RegionUnit,
    SamplingStudyReport,
    StrategyOutcome,
    TilePhaseRow,
    certified_adversary_bits,
    coarse_lp_outcome,
    expected_checked_gates,
    greedy_protection_probabilities,
    tile_phase_table,
    uniform_probabilities,
)

__all__ = [
    "GPT2SamplingStudy",
    "GranularitySummary",
    "RegionUnit",
    "StrategyOutcome",
    "TilePhaseRow",
    "build_gpt2_region_units",
    "certified_adversary_bits",
    "expected_checked_gates",
    "greedy_protection_probabilities",
    "output_frontier_bits",
    "run_gpt2_sampling_study",
    "tile_phase_table",
    "uniform_probabilities",
    "write_gpt2_sampling_study",
]

_SPECIAL_LOCALIZED_ROWS = frozenset(
    {
        "dead-empty-cut",
        "token-output",
        "final-token-output",
        "penultimate-output-pair",
        "length-one-attention-probability",
        "length-two-softmax-probability-pair",
    }
)


GPT2SamplingStudy = SamplingStudyReport
"""The GPT-2 study emits the shared cross-model report type."""


def _row_capacity_bits(
    indexed: GPT2IndexedCircuit,
    row_id: str,
    *,
    layer_scoped: bool,
) -> float:
    circuit = indexed.circuit
    token = circuit.families["output/argmax"].capacity.width_bits
    probability = circuit.families[
        "blocks/attention/softmax/probability"
    ].capacity.width_bits
    head_cuts = indexed.config.heads * (1 if layer_scoped else indexed.config.layers)
    if row_id == "dead-empty-cut":
        return 0.0
    if row_id == "token-output":
        return (indexed.generated_tokens - 1) * float(token)
    if row_id == "final-token-output":
        return float(token)
    if row_id == "penultimate-output-pair":
        return 2.0 * float(token)
    if row_id == "length-one-attention-probability":
        return head_cuts * float(probability)
    if row_id == "length-two-softmax-probability-pair":
        return head_cuts * 2.0 * float(probability)
    raise AssertionError(f"row {row_id!r} is not a specially localized row")


def _row_single_cut_bits(indexed: GPT2IndexedCircuit, row_id: str) -> float:
    circuit = indexed.circuit
    token = float(circuit.families["output/argmax"].capacity.width_bits)
    probability = float(
        circuit.families["blocks/attention/softmax/probability"].capacity.width_bits
    )
    if row_id == "dead-empty-cut":
        return 0.0
    if row_id in {"token-output", "final-token-output"}:
        return token
    if row_id == "penultimate-output-pair":
        return 2.0 * token
    if row_id == "length-one-attention-probability":
        return probability
    if row_id == "length-two-softmax-probability-pair":
        return 2.0 * probability
    raise AssertionError(f"row {row_id!r} is not a specially localized row")


def _boundary_chunk_occurrences(
    indexed: GPT2IndexedCircuit,
    family_name: str,
    plan_boundary: str,
    boundary_mode: str,
    *,
    layer: int | None,
    position: int,
    source_count: int,
) -> int:
    if family_name != plan_boundary:
        return 0
    if boundary_mode != "causal-final":
        return source_count
    if position <= 0:
        return 0
    family = indexed.circuit.families[family_name]
    fixed: dict[str, int] = {"query": position, "key": position}
    if layer is not None:
        fixed["layer"] = layer
    return _count_family(family, **fixed)


def build_gpt2_region_units(
    indexed: GPT2IndexedCircuit,
    *,
    granularity: str,
    position_bands: int = 8,
) -> tuple[RegionUnit, ...]:
    """Aggregate the certified lifted regions into disjoint sampling units.

    ``granularity`` is one of ``"row"``, ``"row-layer"``, or
    ``"row-layer-band"``.  Units always keep a source and its canonical
    boundary gates together, so each unit's capacity upper bound is the
    certified union-of-cuts bound for corrupting every gate it contains.
    """

    if granularity not in {"row", "row-layer", "row-layer-band"}:
        raise ValueError(f"unknown granularity {granularity!r}")
    reasons = lifted_certificate_reasons(indexed)
    if reasons:
        raise ValueError(f"lifted certificates do not apply: {reasons!r}")

    UnitKey = tuple[str | int | None, ...]
    circuit = indexed.circuit
    costs: dict[UnitKey, int] = defaultdict(int)
    local_occurrences: dict[UnitKey, int] = defaultdict(int)
    local_cut_bits: dict[UnitKey, float] = {}
    special_rows_touched: dict[UnitKey, set[str]] = defaultdict(set)
    row_sets: dict[UnitKey, set[str]] = defaultdict(set)

    def unit_key(row_id: str, layer: int | None, position: int) -> UnitKey:
        if granularity == "row":
            return (row_id,)
        if granularity == "row-layer":
            return (row_id, layer)
        band = position * position_bands // max(indexed.processed_positions, 1)
        return (row_id, layer, band)

    for family in circuit.families.values():
        if family.op == "input":
            continue
        plan = _owner_plan(family.name)
        for layer, position, source_count in _iter_family_chunks(indexed, family):
            row_id = _assignment_row(indexed, plan, position=position, layer=layer)
            key = unit_key(row_id, layer, position)
            costs[key] += source_count
            row_sets[key].add(row_id)
            is_local = row_id == plan.row_id and plan.stage != "output"
            if is_local:
                local_occurrences[key] += _boundary_chunk_occurrences(
                    indexed,
                    family.name,
                    plan.boundary_family,
                    plan.boundary_mode,
                    layer=layer,
                    position=position,
                    source_count=source_count,
                )
                local_cut_bits[key] = float(
                    _capacity_for_plan(indexed, plan).width_bits
                )
            else:
                special_rows_touched[key].add(row_id)

    units: list[RegionUnit] = []
    for key in sorted(costs, key=repr):
        capacity = local_occurrences.get(key, 0) * local_cut_bits.get(key, 0.0)
        single = local_cut_bits.get(key, 0.0) if local_occurrences.get(key, 0) else 0.0
        layer_scoped = granularity != "row" and len(key) > 1 and key[1] is not None
        for row_id in sorted(special_rows_touched.get(key, ())):
            if row_id not in _SPECIAL_LOCALIZED_ROWS:
                raise AssertionError(f"unhandled non-local row {row_id!r}")
            capacity += _row_capacity_bits(indexed, row_id, layer_scoped=layer_scoped)
            single = max(single, _row_single_cut_bits(indexed, row_id))
        units.append(
            RegionUnit(
                id="/".join(str(part) for part in key),
                row_ids=tuple(sorted(row_sets[key])),
                checked_gate_count=costs[key],
                capacity_upper_bits=capacity,
                max_single_cut_bits=single,
            )
        )
    total = sum(unit.checked_gate_count for unit in units)
    if total != circuit.computed_gate_count:
        raise AssertionError(
            f"units cover {total:,} gates, expected {circuit.computed_gate_count:,}"
        )
    return tuple(units)


def output_frontier_bits(indexed: GPT2IndexedCircuit) -> float:
    """Certified capacity of the always-valid designated-output cut."""

    token = float(indexed.circuit.families["output/argmax"].capacity.width_bits)
    return indexed.generated_tokens * token


def run_gpt2_sampling_study(
    indexed: GPT2IndexedCircuit,
    *,
    granularities: Iterable[str] = ("row", "row-layer", "row-layer-band"),
    budget_fractions: Iterable[float] = (1e-3, 1e-1, 0.5, 0.9, 0.99),
    detection_thresholds: Iterable[float] = (0.5, 0.9, 0.99),
    position_bands: int = 8,
) -> GPT2SamplingStudy:
    """Compare certified strategy families across granularities and budgets."""

    started = time.perf_counter()
    frontier = output_frontier_bits(indexed)
    total_gates = indexed.circuit.computed_gate_count
    outcomes: list[StrategyOutcome] = []
    summaries: list[GranularitySummary] = []
    row_total_capacity = 0.0

    for granularity in granularities:
        units = build_gpt2_region_units(
            indexed,
            granularity=granularity,
            position_bands=position_bands,
        )
        capacities = [unit.capacity_upper_bits for unit in units]
        positive = [bits for bits in capacities if bits > 0]
        summaries.append(
            GranularitySummary(
                granularity=granularity,
                unit_count=len(units),
                units_above_frontier=sum(bits > frontier for bits in capacities),
                min_unit_capacity_bits=min(positive) if positive else 0.0,
                max_unit_capacity_bits=max(capacities) if capacities else 0.0,
                total_capacity_bits=sum(capacities),
            )
        )
        if granularity == "row":
            row_total_capacity = sum(capacities)
        for threshold in detection_thresholds:
            for fraction in budget_fractions:
                budget = fraction * total_gates

                uniform = uniform_probabilities(units, budget)
                certified, witness, residual = certified_adversary_bits(
                    units,
                    uniform,
                    detection_threshold=threshold,
                    frontier_bits=frontier,
                )
                outcomes.append(
                    StrategyOutcome(
                        strategy="uniform-independent",
                        granularity=granularity,
                        detection_threshold=threshold,
                        budget_fraction=fraction,
                        expected_checked_gates=expected_checked_gates(units, uniform),
                        certified_upper_bits=certified,
                        greedy_attack_witness_bits=witness,
                        residual_single_cut_bits=residual,
                        protected_unit_count=0,
                        unit_count=len(units),
                    )
                )

                protected, protected_count = greedy_protection_probabilities(
                    units,
                    budget,
                    detection_threshold=threshold,
                    frontier_bits=frontier,
                )
                certified, witness, residual = certified_adversary_bits(
                    units,
                    protected,
                    detection_threshold=threshold,
                    frontier_bits=frontier,
                )
                outcomes.append(
                    StrategyOutcome(
                        strategy="greedy-protection",
                        granularity=granularity,
                        detection_threshold=threshold,
                        budget_fraction=fraction,
                        expected_checked_gates=expected_checked_gates(units, protected),
                        certified_upper_bits=certified,
                        greedy_attack_witness_bits=witness,
                        residual_single_cut_bits=residual,
                        protected_unit_count=protected_count,
                        unit_count=len(units),
                    )
                )

                if granularity == "row":
                    lp_outcome = coarse_lp_outcome(
                        units,
                        budget_gates=budget,
                        detection_threshold=threshold,
                        frontier_bits=frontier,
                    )
                    if lp_outcome is not None:
                        lp_bits, lp_cost = lp_outcome
                        outcomes.append(
                            StrategyOutcome(
                                strategy="coarse-correlated-lp",
                                granularity=granularity,
                                detection_threshold=threshold,
                                budget_fraction=fraction,
                                expected_checked_gates=lp_cost,
                                certified_upper_bits=lp_bits,
                                greedy_attack_witness_bits=lp_bits,
                                residual_single_cut_bits=0.0,
                                protected_unit_count=0,
                                unit_count=len(units),
                            )
                        )

    phase_rows = tile_phase_table(
        total_capacity_bits=row_total_capacity or 1.0,
        total_gates=total_gates,
        frontier_bits=frontier,
        detection_thresholds=detection_thresholds,
        sampling_probabilities=(0.5, 0.9, 0.99, 0.999),
        tile_bit_sizes=(16.0, 160.0, 1600.0),
    )
    return SamplingStudyReport(
        model_id=indexed.config.model_id,
        prompt_tokens=indexed.prompt_tokens,
        generated_tokens=indexed.generated_tokens,
        total_checked_gate_count=total_gates,
        output_frontier_bits=frontier,
        capacity_bound_kind="union-of-certified-canonical-cuts",
        granularity_summaries=tuple(summaries),
        outcomes=tuple(outcomes),
        tile_phase_table=phase_rows,
        elapsed_seconds=time.perf_counter() - started,
        notes=(
            "units aggregate the certified lifted canonical partition; "
            "unit capacity is the union-of-canonical-cuts bound",
        ),
    )


def write_gpt2_sampling_study(
    study: GPT2SamplingStudy,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write the study as JSON and Markdown; return both paths."""

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "gpt2_sampling_study.json"
    json_path.write_text(json.dumps(asdict(study), indent=2) + "\n")

    lines = [
        "# GPT-2 certified verification-sampling study",
        "",
        f"Model `{study.model_id}`, {study.prompt_tokens} prompt tokens, "
        f"{study.generated_tokens} generated tokens, "
        f"{study.total_checked_gate_count:,} checkable gates.",
        "",
        "Every bound is a certified *upper* limit on what any attack the "
        "strategy fails to detect (at its threshold) can structurally reach. "
        f"The designated-output frontier caps all attacks at "
        f"{study.output_frontier_bits:,.1f} bits.",
        "",
        "## The phase transition",
        "",
        "At every granularity studied, all but a few output-adjacent units "
        "(the terminal token rows, whose capacities are inherently at most "
        "the frontier) individually exceed the output frontier. Leaving any "
        "such unit below the detection threshold saturates the frontier "
        "bound, so certifying anything below the frontier requires detection "
        "probability at least the threshold on essentially every unit, which "
        "costs at least `threshold x total gates` in expectation for any "
        "strategy. The strategy tables below show exactly this step: the "
        "certified bound stays at the frontier until the budget fraction "
        "reaches the threshold, then drops to zero. Below the transition the "
        "correlated LP correctly spends nothing, because no intermediate "
        "bound is certifiable; at the transition it certifies zero bits at "
        "expected cost `threshold x total`, cheaper than uniform sampling.",
        "",
        "| Granularity | Units | Units > frontier | "
        "Min unit capacity (bits) | Max unit capacity (bits) | "
        "Total capacity (bits) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in study.granularity_summaries:
        lines.append(
            f"| {summary.granularity} | {summary.unit_count} | "
            f"{summary.units_above_frontier} | "
            f"{summary.min_unit_capacity_bits:,.0f} | "
            f"{summary.max_unit_capacity_bits:,.3g} | "
            f"{summary.total_capacity_bits:,.3g} |"
        )
    lines.extend(
        [
            "",
            "## Strategy comparison",
            "",
            "| Granularity | Units | Threshold | Budget | Strategy | "
            "Protected | Expected gates | Certified bound (bits) | "
            "Greedy witness (bits) |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for outcome in study.outcomes:
        lines.append(
            f"| {outcome.granularity} | {outcome.unit_count} | "
            f"{outcome.detection_threshold:.2f} | "
            f"{outcome.budget_fraction:.3g} | {outcome.strategy} | "
            f"{outcome.protected_unit_count} | "
            f"{outcome.expected_checked_gates:,.0f} | "
            f"{outcome.certified_upper_bits:,.1f} | "
            f"{outcome.greedy_attack_witness_bits:,.1f} |"
        )
    lines.extend(
        [
            "",
            "## What finer units would buy",
            "",
            "Intermediate bounds require units whose certified capacity is "
            "*below* the frontier. The exact table below assumes equal tiles "
            "of the stated capacity sampled independently; gates per tile "
            "scale from the row-level total capacity. Even 16-bit tiles "
            "(one FP16 boundary value each) certify below-frontier bounds "
            "only once per-tile sampling probability approaches the "
            "detection threshold.",
            "",
            "| Threshold | Tile sampling q | Tile bits | ~Gates/tile | "
            "Max undetected tiles | Certified (bits) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in study.tile_phase_table:
        lines.append(
            f"| {row.detection_threshold:.2f} | "
            f"{row.sampling_probability:.3f} | {row.tile_bits:,.0f} | "
            f"{row.approx_gates_per_tile:,.0f} | "
            f"{row.max_undetected_tiles:,} | {row.certified_bits:,.1f} |"
        )
    lines.append("")
    markdown_path = output_directory / "gpt2_sampling_study.md"
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path
