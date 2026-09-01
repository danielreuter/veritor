"""Model-agnostic certified verification-sampling study machinery.

The study operates on *region units*: disjoint groups of computed gates with
an exact gate count and a certified upper bound on the structural capacity an
adversary gains by corrupting every gate in the unit.  Two upper-bound
sources are supported:

* **union-of-canonical-cuts** (tight): available when a certified canonical
  partition exists, as for GPT-2;
* **self-cut** (loose but assumption-free): the corrupted gates themselves
  always form a valid downstream cut, so the sum of their own boundary
  widths, capped by the designated-output frontier, is always sound.

All detection probabilities are exact for the strategy families studied, so
every reported bound is a sound verifier guarantee; optimality is not
claimed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.sampling import (
    SamplingBudgetMode,
    SamplingStrategy,
    _AttackScenario,
    _solve_bound_minimizing_lp,
    _unit_subsets,
)


@dataclass(frozen=True, slots=True)
class RegionUnit:
    """One disjoint sampling unit with a certified capacity upper bound."""

    id: str
    row_ids: tuple[str, ...]
    checked_gate_count: int
    capacity_upper_bits: float
    max_single_cut_bits: float


@dataclass(frozen=True, slots=True)
class StrategyOutcome:
    """One certified evaluation of a sampling strategy on region units."""

    strategy: str
    granularity: str
    detection_threshold: float
    budget_fraction: float
    expected_checked_gates: float
    certified_upper_bits: float
    greedy_attack_witness_bits: float
    residual_single_cut_bits: float
    protected_unit_count: int
    unit_count: int


@dataclass(frozen=True, slots=True)
class GranularitySummary:
    """Unit statistics that explain when the output frontier cap binds."""

    granularity: str
    unit_count: int
    units_above_frontier: int
    min_unit_capacity_bits: float
    max_unit_capacity_bits: float
    total_capacity_bits: float


@dataclass(frozen=True, slots=True)
class TilePhaseRow:
    """Analytic certified bound for uniform tiles of one capacity size.

    For units of certified capacity ``tile_bits`` sampled independently with
    probability ``sampling_probability``, an attack corrupting ``m`` units is
    detected with probability ``1 - (1 - q)^m``; the largest undetected count
    is ``m_max`` and the certified bound is ``min(m_max x tile_bits,
    frontier)``.  This is exact for equal-size tiles and shows what unit
    granularity intermediate bounds require.
    """

    detection_threshold: float
    sampling_probability: float
    tile_bits: float
    approx_gates_per_tile: float
    max_undetected_tiles: int
    certified_bits: float


def certified_adversary_bits(
    units: Sequence[RegionUnit],
    unit_probability: Mapping[str, float],
    *,
    detection_threshold: float,
    frontier_bits: float,
) -> tuple[float, float, float]:
    """Certify the best sub-threshold attack against independent sampling.

    Returns ``(certified_upper, greedy_witness, residual_single_cut)`` bits.
    The certified upper bound relaxes the adversary's detection-budget
    knapsack fractionally, which can only overstate the attack, so the
    verifier-side guarantee remains sound.  The greedy witness is one
    concrete admissible attack; the residual value is the largest certified
    single-cut capacity among units an adversary can still touch.
    """

    if not 0 < detection_threshold < 1:
        raise ValueError("detection threshold must be inside (0, 1)")
    weight_budget = -math.log1p(-detection_threshold)

    weighted: list[tuple[float, float, float]] = []
    free_bits = 0.0
    residual_single = 0.0
    for unit in units:
        probability = unit_probability.get(unit.id, 0.0)
        if not 0 <= probability <= 1:
            raise ValueError(f"unit {unit.id!r} has invalid probability")
        if probability >= detection_threshold:
            continue
        residual_single = max(residual_single, unit.max_single_cut_bits)
        weight = -math.log1p(-probability) if probability > 0 else 0.0
        if weight == 0.0:
            free_bits += unit.capacity_upper_bits
        else:
            weighted.append((unit.capacity_upper_bits, weight, probability))

    weighted.sort(key=lambda item: item[0] / item[1], reverse=True)
    fractional = free_bits
    remaining = weight_budget
    for value, weight, _ in weighted:
        if remaining <= 0:
            break
        take = min(1.0, remaining / weight)
        fractional += take * value
        remaining -= take * weight

    greedy = free_bits
    remaining = weight_budget
    for value, weight, _ in weighted:
        if weight < remaining:
            greedy += value
            remaining -= weight
    greedy = min(greedy, frontier_bits)
    return min(fractional, frontier_bits), greedy, residual_single


def uniform_probabilities(
    units: Sequence[RegionUnit],
    budget_gates: float,
) -> dict[str, float]:
    total = sum(unit.checked_gate_count for unit in units)
    fraction = min(budget_gates / total, 1.0)
    return {unit.id: fraction for unit in units}


def greedy_protection_probabilities(
    units: Sequence[RegionUnit],
    budget_gates: float,
    *,
    detection_threshold: float,
    frontier_bits: float,
) -> tuple[dict[str, float], int]:
    """Protect the densest units at the detection threshold, sweep the count.

    Any attack touching a unit sampled with probability at least the
    threshold is detected at or above the threshold, so protected units are
    unattackable.  The sweep picks the protected-prefix size whose certified
    adversary bound is smallest; leftover budget is spread uniformly across
    the unprotected remainder.
    """

    order = sorted(
        units,
        key=lambda unit: (
            unit.capacity_upper_bits / max(unit.checked_gate_count, 1),
            unit.capacity_upper_bits,
        ),
        reverse=True,
    )
    best: tuple[float, dict[str, float], int] | None = None
    for protected_count in range(len(order) + 1):
        protected = order[:protected_count]
        cost = detection_threshold * sum(unit.checked_gate_count for unit in protected)
        if cost > budget_gates:
            break
        rest = order[protected_count:]
        rest_gates = sum(unit.checked_gate_count for unit in rest)
        leftover_fraction = (
            min((budget_gates - cost) / rest_gates, 1.0) if rest_gates else 0.0
        )
        probabilities = {unit.id: detection_threshold for unit in protected}
        probabilities.update({unit.id: leftover_fraction for unit in rest})
        certified, _, _ = certified_adversary_bits(
            units,
            probabilities,
            detection_threshold=detection_threshold,
            frontier_bits=frontier_bits,
        )
        if best is None or certified < best[0]:
            best = (certified, probabilities, protected_count)
    if best is None:
        raise AssertionError("protecting zero units must always be affordable")
    return best[1], best[2]


def expected_checked_gates(
    units: Sequence[RegionUnit],
    probabilities: Mapping[str, float],
) -> float:
    return sum(
        probabilities.get(unit.id, 0.0) * unit.checked_gate_count for unit in units
    )


def granularity_summary(
    granularity: str,
    units: Sequence[RegionUnit],
    *,
    frontier_bits: float,
) -> GranularitySummary:
    capacities = [unit.capacity_upper_bits for unit in units]
    positive = [bits for bits in capacities if bits > 0]
    return GranularitySummary(
        granularity=granularity,
        unit_count=len(units),
        units_above_frontier=sum(bits > frontier_bits for bits in capacities),
        min_unit_capacity_bits=min(positive) if positive else 0.0,
        max_unit_capacity_bits=max(capacities) if capacities else 0.0,
        total_capacity_bits=sum(capacities),
    )


def tile_phase_table(
    *,
    total_capacity_bits: float,
    total_gates: int,
    frontier_bits: float,
    detection_thresholds: Iterable[float],
    sampling_probabilities: Iterable[float],
    tile_bit_sizes: Iterable[float],
) -> tuple[TilePhaseRow, ...]:
    """Exact certified bounds for hypothetical equal-capacity tile partitions."""

    rows: list[TilePhaseRow] = []
    for threshold in detection_thresholds:
        for probability in sampling_probabilities:
            for tile_bits in tile_bit_sizes:
                if probability >= threshold:
                    max_tiles = 0
                elif probability <= 0:
                    max_tiles = math.ceil(frontier_bits / tile_bits)
                else:
                    # Largest m with 1 - (1 - q)^m < threshold.
                    exact = math.log1p(-threshold) / math.log1p(-probability)
                    max_tiles = math.ceil(exact) - 1
                    if 1 - (1 - probability) ** (max_tiles + 1) < threshold:
                        max_tiles += 1
                certified = min(max_tiles * tile_bits, frontier_bits)
                rows.append(
                    TilePhaseRow(
                        detection_threshold=threshold,
                        sampling_probability=probability,
                        tile_bits=tile_bits,
                        approx_gates_per_tile=(
                            tile_bits / total_capacity_bits * total_gates
                        ),
                        max_undetected_tiles=max_tiles,
                        certified_bits=certified,
                    )
                )
    return tuple(rows)


def coarse_lp_outcome(
    units: Sequence[RegionUnit],
    *,
    budget_gates: float,
    detection_threshold: float,
    frontier_bits: float,
    max_units: int = 12,
) -> tuple[float, float] | None:
    """Solve the correlated-strategy LP over at most ``max_units`` supergroups.

    Units are merged into supergroups by descending capacity density.  The
    scenario capacities round up to whole bits, so the certified bound can
    only be conservative.  Returns ``(certified_bits, expected_gates)`` or
    ``None`` when the LP has no sub-budget feasible point.
    """

    order = sorted(
        units,
        key=lambda unit: unit.capacity_upper_bits / max(unit.checked_gate_count, 1),
        reverse=True,
    )
    groups: list[list[RegionUnit]] = [[] for _ in range(min(max_units, len(order)))]
    for index, unit in enumerate(order):
        groups[index % len(groups)].append(unit)

    group_ids = tuple(f"group-{index}" for index in range(len(groups)))
    group_cost = {
        group_id: sum(unit.checked_gate_count for unit in group)
        for group_id, group in zip(group_ids, groups, strict=True)
    }
    group_bits = {
        group_id: min(
            sum(unit.capacity_upper_bits for unit in group),
            frontier_bits,
        )
        for group_id, group in zip(group_ids, groups, strict=True)
    }

    actions = _unit_subsets(group_ids)
    costs = tuple(
        sum(group_cost[group_id] for group_id in action) for action in actions
    )
    scenarios = tuple(
        _AttackScenario(
            attacked,
            LogCardinality.bits(
                math.ceil(
                    min(
                        sum(group_bits[group_id] for group_id in attacked),
                        frontier_bits,
                    )
                )
            ),
        )
        for attacked in actions
        if attacked
    )

    def strategy_cost(strategy: SamplingStrategy) -> Fraction:
        return sum(
            (
                outcome.probability
                * sum(group_cost[group_id] for group_id in outcome.sampled_units)
                for outcome in strategy.outcomes
            ),
            Fraction(),
        )

    try:
        strategy, bound = _solve_bound_minimizing_lp(
            actions,
            costs,
            scenarios,
            budget=Fraction(str(budget_gates)),
            threshold=Fraction(str(detection_threshold)),
            budget_mode=SamplingBudgetMode.EXPECTED,
            numerical_tolerance=1e-9,
            expected_cost=strategy_cost,
        )
    except RuntimeError:
        return None
    # The frontier is itself a certified cut, so capping stays sound and
    # avoids reporting the LP's whole-bit round-up above the frontier.
    return (
        min(float(bound.width_bits), frontier_bits),
        float(strategy_cost(strategy)),
    )


@dataclass(frozen=True, slots=True)
class SamplingStudyReport:
    """Full study output plus the model-level constants used to certify it."""

    model_id: str
    prompt_tokens: int
    generated_tokens: int
    total_checked_gate_count: int
    output_frontier_bits: float
    capacity_bound_kind: str
    granularity_summaries: tuple[GranularitySummary, ...]
    outcomes: tuple[StrategyOutcome, ...]
    tile_phase_table: tuple[TilePhaseRow, ...]
    elapsed_seconds: float
    notes: tuple[str, ...] = ()


def evaluate_strategies(
    units: Sequence[RegionUnit],
    *,
    granularity: str,
    total_gates: int,
    frontier_bits: float,
    budget_fractions: Iterable[float],
    detection_thresholds: Iterable[float],
) -> tuple[StrategyOutcome, ...]:
    """Run the uniform and greedy-protection families over one unit set."""

    outcomes: list[StrategyOutcome] = []
    for threshold in detection_thresholds:
        for fraction in budget_fractions:
            budget = fraction * total_gates

            uniform = uniform_probabilities(units, budget)
            certified, witness, residual = certified_adversary_bits(
                units,
                uniform,
                detection_threshold=threshold,
                frontier_bits=frontier_bits,
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
                frontier_bits=frontier_bits,
            )
            certified, witness, residual = certified_adversary_bits(
                units,
                protected,
                detection_threshold=threshold,
                frontier_bits=frontier_bits,
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
    return tuple(outcomes)


def write_study_report(
    study: SamplingStudyReport,
    output_directory: Path,
    *,
    stem: str,
) -> tuple[Path, Path]:
    """Write one study as JSON and Markdown; return both paths."""

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{stem}.json"
    json_path.write_text(json.dumps(asdict(study), indent=2) + "\n")

    lines = [
        f"# {study.model_id} certified verification-sampling study",
        "",
        f"Model `{study.model_id}`, {study.prompt_tokens} prompt tokens, "
        f"{study.generated_tokens} generated tokens, "
        f"{study.total_checked_gate_count:,} checkable gates. "
        f"Capacity bounds: {study.capacity_bound_kind}.",
        "",
        "Every bound is a certified *upper* limit on what any attack the "
        "strategy fails to detect (at its threshold) can structurally reach. "
        f"The designated-output frontier caps all attacks at "
        f"{study.output_frontier_bits:,.1f} bits.",
        "",
    ]
    if study.notes:
        lines.append("Assumptions and semantics:")
        lines.extend(f"- {note}" for note in study.notes)
        lines.append("")
    lines += [
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
    markdown_path = output_directory / f"{stem}.md"
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path
