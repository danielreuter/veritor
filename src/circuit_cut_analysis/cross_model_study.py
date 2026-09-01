"""Certified verification-sampling studies over model capacity profiles.

This is the assumption-light path used for models without an exact indexed
circuit (Kimi-K3, DeepSeek-V4-Pro, Inkling).  Units come from a
:class:`~circuit_cut_analysis.models.capacity_profile.ModelCapacityProfile`
at two granularities:

* **region**: one unit per architectural region, self-cut bound;
* **region tiles**: regions split into near-equal gate tiles, which keeps
  the self-cut bounds exact because they are linear in the gates a unit
  contains.

The same strategy families as the GPT-2 study run on top (uniform
independent, greedy protection, coarse correlated LP), so reports are
directly comparable across models.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from circuit_cut_analysis.models.capacity_profile import (
    ModelCapacityProfile,
    tiled_region_units,
)
from circuit_cut_analysis.sampling_study import (
    GranularitySummary,
    SamplingStudyReport,
    StrategyOutcome,
    coarse_lp_outcome,
    evaluate_strategies,
    granularity_summary,
    tile_phase_table,
)


def run_profile_sampling_study(
    profile: ModelCapacityProfile,
    *,
    budget_fractions: Iterable[float] = (1e-3, 1e-1, 0.5, 0.9, 0.99),
    detection_thresholds: Iterable[float] = (0.5, 0.9, 0.99),
    tile_frontier_multiples: Iterable[float] = (),
    max_tiled_units: int = 4096,
    lp_max_units: int = 8,
) -> SamplingStudyReport:
    """Run the certified strategy comparison for one capacity profile.

    ``tile_frontier_multiples`` selects extra tiled granularities whose
    target unit capacity is that multiple of the output frontier (subject
    to the ``max_tiled_units`` cap on unit count).  At frontier-scale
    targets the cap forces tiles far above the frontier for trillion-gate
    models, adding nothing over region granularity, so no tiled
    granularity runs by default; the analytic tile phase table covers the
    sub-frontier hypotheticals instead.
    """

    started = time.perf_counter()
    frontier = profile.output_frontier_bits
    total_gates = profile.total_gate_count
    budgets = tuple(budget_fractions)
    thresholds = tuple(detection_thresholds)

    unit_sets = {"region": profile.region_units()}
    for multiple in tile_frontier_multiples:
        label = f"region-tiled-{multiple:g}x-frontier"
        unit_sets[label] = tiled_region_units(
            profile,
            target_bits=multiple * frontier,
            max_units=max_tiled_units,
        )

    summaries: list[GranularitySummary] = []
    outcomes: list[StrategyOutcome] = []
    for granularity, units in unit_sets.items():
        summaries.append(
            granularity_summary(granularity, units, frontier_bits=frontier)
        )
        outcomes.extend(
            evaluate_strategies(
                units,
                granularity=granularity,
                total_gates=total_gates,
                frontier_bits=frontier,
                budget_fractions=budgets,
                detection_thresholds=thresholds,
            )
        )

    region_units = unit_sets["region"]
    for threshold in thresholds:
        for fraction in budgets:
            lp = coarse_lp_outcome(
                region_units,
                budget_gates=fraction * total_gates,
                detection_threshold=threshold,
                frontier_bits=frontier,
                max_units=lp_max_units,
            )
            if lp is None:
                continue
            lp_bits, lp_cost = lp
            outcomes.append(
                StrategyOutcome(
                    strategy="coarse-correlated-lp",
                    granularity="region",
                    detection_threshold=threshold,
                    budget_fraction=fraction,
                    expected_checked_gates=lp_cost,
                    certified_upper_bits=lp_bits,
                    greedy_attack_witness_bits=lp_bits,
                    residual_single_cut_bits=0.0,
                    protected_unit_count=0,
                    unit_count=len(region_units),
                )
            )

    total_capacity = sum(unit.capacity_upper_bits for unit in unit_sets["region"])
    max_width = max(
        (region.self_cut_bits_per_gate for region in profile.regions), default=1.0
    )
    phase_rows = tile_phase_table(
        total_capacity_bits=total_capacity or 1.0,
        total_gates=total_gates,
        frontier_bits=frontier,
        detection_thresholds=thresholds,
        sampling_probabilities=(0.5, 0.9, 0.99, 0.999),
        tile_bit_sizes=(max_width, frontier / 10.0, frontier),
    )
    return SamplingStudyReport(
        model_id=profile.model_id,
        prompt_tokens=profile.prompt_tokens,
        generated_tokens=profile.generated_tokens,
        total_checked_gate_count=total_gates,
        output_frontier_bits=frontier,
        capacity_bound_kind=(
            f"self-cut (numerical profile {profile.numerical_profile_id})"
        ),
        granularity_summaries=tuple(summaries),
        outcomes=tuple(outcomes),
        tile_phase_table=phase_rows,
        elapsed_seconds=time.perf_counter() - started,
        notes=profile.assumptions,
    )


def write_cross_model_summary(
    studies: Sequence[SamplingStudyReport],
    output_directory: Path,
    *,
    stem: str = "cross_model_sampling_summary",
) -> Path:
    """Write one Markdown comparison across per-model study reports.

    For every (threshold, budget) pair present in a study, the table shows
    the best certified bound over all strategies and granularities in that
    study, so models with different unit structures stay comparable.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-model certified verification-sampling comparison",
        "",
        "Game: the circuit, input, and sampling strategy are fixed and known",
        "to the adversary.  The verifier re-checks each verification unit",
        "with the strategy's probability under an expected checked-gate",
        "budget (a fraction of all computed gates).  The adversary corrupts",
        "any gate set whose detection probability stays below the threshold",
        "and is credited the structural reachable-output bound for that set,",
        "assumed attained.  Reported bits are certified upper limits on any",
        "such sub-threshold attack.",
        "",
        "| Model | Checkable gates | Frontier (bits) | Capacity bound |",
        "| --- | ---: | ---: | --- |",
    ]
    for study in studies:
        lines.append(
            f"| {study.model_id} | {study.total_checked_gate_count:,} | "
            f"{study.output_frontier_bits:,.1f} | {study.capacity_bound_kind} |"
        )
    lines.extend(
        [
            "",
            "Reading the table below: certified bounds are all-or-nothing at",
            "the studied granularities.  In every model, units whose",
            "certified capacity exceeds the output frontier carry almost all",
            "of the gate mass, and a single undetected such unit reaches the",
            "frontier cap.  The certified bound therefore stays at the",
            "frontier while any of them is sampled below the detection",
            "threshold, which persists until the expected checked-gate",
            "budget approaches threshold x total gates; there it drops to",
            "zero.  Intermediate guarantees require every attackable unit to",
            "certify below the frontier; each per-model report's tile phase",
            "table quantifies the tile size and per-tile sampling rate that",
            "would deliver them.",
        ]
    )
    for study in studies:
        finest = min(
            (
                summary.min_unit_capacity_bits
                for summary in study.granularity_summaries
                if summary.min_unit_capacity_bits > 0
            ),
            default=0.0,
        )
        lines.append(
            f"- {study.model_id}: smallest positive-capacity unit certifies "
            f"{finest:,.1f} bits against a {study.output_frontier_bits:,.1f}-bit "
            "frontier."
        )
    lines.extend(
        [
            "",
            "Best certified adversary bound (bits) over every strategy and",
            "granularity studied, per detection threshold and budget fraction.",
            "",
            "| Model | Threshold | Budget | Best certified (bits) | "
            "Fraction of frontier | Best strategy |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for study in studies:
        pairs = sorted(
            {
                (outcome.detection_threshold, outcome.budget_fraction)
                for outcome in study.outcomes
            }
        )
        for threshold, fraction in pairs:
            candidates = [
                outcome
                for outcome in study.outcomes
                if outcome.detection_threshold == threshold
                and outcome.budget_fraction == fraction
            ]
            best = min(candidates, key=lambda outcome: outcome.certified_upper_bits)
            share = (
                best.certified_upper_bits / study.output_frontier_bits
                if study.output_frontier_bits
                else 0.0
            )
            lines.append(
                f"| {study.model_id} | {threshold:.2f} | {fraction:.3g} | "
                f"{best.certified_upper_bits:,.1f} | {share:.3f} | "
                f"{best.strategy} ({best.granularity}) |"
            )
    lines.append("")
    path = output_directory / f"{stem}.md"
    path.write_text("\n".join(lines))
    return path
