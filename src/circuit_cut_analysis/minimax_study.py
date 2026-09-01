"""Certified class-symmetric, exact-budget minimax studies.

This module is deliberately separate from :mod:`sampling_study`.  The older
study compares heuristic strategies over coarse all-or-nothing verification
units under expected budgets.  Here scalar gates are sampled individually,
classes only tie their treatment, every run checks exactly ``B`` gates, and
the adversary ranges over every class-count attack.

For each budget this module reports:

* a global lower bound that applies to every randomized verifier strategy;
* a replay-certified upper bound from one deterministic class quota; and
* a global minimax certificate when the two bounds coincide.

When they do not coincide, the interval is honest: quota randomization or a
stronger lower certificate may close it, and no optimality claim is made.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.weighted_sampling import (
    FixedQuotaOptimization,
    PureClassMomentLowerBound,
    UniversalMinimaxLowerBound,
    WeightedGateClass,
    WeightedGateClassPartition,
    coalesce_frontier_equivalent_classes,
    optimize_fixed_quota_strategy,
    pure_class_moment_lower_bound,
    universal_minimax_lower_bound,
)

DEFAULT_MINIMAX_BUDGET_FRACTIONS = (
    Fraction(1, 1_000),
    Fraction(1, 100),
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(1, 2),
    Fraction(9, 10),
    Fraction(99, 100),
)


@dataclass(frozen=True, slots=True)
class MinimaxBudgetResult:
    """One exact-budget minimax interval and its witnesses."""

    budget_fraction: Fraction
    realized_budget_fraction: Fraction
    budget_source: str
    checked_gate_budget: int
    detection_threshold: Fraction
    global_lower_bound: LogCardinality
    certified_upper_bound: LogCardinality
    lower_bound_method: str
    universal_pool_bound: UniversalMinimaxLowerBound
    pure_class_moment_bound: PureClassMomentLowerBound
    quota_optimization: FixedQuotaOptimization

    @property
    def globally_solved(self) -> bool:
        return self.global_lower_bound == self.certified_upper_bound

    @property
    def gap_bits(self) -> float:
        return float(self.certified_upper_bound.width_bits) - float(
            self.global_lower_bound.width_bits
        )


@dataclass(frozen=True, slots=True)
class MinimaxStudyReport:
    """Results for one probability-class partition."""

    model_id: str
    total_gate_count: int
    original_class_count: int
    analyzed_class_count: int
    analyzed_classes: tuple[WeightedGateClass, ...]
    output_frontier: LogCardinality
    partition_certificate: str
    detection_threshold: Fraction
    results: tuple[MinimaxBudgetResult, ...]
    elapsed_seconds: float
    run_configuration_json: str
    configuration_fingerprint: str
    implementation_fingerprint: str
    max_quota_evaluations: int
    assumptions: tuple[str, ...] = ()

    @property
    def solved_budget_count(self) -> int:
        return sum(result.globally_solved for result in self.results)


@dataclass(frozen=True, slots=True)
class MinimaxReportPaths:
    json: Path
    csv: Path
    markdown: Path


def _as_probability(value: int | float | Fraction) -> Fraction:
    parsed = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    if not 0 < parsed <= 1:
        raise ValueError("probability must lie in (0, 1]")
    return parsed


def _implementation_fingerprint() -> str:
    """Hash package source so reports identify their generating implementation."""

    package_root = Path(__file__).parent
    digest = hashlib.sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        digest.update(str(source_path.relative_to(package_root)).encode())
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_run_configuration(
    configuration: Mapping[str, object] | None,
    *,
    budget_specs: Sequence[tuple[Fraction, int, str]],
    detection_threshold: Fraction,
    coalesce_classes: bool,
    max_quota_evaluations: int,
) -> str:
    payload = dict(configuration or {})
    payload.update(
        {
            "budgets": [
                {
                    "requested_fraction": (
                        f"{fraction.numerator}/{fraction.denominator}"
                    ),
                    "checked_gates": checked,
                    "source": source,
                }
                for fraction, checked, source in budget_specs
            ],
            "detection_threshold": (
                f"{detection_threshold.numerator}/{detection_threshold.denominator}"
            ),
            "coalescing": ("exact-frontier-equivalent" if coalesce_classes else "none"),
            "max_quota_evaluations": max_quota_evaluations,
        }
    )
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except TypeError as error:
        raise ValueError(
            "run configuration must contain JSON-serializable values"
        ) from error


def run_weighted_minimax_study(
    partition: WeightedGateClassPartition,
    *,
    budget_fractions: Iterable[int | float | Fraction] | None = None,
    checked_gate_budgets: Iterable[int] | None = None,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
    coalesce_classes: bool = True,
    max_quota_evaluations: int = 6,
    assumptions: Iterable[str] = (),
    run_configuration: Mapping[str, object] | None = None,
) -> MinimaxStudyReport:
    """Compute globally bracketed minimax values across exact gate budgets."""

    started = time.perf_counter()
    threshold = _as_probability(detection_threshold)
    if budget_fractions is not None and checked_gate_budgets is not None:
        raise ValueError("provide budget fractions or checked-gate budgets, not both")
    if checked_gate_budgets is not None:
        exact_budgets = tuple(checked_gate_budgets)
        if any(
            not 0 <= budget <= partition.total_gate_count for budget in exact_budgets
        ):
            raise ValueError("checked-gate budget is outside partition")
        budget_specs = tuple(
            (
                Fraction(budget, partition.total_gate_count),
                budget,
                "exact-checked-gates",
            )
            for budget in exact_budgets
        )
    else:
        fractions = tuple(
            _as_probability(value)
            for value in (
                DEFAULT_MINIMAX_BUDGET_FRACTIONS
                if budget_fractions is None
                else budget_fractions
            )
        )
        budget_specs = tuple(
            (
                fraction,
                (fraction * partition.total_gate_count).numerator
                // (fraction * partition.total_gate_count).denominator,
                "floor-requested-fraction",
            )
            for fraction in fractions
        )
    configuration_json = _canonical_run_configuration(
        run_configuration,
        budget_specs=budget_specs,
        detection_threshold=threshold,
        coalesce_classes=coalesce_classes,
        max_quota_evaluations=max_quota_evaluations,
    )
    upper_partition = (
        coalesce_frontier_equivalent_classes(partition)
        if coalesce_classes
        else partition
    )
    if upper_partition.total_gate_count != partition.total_gate_count:
        raise AssertionError("class coalescing must preserve every scalar gate")
    results: list[MinimaxBudgetResult] = []
    for fraction, checked_gate_budget, budget_source in budget_specs:
        pool_bound = universal_minimax_lower_bound(
            upper_partition,
            checked_gate_budget=checked_gate_budget,
            detection_threshold=threshold,
        )
        moment_bound = pure_class_moment_lower_bound(
            upper_partition,
            checked_gate_budget=checked_gate_budget,
            detection_threshold=threshold,
        )
        if moment_bound.capacity_lower_bound >= pool_bound.capacity_lower_bound:
            lower_bound = moment_bound.capacity_lower_bound
            lower_method = "pure-class-moment"
        else:
            lower_bound = pool_bound.capacity_lower_bound
            lower_method = "uniform-pool-averaging"
        quota = optimize_fixed_quota_strategy(
            upper_partition,
            checked_gate_budget=checked_gate_budget,
            detection_threshold=threshold,
            max_evaluations=max_quota_evaluations,
        )
        if lower_bound > quota.certified_upper_bound:
            raise AssertionError(
                "global lower bound exceeds replay-certified strategy upper bound"
            )
        results.append(
            MinimaxBudgetResult(
                budget_fraction=fraction,
                realized_budget_fraction=Fraction(
                    checked_gate_budget,
                    partition.total_gate_count,
                ),
                budget_source=budget_source,
                checked_gate_budget=checked_gate_budget,
                detection_threshold=threshold,
                global_lower_bound=lower_bound,
                certified_upper_bound=quota.certified_upper_bound,
                lower_bound_method=lower_method,
                universal_pool_bound=pool_bound,
                pure_class_moment_bound=moment_bound,
                quota_optimization=quota,
            )
        )
    return MinimaxStudyReport(
        model_id=partition.model_id,
        total_gate_count=partition.total_gate_count,
        original_class_count=len(partition.classes),
        analyzed_class_count=len(upper_partition.classes),
        analyzed_classes=upper_partition.classes,
        output_frontier=partition.output_frontier,
        partition_certificate=(
            partition.certificate
            + (
                "; exactly coalesced equal-width classes with no sub-frontier local cap"
                if coalesce_classes
                else ""
            )
        ),
        detection_threshold=threshold,
        results=tuple(results),
        elapsed_seconds=time.perf_counter() - started,
        run_configuration_json=configuration_json,
        configuration_fingerprint=hashlib.sha256(
            configuration_json.encode()
        ).hexdigest(),
        implementation_fingerprint=_implementation_fingerprint(),
        max_quota_evaluations=max_quota_evaluations,
        assumptions=tuple(assumptions),
    )


def _capacity_payload(capacity: LogCardinality) -> dict[str, object]:
    return {
        "bits": float(capacity.width_bits),
        "exact_expression": capacity.expression,
        "multiplier_numerator": capacity.multiplier.numerator,
        "multiplier_denominator": capacity.multiplier.denominator,
    }


def _fraction_payload(value: Fraction) -> dict[str, object]:
    return {
        "value": float(value),
        "exact": f"{value.numerator}/{value.denominator}",
    }


def _result_payload(result: MinimaxBudgetResult) -> dict[str, object]:
    attack = result.quota_optimization.worst_attack
    return {
        "requested_budget_fraction": _fraction_payload(result.budget_fraction),
        "realized_budget_fraction": _fraction_payload(result.realized_budget_fraction),
        "budget_source": result.budget_source,
        "checked_gate_budget": result.checked_gate_budget,
        "detection_threshold": _fraction_payload(result.detection_threshold),
        "global_lower_bound": _capacity_payload(result.global_lower_bound),
        "certified_upper_bound": _capacity_payload(result.certified_upper_bound),
        "gap_bits": result.gap_bits,
        "globally_solved": result.globally_solved,
        "lower_bound_method": result.lower_bound_method,
        "universal_pool_certificate": {
            "capacity": _capacity_payload(
                result.universal_pool_bound.capacity_lower_bound
            ),
            "attack_size": result.universal_pool_bound.attack_size,
            "pool_gate_count": result.universal_pool_bound.pool_gate_count,
            "pool_class_ids": result.universal_pool_bound.pool_class_ids,
            "miss_probability": _fraction_payload(
                result.universal_pool_bound.miss_probability
            ),
        },
        "pure_class_moment_certificate": {
            "capacity": _capacity_payload(
                result.pure_class_moment_bound.capacity_lower_bound
            ),
            "required_expected_checks": _fraction_payload(
                result.pure_class_moment_bound.required_expected_checks
            ),
            "budget_shortfall": _fraction_payload(
                result.pure_class_moment_bound.budget_shortfall
            ),
            "requirements": [
                {
                    "class_id": requirement.class_id,
                    "attack_size": requirement.attack_size,
                    "gate_count": requirement.gate_count,
                    "required_expected_checks": _fraction_payload(
                        requirement.required_expected_checks
                    ),
                }
                for requirement in result.pure_class_moment_bound.requirements
            ],
        },
        "upper_bound_strategy": {
            "kind": "deterministic-class-quota",
            "class_ids": result.quota_optimization.strategy.class_ids,
            "class_sizes": result.quota_optimization.strategy.class_sizes,
            "checked_counts": result.quota_optimization.strategy.checked_counts,
            "inclusion_probabilities": [
                _fraction_payload(probability)
                for probability in (
                    result.quota_optimization.strategy.inclusion_probabilities
                )
            ],
            "evaluated_quota_count": (result.quota_optimization.evaluated_quota_count),
            "globally_optimal_from_search_alone": (
                result.quota_optimization.globally_optimal
            ),
        },
        "upper_bound_worst_attack": (
            None
            if attack is None
            else {
                "attacked_counts": attack.attacked_counts,
                "detection_probability": _fraction_payload(
                    attack.detection_probability
                ),
                "capacity": _capacity_payload(attack.capacity_upper_bound),
            }
        ),
    }


def _report_payload(report: MinimaxStudyReport) -> dict[str, object]:
    return {
        "schema": "class-symmetric-exact-budget-minimax-v2",
        "model_id": report.model_id,
        "run_configuration": json.loads(report.run_configuration_json),
        "configuration_fingerprint": report.configuration_fingerprint,
        "implementation_fingerprint": report.implementation_fingerprint,
        "max_quota_evaluations": report.max_quota_evaluations,
        "total_gate_count": report.total_gate_count,
        "original_class_count": report.original_class_count,
        "analyzed_class_count": report.analyzed_class_count,
        "output_frontier": _capacity_payload(report.output_frontier),
        "partition_certificate": report.partition_certificate,
        "analyzed_class_catalog": [
            {
                "id": gate_class.id,
                "gate_count": gate_class.gate_count,
                "singleton_capacity": _capacity_payload(
                    gate_class.effective_singleton_capacity
                ),
                "aggregate_capacity": _capacity_payload(gate_class.full_class_capacity),
                "source_class_ids": (gate_class.source_class_ids or (gate_class.id,)),
                "description": gate_class.description,
                "selection_predicate": (
                    "uniformly sample scalar gates whose original probability "
                    "class is in source_class_ids"
                ),
            }
            for gate_class in report.analyzed_classes
        ],
        "detection_threshold": _fraction_payload(report.detection_threshold),
        "solved_budget_count": report.solved_budget_count,
        "budget_count": len(report.results),
        "assumptions": report.assumptions,
        "results": [_result_payload(result) for result in report.results],
    }


def write_minimax_study_report(
    report: MinimaxStudyReport,
    output_directory: Path,
    *,
    stem: str | None = None,
) -> MinimaxReportPaths:
    """Write JSON, CSV, and Markdown minimax reports."""

    output_directory.mkdir(parents=True, exist_ok=True)
    if stem is None:
        configuration = json.loads(report.run_configuration_json)
        prompt = configuration.get("prompt_tokens")
        generated = configuration.get("generated_tokens")
        schedule = (
            f"_p{prompt}_g{generated}"
            if isinstance(prompt, int) and isinstance(generated, int)
            else ""
        )
        resolved_stem = (
            f"{report.model_id.replace('-', '_')}{schedule}_"
            f"{report.configuration_fingerprint[:12]}_minimax_study"
        )
    else:
        resolved_stem = stem
    json_path = output_directory / f"{resolved_stem}.json"
    csv_path = output_directory / f"{resolved_stem}.csv"
    markdown_path = output_directory / f"{resolved_stem}.md"

    json_path.write_text(json.dumps(_report_payload(report), indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "model_id",
                "requested_budget_fraction",
                "realized_budget_fraction",
                "budget_source",
                "checked_gate_budget",
                "detection_threshold",
                "global_lower_bits",
                "certified_upper_bits",
                "gap_bits",
                "globally_solved",
                "lower_bound_method",
            )
        )
        for result in report.results:
            writer.writerow(
                (
                    report.model_id,
                    float(result.budget_fraction),
                    float(result.realized_budget_fraction),
                    result.budget_source,
                    result.checked_gate_budget,
                    float(result.detection_threshold),
                    float(result.global_lower_bound.width_bits),
                    float(result.certified_upper_bound.width_bits),
                    result.gap_bits,
                    result.globally_solved,
                    result.lower_bound_method,
                )
            )

    lines = [
        f"# {report.model_id}: scalar-gate minimax study",
        "",
        "This is the class-symmetric, exact-count game—not the coarse-unit",
        "expected-budget baseline. Each run checks exactly B scalar gates",
        "uniformly without replacement inside each probability class. Every",
        "computed gate represented by the partition is attackable.",
        "",
        f"- Gates: {report.total_gate_count:,}",
        f"- Original classes: {report.original_class_count}",
        f"- Exact minimax-reduction classes: {report.analyzed_class_count} after "
        "frontier-equivalent coalescing",
        f"- Detection threshold: {float(report.detection_threshold):.2%}",
        f"- Output frontier: {float(report.output_frontier.width_bits):,.6g} bits",
        f"- Globally solved budgets: {report.solved_budget_count}/"
        f"{len(report.results)}",
        f"- Configuration fingerprint: `{report.configuration_fingerprint}`",
        f"- Implementation fingerprint: `{report.implementation_fingerprint}`",
        f"- Quota candidates replayed per budget: at most "
        f"{report.max_quota_evaluations}",
        "",
        "A matching lower and upper bound is a global minimax certificate over",
        "all randomized exact-budget verifier strategies, even though the",
        "displayed achieving strategy is a deterministic quota. A nonzero gap",
        "is unresolved; the upper endpoint remains rigorously quantified over",
        "all class-count attacks, but quota randomization may improve it.",
        "",
        "Requested budget fractions use `B = floor(qN)`; the realized fraction",
        "is reported separately. Runs requested with an exact checked-gate count",
        "have identical requested and realized fractions.",
        "",
        "| Requested | Realized | Exact B | Global lower (bits) | "
        "Certified upper (bits) | Gap | Status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.results:
        status = "globally solved" if result.globally_solved else "bracketed"
        lines.append(
            f"| {float(result.budget_fraction):.3%} | "
            f"{float(result.realized_budget_fraction):.8%} | "
            f"{result.checked_gate_budget:,} | "
            f"{float(result.global_lower_bound.width_bits):,.6g} | "
            f"{float(result.certified_upper_bound.width_bits):,.6g} | "
            f"{result.gap_bits:,.6g} | {status} |"
        )
    lines.extend(
        (
            "",
            "## Exact-reduction class catalog",
            "",
            "| Class | Gates | Singleton bits | Aggregate bits | Source classes |",
            "| --- | ---: | ---: | ---: | --- |",
        )
    )
    for gate_class in report.analyzed_classes:
        source_ids = gate_class.source_class_ids or (gate_class.id,)
        lines.append(
            f"| {gate_class.id} | {gate_class.gate_count:,} | "
            f"{float(gate_class.effective_singleton_capacity.width_bits):,.6g} | "
            f"{float(gate_class.full_class_capacity.width_bits):,.6g} | "
            f"{', '.join(source_ids)} |"
        )
    lines.extend(
        (
            "",
            "## Achieving upper-bound quotas",
            "",
            "Each row is a scalar inclusion probability induced by uniform",
            "sampling without replacement inside the displayed exact-reduction",
            "class. The checked counts in each budget block sum to exact B.",
            "",
            "| Budget | Class | Gates | Checked | Inclusion |",
            "| ---: | --- | ---: | ---: | ---: |",
        )
    )
    for result in report.results:
        strategy = result.quota_optimization.strategy
        for class_id, size, checked, probability in zip(
            strategy.class_ids,
            strategy.class_sizes,
            strategy.checked_counts,
            strategy.inclusion_probabilities,
            strict=True,
        ):
            lines.append(
                f"| {float(result.budget_fraction):.3%} | {class_id} | "
                f"{size:,} | {checked:,} | {float(probability):.8%} |"
            )
    lines.extend(
        (
            "",
            "## Replayed worst attacks",
            "",
            "| Budget | Attacked counts (class order above) | Detection | "
            "Capacity (bits) |",
            "| ---: | --- | ---: | ---: |",
        )
    )
    for result in report.results:
        attack = result.quota_optimization.worst_attack
        if attack is None:
            lines.append(
                f"| {float(result.budget_fraction):.3%} | no positive-capacity "
                "attack | — | 0 |"
            )
        else:
            counts = ", ".join(str(count) for count in attack.attacked_counts)
            lines.append(
                f"| {float(result.budget_fraction):.3%} | {counts} | "
                f"{float(attack.detection_probability):.8%} | "
                f"{float(attack.capacity_upper_bound.width_bits):,.6g} |"
            )
    if report.assumptions:
        lines.extend(("", "## Model assumptions"))
        lines.extend(f"- {assumption}" for assumption in report.assumptions)
    lines.extend(
        (
            "",
            "## Certificate scope",
            "- Lower endpoints are verifier-independent averaging or convex-moment "
            "certificates on an exact symmetry reduction of the original class "
            "partition.",
            "- Upper endpoints replay one exact quota against every attack-count "
            "vector under the same capped-linear structural bound.",
            "- A coalesced quota means sampling uniformly from the corresponding "
            "union of original classes; it is a valid randomized-quota strategy "
            "for the original game.",
            "- The structural bound is treated as the adversary payoff, as specified "
            "for this study; these reports do not claim every real circuit attains it.",
            "- Fixed-quota search is heuristic unless the global endpoints meet.",
            "",
        )
    )
    markdown_path.write_text("\n".join(lines))
    return MinimaxReportPaths(
        json=json_path,
        csv=csv_path,
        markdown=markdown_path,
    )


def write_cross_model_minimax_summary(
    reports: Sequence[MinimaxStudyReport],
    output_directory: Path,
    *,
    stem: str = "cross_model_minimax_summary",
) -> Path:
    """Write a concise comparison of global minimax intervals."""

    if not reports:
        raise ValueError("cross-model summary needs at least one report")
    thresholds = {report.detection_threshold for report in reports}
    if len(thresholds) != 1:
        raise ValueError("cross-model reports must use the same detection threshold")
    threshold = next(iter(thresholds))
    output_directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-model scalar-gate minimax comparison",
        "",
        "All rows use exact checked-gate budgets and a class-symmetric verifier",
        f"at a {float(threshold):.2%} detection threshold.",
        "Solved rows have matching verifier-independent lower and replayed upper",
        "bounds; bracketed rows make no global-optimality claim.",
        "",
        "GPT-2 uses lifted canonical wiring cuts. Kimi-K3, DeepSeek-V4-Pro,",
        "and Inkling use the looser self-cut envelope, so cross-model differences",
        "also reflect certificate strength and should not be read as intrinsic",
        "model-security rankings.",
        "",
        "| Model | P/G | Certificate | Frontier | Requested | Realized | "
        "Exact B | Lower | Upper | Gap | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in reports:
        configuration = json.loads(report.run_configuration_json)
        certificate_kind = str(
            configuration.get("certificate_kind", "declared structural bound")
        )
        schedule = (
            f"{configuration.get('prompt_tokens', '?')}/"
            f"{configuration.get('generated_tokens', '?')}"
        )
        for result in report.results:
            status = "solved" if result.globally_solved else "bracketed"
            lines.append(
                f"| {report.model_id} | "
                f"{schedule} | "
                f"{certificate_kind} | "
                f"{float(report.output_frontier.width_bits):,.6g} | "
                f"{float(result.budget_fraction):.3%} | "
                f"{float(result.realized_budget_fraction):.8%} | "
                f"{result.checked_gate_budget:,} | "
                f"{float(result.global_lower_bound.width_bits):,.6g} | "
                f"{float(result.certified_upper_bound.width_bits):,.6g} | "
                f"{result.gap_bits:,.6g} | {status} |"
            )
    lines.append("")
    path = output_directory / f"{stem}.md"
    path.write_text("\n".join(lines))
    return path
