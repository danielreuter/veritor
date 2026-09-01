from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.minimax_study import (
    run_weighted_minimax_study,
    write_cross_model_minimax_summary,
    write_minimax_study_report,
)
from circuit_cut_analysis.weighted_sampling import (
    WeightedGateClass,
    WeightedGateClassPartition,
    coalesce_frontier_equivalent_classes,
    pure_class_moment_lower_bound,
)


def _partition() -> WeightedGateClassPartition:
    return WeightedGateClassPartition(
        model_id="tiny",
        classes=(
            WeightedGateClass("left", 60, LogCardinality.bits(2)),
            WeightedGateClass("right", 40, LogCardinality.bits(2)),
        ),
        output_frontier=LogCardinality.bits(20),
        certificate="tiny capped-linear test certificate",
    )


def test_minimax_study_certifies_exact_budget_endpoints() -> None:
    report = run_weighted_minimax_study(
        _partition(),
        budget_fractions=(Fraction(1, 10), Fraction(99, 100)),
        max_quota_evaluations=8,
    )

    low_budget, high_budget = report.results
    assert low_budget.checked_gate_budget == 10
    assert low_budget.global_lower_bound == LogCardinality.bits(20)
    assert low_budget.certified_upper_bound == LogCardinality.bits(20)
    assert low_budget.globally_solved
    assert high_budget.checked_gate_budget == 99
    assert high_budget.global_lower_bound.is_zero
    assert high_budget.certified_upper_bound.is_zero
    assert high_budget.globally_solved
    assert report.solved_budget_count == 2


def test_minimax_study_writes_distinct_machine_readable_reports(
    tmp_path: Path,
) -> None:
    report = run_weighted_minimax_study(
        _partition(),
        budget_fractions=(Fraction(1, 10),),
        max_quota_evaluations=8,
    )
    paths = write_minimax_study_report(report, tmp_path)
    summary = write_cross_model_minimax_summary((report,), tmp_path)

    payload = json.loads(paths.json.read_text())
    assert payload["schema"] == "class-symmetric-exact-budget-minimax-v2"
    assert payload["configuration_fingerprint"]
    assert payload["implementation_fingerprint"]
    assert "elapsed_seconds" not in payload
    assert payload["analyzed_class_catalog"][0]["source_class_ids"]
    assert payload["results"][0]["globally_solved"]
    assert payload["results"][0]["upper_bound_strategy"]["class_sizes"]
    assert (
        payload["results"][0]["upper_bound_strategy"]["kind"]
        == "deterministic-class-quota"
    )
    with paths.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["globally_solved"] == "True"
    markdown = paths.markdown.read_text()
    assert "not the coarse-unit" in markdown
    assert "global minimax certificate" in markdown
    assert "Achieving upper-bound quotas" in markdown
    assert "Exact-reduction class catalog" in markdown
    assert "Replayed worst attacks" in markdown
    summary_text = summary.read_text()
    assert "Cross-model scalar-gate minimax" in summary_text
    assert "certificate strength" in summary_text


def test_lower_certificate_uses_exact_frontier_symmetry_reduction() -> None:
    partition = WeightedGateClassPartition(
        model_id="capped",
        classes=(
            WeightedGateClass(
                "shared-funnel",
                100,
                LogCardinality.bits(4),
                aggregate_capacity=LogCardinality.bits(4),
            ),
            WeightedGateClass("linear", 100, LogCardinality.bits(4)),
            WeightedGateClass("linear-two", 100, LogCardinality.bits(4)),
        ),
        output_frontier=LogCardinality.bits(40),
        certificate="unequal aggregate caps",
    )
    budget = 30
    reduced = coalesce_frontier_equivalent_classes(partition)
    expected = pure_class_moment_lower_bound(
        reduced,
        checked_gate_budget=budget,
    )
    report = run_weighted_minimax_study(
        partition,
        budget_fractions=(Fraction(1, 10),),
        max_quota_evaluations=8,
    )

    assert report.original_class_count == 3
    assert report.analyzed_class_count == 2
    assert report.results[0].checked_gate_budget == budget
    assert report.results[0].pure_class_moment_bound == expected


def test_fraction_budget_reports_floor_and_realized_fraction() -> None:
    partition = WeightedGateClassPartition(
        model_id="three",
        classes=(WeightedGateClass("all", 3, LogCardinality.bits(1)),),
        output_frontier=LogCardinality.bits(3),
        certificate="three gates",
    )
    fractional = run_weighted_minimax_study(
        partition,
        budget_fractions=(Fraction(1, 2),),
    ).results[0]
    exact = run_weighted_minimax_study(
        partition,
        checked_gate_budgets=(2,),
    ).results[0]

    assert fractional.checked_gate_budget == 1
    assert fractional.budget_fraction == Fraction(1, 2)
    assert fractional.realized_budget_fraction == Fraction(1, 3)
    assert fractional.budget_source == "floor-requested-fraction"
    assert exact.checked_gate_budget == 2
    assert exact.budget_fraction == Fraction(2, 3)
    assert exact.realized_budget_fraction == Fraction(2, 3)
    assert exact.budget_source == "exact-checked-gates"


def test_machine_readable_reports_are_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = run_weighted_minimax_study(
        _partition(),
        budget_fractions=(Fraction(1, 10),),
        run_configuration={"prompt_tokens": 2, "generated_tokens": 3},
    )
    first_paths = write_minimax_study_report(first, tmp_path)
    first_bytes = (
        first_paths.json.read_bytes(),
        first_paths.csv.read_bytes(),
        first_paths.markdown.read_bytes(),
    )
    second = run_weighted_minimax_study(
        _partition(),
        budget_fractions=(Fraction(1, 10),),
        run_configuration={"prompt_tokens": 2, "generated_tokens": 3},
    )
    second_paths = write_minimax_study_report(second, tmp_path)

    assert second.configuration_fingerprint == first.configuration_fingerprint
    assert second.implementation_fingerprint == first.implementation_fingerprint
    assert (
        second_paths.json.read_bytes(),
        second_paths.csv.read_bytes(),
        second_paths.markdown.read_bytes(),
    ) == first_bytes
