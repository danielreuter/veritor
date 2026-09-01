from __future__ import annotations

import math
from pathlib import Path

from circuit_cut_analysis.cross_model_study import run_profile_sampling_study
from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.sampling_study import write_study_report


def toy_profile() -> ModelCapacityProfile:
    return ModelCapacityProfile(
        model_id="toy-transformer",
        prompt_tokens=8,
        generated_tokens=4,
        logical_vocabulary_size=256,
        numerical_profile_id="toy-mixed",
        regions=(
            CapacityRegion("macs", "wide multiply-accumulate", 100_000, 32.0),
            CapacityRegion("norms", "normalization internals", 2_000, 32.0),
            CapacityRegion("boundaries", "activation boundaries", 10_000, 16.0),
            CapacityRegion("argmax", "token selection", 4, 8.0),
        ),
        assumptions=("toy assumption",),
    )


def test_profile_study_runs_and_is_frontier_capped() -> None:
    study = run_profile_sampling_study(
        toy_profile(),
        budget_fractions=(1e-2, 0.5, 0.99),
        detection_thresholds=(0.9,),
    )
    frontier = study.output_frontier_bits
    assert math.isclose(frontier, 32.0)
    assert study.total_checked_gate_count == 112_004
    assert study.granularity_summaries[0].granularity == "region"
    strategies = {outcome.strategy for outcome in study.outcomes}
    assert strategies == {
        "uniform-independent",
        "greedy-protection",
        "coarse-correlated-lp",
    }
    for outcome in study.outcomes:
        assert 0.0 <= outcome.certified_upper_bits <= frontier + 1e-9
        assert outcome.greedy_attack_witness_bits <= frontier + 1e-9


def test_full_threshold_budget_certifies_zero_bits() -> None:
    study = run_profile_sampling_study(
        toy_profile(),
        budget_fractions=(0.95,),
        detection_thresholds=(0.9,),
        tile_frontier_multiples=(),
    )
    greedy = [
        outcome for outcome in study.outcomes if outcome.strategy == "greedy-protection"
    ]
    assert greedy
    # Budget 0.95 x total exceeds threshold 0.9 x total: everything can be
    # protected at the detection threshold, so no sub-threshold attack exists.
    assert all(outcome.certified_upper_bits == 0.0 for outcome in greedy)


def test_certified_bounds_are_monotone_in_budget() -> None:
    study = run_profile_sampling_study(
        toy_profile(),
        budget_fractions=(1e-3, 1e-1, 0.5, 0.9),
        detection_thresholds=(0.5,),
        tile_frontier_multiples=(1.0,),
    )
    by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for outcome in study.outcomes:
        key = (outcome.strategy, outcome.granularity)
        by_key.setdefault(key, []).append(
            (outcome.budget_fraction, outcome.certified_upper_bits)
        )
    for pairs in by_key.values():
        pairs.sort()
        bounds = [bits for _, bits in pairs]
        assert bounds == sorted(bounds, reverse=True)


def test_study_report_writes_notes(tmp_path: Path) -> None:
    study = run_profile_sampling_study(
        toy_profile(),
        budget_fractions=(0.5,),
        detection_thresholds=(0.9,),
        tile_frontier_multiples=(),
    )
    json_path, markdown_path = write_study_report(study, tmp_path, stem="toy_study")
    assert json_path.exists()
    text = markdown_path.read_text()
    assert "toy assumption" in text
    assert "self-cut" in text
