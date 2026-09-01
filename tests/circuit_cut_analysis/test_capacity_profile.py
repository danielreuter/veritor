from __future__ import annotations

import math

import pytest

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
    tiled_region_units,
)


def make_profile(regions: tuple[CapacityRegion, ...]) -> ModelCapacityProfile:
    return ModelCapacityProfile(
        model_id="toy",
        prompt_tokens=10,
        generated_tokens=5,
        logical_vocabulary_size=1024,
        numerical_profile_id="toy-profile",
        regions=regions,
        assumptions=("toy",),
    )


def test_profile_totals_and_frontier() -> None:
    profile = make_profile(
        (
            CapacityRegion("macs", "multiply-accumulate", 1_000, 32.0),
            CapacityRegion("argmax", "token selection", 5, 10.0),
            CapacityRegion("dead", "unused", 0, 16.0),
        )
    )
    assert profile.total_gate_count == 1_005
    assert math.isclose(profile.token_bits, 10.0)
    assert math.isclose(profile.output_frontier_bits, 50.0)
    units = profile.region_units()
    assert [unit.id for unit in units] == ["macs", "argmax"]
    assert units[0].capacity_upper_bits == 32_000.0
    assert units[1].max_single_cut_bits == 10.0


def test_profile_rejects_duplicates_and_bad_regions() -> None:
    with pytest.raises(ValueError, match="duplicate region id"):
        make_profile(
            (
                CapacityRegion("a", "x", 1, 16.0),
                CapacityRegion("a", "y", 1, 16.0),
            )
        )
    with pytest.raises(ValueError, match="width must be positive"):
        CapacityRegion("a", "x", 1, 0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        CapacityRegion("a", "x", -1, 16.0)
    with pytest.raises(ValueError, match="must be an integer"):
        CapacityRegion("a", "x", 1.5, 16.0)  # type: ignore[arg-type]
    for invalid_width in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            CapacityRegion("a", "x", 1, invalid_width)
    with pytest.raises(ValueError, match="value cardinality must be an integer"):
        CapacityRegion("a", "x", 1, 1.0, 2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"log2\(value cardinality\)"):
        CapacityRegion("a", "x", 1, 32.0, 3)


def test_tiling_preserves_gates_and_capacity_exactly() -> None:
    profile = make_profile(
        (
            CapacityRegion("wide", "macs", 10_007, 32.0),
            CapacityRegion("narrow", "norms", 13, 16.0),
        )
    )
    units = tiled_region_units(profile, target_bits=1_000.0, max_units=4096)
    assert sum(unit.checked_gate_count for unit in units) == profile.total_gate_count
    assert math.isclose(
        sum(unit.capacity_upper_bits for unit in units),
        sum(region.self_cut_bits for region in profile.regions),
    )
    for unit in units:
        assert unit.checked_gate_count > 0
        assert math.isclose(
            unit.capacity_upper_bits,
            unit.checked_gate_count * unit.max_single_cut_bits,
        )
    # Near-equal split: tile gate counts within one region differ by <= 1.
    wide_counts = {
        unit.checked_gate_count for unit in units if unit.row_ids == ("wide",)
    }
    assert len(wide_counts) <= 2
    assert max(wide_counts) - min(wide_counts) <= 1


def test_tiling_respects_the_unit_cap() -> None:
    profile = make_profile((CapacityRegion("wide", "macs", 10**9, 32.0),))
    units = tiled_region_units(profile, target_bits=1.0, max_units=64)
    assert len(units) <= 65
    assert sum(unit.checked_gate_count for unit in units) == 10**9


def test_tiling_never_splits_below_single_gates() -> None:
    profile = make_profile((CapacityRegion("tiny", "selectors", 3, 32.0),))
    units = tiled_region_units(profile, target_bits=1.0, max_units=4096)
    assert len(units) == 3
    assert all(unit.checked_gate_count == 1 for unit in units)
