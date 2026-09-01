from __future__ import annotations

import math

import pytest

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.models.inkling import (
    INKLING,
    InklingConfig,
    build_inkling_capacity_profile,
    global_attention_pair_count,
    local_attention_pair_count,
)

TINY = InklingConfig(
    model_id="inkling-tiny",
    layers=6,
    hidden_size=4,
    query_heads=2,
    head_dim=2,
    global_layer_period=6,
    local_kv_heads=2,
    global_kv_heads=1,
    window=2,
    relative_rank=2,
    local_relative_extent=2,
    global_relative_extent=4,
    sconv_kernel=4,
    dense_layers=2,
    dense_intermediate=8,
    routed_experts=4,
    routed_top_k=2,
    shared_experts=2,
    router_logits=6,
    expert_intermediate=3,
    physical_vocabulary=11,
    logical_vocabulary=9,
    max_context=64,
)


def _regions_by_id(profile: ModelCapacityProfile) -> dict[str, CapacityRegion]:
    return {region.id: region for region in profile.regions}


def test_tiny_config_region_counts_by_hand() -> None:
    """Hand-computed expectations for a 6-layer tiny model, N=4, G=2.

    Tiny schedule: layers 0-4 are sliding-window (window 2, 2 KV heads of
    dimension 2), layer 5 is global (1 KV head); layers 0-1 dense, 2-5
    sparse.  All arithmetic below is written out independently of the
    module's helper functions.
    """

    profile = build_inkling_capacity_profile(3, 2, config=TINY)
    assert profile.prompt_tokens + profile.generated_tokens - 1 == 4
    regions = _regions_by_id(profile)

    # Fixed-prompt rows are inputs; one generated token is fed back.
    assert regions["embedding-lookup"].gate_count == (2 - 1) * 4
    assert regions["embedding-lookup"].self_cut_bits_per_gate == 16.0

    # Two residual merges per layer, 6 layers, 4 positions, 4 coordinates.
    assert regions["residual-merges"].gate_count == 2 * 6 * 4 * 4
    assert regions["residual-merges"].self_cut_bits_per_gate == 16.0

    # Q width is 2 heads x head dim 2 = 4 outputs; each output is a
    # bias-free length-4 dot product (4 muls + 3 adds = 7 gates), in every
    # of 6 layers at 4 positions.
    assert regions["q-projections"].gate_count == 6 * 4 * 4 * 7
    assert regions["q-projections"].self_cut_bits_per_gate == 32.0

    # Local pairs with window 2 over positions 0..3: 1 + 2 + 2 + 2 = 7.
    # Per pair: length-2 dot (2 muls + 1 add) + scale + bias = 5.
    # 5 local layers x 2 heads.
    assert regions["attention-scores-local"].gate_count == 5 * 2 * 7 * 5

    # Global pairs: 1 + 2 + 3 + 4 = 10 on the single global layer.
    assert regions["attention-scores-global"].gate_count == 1 * 2 * 10 * 6

    # Local softmax rows (5n - 1 gates for n keys): key counts per query
    # are 1, 2, 2, 2, so one row of 4 gates and three rows of 9 gates.
    assert regions["attention-softmax-local"].gate_count == 5 * 2 * (4 + 9 + 9 + 9)

    # Local value reductions: per head coordinate (2 per head) with n keys,
    # n muls + (n - 1) adds: (2*1-1) + 3*(2*2-1) = 10 per coordinate.
    assert regions["attention-value-reductions-local"].gate_count == (
        5 * 2 * 2 * (1 + 3 + 3 + 3)
    )

    # Per-head Q/K RMSNorms of width 2 (4*2 + 2 = 10 gates each): per
    # position 6*2 query-head instances, 5*2 local and 1*1 global KV-head
    # instances, at 4 positions.
    assert regions["qk-head-rmsnorms"].gate_count == 4 * (12 + 10 + 1) * 10

    # Embedding RMSNorm of width 4 (4*4 + 2 = 18 gates) at 4 positions.
    assert regions["embedding-rmsnorm"].gate_count == 4 * 18

    # Two selected expert-index outputs per sparse layer (4) per position (4).
    assert regions["moe-top6-selectors"].gate_count == 4 * 4 * TINY.routed_top_k
    assert regions["moe-top6-selectors"].self_cut_bits_per_gate == math.log2(
        TINY.routed_experts
    )
    assert (
        regions["moe-top6-selectors"].value_cardinality_upper_bound
        == TINY.routed_experts
    )

    # LM head at 2 prediction positions: 4 pre-scale muls plus 11 physical
    # rows of bias-free length-4 dot products (7 gates each).
    assert regions["lm-head"].gate_count == 2 * (4 + 11 * 7)

    # One atomic argmax per generated token over the 9 logical tokens.
    assert regions["argmax-tokens"].gate_count == 2
    assert math.isclose(regions["argmax-tokens"].self_cut_bits_per_gate, math.log2(9))
    assert math.isclose(profile.output_frontier_bits, 2 * math.log2(9))


def test_default_profile_smoke() -> None:
    profile = build_inkling_capacity_profile()

    assert profile.model_id == "inkling"
    assert profile.prompt_tokens == 100
    assert profile.generated_tokens == 100
    assert profile.logical_vocabulary_size == 200_058
    assert profile.numerical_profile_id == "inkling-bf16-reference"
    assert profile.total_gate_count > 10**10

    ids = [region.id for region in profile.regions]
    assert len(ids) == len(set(ids))
    assert 12 <= len(ids) <= 30
    assert all(region.gate_count > 0 for region in profile.regions)
    assert math.isclose(profile.output_frontier_bits, 100 * math.log2(200_058))
    assert INKLING.global_layer_indices == (5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65)


def test_region_units_gate_counts_sum_to_total() -> None:
    profile = build_inkling_capacity_profile()
    units = profile.region_units()

    assert len(units) == len(profile.regions)
    assert sum(unit.checked_gate_count for unit in units) == profile.total_gate_count
    by_id = _regions_by_id(profile)
    for unit in units:
        region = by_id[unit.id]
        assert unit.capacity_upper_bits == region.gate_count * (
            region.self_cut_bits_per_gate
        )
        assert unit.max_single_cut_bits == region.self_cut_bits_per_gate


def test_local_window_pair_count_matches_explicit_enumeration() -> None:
    window = TINY.window
    positions = 4
    enumerated = 0
    for q in range(positions):
        for k in range(positions):
            if max(0, q - window + 1) <= k <= q:
                enumerated += 1

    assert enumerated == sum(min(q + 1, window) for q in range(positions))
    assert local_attention_pair_count(positions, window) == enumerated
    assert global_attention_pair_count(positions) == 10

    # The tiny profile's local score region is exactly (local layers) x
    # (heads) x (enumerated pairs) x (per-pair gates).
    profile = build_inkling_capacity_profile(3, 2, config=TINY)
    regions = _regions_by_id(profile)
    per_pair_gates = 2 * TINY.head_dim + 1
    assert regions["attention-scores-local"].gate_count == (
        TINY.local_layer_count * TINY.query_heads * enumerated * per_pair_gates
    )

    # A longer schedule where the window binds on both sides.
    long_enumerated = 0
    for q in range(9):
        for k in range(9):
            if max(0, q - 3 + 1) <= k <= q:
                long_enumerated += 1
    assert local_attention_pair_count(9, 3) == long_enumerated
    assert long_enumerated == 1 + 2 + 3 + 3 + 3 + 3 + 3 + 3 + 3


def test_rejects_unsupported_numerical_profile() -> None:
    with pytest.raises(ValueError, match="numerical profile"):
        build_inkling_capacity_profile(numerical_profile_id="inkling-unknown")
