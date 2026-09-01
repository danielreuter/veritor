from __future__ import annotations

import math

import pytest

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.models.deepseek_v4_pro import (
    DEEPSEEK_V4_PRO,
    DeepSeekV4ProConfig,
    build_deepseek_v4_pro_capacity_profile,
    csa_core_candidates,
    hca_core_candidates,
)

TINY = DeepSeekV4ProConfig(
    model_id="deepseek-v4-pro-tiny",
    vocabulary_size=11,
    hidden_size=4,
    layers=3,
    residual_streams=4,
    query_heads=2,
    kv_heads=1,
    head_dim=4,
    rope_dims=2,
    query_latent_rank=2,
    sliding_window=2,
    csa_layer_ids=(1,),
    csa_compression=2,
    csa_index_top_k=2,
    hca_compression=2,
    indexer_heads=2,
    indexer_head_dim=2,
    indexer_rope_dims=2,
    routed_experts=4,
    experts_per_token=2,
    shared_experts=1,
    expert_intermediate_size=2,
    hash_routed_layers=(0,),
    output_groups=1,
    group_output_dim=2,
    max_positions=64,
)


def region(profile: ModelCapacityProfile, region_id: str) -> CapacityRegion:
    matches = [item for item in profile.regions if item.id == region_id]
    assert len(matches) == 1, f"expected exactly one region {region_id!r}"
    return matches[0]


def test_tiny_config_hand_computed_region_counts() -> None:
    """Independent hand arithmetic for the tiny 3-layer [HCA, CSA, HCA] model.

    Tiny dimensions: hidden 4, 4 residual streams, 2 heads of dim 4 (2 RoPE
    coordinates), window 2, CSA/HCA compression 2, indexer 2 heads x 2 dims,
    4 routed experts top-2 plus 1 shared, hash-routed layer 0, vocabulary 11.
    Prompt 3 and generation 2 give N = 3 + 2 - 1 = 4 processed positions and
    G = 2 prediction positions.
    """

    profile = build_deepseek_v4_pro_capacity_profile(3, 2, config=TINY)

    # Fixed-prompt rows are inputs; only one generated token is fed back,
    # producing four BF16 lookup write-outs.
    embedding = region(profile, "embedding")
    assert embedding.gate_count == (2 - 1) * 4
    assert embedding.self_cut_bits_per_gate == 16.0

    # LM head: 2 prediction positions x 11 logits, each an unbiased
    # 4-wide dot product (4 multiplies + 3 additions).
    assert region(profile, "lm-head").gate_count == 2 * 11 * (4 + 3)

    # Sinkhorn: 2 transitions per layer per token x 3 layers x 4 tokens,
    # each the released 4x4 kernel expansion
    # 12 max + 16 exp + 668 add + 160 reciprocal + 640 mul = 1496 gates.
    assert region(profile, "mhc-sinkhorn").gate_count == 2 * 3 * 4 * (
        12 + 16 + 668 + 160 + 640
    )

    # The flattened four-stream mHC generator normalization is gain-free;
    # the separate reduced branch input has a gained width-4 RMSNorm.
    transitions = 2 * 3 * 4
    assert region(profile, "mhc-generator-rmsnorm").gate_count == transitions * 50
    assert region(profile, "branch-input-rmsnorm").gate_count == transitions * 18

    # Per prediction: 34 flattened-state RMS-scalar gates, 128 normalized
    # affine gates, 16 weight finishers, 28 weighted-reduction gates, and an
    # 18-gate gained output RMSNorm.
    assert region(profile, "mhc-final-reduction").gate_count == 2 * 224

    # CSA index queries project the normalized rank-2 query latent (not the
    # width-4 hidden state). Per-token features cost 104 gates; each of the
    # two completed index blocks costs 80 gates.
    assert region(profile, "indexer-features").gate_count == 4 * 104 + 2 * 80

    # HCA core attention (layers 0 and 2).  Candidates at position n with
    # window 2 and 2-token blocks: min(n, 2) + floor(n / 2):
    #   n=1 -> 1, n=2 -> 3, n=3 -> 3, n=4 -> 4.
    # Per head with M candidates (head dim 4, 2 RoPE coordinates):
    #   scores M x (4 mul + 3 add + 1 scale) = 8M, exponentials M,
    #   denominator adds M, one reciprocal, probability multiplies M,
    #   value reduction 4 x (2M - 1) = 8M - 4, inverse RoPE 6
    #   => 19M + 3 gates.
    per_head = {1: 19 * 1 + 3, 3: 19 * 3 + 3, 4: 19 * 4 + 3}
    expected_hca_core = 2 * 2 * (per_head[1] + per_head[3] + per_head[3] + per_head[4])
    assert region(profile, "attention-core-hca").gate_count == expected_hca_core
    assert expected_hca_core == 884

    # MoE router: per token per layer, 4 affinity dots x (4 mul + 3 add)
    # = 28, plus 4 softplus + 4 sqrt = 8, plus top-2 weight normalization
    # (1 sum add + 1 reciprocal + 2 affinity muls + 2 scale muls) = 6,
    # totalling 42 over 3 layers x 4 tokens; plus selection-bias adds on
    # the two score-routed layers (1 and 2): 2 layers x 4 tokens x 4 adds.
    assert region(profile, "moe-router").gate_count == 3 * 4 * 42 + 2 * 4 * 4

    # Indexer scores on the single CSA layer: completed 2-token blocks per
    # position are 0, 1, 1, 2 -> 4 (position, block) pairs.  Each pair:
    # 2 heads x (2 mul + 1 add + 1 ReLU) = 8, plus the weighted combine
    # (2 weight muls + 1 reduction add) = 3, so 11 gates.
    assert region(profile, "indexer-scores").gate_count == 4 * 11

    # Scalar selector outputs: one per retained CSA block index and two
    # selected expert IDs per (token, layer), with exact finite alphabets.
    csa_selector = region(profile, "csa-index-selector")
    moe_selector = region(profile, "moe-top6-selector")
    assert csa_selector.gate_count == 1 * 4
    assert moe_selector.gate_count == 3 * 4 * TINY.experts_per_token
    assert csa_selector.self_cut_bits_per_gate == math.log2(2)
    assert csa_selector.value_cardinality_upper_bound == 2
    assert moe_selector.self_cut_bits_per_gate == math.log2(TINY.routed_experts)
    assert moe_selector.value_cardinality_upper_bound == TINY.routed_experts

    # Argmax: one atomic gate per generated token at the semantic token
    # capacity, log2(11) bits.
    argmax = region(profile, "argmax")
    assert argmax.gate_count == 2
    assert math.isclose(argmax.self_cut_bits_per_gate, math.log2(11))


def test_csa_selector_emits_no_index_before_first_completed_block() -> None:
    profile = build_deepseek_v4_pro_capacity_profile(1, 1, config=TINY)

    assert region(profile, "csa-index-selector").gate_count == 0


def test_default_profile_smoke() -> None:
    profile = build_deepseek_v4_pro_capacity_profile()
    assert profile.model_id == "deepseek-v4-pro-0813"
    assert profile.numerical_profile_id == "deepseek-v4-pro-bundled-reference"
    assert profile.prompt_tokens == 100
    assert profile.generated_tokens == 100
    assert profile.total_gate_count > 10**10
    ids = [item.id for item in profile.regions]
    assert len(ids) == len(set(ids))
    assert 12 <= len(ids) <= 30
    assert math.isclose(profile.output_frontier_bits, 100 * math.log2(129_280))
    assert any("TRACE_CONDITIONAL" in text for text in profile.assumptions)


def test_rejects_unsupported_numerical_profile() -> None:
    with pytest.raises(ValueError, match="numerical profile"):
        build_deepseek_v4_pro_capacity_profile(numerical_profile_id="deepseek-unknown")


def test_default_config_matches_spec_schedule() -> None:
    config = DEEPSEEK_V4_PRO
    assert config.layers == 61
    assert config.csa_layer_ids == tuple(range(2, 61, 2))
    assert config.csa_layer_count == 30
    assert config.hca_layer_count == 31
    assert config.hca_layer_ids[:2] == (0, 1)
    assert config.hca_layer_ids[2:] == tuple(range(3, 60, 2))
    assert config.score_routed_layer_count == 58
    assert config.query_heads * config.head_dim == 65_536
    assert config.indexer_heads * config.indexer_head_dim == 8_192
    assert config.residual_state_width == 28_672


def test_region_units_sum_to_total_gate_count() -> None:
    for profile in (
        build_deepseek_v4_pro_capacity_profile(),
        build_deepseek_v4_pro_capacity_profile(7, 5, config=TINY),
    ):
        units = profile.region_units()
        assert sum(unit.checked_gate_count for unit in units) == (
            profile.total_gate_count
        )
        assert all(unit.checked_gate_count > 0 for unit in units)


def test_candidate_count_formulas_match_explicit_enumeration() -> None:
    """N_CSA and N_HCA equal enumerated window entries plus completed blocks.

    The enumeration is semantic: sliding-window candidates are the causal
    positions within the window (including the current token), and a block
    is complete exactly when its last member has been processed.
    """

    for config in (DEEPSEEK_V4_PRO, TINY):
        window = config.sliding_window
        for n in range(1, 700):
            window_entries = sum(1 for position in range(max(1, n - window + 1), n + 1))
            csa_blocks = sum(
                1 for block in range(n) if (block + 1) * config.csa_compression <= n
            )
            hca_blocks = sum(
                1 for block in range(n) if (block + 1) * config.hca_compression <= n
            )
            assert csa_core_candidates(n, config) == window_entries + min(
                config.csa_index_top_k, csa_blocks
            )
            assert hca_core_candidates(n, config) == window_entries + hca_blocks

    # Spec cross-checks at the maximum context length.
    assert csa_core_candidates(1_048_576, DEEPSEEK_V4_PRO) == 1_152
    assert hca_core_candidates(1_048_576, DEEPSEEK_V4_PRO) == 8_320
