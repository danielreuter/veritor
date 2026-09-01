"""Tests for the exact Kimi-K3 capacity-profile adapter."""

from __future__ import annotations

import math

import pytest

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)
from circuit_cut_analysis.models.kimi_k3 import (
    KIMI_K3,
    KimiK3Config,
    build_kimi_k3_capacity_profile,
    mla_causal_pair_count,
)

TINY = KimiK3Config(
    model_id="kimi-k3-tiny",
    layers=2,
    hidden_size=4,
    vocabulary_size=8,
    max_context=64,
    mla_layer_indices=(1,),
    dense_ffn_layer_indices=(0,),
    attn_res_block_size=2,
    kda_heads=2,
    kda_key_dim=2,
    kda_value_dim=2,
    kda_conv_kernel=2,
    kda_decay_rank=2,
    mla_query_heads=2,
    mla_q_lora_rank=3,
    mla_kv_lora_rank=3,
    mla_qk_content_dim=2,
    mla_qk_direct_dim=1,
    mla_value_dim=2,
    dense_intermediate_size=5,
    routed_experts=4,
    active_routed_experts=2,
    shared_experts=1,
    routed_latent_width=3,
    expert_intermediate_size=2,
    shared_expert_intermediate_size=2,
)


def _regions_by_id(profile: ModelCapacityProfile) -> dict[str, CapacityRegion]:
    return {region.id: region for region in profile.regions}


def test_default_config_matches_released_checkpoint_schedule() -> None:
    assert KIMI_K3.layers == 93
    assert KIMI_K3.mla_layer_indices == (*range(3, 92, 4), 92)
    assert KIMI_K3.mla_layer_count == 24
    assert KIMI_K3.kda_layer_count == 69
    assert KIMI_K3.dense_ffn_layer_indices == (0,)
    assert KIMI_K3.moe_layer_count == 92
    assert KIMI_K3.attn_res_block_count == 8
    assert KIMI_K3.hidden_size == 7168
    assert KIMI_K3.vocabulary_size == 163_840
    assert KIMI_K3.mla_qk_total_dim == 192
    assert KIMI_K3.kda_qk_width == 96 * 128
    assert KIMI_K3.routed_experts == 896
    assert KIMI_K3.active_routed_experts == 16


def test_tiny_config_counts_match_hand_arithmetic() -> None:
    """Independent literal arithmetic for the tiny run (prompt 3, generated 2).

    Processed positions N = 3 + 2 - 1 = 4; hidden d = 4; layer 0 is
    KDA + dense FFN, layer 1 is MLA + MoE; AttnRes block size 2 gives one
    (partial-free) block, so the final retrieval reads 2 candidate streams.
    """

    profile = build_kimi_k3_capacity_profile(3, 2, config=TINY)
    regions = _regions_by_id(profile)
    positions = 4

    # Fixed-prompt rows are inputs; only one generated token is fed back.
    assert regions["embedding-lookup"].gate_count == (2 - 1) * 4

    # LM head: 8 logits, each a length-4 bias-free dot product (4 muls +
    # 3 adds), at each of the 2 prediction positions. The region uses the
    # maximum 32-bit accumulator width, not only its final BF16 write-out.
    assert regions["lm-head"].gate_count == 2 * 8 * (4 + 3)
    assert regions["lm-head"].self_cut_bits_per_gate == 32.0
    assert regions["kda-projections"].self_cut_bits_per_gate == 32.0
    assert regions["mla-projections"].self_cut_bits_per_gate == 32.0

    # Argmax: one atomic token gate per generated token, width log2(8).
    assert regions["argmax-token"].gate_count == 2
    assert regions["argmax-token"].self_cut_bits_per_gate == pytest.approx(math.log2(8))

    # MoE router (one MoE layer): 4 expert scores of (4 muls + 3 adds) each,
    # 4 sigmoids, 4 correction-bias adds, then top-2 mixture weights:
    # 1 denominator add + 1 epsilon add + 1 reciprocal + 2 multiplies.
    per_token_router = 4 * (4 + 3) + 4 + 4 + (1 + 1 + 1 + 2)
    assert regions["moe-router"].gate_count == positions * per_token_router
    assert (
        regions["moe-top-k-selectors"].gate_count
        == positions * TINY.active_routed_experts
    )
    assert regions["moe-top-k-selectors"].self_cut_bits_per_gate == math.log2(
        TINY.routed_experts
    )
    assert (
        regions["moe-top-k-selectors"].value_cardinality_upper_bound
        == TINY.routed_experts
    )

    # KDA decay is vector-valued over heads*key_dim. The biased rank-2 decay
    # up-projection therefore has 4 outputs x (2 muls + 2 adds), and each of
    # the 4 decay coordinates has four nonlinear gates plus one beta sigmoid
    # per head.
    expected_kda_projections = (
        2 * 4 * 7  # q and k
        + 4 * 7  # v
        + 4 * 7  # output gate
        + 4 * 7  # output projection
        + 2 * 7  # decay down
        + 4 * 4  # biased decay up
        + 2 * 7  # beta
    )
    assert regions["kda-projections"].gate_count == positions * expected_kda_projections
    assert regions["kda-decay-gates"].gate_count == positions * (4 * 4 + 2)

    # MLA attention (one MLA layer, two heads).  Per query with n visible
    # keys and per head: n score dots of length 3 (3 muls + 2 adds) plus one
    # scale multiply each; softmax (n-1 max, n shifts, n exps, n-1
    # denominator adds, 1 reciprocal, n multiplies); and 2 value coordinates
    # of (n muls + n-1 adds) each.
    per_head_attention = 0
    for n in (1, 2, 3, 4):
        scores = n * ((3 + 2) + 1)
        softmax = (n - 1) + n + n + (n - 1) + 1 + n
        value = 2 * (n + (n - 1))
        per_head_attention += scores + softmax + value
    assert regions["mla-attention"].gate_count == 2 * per_head_attention

    # Dense layer-0 FFN matmuls: gate and up projections are 5 outputs of
    # (4 muls + 3 adds) each; the w2 down projection is 4 outputs of
    # (5 muls + 4 adds).  SiTU gates live in the situ-activations region.
    per_position_dense = 5 * (4 + 3) + 5 * (4 + 3) + 4 * (5 + 4)
    assert regions["dense-ffn"].gate_count == positions * per_position_dense

    # AttnRes: layer-0 pre-attention is the identity; the other four
    # retrievals (layer-0 pre-MLP, layer-1 pre-attention, layer-1 pre-MLP,
    # final output) each score 2 candidates.  Per candidate: a gamma-free
    # RMSNorm (4 squares + 1 mean scale + 4 normalize muls; 3 sum + 1
    # epsilon adds; 1 rsqrt) plus a length-4 score dot (4 muls + 3 adds).
    # Then softmax over 2 (1 max + 2 shifts + 2 exps + 1 denominator add +
    # 1 reciprocal + 2 muls) and the weighted sum (2*4 muls + 1*4 adds).
    per_candidate = (4 + 1 + 4) + (3 + 1) + 1 + (4 + 3)
    per_retrieval = 2 * per_candidate + (1 + 2 + 2 + 1 + 1 + 2) + (2 * 4 + 1 * 4)
    assert per_retrieval == 63
    assert regions["attn-res-retrieval"].gate_count == positions * 4 * per_retrieval

    # KDA recurrent state update (one KDA layer, two heads, dk = dv = 2):
    # decay 2*2 muls; error: 2 dots of (2 muls + 1 add) plus 2 subtractions;
    # write: 2 beta*k muls + 2*2 outer muls + 2*2 accumulate adds; read:
    # 2 query-scale muls + 2 dots of (2 muls + 1 add).
    per_head_update = (
        (2 * 2) + (2 * (2 + 1) + 2) + (2 + 2 * 2 + 2 * 2) + (2 + 2 * (2 + 1))
    )
    assert per_head_update == 30
    assert regions["kda-state-update"].gate_count == positions * 2 * per_head_update


def test_default_profile_smoke() -> None:
    profile = build_kimi_k3_capacity_profile()

    assert profile.model_id == "kimi-k3"
    assert profile.prompt_tokens == 100
    assert profile.generated_tokens == 100
    assert profile.numerical_profile_id == "kimi-k3-semantic-mixed"
    assert profile.logical_vocabulary_size == 163_840
    assert profile.total_gate_count > 10**10
    region_ids = [region.id for region in profile.regions]
    assert len(set(region_ids)) == len(region_ids)
    assert all(region.gate_count > 0 for region in profile.regions)
    assert all(region.self_cut_bits_per_gate > 0 for region in profile.regions)
    assert math.isclose(profile.output_frontier_bits, 100 * math.log2(163_840))
    assert profile.assumptions


def test_region_units_gate_counts_sum_to_total() -> None:
    profile = build_kimi_k3_capacity_profile()
    units = profile.region_units()

    assert sum(unit.checked_gate_count for unit in units) == profile.total_gate_count
    assert {unit.id for unit in units} == {region.id for region in profile.regions}
    for unit in units:
        assert unit.capacity_upper_bits == pytest.approx(
            unit.checked_gate_count * unit.max_single_cut_bits
        )


def test_mla_pair_counts_follow_causal_triangular_sum() -> None:
    for processed in range(1, 9):
        assert mla_causal_pair_count(processed) == sum(q + 1 for q in range(processed))

    # Tiny run with prompt 4 + generated 4 -> N = 7 processed positions.
    # Enumerate every causal (query, key) pair explicitly: query q attends
    # to n = q + 1 keys, costing n*6 score gates (length-3 dot plus scale),
    # 5n - 1 softmax gates, and 2*(2n - 1) value-reduction gates per head.
    profile = build_kimi_k3_capacity_profile(4, 4, config=TINY)
    regions = _regions_by_id(profile)
    enumerated = 0
    for q in range(7):
        n = q + 1
        enumerated += n * 6 + (5 * n - 1) + 2 * (2 * n - 1)
    assert regions["mla-attention"].gate_count == 2 * enumerated  # 1 layer, 2 heads


def test_execution_and_profile_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_kimi_k3_capacity_profile(0, 1)
    with pytest.raises(ValueError, match="context window"):
        build_kimi_k3_capacity_profile(60, 10, config=TINY)
    boundary = build_kimi_k3_capacity_profile(60, 5, config=TINY)
    assert boundary.prompt_tokens + boundary.generated_tokens - 1 == TINY.max_context
    with pytest.raises(ValueError, match="numerical profile"):
        build_kimi_k3_capacity_profile(
            numerical_profile_id="kimi-k3-all-fp32-reference"
        )
