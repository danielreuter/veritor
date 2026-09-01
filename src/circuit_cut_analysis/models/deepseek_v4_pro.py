"""Exact capacity profile for DeepSeek-V4-Pro-0813 greedy text generation.

DeepSeek-V4-Pro has no exact indexed circuit in this project.  This module
provides the assumption-light alternative defined by
:mod:`circuit_cut_analysis.models.capacity_profile`: exact computed-gate
counts per architectural region for one fixed execution, plus declared
per-gate value widths, from which certified self-cut capacity upper bounds
follow (the corrupted gates themselves always form a valid downstream cut).

Execution semantics (TRACE_CONDITIONAL executed work)
=====================================================

The accounting models the spec's "exact fixed-circuit semantics" for the
batch-1, temperature-0, text-only 0813 backbone:

* Fixed prompt length ``P`` and output horizon ``G``.  Processed positions
  number ``N = P + G - 1``: the final generated position receives no
  forward pass.  The LM head, the final stream reduction, and one atomic
  greedy argmax run at the ``G`` prediction positions (the final prompt
  position plus every decode position).
* DSpark draft blocks and native MTP are disabled and contribute no gates.
* Selector outcomes are assumed to match an observed greedy trace
  (``TRACE_CONDITIONAL``): all selector score arithmetic is retained, but
  only traced sparse branches execute.  The result is a realized-execution
  gate count, not a globally function-equivalent circuit size.
* MoE executed work: all routed-expert affinities are computed for every
  token at every layer (including the hash-routed layers), while expert
  arithmetic runs only for the selected routed experts plus the
  always-active shared expert.
* CSA executed work: indexer scores are computed over every completed
  compressed block at every position (before top-k retention).  Core
  attention at a position with ``n`` visible tokens covers
  ``N_CSA(n) = min(n, w) + min(k, floor(n / m))`` candidates on CSA layers
  and ``N_HCA(n) = min(n, w) + floor(n / m')`` candidates on HCA layers.
* Every completed compression block (main and indexer) is pooled exactly
  once and cached across steps.

Width policy (bundled-reference arithmetic profile)
===================================================

Embedding write-outs are BF16 boundaries (16 bits).  Every other
arithmetic region mixes BF16 write-outs with FP32 accumulation, softmax,
normalization, mHC, router, or logit arithmetic, so the larger FP32 width
(32 bits) is declared for the whole region; this is loose only in the
sound direction for self-cut capacity bounds. Top-k results use one
attackable scalar gate per emitted selected index with an exact finite index
alphabet bound. The greedy argmax token gate carries
the semantic token capacity ``log2(vocabulary)``, not a storage container
width.  FP8/FP4 storage formats and all quantize/dequantize/cast
boundaries are excluded from the primitive basis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)

BF16_BITS = 16.0
FP32_BITS = 32.0
_SUPPORTED_NUMERICAL_PROFILE_ID = "deepseek-v4-pro-bundled-reference"

# Released-kernel expansion of one 4x4 Sinkhorn projection under the
# project's shared-reciprocal convention, excluding raw affine generation.
SINKHORN_MAX_GATES = 12
SINKHORN_EXP_GATES = 16
SINKHORN_ADD_GATES = 668
SINKHORN_RECIPROCAL_GATES = 160
SINKHORN_MUL_GATES = 640
SINKHORN_GATES_PER_KERNEL = (
    SINKHORN_MAX_GATES
    + SINKHORN_EXP_GATES
    + SINKHORN_ADD_GATES
    + SINKHORN_RECIPROCAL_GATES
    + SINKHORN_MUL_GATES
)


@dataclass(frozen=True, slots=True)
class DeepSeekV4ProConfig:
    """Architecture dimensions of the DeepSeek-V4-Pro-0813 backbone.

    Defaults follow the pinned 0813 checkpoint configuration.  Layer IDs
    are zero-based; ``csa_layer_ids`` lists the compressed-sparse-attention
    layers and every other layer is hierarchically-compressed attention.
    """

    model_id: str = "deepseek-v4-pro-0813"
    vocabulary_size: int = 129_280
    hidden_size: int = 7_168
    layers: int = 61
    residual_streams: int = 4
    query_heads: int = 128
    kv_heads: int = 1
    head_dim: int = 512
    rope_dims: int = 64
    query_latent_rank: int = 1_536
    sliding_window: int = 128
    csa_layer_ids: tuple[int, ...] = tuple(range(2, 61, 2))
    csa_compression: int = 4
    csa_index_top_k: int = 1_024
    hca_compression: int = 128
    indexer_heads: int = 64
    indexer_head_dim: int = 128
    indexer_rope_dims: int = 64
    routed_experts: int = 384
    experts_per_token: int = 6
    shared_experts: int = 1
    expert_intermediate_size: int = 3_072
    hash_routed_layers: tuple[int, ...] = (0, 1, 2)
    output_groups: int = 16
    group_output_dim: int = 1_024
    max_positions: int = 1_048_576

    def __post_init__(self) -> None:
        positive_dimensions = {
            "vocabulary_size": self.vocabulary_size,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "residual_streams": self.residual_streams,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "rope_dims": self.rope_dims,
            "query_latent_rank": self.query_latent_rank,
            "sliding_window": self.sliding_window,
            "csa_compression": self.csa_compression,
            "csa_index_top_k": self.csa_index_top_k,
            "hca_compression": self.hca_compression,
            "indexer_heads": self.indexer_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_rope_dims": self.indexer_rope_dims,
            "routed_experts": self.routed_experts,
            "experts_per_token": self.experts_per_token,
            "shared_experts": self.shared_experts,
            "expert_intermediate_size": self.expert_intermediate_size,
            "output_groups": self.output_groups,
            "group_output_dim": self.group_output_dim,
            "max_positions": self.max_positions,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary must contain at least two tokens")
        if self.residual_streams != 4:
            raise ValueError(
                "the Sinkhorn kernel expansion constants are specific to 4x4 "
                "kernels; residual_streams must be 4"
            )
        if self.rope_dims % 2 or self.rope_dims > self.head_dim:
            raise ValueError("rope_dims must be even and at most head_dim")
        if self.indexer_rope_dims % 2 or self.indexer_rope_dims > self.indexer_head_dim:
            raise ValueError(
                "indexer_rope_dims must be even and at most indexer_head_dim"
            )
        if self.indexer_head_dim & (self.indexer_head_dim - 1):
            raise ValueError(
                "indexer_head_dim must be a power of two for the Hadamard rotation"
            )
        if self.query_heads % self.output_groups:
            raise ValueError("output_groups must divide query_heads evenly")
        if self.experts_per_token > self.routed_experts:
            raise ValueError("experts_per_token cannot exceed routed_experts")
        for field_name, layer_ids in (
            ("csa_layer_ids", self.csa_layer_ids),
            ("hash_routed_layers", self.hash_routed_layers),
        ):
            if len(set(layer_ids)) != len(layer_ids):
                raise ValueError(f"{field_name} must not contain duplicates")
            if any(not 0 <= layer < self.layers for layer in layer_ids):
                raise ValueError(f"{field_name} must lie in [0, layers)")

    @property
    def csa_layer_count(self) -> int:
        return len(self.csa_layer_ids)

    @property
    def hca_layer_ids(self) -> tuple[int, ...]:
        csa = set(self.csa_layer_ids)
        return tuple(layer for layer in range(self.layers) if layer not in csa)

    @property
    def hca_layer_count(self) -> int:
        return self.layers - self.csa_layer_count

    @property
    def score_routed_layer_count(self) -> int:
        return self.layers - len(self.hash_routed_layers)

    @property
    def residual_state_width(self) -> int:
        return self.residual_streams * self.hidden_size


DEEPSEEK_V4_PRO = DeepSeekV4ProConfig()


def completed_compression_blocks(visible_tokens: int, block_tokens: int) -> int:
    """Completed ``block_tokens``-token blocks among ``visible_tokens``.

    Block ``b`` (zero-based) is complete once token ``(b + 1) * block_tokens``
    has been processed, so ``floor(n / block_tokens)`` blocks are complete at
    a position with ``n`` visible tokens (including itself).
    """

    if visible_tokens < 0:
        raise ValueError("visible token count cannot be negative")
    return visible_tokens // block_tokens


def csa_core_candidates(
    visible_tokens: int,
    config: DeepSeekV4ProConfig = DEEPSEEK_V4_PRO,
) -> int:
    """``N_CSA(n) = min(n, window) + min(top_k, floor(n / m))``."""

    window = min(visible_tokens, config.sliding_window)
    blocks = completed_compression_blocks(visible_tokens, config.csa_compression)
    return window + min(config.csa_index_top_k, blocks)


def hca_core_candidates(
    visible_tokens: int,
    config: DeepSeekV4ProConfig = DEEPSEEK_V4_PRO,
) -> int:
    """``N_HCA(n) = min(n, window) + floor(n / m')``."""

    window = min(visible_tokens, config.sliding_window)
    return window + completed_compression_blocks(visible_tokens, config.hca_compression)


def _projection_gates(input_width: int, output_width: int) -> int:
    """Unbiased linear map: per output, ``input_width`` multiplies and
    ``input_width - 1`` accumulator additions."""

    return output_width * (2 * input_width - 1)


def _rmsnorm_gates(width: int, *, gain: bool) -> int:
    """RMSNorm without mean or beta: squares, sum, mean scale, epsilon add,
    rsqrt, one normalize multiply per coordinate, and (if ``gain``) one gain
    multiply per coordinate."""

    squares = width
    reduction_adds = width - 1
    mean_scale = 1
    epsilon_add = 1
    rsqrt = 1
    normalize = width
    gain_scale = width if gain else 0
    total = (
        squares
        + reduction_adds
        + mean_scale
        + epsilon_add
        + rsqrt
        + normalize
        + gain_scale
    )
    expected = 4 * width + 2 if gain else 3 * width + 2
    if total != expected:
        raise AssertionError("RMSNorm accounting disagrees with its closed form")
    return total


def _partial_rope_gates(rope_dims: int) -> int:
    """Each rotated coordinate pair costs four multiplies and two additions;
    sin/cos values are fixed positional constants, not gates."""

    if rope_dims % 2:
        raise ValueError("RoPE requires an even number of rotated coordinates")
    return (rope_dims // 2) * 6


def _hadamard_gates(dim: int) -> int:
    """Fast Hadamard butterfly (``dim * log2(dim)`` add/sub) plus one
    normalization multiply per coordinate."""

    if dim & (dim - 1):
        raise ValueError("Hadamard rotation requires a power-of-two dimension")
    return dim * (dim.bit_length() - 1) + dim


def _softmax_pool_gates_per_coordinate(lanes: int) -> int:
    """One coordinate of biased softmax pooling over ``lanes`` source lanes:
    bias adds, stable max, shift, exp, denominator adds, one shared
    reciprocal, probability multiplies, and the weighted-sum reduction."""

    bias_adds = lanes
    running_max = lanes - 1
    shift_adds = lanes
    exponentials = lanes
    denominator_adds = lanes - 1
    reciprocal = 1
    probability_muls = lanes
    weighted_sum = 2 * lanes - 1
    total = (
        bias_adds
        + running_max
        + shift_adds
        + exponentials
        + denominator_adds
        + reciprocal
        + probability_muls
        + weighted_sum
    )
    if total != 8 * lanes - 2:
        raise AssertionError("pooling accounting disagrees with its closed form")
    return total


def _core_attention_gates_per_head(candidates: int, config: DeepSeekV4ProConfig) -> int:
    """One head's executed core-attention work at one position.

    Per candidate: one ``head_dim`` dot product plus one scale multiply
    (``2 * head_dim`` gates), one exponential, one denominator addition
    (the sink term ``exp(z')`` is a fixed per-head constant), and one
    probability multiply.  The value reduction runs over the same candidate
    vectors for each of the ``head_dim`` output coordinates, and the last
    ``rope_dims`` output coordinates receive inverse RoPE.
    """

    if candidates <= 0:
        raise ValueError("candidate count must be positive")
    head_dim = config.head_dim
    scores = candidates * 2 * head_dim
    exponentials = candidates
    denominator_adds = candidates
    reciprocal = 1
    probability_muls = candidates
    value_reduction = head_dim * (2 * candidates - 1)
    inverse_rope = _partial_rope_gates(config.rope_dims)
    total = (
        scores
        + exponentials
        + denominator_adds
        + reciprocal
        + probability_muls
        + value_reduction
        + inverse_rope
    )
    closed_form = (4 * head_dim + 3) * candidates + 1 - head_dim + 3 * config.rope_dims
    if total != closed_form:
        raise AssertionError("core-attention accounting disagrees with its closed form")
    return total


def _mhc_generator_affine_gates(config: DeepSeekV4ProConfig) -> int:
    """Dynamic-parameter affine per mHC transition: ``S*(S+2)`` outputs from
    the normalized flattened state (unbiased dot, alpha scale, and static
    offset each), plus sigmoid/epsilon finishers for ``A`` and doubled
    sigmoids for ``C``."""

    state = config.residual_state_width
    streams = config.residual_streams
    outputs = streams * (streams + 2)
    per_output = (2 * state - 1) + 1 + 1
    finishers = 4 * streams
    return outputs * per_output + finishers


def _mhc_mixing_gates(config: DeepSeekV4ProConfig) -> int:
    """Stream mixing per mHC transition: ``A_l X_l`` (S -> 1 mix per hidden
    coordinate), ``B_l X_l`` (S x S mix), the ``C_l`` broadcast of the block
    output, and the residual addition ``B X + C F(A X)``."""

    streams = config.residual_streams
    hidden = config.hidden_size
    a_mix = hidden * (2 * streams - 1)
    b_mix = streams * hidden * (2 * streams - 1)
    c_broadcast = streams * hidden
    residual_add = streams * hidden
    total = a_mix + b_mix + c_broadcast + residual_add
    expected = hidden * (2 * streams - 1) * (streams + 1) + 2 * streams * hidden
    if total != expected:
        raise AssertionError("mHC mixing disagrees with its closed form")
    return total


def _mhc_final_head_gates(config: DeepSeekV4ProConfig) -> int:
    """Dynamic weighted mHC head plus the gained model-output RMSNorm.

    The bundled implementation computes one shared inverse RMS from the
    flattened ``S*d`` residual state, applies ``S`` unbiased affine rows and
    that inverse RMS, finishes each mixing weight with
    scale/base/sigmoid/epsilon, reduces the weighted ``S`` streams, and then
    applies the ordinary gained width-``d`` output RMSNorm.
    """

    streams = config.residual_streams
    hidden = config.hidden_size
    residual_width = config.residual_state_width

    normalization_scalar = 2 * residual_width + 2
    normalized_affine = 2 * streams * residual_width
    weight_finishers = 4 * streams
    weighted_reduction = hidden * (2 * streams - 1)
    output_norm = _rmsnorm_gates(hidden, gain=True)
    return (
        normalization_scalar
        + normalized_affine
        + weight_finishers
        + weighted_reduction
        + output_norm
    )


def _query_projection_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per layer: query down-projection, gained latent RMSNorm,
    up-projection to all heads, unparameterized per-head RMS normalization,
    and partial RoPE on each head's last ``rope_dims`` coordinates."""

    down = _projection_gates(config.hidden_size, config.query_latent_rank)
    latent_norm = _rmsnorm_gates(config.query_latent_rank, gain=True)
    up = _projection_gates(
        config.query_latent_rank, config.query_heads * config.head_dim
    )
    head_norms = config.query_heads * _rmsnorm_gates(config.head_dim, gain=False)
    rope = config.query_heads * _partial_rope_gates(config.rope_dims)
    return down + latent_norm + up + head_norms + rope


def _kv_projection_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per layer: one direct hidden -> head_dim projection per KV
    head, gained RMSNorm, and partial RoPE; the same vector serves as both
    key and value."""

    per_head = (
        _projection_gates(config.hidden_size, config.head_dim)
        + _rmsnorm_gates(config.head_dim, gain=True)
        + _partial_rope_gates(config.rope_dims)
    )
    return config.kv_heads * per_head


def _output_projection_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per layer: grouped head-output projections followed by one
    projection of the concatenated group outputs back to the hidden size."""

    group_input = (config.query_heads // config.output_groups) * config.head_dim
    per_group = _projection_gates(group_input, config.group_output_dim)
    concat_width = config.output_groups * config.group_output_dim
    final = _projection_gates(concat_width, config.hidden_size)
    return config.output_groups * per_group + final


def _csa_compressor_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per CSA layer: four dedicated hidden -> head_dim C/Z
    feature projections (current-block and previous-block branches)."""

    return 4 * _projection_gates(config.hidden_size, config.head_dim)


def _csa_compressor_block_gates(config: DeepSeekV4ProConfig) -> int:
    """Per completed CSA block: coordinate-wise ``2m``-lane biased softmax
    pooling, gained RMSNorm of the pooled vector, and block-start RoPE."""

    lanes = 2 * config.csa_compression
    pool = config.head_dim * _softmax_pool_gates_per_coordinate(lanes)
    norm = _rmsnorm_gates(config.head_dim, gain=True)
    rope = _partial_rope_gates(config.rope_dims)
    return pool + norm + rope


def _hca_compressor_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per HCA layer: two dedicated hidden -> head_dim C/Z
    feature projections (no overlap branch)."""

    return 2 * _projection_gates(config.hidden_size, config.head_dim)


def _hca_compressor_block_gates(config: DeepSeekV4ProConfig) -> int:
    """Per completed HCA block: coordinate-wise ``m'``-lane biased softmax
    pooling, gained RMSNorm of the pooled vector, and block-start RoPE."""

    lanes = config.hca_compression
    pool = config.head_dim * _softmax_pool_gates_per_coordinate(lanes)
    norm = _rmsnorm_gates(config.head_dim, gain=True)
    rope = _partial_rope_gates(config.rope_dims)
    return pool + norm + rope


def _indexer_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per CSA layer: expanded index-query projection, head-weight
    projection with its shared scale, per-head partial RoPE and normalized
    Hadamard rotation, and four index-key C/Z feature projections."""

    query = _projection_gates(
        config.query_latent_rank,
        config.indexer_heads * config.indexer_head_dim,
    )
    weights = (
        _projection_gates(config.hidden_size, config.indexer_heads)
        + config.indexer_heads
    )
    rope = config.indexer_heads * _partial_rope_gates(config.indexer_rope_dims)
    hadamard = config.indexer_heads * _hadamard_gates(config.indexer_head_dim)
    keys = 4 * _projection_gates(config.hidden_size, config.indexer_head_dim)
    return query + weights + rope + hadamard + keys


def _indexer_block_gates(config: DeepSeekV4ProConfig) -> int:
    """Per completed index-key block: ``2m``-lane pooling at the indexer
    width, gained RMSNorm, block-start RoPE, and one Hadamard rotation."""

    lanes = 2 * config.csa_compression
    pool = config.indexer_head_dim * _softmax_pool_gates_per_coordinate(lanes)
    norm = _rmsnorm_gates(config.indexer_head_dim, gain=True)
    rope = _partial_rope_gates(config.indexer_rope_dims)
    hadamard = _hadamard_gates(config.indexer_head_dim)
    return pool + norm + rope + hadamard


def _indexer_score_gates_per_block(config: DeepSeekV4ProConfig) -> int:
    """One (position, completed block) index score: per head, one
    ``indexer_head_dim`` dot product and one ReLU compare/select; then the
    head-weight multiplies and the final reduction."""

    heads = config.indexer_heads
    width = config.indexer_head_dim
    per_head = (2 * width - 1) + 1
    combine = heads + (heads - 1)
    total = heads * per_head + combine
    if total != 2 * heads * width + 2 * heads - 1:
        raise AssertionError("indexer score disagrees with its closed form")
    return total


def _router_token_gates(config: DeepSeekV4ProConfig) -> int:
    """Per token per layer, excluding selection-bias adds: all routed-expert
    affinity dot products, one softplus and one sqrt per affinity, and
    normalization of the selected weights (sum, shared reciprocal, one
    affinity multiply and one 2.5 scale multiply per selected weight)."""

    affinities = _projection_gates(config.hidden_size, config.routed_experts)
    softplus_and_sqrt = 2 * config.routed_experts
    normalize = 3 * config.experts_per_token
    return affinities + softplus_and_sqrt + normalize


def _expert_gates(config: DeepSeekV4ProConfig, *, routed: bool) -> int:
    """One clamped-SwiGLU expert: gate and linear up-projections, the gate
    upper clamp at 10, SiLU (sigmoid plus multiply), the linear-branch clamp
    to [-10, 10] (two compare/selects), the gating product, the routing
    weight multiply (routed experts only), and the down-projection."""

    hidden = config.hidden_size
    intermediate = config.expert_intermediate_size
    gate_up = _projection_gates(hidden, intermediate)
    linear_up = _projection_gates(hidden, intermediate)
    gate_clamp = intermediate
    silu = 2 * intermediate
    linear_clamp = 2 * intermediate
    gating_product = intermediate
    routing_weight = intermediate if routed else 0
    down = _projection_gates(intermediate, hidden)
    total = (
        gate_up
        + linear_up
        + gate_clamp
        + silu
        + linear_clamp
        + gating_product
        + routing_weight
        + down
    )
    expected = (
        2 * _projection_gates(hidden, intermediate)
        + _projection_gates(intermediate, hidden)
        + (7 if routed else 6) * intermediate
    )
    if total != expected:
        raise AssertionError("expert accounting disagrees with its closed form")
    return total


def build_deepseek_v4_pro_capacity_profile(
    prompt_tokens: int = 100,
    generated_tokens: int = 100,
    *,
    config: DeepSeekV4ProConfig = DEEPSEEK_V4_PRO,
    numerical_profile_id: str = _SUPPORTED_NUMERICAL_PROFILE_ID,
) -> ModelCapacityProfile:
    """Build the exact TRACE_CONDITIONAL executed-work capacity profile.

    ``N = prompt_tokens + generated_tokens - 1`` positions are processed;
    the LM head, final stream reduction, and atomic greedy argmax run at
    the ``generated_tokens`` prediction positions.  Per-position candidate
    counts are summed with exact integer loops over all processed
    positions.  See the module docstring for the full declared semantics.
    """

    if numerical_profile_id != _SUPPORTED_NUMERICAL_PROFILE_ID:
        raise ValueError(
            "unsupported numerical profile "
            f"{numerical_profile_id!r}; only "
            f"{_SUPPORTED_NUMERICAL_PROFILE_ID!r} has declared widths"
        )
    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise ValueError("prompt and generated token counts must be positive")
    processed = prompt_tokens + generated_tokens - 1
    if processed > config.max_positions:
        raise ValueError(
            "last processed position must fit the declared context window: "
            f"{processed} > {config.max_positions}"
        )

    hidden = config.hidden_size
    heads = config.query_heads
    streams = config.residual_streams
    transitions = 2 * config.layers * processed
    token_layers = config.layers * processed

    csa_core_per_layer = 0
    hca_core_per_layer = 0
    indexer_block_pairs = 0
    csa_selected_indices_per_layer = 0
    for visible in range(1, processed + 1):
        csa_core_per_layer += heads * _core_attention_gates_per_head(
            csa_core_candidates(visible, config), config
        )
        hca_core_per_layer += heads * _core_attention_gates_per_head(
            hca_core_candidates(visible, config), config
        )
        indexer_block_pairs += completed_compression_blocks(
            visible, config.csa_compression
        )
        csa_selected_indices_per_layer += min(
            config.csa_index_top_k,
            completed_compression_blocks(visible, config.csa_compression),
        )

    csa_blocks = completed_compression_blocks(processed, config.csa_compression)
    hca_blocks = completed_compression_blocks(processed, config.hca_compression)

    regions = (
        CapacityRegion(
            id="embedding",
            description=(
                "One lookup write-out per hidden coordinate for each of the "
                f"{generated_tokens - 1} generated tokens fed back into the "
                "backbone; fixed-prompt embedding rows are circuit inputs. "
                "Residual-stream replication is a copy, not a gate. BF16."
            ),
            gate_count=(generated_tokens - 1) * hidden,
            self_cut_bits_per_gate=BF16_BITS,
        ),
        CapacityRegion(
            id="mhc-generator-rmsnorm",
            description=(
                "Gain-free RMSNorm over the flattened "
                f"{streams}x{hidden} residual state feeding each dynamic "
                "mHC parameter generation: squares, sum, mean scale, epsilon "
                f"add, rsqrt, and normalize multiplies; "
                f"{2 * config.layers} transitions per token.  FP32 mHC "
                "arithmetic."
            ),
            gate_count=transitions
            * _rmsnorm_gates(config.residual_state_width, gain=False),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="branch-input-rmsnorm",
            description=(
                f"Separate gained width-{hidden} RMSNorm of each mHC-reduced "
                "branch input before attention or FFN, one per transition. "
                "FP32 normalization arithmetic."
            ),
            gate_count=transitions * _rmsnorm_gates(hidden, gain=True),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="mhc-generator-affine",
            description=(
                f"Dynamic mHC parameter affine {config.residual_state_width} "
                f"-> {streams * (streams + 2)} per transition (unbiased dot, "
                "alpha scale, and static offset per output) plus "
                "sigmoid-plus-epsilon finishers for A and doubled sigmoids "
                "for C.  FP32 mHC arithmetic."
            ),
            gate_count=transitions * _mhc_generator_affine_gates(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="mhc-sinkhorn",
            description=(
                "Released-kernel 4x4 Sinkhorn projection of B per mHC "
                f"transition: {SINKHORN_MAX_GATES} max, {SINKHORN_EXP_GATES} "
                f"exp, {SINKHORN_ADD_GATES} add/sub, "
                f"{SINKHORN_RECIPROCAL_GATES} reciprocal, "
                f"{SINKHORN_MUL_GATES} mul "
                f"({SINKHORN_GATES_PER_KERNEL} gates per kernel).  FP32 mHC "
                "arithmetic."
            ),
            gate_count=transitions * SINKHORN_GATES_PER_KERNEL,
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="mhc-mixing",
            description=(
                "Stream mixing per mHC transition: A_l X_l "
                f"({streams} -> 1), B_l X_l ({streams}x{streams}), the C_l "
                "broadcast of the block output, and the residual addition "
                "B X + C F(A X).  FP32 (sound larger choice over the BF16 "
                "residual boundary)."
            ),
            gate_count=transitions * _mhc_mixing_gates(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="mhc-final-reduction",
            description=(
                f"Dynamic weighted {streams}-stream output head at each of "
                f"the {generated_tokens} prediction positions: shared "
                f"gain-free flattened-state RMS scalar, {streams} affine mixing "
                "logits with scale/base/sigmoid/epsilon finishers, weighted "
                f"stream reduction, and gained width-{hidden} RMSNorm. FP32."
            ),
            gate_count=generated_tokens * _mhc_final_head_gates(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="attention-query-projection",
            description=(
                f"Per token per layer: {hidden} -> {config.query_latent_rank} "
                "query down-projection, gained latent RMSNorm, "
                f"{config.query_latent_rank} -> {heads * config.head_dim} "
                "up-projection, unparameterized per-head RMS normalization, "
                f"and partial RoPE on the last {config.rope_dims} coordinates "
                f"of each of {heads} heads.  FP32 accumulators (sound over "
                "BF16 write-outs)."
            ),
            gate_count=token_layers * _query_projection_token_gates(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="attention-kv-projection",
            description=(
                f"Per token per layer: direct {hidden} -> {config.head_dim} "
                f"projection for {config.kv_heads} shared KV head, gained "
                f"RMSNorm, and partial RoPE on {config.rope_dims} "
                "coordinates; the same vector serves as key and value.  "
                "FP32 accumulators (sound over the BF16/FP8 cache boundary)."
            ),
            gate_count=token_layers * _kv_projection_token_gates(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="csa-compressor",
            description=(
                "CSA layers only.  Per token: four dedicated "
                f"{hidden} -> {config.head_dim} C/Z feature projections "
                "(current and previous-block branches).  Per completed "
                f"{config.csa_compression}-token block: coordinate-wise "
                f"{2 * config.csa_compression}-lane biased softmax pooling, "
                "gained RMSNorm of the pooled vector, and block-start "
                "partial RoPE.  FP32 compression pooling."
            ),
            gate_count=config.csa_layer_count
            * (
                processed * _csa_compressor_token_gates(config)
                + csa_blocks * _csa_compressor_block_gates(config)
            ),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="hca-compressor",
            description=(
                "HCA layers only.  Per token: two dedicated "
                f"{hidden} -> {config.head_dim} C/Z feature projections.  "
                f"Per completed {config.hca_compression}-token block: "
                f"coordinate-wise {config.hca_compression}-lane biased "
                "softmax pooling (no overlap), gained RMSNorm of the pooled "
                "vector, and block-start partial RoPE.  FP32 compression "
                "pooling."
            ),
            gate_count=config.hca_layer_count
            * (
                processed * _hca_compressor_token_gates(config)
                + hca_blocks * _hca_compressor_block_gates(config)
            ),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="indexer-features",
            description=(
                "CSA layers only. Per token: normalized rank-"
                f"{config.query_latent_rank} query latent -> "
                f"{config.indexer_heads * config.indexer_head_dim} "
                f"index-query projection, {hidden} -> {config.indexer_heads} "
                "head-weight projection with its shared 1/sqrt scale, "
                "per-head partial RoPE and normalized Hadamard rotation, and "
                f"four {hidden} -> {config.indexer_head_dim} index-key C/Z "
                "projections.  Per completed block: "
                f"{2 * config.csa_compression}-lane pooling at width "
                f"{config.indexer_head_dim}, gained RMSNorm, block-start "
                "RoPE, and one Hadamard rotation.  FP32; FP4 Q/K simulation "
                "casts excluded."
            ),
            gate_count=config.csa_layer_count
            * (
                processed * _indexer_token_gates(config)
                + csa_blocks * _indexer_block_gates(config)
            ),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="indexer-scores",
            description=(
                f"ReLU'd {config.indexer_head_dim}-dim dot product per "
                "(index head, completed block) plus the "
                f"{config.indexer_heads}-way weighted combine, computed over "
                "every completed compressed block at every position of each "
                f"CSA layer (executed work before top-"
                f"{config.csa_index_top_k} retention).  FP32."
            ),
            gate_count=config.csa_layer_count
            * indexer_block_pairs
            * _indexer_score_gates_per_block(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="csa-index-selector",
            description=(
                f"The top-{config.csa_index_top_k} retained block indices, "
                "shared by all query heads, represented by one attackable "
                "scalar gate per emitted index. At a position this emits "
                "min(k, completed-block-count) indices; each is bounded by "
                f"the fixed execution's {csa_blocks}-block index alphabet."
            ),
            gate_count=(config.csa_layer_count * csa_selected_indices_per_layer),
            self_cut_bits_per_gate=math.log2(max(1, csa_blocks)),
            value_cardinality_upper_bound=max(1, csa_blocks),
        ),
        CapacityRegion(
            id="attention-core-csa",
            description=(
                "CSA core attention per (position, head): candidate scores "
                f"q.c/sqrt({config.head_dim}), sink-denominator softmax "
                "(exp per candidate, denominator adds, one shared "
                "reciprocal, probability multiplies; no max subtraction), "
                f"{config.head_dim}-wide value reduction over the same "
                f"candidates, and inverse RoPE on {config.rope_dims} output "
                "coordinates.  Candidates per position n: N_CSA(n) = "
                f"min(n, {config.sliding_window}) + "
                f"min({config.csa_index_top_k}, "
                f"floor(n / {config.csa_compression})), summed exactly over "
                "positions.  FP32 logits/online softmax/accumulation."
            ),
            gate_count=config.csa_layer_count * csa_core_per_layer,
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="attention-core-hca",
            description=(
                "HCA core attention per (position, head): same score, "
                "sink-denominator softmax, value-reduction, and inverse-RoPE "
                "arithmetic as CSA.  Candidates per position n: N_HCA(n) = "
                f"min(n, {config.sliding_window}) + "
                f"floor(n / {config.hca_compression}), summed exactly over "
                "positions.  FP32 logits/online softmax/accumulation."
            ),
            gate_count=config.hca_layer_count * hca_core_per_layer,
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="attention-output-projection",
            description=(
                f"Per token per layer: {config.output_groups} grouped "
                f"{(heads // config.output_groups) * config.head_dim} -> "
                f"{config.group_output_dim} projections, then one "
                f"{config.output_groups * config.group_output_dim} -> "
                f"{hidden} projection of the concatenated group outputs.  "
                "FP32 accumulators (sound over BF16 write-outs)."
            ),
            gate_count=token_layers * _output_projection_token_gates(config),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="moe-router",
            description=(
                f"All {config.routed_experts} sqrt-softplus affinities per "
                f"token on every layer ({hidden}-wide dots plus one softplus "
                "and one sqrt each), selection-bias additions on the "
                f"{config.score_routed_layer_count} score-routed layers, and "
                f"normalization of the {config.experts_per_token} selected "
                "weights (sum, shared reciprocal, per-weight affinity "
                "multiply and 2.5 scale).  FP32 router arithmetic."
            ),
            gate_count=token_layers * _router_token_gates(config)
            + config.score_routed_layer_count * processed * config.routed_experts,
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="moe-top6-selector",
            description=(
                f"{config.experts_per_token} attackable scalar expert "
                "indices per token per layer: "
                f"top-{config.experts_per_token} comparator selection on "
                "score-routed layers, tid2eid hash lookup on layers "
                f"{list(config.hash_routed_layers)}; each index ranges over "
                f"{config.routed_experts} routed experts."
            ),
            gate_count=token_layers * config.experts_per_token,
            self_cut_bits_per_gate=math.log2(config.routed_experts),
            value_cardinality_upper_bound=config.routed_experts,
        ),
        CapacityRegion(
            id="moe-routed-experts",
            description=(
                f"Executed work for the {config.experts_per_token} selected "
                "routed experts per token per layer: gate and linear "
                f"{hidden} -> {config.expert_intermediate_size} projections, "
                "gate upper clamp at 10, SiLU (sigmoid plus multiply), "
                "linear-branch clamp to [-10, 10], gating product, routing "
                f"weight multiply, and {config.expert_intermediate_size} -> "
                f"{hidden} down-projection.  FP32 accumulation (sound over "
                "BF16 outputs)."
            ),
            gate_count=token_layers
            * config.experts_per_token
            * _expert_gates(config, routed=True),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="moe-shared-expert",
            description=(
                f"The {config.shared_experts} always-active shared expert "
                "per token per layer: identical clamped-SwiGLU arithmetic "
                "without the routing-weight multiply.  FP32 accumulation "
                "(sound over BF16 outputs)."
            ),
            gate_count=token_layers
            * config.shared_experts
            * _expert_gates(config, routed=False),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="moe-output-combine",
            description=(
                f"FP32 accumulation of the {config.experts_per_token} routed "
                f"outputs plus {config.shared_experts} shared-expert output "
                "per hidden coordinate per token per layer."
            ),
            gate_count=token_layers
            * hidden
            * (config.experts_per_token + config.shared_experts - 1),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="lm-head",
            description=(
                f"Full {config.vocabulary_size}-way logit projection "
                f"({hidden}-wide unbiased dots) at each of the "
                f"{generated_tokens} prediction positions.  FP32 logits."
            ),
            gate_count=generated_tokens
            * _projection_gates(hidden, config.vocabulary_size),
            self_cut_bits_per_gate=FP32_BITS,
        ),
        CapacityRegion(
            id="argmax",
            description=(
                "One atomic greedy argmax over "
                f"{config.vocabulary_size} logits per generated token; the "
                "width is the semantic token capacity "
                f"log2({config.vocabulary_size})."
            ),
            gate_count=generated_tokens,
            self_cut_bits_per_gate=math.log2(config.vocabulary_size),
        ),
    )

    assumptions = (
        "TRACE_CONDITIONAL executed-work semantics: recomputed selectors are "
        "assumed to match an observed greedy trace, and only the traced "
        "sparse branches execute; the profile is a realized-execution gate "
        "count, not a globally function-equivalent circuit size.",
        f"Processed positions N = prompt + generated - 1 = {processed}; the "
        "final generated position receives no forward pass; the LM head, "
        "final stream reduction, and argmax run at the "
        f"{generated_tokens} prediction positions.",
        "MoE executed-work semantics: all routed-expert affinities are "
        "computed per token per layer (hash-routed layers included), but "
        "expert arithmetic runs only for the selected routed experts plus "
        "the shared expert.",
        "CSA executed-work semantics: indexer scores cover every completed "
        "compressed block per position before top-k retention; core "
        "attention covers N_CSA(n) = min(n, window) + min(top_k, "
        "floor(n / m)) candidates, and HCA covers N_HCA(n) = min(n, window) "
        "+ floor(n / m').",
        "Each completed compression block (main and indexer) is pooled "
        "exactly once and cached; pooling counts uniform lane arithmetic "
        "including the -inf/0 padded previous-block lanes of the first CSA "
        "block.",
        "Each top-k result is represented by one attackable scalar gate per "
        "emitted selected index: min(k, completed blocks) CSA "
        "indices per (position, CSA layer), and experts_per_token MoE indices "
        "per (token, layer). Singleton capacities use the exact fixed-run "
        "block-index and routed-expert alphabets; comparator/tie-breaking "
        "arithmetic is not expanded.",
        "DSpark draft blocks and native MTP are disabled and contribute no gates.",
        "Fixed-prompt embedding rows are circuit inputs; only the G-1 "
        "generated-token feedback lookups are attackable computed lookup gates.",
        "FP8/FP4 storage is not modeled; bundled-reference compute dtypes "
        "are used, and the larger FP32 width is declared wherever a region "
        "mixes BF16 write-outs with FP32 accumulation, softmax, "
        "normalization, mHC, router, or logit arithmetic (sound for "
        "self-cut upper bounds).",
        "Casts, quantize/dequantize simulation, loads, stores, gathers, "
        "table indexing, and control flow are excluded from the primitive "
        "basis.",
        "All projections are counted bias-free (the configuration declares "
        "no attention bias) and the untied LM head is unbiased.",
        "sigmoid, softplus, sqrt, exp, reciprocal, rsqrt, and each "
        "compare/select count as one primitive gate; SiLU = sigmoid + "
        "multiply, ReLU = one compare/select, clamp = one compare/select "
        "per bound, and each RoPE pair costs four multiplies and two adds "
        "(sin/cos are fixed constants).",
        "Attention softmax follows the spec's sink-denominator form without "
        "max subtraction; exp(sink) is a fixed per-head constant with no "
        "gate and no value-vector numerator.",
        "The mHC final output is the released dynamic weighted stream head "
        "followed by gained RMSNorm. It is counted at prediction positions "
        "only: this semantic optimized schedule skips unused intermediate "
        "prefill rows that the bundled eager implementation materializes.",
        "Sinkhorn kernels use the released-kernel expansion (12 max, 16 "
        "exp, 668 add/sub, 160 reciprocal, 640 mul per 4x4 kernel); the "
        "accounting is fixed to four residual streams.",
        "Compressor and indexer feature projections (four C/Z maps per CSA "
        "layer, two per HCA layer, four index-key maps per CSA layer) and "
        "the expanded index query and head weights are computed from the "
        "hidden state at every processed position of the owning layer.",
        f"Indexer partial RoPE spans {config.indexer_rope_dims} of "
        f"{config.indexer_head_dim} coordinates per index head (declared; "
        "the spec fixes only 'partial RoPE'); index queries and compressed "
        "index keys each receive one normalized Hadamard rotation counted "
        "as butterfly adds plus one scale multiply per coordinate.",
        "Compressor pooled-vector normalization and the KV/latent norms are "
        "counted as gained RMSNorms; the per-head query normalization is "
        "unparameterized, per the spec.",
        "Embedding lookups are zero-arithmetic materialization boundaries, "
        "one gate per produced coordinate; residual-stream replication is a "
        "copy, not a gate.",
        "Hash-routed layers still compute and normalize all affinities; "
        "selection-bias additions are counted only on score-routed layers.",
    )

    return ModelCapacityProfile(
        model_id=config.model_id,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        logical_vocabulary_size=config.vocabulary_size,
        numerical_profile_id=numerical_profile_id,
        regions=regions,
        assumptions=assumptions,
    )
