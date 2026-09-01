"""Exact capacity-profile accounting for Kimi-K3 text-only greedy generation.

Kimi-K3 has no exact indexed circuit in this project.  This module instead
builds a :class:`ModelCapacityProfile`: exact per-region computed-gate
counts from closed-form architecture formulas plus one declared value width
per region, from which certified self-cut capacity upper bounds follow
(corrupted gates always form a valid downstream cut).

Execution semantics
===================

One batch-1 text-only greedy run with prompt length ``P`` and exactly ``G``
generated tokens is modeled:

* processed backbone positions: ``N = P + G - 1``;
* the final AttnRes output retrieval and final RMSNorm run at every
  processed position, while the LM head and the atomic argmax token gate
  run only at the ``G`` prediction positions;
* each generated token feeds the next embedding lookup (greedy argmax,
  lowest-token-id tie-break, fixed ``G``, EOS termination disabled in the
  topology).

MoE executed-work semantics
===========================

MoE layers use **executed-work semantics**: all 896 router scores (sigmoid
scores plus correction-bias additions and the selected mixture-weight
normalization) are counted per token per MoE layer, but only the 16
selected routed experts' arithmetic plus both shared full-width experts is
counted.  The top-k result is represented by 16 attackable scalar
expert-index outputs per token per MoE layer, each with the exact
``log2(routed_experts)`` alphabet bound. Routing changes induced by
upstream perturbations are a recorded semantics caveat of executed-work
accounting, not a counting error.

Width semantics (``kimi-k3-semantic-mixed``)
============================================

Each region declares one width for all of its gates:

* 32 bits (FP32) for every arithmetic region containing reductions or
  accumulators, including all dot-product/matmul regions, softmax,
  normalization, router scoring, KDA decay gates, and KDA state updates;
* 16 bits (BF16) only for explicit lookup or boundary-only gates;
* ``log2(163840)`` bits for each atomic argmax token gate.

Where a region mixes width classes the larger (sound) width is chosen.
MXFP4/MXFP8 storage formats of routed experts are *not* modeled; arithmetic
regions use the conservative 32-bit internal width.  All counts are exact
integers for the declared semantics; every simplification is recorded in
the profile's ``assumptions`` tuple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)

_BOUNDARY_BITS = 16.0
_INTERNAL_BITS = 32.0
_SUPPORTED_NUMERICAL_PROFILE_ID = "kimi-k3-semantic-mixed"
_DEFAULT_MLA_LAYER_INDICES: tuple[int, ...] = (*range(3, 92, 4), 92)


@dataclass(frozen=True, slots=True)
class KimiK3Config:
    """Released text-only Kimi-K3 dimensions (HF ``config.json`` at a590ce0).

    The default layer schedule repeats KDA,KDA,KDA,MLA through internal
    layer 91 and appends one extra MLA at layer 92; layer 0 carries the only
    dense FFN, and layers 1..92 carry Stable LatentMoE.
    """

    model_id: str = "kimi-k3"
    layers: int = 93
    hidden_size: int = 7168
    vocabulary_size: int = 163_840
    max_context: int = 1_048_576
    mla_layer_indices: tuple[int, ...] = _DEFAULT_MLA_LAYER_INDICES
    dense_ffn_layer_indices: tuple[int, ...] = (0,)
    attn_res_block_size: int = 12
    kda_heads: int = 96
    kda_key_dim: int = 128
    kda_value_dim: int = 128
    kda_conv_kernel: int = 4
    kda_decay_rank: int = 128
    mla_query_heads: int = 96
    mla_q_lora_rank: int = 1536
    mla_kv_lora_rank: int = 512
    mla_qk_content_dim: int = 128
    mla_qk_direct_dim: int = 64
    mla_value_dim: int = 128
    dense_intermediate_size: int = 33_792
    routed_experts: int = 896
    active_routed_experts: int = 16
    shared_experts: int = 2
    routed_latent_width: int = 3584
    expert_intermediate_size: int = 3072
    shared_expert_intermediate_size: int = 3072

    def __post_init__(self) -> None:
        dimensions = (
            self.layers,
            self.hidden_size,
            self.vocabulary_size,
            self.max_context,
            self.attn_res_block_size,
            self.kda_heads,
            self.kda_key_dim,
            self.kda_value_dim,
            self.kda_conv_kernel,
            self.kda_decay_rank,
            self.mla_query_heads,
            self.mla_q_lora_rank,
            self.mla_kv_lora_rank,
            self.mla_qk_content_dim,
            self.mla_value_dim,
            self.dense_intermediate_size,
            self.routed_experts,
            self.active_routed_experts,
            self.routed_latent_width,
            self.expert_intermediate_size,
            self.shared_expert_intermediate_size,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all Kimi-K3 dimensions must be positive")
        if self.mla_qk_direct_dim < 0 or self.shared_experts < 0:
            raise ValueError(
                "direct QK dimension and shared expert count cannot be negative"
            )
        if self.active_routed_experts > self.routed_experts:
            raise ValueError(
                "active routed experts cannot exceed the routed expert count"
            )
        for name, indices in (
            ("mla_layer_indices", self.mla_layer_indices),
            ("dense_ffn_layer_indices", self.dense_ffn_layer_indices),
        ):
            if list(indices) != sorted(set(indices)):
                raise ValueError(f"{name} must be strictly increasing and unique")
            if indices and not 0 <= indices[0] <= indices[-1] < self.layers:
                raise ValueError(f"{name} must lie in [0, {self.layers})")

    @property
    def mla_layer_count(self) -> int:
        return len(self.mla_layer_indices)

    @property
    def kda_layer_count(self) -> int:
        return self.layers - self.mla_layer_count

    @property
    def dense_layer_count(self) -> int:
        return len(self.dense_ffn_layer_indices)

    @property
    def moe_layer_count(self) -> int:
        return self.layers - self.dense_layer_count

    @property
    def mla_qk_total_dim(self) -> int:
        return self.mla_qk_content_dim + self.mla_qk_direct_dim

    @property
    def mla_output_width(self) -> int:
        return self.mla_query_heads * self.mla_value_dim

    @property
    def kda_qk_width(self) -> int:
        return self.kda_heads * self.kda_key_dim

    @property
    def kda_value_width(self) -> int:
        return self.kda_heads * self.kda_value_dim

    @property
    def attn_res_block_count(self) -> int:
        """Number of (possibly partial) AttnRes blocks: ceil(layers / size)."""

        return -(-self.layers // self.attn_res_block_size)


KIMI_K3 = KimiK3Config()


def _dot_product_gates(width: int) -> int:
    """Bias-free dot product of length ``width``: 2*width - 1.

    ``width`` multiplies plus ``width - 1`` accumulator additions.
    """

    return 2 * width - 1


def _rms_norm_gates(dim: int) -> int:
    """Affine RMSNorm over ``dim`` coordinates: 4*dim + 2.

    Multiplies: ``dim`` squares, one 1/dim mean scale, ``dim`` inverse-RMS
    scalings, and ``dim`` gamma scalings (3*dim + 1).  Additions: ``dim - 1``
    sum-of-squares reductions plus one epsilon (dim).  One rsqrt.
    """

    return 4 * dim + 2


def _gamma_free_rms_norm_gates(dim: int) -> int:
    """RMSNorm without an affine weight: 3*dim + 2.

    Used inside AttnRes scoring, where the score vector absorbs any gamma:
    2*dim + 1 multiplies, ``dim`` additions, and one rsqrt.
    """

    return 3 * dim + 2


def _l2_norm_gates(dim: int) -> int:
    """L2 normalization over ``dim`` coordinates: 3*dim + 1.

    ``dim`` squares plus ``dim`` scalings (2*dim multiplies), ``dim - 1``
    reduction additions plus one epsilon (dim additions), and one rsqrt.
    """

    return 3 * dim + 1


def _softmax_gates(count: int) -> int:
    """Max-shifted softmax over ``count`` entries: 5*count - 1.

    ``count - 1`` value-producing max compare/selects, ``count`` shift
    subtractions, ``count`` exponentials, ``count - 1`` denominator
    additions, one reciprocal, and ``count`` probability multiplies.
    """

    return 5 * count - 1


def mla_causal_pair_count(processed_positions: int) -> int:
    """Dense causal (query, key) pairs: sum_{q=0}^{N-1} (q+1) = N*(N+1)/2."""

    if processed_positions < 0:
        raise ValueError("processed position count cannot be negative")
    return processed_positions * (processed_positions + 1) // 2


def _attn_res_retrieval_gates(candidates: int, dim: int) -> int:
    """One AttnRes depth retrieval over ``candidates`` streams of width ``dim``.

    Per candidate: one gamma-free RMSNorm (3*dim + 2) and one length-``dim``
    score dot product (2*dim - 1).  Then one softmax over the candidates
    (5c - 1) and the softmax-weighted stream sum (c*dim multiplies plus
    (c-1)*dim additions).  Closed form: 7*c*dim + 6*c - dim - 1.
    """

    per_candidate = _gamma_free_rms_norm_gates(dim) + _dot_product_gates(dim)
    total = (
        candidates * per_candidate
        + _softmax_gates(candidates)
        + candidates * dim
        + (candidates - 1) * dim
    )
    if total != 7 * candidates * dim + 6 * candidates - dim - 1:
        raise AssertionError("AttnRes retrieval gadget disagrees with its closed form")
    return total


def _attn_res_pre_attention_candidates(config: KimiK3Config, layer: int) -> int:
    """Spec candidate schedule: 0 (identity) at layer 0; 1+b at r=0; else 2+b."""

    if layer == 0:
        return 0
    block, offset = divmod(layer, config.attn_res_block_size)
    if offset == 0:
        return 1 + block
    return 2 + block


def _attn_res_pre_mlp_candidates(config: KimiK3Config, layer: int) -> int:
    """Spec candidate schedule: 2 + b (the partial sum is already populated)."""

    return 2 + layer // config.attn_res_block_size


def _attn_res_gates_per_position(config: KimiK3Config) -> int:
    """All AttnRes retrieval gates at one processed position.

    Sums the exact per-layer pre-attention and pre-MLP retrievals plus the
    final output retrieval over 1 + block-count candidate streams, then
    cross-checks the aggregate against 7*d*C + 6*C - (d+1)*R for ``C`` total
    candidates over ``R`` performed retrievals.
    """

    dim = config.hidden_size
    candidate_counts = [
        count
        for layer in range(config.layers)
        for count in (
            _attn_res_pre_attention_candidates(config, layer),
            _attn_res_pre_mlp_candidates(config, layer),
        )
    ]
    candidate_counts.append(1 + config.attn_res_block_count)
    total = 0
    candidate_total = 0
    retrieval_count = 0
    for count in candidate_counts:
        if count == 0:
            continue
        total += _attn_res_retrieval_gates(count, dim)
        candidate_total += count
        retrieval_count += 1
    expected = (
        7 * dim * candidate_total + 6 * candidate_total - (dim + 1) * retrieval_count
    )
    if total != expected:
        raise AssertionError("AttnRes per-position sum disagrees with its closed form")
    return total


def _kda_projection_gates_per_layer_position(config: KimiK3Config) -> int:
    """Dense KDA matmul gates for one position in one KDA layer.

    Bias-free length-d dot products (2d - 1 gates per output) for W_q and
    W_k (d -> heads*key_dim each), W_v and the full-rank output gate W_g
    (d -> heads*value_dim each), and W_beta (d -> heads); the output
    projection W_o (heads*value_dim -> d, 2*width - 1 per output); the decay
    down-projection (d -> rank); and the biased vector decay up-projection
    (rank -> heads*key_dim, 2*rank gates per coordinate including b_alpha).
    """

    d = config.hidden_size
    return (
        2 * config.kda_qk_width * _dot_product_gates(d)  # W_q and W_k
        + config.kda_value_width * _dot_product_gates(d)  # W_v
        + config.kda_value_width * _dot_product_gates(d)  # output gate W_g
        + d * _dot_product_gates(config.kda_value_width)  # W_o
        + config.kda_decay_rank * _dot_product_gates(d)  # decay down
        + config.kda_qk_width * 2 * config.kda_decay_rank  # biased decay up
        + config.kda_heads * _dot_product_gates(d)  # W_beta
    )


def _kda_conv_gating_gates_per_layer_position(config: KimiK3Config) -> int:
    """Short convolutions, SiLU evaluations, and output gating per position.

    Each of the 2*heads*key_dim + heads*value_dim projected q/k/v channels
    has one causal depthwise width-w convolution (w multiplies plus w - 1
    additions, zero-padded at early positions) and one SiLU (sigmoid plus
    multiply).  The output gate adds one sigmoid and one gating multiply per
    heads*value_dim coordinate.
    """

    channels = 2 * config.kda_qk_width + config.kda_value_width
    return (
        channels * (2 * config.kda_conv_kernel - 1)
        + channels * 2
        + config.kda_value_width * 2
    )


def _kda_norm_gates_per_layer_position(config: KimiK3Config) -> int:
    """Per-head q/k L2 normalizations plus the head-wise output RMSNorm.

    Per head: two L2 normalizations over key_dim (3*dk + 1 each) and one
    affine RMSNorm over value_dim (4*dv + 2) on the recurrent read.
    """

    return config.kda_heads * (
        2 * _l2_norm_gates(config.kda_key_dim) + _rms_norm_gates(config.kda_value_dim)
    )


def _kda_decay_gate_gates_per_layer_position(config: KimiK3Config) -> int:
    """Vector decay and scalar write-strength gates for one KDA layer.

    Every one of ``heads * key_dim`` decay coordinates computes
    ``exp(A)*z``, sigmoid, ``-5`` scaling, and the final exponential producing
    alpha. Beta contributes one sigmoid per head.
    """

    return config.kda_heads * (4 * config.kda_key_dim + 1)


def _kda_state_update_gates_per_layer_position(config: KimiK3Config) -> int:
    """Semantic KDA recurrence per position in one layer: 7*dk*dv + 2*dk - dv
    gates per head, derived term by term from the recurrence.

    Decay application D = Diag(alpha) S_{t-1}: dk*dv multiplies.  Delta error
    e = v - D^T k: dv length-dk dot products (2*dk - 1 each) plus dv
    subtractions = 2*dk*dv.  Outer-product write S = D + beta k e^T: dk
    beta*k multiplies, dk*dv outer multiplies, and dk*dv accumulate
    additions.  Read o = S^T (q / sqrt(dk)): dk query-scale multiplies plus
    dv length-dk dot products = 2*dk*dv + dk - dv.
    """

    dk = config.kda_key_dim
    dv = config.kda_value_dim
    decay = dk * dv
    error = dv * _dot_product_gates(dk) + dv
    write = dk + 2 * dk * dv
    read = dk + dv * _dot_product_gates(dk)
    per_head = decay + error + write + read
    if per_head != 7 * dk * dv + 2 * dk - dv:
        raise AssertionError(
            "KDA state-update decomposition disagrees with its closed form"
        )
    return config.kda_heads * per_head


def _mla_projection_gates_per_layer_position(config: KimiK3Config) -> int:
    """Bias-free MLA matmuls and output gating per position in one MLA layer.

    Query down (d -> q_lora) and up (q_lora -> heads*qk_total); KV down
    (d -> kv_lora + qk_direct, the latent plus the shared direct key); KV up
    (kv_lora -> heads*(qk_content + value)); the full-rank output gate W_g
    (d -> heads*value) with one sigmoid and one gating multiply per
    coordinate; and W_o (heads*value -> d).
    """

    d = config.hidden_size
    q_up_width = config.mla_query_heads * config.mla_qk_total_dim
    kv_down_width = config.mla_kv_lora_rank + config.mla_qk_direct_dim
    kv_up_width = config.mla_query_heads * (
        config.mla_qk_content_dim + config.mla_value_dim
    )
    return (
        config.mla_q_lora_rank * _dot_product_gates(d)
        + q_up_width * _dot_product_gates(config.mla_q_lora_rank)
        + kv_down_width * _dot_product_gates(d)
        + kv_up_width * _dot_product_gates(config.mla_kv_lora_rank)
        + config.mla_output_width * _dot_product_gates(d)  # output gate W_g
        + config.mla_output_width * 2  # gate sigmoid and multiply
        + d * _dot_product_gates(config.mla_output_width)  # W_o
    )


def _mla_latent_norm_gates_per_layer_position(config: KimiK3Config) -> int:
    """Affine RMSNorms on the q_lora and kv_lora latents (reference code)."""

    return _rms_norm_gates(config.mla_q_lora_rank) + _rms_norm_gates(
        config.mla_kv_lora_rank
    )


def _mla_attention_gates(config: KimiK3Config, processed_positions: int) -> int:
    """All MLA score/softmax/value-reduction gates across the execution.

    Per (query, head) with ``n`` visible keys (current token included):
    ``n`` score gates of 2*qk_total each (the length-qk_total dot product,
    2*qk_total - 1, plus one 1/sqrt(qk_total) scale multiply); one softmax
    (5n - 1); and the value reduction (value_dim * (2n - 1)).  Closed form
    over n = 1..N with T = N*(N+1)/2 causal pairs:

        layers * heads * ((2*qk_total + 5 + 2*value_dim) * T
                          - (1 + value_dim) * N)

    cross-checked against explicit per-position enumeration.
    """

    pairs = mla_causal_pair_count(processed_positions)
    per_head = (
        2 * config.mla_qk_total_dim * pairs
        + (5 * pairs - processed_positions)
        + config.mla_value_dim * (2 * pairs - processed_positions)
    )
    enumerated = 0
    for visible in range(1, processed_positions + 1):
        enumerated += (
            visible * 2 * config.mla_qk_total_dim
            + _softmax_gates(visible)
            + config.mla_value_dim * (2 * visible - 1)
        )
    if per_head != enumerated:
        raise AssertionError("MLA causal closed form disagrees with enumeration")
    return config.mla_layer_count * config.mla_query_heads * per_head


def _moe_router_gates_per_layer_position(config: KimiK3Config) -> int:
    """Executed-work router gates per token: E*(2d + 1) + 2A + 1.

    All ``E`` routed scores (one length-d dot product, 2d - 1, plus one
    sigmoid each) and ``E`` correction-bias additions for selection, then
    the selected mixture-weight normalization: A - 1 denominator additions,
    one 1e-20 epsilon addition, one reciprocal, and ``A`` multiplies.
    """

    scores = config.routed_experts * (_dot_product_gates(config.hidden_size) + 1)
    bias = config.routed_experts
    normalization = (
        (config.active_routed_experts - 1) + 1 + 1 + config.active_routed_experts
    )
    return scores + bias + normalization


def _moe_latent_projection_gates_per_layer_position(config: KimiK3Config) -> int:
    """Routed-path latent matmuls per token: down (d -> latent) and the
    post-aggregation up-projection (latent -> d), both bias-free."""

    return config.routed_latent_width * _dot_product_gates(
        config.hidden_size
    ) + config.hidden_size * _dot_product_gates(config.routed_latent_width)


def _moe_aggregation_gates_per_layer_position(config: KimiK3Config) -> int:
    """Mixture-weighted routed sum plus the post-aggregation RMSNorm.

    Per latent coordinate: A multiplies and A - 1 additions (2A - 1); then
    one affine RMSNorm over the latent width (4*latent + 2).
    """

    return config.routed_latent_width * (
        2 * config.active_routed_experts - 1
    ) + _rms_norm_gates(config.routed_latent_width)


def _moe_routed_expert_gates_per_layer_position(config: KimiK3Config) -> int:
    """Matmul gates of the A executed routed experts per token.

    Per expert: W_gate and W_up (latent -> intermediate, 2*latent - 1 gates
    per output each) plus the w2 down-projection (intermediate -> latent),
    which report Eq. 11 requires even though HF revision a590ce0 appears to
    omit the call.  SiTU activation gates live in the situ-activations
    region.
    """

    per_expert = 2 * config.expert_intermediate_size * _dot_product_gates(
        config.routed_latent_width
    ) + config.routed_latent_width * _dot_product_gates(config.expert_intermediate_size)
    return config.active_routed_experts * per_expert


def _moe_shared_expert_gates_per_layer_position(config: KimiK3Config) -> int:
    """Both full-width shared experts per token.

    Per shared expert: W_gate and W_up (d -> shared intermediate), the w2
    down-projection (shared intermediate -> d), and one addition per hidden
    coordinate merging the expert output into the FFN output.
    """

    per_expert = 2 * config.shared_expert_intermediate_size * _dot_product_gates(
        config.hidden_size
    ) + config.hidden_size * _dot_product_gates(config.shared_expert_intermediate_size)
    return config.shared_experts * (per_expert + config.hidden_size)


def _dense_ffn_gates_per_layer_position(config: KimiK3Config) -> int:
    """Dense SiTU-GLU FFN matmuls per token at each dense layer.

    W_gate and W_up (d -> dense intermediate, 2d - 1 gates per output each)
    plus the w2 down-projection (dense intermediate -> d).  SiTU activation
    gates live in the situ-activations region.
    """

    f = config.dense_intermediate_size
    return 2 * f * _dot_product_gates(
        config.hidden_size
    ) + config.hidden_size * _dot_product_gates(f)


def _situ_activation_gates_per_position(config: KimiK3Config) -> int:
    """SiTU-GLU activation gates at one position: 9 per intermediate coordinate.

    Gate branch: a/4 multiply, tanh, x4 multiply, sigmoid(a), and the branch
    product (5 gates).  Up branch: u/25 multiply, tanh, x25 multiply
    (3 gates).  One final elementwise product.  Applied to every dense-layer
    intermediate coordinate and, at every MoE layer, to the A executed
    routed experts and all shared experts.
    """

    coordinates = (
        config.dense_layer_count * config.dense_intermediate_size
        + config.moe_layer_count
        * (
            config.active_routed_experts * config.expert_intermediate_size
            + config.shared_experts * config.shared_expert_intermediate_size
        )
    )
    return 9 * coordinates


def build_kimi_k3_capacity_profile(
    prompt_tokens: int = 100,
    generated_tokens: int = 100,
    *,
    config: KimiK3Config = KIMI_K3,
    numerical_profile_id: str = "kimi-k3-semantic-mixed",
) -> ModelCapacityProfile:
    """Build the exact executed-work capacity profile for one greedy run.

    ``N = prompt_tokens + generated_tokens - 1`` backbone positions are
    processed; the LM head and the atomic argmax run at the
    ``generated_tokens`` prediction positions.  Counts are exact integers
    for the documented semantics; widths follow the module-level
    ``kimi-k3-semantic-mixed`` policy.
    """

    if numerical_profile_id != _SUPPORTED_NUMERICAL_PROFILE_ID:
        raise ValueError(
            "unsupported numerical profile "
            f"{numerical_profile_id!r}; only {_SUPPORTED_NUMERICAL_PROFILE_ID!r} "
            "has declared widths"
        )
    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise ValueError("prompt and generated token counts must be positive")
    positions = prompt_tokens + generated_tokens - 1
    if positions > config.max_context:
        raise ValueError(
            "processed positions must fit the declared context window: "
            f"{positions} > {config.max_context}"
        )

    d = config.hidden_size
    kda_layer_positions = config.kda_layer_count * positions
    mla_layer_positions = config.mla_layer_count * positions
    moe_layer_positions = config.moe_layer_count * positions
    dense_layer_positions = config.dense_layer_count * positions
    token_bits = math.log2(config.vocabulary_size)

    regions = (
        CapacityRegion(
            id="embedding-lookup",
            description=(
                "One table-read lookup gate per hidden coordinate for each of "
                f"the {generated_tokens - 1} generated tokens fed back into the "
                f"backbone; fixed-prompt embedding rows are circuit inputs. The "
                f"table has {config.vocabulary_size} rows."
            ),
            gate_count=(generated_tokens - 1) * d,
            self_cut_bits_per_gate=_BOUNDARY_BITS,
        ),
        CapacityRegion(
            id="attn-res-retrieval",
            description=(
                "AttnRes depth retrievals (pre-attention, pre-MLP, and final "
                "output) at every processed position: per candidate one "
                "gamma-free RMSNorm and one score dot product, then softmax and "
                "the weighted stream sum.  Candidate counts follow the spec "
                "schedule: identity at layer 0, 1+b at later block starts, "
                "otherwise 2+b pre-attention; 2+b pre-MLP; "
                f"{1 + config.attn_res_block_count} final candidates."
            ),
            gate_count=positions * _attn_res_gates_per_position(config),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="layer-input-rmsnorm",
            description=(
                "Affine RMSNorm before attention and before the FFN in each of "
                f"the {config.layers} layers at every processed position "
                "(4d + 2 gates each)."
            ),
            gate_count=positions * 2 * config.layers * _rms_norm_gates(d),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="block-partial-sum",
            description=(
                "AttnRes block accumulator additions: each layer adds its "
                "attention output and its FFN output into the current block "
                "partial sum (2 additions per hidden coordinate per layer per "
                "position)."
            ),
            gate_count=positions * 2 * config.layers * d,
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="kda-projections",
            description=(
                f"Dense matmuls in each of the {config.kda_layer_count} KDA "
                f"layers: W_q/W_k (d->{config.kda_qk_width}), W_v and output "
                f"gate W_g (d->{config.kda_value_width}), W_o "
                f"({config.kda_value_width}->d), rank-{config.kda_decay_rank} "
                "decay down/up (up biased per head), and W_beta "
                f"(d->{config.kda_heads})."
            ),
            gate_count=(
                kda_layer_positions * _kda_projection_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="kda-conv-gating",
            description=(
                f"Width-{config.kda_conv_kernel} causal depthwise short "
                "convolutions and SiLU on every projected q/k/v channel, plus "
                "the output-gate sigmoid and gating multiply."
            ),
            gate_count=(
                kda_layer_positions * _kda_conv_gating_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="kda-norms",
            description=(
                "Per-head L2 normalization of post-conv q and k "
                f"(3*{config.kda_key_dim}+1 each) and the per-head affine "
                f"RMSNorm of the recurrent read (4*{config.kda_value_dim}+2)."
            ),
            gate_count=kda_layer_positions * _kda_norm_gates_per_layer_position(config),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="kda-decay-gates",
            description=(
                "Per-coordinate vector decay gates: exp(A)*z multiply, sigmoid, "
                "-5 scale multiply, and exp producing alpha, plus one scalar "
                "beta sigmoid per head."
            ),
            gate_count=(
                kda_layer_positions * _kda_decay_gate_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="kda-state-update",
            description=(
                "Semantic KDA recurrence per position, head, and KDA layer: "
                "decay application Diag(alpha)S (dk*dv), delta error v - D^T k "
                "(2*dk*dv), outer-product write D + beta k e^T (dk + 2*dk*dv), "
                "and the scaled read S^T(q/sqrt(dk)) (dk + 2*dk*dv - dv)."
            ),
            gate_count=(
                kda_layer_positions * _kda_state_update_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="mla-projections",
            description=(
                f"Bias-free matmuls in each of the {config.mla_layer_count} MLA "
                f"layers: q down/up (d->{config.mla_q_lora_rank}->"
                f"{config.mla_query_heads * config.mla_qk_total_dim}), kv down "
                f"(d->{config.mla_kv_lora_rank}+{config.mla_qk_direct_dim} "
                "latent plus shared direct key), kv up "
                f"({config.mla_kv_lora_rank}->"
                f"{config.mla_query_heads * (config.mla_qk_content_dim + config.mla_value_dim)}), "
                "output gate W_g with sigmoid and gating multiply, and W_o "
                f"({config.mla_output_width}->d)."
            ),
            gate_count=(
                mla_layer_positions * _mla_projection_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="mla-latent-rmsnorm",
            description=(
                f"Affine RMSNorms on the q_lora ({config.mla_q_lora_rank}) and "
                f"kv_lora ({config.mla_kv_lora_rank}) latents, per position per "
                "MLA layer (DeepSeek-style reference implementation)."
            ),
            gate_count=(
                mla_layer_positions * _mla_latent_norm_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="mla-attention",
            description=(
                "Dense causal MLA attention: per (query, head) with n visible "
                f"keys, n length-{config.mla_qk_total_dim} score dots with "
                f"1/sqrt({config.mla_qk_total_dim}) scaling, an FP32 softmax "
                f"(5n - 1), and the value reduction ({config.mla_value_dim} * "
                "(2n - 1)); pair counts follow sum_q (q+1) = N(N+1)/2."
            ),
            gate_count=_mla_attention_gates(config, positions),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="moe-router",
            description=(
                f"Executed-work router per token per MoE layer: all "
                f"{config.routed_experts} sigmoid scores over the normalized "
                f"input, {config.routed_experts} correction-bias additions for "
                "selection, and the selected mixture-weight normalization "
                f"({config.active_routed_experts - 1} + 1 additions, one "
                f"reciprocal, {config.active_routed_experts} multiplies)."
            ),
            gate_count=(
                moe_layer_positions * _moe_router_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="moe-top-k-selectors",
            description=(
                f"The top-{config.active_routed_experts} route result as "
                f"{config.active_routed_experts} attackable scalar expert-index "
                f"outputs per token per MoE layer, each over {config.routed_experts} "
                "possible expert IDs."
            ),
            gate_count=moe_layer_positions * config.active_routed_experts,
            self_cut_bits_per_gate=math.log2(config.routed_experts),
            value_cardinality_upper_bound=config.routed_experts,
        ),
        CapacityRegion(
            id="moe-latent-projections",
            description=(
                f"Routed-path latent matmuls per token per MoE layer: down "
                f"(d->{config.routed_latent_width}) and post-aggregation up "
                f"({config.routed_latent_width}->d)."
            ),
            gate_count=(
                moe_layer_positions
                * _moe_latent_projection_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="moe-aggregation",
            description=(
                f"Mixture-weighted sum of the {config.active_routed_experts} "
                f"routed expert outputs (2A - 1 gates per latent coordinate) "
                "plus the post-aggregation affine RMSNorm over the "
                f"{config.routed_latent_width}-wide latent."
            ),
            gate_count=(
                moe_layer_positions * _moe_aggregation_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="moe-routed-experts",
            description=(
                f"Matmuls of the {config.active_routed_experts} executed routed "
                f"experts (of {config.routed_experts}) per token per MoE layer: "
                f"W_gate/W_up ({config.routed_latent_width}->"
                f"{config.expert_intermediate_size}) and the w2 down-projection "
                f"({config.expert_intermediate_size}->"
                f"{config.routed_latent_width}) per report Eq. 11."
            ),
            gate_count=(
                moe_layer_positions
                * _moe_routed_expert_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="moe-shared-experts",
            description=(
                f"All {config.shared_experts} full-width shared experts per "
                f"token per MoE layer: W_gate/W_up (d->"
                f"{config.shared_expert_intermediate_size}), w2 "
                f"({config.shared_expert_intermediate_size}->d), and one "
                "addition per hidden coordinate merging each expert into the "
                "FFN output."
            ),
            gate_count=(
                moe_layer_positions
                * _moe_shared_expert_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="dense-ffn",
            description=(
                f"Dense SiTU-GLU FFN matmuls at the {config.dense_layer_count} "
                f"dense layer(s): W_gate/W_up (d->"
                f"{config.dense_intermediate_size}) and w2 "
                f"({config.dense_intermediate_size}->d)."
            ),
            gate_count=(
                dense_layer_positions * _dense_ffn_gates_per_layer_position(config)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="situ-activations",
            description=(
                "SiTU-GLU activation evaluations, 9 gates per intermediate "
                "coordinate (gate branch: a/4 multiply, tanh, x4 multiply, "
                "sigmoid, branch product; up branch: u/25 multiply, tanh, x25 "
                "multiply; final product), for the dense FFN, the executed "
                "routed experts, and the shared experts."
            ),
            gate_count=positions * _situ_activation_gates_per_position(config),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="final-rmsnorm",
            description=(
                "Final affine RMSNorm after the final output retrieval at every "
                "processed position (4d + 2 gates)."
            ),
            gate_count=positions * _rms_norm_gates(d),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="lm-head",
            description=(
                "Untied bias-free LM head: one length-d dot product per "
                f"vocabulary logit ({config.vocabulary_size} logits) at each of "
                f"the {generated_tokens} prediction positions."
            ),
            gate_count=(
                generated_tokens * config.vocabulary_size * _dot_product_gates(d)
            ),
            self_cut_bits_per_gate=_INTERNAL_BITS,
        ),
        CapacityRegion(
            id="argmax-token",
            description=(
                "One atomic greedy argmax token gate per generated token over "
                f"{config.vocabulary_size} logits (lowest-token-id tie-break)."
            ),
            gate_count=generated_tokens,
            self_cut_bits_per_gate=token_bits,
        ),
    )

    assumptions = (
        (
            "MoE executed-work semantics: all "
            f"{config.routed_experts} router scores (sigmoids plus "
            "correction-bias additions and the selected mixture normalization) "
            "are counted per token per MoE layer, but only the "
            f"{config.active_routed_experts} selected routed experts plus "
            f"{config.shared_experts} shared expert(s) execute and are counted."
        ),
        (
            f"Each top-{config.active_routed_experts} expert-selection result "
            f"is represented by {config.active_routed_experts} attackable "
            f"scalar expert-index gates over {config.routed_experts} values "
            "per token per MoE layer; "
            "tie-breaking logic and other indexing/control arithmetic are not "
            "expanded. Perturbation-induced routing changes remain a caveat of "
            "trace-conditional executed-work accounting."
        ),
        (
            "MXFP4 block scales of routed-expert weights and MXFP8 activation "
            "scales are not modeled; widths use compute dtypes (BF16/FP32 "
            "region classes), the sound larger direction."
        ),
        (
            "KDA chunkwise kernel realization is not modeled; the semantic "
            "per-position recurrence (decay application, delta error, "
            "outer-product write, read contraction) is counted with FP32 "
            "state-update widths."
        ),
        (
            f"KDA short convolutions are counted with all "
            f"{config.kda_conv_kernel} taps at every position (zero-padded "
            f"starts), matching the persistent width-{config.kda_conv_kernel} "
            "convolution state."
        ),
        (
            "All linear projections are counted bias-free except the KDA decay "
            "up-projection, which carries the per-coordinate bias b_alpha declared "
            "by the recurrence."
        ),
        (
            "MLA latent RMSNorms on the q_lora and kv_lora streams follow the "
            "DeepSeek-style reference implementation; the report equations "
            "omit them."
        ),
        (
            "Shared experts are read as full-width SiTU-GLU experts (hidden -> "
            f"{config.shared_expert_intermediate_size} -> hidden each); the "
            "checkpoint publishes no separate shared intermediate width."
        ),
        (
            "Routed experts include the w2 down-projection "
            f"({config.expert_intermediate_size} -> "
            f"{config.routed_latent_width}) required by report Eq. 11, although "
            "HF revision a590ce0 appears to omit that call."
        ),
        (
            "AttnRes retrieval RMSNorms are counted gamma-free because the "
            "score vector absorbs any affine scale; layer-input, latent, "
            "aggregation, and head-wise norms carry affine gamma (4*dim + 2 "
            "gates)."
        ),
        (
            "The final AttnRes retrieval and the final RMSNorm run at all "
            "N = P + G - 1 processed positions; the LM head and argmax run "
            "only at the G prediction positions."
        ),
        (
            "Fixed-prompt embedding rows are circuit inputs; only the G-1 "
            "generated-token feedback lookups are attackable computed lookup "
            "gates."
        ),
        (
            "Block partial-sum accumulators start at zero and each layer "
            "contribution costs one addition per hidden coordinate."
        ),
        (
            "Widths are per-region declarations under kimi-k3-semantic-mixed: "
            "32-bit FP32 for every arithmetic region containing reductions or "
            "accumulators, including matmul/dot-product regions, softmax, "
            "normalization, router, decay gates, and state updates; 16-bit "
            "BF16 only for explicit lookup or boundary-only gates; "
            "log2(vocabulary) for each atomic argmax token gate; regions "
            "mixing classes take the larger width."
        ),
        (
            "Casts, loads, stores, KV/conv-cache appends, and the MLA cache "
            "representation (expanded vs latent) contribute no computed gates."
        ),
        (
            "Each declared scalar arithmetic or activation evaluation is one "
            "computed gate; subtraction counts in the add class."
        ),
        (
            "Greedy argmax generation with lowest-token-id tie-break and fixed "
            "G; EOS termination is disabled in the topology and each token "
            f"gate is one atomic {config.vocabulary_size}-valued primitive."
        ),
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
