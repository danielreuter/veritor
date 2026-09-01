"""Exact executed-work capacity profile for Thinking Machines Inkling.

This module targets the assumption-light :class:`ModelCapacityProfile`
interface: exact per-region computed-gate counts plus declared per-gate value
widths, from which certified self-cut capacity upper bounds follow.  There is
no indexed circuit here; gate counts come from closed-form architecture
formulas with internal cross-check assertions.

Declared execution semantics (the architecture report's "text-only
fixed-circuit semantics"):

* Batch 1, fully rendered text tokens only, greedy decoding, MTP disabled,
  no padding, and no vision or audio tower gates.
* Fixed prompt length ``P > 0`` and generated-token count ``G > 0``.
  Processed positions ``N = P + G - 1``; the first output token is produced
  at prompt position ``P - 1`` and the final output is designated but not
  forwarded.
* The LM head runs only at the ``G`` prediction positions ``P-1 .. P+G-2``
  and computes every **physical** unembedding row (201,024 by default);
  greedy argmax selects over the **logical** output vocabulary (200,058 by
  default) and is one atomic gate per generated token of width
  ``log2(logical vocabulary)``.
* Executed-work MoE semantics per token per sparse layer: all 258 router
  logits are computed, the route-set selection emits six attackable scalar
  expert-index gates with the exact routed-expert alphabet, exactly six
  routed plus two shared experts
  execute their arithmetic, and the joint eight-way weighted combine runs.
* Fixed weights, router correction biases, learned scalar gains, relative
  banks, and the per-position ``tau_q`` length scale are circuit constants
  represented as literals, not computed gates.

Per-gate widths follow the BF16 reference numerical profile: 16 bits for
ordinary weights, activation boundaries, residuals, probabilities, and
logits; 32 bits for dot-product accumulators and RMSNorm/softmax/
short-convolution internal arithmetic.  Regions that mix 16-bit boundary
gates with 32-bit internal gates declare the larger 32-bit width, which is
loose only in the sound direction for self-cut capacity bounds.  Every
further simplification is recorded in the profile's ``assumptions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from circuit_cut_analysis.models.capacity_profile import (
    CapacityRegion,
    ModelCapacityProfile,
)

_BOUNDARY_BITS = 16.0
_ACCUMULATOR_BITS = 32.0
_SUPPORTED_NUMERICAL_PROFILE_ID = "inkling-bf16-reference"


@dataclass(frozen=True, slots=True)
class InklingConfig:
    """Inkling text-backbone dimensions.

    Defaults match the released BF16 checkpoint: 66 decoder layers, hidden
    width 6144, 64 query heads of head dimension 128, global attention on
    layers ``{l : (l + 1) % 6 == 0}`` (11 layers, 8 KV heads, full causal),
    sliding-window attention elsewhere (55 layers, 16 KV heads, window 512
    including the current token), rank-16 relative-position profiles with
    extents 512 (local) and 1024 (global), four causal depthwise kernel-4
    short convolutions per layer, dense SwiGLU MLPs of width 24,576 on
    layers 0-1, sparse MoE on layers 2-65 (256 routed experts top-6 plus 2
    shared experts, expert intermediate width 3072, 258 router logits), and
    a physical vocabulary of 201,024 rows with logical output vocabulary
    200,058.
    """

    model_id: str = "inkling"
    layers: int = 66
    hidden_size: int = 6144
    query_heads: int = 64
    head_dim: int = 128
    global_layer_period: int = 6
    local_kv_heads: int = 16
    global_kv_heads: int = 8
    window: int = 512
    relative_rank: int = 16
    local_relative_extent: int = 512
    global_relative_extent: int = 1024
    sconv_kernel: int = 4
    dense_layers: int = 2
    dense_intermediate: int = 24_576
    routed_experts: int = 256
    routed_top_k: int = 6
    shared_experts: int = 2
    router_logits: int = 258
    expert_intermediate: int = 3072
    physical_vocabulary: int = 201_024
    logical_vocabulary: int = 200_058
    max_context: int = 1_048_576

    def __post_init__(self) -> None:
        dimensions = (
            self.layers,
            self.hidden_size,
            self.query_heads,
            self.head_dim,
            self.global_layer_period,
            self.local_kv_heads,
            self.global_kv_heads,
            self.window,
            self.relative_rank,
            self.local_relative_extent,
            self.global_relative_extent,
            self.sconv_kernel,
            self.dense_intermediate,
            self.routed_experts,
            self.routed_top_k,
            self.shared_experts,
            self.router_logits,
            self.expert_intermediate,
            self.physical_vocabulary,
            self.logical_vocabulary,
            self.max_context,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all Inkling dimensions must be positive")
        if not 0 <= self.dense_layers <= self.layers:
            raise ValueError("dense layer count must lie in [0, layers]")
        if self.query_heads % self.local_kv_heads:
            raise ValueError("query heads must be divisible by local KV heads")
        if self.query_heads % self.global_kv_heads:
            raise ValueError("query heads must be divisible by global KV heads")
        if self.routed_top_k > self.routed_experts:
            raise ValueError("top-k cannot exceed the routed expert count")
        if self.router_logits != self.routed_experts + self.shared_experts:
            raise ValueError("router logits must equal routed plus shared experts")
        if self.logical_vocabulary < 2:
            raise ValueError("logical vocabulary needs at least two tokens")
        if self.logical_vocabulary > self.physical_vocabulary:
            raise ValueError("logical vocabulary cannot exceed physical rows")

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def relative_width(self) -> int:
        return self.query_heads * self.relative_rank

    @property
    def local_kv_width(self) -> int:
        return self.local_kv_heads * self.head_dim

    @property
    def global_kv_width(self) -> int:
        return self.global_kv_heads * self.head_dim

    @property
    def global_layer_indices(self) -> tuple[int, ...]:
        indices = tuple(
            layer
            for layer in range(self.layers)
            if (layer + 1) % self.global_layer_period == 0
        )
        if len(indices) != self.layers // self.global_layer_period:
            raise AssertionError("global layer schedule disagrees with its count")
        return indices

    @property
    def global_layer_count(self) -> int:
        return len(self.global_layer_indices)

    @property
    def local_layer_count(self) -> int:
        return self.layers - self.global_layer_count

    @property
    def sparse_layer_count(self) -> int:
        return self.layers - self.dense_layers

    @property
    def active_experts(self) -> int:
        """Experts whose arithmetic executes per token: top-k routed + shared."""

        return self.routed_top_k + self.shared_experts


INKLING = InklingConfig()


def local_attention_pair_count(positions: int, window: int) -> int:
    """Exact allowed (query, key) pairs per sliding-window layer and head.

    The window includes the current token, so query ``q`` sees
    ``min(q + 1, window)`` keys.  Summation is an explicit exact-integer
    loop, cross-checked against the closed form.
    """

    if positions < 0:
        raise ValueError("position count cannot be negative")
    if window <= 0:
        raise ValueError("window must be positive")
    total = sum(min(q + 1, window) for q in range(positions))
    if positions <= window:
        closed_form = positions * (positions + 1) // 2
    else:
        closed_form = window * (window + 1) // 2 + (positions - window) * window
    if total != closed_form:
        raise AssertionError("windowed pair sum disagrees with its closed form")
    return total


def global_attention_pair_count(positions: int) -> int:
    """Exact allowed causal (query, key) pairs per global layer and head."""

    if positions < 0:
        raise ValueError("position count cannot be negative")
    total = sum(q + 1 for q in range(positions))
    if total != positions * (positions + 1) // 2:
        raise AssertionError("causal pair sum disagrees with its closed form")
    return total


def _contraction_gates(length: int) -> int:
    """Bias-free length-n dot product: n multiplies plus n-1 accumulator adds."""

    return 2 * length - 1


def _rmsnorm_gates(width: int) -> int:
    """One RMSNorm instance: 4 * width + 2 gates.

    width squares, width-1 sum adds, one mean-scale multiply, one epsilon
    add, one rsqrt, then one inverse-rms multiply and one gamma multiply per
    coordinate.
    """

    return 4 * width + 2


def _swiglu_gates(hidden: int, intermediate: int) -> int:
    """One SwiGLU block, excluding any external output scaling.

    Gate and up projections (bias-free length-``hidden`` dot products), SiLU
    as one sigmoid primitive plus one multiply per intermediate coordinate,
    the gating elementwise multiply, and the down projection.
    """

    return (
        2 * intermediate * _contraction_gates(hidden)
        + 2 * intermediate
        + intermediate
        + hidden * _contraction_gates(intermediate)
    )


def build_inkling_capacity_profile(
    prompt_tokens: int = 100,
    generated_tokens: int = 100,
    *,
    config: InklingConfig = INKLING,
    numerical_profile_id: str = _SUPPORTED_NUMERICAL_PROFILE_ID,
) -> ModelCapacityProfile:
    """Build the exact executed-work capacity profile for one execution.

    All gate counts are exact integers for the declared semantics in the
    module docstring; per-position attention pair counts use explicit
    exact-integer summation over the processed positions.
    """

    if numerical_profile_id != _SUPPORTED_NUMERICAL_PROFILE_ID:
        raise ValueError(
            "unsupported numerical profile "
            f"{numerical_profile_id!r}; only "
            f"{_SUPPORTED_NUMERICAL_PROFILE_ID!r} has declared widths"
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
    heads = config.query_heads
    head_dim = config.head_dim
    layers = config.layers
    local_layers = config.local_layer_count
    global_layers = config.global_layer_count
    sparse_layers = config.sparse_layer_count
    active = config.active_experts
    local_pairs = local_attention_pair_count(positions, config.window)
    global_pairs = global_attention_pair_count(positions)

    softmax_local_direct = sum(
        5 * min(q + 1, config.window) - 1 for q in range(positions)
    )
    if softmax_local_direct != 5 * local_pairs - positions:
        raise AssertionError("local softmax row sum disagrees with the pair form")
    value_local_direct = sum(
        head_dim * (2 * min(q + 1, config.window) - 1) for q in range(positions)
    )
    if value_local_direct != head_dim * (2 * local_pairs - positions):
        raise AssertionError("local value reduction disagrees with the pair form")

    # Local scores omit log-length scaling. Global scores include tau.
    local_score_pair_gates = _contraction_gates(head_dim) + 2
    global_score_pair_gates = _contraction_gates(head_dim) + 3
    qk_norm_instances_per_position = (
        layers * heads
        + local_layers * config.local_kv_heads
        + global_layers * config.global_kv_heads
    )
    sconv_elements_per_position = local_layers * (
        2 * config.local_kv_width + 2 * d
    ) + global_layers * (2 * config.global_kv_width + 2 * d)
    relative_extent_rows = (
        local_layers * config.local_relative_extent
        + global_layers * config.global_relative_extent
    )
    router_gates_per_token = (
        config.router_logits * _contraction_gates(d)  # router logits
        + config.router_logits  # one sigmoid per logit
        + config.routed_experts  # correction-bias adds for selection
        + (active - 1)  # normalization denominator adds
        + 1  # denominator reciprocal
        + 2 * active  # per-active normalize and 8g-scale multiplies
        + d * (2 * active - 1)  # joint eight-way weighted combine
    )
    expert_gates = _swiglu_gates(d, config.expert_intermediate)

    regions = (
        CapacityRegion(
            id="embedding-lookup",
            description=(
                "One table-read lookup gate per hidden coordinate for each of "
                f"the {generated_tokens - 1} generated tokens fed back into the "
                "backbone; fixed-prompt embedding rows are circuit inputs. "
                "BF16 boundary width."
            ),
            gate_count=(generated_tokens - 1) * d,
            self_cut_bits_per_gate=_BOUNDARY_BITS,
        ),
        CapacityRegion(
            id="embedding-rmsnorm",
            description=(
                f"Embedding-row RMSNorm of width {d} at each of the "
                f"{positions} processed positions, separate from any feedback "
                "lookup gate. 32-bit RMSNorm internals dominate the 16-bit "
                "output boundary."
            ),
            gate_count=positions * _rmsnorm_gates(d),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="pre-branch-rmsnorms",
            description=(
                f"Two pre-branch RMSNorms of width {d} per layer per "
                "position (attention branch and MLP/MoE branch). 32-bit "
                "internals dominate the 16-bit output boundary."
            ),
            gate_count=2 * layers * positions * _rmsnorm_gates(d),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="qk-head-rmsnorms",
            description=(
                f"Per-head width-{head_dim} RMSNorms on every query head and "
                "on every post-sconv key head (16 KV heads local, 8 global) "
                "per position. 32-bit internals."
            ),
            gate_count=(
                positions * qk_norm_instances_per_position * _rmsnorm_gates(head_dim)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="q-projections",
            description=(
                f"Query projections: {config.query_width} bias-free "
                f"length-{d} dot products per layer per position. 32-bit "
                "accumulator gates; the 16-bit write-out cast is excluded."
            ),
            gate_count=(
                layers * positions * config.query_width * _contraction_gates(d)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="kv-projections-local",
            description=(
                f"K and V projections on the {local_layers} sliding-window "
                f"layers: 2 x {config.local_kv_width} bias-free length-{d} "
                "dot products per layer per position. 32-bit accumulators."
            ),
            gate_count=(
                local_layers
                * positions
                * 2
                * config.local_kv_width
                * _contraction_gates(d)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="kv-projections-global",
            description=(
                f"K and V projections on the {global_layers} global layers: "
                f"2 x {config.global_kv_width} bias-free length-{d} dot "
                "products per layer per position. 32-bit accumulators."
            ),
            gate_count=(
                global_layers
                * positions
                * 2
                * config.global_kv_width
                * _contraction_gates(d)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="r-projections",
            description=(
                f"Relative projections R: {config.relative_width} bias-free "
                f"length-{d} dot products per layer per position "
                f"({config.relative_rank} coordinates per query head). "
                "32-bit accumulators."
            ),
            gate_count=(
                layers * positions * config.relative_width * _contraction_gates(d)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="short-convolutions",
            description=(
                "Four causal depthwise kernel-4 short convolutions per layer "
                "(K stream, V stream, attention output, MLP output): "
                f"{2 * config.sconv_kernel} gates per output element "
                f"({config.sconv_kernel} multiplies, "
                f"{config.sconv_kernel - 1} kernel-sum adds, one internal "
                "residual add). 32-bit FP32 sconv internals."
            ),
            gate_count=(
                positions * 2 * config.sconv_kernel * sconv_elements_per_position
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="relative-position-profiles",
            description=(
                "Relative-position profiles rho: one bias-free "
                f"length-{config.relative_rank} contraction per (layer, "
                "position, query head, delta) over the full extent "
                f"({config.local_relative_extent} local, "
                f"{config.global_relative_extent} global), as materialized "
                "by the reference implementation. 32-bit accumulators."
            ),
            gate_count=(
                positions
                * heads
                * relative_extent_rows
                * _contraction_gates(config.relative_rank)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-scores-local",
            description=(
                f"Sliding-window attention scores: per allowed pair, one "
                f"length-{head_dim} QK dot product, one 1/{head_dim} scale "
                "multiply and one relative-bias add. Local layers do not "
                "execute log-length tau scaling. Allowed keys at query q: min(q+1, "
                f"{config.window}). 32-bit accumulators dominate the 16-bit "
                "score boundary."
            ),
            gate_count=(local_layers * heads * local_pairs * local_score_pair_gates),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-scores-global",
            description=(
                f"Global attention scores: per causal pair, one "
                f"length-{head_dim} QK dot product, one 1/{head_dim} scale "
                "multiply, one relative-bias add (zero beyond the extent), "
                "and one tau multiply. Allowed keys at query q: q+1. 32-bit "
                "accumulators dominate the 16-bit score boundary."
            ),
            gate_count=(global_layers * heads * global_pairs * global_score_pair_gates),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-softmax-local",
            description=(
                "Sliding-window softmax rows in max-shifted form: for n "
                "visible keys, n-1 max gates, n shifted subtracts, n exps, "
                "n-1 denominator adds, one reciprocal, and n probability "
                "multiplies (5n - 1 gates). 32-bit FP32 softmax internals "
                "dominate the 16-bit probability boundary."
            ),
            gate_count=local_layers * heads * (5 * local_pairs - positions),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-softmax-global",
            description=(
                "Global softmax rows in max-shifted form over all causal "
                "keys (5n - 1 gates for n keys). 32-bit FP32 softmax "
                "internals dominate the 16-bit probability boundary."
            ),
            gate_count=global_layers * heads * (5 * global_pairs - positions),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-value-reductions-local",
            description=(
                "Sliding-window probability-value reductions: per head "
                f"output coordinate ({head_dim} per head) with n visible "
                "keys, n multiplies and n-1 accumulator adds. 32-bit "
                "accumulators."
            ),
            gate_count=(
                local_layers * heads * head_dim * (2 * local_pairs - positions)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-value-reductions-global",
            description=(
                "Global probability-value reductions over all causal keys: "
                f"per head output coordinate ({head_dim} per head), n "
                "multiplies and n-1 accumulator adds. 32-bit accumulators."
            ),
            gate_count=(
                global_layers * heads * head_dim * (2 * global_pairs - positions)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="attention-output-projections",
            description=(
                f"Attention output projections: {d} bias-free "
                f"length-{config.query_width} dot products per layer per "
                "position. 32-bit accumulators."
            ),
            gate_count=(
                layers * positions * d * _contraction_gates(config.query_width)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="residual-merges",
            description=(
                "Two residual additions per layer per position per hidden "
                "coordinate (post-sconv attention merge and post-sconv "
                "MLP/MoE merge); the sconv-internal residual adds are "
                "counted in the short-convolution region. 16-bit residual "
                "boundary values."
            ),
            gate_count=2 * layers * positions * d,
            self_cut_bits_per_gate=_BOUNDARY_BITS,
        ),
        CapacityRegion(
            id="dense-mlp",
            description=(
                f"Dense SwiGLU MLPs on the first {config.dense_layers} "
                f"layers: gate/up projections of width "
                f"{config.dense_intermediate}, SiLU (one sigmoid plus one "
                "multiply per coordinate), gating multiply, down "
                "projection, and one learned scalar gain multiply per "
                "output coordinate. 32-bit accumulators dominate the "
                "16-bit boundaries."
            ),
            gate_count=(
                config.dense_layers
                * positions
                * (_swiglu_gates(d, config.dense_intermediate) + d)
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="moe-router-and-combine",
            description=(
                f"Per token per sparse layer: {config.router_logits} router "
                f"logits (bias-free length-{d} dot products), one sigmoid "
                f"per logit, {config.routed_experts} correction-bias adds "
                f"for selection, weight normalization over the {active} "
                "active experts (denominator adds, one reciprocal, one "
                "normalization multiply and one fixed 8g scale multiply "
                "each), and the joint eight-way weighted output combine. "
                "32-bit internals."
            ),
            gate_count=sparse_layers * positions * router_gates_per_token,
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="moe-top6-selectors",
            description=(
                f"The Top{config.routed_top_k}Select route set as "
                f"{config.routed_top_k} attackable scalar expert-index outputs "
                f"per token per sparse layer, each over {config.routed_experts} "
                "routed-expert IDs."
            ),
            gate_count=sparse_layers * positions * config.routed_top_k,
            self_cut_bits_per_gate=math.log2(config.routed_experts),
            value_cardinality_upper_bound=config.routed_experts,
        ),
        CapacityRegion(
            id="moe-routed-experts",
            description=(
                f"Executed routed experts: {config.routed_top_k} generic "
                "SwiGLU expert slots per token per sparse layer, "
                f"intermediate width {config.expert_intermediate}; the "
                "route result selects the weight bank. 32-bit accumulators."
            ),
            gate_count=(sparse_layers * positions * config.routed_top_k * expert_gates),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="moe-shared-experts",
            description=(
                f"Shared experts: {config.shared_experts} fixed SwiGLU "
                "expert slots per token per sparse layer, intermediate "
                f"width {config.expert_intermediate}. 32-bit accumulators "
                "(the reference reduces shared experts in FP32)."
            ),
            gate_count=(
                sparse_layers * positions * config.shared_experts * expert_gates
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="final-rmsnorm",
            description=(
                f"Final RMSNorm of width {d} at every processed position "
                "(executed work; only prediction positions feed the LM "
                "head). 32-bit internals."
            ),
            gate_count=positions * _rmsnorm_gates(d),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="lm-head",
            description=(
                f"LM head at each of the {generated_tokens} prediction "
                f"positions: one 1/24 pre-scale multiply per hidden "
                f"coordinate plus {config.physical_vocabulary} physical "
                f"bias-free length-{d} dot products. 32-bit accumulators "
                "dominate the 16-bit logit boundary."
            ),
            gate_count=(
                generated_tokens
                * (d + config.physical_vocabulary * _contraction_gates(d))
            ),
            self_cut_bits_per_gate=_ACCUMULATOR_BITS,
        ),
        CapacityRegion(
            id="argmax-tokens",
            description=(
                "One atomic greedy argmax gate per generated token over the "
                f"{config.logical_vocabulary} logical logits; width "
                "log2(logical vocabulary)."
            ),
            gate_count=generated_tokens,
            self_cut_bits_per_gate=math.log2(config.logical_vocabulary),
        ),
    )

    assumptions = (
        "Text-only fixed-circuit semantics: batch 1, greedy argmax decoding, "
        "MTP disabled, no padding, and no vision or audio tower gates.",
        "Processed positions N = prompt + generated - 1 = "
        f"{positions}; the LM head and argmax run only at the "
        f"{generated_tokens} prediction positions.",
        "Fixed-prompt embedding rows are circuit inputs; only the G-1 "
        "generated-token feedback lookups are attackable computed lookup gates.",
        "The LM head counts every physical unembedding row "
        f"({config.physical_vocabulary}) per prediction position; each "
        "greedy argmax is one atomic gate over the logical vocabulary "
        f"({config.logical_vocabulary}) of width log2(logical).",
        "Executed-work MoE semantics; each route-set selector emits "
        f"{config.routed_top_k} attackable scalar expert-index gates over "
        f"{config.routed_experts} routed-expert IDs per token per sparse "
        "layer. All router logits and exactly top-k routed plus all shared "
        "experts execute.",
        "NVFP4 storage not modeled; BF16 compute profile: 16-bit weights, "
        "boundaries, residuals, probabilities, and logits; 32-bit "
        "dot-product accumulators and RMSNorm/softmax/sconv internals.",
        "Relative-bias additions are counted for every scored pair, including "
        "where the bias is constant zero. Log-length tau scaling contributes "
        "one multiply per global-attention pair only; local layers skip it.",
        "Weights, router correction biases, learned scalar gains, relative "
        "banks, and the per-position tau_q scalar are fixed circuit "
        "constants, not computed gates.",
        "Relative profiles rho are materialized over the full relative "
        "extent per (layer, position, query head) as in the reference "
        "implementation, even where fewer key offsets are reachable.",
        "Short convolutions count 2 * kernel gates per output element "
        "(kernel multiplies, kernel - 1 sum adds, one internal residual "
        "add); zero-padded warm-up positions are counted at full kernel "
        "width.",
        "RMSNorm instances count 4 * width + 2 gates: squares, sum, mean "
        "scaling, epsilon add, rsqrt, and per-coordinate inverse-rms and "
        "gamma multiplies.",
        "SiLU is one sigmoid primitive plus one multiply per coordinate; "
        "softmax rows use the max-shifted form (5n - 1 gates for n keys).",
        "Router weighting uses the mathematical sigmoid-normalization form "
        "(denominator adds, one reciprocal, one normalization multiply and "
        "one fixed 8g scale multiply per active expert); the reference's "
        "logsumexp-stabilized rewrite is treated as numerically equivalent.",
        "The final RMSNorm is counted at all N processed positions as "
        "executed work, although only prediction positions feed the LM "
        "head.",
        "Casts, loads, stores, indexing, KV/sconv cache traffic, control "
        "flow and fixed-prompt embedding lookup are excluded; regions "
        "mixing 16-bit boundaries with 32-bit internals declare the larger "
        "32-bit width, which is loose only in the sound direction.",
    )

    return ModelCapacityProfile(
        model_id=config.model_id,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        logical_vocabulary_size=config.logical_vocabulary,
        numerical_profile_id=numerical_profile_id,
        regions=regions,
        assumptions=assumptions,
    )
