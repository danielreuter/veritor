"""Symbolic operation accounting for GPT-2 Small generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from circuit_cut_analysis.accounting import (
    BottleneckRecord,
    ExactPartitionStatus,
    ExecutionAnalysis,
    ModelAnalysis,
    OperationLedgerRecord,
    PrimitiveVector,
    WiringBottleneckRecord,
)
from circuit_cut_analysis.indexed import IndexedCircuit
from circuit_cut_analysis.profiles import VLLM_FP16_REFERENCE, ServingProfile


@dataclass(frozen=True, slots=True)
class GPT2Config:
    model_id: str = "gpt2-small"
    layers: int = 12
    hidden_size: int = 768
    heads: int = 12
    intermediate_size: int = 3072
    vocabulary_size: int = 50_257
    max_context: int = 1024

    def __post_init__(self) -> None:
        dimensions = (
            self.layers,
            self.hidden_size,
            self.heads,
            self.intermediate_size,
            self.vocabulary_size,
            self.max_context,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all GPT-2 dimensions must be positive")
        if self.hidden_size % self.heads:
            raise ValueError("hidden size must be divisible by the number of heads")

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.heads


GPT2_SMALL = GPT2Config()


def _row(
    *,
    row_id: str,
    location: str,
    component: str,
    cut_type: str,
    cut_width: int,
    occurrences: int,
    gates: PrimitiveVector,
    ownership: str,
    profile: ServingProfile,
    assumptions: tuple[str, ...] = (),
    logical_or_materialized: str = "logical",
) -> BottleneckRecord:
    return BottleneckRecord(
        row_id=row_id,
        location=location,
        component=component,
        cut_type=cut_type,
        cut_gate_widths_bits=(cut_width,),
        occurrence_count=occurrences,
        gates_per_occurrence=gates,
        ownership_scope=(
            "Half-open motif region (predecessor cuts excluded; named cut "
            f"included): {ownership}"
        ),
        profile_id=profile.id,
        cut_basis=(
            "declared symbolic candidate; local minimum-cut realizability not evaluated"
        ),
        logical_or_materialized=logical_or_materialized,
        assumptions=assumptions,
    )


def _contraction_flops(config: GPT2Config, context_length: int) -> int:
    d = config.hidden_size
    f = config.intermediate_size
    layers = config.layers
    vocabulary = config.vocabulary_size
    projection_flops = layers * (
        2 * d * (3 * d)  # QKV
        + 2 * d * d  # attention output
        + 2 * d * f  # MLP expansion
        + 2 * f * d  # MLP contraction
    )
    attention_flops = layers * 2 * d * context_length * 2  # QK and PV
    logits_flops = 2 * d * vocabulary
    return projection_flops + attention_flops + logits_flops


def _logical_footprint(value_count: int, width_bits: int) -> dict[str, int]:
    footprint = {
        "values": value_count,
        "bits": value_count * width_bits,
    }
    if width_bits % 8 == 0:
        footprint["bytes"] = value_count * (width_bits // 8)
    return footprint


def expected_decode_step_unit_gate_total(
    context_length: int,
    config: GPT2Config = GPT2_SMALL,
) -> int:
    """Closed form for the default GPT-2 Small dimensions.

    For custom configurations the analysis rows remain authoritative; this
    helper computes their equivalent formula directly.
    """

    return _expected_constant(config) + _expected_context_slope(config) * context_length


def _expected_context_slope(config: GPT2Config) -> int:
    # Per layer and visible key: QK (2k), softmax (5), and PV (2d).
    return config.layers * (
        config.heads * (2 * config.head_size + 5) + config.hidden_size * 2
    )


def _expected_constant(config: GPT2Config) -> int:
    d = config.hidden_size
    f = config.intermediate_size
    layers = config.layers
    heads = config.heads
    vocabulary = config.vocabulary_size
    layer_norms = 2 * layers + 1
    return (
        d  # token + position embedding merge
        + layer_norms * (7 * d + 2)
        + layers * (3 * d) * (2 * d)  # QKV
        - layers * heads  # softmax contributes 5n - 1 per head
        - layers * d  # PV contributes 2n - 1 per output coordinate
        + layers * d * (2 * d)  # attention output projection
        + layers * d * 2  # two residual merges
        + layers * f * (2 * d)  # MLP expansion
        + layers * f * 9  # tanh GELU
        + layers * d * (2 * f)  # MLP contraction
        + vocabulary * (2 * d - 1)  # unbiased LM head
    )


def analyze_gpt2_decode_step(
    context_length: int,
    *,
    config: GPT2Config = GPT2_SMALL,
    profile: ServingProfile = VLLM_FP16_REFERENCE,
) -> ModelAnalysis:
    """Analyze one batch-1 decode token with ``context_length`` visible keys."""

    if not 1 <= context_length <= config.max_context:
        raise ValueError(
            f"context length must be in [1, {config.max_context}], got {context_length}"
        )

    n = context_length
    d = config.hidden_size
    h = config.heads
    k = config.head_size
    f = config.intermediate_size
    layers = config.layers
    vocabulary = config.vocabulary_size
    layer_norms = 2 * layers + 1
    boundary = profile.activation_boundary_bits
    reduction = profile.reduction_bits
    softmax_maximum_rows = (
        (
            _row(
                row_id="softmax-maximum",
                location="attention softmax maximum",
                component="softmax",
                cut_type="shared FP reduction scalar",
                cut_width=reduction,
                occurrences=layers * h,
                gates=PrimitiveVector(max=n - 1),
                ownership="one value-producing max reduction over visible keys",
                profile=profile,
                assumptions=("MAX is one value-producing compare/select primitive.",),
            ),
        )
        if n > 1
        else ()
    )

    rows = (
        _row(
            row_id="embedding-merge",
            location="token + position embedding",
            component="embedding",
            cut_type="scalar activation boundary",
            cut_width=profile.residual_bits,
            occurrences=d,
            gates=PrimitiveVector(add=1),
            ownership="one embedding addition per hidden coordinate",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="layernorm-mean",
            location="LayerNorm mean",
            component="layernorm",
            cut_type="shared FP reduction scalar",
            cut_width=reduction,
            occurrences=layer_norms,
            gates=PrimitiveVector(add=d - 1, mul=1),
            ownership="mean sum and reciprocal-size scaling through the mean scalar",
            profile=profile,
            assumptions=("LayerNorm uses a two-pass mean/variance decomposition.",),
        ),
        _row(
            row_id="layernorm-centered",
            location="LayerNorm centered coordinate",
            component="layernorm",
            cut_type="branching FP intermediate self-cut",
            cut_width=reduction,
            occurrences=layer_norms * d,
            gates=PrimitiveVector(add=1),
            ownership=(
                "one centered value; it branches to both variance and normalization"
            ),
            profile=profile,
            assumptions=("Subtraction is counted in the add primitive class.",),
        ),
        _row(
            row_id="layernorm-variance",
            location="LayerNorm inverse standard deviation",
            component="layernorm",
            cut_type="shared FP reduction scalar",
            cut_width=reduction,
            occurrences=layer_norms,
            gates=PrimitiveVector(add=d, mul=d + 1, rsqrt=1),
            ownership=(
                "squares, variance reduction, reciprocal-size scaling, epsilon, "
                "and rsqrt"
            ),
            profile=profile,
            assumptions=("LayerNorm uses a two-pass mean/variance decomposition.",),
        ),
        _row(
            row_id="layernorm-coordinate-output",
            location="LayerNorm output coordinate",
            component="layernorm",
            cut_type="scalar activation boundary",
            cut_width=boundary,
            occurrences=layer_norms * d,
            gates=PrimitiveVector(add=1, mul=2),
            ownership="normalize, multiply by gamma, and add beta for one coordinate",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="qkv-projection",
            location="QKV projection output coordinate",
            component="attention",
            cut_type="biased dot-product write-out",
            cut_width=boundary,
            occurrences=layers * 3 * d,
            gates=PrimitiveVector(add=d, mul=d),
            ownership="one length-d biased dot product",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="attention-score",
            location="scaled query-key score",
            component="attention",
            cut_type="scaled dot-product output",
            cut_width=boundary,
            occurrences=layers * h * n,
            gates=PrimitiveVector(add=k - 1, mul=k + 1),
            ownership="one length-head-dimension dot product and scale multiply",
            profile=profile,
            assumptions=(
                "The logical score boundary rounds to the activation dtype before softmax.",
            ),
        ),
        *softmax_maximum_rows,
        _row(
            row_id="softmax-exponential",
            location="shifted score and exponential",
            component="softmax",
            cut_type="branching FP exponential value",
            cut_width=profile.nonlinear_internal_bits,
            occurrences=layers * h * n,
            gates=PrimitiveVector(add=1, exp=1),
            ownership=(
                "one score-minus-maximum subtraction and exp; the exp branches "
                "to numerator and denominator"
            ),
            profile=profile,
            assumptions=("Subtraction is counted in the add primitive class.",),
        ),
        _row(
            row_id="softmax-denominator",
            location="attention softmax reciprocal denominator",
            component="softmax",
            cut_type="shared FP reduction scalar",
            cut_width=reduction,
            occurrences=layers * h,
            gates=PrimitiveVector(add=n - 1, reciprocal=1),
            ownership="denominator reduction and reciprocal",
            profile=profile,
        ),
        _row(
            row_id="softmax-probability",
            location="attention probability",
            component="softmax",
            cut_type="scalar probability boundary",
            cut_width=profile.probability_boundary_bits,
            occurrences=layers * h * n,
            gates=PrimitiveVector(mul=1),
            ownership="one exponential-times-reciprocal probability output",
            profile=profile,
            assumptions=(
                "Probabilities round to the declared probability boundary dtype.",
            ),
        ),
        _row(
            row_id="attention-value-reduction",
            location="per-head attention output coordinate",
            component="attention",
            cut_type="unbiased dot-product write-out",
            cut_width=boundary,
            occurrences=layers * d,
            gates=PrimitiveVector(add=n - 1, mul=n),
            ownership="one probability-value reduction over visible keys",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="attention-output-projection",
            location="attention output-projection coordinate",
            component="attention",
            cut_type="biased dot-product write-out",
            cut_width=boundary,
            occurrences=layers * d,
            gates=PrimitiveVector(add=d, mul=d),
            ownership="one length-d biased dot product",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="residual-merges",
            location="attention and MLP residual coordinates",
            component="residual",
            cut_type="scalar activation boundary",
            cut_width=profile.residual_bits,
            occurrences=2 * layers * d,
            gates=PrimitiveVector(add=1),
            ownership="one residual addition",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="mlp-expansion",
            location="MLP expansion coordinate",
            component="mlp",
            cut_type="biased dot-product write-out",
            cut_width=boundary,
            occurrences=layers * f,
            gates=PrimitiveVector(add=d, mul=d),
            ownership="one length-d biased dot product",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="gelu",
            location="MLP GELU output coordinate",
            component="mlp",
            cut_type="scalar activation boundary",
            cut_width=boundary,
            occurrences=layers * f,
            gates=PrimitiveVector(add=2, mul=6, tanh=1),
            ownership="GPT-2 gelu_new tanh approximation for one coordinate",
            profile=profile,
            assumptions=("GPT-2 uses the gelu_new tanh approximation.",),
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="mlp-contraction",
            location="MLP contraction coordinate",
            component="mlp",
            cut_type="biased dot-product write-out",
            cut_width=boundary,
            occurrences=layers * d,
            gates=PrimitiveVector(add=f, mul=f),
            ownership="one length-intermediate-width biased dot product",
            profile=profile,
            logical_or_materialized="materialized",
        ),
        _row(
            row_id="lm-head",
            location="logit coordinate",
            component="output",
            cut_type="unbiased dot-product output",
            cut_width=profile.logit_bits,
            occurrences=vocabulary,
            gates=PrimitiveVector(add=d - 1, mul=d),
            ownership="one length-d unbiased vocabulary projection",
            profile=profile,
            logical_or_materialized="materialized",
        ),
    )

    total = sum(row.represented_unit_gates for row in rows)
    expected = expected_decode_step_unit_gate_total(n, config)
    if total != expected:
        raise AssertionError(f"row ownership totals {total:,}, expected {expected:,}")

    metadata = {
        "architecture": {
            "layers": layers,
            "hidden_size": d,
            "heads": h,
            "head_size": k,
            "intermediate_size": f,
            "vocabulary_size": vocabulary,
            "max_context": config.max_context,
        },
        "decode": {
            "batch_size": 1,
            "visible_kv_positions": n,
            "preexisting_kv_positions": n - 1,
            "includes_current_token": True,
            "dropout": False,
            "full_logits": True,
        },
        "accounting": {
            "primitive_basis": [
                "add/subtract",
                "multiply",
                "value-producing max compare/select",
                "exp",
                "reciprocal",
                "rsqrt",
                "tanh",
            ],
            "casts_loads_stores_indexing_control_excluded": True,
            "primary_measure": "declared scalar primitive-gate count",
            "work_share": (
                "owned primitive gates divided by all declared primitive gates"
            ),
            "source_selection": (
                "one computed primitive gate at a time, grouped into semantic "
                "motif regions"
            ),
            "ownership": (
                "non-overlapping half-open regions: predecessor cuts excluded, "
                "named cut included"
            ),
            "cut_basis": (
                "declared symbolic candidates; local Transformer-gadget "
                "minimum-cut realizability is not evaluated"
            ),
            "minimum_cut_ties": (
                "not evaluated for symbolic rows; the exact finite-DAG solver "
                "reports algorithmic ties"
            ),
            "contraction_flops": (
                "conventional multiply-add contraction count with one multiply "
                "and one add counted as two FLOPs"
            ),
        },
        "logical_value_footprints": {
            "visible_kv_operands": _logical_footprint(
                layers * 2 * d * n,
                profile.kv_cache_bits,
            ),
            "preexisting_kv_cache": _logical_footprint(
                layers * 2 * d * (n - 1),
                profile.kv_cache_bits,
            ),
            "current_kv_produced": _logical_footprint(
                layers * 2 * d,
                profile.kv_cache_bits,
            ),
            "full_logits": _logical_footprint(vocabulary, profile.logit_bits),
            "interpretation": (
                "logical value footprints, not hardware traffic measurements"
            ),
        },
        "profile": profile.as_dict(),
        "closed_form": {
            "constant": _expected_constant(config),
            "context_slope": _expected_context_slope(config),
            "expression": (
                f"{_expected_constant(config)} + {_expected_context_slope(config)} * n"
            ),
        },
    }
    assumptions = (
        "One batch-1 autoregressive decode token is analyzed.",
        "Context length is the number of visible KV positions, including the current token.",
        "The output is the complete logit vector; sampling and argmax are excluded.",
        "Weights, constants, embedding reads, and the pre-existing KV cache are fixed inputs.",
        "Dropout, hidden positions, casts, loads, stores, indexing, and control are excluded.",
        "Each declared scalar arithmetic operation counts as one primitive gate.",
        "Primitive-gate work share is not contraction FLOPs, latency, or GPU utilization.",
        "Rows partition singleton source locations into non-overlapping half-open motif regions.",
        "Symbolic cuts are declared candidates; local minimum-cut realizability and ties are not asserted.",
        *profile.assumptions,
    )
    return ModelAnalysis(
        model_id=config.model_id,
        profile_id=profile.id,
        context_length=n,
        context_includes_current_token=True,
        output_semantics="full logit vector",
        rows=rows,
        contraction_flops=_contraction_flops(config, n),
        metadata=metadata,
        assumptions=assumptions,
    )


def _lm_head_primitives(config: GPT2Config) -> PrimitiveVector:
    return PrimitiveVector(
        add=config.vocabulary_size * (config.hidden_size - 1),
        mul=config.vocabulary_size * config.hidden_size,
    )


def _body_primitives(config: GPT2Config, context_length: int) -> PrimitiveVector:
    """Primitive counts for one processed position, excluding its LM head."""

    if context_length <= 0:
        raise ValueError("context length must be positive")
    d = config.hidden_size
    h = config.heads
    k = config.head_size
    f = config.intermediate_size
    layers = config.layers
    layer_norms = 2 * layers + 1
    n = context_length
    result = PrimitiveVector(
        add=(
            d
            + layer_norms * (4 * d - 1)
            + layers * 3 * d * d
            + layers * h * n * (k - 1)
            + layers * h * n
            + layers * h * (n - 1)
            + layers * d * (n - 1)
            + layers * d * d
            + 2 * layers * d
            + layers * f * d
            + 2 * layers * f
            + layers * d * f
        ),
        mul=(
            layer_norms * (3 * d + 2)
            + layers * 3 * d * d
            + layers * h * n * (k + 1)
            + layers * h * n
            + layers * d * n
            + layers * d * d
            + layers * f * d
            + 6 * layers * f
            + layers * d * f
        ),
        max=layers * h * (n - 1),
        exp=layers * h * n,
        reciprocal=layers * h,
        rsqrt=layer_norms,
        tanh=layers * f,
    )
    expected = (
        _expected_constant(config)
        - _lm_head_primitives(config).total
        + _expected_context_slope(config) * n
    )
    if result.total != expected:
        raise AssertionError(
            f"body primitive formulas total {result.total:,}, expected {expected:,}"
        )
    return result


def _sum_primitives(
    config: GPT2Config,
    contexts: range,
) -> PrimitiveVector:
    total = PrimitiveVector()
    for context_length in contexts:
        total += _body_primitives(config, context_length)
    return total


def _body_contraction_flops(config: GPT2Config, context_length: int) -> int:
    d = config.hidden_size
    f = config.intermediate_size
    layers = config.layers
    projection_flops = layers * (8 * d * d + 4 * d * f)
    attention_flops = 4 * layers * d * context_length
    return projection_flops + attention_flops


def _lm_head_contraction_flops(config: GPT2Config) -> int:
    return 2 * config.hidden_size * config.vocabulary_size


def _primitive_vector_for_families(
    circuit: IndexedCircuit,
    family_names: set[str],
) -> PrimitiveVector:
    counts: dict[str, int] = {}
    for family_name in family_names:
        family = circuit.families[family_name]
        if family.primitive is not None:
            counts[family.primitive] = counts.get(family.primitive, 0) + family.count
    return PrimitiveVector(
        add=counts.get("add", 0),
        mul=counts.get("mul", 0),
        max=counts.get("max", 0),
        exp=counts.get("exp", 0),
        reciprocal=counts.get("reciprocal", 0),
        rsqrt=counts.get("rsqrt", 0),
        tanh=counts.get("tanh", 0),
        argmax=counts.get("argmax", 0),
    )


def _wiring_bottleneck_rows(
    circuit: IndexedCircuit,
    *,
    config: GPT2Config,
    profile: ServingProfile,
    processed_positions: int,
    generated_tokens: int,
) -> tuple[WiringBottleneckRecord, ...]:
    """Partition primitive families into locally certified wire separators."""

    claimed: set[str] = set()
    rows: list[WiringBottleneckRecord] = []
    local_certificate = (
        "The named boundary is an exact downstream separator for the owned "
        "primitive families in the indexed wire relation. This certifies a "
        "local cut, not global minimality."
    )
    pending_global = (
        "Pending the full succinct-graph minimum-cut computation; a narrower "
        "or equally wide later separator may supersede this local cut."
    )

    def add_row(
        *,
        row_id: str,
        bottleneck: str,
        prefixes: tuple[str, ...] = (),
        exact: tuple[str, ...] = (),
        boundaries: tuple[str, ...],
        width_expression: str,
        width_min: float,
        width_max: float,
        occurrences: int,
        upstream: str,
        certificate: str = local_certificate,
        global_status: str = pending_global,
    ) -> None:
        selected = {
            name
            for name, family in circuit.families.items()
            if family.primitive is not None
            and family.count > 0
            and (name in exact or any(name.startswith(prefix) for prefix in prefixes))
        }
        overlap = claimed.intersection(selected)
        if overlap:
            raise AssertionError(
                f"primitive families assigned to multiple bottlenecks: {sorted(overlap)}"
            )
        represented = _primitive_vector_for_families(circuit, selected)
        if represented.total <= 0:
            raise AssertionError(f"{row_id} owns no primitive gate families")
        claimed.update(selected)
        rows.append(
            WiringBottleneckRecord(
                row_id=row_id,
                bottleneck=bottleneck,
                boundary_families=boundaries,
                cut_width_expression_bits=width_expression,
                cut_width_min_bits=width_min,
                cut_width_max_bits=width_max,
                occurrence_count=occurrences,
                represented_primitives=represented,
                upstream_operations_per_cut=upstream,
                cut_certificate=certificate,
                global_minimum_status=global_status,
            )
        )

    positions = processed_positions
    layers = config.layers
    hidden = config.hidden_size
    heads = config.heads
    head_size = config.head_size
    intermediate = config.intermediate_size
    vocabulary = config.vocabulary_size
    activation = profile.activation_boundary_bits
    residual = profile.residual_bits
    probability = profile.probability_boundary_bits
    kv = profile.kv_cache_bits
    logit = profile.logit_bits
    context_occurrences = layers * heads * positions

    add_row(
        row_id="embedding-add",
        bottleneck="Embedding/residual coordinate",
        exact=("embedding/add",),
        boundaries=("embedding/add",),
        width_expression=str(residual),
        width_min=float(residual),
        width_max=float(residual),
        occurrences=positions * hidden,
        upstream="one token-plus-position embedding addition",
    )
    norm_prefixes = ("blocks/ln1", "blocks/ln2", "final_ln")
    norm_occurrences = (2 * layers + 1) * positions
    add_row(
        row_id="layernorm-mean",
        bottleneck="LayerNorm shared mean scalar",
        exact=tuple(
            f"{prefix}/{stage}"
            for prefix in norm_prefixes
            for stage in ("mean_sum", "mean")
        ),
        boundaries=tuple(f"{prefix}/mean" for prefix in norm_prefixes),
        width_expression=str(profile.reduction_bits),
        width_min=float(profile.reduction_bits),
        width_max=float(profile.reduction_bits),
        occurrences=norm_occurrences,
        upstream=(
            f"one {hidden}-coordinate sum ({hidden - 1} additions) and one "
            "mean-scale multiply"
        ),
    )
    add_row(
        row_id="layernorm-centered-coordinate",
        bottleneck="LayerNorm centered coordinate",
        exact=tuple(f"{prefix}/centered" for prefix in norm_prefixes),
        boundaries=tuple(f"{prefix}/centered" for prefix in norm_prefixes),
        width_expression=str(profile.reduction_bits),
        width_min=float(profile.reduction_bits),
        width_max=float(profile.reduction_bits),
        occurrences=norm_occurrences * hidden,
        upstream=(
            "one subtraction; the centered value branches to the variance "
            "calculation and its coordinate normalization"
        ),
    )
    add_row(
        row_id="layernorm-inverse-std",
        bottleneck="LayerNorm shared inverse-standard-deviation scalar",
        exact=tuple(
            f"{prefix}/{stage}"
            for prefix in norm_prefixes
            for stage in (
                "square",
                "variance_sum",
                "variance",
                "stabilized_variance",
                "inverse_std",
            )
        ),
        boundaries=tuple(f"{prefix}/inverse_std" for prefix in norm_prefixes),
        width_expression=str(profile.reduction_bits),
        width_min=float(profile.reduction_bits),
        width_max=float(profile.reduction_bits),
        occurrences=norm_occurrences,
        upstream=(
            f"{hidden} squares, {hidden - 1} variance-sum additions, one "
            "variance-scale multiply, one epsilon addition, and one rsqrt"
        ),
    )
    add_row(
        row_id="layernorm-coordinate-output",
        bottleneck="LayerNorm coordinate output",
        exact=tuple(
            f"{prefix}/{stage}"
            for prefix in norm_prefixes
            for stage in ("normalized", "affine_scale", "write")
        ),
        boundaries=tuple(f"{prefix}/write" for prefix in norm_prefixes),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=norm_occurrences * hidden,
        upstream=(
            "one inverse-standard-deviation multiply, one gamma multiply, "
            f"and one beta addition ending at a {activation}-bit write"
        ),
    )
    projection_specs = (
        (
            "q-projection-output",
            "Q projection inner-product output",
            "blocks/attention/q_projection/",
            "blocks/attention/q_projection/write",
            activation,
        ),
        (
            "k-cache-entry",
            "K projection/cache entry",
            "blocks/attention/k_projection/",
            "blocks/attention/k_projection/write",
            kv,
        ),
        (
            "v-cache-entry",
            "V projection/cache entry",
            "blocks/attention/v_projection/",
            "blocks/attention/v_projection/write",
            kv,
        ),
    )
    for row_id, bottleneck, prefix, boundary, width in projection_specs:
        add_row(
            row_id=row_id,
            bottleneck=bottleneck,
            prefixes=(prefix,),
            boundaries=(boundary,),
            width_expression=str(width),
            width_min=float(width),
            width_max=float(width),
            occurrences=layers * positions * hidden,
            upstream=(
                f"one length-{hidden} biased inner product: {hidden} multiplies "
                f"and {hidden} accumulator additions, then one {width}-bit write-out"
            ),
        )
    score_occurrences = layers * heads * positions * (positions + 1) // 2
    add_row(
        row_id="attention-score-output",
        bottleneck="Scaled QK inner-product output",
        prefixes=("blocks/attention/score/",),
        boundaries=("blocks/attention/score/write",),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=score_occurrences,
        upstream=(
            f"one length-{head_size} unbiased inner product "
            f"({head_size} multiplies and {head_size - 1} additions), one scale "
            f"multiply, then one {activation}-bit write-out"
        ),
    )
    if positions > 1:
        add_row(
            row_id="softmax-maximum",
            bottleneck="Softmax shared maximum scalar",
            exact=("blocks/attention/softmax/max",),
            boundaries=("blocks/attention/softmax/max",),
            width_expression=str(profile.reduction_bits),
            width_min=float(profile.reduction_bits),
            width_max=float(profile.reduction_bits),
            occurrences=layers * heads * (positions - 1),
            upstream="for visible-key length n>1: one n-1-gate max reduction",
        )
    add_row(
        row_id="softmax-exponential",
        bottleneck="Softmax exponential value",
        exact=(
            "blocks/attention/softmax/shifted",
            "blocks/attention/softmax/exp",
        ),
        boundaries=("blocks/attention/softmax/exp",),
        width_expression=str(profile.nonlinear_internal_bits),
        width_min=float(profile.nonlinear_internal_bits),
        width_max=float(profile.nonlinear_internal_bits),
        occurrences=score_occurrences,
        upstream=(
            "one score-minus-maximum subtraction and one exponential; the "
            "exponential branches to numerator and denominator paths"
        ),
    )
    add_row(
        row_id="softmax-reciprocal-denominator",
        bottleneck="Softmax reciprocal denominator scalar",
        exact=(
            "blocks/attention/softmax/denominator",
            "blocks/attention/softmax/reciprocal",
        ),
        boundaries=("blocks/attention/softmax/reciprocal",),
        width_expression=str(profile.reduction_bits),
        width_min=float(profile.reduction_bits),
        width_max=float(profile.reduction_bits),
        occurrences=context_occurrences,
        upstream=(
            "for visible-key length n: n-1 denominator additions and one reciprocal"
        ),
    )
    add_row(
        row_id="softmax-probability",
        bottleneck="Softmax probability output",
        exact=("blocks/attention/softmax/probability",),
        boundaries=("blocks/attention/softmax/probability",),
        width_expression=str(probability),
        width_min=float(probability),
        width_max=float(probability),
        occurrences=score_occurrences,
        upstream="one exponential-times-reciprocal normalization multiply",
    )
    add_row(
        row_id="attention-value-output",
        bottleneck="Attention value-reduction output",
        prefixes=("blocks/attention/value_reduction/",),
        boundaries=("blocks/attention/value_reduction/write",),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=layers * positions * hidden,
        upstream=(
            f"for visible-key length n: n multiplies and n-1 accumulator "
            f"additions, then one {activation}-bit write-out"
        ),
    )
    add_row(
        row_id="attention-output-projection",
        bottleneck="Attention output-projection inner product",
        prefixes=("blocks/attention/output_projection/",),
        boundaries=("blocks/attention/output_projection/write",),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=layers * positions * hidden,
        upstream=(
            f"one length-{hidden} biased inner product: {hidden} multiplies and "
            f"{hidden} accumulator additions, then one {activation}-bit write-out"
        ),
    )
    add_row(
        row_id="attention-residual",
        bottleneck="Attention residual coordinate",
        exact=("blocks/residual1",),
        boundaries=("blocks/residual1",),
        width_expression=str(residual),
        width_min=float(residual),
        width_max=float(residual),
        occurrences=layers * positions * hidden,
        upstream="one residual addition",
    )
    add_row(
        row_id="mlp-expansion-output",
        bottleneck="MLP expansion inner-product output",
        prefixes=("blocks/mlp/expansion/",),
        boundaries=("blocks/mlp/expansion/write",),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=layers * positions * intermediate,
        upstream=(
            f"one length-{hidden} biased inner product: {hidden} multiplies and "
            f"{hidden} accumulator additions, then one {activation}-bit write-out"
        ),
    )
    add_row(
        row_id="gelu-output",
        bottleneck="GELU output coordinate",
        prefixes=("blocks/mlp/gelu/",),
        boundaries=("blocks/mlp/gelu/write",),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=layers * positions * intermediate,
        upstream="one gelu_new gadget: six multiplies, two additions, and one tanh",
    )
    add_row(
        row_id="mlp-contraction-output",
        bottleneck="MLP contraction inner-product output",
        prefixes=("blocks/mlp/contraction/",),
        boundaries=("blocks/mlp/contraction/write",),
        width_expression=str(activation),
        width_min=float(activation),
        width_max=float(activation),
        occurrences=layers * positions * hidden,
        upstream=(
            f"one length-{intermediate} biased inner product: {intermediate} "
            f"multiplies and {intermediate} accumulator additions, then one "
            f"{activation}-bit write-out"
        ),
    )
    add_row(
        row_id="mlp-residual",
        bottleneck="MLP residual coordinate",
        exact=("blocks/residual2",),
        boundaries=("blocks/residual2",),
        width_expression=str(residual),
        width_min=float(residual),
        width_max=float(residual),
        occurrences=layers * positions * hidden,
        upstream="one residual addition",
    )
    add_row(
        row_id="lm-head-output",
        bottleneck="LM-head inner-product/logit output",
        prefixes=("lm_head/",),
        boundaries=("lm_head/write",),
        width_expression=str(logit),
        width_min=float(logit),
        width_max=float(logit),
        occurrences=generated_tokens * vocabulary,
        upstream=(
            f"one length-{hidden} unbiased inner product: {hidden} multiplies and "
            f"{hidden - 1} additions, then one {logit}-bit logit write-out"
        ),
    )
    token_width = math.log2(vocabulary)
    add_row(
        row_id="argmax-token",
        bottleneck="Atomic greedy token output",
        exact=("output/argmax",),
        boundaries=("output/argmax",),
        width_expression=f"log2({vocabulary})",
        width_min=token_width,
        width_max=token_width,
        occurrences=generated_tokens,
        upstream=f"one atomic argmax over {vocabulary:,} logits",
        certificate=(
            "Each argmax is itself a designated output, so its self-cut is an "
            "exact global canonical cut under the all-gates policy."
        ),
        global_status="Globally computed by the zero-edge output self-cut.",
    )

    primitive_families = {
        name
        for name, family in circuit.families.items()
        if family.primitive is not None and family.count > 0
    }
    missing = primitive_families.difference(claimed)
    if missing:
        raise AssertionError(f"unassigned primitive families: {sorted(missing)}")
    if claimed != primitive_families:
        raise AssertionError(
            "bottleneck ownership does not partition primitive families"
        )
    return tuple(rows)


def analyze_gpt2_execution(
    prompt_tokens: int = 100,
    generated_tokens: int = 100,
    *,
    config: GPT2Config = GPT2_SMALL,
    profile: ServingProfile = VLLM_FP16_REFERENCE,
) -> ExecutionAnalysis:
    """Analyze the selected prefill-plus-greedy-generation circuit.

    The first generated token is produced by the final prefill position.
    Therefore ``generated_tokens`` outputs require only
    ``generated_tokens - 1`` decode forwards.
    """

    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise ValueError("prompt and generated token counts must be positive")
    last_processed_position = prompt_tokens + generated_tokens - 1
    if last_processed_position > config.max_context:
        raise ValueError(
            "last processed position must fit the declared context window: "
            f"{last_processed_position} > {config.max_context}"
        )

    prefill_contexts = range(1, prompt_tokens + 1)
    decode_contexts = range(prompt_tokens + 1, last_processed_position + 1)
    prefill_context_sum = prompt_tokens * (prompt_tokens + 1) // 2
    decode_forward_count = generated_tokens - 1
    decode_context_sum = sum(decode_contexts)

    body_constant = _expected_constant(config) - _lm_head_primitives(config).total
    context_slope = _expected_context_slope(config)
    prefill_body = _sum_primitives(config, prefill_contexts)
    decode_body = _sum_primitives(config, decode_contexts)
    lm_head = _lm_head_primitives(config)

    expected_prefill_body = (
        prompt_tokens * body_constant + context_slope * prefill_context_sum
    )
    expected_decode_body = (
        decode_forward_count * body_constant + context_slope * decode_context_sum
    )
    if prefill_body.total != expected_prefill_body:
        raise AssertionError("prefill body sum disagrees with its closed form")
    if decode_body.total != expected_decode_body:
        raise AssertionError("decode body sum disagrees with its closed form")

    rows: list[OperationLedgerRecord] = [
        OperationLedgerRecord(
            row_id="prefill-transformer-body",
            phase="prefill",
            component="transformer-body",
            occurrence_count=prompt_tokens,
            primitives=prefill_body,
            description=(
                "Causal prompt rows with visible-key lengths 1 through "
                f"{prompt_tokens}; LM-head work is excluded from this row."
            ),
        ),
        OperationLedgerRecord(
            row_id="prefill-final-lm-head",
            phase="prefill",
            component="lm-head",
            occurrence_count=1,
            primitives=lm_head,
            description=(
                "One complete vocabulary projection at the final prompt position."
            ),
        ),
        OperationLedgerRecord(
            row_id="prefill-first-argmax",
            phase="prefill",
            component="output",
            occurrence_count=1,
            primitives=PrimitiveVector(argmax=1),
            description="One atomic greedy argmax producing y1.",
        ),
    ]
    if decode_forward_count:
        rows.extend(
            (
                OperationLedgerRecord(
                    row_id="decode-transformer-body",
                    phase="decode",
                    component="transformer-body",
                    occurrence_count=decode_forward_count,
                    primitives=decode_body,
                    description=(
                        "Autoregressive forwards whose visible-KV lengths run from "
                        f"{prompt_tokens + 1} through {last_processed_position}."
                    ),
                ),
                OperationLedgerRecord(
                    row_id="decode-lm-heads",
                    phase="decode",
                    component="lm-head",
                    occurrence_count=decode_forward_count,
                    primitives=lm_head.scale(decode_forward_count),
                    description="One complete vocabulary projection per decode forward.",
                ),
                OperationLedgerRecord(
                    row_id="decode-argmax",
                    phase="decode",
                    component="output",
                    occurrence_count=decode_forward_count,
                    primitives=PrimitiveVector(argmax=decode_forward_count),
                    description=(
                        "Atomic greedy argmax gates producing y2 through "
                        f"y{generated_tokens}."
                    ),
                ),
            )
        )

    prefill_flops = sum(
        _body_contraction_flops(config, n) for n in prefill_contexts
    ) + _lm_head_contraction_flops(config)
    decode_flops = sum(
        _body_contraction_flops(config, n) + _lm_head_contraction_flops(config)
        for n in decode_contexts
    )
    contraction_flops = prefill_flops + decode_flops

    token_capacity = math.log2(config.vocabulary_size)
    minimum_storage_bits = (config.vocabulary_size - 1).bit_length()
    outputs = [f"y{index}" for index in range(1, generated_tokens + 1)]
    generated_positions = list(
        range(prompt_tokens + 1, prompt_tokens + generated_tokens + 1)
    )
    decode_lengths = list(decode_contexts)
    from circuit_cut_analysis.models.gpt2_circuit import (
        build_gpt2_indexed_circuit,
    )

    indexed_model = build_gpt2_indexed_circuit(
        prompt_tokens,
        generated_tokens,
        config=config,
        profile=profile,
    )
    indexed_circuit = indexed_model.circuit
    from circuit_cut_analysis.models.gpt2_partition import (
        compute_gpt2_canonical_partition,
    )

    canonical_partition = compute_gpt2_canonical_partition(indexed_model)
    bottlenecks = canonical_partition.rows or _wiring_bottleneck_rows(
        indexed_circuit,
        config=config,
        profile=profile,
        processed_positions=last_processed_position,
        generated_tokens=generated_tokens,
    )
    partition_reasons = canonical_partition.reasons
    metadata = {
        "architecture": {
            "layers": config.layers,
            "hidden_size": config.hidden_size,
            "heads": config.heads,
            "head_size": config.head_size,
            "intermediate_size": config.intermediate_size,
            "vocabulary_size": config.vocabulary_size,
            "max_context": config.max_context,
        },
        "schedule": {
            "batch_size": 1,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "designated_outputs": outputs,
            "generated_output_positions": generated_positions,
            "prefill_visible_kv_lengths": {
                "first": 1,
                "last": prompt_tokens,
                "sum": prefill_context_sum,
            },
            "prefill_lm_head_positions": [prompt_tokens],
            "decode_forward_count": decode_forward_count,
            "decode_visible_kv_lengths": decode_lengths,
            "decode_visible_kv_length_sum": decode_context_sum,
            "last_processed_position": last_processed_position,
            "final_generated_position": prompt_tokens + generated_tokens,
            "forward_at_final_generated_position": False,
            "context_includes_current_processed_token": True,
        },
        "cross_step_dependencies": {
            "token_feedback": {
                "relation": (
                    "y_j -> embedding_lookup_add(position=prompt_tokens+j), "
                    "for 1 <= j < generated_tokens"
                ),
                "dynamic_feedback_edges": decode_forward_count * config.hidden_size,
                "embedding_table": "fixed constant",
                "lookup_index": "generated token ID",
            },
            "kv_cache": {
                "key_relation": (
                    "K(layer, position, coordinate) -> every compatible current "
                    "or later attention-score use"
                ),
                "value_relation": (
                    "V(layer, position, coordinate) -> every compatible current "
                    "or later attention-value use"
                ),
                "computed_kv_values": (
                    last_processed_position * config.layers * 2 * config.hidden_size
                ),
                "cache_copy_or_load_gates": False,
            },
        },
        "token_gate": {
            "operation": "atomic greedy argmax",
            "alphabet_cardinality": config.vocabulary_size,
            "semantic_capacity_expression_bits": (f"log2({config.vocabulary_size})"),
            "semantic_capacity_bits": token_capacity,
            "minimum_fixed_width_storage_bits": minimum_storage_bits,
            "possible_runtime_storage_widths_bits": [16, 32, 64],
            "separator_capacity_uses_storage_width": False,
        },
        "capacity_expressions": {
            "token": f"log2({config.vocabulary_size})",
            "fp16_boundary": "16",
            "fp32_internal": "32",
            "two_tokens": f"2*log2({config.vocabulary_size})",
            "token_plus_fp16": f"log2({config.vocabulary_size})+16",
        },
        "accounting": {
            "body_expression": f"{body_constant} + {context_slope} * n",
            "lm_head_unit_gates": lm_head.total,
            "prefill_unit_gates_excluding_argmax": (prefill_body.total + lm_head.total),
            "decode_unit_gates_excluding_argmax": (
                decode_body.total + lm_head.total * decode_forward_count
            ),
            "prefill_contraction_flops": prefill_flops,
            "decode_contraction_flops": decode_flops,
            "casts_loads_stores_indexing_control_excluded": True,
            "prefill_materializes_only_final_position_lm_head": True,
        },
        "indexed_circuit": {
            "representation": (
                "exact indexed computed-source dependency projection with "
                "bidirectional rules"
            ),
            "scalar_gate_count_including_zero-work_boundaries": (
                indexed_circuit.gate_count
            ),
            "primitive_gate_count": indexed_circuit.primitive_gate_count,
            "primitive_counts": dict(indexed_circuit.primitive_counts),
            "gate_family_count": len(indexed_circuit.families),
            "edge_rule_count": len(indexed_circuit.edge_rules),
            "allocated_gate_objects_after_build": (
                indexed_circuit.cache_info().gate_entries
            ),
            "allocated_predecessor_lists_after_build": (
                indexed_circuit.cache_info().predecessor_entries
            ),
            "allocated_successor_lists_after_build": (
                indexed_circuit.cache_info().successor_entries
            ),
            "cache_policy": {
                "maximum_entries_per_cache": indexed_circuit.cache_capacity,
                "maximum_neighbors_per_cached_adjacency": (
                    indexed_circuit.max_cached_adjacency
                ),
                "oversized_adjacencies_are_retained": False,
            },
            "fixed_weights_and_constants": (
                "embedded literals, not vertices in this computed-source projection"
            ),
            "embedding_lookup_abstraction": (
                "one zero-work scalar lookup boundary per output coordinate; "
                "mux/decoder/select internals are outside the primitive basis"
            ),
            "explicit_zero_work_boundaries": (
                "contraction write-outs and embedding lookups"
            ),
        },
        "exact_partition_support": {
            "explicit_api": "partition_gate_cuts",
            "indexed_api": "minimum_vertex_cut_indexed",
            "hierarchical_api": "compute_gpt2_canonical_partition",
            "status": canonical_partition.status.value,
            "operation_ledger_is_a_cut_partition": False,
            "global_partition_rows_reported": bool(canonical_partition.rows),
            "global_partition_rows_partition_primitive_work": bool(
                canonical_partition.rows
            ),
            "global_canonical_partition_computed": (
                canonical_partition.status is ExactPartitionStatus.COMPUTED
            ),
            "solver": canonical_partition.solver,
            "covered_computed_source_gates": (
                canonical_partition.covered_source_gate_count
            ),
            "covered_primitive_gates": canonical_partition.covered_primitive_count,
            "evaluated_index_regions": canonical_partition.evaluated_index_regions,
            "retained_region_descriptors": (
                canonical_partition.retained_region_descriptors
            ),
            "validated_edge_rules": canonical_partition.validated_edge_rule_count,
            "materialized_scalar_nodes": (
                indexed_circuit.gate_count
                if canonical_partition.explicit_partition is not None
                else 0
            ),
        },
        "profile": profile.as_dict(),
    }
    assumptions = (
        "Prompt token IDs, weights, constants, and position embeddings are fixed inputs.",
        "All generated token IDs are designated outputs even when they feed later work.",
        "Generated-token lookup dependencies and computed K/V reuse are exact indexed edge relations.",
        "Only causal visible attention positions are counted.",
        "Prefill computes a complete LM head only at its final prompt position.",
        (
            f"Each greedy argmax is one atomic {config.vocabulary_size:,}-valued "
            "primitive gate."
        ),
        (
            "Bottleneck rows are global canonical cut groups when partition status "
            "is COMPUTED; otherwise their status text is authoritative."
        ),
        *profile.assumptions,
    )
    analysis = ExecutionAnalysis(
        model_id=config.model_id,
        profile_id=profile.id,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        output_semantics="all greedy generated token IDs y1..yG",
        rows=tuple(rows),
        bottlenecks=bottlenecks,
        contraction_flops=contraction_flops,
        partition_status=canonical_partition.status,
        partition_reasons=partition_reasons,
        metadata=metadata,
        assumptions=assumptions,
    )

    expected_total = (
        prefill_body.total
        + decode_body.total
        + generated_tokens * lm_head.total
        + generated_tokens
    )
    if analysis.total_unit_gates != expected_total:
        raise AssertionError("execution rows do not match the independent total")
    if indexed_circuit.primitive_gate_count != expected_total:
        raise AssertionError("indexed circuit does not match independent accounting")
    if dict(indexed_circuit.primitive_counts) != analysis.total_primitives.as_dict():
        raise AssertionError("indexed circuit primitive mix disagrees with accounting")
    return analysis
