"""Structural accounting for a no-op-padded inference envelope.

The experiment asks what happens when completion lengths and MoE routes are not
given to ``G`` as advice and inference is not replayed in ``G``.  A
maximum-length, all-expert envelope whose inactive alternatives are explicit
no-ops is the direct construction under the current prototype primitive set.
The abstract theorem can also admit approved coarse deterministic functions
that the prototype does not implement.

All Kimi-2.5 quantities in this module are occurrence counts.  Gate counts stay
symbolic unless the caller supplies all five gate coefficients.  In particular,
serialization or definition-DAG compression can reduce bytes and records but
cannot reduce the conceptual ``n`` used by a capacity bound.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence


class EnvelopeError(ValueError):
    """A no-op-envelope input or accounting request is malformed."""


TOKEN_ID_BYTES = 4
INSTRUCTION_BYTES = 11
CELL_BYTES = 4
HASH_BYTES = 32
CELL_BITS = 32


def _require_int(value: int, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise EnvelopeError(f"{name} must be a {qualifier} integer")


def _require_probability(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise EnvelopeError(f"{name} must be a finite probability in [0, 1]")


@dataclass(frozen=True)
class WorkloadShape:
    """Finite-epoch workload and MoE architecture shape."""

    requests: int
    actual_output_tokens_per_request: int
    max_output_tokens_per_request: int
    prefill_recomputation_tokens: int
    moe_layers: int
    routed_experts: int
    top_k_experts: int

    def __post_init__(self) -> None:
        _require_int(self.requests, "requests", minimum=1)
        _require_int(
            self.actual_output_tokens_per_request,
            "actual_output_tokens_per_request",
            minimum=1,
        )
        _require_int(
            self.max_output_tokens_per_request,
            "max_output_tokens_per_request",
            minimum=1,
        )
        _require_int(
            self.prefill_recomputation_tokens,
            "prefill_recomputation_tokens",
            minimum=0,
        )
        _require_int(self.moe_layers, "moe_layers", minimum=1)
        _require_int(self.routed_experts, "routed_experts", minimum=1)
        _require_int(self.top_k_experts, "top_k_experts", minimum=1)
        if (
            self.actual_output_tokens_per_request
            > self.max_output_tokens_per_request
        ):
            raise EnvelopeError("actual output tokens cannot exceed the maximum")
        if self.top_k_experts > self.routed_experts:
            raise EnvelopeError("top_k_experts cannot exceed routed_experts")


@dataclass(frozen=True)
class SamplingMerkleParameters:
    """Bernoulli sampling and direct-opening representation parameters."""

    p: float = 1e-9
    instruction_bytes: int = INSTRUCTION_BYTES
    cell_bytes: int = CELL_BYTES
    hash_bytes: int = HASH_BYTES
    instruction_depth: int = 35
    value_depth: int = 35
    arity: int = 2

    def __post_init__(self) -> None:
        _require_probability(self.p, "p")
        _require_int(self.instruction_bytes, "instruction_bytes", minimum=1)
        _require_int(self.cell_bytes, "cell_bytes", minimum=1)
        _require_int(self.hash_bytes, "hash_bytes", minimum=1)
        _require_int(self.instruction_depth, "instruction_depth", minimum=0)
        _require_int(self.value_depth, "value_depth", minimum=0)
        _require_int(self.arity, "arity", minimum=0)


@dataclass(frozen=True)
class GateCoefficients:
    """Optional conceptual gate counts for one occurrence of each operation."""

    g_common: int | None = None
    g_mask: int | None = None
    g_router: int | None = None
    g_expert: int | None = None
    g_branch: int | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None:
                _require_int(value, name, minimum=0)

    @property
    def complete(self) -> bool:
        return all(value is not None for value in asdict(self).values())


@dataclass(frozen=True)
class EnvelopeCounts:
    """Exact occurrence counts for one workload shape."""

    length_advice_comparison_bits: int
    actual_decode_tokens: int
    max_decode_tokens: int
    padded_decode_tokens: int
    response_padding_bytes: int
    materialized_token_cell_padding_bytes: int
    server_to_receiver_padding_bytes_if_fixed_width_wire: int
    server_to_receiver_padding_bytes_if_local_canonicalization: int
    actual_model_tokens: int
    envelope_model_tokens: int
    total_model_token_inflation_tokens: int
    total_model_token_inflation_fraction: float
    total_model_token_inflation_ratio: float
    active_moe_layer_token_occurrences: int
    padded_moe_layer_token_occurrences: int
    envelope_moe_layer_token_occurrences: int
    active_top_k_calls: int
    unselected_branches_on_active_tokens: int
    all_branches_on_padded_tokens: int
    total_noop_expert_branches: int
    total_envelope_expert_branches: int
    noop_fraction: float
    noop_branches_per_useful_expert_call: float
    expansion_over_useful_expert_calls: float


@dataclass(frozen=True)
class StrategyCompatibility:
    """Compatibility disposition for one structural strategy."""

    name: str
    compatibility_label: str
    admissible_under_user_constraints: bool
    compatible_with_current_prototype_primitives: bool
    compatible_with_abstract_theorem: bool
    length_advice: str
    route_advice: str
    explanation: str


ANCHOR_WORKLOAD = WorkloadShape(
    requests=128,
    actual_output_tokens_per_request=900,
    max_output_tokens_per_request=1_024,
    prefill_recomputation_tokens=912_224,
    moe_layers=60,
    routed_experts=384,
    top_k_experts=8,
)

DEFAULT_SAMPLING_MERKLE = SamplingMerkleParameters()

STRATEGIES = (
    StrategyCompatibility(
        name="flat_guarded_all_experts",
        compatibility_label="direct_current_prototype_construction",
        admissible_under_user_constraints=True,
        compatible_with_current_prototype_primitives=True,
        compatible_with_abstract_theorem=True,
        length_advice="zero",
        route_advice="zero",
        explanation=(
            "The direct construction under the current prototype primitive "
            "set: canonical maximum-length padding and every expert alternative "
            "guarded. It is compatible but structurally huge, not the only "
            "construction the abstract theorem could admit."
        ),
    ),
    StrategyCompatibility(
        name="expert_at_public_primitive",
        compatibility_label=(
            "conditionally_abstract_theorem_compatible_public_primitive"
        ),
        admissible_under_user_constraints=True,
        compatible_with_current_prototype_primitives=False,
        compatible_with_abstract_theorem=True,
        length_advice="zero with maximum-length padding",
        route_advice="zero",
        explanation=(
            "Compatible with the abstract theorem if immutable public model "
            "weights are embedded in or bound to an approved deterministic "
            "model-specific function and the selected expert ID is a runtime "
            "router output. It can execute top-8 without 384 explicit branches, "
            "but its cost and static read list remain fixed and it needs a "
            "realistic coarse gate/leaf definition."
        ),
    ),
    StrategyCompatibility(
        name="dynamic_expert_lookup",
        compatibility_label=(
            "requires_authenticated_indexed_memory_lookup_extension"
        ),
        admissible_under_user_constraints=True,
        compatible_with_current_prototype_primitives=False,
        compatible_with_abstract_theorem=False,
        length_advice="zero with maximum-length padding",
        route_advice="zero",
        explanation=(
            "For mutable or input-resident weights, a runtime-selected expert "
            "requires authenticated indexed memory/lookup semantics beyond the "
            "current prototype and the theorem as currently instantiated."
        ),
    ),
    StrategyCompatibility(
        name="route_specialized",
        compatibility_label="inadmissible_no_route_advice",
        admissible_under_user_constraints=False,
        compatible_with_current_prototype_primitives=True,
        compatible_with_abstract_theorem=True,
        length_advice="zero with maximum-length padding",
        route_advice="required and trace-dependent",
        explanation=(
            "Inadmissible under the user's no-route-advice constraint even "
            "though a route-specialized straight-line circuit can be smaller."
        ),
    ),
    StrategyCompatibility(
        name="replay_in_g",
        compatibility_label="inadmissible_no_replay_in_g",
        admissible_under_user_constraints=False,
        compatible_with_current_prototype_primitives=True,
        compatible_with_abstract_theorem=True,
        length_advice="zero by replay",
        route_advice="zero by replay",
        explanation=(
            "Inadmissible because the user explicitly forbids replaying "
            "inference in G."
        ),
    ),
)

TOKEN_ENVELOPE_SEMANTICS = {
    "frame": (
        "C consumes a fixed request-major R-by-M canonical frame; each slot "
        "contains a token cell and constrained activity/control values. The "
        "server-to-receiver wire representation need not itself be fixed-width "
        "if the protocol permits deterministic receiver-side canonicalization."
    ),
    "initial_activity": "active_0 = 1 for each admitted request",
    "stop_derivation": (
        "stop_t is derived from the candidate token and the public stop/EOS "
        "policy; it is not supplied as advice or a free witness."
    ),
    "monotone_chain": "active_(t+1) = active_t AND NOT stop_t",
    "output_select": "output_t = Select(active_t, candidate_t, canonical_PAD)",
    "state_freeze": (
        "state_(t+1) = Select(active_(t+1), proposed_state_(t+1), state_t)"
    ),
    "forced_cap": (
        "The public cap forces termination at M even if no earlier stop fires."
    ),
    "mask_status": (
        "Activity, EOS, stop, and select masks are constrained runtime circuit "
        "values, not advice and not free witnesses."
    ),
    "length_visibility": (
        "The EOS/PAD boundary keeps completion length receiver-visible in y*, "
        "not in a. A public Pad(y*) can make C max-shaped and independent of "
        "that length, but framing or compression cannot erase the length the "
        "receiver already observed."
    ),
}

PROTOCOL_INVARIANTS = {
    "zero_cost_noops": (
        "Even a semantically zero-cost no-op occupies a conceptual position in "
        "n and remains in the sampling population."
    ),
    "guard_static_cost_and_reads": (
        "A false runtime guard does not shrink the approved function's static "
        "cost or static read list."
    ),
    "all_named_reads_opened": (
        "Every read named by a sampled instruction or approved function is "
        "opened and checked, including reads behind a false guard."
    ),
    "one_zero_position_limit": (
        "Reusing one committed zero position handles only statically known "
        "outputs; runtime-dependent output positions still require constrained "
        "Select operations."
    ),
    "definition_compression": (
        "Definition or serialization compression does not change conceptual n."
    ),
    "live_only_sampling": (
        "Sampling only live gates is a theorem/protocol extension, not an "
        "encoding trick. It must always constrain mask/control gates and the "
        "full dependency closure needed to justify liveness."
    ),
}

IMPLEMENTATION_SCOPE = {
    "current_prototype": {
        "path": "src/veritor/machine.py",
        "model": (
            "append-only SSA straight-line instructions from a finite primitive "
            "registry"
        ),
        "max_arity": 2,
        "instruction_bytes": 11,
        "limitation": (
            "No model-specific expert_at primitive or authenticated indexed "
            "memory exists."
        ),
    },
    "draft_v4_abstract_theorem": {
        "model": (
            "arbitrary approved deterministic functions with protocol-fixed "
            "semantics, cost, and static read declarations"
        ),
        "distinction": (
            "Abstract admissibility does not imply support in machine.py and "
            "does not waive a realistic coarse gate/leaf cost definition."
        ),
    },
}

RUNTIME_FAILURE_HANDLING = {
    "availability_fixed_in_x_or_epoch": {
        "advice_bits": 0,
        "disposition": (
            "Zero advice: the availability map and deterministic reroute policy "
            "are fixed in x for the epoch."
        ),
    },
    "post_x_authenticated_failure": {
        "advice_bits": "zero only when canonical reroute follows authenticated input",
        "disposition": (
            "A post-x failure requires an authenticated event input plus a "
            "canonical reroute, or else the resulting choice is charged as advice."
        ),
    },
    "self_asserted_failure": {
        "advice_bits": "not free",
        "disposition": (
            "A prover's self-asserted failure cannot be treated as a free input "
            "or free structural choice."
        ),
    },
}

SP1_CALIBRATION = {
    "source": "/Users/danielreuter/projects/sp1-op-bench/results/report.md",
    "standalone_noop_execution": {
        "cycles": 4_886,
        "gas": 6_859,
        "evidence_kind": "measured_execution",
        "source_location": "iteration 1, section A",
    },
    "sha_precompile_path_folding": {
        "cycles_per_level_approx": 1_620,
        "evidence_kind": "measured_execution_summary",
        "source_location": "iteration 2, section E",
    },
    "large_batch_cpu_marginal": {
        "cycles_per_second_approx": 105_000,
        "evidence_kind": "measured_proving_throughput_summary",
        "source_location": "iteration 3, section I",
    },
}

FORMULAS = {
    "length_advice_comparison_bits": "ceil(log2(M^R))",
    "actual_decode_tokens": "R * A",
    "max_decode_tokens": "R * M",
    "padded_decode_tokens": "R * (M - A)",
    "response_padding_bytes": "4 * R * (M - A)",
    "materialized_token_cell_padding_bytes": "4 * R * (M - A)",
    "server_to_receiver_padding_bytes_if_fixed_width_wire": "4 * R * (M - A)",
    "server_to_receiver_padding_bytes_if_local_canonicalization": "0",
    "actual_model_tokens": "P + R * A",
    "envelope_model_tokens": "P + R * M",
    "active_moe_layer_token_occurrences": "L_moe * (P + R * A)",
    "active_top_k_calls": "L_moe * (P + R * A) * k",
    "unselected_branches_on_active_tokens": (
        "L_moe * (P + R * A) * (E - k)"
    ),
    "all_branches_on_padded_tokens": "L_moe * R * (M - A) * E",
    "total_noop_expert_branches": (
        "L_moe * ((P + R * A) * (E - k) + R * (M - A) * E)"
    ),
}


def ceil_log2(value: int) -> int:
    """Return the exact integer ceiling of log2(value)."""

    _require_int(value, "value", minimum=1)
    return (value - 1).bit_length()


def length_advice_comparison_bits(max_length: int, requests: int) -> int:
    """Exact ``ceil(log2(M**R))`` comparison for unpadded lengths."""

    _require_int(max_length, "max_length", minimum=1)
    _require_int(requests, "requests", minimum=1)
    return ceil_log2(pow(max_length, requests))


def envelope_counts(workload: WorkloadShape) -> EnvelopeCounts:
    """Compute exact token, MoE occurrence, and no-op-branch counts."""

    actual_decode = (
        workload.requests * workload.actual_output_tokens_per_request
    )
    max_decode = workload.requests * workload.max_output_tokens_per_request
    padded_decode = max_decode - actual_decode
    materialized_padding_bytes = padded_decode * TOKEN_ID_BYTES
    actual_model = workload.prefill_recomputation_tokens + actual_decode
    envelope_model = workload.prefill_recomputation_tokens + max_decode

    active_layer_tokens = actual_model * workload.moe_layers
    padded_layer_tokens = padded_decode * workload.moe_layers
    envelope_layer_tokens = envelope_model * workload.moe_layers
    active_calls = active_layer_tokens * workload.top_k_experts
    unselected_active = active_layer_tokens * (
        workload.routed_experts - workload.top_k_experts
    )
    all_padded = padded_layer_tokens * workload.routed_experts
    noop_branches = unselected_active + all_padded
    all_branches = envelope_layer_tokens * workload.routed_experts

    return EnvelopeCounts(
        length_advice_comparison_bits=length_advice_comparison_bits(
            workload.max_output_tokens_per_request, workload.requests
        ),
        actual_decode_tokens=actual_decode,
        max_decode_tokens=max_decode,
        padded_decode_tokens=padded_decode,
        response_padding_bytes=materialized_padding_bytes,
        materialized_token_cell_padding_bytes=materialized_padding_bytes,
        server_to_receiver_padding_bytes_if_fixed_width_wire=(
            materialized_padding_bytes
        ),
        server_to_receiver_padding_bytes_if_local_canonicalization=0,
        actual_model_tokens=actual_model,
        envelope_model_tokens=envelope_model,
        total_model_token_inflation_tokens=padded_decode,
        total_model_token_inflation_fraction=padded_decode / actual_model,
        total_model_token_inflation_ratio=envelope_model / actual_model,
        active_moe_layer_token_occurrences=active_layer_tokens,
        padded_moe_layer_token_occurrences=padded_layer_tokens,
        envelope_moe_layer_token_occurrences=envelope_layer_tokens,
        active_top_k_calls=active_calls,
        unselected_branches_on_active_tokens=unselected_active,
        all_branches_on_padded_tokens=all_padded,
        total_noop_expert_branches=noop_branches,
        total_envelope_expert_branches=all_branches,
        noop_fraction=noop_branches / all_branches,
        noop_branches_per_useful_expert_call=noop_branches / active_calls,
        expansion_over_useful_expert_calls=all_branches / active_calls,
    )


def conceptual_gate_accounting(
    counts: EnvelopeCounts,
    coefficients: GateCoefficients | None = None,
) -> dict[str, object]:
    """Return symbolic gate formulas and optional caller-supplied evaluation."""

    selected = coefficients if coefficients is not None else GateCoefficients()
    total_multipliers = {
        "g_common": counts.envelope_model_tokens,
        "g_mask": counts.max_decode_tokens,
        "g_router": counts.envelope_moe_layer_token_occurrences,
        "g_expert": counts.total_envelope_expert_branches,
        "g_branch": counts.total_envelope_expert_branches,
    }
    total_term_values = {
        name: (
            multiplier * getattr(selected, name)
            if getattr(selected, name) is not None
            else None
        )
        for name, multiplier in total_multipliers.items()
    }
    if selected.complete:
        g_common = int(selected.g_common)
        g_mask = int(selected.g_mask)
        g_router = int(selected.g_router)
        g_expert = int(selected.g_expert)
        g_branch = int(selected.g_branch)
        useful_body = (
            counts.actual_model_tokens * g_common
            + counts.active_top_k_calls * g_expert
        )
        noop_body = (
            counts.padded_decode_tokens * g_common
            + counts.padded_moe_layer_token_occurrences * g_router
            + counts.total_noop_expert_branches * g_expert
        )
        always_live_control = (
            counts.max_decode_tokens * g_mask
            + counts.active_moe_layer_token_occurrences * g_router
            + counts.total_envelope_expert_branches * g_branch
        )
        total_conceptual = sum(int(value) for value in total_term_values.values())
        if useful_body + noop_body + always_live_control != total_conceptual:
            raise AssertionError("gate decomposition must equal the envelope total")
        live_plus_control = useful_body + always_live_control
    else:
        useful_body = None
        noop_body = None
        always_live_control = None
        total_conceptual = None
        live_plus_control = None

    return {
        "coefficient_semantics": {
            "g_common": "common-path gates per model token",
            "g_mask": (
                "active/EOS/select control gates per maximum decode slot"
            ),
            "g_router": "router/top-k gates per MoE layer-token",
            "g_expert": "expert-body gates per expert alternative",
            "g_branch": (
                "guard/select/aggregation gates per expert alternative"
            ),
        },
        "coefficients": asdict(selected),
        "formula": {
            "total_conceptual_gates": (
                "T_envelope*g_common + D_max*g_mask + "
                "A_envelope*g_router + "
                "A_envelope*E*(g_expert+g_branch)"
            ),
            "useful_body_gates": (
                "T_actual*g_common + C_useful*g_expert"
            ),
            "noop_body_gates": (
                "P_decode*g_common + A_padded*g_router + B_noop*g_expert"
            ),
            "always_live_control_gates": (
                "D_max*g_mask + A_active*g_router + B_all*g_branch"
            ),
            "decomposition": (
                "total = useful_body + noop_body + always_live_control"
            ),
        },
        "total_multipliers": total_multipliers,
        "total_term_values": total_term_values,
        "evaluated": {
            "useful_body_gates": useful_body,
            "noop_body_gates": noop_body,
            "always_live_control_gates": always_live_control,
            "n_live_plus_control": live_plus_control,
            "N0_noop_body": noop_body,
            "total_conceptual_gates": total_conceptual,
        },
        "complete": selected.complete,
        "note": (
            "Padded common/router bodies and all inactive expert bodies remain "
            "conceptual no-op gates. Mask, active-router, branch-select, and "
            "aggregation gates are always-live control. No Kimi coefficients "
            "are assumed, and compression cannot reduce conceptual n."
        ),
    }


def expected_sampled_noop_gates(p: float, noop_gates: int) -> float:
    """Return ``p * N0`` for independent Bernoulli gate sampling."""

    _require_probability(p, "p")
    _require_int(noop_gates, "noop_gates", minimum=0)
    return float(p) * noop_gates


def challenged_unit_probability(p: float, gates_in_unit: int) -> float:
    """Stable evaluation of ``1 - (1 - p)**m``."""

    _require_probability(p, "p")
    _require_int(gates_in_unit, "gates_in_unit", minimum=0)
    if gates_in_unit == 0 or p == 0:
        return 0.0
    if p == 1:
        return 1.0
    return -math.expm1(gates_in_unit * math.log1p(-float(p)))


def direct_merkle_opening_bytes(
    parameters: SamplingMerkleParameters,
) -> dict[str, int]:
    """Bytes for independent paths to one instruction, inputs, and output."""

    instruction_opening = (
        parameters.instruction_bytes
        + parameters.instruction_depth * parameters.hash_bytes
    )
    one_value_opening = (
        parameters.cell_bytes + parameters.value_depth * parameters.hash_bytes
    )
    value_openings = parameters.arity + 1
    return {
        "instruction_opening_bytes": instruction_opening,
        "one_value_opening_bytes": one_value_opening,
        "value_openings": value_openings,
        "total_bytes": instruction_opening
        + value_openings * one_value_opening,
    }


def naive_exact_padding_opening_bytes(
    padded_cells: int,
    parameters: SamplingMerkleParameters,
) -> int:
    """Open every post-boundary padding cell with an independent value path."""

    _require_int(padded_cells, "padded_cells", minimum=0)
    one_value = parameters.cell_bytes + (
        parameters.value_depth * parameters.hash_bytes
    )
    return padded_cells * one_value


def range_multiproof_proxy_bytes(
    padding_ranges: int,
    parameters: SamplingMerkleParameters,
) -> int:
    """Proxy for one canonical contiguous zero-range proof per response."""

    _require_int(padding_ranges, "padding_ranges", minimum=0)
    # A canonical interval decomposes into O(depth) subtrees.  Two boundary
    # cells plus at most two hash-frontier entries per level is a transparent
    # scale proxy, not a universal multiproof size or bound.
    per_range = (
        2 * parameters.cell_bytes
        + 2 * parameters.value_depth * parameters.hash_bytes
    )
    return padding_ranges * per_range


def zero_commitment_comparison(
    zero_leaves: int,
    configured_depth: int,
) -> dict[str, object]:
    """Compare dense indexed-zero hashing with an implicit-default capability."""

    _require_int(zero_leaves, "zero_leaves", minimum=0)
    _require_int(configured_depth, "configured_depth", minimum=0)
    if zero_leaves == 0:
        minimum_depth = 0
        effective_depth = configured_depth
        dense_tree_hashes = 0
        default_hashes = effective_depth + 1
    else:
        minimum_depth = ceil_log2(zero_leaves)
        effective_depth = max(configured_depth, minimum_depth)
        dense_tree_hashes = 2 * (1 << effective_depth) - 1
        default_hashes = effective_depth + 1
    return {
        "zero_leaves": zero_leaves,
        "configured_depth": configured_depth,
        "minimum_depth": minimum_depth,
        "configured_depth_sufficient": configured_depth >= minimum_depth,
        "effective_depth_for_comparison": effective_depth,
        "dense_indexed_zero_leaf_hashes": zero_leaves,
        "dense_full_binary_tree_hashes": dense_tree_hashes,
        "default_zero_precomputed_hashes": default_hashes,
        "default_zero_explicit_occurrence_hashes": 0,
        "default_zero_capability_required": True,
        "note": (
            "Indexed dense leaves cannot be removed merely because their values "
            "match. The O(depth) representation requires an authenticated "
            "implicit-default-zero commitment semantics; it is not ordinary "
            "serialization compression."
        ),
    }


def sampling_sweep(
    noop_gates: int,
    unit_sizes: dict[str, int],
    probabilities: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Build a deterministic Bernoulli-p sweep."""

    _require_int(noop_gates, "noop_gates", minimum=0)
    for label, size in unit_sizes.items():
        if not label:
            raise EnvelopeError("sampling unit labels must be nonempty")
        _require_int(size, f"unit_sizes[{label!r}]", minimum=0)
    if probabilities is None:
        scale = max(1, noop_gates)
        probabilities = tuple(
            min(1.0, max(0.0, probability))
            for probability in (0.0, 1 / scale, 10 / scale, 100 / scale, 1e-9)
        )

    normalized: set[float] = set()
    for probability in probabilities:
        _require_probability(probability, "sweep probability")
        normalized.add(float(probability))

    return [
        {
            "p": probability,
            "expected_sampled_noop_gates": expected_sampled_noop_gates(
                probability, noop_gates
            ),
            "challenged_unit_probability": {
                label: challenged_unit_probability(probability, size)
                for label, size in sorted(unit_sizes.items())
            },
        }
        for probability in sorted(normalized)
    ]


def K_B(n: int, limit: int, cell_bits: int = CELL_BITS) -> int:
    """Exact small-case Draft-v4 weighted wrong-gate capacity object."""

    _require_int(n, "n", minimum=0)
    _require_int(limit, "limit", minimum=0)
    _require_int(cell_bits, "cell_bits", minimum=1)
    return sum(
        math.comb(n, j) * (1 << (j * cell_bits))
        for j in range(min(n, limit) + 1)
    )


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _log_k_b(n: int, limit: int, cell_bits: int) -> float:
    """Natural log of ``K_B`` without constructing large combinations."""

    upper = min(n, limit)
    log_terms = [0.0]
    log_combination = 0.0
    for j in range(1, upper + 1):
        log_combination += math.log(n - j + 1) - math.log(j)
        log_terms.append(log_combination + j * cell_bits * math.log(2))
    return _logsumexp(log_terms)


def capacity_bound_penalty_bits(
    n: int,
    noop_gates: int,
    limit: int,
    cell_bits: int = CELL_BITS,
) -> float:
    """Stable weighted ``log2(K_B(n + N0, L, B) / K_B(n, L, B))``."""

    _require_int(n, "n", minimum=1)
    _require_int(noop_gates, "noop_gates", minimum=0)
    _require_int(limit, "limit", minimum=0)
    _require_int(cell_bits, "cell_bits", minimum=1)
    if noop_gates == 0 or limit == 0:
        return 0.0
    penalty = (
        _log_k_b(n + noop_gates, limit, cell_bits)
        - _log_k_b(n, limit, cell_bits)
    ) / math.log(2)
    # Monotonicity is exact; clamp only a possible sub-ulp subtraction artifact.
    return max(0.0, penalty)


def sparse_capacity_penalty_approximation_bits(
    n: int,
    noop_gates: int,
    limit: int,
    cell_bits: int = CELL_BITS,
) -> float:
    """Highest-j approximation ``L * log2(1 + N0 / n)``."""

    _require_int(n, "n", minimum=1)
    _require_int(noop_gates, "noop_gates", minimum=0)
    _require_int(limit, "limit", minimum=0)
    _require_int(cell_bits, "cell_bits", minimum=1)
    if noop_gates == 0 or limit == 0:
        return 0.0
    return limit * math.log1p(noop_gates / n) / math.log(2)


def capacity_bound_penalty(
    n: int,
    noop_gates: int,
    limit: int,
    cell_bits: int = CELL_BITS,
) -> dict[str, object]:
    """Return exact stable and sparse-approximation capacity penalties."""

    exact = capacity_bound_penalty_bits(n, noop_gates, limit, cell_bits)
    approximation = sparse_capacity_penalty_approximation_bits(
        n, noop_gates, limit, cell_bits
    )
    return {
        "n": n,
        "N0": noop_gates,
        "L": limit,
        "cell_bits": cell_bits,
        "K_B_definition": (
            "sum(comb(n, j) * 2**(j*cell_bits), j=0..min(n,L))"
        ),
        "exact_penalty_bits": exact,
        "sparse_approximation_bits": approximation,
        "approximation_formula": "L * log2(1 + N0 / n)",
        "approximation_regime": (
            "The j=L weighted term dominates both K_B sums; the 2**(L*B) "
            "factor then cancels from their ratio."
        ),
        "approximation_minus_exact_bits": approximation - exact,
    }


def sp1_constant_noop_local_check_projection(
    parameters: SamplingMerkleParameters,
) -> dict[str, object]:
    """Derived calibration projection; deliberately not a proof measurement."""

    path_levels = parameters.instruction_depth + parameters.value_depth
    path_cycles = (
        path_levels
        * SP1_CALIBRATION["sha_precompile_path_folding"][
            "cycles_per_level_approx"
        ]
    )
    projected_cycles_with_floor = (
        SP1_CALIBRATION["standalone_noop_execution"]["cycles"] + path_cycles
    )
    cycles_per_second = SP1_CALIBRATION["large_batch_cpu_marginal"][
        "cycles_per_second_approx"
    ]
    return {
        "evidence_kind": "derived_projection_from_separate_measurements",
        "local_check": (
            "one constant no-op check plus one instruction path and one value path"
        ),
        "path_levels": path_levels,
        "path_only_cycles_approx": path_cycles,
        "standalone_program_floor_cycles": (
            SP1_CALIBRATION["standalone_noop_execution"]["cycles"]
        ),
        "projected_cycles_with_conservative_floor_approx": (
            projected_cycles_with_floor
        ),
        "projected_gas": None,
        "path_only_large_batch_cpu_seconds_approx": (
            path_cycles / cycles_per_second
        ),
        "projected_large_batch_cpu_seconds_approx": (
            projected_cycles_with_floor / cycles_per_second
        ),
        "measured_noop_proof": None,
        "is_current_protocol_verifier_cost": False,
        "note": (
            "There is no measured no-op proof. This combines a standalone "
            "execution floor, a SHA path-folding slope, and large-batch CPU "
            "throughput from separate SP1 measurements. Adding the 4,886-cycle "
            "standalone program floor to every check is conservative and is not "
            "a measured marginal no-op cost; this is not the current protocol "
            "verifier cost."
        ),
    }


def _sampling_report(
    workload: WorkloadShape,
    counts: EnvelopeCounts,
    parameters: SamplingMerkleParameters,
) -> dict[str, object]:
    padded_ranges = workload.requests if counts.padded_decode_tokens else 0
    unit_sizes = {
        "active_moe_layer_token_unselected_branches": (
            workload.routed_experts - workload.top_k_experts
        ),
        "padded_decode_token_all_expert_branches": (
            workload.moe_layers * workload.routed_experts
        ),
    }
    return {
        "parameters": asdict(parameters),
        "population": {
            "N0": counts.total_noop_expert_branches,
            "unit": "guarded no-op expert branch",
            "evidence_kind": "structural_branch_unit_proxy",
            "note": (
                "The default numeric sweep treats each no-op expert branch as "
                "one sampling unit. It is not a Kimi gate count. For actual "
                "no-op gates, substitute the caller-evaluated conceptual N0."
            ),
        },
        "bernoulli_formulas": {
            "expected_sampled_noop_gates": "p * N0",
            "challenged_unit_probability": "1 - (1 - p)^m",
        },
        "at_configured_p": {
            "expected_sampled_noop_gates": expected_sampled_noop_gates(
                parameters.p, counts.total_noop_expert_branches
            ),
            "challenged_unit_probability": {
                label: challenged_unit_probability(parameters.p, size)
                for label, size in sorted(unit_sizes.items())
            },
        },
        "p_sweep": sampling_sweep(
            counts.total_noop_expert_branches, unit_sizes
        ),
        "direct_sampled_gate_opening": {
            **direct_merkle_opening_bytes(parameters),
            "formula": (
                "(instruction_bytes + instruction_depth*hash_bytes) + "
                "(arity+1)*(cell_bytes + value_depth*hash_bytes)"
            ),
            "path_sharing": "none",
        },
        "exact_padding_openings": {
            "policy": "open every post-boundary padded response cell",
            "evidence_kind": "exact_for_stated_naive_policy",
            "padded_cells": counts.padded_decode_tokens,
            "bytes": naive_exact_padding_opening_bytes(
                counts.padded_decode_tokens, parameters
            ),
        },
        "range_multiproof": {
            "evidence_kind": "proxy",
            "layout_assumption": (
                "fixed request-major output cells with each request's PAD suffix "
                "contiguous"
            ),
            "available_in_current_index_bound_merkle": False,
            "padding_ranges": padded_ranges,
            "bytes_proxy": range_multiproof_proxy_bytes(
                padded_ranges, parameters
            ),
            "note": (
                "A canonical interval uses O(depth) subtree/frontier hashes. "
                "Actual bytes depend on layout and path sharing; this is not a "
                "measured proof size or a universal bound. The current "
                "index-bound Merkle implementation does not provide this "
                "optimization."
            ),
        },
        "zero_commitment": zero_commitment_comparison(
            counts.total_noop_expert_branches,
            parameters.instruction_depth,
        ),
    }


def padding_communication_semantics(
    counts: EnvelopeCounts,
) -> dict[str, object]:
    """Separate materialized padding from conditional wire transmission."""

    materialized = counts.materialized_token_cell_padding_bytes
    return {
        "legacy_field": {
            "name": "response_padding_bytes",
            "value": counts.response_padding_bytes,
            "meaning": (
                "Exact materialized canonical token-cell padding at four bytes "
                "per cell; not an unconditional wire-communication claim."
            ),
        },
        "materialized_canonical_frame": {
            "padding_token_cells": counts.padded_decode_tokens,
            "bytes": materialized,
            "evidence_kind": "exact",
        },
        "server_to_receiver_wire_modes": {
            "fixed_width_response": {
                "additional_padding_bytes": materialized,
                "condition": (
                    "The server transmits the complete fixed-width canonical "
                    "R-by-M token frame."
                ),
            },
            "receiver_local_canonicalization": {
                "additional_padding_bytes": 0,
                "condition": (
                    "The protocol permits a public deterministic Pad(y*) after "
                    "the receiver has observed ordinary variable-length y*."
                ),
                "effect": (
                    "The receiver synthesizes canonical PAD cells locally; "
                    "completion length remains visible in y*, while C consumes "
                    "a max-shaped canonical frame independent of that length."
                ),
            },
        },
        "boundary_costs": (
            "Local canonicalization does not make boundary commitment, "
            "authentication, or opening costs disappear."
        ),
        "prototype_status": (
            "Whether receiver-local canonicalization is allowed is a "
            "protocol-format choice not established by the current prototype."
        ),
    }


def analyze_noop_envelope(
    workload: WorkloadShape = ANCHOR_WORKLOAD,
    sampling: SamplingMerkleParameters = DEFAULT_SAMPLING_MERKLE,
    gate_coefficients: GateCoefficients | None = None,
    cell_bits: int = CELL_BITS,
) -> dict[str, object]:
    """Analyze one no-op-envelope workload."""

    _require_int(cell_bits, "cell_bits", minimum=1)
    counts = envelope_counts(workload)
    gate_accounting = conceptual_gate_accounting(counts, gate_coefficients)
    capacity = capacity_bound_penalty(
        counts.active_top_k_calls,
        counts.total_noop_expert_branches,
        16,
        cell_bits,
    )
    capacity.update(
        {
            "unit": "expert-branch occurrence",
            "evidence_kind": "structural_branch_unit_projection",
            "note": (
                "This uses useful expert calls as n and no-op expert branches "
                "as N0. It is not a gate-level Kimi result; gate-level use "
                "requires supplied coefficients."
            ),
        }
    )
    evaluated_gates = gate_accounting["evaluated"]
    gate_capacity: dict[str, object] | None
    if (
        gate_accounting["complete"]
        and isinstance(evaluated_gates, dict)
        and int(evaluated_gates["n_live_plus_control"]) > 0
    ):
        gate_capacity = capacity_bound_penalty(
            int(evaluated_gates["n_live_plus_control"]),
            int(evaluated_gates["N0_noop_body"]),
            16,
            cell_bits,
        )
        gate_capacity.update(
            {
                "unit": "conceptual gate",
                "evidence_kind": "caller_parameterized_projection",
            }
        )
    else:
        gate_capacity = None
    return {
        "schema_version": 2,
        "experiment": "focused no-op-envelope structural experiment",
        "scope": {
            "replay_in_g": "forbidden",
            "completion_length_advice": "forbidden",
            "expert_route_advice": "forbidden",
            "claim": "structural occurrence and formula accounting",
            "not_claimed": [
                "Kimi gate counts",
                "latency",
                "GPU work",
                "proof size",
                "current-protocol verifier cost",
            ],
            "implementation_scope": IMPLEMENTATION_SCOPE,
        },
        "workload": asdict(workload),
        "symbols": {
            "R": "requests",
            "A": "actual output tokens per request",
            "M": "maximum output tokens per request",
            "P": "prefill recomputation tokens",
            "L_moe": "MoE layers",
            "E": "routed experts",
            "k": "selected experts",
        },
        "formulas": FORMULAS,
        "counts": asdict(counts),
        "padding_communication_semantics": padding_communication_semantics(
            counts
        ),
        "strategies": [asdict(strategy) for strategy in STRATEGIES],
        "token_envelope_semantics": TOKEN_ENVELOPE_SEMANTICS,
        "protocol_invariants": PROTOCOL_INVARIANTS,
        "conceptual_gate_accounting": gate_accounting,
        "sampling_and_merkle": _sampling_report(
            workload, counts, sampling
        ),
        "capacity_bound_branch_unit_projection": capacity,
        "capacity_bound_gate_projection": gate_capacity,
        "sp1_calibration": SP1_CALIBRATION,
        "sp1_constant_noop_local_check_projection": (
            sp1_constant_noop_local_check_projection(sampling)
        ),
        "runtime_failure_handling": RUNTIME_FAILURE_HANDLING,
    }


def _sweep_record(
    label: str,
    workload: WorkloadShape,
    *,
    prefill_treatment: str,
) -> dict[str, object]:
    counts = envelope_counts(workload)
    return {
        "label": label,
        "workload": asdict(workload),
        "prefill_treatment": prefill_treatment,
        "metrics": {
            "length_advice_comparison_bits": (
                counts.length_advice_comparison_bits
            ),
            "padded_decode_tokens": counts.padded_decode_tokens,
            "actual_model_tokens": counts.actual_model_tokens,
            "envelope_model_tokens": counts.envelope_model_tokens,
            "total_model_token_inflation_ratio": (
                counts.total_model_token_inflation_ratio
            ),
            "total_noop_expert_branches": (
                counts.total_noop_expert_branches
            ),
            "noop_fraction": counts.noop_fraction,
            "expansion_over_useful_expert_calls": (
                counts.expansion_over_useful_expert_calls
            ),
        },
    }


def default_anchor_sweeps() -> dict[str, object]:
    """Small deterministic output-length and request-count sweeps."""

    sequence_points = []
    for actual in (128, 512, 900, 1_024):
        workload = WorkloadShape(
            requests=ANCHOR_WORKLOAD.requests,
            actual_output_tokens_per_request=actual,
            max_output_tokens_per_request=(
                ANCHOR_WORKLOAD.max_output_tokens_per_request
            ),
            prefill_recomputation_tokens=(
                ANCHOR_WORKLOAD.prefill_recomputation_tokens
            ),
            moe_layers=ANCHOR_WORKLOAD.moe_layers,
            routed_experts=ANCHOR_WORKLOAD.routed_experts,
            top_k_experts=ANCHOR_WORKLOAD.top_k_experts,
        )
        sequence_points.append(
            _sweep_record(
                f"actual-output-{actual}",
                workload,
                prefill_treatment="held at the anchor aggregate",
            )
        )

    request_points = []
    for requests in (32, 64, 128, 256, 512):
        scaled_prefill_numerator = (
            ANCHOR_WORKLOAD.prefill_recomputation_tokens * requests
        )
        if scaled_prefill_numerator % ANCHOR_WORKLOAD.requests:
            raise AssertionError("default request sweep must scale prefill exactly")
        workload = WorkloadShape(
            requests=requests,
            actual_output_tokens_per_request=(
                ANCHOR_WORKLOAD.actual_output_tokens_per_request
            ),
            max_output_tokens_per_request=(
                ANCHOR_WORKLOAD.max_output_tokens_per_request
            ),
            prefill_recomputation_tokens=(
                scaled_prefill_numerator // ANCHOR_WORKLOAD.requests
            ),
            moe_layers=ANCHOR_WORKLOAD.moe_layers,
            routed_experts=ANCHOR_WORKLOAD.routed_experts,
            top_k_experts=ANCHOR_WORKLOAD.top_k_experts,
        )
        request_points.append(
            _sweep_record(
                f"requests-{requests}",
                workload,
                prefill_treatment=(
                    "exact proportional projection from the 128-request anchor"
                ),
            )
        )
    return {
        "sequence_length": sequence_points,
        "request_count": request_points,
        "note": (
            "These are structural sensitivity sweeps, not measured deployment "
            "workloads or performance projections."
        ),
    }


def run_experiment(
    workload: WorkloadShape = ANCHOR_WORKLOAD,
    sampling: SamplingMerkleParameters = DEFAULT_SAMPLING_MERKLE,
    gate_coefficients: GateCoefficients | None = None,
    cell_bits: int = CELL_BITS,
    *,
    include_sweeps: bool = True,
) -> dict[str, object]:
    """Run the deterministic experiment, with anchor sweeps by default."""

    result = analyze_noop_envelope(
        workload, sampling, gate_coefficients, cell_bits
    )
    if include_sweeps:
        result["sweeps"] = (
            default_anchor_sweeps()
            if workload == ANCHOR_WORKLOAD
            else {
                "note": (
                    "Default sweeps are emitted only for the published anchor; "
                    "this custom workload has no implied scaling law."
                )
            }
        )
    return result


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Serialize deterministically and reject non-JSON floating-point values."""

    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def _optional_gate_coefficients(args: argparse.Namespace) -> GateCoefficients | None:
    names = ("g_common", "g_mask", "g_router", "g_expert", "g_branch")
    values = [getattr(args, name) for name in names]
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise EnvelopeError("gate coefficients must be supplied all together")
    return GateCoefficients(**dict(zip(names, values)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit deterministic structural accounting for a no-op-padded "
            "Kimi-2.5 envelope."
        )
    )
    parser.add_argument("--requests", type=int, default=ANCHOR_WORKLOAD.requests)
    parser.add_argument(
        "--actual-output-tokens",
        type=int,
        default=ANCHOR_WORKLOAD.actual_output_tokens_per_request,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=ANCHOR_WORKLOAD.max_output_tokens_per_request,
    )
    parser.add_argument(
        "--prefill-recomputation-tokens",
        type=int,
        default=ANCHOR_WORKLOAD.prefill_recomputation_tokens,
    )
    parser.add_argument(
        "--instruction-depth",
        type=int,
        default=DEFAULT_SAMPLING_MERKLE.instruction_depth,
    )
    parser.add_argument(
        "--value-depth",
        type=int,
        default=DEFAULT_SAMPLING_MERKLE.value_depth,
    )
    parser.add_argument(
        "--arity", type=int, default=DEFAULT_SAMPLING_MERKLE.arity
    )
    parser.add_argument("--p", type=float, default=DEFAULT_SAMPLING_MERKLE.p)
    parser.add_argument("--cell-bits", type=int, default=CELL_BITS)
    for coefficient in (
        "g-common",
        "g-mask",
        "g-router",
        "g-expert",
        "g-branch",
    ):
        parser.add_argument(f"--{coefficient}", type=int)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    workload = WorkloadShape(
        requests=args.requests,
        actual_output_tokens_per_request=args.actual_output_tokens,
        max_output_tokens_per_request=args.max_output_tokens,
        prefill_recomputation_tokens=args.prefill_recomputation_tokens,
        moe_layers=ANCHOR_WORKLOAD.moe_layers,
        routed_experts=ANCHOR_WORKLOAD.routed_experts,
        top_k_experts=ANCHOR_WORKLOAD.top_k_experts,
    )
    sampling = SamplingMerkleParameters(
        p=args.p,
        instruction_depth=args.instruction_depth,
        value_depth=args.value_depth,
        arity=args.arity,
    )
    result = run_experiment(
        workload,
        sampling,
        _optional_gate_coefficients(args),
        args.cell_bits,
        include_sweeps=not args.no_sweeps,
    )
    print(canonical_json(result, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
