"""Sourced structural frontier sweep for a Kimi-2.5 Mooncake deployment.

The experiment models descriptions consumed by ``Compile(G, x, a)`` and keeps
the following quantities separate:

* static ``G`` program records;
* finite-epoch ``x`` metadata and prompt/block data;
* ``a`` advice bits;
* reusable definitions;
* occurrence and routing records;
* runtime values/events, which are not advice;
* expanded logical model-token work; and
* projected physical KV/network and MoE units.

This is not a latency, throughput-capacity, or GPU-hour model.  The public
deployment report does not expose service curves, DRAM capacity, or a
contention model.  Byte counts based on the report's approximate 3.8 GB per
100K-token Kimi-2.5 FP8 KV example are explicitly marked as proxies.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from prototypes.advice_workload import (
    RequestInput,
    WorkloadInput,
    ceil_log2,
    continual_batch_schedule,
    minimum_length_advice_bits,
)


class SweepError(ValueError):
    """The sweep design, trace, or accounting request is malformed."""


EVIDENCE_KINDS = frozenset(
    {
        "reported",
        "configuration",
        "assumption",
        "derived",
        "proxy",
        "unsupported",
    }
)
CLAIM_LEVELS = ("logical", "physical")
ROUTING_POLICIES = ("sticky", "cache-aware", "round-robin")
LOGICAL_STRATEGIES = (
    "replay_in_g",
    "maximum_envelope",
    "geometric_bucket",
    "exact_logical",
)
KV_BLOCK_TOKENS = 64
FAST25_HASH_BLOCK_TOKENS = 512
FAST25_CHILDREN_PER_HASH = FAST25_HASH_BLOCK_TOKENS // KV_BLOCK_TOKENS
MODEL_CONTEXT_TOKENS = 256_000
LOGICAL_DECODE_SLOTS = 32
PREFILL_CHUNK_TOKENS = 2_048
TOKEN_ID_BYTES = 4
BLOCK_HASH_BYTES = 32

# The blog says a 100K-token Kimi-2.5 context can occupy about 3.8 GB with
# FP8 KV caches.  Dividing the two reported rounded values is useful for scale,
# but it is not an architecture-exact byte count for every deployment.
KV_BYTES_PER_TOKEN_PROXY = 3_800_000_000 // 100_000
KV_BYTES_PER_BLOCK_PROXY = KV_BYTES_PER_TOKEN_PROXY * KV_BLOCK_TOKENS

MOE_LAYERS = 60
ROUTED_EXPERTS = 384
TOP_K_EXPERTS = 8
SHARED_EXPERTS = 1
EXPERT_HIDDEN_SIZE = 2_048
ROUTE_CODE_BITS_PER_MOE_TOKEN_PROXY = ceil_log2(
    math.comb(ROUTED_EXPERTS, TOP_K_EXPERTS)
)


SOURCES: dict[str, dict[str, str]] = {
    "vllm_mooncake_blog": {
        "title": "Serving Agentic Workloads at Scale with vLLM x Mooncake",
        "url": "https://vllm.ai/blog/2026-05-06-mooncake-store",
        "publisher": "vLLM",
        "date": "2026-05-06",
    },
    "kimi_k2_5_card": {
        "title": "Kimi K2.5 official model card",
        "url": "https://github.com/moonshotai/Kimi-K2.5",
        "publisher": "Moonshot AI",
        "date": "undated",
    },
    "mooncake_fast25": {
        "title": "Mooncake FAST'25 trace release",
        "url": "https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release",
        "publisher": "Mooncake",
        "date": "2025-02-21",
    },
    "vllm_scheduler_docs": {
        "title": "vLLM optimization and scheduling documentation",
        "url": "https://docs.vllm.ai/en/stable/configuration/optimization/",
        "publisher": "vLLM",
        "date": "rolling",
    },
}


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    label: str
    kind: str
    claim: str
    source_ids: tuple[str, ...] = ()
    used_as_model_input: bool = True
    note: str = ""


EVIDENCE_LEDGER = (
    EvidenceRecord(
        "kimi_architecture",
        "Kimi-2.5 architecture and 256K context",
        "reported",
        (
            "1T total/32B active parameters; 61 layers (1 dense, 60 MoE); "
            "384 routed experts with top-8 plus one shared expert; hidden "
            "7168; expert hidden 2048; MLA; 64 heads; 256K context"
        ),
        ("kimi_k2_5_card",),
    ),
    EvidenceRecord(
        "deployment_1p1d",
        "May 2026 1P1D deployment",
        "configuration",
        (
            "text-only Kimi-2.5 NVFP4; prefill TP4 on four GB200 GPUs; "
            "decode DP8+EP8 on eight GB200 GPUs; 12 GPUs total"
        ),
        ("vllm_mooncake_blog",),
    ),
    EvidenceRecord(
        "agentic_anchor",
        "Published synthetic agentic scaling workload",
        "configuration",
        (
            "20K common tokens, 10K first input, 2,048 new input tokens per "
            "turn, 900 output tokens, and 30 turns"
        ),
        ("vllm_mooncake_blog",),
    ),
    EvidenceRecord(
        "codex_trace_report",
        "Codex/SWE-bench Pro corpus summary",
        "reported",
        (
            "610 traces; 94.2% potential cache hit; about 131:1 input:output; "
            "median context growth around 12K to 80K"
        ),
        ("vllm_mooncake_blog",),
        used_as_model_input=False,
    ),
    EvidenceRecord(
        "reported_relative_performance",
        "Published relative performance outcomes",
        "reported",
        "3.8x throughput, 46x P50 TTFT, and 8.6x E2E improvements",
        ("vllm_mooncake_blog",),
        used_as_model_input=False,
        note="Reported for context only; never fitted or used as a target.",
    ),
    EvidenceRecord(
        "fast25_schema",
        "Mooncake FAST'25 JSONL request schema",
        "reported",
        (
            "timestamp milliseconds, input_length, output_length, and hash_ids "
            "where each hash identifies a 512-token block"
        ),
        ("mooncake_fast25",),
    ),
    EvidenceRecord(
        "logical_scheduler",
        "Canonical deterministic continuous-batch scheduler",
        "assumption",
        (
            "FCFS logical slot replacement with deterministic tie-breaking; "
            "each active sequence emits one token per logical tick"
        ),
        ("vllm_scheduler_docs",),
    ),
    EvidenceRecord(
        "finite_epoch",
        "Finite offline epoch",
        "assumption",
        (
            "all arrivals and replayed prompts are fixed in x; adaptive tool "
            "turns require a separate epoch"
        ),
    ),
    EvidenceRecord(
        "fixed_sampling",
        "Fixed model and sampling policy",
        "assumption",
        "model, sampling policy, and seeds are fixed in x",
    ),
    EvidenceRecord(
        "text_dram_only",
        "Text-only external DRAM tier",
        "assumption",
        "SSD, multimodal inputs, faults, and speculative decoding are disabled",
    ),
    EvidenceRecord(
        "canonical_outputs",
        "Canonical output representation",
        "assumption",
        (
            "request/output order and post-stop padding are canonical; logical "
            "scheduling has no wall-clock races"
        ),
    ),
    EvidenceRecord(
        "vllm_64_token_blocks",
        "64-token vLLM KV block size",
        "assumption",
        "explicit structural configuration for this projection",
    ),
    EvidenceRecord(
        "fast25_child_mapping",
        "Conservative FAST'25 hash expansion",
        "derived",
        "each 512-token hash ID maps to eight namespaced 64-token child IDs",
        ("mooncake_fast25",),
    ),
    EvidenceRecord(
        "kv_byte_scale",
        "Approximate Kimi-2.5 KV byte scale",
        "proxy",
        (
            "38,000 bytes/token, derived from the blog's rounded example of "
            "about 3.8 GB for a 100K-token FP8 KV context"
        ),
        ("vllm_mooncake_blog",),
        note="Not an exact NVFP4 deployment allocation or capacity measurement.",
    ),
    EvidenceRecord(
        "moe_route_code",
        "MoE route-description code length",
        "proxy",
        (
            "ceil(log2(comb(384, 8))) bits per MoE-layer token as a fixed-field "
            "ceiling; not a proven lower bound for Kimi routes"
        ),
        ("kimi_k2_5_card",),
    ),
    EvidenceRecord(
        "absolute_service_metrics",
        "Latency, capacity-normalized load, and GPU-hours",
        "unsupported",
        (
            "no public absolute GB200 prefill/decode service curves, DRAM "
            "capacity model, or RDMA contention curve is used"
        ),
        ("vllm_mooncake_blog",),
        used_as_model_input=False,
    ),
)


MODEL_CONFIGURATION: dict[str, object] = {
    "name": "Kimi-2.5",
    "total_parameters": 1_000_000_000_000,
    "active_parameters": 32_000_000_000,
    "layers": 61,
    "dense_layers": 1,
    "moe_layers": MOE_LAYERS,
    "routed_experts": ROUTED_EXPERTS,
    "top_k_experts": TOP_K_EXPERTS,
    "shared_experts": SHARED_EXPERTS,
    "hidden_size": 7_168,
    "expert_hidden_size": EXPERT_HIDDEN_SIZE,
    "attention": "MLA",
    "attention_heads": 64,
    "context_tokens": MODEL_CONTEXT_TOKENS,
    "evidence_kind": "reported",
    "source_id": "kimi_k2_5_card",
}

DEPLOYMENT_CONFIGURATION: dict[str, object] = {
    "model_format": "Kimi-2.5 NVFP4",
    "modality": "text-only",
    "prefill": {"instances": 1, "parallelism": "TP4", "gpus": 4},
    "decode": {
        "instances": 1,
        "parallelism": "DP8+EP8",
        "gpus": 8,
        "ep_ranks_synchronized": True,
        "independent_decode_servers": 1,
    },
    "gpu": "GB200",
    "total_gpus": 12,
    "external_cache_tier": "distributed DRAM",
    "ssd_enabled": False,
    "flow": [
        "session turn arrives in the finite epoch",
        "hash prompt blocks and query distributed DRAM",
        "fetch cached prefix and chunk-prefill the missing suffix",
        "transfer prompt KV from prefill to decode",
        "continuous-batch one token per active sequence per logical tick",
        "publish newly completed full KV blocks asynchronously",
    ],
    "evidence_kind": "configuration",
    "source_id": "vllm_mooncake_blog",
}

REPORTED_ANCHOR: dict[str, object] = {
    "agentic_workload": {
        "common_tokens": 20_000,
        "first_input_tokens": 10_000,
        "new_input_tokens_per_turn": 2_048,
        "output_tokens": 900,
        "turns": 30,
        "evidence_kind": "configuration",
        "source_id": "vllm_mooncake_blog",
    },
    "codex_swebench_pro": {
        "traces": 610,
        "potential_cache_hit_fraction": 0.942,
        "input_output_ratio": "approximately 131:1",
        "median_context_start_tokens": 12_000,
        "median_context_end_tokens": 80_000,
        "evidence_kind": "reported",
        "used_as_generated_evidence": False,
        "source_id": "vllm_mooncake_blog",
    },
    "benchmark_outcomes_not_fitted": {
        "throughput_multiplier": 3.8,
        "p50_ttft_reduction_multiplier": 46.0,
        "e2e_latency_reduction_multiplier": 8.6,
        "evidence_kind": "reported",
        "used_as_model_input": False,
        "source_id": "vllm_mooncake_blog",
    },
}

STATIC_PROGRAM_DESCRIPTOR: dict[str, object] = {
    "program": "Compile(G,x,a) frontier inference epoch",
    "canonical_order": "session, turn, request, token, layer, expert",
    "operators": [
        "prompt_block_hash",
        "distributed_dram_lookup",
        "chunked_prefill",
        "prefill_to_decode_transfer",
        "continuous_decode_tick",
        "moe_router_or_envelope",
        "full_block_publish",
        "canonical_stop_padding",
    ],
    "claim_levels": list(CLAIM_LEVELS),
}


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    common_prefix_tokens: int
    first_input_tokens: int
    incremental_input_tokens: int
    realized_output_tokens: int
    max_output_tokens: int
    evidence_kind: str
    note: str


WORKLOAD_PROFILES = (
    WorkloadProfile(
        "short",
        common_prefix_tokens=0,
        first_input_tokens=2_048,
        incremental_input_tokens=512,
        realized_output_tokens=128,
        max_output_tokens=192,
        evidence_kind="assumption",
        note="Short synthetic screening case with a non-power-of-two cap.",
    ),
    WorkloadProfile(
        "medium",
        common_prefix_tokens=8_192,
        first_input_tokens=4_096,
        incremental_input_tokens=1_024,
        realized_output_tokens=384,
        max_output_tokens=512,
        evidence_kind="assumption",
        note="Medium synthetic screening case.",
    ),
    WorkloadProfile(
        "agentic_anchor",
        common_prefix_tokens=20_000,
        first_input_tokens=10_000,
        incremental_input_tokens=2_048,
        realized_output_tokens=900,
        max_output_tokens=1_024,
        evidence_kind="configuration",
        note=(
            "Published scaling-workload lengths; the 1,024 output cap is an "
            "explicit projection around the reported 900-token output."
        ),
    ),
    WorkloadProfile(
        "long",
        common_prefix_tokens=32_768,
        first_input_tokens=32_768,
        incremental_input_tokens=4_096,
        realized_output_tokens=1_536,
        max_output_tokens=2_048,
        evidence_kind="assumption",
        note="Long synthetic corner kept below the 256K context limit.",
    ),
)
PROFILE_BY_NAME = {profile.name: profile for profile in WORKLOAD_PROFILES}


@dataclass(frozen=True)
class SweepSpec:
    point_id: str
    requests_per_epoch: int
    profile_name: str
    turns_per_session: int
    cache_hit_fraction: float
    routing_policy: str
    sampling_seed: int = 0
    adaptive_turns: bool = False

    @property
    def profile(self) -> WorkloadProfile:
        try:
            return PROFILE_BY_NAME[self.profile_name]
        except KeyError as exc:
            raise SweepError(f"unknown workload profile: {self.profile_name}") from exc


@dataclass(frozen=True)
class LengthAccounting:
    strategy: str
    worst_case_advice_bits: int
    realized_advice_bits: int
    realized_output_tokens: int
    represented_output_tokens: int
    padding_output_tokens: int
    represented_lengths: tuple[int, ...]
    evidence_kind: str
    note: str


@dataclass(frozen=True)
class MooncakeTraceRecord:
    timestamp_ms: float
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]

    @property
    def child_hash_ids_64(self) -> tuple[str, ...]:
        return tuple(
            child_id
            for hash_id in self.hash_ids
            for child_id in expand_fast25_hash_id(hash_id)
        )


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Serialize deterministically with no non-JSON values."""

    return json.dumps(
        value,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def measured_json_bytes(value: object) -> int:
    """Return the exact UTF-8 size of this prototype's canonical JSON."""

    return len(canonical_json(value).encode("utf-8"))


def validate_evidence_ledger() -> None:
    seen: set[str] = set()
    for record in EVIDENCE_LEDGER:
        if not record.evidence_id or record.evidence_id in seen:
            raise SweepError("evidence IDs must be nonempty and unique")
        seen.add(record.evidence_id)
        if record.kind not in EVIDENCE_KINDS:
            raise SweepError(f"invalid evidence kind: {record.kind}")
        if not record.label or not record.claim:
            raise SweepError(f"incomplete evidence record: {record.evidence_id}")
        if record.kind in {"reported", "configuration"} and not record.source_ids:
            raise SweepError(f"sourced evidence lacks a source: {record.evidence_id}")
        unknown_sources = set(record.source_ids) - set(SOURCES)
        if unknown_sources:
            raise SweepError(
                f"unknown sources for {record.evidence_id}: {sorted(unknown_sources)}"
            )
    for source_id, source in SOURCES.items():
        if not all(source.get(field) for field in ("title", "url", "publisher", "date")):
            raise SweepError(f"incomplete source metadata: {source_id}")


def _validate_positive_int(value: int, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise SweepError(f"{label} must be a positive integer")


def validate_sweep_spec(spec: SweepSpec) -> None:
    _validate_positive_int(spec.requests_per_epoch, "requests_per_epoch")
    _validate_positive_int(spec.turns_per_session, "turns_per_session")
    if spec.adaptive_turns:
        raise SweepError(
            "adaptive tool turns are outside one finite offline epoch"
        )
    if type(spec.sampling_seed) is not int:
        raise SweepError("sampling_seed must be an integer fixed in x")
    if spec.routing_policy not in ROUTING_POLICIES:
        raise SweepError(f"unknown routing policy: {spec.routing_policy}")
    if not (0.0 <= spec.cache_hit_fraction <= 1.0):
        raise SweepError("cache_hit_fraction must lie in [0, 1]")

    profile = spec.profile
    for field_name in (
        "first_input_tokens",
        "incremental_input_tokens",
        "realized_output_tokens",
        "max_output_tokens",
    ):
        _validate_positive_int(getattr(profile, field_name), field_name)
    if (
        type(profile.common_prefix_tokens) is not int
        or profile.common_prefix_tokens < 0
    ):
        raise SweepError("common_prefix_tokens must be a nonnegative integer")
    if profile.realized_output_tokens > profile.max_output_tokens:
        raise SweepError("realized output exceeds its public cap")
    if profile.evidence_kind not in EVIDENCE_KINDS:
        raise SweepError("workload profile has an invalid evidence kind")

    worst_case_last_prompt = (
        profile.common_prefix_tokens
        + profile.first_input_tokens
        + (spec.turns_per_session - 1)
        * (profile.incremental_input_tokens + profile.max_output_tokens)
    )
    if worst_case_last_prompt + profile.max_output_tokens > MODEL_CONTEXT_TOKENS:
        raise SweepError(
            "worst-case prompt plus output cap exceeds Kimi-2.5's 256K context"
        )


def exact_length_advice_bits(max_lengths: Sequence[int]) -> int:
    """Exact mixed-radix fixed-code bound for lengths in ``1..M_i``."""

    if not max_lengths:
        raise SweepError("at least one length cap is required")
    for cap in max_lengths:
        _validate_positive_int(cap, "length cap")
    return ceil_log2(math.prod(max_lengths))


def geometric_bucket_count(max_length: int) -> int:
    """Number of power-of-two ceiling buckets for lengths in ``1..M``."""

    _validate_positive_int(max_length, "max_length")
    return 1 + ceil_log2(max_length)


def geometric_bucket_ceiling(length: int, max_length: int) -> int:
    """Smallest power-of-two envelope, clipped at a non-power-of-two cap."""

    _validate_positive_int(length, "length")
    _validate_positive_int(max_length, "max_length")
    if length > max_length:
        raise SweepError("length exceeds its cap")
    return min(max_length, 1 << ceil_log2(length))


def geometric_length_advice_bits(max_lengths: Sequence[int]) -> int:
    """Exact joint fixed-code size for one geometric bucket per request."""

    if not max_lengths:
        raise SweepError("at least one length cap is required")
    return ceil_log2(
        math.prod(geometric_bucket_count(cap) for cap in max_lengths)
    )


def account_length_strategy(
    max_lengths: Sequence[int],
    realized_lengths: Sequence[int],
    strategy: str,
) -> LengthAccounting:
    if len(max_lengths) != len(realized_lengths) or not max_lengths:
        raise SweepError("caps and realized lengths must have equal nonzero size")
    for cap, realized in zip(max_lengths, realized_lengths):
        _validate_positive_int(cap, "length cap")
        _validate_positive_int(realized, "realized length")
        if realized > cap:
            raise SweepError("realized length exceeds its cap")

    actual = sum(realized_lengths)
    if strategy in {"replay_in_g", "exact_logical"}:
        represented = tuple(realized_lengths)
    elif strategy == "geometric_bucket":
        represented = tuple(
            geometric_bucket_ceiling(realized, cap)
            for cap, realized in zip(max_lengths, realized_lengths)
        )
    elif strategy == "maximum_envelope":
        represented = tuple(max_lengths)
    else:
        raise SweepError(f"unknown length strategy: {strategy}")

    if strategy == "exact_logical":
        bits = exact_length_advice_bits(max_lengths)
        kind = "derived"
        note = "Exact mixed-radix fixed-code bound; scheduler adds zero bits."
    elif strategy == "geometric_bucket":
        bits = geometric_length_advice_bits(max_lengths)
        kind = "derived"
        note = (
            "Exact joint bucket-code size; each represented length is less "
            "than twice the realized length."
        )
    elif strategy == "maximum_envelope":
        bits = 0
        kind = "derived"
        note = "Zero advice with canonical masks and post-stop padding."
    else:
        bits = 0
        kind = "derived"
        note = (
            "Zero advice because Compile replays deterministic inference and "
            "the scheduler; replay work is charged separately."
        )

    return LengthAccounting(
        strategy=strategy,
        worst_case_advice_bits=bits,
        realized_advice_bits=bits,
        realized_output_tokens=actual,
        represented_output_tokens=sum(represented),
        padding_output_tokens=sum(represented) - actual,
        represented_lengths=represented,
        evidence_kind=kind,
        note=note,
    )


def _candidate_distance(
    left: tuple[int, int, int, int, int],
    right: tuple[int, int, int, int, int],
) -> float:
    axis_sizes = (3, 4, 4, 4)
    ordered_distance = sum(
        ((left[index] - right[index]) / (axis_sizes[index] - 1)) ** 2
        for index in range(4)
    )
    routing_distance = 1.0 if left[4] != right[4] else 0.0
    return ordered_distance + routing_distance


def screening_design(size: int = 28) -> tuple[SweepSpec, ...]:
    """Build a deterministic maximin screening design plus explicit corners."""

    if type(size) is not int or not (4 <= size <= 32):
        raise SweepError("screening design size must lie in [4, 32]")
    requests_axis = (32, 128, 512)
    profile_axis = tuple(profile.name for profile in WORKLOAD_PROFILES)
    turns_axis = (1, 4, 16, 30)
    cache_axis = (0.0, 0.5, 0.9, 0.99)
    routing_axis = ROUTING_POLICIES
    candidates = [
        (request_i, profile_i, turn_i, cache_i, routing_i)
        for request_i in range(len(requests_axis))
        for profile_i in range(len(profile_axis))
        for turn_i in range(len(turns_axis))
        for cache_i in range(len(cache_axis))
        for routing_i in range(len(routing_axis))
    ]
    selected = [
        # Four broad corners.
        (0, 0, 0, 0, 0),
        (2, 3, 3, 3, 2),
        (2, 0, 3, 0, 1),
        (0, 3, 0, 3, 2),
        # Controlled slices around the published 30-turn workload shape.
        (1, 2, 3, 2, 2),
        (0, 2, 3, 2, 2),
        (2, 2, 3, 2, 2),
        (1, 0, 3, 2, 2),
        (1, 1, 3, 2, 2),
        (1, 3, 3, 2, 2),
        (1, 2, 0, 2, 2),
        (1, 2, 1, 2, 2),
        (1, 2, 2, 2, 2),
        (1, 2, 3, 0, 2),
        (1, 2, 3, 1, 2),
        (1, 2, 3, 3, 2),
        (1, 2, 3, 2, 0),
        (1, 2, 3, 2, 1),
    ]
    selected = selected[:size]
    remaining = set(candidates) - set(selected)
    while len(selected) < size:
        candidate = max(
            remaining,
            key=lambda item: (
                min(_candidate_distance(item, chosen) for chosen in selected),
                item,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)

    specs: list[SweepSpec] = []
    for index, indices in enumerate(selected):
        request_i, profile_i, turn_i, cache_i, routing_i = indices
        cache = cache_axis[cache_i]
        cache_label = str(cache).replace(".", "p")
        point_id = (
            f"s{index:02d}-r{requests_axis[request_i]}-"
            f"{profile_axis[profile_i]}-t{turns_axis[turn_i]}-"
            f"h{cache_label}-{routing_axis[routing_i]}"
        )
        spec = SweepSpec(
            point_id=point_id,
            requests_per_epoch=requests_axis[request_i],
            profile_name=profile_axis[profile_i],
            turns_per_session=turns_axis[turn_i],
            cache_hit_fraction=cache,
            routing_policy=routing_axis[routing_i],
        )
        validate_sweep_spec(spec)
        specs.append(spec)
    return tuple(specs)


def _build_workload(
    spec: SweepSpec,
) -> tuple[WorkloadInput, dict[int, int], dict[int, tuple[int, int]]]:
    validate_sweep_spec(spec)
    profile = spec.profile
    session_count = math.ceil(spec.requests_per_epoch / spec.turns_per_session)
    turn_spacing = (
        math.ceil(session_count / LOGICAL_DECODE_SLOTS)
        * profile.max_output_tokens
        + 1
    )
    requests: list[RequestInput] = []
    realized_lengths: dict[int, int] = {}
    session_turn_by_request: dict[int, tuple[int, int]] = {}
    for request_id in range(spec.requests_per_epoch):
        session_id, turn = divmod(request_id, spec.turns_per_session)
        prompt_length = (
            profile.common_prefix_tokens
            + profile.first_input_tokens
            + turn
            * (profile.incremental_input_tokens + profile.realized_output_tokens)
        )
        arrival_tick = turn * turn_spacing + session_id // LOGICAL_DECODE_SLOTS
        requests.append(
            RequestInput(
                request_id=request_id,
                arrival_tick=arrival_tick,
                prompt_length=prompt_length,
                max_new_tokens=profile.max_output_tokens,
            )
        )
        realized_lengths[request_id] = profile.realized_output_tokens
        session_turn_by_request[request_id] = (session_id, turn)
    workload = WorkloadInput(tuple(requests), LOGICAL_DECODE_SLOTS)
    return workload, realized_lengths, session_turn_by_request


def _fraction_floor(value: int, fraction: float) -> int:
    exact_fraction = Fraction(str(fraction))
    return value * exact_fraction.numerator // exact_fraction.denominator


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _metadata_descriptions(
    spec: SweepSpec,
    workload: WorkloadInput,
    session_turn_by_request: Mapping[int, tuple[int, int]],
) -> dict[str, object]:
    session_turn_counts: dict[int, int] = {}
    request_records: list[dict[str, int]] = []
    for request in workload.requests:
        session_id, turn = session_turn_by_request[request.request_id]
        session_turn_counts[session_id] = session_turn_counts.get(session_id, 0) + 1
        request_records.append(
            {
                "arrival_tick": request.arrival_tick,
                "max_new_tokens": request.max_new_tokens,
                "prompt_length": request.prompt_length,
                "request_id": request.request_id,
                "session_id": session_id,
                "turn": turn,
            }
        )
    session_records = [
        {"session_id": session_id, "turns_in_epoch": turns}
        for session_id, turns in sorted(session_turn_counts.items())
    ]
    request_metadata = {
        "canonical_order": "request_id (session-major, then turn)",
        "records": request_records,
    }
    session_metadata = {"records": session_records}

    profile = spec.profile
    first_turn_requests = sum(
        turn == 0 for _, turn in session_turn_by_request.values()
    )
    later_turn_requests = len(workload.requests) - first_turn_requests
    external_prompt_token_units = (
        profile.common_prefix_tokens
        + first_turn_requests * profile.first_input_tokens
        + later_turn_requests * profile.incremental_input_tokens
    )
    prompt_hash_records = (
        _ceil_div(profile.common_prefix_tokens, KV_BLOCK_TOKENS)
        if profile.common_prefix_tokens
        else 0
    )
    prompt_hash_records += first_turn_requests * _ceil_div(
        profile.first_input_tokens, KV_BLOCK_TOKENS
    )
    prompt_hash_records += later_turn_requests * _ceil_div(
        profile.incremental_input_tokens, KV_BLOCK_TOKENS
    )
    request_metadata_bytes = measured_json_bytes(request_metadata)
    session_metadata_bytes = measured_json_bytes(session_metadata)
    token_bytes = external_prompt_token_units * TOKEN_ID_BYTES
    hash_bytes = prompt_hash_records * BLOCK_HASH_BYTES
    return {
        "request_metadata_records": len(request_records),
        "request_metadata_bytes_measured": request_metadata_bytes,
        "session_metadata_records": len(session_records),
        "session_metadata_bytes_measured": session_metadata_bytes,
        "external_prompt_token_units": external_prompt_token_units,
        "prompt_token_bytes_u32_projection": token_bytes,
        "prompt_block_hash_records": prompt_hash_records,
        "prompt_block_hash_bytes_projection": hash_bytes,
        "total_x_bytes_projection": (
            request_metadata_bytes + session_metadata_bytes + token_bytes + hash_bytes
        ),
        "expanded_prompt_token_occurrences": sum(
            request.prompt_length for request in workload.requests
        ),
        "prior_output_token_references": sum(
            turn * profile.realized_output_tokens
            for _, turn in session_turn_by_request.values()
        ),
        "measurement_kind": "measured_serialization_and_explicit_width_projection",
        "evidence_kind": "derived",
        "note": (
            "Metadata bytes are measured canonical JSON. Token/hash widths are "
            "explicit projections. Prior model outputs are runtime values "
            "referenced by later prompts, not charged again as advice."
        ),
    }


def _scenario_base(
    spec: SweepSpec,
    workload: WorkloadInput,
    realized_lengths: Mapping[int, int],
    session_turn_by_request: Mapping[int, tuple[int, int]],
) -> dict[str, object]:
    hit_blocks_by_request: dict[int, int] = {}
    recompute_tokens_by_request: dict[int, int] = {}
    candidate_blocks = 0
    hit_blocks = 0
    prefill_chunks = 0
    prefill_chunk_shapes: set[int] = set()
    pd_blocks = 0
    publish_blocks = 0
    for request in workload.requests:
        full_prompt_blocks = request.prompt_length // KV_BLOCK_TOKENS
        request_hit_blocks = _fraction_floor(
            full_prompt_blocks, spec.cache_hit_fraction
        )
        recompute_tokens = (
            request.prompt_length - request_hit_blocks * KV_BLOCK_TOKENS
        )
        hit_blocks_by_request[request.request_id] = request_hit_blocks
        recompute_tokens_by_request[request.request_id] = recompute_tokens
        candidate_blocks += full_prompt_blocks
        hit_blocks += request_hit_blocks
        if recompute_tokens:
            chunks = _ceil_div(recompute_tokens, PREFILL_CHUNK_TOKENS)
            prefill_chunks += chunks
            if chunks > 1:
                prefill_chunk_shapes.add(PREFILL_CHUNK_TOKENS)
            remainder = recompute_tokens % PREFILL_CHUNK_TOKENS
            prefill_chunk_shapes.add(remainder or PREFILL_CHUNK_TOKENS)
        request_pd_blocks = _ceil_div(request.prompt_length, KV_BLOCK_TOKENS)
        pd_blocks += request_pd_blocks
        full_after_decode = (
            request.prompt_length + realized_lengths[request.request_id]
        ) // KV_BLOCK_TOKENS
        publish_blocks += max(0, full_after_decode - request_hit_blocks)

    realized_hit_fraction = (
        hit_blocks / candidate_blocks if candidate_blocks else 0.0
    )
    prompt_tokens = sum(request.prompt_length for request in workload.requests)
    prefill_recomputation_tokens = sum(recompute_tokens_by_request.values())
    return {
        "session_count": len(
            {session for session, _ in session_turn_by_request.values()}
        ),
        "actual_session_count": 1
        + max(session for session, _ in session_turn_by_request.values()),
        "prompt_token_occurrences": prompt_tokens,
        "realized_output_tokens": sum(realized_lengths.values()),
        "candidate_cache_blocks": candidate_blocks,
        "dram_hit_blocks": hit_blocks,
        "realized_cache_hit_fraction": realized_hit_fraction,
        "prefill_recomputation_tokens": prefill_recomputation_tokens,
        "prefill_chunk_records": prefill_chunks,
        "prefill_chunk_shape_count": len(prefill_chunk_shapes),
        "prefill_to_decode_blocks": pd_blocks,
        "publish_blocks": publish_blocks,
        "hit_blocks_by_request": hit_blocks_by_request,
        "recompute_tokens_by_request": recompute_tokens_by_request,
        "x_description": _metadata_descriptions(
            spec, workload, session_turn_by_request
        ),
    }


@lru_cache(maxsize=256)
def _schedule_summary(
    workload: WorkloadInput, represented_lengths: tuple[int, ...]
) -> dict[str, int]:
    length_map = {
        request.request_id: represented_lengths[index]
        for index, request in enumerate(workload.requests)
    }
    schedule = continual_batch_schedule(workload, length_map)
    return {
        "decode_ticks": len(schedule.steps),
        "decode_active_token_cells": schedule.token_occurrences,
        "decode_event_slots": len(schedule.steps) * workload.decode_slots,
        "batch_composition_changes": sum(
            left.request_ids != right.request_ids
            for left, right in zip(schedule.steps, schedule.steps[1:])
        ),
    }


def _static_g_description() -> dict[str, object]:
    return {
        "program_units": len(STATIC_PROGRAM_DESCRIPTOR["operators"]),
        "canonical_json_bytes_measured": measured_json_bytes(
            STATIC_PROGRAM_DESCRIPTOR
        ),
        "measurement_kind": "measured_serialization",
        "evidence_kind": "derived",
        "note": (
            "Prototype descriptor bytes, not Python source size or a compiled "
            "production binary."
        ),
    }


def _kv_network_description(
    *,
    dram_fetch_blocks: int,
    pd_blocks: int,
    publish_blocks: int,
    included_in_claim: bool,
) -> dict[str, object]:
    total_blocks = dram_fetch_blocks + pd_blocks + publish_blocks
    return {
        "included_in_claim_level": included_in_claim,
        "kv_block_tokens": KV_BLOCK_TOKENS,
        "kv_bytes_per_token_proxy": KV_BYTES_PER_TOKEN_PROXY,
        "kv_bytes_per_block_proxy": KV_BYTES_PER_BLOCK_PROXY,
        "dram_fetch_blocks": dram_fetch_blocks,
        "dram_fetch_bytes_proxy": dram_fetch_blocks * KV_BYTES_PER_BLOCK_PROXY,
        "prefill_to_decode_blocks": pd_blocks,
        "prefill_to_decode_bytes_proxy": pd_blocks * KV_BYTES_PER_BLOCK_PROXY,
        "publish_blocks": publish_blocks,
        "publish_bytes_proxy": publish_blocks * KV_BYTES_PER_BLOCK_PROXY,
        "network_block_transfers": total_blocks,
        "network_transfer_bytes_proxy": total_blocks
        * KV_BYTES_PER_BLOCK_PROXY,
        "evidence_kind": "proxy",
        "note": (
            "Block counts are structural derivations. Bytes use the rounded "
            "38,000-byte/token scale and are not measured traffic."
        ),
    }


def _logical_point(
    spec: SweepSpec,
    workload: WorkloadInput,
    realized_lengths: Mapping[int, int],
    base: Mapping[str, object],
    strategy: str,
) -> dict[str, object]:
    caps = tuple(request.max_new_tokens for request in workload.requests)
    actual = tuple(realized_lengths[request.request_id] for request in workload.requests)
    accounting = account_length_strategy(caps, actual, strategy)
    schedule = _schedule_summary(workload, accounting.represented_lengths)
    prefill_tokens = int(base["prefill_recomputation_tokens"])
    runtime_model_tokens = prefill_tokens + accounting.represented_output_tokens
    actual_runtime_model_tokens = prefill_tokens + accounting.realized_output_tokens
    compile_replay_model_tokens = (
        actual_runtime_model_tokens if strategy == "replay_in_g" else 0
    )
    compile_replay_occurrences = (
        int(base["prefill_chunk_records"])
        + schedule["decode_ticks"]
        + len(workload.requests)
        if strategy == "replay_in_g"
        else 0
    )
    reusable_definition_records = (
        5
        + int(base["prefill_chunk_shape_count"])
        + len(set(accounting.represented_lengths))
    )
    logical_occurrence_records = (
        len(workload.requests)
        + int(base["prefill_chunk_records"])
        + schedule["decode_ticks"]
    )
    runtime_value_cells = {
        "actual_output_token_cells": accounting.realized_output_tokens,
        "represented_output_cells": accounting.represented_output_tokens,
        "canonical_post_stop_padding_cells": accounting.padding_output_tokens,
        "stop_indicator_cells": len(workload.requests),
        "cache_lookup_result_cells": int(base["candidate_cache_blocks"]),
        "decode_event_slots": schedule["decode_event_slots"],
    }
    runtime_value_cells["total_runtime_value_event_cells"] = sum(
        int(value) for value in runtime_value_cells.values()
    )
    kv_description = _kv_network_description(
        dram_fetch_blocks=int(base["dram_hit_blocks"]),
        pd_blocks=int(base["prefill_to_decode_blocks"]),
        publish_blocks=int(base["publish_blocks"]),
        included_in_claim=False,
    )
    description = {
        "static_g": _static_g_description(),
        "x_input": base["x_description"],
        "a_advice": {
            "length_code_bits": accounting.realized_advice_bits,
            "scheduler_choice_bits": 0,
            "worst_case_bits": accounting.worst_case_advice_bits,
            "realized_bits": accounting.realized_advice_bits,
            "evidence_kind": accounting.evidence_kind,
            "note": accounting.note,
        },
        "reusable_definitions": {
            "base_template_records": 5,
            "prefill_chunk_shape_records": int(base["prefill_chunk_shape_count"]),
            "output_shape_records": len(set(accounting.represented_lengths)),
            "total_records": reusable_definition_records,
            "evidence_kind": "derived",
        },
        "occurrence_routing_records": {
            "request_records": len(workload.requests),
            "prefill_chunk_occurrences": int(base["prefill_chunk_records"]),
            "decode_tick_occurrences": schedule["decode_ticks"],
            "compile_replay_occurrences": compile_replay_occurrences,
            "physical_routing_records": 0,
            "total_logical_occurrence_records": logical_occurrence_records,
            "evidence_kind": "derived",
        },
        "runtime_value_event_cells": {
            **runtime_value_cells,
            "evidence_kind": "derived",
            "note": "Runtime cells are circuit values/events, never advice.",
        },
        "expanded_logical_work_units": {
            "prefill_model_tokens": prefill_tokens,
            "realized_decode_model_tokens": accounting.realized_output_tokens,
            "represented_decode_model_tokens": accounting.represented_output_tokens,
            "runtime_model_tokens": runtime_model_tokens,
            "compile_replay_model_tokens": compile_replay_model_tokens,
            "constructor_plus_runtime_model_tokens": (
                runtime_model_tokens + compile_replay_model_tokens
            ),
            "decode_ticks": schedule["decode_ticks"],
            "decode_active_token_cells": schedule["decode_active_token_cells"],
            "decode_event_slots": schedule["decode_event_slots"],
            "evidence_kind": "derived",
        },
        "physical_kv_network_units": kv_description,
    }
    advice_bits = accounting.realized_advice_bits
    return {
        "point_id": f"{spec.point_id}::{strategy}",
        "scenario_id": spec.point_id,
        "claim_level": "logical",
        "strategy_family": strategy,
        "mechanisms": {
            "length": strategy,
            "scheduler": "deterministic_fcfs",
            "cache": "deterministic_prefix_policy",
        },
        "description_decomposition": description,
        "metrics": {
            "worst_case_advice_bits": accounting.worst_case_advice_bits,
            "realized_advice_bits": advice_bits,
            "advice_bits_per_request": advice_bits / len(workload.requests),
            "advice_bits_per_output_token": (
                advice_bits / accounting.realized_output_tokens
            ),
            "realized_output_tokens": accounting.realized_output_tokens,
            "represented_output_tokens": accounting.represented_output_tokens,
            "padded_output_tokens": accounting.padding_output_tokens,
            "output_work_inflation": (
                accounting.represented_output_tokens
                / accounting.realized_output_tokens
            ),
            "prefill_recomputation_tokens": prefill_tokens,
            "runtime_model_token_work": runtime_model_tokens,
            "compile_replay_model_token_work": compile_replay_model_tokens,
            "total_model_token_work": (
                runtime_model_tokens + compile_replay_model_tokens
            ),
            "decode_ticks": schedule["decode_ticks"],
            "decode_event_slots": schedule["decode_event_slots"],
            "reusable_definition_records": reusable_definition_records,
            "logical_occurrence_records": logical_occurrence_records,
            "latency_seconds": None,
            "gpu_hours": None,
        },
        "metric_evidence": {
            "advice": accounting.evidence_kind,
            "metadata_bytes": "derived",
            "logical_work": "derived",
            "kv_bytes": "proxy",
            "latency_seconds": "unsupported",
            "gpu_hours": "unsupported",
        },
    }


PHYSICAL_VARIANTS = (
    (
        "physical_recompute_compute_router",
        "recompute_prefix",
        "compute_moe_router",
    ),
    (
        "physical_cache_compute_router",
        "cache_selectors",
        "compute_moe_router",
    ),
    (
        "physical_cache_route_specialized",
        "cache_selectors",
        "route_specialized",
    ),
    (
        "physical_cache_all_expert_envelope",
        "cache_selectors",
        "all_expert_envelope",
    ),
)


def _physical_point(
    spec: SweepSpec,
    workload: WorkloadInput,
    realized_lengths: Mapping[int, int],
    base: Mapping[str, object],
    variant: tuple[str, str, str],
) -> dict[str, object]:
    strategy_name, cache_mechanism, moe_mechanism = variant
    caps = tuple(request.max_new_tokens for request in workload.requests)
    actual = tuple(realized_lengths[request.request_id] for request in workload.requests)
    exact = account_length_strategy(caps, actual, "exact_logical")
    schedule = _schedule_summary(workload, exact.represented_lengths)
    prompt_tokens = int(base["prompt_token_occurrences"])
    if cache_mechanism == "recompute_prefix":
        prefill_tokens = prompt_tokens
        dram_fetch_blocks = 0
        cache_selector_bits = 0
        prefill_chunks = sum(
            _ceil_div(request.prompt_length, PREFILL_CHUNK_TOKENS)
            for request in workload.requests
        )
        prefill_chunk_shapes = len(
            {
                request.prompt_length % PREFILL_CHUNK_TOKENS
                or PREFILL_CHUNK_TOKENS
                for request in workload.requests
            }
            | (
                {PREFILL_CHUNK_TOKENS}
                if any(
                    request.prompt_length > PREFILL_CHUNK_TOKENS
                    for request in workload.requests
                )
                else set()
            )
        )
    else:
        prefill_tokens = int(base["prefill_recomputation_tokens"])
        dram_fetch_blocks = int(base["dram_hit_blocks"])
        # A direct per-block bitmap is a transparent code-length ceiling.  It
        # is not a lower bound, and it vanishes if the entire cache state is
        # instead fixed in x and deterministically replayed.
        cache_selector_bits = int(base["candidate_cache_blocks"])
        prefill_chunks = int(base["prefill_chunk_records"])
        prefill_chunk_shapes = int(base["prefill_chunk_shape_count"])

    runtime_model_tokens = prefill_tokens + exact.realized_output_tokens
    moe_token_occurrences = runtime_model_tokens * MOE_LAYERS
    if moe_mechanism == "compute_moe_router":
        route_code_bits = 0
        router_score_cells = moe_token_occurrences * ROUTED_EXPERTS
        routed_expert_token_work = moe_token_occurrences * TOP_K_EXPERTS
    elif moe_mechanism == "route_specialized":
        route_code_bits = (
            moe_token_occurrences * ROUTE_CODE_BITS_PER_MOE_TOKEN_PROXY
        )
        router_score_cells = 0
        routed_expert_token_work = moe_token_occurrences * TOP_K_EXPERTS
    elif moe_mechanism == "all_expert_envelope":
        route_code_bits = 0
        router_score_cells = 0
        routed_expert_token_work = moe_token_occurrences * ROUTED_EXPERTS
    else:
        raise SweepError(f"unknown MoE mechanism: {moe_mechanism}")
    shared_expert_token_work = moe_token_occurrences * SHARED_EXPERTS
    moe_structural_work_proxy = router_score_cells + (
        routed_expert_token_work + shared_expert_token_work
    ) * EXPERT_HIDDEN_SIZE

    advice_bits = (
        exact.realized_advice_bits + cache_selector_bits + route_code_bits
    )
    advice_kind = (
        "derived"
        if cache_selector_bits == 0 and route_code_bits == 0
        else "proxy"
    )
    kv_description = _kv_network_description(
        dram_fetch_blocks=dram_fetch_blocks,
        pd_blocks=int(base["prefill_to_decode_blocks"]),
        publish_blocks=int(base["publish_blocks"]),
        included_in_claim=True,
    )
    kv_transfer_records = int(kv_description["network_block_transfers"])
    physical_routing_records = len(workload.requests)
    moe_route_occurrence_records = moe_token_occurrences
    physical_occurrence_records = (
        physical_routing_records
        + len(workload.requests)
        + kv_transfer_records
        + moe_route_occurrence_records
    )
    reusable_definition_records = 7 + prefill_chunk_shapes
    runtime_event_cells = {
        "output_token_cells": exact.realized_output_tokens,
        "stop_indicator_cells": len(workload.requests),
        "cache_lookup_result_cells": int(base["candidate_cache_blocks"]),
        "decode_event_slots": schedule["decode_event_slots"],
        "router_score_cells_proxy": router_score_cells,
    }
    runtime_event_cells["total_runtime_value_event_cells_proxy"] = sum(
        runtime_event_cells.values()
    )
    description = {
        "static_g": _static_g_description(),
        "x_input": base["x_description"],
        "a_advice": {
            "length_code_bits_exact": exact.realized_advice_bits,
            "cache_selector_code_bits_proxy": cache_selector_bits,
            "moe_route_code_bits_proxy": route_code_bits,
            "scheduler_choice_bits": 0,
            "routing_policy_choice_bits": 0,
            "worst_case_bits": advice_bits,
            "realized_bits": advice_bits,
            "evidence_kind": advice_kind,
            "note": (
                "Length bits are exact. Cache bitmaps and route fields are "
                "transparent code-length ceilings, not tight lower bounds."
            ),
        },
        "reusable_definitions": {
            "base_template_records": 7,
            "prefill_chunk_shape_records": prefill_chunk_shapes,
            "total_records": reusable_definition_records,
            "evidence_kind": "derived",
        },
        "occurrence_routing_records": {
            "request_routing_records": physical_routing_records,
            "cache_lookup_records": len(workload.requests),
            "prefill_chunk_occurrences": prefill_chunks,
            "decode_tick_occurrences": schedule["decode_ticks"],
            "kv_block_transfer_records": kv_transfer_records,
            "moe_route_occurrence_records": moe_route_occurrence_records,
            "total_physical_occurrence_records": physical_occurrence_records,
            "evidence_kind": "derived",
        },
        "runtime_value_event_cells": {
            **runtime_event_cells,
            "evidence_kind": "proxy",
            "note": (
                "Token/stop/cache cells are derived counts; router score cells "
                "are a structural proxy, not measured arithmetic cost."
            ),
        },
        "expanded_logical_work_units": {
            "prefill_model_tokens": prefill_tokens,
            "decode_model_tokens": exact.realized_output_tokens,
            "runtime_model_tokens": runtime_model_tokens,
            "decode_ticks": schedule["decode_ticks"],
            "decode_active_token_cells": schedule["decode_active_token_cells"],
            "decode_event_slots": schedule["decode_event_slots"],
            "evidence_kind": "derived",
        },
        "physical_kv_network_units": kv_description,
        "moe_physical_units": {
            "moe_layer_token_occurrences": moe_token_occurrences,
            "router_score_cells_proxy": router_score_cells,
            "routed_expert_token_work": routed_expert_token_work,
            "shared_expert_token_work": shared_expert_token_work,
            "moe_structural_work_proxy": moe_structural_work_proxy,
            "route_bits_per_moe_token_proxy": (
                ROUTE_CODE_BITS_PER_MOE_TOKEN_PROXY
                if moe_mechanism == "route_specialized"
                else 0
            ),
            "evidence_kind": "proxy",
        },
        "unmodeled_physical_events": {
            "speculation": "disabled by assumption",
            "faults": "disabled by assumption",
            "wall_clock_timing": "unsupported without service curves",
            "evidence_kind": "unsupported",
        },
    }
    return {
        "point_id": f"{spec.point_id}::{strategy_name}",
        "scenario_id": spec.point_id,
        "claim_level": "physical",
        "strategy_family": "trace_specialized_physical",
        "strategy_variant": strategy_name,
        "mechanisms": {
            "length": "exact_logical",
            "cache": cache_mechanism,
            "moe": moe_mechanism,
            "routing": spec.routing_policy,
        },
        "description_decomposition": description,
        "metrics": {
            "worst_case_advice_bits": advice_bits,
            "realized_advice_bits": advice_bits,
            "advice_bits_per_request": advice_bits / len(workload.requests),
            "advice_bits_per_output_token": advice_bits
            / exact.realized_output_tokens,
            "realized_output_tokens": exact.realized_output_tokens,
            "represented_output_tokens": exact.represented_output_tokens,
            "padded_output_tokens": 0,
            "prefill_recomputation_tokens": prefill_tokens,
            "runtime_model_token_work": runtime_model_tokens,
            "total_model_token_work": runtime_model_tokens,
            "decode_ticks": schedule["decode_ticks"],
            "decode_event_slots": schedule["decode_event_slots"],
            "dram_fetch_kv_blocks": dram_fetch_blocks,
            "network_kv_block_transfers": kv_transfer_records,
            "network_transfer_bytes_proxy": kv_description[
                "network_transfer_bytes_proxy"
            ],
            "router_score_cells_proxy": router_score_cells,
            "routed_expert_token_work": routed_expert_token_work,
            "shared_expert_token_work": shared_expert_token_work,
            "moe_structural_work_proxy": moe_structural_work_proxy,
            "reusable_definition_records": reusable_definition_records,
            "physical_occurrence_records": physical_occurrence_records,
            "latency_seconds": None,
            "gpu_hours": None,
        },
        "metric_evidence": {
            "length_advice": "derived",
            "cache_selector_advice": (
                "proxy" if cache_selector_bits else "derived"
            ),
            "moe_route_advice": "proxy" if route_code_bits else "derived",
            "logical_work": "derived",
            "kv_blocks": "derived",
            "kv_bytes": "proxy",
            "moe_work": "proxy",
            "latency_seconds": "unsupported",
            "gpu_hours": "unsupported",
        },
    }


def extract_pareto_front(
    points: Iterable[Mapping[str, object]],
    *,
    claim_level: str,
    overhead_metric: str,
) -> list[str]:
    """Return nondominated point IDs for advice versus one overhead metric."""

    if claim_level not in CLAIM_LEVELS:
        raise SweepError(f"unknown claim level: {claim_level}")
    candidates: list[Mapping[str, object]] = []
    for point in points:
        if point.get("claim_level") != claim_level:
            continue
        metrics = point.get("metrics")
        if not isinstance(metrics, Mapping):
            raise SweepError("point metrics must be a mapping")
        advice = metrics.get("realized_advice_bits")
        overhead = metrics.get(overhead_metric)
        if not isinstance(advice, (int, float)) or not isinstance(
            overhead, (int, float)
        ):
            raise SweepError(
                f"Pareto metric {overhead_metric!r} must be numeric"
            )
        candidates.append(point)

    front: list[Mapping[str, object]] = []
    for point in candidates:
        metrics = point["metrics"]
        advice = metrics["realized_advice_bits"]
        overhead = metrics[overhead_metric]
        dominated = False
        for other in candidates:
            if other is point:
                continue
            other_metrics = other["metrics"]
            other_advice = other_metrics["realized_advice_bits"]
            other_overhead = other_metrics[overhead_metric]
            if (
                other_advice <= advice
                and other_overhead <= overhead
                and (other_advice < advice or other_overhead < overhead)
            ):
                dominated = True
                break
        if not dominated:
            front.append(point)
    front.sort(
        key=lambda point: (
            point["metrics"]["realized_advice_bits"],
            point["metrics"][overhead_metric],
            str(point["point_id"]),
        )
    )
    return [str(point["point_id"]) for point in front]


def latency_projection_without_service_curve() -> dict[str, object]:
    return {
        "value": None,
        "unit": "seconds",
        "evidence_kind": "unsupported",
        "reason": (
            "No absolute GB200 prefill/decode service curve or RDMA contention "
            "curve is public in the cited deployment report."
        ),
    }


def expand_fast25_hash_id(hash_id: int) -> tuple[str, ...]:
    if type(hash_id) is not int or hash_id < 0:
        raise SweepError("FAST'25 hash IDs must be nonnegative integers")
    return tuple(
        f"fast25:{hash_id}:child64:{child_index}"
        for child_index in range(FAST25_CHILDREN_PER_HASH)
    )


def parse_mooncake_jsonl(path: str | Path) -> tuple[MooncakeTraceRecord, ...]:
    """Parse direct Mooncake FAST'25 JSONL without loading vendor data."""

    trace_path = Path(path)
    records: list[MooncakeTraceRecord] = []
    with trace_path.open("r", encoding="utf-8") as trace_file:
        for line_number, raw_line in enumerate(trace_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SweepError(
                    f"invalid JSON on trace line {line_number}"
                ) from exc
            required = {"timestamp", "input_length", "output_length", "hash_ids"}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise SweepError(
                    f"trace line {line_number} lacks required FAST'25 fields"
                )
            timestamp = payload["timestamp"]
            input_length = payload["input_length"]
            output_length = payload["output_length"]
            hash_ids = payload["hash_ids"]
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise SweepError(
                    f"trace line {line_number} has an invalid timestamp"
                )
            if type(input_length) is not int or input_length <= 0:
                raise SweepError(
                    f"trace line {line_number} has an invalid input_length"
                )
            if type(output_length) is not int or output_length < 0:
                raise SweepError(
                    f"trace line {line_number} has an invalid output_length"
                )
            if not isinstance(hash_ids, list) or any(
                type(hash_id) is not int or hash_id < 0 for hash_id in hash_ids
            ):
                raise SweepError(
                    f"trace line {line_number} has invalid hash_ids"
                )
            expected_hashes = _ceil_div(input_length, FAST25_HASH_BLOCK_TOKENS)
            if len(hash_ids) != expected_hashes:
                raise SweepError(
                    f"trace line {line_number} has {len(hash_ids)} hashes; "
                    f"expected {expected_hashes} for 512-token blocks"
                )
            records.append(
                MooncakeTraceRecord(
                    timestamp_ms=float(timestamp),
                    input_length=input_length,
                    output_length=output_length,
                    hash_ids=tuple(hash_ids),
                )
            )
    if not records:
        raise SweepError("Mooncake trace contains no requests")
    return tuple(records)


def summarize_mooncake_projection(
    records: Sequence[MooncakeTraceRecord],
) -> dict[str, object]:
    if not records:
        raise SweepError("Mooncake projection requires trace records")
    timestamps = [record.timestamp_ms for record in records]
    return {
        "requests": len(records),
        "timestamp_min_ms": min(timestamps),
        "timestamp_max_ms": max(timestamps),
        "input_tokens": sum(record.input_length for record in records),
        "output_tokens": sum(record.output_length for record in records),
        "fast25_hash_records_512": sum(len(record.hash_ids) for record in records),
        "conservative_child_records_64": sum(
            len(record.child_hash_ids_64) for record in records
        ),
        "source_trace_model": "Mooncake FAST'25 anonymized trace",
        "projected_model": "Kimi-2.5",
        "evidence_kind": "proxy",
        "note": (
            "Combining FAST'25 request shapes with Kimi-2.5 is a projection, "
            "not the measured May 2026 Codex benchmark."
        ),
    }


def _scenario_record(
    spec: SweepSpec,
    workload: WorkloadInput,
    base: Mapping[str, object],
) -> dict[str, object]:
    profile = spec.profile
    return {
        "scenario_id": spec.point_id,
        "requests_per_epoch": spec.requests_per_epoch,
        "sessions": int(base["actual_session_count"]),
        "turns_per_session_cap": spec.turns_per_session,
        "profile": asdict(profile),
        "cache_hit_fraction_target": spec.cache_hit_fraction,
        "cache_hit_fraction_realized": base["realized_cache_hit_fraction"],
        "routing_policy": spec.routing_policy,
        "sampling_seed_fixed_in_x": spec.sampling_seed,
        "adaptive_turns": False,
        "decode_logical_slots": workload.decode_slots,
        "decode_ep_ranks_synchronized": True,
        "prompt_token_occurrences": base["prompt_token_occurrences"],
        "realized_output_tokens": base["realized_output_tokens"],
        "candidate_cache_blocks": base["candidate_cache_blocks"],
        "dram_hit_blocks": base["dram_hit_blocks"],
        "prefill_recomputation_tokens": base["prefill_recomputation_tokens"],
        "x_description": base["x_description"],
        "evidence_kind": (
            "configuration"
            if profile.name == "agentic_anchor"
            else "assumption"
        ),
        "note": (
            "Structural screening projection; cache hit target is realized "
            "deterministically per request at full 64-token block granularity."
        ),
    }


def run_sweep(
    specs: Sequence[SweepSpec] | None = None,
    *,
    trace_path: str | Path | None = None,
) -> dict[str, object]:
    validate_evidence_ledger()
    selected_specs = tuple(specs) if specs is not None else screening_design()
    if not selected_specs:
        raise SweepError("sweep requires at least one design point")

    scenarios: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for spec in selected_specs:
        validate_sweep_spec(spec)
        workload, realized_lengths, session_turns = _build_workload(spec)
        base = _scenario_base(
            spec, workload, realized_lengths, session_turns
        )
        scenarios.append(_scenario_record(spec, workload, base))
        points.extend(
            _logical_point(
                spec, workload, realized_lengths, base, strategy
            )
            for strategy in LOGICAL_STRATEGIES
        )
        points.extend(
            _physical_point(
                spec, workload, realized_lengths, base, variant
            )
            for variant in PHYSICAL_VARIANTS
        )

    frontiers: list[dict[str, object]] = []
    logical_metrics = (
        "total_model_token_work",
        "decode_event_slots",
        "logical_occurrence_records",
    )
    physical_metrics = (
        "total_model_token_work",
        "prefill_recomputation_tokens",
        "router_score_cells_proxy",
        "routed_expert_token_work",
        "moe_structural_work_proxy",
        "physical_occurrence_records",
    )
    for spec in selected_specs:
        scenario_points = [
            point for point in points if point["scenario_id"] == spec.point_id
        ]
        for claim_level, metrics in (
            ("logical", logical_metrics),
            ("physical", physical_metrics),
        ):
            for metric in metrics:
                frontiers.append(
                    {
                        "scenario_id": spec.point_id,
                        "claim_level": claim_level,
                        "overhead_metric": metric,
                        "point_ids": extract_pareto_front(
                            scenario_points,
                            claim_level=claim_level,
                            overhead_metric=metric,
                        ),
                    }
                )

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": (
            "Compile(G,x,a) sourced frontier-inference-cluster structural sweep"
        ),
        "scope": {
            "claim": (
                "description and structural-work decomposition for a finite, "
                "deterministic offline epoch"
            ),
            "not_claimed": [
                "latency",
                "throughput capacity",
                "GPU-hours",
                "DRAM capacity sufficiency",
                "RDMA contention",
                "adaptive cross-epoch tool behavior",
            ],
            "claim_levels": {
                "logical": (
                    "outputs/model-token work, stopping, deterministic "
                    "scheduling, and canonical cache policy"
                ),
                "physical": (
                    "logical claim plus selected routing, cache lifecycle, "
                    "prefill chunks, KV transfers, and MoE mechanisms"
                ),
            },
        },
        "sources": SOURCES,
        "evidence_ledger": [
            {
                **asdict(record),
                "source_ids": list(record.source_ids),
            }
            for record in EVIDENCE_LEDGER
        ],
        "model_configuration": MODEL_CONFIGURATION,
        "deployment_configuration": DEPLOYMENT_CONFIGURATION,
        "reported_anchor": REPORTED_ANCHOR,
        "assumptions": {
            "finite_offline_epoch": True,
            "adaptive_turns_in_epoch": False,
            "model_sampling_and_seed_fixed_in_x": True,
            "deterministic_logical_tie_breaking": True,
            "text_only": True,
            "external_cache_tier": "distributed DRAM only",
            "ssd_enabled": False,
            "canonical_request_output_order": True,
            "canonical_post_stop_padding": True,
            "vllm_kv_block_tokens": KV_BLOCK_TOKENS,
            "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
            "logical_decode_slots_not_capacity_claim": LOGICAL_DECODE_SLOTS,
            "evidence_kind": "assumption",
        },
        "sweep_design": {
            "kind": "deterministic maximin screening design plus corners",
            "points": len(selected_specs),
            "axes": {
                "requests_per_epoch": [32, 128, 512],
                "profiles": [profile.name for profile in WORKLOAD_PROFILES],
                "turns": [1, 4, 16, 30],
                "cache_hit_fraction": [0.0, 0.5, 0.9, 0.99],
                "routing": list(ROUTING_POLICIES),
                "length_strategy": list(LOGICAL_STRATEGIES),
            },
            "offered_load": {
                "requested_normalized_values": [0.5, 0.8, 0.95, 1.1],
                "value": None,
                "evidence_kind": "unsupported",
                "reason": (
                    "No explicit capacity normalization or absolute service "
                    "curve is available."
                ),
            },
        },
        "metric_provenance": {
            "exact_or_derived": [
                "mixed-radix and bucket advice bits",
                "canonical JSON byte measurements",
                "record, token, block, chunk, and event counts",
                "Pareto nondominance within each claim level",
            ],
            "proxy": [
                "KV bytes and transfer bytes",
                "MoE route code lengths",
                "router score cells and expert-token work",
                "u32 token and 32-byte hash serialization widths",
            ],
            "reported_not_generated": [
                "610 traces and 94.2% potential hit rate",
                "roughly 131:1 input:output and 12K to 80K context growth",
                "3.8x/46x/8.6x benchmark outcomes",
            ],
            "unsupported": latency_projection_without_service_curve(),
        },
        "scenarios": scenarios,
        "strategy_points": points,
        "pareto_frontiers": frontiers,
    }
    if trace_path is not None:
        result["external_trace_projection"] = summarize_mooncake_projection(
            parse_mooncake_jsonl(trace_path)
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic structural frontier sweep. Absolute "
            "latency and GPU-hour results remain unsupported."
        )
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="optional Mooncake FAST'25 JSONL path to parse and summarize",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact canonical JSON",
    )
    args = parser.parse_args(argv)
    result = run_sweep(trace_path=args.trace)
    print(canonical_json(result, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
