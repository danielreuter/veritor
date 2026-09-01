import json
import math
from dataclasses import replace

import pytest

from prototypes.advice_workload import (
    RequestInput,
    WorkloadInput,
    minimum_length_advice_bits,
)
from prototypes.frontier_cluster_sweep import (
    CLAIM_LEVELS,
    DEPLOYMENT_CONFIGURATION,
    EVIDENCE_KINDS,
    EVIDENCE_LEDGER,
    FAST25_CHILDREN_PER_HASH,
    KV_BLOCK_TOKENS,
    LOGICAL_STRATEGIES,
    MODEL_CONTEXT_TOKENS,
    PHYSICAL_VARIANTS,
    SOURCES,
    SweepError,
    SweepSpec,
    account_length_strategy,
    canonical_json,
    exact_length_advice_bits,
    expand_fast25_hash_id,
    extract_pareto_front,
    geometric_bucket_ceiling,
    geometric_bucket_count,
    geometric_length_advice_bits,
    parse_mooncake_jsonl,
    run_sweep,
    screening_design,
    summarize_mooncake_projection,
    validate_evidence_ledger,
    validate_sweep_spec,
)


def tiny_spec(
    *,
    point_id: str = "tiny",
    cache_hit_fraction: float = 0.5,
) -> SweepSpec:
    return SweepSpec(
        point_id=point_id,
        requests_per_epoch=4,
        profile_name="short",
        turns_per_session=4,
        cache_hit_fraction=cache_hit_fraction,
        routing_policy="sticky",
        sampling_seed=17,
    )


def test_hand_calculated_exact_geometric_and_max_accounting():
    caps = (13, 13, 13)
    realized = (1, 3, 13)

    exact = account_length_strategy(caps, realized, "exact_logical")
    geometric = account_length_strategy(caps, realized, "geometric_bucket")
    maximum = account_length_strategy(caps, realized, "maximum_envelope")
    replay = account_length_strategy(caps, realized, "replay_in_g")

    assert exact_length_advice_bits(caps) == math.ceil(math.log2(13**3)) == 12
    assert exact.worst_case_advice_bits == exact.realized_advice_bits == 12
    assert exact.represented_lengths == realized
    assert geometric_bucket_count(13) == 5
    assert geometric_length_advice_bits(caps) == math.ceil(math.log2(5**3)) == 7
    assert geometric.represented_lengths == (1, 4, 13)
    assert geometric.padding_output_tokens == 1
    assert maximum.realized_advice_bits == 0
    assert maximum.represented_lengths == caps
    assert maximum.padding_output_tokens == 22
    assert replay.realized_advice_bits == 0
    assert replay.represented_lengths == realized


def test_geometric_bucket_bound_handles_non_power_of_two_caps_exactly():
    cap = 13
    for realized in range(1, cap + 1):
        represented = geometric_bucket_ceiling(realized, cap)
        assert realized <= represented <= cap
        assert represented < 2 * realized
    assert geometric_bucket_ceiling(9, cap) == 13
    assert geometric_bucket_ceiling(13, cap) == 13


def test_exact_bound_matches_existing_mixed_radix_primitive():
    workload = WorkloadInput(
        requests=(
            RequestInput(0, 0, 8, 2),
            RequestInput(1, 0, 8, 3),
            RequestInput(2, 0, 8, 2),
        ),
        decode_slots=2,
    )
    caps = tuple(request.max_new_tokens for request in workload.requests)

    assert exact_length_advice_bits(caps) == 4
    assert exact_length_advice_bits(caps) == minimum_length_advice_bits(workload)


def test_advice_and_work_monotonicity():
    assert exact_length_advice_bits((2, 2, 2)) >= exact_length_advice_bits((2, 2))
    assert exact_length_advice_bits((3, 2)) >= exact_length_advice_bits((2, 2))

    realized = (3, 5)
    exact = account_length_strategy((13, 13), realized, "exact_logical")
    geometric = account_length_strategy((13, 13), realized, "geometric_bucket")
    maximum = account_length_strategy((13, 13), realized, "maximum_envelope")
    assert (
        exact.represented_output_tokens
        <= geometric.represented_output_tokens
        <= maximum.represented_output_tokens
    )


def test_finite_epoch_and_context_validation():
    with pytest.raises(SweepError, match="positive"):
        validate_sweep_spec(replace(tiny_spec(), requests_per_epoch=0))
    with pytest.raises(SweepError, match="adaptive"):
        validate_sweep_spec(replace(tiny_spec(), adaptive_turns=True))
    with pytest.raises(SweepError, match="256K"):
        validate_sweep_spec(
            SweepSpec(
                point_id="too-long",
                requests_per_epoch=32,
                profile_name="long",
                turns_per_session=32,
                cache_hit_fraction=0.9,
                routing_policy="cache-aware",
            )
        )

    valid_long = replace(
        tiny_spec(),
        profile_name="long",
        turns_per_session=30,
    )
    validate_sweep_spec(valid_long)
    profile = valid_long.profile
    worst_case = (
        profile.common_prefix_tokens
        + profile.first_input_tokens
        + 29 * (profile.incremental_input_tokens + profile.max_output_tokens)
        + profile.max_output_tokens
    )
    assert worst_case <= MODEL_CONTEXT_TOKENS


def test_screening_design_and_json_output_are_deterministic():
    left_design = screening_design()
    right_design = screening_design()
    assert left_design == right_design
    assert len(left_design) == 28
    assert len({spec.point_id for spec in left_design}) == 28
    assert {spec.requests_per_epoch for spec in left_design} == {32, 128, 512}

    first = run_sweep((tiny_spec(),))
    second = run_sweep((tiny_spec(),))
    serialized = canonical_json(first)
    assert serialized == canonical_json(second)
    assert json.loads(serialized) == first
    assert len(first["strategy_points"]) == len(LOGICAL_STRATEGIES) + len(
        PHYSICAL_VARIANTS
    )


def test_mooncake_jsonl_parsing_and_conservative_child_mapping(tmp_path):
    trace_path = tmp_path / "tiny.jsonl"
    trace_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "timestamp": 12,
                        "input_length": 600,
                        "output_length": 9,
                        "hash_ids": [46, 47],
                    }
                ),
                json.dumps(
                    {
                        "timestamp": 25.5,
                        "input_length": 512,
                        "output_length": 3,
                        "hash_ids": [46],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = parse_mooncake_jsonl(trace_path)
    assert len(records) == 2
    assert records[0].timestamp_ms == 12.0
    assert len(records[0].child_hash_ids_64) == 2 * FAST25_CHILDREN_PER_HASH
    assert len(expand_fast25_hash_id(46)) == 8
    assert len(set(expand_fast25_hash_id(46))) == 8
    assert expand_fast25_hash_id(46)[0] == "fast25:46:child64:0"

    summary = summarize_mooncake_projection(records)
    assert summary["fast25_hash_records_512"] == 3
    assert summary["conservative_child_records_64"] == 24
    assert summary["evidence_kind"] == "proxy"
    assert "not the measured May 2026" in summary["note"]


def test_mooncake_parser_rejects_inconsistent_hash_count(tmp_path):
    trace_path = tmp_path / "bad.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "timestamp": 0,
                "input_length": 513,
                "output_length": 1,
                "hash_ids": [1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SweepError, match="expected 2"):
        parse_mooncake_jsonl(trace_path)


def test_source_metadata_and_evidence_kinds_are_complete():
    validate_evidence_ledger()
    assert EVIDENCE_KINDS == {
        "reported",
        "configuration",
        "assumption",
        "derived",
        "proxy",
        "unsupported",
    }
    for source in SOURCES.values():
        assert source["title"]
        assert source["url"].startswith("https://")
        assert source["publisher"]
        assert source["date"]
    for evidence in EVIDENCE_LEDGER:
        assert evidence.kind in EVIDENCE_KINDS
        assert evidence.label and evidence.claim
        if evidence.kind in {"reported", "configuration"}:
            assert evidence.source_ids
        assert set(evidence.source_ids) <= set(SOURCES)


def test_description_decomposition_keeps_components_independent():
    result = run_sweep((tiny_spec(),))
    for point in result["strategy_points"]:
        decomposition = point["description_decomposition"]
        assert set(decomposition) >= {
            "static_g",
            "x_input",
            "a_advice",
            "reusable_definitions",
            "occurrence_routing_records",
            "runtime_value_event_cells",
            "expanded_logical_work_units",
            "physical_kv_network_units",
        }
        assert decomposition["static_g"]["canonical_json_bytes_measured"] > 0
        assert decomposition["x_input"]["request_metadata_bytes_measured"] > 0
        assert decomposition["a_advice"]["realized_bits"] >= 0
        assert decomposition["reusable_definitions"]["total_records"] > 0
        assert (
            decomposition["runtime_value_event_cells"]["evidence_kind"]
            in EVIDENCE_KINDS
        )


def test_pareto_extraction_never_mixes_claim_levels():
    points = [
        {
            "point_id": "logical-zero",
            "claim_level": "logical",
            "metrics": {"realized_advice_bits": 0, "work": 1},
        },
        {
            "point_id": "physical-low-advice",
            "claim_level": "physical",
            "metrics": {"realized_advice_bits": 5, "work": 10},
        },
        {
            "point_id": "physical-low-work",
            "claim_level": "physical",
            "metrics": {"realized_advice_bits": 7, "work": 5},
        },
        {
            "point_id": "physical-dominated",
            "claim_level": "physical",
            "metrics": {"realized_advice_bits": 8, "work": 12},
        },
    ]

    assert extract_pareto_front(
        points, claim_level="logical", overhead_metric="work"
    ) == ["logical-zero"]
    assert extract_pareto_front(
        points, claim_level="physical", overhead_metric="work"
    ) == ["physical-low-advice", "physical-low-work"]
    assert CLAIM_LEVELS == ("logical", "physical")


def test_no_latency_or_gpu_hour_claim_without_service_curve():
    result = run_sweep((tiny_spec(),))
    assert result["sweep_design"]["offered_load"]["value"] is None
    assert (
        result["sweep_design"]["offered_load"]["evidence_kind"]
        == "unsupported"
    )
    assert (
        result["metric_provenance"]["unsupported"]["evidence_kind"]
        == "unsupported"
    )
    for point in result["strategy_points"]:
        assert point["metrics"]["latency_seconds"] is None
        assert point["metrics"]["gpu_hours"] is None
        assert point["metric_evidence"]["latency_seconds"] == "unsupported"
        assert point["metric_evidence"]["gpu_hours"] == "unsupported"


def test_cache_hit_reduces_recomputation_but_byte_result_remains_proxy():
    no_hit = run_sweep((tiny_spec(point_id="h0", cache_hit_fraction=0.0),))
    high_hit = run_sweep((tiny_spec(point_id="h9", cache_hit_fraction=0.9),))

    def cache_router(result):
        return next(
            point
            for point in result["strategy_points"]
            if point.get("strategy_variant")
            == "physical_cache_compute_router"
        )

    no_hit_point = cache_router(no_hit)
    high_hit_point = cache_router(high_hit)
    assert (
        high_hit_point["metrics"]["prefill_recomputation_tokens"]
        < no_hit_point["metrics"]["prefill_recomputation_tokens"]
    )
    assert (
        high_hit_point["description_decomposition"]["physical_kv_network_units"][
            "kv_block_tokens"
        ]
        == KV_BLOCK_TOKENS
    )
    assert high_hit_point["metric_evidence"]["kv_bytes"] == "proxy"


def test_decode_ep_ranks_are_not_modeled_as_eight_servers():
    decode = DEPLOYMENT_CONFIGURATION["decode"]
    assert decode["parallelism"] == "DP8+EP8"
    assert decode["ep_ranks_synchronized"] is True
    assert decode["independent_decode_servers"] == 1
