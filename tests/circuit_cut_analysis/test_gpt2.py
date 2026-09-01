from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import pytest

from circuit_cut_analysis.accounting import ExactPartitionStatus, PrimitiveVector
from circuit_cut_analysis.cli import main
from circuit_cut_analysis.indexed import IndexedCircuit
from circuit_cut_analysis.models.gpt2 import (
    GPT2Config,
    analyze_gpt2_decode_step,
    analyze_gpt2_execution,
    expected_decode_step_unit_gate_total,
)
from circuit_cut_analysis.profiles import (
    ALL_FP32_REFERENCE,
    VLLM_FP16_REFERENCE,
)
from circuit_cut_analysis.report import (
    render_csv,
    render_json,
    render_markdown,
    write_reports,
)

EXPECTED_TOTAL = 42_361_101_422
EXPECTED_CONTRACTION_FLOPS = 42_257_061_888
EXPECTED_PREFILL_ARITHMETIC = 17_301_481_495
EXPECTED_DECODE_ARITHMETIC = 25_059_619_827


def _rows_by_id() -> dict[str, object]:
    return {row.row_id: row for row in analyze_gpt2_execution().rows}


def test_gpt2_small_dimensions() -> None:
    config = GPT2Config()

    assert config.model_id == "gpt2-small"
    assert config.layers == 12
    assert config.hidden_size == 768
    assert config.heads == 12
    assert config.head_size == 64
    assert config.intermediate_size == 3072
    assert config.vocabulary_size == 50_257
    assert config.max_context == 1024


@pytest.mark.parametrize("context", [1, 128, 1024])
def test_lower_level_decode_step_keeps_audited_affine_formula(context: int) -> None:
    analysis = analyze_gpt2_decode_step(context)

    assert analysis.total_unit_gates == 247_489_873 + 37_584 * context
    assert expected_decode_step_unit_gate_total(context) == analysis.total_unit_gates
    assert analysis.contraction_flops == 247_064_064 + 36_864 * context
    assert analysis.output_semantics == "full logit vector"


@pytest.mark.parametrize("context", [0, 1025])
def test_lower_level_context_must_fit_declared_window(context: int) -> None:
    with pytest.raises(ValueError, match=r"\[1, 1024\]"):
        analyze_gpt2_decode_step(context)


def test_small_schedule_has_no_extra_final_forward() -> None:
    analysis = analyze_gpt2_execution(prompt_tokens=2, generated_tokens=3)
    schedule = analysis.metadata["schedule"]

    assert schedule["designated_outputs"] == ["y1", "y2", "y3"]
    assert schedule["generated_output_positions"] == [3, 4, 5]
    assert schedule["prefill_visible_kv_lengths"] == {
        "first": 1,
        "last": 2,
        "sum": 3,
    }
    assert schedule["prefill_lm_head_positions"] == [2]
    assert schedule["decode_forward_count"] == 2
    assert schedule["decode_visible_kv_lengths"] == [3, 4]
    assert schedule["decode_visible_kv_length_sum"] == 7
    assert schedule["last_processed_position"] == 4
    assert schedule["final_generated_position"] == 5
    assert schedule["forward_at_final_generated_position"] is False
    assert 5 not in schedule["decode_visible_kv_lengths"]


def test_default_full_execution_exact_totals() -> None:
    analysis = analyze_gpt2_execution()
    accounting = analysis.metadata["accounting"]

    assert accounting["body_expression"] == "170345378 + 37584 * n"
    assert accounting["lm_head_unit_gates"] == 77_144_495
    assert (
        accounting["prefill_unit_gates_excluding_argmax"] == EXPECTED_PREFILL_ARITHMETIC
    )
    assert (
        accounting["decode_unit_gates_excluding_argmax"] == EXPECTED_DECODE_ARITHMETIC
    )
    assert accounting["prefill_contraction_flops"] == 17_250_289_152
    assert accounting["decode_contraction_flops"] == 25_006_772_736
    assert analysis.total_unit_gates == EXPECTED_TOTAL
    assert analysis.contraction_flops == EXPECTED_CONTRACTION_FLOPS
    assert analysis.total_primitives.argmax == 100


def test_phase_rows_independently_sum_to_total() -> None:
    analysis = analyze_gpt2_execution()
    rows = {row.row_id: row for row in analysis.rows}

    assert rows["prefill-transformer-body"].primitives.total == 17_224_337_000
    assert rows["prefill-final-lm-head"].primitives.total == 77_144_495
    assert rows["prefill-first-argmax"].primitives == PrimitiveVector(argmax=1)
    assert rows["decode-transformer-body"].primitives.total == 17_422_314_822
    assert rows["decode-lm-heads"].primitives.total == 7_637_305_005
    assert rows["decode-argmax"].primitives == PrimitiveVector(argmax=99)
    assert sum(row.primitives.total for row in analysis.rows) == EXPECTED_TOTAL


def test_token_capacity_is_semantic_not_storage_width() -> None:
    token = analyze_gpt2_execution().metadata["token_gate"]
    capacity = token["semantic_capacity_bits"]

    assert token["alphabet_cardinality"] == 50_257
    assert token["semantic_capacity_expression_bits"] == "log2(50257)"
    assert capacity == math.log2(50_257)
    assert token["minimum_fixed_width_storage_bits"] == 16
    assert token["possible_runtime_storage_widths_bits"] == [16, 32, 64]
    assert token["separator_capacity_uses_storage_width"] is False
    assert capacity < 16 < 2 * capacity < capacity + 16 < 32 < 3 * capacity


def test_token_and_kv_feedback_are_explicit_macro_dependencies() -> None:
    analysis = analyze_gpt2_execution(prompt_tokens=2, generated_tokens=3)
    dependencies = analysis.metadata["cross_step_dependencies"]

    assert dependencies["token_feedback"]["dynamic_feedback_edges"] == 2 * 768
    assert "y_j -> embedding_lookup_add" in dependencies["token_feedback"]["relation"]
    assert dependencies["token_feedback"]["lookup_index"] == "generated token ID"
    assert dependencies["kv_cache"]["computed_kv_values"] == 4 * 12 * 2 * 768
    assert "later attention-score" in dependencies["kv_cache"]["key_relation"]
    assert "later attention-value" in dependencies["kv_cache"]["value_relation"]
    assert dependencies["kv_cache"]["cache_copy_or_load_gates"] is False


def test_full_indexed_graph_has_a_covered_global_partition() -> None:
    analysis = analyze_gpt2_execution()
    support = analysis.metadata["exact_partition_support"]
    payload = analysis.as_dict()["exact_canonical_partition"]

    assert analysis.partition_status is ExactPartitionStatus.COMPUTED
    assert support["status"] == "COMPUTED"
    assert support["solver"] == "bounded-cut-order hierarchical lift"
    assert support["global_canonical_partition_computed"] is True
    assert support["global_partition_rows_reported"] is True
    assert support["covered_computed_source_gates"] == 42_387_408_594
    assert support["covered_primitive_gates"] == EXPECTED_TOTAL
    assert support["materialized_scalar_nodes"] == 0
    assert support["evaluated_index_regions"] == 155_683
    assert support["retained_region_descriptors"] == 304
    assert support["validated_edge_rules"] == 119
    assert payload["status"] == "COMPUTED"
    assert len(payload["reasons"]) >= 3
    assert len(analysis.bottlenecks) == 23
    assert (
        sum(row.represented_primitives.total for row in analysis.bottlenecks)
        == analysis.total_unit_gates
    )
    assert (
        sum(row.source_gate_count for row in analysis.bottlenecks)
        == support["covered_computed_source_gates"]
    )


def test_default_global_pass_never_materializes_scalar_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_materialization(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("full analysis attempted scalar materialization")

    monkeypatch.setattr(IndexedCircuit, "materialize", reject_materialization)
    analysis = analyze_gpt2_execution()

    assert analysis.partition_status is ExactPartitionStatus.COMPUTED
    assert (
        analysis.metadata["exact_partition_support"]["materialized_scalar_nodes"] == 0
    )


def test_global_rows_merge_sources_by_exact_canonical_cut_type() -> None:
    analysis = analyze_gpt2_execution()
    rows = {row.row_id: row for row in analysis.bottlenecks}

    q = rows["q-projection-output"]
    residual = rows["mlp-residual"]
    layernorm = rows["layernorm-inverse-std"]
    softmax = rows["softmax-probability"]
    token = rows["token-output"]
    pair = rows["penultimate-output-pair"]
    dead = rows["dead-empty-cut"]

    assert q.occurrence_count == 1_664_256
    assert q.cut_width_expression_bits == "16"
    assert q.represented_primitives == PrimitiveVector(
        add=1_278_148_608,
        mul=1_278_148_608,
    )
    assert residual.represented_primitives.total == 10_278_766_080
    assert residual.cut_gate_count == 1
    assert layernorm.cut_width_expression_bits == "32"
    assert softmax.cut_width_expression_bits == "16"
    assert token.cut_width_expression_bits == "log2(50257)"
    assert pair.cut_gate_count == 2
    assert pair.cut_width_min_bits == pytest.approx(2 * math.log2(50_257))
    assert dead.cut_gate_count == 0
    assert dead.cut_width_min_bits == 0
    assert "global" in token.global_minimum_status.lower()


def test_one_generated_output_requires_no_decode_forward() -> None:
    analysis = analyze_gpt2_execution(prompt_tokens=4, generated_tokens=1)

    assert analysis.metadata["schedule"]["decode_forward_count"] == 0
    assert analysis.metadata["schedule"]["decode_visible_kv_lengths"] == []
    assert len(analysis.rows) == 3
    assert analysis.total_primitives.argmax == 1


@pytest.mark.parametrize(
    ("prompt", "generated", "match"),
    [
        (0, 1, "positive"),
        (1, 0, "positive"),
        (1024, 2, "context window"),
    ],
)
def test_execution_schedule_validation(
    prompt: int,
    generated: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        analyze_gpt2_execution(prompt, generated)


def test_profiles_change_width_metadata_not_operation_counts() -> None:
    mixed = analyze_gpt2_execution(profile=VLLM_FP16_REFERENCE)
    fp32 = analyze_gpt2_execution(profile=ALL_FP32_REFERENCE)

    assert mixed.total_primitives == fp32.total_primitives
    assert mixed.contraction_flops == fp32.contraction_flops
    assert mixed.metadata["profile"]["widths_bits"]["kv_cache"] == 16
    assert fp32.metadata["profile"]["widths_bits"]["kv_cache"] == 32
    assert mixed.partition_status is ExactPartitionStatus.COMPUTED
    assert fp32.partition_status is ExactPartitionStatus.UNSUPPORTED
    assert "two token-capacity gates" in " ".join(fp32.partition_reasons)


def test_report_rendering_and_paths_are_reproducible(tmp_path: Path) -> None:
    analysis = analyze_gpt2_execution()
    paths = write_reports(analysis, tmp_path)
    stem = "gpt2-small-vllm-fp16-reference-p100-g100"

    assert paths.json == tmp_path / f"{stem}.json"
    assert paths.csv == tmp_path / f"{stem}.csv"
    assert paths.markdown == tmp_path / f"{stem}.md"
    assert paths.json.read_text(encoding="utf-8") == render_json(analysis)
    assert paths.csv.read_text(encoding="utf-8") == render_csv(analysis)
    assert paths.markdown.read_text(encoding="utf-8") == render_markdown(analysis)

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["report_kind"] == "gpt2-full-execution-canonical-cut-partition"
    assert payload["analysis"]["total_unit_gates"] == EXPECTED_TOTAL
    assert payload["analysis"]["exact_canonical_partition"]["status"] == "COMPUTED"

    csv_rows = list(csv.DictReader(io.StringIO(paths.csv.read_text(encoding="utf-8"))))
    assert len(csv_rows) == 23
    assert {row["exact_partition_status"] for row in csv_rows} == {"COMPUTED"}
    assert any(row["bottleneck"].startswith("MLP residual") for row in csv_rows)

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "42,361,101,422" in markdown
    assert "42,387,485,394" in markdown
    assert "COMPUTED" in markdown
    assert "MLP residual coordinate" in markdown
    assert "304 retained source-family descriptors" in markdown
    assert "15.617036934" in markdown
    assert "```" not in markdown


def test_cli_generates_default_full_execution_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(("gpt2", "--output-dir", str(tmp_path)))

    assert result == 0
    assert len(tuple(tmp_path.iterdir())) == 3
    output = capsys.readouterr().out
    assert "prompt=100, generated=100: 42,361,101,422 primitive gates" in output
    assert "global canonical partition: COMPUTED" in output
    assert "155,683 evaluated, 304 retained; 0 scalar nodes materialized" in output
