"""Deterministic full-execution GPT-2 JSON, CSV, and Markdown reports."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from circuit_cut_analysis.accounting import ExecutionAnalysis, PrimitiveVector

REPORT_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class ReportPaths:
    json: Path
    csv: Path
    markdown: Path


def _safe_stem_part(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    if not sanitized:
        raise ValueError("report stem component must contain a safe character")
    return sanitized


def report_stem(analysis: ExecutionAnalysis) -> str:
    return "-".join(
        (
            _safe_stem_part(analysis.model_id),
            _safe_stem_part(analysis.profile_id),
            f"p{analysis.prompt_tokens}",
            f"g{analysis.generated_tokens}",
        )
    )


def report_payload(analysis: ExecutionAnalysis) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": "gpt2-full-execution-canonical-cut-partition",
        "analysis": analysis.as_dict(),
    }


def render_json(analysis: ExecutionAnalysis) -> str:
    return (
        json.dumps(
            report_payload(analysis),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


_CSV_FIELDS = (
    "model_id",
    "profile_id",
    "prompt_tokens",
    "generated_tokens",
    "output_semantics",
    "exact_partition_status",
    "total_unit_gates",
    "contraction_flops",
    "row_id",
    "bottleneck",
    "boundary_families",
    "cut_width_expression_bits",
    "cut_width_min_bits",
    "cut_width_max_bits",
    "cut_gate_count",
    "occurrence_count",
    "source_gate_count",
    "represented_unit_gates",
    "represented_primitives",
    "unit_primitive_gate_share_fraction",
    "unit_primitive_gate_percentage",
    "upstream_operations_per_cut",
    "cut_certificate",
    "certificate_kind",
    "global_minimum_status",
)


def _percentage(numerator: int, denominator: int, places: int = 9) -> str:
    with localcontext() as context:
        context.prec = 40
        percentage = Decimal(100 * numerator) / Decimal(denominator)
        return f"{percentage:.{places}f}"


def _compact_primitives(primitives: PrimitiveVector) -> str:
    return json.dumps(
        primitives.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def render_csv(analysis: ExecutionAnalysis) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    total = analysis.total_unit_gates
    for row in analysis.bottlenecks:
        represented = row.represented_primitives.total
        writer.writerow(
            {
                "model_id": analysis.model_id,
                "profile_id": analysis.profile_id,
                "prompt_tokens": analysis.prompt_tokens,
                "generated_tokens": analysis.generated_tokens,
                "output_semantics": analysis.output_semantics,
                "exact_partition_status": analysis.partition_status.value,
                "total_unit_gates": total,
                "contraction_flops": analysis.contraction_flops,
                "row_id": row.row_id,
                "bottleneck": row.bottleneck,
                "boundary_families": json.dumps(
                    row.boundary_families,
                    separators=(",", ":"),
                ),
                "cut_width_expression_bits": row.cut_width_expression_bits,
                "cut_width_min_bits": f"{row.cut_width_min_bits:.12f}",
                "cut_width_max_bits": f"{row.cut_width_max_bits:.12f}",
                "cut_gate_count": row.cut_gate_count,
                "occurrence_count": row.occurrence_count,
                "source_gate_count": row.source_gate_count,
                "represented_unit_gates": represented,
                "represented_primitives": _compact_primitives(
                    row.represented_primitives
                ),
                "unit_primitive_gate_share_fraction": f"{represented}/{total}",
                "unit_primitive_gate_percentage": _percentage(represented, total),
                "upstream_operations_per_cut": row.upstream_operations_per_cut,
                "cut_certificate": row.cut_certificate,
                "certificate_kind": row.certificate_kind,
                "global_minimum_status": row.global_minimum_status,
            }
        )
    return output.getvalue()


def _markdown_primitives(primitives: PrimitiveVector) -> str:
    return ", ".join(
        f"{name}={count:,}" for name, count in primitives.as_dict().items()
    )


def render_markdown(analysis: ExecutionAnalysis) -> str:
    metadata = analysis.metadata
    schedule = metadata["schedule"]
    accounting = metadata["accounting"]
    token = metadata["token_gate"]
    indexed = metadata["indexed_circuit"]
    support = metadata["exact_partition_support"]
    partition_computed = analysis.partition_status.value == "COMPUTED"
    cut_section = (
        "## Global canonical cut groups"
        if partition_computed
        else "## Wiring-derived candidate separators"
    )
    cut_intro = (
        "Each row aggregates parameterized concrete occurrences of the same exact "
        "canonical cut type. Owned source regions are disjoint and cover every "
        "computed scalar gate; owned primitive counts cover all declared work."
        if partition_computed
        else "The global partition is unsupported for this configuration. These "
        "rows are local wiring separators only; their status fields do not claim "
        "global minimality."
    )
    width_label = "canonical width" if partition_computed else "local width"
    lines = [
        "# GPT-2 Small global canonical-cut analysis",
        "",
        f"- Profile: `{analysis.profile_id}`",
        (
            f"- Execution: fixed {analysis.prompt_tokens}-token prompt, then "
            f"{analysis.generated_tokens} greedy generated token IDs."
        ),
        (
            "- Designated outputs: every generated token gate "
            f"`y1` through `y{analysis.generated_tokens}`."
        ),
        (
            "- Indexed computed-source graph: "
            f"**{indexed['scalar_gate_count_including_zero-work_boundaries']:,} "
            "scalar value gates** represented by "
            f"{indexed['gate_family_count']} gate families and "
            f"{indexed['edge_rule_count']} exact bidirectional wire rules."
        ),
        (
            "- Full global canonical partition: "
            f"**{analysis.partition_status.value}** using "
            f"`{support['solver']}`."
        ),
        (
            "- Partition descriptor: "
            f"{support['evaluated_index_regions']:,} evaluated algebraic regions, "
            f"{support['retained_region_descriptors']:,} retained source-family "
            f"descriptors, and {support['materialized_scalar_nodes']:,} "
            "materialized scalar nodes."
        ),
        (
            "- Atomic token capacity: "
            f"`{token['semantic_capacity_expression_bits']}` = "
            f"{token['semantic_capacity_bits']:.12f} bits; minimum fixed-width "
            f"storage is {token['minimum_fixed_width_storage_bits']} bits."
        ),
        (
            "- The token separator capacity is semantic log-cardinality and is "
            "not rounded to a 16-, 32-, or 64-bit runtime storage width."
        ),
        (
            "- Generated-output positions run from "
            f"{schedule['generated_output_positions'][0]} through "
            f"{schedule['generated_output_positions'][-1]}."
        ),
        (
            "- Decode forwards use visible-KV lengths "
            f"{schedule['decode_visible_kv_lengths'][0]} through "
            f"{schedule['decode_visible_kv_lengths'][-1]}; there is no forward "
            "at the final generated position."
            if schedule["decode_visible_kv_lengths"]
            else "- No decode forwards are required for one generated output."
        ),
        (
            "- Generated-token embedding feedback and persistent K/V reuse are "
            "indexed edges under the declared atomic lookup abstraction."
        ),
        "",
        cut_section,
        "",
        cut_intro,
        "",
        (
            f"| bottleneck | {width_label} (bits) | cut gates | occurrences | "
            "owned sources | represented gates | share | certificate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in analysis.bottlenecks:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.bottleneck,
                    f"`{row.cut_width_expression_bits}`",
                    f"{row.cut_gate_count}",
                    f"{row.occurrence_count:,}",
                    f"{row.source_gate_count:,}",
                    f"{row.represented_primitives.total:,}",
                    (
                        f"{_percentage(row.represented_primitives.total, analysis.total_unit_gates, 6)}%"
                    ),
                    row.certificate_kind,
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Exact accounting",
            "",
        )
    )
    lines.extend(
        [
            (
                f"- Transformer body formula: `{accounting['body_expression']}` "
                "primitive gates for a processed position with `n` visible KV entries."
            ),
            f"- LM head: {accounting['lm_head_unit_gates']:,} primitive gates.",
            (
                "- Prefill arithmetic, excluding its argmax: "
                f"{accounting['prefill_unit_gates_excluding_argmax']:,} gates."
            ),
            (
                "- Decode arithmetic, excluding argmax gates: "
                f"{accounting['decode_unit_gates_excluding_argmax']:,} gates."
            ),
            (
                f"- Total: **{analysis.total_unit_gates:,} primitive gates**, including "
                f"{analysis.generated_tokens} atomic argmax gates."
            ),
            (
                f"- Conventional contraction work: **{analysis.contraction_flops:,} FLOPs**."
            ),
        ]
    )

    lines.extend(("", "## Global-partition status", ""))
    lines.extend(f"- {reason}" for reason in analysis.partition_reasons)
    lines.extend(
        (
            "",
            (
                "`minimum_vertex_cut_indexed` expands bounded corridors and supports "
                "exact non-power-of-two capacities. The full default uses the "
                "capacity-bounded hierarchical lift and materializes no scalar nodes."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The wiring and primitive counts are exact for the declared logical "
                "arithmetic profile. Shares are primitive-gate shares, not latency, "
                "memory traffic, or GPU-utilization estimates. `COMPUTED` means every "
                "non-input family region has a minimum, downstream-most lifted "
                "certificate; other statuses must be read as incomplete."
            ),
            "",
        )
    )
    return "\n".join(lines)


def write_reports(
    analysis: ExecutionAnalysis,
    output_dir: str | Path = "reports",
) -> ReportPaths:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = report_stem(analysis)
    paths = ReportPaths(
        json=destination / f"{stem}.json",
        csv=destination / f"{stem}.csv",
        markdown=destination / f"{stem}.md",
    )
    paths.json.write_text(render_json(analysis), encoding="utf-8")
    paths.csv.write_text(render_csv(analysis), encoding="utf-8")
    paths.markdown.write_text(render_markdown(analysis), encoding="utf-8")
    return paths
