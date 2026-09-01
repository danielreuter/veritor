"""Compressed, additive accounting records for repeated circuit motifs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any


class MinimumCutTieStatus(StrEnum):
    """Tie evidence available for a reported cut."""

    NOT_EVALUATED = "not_evaluated"
    UNIQUE = "unique"
    TIED = "tied"


class ExactPartitionStatus(StrEnum):
    """Whether an exact canonical partition was actually computed."""

    COMPUTED = "COMPUTED"
    GRAPH_READY = "GRAPH_READY"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class PrimitiveVector:
    """Exact counts in the declared unit-gate basis."""

    add: int = 0
    mul: int = 0
    max: int = 0
    exp: int = 0
    reciprocal: int = 0
    rsqrt: int = 0
    tanh: int = 0
    argmax: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value < 0:
                raise ValueError(f"primitive count {field.name} cannot be negative")

    @property
    def total(self) -> int:
        return sum(getattr(self, field.name) for field in fields(self))

    def scale(self, multiplier: int) -> PrimitiveVector:
        if multiplier < 0:
            raise ValueError("primitive multiplier cannot be negative")
        return PrimitiveVector(
            **{
                field.name: getattr(self, field.name) * multiplier
                for field in fields(self)
            }
        )

    def __add__(self, other: PrimitiveVector) -> PrimitiveVector:
        if not isinstance(other, PrimitiveVector):
            return NotImplemented
        return PrimitiveVector(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )

    def as_dict(self, *, include_zeros: bool = False) -> dict[str, int]:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        if include_zeros:
            return values
        return {name: value for name, value in values.items() if value}


@dataclass(frozen=True, slots=True)
class OperationLedgerRecord:
    """One additive operation-accounting row, without an inferred cut."""

    row_id: str
    phase: str
    component: str
    occurrence_count: int
    primitives: PrimitiveVector
    description: str

    def __post_init__(self) -> None:
        if not self.row_id or not self.phase or not self.component:
            raise ValueError("operation ledger identifiers must be non-empty")
        if self.occurrence_count <= 0:
            raise ValueError("operation occurrence count must be positive")
        if self.primitives.total <= 0:
            raise ValueError("operation ledger row must contain work")
        if not self.description:
            raise ValueError("operation description must be non-empty")

    def as_dict(self, *, total_unit_gates: int) -> dict[str, Any]:
        if total_unit_gates <= 0:
            raise ValueError("total unit gates must be positive")
        represented = self.primitives.total
        return {
            "row_id": self.row_id,
            "phase": self.phase,
            "component": self.component,
            "occurrence_count": self.occurrence_count,
            "represented_unit_gates": represented,
            "represented_primitives": self.primitives.as_dict(),
            "unit_primitive_gate_share": represented / total_unit_gates,
            "unit_primitive_gate_share_numerator": represented,
            "unit_primitive_gate_share_denominator": total_unit_gates,
            "unit_primitive_gate_percentage": 100 * represented / total_unit_gates,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class WiringBottleneckRecord:
    """Aggregated exact cut occurrences and their owned primitive sources."""

    row_id: str
    bottleneck: str
    boundary_families: tuple[str, ...]
    cut_width_expression_bits: str
    cut_width_min_bits: float
    cut_width_max_bits: float
    occurrence_count: int
    represented_primitives: PrimitiveVector
    upstream_operations_per_cut: str
    cut_certificate: str
    global_minimum_status: str
    source_gate_count: int = 0
    cut_gate_count: int = 1
    certificate_kind: str = "local-separator"

    def __post_init__(self) -> None:
        if not self.row_id or not self.bottleneck:
            raise ValueError("bottleneck identifiers must be non-empty")
        if not self.boundary_families:
            raise ValueError(f"{self.row_id}: boundary families cannot be empty")
        if not self.cut_width_expression_bits:
            raise ValueError(f"{self.row_id}: cut width expression cannot be empty")
        if self.cut_width_min_bits < 0:
            raise ValueError(f"{self.row_id}: cut width cannot be negative")
        if self.cut_width_max_bits < self.cut_width_min_bits:
            raise ValueError(f"{self.row_id}: invalid cut width interval")
        if self.occurrence_count <= 0:
            raise ValueError(f"{self.row_id}: occurrence count must be positive")
        if self.represented_primitives.total <= 0:
            raise ValueError(f"{self.row_id}: represented work cannot be empty")
        if not self.upstream_operations_per_cut:
            raise ValueError(f"{self.row_id}: upstream operation summary is required")
        if not self.cut_certificate or not self.global_minimum_status:
            raise ValueError(f"{self.row_id}: cut status fields must be non-empty")
        if self.source_gate_count < 0:
            raise ValueError(f"{self.row_id}: source gate count cannot be negative")
        if self.cut_gate_count < 0:
            raise ValueError(f"{self.row_id}: cut gate count cannot be negative")
        if self.cut_width_max_bits == 0 and self.cut_gate_count != 0:
            raise ValueError(f"{self.row_id}: zero-width cuts cannot contain gates")
        if self.cut_width_min_bits > 0 and self.cut_gate_count == 0:
            raise ValueError(f"{self.row_id}: positive-width cuts need gates")
        if not self.certificate_kind:
            raise ValueError(f"{self.row_id}: certificate kind is required")

    def as_dict(self, *, total_unit_gates: int) -> dict[str, Any]:
        represented = self.represented_primitives.total
        return {
            "row_id": self.row_id,
            "bottleneck": self.bottleneck,
            "boundary_families": list(self.boundary_families),
            "cut_width_expression_bits": self.cut_width_expression_bits,
            "cut_width_min_bits": self.cut_width_min_bits,
            "cut_width_max_bits": self.cut_width_max_bits,
            "occurrence_count": self.occurrence_count,
            "source_gate_count": self.source_gate_count,
            "cut_gate_count": self.cut_gate_count,
            "represented_unit_gates": represented,
            "represented_primitives": self.represented_primitives.as_dict(),
            "unit_primitive_gate_share": represented / total_unit_gates,
            "unit_primitive_gate_share_numerator": represented,
            "unit_primitive_gate_share_denominator": total_unit_gates,
            "unit_primitive_gate_percentage": 100 * represented / total_unit_gates,
            "upstream_operations_per_cut": self.upstream_operations_per_cut,
            "cut_certificate": self.cut_certificate,
            "certificate_kind": self.certificate_kind,
            "global_minimum_status": self.global_minimum_status,
        }


@dataclass(frozen=True, slots=True)
class ExecutionAnalysis:
    """A complete multi-step execution ledger and partition-support statement."""

    model_id: str
    profile_id: str
    prompt_tokens: int
    generated_tokens: int
    output_semantics: str
    rows: tuple[OperationLedgerRecord, ...]
    bottlenecks: tuple[WiringBottleneckRecord, ...]
    contraction_flops: int
    partition_status: ExactPartitionStatus
    partition_reasons: tuple[str, ...]
    metadata: dict[str, Any]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.generated_tokens <= 0:
            raise ValueError("prompt and generated token counts must be positive")
        if not self.rows:
            raise ValueError("execution analysis must contain operation rows")
        if not self.bottlenecks:
            raise ValueError("execution analysis must contain bottleneck rows")
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("execution row ids must be unique")
        bottleneck_ids = [row.row_id for row in self.bottlenecks]
        if len(bottleneck_ids) != len(set(bottleneck_ids)):
            raise ValueError("bottleneck row ids must be unique")
        if self.contraction_flops < 0:
            raise ValueError("contraction FLOPs cannot be negative")
        if self.partition_status is not ExactPartitionStatus.COMPUTED:
            if not self.partition_reasons:
                raise ValueError("incomplete partitions require concrete reasons")
        represented = sum(row.represented_primitives.total for row in self.bottlenecks)
        if represented != self.total_unit_gates:
            raise ValueError(
                "bottleneck rows must partition primitive work exactly: "
                f"{represented} != {self.total_unit_gates}"
            )

    @property
    def total_primitives(self) -> PrimitiveVector:
        total = PrimitiveVector()
        for row in self.rows:
            total += row.primitives
        return total

    @property
    def total_unit_gates(self) -> int:
        return self.total_primitives.total

    def as_dict(self) -> dict[str, Any]:
        total = self.total_unit_gates
        return {
            "model_id": self.model_id,
            "profile_id": self.profile_id,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "output_semantics": self.output_semantics,
            "total_unit_gates": total,
            "total_primitives": self.total_primitives.as_dict(include_zeros=True),
            "contraction_flops": self.contraction_flops,
            "exact_canonical_partition": {
                "status": self.partition_status.value,
                "reasons": list(self.partition_reasons),
            },
            "metadata": self.metadata,
            "assumptions": list(self.assumptions),
            "bottlenecks": [
                row.as_dict(total_unit_gates=total) for row in self.bottlenecks
            ],
            "rows": [row.as_dict(total_unit_gates=total) for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class BottleneckRecord:
    """One repeated, non-overlapping cut-owned region."""

    row_id: str
    location: str
    component: str
    cut_type: str
    cut_gate_widths_bits: tuple[int, ...]
    occurrence_count: int
    gates_per_occurrence: PrimitiveVector
    ownership_scope: str
    profile_id: str
    cut_basis: str
    logical_or_materialized: str = "logical"
    minimum_cut_tie_status: MinimumCutTieStatus = MinimumCutTieStatus.NOT_EVALUATED
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row id must be non-empty")
        if not self.cut_gate_widths_bits:
            raise ValueError(f"{self.row_id}: a cut must contain at least one gate")
        if any(width <= 0 for width in self.cut_gate_widths_bits):
            raise ValueError(f"{self.row_id}: cut widths must be positive")
        if self.occurrence_count <= 0:
            raise ValueError(f"{self.row_id}: occurrence count must be positive")
        if self.gates_per_occurrence.total <= 0:
            raise ValueError(f"{self.row_id}: owned region must contain work")
        if not self.ownership_scope or not self.cut_basis:
            raise ValueError(
                f"{self.row_id}: ownership and cut basis must be non-empty"
            )

    @property
    def cut_gate_count_each(self) -> int:
        return len(self.cut_gate_widths_bits)

    @property
    def cut_width_bits_each(self) -> int:
        return sum(self.cut_gate_widths_bits)

    @property
    def represented_primitives(self) -> PrimitiveVector:
        return self.gates_per_occurrence.scale(self.occurrence_count)

    @property
    def represented_unit_gates(self) -> int:
        return self.represented_primitives.total

    def as_dict(self, *, total_unit_gates: int) -> dict[str, Any]:
        if total_unit_gates <= 0:
            raise ValueError("total unit gates must be positive")
        return {
            "row_id": self.row_id,
            "location": self.location,
            "component": self.component,
            "cut_type": self.cut_type,
            "cut_gate_count_each": self.cut_gate_count_each,
            "cut_gate_widths_bits": list(self.cut_gate_widths_bits),
            "cut_width_bits_each": self.cut_width_bits_each,
            "occurrence_count": self.occurrence_count,
            "unit_gates_per_occurrence": self.gates_per_occurrence.total,
            "primitive_gates_per_occurrence": self.gates_per_occurrence.as_dict(),
            "represented_unit_gates": self.represented_unit_gates,
            "represented_primitives": self.represented_primitives.as_dict(),
            "unit_primitive_gate_share": (
                self.represented_unit_gates / total_unit_gates
            ),
            "unit_primitive_gate_share_numerator": self.represented_unit_gates,
            "unit_primitive_gate_share_denominator": total_unit_gates,
            "unit_primitive_gate_percentage": (
                100 * self.represented_unit_gates / total_unit_gates
            ),
            "ownership_scope": self.ownership_scope,
            "profile_id": self.profile_id,
            "cut_basis": self.cut_basis,
            "logical_or_materialized": self.logical_or_materialized,
            "minimum_cut_tie_status": self.minimum_cut_tie_status.value,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class ModelAnalysis:
    model_id: str
    profile_id: str
    context_length: int
    context_includes_current_token: bool
    output_semantics: str
    rows: tuple[BottleneckRecord, ...]
    contraction_flops: int
    metadata: dict[str, Any]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("analysis row ids must be unique")
        if not self.rows:
            raise ValueError("analysis must contain at least one row")
        if self.context_length <= 0:
            raise ValueError("context length must be positive")
        if self.contraction_flops < 0:
            raise ValueError("contraction FLOPs cannot be negative")
        mismatched_profiles = {
            row.profile_id for row in self.rows if row.profile_id != self.profile_id
        }
        if mismatched_profiles:
            raise ValueError(
                "row profile ids must match the analysis profile: "
                f"{sorted(mismatched_profiles)!r}"
            )

    @property
    def total_primitives(self) -> PrimitiveVector:
        total = PrimitiveVector()
        for row in self.rows:
            total += row.represented_primitives
        return total

    @property
    def total_unit_gates(self) -> int:
        return self.total_primitives.total

    def as_dict(self) -> dict[str, Any]:
        total = self.total_unit_gates
        return {
            "model_id": self.model_id,
            "profile_id": self.profile_id,
            "context_length": self.context_length,
            "context_includes_current_token": self.context_includes_current_token,
            "output_semantics": self.output_semantics,
            "total_unit_gates": total,
            "total_primitives": self.total_primitives.as_dict(include_zeros=True),
            "contraction_flops": self.contraction_flops,
            "metadata": self.metadata,
            "assumptions": list(self.assumptions),
            "rows": [row.as_dict(total_unit_gates=total) for row in self.rows],
        }
