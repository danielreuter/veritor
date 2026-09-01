"""Internal deterministic manifest and capability helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import StrEnum
from fractions import Fraction
from typing import cast

from circuit_cut_analysis.models.capacity_profile import ModelCapacityProfile
from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    CapabilityStatus,
    EvidenceStatus,
    JSONValue,
    SupportState,
)

from .api import AssumptionRecord


def manifest_value(value: object) -> JSONValue:
    """Convert immutable request metadata to canonical identity data.

    Floats are represented by their exact hexadecimal spelling because the
    core canonical JSON contract intentionally rejects binary floats.
    """

    if value is None or type(value) in (bool, int, str):
        return cast(JSONValue, value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Fraction):
        return {
            "denominator": value.denominator,
            "numerator": value.numerator,
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity metadata cannot contain non-finite floats")
        return {"binary64_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: manifest_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("identity mappings require string keys")
            result[key] = manifest_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [manifest_value(item) for item in value]
    raise TypeError(f"unsupported identity metadata {type(value).__name__}")


def profile_manifest(profile: ModelCapacityProfile) -> dict[str, JSONValue]:
    """Return all aggregate profile claims in deterministic identity form."""

    return {
        "generated_tokens": profile.generated_tokens,
        "logical_vocabulary_size": profile.logical_vocabulary_size,
        "model_id": profile.model_id,
        "numerical_profile_id": profile.numerical_profile_id,
        "prompt_tokens": profile.prompt_tokens,
        "regions": [
            {
                "description": region.description,
                "gate_count": region.gate_count,
                "id": region.id,
                "self_cut_bits_per_gate": manifest_value(region.self_cut_bits_per_gate),
                "value_cardinality_upper_bound": (region.value_cardinality_upper_bound),
            }
            for region in profile.regions
        ],
        "assumptions": list(profile.assumptions),
    }


def assumption_records(
    assumptions: Sequence[str],
    *,
    source: str,
    prefix: str = "assumption",
) -> tuple[AssumptionRecord, ...]:
    return tuple(
        AssumptionRecord(
            code=f"{prefix}-{index:03d}",
            statement=statement,
            source=source,
        )
        for index, statement in enumerate(assumptions)
    )


def capability(
    capability_value: Capability,
    state: SupportState,
    artifact_kind: ArtifactKind,
    *,
    guarantee: str,
    evidence: EvidenceStatus,
    reason_code: str | None = None,
    detail: str = "",
) -> CapabilityStatus:
    return CapabilityStatus(
        capability=capability_value,
        state=state,
        artifact_kind=artifact_kind,
        guarantee=guarantee,
        reason_code=reason_code,
        detail=detail,
        evidence=evidence,
    )


def capability_report(
    plugin_id: str,
    artifact_kind: ArtifactKind,
    statuses: Sequence[CapabilityStatus],
) -> CapabilityReport:
    return CapabilityReport(plugin_id, artifact_kind, statuses)
