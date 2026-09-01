"""Verifier-local trust roots and backend allowlists."""

from __future__ import annotations

from dataclasses import dataclass, field

from veritor.commitment import ValueCommitmentRegistry

from .artifact import TrustedArtifactResolver
from .evidence import SampleEvidenceRegistry
from .model import StagedProtocolError


@dataclass(frozen=True, slots=True)
class TrustedVerificationContext:
    """Everything resolved locally rather than trusted from transcript bytes."""

    artifact_resolver: TrustedArtifactResolver
    value_commitment_backends: ValueCommitmentRegistry = field(
        default_factory=ValueCommitmentRegistry.with_defaults
    )
    sample_evidence_backends: SampleEvidenceRegistry = field(
        default_factory=SampleEvidenceRegistry.with_defaults
    )

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_resolver, TrustedArtifactResolver):
            raise StagedProtocolError(
                "artifact_resolver does not satisfy TrustedArtifactResolver"
            )
        if not isinstance(
            self.value_commitment_backends,
            ValueCommitmentRegistry,
        ):
            raise StagedProtocolError("value_commitment_backends has the wrong type")
        if not isinstance(
            self.sample_evidence_backends,
            SampleEvidenceRegistry,
        ):
            raise StagedProtocolError("sample_evidence_backends has the wrong type")
