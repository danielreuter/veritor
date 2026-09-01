"""Immutable messages and verifier-owned outcomes for the staged protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from veritor.commitment import (
    MERKLE_SHA256_V1,
    ValueCommitment,
    ValueOpening,
)
from veritor.core import (
    ArtifactKind,
    Capability,
    InvalidArtifact,
    Unsupported,
    VerificationPolicy,
    validate_digest,
)

STAGED_TRANSCRIPT_VERSION = "veritor.staged-transcript/v1"
TRANSPARENT_LOCAL_CHECK_V1 = "veritor.staged.transparent-local-check/v1"


class StagedProtocolError(InvalidArtifact):
    """Base error for malformed staged-protocol objects."""


def _bytes(
    value: object,
    name: str,
    *,
    length: int | None = None,
    allow_empty: bool = False,
) -> bytes:
    if type(value) is not bytes:
        raise StagedProtocolError(f"{name} must be bytes")
    if length is not None and len(value) != length:
        raise StagedProtocolError(f"{name} must be exactly {length} bytes")
    if not allow_empty and not value:
        raise StagedProtocolError(f"{name} must not be empty")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise StagedProtocolError(f"{name} must be a nonempty string")
    return value


def _indices(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise StagedProtocolError(f"{name} must be a tuple")
    for item in value:
        if type(item) is not int or item < 0:
            raise StagedProtocolError(
                f"{name} must contain nonnegative integer indices"
            )
    return value


def _encoded_values(value: object, name: str) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise StagedProtocolError(f"{name} must be a tuple")
    for item in value:
        _bytes(item, f"{name} entry", allow_empty=True)
    return value


@dataclass(frozen=True, slots=True)
class SessionParameters:
    """Public session and trusted-backend binding."""

    session_id: bytes
    compiled_result_digest: str
    policy_digest: str
    value_commitment_backend_id: str
    sample_evidence_backend_id: str
    protocol_version: str = STAGED_TRANSCRIPT_VERSION

    def __post_init__(self) -> None:
        _bytes(self.session_id, "session_id")
        validate_digest(self.compiled_result_digest, "compiled_result_digest")
        validate_digest(self.policy_digest, "policy_digest")
        _identifier(
            self.value_commitment_backend_id,
            "value_commitment_backend_id",
        )
        _identifier(self.sample_evidence_backend_id, "sample_evidence_backend_id")
        if self.protocol_version != STAGED_TRANSCRIPT_VERSION:
            raise StagedProtocolError("unknown staged transcript version")


@dataclass(frozen=True, slots=True)
class PublicStatement:
    """Canonical encoded public inputs and ordered claimed outputs."""

    public_inputs: tuple[bytes, ...]
    claimed_outputs: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _encoded_values(self.public_inputs, "public_inputs")
        _encoded_values(self.claimed_outputs, "claimed_outputs")


@dataclass(frozen=True, slots=True)
class BoundaryMessage:
    """First prover message: boundary root and mandatory public-I/O openings."""

    commitment: ValueCommitment
    public_io_openings: tuple[ValueOpening, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, ValueCommitment):
            raise StagedProtocolError("boundary commitment has the wrong type")
        if type(self.public_io_openings) is not tuple or any(
            not isinstance(opening, ValueOpening) for opening in self.public_io_openings
        ):
            raise StagedProtocolError(
                "public_io_openings must be a tuple of ValueOpening values"
            )


@dataclass(frozen=True, slots=True)
class ReplayChallengeMessage:
    """Verifier q-seed reveal and the exact derived replay-unit set J."""

    seed: bytes
    boundary_phase_digest: bytes
    selected_replay_units: tuple[int, ...]
    phase_digest: bytes

    def __post_init__(self) -> None:
        _bytes(self.seed, "q seed", length=32)
        _bytes(self.boundary_phase_digest, "boundary_phase_digest", length=32)
        _indices(self.selected_replay_units, "selected_replay_units")
        _bytes(self.phase_digest, "q phase_digest", length=32)


@dataclass(frozen=True, slots=True)
class OwnedValueCommitment:
    """One selected replay unit and its interior commitment."""

    replay_unit_index: int
    commitment: ValueCommitment

    def __post_init__(self) -> None:
        if type(self.replay_unit_index) is not int or self.replay_unit_index < 0:
            raise StagedProtocolError("replay_unit_index must be a nonnegative integer")
        if not isinstance(self.commitment, ValueCommitment):
            raise StagedProtocolError("unit commitment has the wrong type")


@dataclass(frozen=True, slots=True)
class UnitCommitmentsMessage:
    """Ordered roots for exactly the replay units selected by q."""

    q_phase_digest: bytes
    commitments: tuple[OwnedValueCommitment, ...]
    phase_digest: bytes

    def __post_init__(self) -> None:
        _bytes(self.q_phase_digest, "q_phase_digest", length=32)
        if type(self.commitments) is not tuple or any(
            not isinstance(item, OwnedValueCommitment) for item in self.commitments
        ):
            raise StagedProtocolError(
                "unit commitments must be a tuple of OwnedValueCommitment values"
            )
        _bytes(self.phase_digest, "unit commitments phase_digest", length=32)


@dataclass(frozen=True, slots=True)
class SampleChallengeMessage:
    """Verifier s-seed reveal and the exact derived verification-unit set T."""

    seed: bytes
    unit_commitments_phase_digest: bytes
    selected_verification_units: tuple[int, ...]
    phase_digest: bytes

    def __post_init__(self) -> None:
        _bytes(self.seed, "s seed", length=32)
        _bytes(
            self.unit_commitments_phase_digest,
            "unit_commitments_phase_digest",
            length=32,
        )
        _indices(
            self.selected_verification_units,
            "selected_verification_units",
        )
        _bytes(self.phase_digest, "sample phase_digest", length=32)


@dataclass(frozen=True, slots=True)
class SampledUnitEvidence:
    """Backend-tagged evidence for one sampled verification unit."""

    verification_unit_index: int
    backend_id: str
    payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.verification_unit_index) is not int
            or self.verification_unit_index < 0
        ):
            raise StagedProtocolError(
                "verification_unit_index must be a nonnegative integer"
            )
        _identifier(self.backend_id, "sample evidence backend_id")
        _bytes(self.payload, "sample evidence payload", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SampleEvidenceMessage:
    """Evidence with exact T coverage and an explicit phase binding."""

    sample_phase_digest: bytes
    units: tuple[SampledUnitEvidence, ...]

    def __post_init__(self) -> None:
        _bytes(self.sample_phase_digest, "sample_phase_digest", length=32)
        if type(self.units) is not tuple or any(
            not isinstance(item, SampledUnitEvidence) for item in self.units
        ):
            raise StagedProtocolError(
                "sample evidence must be a tuple of SampledUnitEvidence values"
            )


@dataclass(frozen=True, slots=True)
class StagedTranscript:
    """Complete evidence transcript; deliberately contains no acceptance bit."""

    session: SessionParameters
    statement: PublicStatement
    policy: VerificationPolicy
    boundary: BoundaryMessage
    replay_challenge: ReplayChallengeMessage
    unit_commitments: UnitCommitmentsMessage
    sample_challenge: SampleChallengeMessage
    sample_evidence: SampleEvidenceMessage
    version: str = STAGED_TRANSCRIPT_VERSION

    def __post_init__(self) -> None:
        if self.version != STAGED_TRANSCRIPT_VERSION:
            raise StagedProtocolError("unknown staged transcript version")
        expected_types = (
            (self.session, SessionParameters, "session"),
            (self.statement, PublicStatement, "statement"),
            (self.policy, VerificationPolicy, "policy"),
            (self.boundary, BoundaryMessage, "boundary"),
            (
                self.replay_challenge,
                ReplayChallengeMessage,
                "replay_challenge",
            ),
            (
                self.unit_commitments,
                UnitCommitmentsMessage,
                "unit_commitments",
            ),
            (
                self.sample_challenge,
                SampleChallengeMessage,
                "sample_challenge",
            ),
            (self.sample_evidence, SampleEvidenceMessage, "sample_evidence"),
        )
        for value, expected, name in expected_types:
            if not isinstance(value, expected):
                raise StagedProtocolError(f"{name} has the wrong type")


@dataclass(frozen=True, slots=True)
class VerificationExpectation:
    """Verifier-local statement, policy, identity, backends, and random seeds.

    The verifier must keep ``q_seed`` secret until the boundary is fixed and
    ``s_seed`` secret until the selected replay-unit roots are fixed.
    """

    session_id: bytes
    compiled_result_digest: str
    policy: VerificationPolicy
    public_inputs: tuple[object, ...]
    claimed_outputs: tuple[object, ...]
    q_seed: bytes
    s_seed: bytes
    value_commitment_backend_id: str = MERKLE_SHA256_V1
    sample_evidence_backend_id: str = TRANSPARENT_LOCAL_CHECK_V1

    def __post_init__(self) -> None:
        _bytes(self.session_id, "session_id")
        validate_digest(self.compiled_result_digest, "compiled_result_digest")
        if not isinstance(self.policy, VerificationPolicy):
            raise StagedProtocolError("policy must be a VerificationPolicy")
        if type(self.public_inputs) is not tuple:
            raise StagedProtocolError("public_inputs must be a tuple")
        if type(self.claimed_outputs) is not tuple:
            raise StagedProtocolError("claimed_outputs must be a tuple")
        _identifier(
            self.value_commitment_backend_id,
            "value_commitment_backend_id",
        )
        _identifier(self.sample_evidence_backend_id, "sample_evidence_backend_id")
        _bytes(self.q_seed, "expected q seed", length=32)
        _bytes(self.s_seed, "expected s seed", length=32)


class VerificationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"


class VerificationCode(StrEnum):
    """Stable machine-readable verifier outcomes."""

    ACCEPTED = "ACCEPTED"
    MALFORMED_TRANSCRIPT = "MALFORMED_TRANSCRIPT"
    NONCANONICAL_TRANSCRIPT = "NONCANONICAL_TRANSCRIPT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    EXPECTATION_MISMATCH = "EXPECTATION_MISMATCH"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    INVALID_COMPILED_RESULT = "INVALID_COMPILED_RESULT"
    UNSUPPORTED_ARTIFACT = "UNSUPPORTED_ARTIFACT"
    UNKNOWN_COMMITMENT_BACKEND = "UNKNOWN_COMMITMENT_BACKEND"
    UNKNOWN_EVIDENCE_BACKEND = "UNKNOWN_EVIDENCE_BACKEND"
    INVALID_BOUNDARY = "INVALID_BOUNDARY"
    INVALID_COMMITMENT = "INVALID_COMMITMENT"
    INVALID_OPENING = "INVALID_OPENING"
    PUBLIC_IO_MISMATCH = "PUBLIC_IO_MISMATCH"
    INVALID_PHASE = "INVALID_PHASE"
    CHALLENGE_MISMATCH = "CHALLENGE_MISMATCH"
    COVERAGE_MISMATCH = "COVERAGE_MISMATCH"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    RELATION_REJECTED = "RELATION_REJECTED"
    TRUSTED_SERVICE_FAILURE = "TRUSTED_SERVICE_FAILURE"


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Pure verifier result, separate from prover-controlled evidence."""

    status: VerificationStatus
    code: VerificationCode
    detail: str
    transcript_digest: str | None = None
    unsupported: Unsupported | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", VerificationStatus(self.status))
            object.__setattr__(self, "code", VerificationCode(self.code))
        except (TypeError, ValueError) as error:
            raise StagedProtocolError(
                "verification report has an unknown enum"
            ) from error
        if type(self.detail) is not str:
            raise StagedProtocolError("verification report detail must be a string")
        if self.transcript_digest is not None:
            validate_digest(self.transcript_digest, "transcript_digest")
        if self.unsupported is not None and not isinstance(
            self.unsupported, Unsupported
        ):
            raise StagedProtocolError("unsupported must be a core Unsupported outcome")
        if self.status is VerificationStatus.ACCEPTED:
            if (
                self.code is not VerificationCode.ACCEPTED
                or self.unsupported is not None
            ):
                raise StagedProtocolError("accepted report has inconsistent fields")
        elif self.status is VerificationStatus.UNSUPPORTED:
            if (
                self.code is not VerificationCode.UNSUPPORTED_ARTIFACT
                or self.unsupported is None
            ):
                raise StagedProtocolError("unsupported report has inconsistent fields")
        elif self.code is VerificationCode.ACCEPTED or self.unsupported is not None:
            raise StagedProtocolError("rejected report has inconsistent fields")

    @property
    def accepted(self) -> bool:
        return self.status is VerificationStatus.ACCEPTED

    @classmethod
    def accept(cls, data: bytes) -> VerificationReport:
        return cls(
            VerificationStatus.ACCEPTED,
            VerificationCode.ACCEPTED,
            "transcript verified",
            hashlib.sha256(data).hexdigest(),
        )

    @classmethod
    def reject(
        cls,
        code: VerificationCode,
        detail: str,
        data: bytes | None = None,
    ) -> VerificationReport:
        return cls(
            VerificationStatus.REJECTED,
            code,
            detail,
            None if data is None else hashlib.sha256(data).hexdigest(),
        )

    @classmethod
    def unsupported_artifact(
        cls,
        outcome: Unsupported,
        data: bytes,
    ) -> VerificationReport:
        return cls(
            VerificationStatus.UNSUPPORTED,
            VerificationCode.UNSUPPORTED_ARTIFACT,
            outcome.detail,
            hashlib.sha256(data).hexdigest(),
            outcome,
        )


def unsupported_execution_artifact(
    *,
    artifact_kind: ArtifactKind,
    detail: str,
) -> Unsupported:
    """Construct the stable typed unsupported result used by verification."""

    return Unsupported(
        capability=Capability.VERIFY,
        plugin_id="veritor.staged",
        reason_code="NO_EXECUTABLE_RELATIONS",
        detail=detail,
        artifact_kind=artifact_kind,
    )


def rational_pair(value: Fraction) -> list[int]:
    normalized = Fraction(value)
    return [normalized.numerator, normalized.denominator]
