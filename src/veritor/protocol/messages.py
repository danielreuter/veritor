"""Messages of the two-stage protocol, verifier outcomes, and errors.

The protocol has five messages after the public header::

    prover   -> verifier   BoundaryMessage    (commit beta, open public I/O)
    verifier -> prover     ReplayChallenge    (q seed, J)
    prover   -> verifier   InteriorMessage    (commit alpha_r for r in J)
    verifier -> prover     SampleChallenge    (s seed, T)
    prover   -> verifier   EvidenceMessage    (openings for every sampled unit)

A :class:`Transcript` is the header plus these five messages in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from veritor.core import (
    Digest,
    InvalidArtifact,
    JSONValue,
    VerificationPolicy,
    identity_digest,
    validate_digest,
)

PROTOCOL_VERSION = "veritor/protocol/v2"


class ProtocolError(InvalidArtifact):
    """A party used the protocol API incorrectly (not a verdict)."""


class VerificationCode(StrEnum):
    """Why the verifier accepted or rejected."""

    ACCEPTED = "accepted"
    EXPECTATION_MISMATCH = "expectation_mismatch"
    POLICY_REJECTED = "policy_rejected"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    INVALID_PHASE = "invalid_phase"
    INVALID_COMMITMENT = "invalid_commitment"
    INVALID_OPENING = "invalid_opening"
    INVALID_VALUE = "invalid_value"
    PUBLIC_IO_MISMATCH = "public_io_mismatch"
    CHALLENGE_MISMATCH = "challenge_mismatch"
    COVERAGE_MISMATCH = "coverage_mismatch"
    RELATION_REJECTED = "relation_rejected"
    INVALID_COMPILED_RESULT = "invalid_compiled_result"
    MALFORMED_TRANSCRIPT = "malformed_transcript"
    NONCANONICAL_TRANSCRIPT = "noncanonical_transcript"
    RESOURCE_LIMIT = "resource_limit"
    TRUSTED_SERVICE_FAILURE = "trusted_service_failure"


class Reject(ProtocolError):
    """The verifier rejects; ``code`` says why."""

    def __init__(self, code: VerificationCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The verifier's verdict for one protocol run."""

    code: VerificationCode
    detail: str = ""
    sampled_replay_units: tuple[int, ...] = ()
    sampled_verification_units: tuple[int, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.code is VerificationCode.ACCEPTED


def _bytes32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ProtocolError(f"{name} must be exactly 32 bytes")
    return value


def _sorted_unique(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int or item < 0 for item in value):
        raise ProtocolError(f"{name} must be a tuple of nonnegative unit indices")
    if tuple(sorted(set(value))) != value:
        raise ProtocolError(f"{name} must be sorted and unique")
    return value


def _bytes_tuple(value: object, name: str) -> tuple[bytes, ...]:
    if type(value) is not tuple or any(type(item) is not bytes for item in value):
        raise ProtocolError(f"{name} must be a tuple of bytes")
    return value


def raw_digest(tag: str, manifest: JSONValue) -> bytes:
    """Tagged canonical-manifest SHA-256 as raw bytes."""

    return bytes.fromhex(identity_digest(tag, manifest))


@dataclass(frozen=True, slots=True)
class Commitment:
    """A Merkle root over one verifier-derived position domain."""

    root: bytes
    count: int

    def __post_init__(self) -> None:
        _bytes32(self.root, "commitment root")
        if type(self.count) is not int or self.count < 0:
            raise ProtocolError("commitment count must be a nonnegative integer")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"count": self.count, "root": self.root.hex()}


@dataclass(frozen=True, slots=True)
class Opening:
    """An authenticated value at one position of one commitment."""

    position: int
    value: bytes
    path: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ProtocolError("opening position must be a nonnegative integer")
        if type(self.value) is not bytes:
            raise ProtocolError("opening value must be bytes")
        if type(self.path) is not tuple or any(
            type(item) is not bytes or len(item) != 32 for item in self.path
        ):
            raise ProtocolError("opening path must be a tuple of 32-byte digests")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "path": [item.hex() for item in self.path],
            "position": self.position,
            "value": self.value.hex(),
        }


@dataclass(frozen=True, slots=True)
class Header:
    """Public parameters both parties fix before any message is sent."""

    session_id: bytes
    compiled_digest: Digest
    policy: VerificationPolicy
    public_inputs: tuple[bytes, ...]
    claimed_outputs: tuple[bytes, ...]
    digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.session_id) is not bytes or not self.session_id:
            raise ProtocolError("session_id must be nonempty bytes")
        object.__setattr__(
            self, "compiled_digest", validate_digest(self.compiled_digest, "compiled digest")
        )
        if not isinstance(self.policy, VerificationPolicy):
            raise ProtocolError("policy must be a VerificationPolicy")
        _bytes_tuple(self.public_inputs, "public_inputs")
        _bytes_tuple(self.claimed_outputs, "claimed_outputs")
        object.__setattr__(
            self,
            "digest",
            raw_digest(
                "veritor/protocol/header/v2",
                {
                    "claimed_outputs": [item.hex() for item in self.claimed_outputs],
                    "compiled_digest": self.compiled_digest,
                    "policy": self.policy.manifest,
                    "protocol_version": PROTOCOL_VERSION,
                    "public_inputs": [item.hex() for item in self.public_inputs],
                    "session_id": self.session_id.hex(),
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class BoundaryMessage:
    commitment: Commitment
    io_openings: tuple[Opening, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, Commitment):
            raise ProtocolError("boundary commitment has the wrong type")
        if type(self.io_openings) is not tuple or any(
            not isinstance(item, Opening) for item in self.io_openings
        ):
            raise ProtocolError("io_openings must be a tuple of openings")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "commitment": self.commitment.manifest,
            "io_openings": [item.manifest for item in self.io_openings],
        }


@dataclass(frozen=True, slots=True)
class ReplayChallenge:
    seed: bytes
    selected: tuple[int, ...]

    def __post_init__(self) -> None:
        _bytes32(self.seed, "q seed")
        _sorted_unique(self.selected, "selected replay units")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"seed": self.seed.hex(), "selected": list(self.selected)}


@dataclass(frozen=True, slots=True)
class InteriorMessage:
    """One interior commitment per selected replay unit, in ``J`` order."""

    commitments: tuple[Commitment, ...]

    def __post_init__(self) -> None:
        if type(self.commitments) is not tuple or any(
            not isinstance(item, Commitment) for item in self.commitments
        ):
            raise ProtocolError("interior commitments must be a tuple of commitments")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"commitments": [item.manifest for item in self.commitments]}


@dataclass(frozen=True, slots=True)
class SampleChallenge:
    seed: bytes
    selected: tuple[int, ...]

    def __post_init__(self) -> None:
        _bytes32(self.seed, "s seed")
        _sorted_unique(self.selected, "selected verification units")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"seed": self.seed.hex(), "selected": list(self.selected)}


@dataclass(frozen=True, slots=True)
class EvidenceMessage:
    """One opening batch per sampled verification unit, in ``T`` order."""

    units: tuple[tuple[Opening, ...], ...]

    def __post_init__(self) -> None:
        if type(self.units) is not tuple or any(
            type(batch) is not tuple
            or any(not isinstance(item, Opening) for item in batch)
            for batch in self.units
        ):
            raise ProtocolError("evidence must be a tuple of opening tuples")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "units": [[item.manifest for item in batch] for batch in self.units]
        }


@dataclass(frozen=True, slots=True)
class Transcript:
    header: Header
    boundary: BoundaryMessage
    replay_challenge: ReplayChallenge
    interiors: InteriorMessage
    sample_challenge: SampleChallenge
    evidence: EvidenceMessage
