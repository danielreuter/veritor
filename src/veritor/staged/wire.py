"""Strict deterministic canonical JSON codec for staged transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from veritor.commitment import ValueCommitment, ValueOpening
from veritor.core import Position, VerificationLimits, VerificationPolicy

from ._json import (
    NonCanonicalWireError,
    WireCodecError,
    array,
    canonical_bytes,
    exact_keys,
    hex_bytes,
    integer,
    load_strict_json,
    text,
)
from .model import (
    STAGED_TRANSCRIPT_VERSION,
    BoundaryMessage,
    OwnedValueCommitment,
    PublicStatement,
    ReplayChallengeMessage,
    SampleChallengeMessage,
    SampledUnitEvidence,
    SampleEvidenceMessage,
    SessionParameters,
    StagedTranscript,
    UnitCommitmentsMessage,
    rational_pair,
)

_ROOT_KEYS = frozenset(
    {
        "boundary",
        "policy",
        "replay_challenge",
        "sample_challenge",
        "sample_evidence",
        "session",
        "statement",
        "unit_commitments",
        "version",
    }
)

TranscriptCodecError = WireCodecError
NonCanonicalTranscript = NonCanonicalWireError


@dataclass(slots=True)
class _DecodeBudget:
    limits: VerificationLimits
    openings: int = 0
    proof_bytes: int = 0

    def opening(self, value_size: int, path_size: int) -> None:
        self.openings += 1
        self.limits.enforce("max_openings", self.openings)
        self.proof_bytes += value_size + path_size
        self.limits.enforce("max_proof_bytes", self.proof_bytes)

    def payload(self, size: int) -> None:
        self.proof_bytes += size
        self.limits.enforce("max_proof_bytes", self.proof_bytes)


def _commitment_manifest(value: ValueCommitment) -> dict[str, object]:
    return {
        "backend_id": value.backend_id,
        "root": value.root.hex(),
        "value_count": value.value_count,
    }


def _opening_manifest(value: ValueOpening) -> dict[str, object]:
    return {
        "path": [item.hex() for item in value.path],
        "position": int(value.position),
        "value": value.value.hex(),
    }


def transcript_manifest(transcript: StagedTranscript) -> dict[str, object]:
    """Return the fixed, known-field transcript wire object."""

    return {
        "boundary": {
            "commitment": _commitment_manifest(transcript.boundary.commitment),
            "public_io_openings": [
                _opening_manifest(item)
                for item in transcript.boundary.public_io_openings
            ],
        },
        "policy": {
            "eta": rational_pair(transcript.policy.eta),
            "q": rational_pair(transcript.policy.q),
            "s": rational_pair(transcript.policy.s),
        },
        "replay_challenge": {
            "boundary_phase_digest": (
                transcript.replay_challenge.boundary_phase_digest.hex()
            ),
            "phase_digest": transcript.replay_challenge.phase_digest.hex(),
            "seed": transcript.replay_challenge.seed.hex(),
            "selected_replay_units": list(
                transcript.replay_challenge.selected_replay_units
            ),
        },
        "sample_challenge": {
            "phase_digest": transcript.sample_challenge.phase_digest.hex(),
            "seed": transcript.sample_challenge.seed.hex(),
            "selected_verification_units": list(
                transcript.sample_challenge.selected_verification_units
            ),
            "unit_commitments_phase_digest": (
                transcript.sample_challenge.unit_commitments_phase_digest.hex()
            ),
        },
        "sample_evidence": {
            "sample_phase_digest": (
                transcript.sample_evidence.sample_phase_digest.hex()
            ),
            "units": [
                {
                    "backend_id": item.backend_id,
                    "payload": item.payload.hex(),
                    "verification_unit_index": (item.verification_unit_index),
                }
                for item in transcript.sample_evidence.units
            ],
        },
        "session": {
            "compiled_result_digest": (transcript.session.compiled_result_digest),
            "policy_digest": transcript.session.policy_digest,
            "protocol_version": transcript.session.protocol_version,
            "sample_evidence_backend_id": (
                transcript.session.sample_evidence_backend_id
            ),
            "session_id": transcript.session.session_id.hex(),
            "value_commitment_backend_id": (
                transcript.session.value_commitment_backend_id
            ),
        },
        "statement": {
            "claimed_outputs": [
                item.hex() for item in transcript.statement.claimed_outputs
            ],
            "public_inputs": [
                item.hex() for item in transcript.statement.public_inputs
            ],
        },
        "unit_commitments": {
            "commitments": [
                {
                    "commitment": _commitment_manifest(item.commitment),
                    "replay_unit_index": item.replay_unit_index,
                }
                for item in transcript.unit_commitments.commitments
            ],
            "phase_digest": transcript.unit_commitments.phase_digest.hex(),
            "q_phase_digest": (transcript.unit_commitments.q_phase_digest.hex()),
        },
        "version": transcript.version,
    }


def encode_transcript(transcript: StagedTranscript) -> bytes:
    if not isinstance(transcript, StagedTranscript):
        raise TypeError("encode_transcript requires a StagedTranscript")
    return canonical_bytes(transcript_manifest(transcript))


encode_transcript_bytes = encode_transcript


def _decode_commitment(value: object, name: str) -> ValueCommitment:
    obj = exact_keys(
        value,
        frozenset({"backend_id", "root", "value_count"}),
        name,
    )
    return ValueCommitment(
        text(obj["backend_id"], f"{name}.backend_id"),
        hex_bytes(obj["root"], f"{name}.root", allow_empty=True),
        integer(obj["value_count"], f"{name}.value_count"),
    )


def _decode_opening(
    value: object,
    name: str,
    budget: _DecodeBudget,
) -> ValueOpening:
    obj = exact_keys(
        value,
        frozenset({"path", "position", "value"}),
        name,
    )
    raw_path = array(obj["path"], f"{name}.path")
    budget.limits.enforce("max_openings", len(raw_path))
    path = tuple(
        hex_bytes(
            item,
            f"{name}.path[{index}]",
            allow_empty=True,
        )
        for index, item in enumerate(raw_path)
    )
    payload = hex_bytes(obj["value"], f"{name}.value", allow_empty=True)
    budget.opening(len(payload), sum(len(item) for item in path))
    return ValueOpening(
        Position(integer(obj["position"], f"{name}.position")),
        payload,
        path,
    )


def _decode_rational(value: object, name: str) -> Fraction:
    pair = array(value, name)
    if len(pair) != 2:
        raise WireCodecError(f"{name} must be an integer pair")
    numerator = integer(pair[0], f"{name}[0]", nonnegative=False)
    denominator = integer(pair[1], f"{name}[1]")
    if denominator == 0:
        raise WireCodecError(f"{name} denominator must be positive")
    result = Fraction(numerator, denominator)
    if [result.numerator, result.denominator] != [numerator, denominator]:
        raise NonCanonicalWireError(f"{name} rational pair is not reduced")
    return result


def _decode_indices(
    value: object,
    name: str,
    limits: VerificationLimits,
) -> tuple[int, ...]:
    raw = array(value, name)
    limits.enforce("max_units", len(raw))
    return tuple(integer(item, f"{name}[{index}]") for index, item in enumerate(raw))


def _decode_session(value: object) -> SessionParameters:
    obj = exact_keys(
        value,
        frozenset(
            {
                "compiled_result_digest",
                "policy_digest",
                "protocol_version",
                "sample_evidence_backend_id",
                "session_id",
                "value_commitment_backend_id",
            }
        ),
        "session",
    )
    return SessionParameters(
        session_id=hex_bytes(obj["session_id"], "session.session_id"),
        compiled_result_digest=text(
            obj["compiled_result_digest"],
            "session.compiled_result_digest",
        ),
        policy_digest=text(obj["policy_digest"], "session.policy_digest"),
        value_commitment_backend_id=text(
            obj["value_commitment_backend_id"],
            "session.value_commitment_backend_id",
        ),
        sample_evidence_backend_id=text(
            obj["sample_evidence_backend_id"],
            "session.sample_evidence_backend_id",
        ),
        protocol_version=text(
            obj["protocol_version"],
            "session.protocol_version",
        ),
    )


def _decode_statement(
    value: object,
    budget: _DecodeBudget,
) -> PublicStatement:
    obj = exact_keys(
        value,
        frozenset({"claimed_outputs", "public_inputs"}),
        "statement",
    )

    def values(field: str) -> tuple[bytes, ...]:
        raw = array(obj[field], f"statement.{field}")
        budget.limits.enforce("max_positions", len(raw))
        result = tuple(
            hex_bytes(
                item,
                f"statement.{field}[{index}]",
                allow_empty=True,
            )
            for index, item in enumerate(raw)
        )
        for payload in result:
            budget.payload(len(payload))
        return result

    return PublicStatement(values("public_inputs"), values("claimed_outputs"))


def _decode_policy(value: object, limits: VerificationLimits) -> VerificationPolicy:
    obj = exact_keys(
        value,
        frozenset({"eta", "q", "s"}),
        "policy",
    )
    policy = VerificationPolicy(
        _decode_rational(obj["q"], "policy.q"),
        _decode_rational(obj["s"], "policy.s"),
        _decode_rational(obj["eta"], "policy.eta"),
    )
    for name, probability in (
        ("q", policy.q),
        ("s", policy.s),
        ("eta", policy.eta),
    ):
        bits = probability.denominator.bit_length()
        if bits > limits.max_manifest_bytes * 8:
            from veritor.core import ResourceLimit

            raise ResourceLimit(
                f"{name}_denominator_bits",
                limit=limits.max_manifest_bytes * 8,
                observed=bits,
            )
    return policy


def _decode_boundary(
    value: object,
    budget: _DecodeBudget,
) -> BoundaryMessage:
    obj = exact_keys(
        value,
        frozenset({"commitment", "public_io_openings"}),
        "boundary",
    )
    raw_openings = array(
        obj["public_io_openings"],
        "boundary.public_io_openings",
    )
    budget.limits.enforce("max_openings", len(raw_openings))
    return BoundaryMessage(
        _decode_commitment(obj["commitment"], "boundary.commitment"),
        tuple(
            _decode_opening(
                item,
                f"boundary.public_io_openings[{index}]",
                budget,
            )
            for index, item in enumerate(raw_openings)
        ),
    )


def _decode_replay_challenge(
    value: object,
    limits: VerificationLimits,
) -> ReplayChallengeMessage:
    obj = exact_keys(
        value,
        frozenset(
            {
                "boundary_phase_digest",
                "phase_digest",
                "seed",
                "selected_replay_units",
            }
        ),
        "replay_challenge",
    )
    return ReplayChallengeMessage(
        seed=hex_bytes(obj["seed"], "replay_challenge.seed", length=32),
        boundary_phase_digest=hex_bytes(
            obj["boundary_phase_digest"],
            "replay_challenge.boundary_phase_digest",
            length=32,
        ),
        selected_replay_units=_decode_indices(
            obj["selected_replay_units"],
            "replay_challenge.selected_replay_units",
            limits,
        ),
        phase_digest=hex_bytes(
            obj["phase_digest"],
            "replay_challenge.phase_digest",
            length=32,
        ),
    )


def _decode_unit_commitments(
    value: object,
    limits: VerificationLimits,
) -> UnitCommitmentsMessage:
    obj = exact_keys(
        value,
        frozenset({"commitments", "phase_digest", "q_phase_digest"}),
        "unit_commitments",
    )
    raw = array(obj["commitments"], "unit_commitments.commitments")
    limits.enforce("max_units", len(raw))
    commitments: list[OwnedValueCommitment] = []
    for index, item in enumerate(raw):
        owned = exact_keys(
            item,
            frozenset({"commitment", "replay_unit_index"}),
            f"unit_commitments.commitments[{index}]",
        )
        commitments.append(
            OwnedValueCommitment(
                integer(
                    owned["replay_unit_index"],
                    f"unit_commitments.commitments[{index}].replay_unit_index",
                ),
                _decode_commitment(
                    owned["commitment"],
                    f"unit_commitments.commitments[{index}].commitment",
                ),
            )
        )
    return UnitCommitmentsMessage(
        q_phase_digest=hex_bytes(
            obj["q_phase_digest"],
            "unit_commitments.q_phase_digest",
            length=32,
        ),
        commitments=tuple(commitments),
        phase_digest=hex_bytes(
            obj["phase_digest"],
            "unit_commitments.phase_digest",
            length=32,
        ),
    )


def _decode_sample_challenge(
    value: object,
    limits: VerificationLimits,
) -> SampleChallengeMessage:
    obj = exact_keys(
        value,
        frozenset(
            {
                "phase_digest",
                "seed",
                "selected_verification_units",
                "unit_commitments_phase_digest",
            }
        ),
        "sample_challenge",
    )
    return SampleChallengeMessage(
        seed=hex_bytes(obj["seed"], "sample_challenge.seed", length=32),
        unit_commitments_phase_digest=hex_bytes(
            obj["unit_commitments_phase_digest"],
            "sample_challenge.unit_commitments_phase_digest",
            length=32,
        ),
        selected_verification_units=_decode_indices(
            obj["selected_verification_units"],
            "sample_challenge.selected_verification_units",
            limits,
        ),
        phase_digest=hex_bytes(
            obj["phase_digest"],
            "sample_challenge.phase_digest",
            length=32,
        ),
    )


def _decode_sample_evidence(
    value: object,
    budget: _DecodeBudget,
) -> SampleEvidenceMessage:
    obj = exact_keys(
        value,
        frozenset({"sample_phase_digest", "units"}),
        "sample_evidence",
    )
    raw = array(obj["units"], "sample_evidence.units")
    budget.limits.enforce("max_units", len(raw))
    units: list[SampledUnitEvidence] = []
    for index, item in enumerate(raw):
        unit = exact_keys(
            item,
            frozenset({"backend_id", "payload", "verification_unit_index"}),
            f"sample_evidence.units[{index}]",
        )
        payload = hex_bytes(
            unit["payload"],
            f"sample_evidence.units[{index}].payload",
            allow_empty=True,
        )
        budget.payload(len(payload))
        units.append(
            SampledUnitEvidence(
                integer(
                    unit["verification_unit_index"],
                    (f"sample_evidence.units[{index}].verification_unit_index"),
                ),
                text(
                    unit["backend_id"],
                    f"sample_evidence.units[{index}].backend_id",
                ),
                payload,
            )
        )
    return SampleEvidenceMessage(
        hex_bytes(
            obj["sample_phase_digest"],
            "sample_evidence.sample_phase_digest",
            length=32,
        ),
        tuple(units),
    )


def decode_transcript(
    data: bytes,
    limits: VerificationLimits | None = None,
) -> StagedTranscript:
    """Decode only the unique canonical encoding of the fixed v1 schema."""

    checked_limits = VerificationLimits() if limits is None else limits
    document = load_strict_json(data, checked_limits)
    root = exact_keys(document, _ROOT_KEYS, "transcript")
    version = text(root["version"], "version")
    if version != STAGED_TRANSCRIPT_VERSION:
        raise WireCodecError(f"unsupported transcript version {version!r}")
    budget = _DecodeBudget(checked_limits)
    transcript = StagedTranscript(
        session=_decode_session(root["session"]),
        statement=_decode_statement(root["statement"], budget),
        policy=_decode_policy(root["policy"], checked_limits),
        boundary=_decode_boundary(root["boundary"], budget),
        replay_challenge=_decode_replay_challenge(
            root["replay_challenge"],
            checked_limits,
        ),
        unit_commitments=_decode_unit_commitments(
            root["unit_commitments"],
            checked_limits,
        ),
        sample_challenge=_decode_sample_challenge(
            root["sample_challenge"],
            checked_limits,
        ),
        sample_evidence=_decode_sample_evidence(
            root["sample_evidence"],
            budget,
        ),
        version=version,
    )
    if encode_transcript(transcript) != data:
        raise NonCanonicalWireError(
            "transcript bytes are not the unique canonical encoding"
        )
    return transcript


decode_transcript_bytes = decode_transcript
