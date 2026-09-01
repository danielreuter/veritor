"""Canonical phase digests and verifier-derived commitment domains."""

from __future__ import annotations

from veritor.commitment import (
    CommitmentDomain,
    CommitmentOwner,
    commitment_manifest,
    opening_manifest,
)
from veritor.core import (
    StructuralCircuit,
    VerificationPolicy,
    identity_digest,
)

from .boundary import CommitmentLayout, value_schema_for_position
from .model import (
    BoundaryMessage,
    OwnedValueCommitment,
    PublicStatement,
    SessionParameters,
    StagedProtocolError,
    rational_pair,
)


def _raw_digest(tag: str, manifest: object) -> bytes:
    return bytes.fromhex(identity_digest(tag, manifest))  # type: ignore[arg-type]


def session_manifest(session: SessionParameters) -> dict[str, object]:
    return {
        "compiled_result_digest": session.compiled_result_digest,
        "policy_digest": session.policy_digest,
        "protocol_version": session.protocol_version,
        "sample_evidence_backend_id": session.sample_evidence_backend_id,
        "session_id": session.session_id.hex(),
        "value_commitment_backend_id": session.value_commitment_backend_id,
    }


def statement_manifest(statement: PublicStatement) -> dict[str, object]:
    return {
        "claimed_outputs": [item.hex() for item in statement.claimed_outputs],
        "public_inputs": [item.hex() for item in statement.public_inputs],
    }


def policy_manifest(policy: VerificationPolicy) -> dict[str, object]:
    return {
        "eta": rational_pair(policy.eta),
        "q": rational_pair(policy.q),
        "s": rational_pair(policy.s),
    }


def derive_statement_digest(statement: PublicStatement) -> bytes:
    if not isinstance(statement, PublicStatement):
        raise StagedProtocolError("statement has the wrong type")
    return _raw_digest(
        "veritor/staged/public-statement/v1",
        statement_manifest(statement),
    )


def derive_initial_phase_digest(
    session: SessionParameters,
    statement: PublicStatement,
    policy: VerificationPolicy,
) -> bytes:
    """Bind the session, statement, exact policy, and backend choices."""

    if session.policy_digest != policy.digest:
        raise StagedProtocolError("session policy digest does not match policy")
    return _raw_digest(
        "veritor/staged/initial-phase/v1",
        {
            "policy": policy_manifest(policy),
            "session": session_manifest(session),
            "statement": statement_manifest(statement),
        },
    )


def derive_boundary_phase_digest(
    initial_phase_digest: bytes,
    boundary: BoundaryMessage,
) -> bytes:
    if type(initial_phase_digest) is not bytes or len(initial_phase_digest) != 32:
        raise StagedProtocolError("initial_phase_digest must be 32 bytes")
    return _raw_digest(
        "veritor/staged/boundary-phase/v1",
        {
            "boundary_commitment": commitment_manifest(boundary.commitment),
            "initial_phase_digest": initial_phase_digest.hex(),
            "public_io_openings": [
                opening_manifest(opening) for opening in boundary.public_io_openings
            ],
        },
    )


def derive_q_phase_digest(
    boundary_phase_digest: bytes,
    seed: bytes,
    selected_replay_units: tuple[int, ...],
) -> bytes:
    if type(boundary_phase_digest) is not bytes or len(boundary_phase_digest) != 32:
        raise StagedProtocolError("boundary_phase_digest must be 32 bytes")
    if type(seed) is not bytes or len(seed) != 32:
        raise StagedProtocolError("q seed must be 32 bytes")
    return _raw_digest(
        "veritor/staged/q-phase/v1",
        {
            "boundary_phase_digest": boundary_phase_digest.hex(),
            "seed": seed.hex(),
            "selected_replay_units": list(selected_replay_units),
        },
    )


def derive_unit_commitments_phase_digest(
    q_phase_digest: bytes,
    commitments: tuple[OwnedValueCommitment, ...],
) -> bytes:
    """Bind ordered selected-unit roots before the s draw."""

    if type(q_phase_digest) is not bytes or len(q_phase_digest) != 32:
        raise StagedProtocolError("q_phase_digest must be 32 bytes")
    return _raw_digest(
        "veritor/staged/unit-commitments-phase/v1",
        {
            "commitments": [
                {
                    "commitment": commitment_manifest(item.commitment),
                    "replay_unit_index": item.replay_unit_index,
                }
                for item in commitments
            ],
            "q_phase_digest": q_phase_digest.hex(),
        },
    )


def derive_sample_phase_digest(
    unit_commitments_phase_digest: bytes,
    seed: bytes,
    selected_verification_units: tuple[int, ...],
) -> bytes:
    if (
        type(unit_commitments_phase_digest) is not bytes
        or len(unit_commitments_phase_digest) != 32
    ):
        raise StagedProtocolError("unit_commitments_phase_digest must be 32 bytes")
    if type(seed) is not bytes or len(seed) != 32:
        raise StagedProtocolError("s seed must be 32 bytes")
    return _raw_digest(
        "veritor/staged/sample-phase/v1",
        {
            "seed": seed.hex(),
            "selected_verification_units": list(selected_verification_units),
            "unit_commitments_phase_digest": (unit_commitments_phase_digest.hex()),
        },
    )


def derive_boundary_commitment_domain(
    session: SessionParameters,
    statement: PublicStatement,
    policy: VerificationPolicy,
    circuit: StructuralCircuit,
    layout: CommitmentLayout,
    initial_phase_digest: bytes,
) -> CommitmentDomain:
    positions = layout.boundary.items
    return CommitmentDomain(
        backend_id=session.value_commitment_backend_id,
        session_id=session.session_id,
        compiled_result_digest=session.compiled_result_digest,
        policy_digest=str(policy.digest),
        statement_digest=derive_statement_digest(statement),
        phase_digest=initial_phase_digest,
        owner=CommitmentOwner.boundary(),
        positions=positions,
        value_schema_ids=(
            value_schema_for_position(circuit, position) for position in positions
        ),
    )


def derive_unit_commitment_domain(
    session: SessionParameters,
    statement: PublicStatement,
    policy: VerificationPolicy,
    circuit: StructuralCircuit,
    layout: CommitmentLayout,
    q_phase_digest: bytes,
    replay_unit_index: int,
) -> CommitmentDomain:
    if (
        type(replay_unit_index) is not int
        or not 0 <= replay_unit_index < layout.replay_unit_count
    ):
        raise StagedProtocolError("replay unit index is out of range")
    positions = layout.interiors[replay_unit_index].items
    return CommitmentDomain(
        backend_id=session.value_commitment_backend_id,
        session_id=session.session_id,
        compiled_result_digest=session.compiled_result_digest,
        policy_digest=str(policy.digest),
        statement_digest=derive_statement_digest(statement),
        phase_digest=q_phase_digest,
        owner=CommitmentOwner.replay_unit(replay_unit_index),
        positions=positions,
        value_schema_ids=(
            value_schema_for_position(circuit, position) for position in positions
        ),
    )
