from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from veritor.commitment import MerkleSha256Backend, ValueCommitmentRegistry
from veritor.core import VerificationLimits, VerificationPolicy
from veritor.staged import (
    OwnedValueCommitment,
    SampledUnitEvidence,
    SampleEvidenceRegistry,
    StagedProtocolBuilder,
    StagedProtocolError,
    TransparentLocalCheckBackend,
    TrustedArtifactRegistry,
    TrustedVerificationContext,
    VerificationCode,
    VerificationStatus,
    decode_transcript,
    decode_transparent_evidence,
    encode_transcript,
    verify_transcript_bytes,
)


def build_bytes(protocol_case, *, assignment=None, policy=None):
    values = protocol_case.assignment if assignment is None else assignment
    expectation = protocol_case.expectation(
        assignment=values,
        policy=policy,
    )
    data = protocol_case.builder().build_bytes(expectation, values)
    return data, expectation


def test_honest_run_round_trips_and_verifies_purely(protocol_case):
    data, expectation = build_bytes(protocol_case)
    trust = protocol_case.trust()

    first = verify_transcript_bytes(
        data,
        expectation,
        trust,
        VerificationLimits(),
    )
    second = verify_transcript_bytes(
        data,
        expectation,
        trust,
        VerificationLimits(),
    )
    transcript = decode_transcript(data)

    assert first == second
    assert first.accepted
    assert first.code is VerificationCode.ACCEPTED
    assert encode_transcript(transcript) == data
    assert not hasattr(transcript, "accepted")
    assert not hasattr(transcript, "__dict__")
    with pytest.raises(FrozenInstanceError):
        transcript.version = "changed"


def test_transcript_preserves_duplicate_outputs_without_duplicate_openings(
    protocol_case,
):
    data, expectation = build_bytes(protocol_case)
    transcript = decode_transcript(data)

    assert transcript.statement.claimed_outputs == (
        b"\x07",
        b"\x07",
        b"\x02",
    )
    assert tuple(
        int(opening.position) for opening in transcript.boundary.public_io_openings
    ) == (10, 20, 50)
    assert verify_transcript_bytes(
        data,
        expectation,
        protocol_case.trust(),
        VerificationLimits(),
    ).accepted

    inconsistent = replace(
        transcript.statement,
        claimed_outputs=(b"\x07", b"\x08", b"\x02"),
    )
    tampered = encode_transcript(replace(transcript, statement=inconsistent))
    report = verify_transcript_bytes(
        tampered,
        replace(expectation, claimed_outputs=(7, 8, 2)),
        protocol_case.trust(),
        VerificationLimits(),
    )
    assert report.code in {
        VerificationCode.INVALID_OPENING,
        VerificationCode.PUBLIC_IO_MISMATCH,
        VerificationCode.INVALID_PHASE,
    }


def test_forged_sampled_gate_rejects_but_unsampled_forgery_survives(
    protocol_case,
):
    forged = {10: 2, 20: 3, 30: 6, 40: 8, 50: 8}
    sampled_data, sampled_expectation = build_bytes(
        protocol_case,
        assignment=forged,
        policy=VerificationPolicy(1, 1, 0),
    )
    unsampled_data, unsampled_expectation = build_bytes(
        protocol_case,
        assignment=forged,
        policy=VerificationPolicy(1, 0, 0),
    )

    sampled = verify_transcript_bytes(
        sampled_data,
        sampled_expectation,
        protocol_case.trust(),
        VerificationLimits(),
    )
    unsampled = verify_transcript_bytes(
        unsampled_data,
        unsampled_expectation,
        protocol_case.trust(),
        VerificationLimits(),
    )

    assert sampled.code is VerificationCode.RELATION_REJECTED
    assert unsampled.accepted


def test_local_evidence_authenticates_exact_cross_unit_values(protocol_case):
    data, _ = build_bytes(protocol_case)
    transcript = decode_transcript(data)
    evidence = {
        item.verification_unit_index: decode_transparent_evidence(
            item.payload,
            VerificationLimits(),
        )
        for item in transcript.sample_evidence.units
    }

    assert tuple(int(opening.position) for opening in evidence[0].openings) == (
        10,
        20,
        30,
    )
    assert tuple(int(opening.position) for opening in evidence[2].openings) == (40, 50)
    boundary_positions = {
        int(opening.position) for opening in transcript.boundary.public_io_openings
    }
    assert 40 not in boundary_positions  # not public I/O, but still boundary-owned


@pytest.mark.parametrize(
    "expectation_change",
    [
        {"session_id": b"another-session"},
        {"compiled_result_digest": "ab" * 32},
        {"policy": VerificationPolicy(1, 0, 0)},
        {"value_commitment_backend_id": "tests/other-commitment"},
        {"sample_evidence_backend_id": "tests/other-evidence"},
        {"q_seed": b"X" * 32},
        {"s_seed": b"Y" * 32},
    ],
)
def test_wrong_session_identity_policy_seed_or_backend_rejects(
    protocol_case,
    expectation_change,
):
    data, expectation = build_bytes(protocol_case)

    report = verify_transcript_bytes(
        data,
        replace(expectation, **expectation_change),
        protocol_case.trust(),
        VerificationLimits(),
    )

    assert not report.accepted
    assert report.code is VerificationCode.EXPECTATION_MISMATCH


def test_verifier_expectation_requires_both_verifier_owned_seeds(protocol_case):
    expectation = protocol_case.expectation()

    with pytest.raises(StagedProtocolError, match="expected q seed"):
        replace(expectation, q_seed=None)  # type: ignore[arg-type]
    with pytest.raises(StagedProtocolError, match="expected s seed"):
        replace(expectation, s_seed=None)  # type: ignore[arg-type]


def test_phase_substitution_invalidates_roots_or_evidence(protocol_case):
    data, expectation = build_bytes(protocol_case)
    transcript = decode_transcript(data)
    changed_q = replace(
        transcript.replay_challenge,
        phase_digest=b"X" * 32,
    )
    changed_s = replace(
        transcript.sample_challenge,
        seed=b"Y" * 32,
    )
    changed_evidence_phase = replace(
        transcript.sample_evidence,
        sample_phase_digest=b"Z" * 32,
    )

    reports = (
        verify_transcript_bytes(
            encode_transcript(replace(transcript, replay_challenge=changed_q)),
            expectation,
            protocol_case.trust(),
            VerificationLimits(),
        ),
        verify_transcript_bytes(
            encode_transcript(replace(transcript, sample_challenge=changed_s)),
            expectation,
            protocol_case.trust(),
            VerificationLimits(),
        ),
        verify_transcript_bytes(
            encode_transcript(
                replace(transcript, sample_evidence=changed_evidence_phase)
            ),
            expectation,
            protocol_case.trust(),
            VerificationLimits(),
        ),
    )

    assert all(not report.accepted for report in reports)
    assert {report.code for report in reports} <= {
        VerificationCode.EXPECTATION_MISMATCH,
        VerificationCode.INVALID_PHASE,
        VerificationCode.CHALLENGE_MISMATCH,
        VerificationCode.INVALID_COMMITMENT,
    }


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "reordered"])
def test_unit_commitments_require_exact_ordered_j_coverage(protocol_case, mode):
    data, expectation = build_bytes(protocol_case)
    transcript = decode_transcript(data)
    commitments = transcript.unit_commitments.commitments
    if mode == "missing":
        changed = commitments[:-1]
    elif mode == "extra":
        changed = (
            *commitments,
            OwnedValueCommitment(99, commitments[0].commitment),
        )
    elif mode == "duplicate":
        changed = (*commitments, commitments[-1])
    else:
        changed = tuple(reversed(commitments))
    message = replace(transcript.unit_commitments, commitments=changed)

    report = verify_transcript_bytes(
        encode_transcript(replace(transcript, unit_commitments=message)),
        expectation,
        protocol_case.trust(),
        VerificationLimits(),
    )

    assert report.code is VerificationCode.COVERAGE_MISMATCH


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_sample_evidence_requires_exact_ordered_t_coverage(protocol_case, mode):
    data, expectation = build_bytes(protocol_case)
    transcript = decode_transcript(data)
    units = transcript.sample_evidence.units
    if mode == "missing":
        changed = units[:-1]
    elif mode == "duplicate":
        changed = (*units, units[-1])
    else:
        changed = (
            *units,
            SampledUnitEvidence(
                99,
                units[0].backend_id,
                units[0].payload,
            ),
        )
    message = replace(transcript.sample_evidence, units=changed)

    report = verify_transcript_bytes(
        encode_transcript(replace(transcript, sample_evidence=message)),
        expectation,
        protocol_case.trust(),
        VerificationLimits(),
    )

    assert report.code is VerificationCode.COVERAGE_MISMATCH


def test_transparent_evidence_rejects_missing_extra_or_duplicate_openings(
    protocol_case,
):
    data, expectation = build_bytes(protocol_case)
    transcript = decode_transcript(data)
    first = transcript.sample_evidence.units[0]
    evidence = decode_transparent_evidence(first.payload, VerificationLimits())

    for changed_openings in (
        evidence.openings[:-1],
        (*evidence.openings, evidence.openings[-1]),
        (*evidence.openings, evidence.openings[0]),
    ):
        from veritor.staged import (
            TransparentLocalCheckEvidence,
            encode_transparent_evidence,
        )

        changed_payload = encode_transparent_evidence(
            TransparentLocalCheckEvidence(
                evidence.instance_digest,
                changed_openings,
            )
        )
        changed_unit = replace(first, payload=changed_payload)
        changed_message = replace(
            transcript.sample_evidence,
            units=(changed_unit, *transcript.sample_evidence.units[1:]),
        )
        report = verify_transcript_bytes(
            encode_transcript(replace(transcript, sample_evidence=changed_message)),
            expectation,
            protocol_case.trust(),
            VerificationLimits(),
        )
        assert report.code is VerificationCode.INVALID_EVIDENCE


def test_empty_executable_circuit_accepts_only_after_public_io_checks(empty_case):
    expectation = empty_case.expectation()
    data = empty_case.builder().build_bytes(expectation, empty_case.assignment)

    accepted = verify_transcript_bytes(
        data,
        expectation,
        empty_case.trust(),
        VerificationLimits(),
    )
    unsupported = verify_transcript_bytes(
        data,
        expectation,
        empty_case.trust(executable=False),
        VerificationLimits(),
    )

    assert accepted.accepted
    assert decode_transcript(data).sample_evidence.units == ()
    assert unsupported.status is VerificationStatus.UNSUPPORTED
    assert unsupported.code is VerificationCode.UNSUPPORTED_ARTIFACT
    assert unsupported.unsupported is not None
    assert unsupported.unsupported.reason_code == "NO_EXECUTABLE_RELATIONS"


def test_empty_circuit_does_not_skip_wrong_public_output(empty_case):
    expectation = replace(empty_case.expectation(), claimed_outputs=(8, 8))
    data = empty_case.builder().build_bytes(expectation, empty_case.assignment)

    report = verify_transcript_bytes(
        data,
        expectation,
        empty_case.trust(),
        VerificationLimits(),
    )

    assert report.code is VerificationCode.PUBLIC_IO_MISMATCH


def test_structural_artifact_is_typed_unsupported_before_empty_sample_shortcuts(
    empty_case,
    structural_empty_artifact,
):
    expectation = empty_case.expectation()
    transcript = decode_transcript(
        empty_case.builder().build_bytes(expectation, empty_case.assignment)
    )
    structural_digest = structural_empty_artifact.compiled_result_digest
    changed_session = replace(
        transcript.session,
        compiled_result_digest=structural_digest,
    )
    data = encode_transcript(replace(transcript, session=changed_session))
    trust = TrustedVerificationContext(
        TrustedArtifactRegistry((structural_empty_artifact,))
    )

    report = verify_transcript_bytes(
        data,
        replace(expectation, compiled_result_digest=structural_digest),
        trust,
        VerificationLimits(),
    )

    assert report.status is VerificationStatus.UNSUPPORTED
    assert report.unsupported is not None
    assert report.unsupported.artifact_kind.value == "structural_circuit"


def test_unknown_locally_untrusted_backend_rejects(protocol_case):
    data, expectation = build_bytes(protocol_case)
    trust = replace(
        protocol_case.trust(),
        value_commitment_backends=ValueCommitmentRegistry(()),
        sample_evidence_backends=SampleEvidenceRegistry(()),
    )

    report = verify_transcript_bytes(
        data,
        expectation,
        trust,
        VerificationLimits(),
    )

    assert report.code is VerificationCode.UNKNOWN_COMMITMENT_BACKEND


def test_wire_rejects_duplicate_unknown_float_noncanonical_and_trailing_data(
    protocol_case,
):
    data, expectation = build_bytes(protocol_case)
    document = json.loads(data)
    document["unknown"] = 1
    unknown = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    duplicate = data[:-1] + b',"version":"veritor.staged-transcript/v1"}'
    float_policy = data.replace(b'"q":[1,1]', b'"q":0.5')
    nonreduced_policy = data.replace(b'"q":[1,1]', b'"q":[2,2]')
    uppercase_hex = data.replace(b'"seed":"5151', b'"seed":"A151', 1)

    cases = (
        (unknown, VerificationCode.MALFORMED_TRANSCRIPT),
        (duplicate, VerificationCode.MALFORMED_TRANSCRIPT),
        (float_policy, VerificationCode.MALFORMED_TRANSCRIPT),
        (nonreduced_policy, VerificationCode.NONCANONICAL_TRANSCRIPT),
        (b" " + data, VerificationCode.NONCANONICAL_TRANSCRIPT),
        (uppercase_hex, VerificationCode.NONCANONICAL_TRANSCRIPT),
        (data + b"\n", VerificationCode.NONCANONICAL_TRANSCRIPT),
        (data + b"trailing", VerificationCode.MALFORMED_TRANSCRIPT),
    )
    for malformed, code in cases:
        report = verify_transcript_bytes(
            malformed,
            expectation,
            protocol_case.trust(),
            VerificationLimits(),
        )
        assert report.code is code


def test_verification_enforces_transcript_and_structure_resource_limits(
    protocol_case,
):
    data, expectation = build_bytes(protocol_case)

    transcript_limit = verify_transcript_bytes(
        data,
        expectation,
        protocol_case.trust(),
        VerificationLimits(max_transcript_bytes=len(data) - 1),
    )
    position_limit = verify_transcript_bytes(
        data,
        expectation,
        protocol_case.trust(),
        VerificationLimits(max_positions=4),
    )

    assert transcript_limit.code is VerificationCode.RESOURCE_LIMIT
    assert position_limit.code is VerificationCode.RESOURCE_LIMIT


def test_builder_accepts_explicit_backend_registries(protocol_case):
    expectation = protocol_case.expectation()
    builder = StagedProtocolBuilder(
        protocol_case.artifact,
        MerkleSha256Backend(),
        TransparentLocalCheckBackend(),
    )

    transcript = builder.build(
        expectation,
        protocol_case.assignment,
    )

    assert verify_transcript_bytes(
        encode_transcript(transcript),
        expectation,
        protocol_case.trust(),
        VerificationLimits(),
    ).accepted
