from __future__ import annotations

from veritor import (
    ArchitectureId,
    Compile,
    Unsupported,
    VerificationCode,
    VerificationPolicy,
    Verify,
    adapt_protocol_artifact,
    build_demo_conformance_transcript,
    create_trusted_verification_context,
    make_verification_expectation,
)
from veritor.staged import build_transcript_bytes


def test_demo_one_shot_conformance_transcript_verifies_purely() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    run = build_demo_conformance_transcript(
        artifact,
        VerificationPolicy(1, 1, 0),
        session_id=b"research-api/conformance",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )

    assert not isinstance(run, Unsupported)
    first = Verify(run.transcript_bytes, run.expectation, run.trust)
    second = Verify(run.data, run.expectation, run.trust)
    assert first == second
    assert first.accepted
    assert first.code is VerificationCode.ACCEPTED


def test_forged_sampled_demo_execution_is_rejected() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    resolved = adapt_protocol_artifact(artifact)
    trust = create_trusted_verification_context(artifact)
    assert not isinstance(resolved, Unsupported)
    assert not isinstance(trust, Unsupported)

    tape = list(artifact.circuit.evaluate_tape(artifact.public_inputs))
    attacked_position = int(artifact.circuit.computed_positions.unrank(0))
    tape[attacked_position] = (tape[attacked_position] + 1) % (
        1 << artifact.circuit.cell_bits
    )
    assignment = dict(enumerate(tape))
    expectation = make_verification_expectation(
        artifact,
        VerificationPolicy(1, 1, 0),
        public_inputs=artifact.public_inputs,
        claimed_outputs=artifact.expected_outputs,
        session_id=b"research-api/forgery",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )
    assert not isinstance(expectation, Unsupported)

    data = build_transcript_bytes(resolved, expectation, assignment)
    report = Verify(data, expectation, trust)

    assert not report.accepted
    assert report.code is VerificationCode.RELATION_REJECTED
