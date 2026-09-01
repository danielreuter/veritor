from __future__ import annotations

import pytest

from veritor import (
    ArchitectureId,
    Compile,
    ProtocolCircuitArtifact,
    Unsupported,
    VerificationCode,
    VerificationPolicy,
    Verify,
    build_executable_conformance_transcript,
    make_verification_expectation,
    run_protocol,
)
from veritor.protocol import assignment_replay, encode_transcript

SEEDS = {"session_id": b"research-api/conformance", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}


@pytest.mark.parametrize("architecture_id", (ArchitectureId.DEMO_G, ArchitectureId.MATMUL))
def test_honest_conformance_transcript_verifies_purely(architecture_id: ArchitectureId) -> None:
    artifact = Compile(architecture_id)
    assert isinstance(artifact, ProtocolCircuitArtifact)

    run = build_executable_conformance_transcript(artifact, VerificationPolicy(1, 1, 0), **SEEDS)

    assert not isinstance(run, Unsupported)
    report = Verify(run.transcript_bytes, run.expectation, artifact.compiled)
    assert report.accepted
    assert report.code is VerificationCode.ACCEPTED
    assert report.sampled_verification_units == tuple(
        range(artifact.compiled.verification.unit_count)
    )
    assert run.expectation.claimed_outputs == artifact.expected_outputs
    assert run.expectation.compiled_digest == artifact.compiled.identity.digest


def test_conformance_transcript_is_deterministic_given_seeds() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)

    first = build_executable_conformance_transcript(artifact, **SEEDS)
    second = build_executable_conformance_transcript(artifact, **SEEDS)

    assert not isinstance(first, Unsupported)
    assert not isinstance(second, Unsupported)
    assert first == second


def test_forged_sampled_demo_execution_is_rejected() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    assert isinstance(artifact, ProtocolCircuitArtifact)
    compiled = artifact.compiled
    expectation = make_verification_expectation(artifact, VerificationPolicy(1, 1, 0), **SEEDS)
    assert not isinstance(expectation, Unsupported)

    tape = list(compiled.circuit.evaluate_tape(artifact.public_inputs))
    attacked = int(compiled.interior(0).unrank(0))
    tape[attacked] = (tape[attacked] + 1) % (1 << artifact.circuit.cell_bits)
    forged = dict(enumerate(tape))

    run = run_protocol(compiled, expectation, forged, replay=assignment_replay(forged))

    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None


def test_verify_rejects_transcript_against_the_wrong_expectation() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    assert isinstance(artifact, ProtocolCircuitArtifact)
    honest = build_executable_conformance_transcript(artifact, **SEEDS)
    assert not isinstance(honest, Unsupported)
    other = make_verification_expectation(
        artifact, session_id=b"research-api/other", q_seed=b"Q" * 32, s_seed=b"S" * 32
    )
    assert not isinstance(other, Unsupported)

    report = Verify(honest.transcript_bytes, other, artifact.compiled)

    assert report.code is VerificationCode.EXPECTATION_MISMATCH


def test_interactive_run_and_pure_verification_agree() -> None:
    artifact = Compile(ArchitectureId.MATMUL)
    assert isinstance(artifact, ProtocolCircuitArtifact)
    compiled = artifact.compiled
    expectation = make_verification_expectation(artifact, **SEEDS)
    assert not isinstance(expectation, Unsupported)
    values = dict(enumerate(compiled.circuit.evaluate_tape(artifact.public_inputs)))

    run = run_protocol(compiled, expectation, values)

    assert run.transcript is not None
    assert Verify(encode_transcript(run.transcript), expectation, compiled) == run.report
