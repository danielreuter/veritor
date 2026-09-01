from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

from veritor import (
    Compile,
    MatmulCompileRequest,
    Unsupported,
    VerificationCode,
    VerificationPolicy,
    Verify,
    adapt_protocol_artifact,
    create_trusted_verification_context,
    make_verification_expectation,
)
from veritor.staged import (
    InteractionError,
    InteractionPhase,
    ReplayService,
    ResolvedExecutableArtifact,
    StagedProverSession,
    StagedVerifierSession,
    ValueSource,
    derive_commitment_ownership,
    run_interactive_protocol,
)


def _artifact():
    return Compile(
        "matmul",
        MatmulCompileRequest(
            (
                (1, 2),
                (3, 4),
                (5, 6),
            ),
            (
                ((1, 2, 3),),
                ((4, 5, 6),),
            ),
        ),
    )


def _parties(
    policy: VerificationPolicy,
    *,
    q_seed: bytes = b"Q" * 32,
    s_seed: bytes = b"S" * 32,
    replay_service: ReplayService | None = None,
):
    artifact = _artifact()
    resolved = adapt_protocol_artifact(artifact)
    trust = create_trusted_verification_context(artifact)
    expectation = make_verification_expectation(
        artifact,
        policy,
        session_id=b"tests/matmul/interactive",
        q_seed=q_seed,
        s_seed=s_seed,
    )
    assert isinstance(resolved, ResolvedExecutableArtifact)
    assert not isinstance(trust, Unsupported)
    assert not isinstance(expectation, Unsupported)
    tape = artifact.circuit.evaluate_tape(artifact.public_inputs)
    assignment = dict(enumerate(tape))
    verifier = StagedVerifierSession(expectation, trust)
    prover = StagedProverSession(
        resolved,
        verifier.public_context,
        assignment,
        replay_service=replay_service,
    )
    return artifact, resolved, expectation, trust, assignment, prover, verifier


def test_honest_interaction_enforces_every_phase_and_verifies() -> None:
    (
        artifact,
        _resolved,
        expectation,
        trust,
        _assignment,
        prover,
        verifier,
    ) = _parties(VerificationPolicy(1, 1, 0))

    assert prover.phase is InteractionPhase.INITIAL
    assert verifier.phase is InteractionPhase.INITIAL
    assert not hasattr(verifier.public_context, "q_seed")
    assert not hasattr(verifier.public_context, "s_seed")
    assert "q_seed" not in StagedProverSession.__slots__
    assert "s_seed" not in StagedProverSession.__slots__

    boundary = prover.commit_boundary()
    assert prover.phase is InteractionPhase.BOUNDARY_COMMITTED
    replay_challenge = verifier.receive_boundary(boundary)
    assert verifier.phase is InteractionPhase.REPLAY_CHALLENGED
    assert replay_challenge.seed == expectation.q_seed

    commitments = prover.answer_replay_challenge(replay_challenge)
    assert prover.phase is InteractionPhase.UNITS_COMMITTED
    assert prover.replayed_unit_indices == (0, 1)
    sample_challenge = verifier.receive_unit_commitments(commitments)
    assert verifier.phase is InteractionPhase.SAMPLE_CHALLENGED
    assert sample_challenge.seed == expectation.s_seed

    evidence = prover.answer_sample_challenge(sample_challenge)
    assert prover.phase is InteractionPhase.COMPLETE
    result = verifier.receive_sample_evidence(evidence)

    assert result.accepted
    assert result.report.code is VerificationCode.ACCEPTED
    assert verifier.phase is InteractionPhase.COMPLETE
    assert result.transcript == prover.transcript()
    assert result.transcript.replay_challenge.selected_replay_units == (0, 1)
    assert result.transcript.sample_challenge.selected_verification_units == (
        0,
        1,
        2,
        3,
    )
    assert result.transcript.statement.claimed_outputs
    assert artifact.expected_outputs

    rerun = run_interactive_protocol(
        adapt_protocol_artifact(artifact),
        expectation,
        trust,
        dict(enumerate(artifact.circuit.evaluate_tape(artifact.public_inputs))),
    )
    assert rerun.accepted
    assert rerun.replayed_unit_indices == (0, 1)
    assert rerun.replayed_gate_count == 20
    assert rerun.replayed_cost == 32
    tampered = rerun.transcript_bytes[:-1] + b"!"
    assert not Verify(tampered, expectation, trust).accepted


def test_zero_q_replays_and_opens_no_units() -> None:
    (
        _artifact_value,
        resolved,
        expectation,
        trust,
        assignment,
        _prover,
        _verifier,
    ) = _parties(VerificationPolicy(0, 1, 0))

    run = run_interactive_protocol(
        resolved,
        expectation,
        trust,
        assignment,
    )

    assert run.accepted
    assert run.replayed_unit_indices == ()
    assert run.replayed_gate_count == 0
    assert run.replayed_cost == 0
    assert run.transcript.replay_challenge.selected_replay_units == ()
    assert run.transcript.unit_commitments.commitments == ()
    assert run.transcript.sample_challenge.selected_verification_units == ()
    assert run.transcript.sample_evidence.units == ()


def test_length_one_inner_product_has_no_hidden_accumulator() -> None:
    artifact = Compile(
        "matmul",
        MatmulCompileRequest(((3,),), (((5,),),)),
    )
    resolved = adapt_protocol_artifact(artifact)
    trust = create_trusted_verification_context(artifact)
    expectation = make_verification_expectation(
        artifact,
        VerificationPolicy(1, 1, 0),
        session_id=b"tests/matmul/length-one",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )
    assert isinstance(resolved, ResolvedExecutableArtifact)
    assert not isinstance(trust, Unsupported)
    assert not isinstance(expectation, Unsupported)
    tape = artifact.circuit.evaluate_tape(artifact.public_inputs)

    run = run_interactive_protocol(
        resolved,
        expectation,
        trust,
        dict(enumerate(tape)),
    )

    assert run.accepted
    assert run.replayed_gate_count == 1
    assert run.transcript.unit_commitments.commitments[0].commitment.value_count == 0
    assert run.transcript.sample_challenge.selected_verification_units == (0,)


def test_zero_s_commits_selected_units_but_opens_no_evidence() -> None:
    (
        _artifact_value,
        resolved,
        expectation,
        trust,
        assignment,
        _prover,
        _verifier,
    ) = _parties(VerificationPolicy(1, 0, 0))

    run = run_interactive_protocol(
        resolved,
        expectation,
        trust,
        assignment,
    )

    assert run.accepted
    assert run.replayed_unit_indices == (0, 1)
    assert len(run.transcript.unit_commitments.commitments) == 2
    assert run.transcript.sample_challenge.selected_verification_units == ()
    assert run.transcript.sample_evidence.units == ()


def test_partial_q_replays_exactly_the_derived_subset() -> None:
    selected_run = None
    for marker in range(1, 33):
        (
            _artifact_value,
            resolved,
            expectation,
            trust,
            assignment,
            _prover,
            _verifier,
        ) = _parties(
            VerificationPolicy(Fraction(1, 2), 1, 0),
            q_seed=marker.to_bytes(32, "big"),
        )
        run = run_interactive_protocol(
            resolved,
            expectation,
            trust,
            assignment,
        )
        if len(run.replayed_unit_indices) == 1:
            selected_run = run
            break

    assert selected_run is not None
    assert selected_run.replayed_unit_indices == (
        selected_run.transcript.replay_challenge.selected_replay_units
    )
    assert selected_run.replayed_gate_count == 10
    assert selected_run.replayed_cost == 16
    selected_replay = set(selected_run.replayed_unit_indices)
    assert all(
        int(resolved.verification_partition.unit_at(index).replay_unit)
        in selected_replay
        for index in (
            selected_run.transcript.sample_challenge.selected_verification_units
        )
    )


def test_out_of_order_calls_do_not_reveal_challenges() -> None:
    (
        _artifact_value,
        _resolved,
        _expectation,
        _trust,
        _assignment,
        prover,
        verifier,
    ) = _parties(VerificationPolicy(1, 1, 0))

    with pytest.raises(InteractionError, match="fixed boundary"):
        prover.answer_replay_challenge(None)  # type: ignore[arg-type]
    with pytest.raises(InteractionError, match="issued replay"):
        verifier.receive_unit_commitments(None)  # type: ignore[arg-type]
    with pytest.raises(InteractionError, match="issued sample"):
        verifier.receive_sample_evidence(None)  # type: ignore[arg-type]

    boundary = prover.commit_boundary()
    with pytest.raises(InteractionError, match="only be committed once"):
        prover.commit_boundary()
    replay_challenge = verifier.receive_boundary(boundary)
    with pytest.raises(InteractionError, match="only be received once"):
        verifier.receive_boundary(boundary)

    commitments = prover.answer_replay_challenge(replay_challenge)
    sample_challenge = verifier.receive_unit_commitments(commitments)
    with pytest.raises(InteractionError, match="issued replay challenge"):
        verifier.receive_unit_commitments(commitments)
    prover.answer_sample_challenge(sample_challenge)


def test_invalid_boundary_is_rejected_before_q_seed_release() -> None:
    (
        _artifact_value,
        _resolved,
        _expectation,
        _trust,
        _assignment,
        prover,
        verifier,
    ) = _parties(VerificationPolicy(1, 1, 0))
    boundary = prover.commit_boundary()
    first = boundary.public_io_openings[0]
    malformed = replace(
        boundary,
        public_io_openings=(
            replace(first, value=bytes([first.value[0] ^ 1]) + first.value[1:]),
            *boundary.public_io_openings[1:],
        ),
    )

    with pytest.raises(InteractionError, match="openings are invalid"):
        verifier.receive_boundary(malformed)
    assert verifier.phase is InteractionPhase.INITIAL


def test_invalid_selected_roots_are_rejected_before_s_seed_release() -> None:
    (
        _artifact_value,
        _resolved,
        _expectation,
        _trust,
        _assignment,
        prover,
        verifier,
    ) = _parties(VerificationPolicy(1, 1, 0))
    replay_challenge = verifier.receive_boundary(prover.commit_boundary())
    commitments = prover.answer_replay_challenge(replay_challenge)
    first = commitments.commitments[0]
    malformed = replace(
        commitments,
        commitments=(
            replace(
                first,
                commitment=replace(first.commitment, root=b"short"),
            ),
            *commitments.commitments[1:],
        ),
    )

    with pytest.raises(InteractionError, match="commitment 0 is invalid"):
        verifier.receive_unit_commitments(malformed)
    assert verifier.phase is InteractionPhase.REPLAY_CHALLENGED


def test_seed_substitution_is_rejected_by_prover() -> None:
    (
        _artifact_value,
        _resolved,
        _expectation,
        _trust,
        _assignment,
        prover,
        verifier,
    ) = _parties(VerificationPolicy(1, 1, 0))
    replay_challenge = verifier.receive_boundary(prover.commit_boundary())
    forged_q = replace(replay_challenge, seed=b"Z" * 32)
    with pytest.raises(InteractionError, match="not correctly derived"):
        prover.answer_replay_challenge(forged_q)
    assert prover.phase is InteractionPhase.BOUNDARY_COMMITTED

    commitments = prover.answer_replay_challenge(replay_challenge)
    sample_challenge = verifier.receive_unit_commitments(commitments)
    forged_s = replace(sample_challenge, seed=b"Y" * 32)
    with pytest.raises(InteractionError, match="not correctly derived"):
        prover.answer_sample_challenge(forged_s)
    assert prover.phase is InteractionPhase.UNITS_COMMITTED


@dataclass(frozen=True, slots=True)
class _FixedReplayService:
    assignment: Mapping[int, object]

    def values_for_unit(
        self,
        artifact: ResolvedExecutableArtifact,
        replay_unit_index: int,
        interior_positions: tuple[int, ...],
        boundary_values: Mapping[int, object],
        source: ValueSource,
    ) -> Mapping[int, object]:
        del artifact, replay_unit_index, boundary_values, source
        return {position: self.assignment[position] for position in interior_positions}


def test_forged_selected_inner_product_is_rejected() -> None:
    artifact = _artifact()
    resolved = adapt_protocol_artifact(artifact)
    trust = create_trusted_verification_context(artifact)
    expectation = make_verification_expectation(
        artifact,
        VerificationPolicy(1, 1, 0),
        session_id=b"tests/matmul/forged-interior",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )
    assert isinstance(resolved, ResolvedExecutableArtifact)
    assert not isinstance(trust, Unsupported)
    assert not isinstance(expectation, Unsupported)
    tape = artifact.circuit.evaluate_tape(artifact.public_inputs)
    forged = dict(enumerate(tape))
    layout = derive_commitment_ownership(
        artifact.circuit,
        artifact.replay_partition,
    )
    attacked_position = int(layout.interiors[0].items[0])
    forged[attacked_position] = (forged[attacked_position] + 1) % 256

    run = run_interactive_protocol(
        resolved,
        expectation,
        trust,
        forged,
        replay_service=_FixedReplayService(forged),
    )

    assert not run.accepted
    assert run.report.code is VerificationCode.RELATION_REJECTED


def test_executing_replay_detects_boundary_output_inconsistency() -> None:
    artifact = _artifact()
    resolved = adapt_protocol_artifact(artifact)
    trust = create_trusted_verification_context(artifact)
    claimed = list(artifact.expected_outputs)
    claimed[0] = (claimed[0] + 1) % 256
    expectation = make_verification_expectation(
        artifact,
        VerificationPolicy(1, 1, 0),
        claimed_outputs=claimed,
        session_id=b"tests/matmul/forged-output",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )
    assert isinstance(resolved, ResolvedExecutableArtifact)
    assert not isinstance(trust, Unsupported)
    assert not isinstance(expectation, Unsupported)
    tape = list(artifact.circuit.evaluate_tape(artifact.public_inputs))
    output_position = int(artifact.circuit.output_ports[0].position)
    tape[output_position] = claimed[0]
    assignment = dict(enumerate(tape))
    verifier = StagedVerifierSession(expectation, trust)
    prover = StagedProverSession(
        resolved,
        verifier.public_context,
        assignment,
    )
    challenge = verifier.receive_boundary(prover.commit_boundary())

    with pytest.raises(InteractionError, match="disagrees with committed boundary"):
        prover.answer_replay_challenge(challenge)
