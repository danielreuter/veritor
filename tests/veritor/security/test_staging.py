"""Component 2: staged commitments and challenge derivation (phases, challenge, session).

``J`` is a function of (q seed, header, boundary commitment) and the seed is
released only after the boundary is received; ``T`` of (s seed, everything
so far).  Both state machines refuse messages out of order.  Seed reuse
across sessions is shown -- as a NEGATIVE result -- to let the prover predict
both selections and pass with a false output.
"""

from __future__ import annotations

import pytest

from veritor.protocol import (
    BoundaryMessage,
    Commitment,
    EvidenceMessage,
    InteriorMessage,
    ProtocolError,
    ProverSession,
    Reject,
    ReplayChallenge,
    VerificationCode,
    VerifierSession,
    assignment_replay,
    make_expectation,
)
from veritor.protocol.phases import boundary_phase


def test_verifier_rejects_messages_out_of_order(model):
    expectation = model.expectation()
    verifier = VerifierSession(expectation, model.compiled)
    prover = ProverSession(model.compiled, verifier.header, model.values, weight_tree=model.tree)
    boundary = prover.boundary()
    with pytest.raises(Reject) as early:
        verifier.receive_interiors(InteriorMessage(()))
    assert early.value.code == VerificationCode.INVALID_PHASE
    with pytest.raises(Reject) as early:
        verifier.receive_evidence(EvidenceMessage(()))
    assert early.value.code == VerificationCode.INVALID_PHASE
    challenge = verifier.receive_boundary(boundary)
    with pytest.raises(Reject) as again:
        verifier.receive_boundary(boundary)
    assert again.value.code == VerificationCode.INVALID_PHASE
    interiors = prover.interiors(challenge)
    sample = verifier.receive_interiors(interiors)
    with pytest.raises(Reject) as again:
        verifier.receive_interiors(interiors)
    assert again.value.code == VerificationCode.INVALID_PHASE
    with pytest.raises(ProtocolError, match="not finished"):
        _ = verifier.transcript
    assert verifier.receive_evidence(prover.evidence(sample)).accepted
    # a stray out-of-order message does not advance (or poison) the session ...
    verifier = VerifierSession(expectation, model.compiled)
    with pytest.raises(Reject):
        verifier.receive_evidence(EvidenceMessage(()))
    verifier.receive_boundary(boundary)
    # ... but a verdict does: after a rejection every message is INVALID_PHASE
    verifier = VerifierSession(expectation, model.compiled)
    bad = BoundaryMessage(
        Commitment(boundary.commitment.root, boundary.commitment.count + 1), boundary.io_openings
    )
    with pytest.raises(Reject) as verdict:
        verifier.receive_boundary(bad)
    assert verdict.value.code == VerificationCode.INVALID_COMMITMENT
    with pytest.raises(Reject) as after:
        verifier.receive_boundary(boundary)
    assert after.value.code == VerificationCode.INVALID_PHASE


def test_prover_rejects_calls_out_of_order(model):
    expectation = model.expectation()
    header = model.header(expectation)
    prover = ProverSession(model.compiled, header, model.values, weight_tree=model.tree)
    challenge = ReplayChallenge(expectation.q_seed, (0, 1, 2))
    with pytest.raises(ProtocolError, match="phase"):
        prover.interiors(challenge)
    with pytest.raises(ProtocolError, match="not finished"):
        _ = prover.transcript
    prover.boundary()
    with pytest.raises(ProtocolError, match="phase"):
        prover.boundary()
    prover.interiors(challenge)
    with pytest.raises(ProtocolError, match="phase"):
        prover.interiors(challenge)


def test_replay_selection_is_derived_from_seed_header_and_boundary_only(honest_run, model, sec):
    """``J`` is recomputed from the boundary phase; the header carries no seed."""

    run, expectation = honest_run
    transcript = run.transcript
    derived = sec.replay_selection(expectation, transcript.header, transcript.boundary, model.compiled)
    assert derived == transcript.replay_challenge.selected
    document = sec.transcript_document(transcript)
    text = sec.encode_transcript(transcript).decode()
    assert expectation.q_seed.hex() not in str(document["header"])
    assert expectation.q_seed.hex() not in str(document["boundary"])
    assert text.count(expectation.q_seed.hex()) == 1 and text.count(expectation.s_seed.hex()) == 1
    assert document["replay_challenge"]["seed"] == expectation.q_seed.hex()
    assert document["sample_challenge"]["seed"] == expectation.s_seed.hex()
    # a different boundary root gives a different phase digest, hence (in general) another J
    other = BoundaryMessage(
        Commitment(bytes(32), transcript.boundary.commitment.count), transcript.boundary.io_openings
    )
    assert boundary_phase(transcript.header, other) != boundary_phase(
        transcript.header, transcript.boundary
    )
    selections = set()
    for index in range(16):
        expectation = model.expectation(sec.HALVES, q_seed=sec.seed("q", index))
        selections.add(
            sec.replay_selection(expectation, transcript.header, transcript.boundary, model.compiled)
        )
    assert len(selections) > 1


def test_prover_changing_its_boundary_after_seeing_j_is_invalid_opening(model, sec):
    """After ``J`` is known the boundary root is fixed: a re-commitment opens nothing."""

    expectation = model.expectation()
    address = model.hidden_boundary_addresses[1]
    forged = dict(model.values)
    forged[address] = (model.values[address] + 1) % (1 << model.width)
    run = model.run(expectation, model.values, prover=sec.TamperingProver, recommit_boundary=forged)
    assert run.report.code == VerificationCode.INVALID_OPENING
    assert run.report.sampled_replay_units == (0, 1, 2)  # J was issued; the evidence failed


@pytest.mark.parametrize("message", ["replay_challenge", "sample_challenge"])
def test_transcript_with_an_altered_selection_is_challenge_mismatch(honest_run, model, sec, message):
    run, expectation = honest_run

    def drop_one(document: dict) -> None:
        selected = document[message]["selected"]
        assert selected
        document[message]["selected"] = selected[1:]

    report = model.verify(sec.mutate_transcript(run.transcript, drop_one), expectation)
    assert report.code == VerificationCode.CHALLENGE_MISMATCH

    def other_seed(document: dict) -> None:
        document[message]["seed"] = sec.seed("other").hex()

    report = model.verify(sec.mutate_transcript(run.transcript, other_seed), expectation)
    assert report.code == VerificationCode.EXPECTATION_MISMATCH


def test_make_expectation_draws_fresh_seeds_by_default(model, sec):
    first = make_expectation(
        model.compiled, sec.CHECK_EVERYTHING, model.inputs, model.outputs, weights=model.kappa
    )
    second = make_expectation(
        model.compiled, sec.CHECK_EVERYTHING, model.inputs, model.outputs, weights=model.kappa
    )
    assert len(first.q_seed) == len(first.s_seed) == 32
    assert first.q_seed != second.q_seed and first.s_seed != second.s_seed
    assert first.q_seed != first.s_seed and first.session_id != second.session_id


def test_reused_seeds_let_the_prover_predict_and_evade_both_selections(model, sec):
    """NEGATIVE RESULT: with last session's seeds reused, a false output is accepted for sure.

    Session 1 reveals the seeds in its challenges.  If session 2 reuses them,
    the prover computes ``J`` from its own boundary message and ``T`` from its
    own interiors before sending them, and grinds a free interior value until
    the unit holding its false output is not sampled.
    """

    policy = sec.HALVES
    first = model.expectation(policy, q_seed=sec.seed("reused-q"), s_seed=sec.seed("reused-s"))
    revealed = model.run(first, model.values).transcript
    assert revealed is not None
    q_seed, s_seed = revealed.replay_challenge.seed, revealed.sample_challenge.seed
    assert (q_seed, s_seed) == (first.q_seed, first.s_seed)

    # the prover's false claim: output 0 (cell (1, 0)'s add) is off by one
    stage, cell = model.stages - 1, 0
    mul, add = model.cell_addresses(stage, cell)
    false_output = (model.outputs[0] + 1) % (1 << model.width)
    unit, replay = model.cell_unit(stage, cell), model.replay_unit_of(stage)
    claimed = (false_output, *model.outputs[1:])
    second = model.expectation(
        policy, claimed_outputs=claimed, session_id=b"session-2", q_seed=q_seed, s_seed=s_seed
    )
    header = model.header(second)  # the verifier sends the header before any commitment

    chosen = None
    for free in range(1 << model.width):  # grind the interior value the false unit may hold
        values, outputs = model.corrupt({mul: free, add: false_output})
        assert outputs == claimed
        prover = ProverSession(
            model.compiled, header, values, replay=assignment_replay(values), weight_tree=model.tree
        )
        boundary = prover.boundary()
        j = sec.replay_selection(second, header, boundary, model.compiled)
        if replay not in j:
            chosen, predicted = values, (j, ())
            break
        challenge = ReplayChallenge(q_seed, j)
        interiors = prover.interiors(challenge)
        t = sec.sample_selection(second, header, boundary, challenge, interiors, model.compiled)
        if unit not in t:
            chosen, predicted = values, (j, t)
            break
    assert chosen is not None, "grinding never fails for long with known seeds"

    run = model.run(second, chosen)
    assert run.report.code == VerificationCode.ACCEPTED
    assert run.report.sampled_replay_units == predicted[0]
    if predicted[1]:
        assert run.report.sampled_verification_units == predicted[1]
    assert unit not in run.report.sampled_verification_units
    assert model.verify(run.transcript, second).code == VerificationCode.ACCEPTED
    assert claimed != model.outputs  # a false output, accepted with certainty
