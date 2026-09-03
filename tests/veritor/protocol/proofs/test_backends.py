"""The reveal step through a backend: coverage, batching, cross-session proofs."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from veritor.core import Compiled, VerificationLimits, VerificationPolicy
from veritor.protocol import (
    EvidenceMessage,
    ProtocolError,
    ProverSession,
    Reject,
    VerificationCode,
    VerifierSession,
    decode_transcript,
    encode_transcript,
    run_protocol,
    verify_transcript,
)
from veritor.protocol.proofs import (
    TRANSPARENT_BACKEND,
    BatchPlan,
    ForeignBatch,
    ProofBackend,
    Statement,
    TransparentBackend,
    Witness,
    decode_statement,
    derive_obligations,
    encode_statement,
    encode_witness,
    statement_digest,
)

from .conftest import RECORDING_BACKEND, RecordingBackend

SAMPLE_SOME = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))


def exchange(compiled, expectation, values, weight_tree, backend, plan=None):
    """Run everything but the reveal; return the verifier and the prover's evidence."""

    verifier = VerifierSession(expectation, compiled, backend=backend)
    prover = ProverSession(
        compiled,
        verifier.header,
        values,
        weight_tree=weight_tree,
        backend=backend,
        plan=plan,
    )
    replay = verifier.receive_boundary(prover.boundary())
    sample = verifier.receive_interiors(prover.interiors(replay))
    return verifier, prover.evidence(sample)


def test_the_recording_backend_satisfies_the_protocol(
    recording: RecordingBackend,
) -> None:
    assert isinstance(recording, ProofBackend)
    assert isinstance(recording.transparent, ProofBackend)
    assert recording.transparent.backend_id == TRANSPARENT_BACKEND


def test_the_default_run_is_transparent_and_byte_identical_in_the_header(
    compiled: Compiled, expect, honest_values, model_weights
) -> None:
    expectation = expect()
    assert expectation.backend == TRANSPARENT_BACKEND
    run = run_protocol(
        compiled, expectation, honest_values, weight_tree=model_weights[1]
    )
    assert run.report.accepted
    assert run.transcript is not None
    assert run.transcript.evidence.proofs == ()
    assert len(run.transcript.evidence.units) == len(
        run.report.sampled_verification_units
    )
    assert b"backend" not in encode_transcript(run.transcript)
    assert b'"proofs"' not in encode_transcript(run.transcript)


@pytest.mark.parametrize("plan_name", ["single", "one_per_unit", "chunked"])
def test_a_zk_style_backend_accepts_an_honest_run_under_any_plan(
    compiled: Compiled,
    expect,
    honest_values,
    model_weights,
    recording: RecordingBackend,
    plan_name,
) -> None:
    expectation = expect(SAMPLE_SOME, backend=RECORDING_BACKEND)
    verifier = VerifierSession(expectation, compiled, backend=recording)
    count = None
    plan = {
        "single": lambda n: BatchPlan.single(n),
        "one_per_unit": lambda n: BatchPlan.one_per_unit(n),
        "chunked": lambda n: BatchPlan.chunked(n, 3),
    }[plan_name]
    # the prover learns the sample only at the reveal, so plan from the challenge
    prover = ProverSession(
        compiled,
        verifier.header,
        honest_values,
        weight_tree=model_weights[1],
        backend=recording,
    )
    replay = verifier.receive_boundary(prover.boundary())
    sample = verifier.receive_interiors(prover.interiors(replay))
    count = len(sample.selected)
    prover._plan = plan(count)
    evidence = prover.evidence(sample)
    report = verifier.receive_evidence(evidence)

    assert report.accepted
    assert evidence.units == ()
    assert len(evidence.proofs) == len(plan(count).groups)
    assert sorted(index for proof in evidence.proofs for index in proof.units) == list(
        range(count)
    )
    assert len(recording.proved) == len(recording.verified) == len(evidence.proofs)
    for (proved, _), verified in zip(recording.proved, recording.verified, strict=True):
        assert proved == verified
        assert statement_digest(proved) == statement_digest(verified)
    # the header binds the backend and the transcript round-trips with proofs
    transcript = verifier.transcript
    assert transcript.header.backend == RECORDING_BACKEND
    data = encode_transcript(transcript)
    assert decode_transcript(data) == transcript
    assert verify_transcript(data, expectation, compiled, backend=recording) == report
    with pytest.raises(ProtocolError, match="no proof backend configured"):
        verify_transcript(data, expectation, compiled)


def test_the_backend_id_must_match_the_header(
    compiled: Compiled, expect, recording: RecordingBackend
) -> None:
    with pytest.raises(ProtocolError, match="binds backend"):
        VerifierSession(expect(), compiled, backend=recording)
    with pytest.raises(ProtocolError, match="binds backend"):
        VerifierSession(
            expect(backend=RECORDING_BACKEND),
            compiled,
            backend=TransparentBackend(compiled.circuit.gate_set, compiled),
        )


def test_the_verifier_derives_obligations_itself(
    compiled: Compiled,
    expect,
    honest_values,
    model_weights,
    recording: RecordingBackend,
) -> None:
    """The prover's statements equal what the verifier derives from the challenge and the Index."""

    verifier, evidence = exchange(
        compiled,
        expect(SAMPLE_SOME, backend=RECORDING_BACKEND),
        honest_values,
        model_weights[1],
        recording,
    )
    demanded, _kinds = derive_obligations(
        verifier._layout,
        verifier.header,
        verifier._commitments,
        verifier.selected_verification_units,
    )
    ((proved, witness),) = recording.proved
    assert proved.obligations == tuple(sorted(demanded, key=lambda item: item.key))
    for obligation in demanded:
        assert obligation.session == verifier.header.digest
        assert obligation.compiled == bytes.fromhex(compiled.digest)
        assert obligation.unit in verifier.selected_verification_units
        assert obligation.kind == bytes.fromhex(
            compiled.index.verification_unit(obligation.unit).kind
        )
    assert verifier.receive_evidence(evidence).accepted
    witness.for_statement(proved)


class TestTamperedEvidence:
    """Every way the prover's proofs can fail to cover the demands is a distinct rejection."""

    @pytest.fixture
    def honest_pair(self, compiled, expect, honest_values, model_weights, recording):
        verifier = VerifierSession(
            expect(SAMPLE_SOME, backend=RECORDING_BACKEND), compiled, backend=recording
        )
        prover = ProverSession(
            compiled,
            verifier.header,
            honest_values,
            weight_tree=model_weights[1],
            backend=recording,
        )
        replay = verifier.receive_boundary(prover.boundary())
        sample = verifier.receive_interiors(prover.interiors(replay))
        prover._plan = BatchPlan.one_per_unit(len(sample.selected))
        return verifier, prover.evidence(sample)

    def reject(self, verifier: VerifierSession, evidence: EvidenceMessage) -> Reject:
        with pytest.raises(Reject) as info:
            verifier.receive_evidence(evidence)
        return info.value

    def test_a_missing_proof_is_a_coverage_mismatch(self, honest_pair) -> None:
        verifier, evidence = honest_pair
        rejection = self.reject(verifier, EvidenceMessage((), evidence.proofs[1:]))
        assert rejection.code is VerificationCode.COVERAGE_MISMATCH

    def test_a_duplicated_proof_is_a_coverage_mismatch(self, honest_pair) -> None:
        verifier, evidence = honest_pair
        rejection = self.reject(
            verifier, EvidenceMessage((), evidence.proofs + evidence.proofs[:1])
        )
        assert rejection.code is VerificationCode.COVERAGE_MISMATCH

    def test_a_proof_for_another_unit_is_a_coverage_mismatch(self, honest_pair) -> None:
        verifier, evidence = honest_pair
        first, second, *rest = evidence.proofs
        swapped = (
            replace(first, units=second.units),
            replace(second, units=first.units),
            *rest,
        )
        rejection = self.reject(verifier, EvidenceMessage((), swapped))
        assert rejection.code in (
            VerificationCode.PROOF_REJECTED,
            VerificationCode.INVALID_OPENING,
        )

    def test_openings_are_refused_by_a_zk_backend(self, honest_pair) -> None:
        verifier, evidence = honest_pair
        transparent_shaped = EvidenceMessage(((),) * len(evidence.proofs))
        rejection = self.reject(verifier, transparent_shaped)
        assert rejection.code is VerificationCode.COVERAGE_MISMATCH
        assert "takes proofs, not openings" in rejection.detail

    def test_a_proof_the_backend_rejects_is_proof_rejected(
        self, honest_pair, recording: RecordingBackend
    ) -> None:
        verifier, evidence = honest_pair
        recording.reject = True
        rejection = self.reject(verifier, evidence)
        assert rejection.code is VerificationCode.PROOF_REJECTED

    def test_a_foreign_obligation_claiming_this_session_is_refused(
        self, honest_pair, recording: RecordingBackend
    ) -> None:
        verifier, evidence = honest_pair
        # take the statement of proof 1 and smuggle it as "foreign" in proof 0
        statement, _ = recording.proved[1]
        smuggled = replace(evidence.proofs[0], foreign=encode_statement(statement))
        rejection = self.reject(
            verifier, EvidenceMessage((), (smuggled, *evidence.proofs[1:]))
        )
        assert rejection.code is VerificationCode.COVERAGE_MISMATCH
        assert "claiming this session" in rejection.detail

    def test_malformed_foreign_bytes_are_malformed(self, honest_pair) -> None:
        verifier, evidence = honest_pair
        broken = replace(evidence.proofs[0], foreign=b"not a statement")
        rejection = self.reject(
            verifier, EvidenceMessage((), (broken, *evidence.proofs[1:]))
        )
        assert rejection.code is VerificationCode.MALFORMED_TRANSCRIPT

    def test_proof_bytes_are_limited(
        self, compiled, expect, honest_values, model_weights, recording
    ) -> None:
        # single Merkle openings (value + path) pass; a whole batch proof does not
        limits = VerificationLimits(max_proof_bytes=256)
        verifier = VerifierSession(
            expect(SAMPLE_SOME, backend=RECORDING_BACKEND),
            compiled,
            backend=recording,
            limits=limits,
        )
        prover = ProverSession(
            compiled,
            verifier.header,
            honest_values,
            weight_tree=model_weights[1],
            backend=recording,
            limits=None,
        )
        replay = verifier.receive_boundary(prover.boundary())
        sample = verifier.receive_interiors(prover.interiors(replay))
        rejection = self.reject(verifier, prover.evidence(sample))
        assert rejection.code is VerificationCode.RESOURCE_LIMIT
        assert "proof_bytes" in rejection.detail


def test_a_corrupted_witness_fails_the_transparent_checker(
    compiled: Compiled,
    expect,
    honest_values,
    model_weights,
    recording: RecordingBackend,
) -> None:
    verifier, evidence = exchange(
        compiled,
        expect(SAMPLE_SOME, backend=RECORDING_BACKEND),
        honest_values,
        model_weights[1],
        recording,
    )
    ((statement, witness),) = recording.proved
    openings = list(witness.obligations[0])
    value, path = openings[0]
    openings[0] = (bytes(byte ^ 1 for byte in value), path)
    corrupted = Witness((tuple(openings), *witness.obligations[1:]))
    with pytest.raises(Reject) as info:
        recording.transparent.verify(statement, encode_witness(corrupted))
    assert info.value.code is VerificationCode.INVALID_OPENING
    # and a proof that is not even a witness
    with pytest.raises(Reject) as info:
        recording.transparent.verify(statement, b"garbage")
    assert info.value.code is VerificationCode.MALFORMED_TRANSCRIPT
    assert verifier.receive_evidence(evidence).accepted


def test_one_proof_can_cover_two_sessions(
    compiled: Compiled,
    expect,
    honest_values,
    model_weights,
    recording: RecordingBackend,
) -> None:
    """Session B's prover folds session A's obligations into its proof; both verifiers accept."""

    first = expect(backend=RECORDING_BACKEND, session_id=b"session-A")
    verifier_a, evidence_a = exchange(
        compiled, first, honest_values, model_weights[1], recording
    )
    ((statement_a, witness_a),) = recording.proved
    recording.proved.clear()
    assert verifier_a.receive_evidence(evidence_a).accepted

    second = expect(SAMPLE_SOME, backend=RECORDING_BACKEND, session_id=b"session-B")
    verifier_b = VerifierSession(second, compiled, backend=recording)
    prover_b = ProverSession(
        compiled,
        verifier_b.header,
        honest_values,
        weight_tree=model_weights[1],
        backend=recording,
    )
    replay = verifier_b.receive_boundary(prover_b.boundary())
    sample = verifier_b.receive_interiors(prover_b.interiors(replay))
    prover_b._plan = BatchPlan(
        (tuple(range(len(sample.selected))),), (ForeignBatch(statement_a, witness_a),)
    )
    evidence_b = prover_b.evidence(sample)

    (proof,) = evidence_b.proofs
    foreign = decode_statement(proof.foreign)
    assert foreign == statement_a
    ((joint, _),) = recording.proved
    assert len(joint.obligations) == len(statement_a.obligations) + len(sample.selected)
    assert {item.session for item in joint.obligations} == {
        verifier_a.header.digest,
        verifier_b.header.digest,
    }
    assert verifier_b.receive_evidence(evidence_b).accepted
    assert recording.verified[-1] == joint
    # the joint transcript round-trips and re-verifies
    data = encode_transcript(verifier_b.transcript)
    assert decode_transcript(data) == verifier_b.transcript
    assert verify_transcript(data, second, compiled, backend=recording).accepted


def test_statements_are_what_a_guest_would_hash(
    recording: RecordingBackend, compiled, expect, honest_values, model_weights
) -> None:
    exchange(
        compiled,
        expect(backend=RECORDING_BACKEND),
        honest_values,
        model_weights[1],
        recording,
    )
    ((statement, witness),) = recording.proved
    assert isinstance(statement, Statement)
    encoded = encode_statement(statement)
    assert decode_statement(encoded) == statement
    assert len(statement_digest(encoded)) == 32
    assert encode_witness(witness)
