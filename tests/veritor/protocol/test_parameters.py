"""The verifier fixes eta and prices every run before accepting a commitment."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from veritor.compile import Compiler
from veritor.constructors import Tracer
from veritor.core import (
    Compiled,
    ResourceLimit,
    VerificationLimits,
    VerificationPolicy,
    make_word_gate_set,
)
from veritor.protocol import (
    Expectation,
    ProtocolError,
    ProverSession,
    Reject,
    VerificationCode,
    VerifierParameters,
    VerifierSession,
    encode_transcript,
    expected_work,
    make_expectation,
    run_protocol,
    verify_transcript,
)
from veritor.protocol.challenge import bernoulli_subset
from veritor.protocol.merkle import merkle_depth

GATE_SET = make_word_gate_set(8)
EIGHTH = Fraction(1, 8)
CHECK_EVERYTHING = VerificationPolicy(1, 1)
SEEDS = {"session_id": b"parameters", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}


def one_unit_compiled(gates: int) -> Compiled:
    """One replay unit holding one verification unit of ``gates`` adds."""

    tracer = Tracer(GATE_SET)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key=("block", gates), role="verification")
    def block(v):
        accumulator = v[0]
        for _ in range(gates):
            accumulator = add(accumulator, v[0])
        return accumulator

    @tracer.definition(input_count=1, key=("root", gates), role="replay")
    def root(v):
        return block(v[0])

    return Compiler(GATE_SET).compile(tracer.serialize(root), (3,))


def expectation_for(compiled: Compiled, **overrides) -> Expectation:
    values = compiled.circuit.evaluate((3,))
    outputs = tuple(values[o] for o in compiled.circuit.outputs)
    return make_expectation(compiled, CHECK_EVERYTHING, (3,), outputs, **{**SEEDS, **overrides})


# -- eta is the verifier's ------------------------------------------------------


def test_default_parameters_fix_eta_at_zero() -> None:
    parameters = VerifierParameters()

    assert parameters.eta == 0
    assert parameters.max_capacity is None
    assert VerifierParameters("1/8", max_capacity=7, max_work=99).eta == EIGHTH
    with pytest.raises(ProtocolError, match="eta"):
        VerifierParameters(1)
    with pytest.raises(ProtocolError, match="max_work"):
        VerifierParameters(0, max_work=-1)


def test_the_proposal_is_theta_alone_and_the_header_binds_the_verifiers_eta(
    compiled, expect
) -> None:
    proposal = VerificationPolicy(1, Fraction(1, 2))
    admitted = expect(proposal, parameters=VerifierParameters(EIGHTH))

    assert admitted.policy == proposal
    assert admitted.parameters.eta == EIGHTH
    header = VerifierSession(admitted, compiled).header
    assert header.policy == proposal and header.eta == EIGHTH
    assert header != VerifierSession(expect(proposal), compiled).header
    with pytest.raises(ProtocolError, match="VerificationPolicy"):
        VerifierParameters().policy((1, 1))  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="VerificationPolicy"):
        Expectation(
            admitted.session_id,
            admitted.compiled_digest,
            (1, 1),  # type: ignore[arg-type]
            admitted.parameters,
            admitted.public_inputs,
            admitted.claimed_outputs,
            admitted.q_seed,
            admitted.s_seed,
        )


def test_a_transcript_recorded_under_another_eta_is_rejected(
    compiled, honest_values, expect
) -> None:
    recorded = expect(parameters=VerifierParameters(EIGHTH))
    run = run_protocol(compiled, recorded, honest_values)
    assert run.transcript is not None
    assert run.transcript.header.eta == EIGHTH
    data = encode_transcript(run.transcript)
    assert verify_transcript(data, recorded, compiled) == run.report

    report = verify_transcript(data, expect(), compiled)

    assert report.code is VerificationCode.EXPECTATION_MISMATCH
    assert "eta 1/8" in report.detail and "0" in report.detail
    assert report.sampled_replay_units == ()

    other = run_protocol(compiled, expect(), honest_values)
    assert other.transcript is not None
    assert other.transcript.header.digest != run.transcript.header.digest
    reverse = verify_transcript(encode_transcript(other.transcript), recorded, compiled)
    assert reverse.code is VerificationCode.EXPECTATION_MISMATCH and "eta" in reverse.detail


# -- denominators are capped ----------------------------------------------------


def test_huge_denominators_are_rejected_at_admission_and_in_derivation(
    compiled, honest_values, expect
) -> None:
    huge = VerificationPolicy(Fraction(1, 1 << 70), 1)

    run = run_protocol(compiled, expect(huge), honest_values)

    assert run.report.code is VerificationCode.RESOURCE_LIMIT
    assert "probability_denominator_bits" in run.report.detail
    assert run.transcript is None
    assert run.report.sampled_replay_units == ()
    # the verifier's own eta is bound into the header and decoded under the same cap
    huge_eta = run_protocol(
        compiled, expect(parameters=VerifierParameters(Fraction(1, 1 << 70))), honest_values
    )
    assert huge_eta.report.code is VerificationCode.RESOURCE_LIMIT

    with pytest.raises(ResourceLimit, match="probability_denominator_bits"):
        bernoulli_subset(b"Q" * 32, b"stage", b"\0" * 32, 5, huge.q, VerificationLimits())
    relaxed = VerificationLimits(max_probability_denominator_bits=71)
    assert bernoulli_subset(b"Q" * 32, b"stage", b"\0" * 32, 5, huge.q, relaxed) == ()


def test_a_transcript_with_a_huge_denominator_is_a_clean_resource_limit(
    compiled, honest_values, expect
) -> None:
    expectation = expect()
    run = run_protocol(compiled, expectation, honest_values)
    assert run.transcript is not None
    document = json.loads(encode_transcript(run.transcript))
    document["header"]["policy"]["q"] = [1, 1 << 4000]
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    report = verify_transcript(data, expectation, compiled)

    assert report.code is VerificationCode.RESOURCE_LIMIT
    assert "probability_denominator_bits" in report.detail


# -- per-unit caps and resource limits are verdicts -----------------------------


def test_oversized_units_are_rejected_at_session_start() -> None:
    compiled = one_unit_compiled(200)
    expectation = expectation_for(compiled)
    limits = VerificationLimits(max_positions_per_unit=200)

    with pytest.raises(Reject) as rejection:
        VerifierSession(expectation, compiled, limits=limits)
    assert rejection.value.code is VerificationCode.RESOURCE_LIMIT
    assert "positions_per_unit is 201" in rejection.value.detail

    values = dict(enumerate(compiled.circuit.evaluate((3,))))
    run = run_protocol(compiled, expectation, values, limits=limits)
    assert run.report.code is VerificationCode.RESOURCE_LIMIT
    assert run.transcript is None

    accepted = run_protocol(
        compiled, expectation, values, limits=VerificationLimits(max_positions_per_unit=201)
    )
    assert accepted.report.accepted


def test_a_limit_hit_during_the_run_is_a_reject_not_an_exception() -> None:
    compiled = one_unit_compiled(200)
    expectation = expectation_for(compiled)
    values = dict(enumerate(compiled.circuit.evaluate((3,))))
    limits = VerificationLimits(max_openings=100)

    run = run_protocol(compiled, expectation, values, limits=limits)

    assert run.report.code is VerificationCode.RESOURCE_LIMIT
    assert "openings is 201" in run.report.detail
    assert run.report.sampled_replay_units == (0,)
    assert run.report.sampled_verification_units == (0,)

    verifier = VerifierSession(expectation, compiled, limits=limits)
    prover = ProverSession(compiled, verifier.header, values)
    replay_challenge = verifier.receive_boundary(prover.boundary())
    sample_challenge = verifier.receive_interiors(prover.interiors(replay_challenge))
    evidence = prover.evidence(sample_challenge)
    with pytest.raises(Reject) as rejection:
        verifier.receive_evidence(evidence)
    assert rejection.value.code is VerificationCode.RESOURCE_LIMIT
    with pytest.raises(Reject) as after:
        verifier.receive_evidence(evidence)
    assert after.value.code is VerificationCode.INVALID_PHASE


# -- W_max -----------------------------------------------------------------------


def test_expected_work_follows_the_documented_formula(compiled, workload) -> None:
    index = compiled.index
    dots = index.verification_unit_count
    dot = index.verification_unit(0)
    size, reads = dot.size, len(compiled.circuit.In(dot))
    io = len(workload.public_inputs) + len(compiled.circuit.outputs)
    depth = merkle_depth(index.n)

    for q, s in ((Fraction(1), Fraction(1)), (Fraction(1, 2), Fraction(1, 3))):
        work = expected_work(compiled, VerificationPolicy(q, s), io)
        assert work == (
            (io + q * s * dots * (size + reads)) * (1 + depth)
            + q * s * dots * size
            + 1
            + q * index.replay_units.count
        )
    assert expected_work(compiled, CHECK_EVERYTHING, io) > expected_work(
        compiled, VerificationPolicy(Fraction(1, 2), 1), io
    )


def test_runs_above_the_work_budget_are_rejected_before_any_commitment(
    compiled, honest_values, workload, expect
) -> None:
    io = len(workload.public_inputs) + len(compiled.circuit.outputs)
    work = expected_work(compiled, CHECK_EVERYTHING, io)
    assert work.denominator == 1

    exact = run_protocol(
        compiled, expect(parameters=VerifierParameters(max_work=int(work))), honest_values
    )
    short = run_protocol(
        compiled, expect(parameters=VerifierParameters(max_work=int(work) - 1)), honest_values
    )

    assert exact.report.accepted
    assert short.report.code is VerificationCode.WORK_BUDGET_EXCEEDED
    assert short.transcript is None
    assert short.report.sampled_replay_units == ()
    assert f"W_max {int(work) - 1}" in short.report.detail

    cheaper = expect(
        VerificationPolicy(Fraction(1, 2), Fraction(1, 2)),
        parameters=VerifierParameters(max_work=int(work) - 1),
    )
    assert run_protocol(compiled, cheaper, honest_values).report.accepted
