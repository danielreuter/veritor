from __future__ import annotations

import pytest

from veritor import (
    Compiled,
    DemoGCompileRequest,
    MatmulCompileRequest,
    VerificationCode,
    VerificationPolicy,
    Verify,
    build_executable_conformance_transcript,
    compile_demo_g,
    compile_matmul,
    make_verification_expectation,
    run_protocol,
)
from veritor.protocol import assignment_replay, encode_transcript

SEEDS = {"session_id": b"research-api/conformance", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}
CHECK_EVERYTHING = VerificationPolicy(1, 1)
WORKLOADS = {
    "demo-g": (DemoGCompileRequest(), compile_demo_g),
    "matmul": (MatmulCompileRequest(), compile_matmul),
}


@pytest.mark.parametrize("name", tuple(WORKLOADS))
def test_honest_conformance_transcript_verifies_purely(name: str) -> None:
    request, compile = WORKLOADS[name]
    compiled = compile(request)
    assert isinstance(compiled, Compiled)

    run = build_executable_conformance_transcript(
        compiled, request.public_inputs, CHECK_EVERYTHING, **SEEDS
    )

    report = Verify(run.transcript_bytes, run.expectation, compiled)
    assert report.accepted
    assert report.code is VerificationCode.ACCEPTED
    assert report.sampled_verification_units == tuple(
        range(compiled.index.verification_unit_count)
    )
    assert run.expectation.claimed_outputs == request.expected_outputs
    assert run.expectation.compiled_digest == compiled.digest


def test_conformance_transcript_is_deterministic_given_seeds() -> None:
    request = DemoGCompileRequest()
    compiled = compile_demo_g(request)

    first = build_executable_conformance_transcript(compiled, request.public_inputs, **SEEDS)
    second = build_executable_conformance_transcript(compiled, request.public_inputs, **SEEDS)

    assert first == second


def test_forged_sampled_demo_execution_is_rejected() -> None:
    request = DemoGCompileRequest()
    compiled = compile_demo_g(request)
    expectation = make_verification_expectation(
        compiled, CHECK_EVERYTHING, request.public_inputs, request.expected_outputs, **SEEDS
    )

    values = list(compiled.circuit.evaluate(request.public_inputs))
    attacked = int(compiled.index.interior(0).unrank(0))
    values[attacked] = (values[attacked] + 1) % (1 << request.width)
    forged = dict(enumerate(values))

    run = run_protocol(compiled, expectation, forged, replay=assignment_replay(forged))

    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None


def test_verify_rejects_transcript_against_the_wrong_expectation() -> None:
    request = DemoGCompileRequest()
    compiled = compile_demo_g(request)
    honest = build_executable_conformance_transcript(compiled, request.public_inputs, **SEEDS)
    other = make_verification_expectation(
        compiled,
        CHECK_EVERYTHING,
        request.public_inputs,
        request.expected_outputs,
        session_id=b"research-api/other",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )

    report = Verify(honest.transcript_bytes, other, compiled)

    assert report.code is VerificationCode.EXPECTATION_MISMATCH


def test_interactive_run_and_pure_verification_agree() -> None:
    request = MatmulCompileRequest()
    compiled = compile_matmul(request)
    expectation = make_verification_expectation(
        compiled, CHECK_EVERYTHING, request.public_inputs, request.expected_outputs, **SEEDS
    )
    values = dict(enumerate(compiled.circuit.evaluate(request.public_inputs)))

    run = run_protocol(compiled, expectation, values)

    assert run.transcript is not None
    assert Verify(encode_transcript(run.transcript), expectation, compiled) == run.report
