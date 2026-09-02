from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import pytest

from veritor import (
    Bound,
    Compilation,
    DemoGCompileRequest,
    MatmulCompileRequest,
    VerificationCode,
    VerificationPolicy,
    VerifierParameters,
    Verify,
    build_executable_conformance_transcript,
    compile_demo_g,
    compile_matmul,
    make_verification_expectation,
    run_protocol,
)
from veritor.constructors import MatmulG
from veritor.protocol import assignment_replay, commit_weights, encode_transcript

SEEDS = {"session_id": b"research-api/conformance", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}
NO_CAPACITY = VerifierParameters(max_capacity=None)
CHECK_EVERYTHING = VerificationPolicy(1, 1)
WORKLOADS = {
    "demo-g": (DemoGCompileRequest(), compile_demo_g, ()),
    "demo-g-advice": (
        DemoGCompileRequest(advice=b"hint", max_advice_bits=32),
        compile_demo_g,
        (),
    ),
    "matmul": (MatmulCompileRequest(), compile_matmul, MatmulCompileRequest().weight_values),
}


@pytest.mark.parametrize("name", tuple(WORKLOADS))
def test_honest_conformance_transcript_verifies_purely(name: str) -> None:
    request, compile, weights = WORKLOADS[name]
    compilation = compile(request)
    assert isinstance(compilation, Compilation)
    compiled = compilation.compiled
    assert compiled.index.weight_count == len(weights)
    advice = getattr(request, "advice", b"")
    parameters = VerifierParameters(max_advice_bits=8 * len(advice), max_capacity=None)

    run = build_executable_conformance_transcript(
        compilation, CHECK_EVERYTHING, weights=weights, parameters=parameters, **SEEDS
    )

    report = Verify(run.transcript_bytes, run.expectation, compiled)
    assert report.accepted
    assert report.code is VerificationCode.ACCEPTED
    assert report.sampled_verification_units == tuple(
        range(compiled.index.verification_unit_count)
    )
    assert run.expectation.claimed_outputs == request.expected_outputs
    assert run.expectation.public_inputs == request.public_inputs
    assert run.expectation.compiled_digest == compiled.digest
    assert run.expectation.constructor == compilation.constructor
    assert run.expectation.advice == advice
    # the weights, if any, are bound to the header as kappa_W and never carried in the run
    if weights:
        assert run.expectation.weights == commit_weights(compiled.circuit.gate_set, weights)[0]
        assert run.expectation.weights.count == len(weights)
    else:
        assert run.expectation.weights is None


def test_conformance_transcript_is_deterministic_given_seeds() -> None:
    compilation = compile_demo_g(DemoGCompileRequest())

    first = build_executable_conformance_transcript(compilation, **SEEDS)
    second = build_executable_conformance_transcript(compilation, **SEEDS)

    assert first == second


def test_forged_sampled_demo_execution_is_rejected() -> None:
    request = DemoGCompileRequest()
    compilation = compile_demo_g(request)
    compiled = compilation.compiled
    expectation = make_verification_expectation(
        compilation, CHECK_EVERYTHING, request.expected_outputs, parameters=NO_CAPACITY, **SEEDS
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
    compilation = compile_demo_g(request)
    honest = build_executable_conformance_transcript(compilation, **SEEDS)
    other = make_verification_expectation(
        compilation,
        CHECK_EVERYTHING,
        request.expected_outputs,
        parameters=NO_CAPACITY,
        session_id=b"research-api/other",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )

    report = Verify(honest.transcript_bytes, other, compilation.compiled)

    assert report.code is VerificationCode.EXPECTATION_MISMATCH


def test_the_header_binds_the_constructor_and_the_advice() -> None:
    """A transcript recorded under ``(G, a)`` fails against ``(G, a')`` and ``(G', a)``."""

    request = DemoGCompileRequest(advice=b"a", max_advice_bits=8)
    compilation = compile_demo_g(request)
    parameters = VerifierParameters(max_advice_bits=8, max_capacity=None)
    honest = build_executable_conformance_transcript(compilation, parameters=parameters, **SEEDS)
    assert Verify(honest.transcript_bytes, honest.expectation, compilation.compiled).accepted

    def expecting(**changes) -> object:
        return make_verification_expectation(
            replace(compilation, **changes),
            CHECK_EVERYTHING,
            request.expected_outputs,
            parameters=parameters,
            **SEEDS,
        )

    for other in (expecting(advice=b"b"), expecting(advice=b""), expecting(constructor=MatmulG(8).digest)):
        report = Verify(honest.transcript_bytes, other, compilation.compiled)
        assert report.code is VerificationCode.EXPECTATION_MISMATCH
        assert "header" in report.detail


def test_advice_over_the_verifiers_bound_is_rejected_at_admission() -> None:
    """``A = max_advice_bits`` is enforced by the transcript verifier, not only by ``Compile``."""

    request = DemoGCompileRequest(advice=b"hint", max_advice_bits=32)
    compilation = compile_demo_g(request)
    compiled = compilation.compiled
    values = dict(enumerate(compiled.circuit.evaluate(compilation.inputs)))
    admitted = build_executable_conformance_transcript(
        compilation, parameters=VerifierParameters(max_advice_bits=32, max_capacity=None), **SEEDS
    )
    assert Verify(admitted.transcript_bytes, admitted.expectation, compiled).accepted

    for bound_bits in (31, 0):
        capped = make_verification_expectation(
            compilation,
            CHECK_EVERYTHING,
            request.expected_outputs,
            parameters=VerifierParameters(max_advice_bits=bound_bits, max_capacity=None),
            **SEEDS,
        )
        rejected = run_protocol(compiled, capped, values)
        assert rejected.report.code is VerificationCode.POLICY_REJECTED
        assert f"32 bits, exceeding max_advice_bits {bound_bits}" in rejected.report.detail
        assert rejected.transcript is None
        assert Verify(admitted.transcript_bytes, capped, compiled) == rejected.report
    # the default verifier admits no advice at all
    assert (
        run_protocol(
            compiled,
            make_verification_expectation(
                compilation, CHECK_EVERYTHING, request.expected_outputs, parameters=NO_CAPACITY, **SEEDS
            ),
            values,
        ).report.code
        is VerificationCode.POLICY_REJECTED
    )
    with pytest.raises(RuntimeError, match="max_advice_bits"):
        build_executable_conformance_transcript(compilation, **SEEDS)


def test_interactive_run_and_pure_verification_agree() -> None:
    request = MatmulCompileRequest()
    compilation = compile_matmul(request)
    compiled = compilation.compiled
    weights, tree = commit_weights(compiled.circuit.gate_set, request.weight_values)
    expectation = make_verification_expectation(
        compilation,
        CHECK_EVERYTHING,
        request.expected_outputs,
        parameters=NO_CAPACITY,
        weights=weights,
        **SEEDS,
    )
    values = dict(enumerate(compiled.circuit.evaluate(request.public_inputs, request.weight_values)))

    run = run_protocol(compiled, expectation, values, weight_tree=tree)

    assert run.transcript is not None
    data = encode_transcript(run.transcript)
    assert Verify(data, expectation, compiled) == run.report

    # ... and at admission: a U_max below Bound(C, I, theta) rejects both ways alike
    theta, eta = VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 4)
    bits = Bound(compiled, theta, eta).bits
    capped = make_verification_expectation(
        compilation,
        theta,
        request.expected_outputs,
        parameters=VerifierParameters(eta, max_capacity=math.ceil(bits) - 1),
        weights=weights,
        **SEEDS,
    )
    rejected = run_protocol(compiled, capped, values, weight_tree=tree)
    assert rejected.report.code is VerificationCode.POLICY_REJECTED
    assert rejected.transcript is None
    assert Verify(data, capped, compiled) == rejected.report
