from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from veritor.constructors import (
    DemoGCompileRequest,
    compile_demo_g,
    expected_dot_outputs,
    expected_matmul_outputs,
)
from veritor.core import Compiled, VerificationPolicy
from veritor.protocol import (
    Expectation,
    InteriorMessage,
    Opening,
    ProverSession,
    ReplayChallenge,
    VerificationCode,
    VerifierSession,
    assignment_replay,
    decode_transcript,
    encode_transcript,
    make_expectation,
    run_protocol,
    verify_transcript,
)

from .conftest import (
    CHECK_EVERYTHING,
    NO_CAPACITY_STATEMENT,
    Q_SEED,
    S_SEED,
    SESSION_ID,
)


@pytest.fixture(scope="module")
def demo() -> tuple[
    Compiled, Callable[[VerificationPolicy], Expectation], dict[int, object]
]:
    """A DemoG run: chained multiply-accumulates, so every dot has an interior.

    The matmul fixture has none: each dot output is a row output, hence a
    boundary position, and (in the fixture) a circuit output as well.
    """

    compilation = compile_demo_g()
    request = DemoGCompileRequest()
    compiled = compilation.compiled
    values = dict(enumerate(compiled.circuit.evaluate(request.public_inputs)))
    outputs = expected_dot_outputs(request.batch, request.width)

    def expectation(policy: VerificationPolicy) -> Expectation:
        return make_expectation(
            compilation,
            policy,
            outputs,
            parameters=NO_CAPACITY_STATEMENT,
            session_id=SESSION_ID,
            q_seed=Q_SEED,
            s_seed=S_SEED,
        )

    return compiled, expectation, values


def forge_interior(compiled: Compiled, values: dict[int, object]) -> dict[int, object]:
    """Corrupt the first interior position: a VU output that is not an RU output.

    Only the VU that computes it (or the one that consumes it) can notice: the
    public-I/O check never sees it and the interior commitment is whatever the
    prover says.
    """

    forged = dict(values)
    index = compiled.index
    unit = next(r for r in range(index.replay_units.count) if index.interior(r).count)
    address = int(index.interior(unit).unrank(0))
    assert address not in set(compiled.circuit.outputs)
    forged[address] = (forged[address] + 1) % 256  # type: ignore[operator]
    return forged


def test_honest_run_accepts_and_transcript_round_trips(
    compiled, honest_values, expect
) -> None:
    expectation = expect()

    run = run_protocol(compiled, expectation, honest_values)

    assert run.report.code is VerificationCode.ACCEPTED
    assert run.report.accepted
    assert run.transcript is not None
    assert run.report.sampled_replay_units == tuple(
        range(compiled.index.replay_units.count)
    )
    assert run.report.sampled_verification_units == tuple(
        range(compiled.index.verification_unit_count)
    )
    data = encode_transcript(run.transcript)
    assert decode_transcript(data) == run.transcript
    assert verify_transcript(data, expectation, compiled) == run.report


def test_sessions_produce_identical_transcripts_on_both_sides(
    compiled, honest_values, expect, model_weights
) -> None:
    verifier = VerifierSession(expect(), compiled)
    prover = ProverSession(
        compiled, verifier.header, honest_values, weight_tree=model_weights[1]
    )

    replay_challenge = verifier.receive_boundary(prover.boundary())
    sample_challenge = verifier.receive_interiors(prover.interiors(replay_challenge))
    report = verifier.receive_evidence(prover.evidence(sample_challenge))

    assert report.code is VerificationCode.ACCEPTED
    assert prover.transcript == verifier.transcript


def test_forged_interior_is_rejected_when_every_unit_is_checked(demo) -> None:
    compiled, expectation, honest = demo
    forged = forge_interior(compiled, honest)

    run = run_protocol(
        compiled,
        expectation(CHECK_EVERYTHING),
        forged,
        replay=assignment_replay(forged),
    )

    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None


def test_forged_interior_survives_when_nothing_is_sampled(demo) -> None:
    compiled, expectation, honest = demo
    forged = forge_interior(compiled, honest)

    run = run_protocol(
        compiled,
        expectation(VerificationPolicy(1, 0)),
        forged,
        replay=assignment_replay(forged),
    )

    assert run.report.code is VerificationCode.ACCEPTED
    assert run.report.sampled_verification_units == ()


def test_matmul_commits_no_interior_because_every_dot_output_is_a_row_output(
    compiled,
) -> None:
    index = compiled.index
    assert all(index.interior(r).count == 0 for r in range(index.replay_units.count))
    assert all(kind.interior_count == 0 for kind in index.kinds())


def test_wrong_claimed_output_is_rejected_at_the_boundary(
    compiled, honest_values, workload, expect
) -> None:
    wrong = tuple(output + 1 for output in expected_matmul_outputs(workload))

    run = run_protocol(compiled, expect(claimed_outputs=wrong), honest_values)

    assert run.report.code is VerificationCode.PUBLIC_IO_MISMATCH
    assert run.report.sampled_replay_units == ()


def test_wrong_input_value_is_rejected_at_the_boundary(
    compiled, honest_values, expect
) -> None:
    """Every ``in`` gate is opened at commit and compared to the public input of its rank."""

    address = compiled.circuit.inputs[2]
    assert compiled.circuit[address].is_input
    forged = dict(honest_values)
    forged[address] = (forged[address] + 1) % 256

    run = run_protocol(compiled, expect(VerificationPolicy(0, 0)), forged)

    assert run.report.code is VerificationCode.PUBLIC_IO_MISMATCH
    assert f"input at address {address}" in run.report.detail
    assert run.report.sampled_replay_units == ()


def test_verifier_rejects_transcript_recorded_under_other_seeds(
    compiled, honest_values, expect
) -> None:
    expectation = expect()
    run = run_protocol(compiled, expectation, honest_values)
    assert run.transcript is not None
    data = encode_transcript(run.transcript)

    other_q = verify_transcript(data, replace(expectation, q_seed=b"R" * 32), compiled)
    other_s = verify_transcript(data, replace(expectation, s_seed=b"T" * 32), compiled)
    other_session = verify_transcript(data, expect(session_id=b"other"), compiled)

    assert other_q.code is VerificationCode.EXPECTATION_MISMATCH
    assert "q seed" in other_q.detail
    assert other_s.code is VerificationCode.EXPECTATION_MISMATCH
    assert "s seed" in other_s.detail
    assert other_session.code is VerificationCode.EXPECTATION_MISMATCH


def test_tampered_replay_selection_is_a_challenge_mismatch(
    compiled, honest_values, expect
) -> None:
    expectation = expect()
    run = run_protocol(compiled, expectation, honest_values)
    assert run.transcript is not None
    tampered = replace(
        run.transcript,
        replay_challenge=ReplayChallenge(expectation.q_seed, ()),
        interiors=InteriorMessage(()),
    )

    report = verify_transcript(encode_transcript(tampered), expectation, compiled)

    assert report.code is VerificationCode.CHALLENGE_MISMATCH


def test_tampered_opening_fails_authentication(compiled, honest_values, expect) -> None:
    expectation = expect()
    run = run_protocol(compiled, expectation, honest_values)
    assert run.transcript is not None
    first = run.transcript.boundary.io_openings[0]
    flipped = bytes((first.value[0] ^ 1,)) + first.value[1:]
    tampered = replace(
        run.transcript,
        boundary=replace(
            run.transcript.boundary,
            io_openings=(Opening(first.position, flipped, first.path),)
            + run.transcript.boundary.io_openings[1:],
        ),
    )

    report = verify_transcript(encode_transcript(tampered), expectation, compiled)

    assert report.code is VerificationCode.INVALID_OPENING
