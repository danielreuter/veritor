"""The datacenter simulation at small scale: every claim the report makes, checked.

The default ``small`` configuration is run once (about 15 s on the reference
machine, most of it the adversary's trials): the honest run is accepted;
every response is a prefix of ``reference_generate`` although requests stop
at EOS and at least one pod failure restarts requests; the adversary's secret
decodes from the streamed tokens; the observed survival of every attack row
lies within :data:`TOLERANCE_SIGMAS` (4) standard deviations of the
prediction, the tolerance the security tests use too; the advice is charged
at the encoded schedule's length.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import pairwise

import pytest

from veritor.analysis.probability import budget, survival, unit_cost
from veritor.constructors import (
    ClusterG,
    LMShape,
    Request,
    random_parameters,
    reference_generate,
)
from veritor.core import VerificationPolicy, make_isa_gate_set
from veritor.protocol import (
    Reject,
    VerificationCode,
    VerifierParameters,
    VerifierSession,
    commit_weights,
    make_expectation,
)
from veritor.research import Compile
from veritor.simulation.datacenter import (
    POLICY,
    TOLERANCE_SIGMAS,
    DemoConfig,
    attack_sizes,
    main,
    render,
    run,
    small_config,
)
from veritor.simulation.workload import EOS, FAILED, simulate


@pytest.fixture(scope="module")
def config() -> DemoConfig:
    return small_config()


@pytest.fixture(scope="module")
def summary(config):
    return run(config)


def test_the_honest_run_is_accepted(summary):
    honest = summary.honest
    assert honest.accepted and honest.code == VerificationCode.ACCEPTED.name
    assert honest.replay_units_opened > 0 and honest.verification_units_opened > 0
    assert honest.openings > honest.verification_units_opened  # every VU opens its gates and inputs
    assert len(honest.interior_roots) == honest.replay_units_opened
    assert honest.prover_seconds > 0 and honest.verifier_seconds > 0
    assert honest.transcript_bytes > sum(honest.message_bytes.values())
    assert honest.max_capacity == math.ceil(summary.bound.bits)


def test_every_response_is_the_reference_despite_eos_and_restarts(config, summary):
    """The circuit's outputs, the simulation's tokens and the oracle agree, request by request."""

    w = summary.workload
    assert w.matches_reference
    assert w.eos_stops >= 1, "no request stopped at EOS: the control flow was not exercised"
    assert w.restarts >= 1 and any(f.aborted for f in w.failures), "no failure restarted a request"
    shape, workload = config.shape, config.workload
    parameters = random_parameters(shape, config.parameters_seed)
    simulation = simulate(workload, shape, parameters)
    reference = reference_generate(shape, parameters, simulation.requests)
    assert w.responses == simulation.streamed
    for tokens, expected in zip(w.responses, reference, strict=True):
        assert tokens == expected[: len(tokens)]
    # a request cut at EOS or by a failure streamed fewer tokens than it asked for
    assert any(
        len(tokens) < request.max_new
        for tokens, request in zip(w.responses, simulation.requests, strict=True)
    )


def test_a_restarted_request_streams_each_position_exactly_once(summary):
    """The aborted attempt's tokens stand; the restart recomputes them and streams only new ones."""

    w = summary.workload
    restarted = {a.request for a in w.attempts if a.outcome == FAILED}
    assert restarted
    for request in restarted:
        attempts = sorted((a for a in w.attempts if a.request == request), key=lambda a: a.step)
        assert len(attempts) >= 2
        streamed = [position for a in attempts for position in a.streamed]
        assert streamed == list(range(len(w.responses[request])))
        for earlier, later in pairwise(attempts):
            assert (
                earlier.step + earlier.length <= later.step
            )  # a restart never overlaps its predecessor


def test_the_secret_decodes_from_the_streamed_tokens(summary):
    a = summary.adversary
    assert a.decoded
    assert a.bits_per_vu == summary.model.vocab_bits
    assert a.kappa_per_vu == summary.compile.head_vu_cut_bits == summary.model.width
    for row in a.rows:
        assert row.bits == row.carriers * a.bits_per_vu
        assert row.vus_corrupted <= row.carriers  # a carrier already spelling its chunk is free
        assert row.decoded == row.secret and len(row.secret) == row.bits
        assert row.honest_tokens_unchanged
        assert sum(row.errors_per_replay_unit) == row.vus_corrupted
        assert len(row.errors_per_replay_unit) == row.replay_units_touched
    assert [row.carriers for row in a.rows] == list(attack_sizes(summary.workload.tokens))
    assert a.rows[-1].carriers == summary.workload.tokens
    assert a.rows[-1].vus_corrupted >= summary.workload.tokens // 2


def test_observed_survival_matches_the_prediction(summary):
    """Detection over fresh challenges follows ``sigma(E)`` within the documented tolerance."""

    a = summary.adversary
    policy = VerificationPolicy(Fraction(summary.policy.q), Fraction(summary.policy.s))
    assert policy == POLICY
    assert a.tolerance_sigmas == TOLERANCE_SIGMAS == 4.0
    for row in a.rows:
        assert row.predicted_survival == pytest.approx(
            float(survival(policy, row.errors_per_replay_unit))
        )
        assert row.trials >= 400
        assert row.deviation_sigmas <= TOLERANCE_SIGMAS, (
            row.vus_corrupted,
            row.predicted_survival,
            row.observed_survival,
        )
        assert row.observed_survival == row.escaped / row.trials
        assert 0 <= row.protocol_accepted <= row.protocol_trials
    # more corrupted VUs, less survival; the whole output is the last row
    predicted = [row.predicted_survival for row in a.rows]
    assert predicted == sorted(predicted, reverse=True)
    assert predicted[-1] < 0.25


def test_the_advice_is_charged_at_the_encoded_schedule(config, summary):
    """``a`` is ``Schedule.encode()``, charged at the schedule's bit length; a shorter ``A`` rejects the run."""

    w = summary.workload
    shape, workload = config.shape, config.workload
    parameters = random_parameters(shape, config.parameters_seed)
    simulation = simulate(workload, shape, parameters)
    advice = simulation.schedule.encode()
    assert len(advice) == w.advice_bytes == summary.compile.advice_bytes == -(-w.advice_bits // 8)
    assert w.advice_bits == simulation.schedule.bit_length() < 8 * w.advice_bytes + 8
    gate_set = make_isa_gate_set(shape.width)
    constructor = ClusterG(shape, workload.pods, workload.slots, workload.steps)
    compilation = Compile(
        constructor, simulation.requests, advice, gate_set, max_advice_bits=8 * len(advice)
    )
    assert compilation.advice_bits == w.advice_bits
    kappa, _tree = commit_weights(gate_set, parameters.flatten())
    outputs = tuple(token for response in simulation.streamed for token in response)
    parameters_short = VerifierParameters(
        config.eta, max_capacity=None, max_advice_bits=compilation.advice_bits - 1
    )
    expectation = make_expectation(
        compilation, POLICY, outputs, parameters=parameters_short, weights=kappa
    )
    with pytest.raises(Reject) as caught:
        VerifierSession(expectation, compilation.compiled)
    assert caught.value.code is VerificationCode.POLICY_REJECTED


def test_bound_and_cost_are_reported(summary):
    b, c, k = summary.bound, summary.compile, summary.cost
    policy = POLICY
    assert b.eta == "2^-40"
    assert b.bits <= b.out_bits == c.out_bits == c.outputs * summary.model.width
    assert b.capped, "at small scale the interface cap binds; the report documents this"
    assert b.budget_nats == pytest.approx(budget(Fraction(1, 2**40)))
    assert b.unit_cost_nats == pytest.approx(unit_cost(policy, 1))
    assert b.vus_to_eta == math.ceil(b.budget_nats / b.unit_cost_nats)
    assert b.bits_charged_to_eta == b.vus_to_eta * c.head_vu_cut_bits
    assert b.bits_realized_to_eta == b.vus_to_eta * summary.model.vocab_bits
    assert k.total == pytest.approx(k.boundary + k.recompute + k.commit_interior + k.proof)
    assert k.weights_per_epoch == summary.model.weights
    assert (
        c.W_R > 0 and c.W_V > 0 and c.positions_per_vu > c.head_vu_gates * 0
    )  # defined and positive
    assert c.gates_per_token_step * summary.workload.token_steps == pytest.approx(
        c.n - c.weight_gates
    )
    assert c.replay_units == 1 + sum(1 for row in summary.workload.occupancy for n in row if n > 0)


def test_the_summary_dumps_to_json_and_renders(summary):
    document = json.loads(summary.to_json())
    assert document["honest"]["accepted"] is True
    assert document["workload"]["tokens"] == summary.workload.tokens
    assert document["adversary"]["rows"][-1]["decoded"] == summary.adversary.rows[-1].secret
    report = render(summary)
    for heading in (
        "1. Workload",
        "5. Compile",
        "6. Honest protocol run",
        "7. Adversary",
        "8. Bound",
    ):
        assert heading in report
    assert "ACCEPTED" in report and "unit" not in report.replace("units", "").replace("unit_", "")


def test_the_command_line_runs_a_reduced_configuration(tmp_path):
    """``--no-sampling`` falls back to the argmax head; the CLI writes the JSON summary."""

    target = tmp_path / "summary.json"
    code = main(
        [
            "--scale",
            "small",
            "--steps",
            "10",
            "--requests",
            "6",
            "--trials",
            "100",
            "--no-sampling",
            "--quiet",
            "--json",
            str(target),
        ]
    )
    assert code == 0
    document = json.loads(target.read_text())
    assert document["model"]["sampling"] is False and document["model"]["random_bits"] == 0
    assert document["honest"]["accepted"] is True
    assert document["workload"]["steps"] == 10 and document["workload"]["arrivals"] == 6
    assert any("argmax" in note for note in document["notes"])
    assert document["adversary"]["rows"][0]["trials"] == 100


def test_the_sampled_head_consumes_the_published_randomness(config):
    """With sampling on, the requests carry one random word per position and the shape checks them."""

    shape = config.shape
    assert shape.sampling and shape.random_bits > 0
    parameters = random_parameters(shape, config.parameters_seed)
    simulation = simulate(config.workload, shape, parameters)
    for request in simulation.requests:
        assert len(request.randomness) == request.max_new
        shape.check_randomness(request)
    with pytest.raises(ValueError):
        LMShape(vocab=8, d_model=4, heads=2, layers=1, context=16, width=16).check_randomness(
            simulation.requests[0]
        )
    assert simulation.eos_stops == sum(a.outcome == EOS for a in simulation.attempts)
    assert Request((1,), 1, (0,)).randomness == (0,)
