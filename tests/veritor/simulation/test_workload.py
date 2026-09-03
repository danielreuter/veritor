"""The simulated server: arrivals, first-come first-served batching, EOS, failures and restarts."""

from __future__ import annotations

import pytest

from veritor.constructors import LMShape, random_parameters, reference_generate
from veritor.simulation.workload import (
    COMPLETE,
    EOS,
    FAILED,
    RUN_END,
    WorkloadConfig,
    check_against_reference,
    generate_arrivals,
    simulate,
)

SAMPLED = LMShape(
    vocab=8, d_model=4, heads=2, layers=1, context=16, width=16, sampling=True
)
ARGMAX = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=16, width=16)


def config(**overrides) -> WorkloadConfig:
    base = {"pods": 2, "slots": 2, "steps": 12, "arrivals": 10, "seed": 3}
    return WorkloadConfig(**{**base, **overrides})


def test_arrivals_are_a_seeded_poisson_process_with_the_configured_lengths():
    arrivals = generate_arrivals(config(), SAMPLED)
    assert len(arrivals) == 10
    times = [a.time for a in arrivals]
    assert times == sorted(times) and times[0] > 0
    for a in arrivals:
        assert 1 <= len(a.request.prompt) <= 4 and 2 <= a.request.max_new <= 8
        assert len(a.request.randomness) == a.request.max_new
        assert a.request_id is None
    assert generate_arrivals(config(), SAMPLED) == arrivals
    assert generate_arrivals(config(seed=4), SAMPLED) != arrivals
    assert all(a.request.randomness == () for a in generate_arrivals(config(), ARGMAX))


def test_requests_are_admitted_first_come_first_served_and_never_double_booked():
    parameters = random_parameters(SAMPLED, seed=1)
    simulation = simulate(config(), SAMPLED, parameters)
    admitted = [a for a in simulation.arrivals if a.request_id is not None]
    assert [a.request_id for a in admitted] == list(range(len(admitted)))
    assert admitted[0].index == 0 and [a.index for a in admitted] == sorted(
        a.index for a in admitted
    )
    schedule = simulation.schedule
    for join in schedule.joins:
        assert (
            join.step * simulation.config.step_seconds
            >= simulation.arrivals[admitted[join.request].index].time
        )
    schedule.validate(
        simulation.requests
    )  # canonical, no double booking, attempts never overlap
    assert len(simulation.attempts) == len(schedule.joins)
    assert simulation.tokens == sum(len(tokens) for tokens in simulation.streamed)
    assert 0 < simulation.utilization <= 1


def test_the_tokens_are_reference_prefixes_and_eos_frees_the_slot():
    parameters = random_parameters(SAMPLED, seed=1)
    simulation = simulate(config(steps=24, arrivals=16), SAMPLED, parameters)
    reference = reference_generate(SAMPLED, parameters, simulation.requests)
    check_against_reference(simulation, reference)
    eos = SAMPLED.vocab - 1
    stopped = [a for a in simulation.attempts if a.outcome == EOS]
    assert stopped, "the fixture should stop at least one request at EOS"
    for attempt in stopped:
        request = attempt.join.request
        tokens = simulation.streamed[request]
        assert tokens[-1] == eos and len(tokens) <= simulation.requests[request].max_new
        assert (
            attempt.join.length < simulation.requests[request].max_new
            or len(tokens) == simulation.requests[request].max_new
        )
    for attempt in simulation.attempts:
        if attempt.outcome == COMPLETE:
            request = attempt.join.request
            assert (
                len(simulation.streamed[request])
                == simulation.requests[request].max_new
            )
    assert {a.outcome for a in simulation.attempts} <= {COMPLETE, EOS, FAILED, RUN_END}


def test_a_forced_failure_aborts_the_occupants_and_restarts_them_from_the_prefill():
    parameters = random_parameters(SAMPLED, seed=1)
    simulation = simulate(config(forced_failures=((0, 3),)), SAMPLED, parameters)
    failures = [f for f in simulation.failures if f.aborted]
    assert failures and failures[0].pod == 0 and failures[0].step >= 3
    failed = [a for a in simulation.attempts if a.outcome == FAILED]
    assert failed and simulation.restarts >= len(failed)
    for attempt in failed:
        request = attempt.join.request
        later = [
            a
            for a in simulation.attempts
            if a.join.request == request and a.join.step > attempt.join.step
        ]
        assert later, "an aborted request is re-queued and joins again"
        restart = min(later, key=lambda a: a.join.step)
        assert restart.join.step >= attempt.join.step + attempt.join.length
        # the restart recomputes what was streamed and streams only what follows
        assert restart.streamed == tuple(
            range(len(attempt.streamed), len(attempt.streamed) + len(restart.streamed))
        )
    # the pod is down for `downtime` steps from the failure
    down = [step for step, n in enumerate(simulation.occupied[0]) if n < 0]
    assert failures[0].step in down and failures[0].step + 1 in down
    check_against_reference(
        simulation, reference_generate(SAMPLED, parameters, simulation.requests)
    )


def test_restarts_are_deterministic_recomputation():
    """Two simulations of the same configuration agree token for token, restarts included."""

    parameters = random_parameters(SAMPLED, seed=1)
    first = simulate(config(failure_rate=0.1), SAMPLED, parameters)
    second = simulate(config(failure_rate=0.1), SAMPLED, parameters)
    assert first == second
    assert first.restarts >= 1


def test_configurations_are_validated():
    with pytest.raises(ValueError):
        config(pods=0)
    with pytest.raises(ValueError):
        config(failure_rate=1.0)
    with pytest.raises(ValueError):
        config(prompt_lengths=(3, 2))
    with pytest.raises(ValueError):
        simulate(config(eos=8), SAMPLED, random_parameters(SAMPLED, seed=1))
    with pytest.raises(ValueError):
        simulate(config(), SAMPLED, random_parameters(ARGMAX, seed=1))
