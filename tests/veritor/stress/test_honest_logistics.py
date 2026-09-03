"""H6: fleet logistics at round close (``docs/honest-prover.md``, section 8).

The simulated datacenter (:mod:`veritor.simulation.workload`) is cut into
rounds and driven through the epoch layer (:mod:`veritor.simulation.epochs`):
each round is a window of fleet time, its requests compile into ``ClusterG``
runs (RU = step, the schedule as advice), every run commits its boundary in
its round, the round closes, the challenged runs answer.  Each scenario is a
fault of *logistics*, not of computation -- a crash, a straggler, a lost KV
transfer, a request longer than a round -- and the honest prover declares
nothing: what happened is in the schedule (M4) or in ``x``.  The rows record
what that costs.

The fleet is two pods of two slots over sixteen synchronous steps, cut into
two rounds of eight; the model is the datacenter demo's small shape.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import mean

import pytest

from veritor.constructors import (
    Join,
    LMShape,
    Request,
    Schedule,
    random_parameters,
    reference_generate,
)
from veritor.protocol import Header, Reject, VerificationCode
from veritor.protocol.epoch import (
    EpochParameters,
    EpochProver,
    EpochVerifier,
    RoundChallenge,
)
from veritor.simulation.epochs import (
    CONTINUE,
    ETA,
    HOLD,
    POLICY,
    SPLIT,
    CompiledRound,
    EpochOutcome,
    RoundTrace,
    compile_rounds,
    epoch_from_simulation,
    partition,
    partition_schedule,
    run_rounds,
    windows,
)
from veritor.simulation.workload import (
    Simulation,
    WorkloadConfig,
    check_against_reference,
    simulate,
)
from veritor.stress.rows import Recorder

SHAPE = LMShape(
    vocab=8, d_model=4, heads=2, layers=1, context=16, width=16, sampling=True
)
PARAMETERS = random_parameters(SHAPE, 0)
STEPS = 16
ROUNDS = 2
ROUND_STEPS = STEPS // ROUNDS
LAG = 3
"""The straggler's lateness at round close, in steps."""


def fleet(
    *,
    seed: int = 0,
    arrivals: int = 12,
    steps: int = STEPS,
    pods: int = 2,
    max_new_lengths: tuple[int, int] = (2, 8),
    failure_rate: float = 0.0,
    forced_failures: tuple[tuple[int, int], ...] = (),
    abandon_rate: float = 0.0,
) -> WorkloadConfig:
    return WorkloadConfig(
        pods=pods,
        slots=2,
        steps=steps,
        arrivals=arrivals,
        seed=seed,
        max_new_lengths=max_new_lengths,
        failure_rate=failure_rate,
        forced_failures=forced_failures,
        abandon_rate=abandon_rate,
    )


def accepted(outcome: EpochOutcome) -> None:
    """Every round closed, every run challenged and accepted, nothing declared."""

    assert outcome.accepted, outcome.report.detail
    assert outcome.declarations == 0
    for round_report in outcome.report.rounds:
        assert round_report.closed
        assert not round_report.refused
        for run in round_report.runs:
            assert run.report is not None and run.report.accepted


def record(
    honest: Recorder,
    outcome: EpochOutcome,
    *,
    id: str,
    what: str,
    mechanism: str,
    verdict: str,
    notes: str,
) -> None:
    uncapped = math.ceil(outcome.uncapped_bits())
    if uncapped > outcome.capacity_bits:
        rounds = outcome.parameters.rounds
        summed = f" summed over {rounds} rounds at eta / {rounds}" if rounds > 1 else ""
        notes = (
            f"U capped at |Out| = {outcome.capacity_bits} bits{summed} "
            f"(uncapped {uncapped} bits); {notes}"
        )
    honest.record(
        id=id,
        what=what,
        mechanism=mechanism,
        advice_bits=outcome.advice_bits,
        capacity_bits=outcome.capacity_bits,
        overhead=outcome.overhead,
        description_bytes=outcome.description_bytes,
        verdict=verdict,
        notes=notes,
        declarations=outcome.declarations,
        charge_bits=0,
        recompute=0.0,
        rounds=outcome.parameters.rounds,
        runs=len(outcome.runs),
        outputs=outcome.outputs,
        check_outputs=outcome.check_outputs,
        honest_cost=outcome.honest_cost,
    )


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def seconds(steps: int, config: WorkloadConfig) -> str:
    return f"{plural(steps, 'step')} ({steps * config.step_seconds:.2f} s)"


def in_flight(
    simulation: Simulation, boundary: int, pod: int | None = None
) -> dict[int, int]:
    """Per request with an attempt (on ``pod``, or anywhere) across ``boundary``, the steps it has left."""

    residual: dict[int, int] = {}
    for join in simulation.schedule.joins:
        end = join.step + join.length
        if join.step < boundary < end and pod in (None, join.pod):
            residual[join.request] = max(residual.get(join.request, 0), end - boundary)
    return residual


def streamed_in(trace: RoundTrace, origin: int) -> int:
    """The positions ``origin`` streams in this run."""

    index = trace.origins.index(origin)
    return trace.schedule.active_steps(trace.requests)[index]


# -- (a) crash mid-request --------------------------------------------------------------


def test_h6a_crash_mid_request(honest: Recorder) -> None:
    """Pod 0 dies at step 4 with two requests in flight: they restart on pod 1 -- or their clients give up."""

    calm = simulate(fleet(), SHAPE, PARAMETERS)
    crashed = simulate(fleet(forced_failures=((0, 4),)), SHAPE, PARAMETERS)
    truncated = simulate(
        fleet(forced_failures=((0, 4),), abandon_rate=1.0), SHAPE, PARAMETERS
    )
    assert not calm.failures and calm.restarts == 0
    (failure,) = crashed.failures
    assert failure.step == 4 and len(failure.aborted) == 2 and not failure.abandoned
    assert crashed.restarts == 2
    (giving_up,) = truncated.failures
    assert giving_up.abandoned == failure.aborted and truncated.restarts == 0
    assert truncated.abandoned == 2
    reference = reference_generate(SHAPE, PARAMETERS, truncated.requests)
    check_against_reference(truncated, reference)
    cut_short = [
        (len(truncated.streamed[r]), len(reference[r])) for r in giving_up.abandoned
    ]
    assert all(got < wanted for got, wanted in cut_short)
    # the restarts land inside round 0: nothing spans the boundary in any of the three
    for simulation in (calm, crashed, truncated):
        assert all(not trace.spanning for trace in partition(simulation, ROUNDS))
    restart_joins = [
        join for join in crashed.schedule.joins if join.request in failure.aborted
    ]
    assert len(restart_joins) == 4 and all(j.step < ROUND_STEPS for j in restart_joins)

    base = epoch_from_simulation(calm, SHAPE, PARAMETERS, ROUNDS, seed="H6a/calm")
    restart = epoch_from_simulation(
        crashed, SHAPE, PARAMETERS, ROUNDS, seed="H6a/restart"
    )
    given_up = epoch_from_simulation(
        truncated, SHAPE, PARAMETERS, ROUNDS, seed="H6a/truncated"
    )
    for outcome in (base, restart, given_up):
        accepted(outcome)
        assert outcome.check_outputs == 0
    assert restart.advice_bits > base.advice_bits
    assert restart.honest_cost > base.honest_cost  # the recomputed positions are gates
    assert given_up.outputs == sum(run.trace.tokens for run in given_up.runs)
    assert given_up.outputs < base.outputs  # the abandoned requests' missing tokens

    record(
        honest,
        restart,
        id="H6a",
        what=(
            f"crash mid-request, ClusterG through the epoch layer: pod 0 dies at step {failure.step} of "
            f"round 0 with {len(failure.aborted)} requests in flight; both restart from the prefill on "
            f"pod 1 (Schedule v3 re-join) and finish inside the round; 2 pods x 2 slots, 2 rounds of "
            f"{ROUND_STEPS} steps, one run per round, {len(crashed.requests)} requests"
        ),
        mechanism="M4 (the failed attempt and the re-join are two joins of the schedule)",
        verdict=(
            f"both rounds ACCEPTED, 0 declarations: a crash is not silent, the schedule says where "
            f"the pod stopped; {restart.advice_bits} advice bits vs {base.advice_bits} over the same "
            f"arrivals without the crash (+{restart.advice_bits - base.advice_bits}: "
            f"{crashed.restarts} re-joins, the queue reshuffled); outputs = the {restart.outputs} "
            f"streamed tokens"
        ),
        notes=(
            f"honest replay cost {restart.honest_cost} vs {base.honest_cost} without the crash (the "
            f"recomputed positions are gates); a run whose window holds the crash needs nothing from "
            f"the epoch layer"
        ),
    )
    record(
        honest,
        given_up,
        id="H6at",
        what=(
            f"the same crash, the clients give up: the {truncated.abandoned} requests stay truncated at "
            f"the {sum(got for got, _ in cut_short)} tokens streamed before step {failure.step} "
            f"(of {sum(wanted for _, wanted in cut_short)} wanted); no restart"
        ),
        mechanism="M4 (the join's length is the truncated request's t; no blank check outputs with RU = step)",
        verdict=(
            f"both rounds ACCEPTED, 0 declarations, 0 check outputs: with RU = step the generated "
            f"length is the join's length field, already in the schedule (S7 pays it as its own advice "
            f"and pads with blank check outputs under RU = request); {given_up.advice_bits} advice bits "
            f"vs {restart.advice_bits} with the restarts; {given_up.outputs} outputs vs {base.outputs}"
        ),
        notes=(
            f"honest replay cost {given_up.honest_cost} vs {base.honest_cost} uninterrupted: the "
            f"truncated requests' remaining steps are not in the circuit"
        ),
    )


# -- (b) straggler at round close ---------------------------------------------------------


def per_pod(
    simulation: Simulation, spanning: str, shifts: Sequence[int]
) -> list[RoundTrace]:
    """One run per pod per round; pod ``p``'s window boundaries pulled ``shifts[p]`` steps earlier."""

    traces: list[RoundTrace] = []
    for pod, shift in enumerate(shifts):
        traces.extend(
            partition(simulation, windows(STEPS, ROUNDS, shift), spanning, pods=[pod])
        )
    return traces


def test_h6b_straggler_at_round_close(honest: Recorder) -> None:
    """Pod 1's boundary values for the last LAG steps of a round are not in hand when the verifier wants to close."""

    config = fleet()
    simulation = simulate(config, SHAPE, PARAMETERS)
    assert not simulation.failures

    # (i) wait: the round closes LAG steps late, the runs are what they would have been
    wait = epoch_from_simulation(simulation, SHAPE, PARAMETERS, ROUNDS, seed="H6b/wait")
    accepted(wait)

    # per-pod runs, nobody late: the reference for (ii) and (iii)
    punctual = per_pod(simulation, HOLD, (0, 0))
    assert [trace.round for trace in punctual] == [0, 1, 0, 1]
    on_time = run_rounds(punctual, SHAPE, PARAMETERS, rounds=ROUNDS, seed="H6b/pods")
    accepted(on_time)

    # (ii) defer: pod 1's runs are admitted one round late; the epoch needs a trailing round
    deferred = run_rounds(
        punctual,
        SHAPE,
        PARAMETERS,
        rounds=ROUNDS + 1,
        admission=[0, 1, 1, 2],
        seed="H6b/defer",
    )
    accepted(deferred)
    assert [len(round.runs) for round in deferred.report.rounds] == [1, 2, 1]
    assert deferred.advice_bits == on_time.advice_bits

    # (iii) truncate: pod 1 commits what it has LAG steps before each close; the requests
    # cut at the shifted boundary continue next round as new requests with the prefix in x
    cut = per_pod(simulation, CONTINUE, (0, LAG))
    boundary = ROUND_STEPS - LAG
    late = in_flight(simulation, boundary, pod=1)
    assert late, "pod 1 has a request across the shifted boundary"
    pod1 = [trace for trace in cut if trace.pods == (1,)]
    assert pod1[0].window.end == boundary and pod1[1].window.start == boundary
    continuations = sorted(o for o in pod1[1].origins if o in pod1[0].origins)
    assert continuations == sorted(late)
    truncated = run_rounds(cut, SHAPE, PARAMETERS, rounds=ROUNDS, seed="H6b/truncate")
    accepted(truncated)
    assert truncated.outputs == on_time.outputs  # nothing streamed twice, nothing lost
    prefix = sum(streamed_in(pod1[0], origin) for origin in continuations)
    assert prefix == sum(
        len(pod1[1].requests[pod1[1].origins.index(origin)].prompt)
        - len(simulation.requests[origin].prompt)
        for origin in continuations
    )

    record(
        honest,
        wait,
        id="H6bw",
        what=(
            f"straggler, wait: pod 1's boundary values for the last {LAG} steps of each round arrive "
            f"{LAG} steps after the verifier wanted to close; the verifier waits; one run per round for "
            f"the fleet, as H6a"
        ),
        mechanism="none (the close is the verifier's move; the runs do not change)",
        verdict=(
            f"both rounds ACCEPTED, 0 declarations; the runs are byte-identical to the fleet without a "
            f"straggler ({wait.advice_bits} advice bits); the cost is {seconds(LAG, config)} of delay "
            f"per close on the challenges and on the next round's opening, nothing on U"
        ),
        notes=(
            "EpochVerifier today: a run admitted but without a boundary at close_round is recorded "
            "INVALID_PHASE ('the boundary never arrived before the round closed'), its late boundary "
            "is refused, and the epoch's verdict is that run's -- so an admitted straggler fails the "
            "epoch; the honest prover admits a run only with its boundary in hand (test_h6b_verifier_today)"
        ),
    )
    record(
        honest,
        deferred,
        id="H6bd",
        what=(
            f"straggler, defer: one run per pod per round; pod 1's run of round r is admitted in round "
            f"r + 1 with the boundary it then has; the epoch closes {ROUNDS} rounds on time and needs a "
            f"trailing round {ROUNDS} for pod 1's last run"
        ),
        mechanism="none (admission is the prover's move; the run is the same, admitted later)",
        verdict=(
            f"all {ROUNDS + 1} rounds ACCEPTED, 0 declarations; runs per round "
            f"{[len(round.runs) for round in deferred.report.rounds]}; the same {deferred.advice_bits} "
            f"advice bits as the per-pod fleet on time ({on_time.advice_bits}); pod 1's tokens are "
            f"committed one round ({seconds(ROUND_STEPS, config)}) late, and each round is bounded at "
            f"eta / {ROUNDS + 1} instead of eta / {ROUNDS} (U {deferred.capacity_bits} vs "
            f"{on_time.capacity_bits} bits, both capped at |Out|)"
        ),
        notes=(
            f"per-pod runs take {on_time.advice_bits} advice bits against the fleet run's "
            f"{wait.advice_bits}: a schedule header per run, no pod field per join; a request "
            f"restarted on the other pod would be in both pods' runs (its recomputed prefix output twice)"
        ),
    )
    record(
        honest,
        truncated,
        id="H6bt",
        what=(
            f"straggler, truncate: one run per pod per round; pod 1's round-0 run ends {LAG} steps "
            f"early (step {boundary}) with the values it has, and its {len(continuations)} request(s) "
            f"across step {boundary} continue in round 1 as new requests whose prompt is the original "
            f"prompt plus the {prefix} tokens already streamed"
        ),
        mechanism="M4 + x (the continuation is a request of the next run; its prefix is public input)",
        verdict=(
            f"both rounds ACCEPTED on time, 0 declarations, 0 check outputs; {truncated.outputs} "
            f"outputs = the fleet's ({on_time.outputs}), nothing streamed twice; "
            f"{truncated.advice_bits} advice bits vs {on_time.advice_bits} for the per-pod fleet "
            f"(+{truncated.advice_bits - on_time.advice_bits}: the continuation joins, and the cut "
            f"run's shorter fields); the {prefix} prefix tokens enter round 1's x as prompt"
        ),
        notes=(
            f"honest replay cost {truncated.honest_cost} vs {on_time.honest_cost}: the continuations' "
            f"prefills re-read the {prefix} prefix tokens the earlier run computed (M1 recompute of the "
            f"prefill, not of the decodes); the verifier can check a continuation's prompt against "
            f"round 0's claimed outputs but no rule does so today (section 8 gap list)"
        ),
    )


def test_h6b_verifier_today() -> None:
    """What ``EpochVerifier`` does with an admitted run whose boundary misses the close."""

    simulation = simulate(fleet(), SHAPE, PARAMETERS)
    traces = per_pod(simulation, HOLD, (0, 0))
    compiled = compile_rounds(traces, SHAPE, PARAMETERS, seed="H6b/today")
    punctual, late = compiled[0], compiled[2]  # pod 0 and pod 1, round 0
    assert punctual.trace.round == late.trace.round == 0
    parameters = EpochParameters(
        ETA,
        POLICY,
        max_capacity=None,
        rounds=ROUNDS,
        max_advice_bits=max(run.advice_bits for run in compiled),
    )
    verifier = EpochVerifier(parameters)
    prover = EpochProver()
    seeds = [bytes([round]) * 32 for round in range(ROUNDS)]

    def admit(run: CompiledRound, session_id: bytes) -> Header:
        return verifier.admit(
            run.compilation,
            run.outputs,
            weights=run.run.weights,
            session_id=session_id.ljust(16, b"-"),
        )

    def send_boundary(run: CompiledRound, header: Header) -> None:
        verifier.receive_boundary(
            header,
            prover.boundary(
                run.compilation.compiled,
                header,
                run.values,
                weight_tree=run.run.weight_tree,
            ),
        )

    def answer(challenge: RoundChallenge) -> None:
        for header, replay in challenge.challenges:
            sample = verifier.receive_interiors(
                header, prover.interiors(header, replay)
            )
            verifier.receive_evidence(header, prover.evidence(header, sample))

    on_time = admit(punctual, b"pod0/round0")
    straggler = admit(late, b"pod1/round0")
    send_boundary(punctual, on_time)
    # the verifier closes without pod 1's boundary
    challenge = verifier.close_round(seeds[0])
    assert [header for header, _ in challenge.challenges] == [on_time]
    report = verifier.report()
    verdicts = {run.header: run.report for run in report.rounds[0].runs}
    assert verdicts[on_time] is None  # challenged, not yet answered
    missed = verdicts[straggler]
    assert missed is not None and missed.code is VerificationCode.INVALID_PHASE
    assert missed.detail == "the boundary never arrived before the round closed"
    # the boundary arrives LAG steps later: refused, the run stays rejected
    with pytest.raises(Reject) as refused:
        send_boundary(late, straggler)
    assert refused.value.code is VerificationCode.INVALID_PHASE
    assert refused.value.detail == "the run was admitted in round 0, which is closed"
    answer(challenge)
    # round 1: the prover re-admits the same compilation under a fresh session id, boundary in hand
    again = admit(late, b"pod1/round0/bis")
    send_boundary(late, again)
    answer(verifier.close_round(seeds[1]))
    final = verifier.report()
    redone = {run.header: run.report for run in final.rounds[1].runs}[again]
    assert redone is not None and redone.accepted
    assert not final.accepted and final.code is VerificationCode.INVALID_PHASE
    assert "the boundary never arrived before the round closed" in final.detail
    assert (
        final.rounds[0].capacity_bits > 0
    )  # the straggler's table counted in round 0's Bound


# -- (c) network partition: the KV transfer of a disaggregated request is lost ----------------


def test_h6c_disaggregated_transfer_lost(honest: Recorder) -> None:
    """Prefill on pod 0, decode on pod 1 (S6 cross-pod port); the transfer fails and pod 1 re-prefills."""

    rng = random.Random("H6c")

    def request(prompt: tuple[int, ...], max_new: int) -> Request:
        return Request(
            prompt,
            max_new,
            tuple(rng.randrange(1 << SHAPE.random_bits) for _ in range(max_new)),
        )

    requests = (request((1, 2, 3), 5), request((4, 5), 3))
    steps = 8
    transferred = Schedule(
        2,
        2,
        steps,
        (
            Join(
                0, 0, 0, 0, 1
            ),  # prefill on pod 0: the first token, the KV of 3 positions
            Join(1, 0, 1, 1, 3),  # an unrelated request on pod 1
            Join(1, 1, 0, 0, 4, resume=True),  # decode on pod 1 from the transferred KV
        ),
    )
    lost = Schedule(
        2,
        2,
        steps,
        (
            Join(0, 0, 0, 0, 1),  # the same prefill; its KV never arrives
            Join(1, 0, 1, 1, 3),
            Join(
                1, 2, 0, 0, 5
            ),  # a fresh attempt on pod 1: re-prefill, then the decodes
        ),
    )
    reference = reference_generate(SHAPE, PARAMETERS, requests)
    outcomes: dict[str, EpochOutcome] = {}
    for label, schedule in (("transferred", transferred), ("lost", lost)):
        schedule.validate(requests)
        active = schedule.active_steps(requests)
        streamed = tuple(reference[r][: active[r]] for r in range(len(requests)))
        assert [len(tokens) for tokens in streamed] == [5, 3]
        (trace,) = partition_schedule(requests, schedule, streamed, 1)
        assert not trace.spanning
        outcomes[label] = run_rounds(
            [trace], SHAPE, PARAMETERS, rounds=1, seed=f"H6c/{label}"
        )
        accepted(outcomes[label])
    before, after = outcomes["transferred"], outcomes["lost"]
    assert before.outputs == after.outputs == 8
    assert (
        after.advice_bits == before.advice_bits
    )  # a fresh join costs what a resume did
    assert after.honest_cost > before.honest_cost  # the re-prefill is gates
    assert after.check_outputs == before.check_outputs == 0
    recompute = (after.honest_cost - before.honest_cost) / before.honest_cost
    record(
        honest,
        after,
        id="H6c",
        what=(
            "KV transfer lost, ClusterG: request 0 prefilled on pod 0 (3 prompt positions, first "
            "token streamed), its KV bound for pod 1 never arrives; pod 1 re-prefills it as a fresh "
            "attempt one step later and decodes the remaining 4 positions; one round, 2 pods x 2 slots, "
            "8 steps, an unrelated request alongside"
        ),
        mechanism="M4 (a fresh join in place of the resumed one) + M1 (the re-prefill recomputes)",
        verdict=(
            f"ACCEPTED, 0 declarations, 0 check outputs; outputs = the same {after.outputs} tokens "
            f"as the disaggregated run whose transfer succeeded (position 0 recomputed, not re-streamed); "
            f"{after.advice_bits} advice bits, exactly the successful transfer's ({before.advice_bits}): "
            f"the resume flag is a bit either way; honest replay cost {after.honest_cost} vs "
            f"{before.honest_cost} (+{recompute:.1%}, one prefill of 3 positions)"
        ),
        notes=(
            "the circuit holds the orphan prefill step (its KV declared, read by nobody) and the "
            "second prefill; nothing distinguishes a lost transfer from a pod that died after its "
            "first step (H6a); with per-pod runs the successful transfer itself is a cross-run read "
            "(the resume would name a KV row of another run's boundary) and is not expressible today"
        ),
    )


# -- (d) a request longer than a round -------------------------------------------------------


def test_h6d_request_longer_than_a_round(honest: Recorder) -> None:
    """One request's generation crosses the round boundary: hold, split, or continue."""

    config = fleet(seed=4, arrivals=10, max_new_lengths=(3, 6))
    simulation = simulate(config, SHAPE, PARAMETERS)
    assert not simulation.failures
    hold_traces = partition(simulation, ROUNDS, HOLD)
    spanning = hold_traces[-1].spanning
    assert len(spanning) == 1, spanning
    (spanner,) = spanning
    (join,) = [j for j in simulation.schedule.joins if j.request == spanner]
    before = ROUND_STEPS - join.step  # positions streamed in round 0
    after = join.step + join.length - ROUND_STEPS  # positions streamed in round 1
    assert (
        before >= 1 and after >= 1 and len(simulation.streamed[spanner]) == join.length
    )
    assert hold_traces[1].base == join.step and hold_traces[0].window.end == ROUND_STEPS

    hold = run_rounds(hold_traces, SHAPE, PARAMETERS, rounds=ROUNDS, seed="H6d/hold")
    split = epoch_from_simulation(
        simulation, SHAPE, PARAMETERS, ROUNDS, spanning=SPLIT, seed="H6d/split"
    )
    cont = epoch_from_simulation(
        simulation, SHAPE, PARAMETERS, ROUNDS, spanning=CONTINUE, seed="H6d/continue"
    )
    for outcome in (hold, split, cont):
        accepted(outcome)
        assert outcome.check_outputs == 0
    total = simulation.tokens
    assert hold.outputs == total and cont.outputs == total
    assert split.outputs == total + before  # the prefix is output again by the re-join
    assert (
        split.runs[1].trace.schedule.slots == config.slots + 1
    )  # the re-join's own slot
    continuation = cont.runs[1].trace
    prompt = len(simulation.requests[spanner].prompt)
    index = continuation.origins.index(spanner)
    assert len(continuation.requests[index].prompt) == prompt + before
    assert cont.honest_cost > hold.honest_cost  # the longer prefill
    assert split.honest_cost > cont.honest_cost  # the recomputed decode steps

    # the round-close delay hold implies, over a long run of the same workload
    long_config = fleet(seed=4, arrivals=1200, steps=800, max_new_lengths=(3, 6))
    long_run = simulate(long_config, SHAPE, PARAMETERS)
    boundaries = range(ROUND_STEPS, long_config.steps, ROUND_STEPS)
    delays = [max(in_flight(long_run, b).values(), default=0) for b in boundaries]
    crossing = [len(in_flight(long_run, b)) for b in boundaries]
    assert len(delays) == long_config.steps // ROUND_STEPS - 1
    busy = sum(1 for d in delays if d > 0) / len(delays)

    record(
        honest,
        hold,
        id="H6dh",
        what=(
            f"request longer than a round, hold: request {spanner} joins at step {join.step} for "
            f"{join.length} steps, crossing the boundary at step {ROUND_STEPS} ({before} tokens "
            f"before, {after} after); its whole attempt goes into round 1's run, whose window opens at "
            f"step {join.step}; {len(simulation.requests)} requests, lengths 3-6 on 2 pods x 2 slots"
        ),
        mechanism="none (the run is compiled when the request completes; the schedule is unchanged)",
        verdict=(
            f"both rounds ACCEPTED, 0 declarations; {hold.outputs} outputs = the streamed tokens, "
            f"{hold.advice_bits} advice bits (the baseline for H6ds, H6dc); the {before} tokens streamed "
            f"in round 0 are committed one round late, or round 0's close waits {seconds(after, config)}"
        ),
        notes=(
            f"over {len(delays)} closes of an 800-step run of this workload, {busy:.0%} of the "
            f"boundaries have a request in flight ({mean(crossing):.1f} on average of "
            f"{config.pods * config.slots} slots); holding the close costs {mean(delays):.1f} steps on "
            f"average and {max(delays)} at most ({max(delays) / ROUND_STEPS:.0%} of a round) for lengths "
            f"uniform on 3-6; the cross-run read would make this 0"
        ),
    )
    record(
        honest,
        split,
        id="H6ds",
        what=(
            f"request longer than a round, split: round 0 commits the {before} positions streamed by "
            f"step {ROUND_STEPS}; in round 1 the request re-joins as a fresh attempt in a slot of its "
            f"own, recomputing the {before} positions one per step (a restart's semantics) before "
            f"{plural(after, 'new one')}"
        ),
        mechanism="M4 (a second join) + M1 (the prefix recomputed)",
        verdict=(
            f"both rounds ACCEPTED, 0 declarations; {split.outputs} outputs: the prefix is output "
            f"twice ({before} tokens, in both runs), since Schedule has no field for positions streamed "
            f"in another run; {split.advice_bits} advice bits (+{split.advice_bits - hold.advice_bits} "
            f"over hold: the re-join and a third slot); honest replay cost {split.honest_cost} vs "
            f"{hold.honest_cost} (+{(split.honest_cost - hold.honest_cost) / hold.honest_cost:.1%})"
        ),
        notes=(
            f"the re-join needs {join.length} steps of slot time in round 1 where the original attempt "
            f"had {after} left, so it does not fit an existing slot (the cluster gets one more); a real "
            f"server would re-prefill prompt + prefix in one step, which is the continuation's circuit "
            f"(H6dc) with the original request in x: a join shape Schedule lacks"
        ),
    )
    record(
        honest,
        cont,
        id="H6dc",
        what=(
            f"request longer than a round, continue: round 0 commits the {before} positions; round 1 "
            f"holds a new request whose prompt is the original prompt plus those {before} tokens, "
            f"prefilled in one step in the original slot, then {plural(after, 'more decode step')}"
        ),
        mechanism="M4 + x (the prefix is public input of the next run)",
        verdict=(
            f"both rounds ACCEPTED, 0 declarations, 0 check outputs; {cont.outputs} outputs = the "
            f"streamed tokens, nothing twice; {cont.advice_bits} advice bits "
            f"(+{cont.advice_bits - hold.advice_bits} over hold), "
            f"{cont.description_bytes - hold.description_bytes} description bytes for the {before} "
            f"prefix tokens and the longer prefill; honest replay cost {cont.honest_cost} vs "
            f"{hold.honest_cost} (+{(cont.honest_cost - hold.honest_cost) / hold.honest_cost:.1%}: one "
            f"prefill over {prompt + before} instead of {prompt} positions)"
        ),
        notes=(
            "the verifier sees two requests where the client saw one; that the continuation's prompt "
            "ends in round 0's claimed outputs for the same request is checkable from the two runs' "
            "public claims and is not checked by any rule today; the cross-run read (a resume join "
            "naming round 0's KV rows through the stream) would cost the resume join's bits and no "
            "recompute, and needs the four pieces listed in docs/honest-prover.md section 8"
        ),
    )


# -- (e) fleet churn -------------------------------------------------------------------------


def test_h6e_fleet_churn(honest: Recorder) -> None:
    """A day's pod failures compressed into a few rounds: restarts are schedule, never declarations."""

    steps, rounds = 48, 6
    churning = fleet(
        seed=7,
        pods=3,
        arrivals=60,
        steps=steps,
        max_new_lengths=(2, 5),
        failure_rate=0.02,
    )
    calm = fleet(seed=7, pods=3, arrivals=60, steps=steps, max_new_lengths=(2, 5))
    simulation = simulate(churning, SHAPE, PARAMETERS)
    failures = [f for f in simulation.failures if f.aborted]
    assert failures and simulation.restarts >= len(failures)
    traces = partition(simulation, rounds, CONTINUE)
    disturbed = {
        trace.round for trace in traces for f in failures if trace.window.holds(f.step)
    }
    outcome = run_rounds(traces, SHAPE, PARAMETERS, rounds=rounds, seed="H6e")
    accepted(outcome)
    baseline = run_rounds(
        partition(simulate(calm, SHAPE, PARAMETERS), rounds, CONTINUE),
        SHAPE,
        PARAMETERS,
        rounds=rounds,
        seed="H6e/calm",
    )
    accepted(baseline)
    assert outcome.outputs == sum(trace.tokens for trace in traces)

    # the literature's rate: Llama 3 405B, 419 unexpected interruptions in 54 days on 16,384 GPUs
    per_gpu_day = 419 / (54 * 16_384)
    per_pod_day = 8 * per_gpu_day  # an eight-GPU node
    fleet_pods = 1_000
    rounds_per_day = 24 * 60  # one-minute rounds
    per_round = fleet_pods * per_pod_day / rounds_per_day
    extra = outcome.advice_bits - baseline.advice_bits
    record(
        honest,
        outcome,
        id="H6e",
        what=(
            f"fleet churn, ClusterG through {rounds} rounds of {steps // rounds} steps on "
            f"{churning.pods} pods x 2 slots: pods fail at {churning.failure_rate:.0%} per step, "
            f"{len(simulation.failures)} failures ({len(failures)} with occupants), "
            f"{simulation.restarts} restarts over {len(simulation.requests)} requests; spanning "
            f"requests continue (H6dc)"
        ),
        mechanism="M4 (every restart is a join; every crossing is a continuation in x)",
        verdict=(
            f"all {rounds} rounds ACCEPTED, 0 declarations; {len(disturbed)} of {rounds} rounds hold a "
            f"restart; {outcome.advice_bits} advice bits vs {baseline.advice_bits} for the same arrivals "
            f"without failures (+{extra}, {extra / max(simulation.restarts, 1):.0f} per restart); "
            f"honest replay cost {outcome.honest_cost} vs {baseline.honest_cost}"
        ),
        notes=(
            f"at the Llama 3 405B rate (419 unexpected interruptions in 54 days on 16,384 H100s: "
            f"{per_gpu_day:.1e} per GPU-day, {per_pod_day:.1e} per 8-GPU pod-day) a {fleet_pods:,}-pod "
            f"fleet sees {fleet_pods * per_pod_day:.1f} pod failures a day, {per_round:.1e} per one-minute "
            f"round: {1 - math.exp(-per_round):.2%} of rounds hold a restart, each a few joins of advice "
            f"and no declaration; the simulation's rate is inflated to see several in {rounds} rounds"
        ),
    )
