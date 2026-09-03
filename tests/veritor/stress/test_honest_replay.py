"""H2, H3: the honest prover's declarations by fault class and at fleet scale.

The prover of these rows records only what :class:`RecordingPolicy` says,
reconstructs every opened RU with :func:`replay_pinned` and declares the VUs
the replay pins (``docs/honest-prover.md``, sections 3 and 5).  ``H2*`` runs
one fault of each class through the protocol with RU = request
(``RequestsG``) and RU = step (``ClusterG``), under tokens-only (``BOUNDARY``)
and every-VU-output (``VU_OUTPUTS``) recording; ``H3*`` draws random faults
over a small fleet of runs through the epoch layer and prices realistic
densities analytically.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

from veritor.analysis.bound import bound
from veritor.analysis.faults import unit_fault_bits
from veritor.constructors import ClusterG, Request, RequestsG, schedule_fcfs
from veritor.core import Compiled, VerificationPolicy
from veritor.core.indexed import iter_members
from veritor.evaluation.global_estimate import Inputs, estimate
from veritor.evaluation.serving import serving_table
from veritor.protocol import (
    Declare,
    ProtocolRun,
    Replay,
    VerificationCode,
    VerifierParameters,
    commit_weights,
    make_expectation,
    replay_unit,
    run_protocol,
    self_check,
)
from veritor.protocol.epoch import EpochParameters, EpochReport, Run, run_epoch
from veritor.simulation.faults import (
    SDC_RATE_PER_DEVICE_HOUR,
    FaultInjector,
    dot_units,
    fault_budget,
)
from veritor.simulation.honest import (
    Capacity,
    Production,
    RecordingPolicy,
    Strategy,
    account,
    boundary_at_rest,
    catastrophic,
    fold_capacity,
    honest_replay,
    input_read,
    interior_flip,
    phase_boundary,
    phase_diagram,
    pin_everything,
    post_j_charge_bits,
    rate_capacity,
    record,
    recorded_addresses,
    replay_pinned,
    token_flip,
    vu_output_read,
    weight_read,
)
from veritor.stress.measure import ETA, POLICY, Measurement, compile_scenario, price
from veritor.stress.models import SHAPE, Model
from veritor.stress.rows import Recorder

REQUESTS = (Request((1, 2, 3), 3), Request((5,), 4))
"""Two requests, ten positions: small enough that every row runs the protocol."""
PODS, SLOTS, STEPS = 1, 2, 4
OPEN_ALL = VerificationPolicy(Fraction(1), Fraction(1, 8))
"""Every RU opened -- the declarations are the run's whole count -- at the catalogue's ``s``."""


# -- workloads ------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """One served workload: its compilation, weights, honest run and fault injector."""

    name: str
    unit_name: str
    measurement: Measurement
    model: Model
    injector: FaultInjector
    unit: int
    """The RU the H2 faults land in."""
    producer: int | None = None
    """For ``ClusterG``: the step whose KV the boundary-at-rest fault corrupts on its way to ``unit``."""

    @property
    def compiled(self) -> Compiled:
        return self.measurement.compiled

    def run(
        self,
        production: Production | None,
        *,
        policy: VerificationPolicy,
        max_faults: int,
        replay: Replay | None,
        declare: Declare | None,
        values: dict[int, object] | None = None,
        label: str,
    ) -> ProtocolRun:
        """Run the protocol on ``production`` (the honest run when ``None``) from the prover's ``values``."""

        honest = self.injector.honest
        outputs = tuple(
            (honest if production is None else production.values)[a]
            for a in self.compiled.circuit.outputs
        )
        seed = hashlib.sha256(f"veritor/stress/honest/{label}".encode()).digest()
        kappa, tree = commit_weights(self.model.gate_set, self.model.weights)
        expectation = make_expectation(
            self.measurement.compilation,
            policy,
            outputs,
            parameters=VerifierParameters(
                ETA,
                max_capacity=1 << 20,
                max_advice_bits=self.measurement.advice_bits,
                max_faults=max_faults,
            ),
            weights=kappa,
            session_id=seed[:16],
            q_seed=seed,
            s_seed=bytes(reversed(seed)),
        )
        return run_protocol(
            self.compiled,
            expectation,
            (honest if production is None else production.values)
            if values is None
            else values,
            replay=replay,
            weight_tree=tree,
            declare=declare,
        )


def _workload(
    name: str,
    unit_name: str,
    model: Model,
    constructor: object,
    advice: bytes,
    unit: int,
    producer: int | None,
) -> Workload:
    measurement = compile_scenario(constructor, REQUESTS, advice, model.gate_set)  # type: ignore[arg-type]
    injector = FaultInjector(
        measurement.compiled, measurement.compilation.inputs, model.weights
    )
    return Workload(name, unit_name, measurement, model, injector, unit, producer)


@pytest.fixture(scope="module")
def per_request(model: Model) -> Workload:
    """RU = request: RU 0 holds the weights, RU 1 the first request (six positions)."""

    return _workload("RequestsG", "request", model, RequestsG(SHAPE), b"", 1, None)


@pytest.fixture(scope="module")
def per_step(model: Model) -> Workload:
    """RU = step: one pod of two slots for four steps; RU 1 is step 0 (both prefills), RU 2 step 1."""

    schedule = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    return _workload(
        "ClusterG",
        "step",
        model,
        ClusterG(SHAPE, PODS, SLOTS, STEPS),
        schedule.encode(),
        2,
        1,
    )


# -- the recording and the pinned replay ------------------------------------------------


def test_boundary_recording_is_tokens_only_per_request_and_kv_per_step(
    per_request: Workload, per_step: Workload
) -> None:
    for workload in (per_request, per_step):
        compiled = workload.compiled
        recorded = set(recorded_addresses(compiled, RecordingPolicy.BOUNDARY))
        sources = set(compiled.circuit.inputs) | set(compiled.circuit.weights)
        computed = recorded - sources
        assert computed == {
            a
            for unit in range(compiled.index.replay_units.count)
            for a in compiled.circuit.Out(compiled.index.replay_units.unit(unit))
        }
        assert set(compiled.circuit.outputs) <= computed
        everything = set(recorded_addresses(compiled, RecordingPolicy.VU_OUTPUTS))
        assert recorded < everything
        assert everything - recorded == {
            a
            for unit in range(compiled.index.replay_units.count)
            for a in iter_members(compiled.index.interior(unit))
        }
    # RU = request: the recorded computed values are the streamed tokens and nothing else.
    compiled = per_request.compiled
    computed = (
        set(recorded_addresses(compiled, RecordingPolicy.BOUNDARY))
        - set(compiled.circuit.inputs)
        - set(compiled.circuit.weights)
    )
    assert computed == set(compiled.circuit.outputs)
    # RU = step: the KV words that cross steps are recorded too.
    compiled = per_step.compiled
    computed = (
        set(recorded_addresses(compiled, RecordingPolicy.BOUNDARY))
        - set(compiled.circuit.inputs)
        - set(compiled.circuit.weights)
    )
    assert set(compiled.circuit.outputs) < computed


def test_pinned_replay_of_a_fault_free_run_is_the_honest_replay(
    per_request: Workload,
) -> None:
    compiled = per_request.compiled
    honest = per_request.injector.honest
    for policy in RecordingPolicy:
        recorded = record(compiled, honest, policy)
        for unit in range(compiled.index.replay_units.count):
            interior, pinned = replay_pinned(compiled, unit, recorded)
            assert pinned == ()
            assert interior == replay_unit(compiled, unit, honest)


def test_honest_replay_runs_the_protocol_from_the_recording_alone(
    per_request: Workload,
) -> None:
    """``ProverSession`` works from the recorded mapping: boundary, weights, every opened interior."""

    compiled = per_request.compiled
    recorded = record(compiled, per_request.injector.honest, RecordingPolicy.BOUNDARY)
    assert len(recorded) < compiled.circuit.n // 10
    server = honest_replay(compiled, recorded)
    run = per_request.run(
        None,
        policy=OPEN_ALL,
        max_faults=0,
        replay=server.replay,
        declare=server.declare,
        values=recorded,
        label="recording-only",
    )
    assert run.report.accepted, run.report
    assert server.pinned == {
        unit: () for unit in range(compiled.index.replay_units.count)
    }


# -- H2: one fault of each class ---------------------------------------------------------

CLASSES: tuple[tuple[str, str, Callable[[Workload], Production]], ...] = (
    ("a", "interior low bit", lambda w: interior_flip(w.injector, w.unit, bit=0)),
    (
        "b",
        "interior high bit",
        lambda w: interior_flip(w.injector, w.unit, bit=SHAPE.width - 1),
    ),
    ("c", "token flip", lambda w: token_flip(w.injector, w.unit)),
    ("d", "catastrophic", lambda w: catastrophic(w.injector, w.unit)),
    ("e", "weight read", lambda w: weight_read(w.injector)),
    ("f", "input read", lambda w: input_read(w.injector)),
    ("g", "VU-output read", lambda w: vu_output_read(w.injector, w.unit)),
    (
        "h",
        "boundary at rest",
        lambda w: boundary_at_rest(w.injector, w.producer, w.unit),  # type: ignore[arg-type]
    ),
)


def _productions(workload: Workload) -> list[tuple[str, str, Production]]:
    return [
        (letter, label, make(workload))
        for letter, label, make in CLASSES
        if letter != "h" or workload.producer is not None
    ]


def test_declarations_are_what_self_check_finds_over_the_committed_interior(
    per_request: Workload, per_step: Workload
) -> None:
    """``HonestReplay.declare`` agrees with ``self_check`` over the overlay the session commits."""

    for workload in (per_request, per_step):
        compiled = workload.compiled
        for _letter, _label, production in _productions(workload):
            for policy in RecordingPolicy:
                recorded = record(compiled, production.values, policy)
                server = honest_replay(compiled, recorded)
                for unit in range(compiled.index.replay_units.count):
                    interior = server.replay(unit, recorded)
                    overlay = {**recorded, **interior}
                    assert tuple(server.declare(unit, overlay)) == self_check(
                        compiled, unit, overlay
                    )
                    # the recorded values are what the interior commits
                    assert all(
                        interior[a] == recorded[a] for a in interior if a in recorded
                    )


def _tokens(workload: Workload, production: Production) -> str:
    changed = [
        i
        for i, (a, b) in enumerate(
            zip(production.outputs, production.honest_outputs, strict=True)
        )
        if a != b
    ]
    return f"{production.changed_outputs} of {len(production.outputs)} streamed tokens changed (positions {changed})"


def _row_for_class(
    honest: Recorder,
    workload: Workload,
    letter: str,
    label: str,
    production: Production,
    policy: RecordingPolicy,
    pins: dict[int, tuple[int, ...]],
    contrast: dict[int, tuple[int, ...]] | None,
    priced: object,
) -> None:
    compiled = workload.compiled
    recorded = record(compiled, production.values, policy)
    server = honest_replay(compiled, recorded)
    count = sum(len(p) for p in pins.values())
    accepted = workload.run(
        production,
        policy=OPEN_ALL,
        max_faults=count,
        replay=server.replay,
        declare=server.declare,
        values=recorded,
        label=f"h2{letter}/{workload.name}/{policy.name}",
    )
    assert accepted.report.accepted, (label, workload.name, policy, accepted.report)
    assert accepted.transcript is not None
    declared = accepted.transcript.interiors.declarations
    assert declared == tuple(sorted(v for p in pins.values() for v in p))
    verdict = f"ACCEPTED at theta = (1, 1/8) with {count} declaration{'s' if count != 1 else ''}"
    if count:
        short = workload.run(
            production,
            policy=OPEN_ALL,
            max_faults=count - 1,
            replay=server.replay,
            declare=server.declare,
            values=recorded,
            label=f"h2{letter}/{workload.name}/{policy.name}/short",
        )
        assert short.report.code is VerificationCode.FAULTS_EXCEEDED, short.report
        verdict += f"; FAULTS_EXCEEDED with f_max = {count - 1}"
    every = set(range(compiled.index.replay_units.count))
    p1 = account(Strategy.P1, compiled, POLICY, ETA, pins, every)
    p3 = account(Strategy.P3, compiled, POLICY, ETA, pins, every)
    where = {unit: len(p) for unit, p in pins.items() if p}
    site = (
        f"{production.fault.value}: word {production.address} of RU {production.replay_unit} "
        f"({compiled.circuit[production.address].op}, correct {production.correct:#x}, "
        f"{'read as' if production.misreaders else 'stored as'} {production.corrupted:#x}"
        + (f" by {len(production.misreaders)} gates" if production.misreaders else "")
        + ")"
    )
    notes = (
        f"{site}; {_tokens(workload, production)}; pinned VUs by RU {where or 'none'}. "
        f"Post-J price at theta = (1/2, 1/8): {p1.charge_bits:.1f} bits for {count} declarations "
        f"(u(1) = {unit_fault_bits(compiled):.1f}); P3 (100% replay, pre-J) would pay {p3.charge_bits:.1f} bits."
    )
    if contrast is not None:
        other = sum(len(p) for p in contrast.values())
        notes += f" Under {'BOUNDARY' if policy is RecordingPolicy.VU_OUTPUTS else 'VU_OUTPUTS'} recording: {other}."
    suffix = "r" if workload.unit_name == "request" else "s"
    honest.record(
        id=f"H2{letter}{suffix}{'v' if policy is RecordingPolicy.VU_OUTPUTS else ''}",
        what=(
            f"{label}, RU = {workload.unit_name} ({workload.name}), recording {policy.name}, P1: "
            f"{production.fault.value}"
        ),
        mechanism="M6",
        advice_bits=workload.measurement.advice_bits,
        capacity_bits=math.ceil(bound(compiled, POLICY, ETA, max_faults=count).bits)
        + workload.measurement.advice_bits,
        overhead=priced.overhead,  # type: ignore[attr-defined]
        description_bytes=workload.measurement.description_bytes,
        verdict=verdict,
        notes=notes,
        declarations=count,
        charge_bits=round(p1.charge_bits),
        recompute=float(p1.recompute),
        fault_class=production.fault.name,
        recording=policy.name,
        replay_unit=workload.unit_name,
        changed_tokens=production.changed_outputs,
    )


@pytest.mark.parametrize("which", ["per_request", "per_step"])
def test_h2_fault_classes(
    which: str, request: pytest.FixtureRequest, honest: Recorder
) -> None:
    workload: Workload = request.getfixturevalue(which)
    compiled = workload.compiled
    priced = price(compiled, POLICY)
    counts: dict[str, dict[RecordingPolicy, int]] = {}
    for letter, label, production in _productions(workload):
        pins = {
            policy: pin_everything(
                compiled, record(compiled, production.values, policy)
            )
            for policy in RecordingPolicy
        }
        counts[letter] = {
            policy: sum(len(p) for p in pins[policy].values())
            for policy in RecordingPolicy
        }
        _row_for_class(
            honest,
            workload,
            letter,
            label,
            production,
            RecordingPolicy.BOUNDARY,
            pins[RecordingPolicy.BOUNDARY],
            pins[RecordingPolicy.VU_OUTPUTS],
            priced,
        )
        if (
            counts[letter][RecordingPolicy.BOUNDARY]
            != counts[letter][RecordingPolicy.VU_OUTPUTS]
        ):
            _row_for_class(
                honest,
                workload,
                letter,
                label,
                production,
                RecordingPolicy.VU_OUTPUTS,
                pins[RecordingPolicy.VU_OUTPUTS],
                pins[RecordingPolicy.BOUNDARY],
                priced,
            )
    boundary = {letter: c[RecordingPolicy.BOUNDARY] for letter, c in counts.items()}
    outputs = {letter: c[RecordingPolicy.VU_OUTPUTS] for letter, c in counts.items()}
    # Section 4 hypotheses.  Tokens-only recording: a flip that changed no recorded value costs nothing.
    assert boundary["a"] == 0
    # Every-VU-output recording pins exactly the faulty VU for a stored corruption ...
    assert outputs["a"] == outputs["b"] == outputs["c"] == outputs["d"] == 1
    # ... and the consumers, never the producer, for a read fault.
    assert outputs["f"] == 1 and outputs["g"] >= 1
    if which == "per_request":
        assert boundary["b"] == 0
        assert boundary["g"] == 0  # a misread whose consequences never reached a token
    else:
        assert boundary["h"] == 0 and outputs["h"] == 0  # the toy attention absorbed it


# -- H3: random silent data corruption at fleet scale --------------------------------------

FLEET_ROUNDS = 3
RUNS_PER_ROUND = 3
FAULTS_PER_RUN = 1.0
"""Poisson mean of faults per run: about a billion times the Llama-3 rate for a run this size."""


@dataclass(frozen=True, slots=True)
class FleetRun:
    """One run of the fleet: how many flips it took, what the server recorded and what that pins."""

    faults: int
    changed_tokens: int
    recorded: dict[int, object]
    pins: dict[int, tuple[int, ...]]

    @property
    def declarations(self) -> int:
        return sum(len(p) for p in self.pins.values())


def poisson(rng: random.Random, mean: float) -> int:
    count, threshold, product = 0, math.exp(-mean), rng.random()
    while product > threshold:
        count += 1
        product *= rng.random()
    return count


def _fleet(workload: Workload, seed: int) -> list[FleetRun]:
    """``FLEET_ROUNDS * RUNS_PER_ROUND`` runs with Poisson-many flips at random dot words and bits."""

    compiled = workload.compiled
    injector = workload.injector
    rng = random.Random(seed)
    sites = [
        compiled.index.verification_unit(unit).interval[-1]
        for replay_unit in range(compiled.index.replay_units.count)
        for unit in dot_units(compiled, replay_unit)
    ]
    outputs = compiled.circuit.outputs
    runs = []
    for _ in range(FLEET_ROUNDS * RUNS_PER_ROUND):
        flips = {
            rng.choice(sites): 1 << rng.randrange(SHAPE.width)
            for _ in range(poisson(rng, FAULTS_PER_RUN))
        }
        values = injector.propagate(flips) if flips else injector.honest
        recorded = record(compiled, values, RecordingPolicy.BOUNDARY)
        runs.append(
            FleetRun(
                len(flips),
                sum(values[a] != injector.honest[a] for a in outputs),
                recorded,
                pin_everything(compiled, recorded),
            )
        )
    return runs


def _epoch(
    workload: Workload, fleet: Sequence[FleetRun], strategy: Strategy, max_faults: int
) -> EpochReport:
    compiled = workload.compiled
    kappa, tree = commit_weights(workload.model.gate_set, workload.model.weights)
    runs = []
    for index, member in enumerate(fleet):
        server = honest_replay(compiled, member.recorded)
        runs.append(
            Run(
                workload.measurement.compilation,
                member.recorded,
                tuple(member.recorded[a] for a in compiled.circuit.outputs),
                weights=kappa,
                weight_tree=tree,
                replay=server.replay,
                declare=server.declare if strategy is Strategy.P1 else None,
                session_id=hashlib.sha256(
                    f"veritor/stress/honest/fleet/{strategy.name}/{index}".encode()
                ).digest()[:16],
            )
        )
    parameters = EpochParameters(
        ETA,
        POLICY,
        max_capacity=None,
        rounds=FLEET_ROUNDS,
        max_advice_bits=workload.measurement.advice_bits,
        max_faults=max_faults,
    )
    schedule = [
        list(range(r * RUNS_PER_ROUND, (r + 1) * RUNS_PER_ROUND))
        for r in range(FLEET_ROUNDS)
    ]
    seeds = [
        hashlib.sha256(f"veritor/stress/honest/fleet/round/{r}".encode()).digest()
        for r in range(FLEET_ROUNDS)
    ]
    return run_epoch(parameters, runs, schedule, seeds)


def _opened_share(compiled: Compiled, report: EpochReport) -> float:
    """The replay cost of the opened RUs over that of every RU of every run."""

    costs = [
        compiled.circuit.Cost(compiled.index.replay_units.unit(unit), "replay")
        for unit in range(compiled.index.replay_units.count)
    ]
    opened = 0
    for round_report in report.rounds:
        for run in round_report.runs:
            assert run.report is not None
            opened += sum(costs[unit] for unit in run.report.sampled_replay_units)
    return opened / (sum(costs) * report.run_count)


def test_h3_fleet_in_protocol(per_request: Workload, honest: Recorder) -> None:
    workload = per_request
    compiled = workload.compiled
    fleet = _fleet(workload, seed=3)
    faulted = sum(1 for member in fleet if member.faults)
    pinning = sum(1 for member in fleet if member.declarations)
    total_pins = sum(member.declarations for member in fleet)
    assert faulted and pinning, "the seed must produce faults that reach a token"
    # The round budget: each pin lands in an opened RU with probability q.
    mean = float(POLICY.q) * total_pins / FLEET_ROUNDS
    f_max = fault_budget(mean, tail=1e-3)
    reports = {
        Strategy.P0: _epoch(workload, fleet, Strategy.P0, 0),
        Strategy.P1: _epoch(workload, fleet, Strategy.P1, f_max),
    }
    priced = price(compiled, POLICY)
    charge_per_round = post_j_charge_bits(compiled, POLICY, ETA / FLEET_ROUNDS, f_max)
    for strategy, report in reports.items():
        per_round = []
        for round_report in report.rounds:
            rejected = sorted(
                {
                    run.report.code.name
                    for run in round_report.runs
                    if run.report is not None and not run.report.accepted
                }
            )
            per_round.append(
                f"round {round_report.index}: {len(round_report.runs)} runs, "
                f"{round_report.declarations} declared, "
                f"{sum(1 for run in round_report.runs if not run.accepted)} rejected"
                + (f" ({', '.join(rejected)})" if rejected else "")
            )
        rejections = sum(1 for r in report.rounds for run in r.runs if not run.accepted)
        declared = sum(r.declarations for r in report.rounds)
        if strategy is Strategy.P1:
            assert report.accepted, report.detail
            assert declared <= total_pins
        else:
            assert declared == 0
        honest.record(
            id="H3a" if strategy is Strategy.P0 else "H3b",
            what=(
                f"random SDC over a fleet of {len(fleet)} runs of {workload.name} in {FLEET_ROUNDS} rounds "
                f"through run_epoch at theta = (1/2, 1/8): Poisson({FAULTS_PER_RUN:g}) bit flips per run at "
                f"random dot words and bits ({faulted} runs faulted, {pinning} changed a streamed token, "
                f"{total_pins} pinned VUs in all); {strategy.value} "
                + (
                    "declares nothing"
                    if strategy is Strategy.P0
                    else f"declares the pins of the opened RUs after J under a round budget f_max = {f_max}"
                )
            ),
            mechanism="M6",
            advice_bits=workload.measurement.advice_bits,
            capacity_bits=math.ceil(report.capacity_bits)
            + FLEET_ROUNDS * RUNS_PER_ROUND * workload.measurement.advice_bits,
            overhead=priced.overhead,
            description_bytes=workload.measurement.description_bytes,
            verdict=f"epoch {report.code.name}: {rejections} of {report.run_count} runs rejected; "
            + "; ".join(per_round),
            notes=(
                f"The rate is about a billion times the Llama-3 SDC rate ({SDC_RATE_PER_DEVICE_HOUR:.1e} per "
                f"device-hour) for a run of this size, so that faults occur at all in a fleet we can afford to "
                f"simulate; the mechanism, not the rate, is what this row checks. Flips land at a uniformly random "
                f"dot word and bit; with tokens-only recording only those that change a streamed token pin anything. "
                f"Round budget f_max = fault_budget(q * pins / rounds = {mean:.2f}, tail 1e-3) = {f_max}; "
                f"u(1) = {unit_fault_bits(compiled):.1f} bits. The capacity is the epoch's Bound at eta summed over "
                f"the rounds, plus the advice of every run"
                + (
                    f"; the budget costs {charge_per_round:.1f} bits per round uncapped (at eta / rounds), "
                    f"{FLEET_ROUNDS * charge_per_round:.1f} for the epoch."
                    if strategy is Strategy.P1
                    else "."
                )
            ),
            declarations=declared,
            charge_bits=round(FLEET_ROUNDS * charge_per_round)
            if strategy is Strategy.P1
            else 0,
            recompute=_opened_share(compiled, report),
            rejections=rejections,
        )


def test_h3_phase_diagram(per_request: Workload, honest: Recorder) -> None:
    compiled = per_request.compiled
    table = compiled.kind_table()
    priced = price(compiled, POLICY)
    headline = estimate()
    inputs = Inputs()
    frontier = serving_table(
        replace(inputs.shape, requests=inputs.requests), "request", "cell"
    )
    headline_policy = VerificationPolicy(Fraction(headline.q), Fraction(headline.s))
    simulation = fold_capacity(table, POLICY, ETA)
    at_headline = rate_capacity(headline.rho, headline.s, headline.inputs.lam)
    fleet_hour = SDC_RATE_PER_DEVICE_HOUR * 16_384
    global_year = SDC_RATE_PER_DEVICE_HOUR * 1e6 * 8760
    where_sim = f"simulation table ({compiled.index.verification_unit_count} VUs), theta = (1/2, 1/8), fold"
    where_headline = f"headline table, theta = ({headline.q:.2g}, {headline.s:.2g}), rate closed form"
    corners: list[
        tuple[str, str, Capacity, VerificationPolicy, float, float | None, str]
    ] = [
        (
            "c",
            where_sim,
            simulation,
            POLICY,
            unit_fault_bits(table),
            fleet_hour,
            "a 16,384-GPU fleet for an hour per round",
        ),
        (
            "d",
            where_sim,
            simulation,
            POLICY,
            unit_fault_bits(table),
            fleet_hour * 24,
            "a 16,384-GPU fleet for a day per round",
        ),
        (
            "e",
            where_sim,
            simulation,
            POLICY,
            unit_fault_bits(table),
            None,
            "the 1% boundary",
        ),
        (
            "f",
            where_headline,
            at_headline,
            headline_policy,
            unit_fault_bits(frontier),
            global_year,
            "a million GPUs for a year per round",
        ),
        (
            "g",
            where_headline,
            at_headline,
            headline_policy,
            unit_fault_bits(frontier),
            None,
            "the 1% boundary",
        ),
        (
            "h",
            where_headline,
            at_headline,
            headline_policy,
            unit_fault_bits(frontier),
            global_year * 1e3,
            "a billion GPUs for a year per round",
        ),
    ]
    boundaries: dict[str, float] = {}
    for letter, where, capacity, policy, u1, density, describe in corners:
        boundary = boundaries.setdefault(
            where, phase_boundary(capacity, policy, share=0.01)
        )
        if density is None:
            density = boundary
        (point,) = phase_diagram(capacity, policy, [density], u1=u1)
        honest.record(
            id=f"H3{letter}",
            what=f"random SDC, {describe}: {density:.3g} faults per round; {where}; P1 against P3",
            mechanism="M6",
            advice_bits=0,
            capacity_bits=math.ceil(point.capacity_bits + point.p1_charge_bits),
            overhead=priced.overhead if policy is POLICY else headline.overhead,
            description_bytes=0,
            verdict=(
                f"P1: q x faults = {point.declarations_mean:.3g} expected declarations, f_max = {point.f_max} at "
                f"tail 1e-6, charge {point.p1_charge_bits:.3g} bits = {point.p1_share:.2e} of U = {point.capacity_bits:.3g}; "
                f"P3: f_max = {point.p3_f_max} at u(1) = {point.u1:.1f} bits, charge {point.p3_charge_bits:.3g} bits, "
                f"recompute 1; P1's charge reaches 1% of U at {boundary:.3g} faults per round"
            ),
            notes=(
                "Faults per round ~ Poisson(rate x device-hours) at the Llama-3 rate; each lands in an opened RU with "
                "probability q, so P1's declarations ~ Poisson(q x faults) and f_max is the smallest budget with tail "
                "below 1e-6; the charge is for the budget the header carries. The simulation table is priced by the "
                "fold (bound/declared_bits, uncapped); the headline by the rate closed form U = rho lambda + log2 e "
                "that docs/global-estimate.md uses, under which a post-J declaration costs rho log2(1 / (1 - s)) bits, "
                "a share log2(1 / (1 - s)) / lambda of U whatever the model. P3 pardons every fault before J at u(1) "
                "and re-executes the whole run."
            ),
            declarations=point.f_max,
            charge_bits=round(point.p1_charge_bits),
            recompute=float(policy.q),
        )
    assert boundaries[where_headline] > boundaries[where_sim]
