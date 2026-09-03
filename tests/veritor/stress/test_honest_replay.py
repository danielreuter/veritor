"""H1, H2, H3: the honest prover's declarations by fault class, by strategy, at fleet scale.

The prover of these rows records only what :class:`RecordingPolicy` says,
reconstructs every opened RU with :func:`replay_pinned` and declares the VUs
the replay pins (``docs/honest-prover.md``, sections 3 and 5).  ``H1*`` runs
one fault of each class through the protocol with RU = request
(``RequestsG``) and RU = step (``ClusterG``) and counts the declarations
under tokens-only (``BOUNDARY``) and every-VU-output (``VU_OUTPUTS``)
recording; ``H2*`` prices the strategies P0-P3 on one run holding four
faults; ``H3*`` drives a small fleet of faulty runs through the epoch layer.
The phase diagram of section 5 is analytic and asserted here, not recorded.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

from veritor.analysis import union
from veritor.analysis.bound import bound
from veritor.analysis.faults import unit_fault_bits
from veritor.analysis.rate import rate
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
    poisson_tail,
)
from veritor.simulation.honest import (
    Production,
    RecordingPolicy,
    Strategy,
    account,
    boundary_at_rest,
    catastrophic,
    combine,
    honest_replay,
    input_read,
    interior_flip,
    phase_diagram,
    pin_everything,
    record,
    recorded_addresses,
    replay_pinned,
    significant_bits,
    token_flip,
    vu_output_read,
    weight_read,
)
from veritor.stress.measure import ETA, POLICY, Measurement, compile_scenario, price
from veritor.stress.models import SHAPE, Model
from veritor.stress.rows import Recorder

REQUESTS = (Request((1, 2, 3), 3), Request((5,), 4))
"""Two requests, seven streamed tokens: small enough that every row runs the protocol."""
FOUR_REQUESTS = REQUESTS + (Request((2, 7), 3), Request((4, 1, 6), 2))
"""H2: four request RUs, one fault each, so that a q = 1/2 challenge opens some and not others."""
PODS, SLOTS, STEPS = 1, 2, 4
OPEN_ALL = VerificationPolicy(Fraction(1), Fraction(1, 8))
"""Every RU opened -- the declarations are the run's whole count -- at the catalogue's ``s``."""
THETA = "theta = (1/2, 1/8)"


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
    """The RU the H1 faults land in."""
    producer: int | None = None
    """For ``ClusterG``: the step whose KV the boundary-at-rest fault corrupts on its way to ``unit``."""

    @property
    def compiled(self) -> Compiled:
        return self.measurement.compiled

    @property
    def u1(self) -> float:
        """``u(1)``: one pre-J declaration."""

        return unit_fault_bits(self.compiled)

    @property
    def u_post(self) -> float:
        """``u_post(1) = rho log2 (1 / (1 - s))`` at the catalogue's policy: one post-J declaration."""

        rho = bound(self.compiled, POLICY, ETA).rho
        return rho * math.log2(1 / (1 - float(POLICY.s)))

    def run(
        self,
        values: dict[int, object],
        outputs: Sequence[int],
        *,
        policy: VerificationPolicy,
        max_faults: int,
        replay: Replay | None,
        declare: Declare | None,
        label: str,
    ) -> ProtocolRun:
        """Run the protocol from the prover's ``values`` (its recording) on the claimed ``outputs``."""

        seed = hashlib.sha256(f"veritor/stress/honest/{label}".encode()).digest()
        kappa, tree = commit_weights(self.model.gate_set, self.model.weights)
        expectation = make_expectation(
            self.measurement.compilation,
            policy,
            tuple(outputs),
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
            values,
            replay=replay,
            weight_tree=tree,
            declare=declare,
        )


def _workload(
    name: str,
    unit_name: str,
    model: Model,
    constructor: object,
    requests: tuple[Request, ...],
    advice: bytes,
    unit: int,
    producer: int | None,
) -> Workload:
    measurement = compile_scenario(constructor, requests, advice, model.gate_set)  # type: ignore[arg-type]
    injector = FaultInjector(
        measurement.compiled, measurement.compilation.inputs, model.weights
    )
    return Workload(name, unit_name, measurement, model, injector, unit, producer)


@pytest.fixture(scope="module")
def per_request(model: Model) -> Workload:
    """RU = request: RU 0 holds the weights, RU 1 the first request (six positions)."""

    return _workload(
        "RequestsG", "request", model, RequestsG(SHAPE), REQUESTS, b"", 1, None
    )


@pytest.fixture(scope="module")
def per_step(model: Model) -> Workload:
    """RU = step: one pod of two slots for four steps; RU 1 is step 0 (both prefills), RU 2 step 1."""

    schedule = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    return _workload(
        "ClusterG",
        "step",
        model,
        ClusterG(SHAPE, PODS, SLOTS, STEPS),
        REQUESTS,
        schedule.encode(),
        2,
        1,
    )


@pytest.fixture(scope="module")
def four_requests(model: Model) -> Workload:
    """RU = request over four requests (RUs 1-4), for the strategy rows."""

    return _workload(
        "RequestsG", "request", model, RequestsG(SHAPE), FOUR_REQUESTS, b"", 1, None
    )


def _outputs(workload: Workload, values: dict[int, int]) -> tuple[int, ...]:
    return tuple(values[a] for a in workload.compiled.circuit.outputs)


def _count(pins: dict[int, tuple[int, ...]]) -> int:
    return sum(len(p) for p in pins.values())


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
        recorded,
        _outputs(per_request, per_request.injector.honest),
        policy=OPEN_ALL,
        max_faults=0,
        replay=server.replay,
        declare=server.declare,
        label="recording-only",
    )
    assert run.report.accepted, run.report
    assert server.pinned == {
        unit: () for unit in range(compiled.index.replay_units.count)
    }


# -- H1: one fault of each class ---------------------------------------------------------

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


def test_boundary_at_rest_flips_a_significant_bit(per_step: Workload) -> None:
    """The toy attention annihilates the top bits of a key at rest; the class flips one that is not."""

    production = boundary_at_rest(per_step.injector, 1, 2)
    width = per_step.compiled.circuit[production.address].width
    significant = significant_bits(
        per_step.injector, production.address, production.misreaders
    )
    inert = tuple(b for b in range(width) if b not in significant)
    flipped = (production.correct ^ production.corrupted).bit_length() - 1
    assert flipped == significant[-1] and width - 1 in inert
    # an inert bit changes nothing any policy records, so no replay pins anything
    for bit in inert:
        misread = production.correct ^ (1 << bit)
        values = per_step.injector.propagate(
            {}, {r: {production.address: misread} for r in production.misreaders}
        )
        for policy in RecordingPolicy:
            pins = pin_everything(
                per_step.compiled, record(per_step.compiled, values, policy)
            )
            assert _count(pins) == 0


def _site(workload: Workload, production: Production) -> str:
    circuit = workload.compiled.circuit
    return (
        f"word {production.address} of RU {production.replay_unit} "
        f"({circuit[production.address].op}, correct {production.correct:#x}, "
        f"{'read as' if production.misreaders else 'stored as'} {production.corrupted:#x}"
        + (f" by {len(production.misreaders)} gates" if production.misreaders else "")
        + ")"
    )


def _tokens(production: Production) -> str:
    changed = [
        i
        for i, (a, b) in enumerate(
            zip(production.outputs, production.honest_outputs, strict=True)
        )
        if a != b
    ]
    return (
        f"{production.changed_outputs} of {len(production.outputs)} streamed tokens changed"
        + (f" (positions {changed})" if changed else "")
    )


@pytest.mark.parametrize("which", ["per_request", "per_step"])
def test_h1_fault_classes(
    which: str, request: pytest.FixtureRequest, honest: Recorder
) -> None:
    """One fault of each class: the tokens-only prover's declarations, and the VU-output prover's."""

    workload: Workload = request.getfixturevalue(which)
    compiled = workload.compiled
    priced = price(compiled, POLICY)
    u1, u_post = workload.u1, workload.u_post
    suffix = "r" if workload.unit_name == "request" else "s"
    counts: dict[str, dict[RecordingPolicy, int]] = {}
    for letter, label, production in _productions(workload):
        pins = {
            policy: pin_everything(
                compiled, record(compiled, production.values, policy)
            )
            for policy in RecordingPolicy
        }
        counts[letter] = {policy: _count(pins[policy]) for policy in RecordingPolicy}
        boundary, outputs = (
            counts[letter][RecordingPolicy.BOUNDARY],
            counts[letter][RecordingPolicy.VU_OUTPUTS],
        )
        # The tokens-only prover through the protocol, every RU opened, with exactly its pins.
        recorded = record(compiled, production.values, RecordingPolicy.BOUNDARY)
        server = honest_replay(compiled, recorded)
        accepted = workload.run(
            recorded,
            production.outputs,
            policy=OPEN_ALL,
            max_faults=boundary,
            replay=server.replay,
            declare=server.declare,
            label=f"h1{letter}/{workload.name}",
        )
        assert accepted.report.accepted, (label, workload.name, accepted.report)
        assert accepted.transcript is not None
        assert accepted.transcript.interiors.declarations == tuple(
            sorted(v for p in pins[RecordingPolicy.BOUNDARY].values() for v in p)
        )
        verdict = (
            f"BOUNDARY {boundary} / VU_OUTPUTS {outputs} declarations; the tokens-only prover "
            f"ACCEPTED at theta = (1, 1/8) with its {boundary} declared"
        )
        if letter == "c":  # once per workload: the budget is binding
            short = workload.run(
                recorded,
                production.outputs,
                policy=OPEN_ALL,
                max_faults=boundary - 1,
                replay=server.replay,
                declare=server.declare,
                label=f"h1{letter}/{workload.name}/short",
            )
            assert short.report.code is VerificationCode.FAULTS_EXCEEDED, short.report
            verdict += f", FAULTS_EXCEEDED at f_max = {boundary - 1}"
        where = {
            policy.name: {unit: len(p) for unit, p in pins[policy].items() if p}
            for policy in RecordingPolicy
        }
        notes = (
            f"{_site(workload, production)}; {_tokens(production)}; pinned VUs by RU: "
            f"BOUNDARY {where['BOUNDARY'] or 'none'}, VU_OUTPUTS {where['VU_OUTPUTS'] or 'none'}."
        )
        if letter == "h":
            significant = significant_bits(
                workload.injector, production.address, production.misreaders
            )
            width = compiled.circuit[production.address].width
            inert = [b for b in range(width) if b not in significant]
            notes += (
                f" Bits {inert} of this key word are inert for its reader (the polynomial softmax's "
                f"square annihilates them): a flip there changes nothing any policy records; "
                f"the class flips bit {significant[-1]}, the most significant live one."
            )
        honest.record(
            id=f"H1{letter}{suffix}",
            what=(
                f"{label}, RU = {workload.unit_name} ({workload.name}): {production.fault.value}"
            ),
            mechanism="M6",
            advice_bits=workload.measurement.advice_bits,
            capacity_bits=math.ceil(
                bound(compiled, POLICY, ETA, max_faults=boundary).bits
            )
            + workload.measurement.advice_bits,
            overhead=priced.overhead,
            description_bytes=workload.measurement.description_bytes,
            verdict=verdict,
            notes=notes,
            declarations=boundary,
            charge_bits=round(boundary * u_post),
            recompute=1.0,
            fault_class=production.fault.name,
            declarations_vu_outputs=outputs,
            changed_tokens=production.changed_outputs,
            u1=round(u1, 1),
            u_post=round(u_post, 1),
        )
    boundary_counts = {k: c[RecordingPolicy.BOUNDARY] for k, c in counts.items()}
    output_counts = {k: c[RecordingPolicy.VU_OUTPUTS] for k, c in counts.items()}
    # Tokens-only recording: a flip that changed no token costs nothing.
    assert boundary_counts["a"] == 0
    # Every-VU-output recording pins exactly the faulty VU for a stored corruption ...
    assert output_counts["a"] == output_counts["b"] == output_counts["c"] == 1
    assert output_counts["d"] == 1
    # ... and the consumers, never the producer, for a read fault: more than the tokens-only prover.
    assert output_counts["e"] > boundary_counts["e"] > 0
    assert output_counts["g"] > boundary_counts["g"]
    # A stored corruption that reached the boundary pins its cascade under tokens-only recording.
    assert boundary_counts["c"] > 1 and boundary_counts["d"] > 1
    if which == "per_request":
        assert boundary_counts["b"] == 0 and boundary_counts["g"] == 0
    else:
        assert boundary_counts["h"] >= 1 and output_counts["h"] >= 1


# -- H2: the four strategies on one run ---------------------------------------------------


def test_h2_strategies(four_requests: Workload, honest: Recorder) -> None:
    """P0-P3 on one run holding four faults: charge, recompute, verdict at ``theta = (1/2, 1/8)``."""

    workload = four_requests
    compiled, injector = workload.compiled, workload.injector
    index = compiled.index
    fourth = next(
        rank
        for rank, address in enumerate(compiled.circuit.inputs)
        if index.replay_units.owner(address) == 4
    )
    faults = (
        ("RU 1", token_flip(injector, 1)),
        ("RU 2", catastrophic(injector, 2)),
        ("RU 3", interior_flip(injector, 3, bit=0)),
        ("RU 4", input_read(injector, fourth)),
    )
    assert {p.replay_unit for _, p in faults} == {1, 2, 3, 4}
    values = combine(injector, [p for _, p in faults])
    outputs = _outputs(workload, values)
    recorded = record(compiled, values, RecordingPolicy.BOUNDARY)
    pins = pin_everything(compiled, recorded)
    every = pin_everything(
        compiled, record(compiled, values, RecordingPolicy.VU_OUTPUTS)
    )
    assert pins[3] == () and _count(every) == 4
    # The signal of P2: a range check before streaming catches a garbage word, not a bit flip.
    flagged = {p.replay_unit for _, p in faults if p.fault.name == "CATASTROPHIC"}
    assert flagged == {2}

    # The header's budget covers every challenge -- every pin of the run -- and is the same
    # for P0 and P1: the header enters the q-challenge's derivation, so one header is one J.
    total = _count(pins)
    # The first seed under which P0 is rejected: a pinned VU opened and sampled.
    for attempt in range(64):
        label = f"h2/{attempt}"
        server = honest_replay(compiled, recorded)
        silent = workload.run(
            recorded,
            outputs,
            policy=POLICY,
            max_faults=total,
            replay=server.replay,
            declare=None,
            label=label,
        )
        opened = set(silent.report.sampled_replay_units)
        if silent.report.code is VerificationCode.RELATION_REJECTED and (
            opened & {1, 2, 4}
        ) != {1, 2, 4}:
            break
    else:
        pytest.fail("no seed sampled a pinned VU while leaving some faulty RU unopened")
    declaring = honest_replay(compiled, recorded)
    p1 = workload.run(
        recorded,
        outputs,
        policy=POLICY,
        max_faults=total,
        replay=declaring.replay,
        declare=declaring.declare,
        label=label,
    )
    assert p1.report.accepted, p1.report
    assert p1.transcript is not None
    assert set(p1.report.sampled_replay_units) == opened
    assert p1.transcript.interiors.declarations == tuple(
        sorted(v for u in opened for v in pins[u])
    )

    priced = price(compiled, POLICY)
    u1, u_post = workload.u1, workload.u_post
    assert u_post > u1  # at q = 1/2 the adaptive price exceeds the fixed-in-advance one
    describe = {
        Strategy.P0: "declares nothing",
        Strategy.P1: "replays the opened RUs pinned and declares their pins after J",
        Strategy.P2: (
            "a value check before streaming flags RU 2 (its garbage word); its pins are declared before "
            "J at u(1), the other opened pins after J"
        ),
        Strategy.P3: "replays every RU before the boundary commitment and declares every pin at u(1)",
    }
    for strategy in Strategy:
        acct = account(
            strategy,
            compiled,
            pins,
            opened,
            u1=u1,
            u_post=u_post,
            flagged=flagged if strategy is Strategy.P2 else (),
        )
        requests_opened = sorted(
            opened - {0}
        )  # RU 0 holds the weights and pins nothing
        if strategy is Strategy.P0:
            verdict = (
                f"RELATION_REJECTED: {silent.report.detail}; opened request RUs {requests_opened} of 1-4 "
                f"(the same header as P1, f_max = {total}, so the same J; nothing declared)"
            )
        elif strategy is Strategy.P1:
            verdict = (
                f"ACCEPTED with {acct.post_j} post-J declarations for the opened request RUs "
                f"{requests_opened} under a header budget f_max = {total} (every pin of the run, so that "
                f"any J is covered); the {total - acct.post_j} pins of the unopened RUs are never declared"
            )
        else:
            verdict = (
                f"ACCEPTED under P1's declarations (the protocol has no pre-J message); counterfactual charge "
                f"{acct.pre_j} pre-J at u(1) = {u1:.1f} + {acct.post_j} post-J at u_post(1) = {u_post:.1f}"
            )
        # The budget a verifier serving this strategy carries in its header: the pins the
        # strategy leaves to post-J declarations, whichever RUs J opens; a pre-J pardon
        # adds u(1) to the fold's bound, under the same cap at the run's outputs.
        certified = bound(
            compiled,
            POLICY,
            ETA,
            max_faults=0 if strategy is Strategy.P0 else total - acct.pre_j,
        )
        honest.record(
            id=f"H2{'abcd'[list(Strategy).index(strategy)]}",
            what=(
                f"{strategy.value}, {describe[strategy]}; one run of four requests (RequestsG, RU = request) "
                f"holding a token flip (RU 1), a catastrophic word (RU 2), a low-bit interior flip (RU 3) "
                f"and a misread prompt token (RU 4); tokens-only recording; {THETA}"
            ),
            mechanism="M6"
            if strategy is Strategy.P1
            else "none"
            if strategy is Strategy.P0
            else "M6 + pre-J pardons (counterfactual)",
            advice_bits=workload.measurement.advice_bits,
            capacity_bits=math.ceil(
                min(certified.bits + acct.pre_j * u1, certified.out_bits)
            )
            + workload.measurement.advice_bits,
            overhead=priced.overhead,
            description_bytes=workload.measurement.description_bytes,
            verdict=verdict,
            notes=(
                f"Pinned VUs by RU under tokens-only recording: { ({u: len(p) for u, p in pins.items() if p}) } "
                f"({total} in all; the low-bit flip of RU 3 changed no token and pins nothing; every-VU-output "
                f"recording would pin {_count(every)}, one per fault). u(1) = {u1:.1f} bits, u_post(1) = "
                f"rho log2(1/(1-s)) = {u_post:.1f} bits (rho = {u_post / math.log2(8 / 7):.0f}); charge is the "
                f"price of the declarations made, recompute the share of the run's replay cost the strategy "
                f"re-executes (the weights RU costs 0). U is the fold with the strategy's header budget "
                f"(the pins it leaves to post-J declarations) plus its pre-J pardons at u(1), capped at the "
                f"run's |Out| = {certified.out_bits} bits, plus advice; the toy's fold is saturated, so every "
                f"strategy certifies the cap and the charge column carries what U would lose."
            )
            if strategy is Strategy.P0
            else "",
            declarations=acct.declarations,
            charge_bits=round(acct.charge_bits),
            recompute=float(acct.recompute),
            pre_j=acct.pre_j,
            post_j=acct.post_j,
            accepted=strategy is not Strategy.P0,
        )


# -- H3: random silent data corruption through the epoch layer -------------------------------

FLEET_ROUNDS = 3
RUNS_PER_ROUND = 3
FAULTS_PER_RUN = 1.0
"""Poisson mean of flips per run: about a billion times the Llama-3 rate for a run this size."""
FLEET_SEED = 3


@dataclass(frozen=True, slots=True)
class FleetRun:
    """One run of the fleet: how many flips it took and what the server recorded under each policy."""

    faults: int
    changed_tokens: int
    values: dict[int, int]

    def recorded(
        self, compiled: Compiled, policy: RecordingPolicy
    ) -> dict[int, object]:
        return record(compiled, self.values, policy)


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
        values = injector.propagate(flips) if flips else dict(injector.honest)
        runs.append(
            FleetRun(
                len(flips),
                sum(values[a] != injector.honest[a] for a in outputs),
                values,
            )
        )
    return runs


def _epoch(
    workload: Workload,
    fleet: Sequence[FleetRun],
    policy: RecordingPolicy,
    declare: bool,
    max_faults: int,
) -> EpochReport:
    compiled = workload.compiled
    kappa, tree = commit_weights(workload.model.gate_set, workload.model.weights)
    runs = []
    for index, member in enumerate(fleet):
        recorded = member.recorded(compiled, policy)
        server = honest_replay(compiled, recorded)
        runs.append(
            Run(
                workload.measurement.compilation,
                recorded,
                tuple(recorded[a] for a in compiled.circuit.outputs),
                weights=kappa,
                weight_tree=tree,
                replay=server.replay,
                declare=server.declare if declare else None,
                session_id=hashlib.sha256(
                    f"veritor/stress/honest/fleet/{index}".encode()
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


def _rounds(report: EpochReport) -> str:
    parts = []
    for round_report in report.rounds:
        rejected = sorted(
            {
                run.report.code.name
                for run in round_report.runs
                if run.report is not None and not run.report.accepted
            }
        )
        parts.append(
            f"round {round_report.index}: {round_report.declarations} declared, "
            f"{sum(1 for run in round_report.runs if not run.accepted)} of {len(round_report.runs)} rejected"
            + (f" ({', '.join(rejected)})" if rejected else "")
        )
    return "; ".join(parts)


def test_h3_fleet_in_protocol(per_request: Workload, honest: Recorder) -> None:
    workload = per_request
    compiled = workload.compiled
    # The first fleet seed under which the silent prover loses a round: a pinned VU opened and sampled.
    for seed in range(FLEET_SEED, FLEET_SEED + 16):
        fleet = _fleet(workload, seed)
        pins = {
            policy: [
                pin_everything(compiled, member.recorded(compiled, policy))
                for member in fleet
            ]
            for policy in RecordingPolicy
        }
        if not any(_count(p) for p in pins[RecordingPolicy.BOUNDARY]):
            continue
        p0 = _epoch(workload, fleet, RecordingPolicy.BOUNDARY, False, 0)
        if not p0.accepted:
            break
    else:
        pytest.fail("no fleet seed produced a pinned VU that was opened and sampled")
    faulted = sum(1 for member in fleet if member.faults)
    flips = sum(member.faults for member in fleet)
    totals = {
        policy: sum(_count(p) for p in pins[policy]) for policy in RecordingPolicy
    }
    pinning = sum(1 for p in pins[RecordingPolicy.BOUNDARY] if _count(p))
    assert (
        totals[RecordingPolicy.VU_OUTPUTS] == flips
    )  # one pin per flip, silent or not
    assert 0 < pinning <= faulted
    priced = price(compiled, POLICY)
    table = union([compiled.kind_table()] * RUNS_PER_ROUND)
    rho = bound(table, POLICY, ETA / FLEET_ROUNDS).rho
    u_post = rho * math.log2(1 / (1 - float(POLICY.s)))
    u1 = unit_fault_bits(table)
    rejections = sum(1 for r in p0.rounds for run in r.runs if not run.accepted)
    assert rejections
    fleet_what = (
        f"random SDC over a fleet of {len(fleet)} runs of {workload.name} (RU = request) in "
        f"{FLEET_ROUNDS} rounds of {RUNS_PER_ROUND} through run_epoch at {THETA}: Poisson({FAULTS_PER_RUN:g}) "
        f"bit flips per run at random dot words and bits ({flips} flips in {faulted} runs, "
        f"{pinning} runs changed a streamed token)"
    )
    honest.record(
        id="H3a",
        what=f"{fleet_what}; P0 declares nothing",
        mechanism="none",
        advice_bits=workload.measurement.advice_bits,
        capacity_bits=math.ceil(p0.capacity_bits)
        + len(fleet) * workload.measurement.advice_bits,
        overhead=priced.overhead,
        description_bytes=workload.measurement.description_bytes,
        verdict=f"epoch {p0.code.name}: {rejections} of {p0.run_count} runs rejected; {_rounds(p0)}",
        notes=(
            f"The rate is about a billion times the Llama-3 SDC rate ({SDC_RATE_PER_DEVICE_HOUR:.1e} per "
            f"device-hour) for a run of this size, so that faults occur at all in a fleet we can afford to "
            f"simulate; the mechanism, not the rate, is what these rows check. Tokens-only recording pins "
            f"{totals[RecordingPolicy.BOUNDARY]} VUs over the fleet (the flips that reached a token, with their "
            f"cascades), every-VU-output recording {totals[RecordingPolicy.VU_OUTPUTS]} (one per flip, silent or "
            f"not). A pinned VU the silent prover neither declares nor recomputes away is a rejection when "
            f"sampled: {rejections} run(s) lost, and with them the epoch."
        ),
        declarations=0,
        charge_bits=0,
        recompute=_opened_share(compiled, p0),
        rejections=rejections,
        accepted=False,
    )
    for letter, policy in (
        ("b", RecordingPolicy.BOUNDARY),
        ("c", RecordingPolicy.VU_OUTPUTS),
    ):
        mean = float(POLICY.q) * totals[policy] / FLEET_ROUNDS
        f_max = fault_budget(mean, tail=1e-3)
        p1 = _epoch(workload, fleet, policy, True, f_max)
        assert p1.accepted, p1.detail
        declared = sum(r.declarations for r in p1.rounds)
        assert 0 < declared <= totals[policy]
        assert max(r.declarations for r in p1.rounds) <= f_max
        honest.record(
            id=f"H3{letter}",
            what=(
                f"{fleet_what}; P1 with {policy.name} recording declares the pins of the opened RUs "
                f"after J under a round budget f_max = {f_max}"
            ),
            mechanism="M6",
            advice_bits=workload.measurement.advice_bits,
            capacity_bits=math.ceil(p1.capacity_bits)
            + len(fleet) * workload.measurement.advice_bits,
            overhead=priced.overhead,
            description_bytes=workload.measurement.description_bytes,
            verdict=(
                f"epoch ACCEPTED: 0 of {p1.run_count} runs rejected, {declared} of the fleet's "
                f"{totals[policy]} pinned VUs declared (those in opened RUs); {_rounds(p1)}"
            ),
            notes=(
                f"Round budget f_max = fault_budget(q x pins / rounds = {mean:.2f}, tail 1e-3) = {f_max}, "
                f"carried by every header of the round and charged once per round: {f_max} x u_post(1) = "
                f"{f_max * u_post:.0f} bits per round at u_post(1) = rho log2(1/(1-s)) = {u_post:.1f} bits "
                f"(rho = {rho:.0f} for the round's union of {RUNS_PER_ROUND} tables at eta / {FLEET_ROUNDS}), "
                f"{FLEET_ROUNDS * f_max * u_post:.0f} for the epoch; the same pins before J would cost "
                f"u(1) = {u1:.1f} bits each. U is the epoch's Bound at eta with the budget, summed over the "
                f"rounds and capped at the outputs, plus the advice of every run. The two recording policies "
                f"need different budgets, so their headers and with them their J differ; recompute is the "
                f"opened share of the fleet's replay cost either way."
            ),
            declarations=declared,
            charge_bits=round(FLEET_ROUNDS * f_max * u_post),
            recompute=_opened_share(compiled, p1),
            rejections=0,
            f_max=f_max,
            accepted=True,
        )


# -- the phase diagram (section 5): analytic on the headline table ----------------------------

DENSITIES = (
    ("a 16,384-GPU fleet for an hour", SDC_RATE_PER_DEVICE_HOUR * 16_384),
    ("a 16,384-GPU fleet for a day", SDC_RATE_PER_DEVICE_HOUR * 16_384 * 24),
    ("a million GPUs for a day", SDC_RATE_PER_DEVICE_HOUR * 1e6 * 24),
    ("a million GPUs for a year", SDC_RATE_PER_DEVICE_HOUR * 1e6 * 8760),
    ("a thousand times that", SDC_RATE_PER_DEVICE_HOUR * 1e9 * 8760),
)
"""Faults per round at the Llama-3 rate, each taken to pin one VU."""


def _headline_policies() -> list[tuple[str, VerificationPolicy, float]]:
    """``(name, policy, rho)`` for the headline operating point and two alternates on its table."""

    headline = estimate()
    inputs = Inputs()
    table = serving_table(
        replace(inputs.shape, requests=inputs.requests), "request", "cell"
    )
    q, s = headline.q, headline.s
    out = [("headline", VerificationPolicy(Fraction(q), Fraction(s)), headline.rho)]
    for name, policy in (
        ("q x 10", VerificationPolicy(Fraction(10 * q), Fraction(s))),
        ("s x 10", VerificationPolicy(Fraction(q), Fraction(10 * s))),
    ):
        out.append((name, policy, rate(table, policy).rho))
    return out


def test_phase_diagram_at_the_headline() -> None:
    """Section 5's table: the post-J price is the floor ``f_max = 1`` at every realistic density."""

    headline = estimate()
    inputs = Inputs()
    table = serving_table(
        replace(inputs.shape, requests=inputs.requests), "request", "cell"
    )
    u1 = unit_fault_bits(table)
    assert u1 == pytest.approx(16 + math.log2(headline.verification_units), abs=0.01)
    policies = _headline_policies()
    diagrams = {
        name: phase_diagram(rho, policy, u1, [d for _, d in DENSITIES])
        for name, policy, rho in policies
    }
    base = diagrams["headline"]
    *realistic, absurd = base
    assert base[0].capacity_bits == pytest.approx(headline.capacity_bits)
    # u_post(1) is (u(1) + 1) / q at the scattered channel, 3.2e-4 of U_0: the price of the floor.
    assert base[0].u_post == pytest.approx((u1 + 1) / headline.q, rel=0.03)
    assert base[0].u_post / base[0].capacity_bits == pytest.approx(3.2e-4, rel=0.02)
    for point in base:
        assert point.exceeded == pytest.approx(
            poisson_tail(point.declarations_mean, point.f_max), rel=1e-6
        )
        # the budget is the floor or the smallest count with a tail below 1e-6
        assert point.exceeded <= 1e-6
        assert point.f_max == 1 or 1e-6 < poisson_tail(
            point.declarations_mean, point.f_max - 1
        )
        # P3 pardons every fault at u(1) and undercuts the floor, at recompute 1 against q ...
        assert point.p3_charge_bits < point.p1_charge_bits
        assert point.p0_rejected == pytest.approx(
            headline.q * headline.s * point.faults_per_round, rel=0.01
        )
        # P0's expected loss of one round is below the floor's price at every density here ...
        assert point.p0_beats_p1()
    for point in realistic:
        assert point.f_max == 1 and point.p1_share == pytest.approx(3.2e-4, rel=0.02)
        # ... by four orders of magnitude while the floor holds (its charge grows with D)
        assert point.p3_charge_bits < 1e-4 * point.p1_charge_bits
    # ... up to a million GPUs for a year; a rejection that forfeits a thousand-round epoch
    # turns the comparison there (q D x 1000 = 0.039 against log2(e) / lam = 0.036).
    assert realistic[-1].p0_beats_p1(rounds_lost=100)
    assert not realistic[-1].p0_beats_p1(rounds_lost=1000)
    assert all(point.p0_beats_p1(rounds_lost=1000) for point in realistic[:-1])
    # A billion GPU-years leaves the floor: q D = 0.039 needs f_max = 3 at tail 1e-6.
    assert absurd.declarations_mean == pytest.approx(0.0389, rel=0.01)
    assert absurd.f_max == 3 and absurd.p1_share == pytest.approx(3 * 3.2e-4, rel=0.02)
    # The floor is left (f_max = 2 at tail 1e-6) only near sqrt(2e-6) / q faults per round ...
    leave = math.sqrt(2e-6) / headline.q
    assert leave == pytest.approx(9.0e4, rel=0.01)
    assert (
        fault_budget(headline.q * leave * 0.5)
        == 1
        < fault_budget(headline.q * leave * 2)
    )
    # ... and P1's charge reaches 1% of U_0 (31 declarations) only past 1e8 faults per round.
    thirty_one = math.ceil(0.01 * base[0].capacity_bits / base[0].u_post)
    assert thirty_one == 31
    assert fault_budget(headline.q * 1e8) < 31 < fault_budget(headline.q * 1e9)
    # Ten times q: the same price per declaration (a share set by s alone), ten times the count.
    tenfold_q = diagrams["q x 10"]
    assert tenfold_q[0].u_post / tenfold_q[0].capacity_bits == pytest.approx(
        base[0].u_post / base[0].capacity_bits, rel=0.02
    )
    assert tenfold_q[-1].declarations_mean == pytest.approx(
        10 * base[-1].declarations_mean
    )
    assert tenfold_q[-1].f_max == 6
    # Ten times s: the same count, ten times the share -- 3.4e-3 of U_0 for the one-declaration floor.
    tenfold_s = diagrams["s x 10"]
    assert tenfold_s[0].p1_share == pytest.approx(10 * base[0].p1_share, rel=0.05)
    assert tenfold_s[-1].declarations_mean == pytest.approx(base[-1].declarations_mean)
    assert tenfold_s[-1].f_max == 3 and tenfold_s[-1].p1_share == pytest.approx(
        0.0101, rel=0.02
    )
    # P0 against P1 does not depend on s: the same verdicts at ten times s.
    assert [p.p0_beats_p1(1000) for p in tenfold_s] == [
        p.p0_beats_p1(1000) for p in base
    ]


@pytest.mark.slow
def test_fold_and_closed_form_price_the_headline_declaration() -> None:
    """``bound(max_faults=1)`` on the headline table is the fold's price; the closed form's is the doc's."""

    headline = estimate()
    inputs = Inputs()
    table = serving_table(
        replace(inputs.shape, requests=inputs.requests), "request", "cell"
    )
    policy = VerificationPolicy(Fraction(headline.q), Fraction(headline.s))
    base = bound(table, policy, ETA)
    declared = bound(table, policy, ETA, max_faults=1)
    fold_price = declared.bits - base.bits
    closed = headline.rho * math.log2(1 / (1 - headline.s))
    # the fold is loose by four orders of magnitude at this policy, its price of a declaration by one
    assert base.bits > 1e3 * headline.capacity_bits
    assert closed < fold_price < 100 * closed
