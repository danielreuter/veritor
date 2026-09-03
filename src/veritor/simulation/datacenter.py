"""``python -m veritor.simulation.datacenter``: a simulated inference datacenter through Verity.

:func:`run` takes a :class:`DemoConfig` and returns a
:class:`~veritor.simulation.summary.Summary` with every number the report prints;
:func:`~veritor.simulation.report.render` turns it into the narrative report.  The
stages, in order:

1. **Workload.** :mod:`veritor.simulation.workload` simulates the server: Poisson
   arrivals over wall-clock time, first-come first-served continual
   batching over ``pods x slots``, one synchronous decode step at a time.
   Timing reaches the verifier only through the :class:`Schedule` -- the
   advice ``a``, charged by its encoded length.
2. **Control flow.** A request leaves its slot when the model emits the
   end-of-sequence token, so the circuit has exactly the steps the server
   executed; every response is checked against ``reference_generate``.
3. **Failures.** Pods fail at random; tokens already streamed stand and the
   occupants restart from the prefill elsewhere as further joins of the same
   request.
4. **Nondeterminism.** Tokens are sampled by the ``sample`` VU over a public
   random word per generated position, published as an ``in`` gate.
5. **Compile.** ``Compile(ClusterG, requests, schedule)``.
6. **Honest run.** The full three-message protocol with timings and message
   sizes, under a fixed policy; ``Optimize`` is run alongside to show why it
   cannot separate policies at this scale.
7. **Adversary.** :mod:`veritor.simulation.adversary` exfiltrates a secret through
   the head VUs and the verifier's detection rate is measured against the
   analysis' prediction.
8. **Bound and cost.** ``Bound(C, I, theta)`` at ``eta`` and ``Cost``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Self

from veritor.analysis import BoundResult, bound, cost, optimize
from veritor.analysis.bound import cut_bits
from veritor.analysis.optimize import PolicyGrid
from veritor.analysis.probability import budget, saturation_cost, unit_cost
from veritor.compile import Compilation
from veritor.constructors import (
    ClusterG,
    LMShape,
    random_parameters,
    reference_generate,
)
from veritor.core import (
    CompilationLimits,
    VerificationLimits,
    VerificationPolicy,
    canonical_json_bytes,
    make_isa_gate_set,
)
from veritor.core.description import REPLAY, VERIFICATION
from veritor.protocol import (
    MerkleTree,
    ProverSession,
    Reject,
    VerificationReport,
    VerifierParameters,
    VerifierSession,
    Weights,
    commit_weights,
    encode_transcript,
    make_expectation,
)
from veritor.protocol.parameters import (
    DEFAULT_MAX_WORK,
    expected_work,
    positions_per_unit,
)
from veritor.research import Compile

from . import adversary
from .report import render
from .summary import (
    AdversarySummary,
    ArrivalRecord,
    AttackRow,
    AttemptRecord,
    BoundSummary,
    CompileSummary,
    CostSummary,
    FailureRecord,
    HonestRunSummary,
    ModelSummary,
    PolicySummary,
    Summary,
    WorkloadSummary,
)
from .workload import COMPLETE, EOS, RUN_END, Simulation, WorkloadConfig, simulate

ETA = Fraction(1, 2**40)
POLICY = VerificationPolicy(Fraction(1, 2), Fraction(1, 8))
"""The run's ``theta``: half the RUs replayed, an eighth of their VUs sampled (``qs = 1/16``)."""
GRID = PolicyGrid(
    (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1)),
    (Fraction(1, 64), Fraction(1, 16), Fraction(1, 4), Fraction(1)),
)
TOLERANCE_SIGMAS = 4.0
"""An observed survival within this many standard deviations of the prediction agrees with it."""


# -- configuration --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Everything a simulation run depends on."""

    scale: str
    shape: LMShape
    workload: WorkloadConfig
    parameters_seed: int = 0
    eta: Fraction = ETA
    policy: VerificationPolicy = POLICY
    grid: PolicyGrid | None = GRID
    """The grid ``Optimize`` searches, for the report; ``None`` skips it."""
    work_budget: int = DEFAULT_MAX_WORK
    """The verifier's ``W_max`` in operations."""
    compilation_limits: CompilationLimits = field(default_factory=CompilationLimits)
    """The verifier's parsing limits; a run with many streamed tokens has many output runs."""
    verification_limits: VerificationLimits = field(default_factory=VerificationLimits)
    """The verifier's protocol limits; a big run has more positions per kind than the defaults."""
    attack_sizes: tuple[int, ...] | None = None
    """Carriers per attack row; ``None`` for ``1, 4, 16, ...`` up to every streamed token."""
    survival_trials: int = 1000
    """Fresh challenge derivations per attack row."""
    protocol_trials: int = 2
    """Full protocol runs against the dishonest server per attack row."""


def small_config(
    *,
    seed: int = 0,
    pods: int = 2,
    slots: int = 2,
    steps: int = 16,
    requests: int = 12,
    failure_rate: float = 0.03,
    sampling: bool = True,
) -> DemoConfig:
    """Runs end to end in well under a minute: two pods, sixteen steps, a dozen requests."""

    shape = LMShape(
        vocab=8, d_model=4, heads=2, layers=1, context=16, width=16, sampling=sampling
    )
    workload = WorkloadConfig(
        pods=pods,
        slots=slots,
        steps=steps,
        arrivals=requests,
        seed=seed,
        failure_rate=failure_rate,
        forced_failures=((0, steps // 3),),
    )
    return DemoConfig("small", shape, workload, parameters_seed=seed)


def medium_config(
    *,
    seed: int = 0,
    pods: int = 3,
    slots: int = 3,
    steps: int = 32,
    requests: int = 48,
    failure_rate: float = 0.02,
    sampling: bool = True,
) -> DemoConfig:
    """A few minutes: three pods of three slots, fifty requests, a wider model (about 500k gates)."""

    shape = LMShape(
        vocab=16, d_model=8, heads=2, layers=1, context=24, width=16, sampling=sampling
    )
    workload = WorkloadConfig(
        pods=pods,
        slots=slots,
        steps=steps,
        arrivals=requests,
        seed=seed,
        prompt_lengths=(2, 6),
        max_new_lengths=(4, 12),
        failure_rate=failure_rate,
        forced_failures=((0, steps // 3),),
    )
    return DemoConfig(
        "medium",
        shape,
        workload,
        parameters_seed=seed,
        compilation_limits=CompilationLimits(
            max_output_runs=1 << 12, max_output_runs_total=1 << 16
        ),
        verification_limits=VerificationLimits(
            max_positions_per_unit=1 << 24, max_positions=1 << 26
        ),
        survival_trials=2000,
        protocol_trials=1,
    )


# -- the run --------------------------------------------------------------------------


@dataclass(slots=True)
class _Timer:
    total: float = 0.0
    _start: float = field(default=0.0, repr=False)

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.total += time.perf_counter() - self._start


def _seed_bytes(seed: str, label: str) -> bytes:
    return hashlib.sha256(f"{seed}/{label}".encode()).digest()


def run(config: DemoConfig) -> Summary:
    """Simulate, compile, verify honestly, attack, price; return every number."""

    started = time.perf_counter()
    shape = config.shape
    parameters = random_parameters(shape, config.parameters_seed)
    gate_set = make_isa_gate_set(shape.width)
    weights = parameters.flatten()
    kappa, weight_tree = commit_weights(gate_set, weights)

    # 1-4. the server's run
    with _Timer() as simulation_timer:
        simulation = simulate(config.workload, shape, parameters)
        reference = reference_generate(shape, parameters, simulation.requests)
    matches = all(
        tuple(tokens) == expected[: len(tokens)]
        for tokens, expected in zip(simulation.streamed, reference, strict=True)
    )
    advice = simulation.schedule.encode()

    # 5. compile
    constructor = ClusterG(
        shape, config.workload.pods, config.workload.slots, config.workload.steps
    )
    with _Timer() as compile_timer:
        compilation = Compile(
            constructor,
            simulation.requests,
            advice,
            gate_set,
            limits=config.compilation_limits,
            max_advice_bits=8 * len(advice),
        )
    compiled = compilation.compiled
    circuit = compiled.circuit
    with _Timer() as evaluate_timer:
        values = dict(enumerate(circuit.evaluate(compilation.inputs, weights)))
    outputs = tuple(values[address] for address in circuit.outputs)
    layout = constructor.output_layout(simulation.requests, simulation.schedule)
    streamed = tuple(token for response in simulation.streamed for token in response)
    if outputs != streamed:
        raise AssertionError("the circuit's outputs are not the streamed tokens")
    description_bytes = len(constructor(simulation.requests, advice)[0])
    compile_summary = _compile_summary(
        compilation,
        description_bytes,
        compile_timer.total,
        evaluate_timer.total,
        simulation,
    )

    # 6. the policy and the honest run
    io_count = len(set(circuit.inputs) | set(circuit.outputs))
    policy = config.policy
    with _Timer() as bound_timer:
        bound_result = bound(compiled, policy, config.eta)
    policy_summary = _policy_summary(
        config, compilation, compile_summary.out_bits, io_count
    )
    verifier_parameters = VerifierParameters(
        config.eta,
        max_capacity=math.ceil(bound_result.bits),
        max_advice_bits=compilation.advice_bits,
        max_work=config.work_budget,
    )
    honest = _honest_run(
        compilation,
        policy,
        verifier_parameters,
        kappa,
        weight_tree,
        values,
        outputs,
        config,
    )

    # 8. bound and cost
    bound_summary = _bound_summary(
        bound_result,
        policy,
        config.eta,
        shape.vocab_bits,
        compile_summary.head_vu_cut_bits,
        bound_timer.total,
    )
    expected = cost(compiled, policy)
    cost_summary = CostSummary(
        boundary=float(expected.boundary),
        recompute=float(expected.recompute),
        commit_interior=float(expected.commit_interior),
        proof=float(expected.proof),
        total=float(expected.total),
        weights_per_epoch=float(expected.weights),
        verifier_expected_work=policy_summary.expected_work,
    )

    # 7. the adversary
    with _Timer() as adversary_timer:
        rows = _attack_rows(
            config,
            compilation,
            policy,
            verifier_parameters,
            kappa,
            weight_tree,
            layout,
            outputs,
            weights,
        )
    adversary_summary = AdversarySummary(
        channel=(
            "the last tokens of the responses, round-robin over the requests; each carrier is the "
            "output of one head VU forced to the next vocab_bits bits of the secret"
        ),
        bits_per_vu=shape.vocab_bits,
        kappa_per_vu=compile_summary.head_vu_cut_bits,
        tolerance_sigmas=TOLERANCE_SIGMAS,
        rows=rows,
        seconds=adversary_timer.total,
    )

    return Summary(
        scale=config.scale,
        seed=config.workload.seed,
        workload=_workload_summary(
            simulation, advice, matches, shape, simulation_timer.total
        ),
        model=ModelSummary(
            vocab=shape.vocab,
            d_model=shape.d_model,
            heads=shape.heads,
            layers=shape.layers,
            context=shape.context,
            width=shape.width,
            sampling=shape.sampling,
            vocab_bits=shape.vocab_bits,
            random_bits=shape.random_bits if shape.sampling else 0,
            weights=len(weights),
            parameters_seed=config.parameters_seed,
            weight_root=kappa.commitment.root.hex(),
        ),
        compile=compile_summary,
        policy=policy_summary,
        bound=bound_summary,
        cost=cost_summary,
        honest=honest,
        adversary=adversary_summary,
        notes=_notes(config, bound_result, compile_summary),
        total_seconds=time.perf_counter() - started,
    )


def _workload_summary(
    simulation: Simulation, advice: bytes, matches: bool, shape: LMShape, seconds: float
) -> WorkloadSummary:
    config = simulation.config
    outcomes = [attempt.outcome for attempt in simulation.attempts]
    return WorkloadSummary(
        pods=config.pods,
        slots=config.slots,
        steps=config.steps,
        step_seconds=config.step_seconds,
        load=config.load,
        arrivals=len(simulation.arrivals),
        admitted=len(simulation.requests),
        unserved=simulation.unserved,
        joins=len(simulation.schedule.joins),
        restarts=simulation.restarts,
        failures=tuple(
            FailureRecord(f.pod, f.step, f.aborted) for f in simulation.failures
        ),
        eos_token=shape.vocab - 1 if config.eos is None else config.eos,
        eos_stops=outcomes.count(EOS),
        completed=outcomes.count(COMPLETE),
        cut_by_run_end=outcomes.count(RUN_END),
        tokens=simulation.tokens,
        token_steps=sum(join.length for join in simulation.schedule.joins),
        utilization=simulation.utilization,
        advice_bytes=len(advice),
        advice_bits=simulation.schedule.bit_length(),
        arrival_records=tuple(
            ArrivalRecord(
                a.index, a.time, len(a.request.prompt), a.request.max_new, a.request_id
            )
            for a in simulation.arrivals
        ),
        attempts=tuple(
            AttemptRecord(
                a.join.pod,
                a.join.step,
                a.join.slot,
                a.join.request,
                a.join.length,
                a.outcome,
                a.streamed,
            )
            for a in simulation.attempts
        ),
        occupancy=simulation.occupied,
        responses=simulation.streamed,
        matches_reference=matches,
        seconds=seconds,
    )


def _compile_summary(
    compilation: Compilation,
    description_bytes: int,
    compile_seconds: float,
    evaluate_seconds: float,
    simulation: Simulation,
) -> CompileSummary:
    compiled = compilation.compiled
    circuit, index = compiled.circuit, compiled.index
    table = compiled.kind_table()
    rows = {row.kind: row for row in table.rows}
    head_address = circuit.outputs[0]
    head_replay = index.replay_units.owner(head_address)
    block = index.verification_units(head_replay)
    head = rows[index.verification_unit(block.first + block.owner(head_address)).kind]
    replay_rows = [row for row in table.rows if row.role == REPLAY]
    verification_rows = [row for row in table.rows if row.role == VERIFICATION]
    replay_units = index.replay_units.count
    verification_units = index.verification_unit_count
    boundary = index.boundary().count
    token_steps = sum(join.length for join in simulation.schedule.joins)
    return CompileSummary(
        constructor=compilation.constructor,
        compiled_digest=compiled.digest,
        n=circuit.n,
        kinds=len(table.rows),
        replay_units=replay_units,
        verification_units=verification_units,
        weight_gates=index.weight_count,
        input_gates=len(circuit.inputs),
        outputs=len(circuit.outputs),
        out_bits=rows[table.root].out_bits,
        description_bytes=description_bytes,
        advice_bytes=len(compilation.advice),
        compile_seconds=compile_seconds,
        evaluate_seconds=evaluate_seconds,
        gates_per_token_step=(circuit.n - index.weight_count) / token_steps,
        vus_per_token_step=verification_units / token_steps,
        head_kind=head.kind[:12],
        head_vu_gates=head.size,
        head_vu_cut_bits=cut_bits(head),
        boundary_positions=boundary,
        interior_positions=circuit.n - boundary - index.weight_count,
        W_R=sum(row.copies * row.out_bits for row in replay_rows) / replay_units,
        W_V=sum(row.copies * cut_bits(row) for row in verification_rows)
        / verification_units,
        positions_per_vu=sum(
            row.copies * positions_per_unit(row) for row in verification_rows
        )
        / verification_units,
    )


def _policy_summary(
    config: DemoConfig, compilation: Compilation, out_bits: int, io_count: int
) -> PolicySummary:
    """The run's fixed ``theta`` and, for the record, what ``Optimize`` would pick.

    ``Optimize`` ranks policies by the certified capacity; at toy scale
    every affordable policy is capped by the output interface itself (see
    :func:`_notes`), so it returns the cheapest grid point and would sample
    almost nothing.  The simulation therefore fixes ``theta`` and shows both.
    """

    compiled = compilation.compiled
    found = None
    evaluated = 0
    if config.grid is not None:
        found = optimize(
            compiled,
            config.eta,
            config.grid,
            max_bits=out_bits,
            accept=lambda policy: (
                expected_work(compiled, policy, io_count) <= config.work_budget
            ),
        )
        evaluated = found.evaluated if found is not None else 0
    return PolicySummary(
        q=str(config.policy.q),
        s=str(config.policy.s),
        rule=(
            "fixed: half the RUs replayed, an eighth of their VUs sampled (every VU sampled with "
            "probability qs = 1/16); Optimize cannot separate policies at this scale"
        ),
        expected_work=float(expected_work(compiled, config.policy, io_count)),
        work_budget=config.work_budget,
        io_count=io_count,
        optimize_q=None if found is None else str(found.policy.q),
        optimize_s=None if found is None else str(found.policy.s),
        optimize_bits=None if found is None else found.bound.bits,
        optimize_cost=None if found is None else float(found.cost.total),
        grid_points=0
        if config.grid is None
        else sum(1 for _ in config.grid.policies()),
        grid_evaluated=evaluated,
    )


def _bound_summary(
    result: BoundResult,
    policy: VerificationPolicy,
    eta: Fraction,
    vocab_bits: int,
    kappa: int,
    seconds: float,
) -> BoundSummary:
    budget_nats = budget(eta)
    c1 = unit_cost(policy, 1)
    vus_to_eta = math.ceil(budget_nats / c1) if c1 > 0 else 0
    return BoundSummary(
        eta=f"2^-{eta.denominator.bit_length() - 1}"
        if eta.numerator == 1
        else str(eta),
        bits=result.bits,
        capped=result.capped,
        out_bits=result.out_bits,
        knapsack_bits=result.knapsack_bits,
        laplace_bits=result.laplace_bits,
        budget_nats=budget_nats,
        unit_cost_nats=c1,
        saturation_cost_nats=saturation_cost(policy),
        vus_to_eta=vus_to_eta,
        bits_realized_to_eta=vus_to_eta * vocab_bits,
        bits_charged_to_eta=vus_to_eta * kappa,
        seconds=seconds,
    )


def _honest_run(
    compilation: Compilation,
    policy: VerificationPolicy,
    parameters: VerifierParameters,
    kappa: Weights,
    weight_tree: MerkleTree,
    values: dict[int, int],
    outputs: tuple[int, ...],
    config: DemoConfig,
) -> HonestRunSummary:
    """The three-message protocol, prover and verifier timed separately."""

    seed = f"veritor/simulation/{config.scale}/{config.workload.seed}"
    expectation = make_expectation(
        compilation,
        policy,
        outputs,
        parameters=parameters,
        weights=kappa,
        session_id=_seed_bytes(seed, "session")[:16],
        q_seed=_seed_bytes(seed, "q"),
        s_seed=_seed_bytes(seed, "s"),
    )
    prover_time, verifier_time = _Timer(), _Timer()
    compiled = compilation.compiled
    interior_roots: tuple[str, ...] = ()
    boundary_root, boundary_positions, openings = "", 0, 0
    message_bytes: dict[str, int] = {}
    transcript_bytes = 0
    try:
        with verifier_time:
            verifier = VerifierSession(
                expectation, compiled, limits=config.verification_limits
            )
        prover = ProverSession(
            compiled,
            verifier.header,
            values,
            limits=config.verification_limits,
            weight_tree=weight_tree,
        )
        with prover_time:
            boundary = prover.boundary()
        boundary_root = boundary.commitment.root.hex()
        boundary_positions = boundary.commitment.count
        with verifier_time:
            replay_challenge = verifier.receive_boundary(boundary)
        with prover_time:
            interiors = prover.interiors(replay_challenge)
        interior_roots = tuple(c.root.hex() for c in interiors.commitments)
        with verifier_time:
            sample_challenge = verifier.receive_interiors(interiors)
        with prover_time:
            evidence = prover.evidence(sample_challenge)
        openings = sum(len(batch) for batch in evidence.units)
        with verifier_time:
            report = verifier.receive_evidence(evidence)
        message_bytes = {
            "boundary": len(canonical_json_bytes(boundary.manifest)),
            "replay_challenge": len(canonical_json_bytes(replay_challenge.manifest)),
            "interiors": len(canonical_json_bytes(interiors.manifest)),
            "sample_challenge": len(canonical_json_bytes(sample_challenge.manifest)),
            "evidence": len(canonical_json_bytes(evidence.manifest)),
        }
        transcript_bytes = len(encode_transcript(verifier.transcript))
    except Reject as rejection:
        report = VerificationReport(rejection.code, rejection.detail)
    return HonestRunSummary(
        accepted=report.accepted,
        code=report.code.name,
        detail=report.detail,
        session_id=expectation.session_id.hex(),
        q_seed=expectation.q_seed.hex(),
        s_seed=expectation.s_seed.hex(),
        weight_root=kappa.commitment.root.hex(),
        boundary_root=boundary_root,
        boundary_positions=boundary_positions,
        interior_roots=interior_roots,
        replay_units_opened=len(report.sampled_replay_units),
        verification_units_opened=len(report.sampled_verification_units),
        openings=openings,
        prover_seconds=prover_time.total,
        verifier_seconds=verifier_time.total,
        message_bytes=message_bytes,
        transcript_bytes=transcript_bytes,
        max_capacity=parameters.max_capacity or 0,
    )


def attack_sizes(tokens: int) -> tuple[int, ...]:
    """``1, 4, 16, ...`` carriers up to, and then, every streamed token."""

    sizes: list[int] = []
    size = 1
    while size < tokens:
        sizes.append(size)
        size *= 4
    sizes.append(tokens)
    return tuple(sizes)


def _attack_rows(
    config: DemoConfig,
    compilation: Compilation,
    policy: VerificationPolicy,
    parameters: VerifierParameters,
    kappa: Weights,
    weight_tree: MerkleTree,
    layout: Sequence[tuple[int, int]],
    honest_outputs: tuple[int, ...],
    weights: Sequence[int],
) -> tuple[AttackRow, ...]:
    compiled = compilation.compiled
    vocab_bits = config.shape.vocab_bits
    sizes = (
        attack_sizes(len(layout))
        if config.attack_sizes is None
        else config.attack_sizes
    )
    rows: list[AttackRow] = []
    for size in sizes:
        secret = adversary.random_secret(
            size * vocab_bits, f"{config.workload.seed}/{size}"
        )
        attack = adversary.plan_attack(
            compiled, compilation.inputs, weights, layout, secret, vocab_bits
        )
        decoded = adversary.decode_secret(attack.outputs, attack.carriers, vocab_bits)
        carriers = set(attack.carriers)
        unchanged = all(
            attack.outputs[i] == honest_outputs[i]
            for i in range(len(honest_outputs))
            if i not in carriers
        )
        predicted = adversary.predicted_survival(policy, attack)
        label = f"{config.scale}/{config.workload.seed}/{size}"
        escaped = adversary.survival_trials(
            compiled,
            policy,
            attack,
            config.survival_trials,
            label=label,
            limits=config.verification_limits,
        )
        reports = adversary.protocol_trials(
            compilation,
            policy,
            parameters,
            kappa,
            weight_tree,
            attack,
            config.protocol_trials,
            label=label,
            limits=config.verification_limits,
        )
        p = float(predicted)
        observed = escaped / config.survival_trials
        sigma = math.sqrt(p * (1 - p) / config.survival_trials)
        rows.append(
            AttackRow(
                bits=attack.bits,
                carriers=len(attack.carriers),
                vus_corrupted=len(attack.verification_units),
                replay_units_touched=len(set(attack.replay_units)),
                errors_per_replay_unit=attack.errors_per_replay_unit,
                predicted_survival=p,
                trials=config.survival_trials,
                escaped=escaped,
                observed_survival=observed,
                sigma=sigma,
                deviation_sigmas=abs(observed - p) / sigma if sigma else 0.0,
                protocol_trials=len(reports),
                protocol_accepted=sum(report.accepted for report in reports),
                secret=secret,
                decoded=decoded,
                honest_tokens_unchanged=unchanged,
            )
        )
    return tuple(rows)


def _notes(
    config: DemoConfig, result: BoundResult, compiled: CompileSummary
) -> tuple[str, ...]:
    sampling = (
        "Nondeterminism: token sampling is the `sample` VU of the toy LM (a division-free CDF "
        "compare over squared scores); the random word of every generated position is a public "
        "input, so it is part of x and not a covert degree of freedom."
    )
    argmax = (
        "Nondeterminism: sampling is off (--no-sampling); tokens are the argmax and the only "
        "nondeterminism shown is the scheduler's, which reaches the verifier through the advice."
    )
    restarts = (
        "Failures and restarts: a pod failure ends the occupants' joins where they are (their "
        "streamed tokens stay circuit outputs) and each request joins again elsewhere from the "
        "prefill; the constructor recomputes the positions already streamed without re-emitting "
        "them and validates that a request's attempts never overlap."
    )
    notes = [sampling if config.shape.sampling else argmax, restarts]
    if result.capped:
        eta = f"2^-{config.eta.denominator.bit_length() - 1}"
        notes.append(
            f"Bound: at eta = {eta} the fold gives {min(result.knapsack_bits, result.laplace_bits):.0f} "
            f"bits, more than this run's whole output ({compiled.out_bits} bits), so U is capped by the "
            "interface: an adversary who corrupts every token decision of a run this small is still "
            "likely to escape, and no affordable policy certifies less than the output itself. The "
            "capacity is a property of the policy (about Lambda / c(1) corrupted VUs, kappa bits each), "
            "not of the run's size, so it becomes a vanishing fraction of the output as the run grows."
        )
    return tuple(notes)


# -- the command line -----------------------------------------------------------------


def build_config(args: argparse.Namespace) -> DemoConfig:
    factory = small_config if args.scale == "small" else medium_config
    options = {
        name: getattr(args, name)
        for name in ("seed", "pods", "slots", "steps", "requests", "failure_rate")
        if getattr(args, name) is not None
    }
    config = factory(sampling=args.sampling, **options)
    if args.trials is not None:
        config = dataclasses.replace(config, survival_trials=args.trials)
    return config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m veritor.simulation.datacenter",
        description="Simulate an inference datacenter and run it through Verity.",
    )
    p.add_argument("--scale", choices=("small", "medium"), default="small")
    p.add_argument("--seed", type=int)
    p.add_argument("--pods", type=int)
    p.add_argument("--slots", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--requests", type=int, help="arrivals to generate")
    p.add_argument("--failure-rate", type=float, dest="failure_rate")
    p.add_argument("--sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trials", type=int, help="challenge derivations per attack row")
    p.add_argument("--json", type=Path, help="also write the summary as JSON")
    p.add_argument("--quiet", action="store_true", help="do not print the report")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = run(build_config(args))
    if not args.quiet:
        print(render(summary))
    if args.json is not None:
        args.json.write_text(summary.to_json(indent=2))
    return 0 if summary.honest.accepted else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ETA",
    "GRID",
    "POLICY",
    "TOLERANCE_SIGMAS",
    "DemoConfig",
    "Summary",
    "attack_sizes",
    "build_config",
    "main",
    "medium_config",
    "render",
    "run",
    "small_config",
]
