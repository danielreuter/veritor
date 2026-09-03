"""The simulated datacenter through the epoch layer: rounds of runs, each a window of fleet time.

:mod:`veritor.simulation.datacenter` compiles a whole simulated run into one
circuit and drives ``run_protocol`` once.  Here the same simulation is cut
into **rounds**: a round is a window of wall-clock time on the fleet (a
range of the simulation's synchronous steps), the requests the fleet served
in that window are compiled into the round's **runs** (``ClusterG``, RU =
step, the window's schedule as advice; one run for the fleet, or one per
pod), and the runs of all rounds go through
:func:`veritor.protocol.epoch.run_epoch` with an honest prover: every run's
boundary is committed in its round, the round closes, and the challenged
runs answer.

A request whose attempts cross a window boundary *spans* two rounds.  With
step RUs its later steps read the KV cache its earlier steps declared -- in
another round's run, under another boundary commitment; nothing in the
protocol lets a run read another run's boundary (the cross-run read,
``docs/honest-prover.md`` section 8).  Three honest policies are
expressible today and :func:`partition` builds each:

* :data:`HOLD` -- the request goes, whole, into the run of the round in
  which it completes; that run's schedule window starts where the request
  did, so the earlier round commits nothing for it and its tokens are
  committed one round late (or the round's close waits for it).
* :data:`SPLIT` -- the attempt is cut at the boundary; the earlier round
  commits the positions streamed so far and the later round holds a fresh
  attempt of the *same* request, which recomputes the prefix step by step
  and, since that run knows nothing of the earlier one, outputs the prefix
  again (``Schedule`` has no field for "positions streamed in another
  run").  The recompute needs slot time the original schedule did not have,
  so the later round's cluster gets extra slots for the re-joins.
* :data:`CONTINUE` -- the attempt is cut at the boundary and the remainder
  is a *new request* whose prompt is the original prompt plus the tokens
  already streamed: one prefill over ``prompt + t`` positions, then the
  remaining decodes, in the slot time the original attempt had left.  The
  earlier run's outputs are the prefix; the later run's are the rest;
  nothing is output twice, and the verifier can check the continuation's
  prompt against the earlier run's public claimed outputs.

:func:`run_rounds` compiles the traces and runs the epoch;
:func:`epoch_from_simulation` does both from a :class:`Simulation`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from veritor.analysis import bound, cost, union
from veritor.compile import Compilation
from veritor.constructors import (
    ClusterG,
    Join,
    LMShape,
    Parameters,
    Request,
    Schedule,
)
from veritor.core import KindTable, VerificationPolicy, make_isa_gate_set
from veritor.protocol import MerkleTree, Weights, commit_weights
from veritor.protocol.epoch import EpochParameters, EpochReport, Run, run_epoch
from veritor.research import Compile

from .workload import Simulation

HOLD = "hold"
"""A spanning request is committed whole in the round it completes in."""
SPLIT = "split"
"""A spanning request is cut at the boundary and re-joins the next round afresh (prefix output twice)."""
CONTINUE = "continue"
"""A spanning request is cut at the boundary; the remainder is a new request with the prefix in ``x``."""
POLICIES = (HOLD, SPLIT, CONTINUE)

ETA = Fraction(1, 2**40)
POLICY = VerificationPolicy(Fraction(1, 2), Fraction(1, 8))
"""The simulated datacenter's ``theta``: half the RUs replayed, an eighth of their VUs sampled."""


@dataclass(frozen=True, slots=True)
class Window:
    """A round's window of fleet time: the simulation's steps ``start .. end - 1``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("a window's steps are integers")
        if not 0 <= self.start < self.end:
            raise ValueError("a window needs at least one step")

    def holds(self, step: int) -> bool:
        return self.start <= step < self.end


def windows(steps: int, rounds: int, shift: int = 0) -> tuple[Window, ...]:
    """``rounds`` consecutive windows over ``steps`` steps, as equal as they can be (the last takes the rest).

    ``shift`` pulls every interior boundary that many steps earlier: the
    windows of a pod whose values for the last ``shift`` steps of a round
    are not in hand when the round closes (a straggler).
    """

    if type(steps) is not int or type(rounds) is not int or not 1 <= rounds <= steps:
        raise ValueError("rounds must be between 1 and the number of steps")
    width = steps // rounds
    if type(shift) is not int or not 0 <= shift < width:
        raise ValueError("shift must be a number of steps shorter than a round")
    bounds = [0] + [index * width - shift for index in range(1, rounds)] + [steps]
    return tuple(Window(start, end) for start, end in pairwise(bounds))


@dataclass(frozen=True, slots=True)
class RoundTrace:
    """What one run compiles: ``x`` (the requests), ``a`` (the schedule) and ``y*``, re-based to the run.

    ``schedule.steps`` counts from ``base``: step ``s`` of the schedule is
    step ``base + s`` of the simulation, and ``base <= window.start`` (a
    held request pulls the base back to its first join).  ``pods`` names the
    simulation's pods the run covers, in the schedule's pod order.
    ``origins`` names, per request of ``x``, the simulation request it comes
    from -- a continuation keeps its origin -- and ``streamed`` holds the
    tokens the run claims per request of ``x``, in the run's position
    numbering.  ``spanning`` lists the origins that crossed one of this
    window's boundaries.
    """

    round: int
    window: Window
    base: int
    pods: tuple[int, ...]
    requests: tuple[Request, ...]
    origins: tuple[int, ...]
    schedule: Schedule
    streamed: tuple[tuple[int, ...], ...]
    spanning: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not len(self.requests) == len(self.origins) == len(self.streamed):
            raise ValueError("one origin and one token sequence per request")
        if self.base > self.window.start:
            raise ValueError("a run's base step precedes or opens its window")
        if self.base + self.schedule.steps != self.window.end:
            raise ValueError("a run's schedule ends with its window")
        if len(self.pods) != self.schedule.pods:
            raise ValueError("one simulation pod per pod of the schedule")

    @property
    def tokens(self) -> int:
        """The positions this run outputs (a SPLIT re-join outputs its prefix again)."""

        return sum(self.schedule.active_steps(self.requests).values())


@dataclass(slots=True)
class _Piece:
    """One join of one run in the simulation's step numbering; ``slot < 0`` asks for a slot of its own."""

    pod: int
    step: int
    slot: int
    length: int
    resume: bool = False
    chunk: int = 0

    @classmethod
    def of(
        cls, join: Join, start: int | None = None, stop: int | None = None
    ) -> _Piece:
        start = join.step if start is None else start
        stop = join.step + join.length if stop is None else stop
        return cls(join.pod, start, join.slot, stop - start, join.resume, join.chunk)


@dataclass(slots=True)
class _Member:
    """One request of a run's ``x``: its pieces and the tokens it streams in this run."""

    origin: int
    request: Request
    tokens: tuple[int, ...]
    pieces: list[_Piece]

    @property
    def first_step(self) -> int:
        return min(piece.step for piece in self.pieces)


def _window_of(wins: Sequence[Window], step: int) -> int:
    for index, window in enumerate(wins):
        if window.holds(step):
            return index
    raise ValueError(f"step {step} lies in no window")


def partition(
    simulation: Simulation,
    rounds: int | Sequence[Window],
    spanning: str = HOLD,
    *,
    pods: Sequence[int] | None = None,
) -> tuple[RoundTrace, ...]:
    """Cut a simulation into one :class:`RoundTrace` per round (rounds with nothing served are skipped).

    ``rounds`` is a count (equal windows over the simulation's steps) or the
    windows themselves.  ``spanning`` is :data:`HOLD`, :data:`SPLIT` or
    :data:`CONTINUE` and says what becomes of a request whose attempts cross
    a window boundary; a request that completes inside a window is that
    window's whatever the policy.  ``pods`` restricts the runs to the joins
    of those pods (the runs *of a pod*, a cluster of ``len(pods)`` pods);
    ``None`` is the fleet.
    """

    return partition_schedule(
        simulation.requests,
        simulation.schedule,
        simulation.streamed,
        rounds,
        spanning,
        pods=pods,
    )


def partition_schedule(
    requests: Sequence[Request],
    schedule: Schedule,
    streamed: Sequence[Sequence[int]],
    rounds: int | Sequence[Window],
    spanning: str = HOLD,
    *,
    pods: Sequence[int] | None = None,
) -> tuple[RoundTrace, ...]:
    """:func:`partition` over a schedule, its requests and what they streamed."""

    if spanning not in POLICIES:
        raise ValueError(f"spanning must be one of {POLICIES}")
    wins = windows(schedule.steps, rounds) if isinstance(rounds, int) else tuple(rounds)
    if wins[0].start != 0 or wins[-1].end != schedule.steps:
        raise ValueError("the windows must cover the schedule's steps")
    if any(a.end != b.start for a, b in pairwise(wins)):
        raise ValueError("the windows must be consecutive")
    covered = tuple(range(schedule.pods)) if pods is None else tuple(pods)
    if len(set(covered)) != len(covered) or any(
        not 0 <= pod < schedule.pods for pod in covered
    ):
        raise ValueError("pods must be distinct pods of the schedule")
    by_origin: dict[int, list[Join]] = {}
    for join in schedule.joins:
        if join.pod in covered:
            by_origin.setdefault(join.request, []).append(join)
    members: list[list[_Member]] = [[] for _ in wins]
    crossed: list[set[int]] = [set() for _ in wins]
    for origin in sorted(by_origin):
        joins = sorted(by_origin[origin], key=lambda j: j.step)
        request, tokens = requests[origin], tuple(streamed[origin])
        first = _window_of(wins, joins[0].step)
        last = _window_of(wins, joins[-1].step + joins[-1].length - 1)
        if first == last or spanning == HOLD:
            pieces = [_Piece.of(join) for join in joins]
            members[last].append(_Member(origin, request, tokens, pieces))
        else:
            _cut(wins, joins, origin, request, tokens, spanning, members)
        for index in range(first, last):
            crossed[index].add(origin)
            crossed[index + 1].add(origin)
    return tuple(
        _trace(
            index,
            window,
            members[index],
            covered,
            schedule.slots,
            tuple(sorted(crossed[index])),
        )
        for index, window in enumerate(wins)
        if members[index]
    )


def _cut(
    wins: Sequence[Window],
    joins: Sequence[Join],
    origin: int,
    request: Request,
    tokens: tuple[int, ...],
    spanning: str,
    members: list[list[_Member]],
) -> None:
    """Spread a spanning request's attempts over the windows under :data:`SPLIT` or :data:`CONTINUE`.

    Every attempt must be fresh (a restart from the prefill), so its step
    ``k`` computes the request's position ``k``.  A piece is the part of one
    attempt inside one window; ``reached[w]`` is the furthest position the
    request had streamed when window ``w`` ended.  Under :data:`CONTINUE`
    a piece must begin at the position the earlier windows streamed up to:
    a restart that recomputes, in a later window, positions streamed in an
    earlier one is not modelled.
    """

    pieces: dict[int, list[tuple[Join, int, int]]] = {}
    reached: dict[int, int] = {}
    furthest = 0
    for join in joins:
        if join.resume or join.chunk:
            raise ValueError(
                "a resumed or chunked attempt of a spanning request is not modelled"
            )
        start, end = join.step, join.step + join.length
        while start < end:
            index = _window_of(wins, start)
            stop = min(end, wins[index].end)
            pieces.setdefault(index, []).append((join, start, stop))
            furthest = max(furthest, stop - join.step)
            reached[index] = furthest
            start = stop
    offset = 0  # positions streamed before the window
    for index in sorted(pieces):
        cut: list[_Piece] = []
        for join, start, stop in pieces[index]:
            if spanning == SPLIT and start > join.step:
                # a fresh attempt of the same request, the whole length again
                cut.append(_Piece(join.pod, start, -1, join.length))
            elif spanning == CONTINUE and start - join.step < offset:
                raise ValueError(
                    f"request {origin} recomputes streamed positions across a window"
                    " boundary: not modelled"
                )
            else:  # the attempt as scheduled, cut at the window's end
                cut.append(_Piece.of(join, start, stop))
        if spanning == CONTINUE:  # the rest is a new request with the prefix in x
            continuation = Request(
                request.prompt + tokens[:offset],
                request.max_new - offset,
                request.randomness[offset:],
                request.banned,
            )
            members[index].append(_Member(origin, continuation, tokens[offset:], cut))
        else:
            members[index].append(_Member(origin, request, tokens, cut))
        offset = reached[index]


def _trace(
    index: int,
    window: Window,
    members: list[_Member],
    pods: tuple[int, ...],
    slots: int,
    spanning: tuple[int, ...],
) -> RoundTrace:
    """One round's trace: its members re-based, the SPLIT re-joins in extra slots, everything sorted."""

    members = sorted(members, key=lambda m: m.origin)
    base = min(window.start, min(m.first_step for m in members))
    joins: list[Join] = []
    extra: dict[int, int] = {}  # per pod, the extra slots handed to re-joins
    for position, member in enumerate(members):
        for piece in member.pieces:
            slot = piece.slot
            if slot < 0:
                slot = slots + extra.get(piece.pod, 0)
                extra[piece.pod] = extra.get(piece.pod, 0) + 1
                if piece.step + piece.length > window.end:
                    raise ValueError(
                        f"request {member.origin} cannot recompute its prefix within round {index}"
                    )
            joins.append(
                Join(
                    pods.index(piece.pod),
                    piece.step - base,
                    slot,
                    position,
                    piece.length,
                    piece.resume,
                    piece.chunk,
                )
            )
    schedule = Schedule(
        len(pods),
        slots + max(extra.values(), default=0),
        window.end - base,
        tuple(sorted(joins)),
    )
    return RoundTrace(
        index,
        window,
        base,
        pods,
        tuple(m.request for m in members),
        tuple(m.origin for m in members),
        schedule,
        tuple(m.tokens for m in members),
        spanning,
    )


# -- compiling and running --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledRound:
    """One run compiled and evaluated: its :class:`Run` for the epoch and the numbers a row reads."""

    trace: RoundTrace
    compilation: Compilation
    values: dict[int, int]
    outputs: tuple[int, ...]
    description_bytes: int
    run: Run

    @property
    def advice_bits(self) -> int:
        return self.compilation.advice_bits

    @property
    def table(self) -> KindTable:
        return self.compilation.compiled.kind_table()

    @property
    def honest_cost(self) -> int:
        """The replay cost of the whole run: the honest computation in the cost's units."""

        table = self.table
        return next(row.replay_cost for row in table.rows if row.kind == table.root)

    @property
    def check_outputs(self) -> int:
        return sum(1 for _ in self.compilation.compiled.check_values())


@dataclass(frozen=True, slots=True)
class EpochOutcome:
    """An epoch of the simulation: the verifier's report and every run that went into it."""

    parameters: EpochParameters
    report: EpochReport
    runs: tuple[CompiledRound, ...]

    @property
    def accepted(self) -> bool:
        return self.report.accepted

    @property
    def declarations(self) -> int:
        return sum(round.declarations for round in self.report.rounds)

    @property
    def advice_bits(self) -> int:
        return sum(run.advice_bits for run in self.runs)

    @property
    def description_bytes(self) -> int:
        return sum(run.description_bytes for run in self.runs)

    @property
    def honest_cost(self) -> int:
        return sum(run.honest_cost for run in self.runs)

    @property
    def check_outputs(self) -> int:
        return sum(run.check_outputs for run in self.runs)

    @property
    def outputs(self) -> int:
        return sum(len(run.outputs) for run in self.runs)

    @property
    def capacity_bits(self) -> int:
        """The epoch's ``Bound``: the sum over rounds at ``eta / rounds``, in whole bits."""

        return math.ceil(self.report.capacity_bits)

    @property
    def overhead(self) -> float:
        """``Cost(...).total`` over the honest replay cost, over the union of the runs."""

        table = union([run.table for run in self.runs])
        total = cost(table, self.parameters.policy).total
        honest = next(row.replay_cost for row in table.rows if row.kind == table.root)
        return float(total / honest)

    def uncapped_bits(self) -> float:
        """The epoch's bound with the interface cap lifted (the fold alone), summed over rounds."""

        total = 0.0
        for round_report in self.report.rounds:
            tables = [
                run.table for run in self.runs if run.trace.round == round_report.index
            ]
            if tables:
                result = bound(
                    union(tables), self.parameters.policy, self.parameters.round_eta
                )
                total += min(result.knapsack_bits, result.laplace_bits)
        return total


def _seed_bytes(seed: str, label: str) -> bytes:
    return hashlib.sha256(f"veritor/simulation/epochs/{seed}/{label}".encode()).digest()


def compile_round(
    trace: RoundTrace,
    shape: LMShape,
    weights: Sequence[int],
    kappa: Weights,
    weight_tree: MerkleTree,
    *,
    session_id: bytes,
) -> CompiledRound:
    """``Compile(ClusterG, x, a)`` for one run, evaluated on ``weights``; the outputs must be what was streamed."""

    schedule = trace.schedule
    constructor = ClusterG(shape, schedule.pods, schedule.slots, schedule.steps)
    advice = schedule.encode()
    compilation = Compile(
        constructor,
        trace.requests,
        advice,
        make_isa_gate_set(shape.width),
        max_advice_bits=8 * len(advice),
    )
    circuit = compilation.compiled.circuit
    values = dict(enumerate(circuit.evaluate(compilation.inputs, weights)))
    outputs = tuple(values[address] for address in circuit.outputs)
    layout = constructor.output_layout(trace.requests, schedule)
    expected = tuple(trace.streamed[r][g] for r, g in layout)
    if outputs != expected:
        raise AssertionError(
            f"round {trace.round}: the circuit's outputs are not the streamed tokens"
        )
    run = Run(
        compilation,
        values,
        outputs,
        weights=kappa,
        weight_tree=weight_tree,
        session_id=session_id,
    )
    return CompiledRound(
        trace,
        compilation,
        values,
        outputs,
        len(constructor(trace.requests, advice)[0]),
        run,
    )


def compile_rounds(
    traces: Sequence[RoundTrace],
    shape: LMShape,
    parameters: Parameters,
    *,
    seed: str = "",
) -> tuple[CompiledRound, ...]:
    """:func:`compile_round` for every trace under one weight commitment; session ids derive from ``seed``."""

    if not traces:
        raise ValueError("an epoch needs at least one run")
    weights = parameters.flatten()
    kappa, weight_tree = commit_weights(make_isa_gate_set(shape.width), weights)
    return tuple(
        compile_round(
            trace,
            shape,
            weights,
            kappa,
            weight_tree,
            session_id=_seed_bytes(seed, f"run/{index}")[:16],
        )
        for index, trace in enumerate(traces)
    )


def run_rounds(
    traces: Sequence[RoundTrace],
    shape: LMShape,
    parameters: Parameters,
    *,
    rounds: int | None = None,
    admission: Sequence[int] | None = None,
    seed: str = "",
) -> EpochOutcome:
    """Compile every trace and run the epoch: each trace is one run, admitted in its round.

    ``admission`` names, per trace, the round the run is admitted in
    (``trace.round`` by default; a straggler's run may be admitted later).
    ``rounds`` is the epoch's round count (one past the last admission
    unless given: a trailing round with nothing served still closes).
    The epoch runs at :data:`ETA` and :data:`POLICY` with no fault budget
    and no ``U_max``.  Round seeds and session ids derive from ``seed``, so
    the outcome is deterministic.
    """

    admitted = (
        tuple(trace.round for trace in traces)
        if admission is None
        else tuple(admission)
    )
    if len(admitted) != len(traces) or any(
        round < trace.round for round, trace in zip(admitted, traces, strict=True)
    ):
        raise ValueError("a run is admitted in its own round or a later one")
    count = 1 + max(admitted, default=0) if rounds is None else rounds
    if any(not 0 <= round < count for round in admitted):
        raise ValueError("every run's round must lie within the epoch's rounds")
    compiled = compile_rounds(traces, shape, parameters, seed=seed)
    epoch = EpochParameters(
        ETA,
        POLICY,
        max_capacity=None,
        rounds=count,
        max_advice_bits=max(run.advice_bits for run in compiled),
    )
    schedule = [
        [index for index, round in enumerate(admitted) if round == r]
        for r in range(count)
    ]
    seeds = [_seed_bytes(seed, f"round/{r}") for r in range(count)]
    report = run_epoch(epoch, [run.run for run in compiled], schedule, seeds)
    return EpochOutcome(epoch, report, compiled)


def epoch_from_simulation(
    simulation: Simulation,
    shape: LMShape,
    parameters: Parameters,
    rounds: int | Sequence[Window],
    *,
    spanning: str = HOLD,
    seed: str = "",
) -> EpochOutcome:
    """Cut ``simulation`` into rounds (:func:`partition`) and run the epoch (:func:`run_rounds`)."""

    return run_rounds(
        partition(simulation, rounds, spanning),
        shape,
        parameters,
        rounds=rounds if isinstance(rounds, int) else len(rounds),
        seed=seed,
    )


__all__ = [
    "CONTINUE",
    "ETA",
    "HOLD",
    "POLICIES",
    "POLICY",
    "SPLIT",
    "CompiledRound",
    "EpochOutcome",
    "RoundTrace",
    "Window",
    "compile_round",
    "compile_rounds",
    "epoch_from_simulation",
    "partition",
    "partition_schedule",
    "run_rounds",
    "windows",
]
