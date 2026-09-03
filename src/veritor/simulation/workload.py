"""A simulated inference datacenter: arrivals over wall-clock time, continual batching, EOS, failures.

The simulation is the *server's* side of a run.  Requests arrive as a
Poisson process; a first-come, first-served scheduler admits them to the
free slots of live pods step by step; every occupant advances one token per
synchronous step by running the reference model
(:class:`~veritor.constructors.lm.Decoder`); a request leaves its slot when
it emits the end-of-sequence token or reaches ``max_new``; pods fail at
random and their occupants are re-queued and restarted from the prefill.

Nothing about timing reaches the verifier except through the
:class:`~veritor.constructors.schedule.Schedule` the simulation produces:
which request joined which slot of which pod at which step and for how many
steps.  That schedule is the advice ``a`` of the cluster constructor, and
the protocol charges its encoded length -- arrival times, the failure
process and the scheduler's decisions are compressed into it.  The tokens
the server streamed are the claimed outputs ``y*``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from veritor.constructors import Join, LMShape, Parameters, Request, Schedule
from veritor.constructors.lm import Decoder

COMPLETE = "complete"
EOS = "eos"
FAILED = "failed"
RUN_END = "run-end"


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """The arrival process, the request distribution, the cluster and its failure process."""

    pods: int
    slots: int
    steps: int
    arrivals: int
    """Requests generated; those arriving too late to be admitted are not part of the run."""
    seed: int
    prompt_lengths: tuple[int, int] = (1, 4)
    """Inclusive range of prompt lengths (uniform)."""
    max_new_lengths: tuple[int, int] = (2, 8)
    """Inclusive range of requested lengths (uniform)."""
    step_seconds: float = 0.05
    """Wall-clock duration of one synchronous decode step."""
    load: float = 1.5
    """Arrival rate as a fraction of the cluster's throughput at the mean requested length.

    Above 1 the queue builds up and first-come, first-served matters; the
    arrivals the window cannot serve are not part of the run.
    """
    eos: int | None = None
    """The end-of-sequence token id; ``None`` for ``vocab - 1``."""
    failure_rate: float = 0.0
    """Probability that a live pod fails at a step."""
    downtime: int = 2
    """Steps a failed pod stays down (the failed step included)."""
    forced_failures: tuple[tuple[int, int], ...] = ()
    """``(pod, step)``: the pod fails at the first step at or after ``step`` at which it has occupants.

    Independent of the random failure process, so a run can be made to
    contain a restart for certain.
    """
    abandon_rate: float = 0.0
    """Probability that the client of a request a pod failure cut off gives up.

    An abandoned request is not re-queued: what it streamed before the
    failure is all it gets (a truncated request, as a client disconnect).
    """

    def __post_init__(self) -> None:
        for name in ("pods", "slots", "steps", "arrivals"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("prompt_lengths", "max_new_lengths"):
            low, high = getattr(self, name)
            if not 1 <= low <= high:
                raise ValueError(
                    f"{name} must be an inclusive range of positive lengths"
                )
        if not 0 <= self.failure_rate < 1:
            raise ValueError("failure_rate must lie in [0, 1)")
        if not 0 <= self.abandon_rate <= 1:
            raise ValueError("abandon_rate must lie in [0, 1]")
        if type(self.downtime) is not int or self.downtime < 1:
            raise ValueError("downtime must be a positive number of steps")
        if self.step_seconds <= 0 or self.load <= 0:
            raise ValueError("step_seconds and load must be positive")


@dataclass(frozen=True, slots=True)
class Arrival:
    """One generated request: when it arrived and, once admitted, its request id in ``x``."""

    index: int
    time: float
    request: Request
    request_id: int | None = None
    """The request's index in the run's ``x``; ``None`` if it was never admitted."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """One join with its outcome: how it ended and which positions it streamed."""

    join: Join
    outcome: str
    """``complete`` (``max_new`` reached), ``eos``, ``failed`` (the pod went down) or ``run-end``."""
    streamed: tuple[int, ...]
    """The positions this attempt streamed (empty when a restart ended before catching up)."""


@dataclass(frozen=True, slots=True)
class Failure:
    pod: int
    step: int
    aborted: tuple[int, ...]
    """The request ids whose attempts the failure cut off."""
    abandoned: tuple[int, ...] = ()
    """Those of ``aborted`` whose clients gave up: truncated, never restarted."""


@dataclass(frozen=True, slots=True)
class Simulation:
    """What the server did: the requests ``x``, the schedule ``a``, the streamed tokens ``y*``."""

    config: WorkloadConfig
    arrivals: tuple[Arrival, ...]
    requests: tuple[Request, ...]
    """``x``: the admitted requests, in arrival order."""
    schedule: Schedule
    attempts: tuple[Attempt, ...]
    """One per join of the schedule, in the schedule's order."""
    failures: tuple[Failure, ...]
    streamed: tuple[tuple[int, ...], ...]
    """Per request, the tokens the user received."""
    occupied: tuple[tuple[int, ...], ...] = field(repr=False)
    """Per pod, per step, the number of occupied slots (``-1`` while the pod is down)."""

    @property
    def eos_stops(self) -> int:
        return sum(attempt.outcome == EOS for attempt in self.attempts)

    @property
    def restarts(self) -> int:
        return len(self.attempts) - len(self.requests)

    @property
    def unserved(self) -> int:
        return sum(arrival.request_id is None for arrival in self.arrivals)

    @property
    def abandoned(self) -> int:
        """Requests truncated by a failure their client did not wait out."""

        return sum(len(failure.abandoned) for failure in self.failures)

    @property
    def tokens(self) -> int:
        return sum(len(tokens) for tokens in self.streamed)

    @property
    def utilization(self) -> float:
        capacity = self.config.pods * self.config.slots * self.config.steps
        return sum(max(count, 0) for pod in self.occupied for count in pod) / capacity


def generate_arrivals(config: WorkloadConfig, shape: LMShape) -> tuple[Arrival, ...]:
    """Poisson arrivals with uniform prompt and requested lengths (and random words when sampling)."""

    rng = random.Random(f"arrivals/{config.seed}")
    mean_new = sum(config.max_new_lengths) / 2
    throughput = (
        config.pods * config.slots / (mean_new * config.step_seconds)
    )  # requests per second
    rate = config.load * throughput
    arrivals: list[Arrival] = []
    clock = 0.0
    for index in range(config.arrivals):
        clock += rng.expovariate(rate)
        prompt = tuple(
            rng.randrange(shape.vocab)
            for _ in range(rng.randint(*config.prompt_lengths))
        )
        max_new = rng.randint(*config.max_new_lengths)
        max_new = min(max_new, shape.context - len(prompt))
        randomness = (
            tuple(rng.randrange(1 << shape.random_bits) for _ in range(max_new))
            if shape.sampling
            else ()
        )
        arrivals.append(Arrival(index, clock, Request(prompt, max_new, randomness)))
    return tuple(arrivals)


@dataclass(slots=True)
class _Running:
    """An attempt in progress: the arrival it serves, its decoder and its progress."""

    arrival: int
    step: int
    slot: int
    decoder: Decoder
    generated: int = 0
    token: int = 0
    streamed: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _Record:
    join: Join
    outcome: str
    streamed: tuple[int, ...]
    arrival: int


def simulate(
    config: WorkloadConfig, shape: LMShape, parameters: Parameters
) -> Simulation:
    """Run the datacenter for ``config.steps`` steps and return everything the verifier will see.

    Each step: pods fail (the random process, plus the injected failures);
    the failed pods' attempts end where they are and their requests go back
    to the head of the queue (unless the client gives up, ``abandon_rate``:
    then the request stays truncated); live pods admit the earliest waiting arrivals
    into their free slots; every occupant generates one token, streamed if
    its position is new for the request; attempts that emitted the EOS token
    or reached ``max_new`` free their slot.  Restarts recompute the positions
    already streamed and must reproduce them: decoding is deterministic.
    """

    if not isinstance(parameters, Parameters) or parameters.shape != shape:
        raise ValueError("parameters must be Parameters of the given shape")
    eos = shape.vocab - 1 if config.eos is None else config.eos
    if not 0 <= eos < shape.vocab:
        raise ValueError("the EOS token must be in the vocabulary")
    rng = random.Random(f"failures/{config.seed}")
    arrivals = generate_arrivals(config, shape)
    forced: dict[int, int] = {}
    for pod, step in config.forced_failures:
        forced[pod] = min(step, forced.get(pod, step))
    pending = list(range(len(arrivals)))  # arrival indices, by arrival time
    running: dict[tuple[int, int], _Running] = {}
    down_until = [0] * config.pods
    records: list[_Record] = []
    failures: list[Failure] = []
    streamed: dict[int, list[int]] = {}
    occupied = [[0] * config.steps for _ in range(config.pods)]

    def finish(item: _Running, pod: int, step: int, outcome: str) -> None:
        length = step - item.step
        assert length >= 1
        records.append(
            _Record(
                Join(pod, item.step, item.slot, item.arrival, length),
                outcome,
                tuple(item.streamed),
                item.arrival,
            )
        )

    for step in range(config.steps):
        now = step * config.step_seconds
        for pod in range(config.pods):
            if down_until[pod] > step:
                occupied[pod][step] = -1
                continue
            busy = any((pod, slot) in running for slot in range(config.slots))
            fails = config.failure_rate > 0 and rng.random() < config.failure_rate
            if busy and forced.get(pod, config.steps) <= step:
                fails = True
                del forced[pod]
            if fails:
                aborted: list[int] = []
                abandoned: list[int] = []
                for slot in range(config.slots):
                    item = running.pop((pod, slot), None)
                    if item is not None:
                        finish(item, pod, step, FAILED)
                        aborted.append(item.arrival)
                        if (
                            config.abandon_rate > 0
                            and rng.random() < config.abandon_rate
                        ):
                            abandoned.append(item.arrival)
                failures.append(Failure(pod, step, tuple(aborted), tuple(abandoned)))
                pending = sorted(  # re-queued at their original arrival time
                    pending + [index for index in aborted if index not in abandoned]
                )
                down_until[pod] = step + config.downtime
                occupied[pod][step] = -1
                continue
            for slot in range(config.slots):
                if (
                    (pod, slot) in running
                    or not pending
                    or arrivals[pending[0]].time > now
                ):
                    continue
                index = pending.pop(0)
                running[(pod, slot)] = _Running(index, step, slot, Decoder(parameters))
            for slot in range(config.slots):
                item = running.get((pod, slot))
                if item is None:
                    continue
                occupied[pod][step] += 1
                request = arrivals[item.arrival].request
                r = request.randomness[item.generated] if shape.sampling else None
                if item.generated == 0:
                    for token in request.prompt[:-1]:
                        item.decoder.logits(token)
                    item.token = item.decoder.forward(request.prompt[-1], r)
                else:
                    item.token = item.decoder.forward(item.token, r)
                seen = streamed.setdefault(item.arrival, [])
                if item.generated < len(seen):
                    assert seen[item.generated] == item.token, (
                        "a restart reproduces the tokens it recomputes"
                    )
                else:
                    seen.append(item.token)
                    item.streamed.append(item.generated)
                item.generated += 1
                if item.token == eos or item.generated == request.max_new:
                    finish(item, pod, step + 1, EOS if item.token == eos else COMPLETE)
                    del running[(pod, slot)]
    for (pod, _slot), item in running.items():
        finish(item, pod, config.steps, RUN_END)

    admitted = sorted({record.arrival for record in records})
    request_id = {index: rank for rank, index in enumerate(admitted)}
    records.sort(
        key=lambda record: (record.join.pod, record.join.step, record.join.slot)
    )
    joins = tuple(
        Join(r.join.pod, r.join.step, r.join.slot, request_id[r.arrival], r.join.length)
        for r in records
    )
    schedule = Schedule(config.pods, config.slots, config.steps, joins)
    requests = tuple(arrivals[index].request for index in admitted)
    schedule.validate(requests)
    return Simulation(
        config=config,
        arrivals=tuple(
            Arrival(a.index, a.time, a.request, request_id.get(a.index))
            for a in arrivals
        ),
        requests=requests,
        schedule=schedule,
        attempts=tuple(
            Attempt(join, r.outcome, r.streamed)
            for join, r in zip(joins, records, strict=True)
        ),
        failures=tuple(
            Failure(
                f.pod,
                f.step,
                tuple(request_id[a] for a in f.aborted),
                tuple(request_id[a] for a in f.abandoned),
            )
            for f in failures
        ),
        streamed=tuple(tuple(streamed[index]) for index in admitted),
        occupied=tuple(tuple(row) for row in occupied),
    )


def check_against_reference(
    simulation: Simulation, reference: Sequence[Sequence[int]]
) -> None:
    """Every streamed token sequence is a prefix of the reference decoding of its request."""

    for request, (tokens, expected) in enumerate(
        zip(simulation.streamed, reference, strict=True)
    ):
        if tuple(tokens) != tuple(expected[: len(tokens)]):
            raise AssertionError(
                f"request {request} streamed {tokens}, the reference is {expected}"
            )


__all__ = [
    "COMPLETE",
    "EOS",
    "FAILED",
    "RUN_END",
    "Arrival",
    "Attempt",
    "Failure",
    "Simulation",
    "WorkloadConfig",
    "check_against_reference",
    "generate_arrivals",
    "simulate",
]
