"""Continual-batching schedules: the advice an inference cluster hands the compiler.

A cluster runs ``pods`` replicas for ``steps`` synchronous decode steps; each
pod has ``slots`` batch positions.  A request *joins* a slot at some step
(its prompt is prefilled and its first token generated) and then occupies the
slot for one generated token per step, for exactly ``length`` steps: until it
has produced ``max_new`` tokens, emitted an end-of-sequence token, the run
ended, or the pod failed under it.  Everything about a run's structure that
is not fixed by the public requests -- which pod, which slot, which step, how
many steps -- is therefore captured by the list of joins, and that list is
what the cluster constructor takes as its advice ``a``.  The scheduler that
produces it (:func:`schedule_fcfs`, or the simulated datacenter of
:mod:`veritor.simulation`) is the client's choice and outside the trust boundary;
the compiler only needs the schedule to be well formed, and the protocol
charges its encoded length.

A request may join more than once: when a pod fails, the tokens its
occupants had already streamed stand, and each occupant is restarted from
its prefill by a later join (on whichever pod is free).  A fresh attempt
recomputes the request from position ``0``; the positions an earlier attempt
already streamed are recomputed but not streamed again, so the run's outputs
for a request are the first positions of its decoding, each produced by
exactly one attempt.

Two more shapes of attempt are the other things production schedulers do
with a slot.  A join with ``resume`` continues the request's latest attempt
from the KV cache it left (a swap back in after a preemption, or the decode
half of a request prefilled on another pod): its steps are all decode steps
reading the cache that attempt declared, however many steps ago and on
whichever pod.  A join with ``chunk > 0`` prefills its prompt ``chunk``
tokens per step (chunked prefill), over ``ceil(prompt / chunk)`` steps of
which the last also generates the first token; each chunk reads the cache
the earlier chunks declared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

_MAGIC = b"veritor/schedule/v3\0"
_WORD = 4
_JOIN_WORDS = 7


class ScheduleError(ValueError):
    """A schedule that is malformed or inconsistent with its requests."""


@dataclass(frozen=True, slots=True)
class Request:
    """A public request: prompt token ids, the number of tokens wanted, and the randomness.

    ``randomness`` holds the public random word of every generated position
    when the model samples (:attr:`~veritor.constructors.lm.LMShape.sampling`);
    it is empty for a model that takes the argmax.  It is part of ``x``: the
    server publishes it, so it is a public input, not a covert degree of
    freedom.  ``banned`` is the request's constrained-decoding list: token
    ids the head may not emit, public too (they enter the circuit as ``in``
    gates and become a mask before the argmax or the sampler).
    """

    prompt: tuple[int, ...]
    max_new: int
    randomness: tuple[int, ...] = ()
    banned: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.prompt) is not tuple or not self.prompt:
            raise ScheduleError("a request needs a nonempty tuple of prompt tokens")
        if any(type(token) is not int or token < 0 for token in self.prompt):
            raise ScheduleError("prompt tokens must be nonnegative integers")
        if type(self.max_new) is not int or self.max_new < 1:
            raise ScheduleError("max_new must be a positive integer")
        if type(self.randomness) is not tuple or any(
            type(word) is not int or word < 0 for word in self.randomness
        ):
            raise ScheduleError("randomness must be a tuple of nonnegative integers")
        if self.randomness and len(self.randomness) != self.max_new:
            raise ScheduleError("randomness must hold one word per generated position")
        if type(self.banned) is not tuple or any(
            type(token) is not int or token < 0 for token in self.banned
        ):
            raise ScheduleError("banned must be a tuple of nonnegative token ids")
        if len(set(self.banned)) != len(self.banned):
            raise ScheduleError("banned tokens must be distinct")


@dataclass(frozen=True, slots=True, order=True)
class Join:
    """Request ``request`` enters ``slot`` of ``pod`` at ``step`` and holds it for ``length`` steps.

    A fresh attempt (``resume`` false) prefills the prompt first: at the
    join step when ``chunk`` is ``0``, otherwise ``chunk`` tokens per step
    over ``ceil(prompt / chunk)`` steps; the step that completes the prompt
    generates the request's first token and each following step one more.
    A resumed attempt (``resume`` true, ``chunk`` ``0``) continues the
    request's latest attempt from its cache: every step is a decode step.
    """

    pod: int
    step: int
    slot: int
    request: int
    length: int
    resume: bool = False
    chunk: int = 0

    def __post_init__(self) -> None:
        if type(self.resume) is not bool:
            raise ScheduleError("resume must be a bool")
        if type(self.chunk) is not int or self.chunk < 0:
            raise ScheduleError("chunk must be a nonnegative integer")
        if self.resume and self.chunk:
            raise ScheduleError("a resumed attempt has nothing to prefill")


@dataclass(frozen=True, slots=True)
class Span:
    """What one join computes for its request: its prefill steps and the positions it generates."""

    prefill_steps: int
    """Steps spent on the prompt: ``1``, ``ceil(prompt / chunk)`` when chunked, ``0`` for a resume."""
    start: int
    """The first position the attempt generates: ``0`` for a fresh attempt."""
    end: int
    """One past the last position it generates (``start`` when it is cut off during its prefill)."""


@dataclass(frozen=True, slots=True)
class Occupant:
    """A slot's state at one step: which request, its progress, and which join placed it."""

    slot: int
    request: int
    generated: int
    """The request's tokens before this step: the position a decode step generates.

    During a prefill it is the position the attempt will generate first
    (``0`` for a fresh attempt); a resumed attempt starts where the attempt
    it continues stopped.
    """
    join: int
    """The index in :attr:`Schedule.joins` of the attempt this occupant belongs to."""
    prefilled: int = 0
    """Prompt positions already in the slot's cache: ``0`` at a fresh join, the
    prompt length once the prompt is prefilled, a multiple of ``chunk`` between."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """The joins of a run, sorted by ``(pod, step, slot)``; canonical bytes on demand.

    Well-formedness is structural and checked at construction: every join is
    in range, ends within the run, and no two joins overlap in a slot, so a
    schedule can never double-book a slot.  :meth:`validate` checks it against
    the requests: every request joins at least once, no attempt generates
    past ``max_new``, a request's attempts never overlap in time -- a
    restart begins no earlier than the step at which the attempt it replaces
    was cut off -- and a resumed attempt has a cache to continue.  Requests
    whose generation is cut short by a failure, an early stop or the end of
    the run simply produce fewer tokens; every such choice is the client's
    and is paid for by the encoded length of the schedule.
    """

    pods: int
    slots: int
    steps: int
    joins: tuple[Join, ...]

    def __post_init__(self) -> None:
        for name in ("pods", "slots", "steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1 or value >= 1 << (8 * _WORD):
                raise ScheduleError(f"{name} must be a positive integer below 2**32")
        if type(self.joins) is not tuple:
            raise ScheduleError("joins must be a tuple")
        self._check_joins()

    def _check_joins(self) -> None:
        previous: Join | None = None
        busy_until: dict[tuple[int, int], int] = {}
        for join in self.joins:
            if type(join) is not Join:
                raise ScheduleError("joins must be Join instances")
            for field, limit in (
                ("pod", self.pods),
                ("step", self.steps),
                ("slot", self.slots),
                ("request", 1 << (8 * _WORD)),
                ("chunk", 1 << (8 * _WORD)),
            ):
                value = getattr(join, field)
                if type(value) is not int or not 0 <= value < limit:
                    raise ScheduleError(f"join {field} {value!r} is out of range")
            if type(join.length) is not int or join.length < 1 or join.step + join.length > self.steps:
                raise ScheduleError(f"join length {join.length!r} does not fit the run")
            if previous is not None and (previous.pod, previous.step, previous.slot) >= (
                join.pod,
                join.step,
                join.slot,
            ):
                raise ScheduleError("joins must be strictly sorted by (pod, step, slot)")
            previous = join
            key = (join.pod, join.slot)
            if busy_until.get(key, 0) > join.step:
                raise ScheduleError(f"slot {join.slot} of pod {join.pod} is double-booked at step {join.step}")
            busy_until[key] = join.step + join.length

    # -- canonical bytes ----------------------------------------------------------

    def encode(self) -> bytes:
        """``magic | pods | slots | steps | count | joins*``, all big-endian ``u32``.

        A join is ``pod | step | slot | request | length | resume | chunk``.
        """

        out = bytearray(_MAGIC)
        for value in (self.pods, self.slots, self.steps, len(self.joins)):
            out.extend(value.to_bytes(_WORD, "big"))
        for join in self.joins:
            words = (join.pod, join.step, join.slot, join.request, join.length, int(join.resume), join.chunk)
            for value in words:
                out.extend(value.to_bytes(_WORD, "big"))
        return bytes(out)

    @classmethod
    def decode(cls, data: bytes) -> Schedule:
        """Parse canonical bytes; anything but the exact encoding of a schedule fails."""

        if type(data) is not bytes or not data.startswith(_MAGIC):
            raise ScheduleError("not a schedule")
        body = memoryview(data)[len(_MAGIC) :]
        if len(body) < 4 * _WORD:
            raise ScheduleError("truncated schedule header")
        words = [int.from_bytes(body[i : i + _WORD], "big") for i in range(0, 4 * _WORD, _WORD)]
        pods, slots, steps, count = words
        rest = body[4 * _WORD :]
        stride = _JOIN_WORDS * _WORD
        if len(rest) != count * stride:
            raise ScheduleError("schedule length does not match its join count")
        joins = []
        for i in range(0, len(rest), stride):
            pod, step, slot, request, length, resume, chunk = (
                int.from_bytes(rest[i + k * _WORD : i + (k + 1) * _WORD], "big") for k in range(_JOIN_WORDS)
            )
            if resume > 1:
                raise ScheduleError("a join's resume flag must be 0 or 1")
            joins.append(Join(pod, step, slot, request, length, bool(resume), chunk))
        schedule = cls(pods, slots, steps, tuple(joins))
        if schedule.encode() != data:
            raise ScheduleError("schedule bytes are not canonical")
        return schedule

    # -- consistency with the requests ------------------------------------------------

    def _checked(self, requests: Sequence[Request]) -> tuple[dict[int, tuple[int, ...]], tuple[Span, ...]]:
        """The attempts per request and the span of every join, checking the schedule against the requests."""

        by_request: dict[int, list[int]] = {}
        for index, join in enumerate(self.joins):
            if join.request >= len(requests):
                raise ScheduleError(f"join names unknown request {join.request}")
            by_request.setdefault(join.request, []).append(index)
        attempts: dict[int, tuple[int, ...]] = {}
        spans: list[Span | None] = [None] * len(self.joins)
        for request in sorted(by_request):
            indices = sorted(by_request[request], key=lambda index: self.joins[index].step)
            for earlier, later in pairwise(indices):
                first, second = self.joins[earlier], self.joins[later]
                if first.step + first.length > second.step:
                    raise ScheduleError(
                        f"request {request} restarts at step {second.step} while its earlier "
                        f"attempt still holds a slot"
                    )
            attempts[request] = tuple(indices)
            prompt, max_new = len(requests[request].prompt), requests[request].max_new
            previous: Span | None = None
            for index in indices:
                join = self.joins[index]
                if join.resume:
                    if previous is None or previous.end == previous.start:
                        raise ScheduleError(
                            f"request {request} resumes at step {join.step} with no cache to continue"
                        )
                    span = Span(0, previous.end, previous.end + join.length)
                else:
                    steps = 1 if join.chunk == 0 else -(-prompt // join.chunk)
                    span = Span(steps, 0, max(0, join.length - steps + 1))
                if span.end > max_new:
                    raise ScheduleError(f"request {request} is scheduled for more than max_new tokens")
                spans[index] = span
                previous = span
        if len(attempts) != len(requests):
            missing = sorted(set(range(len(requests))) - set(attempts))
            raise ScheduleError(f"requests never scheduled: {missing}")
        return attempts, tuple(span for span in spans if span is not None)

    def attempts(self, requests: Sequence[Request]) -> dict[int, tuple[int, ...]]:
        """``request -> join indices`` in step order, after checking the schedule against the requests.

        Every join names an existing request; every request joins at least
        once; a request's attempts do not overlap in time; no attempt
        generates past ``max_new``; and a resumed attempt has a cache to
        continue: the attempt before it generated at least one token.
        """

        return self._checked(requests)[0]

    def spans(self, requests: Sequence[Request]) -> tuple[Span, ...]:
        """Per join, the :class:`Span` it computes, after :meth:`attempts`' checks."""

        return self._checked(requests)[1]

    def validate(self, requests: Sequence[Request]) -> None:
        """Raise :class:`ScheduleError` unless the schedule is consistent with ``requests``."""

        self._checked(requests)

    def streamed_before(self, requests: Sequence[Request]) -> tuple[int, ...]:
        """Per join, the positions earlier attempts of its request already streamed.

        Only the positions from ``streamed_before`` up to the end of its
        span are new outputs of an attempt.  ``0`` for a request's first
        attempt.
        """

        attempts, spans = self._checked(requests)
        before = [0] * len(self.joins)
        for indices in attempts.values():
            streamed = 0
            for index in indices:
                before[index] = streamed
                streamed = max(streamed, spans[index].end)
        return tuple(before)

    def active_steps(self, requests: Sequence[Request]) -> dict[int, int]:
        """Tokens each request streams: the furthest position any of its attempts reaches."""

        attempts, spans = self._checked(requests)
        return {request: max(spans[index].end for index in indices) for request, indices in attempts.items()}

    def occupancy(self, requests: Sequence[Request]) -> dict[tuple[int, int], tuple[Occupant, ...]]:
        """``(pod, step) -> occupants`` in slot order, for every step with an active slot."""

        spans = self.spans(requests)
        table: dict[tuple[int, int], list[Occupant]] = {}
        for index, join in enumerate(self.joins):
            span, prompt = spans[index], len(requests[join.request].prompt)
            for offset in range(join.length):
                if offset < span.prefill_steps:  # chunk ``offset`` of the prompt (the whole prompt at chunk 0)
                    occupant = Occupant(join.slot, join.request, span.start, index, offset * join.chunk)
                else:
                    position = span.start + offset - max(span.prefill_steps - 1, 0)
                    occupant = Occupant(join.slot, join.request, position, index, prompt)
                table.setdefault((join.pod, join.step + offset), []).append(occupant)
        return {key: tuple(sorted(items, key=lambda o: o.slot)) for key, items in table.items()}


def schedule_fcfs(requests: Sequence[Request], pods: int, slots: int, steps: int) -> Schedule:
    """First-come, first-served continual batching.

    Requests are admitted in index order; at each step every pod fills its
    free slots (lowest pod, lowest slot first) from the head of the queue.  A
    request holds its slot for ``max_new`` steps or until the run ends, and
    the slot is free again the step after.  Requests still queued when the
    run ends are not scheduled, and :meth:`Schedule.validate` will say so.
    """

    if any(type(item) is not Request for item in requests):
        raise ScheduleError("requests must be Request instances")
    free_at = {(pod, slot): 0 for pod in range(pods) for slot in range(slots)}
    queue = list(range(len(requests)))
    joins: list[Join] = []
    for step in range(steps):
        for pod in range(pods):
            for slot in range(slots):
                if not queue or free_at[(pod, slot)] > step:
                    continue
                request = queue.pop(0)
                length = min(requests[request].max_new, steps - step)
                joins.append(Join(pod, step, slot, request, length))
                free_at[(pod, slot)] = step + length
    return Schedule(pods, slots, steps, tuple(sorted(joins)))
