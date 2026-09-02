"""Continual-batching schedules: the advice an inference cluster hands the compiler.

A cluster runs ``pods`` replicas for ``steps`` synchronous decode steps; each
pod has ``slots`` batch positions.  A request *joins* a slot at some step
(its prompt is prefilled and its first token generated) and then occupies the
slot for one generated token per step until it has produced ``max_new``
tokens, the slot is reassigned, or the run ends.  Everything about a run's
structure that is not fixed by the public requests -- which pod, which slot,
which step -- is therefore captured by the list of joins, and that list is
what the cluster constructor takes as its advice ``a``.  The scheduler that
produces it (:func:`schedule_fcfs`) is the client's choice and outside the
trust boundary; the compiler only needs the schedule to be well formed, and
the protocol charges its encoded length.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_MAGIC = b"veritor/schedule/v1\0"
_WORD = 4


class ScheduleError(ValueError):
    """A schedule that is malformed or inconsistent with its requests."""


@dataclass(frozen=True, slots=True)
class Request:
    """A public request: prompt token ids and the number of tokens wanted."""

    prompt: tuple[int, ...]
    max_new: int

    def __post_init__(self) -> None:
        if type(self.prompt) is not tuple or not self.prompt:
            raise ScheduleError("a request needs a nonempty tuple of prompt tokens")
        if any(type(token) is not int or token < 0 for token in self.prompt):
            raise ScheduleError("prompt tokens must be nonnegative integers")
        if type(self.max_new) is not int or self.max_new < 1:
            raise ScheduleError("max_new must be a positive integer")


@dataclass(frozen=True, slots=True, order=True)
class Join:
    """Request ``request`` enters ``slot`` of ``pod`` at ``step``."""

    pod: int
    step: int
    slot: int
    request: int


@dataclass(frozen=True, slots=True)
class Occupant:
    """A slot's state at one step: which request and how many tokens it has generated."""

    slot: int
    request: int
    generated: int
    """Tokens generated before this step: ``0`` at the join (prefill) step."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """The joins of a run, sorted by ``(pod, step, slot)``; canonical bytes on demand.

    Occupancy is derived: a join holds its slot for ``min(max_new, next join in
    the same slot - step, steps - step)`` steps, so no explicit leave events
    are needed and a schedule can never double-book a slot.  Requests whose
    generation is cut short by a reassignment or the end of the run simply
    produce fewer tokens; that choice is the client's and is paid for by the
    encoded length of the schedule.
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
        previous: Join | None = None
        for join in self.joins:
            if type(join) is not Join:
                raise ScheduleError("joins must be Join instances")
            for field, limit in (
                ("pod", self.pods),
                ("step", self.steps),
                ("slot", self.slots),
                ("request", 1 << (8 * _WORD)),
            ):
                value = getattr(join, field)
                if type(value) is not int or not 0 <= value < limit:
                    raise ScheduleError(f"join {field} {value!r} is out of range")
            if previous is not None and (previous.pod, previous.step, previous.slot) >= (
                join.pod,
                join.step,
                join.slot,
            ):
                raise ScheduleError("joins must be strictly sorted by (pod, step, slot)")
            previous = join

    # -- canonical bytes ----------------------------------------------------------

    def encode(self) -> bytes:
        """``magic | pods | slots | steps | count | joins*``, all big-endian ``u32``."""

        out = bytearray(_MAGIC)
        for value in (self.pods, self.slots, self.steps, len(self.joins)):
            out.extend(value.to_bytes(_WORD, "big"))
        for join in self.joins:
            for value in (join.pod, join.step, join.slot, join.request):
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
        if len(rest) != count * 4 * _WORD:
            raise ScheduleError("schedule length does not match its join count")
        joins = tuple(
            Join(*(int.from_bytes(rest[i + k * _WORD : i + (k + 1) * _WORD], "big") for k in range(4)))
            for i in range(0, len(rest), 4 * _WORD)
        )
        schedule = cls(pods, slots, steps, joins)
        if schedule.encode() != data:
            raise ScheduleError("schedule bytes are not canonical")
        return schedule

    # -- derived occupancy ----------------------------------------------------------

    def validate(self, requests: Sequence[Request]) -> None:
        """Every request joins exactly once and names an existing request."""

        seen: set[int] = set()
        for join in self.joins:
            if join.request >= len(requests):
                raise ScheduleError(f"join names unknown request {join.request}")
            if join.request in seen:
                raise ScheduleError(f"request {join.request} joins more than once")
            seen.add(join.request)
        if len(seen) != len(requests):
            missing = sorted(set(range(len(requests))) - seen)
            raise ScheduleError(f"requests never scheduled: {missing}")

    def active_steps(self, requests: Sequence[Request]) -> dict[int, int]:
        """Steps each request is active for, which is the number of tokens it generates."""

        self.validate(requests)
        following: dict[tuple[int, int], int] = {}
        active: dict[int, int] = {}
        for join in reversed(self.joins):
            key = (join.pod, join.slot)
            window = following.get(key, self.steps) - join.step
            following[key] = join.step
            active[join.request] = min(requests[join.request].max_new, window)
        return active

    def occupancy(self, requests: Sequence[Request]) -> dict[tuple[int, int], tuple[Occupant, ...]]:
        """``(pod, step) -> occupants`` in slot order, for every step with an active slot."""

        active = self.active_steps(requests)
        table: dict[tuple[int, int], list[Occupant]] = {}
        for join in self.joins:
            for offset in range(active[join.request]):
                table.setdefault((join.pod, join.step + offset), []).append(
                    Occupant(join.slot, join.request, offset)
                )
        return {key: tuple(sorted(items, key=lambda o: o.slot)) for key, items in table.items()}


def schedule_fcfs(requests: Sequence[Request], pods: int, slots: int, steps: int) -> Schedule:
    """First-come, first-served continual batching.

    Requests are admitted in index order; at each step every pod fills its
    free slots (lowest pod, lowest slot first) from the head of the queue.  A
    slot is free again the step after its request has generated ``max_new``
    tokens.  Requests still queued when the run ends are not scheduled, and
    :meth:`Schedule.validate` will say so.
    """

    for request in requests:
        if type(request) is not Request:
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
                joins.append(Join(pod, step, slot, request))
                free_at[(pod, slot)] = step + requests[request].max_new
    return Schedule(pods, slots, steps, tuple(sorted(joins)))
