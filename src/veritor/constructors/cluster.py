"""The cluster constructor: continual batching of the toy decoder, a schedule as advice.

``ClusterG(shape, pods, slots, steps)`` traces a run of ``pods`` replicas of
the toy decoder (:mod:`veritor.constructors.lm`) for ``steps`` synchronous
decode steps.  The public inputs ``x`` are the requests; the advice ``a`` is
a :class:`~veritor.constructors.schedule.Schedule` -- which request joins
which slot of which pod at which step -- and everything about the circuit
that ``x`` does not fix follows from it.  A bad schedule is the client's
fault: it fails to trace (:class:`TracerError`), so it never compiles.

Structure.  The root has no ports.  It calls the ``weights`` unit once, then
one ``step`` per ``(pod, step)`` with occupants, in that order.  A step is a
replay unit holding its occupants: a *prefill* (the request's prompt tokens
are ``in`` gates inside the step, the first token comes out) or a *decode*
(the previous token and the KV cache come in through ports from earlier
steps' declared outputs, the next token comes out).  The number of occupants
varies from step to step -- that is continual batching -- and two steps with
the same tuple of occupant shapes are the same kind.  Steps of one pod are
chained through the tokens and the KV cache they declare; pods share
nothing but the weights.  For a sampling shape each occupant also reads its
position's public random word, an ``in`` gate inside the step.

Restarts.  A request may join more than once (a pod failed under it): each
attempt is a fresh prefill and its own chain of steps, and the run's outputs
for the request are its first ``max(length)`` positions, each taken from the
attempt that streamed it first (:meth:`Schedule.streamed_before`).  An
attempt's recomputation of positions already streamed is in the circuit --
the server executed it -- but declares no circuit output.

Marks.  "Replay decode step ``t`` of pod ``p``" is the unit a server can be
asked for and explain, so ``step`` is the replay unit (with ``weights``, the
one unit of source gates); the verification units are the row-sized kinds
of :class:`~veritor.constructors.lm.ToyLM`.

Restarts.  A request the schedule joins more than once (a pod failed under
it) is prefilled again by its later attempt; the attempt recomputes the
positions already streamed -- they stay inside the pod's chain of steps,
declared like every token -- and the circuit's outputs for the request are
its positions ``0 .. max(length) - 1``, each taken from the attempt that
streamed it (:meth:`~veritor.constructors.schedule.Schedule.streamed_before`).
The aborted attempt's steps stay in the circuit: their tokens were observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue
from veritor.core.description import REPLAY

from .lm import LMShape, ToyLM, wires
from .schedule import Request, Schedule, ScheduleError
from .tracer import TracedDefinition, TracerError, Wire, Wires

PREFILL = "prefill"
DECODE = "decode"

type OccupantShape = tuple[str, int]
"""``("prefill", prompt length)`` or ``("decode", context length)``."""


@dataclass(slots=True)
class _Slot:
    """What a slot's request has produced so far: its cache blocks per layer and last token."""

    request: int
    keys: list[list[Wires]]
    values: list[list[Wires]]
    token: Wire | None = None

    @staticmethod
    def fresh(request: int, layers: int) -> _Slot:
        return _Slot(request, [[] for _ in range(layers)], [[] for _ in range(layers)])


@dataclass(frozen=True, slots=True)
class _Plan:
    """A validated request tuple and schedule with the derived tables every method needs."""

    requests: tuple[Request, ...]
    schedule: Schedule
    active: dict[int, int] = field(compare=False)
    """Per request, the positions the run streams (its outputs): the longest of its attempts."""
    streamed_before: tuple[int, ...] = field(compare=False)
    """Per join, the positions earlier attempts of its request already streamed."""
    occupancy: dict[tuple[int, int], tuple] = field(compare=False)


class ClusterG:
    """``G(x, a)`` for a cluster run: ``x`` the requests, ``a`` the encoded schedule.

    A :class:`veritor.compile.Constructor`: ``digest`` names the class, its
    version and ``(shape, pods, slots, steps)``; ``G(x, a)`` returns the
    description bytes and the public inputs -- the prompt tokens and, for a
    sampling shape, the random words -- as the ``in`` gates consume them.
    """

    VERSION = "2"

    def __init__(self, shape: LMShape, pods: int, slots: int, steps: int) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        for name, value in (("pods", pods), ("slots", slots), ("steps", steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.shape = shape
        self.pods, self.slots, self.steps = pods, slots, steps
        self.lm = ToyLM(shape)
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"pods": self.pods, "shape": self.shape.manifest, "slots": self.slots, "steps": self.steps}

    # -- validation -----------------------------------------------------------------

    def _requests(self, x: object) -> tuple[Request, ...]:
        if type(x) is not tuple or not x or any(type(item) is not Request for item in x):
            raise TracerError("ClusterG expects a nonempty tuple of Request")
        for index, request in enumerate(x):
            if any(token >= self.shape.vocab for token in request.prompt):
                raise TracerError(f"request {index} has a prompt token outside the vocabulary")
            try:
                self.shape.check_randomness(request)
            except ValueError as error:
                raise TracerError(f"request {index}: {error}") from error
        return x

    def _plan(self, x: object, schedule: object) -> _Plan:
        requests = self._requests(x)
        if not isinstance(schedule, Schedule):
            raise TracerError("the advice must decode to a Schedule")
        if (schedule.pods, schedule.slots, schedule.steps) != (self.pods, self.slots, self.steps):
            raise TracerError("the schedule is for another cluster (pods, slots, steps)")
        try:
            active = schedule.active_steps(requests)
            before = schedule.streamed_before(requests)
        except ScheduleError as error:
            raise TracerError(f"bad schedule: {error}") from error
        for index, request in enumerate(requests):
            if len(request.prompt) + active[index] > self.shape.context:
                raise TracerError(
                    f"request {index} needs {len(request.prompt) + active[index]} positions; "
                    f"the context is {self.shape.context}"
                )
        return _Plan(requests, schedule, active, before, schedule.occupancy(requests))

    def _decode_advice(self, a: object) -> Schedule:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        try:
            return Schedule.decode(a)
        except ScheduleError as error:
            raise TracerError(f"malformed advice: {error}") from error

    # -- layouts ---------------------------------------------------------------------

    def output_layout(self, x: object, schedule: Schedule) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order.

        A request streams ``max(length)`` positions over its attempts; each
        position is the output of the one attempt that streamed it.
        """

        plan = self._plan(x, schedule)
        return tuple((r, g) for r in range(len(plan.requests)) for g in range(plan.active[r]))

    def flatten_inputs(self, x: object, schedule: Schedule) -> tuple[int, ...]:
        """The public inputs in ``in``-gate address order: by ``(pod, step)``, then slot.

        A prefill occupant contributes its prompt tokens (then, for a sampling
        shape, the random word of position ``0``); a decode occupant at
        position ``g`` contributes the random word of position ``g``.
        """

        plan = self._plan(x, schedule)
        values: list[int] = []
        for key in sorted(plan.occupancy):
            for occupant in plan.occupancy[key]:
                request = plan.requests[occupant.request]
                if occupant.generated == 0:
                    values.extend(request.prompt)
                if self.shape.sampling:
                    values.append(request.randomness[occupant.generated])
        return tuple(values)

    # -- kinds -----------------------------------------------------------------------

    def _decode_ports(self, c: int) -> int:
        """Ports a decode occupant adds to its step: the token and the cache of ``c - 1``."""

        return 1 + self.shape.state_size(c - 1)

    def step(self, shapes: tuple[OccupantShape, ...]) -> TracedDefinition:
        """One decode step of one pod: its occupants over the shared weights.

        Ports: the weights, then per decode occupant its token and cache.
        Outputs: each occupant's new cache entries and token, in slot order.
        """

        if not shapes:
            raise TracerError("a step needs at least one occupant")
        weights = self.shape.weight_count
        extra = sum(self._decode_ports(size) for kind, size in shapes if kind == DECODE)

        @self.lm.tracer.definition(input_count=weights + extra, key=("step", shapes), role=REPLAY)
        def step(v: Wires) -> object:
            w, cursor, outputs = v[:weights], weights, []
            for kind, size in shapes:
                if kind == PREFILL:
                    outputs.append(self.lm.prefill(size)(w))
                else:
                    ports = self._decode_ports(size)
                    outputs.append(self.lm.decode(size)(w, v[cursor : cursor + ports]))
                    cursor += ports
            return outputs

        return step

    def root(self, plan: _Plan) -> TracedDefinition:
        """The run: the weights, then every step of every pod in order, wired through the caches."""

        shape, layers, d = self.shape, self.shape.layers, self.shape.d_model
        requests, occupancy, before = plan.requests, plan.occupancy, plan.streamed_before

        @self.lm.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = wires(self.lm.weights_unit()())
            slots: dict[tuple[int, int], _Slot] = {}
            tokens: dict[tuple[int, int], Wire] = {}
            for key in sorted(occupancy):
                pod, occupants = key[0], occupancy[key]
                shapes: list[OccupantShape] = []
                args: list[Wire | Wires] = [w]
                for occupant in occupants:
                    prompt = len(requests[occupant.request].prompt)
                    if occupant.generated == 0:
                        shapes.append((PREFILL, prompt))
                        continue
                    slot = slots[(pod, occupant.slot)]
                    assert slot.request == occupant.request and slot.token is not None
                    shapes.append((DECODE, prompt + occupant.generated))
                    args.append(slot.token)
                    for layer in range(layers):
                        args.extend(slot.keys[layer])
                        args.extend(slot.values[layer])
                outputs = wires(self.step(tuple(shapes))(*args))
                cursor = 0
                for occupant, (kind, size) in zip(occupants, shapes, strict=True):
                    positions = size if kind == PREFILL else 1
                    produced = shape.state_size(positions) + 1
                    block = outputs[cursor : cursor + produced]
                    cursor += produced
                    if kind == PREFILL:
                        slot = slots[(pod, occupant.slot)] = _Slot.fresh(occupant.request, layers)
                    else:
                        slot = slots[(pod, occupant.slot)]
                    for layer in range(layers):
                        start = 2 * layer * positions * d
                        slot.keys[layer].append(block[start : start + positions * d])
                        slot.values[layer].append(block[start + positions * d : start + 2 * positions * d])
                    token = block[-1]
                    assert isinstance(token, Wire)
                    slot.token = token
                    if occupant.generated >= before[occupant.join]:  # streamed by this attempt
                        key = (occupant.request, occupant.generated)
                        assert key not in tokens, "a position is streamed by one attempt only"
                        tokens[key] = token
            return [tokens[key] for key in self.output_layout(requests, plan.schedule)]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        plan = self._plan(x, self._decode_advice(a))
        description = self.lm.tracer.serialize(self.root(plan))
        return description, self.flatten_inputs(plan.requests, plan.schedule)


__all__ = ["DECODE", "PREFILL", "ClusterG", "OccupantShape"]
