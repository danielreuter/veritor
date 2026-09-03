"""The cluster constructor: continual batching of the toy decoder, a schedule as advice.

``ClusterG(shape, pods, slots, steps)`` traces a run of ``pods`` replicas of
the toy decoder (:mod:`veritor.constructors.lm`) for ``steps`` synchronous
decode steps.  The public inputs ``x`` are the requests; the advice ``a`` is
a :class:`~veritor.constructors.schedule.Schedule` -- which request joins
which slot of which pod at which step -- and everything about the circuit
that ``x`` does not fix follows from it.  A bad schedule is the client's
fault: it fails to trace (:class:`TracerError`), so it never compiles.

Structure.  The root has no ports.  It calls the ``weights`` unit once, then
one ``step`` per ``(pod, step)`` with occupants, in time order (step, then
pod: the cluster is synchronous, and a step may read what any earlier step
of any pod declared).  A step is a replay unit holding its occupants, each
one of three shapes: a *prefill* (the prompt tokens are ``in`` gates inside
the step, the first token comes out), a *chunk* (some prompt tokens, no
token out: chunked prefill) or a *decode* (the previous token and the KV
cache come in through ports from earlier steps' declared outputs, the next
token comes out).  The number of occupants varies from step to step -- that
is continual batching -- and two steps with the same tuple of occupant
shapes are the same kind.  For a sampling shape each occupant that
generates a token also reads its position's public random word, an ``in``
gate inside the step.

Attempts.  A request's steps are chained through the tokens and the KV cache
they declare, on whichever pods the schedule puts them: a fresh join starts
from its prompt, a resumed join (:attr:`Join.resume`) continues the cache of
the request's latest attempt -- declared by steps possibly many steps
earlier and on another pod, as after a preemption that swapped the cache
out or a prefill on one pod decoded on another.  A request the schedule
joins afresh more than once (a pod failed under it) is prefilled again and
recomputes the positions already streamed; they stay in the circuit,
declared like every token, but the circuit's outputs for the request are
its positions ``0 .. max(end) - 1``, each taken from the attempt that
streamed it (:meth:`~veritor.constructors.schedule.Schedule.streamed_before`).
An aborted attempt's steps stay in the circuit: their tokens were observed.

Marks.  "Replay decode step ``t`` of pod ``p``" is the unit a server can be
asked for and explain, so ``step`` is the replay unit (with ``weights``, the
one unit of source gates); the verification units are the row-sized kinds
of :class:`~veritor.constructors.lm.ToyLM`.

A heterogeneous fleet.  With ``arches`` (one architecture name per pod) the
gate set is the union of one namespaced copy of the toy ISA per
architecture (:func:`~veritor.core.gates.union_gate_set`), each pod's steps
are traced with its architecture's gates, and a step's kind carries the
architecture: the same occupants on two architectures are two kinds.  The
weights, the caches and the tokens are shared as before -- ``in`` and
``weight`` gates are one environment -- so a request may be prefilled on
one architecture and decoded on another.

Constrained decoding (:attr:`Request.banned`) is served by
:class:`~veritor.constructors.requests.RequestsG` only: a step would have to
declare each occupant's mask as outputs and read it back through ports,
which the step kinds do not carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue, make_isa_gate_set
from veritor.core.description import REPLAY
from veritor.core.gates import union_gate_set

from .lm import LMShape, ToyLM
from .schedule import Join, Occupant, Request, Schedule, ScheduleError
from .tracer import TracedDefinition, Tracer, TracerError, Wire, Wires

FLEET_GATE_SET = ("veritor.toy-isa-fleet", "1")
"""Name and version of the union gate set of a heterogeneous cluster."""

PREFILL = "prefill"
CHUNK = "chunk"
DECODE = "decode"

type OccupantShape = tuple[str, int, int]
"""``(kind, positions, cached)``: ``("prefill", n, c)`` completes a prompt with ``n``
tokens over ``c`` cached prompt positions and generates the first token;
``("chunk", n, c)`` processes ``n`` prompt tokens over ``c`` without generating;
``("decode", 1, c)`` generates the next token over a cache of ``c`` positions."""


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
    """Per request, the positions the run streams (its outputs): the furthest any attempt reaches."""
    streamed_before: tuple[int, ...] = field(compare=False)
    """Per join, the positions earlier attempts of its request already streamed."""
    occupancy: dict[tuple[int, int], tuple[Occupant, ...]] = field(compare=False)
    order: tuple[tuple[int, int], ...] = field(compare=False)
    """The ``(pod, step)`` keys of ``occupancy`` in time order: by step, then pod."""

    def shape_of(self, occupant: Occupant) -> OccupantShape:
        """What the occupant's step computes for it, from its progress and its join."""

        prompt, join = len(self.requests[occupant.request].prompt), self.schedule.joins[occupant.join]
        if occupant.prefilled < prompt:
            left = prompt - occupant.prefilled
            positions = left if join.chunk == 0 else min(join.chunk, left)
            return (PREFILL if positions == left else CHUNK, positions, occupant.prefilled)
        return (DECODE, 1, prompt + occupant.generated - 1)


class ClusterG:
    """``G(x, a)`` for a cluster run: ``x`` the requests, ``a`` the encoded schedule.

    A :class:`veritor.compile.Constructor`: ``digest`` names the class, its
    version and ``(shape, pods, slots, steps)`` (and ``arches`` for a
    fleet); ``G(x, a)`` returns the description bytes and the public inputs
    -- the prompt tokens and, for a sampling shape, the random words -- as
    the ``in`` gates consume them.  ``gate_set`` is the Σ its descriptions
    are written over.
    """

    VERSION = "3"

    def __init__(
        self,
        shape: LMShape,
        pods: int,
        slots: int,
        steps: int,
        *,
        arches: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        for name, value in (("pods", pods), ("slots", slots), ("steps", steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.shape = shape
        self.pods, self.slots, self.steps = pods, slots, steps
        self.arches = arches
        if arches is None:
            self.lm = ToyLM(shape)
            self.models: dict[str | None, ToyLM] = {None: self.lm}
        else:
            if type(arches) is not tuple or len(arches) != pods or not all(type(a) is str for a in arches):
                raise ValueError("arches must name one architecture per pod")
            members = {arch: make_isa_gate_set(shape.width) for arch in sorted(set(arches))}
            tracer = Tracer(union_gate_set(members, name=FLEET_GATE_SET[0], version=FLEET_GATE_SET[1]))
            self.models = {arch: ToyLM(shape, tracer=tracer, namespace=arch) for arch in members}
            self.lm = self.models[arches[0]]
        self.gate_set = self.lm.tracer.gate_set
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {
            "pods": self.pods,
            "shape": self.shape.manifest,
            "slots": self.slots,
            "steps": self.steps,
        }
        if self.arches is not None:
            manifest["arches"] = list(self.arches)
        return manifest

    # -- validation -----------------------------------------------------------------

    def _requests(self, x: object) -> tuple[Request, ...]:
        if type(x) is not tuple or not x or any(type(item) is not Request for item in x):
            raise TracerError("ClusterG expects a nonempty tuple of Request")
        for index, request in enumerate(x):
            if any(token >= self.shape.vocab for token in request.prompt):
                raise TracerError(f"request {index} has a prompt token outside the vocabulary")
            if request.banned:
                raise TracerError(
                    f"request {index} bans tokens: ClusterG steps carry no mask; use RequestsG"
                )
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
        occupancy = schedule.occupancy(requests)
        order = tuple(sorted(occupancy, key=lambda key: (key[1], key[0])))
        return _Plan(requests, schedule, active, before, occupancy, order)

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
        """The public inputs in ``in``-gate address order: by step, then pod, then slot.

        A prefill or chunk occupant contributes the prompt tokens it
        processes; every occupant that generates a token (a prefill or a
        decode at position ``g``) then contributes, for a sampling shape, the
        random word of position ``g``.
        """

        plan = self._plan(x, schedule)
        values: list[int] = []
        for key in plan.order:
            for occupant in plan.occupancy[key]:
                request = plan.requests[occupant.request]
                kind, positions, cached = plan.shape_of(occupant)
                if kind != DECODE:
                    values.extend(request.prompt[cached : cached + positions])
                if self.shape.sampling and kind != CHUNK:
                    values.append(request.randomness[occupant.generated])
        return tuple(values)

    # -- kinds -----------------------------------------------------------------------

    def _ports(self, occupant: OccupantShape) -> int:
        """Ports an occupant adds to its step: its cache, and for a decode its token."""

        kind, _, cached = occupant
        return self.shape.state_size(cached) + (1 if kind == DECODE else 0)

    def _produced(self, occupant: OccupantShape) -> int:
        """Outputs an occupant adds to its step: its new cache entries, and a token unless it is a chunk."""

        kind, positions, _ = occupant
        return self.shape.state_size(positions) + (0 if kind == CHUNK else 1)

    def step(self, shapes: tuple[OccupantShape, ...], arch: str | None = None) -> TracedDefinition:
        """One decode step of one pod: its occupants over the shared weights.

        Ports: the weights, then per occupant its cache (a decode's token
        first).  Outputs: each occupant's new cache entries and token, in
        slot order.  ``arch`` is the pod's architecture in a fleet.
        """

        if not shapes:
            raise TracerError("a step needs at least one occupant")
        model = self.models[arch]
        weights = self.shape.weight_count
        extra = sum(self._ports(occupant) for occupant in shapes)
        key = ("step", shapes) if arch is None else ("step", arch, shapes)

        @model.tracer.definition(input_count=weights + extra, key=key, role=REPLAY)
        def step(v: Wires) -> object:
            w, cursor, outputs = v[:weights], weights, []
            for occupant in shapes:
                kind, positions, cached = occupant
                ports = self._ports(occupant)
                args = (w, v[cursor : cursor + ports]) if ports else (w,)
                cursor += ports
                if kind == PREFILL:
                    outputs.append(model.prefill(positions, cached=cached)(*args))
                elif kind == CHUNK:
                    outputs.append(model.chunk(positions, cached)(*args))
                else:
                    outputs.append(model.decode(cached + 1)(*args))
            return outputs

        return step

    def root(self, plan: _Plan) -> TracedDefinition:
        """The run: the weights, then every step in time order, wired through the caches."""

        layers, d = self.shape.layers, self.shape.d_model
        requests, occupancy, before, joins = plan.requests, plan.occupancy, plan.streamed_before, plan.schedule.joins

        @self.lm.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = self.lm.weights_unit()()
            slots: dict[tuple[int, int], _Slot] = {}
            parked: dict[int, _Slot] = {}  # request -> the cache its latest attempt left
            tokens: dict[tuple[int, int], Wire] = {}
            for key in plan.order:
                pod, step_index = key
                occupants = occupancy[key]
                shapes: list[OccupantShape] = []
                args: list[Wire | Wires] = [w]
                for occupant in occupants:
                    join: Join = joins[occupant.join]
                    if step_index == join.step:  # the attempt's first step
                        slot = parked.pop(occupant.request) if join.resume else _Slot.fresh(occupant.request, layers)
                        slots[(pod, occupant.slot)] = slot
                    else:
                        slot = slots[(pod, occupant.slot)]
                    assert slot.request == occupant.request
                    occupant_shape = plan.shape_of(occupant)
                    shapes.append(occupant_shape)
                    if occupant_shape[0] == DECODE:
                        assert slot.token is not None
                        args.append(slot.token)
                    for layer in range(layers):
                        args.extend(slot.keys[layer])
                        args.extend(slot.values[layer])
                arch = None if self.arches is None else self.arches[pod]
                outputs = self.step(tuple(shapes), arch)(*args)
                cursor = 0
                for occupant, occupant_shape in zip(occupants, shapes, strict=True):
                    kind, positions, _ = occupant_shape
                    produced = self._produced(occupant_shape)
                    block = outputs[cursor : cursor + produced]
                    cursor += produced
                    slot = slots[(pod, occupant.slot)]
                    for layer in range(layers):
                        start = 2 * layer * positions * d
                        slot.keys[layer].append(block[start : start + positions * d])
                        slot.values[layer].append(block[start + positions * d : start + 2 * positions * d])
                    if kind != CHUNK:
                        token = block[-1]
                        assert isinstance(token, Wire)
                        slot.token = token
                        if occupant.generated >= before[occupant.join]:  # streamed by this attempt
                            streamed = (occupant.request, occupant.generated)
                            assert streamed not in tokens, "a position is streamed by one attempt only"
                            tokens[streamed] = token
                    join = joins[occupant.join]
                    if step_index == join.step + join.length - 1:  # the attempt's last step
                        parked[occupant.request] = slots.pop((pod, occupant.slot))
            return [tokens[key] for key in self.output_layout(requests, plan.schedule)]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        plan = self._plan(x, self._decode_advice(a))
        description = self.lm.tracer.serialize(self.root(plan))
        return description, self.flatten_inputs(plan.requests, plan.schedule)


__all__ = ["CHUNK", "DECODE", "FLEET_GATE_SET", "PREFILL", "ClusterG", "OccupantShape"]
