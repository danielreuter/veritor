"""The tracer: an untrusted, client-side convenience for writing descriptions.

A constructor ``G`` traces Python functions over symbolic wires and serializes
the result as canonical description bytes.  Nothing here is trusted: the
compiler re-validates every byte.  Tracer cache keys only affect which
functions are traced; the definition identity is always the digest of its
canonical body.

Values are ranges.  A definition's ports are a :class:`Wires` vector; slicing
it (with a step) yields another range, indexing yields a :class:`Wire`, and
``by(j)`` marks a range that shifts by ``j`` per copy of a
:meth:`Tracer.repeat`.  A call passing a whole vector or a strided column is
therefore one range in the description regardless of its length.

The circuit's inputs and weights are source gates: :meth:`Tracer.inputs` and
:meth:`Tracer.weights` emit ``n`` of them inside the current body as one
``repeat`` of a canonical one-gate definition marked ``verification`` (so a
block of ``10**9`` weights is ``O(1)`` description and every source gate is
its own verification unit by default).  A caller who wants a wider
verification unit around its inputs calls the ``in``/``weight`` gate of the
gate set directly inside its own verification-marked definition.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, replace

from veritor.compile.description import (
    FORMAT_VERSION,
    canonical_description,
    definition_digest,
)
from veritor.core import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet
from veritor.core.description import INPUT, LOCAL, ROLES, VERIFICATION


class TracerError(ValueError):
    """The tracer was used incorrectly."""


@dataclass(frozen=True, slots=True)
class Wire:
    """One symbolic value of the trace being recorded."""

    trace: object
    space: str
    index: int

    def by(self, jstride: int) -> Wires:
        """This value shifted by ``jstride`` per copy of a ``repeat``."""

        return Wires(self.trace, self.space, self.index, 1, 0, jstride)


@dataclass(frozen=True, slots=True)
class Wires:
    """``count`` symbolic values at ``start + k * stride`` (``+ j * jstride`` per copy)."""

    trace: object
    space: str
    start: int
    count: int
    stride: int
    jstride: int = 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Wire]:
        for k in range(self.count):
            yield Wire(self.trace, self.space, self.start + k * self.stride)

    def __getitem__(self, item: int | slice) -> Wire | Wires:
        if isinstance(item, slice):
            start, stop, step = item.indices(self.count)
            if step <= 0:
                raise TracerError("wire slices must step forward")
            count = len(range(start, stop, step))
            if count == 0:
                raise TracerError("an empty wire range has no description")
            return Wires(
                self.trace,
                self.space,
                self.start + start * self.stride,
                count,
                self.stride * step,
                self.jstride,
            )
        if type(item) is not int:
            raise TracerError("wires are indexed by integers or slices")
        if item < 0:
            item += self.count
        if not 0 <= item < self.count:
            raise IndexError(item)
        return Wire(self.trace, self.space, self.start + item * self.stride)

    def by(self, jstride: int) -> Wires:
        """This range shifted by ``jstride`` per copy of a ``repeat``."""

        return replace(self, jstride=jstride)


type Argument = Wire | Wires


@dataclass(frozen=True, slots=True)
class TracedDefinition:
    """A traced definition: call it inside another trace of the same tracer."""

    tracer: Tracer
    digest: str
    input_count: int
    output_count: int
    role: str | None

    def __call__(self, *args: Argument) -> Wire | Wires:
        return self.tracer._active().emit_call(self, args)


@dataclass(frozen=True, slots=True)
class TracerGate:
    """A gate of the gate set; calling it on wires records a gate step."""

    tracer: Tracer
    gate: Gate

    def __call__(self, *args: Argument) -> Wire:
        return self.tracer._active().emit_gate(self.gate, args)


@dataclass(slots=True)
class _Run:
    space: str
    start: int
    count: int
    stride: int
    jstride: int

    def extend(self, other: _Run) -> bool:
        """Absorb a single wire that continues this run."""

        if other.count != 1 or (self.space, self.jstride) != (other.space, other.jstride):
            return False
        if self.count == 1 and other.start >= self.start:
            self.count, self.stride = 2, other.start - self.start
            return True
        if self.count > 1 and other.start == self.start + self.count * self.stride:
            self.count += 1
            return True
        return False

    def encode(self, copies: bool) -> list[object]:
        item: list[object] = [self.space, self.start, self.count, self.stride]
        return [*item, self.jstride] if copies else item


def _ranges(trace: _Trace, args: Sequence[Argument], *, copies: bool) -> list[_Run]:
    """Turn arguments into ranges, merging runs of single wires."""

    runs: list[_Run] = []
    for arg in args:
        if isinstance(arg, Wire):
            run = _Run(arg.space, arg.index, 1, 0, 0)
        elif isinstance(arg, Wires):
            run = _Run(arg.space, arg.start, arg.count, arg.stride, arg.jstride)
        else:
            raise TracerError("arguments must be wires")
        if arg.trace is not trace.identity:
            raise TracerError("all arguments must be wires of the current trace")
        if run.jstride and not copies:
            raise TracerError("only repeat arguments may shift per copy")
        if not (runs and runs[-1].extend(run)):
            runs.append(run)
    return runs


def _encode(runs: list[_Run], expected: int, what: str, *, copies: bool) -> list[object]:
    total = sum(run.count for run in runs)
    if total != expected:
        raise TracerError(f"{what} takes {expected} arguments, got {total}")
    return [run.encode(copies) for run in runs]


class _Trace:
    """The steps of one definition while its function runs."""

    def __init__(self, tracer: Tracer, input_count: int) -> None:
        self.tracer = tracer
        self.identity = object()
        self.inputs = Wires(self.identity, INPUT, 0, input_count, 1)
        self.steps: list[dict[str, object]] = []
        self.slots = 0

    def _outputs(self, count: int) -> Wire | Wires:
        start = self.slots
        self.slots += count
        if count == 1:
            return Wire(self.identity, LOCAL, start)
        return Wires(self.identity, LOCAL, start, count, 1)

    def emit_gate(self, gate: Gate, args: Sequence[Argument]) -> Wire:
        ranges = _encode(_ranges(self, args, copies=False), gate.arity, gate.name, copies=False)
        self.steps.append({"kind": "gate", "gate": gate.name, "args": ranges})
        result = self._outputs(1)
        assert isinstance(result, Wire)
        return result

    def emit_call(
        self, definition: TracedDefinition, args: Sequence[Argument]
    ) -> Wire | Wires:
        if definition.tracer is not self.tracer:
            raise TracerError("the definition belongs to another tracer")
        ranges = _encode(
            _ranges(self, args, copies=False),
            definition.input_count,
            "the definition",
            copies=False,
        )
        self.steps.append({"kind": "call", "digest": definition.digest, "args": ranges})
        return self._outputs(definition.output_count)

    def emit_repeat(
        self, count: int, definition: TracedDefinition, args: Sequence[Argument]
    ) -> Wires:
        if definition.tracer is not self.tracer:
            raise TracerError("the definition belongs to another tracer")
        if type(count) is not int or count < 1:
            raise TracerError("repeat count must be a positive integer")
        ranges = _encode(
            _ranges(self, args, copies=True),
            definition.input_count,
            "the definition",
            copies=True,
        )
        self.steps.append(
            {"kind": "repeat", "count": count, "digest": definition.digest, "args": ranges}
        )
        start = self.slots
        self.slots += count * definition.output_count
        return Wires(self.identity, LOCAL, start, count * definition.output_count, 1)


_ACTIVE: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar(
    "veritor_tracer_active_trace", default=None
)


class Tracer:
    """Records definitions over a gate set and serializes a description."""

    def __init__(self, gate_set: GateSet) -> None:
        if not isinstance(gate_set, GateSet):
            raise TracerError("Tracer requires a GateSet")
        self.gate_set = gate_set
        self._bodies: dict[str, dict[str, object]] = {}
        self._by_key: dict[Hashable, TracedDefinition] = {}

    def _active(self) -> _Trace:
        trace = _ACTIVE.get()
        if trace is None or trace.tracer is not self:
            raise TracerError("wires may only be used while tracing a definition")
        return trace

    @property
    def definition_count(self) -> int:
        return len(self._bodies)

    def gate(self, name: str) -> TracerGate:
        if name not in self.gate_set:
            raise TracerError(f"the gate set has no gate {name!r}")
        return TracerGate(self, self.gate_set[name])

    def definition(
        self,
        *,
        input_count: int,
        key: Hashable | None = None,
        role: str | None = None,
    ) -> Callable[[Callable[[Wires], object]], TracedDefinition]:
        """Trace ``fn(inputs)`` once per ``key`` into a definition marked ``role``."""

        if type(input_count) is not int or input_count < 0:
            raise TracerError("input_count must be a nonnegative integer")
        if role is not None and role not in ROLES:
            raise TracerError(f"role must be None or one of {list(ROLES)}")
        if key is not None:
            try:
                hash(key)
            except TypeError as error:
                raise TracerError("definition keys must be hashable") from error

        def decorate(fn: Callable[[Wires], object]) -> TracedDefinition:
            if key is not None and key in self._by_key:
                cached = self._by_key[key]
                if (cached.input_count, cached.role) != (input_count, role):
                    raise TracerError("a definition key was reused with another signature")
                return cached
            trace = _Trace(self, input_count)
            token = _ACTIVE.set(trace)
            try:
                result = fn(trace.inputs)
            finally:
                _ACTIVE.reset(token)
            if isinstance(result, (Wire, Wires)):
                returned: Sequence[Argument] = (result,)
            elif isinstance(result, Sequence) and result and not isinstance(result, str):
                returned = tuple(result)
            else:
                raise TracerError("a definition must return one or more wires")
            runs = _ranges(trace, returned, copies=False)
            output_count = sum(run.count for run in runs)
            body: dict[str, object] = {
                "input_count": input_count,
                "role": role,
                "steps": trace.steps,
                "outputs": [run.encode(False) for run in runs],
            }
            digest = definition_digest(body)  # type: ignore[arg-type]
            self._bodies.setdefault(digest, body)
            definition = TracedDefinition(self, digest, input_count, output_count, role)
            if key is not None:
                self._by_key[key] = definition
            return definition

        return decorate

    def repeat(self, count: int, definition: TracedDefinition, *args: Argument) -> Wires:
        """``count`` copies of ``definition``; ``by(j)`` arguments shift per copy."""

        return self._active().emit_repeat(count, definition, args)

    def source_cell(self, source: str) -> TracedDefinition:
        """The canonical verification unit holding one ``source`` gate (``"input"``/``"weight"``)."""

        names = {INPUT_SOURCE: self.gate_set.input_gates, WEIGHT_SOURCE: self.gate_set.weight_gates}
        if source not in names:
            raise TracerError(f"source must be one of {sorted(names)}")
        if not names[source]:
            raise TracerError(f"the gate set has no {source} gate")
        gate = self.gate(names[source][0])
        return self.definition(input_count=0, key=("veritor.source", source), role=VERIFICATION)(
            lambda _: gate()
        )

    def sources(self, source: str, count: int) -> Wires:
        """``count`` source gates in the current body: one call or ``repeat`` of the canonical cell."""

        if type(count) is not int or count < 1:
            raise TracerError("the number of source gates must be a positive integer")
        cell = self.source_cell(source)
        trace = self._active()
        if count == 1:
            wire = trace.emit_call(cell, ())
            assert isinstance(wire, Wire)
            return Wires(trace.identity, LOCAL, wire.index, 1, 0)
        return trace.emit_repeat(count, cell, ())

    def inputs(self, count: int) -> Wires:
        """``count`` input gates (``x`` by rank) in the current body."""

        return self.sources(INPUT_SOURCE, count)

    def weights(self, count: int) -> Wires:
        """``count`` weight gates (``W`` by rank) in the current body."""

        return self.sources(WEIGHT_SOURCE, count)

    def serialize(self, root: TracedDefinition) -> bytes:
        """Canonical description bytes for ``root`` and everything it calls."""

        if not isinstance(root, TracedDefinition) or root.tracer is not self:
            raise TracerError("root belongs to another tracer")
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(digest: str) -> None:
            if digest in seen:
                return
            seen.add(digest)
            steps = self._bodies[digest]["steps"]
            assert isinstance(steps, list)
            for step in steps:
                if step["kind"] != "gate":
                    visit(str(step["digest"]))
            ordered.append(digest)

        visit(root.digest)
        return canonical_description(
            {
                "version": FORMAT_VERSION,
                "definitions": [
                    {"digest": digest, "body": self._bodies[digest]} for digest in ordered
                ],
                "root": root.digest,
            }
        )
