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
therefore one range in the description regardless of its length.  A range of
one value *is* a :class:`Wire`: ``Wires(...)`` with ``count == 1`` constructs
a ``Wire`` (stride ``0``), so a gate, a one-output call, a one-element slice
and a one-gate ``repeat`` all yield the same kind of object and constructors
never special-case the count.

Check outputs.  Inside the root's trace, :meth:`Tracer.check` marks wires
that the definition will return as outputs the verifier requires to equal a
constant (``ok`` words, blank slots after an advised stop).  The marks are
resolved against the returned outputs when the definition is traced and
emitted as the body's ``checks``; only a root may carry them.

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
from typing import cast, overload

from veritor.compile.description import (
    FORMAT_VERSION,
    canonical_description,
    definition_digest,
)
from veritor.core import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet, JSONValue
from veritor.core.description import INPUT, LOCAL, ROLES, VERIFICATION


class TracerError(ValueError):
    """The tracer was used incorrectly."""


@dataclass(frozen=True, slots=True)
class Wires:
    """``count`` symbolic values at ``start + k * stride`` (``+ j * jstride`` per copy).

    A range of one value is a :class:`Wire`: constructing ``Wires`` with
    ``count == 1`` yields a ``Wire`` with stride ``0``, whatever stride was
    given, so equal ranges are equal objects.
    """

    trace: object
    space: str
    start: int
    count: int
    stride: int
    jstride: int = 0

    def __new__(  # type: ignore[no-untyped-def]
        cls, trace: object, space: str, start: int, count: int = 1, stride: int = 0, jstride: int = 0
    ):
        made = Wire if cls is Wires and count == 1 else cls
        return object.__new__(made)

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise TracerError("a wire range has a nonnegative count")
        if self.count == 1 and self.stride:
            object.__setattr__(self, "stride", 0)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Wire]:
        for k in range(self.count):
            yield Wire(self.trace, self.space, self.start + k * self.stride, jstride=self.jstride)

    @overload
    def __getitem__(self, item: int) -> Wire: ...

    @overload
    def __getitem__(self, item: slice) -> Wires: ...

    def __getitem__(self, item: int | slice) -> Wires:
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
        return Wire(self.trace, self.space, self.start + item * self.stride, jstride=self.jstride)

    def by(self, jstride: int) -> Wires:
        """This range shifted by ``jstride`` per copy of a ``repeat``."""

        return replace(self, jstride=jstride)


@dataclass(frozen=True, slots=True, init=False)
class Wire(Wires):
    """One symbolic value of the trace being recorded: a :class:`Wires` of count one.

    ``Wire(trace, space, index)`` names the value at ``index``; ``count`` and
    ``stride`` are accepted (as ``1`` and ``0``) so that ``Wires(...)`` of
    count one and :func:`dataclasses.replace` construct a ``Wire`` too.
    """

    def __init__(
        self,
        trace: object,
        space: str,
        start: int,
        count: int = 1,
        stride: int = 0,
        jstride: int = 0,
    ) -> None:
        if count != 1:
            raise TracerError("a Wire is one value")
        set_ = object.__setattr__
        set_(self, "trace", trace)
        set_(self, "space", space)
        set_(self, "start", start)
        set_(self, "count", 1)
        set_(self, "stride", 0)
        set_(self, "jstride", jstride)

    @property
    def index(self) -> int:
        return self.start

    def by(self, jstride: int) -> Wire:
        """This value shifted by ``jstride`` per copy of a ``repeat``."""

        return Wire(self.trace, self.space, self.start, jstride=jstride)


type Argument = Wires


@dataclass(frozen=True, slots=True)
class TracedDefinition:
    """A traced definition: call it inside another trace of the same tracer."""

    tracer: Tracer
    digest: str
    input_count: int
    output_count: int
    role: str | None

    def __call__(self, *args: Argument) -> Wires:
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
        if not isinstance(arg, Wires):
            raise TracerError("arguments must be wires")
        run = _Run(arg.space, arg.start, arg.count, arg.stride, arg.jstride)
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


def _ordinal(outputs: list[_Run], space: str, index: int) -> int:
    """The declared-output ordinal of the coordinate ``(space, index)`` among ``outputs``."""

    base = 0
    for run in outputs:
        if run.space == space:
            offset = index - run.start
            if run.count == 1 or run.stride == 0:
                if offset == 0:
                    return base
            elif offset >= 0 and offset % run.stride == 0 and offset // run.stride < run.count:
                return base + offset // run.stride
        base += run.count
    raise TracerError("a checked wire must be one of the definition's declared outputs")


def _encode_checks(outputs: list[_Run], checks: list[tuple[Wires, int]]) -> list[object]:
    """``[start, count, stride, value]`` over output ordinals for the checked wires, runs merged."""

    encoded: list[list[object]] = []
    for wires, value in checks:
        ordinals = [_ordinal(outputs, wires.space, wires.start + k * wires.stride) for k in range(wires.count)]
        held: _Run | None = None
        for ordinal in ordinals:
            single = _Run("", ordinal, 1, 0, 0)
            if held is not None and held.extend(single):
                continue
            if held is not None:
                encoded.append([held.start, held.count, held.stride, value])
            held = single
        if held is not None:
            encoded.append([held.start, held.count, held.stride, value])
    return list(encoded)


class _Trace:
    """The steps of one definition while its function runs."""

    def __init__(self, tracer: Tracer, input_count: int) -> None:
        self.tracer = tracer
        self.identity = object()
        self.inputs = Wires(self.identity, INPUT, 0, input_count, 1)
        self.steps: list[dict[str, object]] = []
        self.slots = 0
        self.checks: list[tuple[Wires, int]] = []

    def _outputs(self, count: int) -> Wires:
        start = self.slots
        self.slots += count
        return Wires(self.identity, LOCAL, start, count, 1)

    def emit_gate(self, gate: Gate, args: Sequence[Argument]) -> Wire:
        ranges = _encode(_ranges(self, args, copies=False), gate.arity, gate.name, copies=False)
        self.steps.append({"kind": "gate", "gate": gate.name, "args": ranges})
        result = self._outputs(1)
        assert isinstance(result, Wire)
        return result

    def emit_call(self, definition: TracedDefinition, args: Sequence[Argument]) -> Wires:
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
        return self._outputs(count * definition.output_count)


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
            if isinstance(result, Wires):
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
            if trace.checks:
                body["checks"] = _encode_checks(runs, trace.checks)
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

    def check(self, wires: Wires, value: int) -> Wires:
        """Mark ``wires`` as check outputs: the verifier requires each to equal ``value``.

        The wires must be among the outputs the definition being traced
        returns (resolved when the trace ends), and only a root may declare
        checks; the compiler validates both and that ``value`` fits the
        gates.  Returns ``wires`` so a mark can wrap an output expression.
        """

        trace = self._active()
        if not isinstance(wires, Wires) or wires.trace is not trace.identity:
            raise TracerError("checks mark wires of the current trace")
        if wires.jstride:
            raise TracerError("a checked range does not shift per copy")
        if type(value) is not int or value < 0:
            raise TracerError("a check value is a nonnegative integer")
        if wires.count:
            trace.checks.append((wires, value))
        return wires

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
            return trace.emit_call(cell, ())
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
                    {"digest": digest, "body": cast("JSONValue", self._bodies[digest])} for digest in ordered
                ],
                "root": root.digest,
            }
        )
