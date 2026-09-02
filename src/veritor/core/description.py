"""Validated circuit descriptions and the lazy descent through their copies.

A description is a list of *definitions* in dependency order.  A definition
has ``input_count`` inputs, a list of steps, declared ``outputs`` and an
optional ``role`` mark.  Steps are:

* ``gate(name, args)`` -- one scalar gate from the gate set;
* ``call(digest, args)`` -- one copy of an earlier definition;
* ``repeat(count, digest, args)`` -- ``count`` copies of an earlier
  definition, copy ``j`` receiving arguments whose start is shifted by
  ``j * jstride``.

Every argument (and every declared output) is a :class:`Range` in *relative*
coordinates: ``space`` is ``"input"`` (the definition's own inputs) or
``"local"`` (output *slots* of earlier steps in the same definition).  A gate
step has one slot; a call has ``child.output_count`` slots; a repeat has
``count * child.output_count`` slots, copy-major.

Layout.  The circuit's addresses are ``[0, input_count)`` for the root inputs
followed by the gates in step order, each call or repeat copy occupying one
contiguous block.  Every copy of a definition is therefore an interval, and
descending from the root to any address costs ``O(depth)``: bisect on step
starts within a definition and divide by the child size within a repeat.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import cached_property

from .gates import Gate

INPUT = "input"
LOCAL = "local"
REPLAY = "replay"
VERIFICATION = "verification"
ROLES = (REPLAY, VERIFICATION)


def _derived():  # type: ignore[no-untyped-def]
    return field(init=False, repr=False, compare=False)


def prefix_sums(counts: list[int]) -> tuple[int, ...]:
    total = 0
    sums = [0]
    for count in counts:
        total += count
        sums.append(total)
    return tuple(sums)


@dataclass(frozen=True, slots=True)
class Range:
    """``count`` relative coordinates ``start + j * jstride + k * stride``."""

    space: str
    start: int
    count: int
    stride: int
    jstride: int = 0

    def element(self, k: int, j: int = 0) -> int:
        return self.start + j * self.jstride + k * self.stride

    def last(self, copies: int = 1) -> int:
        """The largest coordinate over ``copies`` copies (``j < copies``)."""

        return self.element(self.count - 1, copies - 1)


def ranges_total(ranges: tuple[Range, ...]) -> int:
    return sum(item.count for item in ranges)


def range_at(
    ranges: tuple[Range, ...], starts: tuple[int, ...], ordinal: int
) -> tuple[Range, int]:
    """Locate ``ordinal`` inside consecutive ranges: ``(range, element)``."""

    index = bisect_right(starts, ordinal) - 1
    return ranges[index], ordinal - starts[index]


@dataclass(frozen=True, slots=True)
class GateStep:
    gate: Gate
    args: tuple[Range, ...]

    @property
    def size(self) -> int:
        return 1

    @property
    def slots(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class CallStep:
    """``call`` (``count == 1``) or ``repeat`` (``count > 1``) of ``child``."""

    child: Definition
    args: tuple[Range, ...]
    count: int
    arg_starts: tuple[int, ...]

    @staticmethod
    def make(child: Definition, args: tuple[Range, ...], count: int = 1) -> CallStep:
        return CallStep(child, args, count, prefix_sums([item.count for item in args]))

    @property
    def size(self) -> int:
        return self.count * self.child.size

    @property
    def slots(self) -> int:
        return self.count * self.child.output_count

    def arg_at(self, ordinal: int) -> tuple[Range, int]:
        """The argument range and element feeding child input ``ordinal``."""

        return range_at(self.args, self.arg_starts, ordinal)


type Step = GateStep | CallStep


@dataclass(frozen=True)
class Definition:
    """One validated definition with its per-kind summaries.

    Summaries are computed once per definition, never per copy: ``size`` is
    the number of gates, the costs sum gate costs, and the unit counts and
    ``out_total`` (the number of boundary addresses contributed by the replay
    units inside) feed the prefix sums that make unit and boundary lookups
    ``O(depth)``.
    """

    digest: str
    input_count: int
    steps: tuple[Step, ...]
    outputs: tuple[Range, ...]
    role: str | None
    step_address: tuple[int, ...] = _derived()
    step_slot: tuple[int, ...] = _derived()
    step_replay: tuple[int, ...] = _derived()
    step_verification: tuple[int, ...] = _derived()
    output_starts: tuple[int, ...] = _derived()
    depth: int = _derived()
    replay_cost: int = _derived()
    proof_cost: int = _derived()

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        steps = self.steps
        set_(self, "step_address", prefix_sums([step.size for step in steps]))
        set_(self, "step_slot", prefix_sums([step.slots for step in steps]))
        set_(self, "output_starts", prefix_sums([item.count for item in self.outputs]))
        calls = [step for step in steps if isinstance(step, CallStep)]
        set_(self, "depth", 1 + max((c.child.depth for c in calls), default=-1))
        set_(
            self,
            "replay_cost",
            sum(
                step.gate.replay_cost
                if isinstance(step, GateStep)
                else step.count * step.child.replay_cost
                for step in steps
            ),
        )
        set_(
            self,
            "proof_cost",
            sum(
                step.gate.proof_cost
                if isinstance(step, GateStep)
                else step.count * step.child.proof_cost
                for step in steps
            ),
        )
        replay = [
            0 if isinstance(s, GateStep) else s.count * s.child.replay_units
            for s in steps
        ]
        verification = [
            0 if isinstance(s, GateStep) else s.count * s.child.verification_units
            for s in steps
        ]
        set_(self, "step_replay", prefix_sums(replay))
        set_(self, "step_verification", prefix_sums(verification))

    # -- sizes ---------------------------------------------------------------

    @property
    def size(self) -> int:
        return self.step_address[-1]

    @property
    def slot_count(self) -> int:
        return self.step_slot[-1]

    @property
    def output_count(self) -> int:
        return self.output_starts[-1]

    @property
    def replay_units(self) -> int:
        return 1 if self.role == REPLAY else self.step_replay[-1]

    @property
    def verification_units(self) -> int:
        return 1 if self.role == VERIFICATION else self.step_verification[-1]

    # -- relative resolution -------------------------------------------------

    def step_at_address(self, offset: int) -> int:
        """Index of the step whose gates contain relative address ``offset``."""

        return bisect_right(self.step_address, offset) - 1

    def output_at(self, ordinal: int) -> tuple[Range, int]:
        return range_at(self.outputs, self.output_starts, ordinal)

    def gate_at(self, offset: int) -> Gate:
        """The gate at relative address ``offset``, descending through calls."""

        definition = self
        while True:
            index = definition.step_at_address(offset)
            step = definition.steps[index]
            if isinstance(step, GateStep):
                return step.gate
            _, offset = divmod(offset - definition.step_address[index], step.child.size)
            definition = step.child

    def slot_source(self, slot: int) -> tuple[bool, int]:
        """Resolve a local slot: ``(True, gate offset)`` or ``(False, input)``."""

        index = bisect_right(self.step_slot, slot) - 1
        step = self.steps[index]
        offset = self.step_address[index]
        if isinstance(step, GateStep):
            return True, offset
        copy, ordinal = divmod(slot - self.step_slot[index], step.child.output_count)
        is_gate, value = step.child.output_source(ordinal)
        if is_gate:
            return True, offset + copy * step.child.size + value
        return self.ref_source(*self._arg_ref(step, value, copy))

    def output_source(self, ordinal: int) -> tuple[bool, int]:
        """Resolve declared output ``ordinal`` like :meth:`slot_source`."""

        item, element = self.output_at(ordinal)
        return self.ref_source(item.space, item.element(element))

    def ref_source(self, space: str, value: int) -> tuple[bool, int]:
        if space == LOCAL:
            return self.slot_source(value)
        return False, value

    @staticmethod
    def _arg_ref(step: CallStep, ordinal: int, copy: int) -> tuple[str, int]:
        item, element = step.arg_at(ordinal)
        return item.space, item.element(element, copy)

    # -- lazy per-kind summaries ---------------------------------------------

    @cached_property
    def local_outputs(self) -> tuple[int, ...]:
        """Sorted distinct gate offsets among the declared outputs.

        ``Out`` of a copy is its declared interface, resolved to the gates it
        owns; outputs that merely pass an input through are not addresses of
        the copy and are excluded.
        """

        found = {
            value
            for is_gate, value in map(self.output_source, range(self.output_count))
            if is_gate
        }
        return tuple(sorted(found))

    @cached_property
    def out_bits(self) -> int:
        """Bits carried by ``Out`` of a copy: the widths of its ``local_outputs``."""

        return sum(self.gate_at(offset).width for offset in self.local_outputs)

    @cached_property
    def out_total(self) -> int:
        """Boundary addresses contributed by the replay units inside."""

        if self.role == REPLAY:
            return len(self.local_outputs)
        return sum(
            step.count * step.child.out_total
            for step in self.steps
            if isinstance(step, CallStep)
        )

    @cached_property
    def step_out(self) -> tuple[int, ...]:
        return prefix_sums(
            [
                0 if isinstance(s, GateStep) else s.count * s.child.out_total
                for s in self.steps
            ]
        )

    @cached_property
    def reads(self) -> tuple[int, ...]:
        """Sorted distinct input ordinals read (transitively) by gates inside.

        Resolved once per definition.  Through a ``repeat`` whose argument
        shifts with the copy, every copy's image is enumerated, so the cost is
        proportional to the size of the answer, not to the circuit.
        """

        found: set[int] = set()

        def read(space: str, value: int) -> None:
            is_gate, source = self.ref_source(space, value)
            if not is_gate:
                found.add(source)

        for step in self.steps:
            if isinstance(step, GateStep):
                for item in step.args:
                    for k in range(item.count):
                        read(item.space, item.element(k))
                continue
            for ordinal in step.child.reads:
                item, element = step.arg_at(ordinal)
                copies = 1 if item.jstride == 0 else step.count
                for copy in range(copies):
                    read(item.space, item.element(element, copy))
        return tuple(sorted(found))


@dataclass(frozen=True, slots=True)
class Frame:
    """One copy of a definition at an absolute position in the layout.

    ``base`` is the address of the copy's first gate.  The ``*_before``
    counters are prefix sums over everything laid out before this copy, so a
    frame reached by descent knows its replay-unit index, the global index of
    its first verification unit, and how many boundary addresses precede it.
    """

    definition: Definition
    base: int
    parent: Frame | None
    step: int
    j: int
    depth: int
    replay_before: int
    verification_before: int
    out_before: int

    @staticmethod
    def root(definition: Definition) -> Frame:
        return Frame(definition, definition.input_count, None, -1, 0, 0, 0, 0, 0)

    @property
    def interval(self) -> range:
        return range(self.base, self.base + self.definition.size)

    def child(self, index: int, copy: int) -> Frame:
        d = self.definition
        step = d.steps[index]
        if not isinstance(step, CallStep):
            raise TypeError("gate steps have no child frame")
        c = step.child
        return Frame(
            c,
            self.base + d.step_address[index] + copy * c.size,
            self,
            index,
            copy,
            self.depth + 1,
            self.replay_before + d.step_replay[index] + copy * c.replay_units,
            self.verification_before
            + d.step_verification[index]
            + copy * c.verification_units,
            self.out_before + d.step_out[index] + copy * c.out_total,
        )

    def input_address(self, ordinal: int) -> int:
        """Absolute address feeding this copy's input ``ordinal``."""

        frame = self
        while frame.parent is not None:
            parent = frame.parent
            step = parent.definition.steps[frame.step]
            assert isinstance(step, CallStep)
            item, element = step.arg_at(ordinal)
            value = item.element(element, frame.j)
            if item.space == LOCAL:
                return parent.slot_address(value)
            ordinal = value
            frame = parent
        return ordinal

    def slot_address(self, slot: int) -> int:
        is_gate, value = self.definition.slot_source(slot)
        return self.base + value if is_gate else self.input_address(value)

    def address(self, space: str, value: int) -> int:
        if space == LOCAL:
            return self.slot_address(value)
        return self.input_address(value)

    def output_address(self, ordinal: int) -> int:
        item, element = self.definition.output_at(ordinal)
        return self.address(item.space, item.element(element))

    def locate(self, address: int) -> tuple[Frame, GateStep]:
        """Descend to the gate at ``address`` inside this copy."""

        frame = self
        offset = address - self.base
        while True:
            d = frame.definition
            index = d.step_at_address(offset)
            step = d.steps[index]
            if isinstance(step, GateStep):
                return frame, step
            copy, offset = divmod(offset - d.step_address[index], step.child.size)
            frame = frame.child(index, copy)

    def gate(self, address: int) -> tuple[Gate, tuple[int, ...]]:
        """The gate at ``address`` with absolute argument addresses."""

        frame, step = self.locate(address)
        return step.gate, tuple(
            frame.address(item.space, item.element(k))
            for item in step.args
            for k in range(item.count)
        )
