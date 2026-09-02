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

Interfaces.  ``Out`` of a definition is its declared outputs resolved to the
gates a copy owns.  Everything in a description is affine, so this is a union
of arithmetic progressions, kept as a tuple of :class:`Run` computed once per
definition in time proportional to the description, never to the number of
outputs; the compiler requires the runs to be pairwise disjoint.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from math import gcd

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
class Run:
    """``count`` gate offsets ``start + k * stride``, each ``width`` bits wide.

    ``Out`` of a definition is a tuple of runs.  A run of one element has
    stride ``0``; a run of several elements has a positive stride.
    """

    start: int
    count: int
    stride: int
    width: int

    def element(self, k: int) -> int:
        return self.start + k * self.stride

    @property
    def last(self) -> int:
        return self.start + (self.count - 1) * self.stride

    def index(self, offset: int) -> int | None:
        """The ``k`` with ``element(k) == offset``, or ``None``."""

        if offset < self.start:
            return None
        if self.stride == 0:
            return 0 if offset == self.start else None
        k, remainder = divmod(offset - self.start, self.stride)
        return k if remainder == 0 and k < self.count else None


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
    def resolved_outputs(self) -> tuple[tuple[bool, Run], ...]:
        """The declared outputs as pieces: ``(True, gate run)`` or ``(False, input run)``.

        An input run (width ``0``) holds the input ordinals that outputs pass
        straight through; a parent resolves them further through its own
        argument ranges.  Adjacent pieces that continue one progression are
        merged.
        """

        return tuple(_merge(_declared_pieces(self, 0, self.output_count, 1)))

    @cached_property
    def out_runs(self) -> tuple[Run, ...]:
        """``Out`` of a copy: runs of the gate offsets it owns, ordered by start.

        The declared interface resolved to gates; outputs that merely pass an
        input through are not addresses of the copy and are dropped, and runs
        left adjacent by that are merged.  The members of ``Out`` are ranked
        run by run in this order, which is address order unless runs
        interleave.
        """

        gates = sorted(
            (run for is_gate, run in self.resolved_outputs if is_gate),
            key=lambda run: (run.start, run.stride),
        )
        return tuple(run for _, run in _merge((True, run) for run in gates))

    @cached_property
    def out_starts(self) -> tuple[int, ...]:
        return prefix_sums([run.count for run in self.out_runs])

    @property
    def out_count(self) -> int:
        """``|Out|`` of a copy; exact once the runs are known to be disjoint."""

        return self.out_starts[-1]

    @cached_property
    def out_bits(self) -> int:
        """Bits carried by ``Out`` of a copy: the widths of its runs."""

        return sum(run.count * run.width for run in self.out_runs)

    def out_offset(self, rank: int) -> int:
        """The gate offset of member ``rank`` of ``Out``: bisect the run prefix sums."""

        index = bisect_right(self.out_starts, rank) - 1
        return self.out_runs[index].element(rank - self.out_starts[index])

    def out_rank(self, offset: int) -> int | None:
        """The rank of gate offset ``offset`` in ``Out``, or ``None``: a scan over runs."""

        for index, run in enumerate(self.out_runs):
            k = run.index(offset)
            if k is not None:
                return self.out_starts[index] + k
        return None

    @cached_property
    def out_total(self) -> int:
        """Boundary addresses contributed by the replay units inside."""

        if self.role == REPLAY:
            return self.out_count
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
        proportional to the size of the answer, ``Theta(|In|)``, not to the
        circuit.  That is affordable only for a *sampled* unit, which has
        already passed the verifier's work budget: the protocol uses it
        through ``DescriptionCircuit.In`` to open what a sampled unit reads.
        Nothing at compile or admission time may call it; the per-kind table
        prices a kind by its declared ``input_count`` instead.
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


# -- resolving declared outputs to runs --------------------------------------
#
# A *piece* is ``(True, run)`` for gate offsets of the definition or
# ``(False, run)`` for input ordinals it passes through (width 0).  Every
# coordinate progression is cut at the segment boundaries it crosses (steps,
# output ranges, argument ranges) and each segment maps affinely onto the
# child's pieces, so the work is proportional to the pieces produced, never to
# the number of outputs.  Only a progression whose elements land on different
# child outputs in a non-affine way (a stride that is not a multiple of the
# child's output count) is split into its residue classes, one per distinct
# child output it visits: exact enumeration when there are as many classes as
# elements.

type _Piece = tuple[bool, Run]


def _split(
    starts: tuple[int, ...], first: int, count: int, stride: int
) -> Iterator[tuple[int, int, int]]:
    """Cut ``first + k * stride`` (``k < count``) at the segment boundaries ``starts``.

    ``starts`` are prefix sums ending with the total.  Yields ``(segment,
    first k, elements)`` for each segment the progression visits, in order.
    """

    k = 0
    while k < count:
        value = first + k * stride
        index = bisect_right(starts, value) - 1
        if stride == 0:
            taken = count - k
        else:
            taken = min(count - k, (starts[index + 1] - 1 - value) // stride + 1)
        yield index, k, taken
        k += taken


def _grid(
    start: int, rows: int, row_stride: int, columns: int, column_stride: int
) -> list[tuple[int, int, int]]:
    """Progressions covering ``start + r * row_stride + c * column_stride``.

    One progression when a dimension is trivial or the rows tile (each row
    continues the previous one), otherwise the smaller of the two families of
    parallel lines.
    """

    if rows == 1:
        return [(start, columns, column_stride if columns > 1 else 0)]
    if columns == 1:
        return [(start, rows, row_stride)]
    if row_stride == columns * column_stride:
        return [(start, rows * columns, column_stride)]
    if columns <= rows:
        return [(start + c * column_stride, rows, row_stride) for c in range(columns)]
    return [(start + r * row_stride, columns, column_stride) for r in range(rows)]


def _merge(pieces: Iterable[_Piece]) -> list[_Piece]:
    """Join adjacent pieces of one kind and width that continue one progression."""

    merged: list[_Piece] = []
    for is_gate, run in pieces:
        if merged:
            last_gate, last = merged[-1]
            gap = run.start - last.last
            if (
                last_gate == is_gate
                and last.width == run.width
                and gap > 0
                and (last.count == 1 or last.stride == gap)
                and (run.count == 1 or run.stride == gap)
            ):
                merged[-1] = (is_gate, Run(last.start, last.count + run.count, gap, run.width))
                continue
        merged.append((is_gate, run))
    return merged


def _pieces(
    definition: Definition, space: str, start: int, count: int, stride: int
) -> Iterator[_Piece]:
    """Pieces for the relative coordinates ``start + k * stride`` (``k < count``)."""

    if space == INPUT:
        yield False, Run(start, count, stride if count > 1 else 0, 0)
        return
    for index, first, taken in _split(definition.step_slot, start, count, stride):
        step = definition.steps[index]
        base = definition.step_address[index]
        if isinstance(step, GateStep):
            # several elements here only when stride is 0: the same gate repeated
            yield True, Run(base, taken, 0, step.gate.width)
            continue
        slot = start + first * stride - definition.step_slot[index]
        yield from _call_pieces(definition, step, base, slot, taken, stride)


def _declared_pieces(
    definition: Definition, first: int, count: int, stride: int
) -> Iterator[_Piece]:
    """Pieces for the declared output ordinals ``first + k * stride`` (``k < count``)."""

    for index, k, taken in _split(definition.output_starts, first, count, stride):
        item = definition.outputs[index]
        element = first + k * stride - definition.output_starts[index]
        yield from _pieces(
            definition, item.space, item.element(element), taken, stride * item.stride
        )


def _output_pieces(
    definition: Definition, first: int, count: int, stride: int
) -> Iterable[_Piece]:
    """Like :func:`_declared_pieces`, served from the cache for the whole interface."""

    if first == 0 and count == definition.output_count and stride == 1:
        return definition.resolved_outputs
    return _declared_pieces(definition, first, count, stride)


def _call_pieces(
    definition: Definition, step: CallStep, base: int, slot: int, count: int, stride: int
) -> Iterator[_Piece]:
    """Pieces for slots ``slot + k * stride`` (``k < count``) of a call or repeat step.

    Consecutive slots are a partial first copy, whole copies and a partial
    last copy; any other stride visits one child output per residue class of
    ``stride`` modulo the child's output count, with the copy advancing
    affinely inside each class.
    """

    child = step.child
    outputs = child.output_count
    if stride == 1:
        copy, ordinal = divmod(slot, outputs)
        last_copy, last_ordinal = divmod(slot + count - 1, outputs)
        if copy == last_copy:
            pieces = _output_pieces(child, ordinal, count, 1)
            yield from _lift(definition, step, base, pieces, copy, 1, 0)
            return
        head = _output_pieces(child, ordinal, outputs - ordinal, 1)
        yield from _lift(definition, step, base, head, copy, 1, 0)
        if last_copy - copy > 1:
            whole = child.resolved_outputs
            yield from _lift(definition, step, base, whole, copy + 1, last_copy - copy - 1, 1)
        tail = _output_pieces(child, 0, last_ordinal + 1, 1)
        yield from _lift(definition, step, base, tail, last_copy, 1, 0)
        return
    divisor = gcd(stride, outputs)
    period = outputs // divisor
    for residue in range(min(period, count)):
        copy, ordinal = divmod(slot + residue * stride, outputs)
        copies = (count - residue + period - 1) // period
        pieces = _output_pieces(child, ordinal, 1, 0)
        yield from _lift(definition, step, base, pieces, copy, copies, stride // divisor)


def _lift(
    definition: Definition,
    step: CallStep,
    base: int,
    pieces: Iterable[_Piece],
    copy: int,
    copies: int,
    copy_stride: int,
) -> Iterator[_Piece]:
    """Child pieces of copies ``copy + j * copy_stride`` (``j < copies``) in the parent.

    Gate runs shift by the copy's offset; pass-through inputs are fed by the
    step's argument ranges, so they become the parent's own coordinates and
    are resolved there in turn.
    """

    child = step.child
    for is_gate, run in pieces:
        if is_gate:
            origin = base + copy * child.size + run.start
            grid = _grid(origin, copies, copy_stride * child.size, run.count, run.stride)
            for start, count, stride in grid:
                yield True, Run(start, count, stride, run.width)
            continue
        for index, first, taken in _split(step.arg_starts, run.start, run.count, run.stride):
            item = step.args[index]
            element = run.start + first * run.stride - step.arg_starts[index]
            origin = item.element(element, copy)
            grid = _grid(
                origin, copies, copy_stride * item.jstride, taken, run.stride * item.stride
            )
            for start, count, stride in grid:
                if item.space == INPUT:
                    yield False, Run(start, count, stride, 0)
                else:
                    yield from _pieces(definition, LOCAL, start, count, stride)


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
