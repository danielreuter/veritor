"""Structural counts for systematic faults: what one wrong cell touches.

A random bit flip corrupts one VU; a *systematic* fault corrupts a *cell* --
a weight in HBM, a KV block at rest, a kernel path -- and every VU that reads
the cell is wrong for as long as the fault lasts.  The number of VU
declarations the protocol's M6 would need is therefore a property of the
circuit's wiring, not of the fault: how many VUs read one weight cell, how
many later steps read one KV boundary value, how many VUs a pod computes in
an hour.  This module computes those counts from the description.

Two ways.  :func:`brute_force_readers` walks every gate of a flat toy
circuit (:class:`~veritor.core.circuit.Circuit` ``args``) and maps the gate
to the VU owning it (:meth:`~veritor.core.index.Units.owner`): exact, and
linear in the circuit, so it is for the toy only.  :func:`readers` instead
propagates *reader multiplicities* through the description's call graph:
for each definition it computes how many VUs read each of its ports, as a
short list of arithmetic progressions with a multiplicity each
(:class:`Progression`), and at every call step maps a child's port
progressions through the step's argument ranges onto the parent's
coordinates (:func:`veritor.core.description._pieces`, the machinery the
index uses to resolve interfaces), collecting the pieces that land on a
pinned source gate (a weight or input cell) or on a gate of the root (a
boundary value produced by one RU and read by another).  Kinds are
resolved once each, so the work is proportional to the description, not the
circuit: GPT-2 Small's 1.9 G gates resolve in seconds where the flat walk
would take days.  On the toy the two agree exactly
(``tests/veritor/stress/test_honest_systematic.py``).

The counts are per *run*: the compiled circuit is one run's worth of
requests or cluster steps, and a per-cell count is the number of VUs of that
run reading the cell.  :func:`pod_scope` scales a run to the fleet and the
hour a systematic fault lasts.

The reader count is what the prover declares only if it recorded every VU's
output.  What an honest prover must declare depends on its recording policy
(``docs/honest-prover.md``, sections 3 and 9; the model is
``veritor.simulation.honest`` on the honest-sim branch): it replays an
opened RU from the values it recorded, pins the recorded values, and
declares the VUs whose recorded output disagrees with the recomputation.
With every VU output recorded (``VU_OUTPUTS``) a systematic fault pins every
reader of the cell whose output changed; with the committed positions only
(``BOUNDARY``: tokens under request RUs, KV and tokens under step RUs) it
pins the recorded outputs that came out wrong, and a weight fault costs the
tokens it flipped, not the readers.  :func:`perturbed_run` produces the
production assignment of a run with a faulty pod (its RUs misread a cell,
or store what a wrong kernel path computes) and :func:`pinned_units` is the
pinned replay over a recording, so both counts are measured on the toy.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from veritor.core import Compiled
from veritor.core.description import (
    INPUT,
    LOCAL,
    VERIFICATION,
    CallStep,
    Definition,
    GateStep,
    PieceKind,
    Range,
    _pieces,
    _split,
)
from veritor.core.indexed import iter_members

__all__ = [
    "Declarations",
    "PodScope",
    "Progression",
    "Readers",
    "SystematicPricing",
    "brute_force_readers",
    "histogram",
    "kv_consumers",
    "late_lowering_bits",
    "perturbed_run",
    "pinned_units",
    "pod_scope",
    "port_pardon_bits",
    "post_challenge_bits",
    "post_j_unit_bits",
    "price_systematic",
    "reader_count",
    "readers",
    "recorded_positions",
    "ru_scope_bits",
    "ru_scope_post_j_bits",
    "ru_scoped_source_bits",
    "source_pardon_bits",
    "weight_readers",
]


@dataclass(frozen=True, slots=True)
class Progression:
    """``count`` coordinates ``start + k * stride`` (``k < count``), each read by ``multiplicity`` VUs."""

    start: int
    count: int
    stride: int
    multiplicity: int

    @property
    def last(self) -> int:
        return self.start + (self.count - 1) * self.stride

    def contains(self, value: int) -> bool:
        if self.count == 1 or self.stride == 0:
            return value == self.start
        offset = value - self.start
        if self.stride > 0:
            return (
                0 <= offset
                and offset % self.stride == 0
                and offset // self.stride < self.count
            )
        return (
            offset <= 0
            and offset % self.stride == 0
            and offset // self.stride < self.count
        )

    def __iter__(self) -> Iterator[int]:
        for k in range(self.count):
            yield self.start + k * self.stride


@dataclass(frozen=True, slots=True)
class Readers:
    """Reader multiplicities of the root's cells, as progressions over absolute addresses.

    ``weights`` and ``inputs`` are the source gates the root itself holds (in
    the LM constructors the weights unit is called at the root, so every
    weight cell is here; the toy's prompt tokens are ``in`` gates inside the
    steps and are summarised per kind in ``pinned``), ``boundary`` the gates
    the root computes -- outputs of one RU -- read by VUs of other RUs, the KV
    cache and the token between steps of ``ClusterG``.  ``pinned`` is, per
    definition digest below the root, the progressions over that
    definition's *relative* offsets that land on source gates inside it, and
    ``orphans`` the number of gate reads by gates outside every VU (zero for
    every constructor here).
    """

    weights: tuple[Progression, ...]
    inputs: tuple[Progression, ...]
    boundary: tuple[Progression, ...]
    pinned: dict[str, tuple[Progression, ...]]
    orphans: int


def reader_count(progressions: Iterable[Progression], address: int) -> int:
    """The number of VUs reading ``address``: the multiplicities of the progressions containing it."""

    return sum(p.multiplicity for p in progressions if p.contains(address))


def histogram(
    progressions: Iterable[Progression], domain: Iterable[int] | None = None
) -> dict[int, int]:
    """``{readers: cells}`` over ``domain`` (or over the addresses the progressions touch).

    A ``range`` domain is accumulated in a NumPy array with one slice per
    progression (GPT-2 Small's 124 M weight cells in well under a second);
    contiguous progressions (stride ``0`` or ``1``) without a domain are swept
    by their end points; otherwise the cells are enumerated.
    """

    items = tuple(progressions)
    if isinstance(domain, range) and domain.step == 1 and len(domain) > 0:
        counts = np.zeros(len(domain), dtype=np.int64)
        low, high = domain.start, domain.stop
        for p in items:
            if p.count == 1 or p.stride == 0:
                if low <= p.start < high:
                    counts[p.start - low] += p.multiplicity
                continue
            first, last = (p.start, p.last) if p.stride > 0 else (p.last, p.start)
            stride = abs(p.stride)
            if last < low or first >= high:
                continue
            if first < low:
                first += ((low - first + stride - 1) // stride) * stride
            counts[first - low : min(last, high - 1) - low + 1 : stride] += (
                p.multiplicity
            )
        sizes = np.bincount(counts)
        return {int(v): int(n) for v, n in enumerate(sizes) if n}
    if domain is None and all(p.stride in (0, 1) for p in items):
        events: Counter[int] = Counter()
        for p in items:
            events[p.start] += p.multiplicity
            events[p.last + 1] -= p.multiplicity
        result: Counter[int] = Counter()
        level, previous = 0, None
        for position in sorted(events):
            if previous is not None and level > 0:
                result[level] += position - previous
            level += events[position]
            previous = position
        return dict(sorted(result.items()))
    cells: Counter[int] = Counter()
    if domain is None:
        for p in items:
            for address in p:
                cells[address] += p.multiplicity
    else:
        for address in domain:
            cells[address] = reader_count(items, address)
    return dict(sorted(Counter(cells.values()).items()))


# -- the structural resolution ---------------------------------------------------------


def _runs(ordinals: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Maximal runs ``(start, count)`` of consecutive integers in sorted ``ordinals``."""

    start = previous = None
    for value in ordinals:
        if previous is not None and value == previous + 1:
            previous = value
            continue
        if start is not None and previous is not None:
            yield start, previous - start + 1
        start = previous = value
    if start is not None and previous is not None:
        yield start, previous - start + 1


_Key = tuple[int, int, int]
"""``(start, count, stride)``: progressions with one key are the same cells and add up."""


def _merged(counts: Counter[_Key]) -> tuple[Progression, ...]:
    return tuple(
        Progression(start, count, stride, multiplicity)
        for (start, count, stride), multiplicity in sorted(counts.items())
    )


class _Resolver:
    """Per-definition reader multiplicities, resolved once per kind (digest)."""

    def __init__(self, root: Definition) -> None:
        self.root = root
        self.ports: dict[str, tuple[Progression, ...]] = {}
        self.pinned: dict[str, Counter[_Key]] = {}
        self.gates: Counter[_Key] = (
            Counter()
        )  # the root's own gates read by its children
        self.orphans = 0

    def port_readers(self, definition: Definition) -> tuple[Progression, ...]:
        cached = self.ports.get(definition.digest)
        if cached is not None:
            return cached
        if definition.role == VERIFICATION:
            result = tuple(
                Progression(start, count, 1 if count > 1 else 0, 1)
                for start, count in _runs(definition.reads)
            )
        else:
            counts: Counter[_Key] = Counter()
            for p in self._resolve(definition):
                counts[(p.start, p.count, p.stride)] += p.multiplicity
            result = _merged(counts)
        self.ports[definition.digest] = result
        return result

    def _resolve(self, definition: Definition) -> Iterator[Progression]:
        at_root = definition is self.root
        pinned = self.pinned.setdefault(definition.digest, Counter())
        droppable = self._droppable(definition)
        for position, step in enumerate(definition.steps):
            if isinstance(step, GateStep):
                if step.pinned:
                    continue
                # A gate outside every VU: its reads belong to no unit.
                self.orphans += sum(item.count for item in step.args)
                continue
            child = step.child
            is_unit = child.role == VERIFICATION
            seen: set[Progression] = set()
            flags = droppable[position]
            for p in self.port_readers(child):
                first = bisect_right(step.arg_starts, p.start) - 1
                last = bisect_right(step.arg_starts, p.last) - 1
                if not at_root and all(flags[first : last + 1]):
                    continue
                for index, k, taken in _split(
                    step.arg_starts, p.start, p.count, p.stride
                ):
                    if not at_root and flags[index]:
                        continue
                    item = step.args[index]
                    element = p.start + k * p.stride - step.arg_starts[index]
                    yield from self._lift(
                        definition, step, item, element, taken, p, is_unit, seen, pinned
                    )

    def _lift(
        self,
        definition: Definition,
        step: CallStep,
        item: Range,
        element: int,
        taken: int,
        p: Progression,
        is_unit: bool,
        seen: set[Progression],
        pinned: Counter[_Key],
    ) -> Iterator[Progression]:
        """Map ``taken`` ports of every copy of the child onto the parent's coordinates."""

        multiplicity = p.multiplicity
        columns, column_stride = taken, p.stride * item.stride
        if column_stride == 0 and columns > 1:
            # Several ports of one copy read the same coordinate: one VU reading it
            # through several ports counts once; distinct VUs behind distinct ports add.
            multiplicity, columns = (
                (multiplicity if is_unit else multiplicity * columns),
                1,
            )
        rows, row_stride = step.count, item.jstride
        if row_stride == 0 and rows > 1:
            multiplicity, rows = (
                multiplicity * rows,
                1,
            )  # every copy reads the same coordinate
        origin = item.element(element, 0)
        for start, count, stride in _grid(
            origin, rows, row_stride, columns, column_stride
        ):
            if item.space == INPUT:
                q = Progression(start, count, stride, multiplicity)
                if is_unit:
                    if q in seen:
                        continue
                    seen.add(q)
                yield q
                continue
            for kind, run in _pieces(definition, LOCAL, start, count, stride):
                q_count, q_multiplicity = run.count, multiplicity
                if run.stride == 0 and run.count > 1:
                    q_count = 1
                    if not is_unit:
                        q_multiplicity *= run.count
                q = Progression(run.start, q_count, run.stride, q_multiplicity)
                if is_unit:
                    if q in seen:
                        continue
                    seen.add(q)
                if kind is PieceKind.PORT:
                    yield q
                elif kind is PieceKind.PINNED:
                    pinned[(q.start, q.count, q.stride)] += q.multiplicity
                elif definition is self.root:
                    self.gates[(q.start, q.count, q.stride)] += q.multiplicity

    def _droppable(self, definition: Definition) -> dict[int, tuple[bool, ...]]:
        """Per call step (by position) and argument: whether the argument's slots
        all resolve to gates the definition computes itself (neither sources nor
        ports), which a definition below the root need not follow."""

        steps = definition.steps
        gate_only = [
            (not s.pinned)
            if isinstance(s, GateStep)
            else all(kind is PieceKind.GATE for kind, _ in s.child.resolved_outputs)
            for s in steps
        ]
        bad_before = [0]
        for flag in gate_only:
            bad_before.append(bad_before[-1] + (0 if flag else 1))
        slots = definition.step_slot
        result: dict[int, tuple[bool, ...]] = {}
        for position, step in enumerate(steps):
            if not isinstance(step, CallStep):
                continue
            flags = []
            for item in step.args:
                if item.space != LOCAL:
                    flags.append(False)
                    continue
                corners = [
                    item.element(e, c)
                    for e in (0, item.count - 1)
                    for c in (0, step.count - 1)
                ]
                low = bisect_right(slots, min(corners)) - 1
                high = bisect_right(slots, max(corners)) - 1
                flags.append(bad_before[high + 1] - bad_before[low] == 0)
            result[position] = tuple(flags)
        return result


def _grid(
    start: int, rows: int, row_stride: int, columns: int, column_stride: int
) -> Iterator[tuple[int, int, int]]:
    """Progressions covering ``start + r * row_stride + c * column_stride``.

    As ``description._grid``, with one more tiling: when the columns tile
    over the rows (``column_stride == rows * row_stride``, the column-strided
    reads of a row-major matrix by the ``m`` dots of a matvec) the family is
    one progression too.  The pieces are read counts, not addresses, so
    order does not matter and any exact cover will do.
    """

    if rows == 1:
        yield start, columns, column_stride if columns > 1 else 0
    elif columns == 1:
        yield start, rows, row_stride
    elif row_stride == columns * column_stride:
        yield start, rows * columns, column_stride
    elif column_stride == rows * row_stride:
        yield start, rows * columns, row_stride
    elif columns <= rows:
        for c in range(columns):
            yield start + c * column_stride, rows, row_stride
    else:
        for r in range(rows):
            yield start + r * row_stride, columns, column_stride


def readers(compiled: Compiled) -> Readers:
    """Reader multiplicities of every source and boundary cell, from the description."""

    root = compiled.index.root.frame.definition
    resolver = _Resolver(root)
    resolver.port_readers(root)
    circuit = compiled.circuit
    weights: list[Progression] = []
    inputs: list[Progression] = []
    for p in _merged(resolver.pinned.get(root.digest, Counter())):
        (weights if circuit[p.start].is_weight else inputs).append(p)
    pinned = {
        digest: _merged(items)
        for digest, items in resolver.pinned.items()
        if digest != root.digest and items
    }
    return Readers(
        tuple(weights), tuple(inputs), _merged(resolver.gates), pinned, resolver.orphans
    )


def weight_readers(compiled: Compiled, address: int) -> int:
    """How many VUs of the run read the weight cell at ``address``."""

    if not compiled.circuit[address].is_weight:
        raise ValueError(f"address {address} is not a weight gate")
    return reader_count(readers(compiled).weights, address)


def kv_consumers(compiled: Compiled, address: int) -> int:
    """How many VUs of *other* RUs read the boundary value at ``address``.

    For ``ClusterG`` a step's outputs are its occupants' new KV entries and
    tokens; a KV entry is read by the attention VUs of every later step of
    the same request, a token by the next step's embedding.
    """

    return reader_count(readers(compiled).boundary, address)


# -- the flat walk, for validation -----------------------------------------------------


def brute_force_readers(compiled: Compiled) -> tuple[dict[int, int], dict[int, int]]:
    """``(sources, boundary)``: distinct VUs reading each source gate, and each RU
    output read from another RU, by walking every gate of the circuit."""

    circuit, index = compiled.circuit, compiled.index
    replay_of: dict[int, int] = {}
    unit_of: dict[int, int] = {}
    for replay in range(index.replay_units.count):
        units = index.verification_units(replay)
        for offset, node in enumerate(units):
            for address in node.interval:
                unit_of[address] = units.first + offset
        for address in index.replay_units.unit(replay).interval:
            replay_of[address] = replay
    sources: dict[int, set[int]] = {}
    boundary: dict[int, set[int]] = {}
    for address in range(index.n):
        gate = circuit[address]
        if gate.is_source:
            continue
        unit = unit_of.get(
            address, -1 - address
        )  # a gate outside every VU reads as itself
        replay = replay_of.get(address, -1)
        for argument in gate.args:
            if circuit[argument].is_source:
                sources.setdefault(argument, set()).add(unit)
            elif replay_of.get(argument, -1) != replay:
                boundary.setdefault(argument, set()).add(unit)
    return (
        {address: len(units_) for address, units_ in sorted(sources.items())},
        {address: len(units_) for address, units_ in sorted(boundary.items())},
    )


# -- scaling a run to a fleet-hour -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class PodScope:
    """What an hour-long fault on one pod touches, in the units the protocol counts."""

    steps: int
    """Cluster steps the pod computes in the window."""
    positions: int
    """Token positions (prefill and decode) it processes."""
    verification_units: int
    """VUs it computes."""
    replay_units: int
    """RUs those VUs lie in (steps under ``ClusterG``; requests under ``RequestsG``)."""
    requests: float
    """Requests served, in whole or in part."""


def pod_scope(
    *,
    slots: int,
    step_seconds: float,
    hours: float,
    units_per_position: float,
    positions_per_request: float,
    replay_unit: str = "step",
) -> PodScope:
    """Scale one pod's hour: ``slots`` occupants advance one position per synchronous step."""

    if slots < 1 or step_seconds <= 0 or hours <= 0:
        raise ValueError("slots, step_seconds and hours must be positive")
    if units_per_position <= 0 or positions_per_request <= 0:
        raise ValueError(
            "units_per_position and positions_per_request must be positive"
        )
    if replay_unit not in ("step", "request"):
        raise ValueError("replay_unit is 'step' or 'request'")
    steps = math.floor(hours * 3600 / step_seconds)
    positions = steps * slots
    requests = positions / positions_per_request
    return PodScope(
        steps=steps,
        positions=positions,
        verification_units=math.ceil(positions * units_per_position),
        replay_units=steps if replay_unit == "step" else math.ceil(requests),
        requests=requests,
    )


# -- the honest prover's declarations, measured ----------------------------------------


def recorded_positions(compiled: Compiled, *, interiors: bool) -> frozenset[int]:
    """The addresses a server keeps of a run under one recording policy.

    The boundary and the weights always: they are the committed positions,
    and the computed ones among them are the streamed tokens under request
    RUs, the KV entries and tokens that cross steps under step RUs.  With
    ``interiors`` every RU's interior positions too, every VU's output word
    (the ``VU_OUTPUTS`` policy); without them the committed positions only
    (``BOUNDARY``).  A VU's internal gates are recorded under neither.
    """

    index = compiled.index
    addresses = set(iter_members(index.boundary())) | set(iter_members(index.weights()))
    if interiors:
        for unit in range(index.replay_units.count):
            addresses.update(iter_members(index.interior(unit)))
    return frozenset(addresses)


def perturbed_run(
    compiled: Compiled,
    inputs: Sequence[int],
    weights: Sequence[int],
    *,
    units: Collection[int],
    misread: Mapping[int, int] | None = None,
    corrupt: Callable[[int, int], int] | None = None,
) -> dict[int, int]:
    """The production assignment of a run whose RUs ``units`` ran on a faulty pod.

    Every gate is evaluated in address order from the true inputs and weights
    (what the boundary and ``kappa_W`` commit).  A gate of one of ``units``
    reads each address in ``misread`` as the value given there instead of
    the stored one -- a weight cell corrupted in the pod's memory, a stale
    version, a KV word rotted at rest -- and, with ``corrupt``, stores
    ``corrupt(address, value)`` in place of what it computed -- a wrong
    kernel path.  Everything downstream is computed from the corrupted
    values, on the faulty pod and off it, as the datacenter did.
    """

    circuit, index = compiled.circuit, compiled.index
    faulty: set[int] = set()
    for unit in units:
        faulty.update(index.replay_units.unit(unit).interval)
    given = {"input": iter(tuple(inputs)), "weight": iter(tuple(weights))}
    reads = dict(misread or {})
    values: dict[int, int] = {}
    for address in range(circuit.n):
        ref = circuit[address]
        if ref.is_source:
            values[address] = next(given[ref.source])  # type: ignore[index]
            continue
        inside = address in faulty
        if inside and reads:
            arguments = tuple(reads.get(a, values[a]) for a in ref.args)
        else:
            arguments = tuple(values[a] for a in ref.args)
        value = circuit.evaluate_gate(address, arguments)
        if inside and corrupt is not None:
            value = corrupt(address, value)
        values[address] = value
    return values


def pinned_units(
    compiled: Compiled,
    values: Mapping[int, int],
    recorded: Collection[int],
    units: Iterable[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Per RU, the VUs the honest prover declares when it replays from its recording.

    The rule is the honest-prover model's pinned replay.  The RU's gates are
    recomputed in address order from the recorded values (sources and
    boundary) and the values recomputed so far; at a recorded address the
    recorded value is what the interior holds and what later gates read, and
    when it differs from the recomputation the VU owning the address is
    pinned.  A reader of a pinned value is recomputed from that value, so
    its relation holds; the pinned VUs are exactly what M6 must declare for
    the RU to be accepted whatever the s-challenge samples.  ``values`` is
    the production assignment (:func:`perturbed_run`), ``recorded`` the
    addresses the server kept (:func:`recorded_positions`): the two policies
    differ only in ``recorded``.  ``units`` restricts the replay to some RUs
    (default: every RU of the run).
    """

    circuit, index = compiled.circuit, compiled.index
    kept = frozenset(recorded)
    result: dict[int, tuple[int, ...]] = {}
    replayed = range(index.replay_units.count) if units is None else units
    for unit in replayed:
        interior: dict[int, int] = {}
        disagreeing: list[int] = []
        for address in index.replay_units.unit(unit).interval:
            gate = circuit[address]
            if gate.is_source:
                continue
            arguments: list[int] = []
            for argument in gate.args:
                if argument in interior:
                    arguments.append(interior[argument])
                elif argument in kept:
                    arguments.append(values[argument])
                else:
                    raise KeyError(
                        f"replay of unit {unit} needs address {argument}, which is "
                        "neither recorded nor computed by the unit"
                    )
            value = circuit.evaluate_gate(address, tuple(arguments))
            if address in kept:
                if values[address] != value:
                    disagreeing.append(address)
                value = values[address]
            interior[address] = value
        nodes = index.verification_units(unit)
        result[unit] = tuple(
            sorted({nodes.first + nodes.owner(a) for a in disagreeing})
        )
    return result


# -- the prices of the mechanisms -------------------------------------------------------
#
# Stage prices as docs/notes/late-advice.md fixes them (docs/honest-prover.md,
# section 2).  A declaration fixed before the q-challenge costs its message:
# log2 of the number of things it could have said.  One made after J is
# adaptive and costs what ``veritor.analysis.faults`` charges; per VU
# declaration that is ``u_post(1) = rho log2 (1 / (1 - s))``, about
# ``(u(1) + 1) / q`` at the scattered channel (``post_j_unit_bits``).  Nothing
# may be declared after the s-challenge.  The kinds of
# docs/notes/declaration-kinds.md are priced by the same two arguments: a
# run-wide source-position pardon makes every opened reader of the cell
# answer to one value, so an adaptive prover cannot mix honest and corrupted
# RUs under it and its post-J price is its message, without the ``1 / q``
# leverage (the note states the condition and flags what is not proved); an
# RU-scoped one has the selective-opening leverage of a VU declaration and is
# priced like ``declared_bits`` with the pardon's message in place of ``u(1)``.
# A structural amendment (a different lowering of an RU) is admitted post-J at
# its compile-time price ``log2 |V_R|`` only under the note's conditions.


def post_j_unit_bits(rho: float, s: Fraction | float) -> float:
    """``u_post(1) = rho log2 (1 / (1 - s))``: what one VU declaration made after ``J`` adds to ``U``.

    Bound (i) of :mod:`veritor.analysis.faults` read off the fold's slope
    ``rho`` (:attr:`~veritor.analysis.bound.BoundResult.rho`): the note's
    price of a post-J declaration, ``145.6`` bits at the simulation policy
    and ``6.1e9`` at the headline.
    """

    if rho < 0 or not 0 <= s < 1:
        raise ValueError("rho must be nonnegative and s in [0, 1)")
    return rho * math.log2(1 / (1 - float(s)))


def post_challenge_bits(
    content_bits: float, q: Fraction | float, count: int = 1
) -> float:
    """``count`` declarations of ``content_bits`` each, made after ``J``: ``count * content / q``,
    the small-``s`` summary of the post-J price (``u_post(1) ~ (u(1) + 1) / q``)."""

    if content_bits < 0 or count < 0 or not 0 < q <= 1:
        raise ValueError("content and count must be nonnegative, q in (0, 1]")
    return count * content_bits / float(q)


def source_pardon_bits(width: int, positions: int, scopes: int = 1) -> float:
    """The message of one source-position pardon: ``log2 (1 + scopes * positions * 2**width)``.

    It names the scope among ``scopes`` (the run, or one of the run's pods),
    the cell among ``positions`` pardonable ones and the value ``v'`` in
    ``width`` bits: about ``width + log2 (scopes * positions)``, ``51.9`` at
    the headline (``16 + log2 n_W``).  It is the pre-J price of the pardon at
    any scope and the post-J price of a run-wide one where every opened
    reader must answer to the same value.
    """

    if width < 1 or positions < 1 or scopes < 1:
        raise ValueError("width, positions and scopes must be positive")
    return math.log2(1 + scopes * positions * 2.0**width)


def ru_scoped_source_bits(
    rho: float,
    s: Fraction | float,
    readers: int,
    content_bits: float,
    choices_bits: float,
) -> float:
    """One RU-scoped source-position pardon made after ``J``, priced like ``declared_bits``.

    The smaller of bound (i), the ``readers`` errors the pardon can remove
    from the opened RU (the cell's readers in it) at ``u_post(1)`` each, and
    bound (ii), the union over the ``2**choices_bits`` pardons it could have
    been (``rho * choices_bits``, the threshold divided by their number) plus
    its own message ``content_bits``.  At the headline (1024 readers) bound
    (i) is the smaller and is ``0.33 U_0``; the note prohibits the kind for
    want of anything cheaper.
    """

    if readers < 0 or content_bits < 0 or choices_bits < 0:
        raise ValueError("readers, content and choices must be nonnegative")
    return min(readers * post_j_unit_bits(rho, s), rho * choices_bits + content_bits)


def port_pardon_bits(width: int, ports: int) -> float:
    """The message of one port pardon: ``v'`` and the port among ``ports`` (``n_RU * n_ports``)."""

    if width < 1 or ports < 1:
        raise ValueError("width and ports must be positive")
    return math.log2(1 + ports * 2.0**width)


def ru_scope_bits(out_bits: int, replay_units: int) -> float:
    """The message of one RU-scope pardon, its pre-J price: the RU among ``replay_units``
    and its ``out_bits`` of now-free outputs."""

    if out_bits < 0 or replay_units < 1:
        raise ValueError("out_bits must be nonnegative and replay_units positive")
    return out_bits + math.log2(1 + replay_units)


def ru_scope_post_j_bits(
    rho: float, q: Fraction | float, out_bits: int, replay_units: int
) -> float:
    """The note's bound for one RU-scope pardon made after ``J``:
    ``rho log2 (1 + q n_RU / (1 - q)) + W_R + log2 n_RU`` (``0.47 U_0`` at the headline)."""

    if rho < 0 or not 0 < q < 1:
        raise ValueError("rho must be nonnegative and q in (0, 1)")
    shift = math.log2(1 + float(q) * replay_units / (1 - float(q)))
    return rho * shift + ru_scope_bits(out_bits, replay_units)


def late_lowering_bits(replay_units: int, variants: int) -> float:
    """A structural amendment naming one of ``variants`` lowerings for each of
    ``replay_units`` RUs, at the compile-time price ``log2 |V_R|`` per RU."""

    if replay_units < 0 or variants < 1:
        raise ValueError("replay_units must be nonnegative and variants positive")
    return replay_units * math.log2(variants)


@dataclass(frozen=True, slots=True)
class Declarations:
    """What M6 costs the honest prover for one systematic fault under one recording policy."""

    faulty: int
    """VUs whose recorded output the fault changed over the window: what pinned replay pins."""
    opened: int
    """``ceil(q * faulty)``: the declarations P1 makes, those in opened RUs."""
    f_max: int
    bits: float
    """``opened * u_post(1)``: the linearised post-J charge, before the interface cap."""

    @property
    def admissible(self) -> bool:
        return self.opened <= self.f_max


@dataclass(frozen=True, slots=True)
class SystematicPricing:
    """One systematic scenario priced under every mechanism of section 6.

    ``readers`` is M6 under the ``VU_OUTPUTS`` recording (every reader whose
    output changed), ``flipped`` M6 under ``BOUNDARY`` (every recorded output
    that came out wrong); each ``*_bits`` is the charge of one alternative,
    ``None`` where it does not apply to the scenario: ``source_bits`` the
    run-wide source-position pardons (one message each), ``ru_scoped_source_bits``
    one RU-scoped pardon per opened affected RU at the ``declared_bits``-like
    price, ``ru_scope_bits`` the pre-J price of withdrawing every affected RU
    (post-J the kind is prohibited), ``lowering_bits`` a late lowering at its
    compile-time price, ``configuration_bits`` a public per-pod configuration
    (M2/M8).  ``reserve_fraction`` is the share of the window's production
    computation that rejection would re-serve.
    """

    readers: Declarations
    flipped: Declarations
    source_bits: float | None
    ru_scoped_source_bits: float | None
    ru_scope_bits: float
    lowering_bits: float | None
    configuration_bits: float | None
    reserve_fraction: float


def price_systematic(
    *,
    readers: int,
    flipped: int,
    q: Fraction | float,
    f_max: int,
    unit_post_bits: float,
    affected_replay_units: int,
    replay_units: int,
    out_bits: int,
    reserve_fraction: float,
    source_pardons: int | None = None,
    source_content_bits: float | None = None,
    ru_scoped_bits_each: float | None = None,
    lowering_variants: int | None = None,
    configurable: bool = False,
) -> SystematicPricing:
    """Price one systematic scenario.

    P1 declares the ``q * readers`` (or ``q * flipped``) pinned VUs that lie
    in opened RUs at ``unit_post_bits`` each; ``source_pardons`` run-wide
    source-position pardons cost their message ``source_content_bits`` each;
    RU-scoped pardons cost ``ru_scoped_bits_each`` for each opened affected
    RU; withdrawing every affected RU costs ``ru_scope_bits`` each before
    ``J``; a late lowering costs ``log2 (variants)`` per affected RU; a public
    per-pod configuration costs nothing when the constructor can carry it.
    """

    if readers < 0 or flipped < 0 or affected_replay_units < 0 or replay_units < 1:
        raise ValueError("counts must be nonnegative and replay_units positive")
    if not 0 <= reserve_fraction <= 1:
        raise ValueError("reserve_fraction is a fraction of the window")
    if unit_post_bits < 0 or not 0 < q <= 1:
        raise ValueError("unit_post_bits must be nonnegative and q in (0, 1]")

    def declarations(faulty: int) -> Declarations:
        opened = math.ceil(faulty * float(q))
        return Declarations(faulty, opened, f_max, opened * unit_post_bits)

    opened_units = math.ceil(affected_replay_units * float(q))
    return SystematicPricing(
        readers=declarations(readers),
        flipped=declarations(flipped),
        source_bits=(
            None
            if source_pardons is None or source_content_bits is None
            else source_pardons * source_content_bits
        ),
        ru_scoped_source_bits=(
            None if ru_scoped_bits_each is None else opened_units * ru_scoped_bits_each
        ),
        ru_scope_bits=affected_replay_units * ru_scope_bits(out_bits, replay_units),
        lowering_bits=(
            None
            if lowering_variants is None
            else late_lowering_bits(affected_replay_units, lowering_variants)
        ),
        configuration_bits=0.0 if configurable else None,
        reserve_fraction=reserve_fraction,
    )
