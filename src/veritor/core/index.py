"""The index ``I``: nested subcircuits over ``C``'s addresses.

Every copy of a definition is a node: an address interval, a kind (the
definition digest), a role mark, lazy children and a depth.  Two antichains
are designated by the role marks: every copy of a definition marked
``"replay"`` is a replay unit and every copy marked ``"verification"`` is a
verification unit.  Both are lazy sequences: ``count``, ``unit(k)`` and
``owner(address)`` all cost ``O(depth)`` through the prefix sums stored on the
definitions (bisect within a step list, divide within a ``repeat``).

The circuit's inputs and weights are source gates inside the units.  The
input gates ``In``, the weight gates ``W``, the boundary
``In ∪ ⋃_r Out(R_r)`` and each interior ``R_r \\ (boundary ∪ W)`` are lazy
address sets built the same way; nothing about them is stored.

The per-kind table (:meth:`Index.kinds`) also records which kinds are
*closed*: fed nothing but source gates at every call site, hence
re-executable from the inputs and the weights alone.  That is what the cost
model needs to price the recomputation behind a sampled replay unit.

Canonical chunking of long step lists (so a cut can fall inside a definition)
is a later phase: today a unit is always a whole copy of a definition.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import gcd

from .description import (
    INPUT,
    REPLAY,
    VERIFICATION,
    CallStep,
    Definition,
    Frame,
    GateStep,
    PieceKind,
    Range,
    Run,
    _call_pieces,
    _sorted_runs,
    _split,
    progression_meet,
)
from .errors import InvalidArtifact
from .gates import INPUT_SOURCE, WEIGHT_SOURCE
from .identity import Digest, identity_digest
from .indexed import IndexedDomain, IntervalDifferenceDomain
from .limits import CompilationLimits


class IndexNode:
    """One copy of a definition: an interval of addresses with a kind and role."""

    __slots__ = ("frame",)

    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    @property
    def interval(self) -> range:
        return self.frame.interval

    @property
    def kind(self) -> str:
        return self.frame.definition.digest

    @property
    def role(self) -> str | None:
        return self.frame.definition.role

    @property
    def depth(self) -> int:
        return self.frame.depth

    @property
    def size(self) -> int:
        return self.frame.definition.size

    @property
    def replay_unit(self) -> int | None:
        """The replay unit this node is, or lies inside; ``None`` above the cut."""

        frame: Frame | None = self.frame
        while frame is not None:
            if frame.definition.role == REPLAY:
                return frame.replay_before
            frame = frame.parent
        return None

    def children(self) -> Iterator[IndexNode]:
        """The copies called by this node's steps, in layout order."""

        for index, step in enumerate(self.frame.definition.steps):
            if isinstance(step, CallStep):
                for copy in range(step.count):
                    yield IndexNode(self.frame.child(index, copy))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IndexNode)
            and self.kind == other.kind
            and self.frame.base == other.frame.base
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.frame.base))

    def __repr__(self) -> str:
        return f"IndexNode({self.kind[:12]}, {self.interval}, role={self.role})"


def _unit_frame_at(frame: Frame, address: int, role: str) -> Frame | None:
    """Descend from ``frame`` to the copy marked ``role`` containing ``address``."""

    offset = address - frame.base
    while frame.definition.role != role:
        definition = frame.definition
        index = definition.step_at_address(offset)
        step = definition.steps[index]
        if isinstance(step, GateStep):
            return None
        copy, offset = divmod(offset - definition.step_address[index], step.child.size)
        frame = frame.child(index, copy)
    return frame


def _frame_by_rank(
    frame: Frame,
    rank: int,
    role: str,
    prefix: Callable[[Definition], tuple[int, ...]],
    total: Callable[[Definition], int],
) -> tuple[Frame, int]:
    """Descend by the prefix sums of a per-definition count to the ``role`` copy.

    Returns the copy and the rank relative to it.
    """

    while frame.definition.role != role:
        definition = frame.definition
        sums = prefix(definition)
        index = bisect_right(sums, rank) - 1
        step = definition.steps[index]
        assert isinstance(step, CallStep)
        copy, rank = divmod(rank - sums[index], total(step.child))
        frame = frame.child(index, copy)
    return frame, rank


class Units:
    """A lazy antichain of unit nodes below ``root``: the copies marked ``role``."""

    __slots__ = ("_root", "count", "first", "role")

    def __init__(self, root: Frame, role: str, first: int) -> None:
        self._root = root
        self.role = role
        self.first = first
        definition = root.definition
        self.count = (
            definition.replay_units if role == REPLAY else definition.verification_units
        )

    def _check(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < self.count:
            raise IndexError(f"{self.role} unit {index} does not exist")
        return index

    def unit(self, index: int) -> IndexNode:
        rank = self._check(index)
        if self.role == REPLAY:
            frame, _ = _frame_by_rank(
                self._root,
                rank,
                REPLAY,
                lambda d: d.step_replay,
                lambda d: d.replay_units,
            )
        else:
            frame, _ = _frame_by_rank(
                self._root,
                rank,
                VERIFICATION,
                lambda d: d.step_verification,
                lambda d: d.verification_units,
            )
        return IndexNode(frame)

    def owner(self, address: int) -> int:
        """The index of the unit containing ``address``."""

        if type(address) is not int or address not in self._root.interval:
            raise KeyError(address)
        frame = _unit_frame_at(self._root, address, self.role)
        if frame is None:
            raise KeyError(address)
        if self.role == REPLAY:
            return frame.replay_before - self._root.replay_before
        return frame.verification_before - self._root.verification_before

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[IndexNode]:
        for index in range(self.count):
            yield self.unit(index)


@dataclass(frozen=True, slots=True)
class KindSummary:
    """The profiler's row for one kind: what every copy shares.

    Copies of a kind are isomorphic, so everything a fold over the index
    needs is computed once per kind here and weighted by ``copies``.
    ``input_count`` and ``out_count`` are the sizes of the *declared*
    interfaces of one copy: its ports as declared (a superset of what its
    gates read, so pricing by it is conservative) and ``Out``, its declared
    outputs resolved to the unpinned gates it owns; ``out_bits`` is the width
    of ``Out`` in bits.  ``source_inputs`` and ``source_weights`` count the
    input and weight gates inside one copy.  ``children`` counts, per child
    kind, the copies one copy of this kind calls directly;
    ``verification_units`` and ``verification_kinds`` describe the
    verification units inside one copy (a verification kind contains itself).

    ``closed`` says whether every port of the kind, at every call site of
    every copy, is fed a *retained* value: a source gate (``in`` or
    ``weight``), directly or through a port of the caller that is itself
    retained.  A closed kind can be re-executed from what an honest prover
    keeps (the circuit's inputs and the weights); anything else needs the
    values of the computation around it.  The root has no ports and is
    closed; so is every kind without ports.
    """

    kind: str
    role: str | None
    copies: int
    size: int
    replay_cost: int
    proof_cost: int
    input_count: int
    out_count: int
    out_bits: int
    source_inputs: int
    source_weights: int
    min_depth: int
    max_depth: int
    children: tuple[tuple[str, int], ...]
    verification_units: int
    verification_kinds: tuple[tuple[str, int], ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class KindTable:
    """The rows of :meth:`Index.kinds` with the totals a fold over them reads.

    ``Bound``, ``Cost`` and ``expected_work`` are functions of this table
    alone, so they accept it in place of a :class:`Compiled` artifact: a
    table written from a model of a circuit (``veritor.evaluation``) is
    priced exactly like one profiled from a compiled description.  ``root``
    is the kind of the root definition, ``n`` the number of gates.
    """

    rows: tuple[KindSummary, ...]
    root: str
    n: int
    input_count: int
    weight_count: int
    replay_unit_count: int
    digest: Digest

    def __post_init__(self) -> None:
        kinds = {row.kind for row in self.rows}
        if len(kinds) != len(self.rows):
            raise ValueError("kind table rows must have distinct kinds")
        if self.root not in kinds:
            raise ValueError("kind table root must be one of its rows")
        for row in self.rows:
            for child, _ in row.children:
                if child not in kinds:
                    raise ValueError(f"kind {row.kind} calls unknown kind {child}")
            if row.input_count == 0 and not row.closed:
                raise ValueError(f"kind {row.kind} has no ports and must be closed")


class Index:
    """The hierarchy of copies over the circuit, with its two unit antichains."""

    __slots__ = ("_frame", "digest", "replay_units", "root")

    def __init__(self, root: Definition, limits: CompilationLimits | None = None) -> None:
        validate_marks(root, CompilationLimits() if limits is None else limits)
        self._frame = Frame.root(root)
        self.root = IndexNode(self._frame)
        self.replay_units = Units(self._frame, REPLAY, 0)
        self.digest: Digest = identity_digest("veritor/index/v1", {"root": root.digest})

    @property
    def input_count(self) -> int:
        """The number of input gates ``|In|`` (the root has no ports)."""

        return self._frame.definition.input_total

    @property
    def weight_count(self) -> int:
        """The number of weight gates ``|W|``."""

        return self._frame.definition.weight_total

    @property
    def n(self) -> int:
        return self._frame.definition.size

    def inputs(self) -> IndexedDomain[int]:
        """``In``: the addresses of the input gates, ranked in address order."""

        return _Sources(self, INPUT_SOURCE)

    def weights(self) -> IndexedDomain[int]:
        """``W``: the addresses of the weight gates, ranked in address order."""

        return _Sources(self, WEIGHT_SOURCE)

    def verification_units(self, replay_unit: int) -> Units:
        """The verification units inside replay unit ``replay_unit``.

        ``first`` is the global index of its unit 0: verification units are
        numbered globally in layout order, block by replay unit.
        """

        frame = self.replay_units.unit(replay_unit).frame
        return Units(frame, VERIFICATION, frame.verification_before)

    @property
    def verification_unit_count(self) -> int:
        return self._frame.definition.verification_units

    def verification_unit(self, index: int) -> IndexNode:
        """The verification unit with global index ``index``."""

        if type(index) is not int or not 0 <= index < self.verification_unit_count:
            raise IndexError(f"verification unit {index} does not exist")
        frame, _ = _frame_by_rank(
            self._frame,
            index,
            VERIFICATION,
            lambda d: d.step_verification,
            lambda d: d.verification_units,
        )
        return IndexNode(frame)

    def boundary(self) -> IndexedDomain[int]:
        """``In ∪ ⋃_r Out(R_r)``: the addresses the boundary commitment covers.

        The input gates by rank, then the units' ``Out`` in unit order.  The
        weight gates are committed under their own root and are not here.
        The circuit outputs are always inside this set: every output resolves
        through the declared interface of the replay unit that owns it.
        """

        return _Boundary(self)

    def interior(self, replay_unit: int) -> IntervalDifferenceDomain:
        """``R_r`` minus the boundary and the weights: its interval minus ``Out`` and its pinned runs."""

        frame = self.replay_units.unit(replay_unit).frame
        definition = frame.definition
        return IntervalDifferenceDomain(
            frame.base,
            frame.base + definition.size,
            (
                (frame.base + run.start, run.count, run.stride)
                for runs in (definition.out_runs, definition.input_runs, definition.weight_runs)
                for run in runs
            ),
        )

    def kinds(self) -> tuple[KindSummary, ...]:
        """One row per kind reachable from the root, in first-visit order.

        ``O(|description|)``: counts flow from parents to children along the
        definition DAG, so a kind reached through many paths is still visited
        once, and every row is a per-definition summary (declared interfaces
        as runs, never enumerated).  ``closed`` comes from
        :func:`transient_ports`, the same top-down pass over the DAG.
        """

        root = self._frame.definition
        parents_first = _reachable(root)[::-1]
        transient = transient_ports(root)
        copies: dict[str, int] = {root.digest: 1}
        min_depth: dict[str, int] = {root.digest: 0}
        max_depth: dict[str, int] = {root.digest: 0}
        for definition in parents_first:
            count = copies[definition.digest]
            for step in definition.steps:
                if isinstance(step, CallStep):
                    child = step.child.digest
                    copies[child] = copies.get(child, 0) + count * step.count
                    depth = min_depth[definition.digest] + 1
                    min_depth[child] = min(min_depth.get(child, depth), depth)
                    depth = max_depth[definition.digest] + 1
                    max_depth[child] = max(max_depth.get(child, depth), depth)
        verification_kinds: dict[str, tuple[tuple[str, int], ...]] = {}
        for definition in reversed(parents_first):
            verification_kinds[definition.digest] = (
                ((definition.digest, 1),)
                if definition.role == VERIFICATION
                else _multiset(definition, verification_kinds)
            )
        return tuple(
            KindSummary(
                kind=definition.digest,
                role=definition.role,
                copies=copies[definition.digest],
                size=definition.size,
                replay_cost=definition.replay_cost,
                proof_cost=definition.proof_cost,
                input_count=definition.input_count,
                out_count=definition.out_count,
                out_bits=definition.out_bits,
                source_inputs=definition.input_total,
                source_weights=definition.weight_total,
                min_depth=min_depth[definition.digest],
                max_depth=max_depth[definition.digest],
                children=_multiset(definition, None),
                verification_units=definition.verification_units,
                verification_kinds=verification_kinds[definition.digest],
                closed=not transient[definition.digest],
            )
            for definition in _preorder(root)
        )

    def kind_table(self) -> KindTable:
        """:meth:`kinds` with the totals the analysis folds read, under the index digest."""

        return KindTable(
            rows=self.kinds(),
            root=self.root.kind,
            n=self.n,
            input_count=self.input_count,
            weight_count=self.weight_count,
            replay_unit_count=self.replay_units.count,
            digest=self.digest,
        )


class _Sources:
    """Lazy addresses of the ``source`` gates (``In`` or ``W``) in address order.

    The rank of a source gate is the number of source gates of its kind laid
    out before it: ``input_before``/``weight_before`` on the frames and the
    per-step prefix sums inside a definition give it by descent, and unrank
    descends the same sums (bisect within a step list, divide within a
    ``repeat``), so everything is ``O(depth)``.
    """

    __slots__ = ("_frame", "count", "identity_digest", "source")

    def __init__(self, index: Index, source: str) -> None:
        self._frame = index._frame
        self.source = source
        self.count = self._frame.definition.source_total(source)
        self.identity_digest = identity_digest(
            "veritor/indexed-domain/sources/v1",
            {"index": index.digest, "source": source},
        )

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def _rank(self, item: int) -> int | None:
        if type(item) is not int or item not in self._frame.interval:
            return None
        return self._frame.source_rank(self.source, item)

    def contains(self, item: int) -> bool:
        return self._rank(item) is not None

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        rank = self._rank(item)
        if rank is None:
            raise KeyError(item)
        return rank

    def unrank(self, rank: int) -> int:
        if type(rank) is not int:
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.count:
            raise IndexError(f"rank {rank} is outside domain of size {self.count}")
        return self._frame.source_address(self.source, rank)

    def __iter__(self) -> Iterator[int]:
        for rank in range(self.count):
            yield self.unrank(rank)

    def __len__(self) -> int:
        return self.count


class _Boundary:
    """Lazy ``In ∪ ⋃_r Out(R_r)`` with ``O(depth)`` rank and unrank.

    The input gates come first, by rank, then the units' declared outputs
    unit by unit, each unit's in the run order of its kind (address order
    unless runs interleave); ``out_before`` on the frames gives the prefix
    sums and a unit's runs the rank within it.  The two parts are disjoint:
    ``Out`` never contains a pinned gate.
    """

    __slots__ = ("_index", "_inputs", "count", "identity_digest")

    def __init__(self, index: Index) -> None:
        self._index = index
        self._inputs = _Sources(index, INPUT_SOURCE)
        self.count = self._inputs.count + index._frame.definition.out_total
        self.identity_digest = identity_digest(
            "veritor/indexed-domain/boundary/v4", {"index": index.digest}
        )

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def _locate(self, address: int) -> tuple[Frame, int] | None:
        frame = _unit_frame_at(self._index._frame, address, REPLAY)
        if frame is None:
            return None
        position = frame.definition.out_rank(address - frame.base)
        if position is None:
            return None
        return frame, position

    def contains(self, item: int) -> bool:
        if type(item) is not int or not 0 <= item < self._index.n:
            return False
        return self._inputs.contains(item) or self._locate(item) is not None

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        if type(item) is not int or not 0 <= item < self._index.n:
            raise KeyError(item)
        input_rank = self._inputs._rank(item)
        if input_rank is not None:
            return input_rank
        located = self._locate(item)
        if located is None:
            raise KeyError(item)
        frame, position = located
        return self._inputs.count + frame.out_before + position

    def unrank(self, rank: int) -> int:
        if type(rank) is not int:
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.count:
            raise IndexError(f"rank {rank} is outside domain of size {self.count}")
        inputs = self._inputs.count
        if rank < inputs:
            return self._inputs.unrank(rank)
        frame, offset = _frame_by_rank(
            self._index._frame,
            rank - inputs,
            REPLAY,
            lambda d: d.step_out,
            lambda d: d.out_total,
        )
        return frame.base + frame.definition.out_offset(offset)

    def __iter__(self) -> Iterator[int]:
        for rank in range(self.count):
            yield self.unrank(rank)

    def __len__(self) -> int:
        return self.count


# -- validity of the role marks ------------------------------------------------


def _reachable(root: Definition) -> list[Definition]:
    """Definitions reachable from ``root``, children before parents."""

    order: list[Definition] = []
    seen: set[str] = set()

    def visit(definition: Definition) -> None:
        if definition.digest in seen:
            return
        seen.add(definition.digest)
        for step in definition.steps:
            if isinstance(step, CallStep):
                visit(step.child)
        order.append(definition)

    visit(root)
    return order


def _preorder(root: Definition) -> list[Definition]:
    """Definitions reachable from ``root`` in first-visit (step) order."""

    order: list[Definition] = []
    seen: set[str] = set()
    pending = [root]
    while pending:
        definition = pending.pop()
        if definition.digest in seen:
            continue
        seen.add(definition.digest)
        order.append(definition)
        for step in reversed(definition.steps):
            if isinstance(step, CallStep):
                pending.append(step.child)
    return order


def _multiset(
    definition: Definition,
    inner: dict[str, tuple[tuple[str, int], ...]] | None,
) -> tuple[tuple[str, int], ...]:
    """Kinds called by one copy of ``definition`` as ``(kind, count)`` pairs.

    With ``inner`` the children are expanded through the given per-kind
    multisets (weighted by the call count); without it the direct children
    are counted.  Pairs appear in first-visit order.
    """

    counts: dict[str, int] = {}
    for step in definition.steps:
        if not isinstance(step, CallStep):
            continue
        expansion = ((step.child.digest, 1),) if inner is None else inner[step.child.digest]
        for kind, count in expansion:
            counts[kind] = counts.get(kind, 0) + step.count * count
    return tuple(counts.items())


# -- retained ports ---------------------------------------------------------------
#
# A value is *retained* when an honest prover still has it after the run: the
# circuit's inputs and the weights, i.e. the source gates.  A port of a
# definition is retained when every call site, in every calling definition and
# every copy of a ``repeat``, feeds it a source gate (a source-gate step of the
# caller, or the pinned output of a call the caller makes) or a port of the
# caller that is itself retained; a computed gate of the caller, an unpinned
# output of another call, or a transient port of the caller make it transient.
# Circuit outputs are not counted as retained.  The pass is top-down over the
# definition DAG (callers before callees), works per step with the run
# arithmetic of the description, and never enumerates copies or ports: each
# argument range is a grid over the copies of the step, cut into at most
# ``min(copies, count)`` progressions, and each progression is intersected
# with the caller's transient port runs (input space) or classified through the
# steps it crosses (local space).  Two approximations, both conservative (more
# ports transient, never fewer): a stretch of a local argument that lies inside
# one call of the caller is transient as a whole when any of its slots is, and
# the ports fed by a tiled ``repeat`` argument are taken by residue class when
# the transient stretch wraps around the copies.


def _argument_grid(item: Range, copies: int) -> Iterator[tuple[int, int, int, int, int]]:
    """The coordinates of argument ``item`` over ``copies`` copies as progressions.

    Yields ``(start, count, stride, k0, kstep)``: element ``e`` of the
    progression is coordinate ``start + e * stride`` and feeds port
    ``k0 + e * kstep`` of the argument, modulo ``item.count`` (the copies of
    a tiled ``repeat`` argument continue one progression, so ``e`` runs over
    copy after copy).  At most ``min(copies, item.count)`` progressions.
    """

    rows = copies if item.jstride else 1
    columns, stride, jstride = item.count, item.stride, item.jstride
    if rows == 1:
        yield item.start, columns, stride, 0, 1
    elif columns == 1:
        yield item.start, rows, jstride, 0, 0
    elif jstride == columns * stride:
        yield item.start, rows * columns, stride, 0, 1
    elif columns <= rows:
        for c in range(columns):
            yield item.start + c * stride, rows, jstride, c, 0
    else:
        for r in range(rows):
            yield item.start + r * jstride, columns, stride, 0, 1


def _port_runs(base: int, count: int, k0: int, kstep: int, first: int, n: int, step: int) -> Iterator[Run]:
    """The ports ``base + ((k0 + e * kstep) mod count)`` for ``e = first + t * step``, ``t < n``, as runs."""

    if n <= 0:
        return
    if kstep == 0 or count == 1:
        yield Run(base + k0, 1, 0, 0)
        return
    start = first % count
    if n == 1 or step % count == 0:
        yield Run(base + start, 1, 0, 0)
    elif start + (n - 1) * step < count:
        yield Run(base + start, n, step, 0)
    else:
        # the progression wraps around the copies: the residue class it fills
        # (exactly when it wraps often enough, conservatively otherwise)
        modulus = gcd(step, count)
        yield Run(base + first % modulus, count // modulus, modulus, 0)


def _retained_slots(
    definition: Definition,
    transient: tuple[Run, ...],
    step: CallStep,
    index: int,
    slot: int,
    count: int,
    stride: int,
) -> bool:
    """Whether slots ``slot + k * stride`` (``k < count``) of call step ``index`` all hold retained values."""

    kinds = {kind for kind, _ in step.child.resolved_outputs}
    if kinds == {PieceKind.PINNED}:
        return True
    if kinds == {PieceKind.GATE}:
        return False
    base = definition.step_address[index]
    for kind, run in _call_pieces(definition, step, base, slot, count, stride):
        if kind is PieceKind.GATE:
            return False
        if kind is PieceKind.PORT and any(
            progression_meet(run.start, run.count, run.stride, item) is not None for item in transient
        ):
            return False
    return True


def _transient_stretches(
    definition: Definition, transient: tuple[Run, ...], start: int, count: int, stride: int
) -> Iterator[tuple[int, int]]:
    """``(first, taken)`` index stretches of local slots ``start + e * stride`` holding transient values.

    The progression is cut at the caller's step boundaries; a gate step is
    retained iff it is a source gate, and a stretch inside a call is retained
    iff every slot of it resolves to a source gate or to a retained port.
    """

    for index, first, taken in _split(definition.step_slot, start, count, stride):
        step = definition.steps[index]
        if isinstance(step, GateStep):
            if not step.pinned:
                yield first, taken
            continue
        slot = start + first * stride - definition.step_slot[index]
        if not _retained_slots(definition, transient, step, index, slot, taken, stride):
            yield first, taken


def _normalized(runs: list[Run]) -> tuple[Run, ...]:
    return _sorted_runs(set(runs))


def transient_ports(root: Definition) -> dict[str, tuple[Run, ...]]:
    """Per reachable definition, runs of the port ordinals fed a transient value somewhere.

    A definition is *closed* iff its tuple is empty.  Top-down over the DAG:
    in reverse post-order every caller precedes its callees, so a
    definition's transient ports are final when it is reached and can be
    propagated through its own call steps.  ``O(|description|)`` up to the
    number of progressions the argument grids cut into (see the module
    comment above); copies and ports are never enumerated.
    """

    pending: dict[str, list[Run]] = {root.digest: []}
    result: dict[str, tuple[Run, ...]] = {}
    for definition in _reachable(root)[::-1]:
        own = result[definition.digest] = _normalized(pending.pop(definition.digest, []))
        for step in definition.steps:
            if not isinstance(step, CallStep):
                continue
            found = pending.setdefault(step.child.digest, [])
            for index, item in enumerate(step.args):
                base = step.arg_starts[index]
                for start, count, stride, k0, kstep in _argument_grid(item, step.count):
                    if item.space == INPUT:
                        for run in own:
                            meet = progression_meet(start, count, stride, run)
                            if meet is not None:
                                found.extend(_port_runs(base, item.count, k0, kstep, *meet))
                    else:
                        for first, taken in _transient_stretches(definition, own, start, count, stride):
                            found.extend(_port_runs(base, item.count, k0, kstep, first, taken, 1))
    return result


def _short(definition: Definition) -> str:
    return f"definition {definition.digest[:12]}"


def _offending_step(
    definition: Definition, tiled: dict[str, bool], unit: str
) -> InvalidArtifact:
    """Explain why ``definition`` (not tiled) is not covered by ``unit`` marks."""

    while True:
        for index, step in enumerate(definition.steps):
            if isinstance(step, GateStep):
                return InvalidArtifact(
                    f"gate step {index} of {_short(definition)} is not inside "
                    f"a {unit} unit"
                )
            if not tiled[step.child.digest]:
                child = step.child
                if child.role is not None:
                    return InvalidArtifact(
                        f"step {index} of {_short(definition)} calls {child.role}-marked "
                        f"{_short(child)} outside any {unit} unit"
                    )
                definition = child
                break
        else:  # pragma: no cover - a non-tiled definition has an offending step
            raise AssertionError("no offending step found")


def validate_marks(root: Definition, limits: CompilationLimits) -> None:
    """Check the role marks once per definition, never per copy.

    1. Replay marks tile the gates: above the replay cut every step is a call
       or repeat into a covered definition.
    2. Verification marks tile every replay unit the same way.
    3. No mark is nested inside a mark of the same role.
    4. Every verification unit's proof cost is within the completeness cap.
    """

    replay_tiled: dict[str, bool] = {}
    verification_tiled: dict[str, bool] = {}
    replay_inside: dict[str, bool] = {}
    verification_inside: dict[str, bool] = {}
    for definition in _reachable(root):
        calls = [step for step in definition.steps if isinstance(step, CallStep)]
        has_gate = len(calls) != len(definition.steps)
        children = [step.child for step in calls]
        r_inside = any(c.role == REPLAY or replay_inside[c.digest] for c in children)
        v_inside = any(
            c.role == VERIFICATION or verification_inside[c.digest] for c in children
        )
        role = definition.role
        if role is not None and definition.size == 0:
            raise InvalidArtifact(f"{_short(definition)} is marked {role} but has no gates")
        if role == REPLAY:
            if r_inside:
                raise InvalidArtifact(
                    f"{_short(definition)} is marked replay and contains a replay mark"
                )
            if has_gate or not all(verification_tiled[c.digest] for c in children):
                raise _offending_step(definition, verification_tiled, "verification")
        elif role == VERIFICATION:
            if r_inside:
                raise InvalidArtifact(
                    f"{_short(definition)} is marked verification and contains "
                    "a replay mark"
                )
            if v_inside:
                raise InvalidArtifact(
                    f"{_short(definition)} is marked verification and contains "
                    "a verification mark"
                )
            if definition.proof_cost > limits.max_verification_unit_proof_cost:
                raise InvalidArtifact(
                    f"{_short(definition)} is a verification unit of proof cost "
                    f"{definition.proof_cost}; the limit is "
                    f"{limits.max_verification_unit_proof_cost}"
                )
        digest = definition.digest
        replay_inside[digest] = r_inside
        verification_inside[digest] = v_inside
        covered = role is None and not has_gate
        replay_tiled[digest] = role == REPLAY or (
            covered and all(replay_tiled[c.digest] for c in children)
        )
        verification_tiled[digest] = role == VERIFICATION or (
            covered and all(verification_tiled[c.digest] for c in children)
        )
    if root.role == VERIFICATION:
        raise InvalidArtifact("the root is marked verification but is not inside a replay unit")
    if not replay_tiled[root.digest]:
        raise _offending_step(root, replay_tiled, "replay")
