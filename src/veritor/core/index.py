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
model needs to price the recomputation behind a sampled replay unit (RU).

It records, too, how much of the circuit's output a copy of each kind can
*reach* (``reach_bits``): the width of the circuit outputs that lie forward
of any gate of the copy, along argument reads.  Those outputs are a
downstream cut for the copy, so ``Bound`` may charge an unsampled RU or
verification unit (VU) ``min(out_bits, reach_bits)`` in place of its
interface.  The reach is computed at *step granularity*: within a
definition, a step (a gate, a call or a whole ``repeat``) depends on every
earlier step whose slots any of its arguments read, and everything
reachable from a step is charged to it; the result is an upper bound that
never enumerates copies (see :func:`output_reach`).  Tracking dataflow
through the individual outputs of a step -- per slot of a batched decode
step, say -- would be a further refinement.

The third downstream cut a copy has is the declared interface of any copy
enclosing it (``ancestor_bits``): a value computed inside a copy of ``D``
can be read outside that copy, or be a circuit output, only through one of
``D``'s declared outputs, so ``Out`` of every proper ancestor is a
downstream cut for the copy.  ``ancestor_bits`` is the narrowest of them,
maximised over the copies of the kind, computed top-down over the
definition DAG at step granularity (see :func:`ancestor_interfaces`); the
bottleneck ``Bound`` charges a RU or VU is
``min(out_bits, reach_bits, ancestor_bits)`` (:attr:`KindSummary.cut_bits`).

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
    LOCAL,
    REPLAY,
    VERIFICATION,
    CallStep,
    Check,
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
    of ``Out`` in bits less the root's check outputs (members of ``Out`` the
    verifier requires to equal a constant of the description, so worth ``0``
    bits here and in the reach).  ``source_inputs`` and ``source_weights`` count the
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

    ``reach_bits`` is an upper bound, over the copies of the kind, on the
    width in bits of the circuit outputs reachable from the copy's gates
    along argument reads (:func:`output_reach`).  The reachable outputs are
    a downstream cut for the copy, so a node may be charged
    ``min(out_bits, reach_bits)`` by ``Bound``; the reach may be far below
    the interface (a decode step of one request reaches only that request's
    remaining tokens) and is never above the root's ``out_bits``, which is
    the root's own ``reach_bits``.  It is computed at step granularity and
    is ``0`` for a kind whose copies never reach an output.

    ``ancestor_bits`` is an upper bound, over the copies of the kind, on the
    narrowest declared interface (``out_bits``) among the copy's *proper*
    ancestors in the hierarchy (:func:`ancestor_interfaces`).  Every value
    a copy computes leaves an enclosing copy through that copy's declared
    outputs, so each ancestor's ``Out`` is a downstream cut for the copy
    too.  The root has no ancestor and carries its own ``out_bits``, which
    is also every kind's ceiling.  The three cuts together are the copy's
    *bottleneck*, :attr:`cut_bits`, what ``Bound`` charges a RU or VU.
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
    reach_bits: int
    ancestor_bits: int
    source_inputs: int
    source_weights: int
    min_depth: int
    max_depth: int
    children: tuple[tuple[str, int], ...]
    verification_units: int
    verification_kinds: tuple[tuple[str, int], ...]
    closed: bool

    @property
    def cut_bits(self) -> int:
        """``kappa`` of a copy of the kind: the narrowest of its three downstream cuts.

        The interface ``Out`` (``out_bits``), the circuit outputs the copy
        can reach (``reach_bits``) and the narrowest interface of a copy
        enclosing it (``ancestor_bits``) are all downstream cuts for every
        gate of the copy, so ``Bound`` may charge any of them; it charges the
        smallest.
        """

        return min(self.out_bits, self.reach_bits, self.ancestor_bits)


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
        kinds = {row.kind: row for row in self.rows}
        if len(kinds) != len(self.rows):
            raise ValueError("kind table rows must have distinct kinds")
        if self.root not in kinds:
            raise ValueError("kind table root must be one of its rows")
        total = kinds[self.root].out_bits
        if kinds[self.root].reach_bits != total:
            raise ValueError("the root reaches exactly its own outputs")
        if kinds[self.root].ancestor_bits != total:
            raise ValueError("the root has no ancestor and carries its own interface")
        for row in self.rows:
            # a child's ancestor bound must cover every call site: the caller's own
            # bound narrowed by the caller's interface (larger is sound, smaller is not)
            site = min(row.ancestor_bits, row.out_bits)
            for child, _ in row.children:
                if child not in kinds:
                    raise ValueError(f"kind {row.kind} calls unknown kind {child}")
                if kinds[child].ancestor_bits < site:
                    raise ValueError(
                        f"kind {child} claims ancestors of {kinds[child].ancestor_bits} bits "
                        f"but is called by {row.kind} through {site}"
                    )
            if row.input_count == 0 and not row.closed:
                raise ValueError(f"kind {row.kind} has no ports and must be closed")
            if not 0 <= row.reach_bits <= total:
                raise ValueError(f"kind {row.kind} reaches {row.reach_bits} bits of a {total}-bit output")
            if not 0 <= row.ancestor_bits <= total:
                raise ValueError(
                    f"kind {row.kind} claims ancestors of {row.ancestor_bits} bits in a {total}-bit output"
                )


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

    @property
    def checks(self) -> tuple[Check, ...]:
        """The root's check outputs: progressions of output ordinals with their constants."""

        return self._frame.definition.checks

    def check_values(self) -> Iterator[tuple[int, int]]:
        """``(output ordinal, constant)`` for every check output, check by check."""

        for check in self.checks:
            for ordinal in check.ordinals():
                yield ordinal, check.value

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
        :func:`transient_ports`, ``reach_bits`` from :func:`output_reach`
        and ``ancestor_bits`` from :func:`ancestor_interfaces`, three more
        top-down passes over the DAG.
        """

        root = self._frame.definition
        parents_first = _reachable(root)[::-1]
        transient = transient_ports(root)
        reach = output_reach(root)
        ancestor = ancestor_interfaces(root)
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
                reach_bits=reach[definition.digest],
                ancestor_bits=ancestor[definition.digest],
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


# -- output reach ------------------------------------------------------------------
#
# For a set of gates ``A``, the circuit outputs reachable from ``A`` along
# argument reads are a downstream cut for ``A``: every path from ``A`` to an
# output ends at one of them.  Their width therefore bounds the capacity of
# ``A`` (paper, section 5.4), beside the two cuts the fold already knows, the
# interface ``Out(A)`` and the whole output.  Any superset of the reachable
# outputs is a cut as well, so a structural over-approximation of the reach is
# sound.
#
# The reach is computed at *step granularity*, per definition, top-down over the
# DAG.  Inside a definition ``D`` a step ``k`` depends on an earlier step ``j``
# when any argument run of ``k`` (over every copy of a ``repeat``) meets the
# slots of ``j``; a call or repeat step is one node, so any input to it may reach
# any of its outputs.  ``Down(j)`` is the transitive closure.  The grammar lets a
# step read only the slots of *earlier* steps, so the copies of one ``repeat``
# never read each other: a chain of decode steps is a sequence of call steps,
# which the closure follows, while the copies of a ``repeat`` are independent
# and one copy's reach is computed from the outputs attributable to one copy.
#
# The closure is kept as *intervals of step indices*, never as a set of steps.
# ``Down(j) ⊆ [j, S)`` for a definition of ``S`` steps, and the two structures
# that matter are (unions of) a few intervals: along a chain (step ``k`` reads
# step ``k - 1``, or every earlier step as a KV cache does) ``Down(j) = [j, S)``;
# for ``N`` independent siblings reading one broadcast step ``Down(sibling) =
# {sibling}`` and ``Down(broadcast) = [broadcast, N + 1)``.  Reads are recorded
# on the *reader's* side as ranges of steps: an argument run of step ``k``
# covers the steps ``[a, b)`` between the ones holding its first and last slot
# (two bisections on ``step_slot``), exactly when the run is dense and as a
# superset when it is strided over more than ``_EXACT_READ_STEPS`` steps (a
# narrower strided run is enumerated through ``_split``).  The steps are then
# swept last to first: a reader ``k`` is *active* while the sweep is inside
# one of its ranges, ``Down(k)`` is already final then (``b <= k``), and a
# segment tree over the step positions counts, per position, the active
# readers whose ``Down`` contains it, so the union ``U_j`` of their closures is
# the set of positions with a positive count.  ``reach(j)`` is ``share(j)``
# plus the output bits over ``U_j`` (read off the root: the bits at count zero
# are the uncovered ones), and ``Down(j) = {j} ∪ U_j`` is extracted as maximal
# intervals by descent, at most ``_MAX_DOWN_INTERVALS`` of them, beyond which
# the hull ``[j, max Down(j) + 1)`` stands in.  Both approximations only add
# edges and only enlarge a closure, which keeps every reach a downstream cut;
# both are exact on every definition with at most ``_EXACT_READ_STEPS`` steps.
# The sweep costs ``O((S + R) · I · log S)`` for ``R`` recorded ranges and
# ``I <= _MAX_DOWN_INTERVALS`` intervals per closure, ``O((S + R) log S)`` on
# the chain and sibling shapes, against ``Θ(S³ / w)`` and ``Θ(N² / w)`` for the
# bitmask closure this replaces.
#
# What a step reaches is measured in bits of the *caller's* declared outputs.
# At the root those are the circuit outputs, cut at step boundaries into
# segments whose widths (through ``_call_pieces``) sum to ``out_bits``; a copy
# of a ``repeat`` is charged at most the widest share one copy can hold of each
# segment.  Below the root all of a definition's outputs at a call site share
# the reach ``R_D`` the caller assigned to the copy (every path out of the copy
# leaves through a declared output, so ``R_D`` bounds what any of them reaches),
# and a step reaches ``R_D`` or nothing.  A child called at step ``j`` of ``D``
# receives ``reach_D(j)``; its own ``reach_bits`` is the maximum over its call
# sites.  Everything inside a copy inherits at most the copy's reach, but is
# still computed through the copy's own steps: a child may reach only some of
# the copy's outputs, or none.


def _segment_bits(definition: Definition, index: int, first: int, taken: int, stride: int) -> tuple[int, int]:
    """Bits of the declared-output slots ``first + t * stride`` (``t < taken``) inside step ``index``.

    Returns ``(width, share)``: the width of the gates the segment resolves
    to (pinned gates and pass-through ports carry nothing) and an upper
    bound on the part of it that lies in any one copy of the step.  The
    copies of a ``repeat`` are independent, so one copy owns at most
    ``ceil(outputs / stride)`` elements of a strided segment (all of them
    when the stride is ``0``), each no wider than the widest gate the
    segment resolves to; for a gate step or a single call the share is the
    width.
    """

    step = definition.steps[index]
    if isinstance(step, GateStep):
        width = 0 if step.pinned else taken * step.gate.width
        return width, width
    slot = first - definition.step_slot[index]
    base = definition.step_address[index]
    runs = [
        run for kind, run in _call_pieces(definition, step, base, slot, taken, stride) if kind is PieceKind.GATE
    ]
    width = sum(run.count * run.width for run in runs)
    if step.count == 1 or not runs:
        return width, width
    outputs = step.child.output_count
    per_copy = taken if stride == 0 else min(taken, -(-outputs // stride))
    return width, min(width, per_copy * max(run.width for run in runs))


_EXACT_READ_STEPS = 64
"""A strided argument run spanning at most this many steps names its steps exactly; wider, its hull."""

_MAX_DOWN_INTERVALS = 64
"""Maximal intervals kept for a step's closure ``Down``; beyond it the hull ``[j, max + 1)`` stands in."""

type _Intervals = list[tuple[int, int]]


class _Coverage:
    """Counts over the step positions ``[0, S)`` under range addition of ``±1``.

    A segment tree with lazy addition (a node's values include its own pending
    addition, not its ancestors').  Per node: the minimum and maximum count in
    its subtree and the ``out`` weight of the positions at the minimum, so the
    weight of the *covered* positions (positive count) is the total weight
    minus the weight at count zero, read off the root, and the covered
    positions come out as maximal intervals by descending into the nodes
    whose maximum is positive and stopping at those whose minimum is.  The
    positions ``[S, size)`` padding the tree to a power of two mirror position
    ``S - 1``: a suffix ``[low, S)`` is added as ``[low, size)``, so they never
    hold the minimum on their own and intervals are clipped to ``S``.
    """

    __slots__ = ("count", "lz", "mn", "mx", "size", "sm", "total")

    def __init__(self, out: list[int]) -> None:
        count = len(out)
        size = 1
        while size < count:
            size <<= 1
        self.count = count
        self.size = size
        self.mn = [0] * (2 * size)
        self.mx = [0] * (2 * size)
        self.lz = [0] * (2 * size)
        sm = [0] * (2 * size)
        sm[size : size + count] = out
        for node in range(size - 1, 0, -1):
            sm[node] = sm[2 * node] + sm[2 * node + 1]
        self.sm = sm
        self.total = sm[1]

    def covered_out(self) -> int:
        """The ``out`` weight of the positions with a positive count."""

        return self.total - self.sm[1] if self.mn[1] == 0 else self.total

    def add(self, low: int, high: int, delta: int, dirty: list[int]) -> None:
        """Add ``delta`` to the counts of positions ``[low, high)`` on the ``O(log S)`` nodes covering it.

        The ancestors of those nodes are stale until :meth:`settle` is
        called with ``dirty``, to which the parents of the boundary leaves
        are appended; several additions may share one settlement.  Every
        stale node is an ancestor of leaf ``low`` or of leaf ``high - 1``,
        and for a suffix (which reaches the last leaf of the tree) the
        latter are ancestors of leaf ``low`` too.
        """

        mn, mx, lz = self.mn, self.mx, self.lz
        size = self.size
        if high == self.count:
            high = size
        left = low + size
        right = high + size
        dirty.append(left >> 1)
        if high != size:
            dirty.append((right - 1) >> 1)
        while left < right:
            if left & 1:
                mn[left] += delta
                mx[left] += delta
                lz[left] += delta
                left += 1
            if right & 1:
                right -= 1
                mn[right] += delta
                mx[right] += delta
                lz[right] += delta
            left >>= 1
            right >>= 1

    def settle(self, dirty: list[int]) -> None:
        """Recompute the ancestors of the boundary leaves recorded in ``dirty``, bottom-up, once each.

        ``O(log S)`` per addition and less for many: the paths merge on the
        way up, so ``N`` point additions over adjacent positions settle in
        ``O(N)``.
        """

        mn, mx, lz, sm = self.mn, self.mx, self.lz, self.sm
        nodes = sorted(set(dirty)) if len(dirty) > 2 else dirty  # all at the level above the leaves
        while nodes[0]:
            parents: list[int] = []
            last = 0
            for node in nodes:
                if node == last:
                    continue
                last = node
                a = node << 1
                b = a | 1
                ma = mn[a]
                mb = mn[b]
                if ma < mb:
                    mn[node] = ma + lz[node]
                    sm[node] = sm[a]
                elif mb < ma:
                    mn[node] = mb + lz[node]
                    sm[node] = sm[b]
                else:
                    mn[node] = ma + lz[node]
                    sm[node] = sm[a] + sm[b]
                xa = mx[a]
                xb = mx[b]
                mx[node] = max(xb, xa) + lz[node]
                parents.append(node >> 1)
            nodes = parents

    def intervals(self, first: int, cap: int) -> _Intervals:
        """``{first}`` and the covered positions as maximal intervals, left to right.

        ``first`` lies left of every covered position.  Past ``cap``
        intervals the hull ``[first, last covered + 1)`` is returned instead.
        """

        mn, mx, lz = self.mn, self.mx, self.lz
        size, count = self.size, self.count
        found: _Intervals = [(first, first + 1)]

        def emit(low: int, high: int) -> bool:
            """Record the fully covered ``[low, high)``, clipped; ``True`` once the cap is exceeded."""

            if low >= count:
                return False
            high = min(high, count)
            start, end = found[-1]
            if end == low:
                found[-1] = (start, high)
            elif len(found) < cap:
                found.append((low, high))
            else:
                return True
            return False

        if mx[1] <= 0:
            return found
        # ``node`` holds covered and uncovered positions both (the root: ``first`` is uncovered);
        # ``acc`` is the pending addition of its proper ancestors.  The right child waits on the
        # stack while the left is done, as itself when partial or as a marker when fully covered.
        stack: list[tuple[int, int, int, int]] = []
        node, low, high, acc = 1, 0, size, 0
        while True:
            acc += lz[node]
            mid = (low + high) >> 1
            a = node << 1
            b = a | 1
            if mx[b] + acc > 0:
                stack.append((-1, mid, high, 0) if mn[b] + acc > 0 else (b, mid, high, acc))
            if mx[a] + acc > 0:
                if mn[a] + acc <= 0:
                    node, high = a, mid
                    continue
                if emit(low, mid):
                    return [(first, self._last_covered() + 1)]
            while stack:
                node, low, high, acc = stack.pop()
                if node >= 0:
                    break
                if emit(low, high):
                    return [(first, self._last_covered() + 1)]
            else:
                return found

    def _last_covered(self) -> int:
        """The rightmost position with a positive count (some position has one)."""

        mx, lz = self.mx, self.lz
        size = self.size
        node, acc = 1, 0
        while node < size:
            acc += lz[node]
            node <<= 1
            if mx[node | 1] + acc > 0:
                node |= 1
        return min(node - size, self.count - 1)


def _read_steps(step_slot: tuple[int, ...], start: int, count: int, stride: int, ranges: _Intervals) -> None:
    """Append, as step ranges ``[a, b)``, the steps whose slots ``start + k * stride`` (``k < count``) visit.

    Exact for a dense run (every step between its first and last slot is
    visited) and for a strided one spanning at most ``_EXACT_READ_STEPS``
    steps; a wider strided run, which may skip a step narrower than its
    stride, is recorded as the hull of the steps it spans.
    """

    a = bisect_right(step_slot, start) - 1
    if count == 1 or stride == 0:
        ranges.append((a, a + 1))
        return
    b = bisect_right(step_slot, start + (count - 1) * stride)
    if stride == 1 or b - a <= 2 or b - a > _EXACT_READ_STEPS:
        ranges.append((a, b))
        return
    low = high = -1
    for index, _, _ in _split(step_slot, start, count, stride):
        if index != high:
            if high >= 0:
                ranges.append((low, high))
            low = index
        high = index + 1
    ranges.append((low, high))


def _disjoint(ranges: _Intervals) -> _Intervals:
    """The union of ``ranges`` as sorted, pairwise disjoint, non-adjacent intervals."""

    if len(ranges) == 1:
        return ranges
    ranges.sort()
    merged = [ranges[0]]
    for low, high in ranges[1:]:
        start, end = merged[-1]
        if low <= end:
            if high > end:
                merged[-1] = (start, high)
        else:
            merged.append((low, high))
    return merged


def _step_reach(definition: Definition, total: int, exact: bool) -> list[int]:
    """Per step of ``definition``, the bits of its declared outputs reachable from one copy of the step.

    ``total`` is the reach of a copy of ``definition`` itself.  With ``exact``
    (the root) the declared outputs are the circuit outputs and their
    segments are weighed by :func:`_segment_bits`; otherwise every output
    segment weighs ``total`` and a step reaches ``total`` or nothing.  Steps
    are swept last to first with the closure ``Down`` as intervals of steps
    (see the comment above :func:`_segment_bits`): ``Down(j)`` is ``j`` with
    the union of the ``Down`` of the steps reading it, and a step reaches its
    own share of the outputs plus the outputs of the rest of its closure,
    capped at ``total``.
    """

    steps = definition.steps
    count = len(steps)
    step_slot = definition.step_slot
    # reader k -> its ranges of read steps, as events of the sweep: k becomes active
    # (its Down is added to the counts) at the last step of a range and inactive
    # (subtracted) below the first; a range from step 0 is never left
    on_at: list[list[int] | None] = [None] * count
    off_at: list[list[int] | None] = [None] * count
    uses = [0] * count  # events left for k: its Down is dropped after the last one
    reads = [False] * count
    for k, step in enumerate(steps):
        copies = step.count if isinstance(step, CallStep) else 1
        ranges: _Intervals = []
        for item in step.args:
            if item.space != LOCAL:
                continue
            for start, run, stride, _, _ in _argument_grid(item, copies):
                _read_steps(step_slot, start, run, stride, ranges)
        if not ranges:
            continue
        reads[k] = True
        for low, high in _disjoint(ranges):
            arriving = on_at[high - 1]
            if arriving is None:
                arriving = on_at[high - 1] = []
            arriving.append(k)
            uses[k] += 1
            if low:
                leaving = off_at[low - 1]
                if leaving is None:
                    leaving = off_at[low - 1] = []
                leaving.append(k)
                uses[k] += 1
    out = [0] * count  # bits of declared outputs held by every copy of the step
    share = [0] * count  # ... and by any one copy of it
    for item in definition.outputs:
        if item.space != LOCAL:
            continue
        for index, first, taken in _split(step_slot, item.start, item.count, item.stride):
            if exact:
                width, single = _segment_bits(definition, index, item.element(first), taken, item.stride)
            else:
                width = single = total
            out[index] += width
            share[index] += single
    if exact:
        # A check output is fixed by the verifier: no bits of it are reachable,
        # so its gates come off the steps that hold them.  A share comes down
        # by the checked gates every copy of a ``repeat`` holds (a run with
        # one element per copy, the pitch of the copies) and is at most what
        # is left.
        for checked in definition.checked_runs:
            for index, _, taken in _split(
                definition.step_address, checked.start, checked.count, checked.stride
            ):
                out[index] -= taken * checked.width
                step = steps[index]
                if (
                    isinstance(step, CallStep)
                    and step.count > 1
                    and taken == step.count
                    and checked.stride % step.child.size == 0
                ):
                    share[index] -= checked.width
                share[index] = min(share[index], out[index])
    cover = _Coverage(out)
    add = cover.add
    down: list[_Intervals | None] = [None] * count
    reach = [0] * count
    dirty: list[int] = []
    for j in range(count - 1, -1, -1):
        for events, delta in ((off_at[j], -1), (on_at[j], 1)):
            if events is None:
                continue
            for k in events:
                intervals = down[k]
                assert intervals is not None  # k > j read a slot of j: its closure is final
                for low, high in intervals:
                    add(low, high, delta, dirty)
                uses[k] -= 1
                if not uses[k]:
                    down[k] = None
        if dirty:
            cover.settle(dirty)
            dirty = []
        bits = share[j] + cover.covered_out()
        reach[j] = min(total, bits)
        if reads[j]:
            down[j] = cover.intervals(j, _MAX_DOWN_INTERVALS)
    return reach


def output_reach(root: Definition) -> dict[str, int]:
    """Per reachable definition, an upper bound on the circuit output bits a copy of it can reach.

    The root reaches its whole output, ``out_bits``.  Top-down over the DAG,
    in reverse post-order, so a definition's reach is final when its own
    call steps are processed: a child called at step ``j`` receives the reach
    of that step (:func:`_step_reach`) and keeps the maximum over its call
    sites.  ``O(|description| · log S)`` for definitions of at most ``S``
    steps, up to the number of progressions the argument grids cut into and
    the intervals (at most ``_MAX_DOWN_INTERVALS``) a step's closure is kept
    as: the closure is swept with a segment tree over the steps, never as
    per-step sets, so a chain of ``S`` dependent steps or ``S`` siblings
    reading one step cost ``O(S log S)``; copies are never enumerated.
    """

    reach: dict[str, int] = {root.digest: root.out_bits}
    for definition in _reachable(root)[::-1]:
        own = reach[definition.digest]
        if not any(isinstance(step, CallStep) for step in definition.steps):
            continue
        per_step = _step_reach(definition, own, definition is root)
        for index, step in enumerate(definition.steps):
            if isinstance(step, CallStep):
                child = step.child.digest
                reach[child] = max(reach.get(child, 0), per_step[index])
    return reach


# -- ancestor interfaces ------------------------------------------------------------
#
# A copy of ``D`` exposes its values only through its declared outputs: the
# grammar lets a step read the slots of earlier steps of the same definition,
# and a call's slots are the child's declared outputs, so a value computed
# inside the copy is read outside it, or is a circuit output, only through
# ``Out(D)`` (a pinned declared output is a source gate, never in error, and is
# not in ``Out``).  For a copy ``x``, then, ``Out`` of every proper ancestor of
# ``x`` is a downstream cut, and the narrowest of them bounds ``x``'s capacity
# beside its own interface and its reach.
#
# Copies of a kind are paths from the root in the definition DAG.  The minimum
# over the ancestors of a copy is the minimum of ``out_bits`` along its path
# (the root's own interface at the top, the whole output), and a kind's value
# is the maximum over its copies, which is the maximum over paths: since
# ``min(max_p f(p), c) = max_p min(f(p), c)``, the recursion ``anc(child at a
# step of D) = min(anc(D), out_bits(D))``, maximised over call sites, computes
# that maximum exactly without enumerating a copy.


def ancestor_interfaces(root: Definition) -> dict[str, int]:
    """Per reachable definition, the widest, over its copies, of the narrowest enclosing interface.

    The root, having no ancestor, carries its own ``out_bits``.  Top-down
    over the DAG in reverse post-order, so a definition's value is final
    when its call steps are processed: a child called from ``D`` receives
    ``min(anc(D), out_bits(D))`` and keeps the maximum over its call sites,
    which is exactly the maximum over its copies (see the comment above).
    ``O(|description|)``.
    """

    ancestor: dict[str, int] = {root.digest: root.out_bits}
    for definition in _reachable(root)[::-1]:
        site = min(ancestor[definition.digest], definition.out_bits)
        for step in definition.steps:
            if isinstance(step, CallStep):
                child = step.child.digest
                ancestor[child] = max(ancestor.get(child, 0), site)
    return ancestor


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
