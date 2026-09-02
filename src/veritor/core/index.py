"""The index ``I``: nested subcircuits over ``C``'s addresses.

Every copy of a definition is a node: an address interval, a kind (the
definition digest), a role mark, lazy children and a depth.  Two antichains
are designated by the role marks: every copy of a definition marked
``"replay"`` is a replay unit and every copy marked ``"verification"`` is a
verification unit.  Both are lazy sequences: ``count``, ``unit(k)`` and
``owner(address)`` all cost ``O(depth)`` through the prefix sums stored on the
definitions (bisect within a step list, divide within a ``repeat``).

The boundary ``inputs ∪ ⋃_r Out(R_r)`` and each interior ``R_r \\ boundary``
are lazy address sets built the same way; nothing about them is stored.

Canonical chunking of long step lists (so a cut can fall inside a definition)
is a later phase: today a unit is always a whole copy of a definition.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .description import REPLAY, VERIFICATION, CallStep, Definition, Frame, GateStep
from .errors import InvalidArtifact
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
    """The profiler's row for one kind: what every copy shares."""

    kind: str
    role: str | None
    copies: int
    size: int
    replay_cost: int
    proof_cost: int
    in_count: int
    out_count: int
    min_depth: int
    max_depth: int


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
        return self._frame.definition.input_count

    @property
    def n(self) -> int:
        return self._frame.base + self._frame.definition.size

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
        """``inputs ∪ ⋃_r Out(R_r)``: the addresses the boundary commitment covers.

        The circuit outputs are always inside this set: every output resolves
        through the declared interface of the replay unit that owns it.
        """

        return _Boundary(self)

    def interior(self, replay_unit: int) -> IntervalDifferenceDomain:
        """``R_r`` minus the boundary."""

        frame = self.replay_units.unit(replay_unit).frame
        return IntervalDifferenceDomain(
            frame.base,
            frame.base + frame.definition.size,
            (frame.base + offset for offset in frame.definition.local_outputs),
        )

    def kinds(self) -> tuple[KindSummary, ...]:
        """One row per kind reachable from the root, in first-visit order."""

        copies: dict[str, int] = {}
        depths: dict[str, list[int]] = {}
        definitions: dict[str, Definition] = {}
        pending: list[tuple[Definition, int, int]] = [(self._frame.definition, 1, 0)]
        order: list[str] = []
        while pending:
            definition, count, depth = pending.pop()
            digest = definition.digest
            if digest not in definitions:
                definitions[digest] = definition
                order.append(digest)
                depths[digest] = [depth, depth]
            copies[digest] = copies.get(digest, 0) + count
            depths[digest][0] = min(depths[digest][0], depth)
            depths[digest][1] = max(depths[digest][1], depth)
            for step in reversed(definition.steps):
                if isinstance(step, CallStep):
                    pending.append((step.child, count * step.count, depth + 1))
        return tuple(
            KindSummary(
                kind=digest,
                role=definitions[digest].role,
                copies=copies[digest],
                size=definitions[digest].size,
                replay_cost=definitions[digest].replay_cost,
                proof_cost=definitions[digest].proof_cost,
                in_count=len(definitions[digest].reads),
                out_count=len(definitions[digest].local_outputs),
                min_depth=depths[digest][0],
                max_depth=depths[digest][1],
            )
            for digest in order
        )


class _Boundary:
    """Lazy ``inputs ∪ ⋃_r Out(R_r)`` with ``O(depth)`` rank and unrank.

    Inputs occupy ``[0, input_count)`` and every unit's ``Out`` lies inside its
    interval, so the boundary in address order is the inputs followed by the
    units' declared outputs unit by unit; ``out_before`` on the frames gives
    the prefix sums.
    """

    __slots__ = ("_index", "count", "identity_digest")

    def __init__(self, index: Index) -> None:
        self._index = index
        frame = index._frame
        self.count = frame.definition.input_count + frame.definition.out_total
        self.identity_digest = identity_digest(
            "veritor/indexed-domain/boundary/v1", {"index": index.digest}
        )

    def _locate(self, address: int) -> tuple[Frame, int] | None:
        frame = _unit_frame_at(self._index._frame, address, REPLAY)
        if frame is None:
            return None
        outputs = frame.definition.local_outputs
        offset = address - frame.base
        position = bisect_left(outputs, offset)
        if position == len(outputs) or outputs[position] != offset:
            return None
        return frame, position

    def contains(self, item: int) -> bool:
        if type(item) is not int or not 0 <= item < self._index.n:
            return False
        return item < self._index.input_count or self._locate(item) is not None

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        if type(item) is not int or not 0 <= item < self._index.n:
            raise KeyError(item)
        inputs = self._index.input_count
        if item < inputs:
            return item
        located = self._locate(item)
        if located is None:
            raise KeyError(item)
        frame, position = located
        return inputs + frame.out_before + position

    def unrank(self, rank: int) -> int:
        if type(rank) is not int:
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.count:
            raise IndexError(f"rank {rank} is outside domain of size {self.count}")
        inputs = self._index.input_count
        if rank < inputs:
            return rank
        frame, offset = _frame_by_rank(
            self._index._frame,
            rank - inputs,
            REPLAY,
            lambda d: d.step_out,
            lambda d: d.out_total,
        )
        return frame.base + frame.definition.local_outputs[offset]

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
