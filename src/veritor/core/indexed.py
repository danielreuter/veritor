"""Deterministic finite indexed domains with materialized and lazy forms."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from sys import maxsize
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from .errors import InvalidArtifact
from .identity import Digest, JSONValue, identity_digest, validate_digest
from .ids import Position, position


@runtime_checkable
class IndexedDomain[T](Protocol):
    """A finite domain with stable zero-based rank and unrank operations."""

    @property
    def count(self) -> int: ...

    @property
    def identity_digest(self) -> Digest: ...

    def contains(self, item: T) -> bool: ...

    def rank(self, item: T) -> int: ...

    def unrank(self, rank: int) -> T: ...


@runtime_checkable
class FiniteIndexedDomain[T](IndexedDomain[T], Protocol):
    """Descriptive spelling for any finite indexed domain."""


@runtime_checkable
class LazyIndexedDomain[T](FiniteIndexedDomain[T], Protocol):
    """Contract for domains that may compute members without materializing all."""


def _checked_rank(rank: object, count: int) -> int:
    if type(rank) is not int:
        raise TypeError("rank must be an integer")
    if rank < 0 or rank >= count:
        raise IndexError(f"rank {rank} is outside domain of size {count}")
    return rank


@dataclass(frozen=True, slots=True, init=False)
class ExplicitIndexedDomain[T]:
    """An immutable ordered domain backed by an explicit tuple."""

    _items: tuple[T, ...]
    _rank_by_item: Mapping[T, int] = field(repr=False, compare=False, hash=False)
    identity_digest: Digest

    def __init__(self, items: Iterable[T]) -> None:
        values = tuple(items)
        ranks: dict[T, int] = {}
        for index, item in enumerate(values):
            try:
                if item in ranks:
                    raise InvalidArtifact(
                        f"explicit indexed domain repeats item {item!r}"
                    )
                ranks[item] = index
            except TypeError as error:
                raise InvalidArtifact(
                    "explicit indexed domain items must be hashable"
                ) from error
        digest = identity_digest(
            "veritor/indexed-domain/explicit/v1",
            {"items": cast(JSONValue, list(values))},
        )
        object.__setattr__(self, "_items", values)
        object.__setattr__(self, "_rank_by_item", MappingProxyType(ranks))
        object.__setattr__(self, "identity_digest", digest)

    @property
    def items(self) -> tuple[T, ...]:
        return self._items

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def contains(self, item: T) -> bool:
        try:
            return item in self._rank_by_item
        except TypeError:
            return False

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: T) -> int:
        try:
            return self._rank_by_item[item]
        except (KeyError, TypeError) as error:
            raise KeyError(item) from error

    def unrank(self, rank: int) -> T:
        return self._items[_checked_rank(rank, self.count)]

    def at_rank(self, rank: int) -> T:
        return self.unrank(rank)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True, slots=True, init=False)
class RangeIndexedDomain:
    """A lazy nonnegative integer domain equivalent to ``range``."""

    start: int
    stop: int
    step: int
    count: int
    identity_digest: Digest

    def __init__(self, start: int, stop: int | None = None, step: int = 1) -> None:
        if stop is None:
            stop = start
            start = 0
        for value, name in ((start, "start"), (stop, "stop"), (step, "step")):
            if type(value) is not int:
                raise InvalidArtifact(f"range domain {name} must be an integer")
        if start < 0 or stop < 0:
            raise InvalidArtifact("range domain bounds must be nonnegative")
        if stop < start:
            raise InvalidArtifact("range domain stop must not precede start")
        if step <= 0:
            raise InvalidArtifact("range domain step must be positive")
        count = 0 if start == stop else (stop - start + step - 1) // step
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "count", count)
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/indexed-domain/range/v1",
                {"start": start, "step": step, "stop": stop},
            ),
        )

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def contains(self, item: int) -> bool:
        return (
            type(item) is int
            and self.start <= item < self.stop
            and (item - self.start) % self.step == 0
        )

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        if not self.contains(item):
            raise KeyError(item)
        return (item - self.start) // self.step

    def unrank(self, rank: int) -> Position:
        checked = _checked_rank(rank, self.count)
        return Position(self.start + checked * self.step)

    def at_rank(self, rank: int) -> Position:
        return self.unrank(rank)

    def __iter__(self) -> Iterator[Position]:
        for rank in range(self.count):
            yield self.unrank(rank)

    def __len__(self) -> int:
        if self.count > maxsize:
            raise OverflowError("domain is too large for len(); use .count")
        return self.count


@dataclass(frozen=True, slots=True, init=False)
class IntervalDomain:
    """A sorted union of disjoint half-open position intervals.

    Rank, unrank, and membership cost ``O(log k)`` in the number of intervals,
    so a replay boundary made of a few long runs is cheap even when it holds
    millions of positions.  ``intervals`` is canonical: sorted, disjoint,
    nonempty, and never adjacent.
    """

    intervals: tuple[tuple[int, int], ...]
    _starts: tuple[int, ...] = field(repr=False, compare=False, hash=False)
    _prefix: tuple[int, ...] = field(repr=False, compare=False, hash=False)
    count: int
    identity_digest: Digest

    def __init__(self, intervals: Iterable[tuple[int, int]]) -> None:
        merged: list[list[int]] = []
        for start, stop in sorted(intervals):
            if type(start) is not int or type(stop) is not int or start < 0:
                raise InvalidArtifact("interval bounds must be nonnegative integers")
            if stop <= start:
                if stop == start:
                    continue
                raise InvalidArtifact("interval stop must not precede start")
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        canonical = tuple((start, stop) for start, stop in merged)
        prefix: list[int] = [0]
        for start, stop in canonical:
            prefix.append(prefix[-1] + stop - start)
        object.__setattr__(self, "intervals", canonical)
        object.__setattr__(self, "_starts", tuple(start for start, _ in canonical))
        object.__setattr__(self, "_prefix", tuple(prefix))
        object.__setattr__(self, "count", prefix[-1])
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/indexed-domain/intervals/v1",
                {"intervals": [list(item) for item in canonical]},
            ),
        )

    @classmethod
    def from_positions(cls, positions: Iterable[int]) -> IntervalDomain:
        """Coalesce arbitrary positions into runs."""

        return cls((item, item + 1) for item in positions)

    @classmethod
    def from_range(cls, start: int, stop: int) -> IntervalDomain:
        return cls(((start, stop),))

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def _interval_index(self, item: int) -> int | None:
        index = bisect_right(self._starts, item) - 1
        if index >= 0 and item < self.intervals[index][1]:
            return index
        return None

    def contains(self, item: int) -> bool:
        return type(item) is int and self._interval_index(item) is not None

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        index = self._interval_index(item) if type(item) is int else None
        if index is None:
            raise KeyError(item)
        return self._prefix[index] + item - self.intervals[index][0]

    def unrank(self, rank: int) -> Position:
        checked = _checked_rank(rank, self.count)
        index = bisect_right(self._prefix, checked) - 1
        return Position(self.intervals[index][0] + checked - self._prefix[index])

    def count_below(self, item: int) -> int:
        """Return how many members are strictly less than ``item``."""

        index = bisect_right(self._starts, item) - 1
        if index < 0:
            return 0
        start, stop = self.intervals[index]
        return self._prefix[index] + min(item, stop) - start

    def intersect(self, start: int, stop: int) -> IntervalDomain:
        """Return the members inside ``[start, stop)``."""

        return IntervalDomain(
            (max(a, start), min(b, stop))
            for a, b in self.intervals
            if a < stop and b > start
        )

    def complement_within(self, start: int, stop: int) -> IntervalDomain:
        """Return ``[start, stop)`` minus this domain."""

        gaps: list[tuple[int, int]] = []
        cursor = start
        for a, b in self.intersect(start, stop).intervals:
            if a > cursor:
                gaps.append((cursor, a))
            cursor = b
        if cursor < stop:
            gaps.append((cursor, stop))
        return IntervalDomain(gaps)

    def __iter__(self) -> Iterator[Position]:
        for start, stop in self.intervals:
            for item in range(start, stop):
                yield Position(item)

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True, slots=True, init=False)
class IntervalDifferenceDomain:
    """A half-open interval minus excluded members given as disjoint runs.

    This is the lazy difference an index needs for a unit's interior: the
    unit's interval minus its interface, ``Out`` as runs ``(start, count,
    stride)`` of positions.  The runs must be pairwise disjoint (the compiler
    guarantees this for an interface); a run of one member has stride ``0``.
    Members are ranked in position order.  Membership and rank cost ``O(k)``
    in the number ``k`` of runs, unrank ``O(k log n)``; nothing is
    materialized.
    """

    start: int
    stop: int
    excluded: tuple[tuple[int, int, int], ...]
    count: int
    identity_digest: Digest

    def __init__(
        self, start: int, stop: int, excluded: Iterable[tuple[int, int, int]]
    ) -> None:
        if type(start) is not int or type(stop) is not int or start < 0:
            raise InvalidArtifact("interval bounds must be nonnegative integers")
        if stop < start:
            raise InvalidArtifact("interval stop must not precede start")
        runs = tuple(tuple(run) for run in excluded)
        for run in runs:
            if len(run) != 3 or any(type(value) is not int for value in run):
                raise InvalidArtifact("excluded runs must be (start, count, stride) integers")
            first, count, stride = run
            if count < 1 or stride < 0 or (stride == 0) != (count == 1):
                raise InvalidArtifact("an excluded run has a positive stride unless it is one member")
            if not start <= first < stop or first + (count - 1) * stride >= stop:
                raise InvalidArtifact("excluded members must lie inside the interval")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "excluded", runs)
        object.__setattr__(self, "count", stop - start - sum(run[1] for run in runs))
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/indexed-domain/interval-difference/v2",
                {"excluded": [list(run) for run in runs], "start": start, "stop": stop},
            ),
        )

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def _excluded_below(self, item: int) -> int:
        """How many excluded members are strictly less than ``item``."""

        total = 0
        for first, count, stride in self.excluded:
            if item > first:
                total += 1 if stride == 0 else min(count, (item - 1 - first) // stride + 1)
        return total

    def _is_excluded(self, item: int) -> bool:
        for first, count, stride in self.excluded:
            if item == first:
                return True
            if stride and item > first:
                k, remainder = divmod(item - first, stride)
                if remainder == 0 and k < count:
                    return True
        return False

    def contains(self, item: int) -> bool:
        if type(item) is not int or not self.start <= item < self.stop:
            return False
        return not self._is_excluded(item)

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        if not self.contains(item):
            raise KeyError(item)
        return item - self.start - self._excluded_below(item)

    def unrank(self, rank: int) -> Position:
        checked = _checked_rank(rank, self.count)
        # ``p - start + 1 - excluded_below(p + 1)`` members lie in ``[start, p]``;
        # it grows only at members, so the least ``p`` reaching ``rank + 1`` is one.
        low, high = self.start + checked, self.stop - 1
        while low < high:
            middle = (low + high) // 2
            if middle - self.start + 1 - self._excluded_below(middle + 1) > checked:
                high = middle
            else:
                low = middle + 1
        return Position(low)

    def __iter__(self) -> Iterator[Position]:
        for item in range(self.start, self.stop):
            if not self._is_excluded(item):
                yield Position(item)

    def __len__(self) -> int:
        return self.count


def iter_domain[T](domain: IndexedDomain[T]) -> Iterator[T]:
    """Iterate a domain in canonical rank order."""

    for rank in range(domain.count):
        yield domain.unrank(rank)


def domains_equal[T](
    left: IndexedDomain[T],
    right: IndexedDomain[T],
) -> bool:
    """Compare represented members and order, independent of implementation."""

    if left.count != right.count:
        return False
    if left.identity_digest == right.identity_digest:
        return True
    return all(left.unrank(rank) == right.unrank(rank) for rank in range(left.count))


def position_domain(
    value: IndexedDomain[Position] | Iterable[int],
    *,
    field_name: str = "positions",
) -> IndexedDomain[Position]:
    """Validate or materialize a domain whose members are positions."""

    domain: IndexedDomain[Position]
    if isinstance(value, IndexedDomain):
        domain = value
    else:
        try:
            checked_values = tuple(
                position(item, field_name=f"{field_name}[{rank}]")
                for rank, item in enumerate(value)
            )
        except TypeError as error:
            raise InvalidArtifact(f"{field_name} must be an indexed domain") from error
        domain = ExplicitIndexedDomain(checked_values)
    if type(domain.count) is not int or domain.count < 0:
        raise InvalidArtifact(f"{field_name} has an invalid count")
    validate_digest(domain.identity_digest, f"{field_name} identity_digest")
    if isinstance(domain, ExplicitIndexedDomain):
        ranks: Iterable[int] = range(domain.count)
    elif domain.count:
        ranks = (0, domain.count - 1)
    else:
        ranks = ()
    for rank in ranks:
        item = position(domain.unrank(rank), field_name=f"{field_name}[{rank}]")
        if domain.rank(item) != rank:
            raise InvalidArtifact(f"{field_name} violates rank/unrank")
    return domain


ExplicitDomain = ExplicitIndexedDomain
RangeDomain = RangeIndexedDomain
