"""Deterministic finite indexed domains with materialized and lazy forms."""

from __future__ import annotations

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
