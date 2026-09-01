"""Lazy, semantics-independent rank/unrank for indexed gate references."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from circuit_cut_analysis.indexed import (
    CausalPairsDomain,
    CausalReductionStepsDomain,
    ExplicitDomain,
    GateRef,
    Index,
    IndexDomain,
    IndexedCircuit,
    ProductDomain,
    RectangularDomain,
)
from veritor.core import Digest, JSONValue, identity_digest


class UnsupportedIndexDomain(TypeError):
    """A domain has no proven stable rank/unrank implementation."""


def _checked_rank(rank: object, count: int) -> int:
    if type(rank) is not int:
        raise TypeError("rank must be an integer")
    if rank < 0 or rank >= count:
        raise IndexError(f"rank {rank} is outside domain of size {count}")
    return rank


def _domain_manifest(domain: IndexDomain) -> dict[str, JSONValue]:
    if isinstance(domain, RectangularDomain):
        return {"kind": "rectangular", "shape": list(domain.shape)}
    if isinstance(domain, CausalPairsDomain):
        return {
            "kind": "causal-pairs",
            "positions": domain.positions,
            "strict": domain.strict,
        }
    if isinstance(domain, CausalReductionStepsDomain):
        return {
            "kind": "causal-reduction-steps",
            "positions": domain.positions,
        }
    if isinstance(domain, ProductDomain):
        return {
            "kind": "product",
            "parts": [_domain_manifest(part) for part in domain.parts],
        }
    if isinstance(domain, ExplicitDomain):
        return {
            "indices": [list(index) for index in domain.indices],
            "kind": "explicit",
        }
    raise UnsupportedIndexDomain(
        f"no sound rank/unrank adapter for {type(domain).__name__}"
    )


def rank_index(domain: IndexDomain, index: Index) -> int:
    """Rank an index in exactly the domain's documented iteration order."""

    if not domain.contains(index):
        raise KeyError(index)
    if isinstance(domain, RectangularDomain):
        rank = 0
        for coordinate, size in zip(index, domain.shape, strict=True):
            rank = rank * size + coordinate
        return rank
    if isinstance(domain, CausalPairsDomain):
        query, key = index
        prefix = query * (query - 1) // 2 if domain.strict else query * (query + 1) // 2
        return prefix + key
    if isinstance(domain, CausalReductionStepsDomain):
        query, step = index
        return query * (query - 1) // 2 + step - 1
    if isinstance(domain, ProductDomain):
        rank = 0
        offset = 0
        for part in domain.parts:
            next_offset = offset + part.arity
            rank = rank * part.count + rank_index(part, index[offset:next_offset])
            offset = next_offset
        return rank
    if isinstance(domain, ExplicitDomain):
        try:
            return domain.indices.index(index)
        except ValueError as error:
            raise KeyError(index) from error
    raise UnsupportedIndexDomain(
        f"no sound rank/unrank adapter for {type(domain).__name__}"
    )


def unrank_index(domain: IndexDomain, rank: int) -> Index:
    """Invert :func:`rank_index` without enumerating the domain."""

    remaining = _checked_rank(rank, domain.count)
    if isinstance(domain, RectangularDomain):
        coordinates = [0] * len(domain.shape)
        for index in range(len(domain.shape) - 1, -1, -1):
            remaining, coordinates[index] = divmod(remaining, domain.shape[index])
        return tuple(coordinates)
    if isinstance(domain, CausalPairsDomain):
        low = 0
        high = domain.positions
        while low < high:
            middle = (low + high) // 2
            stop = (
                (middle + 1) * middle // 2
                if domain.strict
                else (middle + 1) * (middle + 2) // 2
            )
            if remaining < stop:
                high = middle
            else:
                low = middle + 1
        query = low
        prefix = query * (query - 1) // 2 if domain.strict else query * (query + 1) // 2
        return query, remaining - prefix
    if isinstance(domain, CausalReductionStepsDomain):
        low = 1
        high = domain.positions
        while low < high:
            middle = (low + high) // 2
            stop = middle * (middle + 1) // 2
            if remaining < stop:
                high = middle
            else:
                low = middle + 1
        query = low
        prefix = query * (query - 1) // 2
        return query, remaining - prefix + 1
    if isinstance(domain, ProductDomain):
        pieces: list[Index] = [()] * len(domain.parts)
        for index in range(len(domain.parts) - 1, -1, -1):
            part = domain.parts[index]
            remaining, local_rank = divmod(remaining, part.count)
            pieces[index] = unrank_index(part, local_rank)
        return tuple(coordinate for piece in pieces for coordinate in piece)
    if isinstance(domain, ExplicitDomain):
        return domain.indices[remaining]
    raise UnsupportedIndexDomain(
        f"no sound rank/unrank adapter for {type(domain).__name__}"
    )


def supports_rank_unrank(domain: IndexDomain) -> bool:
    """Return whether the domain is composed entirely of supported forms."""

    try:
        _domain_manifest(domain)
    except UnsupportedIndexDomain:
        return False
    return True


@dataclass(frozen=True, slots=True, init=False)
class GateRefIndexedDomain:
    """Global lazy gate domain in stable family/intra-family order."""

    circuit: IndexedCircuit
    computed_only: bool
    family_names: tuple[str, ...]
    _offsets: tuple[int, ...]
    _family_index: Mapping[str, int]
    count: int
    identity_digest: Digest

    def __init__(
        self,
        circuit: IndexedCircuit,
        *,
        computed_only: bool = False,
    ) -> None:
        if not isinstance(circuit, IndexedCircuit):
            raise TypeError("GateRefIndexedDomain requires an IndexedCircuit")
        families = tuple(
            family
            for family in circuit.families.values()
            if not computed_only or family.op != "input"
        )
        for family in families:
            _domain_manifest(family.domain)
        offsets = [0]
        for family in families:
            offsets.append(offsets[-1] + family.count)
        names = tuple(family.name for family in families)
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "computed_only", computed_only)
        object.__setattr__(self, "family_names", names)
        object.__setattr__(self, "_offsets", tuple(offsets))
        object.__setattr__(
            self,
            "_family_index",
            MappingProxyType({name: index for index, name in enumerate(names)}),
        )
        object.__setattr__(self, "count", offsets[-1])
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/plugins/gate-ref-indexed-domain/v1",
                {
                    "computed_only": computed_only,
                    "families": [
                        {
                            "count": family.count,
                            "domain": _domain_manifest(family.domain),
                            "name": family.name,
                        }
                        for family in families
                    ],
                },
            ),
        )

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def contains(self, item: GateRef) -> bool:
        if not isinstance(item, GateRef):
            return False
        family_index = self._family_index.get(item.family)
        if family_index is None:
            return False
        family = self.circuit.families[item.family]
        return family.domain.contains(item.index)

    def __contains__(self, item: object) -> bool:
        return isinstance(item, GateRef) and self.contains(item)

    def rank(self, item: GateRef) -> int:
        if not self.contains(item):
            raise KeyError(item)
        family_index = self._family_index[item.family]
        family = self.circuit.families[item.family]
        return self._offsets[family_index] + rank_index(family.domain, item.index)

    def unrank(self, rank: int) -> GateRef:
        checked = _checked_rank(rank, self.count)
        family_index = bisect_right(self._offsets, checked) - 1
        family_name = self.family_names[family_index]
        family = self.circuit.families[family_name]
        local_rank = checked - self._offsets[family_index]
        return GateRef(family_name, unrank_index(family.domain, local_rank))

    def at_rank(self, rank: int) -> GateRef:
        return self.unrank(rank)

    def __iter__(self) -> Iterator[GateRef]:
        for rank in range(self.count):
            yield self.unrank(rank)
