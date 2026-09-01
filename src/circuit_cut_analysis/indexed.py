"""Exact, lazily expanded circuits represented by indexed gate families.

The representation is *succinct*, not approximate.  A :class:`GateFamily`
stands for a finite set of scalar gates, while each :class:`EdgeRule` gives
both directions of an exact relation between two families. Individual gates
and adjacency lists are allocated only when queried. Small results enter a
bounded LRU; oversized neighborhoods are returned without being retained.

This module deliberately does not claim that a succinct graph makes every
minimum-cut query cheap.  ``materialize_corridor`` expands the exact portion
needed by the existing finite-DAG solver and enforces an explicit safety
limit.  Model-specific quotienting can be layered on top without changing the
underlying gate identities or wire relation.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from circuit_cut_analysis.capacity import GateCapacity as GateCapacity
from circuit_cut_analysis.circuit import CircuitDAG, Edge, Gate

Index = tuple[int, ...]
_FAMILY_NAME = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class IndexDomain(Protocol):
    """Finite index set used by one gate family."""

    @property
    def arity(self) -> int:
        """Number of integer coordinates in each index."""

    @property
    def count(self) -> int:
        """Number of indices in the domain."""

    def contains(self, index: Index) -> bool:
        """Whether ``index`` belongs to this domain."""

    def iter_indices(self) -> Iterator[Index]:
        """Iterate every index in deterministic order."""


@dataclass(frozen=True, slots=True)
class RectangularDomain:
    """Cartesian product ``range(shape[0]) × ... × range(shape[-1])``."""

    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(size < 0 for size in self.shape):
            raise ValueError("rectangular dimensions cannot be negative")

    @property
    def arity(self) -> int:
        return len(self.shape)

    @property
    def count(self) -> int:
        return math.prod(self.shape)

    def contains(self, index: Index) -> bool:
        return len(index) == self.arity and all(
            0 <= coordinate < size
            for coordinate, size in zip(index, self.shape, strict=True)
        )

    def iter_indices(self) -> Iterator[Index]:
        yield from itertools.product(*(range(size) for size in self.shape))


@dataclass(frozen=True, slots=True)
class CausalPairsDomain:
    """Pairs ``(query, key)`` with ``key <= query`` (or ``key < query``)."""

    positions: int
    strict: bool = False

    def __post_init__(self) -> None:
        if self.positions < 0:
            raise ValueError("the number of positions cannot be negative")

    @property
    def arity(self) -> int:
        return 2

    @property
    def count(self) -> int:
        if self.strict:
            return self.positions * (self.positions - 1) // 2
        return self.positions * (self.positions + 1) // 2

    def contains(self, index: Index) -> bool:
        if len(index) != 2:
            return False
        query, key = index
        relation = key < query if self.strict else key <= query
        return 0 <= key < self.positions and 0 <= query < self.positions and relation

    def iter_indices(self) -> Iterator[Index]:
        offset = 1 if self.strict else 0
        for query in range(self.positions):
            for key in range(query + 1 - offset):
                yield (query, key)


@dataclass(frozen=True, slots=True)
class CausalReductionStepsDomain:
    """Pairs ``(query, step)`` for reduction steps ``1 <= step <= query``."""

    positions: int

    def __post_init__(self) -> None:
        if self.positions < 0:
            raise ValueError("the number of positions cannot be negative")

    @property
    def arity(self) -> int:
        return 2

    @property
    def count(self) -> int:
        return self.positions * (self.positions - 1) // 2

    def contains(self, index: Index) -> bool:
        if len(index) != 2:
            return False
        query, step = index
        return 0 <= query < self.positions and 1 <= step <= query

    def iter_indices(self) -> Iterator[Index]:
        for query in range(self.positions):
            for step in range(1, query + 1):
                yield (query, step)


@dataclass(frozen=True, slots=True)
class ProductDomain:
    """Concatenated Cartesian product of finite index domains."""

    parts: tuple[IndexDomain, ...]

    @property
    def arity(self) -> int:
        return sum(part.arity for part in self.parts)

    @property
    def count(self) -> int:
        return math.prod(part.count for part in self.parts)

    def contains(self, index: Index) -> bool:
        if len(index) != self.arity:
            return False
        offset = 0
        for part in self.parts:
            next_offset = offset + part.arity
            if not part.contains(index[offset:next_offset]):
                return False
            offset = next_offset
        return True

    def iter_indices(self) -> Iterator[Index]:
        iterables = tuple(part.iter_indices() for part in self.parts)
        for pieces in itertools.product(*iterables):
            yield tuple(itertools.chain.from_iterable(pieces))


@dataclass(frozen=True, slots=True)
class ExplicitDomain:
    """Small irregular domain with explicitly listed integer tuples."""

    indices: tuple[Index, ...]

    def __post_init__(self) -> None:
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("explicit domain indices must be unique")
        arities = {len(index) for index in self.indices}
        if len(arities) > 1:
            raise ValueError("explicit domain indices must have one arity")
        if any(coordinate < 0 for index in self.indices for coordinate in index):
            raise ValueError("explicit domain coordinates cannot be negative")

    @property
    def arity(self) -> int:
        return len(self.indices[0]) if self.indices else 0

    @property
    def count(self) -> int:
        return len(self.indices)

    def contains(self, index: Index) -> bool:
        return index in self.indices

    def iter_indices(self) -> Iterator[Index]:
        yield from self.indices


@dataclass(frozen=True, slots=True, order=True)
class GateRef:
    """Stable identity of one scalar gate in an indexed family."""

    family: str
    index: Index = ()

    @property
    def id(self) -> str:
        if not self.index:
            return self.family
        coordinates = ",".join(str(value) for value in self.index)
        return f"{self.family}[{coordinates}]"

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class GateFamily:
    """Metadata shared by a finite indexed collection of scalar gates."""

    name: str
    domain: IndexDomain
    index_names: tuple[str, ...]
    capacity: GateCapacity
    op: str
    primitive: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _FAMILY_NAME.fullmatch(self.name):
            raise ValueError(
                "family names must be slash-separated alphanumeric identifiers: "
                f"{self.name!r}"
            )
        if len(self.index_names) != self.domain.arity:
            raise ValueError(
                f"{self.name}: {len(self.index_names)} index names for "
                f"arity-{self.domain.arity} domain"
            )
        if len(set(self.index_names)) != len(self.index_names):
            raise ValueError(f"{self.name}: index coordinate names must be unique")
        if not self.op:
            raise ValueError(f"{self.name}: operation must be non-empty")
        if self.primitive == "":
            raise ValueError(f"{self.name}: primitive cannot be empty")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError(f"{self.name}: tags must be unique")

    @property
    def count(self) -> int:
        return self.domain.count

    @property
    def scope(self) -> str:
        return self.name.rpartition("/")[0]

    def ref(self, *index: int) -> GateRef:
        result = GateRef(self.name, tuple(index))
        if not self.domain.contains(result.index):
            raise ValueError(f"index {result.index!r} is outside family {self.name!r}")
        return result


@dataclass(frozen=True, slots=True)
class IndexedGate:
    """One lazily instantiated scalar gate."""

    ref: GateRef
    capacity: GateCapacity
    op: str
    primitive: str | None
    tags: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.ref.id


IndexMapper = Callable[[Index], Iterable[Index]]


@dataclass(frozen=True, slots=True)
class EdgeRule:
    """Bidirectional exact relation between two gate families."""

    name: str
    source_family: str
    target_family: str
    sources_for_target: IndexMapper
    targets_for_source: IndexMapper

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("edge rule name must be non-empty")
        if not self.source_family or not self.target_family:
            raise ValueError(f"{self.name}: edge-rule family names must be non-empty")


@dataclass(frozen=True, slots=True)
class CacheInfo:
    capacity: int
    max_cached_adjacency: int
    gate_entries: int
    predecessor_entries: int
    successor_entries: int
    gate_hits: int
    predecessor_hits: int
    successor_hits: int
    gate_evictions: int
    predecessor_evictions: int
    successor_evictions: int
    oversized_predecessors_skipped: int
    oversized_successors_skipped: int


@dataclass(frozen=True, slots=True)
class ScopeSummary:
    """Compressed inventory for one slash-delimited hierarchy scope."""

    scope: str
    family_count: int
    gate_count: int
    primitive_counts: Mapping[str, int]


class ExpansionLimitExceeded(RuntimeError):
    """Raised before an indexed expansion exceeds its declared safety budget."""


class IndexedCircuit:
    """Finite circuit exposed through exact, cached local queries."""

    __slots__ = (
        "_families",
        "_cache_capacity",
        "_gate_cache",
        "_gate_evictions",
        "_gate_hits",
        "_incoming",
        "_max_cached_adjacency",
        "_outputs",
        "_outgoing",
        "_pred_cache",
        "_pred_evictions",
        "_pred_hits",
        "_pred_oversized",
        "_rules",
        "_succ_cache",
        "_succ_evictions",
        "_succ_hits",
        "_succ_oversized",
    )

    def __init__(
        self,
        families: Iterable[GateFamily],
        edge_rules: Iterable[EdgeRule],
        outputs: Iterable[GateRef],
        *,
        cache_capacity: int = 4_096,
        max_cached_adjacency: int = 4_096,
    ) -> None:
        if cache_capacity <= 0:
            raise ValueError("cache capacity must be positive")
        if max_cached_adjacency < 0:
            raise ValueError("maximum cached adjacency cannot be negative")
        family_map: dict[str, GateFamily] = {}
        for family in families:
            if family.name in family_map:
                raise ValueError(f"duplicate gate family: {family.name!r}")
            family_map[family.name] = family
        if not family_map:
            raise ValueError("an indexed circuit needs at least one gate family")

        rules = tuple(edge_rules)
        rule_names: set[str] = set()
        incoming: dict[str, list[EdgeRule]] = {
            family_name: [] for family_name in family_map
        }
        outgoing: dict[str, list[EdgeRule]] = {
            family_name: [] for family_name in family_map
        }
        for rule in rules:
            if rule.name in rule_names:
                raise ValueError(f"duplicate edge rule: {rule.name!r}")
            rule_names.add(rule.name)
            if rule.source_family not in family_map:
                raise ValueError(
                    f"{rule.name}: unknown source family {rule.source_family!r}"
                )
            if rule.target_family not in family_map:
                raise ValueError(
                    f"{rule.name}: unknown target family {rule.target_family!r}"
                )
            outgoing[rule.source_family].append(rule)
            incoming[rule.target_family].append(rule)

        output_set = frozenset(outputs)
        if not output_set:
            raise ValueError("an indexed circuit needs at least one output gate")
        for output in output_set:
            self._validate_ref_from_map(output, family_map)

        self._families = MappingProxyType(family_map)
        self._rules = rules
        self._outputs = output_set
        self._incoming = {
            name: tuple(family_rules) for name, family_rules in incoming.items()
        }
        self._outgoing = {
            name: tuple(family_rules) for name, family_rules in outgoing.items()
        }
        self._cache_capacity = cache_capacity
        self._max_cached_adjacency = max_cached_adjacency
        self._gate_cache: OrderedDict[GateRef, IndexedGate] = OrderedDict()
        self._pred_cache: OrderedDict[GateRef, frozenset[GateRef]] = OrderedDict()
        self._succ_cache: OrderedDict[GateRef, frozenset[GateRef]] = OrderedDict()
        self._gate_hits = 0
        self._pred_hits = 0
        self._succ_hits = 0
        self._gate_evictions = 0
        self._pred_evictions = 0
        self._succ_evictions = 0
        self._pred_oversized = 0
        self._succ_oversized = 0

    @staticmethod
    def _validate_ref_from_map(
        ref: GateRef,
        families: Mapping[str, GateFamily],
    ) -> GateFamily:
        try:
            family = families[ref.family]
        except KeyError:
            raise ValueError(f"unknown gate family: {ref.family!r}") from None
        if not family.domain.contains(ref.index):
            raise ValueError(f"index {ref.index!r} is outside family {ref.family!r}")
        return family

    @property
    def families(self) -> Mapping[str, GateFamily]:
        return self._families

    @property
    def edge_rules(self) -> tuple[EdgeRule, ...]:
        return self._rules

    @property
    def outputs(self) -> frozenset[GateRef]:
        return self._outputs

    @property
    def cache_capacity(self) -> int:
        return self._cache_capacity

    @property
    def max_cached_adjacency(self) -> int:
        return self._max_cached_adjacency

    @property
    def gate_count(self) -> int:
        return sum(family.count for family in self._families.values())

    @property
    def computed_gate_count(self) -> int:
        return sum(
            family.count for family in self._families.values() if family.op != "input"
        )

    @property
    def primitive_counts(self) -> Mapping[str, int]:
        counts: Counter[str] = Counter()
        for family in self._families.values():
            if family.primitive is not None and family.count:
                counts[family.primitive] += family.count
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def primitive_gate_count(self) -> int:
        return sum(self.primitive_counts.values())

    def scope_summaries(self) -> tuple[ScopeSummary, ...]:
        scopes: dict[str, list[GateFamily]] = {}
        for family in self._families.values():
            pieces = family.name.split("/")[:-1]
            for depth in range(len(pieces) + 1):
                scope = "/".join(pieces[:depth])
                scopes.setdefault(scope, []).append(family)
        summaries: list[ScopeSummary] = []
        for scope, families in sorted(scopes.items()):
            primitives: Counter[str] = Counter()
            for family in families:
                if family.primitive is not None:
                    primitives[family.primitive] += family.count
            summaries.append(
                ScopeSummary(
                    scope=scope,
                    family_count=len(families),
                    gate_count=sum(family.count for family in families),
                    primitive_counts=MappingProxyType(dict(sorted(primitives.items()))),
                )
            )
        return tuple(summaries)

    def require_ref(self, ref: GateRef) -> GateFamily:
        return self._validate_ref_from_map(ref, self._families)

    def ref_from_id(self, gate_id: str) -> GateRef:
        """Invert the stable textual ID without enumerating its family."""

        if gate_id.endswith("]"):
            family_name, separator, coordinates = gate_id[:-1].rpartition("[")
            if not separator:
                raise ValueError(f"invalid indexed gate id: {gate_id!r}")
            try:
                index = tuple(int(value) for value in coordinates.split(","))
            except ValueError as error:
                raise ValueError(f"invalid indexed gate id: {gate_id!r}") from error
            ref = GateRef(family_name, index)
        else:
            ref = GateRef(gate_id)
        self.require_ref(ref)
        return ref

    def gate(self, ref: GateRef) -> IndexedGate:
        try:
            cached = self._gate_cache.pop(ref)
        except KeyError:
            cached = None
        if cached is not None:
            self._gate_hits += 1
            self._gate_cache[ref] = cached
            return cached
        family = self.require_ref(ref)
        gate = IndexedGate(
            ref=ref,
            capacity=family.capacity,
            op=family.op,
            primitive=family.primitive,
            tags=family.tags,
        )
        self._gate_cache[ref] = gate
        if len(self._gate_cache) > self._cache_capacity:
            self._gate_cache.popitem(last=False)
            self._gate_evictions += 1
        return gate

    def _related_refs(
        self,
        ref: GateRef,
        *,
        predecessor_direction: bool,
    ) -> frozenset[GateRef]:
        self.require_ref(ref)
        rules = (
            self._incoming[ref.family]
            if predecessor_direction
            else self._outgoing[ref.family]
        )
        related: set[GateRef] = set()
        for rule in rules:
            family_name = (
                rule.source_family if predecessor_direction else rule.target_family
            )
            mapper = (
                rule.sources_for_target
                if predecessor_direction
                else rule.targets_for_source
            )
            for index in mapper(ref.index):
                related_ref = GateRef(family_name, tuple(index))
                try:
                    self.require_ref(related_ref)
                except ValueError as error:
                    direction = "source" if predecessor_direction else "target"
                    raise ValueError(
                        f"edge rule {rule.name!r} generated invalid {direction} "
                        f"{related_ref.id!r} for {ref.id!r}"
                    ) from error
                related.add(related_ref)
        return frozenset(related)

    def predecessors(self, ref: GateRef) -> frozenset[GateRef]:
        try:
            cached = self._pred_cache.pop(ref)
        except KeyError:
            cached = None
        if cached is not None:
            self._pred_hits += 1
            self._pred_cache[ref] = cached
            return cached
        result = self._related_refs(ref, predecessor_direction=True)
        if len(result) <= self._max_cached_adjacency:
            self._pred_cache[ref] = result
            if len(self._pred_cache) > self._cache_capacity:
                self._pred_cache.popitem(last=False)
                self._pred_evictions += 1
        else:
            self._pred_oversized += 1
        return result

    def successors(self, ref: GateRef) -> frozenset[GateRef]:
        try:
            cached = self._succ_cache.pop(ref)
        except KeyError:
            cached = None
        if cached is not None:
            self._succ_hits += 1
            self._succ_cache[ref] = cached
            return cached
        result = self._related_refs(ref, predecessor_direction=False)
        if len(result) <= self._max_cached_adjacency:
            self._succ_cache[ref] = result
            if len(self._succ_cache) > self._cache_capacity:
                self._succ_cache.popitem(last=False)
                self._succ_evictions += 1
        else:
            self._succ_oversized += 1
        return result

    def cache_info(self) -> CacheInfo:
        return CacheInfo(
            capacity=self._cache_capacity,
            max_cached_adjacency=self._max_cached_adjacency,
            gate_entries=len(self._gate_cache),
            predecessor_entries=len(self._pred_cache),
            successor_entries=len(self._succ_cache),
            gate_hits=self._gate_hits,
            predecessor_hits=self._pred_hits,
            successor_hits=self._succ_hits,
            gate_evictions=self._gate_evictions,
            predecessor_evictions=self._pred_evictions,
            successor_evictions=self._succ_evictions,
            oversized_predecessors_skipped=self._pred_oversized,
            oversized_successors_skipped=self._succ_oversized,
        )

    def clear_cache(self) -> None:
        self._gate_cache.clear()
        self._pred_cache.clear()
        self._succ_cache.clear()
        self._gate_hits = 0
        self._pred_hits = 0
        self._succ_hits = 0
        self._gate_evictions = 0
        self._pred_evictions = 0
        self._succ_evictions = 0
        self._pred_oversized = 0
        self._succ_oversized = 0

    def iter_gate_refs(self, *, max_gates: int | None = None) -> Iterator[GateRef]:
        if max_gates is not None and self.gate_count > max_gates:
            raise ExpansionLimitExceeded(
                f"indexed circuit has {self.gate_count:,} gates; limit is {max_gates:,}"
            )
        for family in self._families.values():
            for index in family.domain.iter_indices():
                yield GateRef(family.name, index)

    def iter_edges(
        self,
        *,
        max_gates: int | None = None,
        max_edges: int | None = None,
    ) -> Iterator[tuple[GateRef, GateRef]]:
        edge_count = 0
        for target in self.iter_gate_refs(max_gates=max_gates):
            for source in sorted(self.predecessors(target)):
                edge_count += 1
                if max_edges is not None and edge_count > max_edges:
                    raise ExpansionLimitExceeded(
                        f"edge expansion exceeded limit {max_edges:,}"
                    )
                yield source, target

    def validate_bidirectional(
        self,
        *,
        max_gates: int,
        max_edges: int,
    ) -> int:
        """Exhaustively verify both enumerators on a safely small instance."""

        forward_edges = set(self.iter_edges(max_gates=max_gates, max_edges=max_edges))
        reverse_edges: set[tuple[GateRef, GateRef]] = set()
        for source in self.iter_gate_refs(max_gates=max_gates):
            for target in self.successors(source):
                reverse_edges.add((source, target))
                if len(reverse_edges) > max_edges:
                    raise ExpansionLimitExceeded(
                        f"reverse edge expansion exceeded limit {max_edges:,}"
                    )
        if forward_edges != reverse_edges:
            only_forward = sorted(forward_edges.difference(reverse_edges))
            only_reverse = sorted(reverse_edges.difference(forward_edges))
            raise ValueError(
                "edge rules disagree by direction; "
                f"predecessor-only={only_forward[:3]!r}, "
                f"successor-only={only_reverse[:3]!r}"
            )
        return len(forward_edges)

    def _reachable(
        self,
        starts: Iterable[GateRef],
        *,
        predecessor_direction: bool,
        max_gates: int,
        allowed: frozenset[GateRef] | None = None,
        stop_at: frozenset[GateRef] = frozenset(),
    ) -> frozenset[GateRef]:
        start_set = frozenset(starts)
        for start in start_set:
            self.require_ref(start)
        if allowed is not None and not start_set.issubset(allowed):
            raise ValueError("reachability starts must belong to the allowed set")
        seen = set(start_set)
        queue = deque(start_set)
        while queue:
            ref = queue.popleft()
            if ref in stop_at:
                continue
            adjacent = (
                self.predecessors(ref)
                if predecessor_direction
                else self.successors(ref)
            )
            for neighbor in adjacent:
                if allowed is not None and neighbor not in allowed:
                    continue
                if neighbor in seen:
                    continue
                if len(seen) >= max_gates:
                    direction = "ancestor" if predecessor_direction else "descendant"
                    raise ExpansionLimitExceeded(
                        f"{direction} expansion exceeded limit {max_gates:,}"
                    )
                seen.add(neighbor)
                queue.append(neighbor)
        return frozenset(seen)

    def materialize(
        self,
        *,
        max_gates: int,
        max_edges: int,
    ) -> CircuitDAG:
        """Expand the complete indexed graph into the existing explicit DAG."""

        refs = tuple(self.iter_gate_refs(max_gates=max_gates))
        return self._materialize_refs(
            refs,
            self._outputs,
            max_edges=max_edges,
        )

    def materialize_corridor(
        self,
        sources: Iterable[GateRef],
        outputs: Iterable[GateRef] | None = None,
        *,
        max_gates: int,
        max_edges: int,
    ) -> CircuitDAG:
        """Expand exactly the selected source-to-output live corridor."""

        source_set = frozenset(sources)
        output_set = self._outputs if outputs is None else frozenset(outputs)
        if not source_set or not output_set:
            raise ValueError("corridor materialization needs sources and outputs")
        descendants = self._reachable(
            source_set,
            predecessor_direction=False,
            max_gates=max_gates,
            stop_at=output_set,
        )
        reachable_outputs = output_set.intersection(descendants)
        if not reachable_outputs:
            raise ValueError(
                "selected indexed sources have no path to selected outputs"
            )
        ancestors = self._reachable(
            reachable_outputs,
            predecessor_direction=True,
            max_gates=max_gates,
            allowed=descendants,
            stop_at=source_set,
        )
        corridor = ancestors
        return self._materialize_refs(
            tuple(sorted(corridor)),
            reachable_outputs,
            max_edges=max_edges,
        )

    def _materialize_refs(
        self,
        refs: tuple[GateRef, ...],
        outputs: Iterable[GateRef],
        *,
        max_edges: int,
    ) -> CircuitDAG:
        ref_set = frozenset(refs)
        gates: list[Gate] = []
        for ref in refs:
            indexed_gate = self.gate(ref)
            gates.append(Gate(ref.id, indexed_gate.capacity, indexed_gate.op))

        edges: list[Edge] = []
        for target in refs:
            for source in self.predecessors(target):
                if source not in ref_set:
                    continue
                edges.append((source.id, target.id))
                if len(edges) > max_edges:
                    raise ExpansionLimitExceeded(
                        f"materialized edges exceeded limit {max_edges:,}"
                    )
        output_ids = [ref.id for ref in outputs if ref in ref_set]
        return CircuitDAG(gates, edges, output_ids)
