"""Guarantee-carrying capacity-oracle contracts and structural adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from circuit_cut_analysis.capacity import GateCapacity, LogCardinality
from circuit_cut_analysis.capacity_oracle import (
    ExplicitCircuitCapacityOracle as CutExplicitCircuitCapacityOracle,
)
from circuit_cut_analysis.circuit import CircuitDAG, Gate
from veritor.core.circuit import Circuit
from veritor.core.ids import Position
from veritor.core.index import Index


@dataclass(frozen=True, slots=True)
class CapacityEvidence[SupportT]:
    """A certified structural-capacity interval for one attacked support.

    ``lower_bound`` and ``upper_bound`` are independently certified.  Exactness
    is a derived property and is never inferred from an oracle's method name.
    A trusted in-process solver result may be retained in ``certificate``;
    portable flow certificates are not required by this analysis layer.
    """

    lower_bound: LogCardinality
    upper_bound: LogCardinality
    requested_support: SupportT
    evaluated_support: SupportT
    method: str
    certificate: object | None = None
    notes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        zero = LogCardinality.zero()
        if not isinstance(self.lower_bound, LogCardinality) or not isinstance(
            self.upper_bound,
            LogCardinality,
        ):
            raise TypeError("capacity bounds must be LogCardinality values")
        if self.lower_bound < zero:
            raise ValueError("capacity lower bound cannot be negative")
        if self.lower_bound > self.upper_bound:
            raise ValueError("capacity lower bound exceeds upper bound")
        if type(self.method) is not str or not self.method:
            raise ValueError("capacity evidence method must be nonempty")
        if any(type(item) is not str or not item for item in self.notes):
            raise ValueError("capacity evidence notes must be nonempty strings")
        if any(type(item) is not str or not item for item in self.assumptions):
            raise ValueError("capacity assumptions must be nonempty strings")

    @property
    def exact_capacity(self) -> LogCardinality | None:
        """Return the capacity iff the certified interval is a singleton."""

        if self.lower_bound == self.upper_bound:
            return self.lower_bound
        return None

    @property
    def is_exact(self) -> bool:
        return self.exact_capacity is not None


CapacityEvaluation = CapacityEvidence


@runtime_checkable
class CapacityOracle[SupportT](Protocol):
    """Monotone structural-capacity oracle used by finite bound solvers.

    The represented capacity must be monotone under support inclusion.  Each
    answer may be exact or a certified interval.
    """

    def evaluate(self, attack_support: frozenset[SupportT]) -> object:
        """Return capacity evidence for ``attack_support``."""


def coerce_capacity_evidence(
    value: object,
    *,
    requested_support: object | None = None,
) -> CapacityEvidence[Any]:
    """Normalize this package's evidence or an existing cut-oracle answer."""

    if isinstance(value, CapacityEvidence):
        return value
    try:
        lower = value.lower_bound  # type: ignore[attr-defined]
        upper = value.upper_bound  # type: ignore[attr-defined]
        source_requested = value.requested_support  # type: ignore[attr-defined]
        source_evaluated = value.evaluated_support  # type: ignore[attr-defined]
        method = value.method  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(
            "capacity oracle must return certified lower/upper bound evidence"
        ) from error
    notes = tuple(getattr(value, "notes", ()))
    assumptions = tuple(getattr(value, "assumptions", ()))
    certificate = getattr(value, "solver_result", None)
    if certificate is None:
        certificate = getattr(value, "certificate", None)
    requested = source_requested if requested_support is None else requested_support
    evaluated = source_evaluated if requested_support is None else requested_support
    if requested_support is not None and source_evaluated != source_requested:
        notes = (
            *notes,
            f"delegate evaluated compressed support {source_evaluated!r}",
        )
    return CapacityEvidence(
        lower_bound=lower,
        upper_bound=upper,
        requested_support=requested,
        evaluated_support=evaluated,
        method=method,
        certificate=certificate if certificate is not None else value,
        notes=notes,
        assumptions=assumptions,
    )


def zero_capacity_evidence[SupportT](
    support: SupportT,
    *,
    method: str = "empty-attack",
) -> CapacityEvidence[SupportT]:
    """Return canonical exact evidence for an empty attack."""

    zero = LogCardinality.zero()
    return CapacityEvidence(
        lower_bound=zero,
        upper_bound=zero,
        requested_support=support,
        evaluated_support=support,
        method=method,
    )


@dataclass(frozen=True, slots=True)
class MappedCapacityOracle[SupportT, MappedT]:
    """Map analysis attack atoms into another certified oracle's support."""

    delegate: CapacityOracle[MappedT]
    mapper: Callable[[frozenset[SupportT]], Iterable[MappedT]]
    method_prefix: str = "mapped"
    assumptions: tuple[str, ...] = ()

    def evaluate(
        self,
        attack_support: frozenset[SupportT],
    ) -> CapacityEvidence[frozenset[SupportT]]:
        mapped = frozenset(self.mapper(attack_support))
        external = coerce_capacity_evidence(self.delegate.evaluate(mapped))
        notes = (
            *external.notes,
            f"delegate requested support {external.requested_support!r}",
        )
        return CapacityEvidence(
            lower_bound=external.lower_bound,
            upper_bound=external.upper_bound,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method=f"{self.method_prefix}:{external.method}",
            certificate=external,
            notes=notes,
            assumptions=(*self.assumptions, *external.assumptions),
        )


@dataclass(frozen=True, slots=True)
class VerificationUnitCapacityOracle:
    """Adapt an address oracle to verification-unit attack IDs."""

    delegate: CapacityOracle[Position]
    index: Index

    def evaluate(
        self,
        attack_support: frozenset[int],
    ) -> CapacityEvidence[frozenset[int]]:
        addresses: set[Position] = set()
        for unit_index in attack_support:
            if type(unit_index) is not int or not (
                0 <= unit_index < self.index.verification_unit_count
            ):
                raise ValueError("capacity attack names an unknown verification unit")
            addresses.update(
                Position(address)
                for address in self.index.verification_unit(unit_index).interval
            )
        external = coerce_capacity_evidence(self.delegate.evaluate(frozenset(addresses)))
        return CapacityEvidence(
            lower_bound=external.lower_bound,
            upper_bound=external.upper_bound,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method=f"verification-unit-to-address:{external.method}",
            certificate=external,
            notes=external.notes,
            assumptions=external.assumptions,
        )


def _position_mapper(
    gate_id_for_position: Mapping[int, str] | Callable[[int], str],
) -> Callable[[int], str]:
    if isinstance(gate_id_for_position, Mapping):

        def from_mapping(position: int) -> str:
            try:
                return gate_id_for_position[position]
            except KeyError as error:
                raise ValueError(
                    f"no structural gate ID is mapped for position {position}"
                ) from error

        return from_mapping
    if not callable(gate_id_for_position):
        raise TypeError("gate_id_for_position must be a mapping or callable")
    return gate_id_for_position


@dataclass(frozen=True, slots=True, init=False)
class ExplicitCircuitCapacityOracleAdapter:
    """Adapt the existing string-gate explicit oracle to core positions."""

    delegate: CutExplicitCircuitCapacityOracle
    gate_id_for_position: Callable[[int], str]

    def __init__(
        self,
        delegate: CutExplicitCircuitCapacityOracle,
        gate_id_for_position: Mapping[int, str] | Callable[[int], str],
    ) -> None:
        object.__setattr__(self, "delegate", delegate)
        object.__setattr__(
            self,
            "gate_id_for_position",
            _position_mapper(gate_id_for_position),
        )

    def evaluate(
        self,
        attack_support: frozenset[Position],
    ) -> CapacityEvidence[frozenset[Position]]:
        mapped = frozenset(
            self.gate_id_for_position(int(position)) for position in attack_support
        )
        external = coerce_capacity_evidence(self.delegate.evaluate(mapped))
        return CapacityEvidence(
            lower_bound=external.lower_bound,
            upper_bound=external.upper_bound,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method=f"explicit-circuit-position-adapter:{external.method}",
            certificate=external,
            notes=external.notes,
            assumptions=external.assumptions,
        )


@dataclass(frozen=True, slots=True, init=False)
class CircuitCapacityOracle:
    """Build an exact explicit cut oracle from a finite :class:`Circuit`.

    Scans every address (``O(n)``), so this is for small circuits.  Every
    value has ``2**width`` possible values; inputs cannot lie on a downstream
    path from an attacked gate, so their width does not affect the queries.
    """

    circuit: Circuit
    delegate: CutExplicitCircuitCapacityOracle

    def __init__(self, circuit: Circuit) -> None:
        gates = []
        edges: set[tuple[str, str]] = set()
        for address in range(circuit.n):
            ref = circuit[address]
            gates.append(
                Gate(_gate_id(address), GateCapacity.values(1 << ref.width), op=ref.op)
            )
            edges.update((_gate_id(arg), _gate_id(address)) for arg in ref.args)
        outputs = {_gate_id(address) for address in circuit.outputs}
        if not outputs:
            raise ValueError("the circuit must designate an output")
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(
            self,
            "delegate",
            CutExplicitCircuitCapacityOracle(CircuitDAG(gates, edges, outputs)),
        )

    def evaluate(
        self,
        attack_support: frozenset[Position],
    ) -> CapacityEvidence[frozenset[Position]]:
        mapped: set[str] = set()
        for address in attack_support:
            if not 0 <= address < self.circuit.n or self.circuit[address].is_input:
                raise ValueError(f"capacity attack names non-gate address {int(address)}")
            mapped.add(_gate_id(address))
        external = coerce_capacity_evidence(self.delegate.evaluate(frozenset(mapped)))
        return CapacityEvidence(
            lower_bound=external.lower_bound,
            upper_bound=external.upper_bound,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method=f"circuit:{external.method}",
            certificate=external,
            notes=external.notes,
            assumptions=external.assumptions,
        )


def _gate_id(address: int) -> str:
    return f"address/{int(address)}"


__all__ = [
    "CapacityEvaluation",
    "CapacityEvidence",
    "CapacityOracle",
    "CircuitCapacityOracle",
    "ExplicitCircuitCapacityOracleAdapter",
    "MappedCapacityOracle",
    "VerificationUnitCapacityOracle",
    "coerce_capacity_evidence",
    "zero_capacity_evidence",
]
