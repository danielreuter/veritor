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
from veritor.core.circuit import (
    StructuralCircuit,
    validate_circuit_contract,
)
from veritor.core.ids import Position
from veritor.core.indexed import iter_domain
from veritor.core.partitions import VerificationPartition


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
    """Adapt a position oracle to verification-unit attack IDs."""

    delegate: CapacityOracle[Position]
    verification_partition: VerificationPartition

    def evaluate(
        self,
        attack_support: frozenset[int],
    ) -> CapacityEvidence[frozenset[int]]:
        positions: set[Position] = set()
        for unit_index in attack_support:
            if type(unit_index) is not int or not (
                0 <= unit_index < self.verification_partition.unit_count
            ):
                raise ValueError("capacity attack names an unknown verification unit")
            unit = self.verification_partition.unit_at(unit_index)
            positions.update(iter_domain(unit.members))
        external = coerce_capacity_evidence(
            self.delegate.evaluate(frozenset(positions))
        )
        return CapacityEvidence(
            lower_bound=external.lower_bound,
            upper_bound=external.upper_bound,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method=f"verification-unit-to-position:{external.method}",
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
class StructuralCircuitCapacityOracle:
    """Build an exact explicit cut oracle from a finite core structural circuit.

    This practical adapter requires every computed position to expose a finite
    cardinality of at least two.  Fixed inputs receive a dummy positive width;
    they cannot lie on a downstream path starting at an attacked computed
    position, so that width does not affect these queries.
    """

    circuit: StructuralCircuit
    delegate: CutExplicitCircuitCapacityOracle
    gate_ids_by_position: Mapping[int, str]

    def __init__(self, circuit: StructuralCircuit) -> None:
        validate_circuit_contract(circuit, exhaustive=True)
        input_positions = {port.position for port in circuit.input_ports}
        computed_positions = tuple(iter_domain(circuit.computed_positions))
        all_positions = input_positions | set(computed_positions)
        ids = {int(position): f"position/{int(position)}" for position in all_positions}
        gates = [
            Gate(ids[int(position)], GateCapacity.values(2), op="input")
            for position in sorted(input_positions)
        ]
        for position in computed_positions:
            structural_gate = circuit.gate_at(position)
            cardinality = structural_gate.capacity_upper_bound
            if cardinality is None:
                raise ValueError(
                    f"position {int(position)} has no finite capacity upper bound"
                )
            if cardinality < 2:
                raise ValueError(
                    "the explicit cut adapter cannot represent zero-capacity gates"
                )
            gates.append(
                Gate(
                    ids[int(position)],
                    GateCapacity.values(cardinality),
                    op=str(structural_gate.operation),
                )
            )
        edges = {
            (ids[int(predecessor)], ids[int(position)])
            for position in computed_positions
            for predecessor in circuit.gate_at(position).predecessors
        }
        outputs = {
            ids[int(port.position)]
            for port in circuit.output_ports
        }
        if not outputs:
            raise ValueError("the structural circuit must designate an output")
        explicit = CircuitDAG(gates, edges, outputs)
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(
            self,
            "delegate",
            CutExplicitCircuitCapacityOracle(explicit),
        )
        object.__setattr__(self, "gate_ids_by_position", ids)

    def evaluate(
        self,
        attack_support: frozenset[Position],
    ) -> CapacityEvidence[frozenset[Position]]:
        mapped: set[str] = set()
        for position in attack_support:
            if not self.circuit.computed_positions.contains(position):
                raise ValueError(
                    f"capacity attack names non-computed position {int(position)}"
                )
            mapped.add(self.gate_ids_by_position[int(position)])
        external = coerce_capacity_evidence(
            self.delegate.evaluate(frozenset(mapped))
        )
        return CapacityEvidence(
            lower_bound=external.lower_bound,
            upper_bound=external.upper_bound,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method=f"core-structural-circuit:{external.method}",
            certificate=external,
            notes=external.notes,
            assumptions=external.assumptions,
        )


__all__ = [
    "CapacityEvaluation",
    "CapacityEvidence",
    "CapacityOracle",
    "ExplicitCircuitCapacityOracleAdapter",
    "MappedCapacityOracle",
    "StructuralCircuitCapacityOracle",
    "VerificationUnitCapacityOracle",
    "coerce_capacity_evidence",
    "zero_capacity_evidence",
]
