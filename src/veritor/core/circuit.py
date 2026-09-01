"""Semantics-independent structural and executable circuit contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .capabilities import ArtifactKind
from .errors import InvalidArtifact
from .identity import JSONScalar, StructureIdentity
from .ids import (
    OperationId,
    Position,
    RelationId,
    ValueTypeId,
    nonempty_identifier,
)
from .ids import position as _as_position
from .indexed import IndexedDomain, iter_domain, position_domain


def _metadata(
    value: Mapping[str, JSONScalar] | Iterable[tuple[str, JSONScalar]],
) -> tuple[tuple[str, JSONScalar], ...]:
    pairs = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    result: dict[str, JSONScalar] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise InvalidArtifact("metadata entries must be (key, value) pairs")
        key, item = pair
        if type(key) is not str or not key:
            raise InvalidArtifact("metadata keys must be nonempty strings")
        if type(item) not in (type(None), bool, int, str):
            raise InvalidArtifact("metadata values must be canonical JSON scalars")
        if key in result:
            raise InvalidArtifact(f"duplicate metadata key {key!r}")
        result[key] = item
    return tuple(sorted(result.items()))


@dataclass(frozen=True, slots=True, init=False)
class Port:
    """A named typed circuit input or output port."""

    name: str
    position: Position
    value_type: ValueTypeId

    def __init__(self, name: str, position: int, value_type: str) -> None:
        object.__setattr__(
            self, "name", nonempty_identifier(name, field_name="port name")
        )
        object.__setattr__(
            self,
            "position",
            _as_position(position, field_name=f"port {name!r} position"),
        )
        object.__setattr__(
            self,
            "value_type",
            ValueTypeId(nonempty_identifier(value_type, field_name="value_type")),
        )


@dataclass(frozen=True, slots=True, init=False)
class StructuralGate:
    """Static dependency and value-domain metadata for one computed position.

    ``predecessors`` is a tuple so adapters may retain operand order and
    repeated reads even when a downstream structural algorithm treats it as a
    set. ``capacity_upper_bound`` is an exact finite value-cardinality bound.
    """

    position: Position
    operation: OperationId
    predecessors: tuple[Position, ...]
    capacity_upper_bound: int | None
    value_type: ValueTypeId | None
    metadata: tuple[tuple[str, JSONScalar], ...]

    def __init__(
        self,
        position: int,
        operation: str,
        predecessors: Iterable[int],
        capacity_upper_bound: int | None = None,
        *,
        value_type: str | None = None,
        metadata: Mapping[str, JSONScalar] | Iterable[tuple[str, JSONScalar]] = (),
    ) -> None:
        checked_position = _as_position(position)
        checked_operation = OperationId(
            nonempty_identifier(operation, field_name="operation")
        )
        checked_predecessors = tuple(
            _as_position(item, field_name=f"predecessors[{index}]")
            for index, item in enumerate(predecessors)
        )
        if capacity_upper_bound is not None and (
            type(capacity_upper_bound) is not int or capacity_upper_bound < 1
        ):
            raise InvalidArtifact(
                "capacity_upper_bound must be a positive integer or None"
            )
        checked_value_type = (
            None
            if value_type is None
            else ValueTypeId(nonempty_identifier(value_type, field_name="value_type"))
        )
        object.__setattr__(self, "position", checked_position)
        object.__setattr__(self, "operation", checked_operation)
        object.__setattr__(self, "predecessors", checked_predecessors)
        object.__setattr__(self, "capacity_upper_bound", capacity_upper_bound)
        object.__setattr__(self, "value_type", checked_value_type)
        object.__setattr__(self, "metadata", _metadata(metadata))

    @property
    def value_cardinality_upper_bound(self) -> int | None:
        """Descriptive alias for ``capacity_upper_bound``."""

        return self.capacity_upper_bound


@dataclass(frozen=True, slots=True, init=False)
class ExecutableGate:
    """One trusted local relation with ordered, duplicate-preserving arguments."""

    position: Position
    operation: OperationId
    arguments: tuple[Position, ...]
    output_type: ValueTypeId
    relation_id: RelationId
    metadata: tuple[tuple[str, JSONScalar], ...]

    def __init__(
        self,
        position: int,
        operation: str,
        arguments: Iterable[int],
        output_type: str,
        relation_id: str,
        *,
        metadata: Mapping[str, JSONScalar] | Iterable[tuple[str, JSONScalar]] = (),
    ) -> None:
        object.__setattr__(self, "position", _as_position(position))
        object.__setattr__(
            self,
            "operation",
            OperationId(nonempty_identifier(operation, field_name="operation")),
        )
        object.__setattr__(
            self,
            "arguments",
            tuple(
                _as_position(item, field_name=f"arguments[{index}]")
                for index, item in enumerate(arguments)
            ),
        )
        object.__setattr__(
            self,
            "output_type",
            ValueTypeId(nonempty_identifier(output_type, field_name="output_type")),
        )
        object.__setattr__(
            self,
            "relation_id",
            RelationId(nonempty_identifier(relation_id, field_name="relation_id")),
        )
        object.__setattr__(self, "metadata", _metadata(metadata))

    @property
    def predecessors(self) -> tuple[Position, ...]:
        return self.arguments


@runtime_checkable
class StructuralCircuit(Protocol):
    """A finite or lazily indexed circuit sufficient for structural analysis."""

    @property
    def identity(self) -> StructureIdentity: ...

    @property
    def computed_positions(self) -> IndexedDomain[Position]: ...

    @property
    def input_ports(self) -> tuple[Port, ...]: ...

    @property
    def output_ports(self) -> tuple[Port, ...]: ...

    def gate_at(self, position: Position) -> StructuralGate: ...


@runtime_checkable
class ExecutableCircuit(StructuralCircuit, Protocol):
    """A structural circuit that also exposes trusted local relations."""

    def executable_gate_at(self, position: Position) -> ExecutableGate: ...


CircuitAccess = StructuralCircuit
StructuralCircuitAccess = StructuralCircuit
ExecutableCircuitAccess = ExecutableCircuit


def ordered_output_positions(circuit: StructuralCircuit) -> tuple[Position, ...]:
    """Return outputs exactly as declared, retaining order and duplicates."""

    if type(circuit.output_ports) is not tuple:
        raise InvalidArtifact("circuit output_ports must be a tuple")
    return tuple(port.position for port in circuit.output_ports)


def validate_circuit_contract(
    circuit: StructuralCircuit,
    *,
    exhaustive: bool = False,
) -> None:
    """Validate representation-independent circuit invariants.

    Exhaustive gate lookup is opt-in so this function remains usable with very
    large lazy circuits.
    """

    if not isinstance(circuit.identity, StructureIdentity):
        raise InvalidArtifact("circuit identity must be a StructureIdentity")
    if circuit.identity.artifact_kind is ArtifactKind.CAPACITY_PROFILE:
        raise InvalidArtifact("a capacity profile is not a structural circuit")
    computed = position_domain(
        circuit.computed_positions,
        field_name="circuit computed_positions",
    )
    if type(circuit.input_ports) is not tuple:
        raise InvalidArtifact("circuit input_ports must be a tuple")
    if type(circuit.output_ports) is not tuple:
        raise InvalidArtifact("circuit output_ports must be a tuple")
    inputs: set[Position] = set()
    for port in circuit.input_ports:
        if not isinstance(port, Port):
            raise InvalidArtifact("every input port must be a Port")
        if port.position in inputs:
            raise InvalidArtifact("input positions must be unique")
        if computed.contains(port.position):
            raise InvalidArtifact("input and computed positions must be disjoint")
        inputs.add(port.position)
    for port in circuit.output_ports:
        if not isinstance(port, Port):
            raise InvalidArtifact("every output port must be a Port")
        if port.position not in inputs and not computed.contains(port.position):
            raise InvalidArtifact("an output port references an unknown position")
    if not exhaustive:
        return
    known = inputs | set(iter_domain(computed))
    for expected_position in iter_domain(computed):
        gate = circuit.gate_at(expected_position)
        if not isinstance(gate, StructuralGate):
            raise InvalidArtifact("gate_at must return StructuralGate values")
        if gate.position != expected_position:
            raise InvalidArtifact("gate_at returned a gate for another position")
        if any(predecessor not in known for predecessor in gate.predecessors):
            raise InvalidArtifact("a structural gate references an unknown position")
