"""Generic finite replay-boundary and commitment-ownership derivation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from veritor.commitment import CommitmentOwner
from veritor.core import (
    ExecutableCircuit,
    ExplicitIndexedDomain,
    InvalidArtifact,
    Position,
    ReplayPartition,
    StructuralCircuit,
    VerificationLimits,
    VerificationUnit,
    domains_equal,
    iter_domain,
    validate_circuit_contract,
)
from veritor.core import position as _position


@dataclass(frozen=True, slots=True)
class CommitmentLayout:
    """Disjoint complete boundary/interior ownership over all value positions."""

    all_positions: ExplicitIndexedDomain[Position]
    boundary: ExplicitIndexedDomain[Position]
    interiors: tuple[ExplicitIndexedDomain[Position], ...]
    _owner_by_position: Mapping[Position, CommitmentOwner] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.all_positions, ExplicitIndexedDomain)
            or not isinstance(self.boundary, ExplicitIndexedDomain)
            or type(self.interiors) is not tuple
            or any(
                not isinstance(interior, ExplicitIndexedDomain)
                for interior in self.interiors
            )
        ):
            raise InvalidArtifact("commitment layout domains have the wrong type")
        owners: dict[Position, CommitmentOwner] = {}
        for position in self.boundary.items:
            owners[position] = CommitmentOwner.boundary()
        for index, interior in enumerate(self.interiors):
            owner = CommitmentOwner.replay_unit(index)
            for position in interior.items:
                if position in owners:
                    raise InvalidArtifact("commitment layout owners are not disjoint")
                owners[position] = owner
        if (
            tuple(item for item in self.all_positions.items if item in owners)
            != (self.all_positions.items)
            or len(owners) != self.all_positions.count
        ):
            raise InvalidArtifact(
                "commitment layout owners are not a complete exact cover"
            )
        object.__setattr__(
            self,
            "_owner_by_position",
            MappingProxyType(owners),
        )

    def owner_of(self, global_position: int) -> CommitmentOwner:
        if type(global_position) is not int or global_position < 0:
            raise KeyError(global_position)
        try:
            return self._owner_by_position[Position(global_position)]
        except (KeyError, TypeError) as error:
            raise KeyError(global_position) from error

    def positions_for(
        self,
        owner: CommitmentOwner,
    ) -> ExplicitIndexedDomain[Position]:
        if owner == CommitmentOwner.boundary():
            return self.boundary
        if owner.replay_unit_index is None or owner.replay_unit_index >= len(
            self.interiors
        ):
            raise KeyError(owner)
        return self.interiors[owner.replay_unit_index]

    @property
    def position_count(self) -> int:
        return self.all_positions.count

    @property
    def replay_unit_count(self) -> int:
        return len(self.interiors)


def _preflight(
    circuit: StructuralCircuit,
    replay_partition: ReplayPartition,
    limits: VerificationLimits,
) -> tuple[
    tuple[Position, ...],
    tuple[Position, ...],
    dict[Position, int],
]:
    try:
        validate_circuit_contract(circuit, exhaustive=False)
    except (AttributeError, TypeError) as error:
        raise InvalidArtifact("circuit does not satisfy StructuralCircuit") from error
    if not isinstance(replay_partition, ReplayPartition):
        raise InvalidArtifact("replay partition has the wrong type")
    if replay_partition.structure_identity != circuit.identity:
        raise InvalidArtifact("replay partition belongs to another structure")
    computed_count = circuit.computed_positions.count
    input_count = len(circuit.input_ports)
    if type(computed_count) is not int or computed_count < 0:
        raise InvalidArtifact("only finite indexed circuits are supported")
    limits.enforce("max_positions", input_count + computed_count)
    limits.enforce("max_units", replay_partition.unit_count)
    limits.enforce(
        "max_positions",
        replay_partition.eligible_positions.count,
    )
    for unit in replay_partition.units:
        limits.enforce("max_positions_per_unit", unit.count)
    if not domains_equal(
        replay_partition.eligible_positions,
        circuit.computed_positions,
    ):
        raise InvalidArtifact(
            "replay partition does not cover the circuit computed positions"
        )

    inputs = tuple(port.position for port in circuit.input_ports)
    computed = tuple(iter_domain(circuit.computed_positions))
    if len(set(inputs)) != len(inputs):
        raise InvalidArtifact("input positions must be unique")
    if set(inputs) & set(computed):
        raise InvalidArtifact("input and computed positions must be disjoint")
    replay_partition.validate()
    replay_owners: dict[Position, int] = {}
    for unit in replay_partition.units:
        for member in iter_domain(unit.members):
            if member in replay_owners:
                raise InvalidArtifact(
                    "replay partition assigns a position more than once"
                )
            replay_owners[member] = int(unit.index)
    if len(replay_owners) != len(computed):
        raise InvalidArtifact("replay partition does not exactly cover the circuit")
    return inputs, computed, replay_owners


def derive_commitment_ownership(
    circuit: StructuralCircuit,
    replay_partition: ReplayPartition,
    limits: VerificationLimits | None = None,
) -> CommitmentLayout:
    """Derive B and every replay interior from only trusted ``(C, R)``.

    Boundary order is global circuit position order (declared inputs followed
    by computed-domain rank order), not numeric sorting.  This supports
    arbitrary verifier-derived position domains.
    """

    checked_limits = VerificationLimits() if limits is None else limits
    if not isinstance(checked_limits, VerificationLimits):
        raise TypeError("limits must be VerificationLimits")
    inputs, computed, replay_owners = _preflight(
        circuit,
        replay_partition,
        checked_limits,
    )
    all_positions = inputs + computed
    known = set(all_positions)
    input_positions = set(inputs)
    boundary_members: set[Position] = set(inputs)
    boundary_members.update(port.position for port in circuit.output_ports)

    for expected_position in computed:
        gate = circuit.gate_at(expected_position)
        if gate.position != expected_position:
            raise InvalidArtifact("gate_at returned a gate for another position")
        try:
            write_owner = replay_owners[expected_position]
        except KeyError as error:
            raise InvalidArtifact("computed position has no replay owner") from error
        for predecessor in gate.predecessors:
            if predecessor not in known:
                raise InvalidArtifact(
                    f"gate {expected_position} references unknown position "
                    f"{predecessor}"
                )
            if (
                predecessor in input_positions
                or replay_owners[predecessor] != write_owner
            ):
                boundary_members.add(predecessor)

    boundary_tuple = tuple(
        position for position in all_positions if position in boundary_members
    )
    interior_lists: list[list[Position]] = [
        [] for _ in range(replay_partition.unit_count)
    ]
    for global_position in computed:
        if global_position not in boundary_members:
            interior_lists[replay_owners[global_position]].append(global_position)

    interiors = tuple(ExplicitIndexedDomain(items) for items in interior_lists)
    boundary = ExplicitIndexedDomain(boundary_tuple)
    all_domain = ExplicitIndexedDomain(all_positions)

    covered = set(boundary.items)
    for interior in interiors:
        overlap = covered.intersection(interior.items)
        if overlap:
            raise InvalidArtifact(
                "derived boundary and replay interiors are not disjoint"
            )
        covered.update(interior.items)
    if covered != set(all_positions):
        raise InvalidArtifact(
            "derived boundary and interiors do not cover all positions"
        )
    return CommitmentLayout(all_domain, boundary, interiors)


derive_commitment_layout = derive_commitment_ownership


def derive_replay_boundary(
    circuit: StructuralCircuit,
    replay_partition: ReplayPartition,
    limits: VerificationLimits | None = None,
) -> tuple[Position, ...]:
    """Return the generic verifier-derived replay boundary."""

    return derive_commitment_ownership(
        circuit,
        replay_partition,
        limits,
    ).boundary.items


def value_schema_for_position(
    circuit: StructuralCircuit,
    global_position: int,
) -> str:
    """Resolve the trusted value schema declared for one position."""

    checked_position = _position(global_position)
    for port in circuit.input_ports:
        if port.position == checked_position:
            return str(port.value_type)
    if not circuit.computed_positions.contains(checked_position):
        raise KeyError(global_position)
    if isinstance(circuit, ExecutableCircuit):
        gate = circuit.executable_gate_at(checked_position)
        if gate.position != checked_position:
            raise InvalidArtifact(
                "executable_gate_at returned a gate for another position"
            )
        return str(gate.output_type)
    structural = circuit.gate_at(checked_position)
    if structural.value_type is None:
        raise InvalidArtifact(
            f"position {global_position} has no declared value schema"
        )
    return str(structural.value_type)


def validate_output_schemas(circuit: StructuralCircuit) -> None:
    """Require every output view to use its position's canonical schema."""

    for port in circuit.output_ports:
        if str(port.value_type) != value_schema_for_position(
            circuit,
            port.position,
        ):
            raise InvalidArtifact(
                f"output {port.name!r} declares a schema different from "
                "its referenced position"
            )


def required_positions_for_verification_unit(
    circuit: StructuralCircuit,
    unit: VerificationUnit,
    layout: CommitmentLayout,
) -> tuple[Position, ...]:
    """Return exactly the values a transparent local check must authenticate."""

    if not isinstance(unit, VerificationUnit):
        raise InvalidArtifact("verification unit has the wrong type")
    required: set[Position] = set()
    for member in iter_domain(unit.members):
        if not circuit.computed_positions.contains(member):
            raise InvalidArtifact("verification unit contains a non-gate position")
        gate = circuit.gate_at(member)
        if gate.position != member:
            raise InvalidArtifact("gate lookup returned the wrong position")
        required.add(member)
        required.update(gate.predecessors)

    try:
        ordered = tuple(sorted(required, key=layout.all_positions.rank))
    except KeyError as error:
        raise InvalidArtifact(
            "verification unit requires a position outside the circuit layout"
        ) from error
    if len(ordered) != len(required):
        raise InvalidArtifact(
            "verification unit requires a position outside the circuit layout"
        )
    expected_owner = CommitmentOwner.replay_unit(int(unit.replay_unit))
    for item in ordered:
        owner = layout.owner_of(item)
        if owner != CommitmentOwner.boundary() and owner != expected_owner:
            raise InvalidArtifact(
                "replay boundary omitted a cross-unit verification dependency"
            )
    return ordered
