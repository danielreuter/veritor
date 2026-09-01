"""The compiled artifact: a circuit, its two partitions, and the replay boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .circuit import (
    ExecutableCircuit,
    StructuralCircuit,
    validate_circuit_contract,
)
from .errors import InvalidArtifact
from .identity import CompiledResultIdentity, PartitionKind
from .ids import Position, UnitIndex
from .indexed import (
    IndexedDomain,
    IntervalDomain,
    domains_equal,
    iter_domain,
    position_domain,
)
from .partitions import (
    ReplayPartition,
    VerificationPartition,
    contiguous_span,
)

BOUNDARY_OWNER = -1
"""Owner index reported by :meth:`CompiledArtifact.value_owner` for boundary positions."""


@dataclass(frozen=True, slots=True, init=False)
class CompiledArtifact:
    """``Compile``'s output: ``(C, S_replay, S_verif)`` plus the replay boundary.

    The boundary is the set of positions whose values are committed before any
    replay unit is selected: every input, every output, and every position read
    by a gate in a different replay unit.  It is produced by the trusted
    compiler, bound into ``identity``, and therefore never rederived by the
    verifier.  Everything the protocol needs about ownership of a position is
    answered here in ``O(log)`` time.
    """

    circuit: StructuralCircuit
    replay: ReplayPartition
    verification: VerificationPartition
    boundary: IndexedDomain[Position]
    identity: CompiledResultIdentity

    def __init__(
        self,
        circuit: StructuralCircuit,
        replay: ReplayPartition,
        verification: VerificationPartition,
        boundary: IndexedDomain[Position] | Iterable[int],
    ) -> None:
        if not isinstance(replay, ReplayPartition):
            raise InvalidArtifact("compiled replay value is not a ReplayPartition")
        if not isinstance(verification, VerificationPartition):
            raise InvalidArtifact(
                "compiled verification value is not a VerificationPartition"
            )
        try:
            validate_circuit_contract(circuit, exhaustive=False)
        except (AttributeError, TypeError) as error:
            raise InvalidArtifact(
                "compiled circuit does not satisfy the circuit contract"
            ) from error
        structure = circuit.identity
        for partition, label in ((replay, "replay"), (verification, "verification")):
            if partition.structure_identity != structure:
                raise InvalidArtifact(f"{label} partition belongs to another structure")
            if not domains_equal(
                circuit.computed_positions, partition.eligible_positions
            ):
                raise InvalidArtifact(
                    f"{label} partition does not cover the circuit's computed positions"
                )
        if verification.replay_partition_identity != replay.identity:
            raise InvalidArtifact(
                "verification partition identifies a different replay partition"
            )
        if replay.identity.partition_kind is not PartitionKind.REPLAY:
            raise InvalidArtifact("replay identity has the wrong partition kind")
        if verification.identity.partition_kind is not PartitionKind.VERIFICATION:
            raise InvalidArtifact("verification identity has the wrong partition kind")
        checked_boundary = position_domain(boundary, field_name="replay boundary")
        inputs = {port.position for port in circuit.input_ports}
        computed = circuit.computed_positions
        for port in circuit.input_ports + circuit.output_ports:
            if not checked_boundary.contains(port.position):
                raise InvalidArtifact(
                    f"replay boundary omits port position {port.position}"
                )
        probes: Iterable[int]
        if isinstance(checked_boundary, IntervalDomain):
            probes = (
                item
                for start, stop in checked_boundary.intervals
                for item in (start, stop - 1)
            )
        else:
            probes = iter_domain(checked_boundary)
        for item in probes:
            if item not in inputs and not computed.contains(item):
                raise InvalidArtifact(
                    f"replay boundary contains unknown position {item}"
                )
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "replay", replay)
        object.__setattr__(self, "verification", verification)
        object.__setattr__(self, "boundary", checked_boundary)
        object.__setattr__(
            self,
            "identity",
            CompiledResultIdentity(
                schema_version=structure.schema_version,
                structure_digest=structure.digest,
                replay_partition_digest=replay.identity.digest,
                verification_partition_digest=verification.identity.digest,
                boundary_digest=checked_boundary.identity_digest,
            ),
        )

    @property
    def executable(self) -> bool:
        return isinstance(self.circuit, ExecutableCircuit)

    def interior(self, unit: int) -> IndexedDomain[Position]:
        """Positions of replay unit ``unit`` that are not boundary positions."""

        members = self.replay.unit_at(unit).members
        span = contiguous_span(members)
        if span is not None and isinstance(self.boundary, IntervalDomain):
            return self.boundary.complement_within(*span)
        return IntervalDomain.from_positions(
            item for item in iter_domain(members) if not self.boundary.contains(item)
        )

    def value_owner(self, position: int) -> int:
        """Return :data:`BOUNDARY_OWNER` for boundary positions, else the replay unit."""

        if self.boundary.contains(position):
            return BOUNDARY_OWNER
        return self.replay.owner_of(position)


def derive_replay_boundary(
    circuit: StructuralCircuit,
    replay: ReplayPartition,
) -> IntervalDomain:
    """Compute the boundary by scanning every gate (``O(n)``; reference only).

    Compilers with structural knowledge (the call-DAG kernel) derive the same
    set without touching individual gates.  This function defines what they
    must agree with.
    """

    positions: set[int] = {port.position for port in circuit.input_ports}
    positions.update(port.position for port in circuit.output_ports)
    for item in iter_domain(circuit.computed_positions):
        owner: UnitIndex = replay.owner_of(item)
        for predecessor in circuit.gate_at(item).predecessors:
            if (
                not replay.eligible_positions.contains(predecessor)
                or replay.owner_of(predecessor) != owner
            ):
                positions.add(predecessor)
    return IntervalDomain.from_positions(positions)


def validate_replay_boundary(artifact: CompiledArtifact) -> None:
    """Exhaustively check that ``artifact.boundary`` is the true replay boundary."""

    expected = derive_replay_boundary(artifact.circuit, artifact.replay)
    if set(expected) != set(iter_domain(artifact.boundary)):
        raise InvalidArtifact("compiled artifact carries an incorrect replay boundary")
