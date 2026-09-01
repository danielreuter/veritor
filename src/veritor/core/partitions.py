"""Concrete finite replay and verification partitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .errors import InvalidArtifact
from .identity import (
    Digest,
    JSONValue,
    PartitionIdentity,
    PartitionKind,
    StructureIdentity,
    identity_digest,
)
from .ids import (
    Position,
    RelationId,
    UnitIndex,
    nonempty_identifier,
    unit_index,
)
from .ids import position as _as_position
from .indexed import (
    ExplicitIndexedDomain,
    IndexedDomain,
    domains_equal,
    iter_domain,
    position_domain,
)


def _relation(value: str | None, field_name: str) -> RelationId | None:
    if value is None:
        return None
    return RelationId(nonempty_identifier(value, field_name=field_name))


@dataclass(frozen=True, slots=True, init=False)
class ReplayUnit:
    """One nonempty first-stage unit of eligible computed positions."""

    index: UnitIndex
    members: IndexedDomain[Position]
    replay_cost: int
    replay_relation_id: RelationId | None
    identity_digest: Digest = field(init=False)

    def __init__(
        self,
        index: int,
        members: IndexedDomain[Position] | Iterable[int],
        replay_cost: int = 0,
        replay_relation_id: str | None = None,
    ) -> None:
        checked_index = unit_index(index, field_name="replay unit index")
        checked_members = position_domain(
            members, field_name=f"replay unit {index} members"
        )
        if checked_members.count == 0:
            raise InvalidArtifact("replay units must not be empty")
        if type(replay_cost) is not int or replay_cost < 0:
            raise InvalidArtifact("replay_cost must be a nonnegative integer")
        checked_relation = _relation(replay_relation_id, "replay_relation_id")
        object.__setattr__(self, "index", checked_index)
        object.__setattr__(self, "members", checked_members)
        object.__setattr__(self, "replay_cost", replay_cost)
        object.__setattr__(self, "replay_relation_id", checked_relation)
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/replay-unit/v1",
                {
                    "index": checked_index,
                    "members_digest": checked_members.identity_digest,
                    "replay_cost": replay_cost,
                    "replay_relation_id": checked_relation,
                },
            ),
        )

    @property
    def count(self) -> int:
        return self.members.count

    @property
    def identity(self) -> Digest:
        return self.identity_digest

    @property
    def unit_id(self) -> UnitIndex:
        return self.index


@dataclass(frozen=True, slots=True, init=False)
class VerificationUnit:
    """One nonempty second-stage unit contained in one replay unit."""

    index: UnitIndex
    replay_unit: UnitIndex
    members: IndexedDomain[Position]
    proof_relation_id: RelationId | None
    identity_digest: Digest = field(init=False)

    def __init__(
        self,
        index: int,
        replay_unit: int,
        members: IndexedDomain[Position] | Iterable[int],
        proof_relation_id: str | None = None,
    ) -> None:
        checked_index = unit_index(index, field_name="verification unit index")
        checked_replay = unit_index(
            replay_unit, field_name="verification unit replay owner"
        )
        checked_members = position_domain(
            members, field_name=f"verification unit {index} members"
        )
        if checked_members.count == 0:
            raise InvalidArtifact("verification units must not be empty")
        checked_relation = _relation(proof_relation_id, "proof_relation_id")
        object.__setattr__(self, "index", checked_index)
        object.__setattr__(self, "replay_unit", checked_replay)
        object.__setattr__(self, "members", checked_members)
        object.__setattr__(self, "proof_relation_id", checked_relation)
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/verification-unit/v1",
                {
                    "index": checked_index,
                    "members_digest": checked_members.identity_digest,
                    "proof_relation_id": checked_relation,
                    "replay_unit": checked_replay,
                },
            ),
        )

    @property
    def count(self) -> int:
        return self.members.count

    @property
    def identity(self) -> Digest:
        return self.identity_digest

    @property
    def unit_id(self) -> UnitIndex:
        return self.index

    @property
    def replay_unit_index(self) -> UnitIndex:
        return self.replay_unit


def _validate_unit_indices(units: tuple[ReplayUnit | VerificationUnit, ...]) -> None:
    for expected, unit in enumerate(units):
        if unit.index != expected:
            raise InvalidArtifact(
                "partition units must have contiguous indices matching tuple order"
            )


def _validate_exact_cover(
    eligible: IndexedDomain[Position],
    units: tuple[ReplayUnit | VerificationUnit, ...],
    *,
    label: str,
) -> None:
    if eligible.count == 0:
        if units:
            raise InvalidArtifact(f"{label} must have zero units for an empty domain")
        return
    if not units:
        raise InvalidArtifact(f"{label} does not cover any eligible positions")
    owners_by_rank: dict[int, UnitIndex] = {}
    for unit in units:
        if unit.members.count == 0:
            raise InvalidArtifact(f"{label} contains an empty unit")
        for member in iter_domain(unit.members):
            try:
                rank = eligible.rank(member)
            except KeyError as error:
                raise InvalidArtifact(
                    f"{label} unit {unit.index} contains ineligible position {member}"
                ) from error
            if not 0 <= rank < eligible.count or eligible.unrank(rank) != member:
                raise InvalidArtifact(f"{label} eligible domain violates rank/unrank")
            if rank in owners_by_rank:
                raise InvalidArtifact(
                    f"{label} assigns position {member} to multiple units"
                )
            owners_by_rank[rank] = unit.index
    if len(owners_by_rank) != eligible.count:
        raise InvalidArtifact(
            f"{label} covers {len(owners_by_rank)} of {eligible.count} positions"
        )


def _configuration_digest(
    kind: PartitionKind,
    configuration: JSONValue | None,
) -> Digest:
    return identity_digest(
        "veritor/partition-configuration/v1",
        {
            "configuration": {} if configuration is None else configuration,
            "partition_kind": kind.value,
        },
    )


def _replay_representation_manifest(
    eligible: IndexedDomain[Position],
    units: tuple[ReplayUnit, ...],
) -> dict[str, JSONValue]:
    return {
        "eligible_count": eligible.count,
        "eligible_digest": eligible.identity_digest,
        "units": [unit.identity_digest for unit in units],
    }


@dataclass(frozen=True, slots=True, init=False)
class ReplayPartition:
    """A validated exact finite partition of eligible computed positions."""

    structure_identity: StructureIdentity
    eligible_positions: IndexedDomain[Position]
    units: tuple[ReplayUnit, ...]
    identity: PartitionIdentity

    def __init__(
        self,
        structure_identity: StructureIdentity,
        eligible_positions: IndexedDomain[Position] | Iterable[int],
        units: Iterable[ReplayUnit],
        *,
        algorithm_id: str = "explicit",
        algorithm_version: str = "1",
        configuration: JSONValue | None = None,
        identity: PartitionIdentity | None = None,
    ) -> None:
        if not isinstance(structure_identity, StructureIdentity):
            raise InvalidArtifact(
                "replay partition structure_identity must be a StructureIdentity"
            )
        checked_eligible = position_domain(
            eligible_positions, field_name="replay eligible_positions"
        )
        checked_units = tuple(units)
        if any(not isinstance(unit, ReplayUnit) for unit in checked_units):
            raise InvalidArtifact("replay partition units must be ReplayUnit values")
        _validate_unit_indices(checked_units)
        _validate_exact_cover(checked_eligible, checked_units, label="replay partition")
        checked_algorithm = nonempty_identifier(algorithm_id, field_name="algorithm_id")
        checked_version = nonempty_identifier(
            algorithm_version, field_name="algorithm_version"
        )
        representation_digest = identity_digest(
            "veritor/replay-partition-representation/v1",
            _replay_representation_manifest(checked_eligible, checked_units),
        )
        expected_identity = PartitionIdentity(
            partition_kind=PartitionKind.REPLAY,
            structure_digest=structure_identity.digest,
            algorithm_id=checked_algorithm,
            algorithm_version=checked_version,
            configuration_digest=_configuration_digest(
                PartitionKind.REPLAY, configuration
            ),
            representation_digest=representation_digest,
        )
        if identity is not None and identity != expected_identity:
            raise InvalidArtifact(
                "provided replay partition identity does not match its contents"
            )
        object.__setattr__(self, "structure_identity", structure_identity)
        object.__setattr__(self, "eligible_positions", checked_eligible)
        object.__setattr__(self, "units", checked_units)
        object.__setattr__(self, "identity", expected_identity)

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def identity_digest(self) -> Digest:
        return self.identity.digest

    def unit_at(self, index: int) -> ReplayUnit:
        checked = unit_index(index)
        if checked >= self.unit_count:
            raise KeyError(index)
        return self.units[checked]

    def owner_of(self, position: int) -> UnitIndex:
        checked = _as_position(position, field_name="owner lookup position")
        if not self.eligible_positions.contains(checked):
            raise KeyError(position)
        owners = tuple(
            unit.index for unit in self.units if unit.members.contains(checked)
        )
        if len(owners) != 1:
            raise InvalidArtifact(
                "replay partition no longer has deterministic ownership"
            )
        return owners[0]

    def validate(self) -> None:
        """Recheck exact-cover and representation-integrity invariants."""

        _validate_unit_indices(self.units)
        _validate_exact_cover(
            self.eligible_positions, self.units, label="replay partition"
        )
        expected_representation = identity_digest(
            "veritor/replay-partition-representation/v1",
            _replay_representation_manifest(self.eligible_positions, self.units),
        )
        if (
            self.identity.partition_kind is not PartitionKind.REPLAY
            or self.identity.structure_digest != self.structure_identity.digest
            or self.identity.representation_digest != expected_representation
        ):
            raise InvalidArtifact("replay partition identity is inconsistent")


def _verification_representation_manifest(
    eligible: IndexedDomain[Position],
    replay_identity: PartitionIdentity,
    units: tuple[VerificationUnit, ...],
) -> dict[str, JSONValue]:
    return {
        "eligible_count": eligible.count,
        "eligible_digest": eligible.identity_digest,
        "replay_partition_digest": replay_identity.digest,
        "units": [unit.identity_digest for unit in units],
    }


@dataclass(frozen=True, slots=True, init=False)
class VerificationPartition:
    """A validated exact finite refinement of a replay partition."""

    structure_identity: StructureIdentity
    replay_partition_identity: PartitionIdentity
    eligible_positions: IndexedDomain[Position]
    units: tuple[VerificationUnit, ...]
    identity: PartitionIdentity
    replay_unit_count: int

    def __init__(
        self,
        structure_identity: StructureIdentity,
        replay_partition: ReplayPartition,
        eligible_positions: IndexedDomain[Position] | Iterable[int],
        units: Iterable[VerificationUnit],
        *,
        algorithm_id: str = "explicit",
        algorithm_version: str = "1",
        configuration: JSONValue | None = None,
        identity: PartitionIdentity | None = None,
    ) -> None:
        if not isinstance(structure_identity, StructureIdentity):
            raise InvalidArtifact(
                "verification structure_identity must be a StructureIdentity"
            )
        if not isinstance(replay_partition, ReplayPartition):
            raise InvalidArtifact("verification partition requires a ReplayPartition")
        checked_eligible = position_domain(
            eligible_positions, field_name="verification eligible_positions"
        )
        checked_units = tuple(units)
        if any(not isinstance(unit, VerificationUnit) for unit in checked_units):
            raise InvalidArtifact(
                "verification partition units must be VerificationUnit values"
            )
        _validate_unit_indices(checked_units)
        _validate_exact_cover(
            checked_eligible, checked_units, label="verification partition"
        )
        checked_algorithm = nonempty_identifier(algorithm_id, field_name="algorithm_id")
        checked_version = nonempty_identifier(
            algorithm_version, field_name="algorithm_version"
        )
        representation_digest = identity_digest(
            "veritor/verification-partition-representation/v1",
            _verification_representation_manifest(
                checked_eligible,
                replay_partition.identity,
                checked_units,
            ),
        )
        expected_identity = PartitionIdentity(
            partition_kind=PartitionKind.VERIFICATION,
            structure_digest=structure_identity.digest,
            algorithm_id=checked_algorithm,
            algorithm_version=checked_version,
            configuration_digest=_configuration_digest(
                PartitionKind.VERIFICATION, configuration
            ),
            representation_digest=representation_digest,
        )
        if identity is not None and identity != expected_identity:
            raise InvalidArtifact(
                "provided verification partition identity does not match its contents"
            )
        object.__setattr__(self, "structure_identity", structure_identity)
        object.__setattr__(self, "replay_partition_identity", replay_partition.identity)
        object.__setattr__(self, "eligible_positions", checked_eligible)
        object.__setattr__(self, "units", checked_units)
        object.__setattr__(self, "identity", expected_identity)
        object.__setattr__(self, "replay_unit_count", replay_partition.unit_count)
        validate_verification_refines_replay(replay_partition, self)

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def identity_digest(self) -> Digest:
        return self.identity.digest

    def unit_at(self, index: int) -> VerificationUnit:
        checked = unit_index(index)
        if checked >= self.unit_count:
            raise KeyError(index)
        return self.units[checked]

    def owner_of(self, position: int) -> UnitIndex:
        checked = _as_position(position, field_name="owner lookup position")
        if not self.eligible_positions.contains(checked):
            raise KeyError(position)
        owners = tuple(
            unit.index for unit in self.units if unit.members.contains(checked)
        )
        if len(owners) != 1:
            raise InvalidArtifact(
                "verification partition no longer has deterministic ownership"
            )
        return owners[0]

    def units_in_replay_unit(
        self, replay_unit: int
    ) -> ExplicitIndexedDomain[UnitIndex]:
        checked = unit_index(replay_unit, field_name="replay unit index")
        if checked >= self.replay_unit_count:
            raise KeyError(replay_unit)
        return ExplicitIndexedDomain(
            unit.index for unit in self.units if unit.replay_unit == checked
        )

    def validate(self, replay_partition: ReplayPartition) -> None:
        """Recheck exact-cover, identity, and refinement invariants."""

        _validate_unit_indices(self.units)
        _validate_exact_cover(
            self.eligible_positions,
            self.units,
            label="verification partition",
        )
        validate_verification_refines_replay(replay_partition, self)
        expected_representation = identity_digest(
            "veritor/verification-partition-representation/v1",
            _verification_representation_manifest(
                self.eligible_positions,
                replay_partition.identity,
                self.units,
            ),
        )
        if (
            self.identity.partition_kind is not PartitionKind.VERIFICATION
            or self.identity.structure_digest != self.structure_identity.digest
            or self.identity.representation_digest != expected_representation
        ):
            raise InvalidArtifact("verification partition identity is inconsistent")


def validate_verification_refines_replay(
    replay_partition: ReplayPartition,
    verification_partition: VerificationPartition,
) -> None:
    """Require each verification unit to lie inside its declared replay unit."""

    if not isinstance(replay_partition, ReplayPartition) or not isinstance(
        verification_partition, VerificationPartition
    ):
        raise InvalidArtifact("refinement validation received a wrong partition type")
    if replay_partition.structure_identity != verification_partition.structure_identity:
        raise InvalidArtifact(
            "verification and replay partitions use different structure identities"
        )
    if verification_partition.replay_unit_count != replay_partition.unit_count:
        raise InvalidArtifact(
            "verification partition records a different replay-unit count"
        )
    if verification_partition.replay_partition_identity != replay_partition.identity:
        raise InvalidArtifact(
            "verification partition identifies a different replay partition"
        )
    if not domains_equal(
        replay_partition.eligible_positions,
        verification_partition.eligible_positions,
    ):
        raise InvalidArtifact(
            "verification and replay partitions have different eligible positions"
        )
    for verification_unit in verification_partition.units:
        try:
            replay_partition.unit_at(verification_unit.replay_unit)
        except KeyError as error:
            raise InvalidArtifact(
                f"verification unit {verification_unit.index} names a missing replay unit"
            ) from error
        for member in iter_domain(verification_unit.members):
            if replay_partition.owner_of(member) != verification_unit.replay_unit:
                raise InvalidArtifact(
                    f"verification unit {verification_unit.index} crosses replay units"
                )


validate_refinement = validate_verification_refines_replay
