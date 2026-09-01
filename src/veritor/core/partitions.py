"""Concrete finite replay and verification partitions.

Both partitions are exact covers of one eligible position domain by nonempty
units.  When every unit is a contiguous run of positions (the common case for
compiled circuits) ownership lookups bisect over unit starts, so the verifier
never scans units or positions.
"""

from __future__ import annotations

from bisect import bisect_right
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
    IndexedDomain,
    IntervalDomain,
    RangeIndexedDomain,
    domains_equal,
    iter_domain,
    position_domain,
)


def _relation(value: str | None, field_name: str) -> RelationId | None:
    if value is None:
        return None
    return RelationId(nonempty_identifier(value, field_name=field_name))


def contiguous_span(domain: IndexedDomain[Position]) -> tuple[int, int] | None:
    """Return ``(start, stop)`` when ``domain`` is exactly one run of positions."""

    if isinstance(domain, RangeIndexedDomain) and domain.step == 1:
        return (domain.start, domain.stop)
    if isinstance(domain, IntervalDomain) and len(domain.intervals) == 1:
        return domain.intervals[0]
    return None


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


type _Unit = ReplayUnit | VerificationUnit


def _validate_unit_indices(units: tuple[_Unit, ...]) -> None:
    for expected, unit in enumerate(units):
        if unit.index != expected:
            raise InvalidArtifact(
                "partition units must have contiguous indices matching tuple order"
            )


def _span_table(units: tuple[_Unit, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(starts, unit indices)`` sorted by start, or empty tuples."""

    spans: list[tuple[int, int]] = []
    for unit in units:
        span = contiguous_span(unit.members)
        if span is None:
            return ((), ())
        spans.append((span[0], unit.index))
    spans.sort()
    return (tuple(start for start, _ in spans), tuple(index for _, index in spans))


def _validate_exact_cover(
    eligible: IndexedDomain[Position],
    units: tuple[_Unit, ...],
    starts: tuple[int, ...],
    order: tuple[int, ...],
    *,
    label: str,
) -> None:
    if eligible.count == 0:
        if units:
            raise InvalidArtifact(f"{label} must have zero units for an empty domain")
        return
    if not units:
        raise InvalidArtifact(f"{label} does not cover any eligible positions")
    eligible_span = contiguous_span(eligible)
    if starts and eligible_span is not None:
        cursor = eligible_span[0]
        for index in order:
            span = contiguous_span(units[index].members)
            assert span is not None
            if span[0] != cursor:
                raise InvalidArtifact(f"{label} does not tile the eligible positions")
            cursor = span[1]
        if cursor != eligible_span[1]:
            raise InvalidArtifact(f"{label} does not tile the eligible positions")
        return
    owners_by_rank: dict[int, UnitIndex] = {}
    for unit in units:
        for member in iter_domain(unit.members):
            try:
                rank = eligible.rank(member)
            except KeyError as error:
                raise InvalidArtifact(
                    f"{label} unit {unit.index} contains ineligible position {member}"
                ) from error
            if rank in owners_by_rank:
                raise InvalidArtifact(
                    f"{label} assigns position {member} to multiple units"
                )
            owners_by_rank[rank] = unit.index
    if len(owners_by_rank) != eligible.count:
        raise InvalidArtifact(
            f"{label} covers {len(owners_by_rank)} of {eligible.count} positions"
        )


def _owner_of(
    eligible: IndexedDomain[Position],
    units: tuple[_Unit, ...],
    starts: tuple[int, ...],
    order: tuple[int, ...],
    position: int,
    *,
    label: str,
) -> UnitIndex:
    checked = _as_position(position, field_name="owner lookup position")
    if not eligible.contains(checked):
        raise KeyError(position)
    if starts:
        index = order[bisect_right(starts, checked) - 1]
        if units[index].members.contains(checked):
            return units[index].index
        raise InvalidArtifact(f"{label} no longer has deterministic ownership")
    owners = tuple(unit.index for unit in units if unit.members.contains(checked))
    if len(owners) != 1:
        raise InvalidArtifact(f"{label} no longer has deterministic ownership")
    return owners[0]


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
    _starts: tuple[int, ...] = field(repr=False, compare=False, hash=False)
    _order: tuple[int, ...] = field(repr=False, compare=False, hash=False)

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
        starts, order = _span_table(checked_units)
        _validate_exact_cover(
            checked_eligible, checked_units, starts, order, label="replay partition"
        )
        checked_algorithm = nonempty_identifier(algorithm_id, field_name="algorithm_id")
        checked_version = nonempty_identifier(
            algorithm_version, field_name="algorithm_version"
        )
        expected_identity = PartitionIdentity(
            partition_kind=PartitionKind.REPLAY,
            structure_digest=structure_identity.digest,
            algorithm_id=checked_algorithm,
            algorithm_version=checked_version,
            configuration_digest=_configuration_digest(
                PartitionKind.REPLAY, configuration
            ),
            representation_digest=identity_digest(
                "veritor/replay-partition-representation/v1",
                _replay_representation_manifest(checked_eligible, checked_units),
            ),
        )
        if identity is not None and identity != expected_identity:
            raise InvalidArtifact(
                "provided replay partition identity does not match its contents"
            )
        object.__setattr__(self, "structure_identity", structure_identity)
        object.__setattr__(self, "eligible_positions", checked_eligible)
        object.__setattr__(self, "units", checked_units)
        object.__setattr__(self, "identity", expected_identity)
        object.__setattr__(self, "_starts", starts)
        object.__setattr__(self, "_order", order)

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
        return _owner_of(
            self.eligible_positions,
            self.units,
            self._starts,
            self._order,
            position,
            label="replay partition",
        )


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
    _starts: tuple[int, ...] = field(repr=False, compare=False, hash=False)
    _order: tuple[int, ...] = field(repr=False, compare=False, hash=False)
    _by_replay: tuple[tuple[UnitIndex, ...], ...] = field(
        repr=False, compare=False, hash=False
    )

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
        starts, order = _span_table(checked_units)
        _validate_exact_cover(
            checked_eligible,
            checked_units,
            starts,
            order,
            label="verification partition",
        )
        checked_algorithm = nonempty_identifier(algorithm_id, field_name="algorithm_id")
        checked_version = nonempty_identifier(
            algorithm_version, field_name="algorithm_version"
        )
        expected_identity = PartitionIdentity(
            partition_kind=PartitionKind.VERIFICATION,
            structure_digest=structure_identity.digest,
            algorithm_id=checked_algorithm,
            algorithm_version=checked_version,
            configuration_digest=_configuration_digest(
                PartitionKind.VERIFICATION, configuration
            ),
            representation_digest=identity_digest(
                "veritor/verification-partition-representation/v1",
                _verification_representation_manifest(
                    checked_eligible, replay_partition.identity, checked_units
                ),
            ),
        )
        if identity is not None and identity != expected_identity:
            raise InvalidArtifact(
                "provided verification partition identity does not match its contents"
            )
        grouped: list[list[UnitIndex]] = [
            [] for _ in range(replay_partition.unit_count)
        ]
        for unit in checked_units:
            if unit.replay_unit >= replay_partition.unit_count:
                raise InvalidArtifact(
                    f"verification unit {unit.index} names a missing replay unit"
                )
            grouped[unit.replay_unit].append(unit.index)
        object.__setattr__(self, "structure_identity", structure_identity)
        object.__setattr__(self, "replay_partition_identity", replay_partition.identity)
        object.__setattr__(self, "eligible_positions", checked_eligible)
        object.__setattr__(self, "units", checked_units)
        object.__setattr__(self, "identity", expected_identity)
        object.__setattr__(self, "replay_unit_count", replay_partition.unit_count)
        object.__setattr__(self, "_starts", starts)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_by_replay", tuple(tuple(g) for g in grouped))
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
        return _owner_of(
            self.eligible_positions,
            self.units,
            self._starts,
            self._order,
            position,
            label="verification partition",
        )

    def units_in_replay_unit(self, replay_unit: int) -> tuple[UnitIndex, ...]:
        checked = unit_index(replay_unit, field_name="replay unit index")
        if checked >= self.replay_unit_count:
            raise KeyError(replay_unit)
        return self._by_replay[checked]


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
    for unit in verification_partition.units:
        span = contiguous_span(unit.members)
        if span is not None and contiguous_span(
            replay_partition.unit_at(unit.replay_unit).members
        ):
            probes: Iterable[int] = (span[0], span[1] - 1)
        else:
            probes = iter_domain(unit.members)
        for member in probes:
            if replay_partition.owner_of(member) != unit.replay_unit:
                raise InvalidArtifact(
                    f"verification unit {unit.index} crosses replay units"
                )
