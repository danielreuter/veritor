"""Exact two-stage survival probabilities for finite error sets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from veritor.core.ids import Position, UnitIndex
from veritor.core.partitions import (
    ReplayPartition,
    VerificationPartition,
    validate_verification_refines_replay,
)
from veritor.core.policy import VerificationPolicy


class AttackSetKind(StrEnum):
    """The atom named by an error-set input."""

    VERIFICATION_UNITS = "verification_units"
    POSITIONS = "positions"


@dataclass(frozen=True, slots=True)
class VerificationUnitErrorSet:
    """An explicitly typed finite set of verification-unit indices."""

    units: frozenset[int]

    def __init__(self, units: Iterable[int]) -> None:
        object.__setattr__(self, "units", frozenset(units))


@dataclass(frozen=True, slots=True)
class PositionErrorSet:
    """An explicitly typed finite set of attacked computed positions."""

    positions: frozenset[int]

    def __init__(self, positions: Iterable[int]) -> None:
        object.__setattr__(self, "positions", frozenset(positions))


type ErrorSetInput = Iterable[int] | VerificationUnitErrorSet | PositionErrorSet


def _attack_kind(value: AttackSetKind | str) -> AttackSetKind:
    aliases = {
        "units": AttackSetKind.VERIFICATION_UNITS,
        "verification_unit_ids": AttackSetKind.VERIFICATION_UNITS,
        "verification_units": AttackSetKind.VERIFICATION_UNITS,
        "positions": AttackSetKind.POSITIONS,
    }
    try:
        return aliases[str(value)]
    except KeyError:
        try:
            return AttackSetKind(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown attack-set kind {value!r}") from error


def normalize_error_units(
    replay_partition: ReplayPartition,
    verification_partition: VerificationPartition,
    error_set: ErrorSetInput,
    *,
    attack_kind: AttackSetKind | str = AttackSetKind.VERIFICATION_UNITS,
) -> tuple[UnitIndex, ...]:
    """Normalize verification-unit IDs or positions to sorted unit indices.

    Plain integer iterables name verification units by default.  Callers using
    positions should pass ``attack_kind="positions"`` or :class:`PositionErrorSet`
    so an integer that is valid in both domains is never interpreted by a
    heuristic.
    """

    validate_verification_refines_replay(
        replay_partition,
        verification_partition,
    )
    if isinstance(error_set, VerificationUnitErrorSet):
        values = error_set.units
        kind = AttackSetKind.VERIFICATION_UNITS
    elif isinstance(error_set, PositionErrorSet):
        values = error_set.positions
        kind = AttackSetKind.POSITIONS
    else:
        try:
            values = frozenset(error_set)
        except TypeError as error:
            raise TypeError("error_set must be a finite iterable of integers") from error
        kind = _attack_kind(attack_kind)

    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("error-set members must be nonnegative integers")

    if kind is AttackSetKind.VERIFICATION_UNITS:
        if any(value >= verification_partition.unit_count for value in values):
            raise ValueError("error set names an unknown verification unit")
        return tuple(UnitIndex(value) for value in sorted(values))

    units: set[UnitIndex] = set()
    for value in values:
        try:
            units.add(verification_partition.owner_of(Position(value)))
        except KeyError as error:
            raise ValueError(
                f"error set names ineligible computed position {value}"
            ) from error
    return tuple(sorted(units))


def survival_from_replay_error_counts(
    policy: VerificationPolicy,
    incorrect_units_per_replay: Iterable[int],
) -> Fraction:
    """Evaluate ``prod_r (1-q + q(1-s)^ell_r)`` exactly."""

    counts = tuple(incorrect_units_per_replay)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("replay error counts must be nonnegative integers")
    q = policy.q
    s = policy.s
    survival = Fraction(1)
    for count in counts:
        survival *= 1 - q + q * (1 - s) ** count
    return survival


def survival_probability(
    replay_partition: ReplayPartition,
    verification_partition: VerificationPartition,
    policy: VerificationPolicy,
    error_set: ErrorSetInput,
    *,
    attack_kind: AttackSetKind | str = AttackSetKind.VERIFICATION_UNITS,
) -> Fraction:
    """Return the exact fixed-policy survival probability for ``error_set``.

    First-stage replay choices are independent across replay units.  All
    verification units inside one replay unit share its first-stage choice,
    while their second-stage choices are independent.  The strict admissibility
    predicate used by bound solvers is ``survival > policy.eta``.
    """

    if not isinstance(policy, VerificationPolicy):
        raise TypeError("policy must be a VerificationPolicy")
    units = normalize_error_units(
        replay_partition,
        verification_partition,
        error_set,
        attack_kind=attack_kind,
    )
    counts = [0] * replay_partition.unit_count
    for unit_index in units:
        owner = verification_partition.unit_at(unit_index).replay_unit
        counts[owner] += 1
    return survival_from_replay_error_counts(policy, counts)


def survives_strict_threshold(
    survival: Fraction,
    policy: VerificationPolicy,
) -> bool:
    """Return the protocol's strict admissibility decision."""

    if not isinstance(survival, Fraction):
        raise TypeError("survival must be an exact Fraction")
    return survival > policy.eta


__all__ = [
    "AttackSetKind",
    "ErrorSetInput",
    "PositionErrorSet",
    "VerificationUnitErrorSet",
    "normalize_error_units",
    "survival_from_replay_error_counts",
    "survival_probability",
    "survives_strict_threshold",
]
