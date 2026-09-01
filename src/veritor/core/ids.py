"""Nominal identifiers used by circuit and partition contracts."""

from __future__ import annotations

from typing import NewType

from .errors import InvalidArtifact

Position = NewType("Position", int)
UnitIndex = NewType("UnitIndex", int)
UnitId = NewType("UnitId", int)
ReplayUnitId = NewType("ReplayUnitId", int)
VerificationUnitId = NewType("VerificationUnitId", int)

OperationId = NewType("OperationId", str)
ValueTypeId = NewType("ValueTypeId", str)
RelationId = NewType("RelationId", str)


def position(value: object, *, field_name: str = "position") -> Position:
    """Validate and return a nonnegative position identifier."""

    if type(value) is not int or value < 0:
        raise InvalidArtifact(f"{field_name} must be a nonnegative integer")
    return Position(value)


def unit_index(value: object, *, field_name: str = "unit index") -> UnitIndex:
    """Validate and return a nonnegative unit index."""

    if type(value) is not int or value < 0:
        raise InvalidArtifact(f"{field_name} must be a nonnegative integer")
    return UnitIndex(value)


def nonempty_identifier(value: object, *, field_name: str) -> str:
    """Validate a stable textual identifier."""

    if type(value) is not str or not value.strip():
        raise InvalidArtifact(f"{field_name} must be a nonempty string")
    return value
