"""Nominal identifiers used by the indexed-domain contracts."""

from __future__ import annotations

from typing import NewType

from .errors import InvalidArtifact

Position = NewType("Position", int)


def position(value: object, *, field_name: str = "position") -> Position:
    """Validate and return a nonnegative position identifier."""

    if type(value) is not int or value < 0:
        raise InvalidArtifact(f"{field_name} must be a nonnegative integer")
    return Position(value)
