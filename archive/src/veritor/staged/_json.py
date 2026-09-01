"""Shared strict canonical-JSON primitives for transcript and evidence codecs."""

from __future__ import annotations

import json

from veritor.core import ResourceLimit, VerificationLimits, canonical_json_bytes


class WireCodecError(ValueError):
    """A JSON document or typed wire value is malformed."""


class NonCanonicalWireError(WireCodecError):
    """A valid typed document did not use its unique canonical encoding."""


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WireCodecError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise WireCodecError(f"floating-point value {value!r} is forbidden")


def _reject_constant(value: str) -> object:
    raise WireCodecError(f"non-JSON numeric constant {value!r} is forbidden")


def _check_depth(value: object, maximum: int) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > maximum:
            raise ResourceLimit(
                "nesting_depth",
                limit=maximum,
                observed=depth,
            )
        if type(current) is dict:
            stack.extend((child, depth + 1) for child in current.values())
        elif type(current) is list:
            stack.extend((child, depth + 1) for child in current)


def load_strict_json(
    data: bytes,
    limits: VerificationLimits,
    *,
    byte_limit: int | None = None,
) -> object:
    if type(data) is not bytes:
        raise WireCodecError("wire data must be bytes")
    maximum = limits.max_transcript_bytes if byte_limit is None else byte_limit
    if len(data) > maximum:
        # Kept as a codec error for standalone callers; the top-level verifier
        # performs the authoritative ResourceLimit check before calling here.
        raise WireCodecError("wire data exceeds its byte limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WireCodecError("wire data is not valid UTF-8") from error
    try:
        result = json.loads(
            text,
            object_pairs_hook=_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except WireCodecError:
        raise
    except (ValueError, RecursionError) as error:
        raise WireCodecError("wire data is not one complete JSON document") from error
    _check_depth(result, limits.max_nesting_depth)
    return result


def canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise WireCodecError("value cannot be canonically encoded") from error


def exact_keys(
    value: object,
    expected: frozenset[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise WireCodecError(f"{name} must be an object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise WireCodecError(
            f"{name} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def integer(value: object, name: str, *, nonnegative: bool = True) -> int:
    if type(value) is not int:
        raise WireCodecError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise WireCodecError(f"{name} must be nonnegative")
    return value


def text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise WireCodecError(f"{name} must be a nonempty string")
    return value


def hex_bytes(
    value: object,
    name: str,
    *,
    length: int | None = None,
    allow_empty: bool = False,
) -> bytes:
    if type(value) is not str:
        raise WireCodecError(f"{name} must be hexadecimal text")
    if value != value.lower() or len(value) % 2:
        raise NonCanonicalWireError(
            f"{name} must use lowercase even-length hexadecimal"
        )
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise WireCodecError(f"{name} is not hexadecimal") from error
    if length is not None and len(result) != length:
        raise WireCodecError(f"{name} must encode exactly {length} bytes")
    if not allow_empty and not result:
        raise WireCodecError(f"{name} must not be empty")
    return result


def array(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise WireCodecError(f"{name} must be an array")
    return value
