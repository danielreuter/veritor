"""Canonical manifests and content identities for protocol artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import NewType

from .errors import InvalidArtifact

Digest = NewType("Digest", str)

type JSONScalar = None | bool | int | str
type JSONValue = JSONScalar | Mapping[str, JSONValue] | Sequence[JSONValue]

_TAGGED_HASH_PREFIX = b"veritor/tagged-sha256/v1\0"


def _normalize_json(value: object, path: str = "$") -> JSONValue:
    if value is None or type(value) in (bool, int, str):
        return value  # type: ignore[return-value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string object key")
            normalized[key] = _normalize_json(child, f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [
            _normalize_json(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, float):
        raise TypeError(f"{path} contains a float; encode exact values explicitly")
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


def canonical_json_text(manifest: JSONValue) -> str:
    """Encode a JSON-compatible identity manifest deterministically.

    Object keys are sorted, insignificant whitespace is omitted, Unicode is
    encoded directly as UTF-8 by :func:`canonical_json_bytes`, and floats are
    rejected. Exact rational values must be represented by explicit integer
    numerator/denominator fields.
    """

    normalized = _normalize_json(manifest)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(manifest: JSONValue) -> bytes:
    """Return the canonical UTF-8 encoding of an identity manifest."""

    try:
        return canonical_json_text(manifest).encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "identity manifest contains an invalid Unicode surrogate"
        ) from error


# The short name follows the existing repository convention of returning bytes.
canonical_json = canonical_json_bytes


def tagged_sha256(tag: str, payload: bytes | bytearray | memoryview) -> Digest:
    """Hash ``payload`` with an unambiguous UTF-8 domain-separation tag."""

    if type(tag) is not str or not tag:
        raise TypeError("hash tag must be a nonempty string")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("tagged hash payload must be bytes-like")
    tag_bytes = tag.encode("utf-8")
    payload_bytes = bytes(payload)
    if len(tag_bytes) >= 1 << 32 or len(payload_bytes) >= 1 << 64:
        raise ValueError("tag or payload is too large for the canonical hash frame")
    framed = b"".join(
        (
            _TAGGED_HASH_PREFIX,
            len(tag_bytes).to_bytes(4, "big"),
            tag_bytes,
            len(payload_bytes).to_bytes(8, "big"),
            payload_bytes,
        )
    )
    return Digest(hashlib.sha256(framed).hexdigest())


def identity_digest(tag: str, manifest: JSONValue) -> Digest:
    """Return a tagged SHA-256 digest of a canonical identity manifest."""

    return tagged_sha256(tag, canonical_json_bytes(manifest))


canonical_manifest_digest = identity_digest


def validate_digest(value: object, field_name: str = "digest") -> Digest:
    """Validate and normalize a lowercase hexadecimal SHA-256 digest."""

    if type(value) is not str or len(value) != 64:
        raise InvalidArtifact(f"{field_name} must be a 64-character SHA-256 digest")
    if value != value.lower():
        raise InvalidArtifact(f"{field_name} must use lowercase hexadecimal")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise InvalidArtifact(f"{field_name} must be hexadecimal") from error
    return Digest(value)
