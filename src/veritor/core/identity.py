"""Canonical manifests and content identities for protocol artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NewType

from .capabilities import ArtifactKind
from .errors import InvalidArtifact

Digest = NewType("Digest", str)

type JSONScalar = None | bool | int | str
type JSONValue = JSONScalar | Mapping[str, JSONValue] | Sequence[JSONValue]

_TAGGED_HASH_PREFIX = b"veritor/tagged-sha256/v1\0"
STRUCTURE_IDENTITY_TAG = "veritor/structure-identity/v1"
PARTITION_IDENTITY_TAG = "veritor/partition-identity/v1"
COMPILED_RESULT_IDENTITY_TAG = "veritor/compiled-result-identity/v2"


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


def _required_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise InvalidArtifact(f"{field_name} must be a nonempty string")
    return value


class PartitionKind(StrEnum):
    """The role played by a partition in the two-stage protocol."""

    REPLAY = "replay"
    VERIFICATION = "verification"


@dataclass(frozen=True, slots=True, init=False)
class StructureIdentity:
    """Identity of a static circuit or aggregate capacity artifact."""

    schema_version: str
    artifact_kind: ArtifactKind
    compiler_id: str
    compiler_version: str
    semantic_scope_id: str
    representation_digest: Digest
    value_registry_digest: Digest | None
    operator_registry_digest: Digest | None
    digest: Digest = field(init=False)

    def __init__(
        self,
        *,
        schema_version: str,
        artifact_kind: ArtifactKind,
        compiler_id: str,
        compiler_version: str,
        semantic_scope_id: str,
        representation_digest: Digest | str,
        value_registry_digest: Digest | str | None = None,
        operator_registry_digest: Digest | str | None = None,
    ) -> None:
        checked_schema = _required_text(schema_version, "schema_version")
        checked_compiler = _required_text(compiler_id, "compiler_id")
        checked_compiler_version = _required_text(compiler_version, "compiler_version")
        checked_scope = _required_text(semantic_scope_id, "semantic_scope_id")
        try:
            checked_kind = ArtifactKind(artifact_kind)
        except (TypeError, ValueError) as error:
            raise InvalidArtifact("artifact_kind is unknown") from error
        checked_representation = validate_digest(
            representation_digest, "representation_digest"
        )
        checked_values = (
            None
            if value_registry_digest is None
            else validate_digest(value_registry_digest, "value_registry_digest")
        )
        checked_operators = (
            None
            if operator_registry_digest is None
            else validate_digest(operator_registry_digest, "operator_registry_digest")
        )
        object.__setattr__(self, "schema_version", checked_schema)
        object.__setattr__(self, "artifact_kind", checked_kind)
        object.__setattr__(self, "compiler_id", checked_compiler)
        object.__setattr__(self, "compiler_version", checked_compiler_version)
        object.__setattr__(self, "semantic_scope_id", checked_scope)
        object.__setattr__(self, "representation_digest", checked_representation)
        object.__setattr__(self, "value_registry_digest", checked_values)
        object.__setattr__(self, "operator_registry_digest", checked_operators)
        object.__setattr__(
            self,
            "digest",
            identity_digest(STRUCTURE_IDENTITY_TAG, self.manifest),
        )

    @property
    def manifest(self) -> dict[str, JSONValue]:
        """Return the canonical manifest whose digest identifies this structure."""

        return {
            "artifact_kind": self.artifact_kind.value,
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "operator_registry_digest": self.operator_registry_digest,
            "representation_digest": self.representation_digest,
            "schema_version": self.schema_version,
            "semantic_scope_id": self.semantic_scope_id,
            "value_registry_digest": self.value_registry_digest,
        }

    @classmethod
    def from_manifest(
        cls,
        representation_manifest: JSONValue,
        *,
        schema_version: str,
        artifact_kind: ArtifactKind,
        compiler_id: str,
        compiler_version: str,
        semantic_scope_id: str,
        value_registry_digest: Digest | str | None = None,
        operator_registry_digest: Digest | str | None = None,
    ) -> StructureIdentity:
        """Build an identity while hashing a representation manifest."""

        representation_digest = identity_digest(
            "veritor/structure-representation/v1",
            representation_manifest,
        )
        return cls(
            schema_version=schema_version,
            artifact_kind=artifact_kind,
            compiler_id=compiler_id,
            compiler_version=compiler_version,
            semantic_scope_id=semantic_scope_id,
            representation_digest=representation_digest,
            value_registry_digest=value_registry_digest,
            operator_registry_digest=operator_registry_digest,
        )


@dataclass(frozen=True, slots=True, init=False)
class PartitionIdentity:
    """Identity of a deterministic replay or verification partition."""

    partition_kind: PartitionKind
    structure_digest: Digest
    algorithm_id: str
    algorithm_version: str
    configuration_digest: Digest
    representation_digest: Digest
    digest: Digest = field(init=False)

    def __init__(
        self,
        *,
        partition_kind: PartitionKind,
        structure_digest: Digest | str,
        algorithm_id: str,
        algorithm_version: str,
        configuration_digest: Digest | str,
        representation_digest: Digest | str,
    ) -> None:
        try:
            checked_kind = PartitionKind(partition_kind)
        except (TypeError, ValueError) as error:
            raise InvalidArtifact("partition_kind is unknown") from error
        checked_structure = validate_digest(structure_digest, "structure_digest")
        checked_algorithm = _required_text(algorithm_id, "algorithm_id")
        checked_version = _required_text(algorithm_version, "algorithm_version")
        checked_configuration = validate_digest(
            configuration_digest, "configuration_digest"
        )
        checked_representation = validate_digest(
            representation_digest, "representation_digest"
        )
        object.__setattr__(self, "partition_kind", checked_kind)
        object.__setattr__(self, "structure_digest", checked_structure)
        object.__setattr__(self, "algorithm_id", checked_algorithm)
        object.__setattr__(self, "algorithm_version", checked_version)
        object.__setattr__(self, "configuration_digest", checked_configuration)
        object.__setattr__(self, "representation_digest", checked_representation)
        object.__setattr__(
            self,
            "digest",
            identity_digest(PARTITION_IDENTITY_TAG, self.manifest),
        )

    @property
    def kind(self) -> PartitionKind:
        """Alias for ``partition_kind`` used by concise callers."""

        return self.partition_kind

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "configuration_digest": self.configuration_digest,
            "partition_kind": self.partition_kind.value,
            "representation_digest": self.representation_digest,
            "structure_digest": self.structure_digest,
        }


@dataclass(frozen=True, slots=True, init=False)
class CompiledResultIdentity:
    """Identity of a compiled ``(C, replay, verification, boundary)`` result."""

    schema_version: str
    structure_digest: Digest
    replay_partition_digest: Digest
    verification_partition_digest: Digest
    boundary_digest: Digest
    digest: Digest = field(init=False)

    def __init__(
        self,
        *,
        schema_version: str,
        structure_digest: Digest | str,
        replay_partition_digest: Digest | str,
        verification_partition_digest: Digest | str,
        boundary_digest: Digest | str,
    ) -> None:
        checked_schema = _required_text(schema_version, "schema_version")
        checked_structure = validate_digest(structure_digest, "structure_digest")
        checked_replay = validate_digest(
            replay_partition_digest, "replay_partition_digest"
        )
        checked_verification = validate_digest(
            verification_partition_digest, "verification_partition_digest"
        )
        checked_boundary = validate_digest(boundary_digest, "boundary_digest")
        object.__setattr__(self, "schema_version", checked_schema)
        object.__setattr__(self, "structure_digest", checked_structure)
        object.__setattr__(self, "replay_partition_digest", checked_replay)
        object.__setattr__(self, "verification_partition_digest", checked_verification)
        object.__setattr__(self, "boundary_digest", checked_boundary)
        object.__setattr__(
            self,
            "digest",
            identity_digest(COMPILED_RESULT_IDENTITY_TAG, self.manifest),
        )

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "boundary_digest": self.boundary_digest,
            "replay_partition_digest": self.replay_partition_digest,
            "schema_version": self.schema_version,
            "structure_digest": self.structure_digest,
            "verification_partition_digest": self.verification_partition_digest,
        }
