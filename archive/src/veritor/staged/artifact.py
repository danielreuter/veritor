"""Trusted executable-artifact resolution and semantic services."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from veritor.core import (
    ArtifactKind,
    ExecutableCircuit,
    ReplayPartition,
    StructuralCircuit,
    VerificationPartition,
    validate_compiled_result,
)

from .model import StagedProtocolError


@runtime_checkable
class ValueEncodingService(Protocol):
    """Trusted canonical encoding and decoding for named value schemas."""

    def encode(self, value_type: str, value: object) -> bytes: ...

    def decode(self, value_type: str, payload: bytes) -> object: ...


@runtime_checkable
class TrustedRelationService(Protocol):
    """Trusted local relation checker resolved with the executable artifact."""

    def check(
        self,
        relation_id: str,
        arguments: Sequence[object],
        output: object,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class UniformValueEncodingService:
    """Adapt a one-schema codec exposing ``encode`` and ``decode`` methods."""

    value_type: str
    codec: object

    def __post_init__(self) -> None:
        if type(self.value_type) is not str or not self.value_type.strip():
            raise StagedProtocolError("uniform value_type must be nonempty")
        if not callable(getattr(self.codec, "encode", None)) or not callable(
            getattr(self.codec, "decode", None)
        ):
            raise StagedProtocolError("codec must expose encode and decode")

    def encode(self, value_type: str, value: object) -> bytes:
        if value_type != self.value_type:
            raise StagedProtocolError(f"unknown value schema {value_type!r}")
        result = self.codec.encode(value)  # type: ignore[attr-defined]
        if type(result) is not bytes or not result:
            raise StagedProtocolError("trusted codec returned a noncanonical payload")
        return result

    def decode(self, value_type: str, payload: bytes) -> object:
        if value_type != self.value_type:
            raise StagedProtocolError(f"unknown value schema {value_type!r}")
        return self.codec.decode(payload)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class EvaluatingRelationService:
    """Turn a trusted evaluator's result into an equality relation check."""

    evaluator: object

    def __post_init__(self) -> None:
        if not callable(getattr(self.evaluator, "evaluate", None)):
            raise StagedProtocolError("relation evaluator must expose evaluate")

    def check(
        self,
        relation_id: str,
        arguments: Sequence[object],
        output: object,
    ) -> bool:
        expected = self.evaluate(relation_id, arguments)
        return bool(expected == output)

    def evaluate(
        self,
        relation_id: str,
        arguments: Sequence[object],
    ) -> object:
        """Evaluate a relation for client-side replay."""

        return self.evaluator.evaluate(  # type: ignore[attr-defined]
            relation_id,
            tuple(arguments),
        )


@dataclass(frozen=True, slots=True)
class ResolvedStructuralArtifact:
    """Trusted compiled tuple without executable local semantics."""

    circuit: StructuralCircuit
    replay_partition: ReplayPartition
    verification_partition: VerificationPartition

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, StructuralCircuit):
            raise StagedProtocolError("structural artifact needs a StructuralCircuit")
        if not isinstance(self.replay_partition, ReplayPartition) or not isinstance(
            self.verification_partition, VerificationPartition
        ):
            raise StagedProtocolError(
                "structural artifact needs core replay and verification partitions"
            )

    @property
    def compiled_result_digest(self) -> str:
        return str(
            validate_compiled_result(
                self.circuit,
                self.replay_partition,
                self.verification_partition,
            ).digest
        )


@dataclass(frozen=True, slots=True)
class ResolvedExecutableArtifact:
    """Trusted compiled tuple plus value and local-relation services."""

    circuit: ExecutableCircuit
    replay_partition: ReplayPartition
    verification_partition: VerificationPartition
    value_service: ValueEncodingService
    relation_service: TrustedRelationService

    def __post_init__(self) -> None:
        if (
            not isinstance(self.circuit, ExecutableCircuit)
            or self.circuit.identity.artifact_kind
            is not ArtifactKind.EXECUTABLE_CIRCUIT
        ):
            raise StagedProtocolError(
                "executable artifact needs an executable-circuit identity"
            )
        if not isinstance(self.replay_partition, ReplayPartition) or not isinstance(
            self.verification_partition, VerificationPartition
        ):
            raise StagedProtocolError(
                "executable artifact needs core replay and verification partitions"
            )
        if not isinstance(self.value_service, ValueEncodingService):
            raise StagedProtocolError(
                "executable artifact needs a ValueEncodingService"
            )
        if not isinstance(self.relation_service, TrustedRelationService):
            raise StagedProtocolError(
                "executable artifact needs a TrustedRelationService"
            )

    @property
    def compiled_result_digest(self) -> str:
        return str(
            validate_compiled_result(
                self.circuit,
                self.replay_partition,
                self.verification_partition,
            ).digest
        )

    @classmethod
    def from_uniform_circuit(
        cls,
        circuit: ExecutableCircuit,
        replay_partition: ReplayPartition,
        verification_partition: VerificationPartition,
        *,
        codec: object,
        relation_evaluator: object,
        value_type: str | None = None,
    ) -> ResolvedExecutableArtifact:
        """Adapt common circuits with one value codec and evaluator."""

        schemas = {
            str(port.value_type)
            for port in (*circuit.input_ports, *circuit.output_ports)
        }
        for rank in range(circuit.computed_positions.count):
            schemas.add(
                str(
                    circuit.executable_gate_at(
                        circuit.computed_positions.unrank(rank)
                    ).output_type
                )
            )
        if value_type is None:
            if len(schemas) != 1:
                raise StagedProtocolError(
                    "value_type is required for a multi-schema circuit"
                )
            value_type = schemas.pop()
        return cls(
            circuit,
            replay_partition,
            verification_partition,
            UniformValueEncodingService(value_type, codec),
            EvaluatingRelationService(relation_evaluator),
        )


ResolvedArtifact = ResolvedStructuralArtifact | ResolvedExecutableArtifact


@runtime_checkable
class TrustedArtifactResolver(Protocol):
    """Resolve a transcript identity using verifier-local trust."""

    def resolve(self, compiled_result_digest: str) -> ResolvedArtifact | None: ...


class TrustedArtifactRegistry:
    """Immutable content-addressed allowlist of resolved artifacts."""

    __slots__ = ("_artifacts",)

    def __init__(self, artifacts: Iterable[ResolvedArtifact]) -> None:
        by_digest: dict[str, ResolvedArtifact] = {}
        for artifact in artifacts:
            if not isinstance(
                artifact,
                (ResolvedStructuralArtifact, ResolvedExecutableArtifact),
            ):
                raise StagedProtocolError("artifact registry entry has the wrong type")
            digest = artifact.compiled_result_digest
            if digest in by_digest:
                raise StagedProtocolError(
                    f"duplicate compiled artifact digest {digest}"
                )
            by_digest[digest] = artifact
        self._artifacts = MappingProxyType(by_digest)

    @property
    def compiled_result_digests(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))

    def resolve(self, compiled_result_digest: str) -> ResolvedArtifact | None:
        if type(compiled_result_digest) is not str:
            return None
        return self._artifacts.get(compiled_result_digest)
