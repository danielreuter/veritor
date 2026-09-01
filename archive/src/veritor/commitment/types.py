"""Backend-neutral value-commitment contracts.

The concrete protocol derives every :class:`CommitmentDomain`; a transcript
never gets to choose its positions, value schemas, or commitment owner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

from veritor.core import (
    ExplicitIndexedDomain,
    IndexedDomain,
    InvalidArtifact,
    Position,
    VerificationLimits,
    canonical_json_bytes,
    identity_digest,
    iter_domain,
    position,
    validate_digest,
)

MERKLE_SHA256_V1 = "veritor.commitment.merkle-sha256/v1"
COMMITMENT_PROTOCOL_ID = "veritor.staged-value-commitment/v1"


class CommitmentError(InvalidArtifact):
    """A value commitment or opening violates its public contract."""


class CommitmentOwnerKind(StrEnum):
    """The verifier-derived owner of a committed position."""

    BOUNDARY = "boundary"
    REPLAY_UNIT = "replay_unit"


@dataclass(frozen=True, slots=True, init=False)
class CommitmentOwner:
    """A boundary owner or a zero-based replay-unit owner."""

    kind: CommitmentOwnerKind
    replay_unit_index: int | None

    def __init__(
        self,
        kind: CommitmentOwnerKind | str,
        replay_unit_index: int | None = None,
    ) -> None:
        try:
            checked_kind = CommitmentOwnerKind(kind)
        except (TypeError, ValueError) as error:
            raise CommitmentError("unknown commitment owner kind") from error
        if checked_kind is CommitmentOwnerKind.BOUNDARY:
            if replay_unit_index is not None:
                raise CommitmentError("the boundary owner has no replay-unit index")
        elif type(replay_unit_index) is not int or replay_unit_index < 0:
            raise CommitmentError(
                "a replay-unit owner needs a nonnegative integer index"
            )
        object.__setattr__(self, "kind", checked_kind)
        object.__setattr__(self, "replay_unit_index", replay_unit_index)

    @classmethod
    def boundary(cls) -> CommitmentOwner:
        return cls(CommitmentOwnerKind.BOUNDARY)

    @classmethod
    def replay_unit(cls, index: int) -> CommitmentOwner:
        return cls(CommitmentOwnerKind.REPLAY_UNIT, index)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "replay_unit_index": self.replay_unit_index,
        }


def _bytes_field(
    value: object,
    name: str,
    *,
    length: int | None = None,
    allow_empty: bool = False,
) -> bytes:
    if type(value) is not bytes:
        raise CommitmentError(f"{name} must be bytes")
    if length is not None and len(value) != length:
        raise CommitmentError(f"{name} must be exactly {length} bytes")
    if length is None and not allow_empty and not value:
        raise CommitmentError(f"{name} must not be empty")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise CommitmentError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True, init=False)
class CommitmentDomain:
    """A fully bound, finite position domain for one value commitment.

    Positions are materialized in verifier-defined rank order.  This makes
    Merkle shape and local ranks independent of the original indexed-domain
    implementation.
    """

    protocol_id: str
    backend_id: str
    session_id: bytes
    compiled_result_digest: str
    policy_digest: str
    statement_digest: bytes
    phase_digest: bytes
    owner: CommitmentOwner
    positions: ExplicitIndexedDomain[Position]
    value_schema_ids: tuple[str, ...]
    position_sequence_digest: str = field(init=False)
    domain_id: bytes = field(init=False)

    def __init__(
        self,
        *,
        backend_id: str,
        session_id: bytes,
        compiled_result_digest: str,
        policy_digest: str,
        statement_digest: bytes,
        phase_digest: bytes,
        owner: CommitmentOwner,
        positions: IndexedDomain[Position] | Iterable[int],
        value_schema_ids: Iterable[str],
        protocol_id: str = COMMITMENT_PROTOCOL_ID,
    ) -> None:
        checked_protocol = _identifier(protocol_id, "protocol_id")
        checked_backend = _identifier(backend_id, "backend_id")
        checked_session = _bytes_field(session_id, "session_id")
        checked_compiled = str(
            validate_digest(compiled_result_digest, "compiled_result_digest")
        )
        checked_policy = str(validate_digest(policy_digest, "policy_digest"))
        checked_statement = _bytes_field(
            statement_digest, "statement_digest", length=32
        )
        checked_phase = _bytes_field(phase_digest, "phase_digest", length=32)
        if not isinstance(owner, CommitmentOwner):
            raise CommitmentError("owner must be a CommitmentOwner")

        if isinstance(positions, IndexedDomain):
            raw_positions = cast(tuple[int, ...], tuple(iter_domain(positions)))
        else:
            try:
                raw_positions = tuple(positions)
            except TypeError as error:
                raise CommitmentError("positions must be a finite domain") from error
        checked_positions = ExplicitIndexedDomain(
            position(item, field_name=f"positions[{rank}]")
            for rank, item in enumerate(raw_positions)
        )
        checked_schemas = tuple(value_schema_ids)
        if len(checked_schemas) != checked_positions.count:
            raise CommitmentError(
                "value schemas must have exactly one entry per position"
            )
        for rank, schema in enumerate(checked_schemas):
            _identifier(schema, f"value_schema_ids[{rank}]")

        sequence_digest = str(
            identity_digest(
                "veritor/commitment/position-sequence/v1",
                {"positions": list(checked_positions.items)},
            )
        )
        manifest = {
            "backend_id": checked_backend,
            "compiled_result_digest": checked_compiled,
            "owner": owner.manifest,
            "phase_digest": checked_phase.hex(),
            "policy_digest": checked_policy,
            "position_count": checked_positions.count,
            "position_sequence_digest": sequence_digest,
            "protocol_id": checked_protocol,
            "session_id": checked_session.hex(),
            "statement_digest": checked_statement.hex(),
            "value_schema_ids": list(checked_schemas),
        }
        domain_id = bytes.fromhex(
            identity_digest(
                "veritor/commitment/domain/v1",
                manifest,  # type: ignore[arg-type]
            )
        )

        object.__setattr__(self, "protocol_id", checked_protocol)
        object.__setattr__(self, "backend_id", checked_backend)
        object.__setattr__(self, "session_id", checked_session)
        object.__setattr__(self, "compiled_result_digest", checked_compiled)
        object.__setattr__(self, "policy_digest", checked_policy)
        object.__setattr__(self, "statement_digest", checked_statement)
        object.__setattr__(self, "phase_digest", checked_phase)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "positions", checked_positions)
        object.__setattr__(self, "value_schema_ids", checked_schemas)
        object.__setattr__(self, "position_sequence_digest", sequence_digest)
        object.__setattr__(self, "domain_id", domain_id)

    @property
    def count(self) -> int:
        return self.positions.count

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "compiled_result_digest": self.compiled_result_digest,
            "domain_id": self.domain_id.hex(),
            "owner": self.owner.manifest,
            "phase_digest": self.phase_digest.hex(),
            "policy_digest": self.policy_digest,
            "position_count": self.count,
            "position_sequence_digest": self.position_sequence_digest,
            "protocol_id": self.protocol_id,
            "session_id": self.session_id.hex(),
            "statement_digest": self.statement_digest.hex(),
            "value_schema_ids": list(self.value_schema_ids),
        }

    def schema_at_rank(self, rank: int) -> str:
        if type(rank) is not int or not 0 <= rank < self.count:
            raise IndexError(rank)
        return self.value_schema_ids[rank]

    def schema_for(self, global_position: int) -> str:
        try:
            rank = self.positions.rank(position(global_position))
        except KeyError as error:
            raise KeyError(global_position) from error
        return self.value_schema_ids[rank]


@dataclass(frozen=True, slots=True)
class ValueCommitment:
    """A backend-tagged root for a verifier-derived domain."""

    backend_id: str
    root: bytes
    value_count: int

    def __post_init__(self) -> None:
        _identifier(self.backend_id, "backend_id")
        _bytes_field(self.root, "root", allow_empty=True)
        if type(self.value_count) is not int or self.value_count < 0:
            raise CommitmentError("value_count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class ValueOpening:
    """One encoded value and its strict backend authentication path."""

    position: Position
    value: bytes
    path: tuple[bytes, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", position(self.position))
        _bytes_field(self.value, "opening value", allow_empty=True)
        if type(self.path) is not tuple:
            raise CommitmentError("opening path must be a tuple")
        for sibling in self.path:
            _bytes_field(sibling, "opening sibling", allow_empty=True)


@runtime_checkable
class ValueCommitmentProver(Protocol):
    """Private commitment material capable of producing openings."""

    @property
    def commitment(self) -> ValueCommitment: ...

    def open(self, global_position: int) -> ValueOpening: ...


@runtime_checkable
class ValueCommitmentBackend(Protocol):
    """Backend-neutral commitment construction and verification contract."""

    @property
    def backend_id(self) -> str: ...

    def commit(
        self,
        domain: CommitmentDomain,
        values: Mapping[int, bytes],
        limits: VerificationLimits,
    ) -> ValueCommitmentProver: ...

    def validate_commitment(
        self,
        domain: CommitmentDomain,
        commitment: ValueCommitment,
        limits: VerificationLimits,
    ) -> None: ...

    def verify_opening(
        self,
        domain: CommitmentDomain,
        commitment: ValueCommitment,
        opening: ValueOpening,
        limits: VerificationLimits,
    ) -> bool: ...

    def verify_openings(
        self,
        domain: CommitmentDomain,
        commitment: ValueCommitment,
        openings: tuple[ValueOpening, ...],
        required_positions: tuple[int, ...],
        limits: VerificationLimits,
    ) -> Mapping[int, bytes]: ...


def commitment_manifest(commitment: ValueCommitment) -> dict[str, object]:
    """Return the fixed canonical representation used by phase hashes."""

    if not isinstance(commitment, ValueCommitment):
        raise CommitmentError("expected a ValueCommitment")
    return {
        "backend_id": commitment.backend_id,
        "root": commitment.root.hex(),
        "value_count": commitment.value_count,
    }


def opening_manifest(opening: ValueOpening) -> dict[str, object]:
    """Return the fixed canonical representation used by phase hashes."""

    if not isinstance(opening, ValueOpening):
        raise CommitmentError("expected a ValueOpening")
    return {
        "path": [item.hex() for item in opening.path],
        "position": int(opening.position),
        "value": opening.value.hex(),
    }


def domain_canonical_bytes(domain: CommitmentDomain) -> bytes:
    """Expose deterministic domain bytes for test vectors and plug-ins."""

    return canonical_json_bytes(domain.manifest)  # type: ignore[arg-type]
