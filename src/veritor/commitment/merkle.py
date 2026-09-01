"""Transparent indexed SHA-256 value commitments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from veritor.core import Position, VerificationLimits

from .types import (
    MERKLE_SHA256_V1,
    CommitmentDomain,
    CommitmentError,
    ValueCommitment,
    ValueOpening,
)

_FRAME_TAG = b"veritor/commitment/hash-frame/v1\0"
_LEAF_TAG = b"veritor/commitment/merkle-sha256/leaf/v1"
_PADDING_TAG = b"veritor/commitment/merkle-sha256/padding/v1"
_NODE_TAG = b"veritor/commitment/merkle-sha256/node/v1"
_EMPTY_TAG = b"veritor/commitment/merkle-sha256/empty/v1"


def _uint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise CommitmentError("Merkle integer fields must be nonnegative integers")
    width = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(width, "big")


def _hash_parts(tag: bytes, *parts: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(_FRAME_TAG)
    digest.update(len(tag).to_bytes(4, "big"))
    digest.update(tag)
    for part in parts:
        if type(part) is not bytes:
            raise TypeError("hash frame parts must be bytes")
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def merkle_depth(value_count: int) -> int:
    """Return the unique authentication-path depth for ``value_count``."""

    if type(value_count) is not int or value_count < 0:
        raise CommitmentError("value_count must be a nonnegative integer")
    return 0 if value_count <= 1 else (value_count - 1).bit_length()


def _leaf_hash(
    domain: CommitmentDomain,
    local_rank: int,
    global_position: int,
    schema: str,
    value: bytes,
) -> bytes:
    # Some fields are redundant with domain_id by design.  The explicit frame
    # is the public binding contract and remains auditable across backends.
    return _hash_parts(
        _LEAF_TAG,
        domain.domain_id,
        domain.session_id,
        bytes.fromhex(domain.compiled_result_digest),
        bytes.fromhex(domain.policy_digest),
        domain.owner.kind.value.encode("utf-8"),
        _uint(
            0
            if domain.owner.replay_unit_index is None
            else domain.owner.replay_unit_index + 1
        ),
        schema.encode("utf-8"),
        _uint(global_position),
        _uint(local_rank),
        value,
    )


def _padding_hash(domain: CommitmentDomain, local_rank: int) -> bytes:
    return _hash_parts(_PADDING_TAG, domain.domain_id, _uint(local_rank))


def _node_hash(
    domain: CommitmentDomain,
    level: int,
    parent_rank: int,
    left: bytes,
    right: bytes,
) -> bytes:
    return _hash_parts(
        _NODE_TAG,
        domain.domain_id,
        _uint(level),
        _uint(parent_rank),
        left,
        right,
    )


def empty_root(domain: CommitmentDomain) -> bytes:
    """Return the sole canonical root for an empty domain."""

    return _hash_parts(_EMPTY_TAG, domain.domain_id)


@dataclass(frozen=True, slots=True)
class MerkleCommittedValues:
    """Private material for one transparent commitment."""

    domain: CommitmentDomain
    values: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]

    @property
    def commitment(self) -> ValueCommitment:
        root = empty_root(self.domain) if not self.levels else self.levels[-1][0]
        return ValueCommitment(MERKLE_SHA256_V1, root, self.domain.count)

    def open(self, global_position: int) -> ValueOpening:
        try:
            local_rank = self.domain.positions.rank(Position(global_position))
        except KeyError as error:
            raise CommitmentError(
                f"position {global_position} is outside the commitment domain"
            ) from error
        rank = local_rank
        path: list[bytes] = []
        for level in self.levels[:-1]:
            path.append(level[rank ^ 1])
            rank >>= 1
        return ValueOpening(
            Position(global_position),
            self.values[local_rank],
            tuple(path),
        )


@dataclass(frozen=True, slots=True)
class MerkleSha256Backend:
    """Canonical power-of-two-padded SHA-256 commitment backend."""

    backend_id: str = MERKLE_SHA256_V1

    def _preflight(
        self,
        domain: CommitmentDomain,
        limits: VerificationLimits,
    ) -> None:
        if not isinstance(domain, CommitmentDomain):
            raise CommitmentError("commitment domain has the wrong type")
        if domain.backend_id != self.backend_id:
            raise CommitmentError("commitment domain names another backend")
        limits.enforce("max_positions", domain.count)

    def commit(
        self,
        domain: CommitmentDomain,
        values: Mapping[int, bytes],
        limits: VerificationLimits,
    ) -> MerkleCommittedValues:
        self._preflight(domain, limits)
        if not isinstance(values, Mapping):
            raise CommitmentError("committed values must be a mapping")
        expected = tuple(int(item) for item in domain.positions.items)
        try:
            actual = tuple(values.keys())
        except TypeError as error:
            raise CommitmentError("committed values have invalid keys") from error
        if any(type(item) is not int or item < 0 for item in actual):
            raise CommitmentError(
                "committed value keys must be nonnegative integer positions"
            )
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise CommitmentError(
                "committed values must exactly cover the verifier-derived domain"
            )

        ordered: list[bytes] = []
        cumulative_value_bytes = 0
        for local_rank, global_position in enumerate(expected):
            value = values[global_position]
            if type(value) is not bytes:
                raise CommitmentError("committed values must be bytes")
            cumulative_value_bytes += len(value)
            ordered.append(value)
            limits.enforce("max_proof_bytes", cumulative_value_bytes)

        if not ordered:
            return MerkleCommittedValues(domain, (), ())

        width = 1 << merkle_depth(len(ordered))
        leaves = [
            _leaf_hash(
                domain,
                local_rank,
                global_position,
                domain.schema_at_rank(local_rank),
                value,
            )
            for local_rank, (global_position, value) in enumerate(
                zip(expected, ordered, strict=True)
            )
        ]
        leaves.extend(
            _padding_hash(domain, local_rank)
            for local_rank in range(len(leaves), width)
        )
        levels: list[tuple[bytes, ...]] = [tuple(leaves)]
        level = leaves
        tree_level = 0
        while len(level) > 1:
            level = [
                _node_hash(
                    domain,
                    tree_level,
                    index // 2,
                    level[index],
                    level[index + 1],
                )
                for index in range(0, len(level), 2)
            ]
            levels.append(tuple(level))
            tree_level += 1
        return MerkleCommittedValues(domain, tuple(ordered), tuple(levels))

    def validate_commitment(
        self,
        domain: CommitmentDomain,
        commitment: ValueCommitment,
        limits: VerificationLimits,
    ) -> None:
        self._preflight(domain, limits)
        if not isinstance(commitment, ValueCommitment):
            raise CommitmentError("commitment has the wrong type")
        if commitment.backend_id != self.backend_id:
            raise CommitmentError("commitment names another backend")
        if commitment.value_count != domain.count:
            raise CommitmentError("commitment value_count does not match its domain")
        if type(commitment.root) is not bytes or len(commitment.root) != 32:
            raise CommitmentError("Merkle root must be exactly 32 bytes")
        if domain.count == 0 and commitment.root != empty_root(domain):
            raise CommitmentError("empty commitment has a noncanonical root")

    def verify_opening(
        self,
        domain: CommitmentDomain,
        commitment: ValueCommitment,
        opening: ValueOpening,
        limits: VerificationLimits,
    ) -> bool:
        try:
            self.validate_commitment(domain, commitment, limits)
            if not isinstance(opening, ValueOpening):
                return False
            if type(opening.path) is not tuple:
                return False
            local_rank = domain.positions.rank(opening.position)
            if len(opening.path) != merkle_depth(domain.count):
                return False
            if type(opening.value) is not bytes:
                return False
            proof_bytes = len(opening.value) + 32 * len(opening.path)
            limits.enforce("max_proof_bytes", proof_bytes)
            value_hash = _leaf_hash(
                domain,
                local_rank,
                int(opening.position),
                domain.schema_at_rank(local_rank),
                opening.value,
            )
            rank = local_rank
            for tree_level, sibling in enumerate(opening.path):
                if type(sibling) is not bytes or len(sibling) != 32:
                    return False
                parent_rank = rank >> 1
                value_hash = (
                    _node_hash(
                        domain,
                        tree_level,
                        parent_rank,
                        value_hash,
                        sibling,
                    )
                    if rank % 2 == 0
                    else _node_hash(
                        domain,
                        tree_level,
                        parent_rank,
                        sibling,
                        value_hash,
                    )
                )
                rank = parent_rank
            return value_hash == commitment.root
        except (CommitmentError, KeyError, IndexError):
            return False

    def verify_openings(
        self,
        domain: CommitmentDomain,
        commitment: ValueCommitment,
        openings: tuple[ValueOpening, ...],
        required_positions: tuple[int, ...],
        limits: VerificationLimits,
    ) -> Mapping[int, bytes]:
        self.validate_commitment(domain, commitment, limits)
        if type(openings) is not tuple or type(required_positions) is not tuple:
            raise CommitmentError("opening batches and requirements must be tuples")
        limits.enforce("max_openings", len(openings))
        if len(openings) != len(required_positions):
            raise CommitmentError("opening batch does not have exact required coverage")
        if any(type(item) is not int or item < 0 for item in required_positions):
            raise CommitmentError("required positions must be nonnegative integers")
        actual_positions = tuple(int(opening.position) for opening in openings)
        if actual_positions != required_positions:
            raise CommitmentError(
                "openings must exactly match required positions in canonical order"
            )
        if len(set(actual_positions)) != len(actual_positions):
            raise CommitmentError("opening batch contains a duplicate position")

        result: dict[int, bytes] = {}
        proof_bytes = 0
        for opening in openings:
            proof_bytes += len(opening.value) + 32 * len(opening.path)
            limits.enforce("max_proof_bytes", proof_bytes)
            if not self.verify_opening(domain, commitment, opening, limits):
                raise CommitmentError(
                    f"opening for position {opening.position} failed authentication"
                )
            result[int(opening.position)] = opening.value
        return result


def build_merkle_commitment(
    domain: CommitmentDomain,
    values: Mapping[int, bytes],
    limits: VerificationLimits | None = None,
) -> MerkleCommittedValues:
    """Convenience constructor for the transparent default backend."""

    return MerkleSha256Backend().commit(
        domain,
        values,
        VerificationLimits() if limits is None else limits,
    )
