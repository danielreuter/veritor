"""Stable value-commitment API."""

from .merkle import (
    MerkleCommittedValues,
    MerkleSha256Backend,
    build_merkle_commitment,
    empty_root,
    merkle_depth,
)
from .registry import ValueCommitmentRegistry
from .types import (
    COMMITMENT_PROTOCOL_ID,
    MERKLE_SHA256_V1,
    CommitmentDomain,
    CommitmentError,
    CommitmentOwner,
    CommitmentOwnerKind,
    ValueCommitment,
    ValueCommitmentBackend,
    ValueCommitmentProver,
    ValueOpening,
    commitment_manifest,
    domain_canonical_bytes,
    opening_manifest,
)

__all__ = [
    "COMMITMENT_PROTOCOL_ID",
    "MERKLE_SHA256_V1",
    "CommitmentDomain",
    "CommitmentError",
    "CommitmentOwner",
    "CommitmentOwnerKind",
    "MerkleCommittedValues",
    "MerkleSha256Backend",
    "ValueCommitment",
    "ValueCommitmentBackend",
    "ValueCommitmentProver",
    "ValueCommitmentRegistry",
    "ValueOpening",
    "build_merkle_commitment",
    "commitment_manifest",
    "domain_canonical_bytes",
    "empty_root",
    "merkle_depth",
    "opening_manifest",
]
