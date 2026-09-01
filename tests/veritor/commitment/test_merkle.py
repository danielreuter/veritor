from __future__ import annotations

from dataclasses import replace

import pytest

from veritor.commitment import (
    MERKLE_SHA256_V1,
    CommitmentDomain,
    CommitmentError,
    CommitmentOwner,
    MerkleSha256Backend,
    ValueCommitment,
    ValueCommitmentRegistry,
    empty_root,
)
from veritor.core import ResourceLimit, VerificationLimits


def domain(**changes) -> CommitmentDomain:
    fields = {
        "backend_id": MERKLE_SHA256_V1,
        "session_id": b"commitment-test-session",
        "compiled_result_digest": "11" * 32,
        "policy_digest": "22" * 32,
        "statement_digest": b"A" * 32,
        "phase_digest": b"B" * 32,
        "owner": CommitmentOwner.boundary(),
        "positions": (9, 2, 100),
        "value_schema_ids": ("u8", "u16", "u8"),
    }
    fields.update(changes)
    return CommitmentDomain(**fields)


def test_indexed_merkle_uses_verifier_rank_for_arbitrary_position_domains():
    backend = MerkleSha256Backend()
    commitment_domain = domain()
    values = {9: b"\x01", 2: b"\x00\x02", 100: b"\x03"}

    first = backend.commit(commitment_domain, values, VerificationLimits())
    second = backend.commit(
        commitment_domain, dict(reversed(values.items())), VerificationLimits()
    )

    assert first.commitment == second.commitment
    assert first.commitment.backend_id == MERKLE_SHA256_V1
    assert first.commitment.root.hex() == (
        "465d9b5cfd372c64cca3d4987b9a88ae5c68173b532d92985f00aa521dfdc17a"
    )
    for position in commitment_domain.positions:
        assert backend.verify_opening(
            commitment_domain,
            first.commitment,
            first.open(int(position)),
            VerificationLimits(),
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"session_id": b"another-session"},
        {"compiled_result_digest": "33" * 32},
        {"policy_digest": "44" * 32},
        {"statement_digest": b"C" * 32},
        {"phase_digest": b"D" * 32},
        {"owner": CommitmentOwner.replay_unit(0)},
        {"positions": (2, 9, 100)},
        {"value_schema_ids": ("u8", "u8", "u8")},
    ],
)
def test_every_domain_identity_component_invalidates_openings(changed):
    backend = MerkleSha256Backend()
    original = domain()
    tree = backend.commit(
        original,
        {9: b"\x01", 2: b"\x00\x02", 100: b"\x03"},
        VerificationLimits(),
    )

    assert not backend.verify_opening(
        domain(**changed),
        tree.commitment,
        tree.open(9),
        VerificationLimits(),
    )


def test_merkle_rejects_value_position_and_path_substitution():
    backend = MerkleSha256Backend()
    commitment_domain = domain()
    tree = backend.commit(
        commitment_domain,
        {9: b"\x01", 2: b"\x00\x02", 100: b"\x03"},
        VerificationLimits(),
    )
    opening = tree.open(2)

    assert not backend.verify_opening(
        commitment_domain,
        tree.commitment,
        replace(opening, value=b"\x00\x04"),
        VerificationLimits(),
    )
    assert not backend.verify_opening(
        commitment_domain,
        tree.commitment,
        replace(opening, position=9),
        VerificationLimits(),
    )
    assert not backend.verify_opening(
        commitment_domain,
        tree.commitment,
        replace(opening, path=opening.path[:-1]),
        VerificationLimits(),
    )


def test_empty_domain_has_one_domain_bound_root_and_no_openings():
    backend = MerkleSha256Backend()
    commitment_domain = domain(positions=(), value_schema_ids=())
    tree = backend.commit(commitment_domain, {}, VerificationLimits())

    assert tree.commitment.root == empty_root(commitment_domain)
    backend.validate_commitment(
        commitment_domain,
        tree.commitment,
        VerificationLimits(),
    )
    with pytest.raises(CommitmentError, match="noncanonical"):
        backend.validate_commitment(
            commitment_domain,
            ValueCommitment(MERKLE_SHA256_V1, b"\x00" * 32, 0),
            VerificationLimits(),
        )
    with pytest.raises(CommitmentError, match="outside"):
        tree.open(9)


def test_empty_byte_value_is_valid_when_the_bound_schema_allows_it():
    backend = MerkleSha256Backend()
    commitment_domain = domain(
        positions=(7,),
        value_schema_ids=("unit",),
    )
    tree = backend.commit(
        commitment_domain,
        {7: b""},
        VerificationLimits(),
    )

    assert backend.verify_opening(
        commitment_domain,
        tree.commitment,
        tree.open(7),
        VerificationLimits(),
    )


def test_commitment_batches_require_exact_ordered_unique_coverage():
    backend = MerkleSha256Backend()
    commitment_domain = domain()
    tree = backend.commit(
        commitment_domain,
        {9: b"\x01", 2: b"\x00\x02", 100: b"\x03"},
        VerificationLimits(),
    )

    assert backend.verify_openings(
        commitment_domain,
        tree.commitment,
        (tree.open(9), tree.open(100)),
        (9, 100),
        VerificationLimits(),
    ) == {9: b"\x01", 100: b"\x03"}
    with pytest.raises(CommitmentError, match="canonical order"):
        backend.verify_openings(
            commitment_domain,
            tree.commitment,
            (tree.open(100), tree.open(9)),
            (9, 100),
            VerificationLimits(),
        )
    with pytest.raises(CommitmentError, match="coverage"):
        backend.verify_openings(
            commitment_domain,
            tree.commitment,
            (tree.open(9),),
            (9, 100),
            VerificationLimits(),
        )


def test_commitment_registry_is_an_immutable_explicit_allowlist():
    registry = ValueCommitmentRegistry.with_defaults()

    assert registry.backend_ids == (MERKLE_SHA256_V1,)
    assert isinstance(registry.require(MERKLE_SHA256_V1), MerkleSha256Backend)
    assert registry.resolve("unknown") is None
    with pytest.raises(CommitmentError, match="duplicate"):
        ValueCommitmentRegistry((MerkleSha256Backend(), MerkleSha256Backend()))


def test_commitment_work_obeys_verifier_resource_limits():
    with pytest.raises(ResourceLimit):
        MerkleSha256Backend().commit(
            domain(),
            {9: b"\x01", 2: b"\x00\x02", 100: b"\x03"},
            VerificationLimits(max_positions=2),
        )
