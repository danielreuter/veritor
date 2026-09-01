"""Merkle tree: openings verify, and every kind of tampering fails."""

import pytest

from veritor.merkle import LEAF_INSTRUCTION, LEAF_VALUE, MerkleTree, verify_leaf


def payloads(n: int) -> list[bytes]:
    return [bytes([i]) * 4 for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 17])
def test_roundtrip_all_indices(n):
    tree = MerkleTree.build(LEAF_VALUE, payloads(n))
    for i in range(n):
        path = tree.prove(i)
        assert verify_leaf(tree.root, LEAF_VALUE, i, payloads(n)[i], path, n)


def test_wrong_payload_rejected():
    tree = MerkleTree.build(LEAF_VALUE, payloads(8))
    path = tree.prove(3)
    assert not verify_leaf(tree.root, LEAF_VALUE, 3, b"\xff\xff\xff\xff", path, 8)


def test_wrong_index_rejected():
    # Serving the (authentic) content of cell 3 when asked for cell 5 fails,
    # because the leaf hash binds the index.
    tree = MerkleTree.build(LEAF_VALUE, payloads(8))
    path = tree.prove(3)
    assert not verify_leaf(tree.root, LEAF_VALUE, 5, payloads(8)[3], path, 8)


def test_wrong_root_rejected():
    tree = MerkleTree.build(LEAF_VALUE, payloads(8))
    other = MerkleTree.build(LEAF_VALUE, payloads(9))
    path = tree.prove(0)
    assert not verify_leaf(other.root, LEAF_VALUE, 0, payloads(8)[0], path, 8)


def test_truncated_path_rejected():
    tree = MerkleTree.build(LEAF_VALUE, payloads(8))
    path = tree.prove(0)
    assert not verify_leaf(tree.root, LEAF_VALUE, 0, payloads(8)[0], path[:-1], 8)


def test_out_of_range_index_rejected():
    tree = MerkleTree.build(LEAF_VALUE, payloads(8))
    path = tree.prove(0)
    assert not verify_leaf(tree.root, LEAF_VALUE, 100, payloads(8)[0], path, 8)


def test_domain_separation():
    # The same payloads under different tags commit to different roots, so an
    # instruction opening can never be replayed as a value opening.
    a = MerkleTree.build(LEAF_INSTRUCTION, payloads(4))
    b = MerkleTree.build(LEAF_VALUE, payloads(4))
    assert a.root != b.root
