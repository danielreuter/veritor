"""Merkle commitments over the two tapes.

A Merkle tree lets the prover commit to a long list of items with a single
32-byte "root", and later *open* any single item with a short authentication
path (one sibling hash per tree level). Once the root is fixed, the prover
cannot change any item without changing the root -- this is the binding
property the whole protocol rests on: the verifier only samples *after* the
roots are fixed.

Two design details that matter even in a toy:

  - Domain separation: instruction leaves and value leaves are hashed with
    different tags, and leaves are hashed differently from internal nodes.
    Without this, an opening for one kind of object could be replayed as an
    opening for another.

  - Index binding: each leaf hash includes its position, so the prover
    cannot serve the (authentic) content of cell j when asked for cell i.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

# Domain-separation tags.
LEAF_INSTRUCTION = b"\x00"  # leaves of the instruction tape
LEAF_VALUE = b"\x01"  # leaves of the value tape
_NODE = b"\x02"  # internal nodes

# Trees are padded to a power of two with this placeholder hash. A real
# leaf hash is a SHA-256 output over a tagged preimage, so a placeholder
# cannot be opened as a real leaf.
_EMPTY = bytes(32)


def _leaf_hash(tag: bytes, index: int, payload: bytes) -> bytes:
    return hashlib.sha256(tag + struct.pack(">Q", index) + payload).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE + left + right).digest()


def depth_for(num_leaves: int) -> int:
    """Tree depth (= authentication path length) for `num_leaves` leaves."""
    d = 0
    while (1 << d) < num_leaves:
        d += 1
    return d


@dataclass(frozen=True)
class MerkleTree:
    """A binary SHA-256 Merkle tree over a list of byte payloads.

    `levels[0]` is the (padded) list of leaf hashes; `levels[-1]` is the
    single root. The prover keeps the whole tree so it can answer openings;
    the verifier only ever sees the root and individual openings.
    """

    tag: bytes
    num_leaves: int
    levels: tuple[tuple[bytes, ...], ...]

    @staticmethod
    def build(tag: bytes, payloads: list[bytes]) -> MerkleTree:
        assert len(payloads) >= 1, "cannot commit to an empty tape"
        n = len(payloads)
        d = depth_for(n)
        level = [_leaf_hash(tag, i, p) for i, p in enumerate(payloads)]
        level += [_EMPTY] * ((1 << d) - n)
        levels = [tuple(level)]
        while len(level) > 1:
            level = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
            levels.append(tuple(level))
        return MerkleTree(tag=tag, num_leaves=n, levels=tuple(levels))

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def prove(self, index: int) -> tuple[bytes, ...]:
        """Authentication path for leaf `index`: one sibling per level."""
        assert 0 <= index < self.num_leaves, "index out of range"
        path = []
        i = index
        for level in self.levels[:-1]:
            path.append(level[i ^ 1])  # sibling of the current node
            i >>= 1
        return tuple(path)


def verify_leaf(
    root: bytes,
    tag: bytes,
    index: int,
    payload: bytes,
    path: tuple[bytes, ...],
    num_leaves: int,
) -> bool:
    """Check that `payload` is leaf `index` of the tree with this `root`.

    The verifier knows `num_leaves` independently (from its own re-trace of
    the public program), so a lying prover cannot present a differently
    sized tree.
    """
    if not (0 <= index < num_leaves):
        return False
    if len(path) != depth_for(num_leaves):
        return False
    h = _leaf_hash(tag, index, payload)
    i = index
    for sibling in path:
        h = _node_hash(h, sibling) if i % 2 == 0 else _node_hash(sibling, h)
        i >>= 1
    return h == root
