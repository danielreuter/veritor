"""A numpy-backed, parallel-hashed weight commitment with the ``MerkleTree`` interface, root-identical to ``commit_weights``.

``commit_weights`` builds the model's tree out of Python objects: for GPT-2
Small's 124.5 M weights that is tens of gigabytes and many minutes.  This
class computes the same leaves (``CommitmentDomain.leaf``) and nodes
(``CommitmentDomain.node``), in chunks across processes, and stores each
level as one ``[n, 32]`` ``uint8`` array (8 GB for the full model).  The
prover opens weights through ``open`` exactly as from a ``MerkleTree``; the
verifier is untouched and checks the openings against the same root.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from veritor.core import GateSet, encode_value, make_pinned_gate_set
from veritor.protocol import Weights, weight_domain
from veritor.protocol.merkle import _PAD, CommitmentDomain, _hash, _uint, merkle_depth
from veritor.protocol.messages import Commitment, Opening

_CHUNK = 1 << 18


def _domain(count: int) -> CommitmentDomain:
    return weight_domain(make_pinned_gate_set(), count)


def _leaf_chunk(args: tuple[int, int, bytes, int]) -> tuple[int, bytes]:
    count, start, words, width = args
    domain = _domain(count)
    values = np.frombuffer(words, dtype=np.uint16)
    schema = f"u{width}"
    out = bytearray()
    for offset, word in enumerate(values.tolist()):
        rank = start + offset
        out += domain.leaf(rank, rank, schema, encode_value(width, word))
    return start, bytes(out)


def _pad_chunk(args: tuple[int, int, int]) -> tuple[int, bytes]:
    count, start, n = args
    domain = _domain(count)
    out = bytearray()
    for rank in range(start, start + n):
        out += _hash(_PAD, domain.domain_id, _uint(rank))
    return start, bytes(out)


def _node_chunk(args: tuple[int, int, int, bytes]) -> tuple[int, bytes]:
    count, depth, first, level = args  # level: the 2 * n children as bytes
    domain = _domain(count)
    out = bytearray()
    for j in range(len(level) // 64):
        left = level[64 * j : 64 * j + 32]
        right = level[64 * j + 32 : 64 * j + 64]
        out += domain.node(depth, first + j, left, right)
    return first, bytes(out)


class WeightTree:
    """The weight commitment of a model given as BF16/uint16 words, ``MerkleTree``-compatible."""

    def __init__(
        self, gate_set: GateSet, words: np.ndarray, processes: int | None = None
    ) -> None:
        if gate_set.digest != make_pinned_gate_set().digest:
            raise ValueError("WeightTree is built for the pinned gate set")
        words = np.ascontiguousarray(words, dtype=np.uint16)
        count = int(words.shape[0])
        self._words = words
        self._domain = weight_domain(gate_set, count)
        width = gate_set["weight"].width
        processes = processes or max(1, (os.cpu_count() or 2) - 2)
        depth = merkle_depth(count)
        full = 1 << depth
        with ProcessPoolExecutor(processes) as pool:
            leaves = np.empty((full, 32), dtype=np.uint8)
            leaf_jobs = [
                (count, s, words[s : s + _CHUNK].tobytes(), width)
                for s in range(0, count, _CHUNK)
            ]
            for start, blob in pool.map(_leaf_chunk, leaf_jobs, chunksize=1):
                n = len(blob) // 32
                leaves[start : start + n] = np.frombuffer(blob, dtype=np.uint8).reshape(
                    n, 32
                )
            pads = [
                (count, s, min(_CHUNK, full - s)) for s in range(count, full, _CHUNK)
            ]
            for start, blob in pool.map(_pad_chunk, pads, chunksize=1):
                n = len(blob) // 32
                leaves[start : start + n] = np.frombuffer(blob, dtype=np.uint8).reshape(
                    n, 32
                )
            levels = [leaves]
            level_depth = 0
            while levels[-1].shape[0] > 1:
                children = levels[-1]
                parents = np.empty((children.shape[0] // 2, 32), dtype=np.uint8)
                pairs = children.reshape(-1, 64)
                node_jobs = [
                    (count, level_depth, f, pairs[f : f + _CHUNK].tobytes())
                    for f in range(0, pairs.shape[0], _CHUNK)
                ]
                for first, blob in pool.map(_node_chunk, node_jobs, chunksize=1):
                    n = len(blob) // 32
                    parents[first : first + n] = np.frombuffer(
                        blob, dtype=np.uint8
                    ).reshape(n, 32)
                levels.append(parents)
                level_depth += 1
        self._levels = levels
        self._width = width

    @property
    def domain(self) -> CommitmentDomain:
        return self._domain

    @property
    def commitment(self) -> Commitment:
        return Commitment(self._levels[-1][0].tobytes(), self._domain.count)

    @property
    def weights(self) -> Weights:
        return Weights(self._domain.count, self.commitment.root)

    def open(self, position: int) -> Opening:
        rank = int(position)
        if not 0 <= rank < self._domain.count:
            raise KeyError(position)
        path = []
        cursor = rank
        for level in self._levels[:-1]:
            path.append(level[cursor ^ 1].tobytes())
            cursor >>= 1
        return Opening(
            position, encode_value(self._width, int(self._words[rank])), tuple(path)
        )
