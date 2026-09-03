"""Benchmark 6 -- Merkle commitments: build throughput and memory, opening size and verification.

``MerkleTree`` hashes one leaf per position (bound to the domain, its rank,
its position and a schema) and one node per internal vertex, so the build is
``2L - 1`` SHA-256 calls and linear in ``L``; an opening is ``depth`` sibling
hashes and its verification ``depth + 1`` hashes, logarithmic in ``L``.
``commit_weights`` (the per-model ``kappa_W``) is timed alongside.
"""

from __future__ import annotations

import random
import time

from veritor.core import RangeIndexedDomain, VerificationLimits, encode_value
from veritor.protocol.domains import commit_weights
from veritor.protocol.merkle import (
    CommitmentDomain,
    MerkleTree,
    merkle_depth,
    verify_opening,
)

from ._harness import Benchmark, Point, Scale, Series, measure, per_call
from ._synthetic import GATE_SET

BINDING = b"\x33" * 32
LIMITS = VerificationLimits()
CALLS = 300


def _domain(count: int) -> CommitmentDomain:
    return CommitmentDomain(BINDING, 7, RangeIndexedDomain(count))


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "merkle",
        "Merkle commitments: build, open, verify",
        "`MerkleTree(domain, values, schema)` over `L` 16-bit leaves on a range domain: build wall time and "
        "`tracemalloc` peak, then the per-call latency of `tree.open(position)` and `verify_opening`, with "
        "the opening's size in bytes.  `commit_weights_s` builds `kappa_W` for `L` weights through "
        "`commit_weights`.",
    )

    leaves = scale.pick(
        [10**2, 10**3, 10**4, 10**5], [10**2, 10**3, 10**4, 10**5, 10**6, 2 * 10**6]
    )
    series = Series(
        "build_vs_leaves",
        "leaves",
        fit_columns=("time_s", "peak_bytes", "commit_weights_s"),
        note="Linear in `L`: `2L - 1` domain-bound SHA-256 calls.  `values_per_s` is the build throughput, "
        "`bytes_per_leaf` the retained tree (`_values` and every level) per leaf.",
    )
    for count in leaves:
        rng = random.Random(count)
        values = {k: encode_value(16, rng.randrange(1 << 16)) for k in range(count)}
        domain = _domain(count)
        build = measure(
            lambda d=domain, v=values: MerkleTree(d, v, lambda _p: "u16"), scale
        )
        weights = [rng.randrange(1 << 16) for _ in range(count)]
        kappa = measure(
            lambda w=weights: commit_weights(GATE_SET, w), scale, memory=False
        )
        series.points.append(
            Point(
                count,
                time_s=build.time_s,
                peak_bytes=build.peak_bytes,
                repeats=len(build.times),
                extra={
                    "values_per_s": count / build.time_s,
                    "hashes_per_s": (2 * (1 << merkle_depth(count)) - 1) / build.time_s,
                    "bytes_per_leaf": (build.peak_bytes or 0) / count,
                    "commit_weights_s": kappa.time_s,
                    "depth": merkle_depth(count),
                },
            )
        )
    bench.series.append(series)

    series = Series(
        "open_verify_vs_leaves",
        "leaves",
        fit_columns=("time_s", "verify_s", "proof_bytes"),
        note="`time_s` is `tree.open(position)` (a rank lookup and `depth` sibling reads), `verify_s` is "
        "`verify_opening` (`depth + 1` hashes): both `O(log L)`.  `proof_bytes` is `2 + 32 * depth`.",
    )
    for count in leaves:
        rng = random.Random(count)
        values = {k: encode_value(16, rng.randrange(1 << 16)) for k in range(count)}
        domain = _domain(count)
        tree = MerkleTree(domain, values, lambda _p: "u16")
        positions = [rng.randrange(count) for _ in range(CALLS)]
        openings = [tree.open(p) for p in positions]
        commitment = tree.commitment
        series.points.append(
            Point(
                count,
                time_s=per_call(tree.open, positions, scale),
                extra={
                    "verify_s": per_call(
                        lambda o, d=domain, c=commitment: verify_opening(
                            d, c, o, "u16", LIMITS
                        ),
                        openings,
                        scale,
                    ),
                    "proof_bytes": len(openings[0].value) + 32 * len(openings[0].path),
                    "depth": merkle_depth(count),
                },
            )
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
