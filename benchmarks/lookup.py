"""Benchmark 2 -- lazy gate lookup: ``circuit[i]`` latency against ``n`` and against depth.

``DescriptionCircuit.__getitem__`` descends from the root, bisecting within a
step list and dividing within a ``repeat``, so its cost is ``O(depth *
arity)`` and independent of ``n``.  Three access patterns are timed: uniform
random addresses, a sequential window, and a strided walk.
"""

from __future__ import annotations

import random
import time

from veritor import Compile
from veritor.compile import Compiler

from ._harness import Benchmark, Point, Scale, Series, per_call
from ._synthetic import CLUSTER_LADDER, GATE_SET, INPUT, ISA, deep_repeat

CALLS = 300


def _patterns(n: int, seed: int = 0) -> dict[str, list[int]]:
    rng = random.Random(seed)
    start = rng.randrange(max(1, n - CALLS))
    stride = max(1, n // CALLS)
    return {
        "random": [rng.randrange(n) for _ in range(CALLS)],
        "sequential": [min(n - 1, start + k) for k in range(CALLS)],
        "strided": [min(n - 1, k * stride) for k in range(CALLS)],
    }


def _lookup_point(compiled, x: float, scale: Scale, extra: dict) -> Point:
    circuit, index = compiled.circuit, compiled.index
    n = circuit.n
    patterns = _patterns(n)
    times = {
        name: per_call(circuit.__getitem__, addresses, scale)
        for name, addresses in patterns.items()
    }
    owner_us = per_call(index.replay_units.owner, patterns["random"], scale)
    rng = random.Random(1)
    ranks = [rng.randrange(index.replay_units.count) for _ in range(CALLS)]
    unit_us = per_call(index.replay_units.unit, ranks, scale)
    return Point(
        x,
        time_s=times["random"],
        extra={
            "sequential_s": times["sequential"],
            "strided_s": times["strided"],
            "owner_s": owner_us,
            "unit_s": unit_us,
            "n": n,
            "depth": compiled.circuit.root.depth,
            **extra,
        },
    )


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "lookup",
        "Lazy gate lookup: `circuit[i]` latency",
        "Per-call latency (seconds) of `DescriptionCircuit.__getitem__` for random, sequential and strided "
        "addresses, of `Index.replay_units.owner(address)` and of `Index.replay_units.unit(k)`.  "
        "`time_s` is the random-access latency.",
    )
    compiler = Compiler(GATE_SET)

    # vs n at fixed depth: repeat(k, repeat(k, repeat(k, cell)))
    factors = scale.pick([10, 100, 1000, 10_000], [10, 100, 1000, 10_000, 100_000])
    series = Series(
        "vs_n",
        "n (gates)",
        fit_columns=("time_s", "sequential_s", "strided_s", "owner_s", "unit_s"),
        note="Three `repeat` levels of factor `k`, `n = 3 k**3 + 1`: depth is fixed, so every latency should be "
        "flat in `n` (exponent near 0).",
    )
    for k in factors:
        compiled = compiler.compile(deep_repeat((k, k, k)), INPUT)
        series.points.append(
            _lookup_point(compiled, compiled.circuit.n, scale, {"k": k})
        )
    bench.series.append(series)

    # vs depth at k = 2
    depths = scale.pick([2, 4, 8, 16, 32], [2, 4, 8, 16, 32, 48])
    series = Series(
        "vs_depth",
        "depth",
        fit_columns=("time_s", "sequential_s", "strided_s", "owner_s", "unit_s"),
        note="`repeat(2, ...)` nested `depth` times (`n = 3 * 2**depth + 1`, up to ~10**15): latency should be "
        "linear in depth (exponent near 1).",
    )
    for depth in depths:
        compiled = compiler.compile(deep_repeat((2,) * depth), INPUT)
        series.points.append(_lookup_point(compiled, depth, scale, {}))
    bench.series.append(series)

    # the toy cluster: real hierarchies of fixed depth
    cases = CLUSTER_LADDER[: 4 if scale.quick else 6]
    series = Series(
        "cluster",
        "n (gates)",
        fit_columns=("time_s", "owner_s", "unit_s"),
        note="`ClusterG` circuits of growing width: the decoder's hierarchy has a fixed depth, so lookups stay "
        "flat while `n` grows four orders of magnitude.",
    )
    for case in cases:
        compilation = Compile(
            case.cluster, case.requests, case.advice, ISA, max_advice_bits=1 << 24
        )
        compiled = compilation.compiled
        series.points.append(
            _lookup_point(compiled, compiled.circuit.n, scale, {"case": case.label})
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
