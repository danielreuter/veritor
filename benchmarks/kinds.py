"""Benchmark 3 -- the per-kind table and the lazy address sets of the index.

``Index.kinds()`` / ``kind_table()`` against the number of definitions and
against the number of runs in a unit's ``Out``; ``Index.boundary()`` and
``Index.interior(r)`` (construction, ``contains``, ``rank``, ``unrank``) and
the unit lookups against the number of replay units (RUs), for a ``repeat``
layout (descent divides) and an unrolled one (descent bisects a step list).
"""

from __future__ import annotations

import random
import time

from veritor.compile import Compiler

from ._harness import Benchmark, Point, Scale, Series, measure, per_call
from ._synthetic import (
    GATE_SET,
    INPUT,
    deep_repeat,
    many_definitions,
    runs_unit,
    unrolled_units,
)

CALLS = 300


def _address_set_point(compiled, x: float, scale: Scale, extra: dict) -> Point:
    index = compiled.index
    rng = random.Random(0)
    units = index.replay_units.count
    n = compiled.circuit.n
    boundary = measure(index.boundary, scale, memory=False).result
    ranks = [rng.randrange(boundary.count) for _ in range(CALLS)]
    members = [boundary.unrank(r) for r in ranks]
    addresses = [rng.randrange(n) for _ in range(CALLS)]
    unit_ranks = [rng.randrange(units) for _ in range(CALLS)]
    vu_ranks = [rng.randrange(index.verification_unit_count) for _ in range(CALLS)]
    interior_of = [index.interior(r) for r in unit_ranks[:32]]
    interior_ranks = [
        (dom, rng.randrange(max(1, dom.count))) for dom in interior_of if dom.count
    ]
    return Point(
        x,
        time_s=per_call(boundary.rank, members, scale),
        extra={
            "boundary_build_s": measure(index.boundary, scale, memory=False).time_s,
            "boundary_unrank_s": per_call(boundary.unrank, ranks, scale),
            "boundary_contains_s": per_call(boundary.contains, addresses, scale),
            "interior_build_s": per_call(index.interior, unit_ranks, scale),
            # an RU whose one VU declares exactly the RU's outputs (the unrolled layout) has no interior
            "interior_unrank_s": per_call(
                lambda pair: pair[0].unrank(pair[1]), interior_ranks, scale
            )
            if interior_ranks
            else None,
            "interior_contains_s": per_call(
                lambda pair: pair[0].contains(pair[0].interval.start + pair[1]),
                interior_ranks,
                scale,
            )
            if interior_ranks
            else None,
            "unit_s": per_call(index.replay_units.unit, unit_ranks, scale),
            "owner_s": per_call(index.replay_units.owner, addresses, scale),
            "verification_unit_s": per_call(index.verification_unit, vu_ranks, scale),
            "verification_units_s": per_call(
                index.verification_units, unit_ranks, scale
            ),
            "replay_units": units,
            "boundary_count": boundary.count,
            "n": n,
            **extra,
        },
    )


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "kinds",
        "Kind table and lazy address sets of the index",
        "`Index.kinds()` / `kind_table()` wall time against the description, and the per-call latency of the "
        "boundary and interior address sets and the unit lookups against the number of RUs.",
    )
    compiler = Compiler(GATE_SET)

    counts = scale.pick([8, 32, 128, 512], [8, 32, 128, 512, 2048, 8192])
    series = Series(
        "kind_table_vs_definitions",
        "kinds",
        fit_columns=("time_s", "kinds_s", "peak_bytes"),
        note="`Index.kind_table()` on a root calling `D` distinct RU kinds: one row per kind, linear in the "
        "description.  `kinds_s` is `Index.kinds()` alone (the table adds the totals).",
    )
    for count in counts:
        compiled = compiler.compile(many_definitions(count), INPUT)
        table = measure(compiled.index.kind_table, scale)
        kinds = measure(compiled.index.kinds, scale, memory=False)
        series.points.append(
            Point(
                len(table.result.rows),
                time_s=table.time_s,
                peak_bytes=table.peak_bytes,
                repeats=len(table.times),
                extra={
                    "kinds_s": kinds.time_s,
                    "definitions": count + 3,
                    "n": compiled.circuit.n,
                },
            )
        )
    bench.series.append(series)

    runs_wanted = scale.pick([2, 8, 32, 128], [2, 8, 32, 128, 256])
    series = Series(
        "vs_output_runs",
        "runs in Out",
        fit_columns=(
            "time_s",
            "boundary_rank_s",
            "boundary_contains_miss_s",
            "boundary_unrank_s",
            "interior_build_s",
            "interior_contains_s",
            "interior_unrank_s",
        ),
        note="1000 copies of one RU whose declared `Out` resolves to `R` runs (outputs at irregular gaps); every "
        "gate is a one-gate VU, so the interior is the RU's gates minus its `Out`.  `time_s` is `kind_table()`.  "
        "`boundary.rank` and a boundary `contains` miss scan the runs (`O(R)`), `boundary.unrank` bisects "
        "(`O(log R)`); the interior's `contains`/`rank` subtract the RU outputs below the address (`O(R)`) after an "
        "`O(depth)` descent to the VU, and its `unrank` bisects the steps with the same subtraction per probe "
        "(`O(R log |R_r|)`).",
    )
    for wanted in runs_wanted:
        compiled = compiler.compile(runs_unit(wanted, 1000), INPUT)
        index = compiled.index
        unit = index.replay_units.unit(500)
        runs = len(unit.frame.definition.out_runs)
        boundary = index.boundary()
        interior = index.interior(500)
        rng = random.Random(0)
        out_addresses = [
            unit.frame.base
            + unit.frame.definition.out_offset(
                rng.randrange(unit.frame.definition.out_count)
            )
            for _ in range(CALLS)
        ]
        misses = [
            int(interior.unrank(rng.randrange(interior.count))) for _ in range(CALLS)
        ]
        ranks = [boundary.rank(a) for a in out_addresses]
        interior_ranks = [rng.randrange(interior.count) for _ in range(CALLS)]
        table = measure(index.kind_table, scale)
        series.points.append(
            Point(
                runs,
                time_s=table.time_s,
                peak_bytes=table.peak_bytes,
                repeats=len(table.times),
                extra={
                    "boundary_rank_s": per_call(boundary.rank, out_addresses, scale),
                    "boundary_contains_miss_s": per_call(
                        boundary.contains, misses, scale
                    ),
                    "boundary_unrank_s": per_call(boundary.unrank, ranks, scale),
                    "interior_build_s": per_call(index.interior, [500] * CALLS, scale),
                    "interior_contains_s": per_call(interior.contains, misses, scale),
                    "interior_rank_s": per_call(interior.rank, misses, scale),
                    "interior_unrank_s": per_call(
                        interior.unrank, interior_ranks, scale
                    ),
                    "unit_size": unit.size,
                    "out_count": unit.frame.definition.out_count,
                    "interior_count": interior.count,
                },
            )
        )
    bench.series.append(series)

    units = scale.pick(
        [100, 1000, 10_000, 100_000],
        [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000],
    )
    series = Series(
        "address_sets_vs_units_repeat",
        "replay units",
        fit_columns=(
            "time_s",
            "boundary_unrank_s",
            "boundary_contains_s",
            "interior_build_s",
            "interior_unrank_s",
            "unit_s",
            "owner_s",
            "verification_unit_s",
        ),
        note="`repeat(U, block)` of an 8-cell RU: every lookup descends by division, so all latencies should be "
        "flat in `U`.  `time_s` is `boundary.rank`.",
    )
    for count in units:
        compiled = compiler.compile(deep_repeat((8, count)), INPUT)
        series.points.append(
            _address_set_point(compiled, count, scale, {"layout": "repeat"})
        )
    bench.series.append(series)

    units = scale.pick([100, 1000, 10_000], [100, 1000, 10_000, 50_000])
    series = Series(
        "address_sets_vs_units_unrolled",
        "replay units",
        fit_columns=(
            "time_s",
            "boundary_unrank_s",
            "boundary_contains_s",
            "interior_build_s",
            "unit_s",
            "owner_s",
            "verification_unit_s",
        ),
        note="A root of `U` separate `call` steps of one RU: descent bisects the root's step list, `O(log U)`; "
        "the description itself is `O(U)` (about 120 bytes per step, so 10^5 unrolled steps exceed "
        "`max_description_bytes = 10 MB`; the sweep stops at 50,000).  `time_s` is `boundary.rank`.",
    )
    for count in units:
        compiled = compiler.compile(unrolled_units(count), INPUT)
        series.points.append(
            _address_set_point(compiled, count, scale, {"layout": "unrolled"})
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
