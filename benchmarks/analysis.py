"""Benchmark 4 -- ``Bound`` (knapsack on and off), ``Cost``, ``expected_work`` and ``Optimize``.

All four are folds over the kind table, so they are timed against the
number of kinds: on the serving tables of :mod:`veritor.evaluation.serving`
from toy dimensions up to the 70B-class ``FRONTIER_SHAPE``, and on
synthetic tables with a controlled number of replay kinds.  ``Bound`` is
also swept over its cost grid (``max_buckets``) and ``Optimize`` over the
policy grid.
"""

from __future__ import annotations

import time
from fractions import Fraction

from veritor.analysis import BoundOptions, PolicyGrid, bound, cost, optimize
from veritor.core import KindSummary, KindTable, VerificationPolicy, identity_digest
from veritor.core.description import REPLAY, VERIFICATION
from veritor.evaluation.frontier import FRONTIER_OPTIONS, FRONTIER_SHAPE
from veritor.evaluation.serving import ServingShape, serving_table
from veritor.protocol.parameters import expected_work

from ._harness import Benchmark, Point, Scale, Series, measure

POLICY = VerificationPolicy(Fraction(1, 128), Fraction(1, 8))
ETA = Fraction(1, 100)

LADDER: tuple[tuple[str, ServingShape], ...] = (
    (
        "toy",
        ServingShape(
            vocab=8,
            d_model=4,
            heads=2,
            layers=1,
            prompt=3,
            generated=3,
            requests=4,
            batch=2,
        ),
    ),
    (
        "small",
        ServingShape(
            vocab=256,
            d_model=64,
            heads=4,
            layers=4,
            prompt=16,
            generated=16,
            requests=32,
            batch=8,
        ),
    ),
    (
        "mid",
        ServingShape(
            vocab=4096,
            d_model=1024,
            heads=16,
            layers=16,
            prompt=128,
            generated=128,
            requests=256,
            batch=32,
        ),
    ),
    ("frontier-70B", FRONTIER_SHAPE),
)


def synthetic_table(
    replay_kinds: int, copies: int = 1000, units_per_copy: int = 64, width: int = 16
) -> KindTable:
    """A root over ``replay_kinds`` distinct RU kinds, each ``copies`` copies of ``units_per_copy`` VUs of one kind."""

    vu = KindSummary(
        kind="vu",
        role=VERIFICATION,
        copies=replay_kinds * copies * units_per_copy,
        size=4,
        replay_cost=4,
        proof_cost=4,
        input_count=2,
        out_count=1,
        out_bits=width,
        interior_count=0,
        reach_bits=width,
        ancestor_bits=width,  # enclosed by its RU's one-value interface
        source_inputs=0,
        source_weights=0,
        min_depth=2,
        max_depth=2,
        children=(),
        verification_units=1,
        verification_kinds=(("vu", 1),),
        closed=False,
    )
    total_out = replay_kinds * copies
    rows = [vu]
    for k in range(replay_kinds):
        rows.append(
            KindSummary(
                kind=f"ru{k}",
                role=REPLAY,
                copies=copies,
                size=4 * units_per_copy,
                replay_cost=4 * units_per_copy,
                proof_cost=4 * units_per_copy,
                input_count=1,
                out_count=1,
                out_bits=width,
                interior_count=units_per_copy
                - 1,  # every VU output but the one the RU declares
                reach_bits=width * (1 + k % 7),
                ancestor_bits=width * total_out,
                source_inputs=0,
                source_weights=0,
                min_depth=1,
                max_depth=1,
                children=(("vu", units_per_copy),),
                verification_units=units_per_copy,
                verification_kinds=(("vu", units_per_copy),),
                closed=False,
            )
        )
    root = KindSummary(
        kind="root",
        role=None,
        copies=1,
        size=replay_kinds * copies * 4 * units_per_copy,
        replay_cost=replay_kinds * copies * 4 * units_per_copy,
        proof_cost=replay_kinds * copies * 4 * units_per_copy,
        input_count=0,
        out_count=total_out,
        out_bits=width * total_out,
        interior_count=total_out * (units_per_copy - 1),
        reach_bits=width * total_out,
        ancestor_bits=width * total_out,
        source_inputs=0,
        source_weights=0,
        min_depth=0,
        max_depth=0,
        children=tuple((f"ru{k}", copies) for k in range(replay_kinds)),
        verification_units=replay_kinds * copies * units_per_copy,
        verification_kinds=(("vu", replay_kinds * copies * units_per_copy),),
        closed=True,
    )
    rows.insert(0, root)
    return KindTable(
        rows=tuple(rows),
        root="root",
        n=root.size,
        input_count=0,
        weight_count=0,
        replay_unit_count=replay_kinds * copies,
        digest=identity_digest(
            "benchmarks/synthetic-table/v1",
            {"kinds": replay_kinds, "copies": copies, "units": units_per_copy},
        ),
    )


def _fold_point(
    table: KindTable, x: float, scale: Scale, extra: dict, *, knapsack: bool = True
) -> Point:
    laplace = measure(lambda: bound(table, POLICY, ETA, FRONTIER_OPTIONS), scale)
    replay_rows = [row for row in table.rows if row.role == REPLAY]
    replay_copies = sum(row.copies for row in replay_rows)
    point = Point(
        x,
        time_s=laplace.time_s,
        peak_bytes=laplace.peak_bytes,
        repeats=len(laplace.times),
        extra={
            "rows": len(table.rows),
            "replay_kinds": len(replay_rows),
            "replay_copies": replay_copies,
            "ru_gates": (
                sum(row.copies * row.size for row in replay_rows) / replay_copies
                if replay_copies
                else None
            ),
            "ru_positions": (
                sum(row.copies * row.interior_count for row in replay_rows)
                / replay_copies
                if replay_copies
                else None
            ),
            "n": table.n,
            "laplace_bits": laplace.result.bits,
            **extra,
        },
    )
    if knapsack:
        knap = measure(lambda: bound(table, POLICY, ETA), scale, memory=False)
        point.extra["knapsack_s"] = knap.time_s
        point.extra["knapsack_bits"] = knap.result.bits
        point.extra["buckets"] = knap.result.buckets
    point.extra["cost_s"] = measure(
        lambda: cost(table, POLICY), scale, memory=False
    ).time_s
    point.extra["work_s"] = measure(
        lambda: expected_work(table, POLICY, 16), scale, memory=False
    ).time_s
    return point


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "analysis",
        "Bound, Cost, expected_work and Optimize",
        "Folds over the kind table.  `time_s` is `Bound` with the Laplace fold alone "
        "(`FRONTIER_OPTIONS`: `knapsack=False, max_buckets=2**22`); `knapsack_s` is `Bound` with the default "
        "options (knapsack on a 2048-bucket grid plus Laplace); `cost_s` and `work_s` are `Cost` and "
        "`expected_work`.  Policy `q = 1/128, s = 1/8`, `eta = 1/100`.",
    )

    ladder = LADDER[:2] if scale.quick else LADDER
    for replay, verification in (("request", "row"), ("step", "row"), ("cell", "gate")):
        series = Series(
            f"serving_{replay}_{verification}",
            "kinds",
            fit_columns=("time_s", "knapsack_s", "cost_s", "work_s", "build_s"),
            note=f"`serving_table(shape, {replay!r}, {verification!r})` from toy dimensions to the 70B-class "
            "frontier shape (`d_model = 8192`, 80 layers, 2048 requests of 512 + 512 tokens, 2.7e17 gates); "
            "the number of kinds grows with the number of distinct decode contexts.  `build_s` is the table "
            "construction.",
        )
        for label, shape in ladder:
            build = measure(
                lambda s=shape, r=replay, v=verification: serving_table(s, r, v),
                scale,
                memory=False,
            )
            table = build.result
            # the knapsack on the cell-level frontier table takes minutes: Laplace only there
            knapsack = not (replay == "cell" and len(table.rows) > 1000)
            point = _fold_point(
                table,
                len(table.rows),
                scale,
                {"shape": label, "build_s": build.time_s},
                knapsack=knapsack,
            )
            series.points.append(point)
        bench.series.append(series)

    counts = scale.pick([4, 16], [4, 16, 64, 256, 1024])
    series = Series(
        "synthetic_vs_replay_kinds",
        "replay kinds",
        fit_columns=("time_s", "knapsack_s", "cost_s", "work_s"),
        note="A synthetic table of `K` distinct RU kinds (1000 copies each, 64 VUs of one kind per copy): the "
        "Laplace fold is one series per kind and ~130 evaluations of a `K`-term sum (linear in `K`); the "
        "knapsack forms a cost polynomial per kind and convolves `K` of them on the grid.",
    )
    for count in counts:
        table = synthetic_table(count)
        # the knapsack costs ~50 ms per replay kind here: minutes at 1024 kinds, so Laplace only there
        series.points.append(
            _fold_point(table, count, scale, {}, knapsack=count <= 256)
        )
    bench.series.append(series)

    buckets = scale.pick([128, 512, 2048], [128, 512, 2048, 8192, 32768])
    table = serving_table(LADDER[1][1], "step", "row")
    series = Series(
        "knapsack_vs_buckets",
        "max_buckets",
        fit_columns=("time_s", "peak_bytes"),
        note="`Bound` with the knapsack on, on the `small` `step/row` table (116 kinds, 17 replay kinds), against "
        "the cost grid: `sparse_power` and the convolutions are `O(buckets * terms)` to `O(buckets**2)`.  "
        "`time_s` is the whole `Bound`.",
    )
    for count in buckets:
        options = BoundOptions(max_buckets=count)
        result = measure(lambda o=options: bound(table, POLICY, ETA, o), scale)
        series.points.append(
            Point(
                count,
                time_s=result.time_s,
                peak_bytes=result.peak_bytes,
                repeats=len(result.times),
                extra={
                    "buckets_used": result.result.buckets,
                    "bits": result.result.bits,
                    "knapsack_bits": result.result.knapsack_bits,
                },
            )
        )
    bench.series.append(series)

    steps = scale.pick([1, 2, 4], [1, 2, 4, 8, 16])
    series = Series(
        "optimize_vs_grid",
        "grid points",
        fit_columns=("time_s",),
        note="`Optimize` (`max_bits` objective, Laplace-only bound) on the `small` `step/row` table over "
        "`PolicyGrid.uniform(steps)`, `(steps + 1)**2` points: one `Cost` and one `Bound` per point.",
    )
    for count in steps:
        grid = PolicyGrid.uniform(count)
        result = measure(
            lambda g=grid: optimize(
                table, ETA, g, max_bits=1e9, bound_options=FRONTIER_OPTIONS
            ),
            scale,
            memory=False,
        )
        evaluated = result.result.evaluated if result.result else 0
        series.points.append(
            Point(
                (count + 1) ** 2,
                time_s=result.time_s,
                repeats=len(result.times),
                extra={"steps": count, "evaluated": evaluated},
            )
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
