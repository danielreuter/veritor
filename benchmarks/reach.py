"""Benchmark 8 -- ``output_reach`` (``veritor.core.index``) vs description size.

``output_reach`` is one of the top-down passes behind ``Index.kinds()`` (with
``transient_ports``); it bounds, per definition, the circuit output bits a
copy can reach, through a per-definition transitive closure over the steps
kept as intervals of steps and swept with a segment tree.  The sweeps
separate the shapes of a definition's step graph: a chain of dependent call
steps (each reads the previous one), independent call steps all reading one
step, many distinct definitions, deep ``repeat`` nesting, and the toy cluster
descriptions.  ``parse_s`` (``parse_description``) and ``kinds_s``
(``Index.kinds()``, which includes the reach) are recorded next to it.
"""

from __future__ import annotations

import time

from veritor.compile.description import parse_description
from veritor.core.index import Index, output_reach, transient_ports
from veritor.core.limits import CompilationLimits

from ._harness import Benchmark, Point, Scale, Series, measure
from ._synthetic import (
    CLUSTER_LADDER,
    GATE_SET,
    ISA,
    chain_steps,
    deep_repeat,
    deep_repeat_gates,
    many_definitions,
    unrolled_units,
)

LIMITS = CompilationLimits()
COLUMNS = ("time_s", "peak_bytes", "transient_ports_s", "kinds_s", "parse_s")


def _point(
    description: bytes, x: float, scale: Scale, gate_set=GATE_SET, **extra
) -> Point:
    parse = measure(
        lambda: parse_description(description, gate_set, LIMITS), scale, memory=False
    )
    root = parse.result.root
    reach = measure(lambda: output_reach(root), scale)
    transient = measure(lambda: transient_ports(root), scale, memory=False)
    index = Index(root, LIMITS)
    kinds = measure(index.kinds, scale, memory=False)
    return Point(
        x,
        time_s=reach.time_s,
        peak_bytes=reach.peak_bytes,
        repeats=len(reach.times),
        extra={
            "description_bytes": len(description),
            "definitions": len(kinds.result),
            "root_steps": len(root.steps),
            "n": index.n,
            "transient_ports_s": transient.time_s,
            "kinds_s": kinds.time_s,
            "parse_s": parse.time_s,
            **extra,
        },
    )


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "reach",
        "`output_reach` vs description size",
        "`time_s` is `output_reach(root)` on the parsed root; `transient_ports_s` the sibling pass, "
        "`kinds_s` the whole `Index.kinds()` (both passes plus the per-kind summaries), `parse_s` "
        "`parse_description`.  `root_steps` is the step count of the root definition, `definitions` "
        "the reachable kinds.",
    )

    steps = scale.pick([64, 256, 1024], [64, 256, 1024, 4096, 8192])
    series = Series(
        "chain_vs_steps",
        "root steps",
        fit_columns=COLUMNS,
        note="A root with `S` sequential `call` steps of one 3-gate RU, each reading the previous step's "
        "output (a decode chain).  The closure `Down(j)` is every later step, one interval `[j, S)` in "
        "the sweep of `_step_reach`: `O(S log S)` (two range additions and one extraction on the segment "
        "tree per step), against `Θ(S)` for the parse.  The bitmask closure this replaced was `Θ(S³ / w)` "
        "here (14.3 s at `S = 8192`).",
    )
    for count in steps:
        series.points.append(_point(chain_steps(count), count, scale))
    bench.series.append(series)

    series = Series(
        "independent_vs_steps",
        "root steps",
        fit_columns=COLUMNS,
        note="A root with `S` independent `call` steps of one RU, every step reading the input step "
        "(siblings over one broadcast step, as requests over a weights step).  `Down(j) = {j}` and "
        "`Down(input) = [0, S + 1)`, so the pass is `O(S log S)`; the bitmask closure ORed `S` masks of "
        "`S` bits here, `Θ(S² / w)`.",
    )
    for count in steps:
        series.points.append(_point(unrolled_units(count), count, scale))
    bench.series.append(series)

    definitions = scale.pick([64, 256, 1024], [64, 256, 1024, 4096])
    series = Series(
        "definitions_vs_count",
        "definitions",
        fit_columns=COLUMNS,
        note="A root calling `D` distinct RU definitions once each, all reading the input: linear in `D` "
        "(one `_step_reach` per definition with call steps, here only the root).",
    )
    for count in definitions:
        series.points.append(_point(many_definitions(count), count, scale))
    bench.series.append(series)

    depths = scale.pick([4, 12, 20], [4, 8, 12, 16, 20, 24])
    series = Series(
        "deep_repeat_vs_depth",
        "depth",
        fit_columns=COLUMNS,
        note="`repeat` nesting `depth` levels deep with branching 4 (`n = 4^depth * 8`): one definition per "
        "level, one step each, so the pass is linear in `depth` whatever `n` is.",
    )
    for depth in depths:
        shape = (4,) * depth
        series.points.append(
            _point(deep_repeat(shape), depth, scale, gates=deep_repeat_gates(shape))
        )
    bench.series.append(series)

    series = Series(
        "cluster_vs_description",
        "description bytes",
        fit_columns=COLUMNS,
        note="The `ClusterG` ladder: the description grows with the layer count and the schedule "
        "(steps of the decode loop), not with the model width.",
    )
    for case in CLUSTER_LADDER[: 3 if scale.quick else 5]:
        description, _ = case.cluster(case.requests, case.advice)
        series.points.append(
            _point(description, len(description), scale, gate_set=ISA, case=case.label)
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
