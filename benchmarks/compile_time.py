"""Benchmark 1 -- ``Compile``: time and memory against description size, not against ``n``.

Three sweeps over the toy cluster (``ClusterG``: model width, run length,
slot count) and two synthetic ones: deep ``repeat`` nesting whose flattened
gate count reaches ``10**12`` while the description stays a few kilobytes,
and a root calling thousands of distinct definitions.
"""

from __future__ import annotations

import time

from veritor import Compile
from veritor.compile import Compiler

from ._harness import Benchmark, Point, Scale, Series, measure
from ._synthetic import (
    GATE_SET,
    INPUT,
    ISA,
    DecoderCase,
    decoder_case,
    deep_repeat,
    many_definitions,
)


def _definitions(description: bytes) -> int:
    return description.count(b'"body":')


def _cluster_point(case: DecoderCase, x: float, scale: Scale) -> Point:
    constructor = case.cluster
    x_inputs, advice = case.requests, case.advice
    trace = measure(lambda: constructor(x_inputs, advice), scale, memory=False)
    description, inputs = trace.result
    compiler = Compiler(ISA)
    compiled_only = measure(lambda: compiler.compile(description, inputs), scale)
    whole = measure(
        lambda: Compile(constructor, x_inputs, advice, ISA, max_advice_bits=1 << 24),
        scale,
        memory=False,
    )
    compiled = compiled_only.result
    return Point(
        x,
        time_s=compiled_only.time_s,
        peak_bytes=compiled_only.peak_bytes,
        repeats=len(compiled_only.times),
        extra={
            "trace_s": trace.time_s,
            "compile_total_s": whole.time_s,
            "description_bytes": len(description),
            "definitions": _definitions(description),
            "n": compiled.circuit.n,
            "replay_units": compiled.index.replay_units.count,
            "verification_units": compiled.index.verification_unit_count,
            "advice_bytes": len(advice),
        },
    )


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "compile",
        "Compile: description in, lazy circuit and index out",
        "Wall time of `Compiler.compile` (parse, validate, summarize, mark check, build `Index` and "
        "`DescriptionCircuit`) with the constructor's trace time (`G(x, a)`) and the whole "
        "`Compile(G, x, a)` alongside.  The x-axis is the parameter swept; `n` is the flattened gate count, "
        "`description_bytes` what the compiler actually reads.",
    )

    # (a) ClusterG: model width
    widths = scale.pick([4, 8, 16, 32], [4, 8, 16, 32, 64, 128])
    series = Series(
        "cluster_d_model",
        "d_model",
        fit_columns=("time_s", "description_bytes", "n", "trace_s"),
        note="ClusterG with 2 layers, 4 requests of 4 + 4 tokens on 2 slots; compile time should follow "
        "`description_bytes` (roughly linear in d_model: the tracer unrolls each dot product), not `n` (quadratic).",
    )
    for d in widths:
        case = decoder_case(
            f"d{d}", d_model=d, layers=2, prompt=4, max_new=4, requests=4, slots=2
        )
        series.points.append(_cluster_point(case, d, scale))
    bench.series.append(series)

    # (b) ClusterG: run length (requests through the same slots)
    counts = scale.pick([2, 8, 32], [2, 8, 16, 32, 64])
    series = Series(
        "cluster_requests",
        "requests",
        fit_columns=("time_s", "description_bytes", "n", "trace_s", "advice_bytes"),
        note="ClusterG d_model=8, 1 layer, 2 slots, requests of 3 + 3 tokens; every extra wave adds root steps and "
        "schedule bytes but no new kinds.  The sweep stops at 64 requests: from 96 on, the root's declared "
        "outputs (every request's tokens) resolve to more than `CompilationLimits.max_output_runs = 256` runs "
        "and the description is rejected, a limit of the toy constructor's output layout rather than of the "
        "compiler.",
    )
    for count in counts:
        case = decoder_case(
            f"r{count}",
            d_model=8,
            layers=1,
            prompt=3,
            max_new=3,
            requests=count,
            slots=2,
        )
        series.points.append(_cluster_point(case, count, scale))
    bench.series.append(series)

    # (c) ClusterG: slot count (distinct step shapes)
    slots = scale.pick([1, 2, 4, 8], [1, 2, 4, 8, 16, 32])
    series = Series(
        "cluster_slots",
        "slots",
        fit_columns=("time_s", "description_bytes", "n"),
        note="ClusterG d_model=8, 1 layer, `2 * slots` requests of 3 + 3 tokens: a step kind per distinct occupant "
        "tuple, so the description grows with the number of slots.",
    )
    for count in slots:
        case = decoder_case(
            f"s{count}",
            d_model=8,
            layers=1,
            prompt=3,
            max_new=3,
            requests=2 * count,
            slots=count,
        )
        series.points.append(_cluster_point(case, count, scale))
    bench.series.append(series)

    # (d) deep repeat nesting: n up to 10**12 from a description of a few KB
    depths = scale.pick([1, 2, 4, 6], [1, 2, 4, 6, 8, 10, 12])
    series = Series(
        "deep_repeat",
        "n (gates)",
        fit_columns=("time_s", "description_bytes", "peak_bytes"),
        note="`repeat(10, ...)` nested `depth` times over a 3-gate VU: `n = 3 * 10**depth + 1`.  The compiler "
        "never touches a gate, so time and memory follow the description (`O(depth)`), not `n`.",
    )
    compiler = Compiler(GATE_SET)
    for depth in depths:
        description = deep_repeat((10,) * depth)
        result = measure(lambda d=description: compiler.compile(d, INPUT), scale)
        compiled = result.result
        series.points.append(
            Point(
                compiled.circuit.n,
                time_s=result.time_s,
                peak_bytes=result.peak_bytes,
                repeats=len(result.times),
                extra={
                    "depth": depth,
                    "description_bytes": len(description),
                    "definitions": _definitions(description),
                    "replay_units": compiled.index.replay_units.count,
                },
            )
        )
    bench.series.append(series)

    # (e) many distinct definitions
    counts = scale.pick([8, 32, 128, 512], [8, 32, 128, 512, 2048, 8192])
    series = Series(
        "definitions",
        "definitions",
        fit_columns=("time_s", "description_bytes", "peak_bytes"),
        note="A root calling `D` distinct RU kinds once each (`repeat(i + 1, cell)`): compile time linear in the "
        "number of definitions, i.e. in the description.",
    )
    for count in counts:
        description = many_definitions(count)
        result = measure(lambda d=description: compiler.compile(d, INPUT), scale)
        compiled = result.result
        series.points.append(
            Point(
                _definitions(description),
                time_s=result.time_s,
                peak_bytes=result.peak_bytes,
                repeats=len(result.times),
                extra={"description_bytes": len(description), "n": compiled.circuit.n},
            )
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
