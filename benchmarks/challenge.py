"""Benchmark 5 -- challenge sampling: ``bernoulli_subset`` against ``N`` and against ``K``.

The selection is ``Binomial(N, p)`` by CDF inversion in 512-bit fixed point
followed by Floyd's subset from exactly ``K`` further HMAC draws, so the cost
must be ``O(K log N)``: flat in ``N`` at fixed expected ``K = pN``, linear in
``K`` at fixed ``N``.  ``derive_sample_selection`` over the verification
units (VUs) of the selected replay units (RUs) is timed on a compiled
circuit as well.
"""

from __future__ import annotations

import time
from fractions import Fraction

from veritor.compile import Compiler
from veritor.core import VerificationLimits, VerificationPolicy
from veritor.protocol.challenge import (
    bernoulli_subset,
    derive_replay_selection,
    derive_sample_selection,
)

from ._harness import Benchmark, Point, Scale, Series, measure
from ._synthetic import GATE_SET, INPUT, deep_repeat

SEED = b"\x11" * 32
PHASE = b"\x22" * 32
LIMITS = VerificationLimits()


def _sample(count: int, probability: Fraction, scale: Scale) -> Point:
    result = measure(
        lambda: bernoulli_subset(
            SEED, b"q/replay-unit", PHASE, count, probability, LIMITS
        ),
        scale,
        memory=False,
    )
    selected = len(result.result)
    return Point(
        count,
        time_s=result.time_s,
        repeats=len(result.times),
        extra={
            "selected": selected,
            "expected": float(count * probability),
            "per_selected_s": result.time_s / max(1, selected),
            "denominator_bits": probability.denominator.bit_length(),
        },
    )


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "challenge",
        "Challenge sampling: binomial count and Floyd subset",
        "Wall time of `bernoulli_subset(seed, stage, phase, N, p)` -- `K ~ Binomial(N, p)` by 512-bit CDF "
        "inversion, then a uniform `K`-subset from `K` HMAC-SHA256 draws with rejection -- and of the two "
        "derivations the verifier runs on a compiled circuit.",
    )

    counts = scale.pick(
        [10**3, 10**5, 10**7, 10**9],
        [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**11, 10**13, 10**15],
    )
    series = Series(
        "vs_N_fixed_K",
        "N (candidates)",
        fit_columns=("time_s", "per_selected_s"),
        note="`p = 64 / N`: sixty-four selections expected whatever `N`; time should be flat in `N` up to the "
        "`O(log N)` initial power (the exponent should be near 0, far below 1).",
    )
    for count in counts:
        series.points.append(_sample(count, Fraction(64, count), scale))
    bench.series.append(series)

    selected = scale.pick([1, 10, 100, 1000], [1, 10, 100, 1000, 10_000, 100_000])
    series = Series(
        "vs_K_fixed_N",
        "K (expected selections)",
        fit_columns=("time_s",),
        note="`N = 10**6` candidates, `p = K / N`: one HMAC draw (plus rejections) per selection, so time is "
        "linear in `K` (exponent near 1) once `K` dominates the fixed cost of the binomial inversion.",
    )
    for k in selected:
        point = _sample(10**6, Fraction(k, 10**6), scale)
        point.x = k
        series.points.append(point)
    bench.series.append(series)

    # both derivations on a compiled tiling: J over the RUs, T over the VUs inside J
    units = scale.pick(
        [10**3, 10**5, 10**7], [10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
    )
    series = Series(
        "derive_selections_vs_units",
        "replay units",
        fit_columns=("time_s", "sample_s"),
        note="`derive_replay_selection` (`q = 64 / #RU`) then `derive_sample_selection` (`s = 1/2`) on a "
        "`repeat(U, block)` of 8-cell RUs; the verifier's `max_units` limit (10**6) is raised for the sweep.  "
        "`time_s` is the replay selection, `sample_s` the VU selection over the 8 * |J| candidates.",
    )
    compiler = Compiler(GATE_SET)
    limits = VerificationLimits(max_units=10**9)
    for count in units:
        compiled = compiler.compile(deep_repeat((8, count)), INPUT)
        policy = VerificationPolicy(
            Fraction(64, compiled.index.replay_units.count), Fraction(1, 2)
        )
        replay = measure(
            lambda c=compiled, p=policy: derive_replay_selection(
                SEED, PHASE, c, p, limits
            ),
            scale,
            memory=False,
        )
        chosen = replay.result
        sample = measure(
            lambda c=compiled, j=chosen, p=policy: derive_sample_selection(
                SEED, PHASE, c, j, p, limits
            ),
            scale,
            memory=False,
        )
        series.points.append(
            Point(
                compiled.index.replay_units.count,
                time_s=replay.time_s,
                repeats=len(replay.times),
                extra={
                    "sample_s": sample.time_s,
                    "selected_replay_units": len(chosen),
                    "sampled_verification_units": len(sample.result),
                    "verification_units": compiled.index.verification_unit_count,
                },
            )
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
