"""Measurement primitives: timing, peak memory, power-law fits and the result records."""

from __future__ import annotations

import gc
import math
import statistics
import time
import tracemalloc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

Scalar = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class Scale:
    """How far a sweep goes.  ``quick`` keeps every benchmark to a few seconds."""

    quick: bool = False
    repeats: int = 3
    budget_s: float = 20.0
    """Soft cap on the time spent repeating one point; a point slower than this runs once."""

    def pick[T](self, quick: Sequence[T], full: Sequence[T]) -> Sequence[T]:
        return quick if self.quick else full


@dataclass(slots=True)
class Fit:
    """``y ~ coefficient * x**exponent`` fitted by least squares in log-log space."""

    exponent: float
    coefficient: float
    r2: float
    points: int

    def as_json(self) -> dict[str, float | int]:
        return {
            "exponent": self.exponent,
            "coefficient": self.coefficient,
            "r2": self.r2,
            "points": self.points,
        }


def fit_power_law(xs: Iterable[float], ys: Iterable[float]) -> Fit | None:
    """Least-squares slope of ``log y`` against ``log x`` over the positive pairs."""

    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=True)
        if x and y and x > 0 and y > 0
    ]
    if len(pairs) < 2:
        return None
    lx = [math.log(x) for x, _ in pairs]
    ly = [math.log(y) for _, y in pairs]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ly)
    ss_res = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(lx, ly, strict=True)
    )
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return Fit(slope, math.exp(intercept), r2, len(pairs))


@dataclass(slots=True)
class Point:
    """One size of one sweep: the size, the median time, the peak memory and the sizes observed."""

    x: float
    time_s: float | None = None
    peak_bytes: int | None = None
    repeats: int = 0
    extra: dict[str, Scalar] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "time_s": self.time_s,
            "peak_bytes": self.peak_bytes,
            "repeats": self.repeats,
            **self.extra,
        }


@dataclass(slots=True)
class Series:
    """One sweep: points over one size parameter, with fits of the columns that should scale."""

    name: str
    x_label: str
    points: list[Point] = field(default_factory=list)
    fit_columns: tuple[str, ...] = ("time_s",)
    note: str = ""
    columns: tuple[str, ...] = ()
    """Extra columns to show in the report, in order (defaults to every extra key)."""

    def column(self, name: str) -> list[float | None]:
        if name == "time_s":
            return [p.time_s for p in self.points]
        if name == "peak_bytes":
            return [p.peak_bytes for p in self.points]
        return [
            p.extra.get(name) if isinstance(p.extra.get(name), (int, float)) else None
            for p in self.points
        ]  # type: ignore[misc]

    def fits(self) -> dict[str, Fit | None]:
        xs = [p.x for p in self.points]
        result: dict[str, Fit | None] = {}
        for name in self.fit_columns:
            ys = self.column(name)
            pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
            result[name] = (
                fit_power_law([x for x, _ in pairs], [y for _, y in pairs])
                if pairs
                else None
            )
        return result

    def as_json(self) -> dict[str, Any]:
        keys: list[str] = list(self.columns)
        for point in self.points:
            for key in point.extra:
                if key not in keys:
                    keys.append(key)
        return {
            "name": self.name,
            "x_label": self.x_label,
            "note": self.note,
            "columns": keys,
            "fit_columns": list(self.fit_columns),
            "fits": {
                name: (fit.as_json() if fit else None)
                for name, fit in self.fits().items()
            },
            "points": [p.as_json() for p in self.points],
        }


@dataclass(slots=True)
class Benchmark:
    name: str
    title: str
    description: str
    series: list[Series] = field(default_factory=list)
    seconds: float = 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "seconds": self.seconds,
            "series": [s.as_json() for s in self.series],
        }


# -- measurement -------------------------------------------------------------------


@dataclass(slots=True)
class Measurement:
    time_s: float
    times: list[float]
    peak_bytes: int | None
    result: Any


def measure(
    fn: Callable[[], Any],
    scale: Scale,
    *,
    repeats: int | None = None,
    memory: bool = True,
    setup: Callable[[], None] | None = None,
) -> Measurement:
    """Median wall time of ``fn`` over a few repeats, then its ``tracemalloc`` peak in one more run.

    The memory run is separate because tracing slows Python two- to
    four-fold; its result is the one returned.  A point that alone exceeds
    ``scale.budget_s`` is not repeated.
    """

    wanted = scale.repeats if repeats is None else repeats
    if scale.quick:
        wanted = min(wanted, 2)
    times: list[float] = []
    result: Any = None
    gc.collect()
    for _ in range(max(1, wanted)):
        if setup is not None:
            setup()
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
        if sum(times) > scale.budget_s:
            break
    peak: int | None = None
    if memory and times[0] < scale.budget_s:
        if setup is not None:
            setup()
        gc.collect()
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            result = fn()
            _, top = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peak = max(0, top - base)
    return Measurement(statistics.median(times), times, peak, result)


def per_call[T](
    fn: Callable[[T], Any],
    items: Sequence[T],
    scale: Scale,
    *,
    repeats: int | None = None,
) -> float:
    """Median per-call latency of ``fn`` applied to every item, over a few repeats (microsecond work)."""

    wanted = scale.repeats if repeats is None else repeats
    if scale.quick:
        wanted = min(wanted, 2)
    samples: list[float] = []
    for _ in range(max(1, wanted)):
        start = time.perf_counter()
        for item in items:
            fn(item)
        samples.append((time.perf_counter() - start) / len(items))
    return statistics.median(samples)


def as_dict(value: Any) -> Any:
    return asdict(value)
