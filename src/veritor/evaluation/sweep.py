"""The frontier sweep in parallel, resumable, with a manifest.

:func:`veritor.evaluation.frontier.sweep` prices every partition, grid policy
and ``eta`` in one process; at the paper's shape a point takes seconds to
minutes, so the full ``DEFAULT_PARTITIONS x DEFAULT_GRID x DEFAULT_ETAS``
sweep is run here across a process pool instead.  The points are the same
(:func:`~veritor.evaluation.frontier.price` on the same serving table), they
are written in a canonical order (partitions, then ``q``, then ``s``, then
``eta``), the file is checkpointed as points complete and a rerun picks up
where the last one stopped, and the file carries a *manifest*: the commit the
package was at (and whether the tree was dirty), the package version, the
shape, the :class:`~veritor.analysis.BoundOptions`, the etas, the grid, the
partitions and the wall time.  ``docs/data/frontier-70b.json`` was produced
this way::

    python -m veritor.evaluation.sweep docs/data/frontier-70b.json --workers 10

A smoke test of the same path is ``--shape toy --grid small --levels
request/row,cell/gate --etas 1/2`` (seconds, not hours).
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from veritor.analysis import BoundOptions, PolicyGrid
from veritor.core import KindTable, VerificationPolicy, exact_fraction

from .frontier import (
    DEFAULT_ETAS,
    DEFAULT_GRID,
    DEFAULT_PARTITIONS,
    FRONTIER_OPTIONS,
    FRONTIER_SHAPE,
    Point,
    load,
    load_manifest,
    price,
    save,
)
from .serving import ReplayLevel, ServingShape, VerificationLevel, serving_table

Key = tuple[str, str, Fraction, Fraction, Fraction]
"""``(replay level, verification level, q, s, eta)``: what identifies a point."""

SMALL_GRID = PolicyGrid(q=(Fraction(1, 2), Fraction(1, 8)), s=(1, Fraction(1, 8)))
"""A ``2 x 2`` grid for smoke tests of the sweep itself."""

TOY_SHAPE = ServingShape(vocab=8, d_model=4, heads=2, layers=1, prompt=2, generated=3, requests=4, batch=2)
"""A run small enough that the whole default sweep takes seconds."""

SHAPES: dict[str, ServingShape] = {"70b": FRONTIER_SHAPE, "toy": TOY_SHAPE}
GRIDS: dict[str, PolicyGrid] = {"full": DEFAULT_GRID, "small": SMALL_GRID}


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """What a sweep prices: a shape under some partitions, at a grid of policies and some etas."""

    shape: ServingShape = FRONTIER_SHAPE
    levels: tuple[tuple[ReplayLevel, VerificationLevel], ...] = DEFAULT_PARTITIONS
    grid: PolicyGrid = DEFAULT_GRID
    etas: tuple[Fraction, ...] = DEFAULT_ETAS
    options: BoundOptions = FRONTIER_OPTIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "etas", tuple(exact_fraction(eta, name="eta") for eta in self.etas))
        object.__setattr__(self, "levels", tuple((str(r), str(v)) for r, v in self.levels))
        if not self.levels:
            raise ValueError("a sweep needs at least one partition")
        if not self.etas:
            raise ValueError("a sweep needs at least one eta")

    def keys(self) -> list[Key]:
        """Every point of the sweep in canonical order: partition, then ``q``, then ``s``, then ``eta``."""

        return [
            (replay, verification, policy.q, policy.s, eta)
            for replay, verification in self.levels
            for policy in self.grid.policies()
            for eta in self.etas
        ]

    def describe(self) -> dict[str, object]:
        """The spec as JSON: the shape, the options, the etas, the grid and the partitions."""

        return {
            "shape": self.shape.manifest,
            "options": asdict(self.options),
            "etas": [str(eta) for eta in self.etas],
            "grid": {"q": [str(q) for q in self.grid.q], "s": [str(s) for s in self.grid.s]},
            "partitions": [f"{replay}/{verification}" for replay, verification in self.levels],
        }


def key_of(point: Point) -> Key:
    return (point.replay, point.verification, point.q, point.s, point.eta)


# -- the worker --------------------------------------------------------------------------

_TABLES: dict[tuple[ServingShape, str, str], KindTable] = {}
"""Serving tables built so far in this process: one per partition, shared by its points."""


def price_key(shape: ServingShape, key: Key, options: BoundOptions = FRONTIER_OPTIONS) -> Point:
    """The point at ``key``: :func:`~veritor.evaluation.frontier.price` on the partition's table.

    A module-level function of picklable arguments, so a process pool can run
    it; the table of a partition is built once per process and reused for
    every point of that partition the process is handed.
    """

    replay, verification, q, s, eta = key
    table = _TABLES.get((shape, replay, verification))
    if table is None:
        table = serving_table(shape, replay, verification)  # type: ignore[arg-type]
        _TABLES[(shape, replay, verification)] = table
    return price(table, shape, replay, verification, VerificationPolicy(q, s), eta, options=options)


# -- the manifest -----------------------------------------------------------------------


def git_state(directory: Path | None = None) -> tuple[str | None, bool | None]:
    """``(commit, dirty)`` of the repository holding ``directory`` (this package by default).

    ``(None, None)`` when there is no ``git`` or no repository, as for an
    installed wheel.
    """

    cwd = directory or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def package_version() -> str | None:
    try:
        return version("veritor")
    except PackageNotFoundError:
        return None


def manifest(
    spec: SweepSpec,
    *,
    wall_seconds: float,
    workers: int,
    points: int,
    previous: dict[str, object] | None = None,
) -> dict[str, object]:
    """The provenance of a file of points.

    ``wall_seconds`` is the wall time of this run; a resumed run adds the
    ``previous`` manifest's wall time, so the total is the time it took to
    produce the file over all its runs, and counts the runs.
    """

    commit, dirty = git_state()
    runs = 1
    total = wall_seconds
    if previous is not None:
        earlier_runs, earlier_seconds = previous.get("runs", 1), previous.get("wall_seconds", 0.0)
        runs += earlier_runs if isinstance(earlier_runs, int) else 1
        total += earlier_seconds if isinstance(earlier_seconds, (int, float)) else 0.0
    return {
        "commit": commit,
        "dirty": dirty,
        "version": package_version(),
        "completed_at": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "wall_seconds": total,
        "runs": runs,
        "workers": workers,
        "points": points,
        **spec.describe(),
    }


# -- the sweep ---------------------------------------------------------------------------


def resume(spec: SweepSpec, out: Path) -> tuple[dict[Key, Point], dict[str, object] | None]:
    """The points of ``out`` that belong to ``spec``, by key, and its manifest.

    Nothing when the file does not exist.  A file for another shape is an
    error: its points would be silently wrong for this one.  Points of
    partitions, policies or etas outside the spec are dropped from the resumed
    set (the file is rewritten from the spec's canonical order).
    """

    if not out.exists():
        return {}, None
    shape, points = load(out)
    if shape != spec.shape:
        raise ValueError(f"{out} holds points of another shape: {shape.manifest} != {spec.shape.manifest}")
    wanted = set(spec.keys())
    return {key_of(point): point for point in points if key_of(point) in wanted}, load_manifest(out)


def parallel_sweep(
    spec: SweepSpec,
    out: Path,
    *,
    workers: int = 1,
    checkpoint_every: int = 20,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Point], int]:
    """Price every point of ``spec`` not already in ``out`` and write the merged file.

    Returns the points in canonical order and how many were computed by this
    call; a second call on a finished file computes nothing and rewrites the
    same points with an updated manifest.  With ``workers <= 1`` the points
    are priced in this process; otherwise in a pool, this process being the
    single writer, checkpointing the file every ``checkpoint_every`` points.
    """

    order = spec.keys()
    rank = {key: i for i, key in enumerate(order)}
    by_key, previous = resume(spec, out)
    missing = [key for key in order if key not in by_key]
    say = log or (lambda _: None)
    say(f"{len(by_key)} points present, {len(missing)} missing, {workers} workers -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    def merged() -> list[Point]:
        return [by_key[key] for key in sorted(by_key, key=rank.__getitem__)]

    def checkpoint() -> None:
        record = manifest(
            spec,
            wall_seconds=time.perf_counter() - started,
            workers=workers,
            points=len(by_key),
            previous=previous,
        )
        save(merged(), spec.shape, out, manifest=record)

    def report(done: int, point: Point) -> None:
        say(
            f"[{time.perf_counter() - started:6.0f}s {done:4d}/{len(missing)}] "
            f"{point.replay:8s}/{point.verification:5s} q={point.q!s:7s} s={point.s!s:6s} "
            f"eta={point.eta!s:10s} U={point.bits:14.1f} ({100 * point.fraction:7.3f}%) "
            f"overhead={float(point.overhead):.5f} recompute={float(point.recompute):.5f} "
            f"work={float(point.work):.5f} ({point.seconds:.1f}s)"
        )

    done = 0
    if workers <= 1:
        for key in missing:
            point = price_key(spec.shape, key, spec.options)
            by_key[key] = point
            done += 1
            report(done, point)
            if done % checkpoint_every == 0:
                checkpoint()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(price_key, spec.shape, key, spec.options): key for key in missing}
            for future in as_completed(futures):
                point = future.result()
                by_key[futures[future]] = point
                done += 1
                report(done, point)
                if done % checkpoint_every == 0:
                    checkpoint()
    checkpoint()
    say(f"done: {len(by_key)} points ({done} new) in {time.perf_counter() - started:.0f}s")
    return merged(), done


# -- the command line ---------------------------------------------------------------------


def _parse_levels(text: str) -> tuple[tuple[ReplayLevel, VerificationLevel], ...]:
    levels = []
    for item in text.split(","):
        replay, _, verification = item.strip().partition("/")
        if not replay or not verification:
            raise argparse.ArgumentTypeError(f"a partition is written REPLAY/VERIFICATION, not {item!r}")
        levels.append((replay, verification))
    return tuple(levels)  # type: ignore[arg-type]


def _parse_fractions(text: str) -> tuple[Fraction, ...]:
    return tuple(exact_fraction(Fraction(item.strip()), name="eta") for item in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m veritor.evaluation.sweep",
        description="Price the honest server's frontier in parallel; resumes a partially written OUT.",
    )
    parser.add_argument("out", type=Path, help="the JSON file to write (and resume from)")
    parser.add_argument("--workers", type=int, default=1, help="processes in the pool (1: this process)")
    parser.add_argument("--shape", choices=sorted(SHAPES), default="70b", help="the serving shape")
    parser.add_argument(
        "--levels",
        type=_parse_levels,
        default=DEFAULT_PARTITIONS,
        help="comma-separated partitions, e.g. request/row,cell/gate (default: every admissible one)",
    )
    parser.add_argument("--grid", choices=sorted(GRIDS), default="full", help="the (q, s) grid")
    parser.add_argument(
        "--etas", type=_parse_fractions, default=DEFAULT_ETAS, help="comma-separated etas, e.g. 1/2,1/100"
    )
    parser.add_argument("--checkpoint-every", type=int, default=20, help="write OUT every N points")
    return parser


def spec_from_args(args: argparse.Namespace) -> SweepSpec:
    return SweepSpec(
        shape=SHAPES[args.shape], levels=tuple(args.levels), grid=GRIDS[args.grid], etas=tuple(args.etas)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = spec_from_args(args)

    def say(line: str) -> None:
        print(line, flush=True)

    parallel_sweep(spec, args.out, workers=args.workers, checkpoint_every=args.checkpoint_every, log=say)
    return 0


__all__ = [
    "GRIDS",
    "SHAPES",
    "SMALL_GRID",
    "TOY_SHAPE",
    "Key",
    "SweepSpec",
    "git_state",
    "key_of",
    "manifest",
    "package_version",
    "parallel_sweep",
    "price_key",
    "resume",
]

if __name__ == "__main__":
    sys.exit(main())
