"""The parallel sweep: its manifest round-trips and a rerun does no work.

Everything here is at toy dimensions on a ``2 x 2`` grid, so a sweep takes a
fraction of a second; the pool path is exercised once with two workers.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from veritor.evaluation import ServingShape
from veritor.evaluation.frontier import Point, load, load_manifest, save, sweep
from veritor.evaluation.sweep import (
    SMALL_GRID,
    TOY_SHAPE,
    SweepSpec,
    key_of,
    parallel_sweep,
    price_key,
)

SERIAL = {"grid": SMALL_GRID, "etas": (Fraction(1, 2),)}
"""``frontier.sweep``'s arguments for the same grid and eta as ``SPEC``."""

SPEC = SweepSpec(
    shape=TOY_SHAPE, levels=(("request", "row"), ("cell", "gate")), grid=SMALL_GRID, etas=(Fraction(1, 2),)
)


def test_the_manifest_records_the_run_and_round_trips_through_save_and_load(tmp_path: Path) -> None:
    out = tmp_path / "points.json"

    points, computed = parallel_sweep(SPEC, out, workers=1)

    assert computed == len(points) == len(SPEC.keys()) == 2 * 4 * 1
    shape, back = load(out)
    manifest = load_manifest(out)
    assert shape == TOY_SHAPE and back == points
    assert manifest is not None
    assert manifest["shape"] == TOY_SHAPE.manifest
    assert manifest["options"] == {"max_buckets": 1 << 22, "resolution": 16, "max_errors": 256, "knapsack": False}
    assert manifest["etas"] == ["1/2"]
    assert manifest["grid"] == {"q": ["1/2", "1/8"], "s": ["1", "1/8"]}
    assert manifest["partitions"] == ["request/row", "cell/gate"]
    assert manifest["points"] == 8 and manifest["runs"] == 1 and manifest["workers"] == 1
    assert isinstance(manifest["wall_seconds"], float) and manifest["wall_seconds"] >= 0
    assert manifest["version"] == "0.2.0"
    # in a checkout the commit is known and the tree may be dirty; from a wheel neither is
    assert (manifest["commit"] is None) == (manifest["dirty"] is None)
    assert manifest["commit"] is None or (len(manifest["commit"]) == 40 and isinstance(manifest["dirty"], bool))
    # the manifest is JSON of the file itself, next to the shape and the points
    record = json.loads(out.read_text())
    assert set(record) == {"shape", "manifest", "points"} and record["manifest"] == manifest


def test_files_without_a_manifest_still_load(tmp_path: Path) -> None:
    points = sweep(TOY_SHAPE, levels=SPEC.levels, **SERIAL)
    out = tmp_path / "old.json"

    save(points, TOY_SHAPE, out)

    record = json.loads(out.read_text())
    assert set(record) == {"shape", "points"}  # the file format before manifests, byte for byte
    assert load(out) == (TOY_SHAPE, points)
    assert load_manifest(out) is None


def test_the_points_are_those_of_the_serial_sweep_in_canonical_order(tmp_path: Path) -> None:
    points, _ = parallel_sweep(SPEC, tmp_path / "points.json", workers=2)

    serial = sweep(TOY_SHAPE, levels=SPEC.levels, **SERIAL)
    assert [key_of(p) for p in points] == [key_of(p) for p in serial] == SPEC.keys()
    assert [_priced(p) for p in points] == [_priced(p) for p in serial]


def test_a_rerun_computes_nothing_and_a_partial_file_is_completed(tmp_path: Path) -> None:
    out = tmp_path / "points.json"
    full, _ = parallel_sweep(SPEC, out, workers=1)
    manifest = load_manifest(out)
    assert manifest is not None

    again, computed = parallel_sweep(SPEC, out, workers=1)

    assert computed == 0 and again == full
    resumed = load_manifest(out)
    assert resumed is not None
    assert resumed["runs"] == 2 and resumed["wall_seconds"] >= manifest["wall_seconds"]

    # the first half of the points, written as a checkpoint would be, then finished
    partial = tmp_path / "partial.json"
    save(full[: len(full) // 2], TOY_SHAPE, partial, manifest=manifest)
    completed, computed = parallel_sweep(SPEC, partial, workers=1)

    assert computed == len(full) - len(full) // 2
    assert [_priced(p) for p in completed] == [_priced(p) for p in full]
    assert completed[: len(full) // 2] == full[: len(full) // 2]  # kept, not recomputed


def test_a_file_of_another_shape_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "points.json"
    shape = ServingShape(**{**TOY_SHAPE.manifest, "requests": 6, "batch": 3})
    other = SweepSpec(shape=shape, levels=SPEC.levels, grid=SMALL_GRID, etas=SPEC.etas)
    parallel_sweep(other, out, workers=1)

    with pytest.raises(ValueError, match="another shape"):
        parallel_sweep(SPEC, out, workers=1)


def test_a_worker_prices_a_key_as_the_frontier_does() -> None:
    key = ("request", "row", Fraction(1, 2), Fraction(1, 8), Fraction(1, 2))
    (point,) = [p for p in sweep(TOY_SHAPE, levels=(("request", "row"),), **SERIAL) if key_of(p) == key]

    assert _priced(price_key(TOY_SHAPE, key)) == _priced(point)


def _priced(point: Point) -> tuple[object, ...]:
    """A point without its timing, which differs run to run."""

    return (*key_of(point), point.bits, point.out_bits, point.overhead, point.work, point.recompute)
