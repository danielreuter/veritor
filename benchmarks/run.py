"""``python -m benchmarks.run [--quick] [--only NAME ...] [--out PATH]``: run the barrage, write JSON."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ._harness import Benchmark, Scale

REGISTRY: dict[str, str] = {
    "compile": "benchmarks.compile_time",
    "lookup": "benchmarks.lookup",
    "kinds": "benchmarks.kinds",
    "analysis": "benchmarks.analysis",
    "challenge": "benchmarks.challenge",
    "merkle": "benchmarks.merkle",
    "protocol": "benchmarks.protocol",
    "reach": "benchmarks.reach",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - the manifest is best effort
        return "unknown"


def manifest(scale: Scale) -> dict[str, object]:
    import veritor

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _git("status", "--porcelain") != "",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "veritor_version": getattr(veritor, "__version__", "unknown"),
        "quick": scale.quick,
        "repeats": scale.repeats,
    }


def run_all(names: list[str], scale: Scale, log=print) -> list[Benchmark]:
    results: list[Benchmark] = []
    for name in names:
        module = importlib.import_module(REGISTRY[name])
        log(f"[{name}] running ...")
        started = time.perf_counter()
        bench = module.run(scale)
        log(f"[{name}] done in {time.perf_counter() - started:.1f}s")
        results.append(bench)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run", description=__doc__
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small sizes, two repeats: seconds per benchmark",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help=f"subset of {sorted(REGISTRY)}"
    )
    parser.add_argument(
        "--out",
        default="docs/data/benchmarks.json",
        help="where to write the raw results",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed repeats per point (median is reported)",
    )
    args = parser.parse_args(argv)
    names = (
        list(REGISTRY)
        if not args.only
        else [n for n in REGISTRY if any(n.startswith(o) for o in args.only)]
    )
    unknown = [
        o for o in (args.only or []) if not any(n.startswith(o) for n in REGISTRY)
    ]
    if unknown:
        parser.error(f"unknown benchmark(s) {unknown}; choose from {sorted(REGISTRY)}")
    scale = Scale(quick=args.quick, repeats=args.repeats)
    started = time.perf_counter()
    results = run_all(names, scale)
    document = {
        "manifest": {
            **manifest(scale),
            "seconds": time.perf_counter() - started,
            "benchmarks": names,
        },
        "benchmarks": {bench.name: bench.as_json() for bench in results},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
