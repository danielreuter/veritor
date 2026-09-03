"""``python -m benchmarks.run [--quick] [--only NAME ...] [--out PATH]``: run the barrage, write JSON."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
import traceback
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


def run_all(
    names: list[str], scale: Scale, log=print
) -> tuple[list[Benchmark], dict[str, str]]:
    """Run every named benchmark; a failure is recorded and the barrage goes on."""

    results: list[Benchmark] = []
    errors: dict[str, str] = {}
    for name in names:
        module = importlib.import_module(REGISTRY[name])
        log(f"[{name}] running ...")
        started = time.perf_counter()
        try:
            bench = module.run(scale)
        except Exception:  # noqa: BLE001 - keep the rest of the barrage
            errors[name] = traceback.format_exc()
            log(
                f"[{name}] FAILED after {time.perf_counter() - started:.1f}s\n{errors[name]}"
            )
            continue
        log(f"[{name}] done in {time.perf_counter() - started:.1f}s")
        results.append(bench)
    return results, errors


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
    parser.add_argument(
        "--update",
        action="store_true",
        help="merge into an existing --out file: replace only the benchmarks run now",
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
    results, errors = run_all(names, scale)
    out = Path(args.out)
    document: dict = {
        "manifest": {
            **manifest(scale),
            "seconds": time.perf_counter() - started,
            "benchmarks": names,
            "errors": errors,
        },
        "benchmarks": {bench.name: bench.as_json() for bench in results},
    }
    if args.update and out.exists():
        previous = json.loads(out.read_text())
        merged_manifest = previous["manifest"]
        merged_manifest["seconds"] = (
            merged_manifest.get("seconds", 0.0) + document["manifest"]["seconds"]
        )
        merged_manifest["benchmarks"] = sorted(
            set(merged_manifest.get("benchmarks", [])) | set(names)
        )
        merged_manifest["errors"] = {
            **{
                k: v
                for k, v in merged_manifest.get("errors", {}).items()
                if k not in names
            },
            **errors,
        }
        merged_manifest.setdefault("updates", []).append(
            {
                k: document["manifest"][k]
                for k in ("git_commit", "git_dirty", "timestamp", "seconds")
            }
            | {"benchmarks": names}
        )
        document = {
            "manifest": merged_manifest,
            "benchmarks": {**previous["benchmarks"], **document["benchmarks"]},
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({time.perf_counter() - started:.1f}s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
