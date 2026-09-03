"""Hash cost: nanoseconds per call and throughput of the hash functions the cost model prices.

``uv run --with blake3 python -m benchmarks.hash_cost`` times ``hashlib``'s
SHA-256, SHA-512, BLAKE2b and SHA3-256 and the ``blake3`` package on payloads
of 32 B, 64 B, 1 KiB and 1 MiB, then the repo's own ``CommitmentDomain.leaf``
and ``.node`` frames (``veritor.protocol.merkle``), prints a table and writes
``docs/data/hash-cost.json``.  Each figure is the median of ``REPEATS`` runs of
a loop sized to take about ``TARGET_SECONDS``; ``python_call_floor_ns`` is the
same loop around ``bytes.__len__`` and bounds the interpreter's share.

The numbers are a CPU proxy for ``docs/cost-model.md``: the small-payload
columns bound one compression-function call plus Python call overhead, the
64 B -> 1 KiB slope (``marginal_block``) prices one more block without that
overhead, the 1 MiB column is the streaming rate.  ``cycles_nominal``
multiplies by the nominal clock reported by the kernel and is only as good as
that clock.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import ssl
import statistics
import subprocess
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

from veritor.core import RangeIndexedDomain
from veritor.protocol.merkle import CommitmentDomain

BLAKE3: tuple[Callable[[bytes], Any], str] | None
try:
    import blake3 as _blake3  # type: ignore[import-not-found]
except ImportError:  # the package is optional: ``uv run --with blake3``
    BLAKE3 = None
else:
    BLAKE3 = (_blake3.blake3, _blake3.__version__)

PAYLOADS: tuple[int, ...] = (32, 64, 1024, 1 << 20)
BLOCK_BYTES: dict[str, int] = {
    "sha256": 64,
    "sha512": 128,
    "blake2b": 128,
    "sha3_256": 136,  # the sponge rate
    "blake3": 64,
}
REPEATS = 7
TARGET_SECONDS = 0.05
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "hash-cost.json"

Hasher = Callable[[bytes], bytes]


def hashers() -> dict[str, Hasher]:
    table: dict[str, Hasher] = {
        "sha256": lambda b: hashlib.sha256(b).digest(),
        "sha512": lambda b: hashlib.sha512(b).digest(),
        "blake2b": lambda b: hashlib.blake2b(b).digest(),
        "sha3_256": lambda b: hashlib.sha3_256(b).digest(),
    }
    if BLAKE3 is not None:
        constructor = BLAKE3[0]
        table["blake3"] = lambda b: constructor(b).digest()
    return table


def ns_per_call(fn: Callable[[], object]) -> float:
    """Median over ``REPEATS`` of the mean nanoseconds per call of ``fn``."""

    count = 256
    while True:
        started = time.perf_counter_ns()
        for _ in range(count):
            fn()
        elapsed = time.perf_counter_ns() - started
        if elapsed >= TARGET_SECONDS * 1e9 or count >= 1 << 22:
            break
        count *= 2
    samples: list[float] = []
    for _ in range(REPEATS):
        started = time.perf_counter_ns()
        for _ in range(count):
            fn()
        samples.append((time.perf_counter_ns() - started) / count)
    return statistics.median(samples)


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "openssl": ssl.OPENSSL_VERSION,
        "blake3": None if BLAKE3 is None else BLAKE3[1],
    }
    try:
        lscpu = subprocess.run(
            ["lscpu"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        lscpu = ""
    for key, label in (("Model name", "cpu"), ("CPU(s)", "cpus")):
        found = re.search(rf"^{re.escape(key)}:\s*(.+)$", lscpu, re.MULTILINE)
        info[label] = found.group(1).strip() if found else None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    mhz = re.search(r"^cpu MHz\s*:\s*([\d.]+)", cpuinfo, re.MULTILINE)
    info["nominal_mhz"] = float(mhz.group(1)) if mhz else None
    flags = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    present = set(flags.group(1).split()) if flags else set()
    info["flags"] = sorted(
        present & {"sha_ni", "avx2", "avx512f", "avx512bw", "avx512vl", "sse4_1"}
    )
    return info


def preimage_bytes(fn: Callable[[], bytes]) -> int:
    """Bytes fed to SHA-256 by one call of ``fn`` (the frame the repo hashes)."""

    total = 0
    real = hashlib.sha256

    class Counting:
        def __init__(self) -> None:
            self.inner = real()

        def update(self, data: bytes) -> None:
            nonlocal total
            total += len(data)
            self.inner.update(data)

        def digest(self) -> bytes:
            return self.inner.digest()

    with mock.patch.object(hashlib, "sha256", Counting):
        fn()
    return total


def sha256_blocks(preimage: int) -> int:
    """64-byte blocks SHA-256 compresses for a message of ``preimage`` bytes."""

    return (preimage + 9 + 63) // 64


def veritor_frames() -> dict[str, Any]:
    domain = CommitmentDomain(b"\x33" * 32, 7, RangeIndexedDomain(1 << 20))
    value = (12345).to_bytes(2, "big")
    left = right = hashlib.sha256(b"sibling").digest()

    def leaf() -> bytes:
        return domain.leaf(999_999, 999_999, "u16", value)

    def node() -> bytes:
        return domain.node(19, 1, left, right)

    leaf_pre, node_pre = preimage_bytes(leaf), preimage_bytes(node)
    leaf_ns, node_ns = ns_per_call(leaf), ns_per_call(node)
    return {
        "leaf_preimage_bytes": leaf_pre,
        "leaf_sha256_blocks": sha256_blocks(leaf_pre),
        "leaf_ns": leaf_ns,
        "node_preimage_bytes": node_pre,
        "node_sha256_blocks": sha256_blocks(node_pre),
        "node_ns": node_ns,
        "per_committed_value_ns": leaf_ns + node_ns,
        "note": "one leaf frame plus one node frame per committed value (a large "
        "binary tree has one internal node per leaf); 16-bit value, schema u16, "
        "depth-20 range domain",
    }


def main() -> None:
    info = machine()
    payloads = {size: bytes(range(256)) * (size // 256) for size in PAYLOADS}
    floor = ns_per_call(lambda: bytes.__len__(payloads[64]))
    mhz = info["nominal_mhz"]
    results: dict[str, dict[str, dict[str, float | None]]] = {}
    marginal: dict[str, dict[str, float | None]] = {}
    for name, fn in hashers().items():
        results[name] = {}
        for size in PAYLOADS:
            ns = ns_per_call(partial(fn, payloads[size]))
            results[name][str(size)] = {
                "ns_per_hash": ns,
                "mb_per_s": size / ns * 1e3,
                "cycles_nominal": ns * mhz / 1e3 if mhz else None,
            }
        # The 64 B -> 1 KiB slope removes the per-call overhead (Python, object
        # setup, finalisation) and prices one more block of a single-lane hash.
        small, kib = results[name]["64"], results[name]["1024"]
        assert small["ns_per_hash"] is not None and kib["ns_per_hash"] is not None
        per_byte = (kib["ns_per_hash"] - small["ns_per_hash"]) / (1024 - 64)
        block_ns = per_byte * BLOCK_BYTES[name]
        marginal[name] = {
            "block_bytes": BLOCK_BYTES[name],
            "ns_per_byte": per_byte,
            "ns_per_block": block_ns,
            "cycles_per_block_nominal": block_ns * mhz / 1e3 if mhz else None,
        }
    frames = veritor_frames()

    widths = [max(len(n) for n in results) + 2, *([22] * len(PAYLOADS))]
    labels = ["hash", *(f"{s} B" if s < 1024 else f"{s >> 10} KiB" for s in PAYLOADS)]
    print(f"{info['cpu']}  ({info['cpus']} CPUs, {mhz} MHz nominal, {info['flags']})")
    print(f"Python {info['python']}, {info['openssl']}, blake3 {info['blake3']}")
    print(f"call floor {floor:.0f} ns")
    print("".join(l.ljust(w) for l, w in zip(labels, widths, strict=True)))
    for name, by_size in results.items():
        cells = [name]
        for size in PAYLOADS:
            r = by_size[str(size)]
            cells.append(f"{r['ns_per_hash']:8.0f} ns {r['mb_per_s']:7.0f} MB/s")
        print("".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)))
    for name, m in marginal.items():
        print(
            f"{name}: marginal {m['ns_per_byte']:.3f} ns/B, "
            f"{m['ns_per_block']:.0f} ns per {m['block_bytes']}-byte block"
        )
    print(
        f"veritor leaf frame {frames['leaf_preimage_bytes']} B "
        f"({frames['leaf_sha256_blocks']} blocks) {frames['leaf_ns']:.0f} ns; "
        f"node frame {frames['node_preimage_bytes']} B "
        f"({frames['node_sha256_blocks']} blocks) {frames['node_ns']:.0f} ns"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "machine": info,
                "method": f"median of {REPEATS} loops of >= {TARGET_SECONDS} s each; "
                "one hash object per call, digest() included",
                "payload_bytes": list(PAYLOADS),
                "python_call_floor_ns": floor,
                "hashes": results,
                "marginal_block": marginal,
                "veritor_frames": frames,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
