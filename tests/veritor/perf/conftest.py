"""Asymptotic shape tests: ratios between sizes, never absolute constants.

The `slow` marker is registered in ``tests/veritor/conftest.py``; slow tests
are skipped unless selected with `-m slow`.  The synthetic descriptions come
from the `benchmarks` package at the repository root, which is put on
`sys.path` because pytest's importlib mode does not add the root itself.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def best_of(fn: Callable[[], object], repeats: int = 3) -> float:
    """The fastest of a few runs: the least noisy estimate of a cost for a shape test."""

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def per_call(fn: Callable[[object], object], items: list, repeats: int = 3) -> float:
    """Fastest per-call latency of ``fn`` over ``items``."""

    return best_of(lambda: [fn(item) for item in items], repeats) / len(items)


@pytest.fixture
def timer() -> Callable[[Callable[[], object]], float]:
    return best_of
