"""Suite-wide pytest hooks.

The ``slow`` marker (declared in ``pyproject.toml``) tags long-running or
environment-dependent tests: larger perf sizes, GPU-simulator comparisons.
They are skipped unless selected with ``-m slow``.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow: select with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
