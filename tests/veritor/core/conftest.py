"""Register the ``slow`` marker used by the Hawkeye-simulator comparison test."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: long-running or environment-dependent test"
    )
