"""The stress catalogue's fixtures.

``scenario`` records the priced rows of one test into ``docs/data/stress.json``
(or ``$VERITOR_STRESS_DATA``) once the test has passed; rows are merged by ID
under a lock so that other suites writing other IDs are never clobbered.
``model`` and ``sampled`` are the toy LMs every scenario runs on
(:mod:`veritor.stress.models`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from veritor.stress.models import SAMPLED, SHAPE, Model, make_model
from veritor.stress.rows import Recorder, record

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(
    os.environ.get("VERITOR_STRESS_DATA", ROOT / "docs" / "data" / "stress.json")
)

_PASSED = pytest.StashKey[bool]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Iterator[None]:
    outcome = yield
    report: pytest.TestReport = outcome.get_result()  # type: ignore[attr-defined]
    if report.when == "call":
        item.stash[_PASSED] = report.passed


@pytest.fixture
def scenario(request: pytest.FixtureRequest) -> Iterator[Recorder]:
    recorder = Recorder()
    yield recorder
    if recorder.rows and request.node.stash.get(_PASSED, False):
        record(DATA, recorder.rows)


@pytest.fixture(scope="session")
def model() -> Model:
    return make_model(SHAPE)


@pytest.fixture(scope="session")
def sampled() -> Model:
    return make_model(SAMPLED)
