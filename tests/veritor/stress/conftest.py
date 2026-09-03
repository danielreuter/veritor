"""The stress catalogue's fixtures.

``scenario`` records the priced rows of one test into ``docs/data/stress.json``
(or ``$VERITOR_STRESS_DATA``) once the test has passed; rows are merged by ID
under a lock so that other suites writing other IDs are never clobbered.
``model`` is the toy LM every scenario runs on, sized so each test stays
well under a few seconds.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from veritor.constructors.lm import LMShape, Parameters, random_parameters
from veritor.core.gates import GateSet, make_isa_gate_set
from veritor.stress.rows import Row, record

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("VERITOR_STRESS_DATA", ROOT / "docs" / "data" / "stress.json"))

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=16, width=16)
"""The catalogue's toy LM: the simulated datacenter's shape, with the argmax head."""

SAMPLED = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=16, width=16, sampling=True)
"""The same model with the ``sample`` VU over public randomness."""


@dataclass
class Recorder:
    """Rows recorded by one test; written to :data:`DATA` if the test passes."""

    rows: list[Row] = field(default_factory=list)

    def record(
        self,
        *,
        id: str,
        what: str,
        mechanism: str,
        advice_bits: int,
        capacity_bits: int,
        overhead: float,
        description_bytes: int,
        verdict: str,
        notes: str = "",
    ) -> Row:
        if any(row.id == id for row in self.rows):
            raise ValueError(f"row {id!r} recorded twice by one test")
        row = Row(
            id=id,
            what=what,
            mechanism=mechanism,
            advice_bits=advice_bits,
            capacity_bits=capacity_bits,
            overhead=overhead,
            description_bytes=description_bytes,
            verdict=verdict,
            notes=notes,
        )
        self.rows.append(row)
        return row


_PASSED = pytest.StashKey[bool]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
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


@dataclass(frozen=True)
class Model:
    """A toy LM shape with its gate set and one draw of parameters."""

    shape: LMShape
    gate_set: GateSet
    parameters: Parameters

    @property
    def weights(self) -> tuple[int, ...]:
        return self.parameters.flatten()


def make_model(shape: LMShape, seed: int = 7) -> Model:
    return Model(shape, make_isa_gate_set(shape.width), random_parameters(shape, seed))


@pytest.fixture(scope="session")
def model() -> Model:
    return make_model(SHAPE)


@pytest.fixture(scope="session")
def sampled() -> Model:
    return make_model(SAMPLED)
