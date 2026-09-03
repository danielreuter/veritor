"""Stress-test bookkeeping: priced scenario rows and the results table they render.

``tests/veritor/stress/`` builds each scenario of ``docs/stress-tests.md``
concretely, compiles it, and records one :class:`Row` per priced circuit in
``docs/data/stress.json``.  ``python -m veritor.stress.report`` renders the
recorded rows into the results section of the catalogue.
"""

from veritor.stress.measure import (
    ETA,
    POLICY,
    Measurement,
    Price,
    compile_scenario,
    evaluate,
    honest_cost,
    price,
)
from veritor.stress.models import SAMPLED, SHAPE, Model, make_model
from veritor.stress.rows import Recorder, Row, dump, load, record, row_key
from veritor.stress.serving import Served, by_request, serve

__all__ = [
    "ETA",
    "POLICY",
    "SAMPLED",
    "SHAPE",
    "Measurement",
    "Model",
    "Price",
    "Recorder",
    "Row",
    "Served",
    "by_request",
    "compile_scenario",
    "dump",
    "evaluate",
    "honest_cost",
    "load",
    "make_model",
    "price",
    "record",
    "row_key",
    "serve",
]
