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
from veritor.stress.rows import Row, dump, load, record, row_key

__all__ = [
    "ETA",
    "POLICY",
    "Measurement",
    "Price",
    "Row",
    "compile_scenario",
    "dump",
    "evaluate",
    "honest_cost",
    "load",
    "price",
    "record",
    "row_key",
]
