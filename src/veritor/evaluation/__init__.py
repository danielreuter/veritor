"""Calibration: what an honest server achieves, to set ``U_max`` and ``W_max``.

Nothing here is part of the protocol.  :mod:`.serving` writes the kind table
of a serving run at any dimensions under each partition an honest server
might mark; :mod:`.frontier` prices every partition and policy with the
protocol's own ``Bound``, ``Cost`` and ``expected_work`` and reports the
capacity an honest server can certify within a prover overhead and a
verifier work budget.
"""

from .frontier import (
    DEFAULT_ETAS,
    DEFAULT_GRID,
    DEFAULT_PARTITIONS,
    FRONTIER_OPTIONS,
    FRONTIER_SHAPE,
    Point,
    calibration_table,
    certify,
    honest_cost,
    partition_table,
    price,
    sweep,
)
from .serving import (
    REPLAY_LEVELS,
    VERIFICATION_LEVELS,
    ReplayLevel,
    ServingShape,
    VerificationLevel,
    partitions,
    serving_table,
)

__all__ = [
    "DEFAULT_ETAS",
    "DEFAULT_GRID",
    "DEFAULT_PARTITIONS",
    "FRONTIER_OPTIONS",
    "FRONTIER_SHAPE",
    "REPLAY_LEVELS",
    "VERIFICATION_LEVELS",
    "Point",
    "ReplayLevel",
    "ServingShape",
    "VerificationLevel",
    "calibration_table",
    "certify",
    "honest_cost",
    "partition_table",
    "partitions",
    "price",
    "serving_table",
    "sweep",
]
