"""Static analysis of a compiled ``(C, I)`` under a policy ``theta``.

``bound`` folds the capacity bound ``U`` over the kinds of the index;
``probability`` holds the survival function the bound rests on; and
``reference`` enumerates everything explicitly on tiny circuits so the
fold can be tested against the definitions.
"""

from veritor.analysis.bound import BoundOptions, BoundResult, bound
from veritor.analysis.probability import (
    admissible,
    budget,
    saturation_cost,
    survival,
    survival_factor,
    unit_cost,
)

__all__ = [
    "BoundOptions",
    "BoundResult",
    "admissible",
    "bound",
    "budget",
    "saturation_cost",
    "survival",
    "survival_factor",
    "unit_cost",
]
