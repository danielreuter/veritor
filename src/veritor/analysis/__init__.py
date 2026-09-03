"""Static analysis of ``(C, I, theta)``: ``Bound``, ``Cost`` and ``Optimize``."""

from veritor.analysis.bound import BoundOptions, BoundResult, bound
from veritor.analysis.cost import CostParameters, ExpectedCost, cost
from veritor.analysis.optimize import Optimization, PolicyGrid, optimize
from veritor.analysis.probability import (
    admissible,
    budget,
    saturation_cost,
    survival,
    survival_factor,
    unit_cost,
)
from veritor.analysis.rate import RateResult, capacity_from_rate, rate
from veritor.analysis.union import union

__all__ = [
    "BoundOptions",
    "BoundResult",
    "CostParameters",
    "ExpectedCost",
    "Optimization",
    "PolicyGrid",
    "RateResult",
    "admissible",
    "bound",
    "budget",
    "capacity_from_rate",
    "cost",
    "optimize",
    "rate",
    "saturation_cost",
    "survival",
    "survival_factor",
    "union",
    "unit_cost",
]
