"""``Optimize``: the client's search for ``theta`` over a rational grid.

Advisory and untrusted: the verifier fixes ``eta`` and checks
``Bound(C, I, theta) <= U_max`` itself, so nothing here is load-bearing.
The search evaluates every grid point (each ``Bound`` takes milliseconds)
and keeps the cheapest policy whose bound fits, or -- given a budget instead
-- the tightest bound the budget affords.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction

from veritor.core.compiled import Compiled
from veritor.core.index import KindTable
from veritor.core.policy import ProbabilityInput, VerificationPolicy, exact_fraction

from .bound import BoundOptions, BoundResult, bound
from .cost import CostParameters, ExpectedCost, cost


@dataclass(frozen=True, slots=True, init=False)
class PolicyGrid:
    """The ``(q, s)`` pairs to try: every combination of the two lists."""

    q: tuple[Fraction, ...]
    s: tuple[Fraction, ...]

    def __init__(
        self, q: tuple[ProbabilityInput, ...], s: tuple[ProbabilityInput, ...]
    ) -> None:
        for name, values in (("q", q), ("s", s)):
            checked = tuple(exact_fraction(value, name=name) for value in values)
            if not checked or any(not 0 <= value <= 1 for value in checked):
                raise ValueError(f"{name} must list probabilities in [0, 1]")
            object.__setattr__(self, name, checked)

    @classmethod
    def uniform(cls, steps: int) -> PolicyGrid:
        """``{0, 1/steps, ..., 1}`` for both rates."""

        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        values = tuple(Fraction(k, steps) for k in range(steps + 1))
        return cls(values, values)

    def policies(self) -> Iterator[VerificationPolicy]:
        for q in self.q:
            for s in self.s:
                yield VerificationPolicy(q, s)


@dataclass(frozen=True, slots=True)
class Optimization:
    """The chosen policy with the bound and cost that justified it."""

    policy: VerificationPolicy
    bound: BoundResult
    cost: ExpectedCost
    evaluated: int


def optimize(
    target: Compiled | KindTable,
    eta: ProbabilityInput,
    grid: PolicyGrid,
    *,
    max_bits: float | None = None,
    max_cost: ProbabilityInput | None = None,
    parameters: CostParameters | None = None,
    bound_options: BoundOptions | None = None,
    accept: Callable[[VerificationPolicy], bool] | None = None,
) -> Optimization | None:
    """The cheapest grid policy with ``Bound <= max_bits``, or the tightest under ``max_cost``.

    Exactly one of ``max_bits`` and ``max_cost`` must be given.  ``accept``
    may veto policies on other grounds (the verifier's ``W_max`` through
    :func:`veritor.protocol.expected_work`, say).  Ties break towards the
    smaller secondary objective, then the earlier grid point.  ``None`` when
    no grid point is feasible.
    """

    if (max_bits is None) == (max_cost is None):
        raise ValueError("give exactly one of max_bits and max_cost")
    budget = None if max_cost is None else exact_fraction(max_cost, name="max_cost")
    best: tuple[tuple[Fraction | float, Fraction | float], Optimization] | None = None
    evaluated = 0
    for policy in grid.policies():
        if accept is not None and not accept(policy):
            continue
        evaluated += 1
        expected = cost(target, policy, parameters)
        if budget is not None and expected.total > budget:
            continue
        result = bound(target, policy, eta, bound_options)
        if max_bits is not None and result.bits > max_bits:
            continue
        key = (
            (expected.total, result.bits)
            if budget is None
            else (result.bits, expected.total)
        )
        if best is None or key < best[0]:
            best = (key, Optimization(policy, result, expected, evaluated))
    if best is None:
        return None
    return replace(best[1], evaluated=evaluated)


__all__ = ["Optimization", "PolicyGrid", "optimize"]
