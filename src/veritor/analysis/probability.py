"""Survival of an error set under ``theta = (q, s)``, exact and in nats.

Each replay unit is selected with probability ``q``; inside a selected
replay unit each verification unit is selected with probability ``s``.  An
error set ``E`` (verification units holding an incorrect gate) escapes
detection with probability

    sigma(E) = prod_r f(l_r),   f(l) = 1 - q + q (1 - s)^l,   l_r = |E ∩ R_r|,

and is *admissible* iff ``sigma(E) > eta`` for the verifier's threshold
``eta``.  Writing ``c(l) = -ln f(l)`` and ``Lambda = ln(1 / eta)`` this is
``sum_r c(l_r) < Lambda``: a knapsack over replay units.  ``c`` is increasing and saturates at ``-ln(1 - q)``, so many
errors in one replay unit cost little more than a few -- concentration is
cheap, and that falls out of the formula.

The exact functions return :class:`~fractions.Fraction`; the ``float``
functions round in the direction that admits more error sets (costs down,
the budget up), so a bound computed from them stays an upper bound.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction

from veritor.core.policy import VerificationPolicy

_REL = 2.0**-40
_ABS = 2.0**-48


def _ln(value: Fraction) -> float:
    """``ln value`` for a positive rational, safe far outside the float range."""

    as_float = float(value)
    if as_float >= 2.0**-1000:
        return math.log(as_float)
    return math.log(value.numerator) - math.log(value.denominator)


def survival_factor(policy: VerificationPolicy, errors: int) -> Fraction:
    """``f(l)``: the chance one replay unit with ``l`` erroneous units escapes."""

    if type(errors) is not int or errors < 0:
        raise ValueError("the error count must be a nonnegative integer")
    return 1 - policy.q + policy.q * (1 - policy.s) ** errors


def survival(policy: VerificationPolicy, errors_per_replay_unit: Iterable[int]) -> Fraction:
    """``sigma(E) = prod_r f(l_r)`` from the per-replay-unit error counts."""

    result = Fraction(1)
    for errors in errors_per_replay_unit:
        result *= survival_factor(policy, errors)
    return result


def admissible(
    policy: VerificationPolicy, eta: Fraction, errors_per_replay_unit: Iterable[int]
) -> bool:
    """Whether ``E`` is accepted with probability strictly above ``eta``."""

    return survival(policy, errors_per_replay_unit) > eta


def unit_cost(policy: VerificationPolicy, errors: int) -> float:
    """``c(l) = -ln f(l)`` in nats, rounded down (``inf`` when ``f(l) = 0``)."""

    factor = survival_factor(policy, errors)
    if factor == 0:
        return math.inf
    if factor == 1:
        return 0.0
    cost = -_ln(factor)
    return max(0.0, cost - cost * _REL - _ABS)


def saturation_cost(policy: VerificationPolicy) -> float:
    """``c(inf) = -ln(1 - q)``, the cost of corrupting a whole replay unit."""

    if policy.q == 1:
        return math.inf
    return -math.log(1 - policy.q)


def budget(eta: Fraction) -> float:
    """``Lambda = ln(1 / eta)`` in nats, rounded up (``inf`` when ``eta = 0``)."""

    if not isinstance(eta, Fraction) or not 0 <= eta < 1:
        raise ValueError("eta must be a Fraction in [0, 1)")
    if eta == 0:
        return math.inf
    value = -_ln(eta)
    return value + value * _REL + _ABS


__all__ = [
    "admissible",
    "budget",
    "saturation_cost",
    "survival",
    "survival_factor",
    "unit_cost",
]
