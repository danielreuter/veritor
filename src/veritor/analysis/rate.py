"""The *rate* of a policy: a closed form for the capacity per bit of threshold.

``Bound`` (:mod:`veritor.analysis.bound`) folds the whole kind table; this
module reads four numbers off it and gives a formula.  Fix a policy
``theta = (q, s)`` and write, for a replay unit (RU) copy holding ``l``
erroneous verification units (VUs), its survival cost in bits

    c(l) = log2 (1 / f(l)),   f(l) = 1 - q + q (1 - s)**l

(:mod:`veritor.analysis.probability`, in nats there), and ``lambda =
log2 (1 / eta)`` for the verifier's threshold.  A VU of nothing but source
gates is never incorrect (see ``bound``); call the others *fallible*.  Let

    R    the number of RU copies holding a fallible VU,
    W_R  the widest bottleneck ``cut_bits`` among their kinds,
    W_V  the widest bottleneck among the fallible VU kinds inside them,
    m    the most fallible VUs inside one RU copy.

Then for every ``eta`` in ``(0, 1)``

    log2 |Y_eta|  <=  rho * lambda + log2 e,                            (1)

    rho  =  max over 1 <= l <= l_0 of
            [ log2 R + log2 (l (l + 1)) + min(l W_V + log2 C(m, l), W_R) ] / c(l),     (2)

where ``l_0`` is the least ``l`` at which ``l W_V + log2 C(m, l) >= W_R``
(the covers of the ``l``-subsets of an RU by their VUs outweigh the RU's
own cover), or ``min(m, W_R)`` if there is none.  :func:`rate` computes
``rho`` and the numbers it is built from; :func:`capacity_from_rate` is the
right-hand side of (1).

Derivation.  ``bound``'s statement: ``|Y_eta| <= sum over admissible
(l_r)_r of prod_r V_r(l_r)``, where ``V_r(l)`` is the total ``2**kappa`` of
the distinct covers assigned to the ``l``-subsets of ``R_r`` and a profile
is admissible when ``sum_r c(l_r) < lambda``.  Assign covers as follows: a
subset of fewer than ``l_0`` VUs is covered by those VUs themselves (each
its own cover, ``2**kappa(VU) <= 2**W_V`` apiece, so ``V_r(l) <= C(m, l)
2**(l W_V)``), and every subset of ``l_0`` or more VUs by the RU node
(*one* cover, ``2**kappa(R_r) <= 2**W_R``).  Group the profiles by ``l_r
-> min(l_r, l_0)``: a group of ``l_0`` or more errors in ``R_r`` costs at
least ``c(l_0)`` (``c`` is increasing), so the sum over admissible profiles
is at most the sum over admissible groups, ``sum_r c(min(l_r, l_0)) <
lambda``, of ``prod_r V_r``.  A Chernoff step at any ``t >= 0`` multiplies
each admissible group by ``2**(t (lambda - sum_r c(l_r))) >= 1`` and drops
admissibility:

    |Y_eta|  <=  2**(t lambda) * prod_r ( 1 + eps_r(t) ),
    eps_r(t)  =  sum_{l < l_0} C(m, l) 2**(l W_V - t c(l))  +  2**(W_R - t c(l_0)).

Now ``prod_r (1 + eps_r) <= exp(sum_r eps_r)``, so if ``sum_r eps_r(t) <=
1`` then ``log2 |Y_eta| <= t lambda + log2 e``.  Choosing ``t = rho`` of (2)
makes the ``l``-th term of every ``eps_r`` at most ``1 / (R l (l + 1))``
(rearrange ``rho c(l) >= log2 R + log2 (l (l + 1)) + ...``), and ``sum_r
sum_l 1 / (R l (l + 1)) = sum_{l >= 1} 1 / (l (l + 1)) = 1``.  ``rho`` does
not depend on ``eta``.  Every step is a bound on ``|Y_eta|`` itself, not
on the fold, so (1) holds whether or not the fold is tighter (on small
circuits it is tested against the exact union of outputs).

Reading (2).  Each term is the *escaping bits* over the *cost in bits* of
one attack channel, "``l`` errors inside one RU": choosing the RU
(``log2 R``) and the VUs inside it (``log2 C(m, l)``) is the position
term, ``l W_V`` the value term, capped by the RU's own bottleneck ``W_R``
once the errors are many; ``log2 (l (l + 1))`` pays for summing the
channels; ``c(l)`` is what the attack forfeits in acceptance probability.
``l = 1`` is the *scattered* channel, single errors in distinct RUs,
``(W_V + log2 R + log2 m + 1) / c(1)``; ``l = l_0`` is the *whole-RU*
channel, ``(W_R + log2 R + log2 (l_0 (l_0 + 1))) / c(l_0)`` with
``c(l_0)`` close to the saturation ``log2 (1 / (1 - q))``.  Since
``c(l) <= min(l c(1), c(inf))`` -- concentration is cheap -- the maximum
can sit strictly between, and ``rho`` is the steepest channel.

What the closed form leaves out.  (i) The position term is charged per
channel as ``log2 R + log2 C(m, l)``, i.e. ``log2 N`` bits per error where
``N = R m``; the sum over profiles the fold minimises has only ``C(N, B) ~
(e N / B)**B`` choices when ``B ~ lambda / c(l*)`` errors are affordable,
so the fold saves about ``log2 B`` bits per error over (1).  This is
visible exactly when many errors are affordable, i.e. at tiny ``s`` where
the scattered channel binds; where a wide RU channel binds, ``log2 B`` is
negligible next to ``W_R``.  (ii) Covers by the kinds between an RU and
its VUs (a step, a layer, a matvec) are not used: the closed form knows
two levels.  (iii) ``R``, ``W_R``, ``W_V`` and ``m`` are maxima over
kinds; a table mixing wide and narrow RUs is charged as if all were wide.
The fold's own slope, :attr:`veritor.analysis.bound.BoundResult.rho`,
solves ``sum_K n_K (Z_K(t) - 1) = 1`` for the fold's exact per-kind series:
the same construction, free of (ii) and (iii) but not of (i).  On the
frontier's request/cell serving table (2048 requests, ``W_R = 8192``,
``W_V = 16``, ``l_0 = 187``) the closed form is within ``0.4%`` of the
fold wherever the whole-RU channel binds (``s >= 1/64``) and ``17--27%``
above it where the scattered channel does (``s = 1/512``), all of that
excess being (i); ``tests/veritor/analysis/test_rate.py`` records the
ranges.

Rounding.  Costs are rounded down, binomials and logarithms up, and the
final ratio up, so the computed ``rho`` is never below the exact value of
(2) and (1) remains a bound; ``rho`` is ``inf`` when some channel is free
(``q = 0`` or ``s = 0`` with a fallible VU) and ``0`` when there is no
fallible VU or every error is caught surely (``q = s = 1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from veritor.core.compiled import Compiled, as_kind_table
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.identity import Digest
from veritor.core.index import KindSummary, KindTable
from veritor.core.policy import ProbabilityInput, VerificationPolicy, exact_fraction

from .probability import budget
from .series import log2_binomials

LOG2E = 1 / math.log(2)
_REL = 2.0**-40
_ABS = 2.0**-48


@dataclass(frozen=True, slots=True)
class RateResult:
    """``rho`` of a policy and the four numbers it is built from.

    ``rho`` is (2) of the module docstring: bits of capacity per bit of
    ``log2 (1 / eta)``, so that ``log2 |Y_eta| <= rho * log2 (1 / eta) +
    log2 e`` (:meth:`capacity`).  ``replay_units`` is ``R``, the RU copies
    holding a fallible VU; ``replay_bits`` is ``W_R`` and
    ``verification_bits`` is ``W_V``, the widest RU and VU bottlenecks
    (``cut_bits``) among them; ``verification_units`` is ``m``, the most
    fallible VUs inside one RU copy.  ``lumped_at`` is ``l_0``, the error
    count from which an RU is charged its own bottleneck, ``binding`` the
    ``l`` whose channel attains ``rho``; ``scattered`` and ``whole`` are the
    two named channels, ``l = 1`` and ``l = l_0``.  All are ``0`` when no RU
    holds a fallible VU.
    """

    rho: float
    replay_units: int
    replay_bits: int
    verification_bits: int
    verification_units: int
    lumped_at: int
    binding: int
    scattered: float
    whole: float
    policy: VerificationPolicy
    digest: Digest

    def capacity(self, eta: ProbabilityInput) -> float:
        """``rho * log2 (1 / eta) + log2 e``, an upper bound on ``log2 |Y_eta|``."""

        return capacity_from_rate(self.rho, eta)


def rate(target: Compiled | KindTable, policy: VerificationPolicy) -> RateResult:
    """The closed-form rate (2) of ``policy`` on the compiled artifact or its kind table."""

    table = as_kind_table(target)
    if not isinstance(policy, VerificationPolicy):
        raise TypeError("policy must be a VerificationPolicy")
    rows = {row.kind: row for row in table.rows}
    fallible = {
        kind
        for kind, row in rows.items()
        if row.role == VERIFICATION
        and row.size > row.source_inputs + row.source_weights
    }
    holders: list[tuple[KindSummary, int]] = []
    for row in rows.values():
        if row.role != REPLAY:
            continue
        inside = sum(
            count for kind, count in row.verification_kinds if kind in fallible
        )
        if inside:
            holders.append((row, inside))
    if not holders:
        return RateResult(0.0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, policy, table.digest)

    replay_units = sum(row.copies for row, _ in holders)
    replay_bits = max(row.cut_bits for row, _ in holders)
    verification_units = max(inside for _, inside in holders)
    verification_bits = max(
        rows[kind].cut_bits
        for row, _ in holders
        for kind, _ in row.verification_kinds
        if kind in fallible
    )

    channels, lumped_at = _channels(
        replay_units, replay_bits, verification_bits, verification_units, policy
    )
    binding = int(np.argmax(channels)) + 1
    return RateResult(
        rho=_up(float(channels[binding - 1])),
        replay_units=replay_units,
        replay_bits=replay_bits,
        verification_bits=verification_bits,
        verification_units=verification_units,
        lumped_at=lumped_at,
        binding=binding,
        scattered=_up(float(channels[0])),
        whole=_up(float(channels[-1])),
        policy=policy,
        digest=table.digest,
    )


def capacity_from_rate(rho: float, eta: ProbabilityInput) -> float:
    """``rho * log2 (1 / eta) + log2 e`` in bits: the right-hand side of (1), rounded up."""

    if not (isinstance(rho, (int, float)) and rho >= 0):
        raise ValueError("rho must be a nonnegative number")
    threshold = exact_fraction(eta, name="eta")
    if not 0 <= threshold < 1:
        raise ValueError("eta must lie in [0, 1)")
    if rho == 0:
        return LOG2E
    bits = budget(threshold) * LOG2E  # rounded up in nats; the product errs by an ulp
    return _up(rho * bits) + LOG2E


# -- the channels ------------------------------------------------------------


def _channels(
    replay_units: int,
    replay_bits: int,
    verification_bits: int,
    verification_units: int,
    policy: VerificationPolicy,
) -> tuple[np.ndarray, int]:
    """The ratios of (2) for ``l = 1 .. l_0`` and ``l_0`` itself.

    An RU is never charged its own bottleneck later than ``W_R`` errors in
    (from then on ``l W_V + log2 C(m, l) >= W_R`` whenever ``W_V >= 1``), nor
    later than it has VUs, so the binomials stop at ``min(m, W_R)``.
    """

    last = max(1, min(verification_units, replay_bits))
    binomials = log2_binomials(verification_units, last)[1:]
    errors = np.arange(1, last + 1)
    by_units = errors * verification_bits + binomials
    crossing = np.flatnonzero(by_units >= replay_bits)
    lumped_at = int(errors[crossing[0]]) if crossing.size else last
    value = np.minimum(by_units[:lumped_at], float(replay_bits))
    if not crossing.size and lumped_at < verification_units:
        value[-1] = replay_bits  # the lumped class still needs the RU's cover
    errors = errors[:lumped_at]
    position = _up_array(
        math.log2(replay_units) + np.log2(errors) + np.log2(errors + 1)
    )
    numerator = position + value
    cost = _cost_bits(policy, errors)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(cost > 0, numerator / cost, math.inf)
    ratios = np.where(np.isinf(cost), 0.0, ratios)
    return ratios, lumped_at


def _cost_bits(policy: VerificationPolicy, errors: np.ndarray) -> np.ndarray:
    """``c(l) = log2 (1 / f(l))`` for each ``l`` of ``errors``, rounded down.

    ``ln f(l) = logaddexp(ln (1 - q), ln q + l ln (1 - s))`` keeps every
    term nonnegative and finite except where ``f`` is exactly ``0`` (``q = 1``
    and ``s = 1``), so ``q = 1`` and tiny ``(1 - s)**l`` are exact to within
    float rounding, which the downward slack dominates.
    """

    q, s = float(policy.q), float(policy.s)
    with np.errstate(divide="ignore"):
        surviving = np.log1p(-q) if q < 1 else -math.inf
        caught = (math.log(q) if q > 0 else -math.inf) + errors * (
            np.log1p(-s) if s < 1 else -math.inf
        )
        ln_f = np.logaddexp(surviving, caught)
    cost = -ln_f * LOG2E
    finite = np.isfinite(cost)
    cost[finite] = np.maximum(0.0, cost[finite] - cost[finite] * _REL - _ABS)
    return cost


def _up(value: float) -> float:
    """``value`` rounded up by more than the float operations behind it can err; ``0`` and ``inf`` stay."""

    return value + abs(value) * _REL + _ABS if value and math.isfinite(value) else value


def _up_array(values: np.ndarray) -> np.ndarray:
    return values + np.abs(values) * _REL + _ABS


__all__ = ["RateResult", "capacity_from_rate", "rate"]
