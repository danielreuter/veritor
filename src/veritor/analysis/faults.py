"""M6 fault declarations: what admitting ``f_max`` declared VUs costs in capacity.

After the q-challenge the prover may declare up to ``f_max`` VUs of the
opened RUs incorrect (:class:`veritor.protocol.InteriorMessage`).  A declared
VU is committed like any other -- its value is authenticated and every VU
reading it is checked against that value -- but its own relation is never
checked, so it is an incorrect VU with zero exposure.  The survival argument
of :mod:`veritor.analysis.bound` therefore sees it as correct, and the
counting argument has to pay for it here.

**Charge.**  Let ``D`` be the declared set (``|D| <= f_max``), ``E`` the
error set of the transcript and ``E' = E \\ D`` what remains exposed.  The
output ``y`` is a function of the correct computation outside ``E``, the
values at ``E'`` and the values at ``D``; ``E'`` must survive the sampling
as before.  Enumerating ``D`` and its values,

    |Y_eta| <= sum_{|D| <= f} 2**kappa(D) * sum_{E' admissible} 2**kappa(E')
            <= (1 + |S| 2**W_V)**f * 2**U_0,

with ``|S|`` the number of VUs and ``W_V = max_kind kappa`` the widest VU
cut (``kappa(D) <= sum_{v in D} kappa(v)`` because the union of downstream
cuts is a downstream cut, and ``sum_{j <= f} C(|S|, j) 2**(j W_V) <= (1 +
|S| 2**W_V)**f``).  So one declared fault costs

    u(1) = log2(1 + |S| 2**W_V) = W_V + log2(|S| + 2**-W_V)  ~  W_V + log2|S|

bits and ``fault_allowance_bits(target, f) = f * u(1)``: a value-level advice
string that names ``f`` VUs (``log2|S|`` each) and their contents (``W_V``
each).  ``W_V`` ranges over the kinds with a relation only: a VU of source
gates has nothing to declare and the verifier rejects its declaration.

**Adaptivity.**  The charge above prices a ``D`` fixed before the q-challenge
(or ``q = 1``, every RU replayed).  The protocol lets the prover choose ``D``
*after* seeing ``J`` -- the honest server cannot know its faults before it
replays the opened RUs -- and an adversary uses this: it corrupts one VU in
each of many RUs and pardons whichever ``f`` of them were opened.  With ``N_J``
corrupted VUs in the opened RUs its acceptance probability is
``(1 - s)**max(0, N_J - f)`` instead of ``(1 - s)**N_J``: a factor
``1 / (1 - s)`` per declaration whenever ``N_J >= f``, which is worth
``ln(1 / (1 - s)) / (-ln(1 - q s))`` additional corrupted VUs -- about
``1 / q`` of them when ``s`` is small, one when ``q = 1``.  Since
``sigma_f(E) <= (1 - s)**-f sigma_0(E)`` for every ``E``, the fold at the
lowered threshold ``eta (1 - s)**f`` is a rigorous upper bound on the
adaptive capacity for ``s < 1``; :func:`adaptive_fault_bound` computes it,
and :func:`adaptive_fault_allowance_bits` is its excess over ``U_0``.  It
coincides with ``f * u(1)`` in order of magnitude only when ``q`` is near
one; for the replay rates a serving run uses it is far larger.  ``bound()``
charges the specified ``f * u(1)``; which of the two the protocol should
charge is a decision for the architect (see ``docs/stress-tests.md``, M6).
"""

from __future__ import annotations

import math
from dataclasses import replace

from veritor.core import (
    Compiled,
    KindTable,
    ProbabilityInput,
    VerificationPolicy,
    as_kind_table,
    exact_fraction,
)
from veritor.core.description import VERIFICATION

from .bound import BoundOptions, BoundResult, bound, cut_bits


def unit_fault_bits(target: Compiled | KindTable) -> float:
    """``u(1) = log2(1 + |S| 2**W_V)``: the capacity of one declared VU.

    ``|S|`` counts every VU of the index and ``W_V`` is the widest cut
    (:func:`~veritor.analysis.bound.cut_bits`) over the VU kinds that have a
    relation.  ``0.0`` for an index without such a kind: nothing can be
    declared.
    """

    table = as_kind_table(target)
    widest = -1
    units = 0
    for row in table.rows:
        if row.role != VERIFICATION:
            continue
        units += row.copies
        if row.size > row.source_inputs + row.source_weights:
            widest = max(widest, cut_bits(row))
    if widest < 0 or units == 0:
        return 0.0
    return widest + math.log2(units + 2.0**-widest)


def fault_allowance_bits(target: Compiled | KindTable, max_faults: int) -> float:
    """``f_max * u(1)``: the bits ``Bound`` gains for admitting ``max_faults`` declarations."""

    if type(max_faults) is not int or max_faults < 0:
        raise ValueError("max_faults must be a nonnegative integer")
    if max_faults == 0:
        return 0.0
    return max_faults * unit_fault_bits(target)


def adaptive_fault_bound(
    target: Compiled | KindTable,
    policy: VerificationPolicy,
    eta: ProbabilityInput,
    max_faults: int,
    options: BoundOptions | None = None,
) -> BoundResult:
    """``Bound`` at threshold ``eta (1 - s)**max_faults``: an upper bound on the
    capacity when the prover chooses its declarations after seeing ``J``.

    Rigorous for ``s < 1``, where a declaration is worth at most ``1 / (1 -
    s)`` in acceptance probability; the returned result reports the lowered
    threshold as its ``eta``.  At ``q = 1`` the q-challenge reveals nothing,
    so ``f * u(1)`` is rigorous too and the smaller of the two is returned.
    At ``s = 1`` with ``q < 1`` the relaxation is vacuous and the result is
    the trivial cap ``out_bits``.
    """

    if type(max_faults) is not int or max_faults < 0:
        raise ValueError("max_faults must be a nonnegative integer")
    if not isinstance(policy, VerificationPolicy):
        raise TypeError("policy must be a VerificationPolicy")
    eta = exact_fraction(eta, name="eta")
    if max_faults == 0:
        return bound(target, policy, eta, options)
    if policy.q == 1:
        specified = bound(target, policy, eta, options, max_faults=max_faults)
        if policy.s == 1:
            return specified
        lowered = bound(target, policy, eta * (1 - policy.s) ** max_faults, options)
        return specified if specified.bits <= lowered.bits else lowered
    if policy.s == 1:
        base = bound(target, policy, eta, options)
        return replace(base, bits=float(base.out_bits), capped=True)
    return bound(target, policy, eta * (1 - policy.s) ** max_faults, options)


def adaptive_fault_allowance_bits(
    target: Compiled | KindTable,
    policy: VerificationPolicy,
    eta: ProbabilityInput,
    max_faults: int,
    options: BoundOptions | None = None,
) -> float:
    """The excess of :func:`adaptive_fault_bound` over ``Bound(C, I, theta)`` at ``eta``."""

    base = bound(target, policy, eta, options).bits
    return adaptive_fault_bound(target, policy, eta, max_faults, options).bits - base
