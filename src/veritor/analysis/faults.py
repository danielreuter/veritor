"""M6 fault declarations: what admitting ``f_max`` declared VUs costs in capacity.

After the q-challenge the prover may declare up to ``f_max`` VUs of the
opened RUs incorrect (:class:`veritor.protocol.InteriorMessage`).  A declared
VU is committed like any other -- its value is authenticated and every VU
reading it is checked against that value -- but its own relation is never
checked, so it is an incorrect VU with zero exposure.  The survival argument
of :mod:`veritor.analysis.bound` therefore sees it as correct, and the
counting argument has to pay for it here.  ``bound(..., max_faults=f)``
charges :func:`declared_bits`; the three bounds it takes the minimum of are
derived below.  Write ``E`` for the error set of a transcript, ``n`` for the
number of VUs (an upper bound on the declarable ones: a declaration of a VU
of source gates is rejected), ``W_V`` for the widest bottleneck cut over the
VU kinds that have a relation, and

    u(1) = log2(1 + n 2**W_V) = W_V + log2(n + 2**-W_V)  ~  W_V + log2 n

for the bits that name one VU and its contents (:func:`unit_fault_bits`).

**A declaration fixed in advance costs ``u(1)``.**  If the declared set ``D``
(``|D| <= f``) does not depend on the challenges, the output is a function
of the correct computation outside ``E``, the values at ``E' = E \\ D``
(which must survive the sampling as before) and the values at ``D``.
Enumerating ``D`` and its values,

    |Y_eta| <= sum_{|D| <= f} 2**kappa(D) * sum_{E' admissible} 2**kappa(E')
            <= (1 + n 2**W_V)**f * 2**U_0 ,

using ``kappa(D) <= sum_{v in D} kappa(v)`` (the union of downstream cuts is
a downstream cut) and ``sum_{j <= f} C(n, j) 2**(j W_V) <= (1 + n 2**W_V)**f``.
So ``U_f <= U_0 + f u(1)`` (:func:`fault_allowance_bits`).  This holds when
``q = 1``: every RU is opened, the q-challenge reveals nothing, and the
prover's best declaration is a function of ``E`` alone.

**A declaration chosen after ``J`` is adaptive.**  For ``q < 1`` the prover
sees which RUs were opened before it declares -- the honest server cannot
know its faults before it replays the opened RUs -- and an adversary uses
this: it corrupts one VU in each of many RUs and pardons whichever ``f`` of
them were opened.  With ``N_J`` corrupted VUs in the opened RUs its
acceptance probability is ``sigma_f(E) = E_J[(1 - s)**max(0, N_J - f)]``
instead of ``sigma_0(E) = E_J[(1 - s)**N_J]``.  Two bounds on it:

* ``(1 - s)**max(0, N_J - f) <= (1 - s)**(N_J - f)``, so ``sigma_f(E) <=
  (1 - s)**-f sigma_0(E)``: every error set admissible with declarations at
  threshold ``eta`` is admissible without them at ``eta (1 - s)**f``, and
  ``U_f <= U_0(eta (1 - s)**f)``.  Rigorous for ``s < 1``; it is the fold at
  a threshold lowered by ``f log2 (1 / (1 - s))`` bits, worth about ``1 / q``
  extra corrupted VUs per declaration when ``s`` is small (one when
  ``q = 1``), and vacuous at ``s = 1``.
* ``(1 - s)**max(0, N_J - f) = max over D subset E, |D| <= f of
  (1 - s)**N_J(E \\ D)`` (declare the opened errors), and a maximum is at most
  the sum, so ``sigma_f(E) <= sum_D sigma_0(E \\ D)``: some ``D`` has
  ``sigma_0(E \\ D) >= eta / (1 + n)**f`` (there are at most ``(1 + n)**f``
  choices), ``E \\ D`` is admissible at that threshold and ``D`` is one of
  the fixed sets above, so ``U_f <= U_0(eta / (1 + n)**f) + f u(1)``.
  Rigorous for every ``s``, including ``s = 1``.

Both are bounds on ``|Y_eta|`` itself, so :func:`declared_bits` returns the
smaller (and, at ``q = 1``, the smaller of those and ``U_0 + f u(1)``),
capped by the interface in ``bound``.  Which is tighter depends on the
policy: for the replay rates a serving run uses, ``f log2 (1 / (1 - s))``
is a fraction of a bit of threshold but the rate ``rho`` multiplies it by
``~ 1 / (q s)``; the second bound's ``f log2 (1 + n)`` bits of threshold
cost more still.  ``U_f - U_0`` is far larger than ``f u(1)`` whenever
``q`` is small: declaring after ``J`` is expensive, and a protocol that
wanted the ``f u(1)`` price would have to take declarations before the
q-challenge, which an honest server can only do for the faults it detects
without replaying (``docs/stress-tests.md``, M6).
"""

from __future__ import annotations

import math
from fractions import Fraction

from veritor.core import (
    Compiled,
    KindTable,
    VerificationPolicy,
    as_kind_table,
)
from veritor.core.description import VERIFICATION

from .bound import BoundOptions, bound, cut_bits


def _units(table: KindTable) -> tuple[int, int]:
    """``(n, W_V)``: the VU copies of the index and the widest cut over the
    kinds with a relation; ``W_V = -1`` if no kind has one."""

    widest = -1
    units = 0
    for row in table.rows:
        if row.role != VERIFICATION:
            continue
        units += row.copies
        if row.size > row.source_inputs + row.source_weights:
            widest = max(widest, cut_bits(row))
    return units, widest


def unit_fault_bits(target: Compiled | KindTable) -> float:
    """``u(1) = log2(1 + n 2**W_V)``: the capacity of one declared VU.

    ``n`` counts every VU of the index (an upper bound on the declarable
    ones) and ``W_V`` is the widest cut (:func:`~veritor.analysis.bound.cut_bits`)
    over the VU kinds that have a relation.  ``0.0`` for an index without
    such a kind: nothing can be declared.
    """

    units, widest = _units(as_kind_table(target))
    if widest < 0 or units == 0:
        return 0.0
    return widest + math.log2(units + 2.0**-widest)


def fault_allowance_bits(target: Compiled | KindTable, max_faults: int) -> float:
    """``f u(1)``: the price of ``max_faults`` declarations fixed before the challenges."""

    if type(max_faults) is not int or max_faults < 0:
        raise ValueError("max_faults must be a nonnegative integer")
    if max_faults == 0:
        return 0.0
    return max_faults * unit_fault_bits(target)


def declared_bits(
    table: KindTable,
    policy: VerificationPolicy,
    eta: Fraction,
    options: BoundOptions,
    max_faults: int,
    base_bits: float,
) -> float:
    """The bits ``bound(table, policy, eta)`` becomes when up to ``max_faults``
    VUs may be declared after the q-challenge; ``base_bits`` is its value
    without declarations.  The minimum of the rigorous bounds of the module
    docstring, before the interface cap (``bound`` applies it).
    """

    if type(max_faults) is not int or max_faults < 0:
        raise ValueError("max_faults must be a nonnegative integer")
    if max_faults == 0:
        return base_bits
    allowance = fault_allowance_bits(table, max_faults)
    units, widest = _units(table)
    if widest < 0 or units == 0:
        return base_bits  # nothing can be declared
    candidates = [bound(table, policy, eta / (1 + units) ** max_faults, options).bits + allowance]
    if policy.s < 1:
        candidates.append(bound(table, policy, eta * (1 - policy.s) ** max_faults, options).bits)
    if policy.q == 1:
        candidates.append(base_bits + allowance)
    return min(candidates)


__all__ = ["declared_bits", "fault_allowance_bits", "unit_fault_bits"]
