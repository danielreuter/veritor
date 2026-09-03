"""``Bound(C, I, theta)``: the output capacity a policy leaves an adversary.

Statement.  Let ``Y_eta`` be the set of outputs of transcripts the verifier
accepts with probability above its threshold ``eta``.  Every such transcript has an error
set ``E`` (verification units (VUs) holding an incorrect gate) with
survival ``sigma(E) > eta`` (see :mod:`veritor.analysis.probability`), and
if all incorrect gates lie inside a union of index nodes ``S_1, ..., S_m``
then at most ``2**(sum_j kappa(S_j))`` outputs are reachable, where
``kappa(S)`` is the width of any downstream cut for ``S``.  Three cuts are
read off the kind table: the node's interface ``Out(S)`` (``out_bits``,
every path out of the node leaves through a declared output), the circuit
outputs reachable from the node (``reach_bits``, every path from the node
to the output ends at one of them) and the narrowest interface of a node
enclosing it (``ancestor_bits``, every path out of the enclosing node
leaves through *its* declared outputs), so a node is charged its
*bottleneck* ``kappa(S) = min(out_bits, reach_bits, ancestor_bits)``
(:attr:`~veritor.core.KindSummary.cut_bits`); the whole output
``out_bits(C)`` caps everything.  Assign every subset ``E'`` of a replay
unit (RU) a cover ``c(E')`` by index nodes, so that ``E`` is covered by
the union of the ``c(E ∩ R_r)``; then

    |Y_eta| <= sum over admissible (l_r)_r of prod_r V_r(l_r),

where ``V_r(l)`` is the total ``2**kappa`` of the *distinct* covers assigned
to the ``l``-subsets of ``R_r``.  The fold chooses covers per kind: the
``l``-subsets of a kind are covered either by the node itself (one cover,
``2**kappa``) or by the covers of their pieces in the child kinds (a
convolution over child copies), whichever total is smaller.  This is at
most the per-set sum ``sum_E 2**kappa(E)`` and can be far below it when
whole RUs may be corrupted.  The reach matters where the interface is
wide but the influence narrow: a decode step, a layer or a matvec inside
one request of a serving run has an interface of the activations and cache
entries it writes, yet can only change that request's remaining tokens.

Admissibility is a knapsack over replay units with costs ``c(l_r)`` and
budget ``Lambda``.  It is solved on a grid of ``buckets`` cost steps of
size ``cost_step`` nats, rounding every cost *down* (this admits more error
sets, never fewer): the grid result is exact for the relaxed survival
``sigma~(E) = prod_r exp(-cost_step * floor(c(l_r) / cost_step))``, which
exceeds ``sigma(E)`` by less than ``exp(cost_step)`` per touched replay
unit.  Independently, a Laplace (Chernoff) bound
``min_t  t Lambda + sum_K n_K ln sum_l V_K(l) e^(-t c(l))`` uses the exact
costs and no grid; the smaller of the two is reported.  Both are capped by
the circuit's own interface ``out_bits(C)``.

Error counts inside one copy are tracked up to ``errors_limit``; subsets
with more errors are lumped together at the cost of ``errors_limit + 1``
errors, again admitting more.  Everything is computed once per kind in
``log2`` with upward rounding (:mod:`veritor.analysis.series`) and weighted
by copy counts; no copy is ever enumerated, and the running time is
polynomial in ``buckets`` and ``errors_limit`` only.

Source gates (``in``/``weight``) are pinned to the boundary or to ``kappa_W``
and never counted in a node's ``out_bits``: a unit holding nothing but
source gates has capacity ``2**0``.  Such a unit is never in the error set
of a transcript the verifier can accept: every input gate is compared with
the public input at the boundary, before any sampling, and a weight gate's
only admissible value is its opening under ``kappa_W``.  The fold therefore
gives a kind with no non-source gate the series of the empty subset alone
(``l = 0``); counting its ``l >= 1`` subsets would add terms of weight one
for error sets that never survive.

Check outputs (:class:`~veritor.core.description.Check`) are declared
outputs the description fixes at a constant and the verifier compares at
the boundary, before any sampling.  They contribute ``0`` to the root's
``out_bits`` and to every node's ``reach_bits`` and ``ancestor_bits``: a
transcript whose check output differs from its constant is rejected
whatever the sampling does, so ``Y_eta`` holds one value at every check
position and a cut may drop those positions without losing a reachable
output.  A node that can only move check outputs therefore has capacity
``2**0`` through them, exactly like one that holds only source gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from veritor.core.compiled import Compiled, as_kind_table
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.identity import Digest
from veritor.core.index import KindSummary, KindTable
from veritor.core.policy import ProbabilityInput, VerificationPolicy, exact_fraction

from .probability import budget, saturation_cost, unit_cost
from .series import (
    NEG_INF,
    ErrorSeries,
    cap,
    convolve,
    empty_series,
    log2_sum,
    multiply,
    power,
    prefix_sums,
    sparse_power,
    unit_series,
)

LOG2E = 1 / math.log(2)


@dataclass(frozen=True, slots=True)
class BoundOptions:
    """Resolution of the fold.

    ``resolution`` cost buckets span the cost ``c(1)`` of one erroneous unit,
    with at most ``max_buckets`` buckets over the whole budget (a coarser
    grid is sound; the Laplace bound then usually wins).  ``max_errors``
    truncates the per-copy error count.  With ``knapsack`` false only the
    Laplace bound is computed: it uses the grid for nothing but the error
    truncation, so ``max_buckets`` can then be large at no cost, which is
    what a run with millions of units per copy or a single-unit cost far
    below ``Lambda / 2048`` needs (the knapsack on a coarse grid would round
    such costs down to zero).
    """

    max_buckets: int = 2048
    resolution: int = 16
    max_errors: int = 256
    knapsack: bool = True

    def __post_init__(self) -> None:
        for name in ("max_buckets", "resolution", "max_errors"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.knapsack) is not bool:
            raise ValueError("knapsack must be a bool")


@dataclass(frozen=True, slots=True)
class BoundResult:
    """``U`` in bits with the relaxations that produced it.

    ``bits`` is the certified capacity: ``min(knapsack_bits, laplace_bits,
    out_bits)`` tightened to an integer count of outputs (a fully checked run
    is exactly ``0.0``); ``capped`` says the circuit interface was the minimum.
    ``knapsack_bits`` is ``inf`` when the knapsack was not computed.
    ``cost_step`` (nats) and ``buckets`` describe the knapsack grid,
    ``errors_limit`` the error-count truncation; ``policy`` and ``eta`` are
    the ``theta`` and threshold bounded.  The result is always an upper
    bound; it is exact for the relaxed admissibility described in the module
    docstring when ``knapsack_bits`` is the minimum.

    ``rho`` is the slope of the Laplace fold: writing its per-kind sum as
    ``Z_K(t) = 1 + eps_K(t)``, the least ``t >= 0`` at which the excess
    ``sum_K n_K eps_K(t)`` is at most one, so that ``laplace_bits <= rho *
    log2 (1 / eta) + log2 e`` (``log2 prod (1 + eps) <= log2 e * sum eps``).
    It does not depend on ``eta`` beyond the error truncation, and is the
    quantity the closed form of :mod:`veritor.analysis.rate` approximates
    from the table's summary numbers; ``inf`` when some error is free.
    """

    bits: float
    capped: bool
    out_bits: int
    knapsack_bits: float
    laplace_bits: float
    cost_step: float
    buckets: int
    errors_limit: int
    policy: VerificationPolicy
    eta: Fraction
    digest: Digest
    rho: float


def bound(
    target: Compiled | KindTable,
    policy: VerificationPolicy,
    eta: ProbabilityInput,
    options: BoundOptions | None = None,
    *,
    max_faults: int = 0,
) -> BoundResult:
    """Fold ``U = Bound(C, I, theta)`` at threshold ``eta`` over the kinds of the index.

    ``target`` is the compiled artifact or its :class:`KindTable`: the fold
    reads nothing else.  ``max_faults`` (``f_max``, mechanism M6) lets the
    prover declare that many VUs after the q-challenge; ``bits`` becomes the
    declared capacity of :mod:`veritor.analysis.faults`, still capped by the
    circuit's interface.
    """

    table = as_kind_table(target)
    if not isinstance(policy, VerificationPolicy):
        raise TypeError("policy must be a VerificationPolicy")
    eta = exact_fraction(eta, name="eta")
    if not 0 <= eta < 1:
        raise ValueError("eta must lie in [0, 1)")
    options = BoundOptions() if options is None else options
    rows = {row.kind: row for row in table.rows}
    out_bits = rows[table.root].out_bits
    fold = _Fold(rows, policy, eta, options)
    replay = [row for row in rows.values() if row.role == REPLAY]
    knapsack = fold.knapsack(replay) if options.knapsack else math.inf
    laplace = fold.laplace(replay)
    bits = min(knapsack, laplace, float(out_bits))
    if max_faults:  # M6: declarations chosen after the q-challenge, see veritor.analysis.faults
        from .faults import declared_bits

        bits = min(declared_bits(table, policy, eta, options, max_faults, bits), float(out_bits))
    return BoundResult(
        bits=_integer_count(max(bits, 0.0)),
        capped=bits >= out_bits,
        out_bits=out_bits,
        knapsack_bits=knapsack,
        laplace_bits=laplace,
        cost_step=fold.step,
        buckets=fold.buckets,
        errors_limit=fold.limit,
        policy=policy,
        eta=eta,
        digest=table.digest,
        rho=fold.slope(replay),
    )


def cut_bits(row: KindSummary) -> int:
    """``kappa`` of a node of the kind: the narrowest of its three downstream cuts.

    The interface ``Out`` (``out_bits``), the circuit outputs the node can
    reach (``reach_bits``) and the narrowest interface of a node enclosing
    it (``ancestor_bits``) are all downstream cuts for every gate of the
    node, so the node may be charged any of them; the fold charges the
    smallest (:attr:`~veritor.core.KindSummary.cut_bits`).
    """

    return row.cut_bits


def _integer_count(bits: float) -> float:
    """``log2`` of the largest integer count consistent with ``bits``.

    ``|Y_eta|`` is an integer, so ``|Y_eta| <= 2**bits`` implies
    ``|Y_eta| <= floor(2**bits)``.  The power is scaled up by ``1 + 2**-45``
    before the floor: that covers the rounding of ``2.0**bits`` itself and
    a few ulps of error in ``bits`` (e.g. ``math.log2(n)`` for an integer
    ``n`` may sit below ``log2 n``), so the count is never below the true
    one; ``log2`` of the count is exact for a power of two and rounded up
    by an ulp otherwise, so the result is never below the true ``log2`` of
    the count; and the result never exceeds ``bits``, which is an upper
    bound by itself.  This removes the upward-rounding slack of the fold
    where that slack is visible, e.g. a fully checked run is exactly ``0.0``
    rather than ``1e-14`` bits.  Above ``2**53`` the count is not an exact
    float and ``bits`` is returned unchanged.
    """

    if not bits < 53.0:
        return bits
    count = math.floor(2.0**bits * (1 + 2.0**-45))
    if count <= 1:
        return 0.0
    log_count = math.log2(count)
    if count & (count - 1):
        log_count = math.nextafter(log_count, math.inf)
    return min(bits, log_count)


class _Fold:
    """The per-policy grid, error truncation and memoised per-kind series."""

    def __init__(
        self,
        rows: dict[str, KindSummary],
        policy: VerificationPolicy,
        eta: Fraction,
        options: BoundOptions,
    ) -> None:
        self.rows = rows
        self.policy = policy
        self.budget = budget(eta)
        first = unit_cost(policy, 1)
        if math.isinf(self.budget) or first == 0.0 or math.isinf(first):
            self.buckets = 1
        else:
            wanted = math.ceil(options.resolution * self.budget / first)
            self.buckets = max(1, min(options.max_buckets, wanted))
        self.step = math.inf if math.isinf(self.budget) else self.budget / self.buckets * (1 + 2.0**-50)
        self.top = self.buckets - 1
        self.limit = self._errors_limit(options.max_errors)
        self._series: dict[str, ErrorSeries] = {}

    # -- costs ---------------------------------------------------------------

    def cost(self, errors: int) -> float:
        return unit_cost(self.policy, errors)

    def bucket(self, cost: float) -> int | None:
        """The grid index of ``cost``, rounded down; ``None`` if inadmissible alone."""

        if math.isinf(cost):
            return None
        if math.isinf(self.step):
            return 0
        index = math.floor(cost / self.step)
        while index > 0 and index * self.step > cost:
            index -= 1
        return index if index <= self.top else None

    def _errors_limit(self, max_errors: int) -> int:
        saturated = self.bucket(saturation_cost(self.policy))
        errors = 1
        while errors < max_errors:
            following = self.bucket(self.cost(errors + 1))
            if following is None or following == saturated:
                break
            errors += 1
        return errors

    # -- per-kind cover weights ---------------------------------------------

    def series(self, kind: str) -> ErrorSeries:
        found = self._series.get(kind)
        if found is not None:
            return found
        row = self.rows[kind]
        if row.role == VERIFICATION:
            if row.size == row.source_inputs + row.source_weights:
                # nothing but source gates: never incorrect, so only l = 0
                result = empty_series()
            else:
                result = unit_series(cut_bits(row))
        else:
            result = empty_series()
            for child, count in row.children:
                if self.rows[child].verification_units == 0:
                    continue
                piece = power(self.series(child), count, self.limit)
                result = piece if len(result.head) == 1 else multiply(result, piece, self.limit)
            result = cap(result, cut_bits(row))
        self._series[kind] = result
        return result

    # -- knapsack over the cost grid ----------------------------------------

    def _sparse(self, series: ErrorSeries) -> tuple[np.ndarray, np.ndarray]:
        """The cost-grid polynomial of one copy: bucket -> log2 weight."""

        merged: dict[int, list[float]] = {}
        for errors, weight in enumerate(series.head):
            index = self.bucket(self.cost(errors))
            if index is not None and weight > NEG_INF:
                merged.setdefault(index, []).append(float(weight))
        if series.tail > NEG_INF:
            index = self.bucket(self.cost(self.limit + 1))
            if index is not None:
                merged.setdefault(index, []).append(series.tail)
        exponents = np.array(sorted(merged), dtype=int)
        values = np.array([float(log2_sum(np.array(merged[e]))) for e in exponents])
        return exponents, values

    def knapsack(self, replay: list[KindSummary]) -> float:
        dense = []
        for row in replay:
            exponents, values = self._sparse(self.series(row.kind))
            dense.append(sparse_power(exponents, values, row.copies, self.top))
        if not dense:
            return 0.0
        if len(dense) == 1:
            return float(log2_sum(dense[0]))
        acc = dense[0]
        for part in dense[1:-1]:
            acc = _pad(convolve(acc, part, self.top), self.top + 1)
        cumulative = prefix_sums(dense[-1])
        return float(log2_sum(acc + cumulative[::-1]))

    # -- Laplace transform bound --------------------------------------------

    def _terms(self, replay: list[KindSummary]) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Per RU kind: copies, ``log2`` cover weights and costs in bits, ``l = 0`` first.

        Terms of infinite cost (error sets that never survive) and of zero
        weight are dropped; the ``l = 0`` term, weight one at cost zero, is
        always kept and first.
        """

        terms = []
        for row in replay:
            series = self.series(row.kind)
            costs = np.array([self.cost(errors) for errors in range(len(series.head))])
            weights = np.append(series.head, series.tail)
            costs = np.append(costs, self.cost(self.limit + 1))
            keep = np.isfinite(costs) & (weights > NEG_INF)
            terms.append((row.copies, weights[keep], costs[keep] * LOG2E))
        return terms

    def slope(self, replay: list[KindSummary]) -> float:
        """The least ``t`` at which the excess ``sum_K n_K (Z_K(t) - 1)`` is at most one.

        The excess is decreasing in ``t`` (every cost is nonnegative), so a
        doubling search brackets the root and a bisection closes in; the
        upper end, where the excess was computed (rounded up) to be at most
        one, is returned.  ``inf`` when a term of positive weight costs
        nothing (``q = 0`` or ``s = 0``): the excess then never falls to one.
        """

        weights = []
        costs = []
        for copies, kind_weights, kind_costs in self._terms(replay):
            if len(kind_weights) > 1:
                weights.append(kind_weights[1:] + math.nextafter(math.log2(copies), math.inf))
                costs.append(kind_costs[1:])
        if not weights:
            return 0.0
        weight = np.concatenate(weights)
        cost = np.concatenate(costs)

        def excess(t: float) -> float:
            return float(log2_sum(weight - t * cost))

        if excess(0.0) <= 0.0:
            return 0.0
        if np.any(cost <= 0.0):
            return math.inf
        low, high = 0.0, 1.0
        while excess(high) > 0.0:
            low, high = high, 2.0 * high
            if high > 2.0**80:
                return math.inf
        for _ in range(64):
            middle = (low + high) / 2
            if excess(middle) <= 0.0:
                high = middle
            else:
                low = middle
        return high

    def laplace(self, replay: list[KindSummary]) -> float:
        terms = self._terms(replay)

        def value(t: float) -> float:
            total = t * self.budget * LOG2E if t else 0.0
            for copies, weights, costs in terms:
                total += copies * float(log2_sum(weights - t * costs))
            return total

        if math.isinf(self.budget):
            return value(0.0)
        scale = 1.0 / self.budget
        samples = [0.0] + [scale * 2.0**k for k in range(-24, 40)]
        values = [value(t) for t in samples]
        best = min(range(len(samples)), key=values.__getitem__)
        low = samples[max(best - 1, 0)]
        high = samples[min(best + 1, len(samples) - 1)]
        golden = (math.sqrt(5) - 1) / 2
        a, b = high - golden * (high - low), low + golden * (high - low)
        fa, fb = value(a), value(b)
        for _ in range(60):
            if fa < fb:
                high, b, fb = b, a, fa
                a = high - golden * (high - low)
                fa = value(a)
            else:
                low, a, fa = a, b, fb
                b = low + golden * (high - low)
                fb = value(b)
        return min(values[best], fa, fb)


def _pad(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return values[:length]
    return np.concatenate([values, np.full(length - len(values), NEG_INF)])


__all__ = ["BoundOptions", "BoundResult", "bound", "cut_bits"]
