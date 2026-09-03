"""The closed-form rate: sound against the exact union, and how far from the fold on the serving tables.

``rho * log2 (1 / eta) + log2 e`` is proved in :mod:`veritor.analysis.rate`
to bound ``log2 |Y_eta|`` directly, so on the small circuits it is checked
against the brute-forced union of accepted outputs, not against the fold.
Against the fold it is a *comparison*: the fold's own slope
:attr:`BoundResult.rho` is the same construction on the exact per-kind
series, and the two are recorded side by side over the frontier's policy
grid.  Observed on ``serving_table`` at request/cell:

* ``FRONTIER_SHAPE`` (2048 requests of 512 + 512 tokens, ``R = 2048``,
  ``W_R = 8192``, ``W_V = 16``, ``m = 1.7e10``, ``l_0 = 187``; the full
  ``DEFAULT_GRID x DEFAULT_ETAS``, 84 points, 20 minutes, run offline).  At
  ``s >= 1/64`` the whole-RU channel binds (``l* = l_0``):
  ``rho_closed / rho_fold`` in ``[0.9960, 1.0018]`` and, on the 42 points
  where the fold is neither capped nor saturated, ``capacity /
  laplace_bits`` in ``[0.9961, 1.0031]`` -- the closed form pays
  ``log2 (l_0 (l_0 + 1)) = 15`` bookkeeping bits on ``8203`` and, at ``s =
  1/8``, saves the fold's per-``l`` caps.  At ``s = 1/512`` the scattered
  channel binds (``l* = 1``): ``rho_closed / rho_fold = 1.0166`` but
  ``capacity / laplace_bits`` in ``[1.17, 1.27]`` on the 12 unsaturated
  points, because the ``rho * lambda + log2 e`` *form* is loose against the
  fold's minimum over ``t`` wherever many errors are affordable (``lambda
  >> c(1)``), by about ``log2 B`` bits per error (module docstring, (i)).
* :data:`GRID_SHAPE` below (the same table at 1/32 scale, run here):
  ``rho_closed / rho_fold`` in ``[1.015, 1.050]``; on the points where the
  fold is neither capped nor saturated, ``capacity / laplace_bits`` in
  ``[1.015, 1.28]``.

The tolerances asserted are these observed ranges with a margin, not a
theorem: the closed form may fall below the fold (it lumps the RU's covers
where the fold caps them per ``l``) and above it (bookkeeping, two levels).
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from veritor.analysis import RateResult, bound, capacity_from_rate, rate
from veritor.analysis.bound import LOG2E
from veritor.analysis.probability import unit_cost
from veritor.analysis.reference import accepted_outputs, transcript_outputs
from veritor.analysis.series import log2_binomials
from veritor.core import Compiled, VerificationPolicy
from veritor.core.description import REPLAY, VERIFICATION
from veritor.evaluation import ServingShape, serving_table
from veritor.evaluation.frontier import DEFAULT_GRID, FRONTIER_OPTIONS, FRONTIER_SHAPE

from .conftest import bottlenecked, build_compiled, paper_example, random_compiled

TOLERANCE = 1e-6

POLICIES = [  # (theta, eta)
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
    (VerificationPolicy(Fraction(1, 3), Fraction(1, 5)), Fraction(1, 100)),
    (VerificationPolicy(Fraction(9, 10), Fraction(9, 10)), Fraction(1, 10)),
    (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    (VerificationPolicy(1, 1), Fraction(1, 2)),
    (VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 3)),
    (VerificationPolicy(Fraction(1, 16), Fraction(1, 64)), Fraction(1, 2)),
    (VerificationPolicy(0, 1), Fraction(1, 2)),
]

GRID_SHAPE = ServingShape(
    vocab=32768,
    d_model=1024,
    heads=8,
    layers=2,
    prompt=32,
    generated=32,
    requests=64,
    batch=8,
    hidden_multiplier=4,
)
"""The frontier's request/cell table at a scale the whole policy grid folds in seconds."""

SMALL = {
    "paper": lambda: paper_example(2, False),
    "paper-split": lambda: paper_example(2, True),
    "chain": lambda: build_compiled((3, 2), width=1),
    "bottlenecked": lambda: bottlenecked(2, width=1),
    **{f"random{seed}": (lambda seed=seed: random_compiled(seed)) for seed in range(6)},
}


def inputs_for(compiled: Compiled) -> list[int]:
    """One nonzero value per input gate, within the gate width (``1`` for one-bit words)."""

    if not compiled.circuit.inputs:
        return []
    width = compiled.circuit[compiled.circuit.inputs[0]].width
    return [1 + i % max(1, (1 << width) - 1) for i in range(compiled.index.input_count)]


@pytest.fixture(scope="module")
def unions():
    """Every transcript's output, per small circuit: the exact ``Y_eta`` for every policy."""

    compiled = {name: make() for name, make in SMALL.items()}
    return compiled, {
        name: transcript_outputs(c, inputs_for(c)) for name, c in compiled.items()
    }


@pytest.mark.parametrize("name", list(SMALL))
def test_the_closed_form_is_above_the_exact_union(unions, name):
    """Soundness, directly: ``log2 |Y_eta| <= rho * log2 (1 / eta) + log2 e`` for every policy."""

    compiled, outputs = unions[0][name], unions[1][name]
    for policy, eta in POLICIES:
        result = rate(compiled, policy)
        assert result == rate(
            compiled.kind_table(), policy
        )  # the fold reads the table alone
        union = math.log2(len(accepted_outputs(outputs, policy, eta)))
        assert union <= result.capacity(eta) + TOLERANCE
        assert result.capacity(eta) == capacity_from_rate(result.rho, eta)
        if policy.q == 0:
            assert math.isinf(result.rho)  # nothing is ever checked: no finite slope
        elif policy.q == policy.s == 1:
            assert result.rho == 0.0 and union == 0.0  # everything is checked
        else:
            assert 0 < result.rho < math.inf


@pytest.mark.parametrize("name", list(SMALL))
def test_the_folds_slope_bounds_its_own_laplace_bound(unions, name):
    """``laplace_bits <= rho_fold * log2 (1 / eta) + log2 e`` is a theorem about the fold; check it."""

    compiled = unions[0][name]
    for policy, eta in POLICIES:
        result = bound(compiled, policy, eta)
        if math.isinf(result.rho):
            assert policy.q == 0
            continue
        assert result.laplace_bits <= capacity_from_rate(result.rho, eta) + TOLERANCE
        assert result.bits <= capacity_from_rate(result.rho, eta) + TOLERANCE


def test_the_four_numbers_and_the_channels_on_the_frontier_table():
    """``R``, ``W_R``, ``W_V``, ``m`` and ``l_0`` read off the request/cell table, and the two named channels."""

    table = serving_table(FRONTIER_SHAPE, "request", "cell")
    tokens = FRONTIER_SHAPE.generated
    result = rate(table, VerificationPolicy(Fraction(1, 2), 1))

    assert result.replay_units == FRONTIER_SHAPE.requests == 2048
    assert (
        result.replay_bits == 8192 == tokens * 16
    )  # a request's bottleneck: its generated tokens
    assert result.verification_bits == 16  # a cell: one dot product, one word
    fallible = {
        row.kind
        for row in table.rows
        if row.role == VERIFICATION
        and row.size > row.source_inputs + row.source_weights
    }
    per_request = {
        sum(count for kind, count in row.verification_kinds if kind in fallible)
        for row in table.rows
        if row.role == REPLAY
        and any(kind in fallible for kind, _ in row.verification_kinds)
    }
    assert (
        per_request == {result.verification_units}
        and result.verification_units == 16936948224
    )
    # l_0: the first l at which l W_V + log2 C(m, l) reaches W_R
    binomials = log2_binomials(result.verification_units, 200)
    crossing = [l for l in range(1, 201) if 16 * l + binomials[l] >= 8192]
    assert result.lumped_at == crossing[0] == 187

    def channel(l: int, value: float, policy: VerificationPolicy) -> float:
        position = math.log2(2048) + math.log2(l * (l + 1))
        return (position + value) / (unit_cost(policy, l) * LOG2E)

    for policy in (
        VerificationPolicy(Fraction(1, 2), 1),
        VerificationPolicy(Fraction(1, 8), Fraction(1, 8)),
    ):
        result = rate(table, policy)
        m = result.verification_units
        assert result.scattered == pytest.approx(
            channel(1, 16 + math.log2(m), policy), rel=1e-9
        )
        assert result.whole == pytest.approx(channel(187, 8192, policy), rel=1e-9)
        assert result.rho >= max(result.scattered, result.whole) * (1 - 1e-12)
        assert 1 <= result.binding <= result.lumped_at
    # s = 1: every l costs the same log2 (1 / (1 - q)) = 1 bit, so the widest cover, the whole RU, binds
    whole = rate(table, VerificationPolicy(Fraction(1, 2), 1))
    assert whole.binding == 187 and whole.rho == pytest.approx(
        8192 + 11 + math.log2(187 * 188), rel=1e-9
    )
    # tiny q s: a single error costs almost nothing while a whole RU still costs about q log2 e
    scattered = rate(table, VerificationPolicy(Fraction(1, 8192), Fraction(1, 512)))
    assert (
        scattered.binding == 1
        and scattered.rho == scattered.scattered > scattered.whole
    )


def test_rho_grows_as_the_policy_relaxes():
    table = serving_table(GRID_SHAPE, "request", "cell")
    for s in (Fraction(1), Fraction(1, 8), Fraction(1, 64)):
        rhos = [rate(table, VerificationPolicy(q, s)).rho for q in DEFAULT_GRID.q]
        assert rhos == sorted(rhos)  # smaller q: cheaper errors, steeper slope
    for q in (Fraction(1, 2), Fraction(1, 128)):
        rhos = [rate(table, VerificationPolicy(q, s)).rho for s in DEFAULT_GRID.s]
        assert rhos == sorted(rhos)


def test_the_closed_form_against_the_fold_over_the_frontier_grid():
    """Recorded, not proved: the ranges of the module docstring, with a margin, on :data:`GRID_SHAPE`."""

    table = serving_table(GRID_SHAPE, "request", "cell")
    out_bits = next(row.out_bits for row in table.rows if row.kind == table.root)
    slopes = []
    capacities = []
    for policy in DEFAULT_GRID.policies():
        closed = rate(table, policy)
        # at eta = 0 the budget is infinite and the Laplace bound is its value at t = 0, the
        # total cover weight: the ceiling the fold saturates at as lambda grows
        ceiling = bound(table, policy, 0, FRONTIER_OPTIONS).laplace_bits
        for eta in (Fraction(1, 2), Fraction(1, 100)):
            fold = bound(table, policy, eta, FRONTIER_OPTIONS)
            slopes.append(closed.rho / fold.rho)
            # past the output cap or within a few bits of the ceiling, rho * lambda says nothing
            saturated = (
                fold.laplace_bits >= out_bits or fold.laplace_bits >= 0.98 * ceiling
            )
            if not saturated:
                capacities.append(closed.capacity(eta) / fold.laplace_bits)
    assert 0.98 <= min(slopes) and max(slopes) <= 1.10  # observed [1.015, 1.050]
    assert len(capacities) >= 12
    assert 0.98 <= min(capacities) and max(capacities) <= 1.40  # observed [1.015, 1.28]


def test_degenerate_tables_and_inputs(make_compiled):
    compiled = make_compiled((2,))
    table = compiled.kind_table()
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
    # no fallible VU: every VU is nothing but source gates, so nothing can ever be wrong
    sourced = replace(
        table,
        rows=tuple(
            replace(row, size=row.source_inputs + row.source_weights)
            if row.role == VERIFICATION
            else row
            for row in table.rows
        ),
    )
    empty = rate(sourced, policy)
    assert empty == RateResult(0.0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, policy, table.digest)
    assert empty.capacity(Fraction(1, 2)) == LOG2E
    assert capacity_from_rate(0.0, 0) == LOG2E
    # eta = 0 admits everything: infinite budget, infinite capacity, unless the slope is zero
    assert capacity_from_rate(1.0, 0) == math.inf
    assert capacity_from_rate(math.inf, Fraction(1, 2)) == math.inf
    assert capacity_from_rate(2.0, Fraction(1, 4)) == pytest.approx(4 + LOG2E, rel=1e-9)
    with pytest.raises(ValueError, match="nonnegative"):
        capacity_from_rate(-1.0, Fraction(1, 2))
    with pytest.raises(ValueError, match="eta"):
        capacity_from_rate(1.0, 1)
    with pytest.raises(TypeError, match="VerificationPolicy"):
        rate(compiled, (1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compiled"):
        rate(compiled.circuit, policy)  # type: ignore[arg-type]
    # s = 0 leaves every VU unchecked at any q: free errors, no slope
    assert math.isinf(rate(table, VerificationPolicy(1, 0)).rho)
    # a table whose VU bottlenecks are zero bits still pays the position term
    zero = replace(
        table,
        rows=tuple(
            replace(row, out_bits=0) if row.role == VERIFICATION else row
            for row in table.rows
        ),
    )
    positional = rate(zero, policy)
    assert positional.verification_bits == 0 and positional.rho > 0
    assert np.isfinite(positional.rho)
