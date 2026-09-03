from fractions import Fraction

import pytest

from veritor.analysis import (
    BoundOptions,
    CostParameters,
    Optimization,
    PolicyGrid,
    bound,
    cost,
    optimize,
)
from veritor.core import VerificationPolicy

ETA = Fraction(1, 4)


def brute_force(compiled, grid, *, max_bits=None, max_cost=None, accept=None):
    """Every grid point, ranked the way ``optimize`` documents."""

    candidates = []
    for policy in grid.policies():
        if accept is not None and not accept(policy):
            continue
        result, expected = bound(compiled, policy, ETA), cost(compiled, policy)
        if max_bits is not None and result.bits <= max_bits:
            candidates.append(((expected.total, result.bits), policy))
        if max_cost is not None and expected.total <= max_cost:
            candidates.append(((result.bits, expected.total), policy))
    if not candidates:
        return None
    return candidates[min(range(len(candidates)), key=lambda i: candidates[i][0])][1]


def test_grid_lists_every_combination_and_validates():
    grid = PolicyGrid.uniform(2)
    assert grid.q == grid.s == (Fraction(0), Fraction(1, 2), Fraction(1))
    assert len(list(grid.policies())) == 9
    assert [(p.q, p.s) for p in grid.policies()] == [
        (q, s) for q in grid.q for s in grid.s
    ]
    assert PolicyGrid(("1/3",), (1,)).q == (Fraction(1, 3),)
    with pytest.raises(ValueError, match="q must list"):
        PolicyGrid((), (1,))
    with pytest.raises(ValueError, match="s must list"):
        PolicyGrid((1,), (2,))
    with pytest.raises(ValueError, match="steps"):
        PolicyGrid.uniform(0)


@pytest.mark.parametrize("sizes", [(3, 2), (2, 2, 2)])
def test_cheapest_policy_under_a_capacity_limit(make_compiled, sizes):
    compiled = make_compiled(sizes)
    grid = PolicyGrid.uniform(4)
    limit = 4 * sum(sizes)  # half of the interface

    chosen = optimize(compiled, ETA, grid, max_bits=limit)

    assert isinstance(chosen, Optimization)
    assert chosen.policy == brute_force(compiled, grid, max_bits=limit)
    assert chosen.bound.bits <= limit
    assert chosen.bound == bound(compiled, chosen.policy, ETA)
    assert chosen.bound.eta == ETA
    assert chosen.cost == cost(compiled, chosen.policy)
    assert chosen.evaluated == 25
    # the dual at the chosen cost cannot beat the chosen bound
    dual = optimize(compiled, ETA, grid, max_cost=chosen.cost.total)
    assert isinstance(dual, Optimization)
    assert dual.policy == brute_force(compiled, grid, max_cost=chosen.cost.total)
    assert dual.bound.bits <= chosen.bound.bits
    assert dual.cost.total <= chosen.cost.total


def test_full_checking_is_the_only_way_to_zero_capacity(make_compiled):
    compiled = make_compiled((2, 2))
    chosen = optimize(compiled, 0, PolicyGrid.uniform(2), max_bits=0.5)

    assert isinstance(chosen, Optimization)
    assert chosen.policy == VerificationPolicy(1, 1)
    assert chosen.bound.eta == 0
    assert optimize(compiled, ETA, PolicyGrid.uniform(2), max_bits=-1) is None
    assert optimize(compiled, ETA, PolicyGrid.uniform(2), max_cost=0) is None


def test_extra_constraints_and_options_are_honoured(make_compiled):
    compiled = make_compiled((3, 2))
    grid = PolicyGrid.uniform(4)
    chosen = optimize(
        compiled,
        ETA,
        grid,
        max_bits=40,
        accept=lambda policy: policy.q <= Fraction(1, 2),
        parameters=CostParameters(3, 2),
        bound_options=BoundOptions(max_buckets=8),
    )

    assert isinstance(chosen, Optimization)
    assert chosen.policy.q <= Fraction(1, 2)
    assert chosen.evaluated == 15
    assert chosen.bound.buckets <= 8
    assert chosen.cost == cost(compiled, chosen.policy, CostParameters(3, 2))


def test_optimize_needs_exactly_one_objective(make_compiled):
    compiled = make_compiled((1,))
    with pytest.raises(ValueError, match="exactly one"):
        optimize(compiled, ETA, PolicyGrid.uniform(1))
    with pytest.raises(ValueError, match="exactly one"):
        optimize(compiled, ETA, PolicyGrid.uniform(1), max_bits=1, max_cost=1)
