from decimal import Decimal, getcontext
from fractions import Fraction

import pytest

from veritor.analysis import (
    admissible,
    budget,
    saturation_cost,
    survival,
    survival_factor,
    unit_cost,
)
from veritor.core import VerificationPolicy

getcontext().prec = 60


def exact_ln(value: Fraction) -> Decimal:
    return (Decimal(value.numerator) / Decimal(value.denominator)).ln()


@pytest.mark.parametrize(
    ("q", "s", "errors", "expected"),
    [
        (0, Fraction(3, 5), 2, Fraction(1)),
        (Fraction(2, 3), 0, 2, Fraction(1)),
        (1, 1, 1, Fraction(0)),
        (1, 1, 0, Fraction(1)),
        (1, Fraction(1, 2), 2, Fraction(1, 4)),
        (Fraction(1, 2), Fraction(1, 2), 1, Fraction(3, 4)),
    ],
)
def test_survival_factor_is_exact_at_the_endpoints(q, s, errors, expected):
    assert survival_factor(VerificationPolicy(q, s), errors) == expected


def test_survival_multiplies_over_replay_units_and_ignores_positions():
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))

    assert survival(policy, (2, 0)) == Fraction(5, 8)
    assert survival(policy, (1, 1)) == Fraction(9, 16)
    assert survival(policy, ()) == 1
    assert survival(policy, (2, 0)) > survival(policy, (1, 1))


def test_admissibility_is_strict_at_eta():
    policy = VerificationPolicy(1, Fraction(1, 2))
    eta = Fraction(1, 2)

    assert survival(policy, (1,)) == eta
    assert not admissible(policy, eta, (1,))
    assert admissible(policy, eta, (0,))
    assert admissible(policy, Fraction(1, 4), (1,))


def test_error_counts_must_be_nonnegative_integers():
    policy = VerificationPolicy(1, 1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        survival_factor(policy, -1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        survival(policy, (True,))


@pytest.mark.parametrize(
    ("policy", "eta"),
    [
        (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
        (VerificationPolicy(Fraction(1, 3), Fraction(1, 7)), Fraction(1, 10**9)),
        (
            VerificationPolicy(Fraction(999, 1000), Fraction(1, 1000)),
            Fraction(1, 10**30),
        ),
        (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    ],
)
def test_costs_round_down_and_the_budget_rounds_up(policy, eta):
    exact_budget = -exact_ln(eta)
    assert Decimal(budget(eta)) >= exact_budget
    assert Decimal(budget(eta)) - exact_budget < Decimal(2) ** -30
    previous = 0.0
    for errors in range(40):
        exact = -exact_ln(survival_factor(policy, errors))
        cost = unit_cost(policy, errors)
        assert Decimal(cost) <= exact
        assert exact - Decimal(cost) < Decimal(2) ** -30
        assert cost >= previous
        previous = cost
    if policy.q < 1:
        assert previous <= saturation_cost(policy)
        assert survival_factor(policy, 10**6) > 1 - policy.q


def test_costs_at_the_endpoints():
    assert unit_cost(VerificationPolicy(1, 1), 1) == float("inf")
    assert unit_cost(VerificationPolicy(1, 1), 0) == 0.0
    assert unit_cost(VerificationPolicy(0, 1), 5) == 0.0
    assert budget(Fraction(0)) == float("inf")
    assert saturation_cost(VerificationPolicy(1, 1)) == float("inf")
    assert saturation_cost(VerificationPolicy(Fraction(1, 2), 1)) == pytest.approx(
        0.6931471805599453
    )


@pytest.mark.parametrize("eta", [Fraction(1), Fraction(5, 4), Fraction(-1, 8), 0, 0.5])
def test_budget_needs_a_fraction_in_the_half_open_unit_interval(eta):
    with pytest.raises(ValueError, match="eta"):
        budget(eta)


def test_costs_stay_finite_far_below_the_float_range():
    policy = VerificationPolicy(1, Fraction(1, 2))
    eta = Fraction(1, 2) ** 2000

    assert unit_cost(policy, 1500) == pytest.approx(1500 * 0.6931471805599453, rel=1e-9)
    assert budget(eta) == pytest.approx(2000 * 0.6931471805599453, rel=1e-9)
    assert admissible(policy, eta, (1999,))
    assert not admissible(policy, eta, (2000,))
