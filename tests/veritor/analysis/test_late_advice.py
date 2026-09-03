"""The prices quoted in ``docs/notes/late-advice.md``, pinned at the two operating points.

The headline policy is the point ``veritor.evaluation.global_estimate``
lands on (its capacity is the closed-form ``rate``, so the adaptive charges
are the same threshold shifts ``declared_bits`` applies, read off the rate);
the simulation policy is ``theta = (1/2, 1/8)`` on the small datacenter
run's kind table, where the fold is exercised directly.  Every number the
note's section 6 quotes is asserted here to the precision the note prints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

from veritor.analysis.bound import BoundOptions, bound
from veritor.analysis.faults import unit_fault_bits
from veritor.analysis.rate import RateResult, rate
from veritor.constructors.cluster import ClusterG
from veritor.constructors.lm import random_parameters
from veritor.core import KindTable, VerificationPolicy, as_kind_table, make_isa_gate_set
from veritor.core.description import REPLAY, VERIFICATION
from veritor.evaluation.global_estimate import Inputs, estimate
from veritor.evaluation.serving import ServingShape, serving_table
from veritor.research import Compile
from veritor.simulation.datacenter import ETA, POLICY, small_config
from veritor.simulation.workload import simulate

LAMBDA = 40.0  # log2 (1 / eta) at eta = 2^-40


def units(table: KindTable) -> int:
    return sum(row.copies for row in table.rows if row.role == VERIFICATION)


def poisson_quantile(pardons: int, eta: float) -> float:
    """The largest mean ``mu`` with ``P[Poisson(mu) <= pardons] > eta``."""

    def cdf(mu: float) -> float:
        return sum(
            math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))
            for k in range(pardons + 1)
        )

    low, high = 0.0, 1000.0
    for _ in range(200):
        middle = (low + high) / 2
        if cdf(middle) > eta:
            low = middle
        else:
            high = middle
    return low


@dataclass(frozen=True, slots=True)
class Point:
    table: KindTable
    policy: VerificationPolicy
    rate: RateResult
    rho: float  # the slope the note quotes for the point
    capacity: float  # U_0 at eta = 2^-40, the note's U

    @property
    def q(self) -> float:
        return float(self.policy.q)

    @property
    def s(self) -> float:
        return float(self.policy.s)

    def post_j_unit(self) -> float:
        """Bound (i): ``rho log2 (1 / (1 - s))`` per declaration."""

        return self.rho * math.log2(1 / (1 - self.s))

    def post_j_union(self) -> float:
        """Bound (ii): ``rho log2 (1 + |S|) + u(1)`` per declaration."""

        return self.rho * math.log2(1 + units(self.table)) + unit_fault_bits(self.table)


# -- the headline policy -------------------------------------------------------


@pytest.fixture(scope="module")
def headline() -> Point:
    est = estimate()
    shape = replace(est.inputs.shape, requests=est.inputs.requests)
    table = serving_table(shape, "request", "cell")
    policy = VerificationPolicy(Fraction(est.q), Fraction(est.s))
    result = rate(table, policy)
    assert result.rho == est.rho
    return Point(table, policy, result, est.rho, est.capacity_bits)


def test_headline_operating_point_is_the_documented_one(headline: Point) -> None:
    assert headline.q == pytest.approx(1.6e-8, rel=0.03)
    assert headline.s == pytest.approx(8.9e-3, rel=0.01)
    assert headline.rho == pytest.approx(4.74e11, rel=0.01)
    assert headline.capacity == pytest.approx(1.90e13, rel=0.01)
    assert headline.rate.binding == 1  # the scattered channel binds
    assert headline.rate.replay_bits == 8192 and headline.rate.verification_bits == 16
    assert headline.rate.replay_units == pytest.approx(2.93e13, rel=0.01)
    assert math.log2(units(headline.table)) == pytest.approx(78.7, abs=0.05)
    assert unit_fault_bits(headline.table) == pytest.approx(94.7, abs=0.05)


def test_headline_post_j_declaration_is_the_unit_price_over_q(headline: Point) -> None:
    unit = unit_fault_bits(headline.table)
    per_declaration = headline.post_j_unit()
    assert per_declaration == pytest.approx(6.12e9, rel=0.01)
    assert headline.post_j_union() == pytest.approx(3.73e13, rel=0.01)
    assert per_declaration < headline.post_j_union()  # bound (i) is the charge
    assert per_declaration / headline.capacity == pytest.approx(3.23e-4, rel=0.01)
    # the conservation law: leverage 1 / q on the pre-J price, within 2%
    assert per_declaration / (unit / headline.q) == pytest.approx(1.015, abs=0.01)


def test_headline_post_s_pardon_is_the_unit_price_over_q_s(headline: Point) -> None:
    unit = unit_fault_bits(headline.table)
    leverage = 1 / (headline.q * headline.s)
    assert unit * leverage == pytest.approx(6.77e11, rel=0.01)
    # at eta = 2^-40 the first pardon buys mu_1 - mu_0 = 3.47 more caught errors
    first = poisson_quantile(1, 2.0**-40) - poisson_quantile(0, 2.0**-40)
    assert poisson_quantile(0, 2.0**-40) == pytest.approx(
        LAMBDA * math.log(2), rel=1e-6
    )
    assert first == pytest.approx(3.47, abs=0.01)
    first_pardon = first * leverage * (unit + 1)
    assert first_pardon == pytest.approx(2.37e12, rel=0.01)
    assert first_pardon / headline.capacity == pytest.approx(0.125, abs=0.003)
    # 1 / s times the post-J price, up to the Poisson factor
    assert first_pardon / headline.post_j_unit() == pytest.approx(
        first / headline.s, rel=0.03
    )


def test_headline_ru_scope_pardons(headline: Point) -> None:
    n = headline.rate.replay_units
    pre_j = headline.rate.replay_bits + math.log2(n)
    assert pre_j == pytest.approx(8237, abs=1)
    shift = math.log2(1 + headline.q * n / (1 - headline.q))
    assert shift == pytest.approx(18.8, abs=0.05)
    assert headline.rho * shift / headline.capacity == pytest.approx(0.47, abs=0.01)
    first = poisson_quantile(1, 2.0**-40) - poisson_quantile(0, 2.0**-40)
    attack = first / headline.q * headline.rate.replay_bits
    assert attack / headline.capacity == pytest.approx(0.095, abs=0.003)


def test_headline_source_and_late_lowering_prices(headline: Point) -> None:
    shape = replace(Inputs().shape, requests=Inputs().requests)
    weights = shape.weight_count
    assert math.log2(weights) + 16 == pytest.approx(51.9, abs=0.05)
    readers = shape.context  # one reader per position of the request
    assert readers == 1024
    assert readers * headline.post_j_unit() / headline.capacity == pytest.approx(
        0.33, abs=0.01
    )
    one_bit_per_ru = headline.rate.replay_units * math.log2(2)
    assert one_bit_per_ru / headline.capacity == pytest.approx(1.55, abs=0.02)


def test_recording_costs_at_the_two_shapes() -> None:
    inputs = Inputs()
    shape = replace(inputs.shape, requests=inputs.requests)
    table = serving_table(shape, "request", "cell")
    request = next(
        row for row in table.rows if row.role == REPLAY and row.replay_cost > 0
    )
    flops = 2 * request.replay_cost / 3  # a cost unit is one gate, a MAC is three
    assert flops == pytest.approx(1.34e14, rel=0.01)
    assert 2 * request.interior_count / flops == pytest.approx(2.53e-4, rel=0.01)
    assert 2 * request.interior_count == pytest.approx(33.9e9, rel=0.01)
    hashing = inputs.hash_units * request.interior_count / request.replay_cost
    assert hashing == pytest.approx(0.79, abs=0.005)
    assert 2 * shape.state_size(shape.context) == pytest.approx(2.68e9, rel=0.01)
    assert 2 * shape.context == 2048

    gpt2 = ServingShape(
        vocab=50257,
        d_model=768,
        heads=12,
        layers=12,
        prompt=100,
        generated=100,
        requests=1,
        hidden_multiplier=4,
    )
    small = next(
        row
        for row in serving_table(gpt2, "request", "cell").rows
        if row.role == REPLAY and row.replay_cost > 0
    )
    small_flops = 2 * small.replay_cost / 3
    assert small_flops == pytest.approx(5.77e10, rel=0.01)
    assert 2 * small.interior_count / small_flops == pytest.approx(3.03e-3, rel=0.01)
    assert 2 * small.interior_count == pytest.approx(175e6, rel=0.01)
    assert (
        inputs.hash_units * small.interior_count / small.replay_cost
        == pytest.approx(9.5, abs=0.05)
    )
    assert 2 * gpt2.state_size(gpt2.context) == pytest.approx(7.4e6, rel=0.01)


# -- the simulation policy -----------------------------------------------------


@pytest.fixture(scope="module")
def small() -> KindTable:
    config = small_config()
    shape = config.shape
    parameters = random_parameters(shape, config.parameters_seed)
    simulation = simulate(config.workload, shape, parameters)
    advice = simulation.schedule.encode()
    constructor = ClusterG(
        shape, config.workload.pods, config.workload.slots, config.workload.steps
    )
    compilation = Compile(
        constructor,
        simulation.requests,
        advice,
        make_isa_gate_set(shape.width),
        limits=config.compilation_limits,
        max_advice_bits=8 * len(advice),
    )
    return as_kind_table(compilation.compiled)


def test_simulation_table_is_the_one_stress_tests_quote(small: KindTable) -> None:
    assert units(small) == 3791
    assert unit_fault_bits(small) == pytest.approx(75.89, abs=0.005)
    result = rate(small, POLICY)
    assert result.verification_bits == 64 and result.replay_bits == 224
    assert sum(row.copies for row in small.rows if row.role == REPLAY) == 20
    assert next(row for row in small.rows if row.kind == small.root).out_bits == 480
    assert result.replay_bits + math.log2(result.replay_units) == pytest.approx(
        228.2, abs=0.05
    )


def test_simulation_fold_is_saturated_so_declarations_add_nothing(
    small: KindTable,
) -> None:
    """Twenty RUs at ``q = 1/2`` all fit in the budget: every error set is
    admissible, the uncapped fold is the sum of the RU covers whatever the
    threshold, and the interface caps it at 480 bits with or without a fault
    budget.  Read at a fixed error truncation (a fine grid): the default grid
    moves the truncation with ``eta`` and with it the slack ``log2 (limit +
    1)`` per saturated RU, so the default fold is not monotone in ``eta``."""

    fine = BoundOptions(knapsack=False, max_buckets=1 << 16)
    at_eta = bound(small, POLICY, ETA, fine)
    lowered = bound(small, POLICY, ETA * (1 - POLICY.s), fine)
    divided = bound(small, POLICY, ETA / (1 + units(small)), fine)
    assert at_eta.laplace_bits == pytest.approx(1894.5, abs=0.05)
    assert lowered.laplace_bits == at_eta.laplace_bits
    assert divided.laplace_bits == at_eta.laplace_bits
    assert at_eta.errors_limit == lowered.errors_limit == divided.errors_limit
    assert at_eta.rho == pytest.approx(755.9, abs=0.05)

    default = bound(small, POLICY, ETA)
    assert default.bits == 480.0 and default.capped
    assert default.knapsack_bits == pytest.approx(1895.8, abs=0.05)
    assert bound(small, POLICY, ETA, max_faults=1).bits == 480.0

    # the slopes the note quotes: the marginal charges the fold would make
    point = Point(small, POLICY, rate(small, POLICY), default.rho, 480.0)
    assert point.post_j_unit() == pytest.approx(145.6, abs=0.05)
    assert unit_fault_bits(small) / point.q == pytest.approx(151.8, abs=0.05)
    assert point.post_j_union() == pytest.approx(9062, abs=1)
    assert unit_fault_bits(small) / (point.q * point.s) == pytest.approx(1214, abs=1)
    n = point.rate.replay_units
    assert default.rho * math.log2(1 + point.q * n / (1 - point.q)) == pytest.approx(
        3267, abs=1
    )


def test_simulation_at_full_verification_pays_exactly_the_unit_price(
    small: KindTable,
) -> None:
    full = VerificationPolicy(Fraction(1), Fraction(1))
    assert bound(small, full, ETA).bits == 0.0
    assert bound(small, full, ETA, max_faults=1).bits == pytest.approx(
        unit_fault_bits(small), abs=1e-9
    )
