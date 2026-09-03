"""H5: detection before the challenge (``docs/honest-prover.md``, section 7).

One catastrophic fault -- bit 12 of a step-1 embedding or projection dot of the
first request, a high-order flip that changes the streamed tokens -- handled
three ways: left to the post-J mechanism (H5a: no action is a rejected
transcript, a declaration after ``J`` costs ``u_post(1)``), caught before the
token was streamed and truncated (H5b, S7's constructor, the length as
advice), or found by re-executing the request before the round closes and
declared before ``J`` at ``u(1)`` (H5c, a priced extension: the protocol has no
pre-J declaration message).  The charges are priced on the toy shape's
1024-request serving table, whose fold is not capped by the interface as the
six-request run's is, and at the headline policy; every number section 7
quotes is pinned here.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from veritor.analysis.bound import bound
from veritor.analysis.faults import unit_fault_bits
from veritor.compile.constructor import Constructor
from veritor.constructors.requests import RequestsG
from veritor.constructors.schedule import Request
from veritor.constructors.truncation import TruncatedRequestsG, field_width
from veritor.core import Compiled, KindTable, VerificationPolicy
from veritor.evaluation.global_estimate import UNITS_PER_MAC, estimate
from veritor.evaluation.serving import ServingShape, serving_table
from veritor.protocol import (
    MerkleTree,
    ProtocolRun,
    VerificationCode,
    VerifierParameters,
    Weights,
    assignment_replay,
    commit_weights,
    honest_declare,
    make_expectation,
    run_protocol,
    self_check,
)
from veritor.simulation.detection import (
    DECLARE,
    FAULT_MIXTURE,
    LLAMA3_GPU_INTERRUPTIONS,
    SILENT_FAULTS,
    TRUNCATE,
    VISIBLE_FAULTS,
    DeclarationPrices,
    HardwareSignals,
    PartialReexecution,
    ValueCheck,
    analytic_prices,
    charge_deviation,
    coverage,
    detected,
    expected_charge,
    fold_prices,
    frontier,
    halving_fraction,
    headline_prices,
    rejection_probability,
    savings_per_fault,
    truncation_charged_bits,
    truncation_information_bits,
)
from veritor.simulation.faults import (
    LLAMA3_SDC_EVENTS,
    Fault,
    FaultInjector,
    expected_faults,
    is_dot_unit,
)
from veritor.stress.measure import ETA, POLICY, Measurement, compile_scenario, price
from veritor.stress.models import Model
from veritor.stress.rows import Recorder

FULL = VerificationPolicy(1, 1)
"""Every RU replayed, every VU sampled: the verdict is deterministic."""
MAX_NEW = 8
REQUESTS = 6
BIT = 12
"""The flipped bit: in the top half of a 16-bit word, the class a value check sees."""
TOY_REQUESTS = 1024
"""Requests on the serving table the charges are priced on (the run's own fold is capped)."""
MACS_PER_SECOND = 1e15
"""An H100-class device (``veritor.evaluation.global_estimate.Inputs.hash_macs``)."""
FAULT_DENSITIES = (1e6, 1e8, 1e9)
"""Faults per round beyond the Llama-3 rate, for the f_max headroom table."""
CHALLENGES = 8
"""Independent q-challenges (fixed seeds) the post-J row is run under."""


# -- the run ------------------------------------------------------------------------------


def _requests(model: Model, seed: int = 7) -> tuple[Request, ...]:
    """The six requests of S7: prompts of two to four tokens, ``max_new`` 8."""

    rng = random.Random(seed)
    vocab = model.shape.vocab
    return tuple(
        Request(tuple(rng.randrange(vocab) for _ in range(rng.randint(2, 4))), MAX_NEW)
        for _ in range(REQUESTS)
    )


@dataclass(frozen=True, slots=True)
class Run:
    """A compiled run with what a protocol run and a row need."""

    measurement: Measurement
    model: Model
    kappa: Weights
    weight_tree: MerkleTree
    values: dict[int, int]

    @property
    def compiled(self) -> Compiled:
        return self.measurement.compiled

    @property
    def outputs(self) -> tuple[int, ...]:
        return tuple(self.values[a] for a in self.compiled.circuit.outputs)

    @property
    def honest_cost(self) -> int:
        table = self.compiled.kind_table()
        return next(row.replay_cost for row in table.rows if row.kind == table.root)

    def opened_fraction(self, replay_units: Sequence[int]) -> float:
        """The replay cost of ``replay_units`` over the whole run's: what replaying them re-executes."""

        table = self.compiled.kind_table()
        costs = {row.kind: row.replay_cost for row in table.rows}
        index = self.compiled.index
        return (
            sum(costs[index.replay_units.unit(r).kind] for r in replay_units)
            / self.honest_cost
        )

    def run(
        self,
        values: Mapping[int, int],
        outputs: Sequence[int],
        *,
        policy: VerificationPolicy,
        max_faults: int,
        declare: bool,
        label: str,
    ) -> ProtocolRun:
        seed = hashlib.sha256(
            f"veritor/stress/honest/detection/{label}".encode()
        ).digest()
        expectation = make_expectation(
            self.measurement.compilation,
            policy,
            tuple(outputs),
            parameters=VerifierParameters(
                ETA,
                max_capacity=1 << 20,
                max_advice_bits=self.measurement.advice_bits,
                max_faults=max_faults,
            ),
            weights=self.kappa,
            session_id=seed[:16],
            q_seed=seed,
            s_seed=bytes(reversed(seed)),
        )
        return run_protocol(
            self.compiled,
            expectation,
            values,
            replay=assignment_replay(values),
            weight_tree=self.weight_tree,
            declare=honest_declare(self.compiled) if declare else None,
        )


def _serve(constructor: Constructor, x: object, a: bytes, model: Model) -> Run:
    measurement = compile_scenario(constructor, x, a, model.gate_set)
    circuit = measurement.compiled.circuit
    values = dict(
        enumerate(circuit.evaluate(measurement.compilation.inputs, model.weights))
    )
    kappa, tree = commit_weights(model.gate_set, model.weights)
    return Run(measurement, model, kappa, tree, values)


@dataclass(frozen=True, slots=True)
class Scenario:
    """The six-request run and the catastrophic fault in its first request."""

    x: tuple[Request, ...]
    run: Run
    fault: Fault
    candidates: int
    """Step-1 embedding and projection dots of the first request (the fault's peers)."""
    catastrophic: int
    """How many of them change a streamed token when bit 12 flips."""

    @property
    def replay_unit(self) -> int:
        return self.fault.replay_unit


def step1_dots(run: Run, x: tuple[Request, ...]) -> tuple[int, list[int]]:
    """The first request's RU and the dot VUs of its first decode step that embed the
    token and project it to ``q``, ``k`` and ``v``, in order."""

    compiled = run.compiled
    circuit, index = compiled.circuit, compiled.index
    layout = RequestsG(run.model.shape).output_layout(x)
    first, second = (circuit.outputs[layout.index((0, g))] for g in (0, 1))
    replay_unit = index.replay_units.owner(first)
    block = index.verification_units(replay_unit)
    shape = run.model.shape
    dots = [
        unit
        for unit in range(block.first, block.first + block.count)
        if is_dot_unit(compiled, node := index.verification_unit(unit))
        and first < node.interval[0] < second
    ][: shape.d_model + 3 * shape.d_model]
    return replay_unit, dots


@pytest.fixture(scope="module")
def scenario(model: Model) -> Scenario:
    x = _requests(model)
    run = _serve(RequestsG(model.shape), x, b"", model)
    replay_unit, dots = step1_dots(run, x)
    assert replay_unit == 1, "RU 0 is the weights; the first request is RU 1"
    injector = FaultInjector(
        run.compiled, run.measurement.compilation.inputs, model.weights
    )
    faults = [injector.inject(unit, BIT) for unit in dots]
    catastrophic = [fault for fault in faults if fault.changed_outputs]
    fault = catastrophic[0]
    assert fault.replay_unit == replay_unit
    return Scenario(x, run, fault, len(dots), len(catastrophic))


def toy_table(requests: int) -> KindTable:
    """The toy shape served to ``requests`` identical requests, request RUs and row VUs."""

    shape = ServingShape(
        vocab=8,
        d_model=4,
        heads=2,
        layers=1,
        prompt=3,
        generated=MAX_NEW,
        requests=requests,
        hidden_multiplier=2,
    )
    return serving_table(shape, "request", "row")


@pytest.fixture(scope="module")
def toy_prices() -> tuple[DeclarationPrices, DeclarationPrices]:
    """The simulation policy's prices on the toy shape at 6 and at 1024 requests."""

    return fold_prices(toy_table(REQUESTS), POLICY), fold_prices(
        toy_table(TOY_REQUESTS), POLICY
    )


@pytest.fixture(scope="module")
def headline() -> DeclarationPrices:
    return headline_prices()


# -- the detector menu ------------------------------------------------------------------


def test_the_fault_mixture_and_the_detectors_coverage() -> None:
    assert LLAMA3_GPU_INTERRUPTIONS == 268 and LLAMA3_SDC_EVENTS == 6
    assert (
        VISIBLE_FAULTS.share == pytest.approx(262 / 268) and not VISIBLE_FAULTS.silent
    )
    assert SILENT_FAULTS.share == pytest.approx(6 / 268) and SILENT_FAULTS.silent
    assert sum(fault.share for fault in FAULT_MIXTURE) == pytest.approx(1.0)
    hardware, check = HardwareSignals(), ValueCheck(7e-6)
    assert hardware.cost == 0 and hardware.outcome == TRUNCATE
    assert (
        hardware.coverage(VISIBLE_FAULTS) == 1 and hardware.coverage(SILENT_FAULTS) == 0
    )
    assert check.coverage(VISIBLE_FAULTS) == 0 and check.coverage(SILENT_FAULTS) == 0.5
    assert check.outcome == TRUNCATE
    assert coverage([hardware]) == pytest.approx(0.9776, abs=5e-4)
    assert coverage([hardware], silent=True) == 0
    assert coverage([check]) == pytest.approx(0.0112, abs=5e-4)
    assert coverage([check], silent=True) == 0.5
    assert coverage([hardware, check]) == pytest.approx(0.9888, abs=5e-4)
    assert coverage([hardware, check], silent=True) == 0.5
    for fraction in (0.1, 0.5, 0.9):
        rerun = PartialReexecution(fraction)
        assert rerun.cost == fraction and rerun.outcome == DECLARE
        assert (
            rerun.coverage(SILENT_FAULTS) == rerun.coverage(VISIBLE_FAULTS) == fraction
        )
        assert coverage([hardware, check, rerun], silent=True) == pytest.approx(
            0.5 + 0.5 * fraction
        )
        assert detected([check, rerun], SILENT_FAULTS) == pytest.approx(
            1 - 0.5 * (1 - fraction)
        )
    assert coverage([], silent=True) == 0 and coverage([hardware], []) == 0
    # bit 12 of a 16-bit word is in the top half: the flip the rows inject is one the check sees
    assert ValueCheck.sees(BIT, 16) and not ValueCheck.sees(7, 16)
    with pytest.raises(ValueError):
        ValueCheck.sees(16, 16)
    with pytest.raises(ValueError):
        ValueCheck(1.5)
    with pytest.raises(ValueError):
        PartialReexecution(-0.1)
    with pytest.raises(ValueError):
        ValueCheck.cost_fraction(1, 1, 0)


def test_the_value_check_costs_millionths_of_the_serving_compute() -> None:
    best = estimate()
    shape = best.inputs.shape
    words = (
        shape.vocab + shape.layers * shape.d_model
    )  # the logits and every layer's residual
    assert words == 688_128
    cost = ValueCheck.cost_fraction(
        words, best.inputs.tokens_per_request, best.ru_cost, compare_cost=2.0
    )
    assert cost == pytest.approx(7.01e-6, rel=0.01)
    assert ValueCheck.cost_fraction(0, 10, 100.0) == 0


# -- the two prices and the conservation law -------------------------------------------


def test_conservation_law_at_the_headline(headline: DeclarationPrices) -> None:
    q, s = float(headline.policy.q), float(headline.policy.s)
    assert q == pytest.approx(1.57e-8, rel=0.01) and s == pytest.approx(
        8.91e-3, rel=0.01
    )
    assert headline.method == "rate" and not headline.capped
    assert headline.rho == pytest.approx(4.74e11, rel=0.01)
    assert headline.pre == pytest.approx(94.7, abs=0.1)  # u(1) = W_V + log2 |S|
    assert headline.post == pytest.approx(
        6.12e9, rel=0.01
    )  # u_post(1) = rho log2 (1 / (1 - s))
    assert headline.post == pytest.approx(
        headline.rho * math.log2(1 / (1 - s)), rel=1e-9
    )
    assert headline.leverage == pytest.approx(6.46e7, rel=0.01)
    assert headline.conservation == pytest.approx(
        1.015, abs=0.002
    )  # q u_post(1) / u(1)
    assert q * headline.post == pytest.approx(96.1, abs=0.1)
    # why it holds: the scattered channel sets rho (rate.py, (2) at l = 1), as
    # (W_V + log2 R + log2 m + 1) / log2 (1 / (1 - q s)) with W_V + log2 R + log2 m = u(1), so
    # q u_post(1) = (u(1) + 1) q log2 (1 / (1 - s)) / log2 (1 / (1 - q s)) ~ (u(1) + 1) (1 + s / 2)
    assert headline.rho == pytest.approx(
        (headline.pre + 1) / math.log2(1 / (1 - q * s)), rel=1e-4
    )
    assert headline.conservation == pytest.approx(
        (headline.pre + 1) / headline.pre * q * math.log(1 - s) / math.log(1 - q * s),
        rel=1e-4,
    )
    assert headline.conservation == pytest.approx(
        (1 + 1 / headline.pre) * (1 + s / 2), rel=1e-4
    )
    # detection moves the expected charge by a bit and a half per fault, never halves it
    assert expected_charge(0.0, headline) == pytest.approx(96.1, abs=0.1)
    assert expected_charge(1.0, headline) == headline.pre
    assert savings_per_fault(1.0, headline) == pytest.approx(1.43, abs=0.01)
    assert savings_per_fault(0.5, headline) == pytest.approx(0.72, abs=0.01)
    assert halving_fraction(headline) is None
    # what it does move: the per-fault charge is 0 or 6e9 bits undetected, 94.7 detected
    assert charge_deviation(0.0, headline) == pytest.approx(7.67e5, rel=0.01)
    assert charge_deviation(0.0, headline) == pytest.approx(
        headline.post * math.sqrt(q * (1 - q)), rel=1e-9
    )
    assert charge_deviation(0.5, headline) == pytest.approx(5.42e5, rel=0.01)
    assert charge_deviation(1.0, headline) == 0
    points = frontier(headline)
    assert [point.fraction for point in points] == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    assert [point.compute for point in points] == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    charges = [point.charge_per_fault for point in points]
    assert charges == sorted(charges, reverse=True)
    assert charges[0] - charges[-1] == pytest.approx(1.43, abs=0.01)
    assert points[-1].savings_per_fault == pytest.approx(1.43, abs=0.01)
    # the truncation's price at the headline: a length for every request, or the sparse floor
    best = estimate()
    requests, generated = best.inputs.requests, best.inputs.shape.generated
    assert truncation_charged_bits(requests, generated) == 9 * requests
    assert truncation_charged_bits(
        requests, generated
    ) / best.capacity_bits == pytest.approx(13.9, abs=0.1)
    assert truncation_information_bits(requests, 1, generated) == pytest.approx(
        53.7, abs=0.1
    )
    assert expected_charge(1.0, headline, pre=53.7) == pytest.approx(53.7)
    # no action: the fault is a rejection iff its RU is opened and its VU sampled
    assert q * s == pytest.approx(1.40e-10, rel=0.01)
    with pytest.raises(ValueError):
        expected_charge(1.5, headline)
    with pytest.raises(ValueError):
        savings_per_fault(-0.1, headline)


def test_prices_on_the_toy_table_at_the_simulation_policy(
    toy_prices: tuple[DeclarationPrices, DeclarationPrices],
) -> None:
    small, large = toy_prices
    # six requests: the fold is saturated, the interface caps U, the marginal is invisible
    assert small.capped and small.post == 0
    assert small.pre == pytest.approx(75.9, abs=0.1)
    assert small.base_bits == pytest.approx(800.8, abs=0.1)
    # 1024 requests: the same kinds, an uncapped fold
    assert large.method == "fold" and not large.capped
    assert large.pre == pytest.approx(83.2, abs=0.1)
    assert large.post == pytest.approx(141.2, abs=0.1)
    assert large.rho == pytest.approx(830, abs=1)
    assert large.base_bits == pytest.approx(30_012, abs=1)
    assert large.leverage == pytest.approx(1.70, abs=0.01)
    assert large.conservation == pytest.approx(
        0.848, abs=0.002
    )  # below 1: detection loses bits
    assert savings_per_fault(1.0, large) == pytest.approx(-12.6, abs=0.1)
    assert expected_charge(0.0, large) == pytest.approx(70.6, abs=0.1)
    assert halving_fraction(large) is None and halving_fraction(small) is None
    assert charge_deviation(0.0, large) == pytest.approx(large.post / 2, rel=1e-9)
    # the closed form is above the fold here (the scattered channel binds; rate.py, (i)):
    # its rho is the scattered channel's, and the conservation law holds for it within 5%
    closed = analytic_prices(toy_table(TOY_REQUESTS), POLICY)
    assert closed.pre == large.pre
    assert closed.post == pytest.approx(174.2, abs=0.1)
    assert closed.rho == pytest.approx((closed.pre + 1) / math.log2(16 / 15), rel=1e-4)
    assert closed.conservation == pytest.approx(1.047, abs=0.001)
    assert large.post / closed.post == pytest.approx(0.81, abs=0.01)
    assert 1.15 < closed.post / large.post < 1.30
    # the truncation's price on the same table
    assert truncation_charged_bits(TOY_REQUESTS, MAX_NEW) == 3072
    assert truncation_information_bits(TOY_REQUESTS, 1, MAX_NEW) == 13.0
    assert truncation_charged_bits(REQUESTS, MAX_NEW) == 18
    assert truncation_information_bits(REQUESTS, 1, MAX_NEW) == pytest.approx(
        5.58, abs=0.01
    )
    assert truncation_information_bits(REQUESTS, 0, MAX_NEW) == 0
    with pytest.raises(ValueError):
        truncation_information_bits(6, 7, MAX_NEW)
    with pytest.raises(ValueError):
        truncation_charged_bits(-1, MAX_NEW)


# -- f_max headroom ---------------------------------------------------------------------


def headline_fault_density() -> float:
    """Silent faults in the headline's year at the Llama-3 SDC rate: the year's requests
    at their cost in MACs, on ``MACS_PER_SECOND`` devices, are device-hours."""

    best = estimate()
    device_hours = (
        best.inputs.requests * best.ru_cost / UNITS_PER_MAC / MACS_PER_SECOND / 3600
    )
    assert device_hours / 8760 == pytest.approx(62_000, rel=0.01)  # GPU-years
    return expected_faults(device_hours)


def test_detection_buys_headroom_under_the_round_cap(
    headline: DeclarationPrices,
) -> None:
    policy = headline.policy
    q = float(policy.q)
    density = headline_fault_density()
    assert density == pytest.approx(154, abs=1)
    assert q * density == pytest.approx(2.42e-6, rel=0.01)
    # f_max = 0: the round is rejected iff any undetected fault is opened, 1 - exp(-q mu (1 - p))
    for faults in (density, *FAULT_DENSITIES):
        for p in (0.0, 0.5, 0.9, 0.99):
            expected = -math.expm1(-q * faults * (1 - p))
            assert rejection_probability(policy, faults, 0, p) == pytest.approx(
                expected, rel=1e-9
            )
    assert rejection_probability(policy, density, 0) == pytest.approx(2.42e-6, rel=0.01)
    assert rejection_probability(policy, 1e8, 0) == pytest.approx(0.792, abs=0.001)
    assert rejection_probability(policy, 1e8, 0, 0.9) == pytest.approx(0.145, abs=0.001)
    # the round's charge at the Llama-3 rate: 6.1e9 bits (0.03% of U_0) with probability
    # 2.4e-6, else nothing; detected, 154 u(1) = 1.46e4 bits with certainty; no action is a
    # rejected round iff a faulty VU is sampled, 154 q s = 2.2e-8 per round
    assert headline.post / headline.base_bits == pytest.approx(3.23e-4, rel=0.01)
    assert density * headline.pre == pytest.approx(1.46e4, rel=0.01)
    assert density * q * headline.post == pytest.approx(1.48e4, rel=0.01)
    assert -math.expm1(-density * q * float(policy.s)) == pytest.approx(
        2.16e-8, rel=0.01
    )
    assert rejection_probability(policy, density, 1) == pytest.approx(
        2.93e-12, rel=0.01
    )
    # f_max = 4, the day's budget of M6: detection matters once q mu is of the order of f_max
    table = {
        faults: [
            rejection_probability(policy, faults, 4, p) for p in (0.0, 0.5, 0.9, 0.99)
        ]
        for faults in (density, *FAULT_DENSITIES)
    }
    assert table[density] == pytest.approx(
        [6.92e-31, 2.16e-32, 6.92e-36, 6.92e-41], rel=0.02
    )
    assert table[1e6] == pytest.approx(
        [7.86e-12, 2.47e-13, 7.96e-17, 7.97e-22], rel=0.02
    )
    assert table[1e8] == pytest.approx([0.0221, 0.0013, 6.99e-7, 7.86e-12], rel=0.02)
    assert table[1e9] == pytest.approx([0.999, 0.892, 0.0221, 6.99e-7], rel=0.02)
    assert rejection_probability(policy, 1e6, 0) == pytest.approx(0.0156, rel=0.02)
    assert 1e8 / density == pytest.approx(6.5e5, rel=0.01)  # times the Llama-3 rate
    for row in table.values():
        assert row == sorted(row, reverse=True)
    assert rejection_probability(policy, 1e9, 4, 0.99) == pytest.approx(
        rejection_probability(policy, 1e8, 4, 0.9), rel=1e-9
    )  # the cap sees only q mu (1 - p)
    # the detection that holds a round's rejection below 1e-6 under f_max = 4 is
    # q mu (1 - p) <= 0.169: p = 0.892 at 1e8 faults per round, 0.989 at 1e9
    for faults, needed in ((1e8, 0.892), (1e9, 0.989)):
        assert rejection_probability(policy, faults, 4, needed) == pytest.approx(
            1e-6, rel=0.12
        )
        assert needed == pytest.approx(1 - 0.169 / (q * faults), abs=0.001)
    # a mercurial core per thousand devices (Meta 2025; datacenter-realities.md, section 7)
    # corrupts every request it serves: 2.9e10 faulty requests per round, 460 of them opened,
    # and no cap holds without p > 0.9996 -- a device to remove, not a fault to declare
    mercurial = estimate().inputs.requests / 1000
    assert q * mercurial == pytest.approx(460, abs=1)
    assert rejection_probability(policy, mercurial, 4) == 1.0
    assert rejection_probability(policy, mercurial, 4, 0.99) == pytest.approx(
        0.487, abs=0.001
    )
    assert rejection_probability(policy, mercurial, 4, 0.9996) == pytest.approx(
        1.5e-6, rel=0.02
    )
    # the simulation policy, q = 1/2: a handful of faults per round already needs detection
    assert rejection_probability(POLICY, 3, 1) == pytest.approx(0.442, abs=0.001)
    assert rejection_probability(POLICY, 3, 1, 0.9) == pytest.approx(0.0102, abs=0.0001)
    assert rejection_probability(POLICY, 1, 1) == pytest.approx(0.0902, abs=0.0005)
    assert rejection_probability(POLICY, 1, 1, 0.9) == pytest.approx(
        0.00121, abs=0.00002
    )
    # a direct sum agrees where 1 - cdf still has digits, and the edges behave
    mean = 0.5 * 10
    direct = 1 - sum(math.exp(-mean) * mean**k / math.factorial(k) for k in range(3))
    assert rejection_probability(POLICY, 10, 2) == pytest.approx(direct, rel=1e-9)
    assert rejection_probability(POLICY, 0, 0) == 0
    assert rejection_probability(POLICY, 1e6, 3) == 1.0
    assert rejection_probability(policy, 1e9, 4, 1.0) == 0
    with pytest.raises(ValueError):
        rejection_probability(POLICY, -1, 0)
    with pytest.raises(ValueError):
        rejection_probability(POLICY, 1, -1)
    with pytest.raises(ValueError):
        rejection_probability(POLICY, 1, 0, 2.0)


# -- H5a: no action, or a declaration after J -----------------------------------------


def test_h5a_post_j_declaration_or_rejected_transcript(
    honest: Recorder,
    scenario: Scenario,
    toy_prices: tuple[DeclarationPrices, DeclarationPrices],
    headline: DeclarationPrices,
) -> None:
    run, fault = scenario.run, scenario.fault
    compiled = run.compiled
    assert scenario.candidates == 16 and scenario.catastrophic == 14
    assert fault.changed_outputs >= 1 and fault.bit == BIT
    assert ValueCheck.sees(fault.bit, compiled.circuit[fault.address].width)
    # the server holds the faulty assignment; replaying its RU finds exactly the flipped VU
    assert self_check(compiled, fault.replay_unit, fault.values) == (
        fault.verification_unit,
    )
    # no action: every VU sampled, the faulty relation fails
    rejected = run.run(
        fault.values,
        fault.outputs,
        policy=FULL,
        max_faults=0,
        declare=False,
        label="h5a/none",
    )
    assert rejected.report.code is VerificationCode.RELATION_REJECTED
    assert fault.verification_unit in rejected.report.sampled_verification_units
    # post-J, over CHALLENGES fixed seeds: when the q-challenge opens the RU the server
    # self-checks it and declares the one VU; when it does not, nothing is seen or charged
    runs = [
        run.run(
            fault.values,
            fault.outputs,
            policy=POLICY,
            max_faults=1,
            declare=True,
            label=f"h5a/post/{attempt}",
        )
        for attempt in range(CHALLENGES)
    ]
    declared = [r for r in runs if fault.replay_unit in r.report.sampled_replay_units]
    unseen = [r for r in runs if fault.replay_unit not in r.report.sampled_replay_units]
    assert len(declared) == 2 and len(unseen) == 6  # q = 1/2, the seeds fixed
    for r in runs:
        assert r.report.accepted, r.report
        assert r.transcript is not None
    assert all(
        r.transcript is not None
        and r.transcript.interiors.declarations == (fault.verification_unit,)
        for r in declared
    )
    assert all(
        r.transcript is not None and r.transcript.interiors.declarations == ()
        for r in unseen
    )
    opened = declared[0].report.sampled_replay_units
    recompute = run.opened_fraction(opened)
    small, large = toy_prices
    priced = price(compiled, POLICY)
    capped = bound(compiled, POLICY, ETA, max_faults=1)
    assert capped.capped and capped.bits == priced.bound.bits
    own = unit_fault_bits(compiled)
    assert own == pytest.approx(small.pre, abs=0.01)
    units = sum(
        row.copies for row in compiled.kind_table().rows if row.role == "verification"
    )
    honest.record(
        id="H5a",
        what=(
            f"catastrophic silent fault: bit {fault.bit} of a step-1 projection dot's output word "
            f"(VU {fault.verification_unit}, RU {fault.replay_unit}, the first of {REQUESTS} requests "
            f"of max_new {MAX_NEW}) flips, {fault.changed_outputs} of the request's "
            f"{MAX_NEW - 1} downstream tokens change and are streamed; nothing is detected before J"
        ),
        mechanism="M6 post-J declaration (as built), or no action",
        advice_bits=run.measurement.advice_bits,
        capacity_bits=math.ceil(capped.bits) + run.measurement.advice_bits,
        overhead=priced.overhead,
        description_bytes=run.measurement.description_bytes,
        verdict=(
            f"no action, theta = (1, 1): RELATION_REJECTED at VU {fault.verification_unit}; "
            f"post-J, theta = (1/2, 1/8), f_max = 1, {CHALLENGES} challenges: {len(declared)} open "
            f"RU {fault.replay_unit} (the first opens {len(opened)} of "
            f"{compiled.index.replay_units.count} RUs), self_check finds the one VU, declared, "
            f"ACCEPTED; {len(unseen)} do not open it, nothing is declared or charged, ACCEPTED"
        ),
        notes=(
            f"Charged only when the RU is opened (probability q). u_post(1) on the toy shape's "
            f"{TOY_REQUESTS}-request serving table at theta = (1/2, 1/8), uncapped fold: "
            f"{large.post:.1f} bits (u(1) = {large.pre:.1f}, leverage {large.leverage:.2f}, "
            f"q u_post(1) / u(1) = {large.conservation:.3f}); this run's own fold is saturated "
            f"(|S| = {units}, u(1) = {own:.1f}, U capped at |Out| = {capped.out_bits} bits with or "
            f"without the declaration, uncapped {capped.knapsack_bits:.0f}), so its marginal is "
            f"invisible. Headline (q = {float(headline.policy.q):.2e}, s = {float(headline.policy.s):.2e}): "
            f"u_post(1) = {headline.post:.2e} bits per declaration, expected q u_post(1) = "
            f"{expected_charge(0.0, headline):.1f} = {headline.conservation:.3f} u(1): the conservation law. "
            f"{scenario.catastrophic} of the {scenario.candidates} step-1 embedding and projection dots "
            f"change a token when bit {BIT} flips; a tokens-only recording policy would have to pin "
            f"up to {MAX_NEW - 1} downstream tokens instead of the one VU (section 9)."
        ),
        declarations=1,
        charge_bits=round(large.post),
        recompute=recompute,
    )


# -- H5b: caught before the token was streamed: truncate -------------------------------


def test_h5b_truncation_after_a_pre_streaming_detection(
    honest: Recorder,
    scenario: Scenario,
    toy_prices: tuple[DeclarationPrices, DeclarationPrices],
    headline: DeclarationPrices,
) -> None:
    model, x, fault = scenario.run.model, scenario.x, scenario.fault
    # the fault sits in the step that would produce the second token: a hardware signal or a
    # value check on that step's activations fires before it is streamed, and the request ends
    # with the one token the prefill produced
    constructor = TruncatedRequestsG(model.shape)
    lengths = (1, *(MAX_NEW,) * (REQUESTS - 1))
    advice = constructor.advice(x, lengths)
    truncated = _serve(constructor, x, advice, model)
    assert truncated.measurement.advice_bits == REQUESTS * field_width(MAX_NEW) == 18
    assert truncated.measurement.advice_bits == truncation_charged_bits(
        REQUESTS, MAX_NEW
    )
    # the truncated circuit is fault-free: the faulty dot is not in it; request 0's outputs are
    # the one streamed token then blank check outputs, and every other request's are unchanged
    # (the circuit order differs: the truncated request is a kind of its own)
    blank = constructor.blank
    honest_tokens = dict(
        zip(RequestsG(model.shape).output_layout(x), scenario.run.outputs, strict=True)
    )
    faulty_tokens = dict(
        zip(RequestsG(model.shape).output_layout(x), fault.outputs, strict=True)
    )
    truncated_tokens = dict(
        zip(constructor.output_layout(x, advice), truncated.outputs, strict=True)
    )
    assert truncated_tokens[0, 0] == faulty_tokens[0, 0] == honest_tokens[0, 0]
    assert all(truncated_tokens[0, g] == blank for g in range(1, MAX_NEW))
    assert sum(faulty_tokens[0, g] != honest_tokens[0, g] for g in range(MAX_NEW)) == (
        fault.changed_outputs
    )
    for r in range(1, REQUESTS):
        for g in range(MAX_NEW):
            assert truncated_tokens[r, g] == faulty_tokens[r, g] == honest_tokens[r, g]
    assert len(constructor.blank_positions(x, advice)) == MAX_NEW - 1
    accepted = truncated.run(
        truncated.values,
        truncated.outputs,
        policy=POLICY,
        max_faults=0,
        declare=True,
        label="h5b",
    )
    assert accepted.report.accepted, accepted.report
    assert accepted.transcript is not None
    assert accepted.transcript.interiors.declarations == ()
    priced = price(truncated.compiled, POLICY)
    assert priced.bound.capped and priced.bound.out_bits == 16 * (
        REQUESTS * MAX_NEW - (MAX_NEW - 1)
    )
    _small, large = toy_prices
    hardware, check = HardwareSignals(), ValueCheck(7.01e-6)
    honest.record(
        id="H5b",
        what=(
            f"the same fault caught before the token was streamed (a hardware signal, or a value check "
            f"on step 1's activations: bit {fault.bit} of 16 is in the top half); the request ends "
            f"after its first token, the {MAX_NEW - 1} absent slots are blank check outputs"
        ),
        mechanism="pre-J truncation: S7 (TruncatedRequestsG), the generated length as advice",
        advice_bits=truncated.measurement.advice_bits,
        capacity_bits=priced.capacity_bits + truncated.measurement.advice_bits,
        overhead=priced.overhead,
        description_bytes=truncated.measurement.description_bytes,
        verdict=(
            f"ACCEPTED with no declaration at theta = (1/2, 1/8); the faulty step is not in the "
            f"circuit; U = {priced.capacity_bits} = the {REQUESTS * MAX_NEW - (MAX_NEW - 1)}-token "
            f"run's, plus {truncated.measurement.advice_bits} advice bits"
        ),
        notes=(
            f"The detector menu: hardware signals see {coverage([hardware]):.1%} of the Llama-3 GPU "
            f"fault mixture ({LLAMA3_GPU_INTERRUPTIONS - LLAMA3_SDC_EVENTS} of {LLAMA3_GPU_INTERRUPTIONS} "
            f"events) at no cost and none of the silent {SILENT_FAULTS.share:.1%}; a pre-streaming "
            f"value check sees the half of the silent faults that blow a value up (an assumption, "
            f"section 7) at {check.cost:.1e} of the serving compute on the 70B shape, "
            f"{coverage([hardware, check]):.1%} of all faults together. The truncation's price as "
            f"built is a length for every request: {truncation_charged_bits(REQUESTS, MAX_NEW)} bits "
            f"here, {truncation_charged_bits(TOY_REQUESTS, MAX_NEW)} on the {TOY_REQUESTS}-request "
            f"table against u_post(1) = {large.post:.1f}, {truncation_charged_bits(estimate().inputs.requests, 512):.1e} "
            f"at the headline (13.9 U_0); naming only the truncated request would cost "
            f"{truncation_information_bits(TOY_REQUESTS, 1, MAX_NEW):.1f} and "
            f"{truncation_information_bits(estimate().inputs.requests, 1, 512):.1f} bits "
            f"(log2 C(requests, 1) + log2 max_new), below u(1) = {headline.pre:.1f}. The toy ISA has "
            f"no NaN: the check's coverage is asserted by the bit position, not evaluated."
        ),
        declarations=0,
        charge_bits=truncated.measurement.advice_bits,
        recompute=0.0,
    )


# -- H5c: found by re-execution before the round closes, declared before J -------------


def test_h5c_pre_j_declaration_after_partial_reexecution(
    honest: Recorder,
    scenario: Scenario,
    toy_prices: tuple[DeclarationPrices, DeclarationPrices],
    headline: DeclarationPrices,
) -> None:
    run, fault = scenario.run, scenario.fault
    compiled = run.compiled
    # re-executing the request and comparing bit for bit: the first disagreeing gate is the
    # flipped output word, its VU the declaration; everything after it is the fault's cone
    disagreeing = sorted(a for a, v in fault.values.items() if v != run.values[a])
    assert disagreeing[0] == fault.address
    index = compiled.index
    block = index.verification_units(fault.replay_unit)
    assert block.first + block.owner(fault.address) == fault.verification_unit
    assert all(index.replay_units.owner(a) == fault.replay_unit for a in disagreeing)
    # the pre-J price is the fixed-in-advance one: exactly what the fold charges at q = 1
    _small, large = toy_prices
    own = unit_fault_bits(compiled)
    full_0 = bound(compiled, FULL, ETA).bits
    full_1 = bound(compiled, FULL, ETA, max_faults=1).bits
    assert full_1 - full_0 == pytest.approx(own) and own == pytest.approx(75.9, abs=0.1)
    priced = price(compiled, POLICY)
    capped = bound(compiled, POLICY, ETA, max_faults=1)
    rerun = PartialReexecution(1.0)
    honest.record(
        id="H5c",
        what=(
            f"the same fault found before J by re-executing the request on idle capacity before the "
            f"round closes (the first disagreeing gate is the flipped word, VU {fault.verification_unit}), "
            f"and declared before the q-challenge"
        ),
        mechanism="pre-J declaration at u(1): a priced extension, not a protocol message",
        advice_bits=run.measurement.advice_bits,
        capacity_bits=math.ceil(capped.bits) + run.measurement.advice_bits,
        overhead=priced.overhead,
        description_bytes=run.measurement.description_bytes,
        verdict=(
            f"priced, not run: the protocol takes declarations only after J (H5a is the run); a "
            f"declaration fixed before the challenges costs u(1) = {large.pre:.1f} bits on the "
            f"{TOY_REQUESTS}-request table ({own:.1f} on this run: Bound at theta = (1, 1) goes "
            f"{full_0:.0f} -> {full_1:.1f})"
        ),
        notes=(
            f"Re-executing a fraction p of the requests finds a fraction p of the faults at a cost of "
            f"p of the serving compute (here p = {rerun.fraction:.0%}); what it finds has been streamed, "
            f"so the outcome is a declaration, not a truncation. Against leaving the fault to M6 it "
            f"saves q u_post(1) - u(1) = {savings_per_fault(1.0, large):.1f} bits per fault at "
            f"theta = (1/2, 1/8) on the {TOY_REQUESTS}-request table and "
            f"{savings_per_fault(1.0, headline):.1f} at the headline: the conservation law "
            f"q u_post(1) = {large.conservation:.3f} u(1) and {headline.conservation:.3f} u(1). No p "
            f"halves the expected charge; p = 1 takes its standard deviation from "
            f"{charge_deviation(0.0, headline):.1e} bits per fault to 0 at the headline and lowers "
            f"the opened-fault count against f_max (section 7)."
        ),
        declarations=1,
        charge_bits=round(large.pre),
        recompute=rerun.cost,
    )
