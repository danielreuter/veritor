"""Detection before the challenge: what an honest prover can know about its faults
before ``J`` is revealed, what knowing it costs, and what it saves
(``docs/honest-prover.md``, section 7).

M6 lets the prover declare a faulty VU *after* the q-challenge, at the
adaptive price ``bound(..., max_faults=f)`` charges: ``u_post(1)``, about
``u(1) / q`` bits per declaration, where ``u(1) = W_V + log2 |S|`` is the
price of a declaration fixed before the challenges
(:mod:`veritor.analysis.faults`).  A prover that knew a fault before ``J``
could declare it at ``u(1)`` -- the protocol has no such message today;
this module prices the extension -- or, if it knew before the garbage was
streamed, end the request there and pay the truncation's advice bits (S7,
:class:`~veritor.constructors.truncation.TruncatedRequestsG`).  Three ways to
know, each a *detector*: the fraction of each fault class it sees and its
cost as a fraction of the serving computation.

* :class:`HardwareSignals`: ECC counters, Xid errors, watchdogs, crashes.
  Free, and they see the whole hardware-visible class -- the fault stops the
  request -- and nothing of the silent one.
* :class:`ValueCheck`: NaN/Inf, range and degenerate-output checks on the
  activations or logits before a token is streamed.  An elementwise pass, so
  ``words checked / compute per token`` of the computation; sees the silent
  faults that move a value by orders of magnitude (a flip in the sign or
  exponent bits of a floating-point word) and none of the small ones.
* :class:`PartialReexecution`: re-run a fraction ``p`` of the requests on
  idle capacity before the round closes and compare bit for bit.  Sees ``p``
  of every class at a cost of ``p``; what it finds has been streamed, so the
  outcome is a declaration.

The fault mixture is the GPU part of Llama-3 405B pre-training's unexpected
interruptions (Dubey et al. 2024, table 5): 268 events, 6 of them silent
data corruption.  The conservation law the programme tests (section 2 of
the document) reads here ``q u_post(1) ~ u(1)``: a fault left to the post-J
mechanism is charged only when its RU is opened, so its expected charge is
``q u_post(1)``, the same bits a pre-J declaration costs with certainty.  It
is the scattered channel's identity, not a coincidence of the operating
point: where that channel sets ``rho`` (:mod:`veritor.analysis.rate`, (2) at
``l = 1``), ``rho = (u(1) + 1) / log2 (1 / (1 - q s))``, so ``q u_post(1) = q
rho log2 (1 / (1 - s)) = (u(1) + 1) (1 + s / 2 + ...)``; the fold sits below
the closed form by (i) of that module, so the fold's ratio is below one.
:func:`expected_charge` is the charge per fault when a fraction ``p`` is
detected before ``J``, :func:`savings_per_fault` its slope in ``p``,
:func:`frontier` the two against the compute the re-execution burns, and
:func:`rejection_probability` what detection does buy: fewer opened faults
against the round's cap ``f_max``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from veritor.analysis.bound import BoundOptions, BoundResult, bound
from veritor.analysis.faults import unit_fault_bits
from veritor.analysis.rate import rate
from veritor.core import Compiled, KindTable, VerificationPolicy, as_kind_table
from veritor.core.description import VERIFICATION
from veritor.core.policy import ProbabilityInput, exact_fraction

from .faults import LLAMA3_SDC_EVENTS

ETA = Fraction(1, 2**40)
"""The catalogue's threshold, ``lambda = 40``."""

# -- fault classes ----------------------------------------------------------------------

LLAMA3_GPU_INTERRUPTIONS = 148 + 72 + 19 + 17 + 6 + LLAMA3_SDC_EVENTS
"""Llama-3 405B pre-training, 54 days on 16,384 H100s: the GPU rows of table 5's
419 unexpected interruptions -- 148 faulty GPU, 72 HBM3 memory, 19 SRAM memory,
17 GPU system processor, 6 thermal interface or sensor, 6 silent data
corruption: 268 (Dubey et al. 2024; ``docs/notes/datacenter-realities.md``
section 7)."""


@dataclass(frozen=True, slots=True)
class FaultClass:
    """A class of hardware fault and how it presents to the server that suffered it.

    ``share`` is its fraction of the fault events a fleet sees (the Llama-3
    GPU mixture by default); ``silent`` says whether the fault reaches the
    streamed computation without any hardware signal; ``blow_up`` is the
    fraction of the class's faults that move a value far enough for a
    pre-streaming value check to see (``0`` for a class that is never
    silently streamed).
    """

    name: str
    share: float
    silent: bool
    blow_up: float
    source: str

    def __post_init__(self) -> None:
        if not 0 <= self.share <= 1 or not 0 <= self.blow_up <= 1:
            raise ValueError("share and blow_up are fractions in [0, 1]")


VISIBLE_FAULTS = FaultClass(
    name="hardware-visible: ECC events, Xid errors, device or NVLink failure, crash",
    share=(LLAMA3_GPU_INTERRUPTIONS - LLAMA3_SDC_EVENTS) / LLAMA3_GPU_INTERRUPTIONS,
    silent=False,
    blow_up=0.0,
    source=(
        "Dubey et al. 2024 (Llama 3), table 5: 148 faulty GPU, 72 HBM3, 19 SRAM, 17 system "
        "processor, 6 thermal of 268 GPU-category interruptions; NVIDIA Xid documentation: "
        "ECC corrects single-bit HBM errors, a double-bit error (Xid 48) or a row remap "
        "(Xid 63/64) kills the context rather than continue silently"
    ),
)
"""Faults the server sees as they happen: nothing wrong is streamed, the request is aborted."""

SILENT_FAULTS = FaultClass(
    name="silent compute-path corruption (datapath, unprotected latches, mercurial cores)",
    share=LLAMA3_SDC_EVENTS / LLAMA3_GPU_INTERRUPTIONS,
    silent=True,
    blow_up=0.5,
    source=(
        "Dubey et al. 2024: 6 SDC events, 2.8e-7 per device-hour; Dixit et al. 2021 and "
        "Hochschild et al. 2021: wrong results with no ECC or parity signal. blow_up 0.5 "
        "is an assumption, the top half of the word: Li et al. 2017 (SC'17) find that in "
        "floating-point DNNs only flips of the high-order exponent bits cause SDCs, "
        "mantissa and sign flips being benign, and that a range check on the activations "
        "detects most of the SDC-causing ones; Chen et al. 2021 (Ranger, DSN'21) cut SDC "
        "rates 3x to 50x by range restriction at negligible overhead"
    ),
)
"""Faults that reach the streamed tokens with no signal: the class M6 exists for."""

FAULT_MIXTURE: tuple[FaultClass, ...] = (VISIBLE_FAULTS, SILENT_FAULTS)
"""The Llama-3 GPU fault mixture: 97.8% visible, 2.2% silent."""

# -- detectors --------------------------------------------------------------------------

TRUNCATE = "truncate"
"""The detector fires before the garbage is streamed: the request ends there (S7)."""
DECLARE = "declare"
"""The detector fires after the tokens are out but before ``J``: a declaration at ``u(1)``."""


@dataclass(frozen=True, slots=True)
class HardwareSignals:
    """ECC counters, Xid errors, watchdogs, crashes: free, and blind to the silent class."""

    name: str = "hardware signals"
    outcome: str = TRUNCATE

    @property
    def cost(self) -> float:
        return 0.0

    def coverage(self, fault_class: FaultClass) -> float:
        return 0.0 if fault_class.silent else 1.0


@dataclass(frozen=True, slots=True)
class ValueCheck:
    """A pre-streaming pass over the logits or activations: NaN/Inf, range, degenerate output.

    ``cost`` is the pass as a fraction of the serving computation
    (:meth:`cost_fraction`).  It sees a fault iff the fault blew a value up
    -- a flip in the high-order bits of a word (:meth:`sees`) -- and it sees
    it *before* the token is streamed, so what it catches is truncated, not
    declared.
    """

    cost: float
    name: str = "pre-streaming value check"
    outcome: str = TRUNCATE

    def __post_init__(self) -> None:
        if not 0 <= self.cost <= 1:
            raise ValueError("the check's cost is a fraction of the computation")

    def coverage(self, fault_class: FaultClass) -> float:
        return fault_class.blow_up if fault_class.silent else 0.0

    @staticmethod
    def cost_fraction(
        words_per_token: int,
        tokens: int,
        compute_units: float,
        compare_cost: float = 1.0,
    ) -> float:
        """``compare_cost`` per checked word per token over the run's compute, in cost units."""

        if words_per_token < 0 or tokens < 0 or compute_units <= 0 or compare_cost < 0:
            raise ValueError("counts must be nonnegative and the compute positive")
        return words_per_token * tokens * compare_cost / compute_units

    @staticmethod
    def sees(bit: int, width: int, blow_up: float = SILENT_FAULTS.blow_up) -> bool:
        """Whether a flip of ``bit`` in a ``width``-bit word is in the high-order class the
        check catches: the top ``round(blow_up * width)`` bits stand for the sign and
        exponent of a floating-point word of that width."""

        if type(bit) is not int or type(width) is not int or not 0 <= bit < width:
            raise ValueError("bit must lie in [0, width)")
        return bit >= width - round(blow_up * width)


@dataclass(frozen=True, slots=True)
class PartialReexecution:
    """Re-run a fraction of the requests on idle capacity before the round closes.

    Sees ``fraction`` of every class with certainty (the re-execution is
    compared bit for bit) and costs ``fraction`` of the computation; the
    round has to stay open for one request's duration after its last request
    finished.  What it finds has already been streamed: the outcome is a
    declaration before ``J``.
    """

    fraction: float
    name: str = "partial re-execution"
    outcome: str = DECLARE

    def __post_init__(self) -> None:
        if not 0 <= self.fraction <= 1:
            raise ValueError("the re-executed fraction lies in [0, 1]")

    @property
    def cost(self) -> float:
        return self.fraction

    def coverage(self, fault_class: FaultClass) -> float:
        del fault_class
        return self.fraction


Detector = HardwareSignals | ValueCheck | PartialReexecution


def detected(detectors: Iterable[Detector], fault_class: FaultClass) -> float:
    """The fraction of ``fault_class`` some detector sees: ``1 - prod (1 - coverage)``."""

    missed = 1.0
    for detector in detectors:
        missed *= 1.0 - detector.coverage(fault_class)
    return 1.0 - missed


def coverage(
    detectors: Iterable[Detector],
    mixture: Sequence[FaultClass] = FAULT_MIXTURE,
    *,
    silent: bool = False,
) -> float:
    """The share-weighted fraction of the faults of ``mixture`` the detectors see,
    of its silent classes alone when ``silent``."""

    detectors = tuple(detectors)
    classes = [fault for fault in mixture if fault.silent or not silent]
    total = sum(fault.share for fault in classes)
    if total == 0:
        return 0.0
    return sum(fault.share * detected(detectors, fault) for fault in classes) / total


# -- the two prices of a declaration ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeclarationPrices:
    """What one declaration costs before and after ``J`` under ``policy``.

    ``pre`` is ``u(1) = W_V + log2 |S|``, the price of a declaration fixed
    before the challenges (:func:`~veritor.analysis.faults.unit_fault_bits`);
    ``post`` is ``u_post(1)``, the marginal charge of one post-J declaration:
    the smaller of the two bounds :func:`~veritor.analysis.faults.declared_bits`
    takes, less the capacity without declarations -- from the *uncapped* fold
    (``method = "fold"``) or from the closed-form rate as ``rho * log2 (1 /
    (1 - s))`` (``method = "rate"``).  ``base_bits`` is the uncapped capacity
    without declarations, ``rho`` the slope behind it, and ``capped`` says the
    run's interface hides the charge in ``bound`` itself (a small run: ``U``
    is ``|Out|`` with or without declarations).
    """

    policy: VerificationPolicy
    pre: float
    post: float
    rho: float
    base_bits: float
    capped: bool
    method: str

    @property
    def leverage(self) -> float:
        """``u_post(1) / u(1)``: how much dearer a declaration is after ``J``."""

        return self.post / self.pre if self.pre else math.inf

    @property
    def conservation(self) -> float:
        """``q u_post(1) / u(1)``: the expected post-J charge of a fault over its pre-J
        charge.  ``1`` is the conservation law exactly; above it detection before
        ``J`` saves bits per fault, below it loses them."""

        return float(self.policy.q) * self.leverage


def _uncapped(result: BoundResult) -> float:
    return min(result.knapsack_bits, result.laplace_bits)


def _verification_units(table: KindTable) -> int:
    return sum(row.copies for row in table.rows if row.role == VERIFICATION)


def fold_prices(
    target: Compiled | KindTable,
    policy: VerificationPolicy,
    eta: ProbabilityInput = ETA,
    options: BoundOptions | None = None,
) -> DeclarationPrices:
    """The prices from the fold itself: three ``bound`` calls at the thresholds
    :func:`~veritor.analysis.faults.declared_bits` uses, uncapped.

    On a run whose fold is saturated (every RU already corruptible within the
    budget) the marginal is not meaningful and is clamped at ``0``; ``capped``
    says so, and a table of many more RUs is the fix.
    """

    table = as_kind_table(target)
    threshold = exact_fraction(eta, name="eta")
    base = bound(table, policy, threshold, options)
    pre = unit_fault_bits(table)
    units = _verification_units(table)
    candidates = [
        _uncapped(bound(table, policy, threshold / (1 + units), options)) + pre
    ]
    if policy.s < 1:
        candidates.append(
            _uncapped(bound(table, policy, threshold * (1 - policy.s), options))
        )
    post = max(0.0, min(candidates) - _uncapped(base))
    return DeclarationPrices(
        policy=policy,
        pre=pre,
        post=post,
        rho=base.rho,
        base_bits=_uncapped(base),
        capped=base.capped,
        method="fold",
    )


def analytic_prices(
    target: Compiled | KindTable,
    policy: VerificationPolicy,
    eta: ProbabilityInput = ETA,
) -> DeclarationPrices:
    """The prices from the closed-form rate: ``u_post(1) = rho * min(log2 (1 / (1 - s)),
    log2 (1 + |S|) + u(1) / rho)``, the two bounds of ``declared_bits`` with the
    fold replaced by ``rho * (bits of threshold)`` (:mod:`veritor.analysis.rate`).

    This is the form ``docs/stress-tests.md`` (M6) quotes for the headline
    policy, where the fold itself is out of reach.
    """

    table = as_kind_table(target)
    threshold = exact_fraction(eta, name="eta")
    result = rate(table, policy)
    pre = unit_fault_bits(table)
    units = _verification_units(table)
    candidates = [result.rho * math.log2(1 + units) + pre]
    if policy.s < 1:
        candidates.append(result.rho * math.log2(1 / (1 - float(policy.s))))
    return DeclarationPrices(
        policy=policy,
        pre=pre,
        post=min(candidates),
        rho=result.rho,
        base_bits=result.capacity(threshold),
        capped=False,
        method="rate",
    )


def headline_prices() -> DeclarationPrices:
    """The prices at the headline policy of ``docs/global-estimate.md``: the year's
    serving as one circuit, ``(q, s)`` the 1% budget admits, ``rho`` from the rate."""

    from dataclasses import replace

    from veritor.evaluation.global_estimate import estimate
    from veritor.evaluation.serving import serving_table

    best = estimate()
    table = serving_table(
        replace(best.inputs.shape, requests=best.inputs.requests), "request", "cell"
    )
    policy = VerificationPolicy(Fraction(best.q), Fraction(best.s))
    return analytic_prices(table, policy, Fraction(1, 2 ** round(best.inputs.lam)))


# -- the frontier -----------------------------------------------------------------------


def _fraction(detected_fraction: float) -> float:
    if not 0 <= detected_fraction <= 1:
        raise ValueError("the detected fraction lies in [0, 1]")
    return detected_fraction


def expected_charge(
    detected_fraction: float, prices: DeclarationPrices, *, pre: float | None = None
) -> float:
    """``p u(1) + (1 - p) q u_post(1)``: what one fault costs the honest prover,
    ``p`` of the faults being detected before ``J``.

    A fault left to M6 is declared, and charged, only if its RU is opened,
    which happens with probability ``q``; a fault detected before ``J`` is
    charged with certainty.  ``pre`` overrides the pre-J price -- the
    truncation's advice bits when the detector fires before the token is
    streamed.
    """

    p = _fraction(detected_fraction)
    before = prices.pre if pre is None else pre
    return p * before + (1 - p) * float(prices.policy.q) * prices.post


def savings_per_fault(detected_fraction: float, prices: DeclarationPrices) -> float:
    """``p (q u_post(1) - u(1))``: the bits a round saves per fault by detecting ``p``
    of them before ``J`` and declaring at ``u(1)`` instead of leaving them to M6.
    Zero under the conservation law, negative when ``q u_post(1) < u(1)``; the
    detector's compute is not in it."""

    p = _fraction(detected_fraction)
    return p * (float(prices.policy.q) * prices.post - prices.pre)


def halving_fraction(prices: DeclarationPrices) -> float | None:
    """The detected fraction at which :func:`expected_charge` is half its ``p = 0``
    value, ``(q u_post / 2) / (q u_post - u(1))``; ``None`` when no ``p <= 1``
    halves it, which is the case unless ``q u_post(1) >= 2 u(1)`` -- twice the
    conservation law."""

    expected = float(prices.policy.q) * prices.post
    if expected <= prices.pre:
        return None
    fraction = (expected / 2) / (expected - prices.pre)
    return fraction if fraction <= 1 else None


def charge_deviation(detected_fraction: float, prices: DeclarationPrices) -> float:
    """The standard deviation of one fault's charge: ``u(1)`` with probability ``p``,
    else ``u_post(1)`` with probability ``q`` and ``0`` otherwise.  What detection
    buys when it does not move the mean: ``u_post(1) sqrt(q (1 - q))`` at ``p = 0``,
    ``0`` at ``p = 1``."""

    p = _fraction(detected_fraction)
    q = float(prices.policy.q)
    mean = expected_charge(p, prices)
    square = p * prices.pre**2 + (1 - p) * q * prices.post**2
    return math.sqrt(max(0.0, square - mean**2))


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """Partial re-execution at one fraction: the charge per fault and the compute it burns."""

    fraction: float
    charge_per_fault: float
    """:func:`expected_charge` at this fraction."""
    savings_per_fault: float
    """:func:`savings_per_fault` at this fraction."""
    compute: float
    """The re-execution as a fraction of the serving computation."""


def frontier(
    prices: DeclarationPrices,
    fractions: Iterable[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
) -> tuple[FrontierPoint, ...]:
    """:class:`FrontierPoint` at each re-execution fraction."""

    return tuple(
        FrontierPoint(
            fraction=fraction,
            charge_per_fault=expected_charge(fraction, prices),
            savings_per_fault=savings_per_fault(fraction, prices),
            compute=PartialReexecution(fraction).cost,
        )
        for fraction in fractions
    )


# -- what detection does buy: headroom under f_max --------------------------------------


def _poisson_tail(mean: float, count: int) -> float:
    """``P[Poisson(mean) > count]``, summed from ``count + 1`` upward so that a tail of
    ``1e-30`` keeps its digits (``1 - cdf`` would not)."""

    if mean == 0:
        return 0.0
    if count < mean - 40 * math.sqrt(mean) - 40:
        return 1.0  # the lower tail is below double precision
    k = count + 1
    term = math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))
    total = 0.0
    while k <= mean or term > total * 2.0**-60:
        total += term
        k += 1
        term *= mean / k
    return min(total, 1.0)


def rejection_probability(
    policy: VerificationPolicy,
    faults: float,
    max_faults: int,
    detected_fraction: float = 0.0,
) -> float:
    """The probability that a round with ``faults`` silent faults on average is
    rejected for exceeding its cap: ``P[Poisson(q faults (1 - p)) > f_max]``.

    Faults arrive as a Poisson process (:mod:`veritor.simulation.faults`); a
    fraction ``p`` is detected before ``J`` and handled at the pre-J price,
    and of the rest the q-challenge opens each independently, so the faults
    the post-J mechanism must pardon are Poisson with mean ``q faults (1 -
    p)``.  More than ``f_max`` of them is one rejected round.
    """

    if faults < 0:
        raise ValueError("faults per round is nonnegative")
    if type(max_faults) is not int or max_faults < 0:
        raise ValueError("max_faults must be a nonnegative integer")
    p = _fraction(detected_fraction)
    return _poisson_tail(float(policy.q) * faults * (1 - p), max_faults)


# -- what a truncation costs ------------------------------------------------------------


def truncation_information_bits(requests: int, truncated: int, max_new: int) -> float:
    """The bits that name which ``truncated`` of ``requests`` requests stopped early and
    where: ``log2 C(requests, truncated) + truncated * log2 max_new``.

    The least a pre-J truncation can be charged; what
    :class:`~veritor.constructors.truncation.TruncatedRequestsG` charges is
    :func:`truncation_charged_bits`, a length for every request.
    """

    if not 0 <= truncated <= requests or max_new < 1:
        raise ValueError("0 <= truncated <= requests and max_new >= 1")
    if truncated == 0:
        return 0.0
    return math.log2(math.comb(requests, truncated)) + truncated * math.log2(max_new)


def truncation_charged_bits(requests: int, max_new: int) -> int:
    """``requests * ceil(log2 max_new)``: the advice ``TruncatedRequestsG`` charges."""

    if requests < 0 or max_new < 1:
        raise ValueError("requests >= 0 and max_new >= 1")
    return requests * (max_new - 1).bit_length()


__all__ = [
    "DECLARE",
    "ETA",
    "FAULT_MIXTURE",
    "LLAMA3_GPU_INTERRUPTIONS",
    "SILENT_FAULTS",
    "TRUNCATE",
    "VISIBLE_FAULTS",
    "DeclarationPrices",
    "Detector",
    "FaultClass",
    "FrontierPoint",
    "HardwareSignals",
    "PartialReexecution",
    "ValueCheck",
    "analytic_prices",
    "charge_deviation",
    "coverage",
    "detected",
    "expected_charge",
    "fold_prices",
    "frontier",
    "halving_fraction",
    "headline_prices",
    "rejection_probability",
    "savings_per_fault",
    "truncation_charged_bits",
    "truncation_information_bits",
]
