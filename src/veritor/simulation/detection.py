"""Detection before the challenge: what an honest prover can know before ``J``,
what knowing it costs, and whether it ever pays (``docs/honest-prover.md``,
section 7).

M6 lets the prover declare a faulty VU *after* the q-challenge, at the
adaptive price ``bound(..., max_faults=f)`` charges: about ``u(1) / q`` bits
per declaration, ``u(1) = W_V + log2 |S|`` being the price of a declaration
fixed before the challenges (:mod:`veritor.analysis.faults`).  A prover that
knew its faults before ``J`` could declare them at ``u(1)`` -- there is no
such message in the protocol today; it is the priced extension this module
prices -- or, if it knew before the garbage was streamed, abort the request
and pay nothing but the truncation's advice bits (S7,
:class:`~veritor.constructors.truncation.TruncatedRequestsG`).  Three ways
to know are modelled here, each with its coverage by fault class and its
cost as a fraction of the honest computation:

* :class:`HardwareSignals`: ECC counters, Xid errors, watchdogs, crashes.
  Free, and they cover the memory and device-failure classes completely,
  but not the compute-path corruption that is silent by definition
  (``docs/notes/datacenter-realities.md`` section 7; Dixit et al. 2021,
  Hochschild et al. 2021).
* :class:`ValueCheck`: NaN/Inf, range and degenerate-output checks on the
  activations or logits before a token is streamed.  An elementwise pass,
  so a fraction ``words checked / compute per token`` of the computation;
  covers the faults that blow a value up (high-order exponent bits: Li et
  al. 2017, Chen et al. 2021) and none of the low-order ones.  A fault it
  catches is never streamed: the request is truncated at that token.
* :class:`PartialReexecution`: re-run a fraction ``p`` of the requests on
  idle capacity before the round closes.  Covers ``p`` of every class with
  certainty (a bit-exact comparison), costs ``p`` of the computation at the
  idle price, and holds the round open for one request's duration.

The conservation law the programme tests (``docs/honest-prover.md``): a
pardon before stage ``k`` has leverage ``1 / p_k`` over one after it, and
buying that leverage costs checking everything still alive at stage ``k``.
Here it reads ``u_post(1) ~ u(1) / q``: a fault the prover leaves to the
post-J mechanism is charged only if its RU is opened (probability ``q``),
so its expected charge ``q u_post(1)`` is ``u(1)`` -- the same bits a
pre-J declaration costs with certainty.  :func:`savings_per_fault` is the
difference, a few bits per fault either way, and :func:`frontier` sets it
against the compute the detector burns.
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

LLAMA3_UNEXPECTED_INTERRUPTIONS = 419
"""Llama-3 405B pre-training: unexpected interruptions in 54 days on 16,384 GPUs."""
LLAMA3_GPU_SHARE = 0.587
"""The share of those attributed to GPUs (faulty GPU/NVLink 148, HBM3 72, SDC 6, other GPU 20)."""
LLAMA3_GPU_INTERRUPTIONS = round(LLAMA3_GPU_SHARE * LLAMA3_UNEXPECTED_INTERRUPTIONS)
"""246 GPU-attributed interruptions, 6 of them silent data corruption."""


@dataclass(frozen=True, slots=True)
class FaultClass:
    """A class of hardware fault and how it presents to the server that suffered it.

    ``share`` is its fraction of the fault events a fleet sees (the
    Llama-3 GPU mixture by default); ``silent`` says whether the fault
    reaches the streamed computation without any hardware signal;
    ``blow_up`` is the fraction of the class's faults that move a value far
    enough for a pre-streaming value check to see (0 for a class that is
    never silently streamed).
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
        "Llama-3 (2024) Table 5 via docs/notes/datacenter-realities.md section 7: 148 faulty "
        "GPU/NVLink, 72 HBM3, 20 other GPU of 246 GPU-attributed interruptions; NVIDIA Xid docs: "
        "ECC corrects single-bit HBM errors, a double-bit error (Xid 48) or row remap (Xid 63/64) "
        "kills the context rather than continue silently"
    ),
)
"""Faults the server sees as they happen: nothing wrong is streamed, the request is aborted."""

SILENT_FAULTS = FaultClass(
    name="silent compute-path corruption (datapath, unprotected latches, mercurial cores)",
    share=LLAMA3_SDC_EVENTS / LLAMA3_GPU_INTERRUPTIONS,
    silent=True,
    blow_up=0.5,
    source=(
        "Llama-3 (2024): 6 SDC events, 2.8e-7 per device-hour; Dixit et al. (2021) and Hochschild "
        "et al. (2021): wrong results with no ECC or parity signal, found by test patterns; Meta "
        "(2025): about one SDC-prone device per thousand. blow_up 0.5 is an assumption: half the "
        "bits of a bf16 word are sign and exponent, and only high-order exponent flips move a value "
        "out of range (Li et al. 2017); range restriction then catches over 97% of the "
        "output-corrupting faults at 0.5% overhead (Chen et al. 2021)"
    ),
)
"""Faults that reach the streamed tokens with no signal: the class M6 exists for."""

FAULT_MIXTURE: tuple[FaultClass, ...] = (VISIBLE_FAULTS, SILENT_FAULTS)
"""The Llama-3 GPU fault mixture: 97.6% visible, 2.4% silent."""

# -- detectors --------------------------------------------------------------------------

TRUNCATE = "truncate"
"""The detector fires before the garbage is streamed: the request ends there (S7, advice bits)."""
DECLARE = "declare"
"""The detector fires after the tokens are out but before ``J``: a pre-J declaration at ``u(1)``."""


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

    ``cost`` is the pass as a fraction of the honest computation
    (:meth:`cost_fraction`).  It sees a fault iff the fault blew a value up
    -- the high-order bits of a word (:meth:`sees`) -- and it sees them
    *before* the token is streamed, so what it catches is truncated, not
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
        words_per_token: int, tokens: int, compute_units: float, compare_cost: float = 1.0
    ) -> float:
        """One comparison per checked word per token over the run's compute, in cost units."""

        if words_per_token < 0 or tokens < 0 or compute_units <= 0 or compare_cost < 0:
            raise ValueError("counts must be nonnegative and the compute positive")
        return words_per_token * tokens * compare_cost / compute_units

    @staticmethod
    def sees(bit: int, width: int, blow_up: float = SILENT_FAULTS.blow_up) -> bool:
        """Whether a flip of ``bit`` in a ``width``-bit word is in the high-order class the check catches.

        The top ``round(blow_up * width)`` bits stand for the sign and exponent
        of a floating-point word of that width: a flip there moves the value
        by orders of magnitude, a flip below it by a fraction (Li et al. 2017).
        """

        if type(bit) is not int or type(width) is not int or not 0 <= bit < width:
            raise ValueError("bit must lie in [0, width)")
        return bit >= width - round(blow_up * width)


@dataclass(frozen=True, slots=True)
class PartialReexecution:
    """Re-run a fraction of the requests on idle capacity before the round closes.

    Covers ``fraction`` of every class with certainty (the re-execution is
    compared bit for bit), costs ``fraction`` of the computation at
    ``idle_price`` (``0`` when the fleet has slack that would otherwise idle,
    ``1`` at the full price), and needs the round to stay open for one
    request's duration after its last request finished (``delay_seconds``).
    What it finds has already been streamed: the outcome is a declaration
    before ``J``.
    """

    fraction: float
    idle_price: float = 1.0
    name: str = "partial re-execution"
    outcome: str = DECLARE

    def __post_init__(self) -> None:
        if not 0 <= self.fraction <= 1 or not 0 <= self.idle_price <= 1:
            raise ValueError("fraction and idle_price are fractions in [0, 1]")

    @property
    def cost(self) -> float:
        return self.fraction * self.idle_price

    def coverage(self, fault_class: FaultClass) -> float:
        del fault_class
        return self.fraction

    @staticmethod
    def delay_seconds(request_seconds: float) -> float:
        """The round-close delay: the re-execution of the last request to finish."""

        if request_seconds < 0:
            raise ValueError("a request's duration is nonnegative")
        return request_seconds


Detector = HardwareSignals | ValueCheck | PartialReexecution


def detected(detectors: Iterable[Detector], fault_class: FaultClass) -> float:
    """The fraction of ``fault_class`` some detector sees: ``1 - prod (1 - coverage)``."""

    missed = 1.0
    for detector in detectors:
        missed *= 1.0 - detector.coverage(fault_class)
    return 1.0 - missed


def silent_coverage(
    detectors: Iterable[Detector], mixture: Sequence[FaultClass] = FAULT_MIXTURE
) -> float:
    """The share-weighted fraction of the *silent* faults of ``mixture`` the detectors see."""

    detectors = tuple(detectors)
    silent = [fault_class for fault_class in mixture if fault_class.silent]
    total = sum(fault_class.share for fault_class in silent)
    if total == 0:
        return 0.0
    return sum(
        fault_class.share * detected(detectors, fault_class) for fault_class in silent
    ) / total


# -- the two prices of a declaration ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeclarationPrices:
    """What one declaration costs before and after ``J`` under ``policy``.

    ``pre`` is ``u(1) = W_V + log2 |S|``, the price of a declaration fixed
    before the challenges (:func:`~veritor.analysis.faults.fault_allowance_bits`);
    ``post`` is the marginal adaptive charge of one post-J declaration, the
    smaller of the two bounds :func:`~veritor.analysis.faults.declared_bits`
    takes, minus the undeclared capacity -- computed on the *uncapped* fold
    (``method = "fold"``) or from the closed-form rate as ``rho * log2 (1 /
    (1 - s))`` (``method = "rate"``).  ``base_bits`` is the uncapped
    capacity without declarations, ``rho`` the slope behind it, and
    ``capped`` says the run's interface hides the charge in ``bound`` itself
    (a small run: ``U`` is ``|Out|`` with or without declarations).
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
        """``q u_post(1) / u(1)``: the expected post-J charge of a fault over its pre-J charge.

        ``1`` is the conservation law exactly; above it detection before
        ``J`` saves bits per fault, below it loses them.
        """

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
    budget, ``U = |Out|``) the marginal is not meaningful and is clamped at
    ``0``; ``capped`` says so, and a table of many more RUs is the fix.
    """

    table = as_kind_table(target)
    threshold = exact_fraction(eta, name="eta")
    base = bound(table, policy, threshold, options)
    pre = unit_fault_bits(table)
    units = _verification_units(table)
    candidates = [_uncapped(bound(table, policy, threshold / (1 + units), options)) + pre]
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


def expected_charge(
    detected_fraction: float, prices: DeclarationPrices, *, pre: float | None = None
) -> float:
    """``p u(1) + (1 - p) u_post(1)``: the charge of one fault, ``p`` of which the
    detectors see before ``J``.

    ``pre`` overrides the pre-J price -- the truncation's advice bits when the
    detector fires before the token is streamed.  This is the charge *given
    that the fault is charged at all*; :func:`expected_round_charge` weighs
    the post-J term by the probability the fault's RU is opened.
    """

    if not 0 <= detected_fraction <= 1:
        raise ValueError("the detected fraction lies in [0, 1]")
    before = prices.pre if pre is None else pre
    return detected_fraction * before + (1 - detected_fraction) * prices.post


def expected_round_charge(
    detected_fraction: float, prices: DeclarationPrices, *, pre: float | None = None
) -> float:
    """``p u(1) + (1 - p) q u_post(1)``: what one fault costs the honest prover per round.

    A fault left to M6 is declared, and charged, only if its RU is opened,
    which happens with probability ``q``; a fault declared before ``J`` is
    charged whether or not its RU is opened.
    """

    if not 0 <= detected_fraction <= 1:
        raise ValueError("the detected fraction lies in [0, 1]")
    before = prices.pre if pre is None else pre
    return detected_fraction * before + (1 - detected_fraction) * float(
        prices.policy.q
    ) * prices.post


def halving_fraction(prices: DeclarationPrices) -> float | None:
    """The detected fraction at which :func:`expected_charge` is half its ``p = 0`` value,
    ``(u_post / 2) / (u_post - u(1))``; ``None`` when no ``p <= 1`` halves it, i.e.
    when ``u_post(1) < 2 u(1)`` -- the case at every ``q >= 1/2``."""

    if prices.post <= prices.pre:
        return None
    fraction = (prices.post / 2) / (prices.post - prices.pre)
    return fraction if fraction <= 1 else None


def savings_per_fault(detected_fraction: float, prices: DeclarationPrices) -> float:
    """Bits a round saves per fault by detecting ``p`` of them before ``J`` and
    declaring at ``u(1)``, instead of leaving them to M6: ``p (q u_post(1) - u(1))``.

    Zero under the conservation law, negative when the adaptive charge is
    below ``u(1) / q``; the detector's compute is not in it.
    """

    if not 0 <= detected_fraction <= 1:
        raise ValueError("the detected fraction lies in [0, 1]")
    return detected_fraction * (float(prices.policy.q) * prices.post - prices.pre)


def expected_opened_faults(policy: VerificationPolicy, faults_per_round: float) -> float:
    """``q * faults``: the faults per round whose RU the q-challenge opens, the only
    ones the post-J mechanism ever charges."""

    if faults_per_round < 0:
        raise ValueError("faults per round is nonnegative")
    return float(policy.q) * faults_per_round


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """Partial re-execution at one fraction: charges per fault and the compute it burns."""

    fraction: float
    charge_per_fault: float
    """:func:`expected_charge` at this fraction."""
    round_charge_per_fault: float
    """:func:`expected_round_charge` at this fraction."""
    savings_per_fault: float
    """:func:`savings_per_fault` at this fraction."""
    compute: float
    """The re-execution as a fraction of the honest computation, at the idle price."""


def frontier(
    prices: DeclarationPrices,
    fractions: Iterable[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    idle_price: float = 1.0,
) -> tuple[FrontierPoint, ...]:
    """:class:`FrontierPoint` at each re-execution fraction."""

    points = []
    for fraction in fractions:
        plan = PartialReexecution(fraction, idle_price)
        points.append(
            FrontierPoint(
                fraction=fraction,
                charge_per_fault=expected_charge(fraction, prices),
                round_charge_per_fault=expected_round_charge(fraction, prices),
                savings_per_fault=savings_per_fault(fraction, prices),
                compute=plan.cost,
            )
        )
    return tuple(points)


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
    "LLAMA3_GPU_SHARE",
    "LLAMA3_UNEXPECTED_INTERRUPTIONS",
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
    "detected",
    "expected_charge",
    "expected_opened_faults",
    "expected_round_charge",
    "fold_prices",
    "frontier",
    "halving_fraction",
    "headline_prices",
    "savings_per_fault",
    "silent_coverage",
    "truncation_charged_bits",
    "truncation_information_bits",
]
