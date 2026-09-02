"""Component 6: admission (session._admit, parameters.py).

``eta`` is the verifier's and is bound into the header; theta's denominators
are capped; ``U_max`` and ``W_max`` are checked from counts alone before any
commitment; a transcript recorded under another ``eta`` is
EXPECTATION_MISMATCH.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from veritor.analysis import bound
from veritor.core import VerificationLimits, VerificationPolicy
from veritor.protocol import (
    ProtocolError,
    VerificationCode,
    VerifierParameters,
    expected_work,
    make_expectation,
)


def test_theta_with_an_enormous_denominator_is_resource_limit(model, sec):
    """A rate ``1/(2**65 + 1)`` exceeds ``max_probability_denominator_bits`` (64)."""

    huge = VerificationPolicy(Fraction(1, (1 << 65) + 1), 1)
    run = model.run(model.expectation(huge), model.values)
    assert run.report.code == VerificationCode.RESOURCE_LIMIT
    assert "probability_denominator_bits" in run.report.detail
    assert run.transcript is None and run.report.sampled_replay_units == ()
    # the cap is the verifier's: a tighter limit rejects a coarser rate
    tight = VerificationLimits(max_probability_denominator_bits=2)
    report = model.run(model.expectation(VerificationPolicy(Fraction(1, 5), 1)), model.values, limits=tight).report
    assert report.code == VerificationCode.RESOURCE_LIMIT
    # ... and so is eta's denominator
    eta = VerifierParameters(Fraction(1, (1 << 65) + 1), max_capacity=None)
    report = model.run(model.expectation(sec.HALVES, parameters=eta), model.values).report
    assert report.code == VerificationCode.RESOURCE_LIMIT
    # a recorded transcript with such a rate is refused by the decoder before anything else
    run, expectation = model.run(model.expectation(), model.values), model.expectation()

    def inflate(document: dict) -> None:
        document["header"]["policy"]["q"] = [1, (1 << 65) + 1]

    report = model.verify(sec.mutate_transcript(run.transcript, inflate), expectation)
    assert report.code == VerificationCode.RESOURCE_LIMIT


def test_policy_whose_bound_exceeds_u_max_is_policy_rejected(model, sec):
    eta = Fraction(1, 4)
    leaky = bound(model.compiled, sec.HALVES, eta).bits
    assert leaky == 2 * model.width == 16  # the whole interface (two 8-bit outputs) leaks
    parameters = VerifierParameters(eta, max_capacity=math.floor(leaky) - 1)
    run = model.run(model.expectation(sec.HALVES, parameters=parameters), model.values)
    assert run.report.code == VerificationCode.POLICY_REJECTED
    assert run.transcript is None and run.report.sampled_replay_units == ()
    # the client cannot lower the bar: only the verifier's parameters carry U_max and eta
    admitted = model.run(model.expectation(sec.CHECK_EVERYTHING, parameters=parameters), model.values)
    assert admitted.report.accepted
    assert bound(model.compiled, sec.CHECK_EVERYTHING, eta).bits == 0.0
    exact = VerifierParameters(eta, max_capacity=math.floor(leaky))
    assert model.run(model.expectation(sec.HALVES, parameters=exact), model.values).report.accepted


def test_run_whose_expected_work_exceeds_w_max_is_work_budget_exceeded(model, sec):
    io = len(set(model.circuit.inputs) | set(model.circuit.outputs))
    work = expected_work(model.compiled, sec.HALVES, io)
    assert work > 1 and work.denominator > 1
    below = VerifierParameters(max_work=math.floor(work), max_capacity=None)
    run = model.run(model.expectation(sec.HALVES, parameters=below), model.values)
    assert run.report.code == VerificationCode.WORK_BUDGET_EXCEEDED
    assert run.transcript is None and run.report.sampled_replay_units == ()
    above = VerifierParameters(max_work=math.ceil(work), max_capacity=None)
    assert model.run(model.expectation(sec.HALVES, parameters=above), model.values).report.accepted
    # priced from the kinds: more sampling is more work, and checking everything is the most
    everything = expected_work(model.compiled, sec.CHECK_EVERYTHING, io)
    assert everything > work > expected_work(model.compiled, VerificationPolicy(0, 0), io) > 0


def test_eta_is_the_verifiers_and_bound_into_the_header(model, sec):
    parameters = VerifierParameters(Fraction(1, 8), max_capacity=None)
    expectation = model.expectation(sec.HALVES, parameters=parameters)
    header = model.header(expectation)
    assert header.eta == Fraction(1, 8) == expectation.parameters.eta
    assert not hasattr(sec.HALVES, "eta")  # the proposal has no eta to carry
    with pytest.raises(ProtocolError, match=r"eta must lie in \[0, 1\)"):
        VerifierParameters(1, max_capacity=None)
    with pytest.raises(TypeError):
        VerifierParameters(0.25, max_capacity=None)  # type: ignore[arg-type]
    # a header (thus a transcript) recorded under another eta does not verify under this one
    run = model.run(expectation, model.values)
    assert run.report.accepted
    other = model.expectation(sec.HALVES, parameters=VerifierParameters(Fraction(1, 16), max_capacity=None))
    report = model.verify(run.transcript, other)
    assert report.code == VerificationCode.EXPECTATION_MISMATCH and "eta" in report.detail

    def rewrite_eta(document: dict) -> None:
        document["header"]["eta"] = [1, 16]

    report = model.verify(sec.mutate_transcript(run.transcript, rewrite_eta), expectation)
    assert report.code == VerificationCode.EXPECTATION_MISMATCH


def test_waiving_u_max_admits_a_policy_that_checks_nothing(model, sec):
    """``max_capacity=None`` waives ``U_max``; ``theta = (0, 0)`` is then admitted.

    Nothing is sampled, so the only checks are the boundary's public I/O
    against the header -- both of which the client supplied.  A false output
    is accepted with certainty, and ``Bound`` says so (``bits == out_bits``).
    That is why the waiver has to be written out: ``VerifierParameters`` has
    no default for ``max_capacity`` and ``make_expectation`` no default
    parameters (F2).
    """

    nothing = VerificationPolicy(0, 0)
    false_claim = tuple((y + 1) % (1 << model.width) for y in model.outputs)
    forged = dict(model.values)
    for address, value in zip(model.circuit.outputs, false_claim, strict=True):
        forged[address] = value
    run = model.run(model.expectation(nothing, claimed_outputs=false_claim), forged)
    assert run.report.code == VerificationCode.ACCEPTED
    assert run.report.sampled_replay_units == () and run.report.sampled_verification_units == ()
    result = bound(model.compiled, nothing, Fraction(1, 2))
    assert result.capped and result.bits == result.out_bits
    # with U_max set, the same proposal is refused before any commitment
    strict = VerifierParameters(Fraction(1, 2), max_capacity=result.out_bits - 1)
    run = model.run(model.expectation(nothing, claimed_outputs=false_claim, parameters=strict), forged)
    assert run.report.code == VerificationCode.POLICY_REJECTED
    # and a verifier cannot forget to decide: there is no default
    with pytest.raises(TypeError, match="max_capacity"):
        VerifierParameters()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="parameters"):
        make_expectation(model.compilation(), nothing, false_claim)  # type: ignore[call-arg]


def test_admission_checks_unit_counts_against_the_limits(model, sec):
    """``max_units`` and ``max_positions_per_unit`` are enforced at admission, from the kinds."""

    few = VerificationLimits(max_units=model.index.verification_unit_count - 1)
    run = model.run(model.expectation(), model.values, limits=few)
    assert run.report.code == VerificationCode.RESOURCE_LIMIT and "units" in run.report.detail
    narrow = VerificationLimits(max_positions_per_unit=1)
    run = model.run(model.expectation(), model.values, limits=narrow)
    assert run.report.code == VerificationCode.RESOURCE_LIMIT
    assert "positions_per_unit" in run.report.detail
    assert run.transcript is None
