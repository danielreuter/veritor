"""Component 3: the sampling distribution (challenge.py).

The marginals and pairs of ``J`` and ``T`` are covered in
``tests/veritor/protocol/test_challenge.py``.  Here the *acceptance* rate of
a prover that corrupts a fixed error set ``E`` is checked, over many fresh
seeds, against the exact survival ``sigma(E)`` of ``veritor.analysis``.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from veritor.analysis.probability import survival
from veritor.core import VerificationLimits
from veritor.protocol import (
    VerificationCode,
    derive_replay_selection,
    derive_sample_selection,
)

TRIALS = 1000
SIGMAS = 4.0


def sigma_of(model, policy, corrupted: list[int]):
    """``sigma(E)`` for the error set of the units holding ``corrupted``."""

    per_replay = Counter(
        model.index.replay_units.owner(address) for address in corrupted
    )
    return survival(policy, per_replay.values())


def within(rate: float, expected: float, trials: int) -> bool:
    deviation = math.sqrt(expected * (1 - expected) / trials)
    return abs(rate - expected) <= SIGMAS * deviation


@pytest.mark.parametrize(
    ("label", "cells"),
    [
        ("distinct-replay-units", [(0, 0), (1, 1)]),  # sigma = (1 - qs)^2 = 9/16
        ("same-replay-unit", [(0, 0), (0, 1)]),  # sigma = 1 - q + q(1 - s)^2 = 5/8
    ],
)
def test_acceptance_rate_matches_survival_of_the_error_set(model, sec, label, cells):
    """A prover with a fixed error set escapes with frequency ``sigma(E)`` (within 4 sigma)."""

    policy = sec.HALVES
    corrupted = [model.cell_addresses(stage, cell)[0] for stage, cell in cells]
    values, outputs = model.corrupt({address: 0 for address in corrupted})
    assert outputs != model.outputs
    expected = sigma_of(model, policy, corrupted)
    accepted = 0
    for trial in range(TRIALS):
        expectation = model.expectation(
            policy,
            claimed_outputs=outputs,
            q_seed=sec.seed(f"{label}/q", trial),
            s_seed=sec.seed(f"{label}/s", trial),
        )
        report = model.run(expectation, values).report
        if report.accepted:
            accepted += 1
        else:  # every rejection is the relation check of a sampled corrupted unit
            assert report.code == VerificationCode.RELATION_REJECTED
            assert set(report.sampled_verification_units) & model.error_units(corrupted)
    rate = accepted / TRIALS
    assert within(rate, float(expected), TRIALS), (label, rate, float(expected))


def test_selection_law_alone_matches_survival_over_many_seeds(model, sec):
    """The sampler by itself: ``P[E ∩ T = ∅]`` over fresh seeds equals ``sigma(E)``."""

    policy = sec.HALVES
    limits = VerificationLimits()
    cases = {
        "distinct": [model.cell_addresses(0, 0)[0], model.cell_addresses(1, 1)[0]],
        "same": [model.cell_addresses(0, 0)[0], model.cell_addresses(0, 1)[0]],
        "three": [
            model.cell_addresses(0, 0)[0],
            model.cell_addresses(0, 1)[0],
            model.cell_addresses(1, 0)[0],
        ],
    }
    trials = 3000
    for label, corrupted in cases.items():
        errors = model.error_units(corrupted)
        expected = float(sigma_of(model, policy, corrupted))
        escaped = 0
        for trial in range(trials):
            phase = sec.seed(f"phase/{label}", trial)
            j = derive_replay_selection(
                sec.seed(f"q/{label}", trial), phase, model.compiled, policy, limits
            )
            t = derive_sample_selection(
                sec.seed(f"s/{label}", trial), phase, model.compiled, j, policy, limits
            )
            if not errors & set(t):
                escaped += 1
        assert within(escaped / trials, expected, trials), (
            label,
            escaped / trials,
            expected,
        )


def test_survival_is_the_product_of_per_replay_unit_factors(model, sec):
    """The formula the tests compare against, spelled out for the fixture."""

    policy = sec.HALVES
    q, s = policy.q, policy.s
    distinct = sigma_of(
        model, policy, [model.cell_addresses(0, 0)[0], model.cell_addresses(1, 1)[0]]
    )
    same = sigma_of(
        model, policy, [model.cell_addresses(0, 0)[0], model.cell_addresses(0, 1)[0]]
    )
    assert distinct == (1 - q * s) ** 2
    assert same == 1 - q + q * (1 - s) ** 2
    assert same > distinct  # concentrating errors in one replay unit is cheaper
