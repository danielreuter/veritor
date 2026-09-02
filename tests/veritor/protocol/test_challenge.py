"""Challenge derivation: independent Bernoulli marginals at a cost set by the selection."""

from __future__ import annotations

import hashlib
import random
import time
from bisect import bisect_right
from fractions import Fraction
from itertools import accumulate, combinations
from math import comb, sqrt
from types import SimpleNamespace

import pytest

from veritor.core import VerificationLimits, VerificationPolicy
from veritor.protocol import (
    ProtocolError,
    VerificationCode,
    derive_replay_selection,
    derive_sample_selection,
    encode_transcript,
    run_protocol,
    verify_transcript,
)
from veritor.protocol.challenge import (
    _Q_STAGE,
    _S_STAGE,
    _UNIFORM_BITS,
    _binomial_count,
    _exceeds,
    bernoulli_subset,
)
from veritor.protocol.phases import boundary_phase, interior_phase, replay_phase

LIMITS = VerificationLimits()
PHASE = bytes(range(32))
STAGE = b"tests/stage"
HALF = Fraction(1, 2)


def seed(index: int) -> bytes:
    return hashlib.sha256(b"test_challenge seed " + index.to_bytes(4, "big")).digest()


def select(
    count: int,
    probability: Fraction,
    index: int = 0,
    *,
    stage: bytes = STAGE,
    phase: bytes = PHASE,
) -> tuple[int, ...]:
    return bernoulli_subset(seed(index), stage, phase, count, probability, LIMITS)


def pmf(count: int, probability: Fraction, k: int) -> Fraction:
    return comb(count, k) * probability**k * (1 - probability) ** (count - k)


def exact_cdf(count: int, probability: Fraction) -> list[Fraction]:
    return list(accumulate(pmf(count, probability, k) for k in range(count + 1)))


def exact_count(uniform: int, cdf: list[Fraction]) -> int:
    """Reference inversion of the exact binomial CDF at ``uniform / 2**_UNIFORM_BITS``."""

    return bisect_right(cdf, Fraction(uniform, 1 << _UNIFORM_BITS))


def within_noise(observed: int, expected: float, trials: int) -> bool:
    """``observed ~ Binomial(trials, expected / trials)`` within five sigma (plus slack)."""

    probability = expected / trials
    sigma = sqrt(trials * probability * (1 - probability))
    return abs(observed - expected) <= 5 * sigma + 3


def test_selection_is_deterministic_and_bound_to_seed_stage_and_phase() -> None:
    first = select(50, Fraction(1, 3), 1)

    assert first == select(50, Fraction(1, 3), 1)
    assert first != select(50, Fraction(1, 3), 2)
    assert first != select(50, Fraction(1, 3), 1, stage=b"other")
    assert first != select(50, Fraction(1, 3), 1, phase=b"\0" * 32)


@pytest.mark.parametrize(
    ("count", "probability"),
    [(1, HALF), (7, Fraction(1, 3)), (100, Fraction(9, 10)), (1000, Fraction(1, 50))],
)
def test_selection_is_sorted_unique_and_in_range(count: int, probability: Fraction) -> None:
    for index in range(10):
        selected = select(count, probability, index)

        assert list(selected) == sorted(set(selected))
        assert all(0 <= item < count for item in selected)
        assert all(type(item) is int for item in selected)


def test_degenerate_probabilities_and_counts() -> None:
    for count in (0, 1, 5):
        assert select(count, Fraction(0)) == ()
        assert select(count, Fraction(1)) == tuple(range(count))
    for probability in (Fraction(1, 7), HALF, Fraction(1, 10**9)):
        assert select(0, probability) == ()


def test_single_candidate_and_extreme_counts_occur() -> None:
    single = {select(1, HALF, index) for index in range(200)}
    assert single == {(), (0,)}

    sizes = {len(select(3, HALF, index)) for index in range(400)}
    assert {0, 3} <= sizes  # K = 0 and K = N each have probability 1 / 8


def test_invalid_arguments_are_protocol_errors() -> None:
    with pytest.raises(ProtocolError, match="seed"):
        bernoulli_subset(b"short", STAGE, PHASE, 3, HALF, LIMITS)
    with pytest.raises(ProtocolError, match="count"):
        bernoulli_subset(seed(0), STAGE, PHASE, -1, HALF, LIMITS)
    with pytest.raises(ProtocolError, match="probability"):
        bernoulli_subset(seed(0), STAGE, PHASE, 3, Fraction(3, 2), LIMITS)
    with pytest.raises(ProtocolError, match="probability"):
        bernoulli_subset(seed(0), STAGE, PHASE, 3, 0.5, LIMITS)  # type: ignore[arg-type]


@pytest.mark.parametrize("count", [1, 2, 6, 17, 300])
@pytest.mark.parametrize(
    "probability",
    [HALF, Fraction(1, 3), Fraction(9, 10), Fraction(1, 1000), Fraction(999, 1000)],
)
def test_count_inversion_agrees_with_exact_rational_inversion(
    count: int, probability: Fraction
) -> None:
    cdf = exact_cdf(count, probability)
    uniforms = [int.from_bytes(seed(index), "big") for index in range(40)]
    uniforms += [0, (1 << _UNIFORM_BITS) - 1]
    for total in cdf[:-1]:  # uniforms straddling every CDF step
        edge = (total.numerator << _UNIFORM_BITS) // total.denominator
        uniforms += [edge, edge + 1]

    for uniform in filter((1 << _UNIFORM_BITS).__gt__, uniforms):
        assert _binomial_count(uniform, count, probability) == exact_count(uniform, cdf)


def test_exceeds_matches_the_exact_comparison() -> None:
    rng = random.Random(0)
    for _ in range(3000):
        total = rng.getrandbits(rng.randint(1, 512))
        shift = rng.randint(200, 900)
        uniform = rng.getrandbits(rng.randint(0, _UNIFORM_BITS)) if rng.random() < 0.9 else 0
        exact = Fraction(total, 2**shift) > Fraction(uniform, 2**_UNIFORM_BITS)

        assert _exceeds(total, shift, uniform) is exact


@pytest.mark.parametrize(
    ("count", "probability"),
    [(4, Fraction(1, 3)), (6, HALF), (5, Fraction(9, 10)), (6, Fraction(1, 5))],
)
def test_marginals_pairs_and_counts_match_independent_coins(
    count: int, probability: Fraction
) -> None:
    trials = 5000
    marginal = [0] * count
    pairs = dict.fromkeys(combinations(range(count), 2), 0)
    sizes = [0] * (count + 1)
    for index in range(trials):
        selected = select(count, probability, index)
        sizes[len(selected)] += 1
        for item in selected:
            marginal[item] += 1
        for pair in combinations(selected, 2):
            pairs[pair] += 1

    for observed in marginal:
        assert within_noise(observed, trials * float(probability), trials)
    for observed in pairs.values():
        assert within_noise(observed, trials * float(probability) ** 2, trials)
    for k, observed in enumerate(sizes):
        assert within_noise(observed, trials * float(pmf(count, probability, k)), trials)


def stub_compiled(replay_units: int, units_per_replay_unit: int) -> SimpleNamespace:
    """Just enough of a ``Compiled`` for the derivations; nothing materialized."""

    def verification_units(unit: int) -> SimpleNamespace:
        return SimpleNamespace(first=unit * units_per_replay_unit, count=units_per_replay_unit)

    return SimpleNamespace(
        index=SimpleNamespace(
            replay_units=SimpleNamespace(count=replay_units),
            verification_units=verification_units,
        )
    )


def test_cost_follows_the_selection_not_the_candidates() -> None:
    billion, per_million = 10**9, Fraction(1, 10**6)
    for stage in (_Q_STAGE, _S_STAGE):
        start = time.perf_counter()
        selected = bernoulli_subset(seed(1), stage, PHASE, billion, per_million, LIMITS)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
        assert 700 <= len(selected) <= 1300  # mean 1000, sigma ~ 32
        assert list(selected) == sorted(set(selected))
        assert all(0 <= item < billion for item in selected)


def test_derivations_touch_only_selected_units_at_scale() -> None:
    limits = VerificationLimits(max_units=10**9)
    policy = VerificationPolicy(Fraction(1, 10**6), Fraction(1, 10**4))
    compiled = stub_compiled(10**9, 1000)  # type: ignore[assignment]

    start = time.perf_counter()
    replay = derive_replay_selection(seed(2), PHASE, compiled, policy, limits)
    assert time.perf_counter() - start < 0.5
    assert 700 <= len(replay) <= 1300

    selected_replay = tuple(range(0, 10**7, 1000))  # 10^4 replay units, 10^7 candidates
    start = time.perf_counter()
    sample = derive_sample_selection(seed(3), PHASE, compiled, selected_replay, policy, limits)
    assert time.perf_counter() - start < 0.5
    assert 700 <= len(sample) <= 1300
    assert list(sample) == sorted(set(sample))
    assert all(unit // 1000 in set(selected_replay) for unit in sample)


def test_sample_selection_ranks_the_selected_replay_units_blocks() -> None:
    blocks = {0: (0, 1, 2), 1: (3,), 2: (4, 5, 6, 7, 8), 3: (9, 10)}
    compiled = SimpleNamespace(
        index=SimpleNamespace(
            verification_units=lambda unit: SimpleNamespace(
                first=blocks[unit][0], count=len(blocks[unit])
            )
        )
    )
    selected_replay = (0, 2, 3)
    candidates = [unit for replay in selected_replay for unit in blocks[replay]]
    everything = VerificationPolicy(1, 1)

    def sample(policy: VerificationPolicy, index: int = 0) -> tuple[int, ...]:
        return derive_sample_selection(
            seed(index), PHASE, compiled, selected_replay, policy, LIMITS  # type: ignore[arg-type]
        )

    assert sample(everything) == tuple(candidates)
    assert sample(VerificationPolicy(1, 0)) == ()
    assert derive_sample_selection(
        seed(0), PHASE, compiled, (), everything, LIMITS  # type: ignore[arg-type]
    ) == ()

    trials = 3000
    hits = dict.fromkeys(candidates, 0)
    for index in range(trials):
        selected = sample(VerificationPolicy(1, HALF), index)
        assert list(selected) == sorted(set(selected))
        for unit in selected:
            hits[unit] += 1  # KeyError if a unit outside the selected blocks appears
    for observed in hits.values():
        assert within_noise(observed, trials / 2, trials)


def test_fractional_policy_runs_and_matches_independent_derivation(
    compiled, honest_values, expect
) -> None:
    policy = VerificationPolicy(HALF, Fraction(2, 3))
    nontrivial = False
    for index in range(8):
        expectation = expect(policy, q_seed=seed(2 * index), s_seed=seed(2 * index + 1))
        run = run_protocol(compiled, expectation, honest_values)

        assert run.report.code is VerificationCode.ACCEPTED
        assert run.transcript is not None
        transcript = run.transcript
        boundary = boundary_phase(transcript.header, transcript.boundary)
        interior = interior_phase(
            replay_phase(boundary, transcript.replay_challenge), transcript.interiors
        )
        replay = derive_replay_selection(
            expectation.q_seed, boundary, compiled, policy, LIMITS
        )
        sample = derive_sample_selection(
            expectation.s_seed, interior, compiled, replay, policy, LIMITS
        )
        assert run.report.sampled_replay_units == replay
        assert run.report.sampled_verification_units == sample
        assert all(
            compiled.index.verification_unit(unit).replay_unit in replay for unit in sample
        )
        recorded = encode_transcript(transcript)
        assert verify_transcript(recorded, expectation, compiled) == run.report
        nontrivial |= 0 < len(replay) < compiled.index.replay_units.count and 0 < len(sample)
    assert nontrivial
