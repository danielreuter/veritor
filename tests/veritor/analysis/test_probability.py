from fractions import Fraction

import pytest

from veritor.analysis import (
    PositionErrorSet,
    VerificationUnitErrorSet,
    survival_probability,
    survives_strict_threshold,
)
from veritor.core import VerificationPolicy


@pytest.mark.parametrize(
    ("q", "s", "expected"),
    [
        (0, Fraction(3, 5), Fraction(1)),
        (Fraction(2, 3), 0, Fraction(1)),
        (1, 1, Fraction(0)),
        (1, Fraction(1, 2), Fraction(1, 4)),
    ],
)
def test_survival_probability_handles_q_s_endpoints(
    make_partitions,
    q,
    s,
    expected,
):
    replay, verification = make_partitions((2,))
    policy = VerificationPolicy(q, s, 0)

    assert survival_probability(
        replay,
        verification,
        policy,
        VerificationUnitErrorSet((0, 1)),
    ) == expected


def test_strict_equality_at_eta_is_not_admissible(make_partitions):
    replay, verification = make_partitions((1,))
    policy = VerificationPolicy(1, Fraction(1, 2), Fraction(1, 2))
    survival = survival_probability(replay, verification, policy, (0,))

    assert survival == policy.eta
    assert not survives_strict_threshold(survival, policy)


def test_replay_correlation_is_not_independent_scalar_sampling(make_partitions):
    replay, verification = make_partitions((2, 1))
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2), 0)

    same_replay = survival_probability(replay, verification, policy, (0, 1))
    across_replays = survival_probability(replay, verification, policy, (0, 2))
    independent_scalar = (1 - policy.q * policy.s) ** 2

    assert same_replay == Fraction(5, 8)
    assert across_replays == Fraction(9, 16)
    assert same_replay > independent_scalar
    assert across_replays == independent_scalar


def test_position_and_unit_error_sets_are_explicit_and_equivalent(make_partitions):
    replay, verification = make_partitions((2, 1))
    policy = VerificationPolicy(Fraction(1, 3), Fraction(2, 5), 0)

    by_units = survival_probability(
        replay,
        verification,
        policy,
        VerificationUnitErrorSet((0, 2)),
    )
    by_positions = survival_probability(
        replay,
        verification,
        policy,
        PositionErrorSet((10, 12)),
    )
    keyword_positions = survival_probability(
        replay,
        verification,
        policy,
        (10, 12),
        attack_kind="positions",
    )

    assert by_units == by_positions == keyword_positions


def test_survival_is_monotone_under_attack_inclusion(make_partitions):
    replay, verification = make_partitions((3, 2))
    policy = VerificationPolicy(Fraction(2, 3), Fraction(1, 4), 0)
    nested_attacks = ((), (0,), (0, 1), (0, 1, 3), (0, 1, 2, 3, 4))
    probabilities = [
        survival_probability(replay, verification, policy, attack)
        for attack in nested_attacks
    ]

    assert probabilities == sorted(probabilities, reverse=True)


def test_error_sets_reject_booleans_and_unknown_members(make_partitions):
    replay, verification = make_partitions((1,))
    policy = VerificationPolicy(1, 1, 0)

    with pytest.raises(ValueError, match="nonnegative integers"):
        survival_probability(replay, verification, policy, (True,))
    with pytest.raises(ValueError, match="unknown verification unit"):
        survival_probability(replay, verification, policy, (1,))
    with pytest.raises(ValueError, match="ineligible computed position"):
        survival_probability(
            replay,
            verification,
            policy,
            PositionErrorSet((999,)),
        )
