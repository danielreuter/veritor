from fractions import Fraction

import pytest

from veritor.commitment import CommitmentOwner
from veritor.core import ResourceLimit, VerificationLimits, VerificationPolicy
from veritor.staged import (
    derive_commitment_ownership,
    derive_q_challenge,
    derive_replay_boundary,
    derive_s_challenge,
    exact_rejection_map,
    survives_acceptance_threshold,
    two_stage_survival_probability,
)


def test_boundary_contains_io_and_cross_unit_reads_with_complete_ownership(
    protocol_case,
):
    circuit = protocol_case.artifact.circuit
    replay = protocol_case.artifact.replay_partition

    layout = derive_commitment_ownership(circuit, replay)

    assert derive_replay_boundary(circuit, replay) == (10, 20, 40, 50)
    assert layout.boundary.items == (10, 20, 40, 50)
    assert tuple(interior.items for interior in layout.interiors) == ((30,), ())
    assert layout.owner_of(10) == CommitmentOwner.boundary()
    assert layout.owner_of(40) == CommitmentOwner.boundary()
    assert layout.owner_of(30) == CommitmentOwner.replay_unit(0)
    assert layout.position_count == 5
    assert set(layout.boundary.items).isdisjoint(layout.interiors[0].items)
    assert set(layout.boundary.items) | set(layout.interiors[0].items) == {
        10,
        20,
        30,
        40,
        50,
    }


def test_boundary_derivation_enforces_finite_resource_limits(protocol_case):
    with pytest.raises(ResourceLimit):
        derive_commitment_ownership(
            protocol_case.artifact.circuit,
            protocol_case.artifact.replay_partition,
            VerificationLimits(max_positions=4),
        )


def test_q_and_s_endpoints_are_exact(protocol_case):
    replay = protocol_case.artifact.replay_partition
    verification = protocol_case.artifact.verification_partition
    phase = b"P" * 32

    assert (
        derive_q_challenge(
            b"Q" * 32,
            phase,
            replay,
            Fraction(0),
        )
        == ()
    )
    assert derive_q_challenge(
        b"Q" * 32,
        phase,
        replay,
        Fraction(1),
    ) == (0, 1)
    assert (
        derive_s_challenge(
            b"S" * 32,
            phase,
            verification,
            (0, 1),
            Fraction(0),
        )
        == ()
    )
    assert derive_s_challenge(
        b"S" * 32,
        phase,
        verification,
        (0, 1),
        Fraction(1),
    ) == (0, 1, 2)


def test_challenge_known_answer_and_stage_separation(protocol_case):
    replay = protocol_case.artifact.replay_partition
    verification = protocol_case.artifact.verification_partition
    phase = bytes(range(32))

    assert derive_q_challenge(
        bytes(range(32, 64)),
        phase,
        replay,
        Fraction(1, 2),
    ) == (1,)
    assert derive_s_challenge(
        bytes(range(64, 96)),
        phase,
        verification,
        (0, 1),
        Fraction(1, 2),
    ) == (0,)
    q_draw = exact_rejection_map(
        b"K" * 32,
        b"q/replay-unit",
        phase,
        b"candidate",
        97,
    )
    s_draw = exact_rejection_map(
        b"K" * 32,
        b"s/verification-unit",
        phase,
        b"candidate",
        97,
    )
    assert q_draw == 18
    assert s_draw == 37


def test_huge_rational_challenges_never_use_float(protocol_case):
    denominator = 1 << 4096
    probability = Fraction(denominator - 1, denominator)
    phase = b"H" * 32

    selected = derive_q_challenge(
        b"Q" * 32,
        phase,
        protocol_case.artifact.replay_partition,
        probability,
    )
    draw = exact_rejection_map(
        b"Q" * 32,
        b"huge",
        phase,
        b"candidate",
        denominator,
    )

    assert selected == (0, 1)
    assert 0 <= draw < denominator


def test_challenge_api_rejects_binary_floats(protocol_case):
    with pytest.raises(TypeError):
        derive_q_challenge(
            b"Q" * 32,
            b"P" * 32,
            protocol_case.artifact.replay_partition,
            0.5,  # type: ignore[arg-type]
        )


def test_survival_probability_and_eta_comparison_are_exact_and_strict(
    protocol_case,
):
    replay = protocol_case.artifact.replay_partition
    verification = protocol_case.artifact.verification_partition
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2), Fraction(3, 4))

    probability = two_stage_survival_probability(
        (0,),
        replay,
        verification,
        policy,
    )

    assert probability == Fraction(3, 4)
    assert not survives_acceptance_threshold(probability, policy)
    assert survives_acceptance_threshold(
        probability,
        VerificationPolicy(Fraction(1, 2), Fraction(1, 2), Fraction(2, 3)),
    )
