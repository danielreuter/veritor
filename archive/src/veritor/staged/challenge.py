"""Exact deterministic q-then-s challenge derivation."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from fractions import Fraction

from veritor.core import (
    ReplayPartition,
    ResourceLimit,
    VerificationLimits,
    VerificationPartition,
    VerificationPolicy,
)

from .model import StagedProtocolError

_HMAC_DOMAIN = b"veritor/staged/exact-bernoulli-hmac-sha256/v1"
_Q_STAGE = b"q/replay-unit"
_S_STAGE = b"s/verification-unit"
_MAX_REJECTION_ATTEMPTS = 4096


def _bytes32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise StagedProtocolError(f"{name} must be exactly 32 bytes")
    return value


def _uint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise StagedProtocolError("challenge integers must be nonnegative")
    width = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(width, "big")


def _frame(*parts: bytes) -> bytes:
    framed = bytearray()
    for part in parts:
        if type(part) is not bytes:
            raise TypeError("challenge frame parts must be bytes")
        framed.extend(len(part).to_bytes(8, "big"))
        framed.extend(part)
    return bytes(framed)


def _exact_probability(value: object, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if type(value) is int:
        return Fraction(value)
    raise TypeError(f"{name} must be an exact int or Fraction")


def exact_rejection_map(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    candidate: bytes,
    denominator: int,
    limits: VerificationLimits | None = None,
) -> int:
    """Map HMAC-SHA256 output uniformly into ``range(denominator)``.

    Arbitrarily large rational denominators are supported by expanding enough
    independently domain-separated HMAC blocks, masking to the denominator's
    bit length, and rejecting the tail.  No binary floating point is used.
    """

    checked_limits = VerificationLimits() if limits is None else limits
    _bytes32(seed, "phase seed")
    _bytes32(phase_digest, "phase_digest")
    if type(stage) is not bytes or not stage:
        raise StagedProtocolError("challenge stage must be nonempty bytes")
    if type(candidate) is not bytes or not candidate:
        raise StagedProtocolError("challenge candidate must be nonempty bytes")
    if type(denominator) is not int or denominator <= 0:
        raise StagedProtocolError("challenge denominator must be positive")
    denominator_bits = denominator.bit_length()
    # The transcript byte bound is also a conservative arithmetic-work bound.
    if denominator_bits > checked_limits.max_manifest_bytes * 8:
        raise ResourceLimit(
            "challenge_denominator_bits",
            limit=checked_limits.max_manifest_bytes * 8,
            observed=denominator_bits,
        )
    if denominator == 1:
        return 0

    byte_count = (denominator_bits + 7) // 8
    block_count = (byte_count + hashlib.sha256().digest_size - 1) // 32
    excess_bits = byte_count * 8 - denominator_bits
    base = _frame(_HMAC_DOMAIN, stage, phase_digest, candidate)

    for attempt in range(_MAX_REJECTION_ATTEMPTS):
        material = bytearray()
        for block_index in range(block_count):
            message = _frame(base, _uint(attempt), _uint(block_index))
            material.extend(hmac.new(seed, message, hashlib.sha256).digest())
        material = material[:byte_count]
        if excess_bits:
            material[0] &= 0xFF >> excess_bits
        mapped = int.from_bytes(material, "big")
        if mapped < denominator:
            return mapped
    raise ResourceLimit(
        "challenge_rejections",
        limit=_MAX_REJECTION_ATTEMPTS,
        observed=_MAX_REJECTION_ATTEMPTS,
    )


def _selected(
    *,
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    probability: Fraction,
    candidates: Iterable[tuple[int, str]],
    limits: VerificationLimits,
) -> tuple[int, ...]:
    normalized = _exact_probability(probability, "challenge probability")
    if not 0 <= normalized <= 1:
        raise StagedProtocolError("challenge probability must lie in [0, 1]")
    candidate_tuple = tuple(candidates)
    if 0 < normalized.numerator < normalized.denominator:
        bytes_per_draw = (normalized.denominator.bit_length() + 7) // 8
        estimated_bytes = bytes_per_draw * len(candidate_tuple)
        if estimated_bytes > limits.max_proof_bytes:
            raise ResourceLimit(
                "challenge_expansion_bytes",
                limit=limits.max_proof_bytes,
                observed=estimated_bytes,
            )
    result: list[int] = []
    for index, identity in candidate_tuple:
        if normalized.numerator == 0:
            continue
        if normalized.numerator == normalized.denominator:
            result.append(index)
            continue
        candidate = _frame(
            _uint(index),
            bytes.fromhex(identity),
        )
        draw = exact_rejection_map(
            seed,
            stage,
            phase_digest,
            candidate,
            normalized.denominator,
            limits,
        )
        if draw < normalized.numerator:
            result.append(index)
    return tuple(result)


def derive_q_challenge(
    seed: bytes,
    boundary_phase_digest: bytes,
    replay_partition: ReplayPartition,
    probability: Fraction | int | VerificationPolicy,
    limits: VerificationLimits | None = None,
) -> tuple[int, ...]:
    """Select each replay unit independently with exact probability ``q``."""

    checked_limits = VerificationLimits() if limits is None else limits
    _bytes32(seed, "q seed")
    _bytes32(boundary_phase_digest, "boundary_phase_digest")
    if not isinstance(replay_partition, ReplayPartition):
        raise StagedProtocolError("q challenge requires a ReplayPartition")
    checked_limits.enforce("max_units", replay_partition.unit_count)
    q = probability.q if isinstance(probability, VerificationPolicy) else probability
    return _selected(
        seed=seed,
        stage=_Q_STAGE,
        phase_digest=boundary_phase_digest,
        probability=_exact_probability(q, "q"),
        candidates=(
            (int(unit.index), str(unit.identity_digest))
            for unit in replay_partition.units
        ),
        limits=checked_limits,
    )


def derive_s_challenge(
    seed: bytes,
    unit_commitments_phase_digest: bytes,
    verification_partition: VerificationPartition,
    selected_replay_units: tuple[int, ...],
    probability: Fraction | int | VerificationPolicy,
    limits: VerificationLimits | None = None,
) -> tuple[int, ...]:
    """Select verification units inside J with exact probability ``s``."""

    checked_limits = VerificationLimits() if limits is None else limits
    _bytes32(seed, "s seed")
    _bytes32(
        unit_commitments_phase_digest,
        "unit_commitments_phase_digest",
    )
    if not isinstance(verification_partition, VerificationPartition):
        raise StagedProtocolError("s challenge requires a VerificationPartition")
    if type(selected_replay_units) is not tuple or any(
        type(index) is not int or index < 0 for index in selected_replay_units
    ):
        raise StagedProtocolError(
            "selected replay units must be a tuple of nonnegative indices"
        )
    if tuple(sorted(set(selected_replay_units))) != selected_replay_units:
        raise StagedProtocolError("selected replay units must be sorted and unique")
    checked_limits.enforce("max_units", verification_partition.unit_count)
    selected = set(selected_replay_units)
    s = probability.s if isinstance(probability, VerificationPolicy) else probability
    return _selected(
        seed=seed,
        stage=_S_STAGE,
        phase_digest=unit_commitments_phase_digest,
        probability=_exact_probability(s, "s"),
        candidates=(
            (int(unit.index), str(unit.identity_digest))
            for unit in verification_partition.units
            if int(unit.replay_unit) in selected
        ),
        limits=checked_limits,
    )


derive_replay_challenge = derive_q_challenge
derive_verification_challenge = derive_s_challenge


def two_stage_survival_probability(
    incorrect_verification_units: Iterable[int],
    replay_partition: ReplayPartition,
    verification_partition: VerificationPartition,
    policy: VerificationPolicy,
) -> Fraction:
    """Return the exact two-stage survival probability for an error set."""

    if not isinstance(replay_partition, ReplayPartition) or not isinstance(
        verification_partition,
        VerificationPartition,
    ):
        raise StagedProtocolError("survival probability needs both partitions")
    if not isinstance(policy, VerificationPolicy):
        raise StagedProtocolError("survival probability needs a policy")
    verification_partition.validate(replay_partition)
    errors = tuple(incorrect_verification_units)
    if any(type(index) is not int or index < 0 for index in errors):
        raise StagedProtocolError("incorrect unit indices must be nonnegative")
    if tuple(sorted(set(errors))) != errors:
        raise StagedProtocolError("incorrect unit indices must be sorted and unique")
    counts = [0] * replay_partition.unit_count
    for index in errors:
        if index >= verification_partition.unit_count:
            raise StagedProtocolError("incorrect unit index is out of range")
        counts[verification_partition.unit_at(index).replay_unit] += 1
    result = Fraction(1)
    for count in counts:
        result *= 1 - policy.q + policy.q * (1 - policy.s) ** count
    return result


def survives_acceptance_threshold(
    probability: Fraction | int,
    policy: VerificationPolicy,
) -> bool:
    """Apply the protocol's strict ``p > eta`` convention."""

    return _exact_probability(probability, "probability") > policy.eta
