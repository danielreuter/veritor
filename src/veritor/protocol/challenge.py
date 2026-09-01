"""Exact Bernoulli sampling of units from a verifier seed and a phase digest.

Each unit is selected independently with an exact rational probability.
Randomness is HMAC-SHA256 keyed by the verifier's seed over the phase digest
and the unit's identity, expanded with rejection sampling so no floating point
is involved.  Because the phase digest covers every earlier message, the
prover cannot learn a selection before the values it depends on are fixed.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from fractions import Fraction

from veritor.core import (
    CompiledArtifact,
    ResourceLimit,
    VerificationLimits,
    VerificationPolicy,
)

from .messages import ProtocolError

_DOMAIN = b"veritor/protocol/exact-bernoulli-hmac-sha256/v2"
_Q_STAGE = b"q/replay-unit"
_S_STAGE = b"s/verification-unit"
_MAX_REJECTIONS = 4096


def _uint(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _frame(*parts: bytes) -> bytes:
    out = bytearray()
    for part in parts:
        out.extend(len(part).to_bytes(8, "big"))
        out.extend(part)
    return bytes(out)


def uniform_below(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    candidate: bytes,
    denominator: int,
    limits: VerificationLimits,
) -> int:
    """Map HMAC output uniformly into ``range(denominator)`` by rejection."""

    if type(seed) is not bytes or len(seed) != 32:
        raise ProtocolError("challenge seed must be 32 bytes")
    if type(denominator) is not int or denominator <= 0:
        raise ProtocolError("challenge denominator must be positive")
    bits = denominator.bit_length()
    if bits > limits.max_manifest_bytes * 8:
        raise ResourceLimit(
            "challenge_denominator_bits", limit=limits.max_manifest_bytes * 8, observed=bits
        )
    if denominator == 1:
        return 0
    byte_count = (bits + 7) // 8
    block_count = (byte_count + 31) // 32
    excess = byte_count * 8 - bits
    base = _frame(_DOMAIN, stage, phase_digest, candidate)
    for attempt in range(_MAX_REJECTIONS):
        material = bytearray()
        for block in range(block_count):
            material.extend(
                hmac.new(seed, _frame(base, _uint(attempt), _uint(block)), hashlib.sha256)
                .digest()
            )
        del material[byte_count:]
        if excess:
            material[0] &= 0xFF >> excess
        drawn = int.from_bytes(material, "big")
        if drawn < denominator:
            return drawn
    raise ResourceLimit(
        "challenge_rejections", limit=_MAX_REJECTIONS, observed=_MAX_REJECTIONS
    )


def _select(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    probability: Fraction,
    candidates: Iterable[tuple[int, str]],
    limits: VerificationLimits,
) -> tuple[int, ...]:
    if probability == 0:
        return ()
    if probability == 1:
        return tuple(index for index, _ in candidates)
    selected: list[int] = []
    for index, identity in candidates:
        drawn = uniform_below(
            seed,
            stage,
            phase_digest,
            _frame(_uint(index), bytes.fromhex(identity)),
            probability.denominator,
            limits,
        )
        if drawn < probability.numerator:
            selected.append(index)
    return tuple(selected)


def derive_replay_selection(
    seed: bytes,
    boundary_phase_digest: bytes,
    artifact: CompiledArtifact,
    policy: VerificationPolicy,
    limits: VerificationLimits,
) -> tuple[int, ...]:
    """``J``: each replay unit independently with probability ``q``."""

    replay = artifact.replay
    limits.enforce("max_units", replay.unit_count)
    return _select(
        seed,
        _Q_STAGE,
        boundary_phase_digest,
        policy.q,
        ((int(unit.index), str(unit.identity_digest)) for unit in replay.units),
        limits,
    )


def derive_sample_selection(
    seed: bytes,
    interior_phase_digest: bytes,
    artifact: CompiledArtifact,
    selected_replay_units: tuple[int, ...],
    policy: VerificationPolicy,
    limits: VerificationLimits,
) -> tuple[int, ...]:
    """``T``: each verification unit inside ``J`` independently with probability ``s``."""

    verification = artifact.verification
    candidates = [
        index
        for replay_unit in selected_replay_units
        for index in verification.units_in_replay_unit(replay_unit)
    ]
    candidates.sort()
    limits.enforce("max_units", len(candidates))
    return _select(
        seed,
        _S_STAGE,
        interior_phase_digest,
        policy.s,
        ((index, str(verification.unit_at(index).identity_digest)) for index in candidates),
        limits,
    )
