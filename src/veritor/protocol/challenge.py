"""Sublinear Bernoulli sampling of units from a verifier seed and a phase digest.

Every candidate unit is selected independently with an exact rational
probability ``p = a / b`` -- the joint distribution of one ``p``-coin per unit
-- but the work is proportional to the number of units *selected*, not to the
number of candidates.  The count ``K ~ Binomial(N, p)`` is drawn first, by
inverting the binomial CDF against a 256-bit uniform, and then a uniformly
random ``K``-subset of ``range(N)`` is drawn with Floyd's algorithm from
exactly ``K`` further uniforms.  All randomness is HMAC-SHA256 keyed by the
verifier's seed over the phase digest and a purpose tag, expanded by rejection
sampling, so no floating point is involved and both parties agree bit for bit.
Because the phase digest covers every earlier message, the prover cannot learn
a selection before the values it depends on are fixed.

Exact rational evaluation of the CDF would need ``N * log2(b)`` bits, so it is
evaluated in 512-bit fixed point instead; :func:`_binomial_count` bounds the
resulting bias below ``2**-190`` in total variation for every ``N < 2**64``.
"""

from __future__ import annotations

import hashlib
import hmac
from bisect import bisect_right
from fractions import Fraction
from itertools import accumulate

from veritor.core import (
    Compiled,
    ResourceLimit,
    VerificationLimits,
    VerificationPolicy,
)

from .messages import ProtocolError

_DOMAIN = b"veritor/protocol/binomial-subset-hmac-sha256/v3"
_Q_STAGE = b"q/replay-unit"
_S_STAGE = b"s/verification-unit"
_COUNT_TAG = b"count"
_INDEX_TAG = b"index"
_MAX_REJECTIONS = 4096
_UNIFORM_BITS = 256
"""Bits of the uniform the binomial CDF is inverted against."""
_PRECISION = 512
"""Mantissa bits carried while evaluating the binomial CDF."""


def _uint(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _frame(*parts: bytes) -> bytes:
    out = bytearray()
    for part in parts:
        out.extend(len(part).to_bytes(8, "big"))
        out.extend(part)
    return bytes(out)


def _prf(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    purpose: bytes,
    attempt: int,
    byte_count: int,
) -> bytearray:
    """``byte_count`` pseudorandom bytes for one purpose and rejection attempt."""

    base = _frame(_DOMAIN, stage, phase_digest, purpose)
    material = bytearray()
    for block in range((byte_count + 31) // 32):
        material.extend(
            hmac.new(
                seed, _frame(base, _uint(attempt), _uint(block)), hashlib.sha256
            ).digest()
        )
    del material[byte_count:]
    return material


def _check_seed(seed: bytes) -> None:
    if type(seed) is not bytes or len(seed) != 32:
        raise ProtocolError("challenge seed must be 32 bytes")


def uniform_below(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    candidate: bytes,
    denominator: int,
    limits: VerificationLimits,
) -> int:
    """Map HMAC output uniformly into ``range(denominator)`` by rejection."""

    _check_seed(seed)
    if type(denominator) is not int or denominator <= 0:
        raise ProtocolError("challenge denominator must be positive")
    bits = denominator.bit_length()
    if bits > limits.max_manifest_bytes * 8:
        raise ResourceLimit(
            "challenge_denominator_bits",
            limit=limits.max_manifest_bytes * 8,
            observed=bits,
        )
    if denominator == 1:
        return 0
    byte_count = (bits + 7) // 8
    excess = byte_count * 8 - bits
    for attempt in range(_MAX_REJECTIONS):
        material = _prf(seed, stage, phase_digest, candidate, attempt, byte_count)
        if excess:
            material[0] &= 0xFF >> excess
        drawn = int.from_bytes(material, "big")
        if drawn < denominator:
            return drawn
    raise ResourceLimit(
        "challenge_rejections", limit=_MAX_REJECTIONS, observed=_MAX_REJECTIONS
    )


def _scaled_power(numerator: int, denominator: int, exponent: int) -> tuple[int, int]:
    """``(numerator / denominator) ** exponent`` as ``mantissa / 2**shift``.

    Square-and-multiply on a ``_PRECISION``-bit mantissa; every rounding is a
    floor, so the relative error is below ``(exponent + 4 * exponent.bit_length())
    * 2**-_PRECISION``.  For ``exponent >= 1`` the mantissa has exactly
    ``_PRECISION`` bits.
    """

    base_shift = _PRECISION + denominator.bit_length()
    base = (numerator << base_shift) // denominator
    mantissa, shift = 1, 0
    for bit in bin(exponent)[2:]:
        mantissa, shift = mantissa * mantissa, 2 * shift
        if bit == "1":
            mantissa, shift = mantissa * base, shift + base_shift
        excess = mantissa.bit_length() - _PRECISION
        if excess > 0:
            mantissa, shift = mantissa >> excess, shift - excess
    return mantissa, shift


def _exceeds(total: int, shift: int, uniform: int) -> bool:
    """Whether ``total / 2**shift > uniform / 2**_UNIFORM_BITS``.

    ``total`` has at most ``_PRECISION`` bits, so once ``uniform /
    2**_UNIFORM_BITS`` reaches ``2**(_PRECISION - shift)`` the answer is no
    without shifting ``uniform`` by a possibly enormous ``shift``.
    """

    if uniform and uniform.bit_length() > _PRECISION - shift + _UNIFORM_BITS:
        return False
    return total << _UNIFORM_BITS > uniform << shift


def _binomial_count(uniform: int, count: int, probability: Fraction) -> int:
    """Invert the ``Binomial(count, probability)`` CDF at ``uniform / 2**_UNIFORM_BITS``.

    Requires ``count >= 1`` and ``0 < probability < 1``.  Returns the least
    ``k`` with ``F(k) > u``, walking the pmf up from ``k = 0`` by the exact
    recurrence ``pmf(k + 1) = pmf(k) * (count - k) * a / ((k + 1) * (b - a))``
    in fixed point whose scale ``2**-shift`` follows the running total, which
    always keeps ``_PRECISION`` significant bits.  Each step floors at most one
    unit of the current scale, so for ``count < 2**64`` every ``F(k)`` is
    accurate to better than ``2**-380`` absolute (the initial power contributes
    about ``count * 2**-512`` relative, the recurrence at most ``(k + 1)**2 *
    2**-511`` relative left of the mode and two units per step right of it).
    With the ``2**-256`` granularity of ``u`` the sampled count is therefore
    within ``(count + 1) * (2**-256 + 2**-379) < 2**-190`` of the exact binomial
    in total variation.  Cost is ``O(K)`` operations on ``_PRECISION``-bit
    integers plus ``O(log count)`` for the initial power.
    """

    a, b = probability.numerator, probability.denominator
    term, shift = _scaled_power(b - a, b, count)  # pmf(0) * 2**shift
    total = term  # F(k) * 2**shift, kept at _PRECISION bits
    for k in range(count):
        if _exceeds(total, shift, uniform):
            return k
        term = term * ((count - k) * a) // ((k + 1) * (b - a))
        if term == 0:
            return k + 1  # the remaining tail is below working precision
        total += term
        excess = total.bit_length() - _PRECISION
        if excess > 0:
            total, term, shift = total >> excess, term >> excess, shift - excess
    return count


def _floyd_subset(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    count: int,
    size: int,
    limits: VerificationLimits,
) -> list[int]:
    """A uniform ``size``-subset of ``range(count)`` from exactly ``size`` draws.

    Bentley and Floyd's algorithm: for each ``j`` from ``count - size`` up,
    draw ``t`` uniformly in ``range(j + 1)`` and take ``t`` unless already
    taken, in which case take ``j``.
    """

    chosen: set[int] = set()
    for j in range(count - size, count):
        purpose = _frame(_INDEX_TAG, _uint(j))
        drawn = uniform_below(seed, stage, phase_digest, purpose, j + 1, limits)
        chosen.add(j if drawn in chosen else drawn)
    return sorted(chosen)


def bernoulli_subset(
    seed: bytes,
    stage: bytes,
    phase_digest: bytes,
    count: int,
    probability: Fraction,
    limits: VerificationLimits,
) -> tuple[int, ...]:
    """Select each index in ``range(count)`` independently with ``probability``.

    Returns the sorted selection.  The joint distribution is that of
    independent coins per index (up to the bias bounded in
    :func:`_binomial_count`), yet the cost is ``O(K log count)`` HMAC and
    big-integer operations for ``K`` selected indices plus ``O(log count)``,
    never ``O(count)``.  Deterministic in ``(seed, stage, phase_digest, count,
    probability)``.
    """

    _check_seed(seed)
    if type(count) is not int or count < 0:
        raise ProtocolError("candidate count must be a nonnegative integer")
    if not isinstance(probability, Fraction) or not 0 <= probability <= 1:
        raise ProtocolError("selection probability must be a Fraction in [0, 1]")
    limits.enforce(
        "max_probability_denominator_bits", probability.denominator.bit_length()
    )
    if count == 0 or probability == 0:
        return ()
    if probability == 1:
        return tuple(range(count))
    a, b = probability.numerator, probability.denominator
    purpose = _frame(_COUNT_TAG, _uint(count), _uint(a), _uint(b))
    uniform = int.from_bytes(
        _prf(seed, stage, phase_digest, purpose, 0, _UNIFORM_BITS // 8), "big"
    )
    size = _binomial_count(uniform, count, probability)
    return tuple(_floyd_subset(seed, stage, phase_digest, count, size, limits))


def derive_replay_selection(
    seed: bytes,
    boundary_phase_digest: bytes,
    compiled: Compiled,
    policy: VerificationPolicy,
    limits: VerificationLimits,
) -> tuple[int, ...]:
    """``J``: each replay unit independently with probability ``q``."""

    count = compiled.index.replay_units.count
    limits.enforce("max_units", count)
    return bernoulli_subset(
        seed, _Q_STAGE, boundary_phase_digest, count, policy.q, limits
    )


def derive_sample_selection(
    seed: bytes,
    interior_phase_digest: bytes,
    compiled: Compiled,
    selected_replay_units: tuple[int, ...],
    policy: VerificationPolicy,
    limits: VerificationLimits,
) -> tuple[int, ...]:
    """``T``: each verification unit inside ``J`` independently with probability ``s``.

    The candidates are the verification units of the selected replay units,
    ranked block by block in ``J`` order; only the ``O(|J|)`` block sizes and
    the ``O(|T|)`` sampled ranks are ever touched.
    """

    blocks = [compiled.index.verification_units(unit) for unit in selected_replay_units]
    ends = list(accumulate(block.count for block in blocks))
    count = ends[-1] if ends else 0
    limits.enforce("max_units", count)
    ranks = bernoulli_subset(
        seed, _S_STAGE, interior_phase_digest, count, policy.s, limits
    )

    def unit_at(rank: int) -> int:
        block = bisect_right(ends, rank)
        start = ends[block] - blocks[block].count
        return blocks[block].first + rank - start

    return tuple(sorted(unit_at(rank) for rank in ranks))
