"""The canonical encoding of advice: ``bits`` bits as ``ceil(bits / 8)`` bytes, zero-padded.

The constructor ``G`` declares how many bits its advice ``a`` carries and the
verifier charges exactly that many (``Compilation.advice_bits``, the header's
``advice_bits``).  For the charge to be sound the encoding must be canonical:
``a`` is the ``bits`` bits followed by the ``8 * len(a) - bits < 8`` zero bits
that complete the last byte, so no two byte strings of the declared length
encode the same advice and the padding is not a free channel.  Both parties
check this; a violation is an invalid artifact, never a verdict.
"""

from __future__ import annotations

from .errors import InvalidArtifact


def advice_byte_length(bits: int) -> int:
    """The bytes a canonical ``bits``-bit advice occupies: ``ceil(bits / 8)``."""

    return (bits + 7) // 8


def validate_advice_bits(advice: bytes, bits: object, where: str = "advice") -> int:
    """``bits`` as the declared bit length of ``advice``, once it is canonical.

    Requires a nonnegative integer ``bits`` with ``len(advice) == ceil(bits /
    8)`` and every bit of ``advice`` past the first ``bits`` (the low bits of
    the last byte) zero; raises :class:`InvalidArtifact` otherwise.
    """

    if type(bits) is not int or bits < 0:
        raise InvalidArtifact(f"{where} bit length must be a nonnegative integer")
    if len(advice) != advice_byte_length(bits):
        raise InvalidArtifact(
            f"{where} is {len(advice)} bytes but declares {bits} bits, which take "
            f"{advice_byte_length(bits)}"
        )
    padding = 8 * len(advice) - bits
    if padding and advice[-1] & ((1 << padding) - 1):
        raise InvalidArtifact(
            f"{where} declares {bits} bits but its {padding} padding bits are not zero"
        )
    return bits


__all__ = ["advice_byte_length", "validate_advice_bits"]
