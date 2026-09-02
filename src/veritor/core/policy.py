"""Exact public probability policy for two-stage verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from .identity import Digest, JSONValue, identity_digest

type ProbabilityInput = int | Fraction | Decimal | str


def exact_fraction(value: ProbabilityInput, *, name: str = "value") -> Fraction:
    """Parse an exact rational while rejecting booleans and binary floats."""

    if isinstance(value, (bool, float)):
        raise TypeError(
            f"{name} must be an int, Fraction, Decimal, or decimal/fraction string"
        )
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        return Fraction(value)
    if type(value) is int:
        return Fraction(value)
    if type(value) is str:
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must not be an empty string")
        try:
            parsed = Fraction(text)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"{name} is not a valid decimal or fraction") from error
        return Fraction(parsed.numerator, parsed.denominator)
    raise TypeError(
        f"{name} must be an int, Fraction, Decimal, or decimal/fraction string"
    )


parse_probability = exact_fraction


def rational_manifest(value: Fraction) -> dict[str, JSONValue]:
    """Return the canonical reduced numerator/denominator representation."""

    if not isinstance(value, Fraction):
        raise TypeError("rational_manifest requires a Fraction")
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


def rational_pair(value: Fraction) -> tuple[int, int]:
    """Return ``(numerator, positive_denominator)`` in lowest terms."""

    normalized = Fraction(value)
    return normalized.numerator, normalized.denominator


@dataclass(frozen=True, slots=True, init=False)
class VerificationPolicy:
    """The client's exact two-stage sampling rates ``theta = (q, s)``.

    ``q`` selects replay units and ``s`` selects verification units within
    each selected replay unit.  The acceptance threshold ``eta`` is not part
    of the policy: it belongs to the verifier
    (:class:`veritor.protocol.VerifierParameters`).
    """

    q: Fraction
    s: Fraction
    digest: Digest = field(init=False)

    def __init__(
        self,
        q: ProbabilityInput,
        s: ProbabilityInput,
    ) -> None:
        checked_q = exact_fraction(q, name="q")
        checked_s = exact_fraction(s, name="s")
        if not 0 <= checked_q <= 1:
            raise ValueError("q must lie in [0, 1]")
        if not 0 <= checked_s <= 1:
            raise ValueError("s must lie in [0, 1]")
        object.__setattr__(self, "q", checked_q)
        object.__setattr__(self, "s", checked_s)
        object.__setattr__(
            self,
            "digest",
            identity_digest("veritor/verification-policy/v2", self.manifest),
        )

    @property
    def replay_probability(self) -> Fraction:
        """Descriptive alias for ``q``."""

        return self.q

    @property
    def within_unit_probability(self) -> Fraction:
        """Descriptive alias for ``s``."""

        return self.s

    @property
    def denominator_bits(self) -> int:
        """Bits of the larger denominator among ``q`` and ``s``."""

        return max(rate.denominator.bit_length() for rate in (self.q, self.s))

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "q": rational_manifest(self.q),
            "s": rational_manifest(self.s),
        }
