"""Exact algebra for logarithmic gate capacities.

A gate with ``n`` possible values has capacity ``log2(n)``.  Sums of such
capacities can be compared without floating-point arithmetic because

``sum(log2(n_i)) < sum(log2(m_j))`` iff ``prod(n_i) < prod(m_j)``.

``LogCardinality`` stores that product as a positive rational.  Multiplication
of rationals is addition in capacity space, so it also supplies the ordered
abelian-group operations needed by the exact non-integral max-flow backend.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from typing import Self


@dataclass(frozen=True, slots=True)
class GateCapacity:
    """Exact finite alphabet cardinality for one scalar gate."""

    cardinality: int
    expression: str | None = None

    def __post_init__(self) -> None:
        if self.cardinality <= 1:
            raise ValueError("a gate alphabet must contain at least two values")

    @classmethod
    def bits(cls, width_bits: int) -> Self:
        if width_bits <= 0:
            raise ValueError("gate width must be positive")
        return cls(1 << width_bits, str(width_bits))

    @classmethod
    def values(cls, cardinality: int) -> Self:
        return cls(cardinality, f"log2({cardinality})")

    @property
    def width_bits(self) -> float:
        return math.log2(self.cardinality)

    @property
    def integral_width_bits(self) -> int | None:
        if self.cardinality & (self.cardinality - 1):
            return None
        return self.cardinality.bit_length() - 1

    @property
    def display(self) -> str:
        return self.expression or f"log2({self.cardinality})"

    @property
    def log_value(self) -> LogCardinality:
        return LogCardinality(Fraction(self.cardinality))


@total_ordering
@dataclass(frozen=True, slots=True)
class LogCardinality:
    """An exact (possibly signed) sum of log-cardinalities.

    ``multiplier == 1`` is zero bits.  Values below one arise only as residual
    differences inside max flow; gate and cut capacities are always positive.
    """

    multiplier: Fraction

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError("a logarithmic multiplier must be positive")

    @classmethod
    def zero(cls) -> Self:
        return cls(Fraction(1))

    @classmethod
    def bits(cls, width_bits: int) -> Self:
        if width_bits < 0:
            raise ValueError("capacity width cannot be negative")
        return cls(Fraction(1 << width_bits))

    @classmethod
    def cardinality(cls, cardinality: int) -> Self:
        if cardinality <= 0:
            raise ValueError("cardinality multiplier must be positive")
        return cls(Fraction(cardinality))

    @property
    def is_zero(self) -> bool:
        return self.multiplier == 1

    @staticmethod
    def _power_of_two_exponent(value: int) -> int | None:
        if value <= 0 or value & (value - 1):
            return None
        return value.bit_length() - 1

    @property
    def integral_width_bits(self) -> int | None:
        numerator = self._power_of_two_exponent(self.multiplier.numerator)
        denominator = self._power_of_two_exponent(self.multiplier.denominator)
        if numerator is None or denominator is None:
            return None
        return numerator - denominator

    @property
    def width_bits(self) -> int | float:
        integral = self.integral_width_bits
        if integral is not None:
            return integral
        return math.log2(self.multiplier.numerator) - math.log2(
            self.multiplier.denominator
        )

    @property
    def expression(self) -> str:
        integral = self.integral_width_bits
        if integral is not None:
            return str(integral)
        if self.multiplier.denominator == 1:
            return f"log2({self.multiplier.numerator})"
        return f"log2({self.multiplier.numerator})-log2({self.multiplier.denominator})"

    def __add__(self, other: LogCardinality) -> LogCardinality:
        if not isinstance(other, LogCardinality):
            return NotImplemented
        return LogCardinality(self.multiplier * other.multiplier)

    def __sub__(self, other: LogCardinality) -> LogCardinality:
        if not isinstance(other, LogCardinality):
            return NotImplemented
        return LogCardinality(self.multiplier / other.multiplier)

    def scale(self, multiplier: int) -> LogCardinality:
        if multiplier < 0:
            return LogCardinality(Fraction(1, 1) / (self.multiplier ** (-multiplier)))
        return LogCardinality(self.multiplier**multiplier)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LogCardinality):
            return NotImplemented
        return self.multiplier < other.multiplier


def sum_capacities(
    capacities: Iterable[GateCapacity | LogCardinality],
) -> LogCardinality:
    """Return an exact sum for an iterable of gate/log capacities."""

    result = LogCardinality.zero()
    for capacity in capacities:
        if isinstance(capacity, GateCapacity):
            result += capacity.log_value
        elif isinstance(capacity, LogCardinality):
            result += capacity
        else:
            raise TypeError(f"unsupported capacity term: {capacity!r}")
    return result
