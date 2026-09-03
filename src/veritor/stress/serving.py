"""Serving one scenario end to end: compile, evaluate, price, and regroup the tokens.

The scenario tests all do the same three things with a constructor and a
workload -- run ``Compile``, evaluate the circuit on the model's weights,
and price the result -- and then compare the tokens request by request with
the reference decoder.  :func:`serve` does the three and
:class:`Served` keeps everything a row needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from veritor.compile.constructor import Constructor
from veritor.core.gates import GateSet
from veritor.core.limits import CompilationLimits
from veritor.core.policy import VerificationPolicy

from .measure import POLICY, Measurement, Price, compile_scenario, evaluate, price

__all__ = ["Served", "by_request", "serve"]

Layout = Sequence[tuple[int, int]]
"""``(request, generated position)`` per circuit output, as the constructors' ``output_layout`` give it."""


def by_request(layout: Layout, outputs: Sequence[int], count: int) -> tuple[tuple[int, ...], ...]:
    """The outputs regrouped as each request's tokens in position order."""

    grouped: list[list[int | None]] = [[] for _ in range(count)]
    for (request, position), token in zip(layout, outputs, strict=True):
        tokens = grouped[request]
        while len(tokens) <= position:
            tokens.append(None)
        tokens[position] = token
    result = []
    for tokens in grouped:
        if any(token is None for token in tokens):
            raise ValueError("the layout leaves a generated position without an output")
        result.append(tuple(token for token in tokens if token is not None))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Served:
    """One scenario served: the compilation, its price, and the tokens by request."""

    measurement: Measurement
    price: Price
    outputs: tuple[int, ...]
    tokens: tuple[tuple[int, ...], ...]

    @property
    def advice_bits(self) -> int:
        return self.measurement.advice_bits

    @property
    def capacity_bits(self) -> int:
        return self.price.capacity_bits

    @property
    def overhead(self) -> float:
        return self.price.overhead

    @property
    def description_bytes(self) -> int:
        return self.measurement.description_bytes

    @property
    def digest(self) -> str:
        """The compiled artifact's digest: two runs with the same circuit share it."""

        return self.measurement.compiled.digest

    def notes(self, *extra: str) -> str:
        """Row notes: the uncapped bound when ``U`` saturates at ``|Out|``, then ``extra``."""

        result = self.price.bound
        parts = list(extra)
        if result.capped:
            parts.insert(0, f"U capped at |Out| = {result.out_bits} bits (uncapped {math.ceil(result.knapsack_bits)} bits)")
        return "; ".join(part for part in parts if part)

    def kinds(self, role: str | None = None) -> dict[str, int]:
        """Kind digest to copy count, for the kinds of ``role`` (every kind when ``None``)."""

        return {
            row.kind: row.copies
            for row in self.measurement.compiled.index.kinds()
            if role is None or row.role == role
        }


def serve(
    constructor: Constructor,
    x: object,
    a: bytes,
    gate_set: GateSet,
    weights: Sequence[int],
    layout: Layout,
    count: int,
    *,
    limits: CompilationLimits | None = None,
    policy: VerificationPolicy = POLICY,
) -> Served:
    """Compile ``G(x, a)``, evaluate it on ``weights``, price it, and regroup its tokens."""

    measurement = compile_scenario(constructor, x, a, gate_set, limits=limits)
    outputs = evaluate(measurement, weights)
    return Served(measurement, price(measurement.compiled, policy), outputs, by_request(layout, outputs, count))
