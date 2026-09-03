"""Pricing a scenario: compile it, bound it, cost it, in the units of the catalogue.

A row of ``docs/stress-tests.md`` reports the advice ``|a|`` in bits, the
capacity ``U = Bound(C, I, theta)`` at ``eta = 2^-40`` as an integer count of
bits, the prover's overhead ``Cost(...).total / honest replay cost`` at the
simulation's policy, and the size of the description ``G(x, a)`` in bytes.
:func:`compile_scenario` runs the constructor and the compiler separately so
that both halves can be timed; :func:`price` folds the two analyses.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from veritor.analysis.bound import BoundOptions, BoundResult, bound
from veritor.analysis.cost import ExpectedCost, cost
from veritor.compile.compiler import Compiler
from veritor.compile.constructor import Compilation, Constructor
from veritor.core.compiled import Compiled
from veritor.core.gates import GateSet
from veritor.core.index import KindTable
from veritor.core.limits import CompilationLimits
from veritor.core.policy import VerificationPolicy

__all__ = [
    "ETA",
    "POLICY",
    "Measurement",
    "Price",
    "compile_scenario",
    "evaluate",
    "honest_cost",
    "price",
]

ETA = Fraction(1, 2**40)
"""The catalogue's threshold: ``lambda = 40``."""

POLICY = VerificationPolicy(Fraction(1, 2), Fraction(1, 8))
"""The simulated datacenter's policy ``theta = (q, s)``, used for every overhead."""


@dataclass(frozen=True, slots=True)
class Measurement:
    """``Compile(G, x, a)`` with the description it consumed and how long each half took."""

    compilation: Compilation
    description: bytes
    trace_seconds: float
    compile_seconds: float

    @property
    def compiled(self) -> Compiled:
        return self.compilation.compiled

    @property
    def advice_bits(self) -> int:
        return self.compilation.advice_bits

    @property
    def description_bytes(self) -> int:
        return len(self.description)


def compile_scenario(
    constructor: Constructor,
    x: object,
    a: bytes,
    gate_set: GateSet,
    *,
    limits: CompilationLimits | None = None,
) -> Measurement:
    """What ``Compile`` does, keeping the bytes of ``G(x, a)`` and timing its two halves."""

    started = time.perf_counter()
    description, inputs = constructor(x, a)
    traced = time.perf_counter()
    compiled = Compiler(gate_set, limits).compile(description, inputs)
    finished = time.perf_counter()
    compilation = Compilation(compiled, constructor.digest, inputs, a)
    return Measurement(compilation, description, traced - started, finished - traced)


def evaluate(measurement: Measurement, weights: Sequence[int]) -> tuple[int, ...]:
    """The circuit's outputs on the compiled inputs and ``weights``."""

    circuit = measurement.compiled.circuit
    values = circuit.evaluate(measurement.compilation.inputs, weights)
    return tuple(values[address] for address in circuit.outputs)


def honest_cost(table: KindTable) -> int:
    """The replay cost of the whole circuit: the honest computation in the cost's units."""

    return next(row.replay_cost for row in table.rows if row.kind == table.root)


@dataclass(frozen=True, slots=True)
class Price:
    """``U`` and the prover's overhead of one compiled circuit under one policy."""

    bound: BoundResult
    cost: ExpectedCost
    honest: int
    bound_seconds: float

    @property
    def capacity_bits(self) -> int:
        """``U`` as the catalogue reports it: a whole number of bits."""

        return math.ceil(self.bound.bits)

    @property
    def overhead(self) -> float:
        return float(self.cost.total / self.honest)


def price(
    target: Compiled | KindTable,
    policy: VerificationPolicy = POLICY,
    eta: Fraction = ETA,
    options: BoundOptions | None = None,
) -> Price:
    """Bound and cost ``target`` (an artifact or its kind table) under ``policy``."""

    table = target if isinstance(target, KindTable) else target.kind_table()
    started = time.perf_counter()
    result = bound(table, policy, eta, options)
    seconds = time.perf_counter() - started
    return Price(result, cost(table, policy), honest_cost(table), seconds)
