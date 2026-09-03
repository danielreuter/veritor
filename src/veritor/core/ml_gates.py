"""A *structural* gate set for real transformer arithmetic: widths and costs, no semantics.

:func:`make_ml_gate_set` declares the scalar gates the structure of a
GPT-2-class decoder needs beyond the toy ISA -- accumulating dot products,
layer norm, softmax, GELU, an argmax -- with the output width and the
replay/proof cost of each, so that a description over it can be compiled,
indexed and priced (``Compile``, ``Index.kinds``, ``Bound``, ``Cost``) and
its bottleneck widths read off the per-kind table.  It has **no executable
semantics**: every gate's evaluator is a stub that raises
``NotImplementedError``.  Whether values are fixed-point words, floats or
something else is an open architectural decision; nothing on the compile,
index or analysis path evaluates a gate, and a circuit over this set can
never be run (``evaluate``/``check`` of every gate raise, so
``Circuit.evaluate`` and the protocol's relation checks fail loudly).

Two widths.  ``width`` is the activation width (16: what a serving profile
calls the FP16 boundary for weights, activations, KV-cache entries, softmax
probabilities, residuals and logits); ``acc_width`` is the accumulator and
reduction width (32: dot-product accumulators, layer norm and softmax
statistics, the internals of ``exp``, ``recip``, ``rsqrt``, ``tanh``).  A
gate's ``width`` is the width of its *output*; argument widths are not
checked structurally (a ``acc_mul`` may read two ``width``-bit values, a
``mul`` may read an ``acc_width``-bit exponential and a reciprocal: what the
model does at that point is a rounding, which is exactly what the explicit
``narrow`` gate stands for where it is the whole operation).  The
comparisons ``lt`` and ``eq`` are one bit wide, so a unit whose interface
is a comparison is charged one bit for it.

Gates.  At ``width``: ``add``, ``sub``, ``mul``, ``max`` (two arguments),
``select(c, a, b)`` (``b`` if ``c`` else ``a``: with ``lt`` this is the
argmax chain), ``narrow(x)`` (round an ``acc_width``-bit value to ``width``
bits: the write-out of a dot product).  At one bit: ``lt``, ``eq``.  At
``acc_width``: ``acc_add``, ``acc_sub``, ``acc_mul``, ``acc_max``, and the
unary ``exp``, ``recip``, ``rsqrt``, ``tanh``.  The sources ``in`` and
``weight`` are ``width`` bits wide, as in :func:`make_isa_gate_set`.

Costs are declared, not measured: ``width``-bit additive gates cost ``1``,
a ``width``-bit multiply ``2``, ``acc_width``-bit gates twice that, and
each transcendental ``16`` (a short polynomial or a table lookup).  They
enter ``Cost`` and the verification-unit (VU) proof cap only.
"""

from __future__ import annotations

from collections.abc import Callable

from .gates import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet

ML_GATE_SET_NAME = "veritor.ml-structural"
ML_GATE_SET_VERSION = "1"
STRUCTURAL_MESSAGE = "structural gate set: no executable semantics"

TRANSCENDENTAL_COST = 16
"""Declared replay and proof cost of ``exp``, ``recip``, ``rsqrt`` and ``tanh``."""


def structural_stub(name: str) -> Callable[[tuple[int, ...]], int]:
    """An evaluator that refuses to run: :class:`Gate` demands a callable, the set has no semantics."""

    def evaluate(_args: tuple[int, ...]) -> int:
        raise NotImplementedError(f"{STRUCTURAL_MESSAGE} ({name})")

    return evaluate


def make_ml_gate_set(width: int = 16, acc_width: int = 32) -> GateSet:
    """The structural ML gate set at activation width ``width`` and accumulator width ``acc_width``.

    See the module docstring for the gates, their widths and their declared
    costs.  ``acc_width`` must be at least ``width``.  Nothing here can be
    evaluated.
    """

    for name, value in (("width", width), ("acc_width", acc_width)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive bit count")
    if acc_width < width:
        raise ValueError("acc_width must be at least width")

    def gate(name: str, arity: int, bits: int, cost: int) -> Gate:
        return Gate(name, arity, bits, replay_cost=cost, proof_cost=cost, evaluate=structural_stub(name))

    return GateSet(
        (
            # activation width
            gate("add", 2, width, 1),
            gate("sub", 2, width, 1),
            gate("mul", 2, width, 2),
            gate("max", 2, width, 1),
            gate("select", 3, width, 1),
            gate("narrow", 1, width, 1),
            # comparisons: one bit
            gate("lt", 2, 1, 1),
            gate("eq", 2, 1, 1),
            # accumulator width
            gate("acc_add", 2, acc_width, 2),
            gate("acc_sub", 2, acc_width, 2),
            gate("acc_mul", 2, acc_width, 4),
            gate("acc_max", 2, acc_width, 2),
            gate("exp", 1, acc_width, TRANSCENDENTAL_COST),
            gate("recip", 1, acc_width, TRANSCENDENTAL_COST),
            gate("rsqrt", 1, acc_width, TRANSCENDENTAL_COST),
            gate("tanh", 1, acc_width, TRANSCENDENTAL_COST),
            # sources
            Gate("in", 0, width, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, width, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=ML_GATE_SET_NAME,
        version=ML_GATE_SET_VERSION,
    )


__all__ = [
    "ML_GATE_SET_NAME",
    "ML_GATE_SET_VERSION",
    "STRUCTURAL_MESSAGE",
    "TRANSCENDENTAL_COST",
    "make_ml_gate_set",
    "structural_stub",
]
