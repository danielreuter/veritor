"""The compiler: description bytes in, ``Compiled(circuit, index, digest)`` out."""

from __future__ import annotations

from collections.abc import Sequence

from veritor.core import (
    CompilationLimits,
    Compiled,
    DescriptionCircuit,
    GateSet,
    Index,
    InvalidArtifact,
)

from .description import CompileError, parse_description


class Compiler:
    """Trusted compilation of a description against a public gate set.

    ``compile`` parses and validates the description (canonical encoding,
    arity, in-range relative references, dependency order, limits), summarizes
    every definition once, checks the role marks, and builds the lazy circuit
    and index.  Only the input *count* is checked here (values are checked
    when they are encoded); ``advice`` is the prover's untrusted hint,
    admitted only within ``advice_bound_bits``.

    Parametric descriptions (integer parameters bound from the input shape or
    the advice) are a later phase; they would be bound here, before parsing.
    """

    __slots__ = ("gate_set", "limits")

    def __init__(self, gate_set: GateSet, limits: CompilationLimits | None = None) -> None:
        if not isinstance(gate_set, GateSet):
            raise TypeError("Compiler requires a GateSet")
        if limits is not None and not isinstance(limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits")
        self.gate_set = gate_set
        self.limits = CompilationLimits() if limits is None else limits

    def compile(
        self,
        description: bytes,
        inputs: Sequence[int],
        advice: bytes | None = None,
        *,
        advice_bound_bits: int = 0,
    ) -> Compiled:
        if advice is not None:
            if type(advice) is not bytes:
                raise CompileError("advice must be bytes")
            if type(advice_bound_bits) is not int or advice_bound_bits < 0:
                raise CompileError("advice bound must be a nonnegative bit count")
            if len(advice) * 8 > advice_bound_bits:
                raise CompileError("advice exceeds the public bit bound")
        parsed = parse_description(description, self.gate_set, self.limits)
        root = parsed.root
        if len(inputs) != root.input_count:
            raise CompileError(
                f"the circuit expects {root.input_count} inputs, got {len(inputs)}"
            )
        try:
            index = Index(root, self.limits)
            circuit = DescriptionCircuit(root, self.gate_set)
        except InvalidArtifact as error:
            raise CompileError(str(error)) from error
        return Compiled(
            circuit, index, Compiled.digest_of(parsed.digest, self.gate_set.digest)
        )

