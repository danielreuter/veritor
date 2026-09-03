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
    arity, in-range relative references, dependency order, limits, a root
    without ports), summarizes every definition once, checks the role marks,
    and builds the lazy circuit and index.  Only the input *count* is checked
    here, against the number of ``in`` gates (values are checked when they
    are encoded); weight values are not a compile input, they enter through
    the protocol's ``Weights``.  Neither is the advice: shaping a description
    by the input or the advice is the constructor's job, done before the
    bytes reach the compiler (:func:`veritor.research.Compile` runs it).
    """

    __slots__ = ("gate_set", "limits")

    def __init__(
        self, gate_set: GateSet, limits: CompilationLimits | None = None
    ) -> None:
        if not isinstance(gate_set, GateSet):
            raise TypeError("Compiler requires a GateSet")
        if limits is not None and not isinstance(limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits")
        self.gate_set = gate_set
        self.limits = CompilationLimits() if limits is None else limits

    def compile(self, description: bytes, inputs: Sequence[int]) -> Compiled:
        parsed = parse_description(description, self.gate_set, self.limits)
        root = parsed.root
        try:
            index = Index(root, self.limits)
            circuit = DescriptionCircuit(root, self.gate_set)
        except InvalidArtifact as error:
            raise CompileError(str(error)) from error
        if len(inputs) != index.input_count:
            raise CompileError(
                f"the circuit expects {index.input_count} inputs, got {len(inputs)}"
            )
        return Compiled(
            circuit, index, Compiled.digest_of(parsed.digest, self.gate_set.digest)
        )
