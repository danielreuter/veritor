"""The compiler: description bytes in, ``Compiled(circuit, index, digest)`` out."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from veritor.core import (
    CompilationLimits,
    Compiled,
    CompiledArtifact,
    DescriptionCircuit,
    GateSet,
    Index,
    InvalidArtifact,
    JSONValue,
)

from .call_dag import CallDagCircuit, Kernel, construct
from .description import CompileError, parse_description
from .partitions import (
    DEFAULT_REPLAY_POLICY,
    DEFAULT_VERIFICATION_POLICY,
    PartitionPolicy,
    compile_partitions_for_policies,
)


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


def compile_call_dag(
    kernel: Kernel,
    constructor: Callable[[object, bytes], bytes],
    x: object,
    a: bytes,
    *,
    input_cells: Sequence[int],
    advice_bound_bits: int,
    replay_policy: PartitionPolicy | str = DEFAULT_REPLAY_POLICY,
    verification_policy: PartitionPolicy | str = DEFAULT_VERIFICATION_POLICY,
    replay_configuration: JSONValue | None = None,
    verification_configuration: JSONValue | None = None,
) -> CompiledArtifact:
    """Run ``G`` and return the compiled ``(C, replay, verification, boundary)``.

    Constructor code is outside the trust boundary.  Its only trusted output
    is the canonical byte document accepted by ``kernel.load``.
    """

    construction = construct(
        kernel,
        constructor,
        x,
        a,
        input_cells=input_cells,
        advice_bound_bits=advice_bound_bits,
    )
    return compile_partitions_for_policies(
        CallDagCircuit(kernel, construction.load.root),
        replay_policy,
        verification_policy,
        replay_configuration=replay_configuration,
        verification_configuration=verification_configuration,
    )
