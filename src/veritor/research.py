"""The paper-level API: ``Compile`` -> ``Verify`` / ``Bound`` / ``Cost`` / ``Optimize``.

``Compile(G, x, a)`` is the verifier's: it runs the client's constructor
``G`` (see :mod:`veritor.constructors` for the built-in ones) on the public
inputs ``x`` and the advice ``a``, compiles the description it produces and
records the result as a :class:`Compilation`, the ``(C, I)`` every other
function consumes together with what it was compiled from.  ``Verify``,
``Bound`` and ``Capacity`` are the verifier's; ``Cost`` and ``Optimize`` are
the client's advisory tools.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from veritor.analysis import (
    BoundOptions,
    BoundResult,
    CostParameters,
    ExpectedCost,
    Optimization,
    PolicyGrid,
    bound,
    cost,
    optimize,
)
from veritor.compile import Compilation, CompileError, Compiler, Constructor
from veritor.core import (
    CompilationLimits,
    Compiled,
    GateSet,
    InvalidArtifact,
    ProbabilityInput,
    VerificationLimits,
    VerificationPolicy,
    validate_digest,
)
from veritor.protocol import (
    Expectation,
    ProtocolRun,
    VerificationCode,
    VerificationReport,
    VerifierParameters,
    Weights,
    commit_weights,
    encode_transcript,
    make_expectation,
    run_protocol,
    verify_transcript,
)

DEFAULT_CONFORMANCE_POLICY = VerificationPolicy(1, 1)


def Compile(
    G: Constructor,
    x: object,
    a: bytes,
    gate_set: GateSet,
    *,
    limits: CompilationLimits | None = None,
    max_advice_bits: int = 0,
) -> Compilation:
    """``Compile(G, x, a) -> (C, I)``: run the client's constructor and compile its output.

    ``G`` is the client's constructor, ``x`` the request's public inputs and
    ``a`` the client's advice, charged at ``8 * len(a)`` bits and admitted
    only up to ``max_advice_bits``.  ``G(x, a)`` returns the description
    bytes and the flat circuit inputs (the ``in`` gates' values in address
    order); the description is compiled against ``gate_set`` under ``limits``
    by :class:`Compiler`.  Anything that goes wrong in ``G`` -- an exception,
    a malformed return -- is a :class:`CompileError`: the client's constructor
    failed, which is a rejection, never a crash.

    Trust model.  In this prototype the verifier executes ``G`` as ordinary
    Python identified by a versioned digest: like the gate set, ``G`` is
    public code both parties hold, and the header binds ``G.digest`` and
    ``a`` so that everything about the run beyond the advice is a
    deterministic function of ``(G, x, a)``.  A deployment would run ``G``
    sandboxed and metered, or have the client prove ``Compile(G, x, a) = (C,
    I)`` (paper §7).  The description-size and compilation limits bound the
    *output* of ``G``; nothing here bounds ``G``'s running time.
    """

    if not isinstance(G, Constructor):
        raise TypeError("expected a Constructor: an object with a digest, called as G(x, a)")
    if type(max_advice_bits) is not int or max_advice_bits < 0:
        raise ValueError("max_advice_bits must be a nonnegative integer")
    if type(a) is not bytes:
        raise CompileError("advice must be bytes")
    if 8 * len(a) > max_advice_bits:
        raise CompileError("advice exceeds the public bit bound")
    try:
        constructor = validate_digest(G.digest, "constructor digest")
    except InvalidArtifact as error:
        raise CompileError(str(error)) from error
    try:
        produced = G(x, a)
    except Exception as error:
        raise CompileError(f"the constructor failed: {error}") from error
    if type(produced) is not tuple or len(produced) != 2:
        raise CompileError("a constructor returns (description, inputs)")
    description, inputs = produced
    if type(description) is not bytes:
        raise CompileError("the constructor's description must be bytes")
    if type(inputs) is not tuple or any(type(value) is not int for value in inputs):
        raise CompileError("the constructor's inputs must be a tuple of integers")
    compiled = Compiler(gate_set, limits).compile(description, inputs)
    return Compilation(compiled, constructor, inputs, a)


def _compiled(value: object) -> Compiled:
    if not isinstance(value, Compiled):
        raise TypeError("expected a Compiled (C, I) from Compile")
    return value


def _compilation(value: object) -> Compilation:
    if not isinstance(value, Compilation):
        raise TypeError("expected a Compilation from Compile")
    return value


def Verify(
    transcript_bytes: bytes,
    expectation: Expectation,
    compiled: Compiled,
    *,
    limits: VerificationLimits | None = None,
) -> VerificationReport:
    """Purely verify canonical transcript bytes against a trusted ``(C, I)``.

    ``expectation`` carries both 32-byte verifier seeds; no transcript field
    is ever treated as their source of truth.
    """

    return verify_transcript(transcript_bytes, expectation, _compiled(compiled), limits)


def Bound(
    compiled: Compiled,
    policy: VerificationPolicy,
    eta: ProbabilityInput,
    options: BoundOptions | None = None,
) -> BoundResult:
    """``U = Bound(C, I, theta)`` at the verifier's ``eta``: see :mod:`veritor.analysis.bound`."""

    return bound(_compiled(compiled), policy, eta, options)


def Capacity(
    compilation: Compilation,
    policy: VerificationPolicy,
    eta: ProbabilityInput,
    options: BoundOptions | None = None,
) -> float:
    """The per-request capacity the paper charges: ``Bound(C, I, theta) + |a|`` bits.

    Beyond the degrees of freedom ``Bound`` leaves uncharged in the circuit,
    the only freedom the client has is the advice; everything else is a
    deterministic function of ``(G, x, a)``.
    """

    checked = _compilation(compilation)
    return bound(checked.compiled, policy, eta, options).bits + checked.advice_bits


def Cost(
    compiled: Compiled,
    policy: VerificationPolicy,
    parameters: CostParameters | None = None,
) -> ExpectedCost:
    """``Cost(C, I, theta)``: see :mod:`veritor.analysis.cost`."""

    return cost(_compiled(compiled), policy, parameters)


def Optimize(
    compiled: Compiled,
    eta: ProbabilityInput,
    grid: PolicyGrid,
    *,
    max_bits: float | None = None,
    max_cost: ProbabilityInput | None = None,
    parameters: CostParameters | None = None,
    bound_options: BoundOptions | None = None,
    accept: Callable[[VerificationPolicy], bool] | None = None,
) -> Optimization | None:
    """The client's advisory search for ``theta``: see :mod:`veritor.analysis.optimize`."""

    return optimize(
        _compiled(compiled),
        eta,
        grid,
        max_bits=max_bits,
        max_cost=max_cost,
        parameters=parameters,
        bound_options=bound_options,
        accept=accept,
    )


def make_verification_expectation(
    compilation: Compilation,
    policy: VerificationPolicy,
    claimed_outputs: Sequence[object],
    *,
    parameters: VerifierParameters | None = None,
    weights: Weights | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
) -> Expectation:
    """The verifier's side of one run: ``Compile(G, x, a)``, ``y*`` and the client's ``theta``.

    ``compilation`` carries ``(C, I)``, ``G``'s digest, the public inputs and
    the advice the header binds; ``theta`` is admitted under ``parameters``.
    Seeds come from the CSPRNG unless given; ``weights`` is the model's
    pre-committed weight root, if any.
    """

    return make_expectation(
        _compilation(compilation),
        policy,
        claimed_outputs,
        parameters=parameters,
        weights=weights,
        session_id=session_id,
        q_seed=q_seed,
        s_seed=s_seed,
    )


@dataclass(frozen=True, slots=True)
class ExecutableConformanceTranscript:
    """One-shot honest fixture, not evidence of interactive phase ordering."""

    transcript_bytes: bytes
    expectation: Expectation


def build_executable_conformance_transcript(
    compilation: Compilation,
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    weights: Sequence[int] = (),
    parameters: VerifierParameters | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
) -> ExecutableConformanceTranscript:
    """Run an honest prover against the verifier in one process.

    The circuit is evaluated on the compilation's inputs and on ``weights``
    (the values of its ``weight`` gates by rank), the claimed outputs are read
    from that evaluation, the weights are committed under ``kappa_W`` when the
    circuit has any, and both protocol parties run locally via
    :func:`run_protocol`.  This cannot demonstrate that either seed was
    withheld until the message it depends on was fixed; it is a conformance
    fixture for :func:`Verify`.
    """

    compiled = _compilation(compilation).compiled
    values = compiled.circuit.evaluate(compilation.inputs, weights)
    outputs = tuple(values[address] for address in compiled.circuit.outputs)
    bound_weights, weight_tree = (
        commit_weights(compiled.circuit.gate_set, weights)
        if compiled.index.weight_count
        else (None, None)
    )
    expectation = make_expectation(
        compilation,
        policy,
        outputs,
        parameters=parameters,
        weights=bound_weights,
        session_id=session_id,
        q_seed=q_seed,
        s_seed=s_seed,
    )
    run = run_protocol(
        compiled, expectation, dict(enumerate(values)), limits=limits, weight_tree=weight_tree
    )
    if run.transcript is None:
        raise RuntimeError(
            f"honest conformance run was rejected: {run.report.code.value}: "
            f"{run.report.detail}"
        )
    return ExecutableConformanceTranscript(encode_transcript(run.transcript), expectation)


__all__ = [
    "DEFAULT_CONFORMANCE_POLICY",
    "Bound",
    "BoundOptions",
    "BoundResult",
    "Capacity",
    "Compilation",
    "CompilationLimits",
    "Compile",
    "Compiled",
    "Constructor",
    "Cost",
    "CostParameters",
    "ExecutableConformanceTranscript",
    "Expectation",
    "ExpectedCost",
    "GateSet",
    "Optimization",
    "Optimize",
    "PolicyGrid",
    "ProtocolRun",
    "VerificationCode",
    "VerificationLimits",
    "VerificationPolicy",
    "VerificationReport",
    "VerifierParameters",
    "Verify",
    "Weights",
    "build_executable_conformance_transcript",
    "make_verification_expectation",
    "run_protocol",
]
