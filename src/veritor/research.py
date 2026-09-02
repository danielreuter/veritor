"""The paper-level API: ``Compile`` -> ``Verify`` / ``Bound`` / ``Cost`` / ``Optimize``.

``Compile`` is the trusted half of compilation: it takes the description
bytes an untrusted constructor ``G`` produced (see
:mod:`veritor.constructors`), the public inputs and the advice, and returns
the ``(C, I)`` pair every other function consumes.  ``Verify`` and ``Bound``
are the verifier's; ``Cost`` and ``Optimize`` are the client's advisory
tools.
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
from veritor.compile import Compiler
from veritor.core import (
    CompilationLimits,
    Compiled,
    GateSet,
    ProbabilityInput,
    VerificationLimits,
    VerificationPolicy,
)
from veritor.protocol import (
    Expectation,
    ProtocolRun,
    VerificationCode,
    VerificationReport,
    VerifierParameters,
    Weights,
    encode_transcript,
    make_expectation,
    run_protocol,
    verify_transcript,
)

DEFAULT_CONFORMANCE_POLICY = VerificationPolicy(1, 1, 0)


def Compile(
    description: bytes,
    inputs: Sequence[int],
    gate_set: GateSet,
    *,
    advice: bytes | None = None,
    advice_bound_bits: int = 0,
    limits: CompilationLimits | None = None,
) -> Compiled:
    """``Compile(G, x, a) -> (C, I)`` for description bytes already produced by ``G``."""

    return Compiler(gate_set, limits).compile(
        description, inputs, advice, advice_bound_bits=advice_bound_bits
    )


def _compiled(value: object) -> Compiled:
    if not isinstance(value, Compiled):
        raise TypeError("expected a Compiled (C, I) from Compile")
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
    options: BoundOptions | None = None,
) -> BoundResult:
    """``U = Bound(C, I, theta)``: see :mod:`veritor.analysis.bound`."""

    return bound(_compiled(compiled), policy, options)


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
    compiled: Compiled,
    policy: VerificationPolicy,
    public_inputs: Sequence[object],
    claimed_outputs: Sequence[object],
    *,
    parameters: VerifierParameters | None = None,
    weights: Weights | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
) -> Expectation:
    """The verifier's side of one run: the client's ``theta`` admitted under ``parameters``.

    Seeds come from the CSPRNG unless given; ``weights`` is the model's
    pre-committed weight root, if any.
    """

    return make_expectation(
        _compiled(compiled),
        policy,
        public_inputs,
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
    compiled: Compiled,
    public_inputs: Sequence[int],
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
) -> ExecutableConformanceTranscript:
    """Run an honest prover against the verifier in one process.

    The circuit is evaluated on ``public_inputs``, the claimed outputs are
    read from that evaluation, and both protocol parties run locally via
    :func:`run_protocol`.  This cannot demonstrate that either seed was
    withheld until the message it depends on was fixed; it is a conformance
    fixture for :func:`Verify`.
    """

    compiled = _compiled(compiled)
    values = compiled.circuit.evaluate(public_inputs)
    outputs = tuple(values[address] for address in compiled.circuit.outputs)
    expectation = make_expectation(
        compiled,
        policy,
        public_inputs,
        outputs,
        session_id=session_id,
        q_seed=q_seed,
        s_seed=s_seed,
    )
    run = run_protocol(compiled, expectation, dict(enumerate(values)), limits=limits)
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
    "CompilationLimits",
    "Compile",
    "Compiled",
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
