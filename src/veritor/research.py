"""Paper-level research facade: ``Compile`` -> ``Verify`` / ``Bound`` / ``Cost`` / ``Optimize``.

DemoG and matmul compile to :class:`~veritor.core.Compiled`, the executable
``(C, I)`` the protocol verifies and the folds analyse.  The LLM plug-ins
carry model configurations only; until a constructor writes their
descriptions, ``Compile`` returns :class:`~veritor.core.Unsupported` and so
does anything asked of that result.
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
from veritor.core import (
    Capability,
    Compiled,
    ProbabilityInput,
    Unsupported,
    VerificationLimits,
    VerificationPolicy,
)
from veritor.plugins import (
    ArchitectureCompileRequest,
    ArchitectureId,
    CompileResult,
    DeepSeekV4ProCompileRequest,
    DemoGCompileRequest,
    GPT2CompileRequest,
    GreedyTextExecutionShape,
    InklingCompileRequest,
    KimiK3CompileRequest,
    MatmulCompileRequest,
    compile_architecture,
)
from veritor.protocol import (
    Expectation,
    ProtocolRun,
    VerificationCode,
    VerificationReport,
    encode_transcript,
    make_expectation,
    run_protocol,
    verify_transcript,
)

DEFAULT_CONFORMANCE_POLICY = VerificationPolicy(1, 1, 0)


def Compile(
    architecture_id: ArchitectureId | str,
    request: ArchitectureCompileRequest | None = None,
) -> CompileResult:
    """Compile one registered architecture or executable workload."""

    return compile_architecture(architecture_id, request)


def _executable(artifact: CompileResult, capability: Capability) -> Compiled | Unsupported:
    if isinstance(artifact, Compiled):
        return artifact
    if isinstance(artifact, Unsupported):
        return Unsupported(
            capability=capability,
            plugin_id=artifact.plugin_id,
            reason_code=artifact.reason_code,
            detail=f"no compiled description: {artifact.detail}",
            artifact_kind=artifact.artifact_kind,
        )
    raise TypeError("artifact must be a Compile result")


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

    return verify_transcript(transcript_bytes, expectation, compiled, limits)


def Bound(
    artifact: CompileResult,
    policy: VerificationPolicy,
    options: BoundOptions | None = None,
) -> BoundResult | Unsupported:
    """``U = Bound(C, I, theta)``: see :mod:`veritor.analysis.bound`."""

    compiled = _executable(artifact, Capability.STATIC_BOUND)
    if isinstance(compiled, Unsupported):
        return compiled
    return bound(compiled, policy, options)


def Cost(
    artifact: CompileResult,
    policy: VerificationPolicy,
    parameters: CostParameters | None = None,
) -> ExpectedCost | Unsupported:
    """``Cost(C, I, theta)``: see :mod:`veritor.analysis.cost`."""

    compiled = _executable(artifact, Capability.STATIC_BOUND)
    if isinstance(compiled, Unsupported):
        return compiled
    return cost(compiled, policy, parameters)


def Optimize(
    artifact: CompileResult,
    eta: ProbabilityInput,
    grid: PolicyGrid,
    *,
    max_bits: float | None = None,
    max_cost: ProbabilityInput | None = None,
    parameters: CostParameters | None = None,
    bound_options: BoundOptions | None = None,
    accept: Callable[[VerificationPolicy], bool] | None = None,
) -> Optimization | None | Unsupported:
    """The client's advisory search for ``theta``: see :mod:`veritor.analysis.optimize`."""

    compiled = _executable(artifact, Capability.STATIC_BOUND)
    if isinstance(compiled, Unsupported):
        return compiled
    return optimize(
        compiled,
        eta,
        grid,
        max_bits=max_bits,
        max_cost=max_cost,
        parameters=parameters,
        bound_options=bound_options,
        accept=accept,
    )


def make_verification_expectation(
    artifact: CompileResult,
    policy: VerificationPolicy,
    public_inputs: Sequence[object],
    claimed_outputs: Sequence[object],
    *,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
) -> Expectation | Unsupported:
    """Build a verifier-local expectation for an executable compile result.

    Seeds are drawn from the CSPRNG unless given.
    """

    executable = _executable(artifact, Capability.VERIFY)
    if isinstance(executable, Unsupported):
        return executable
    return make_expectation(
        executable,
        policy,
        public_inputs,
        claimed_outputs,
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
    artifact: CompileResult,
    public_inputs: Sequence[int],
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
) -> ExecutableConformanceTranscript | Unsupported:
    """Run an honest prover against the verifier in one process.

    The circuit is evaluated on ``public_inputs``, the claimed outputs are
    read from that evaluation, and both protocol parties run locally via
    :func:`run_protocol`.  This cannot demonstrate that either seed was
    withheld until the message it depends on was fixed; it is a conformance
    fixture for :func:`Verify`.
    """

    compiled = _executable(artifact, Capability.VERIFY)
    if isinstance(compiled, Unsupported):
        return compiled
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
    "ArchitectureCompileRequest",
    "ArchitectureId",
    "Bound",
    "BoundOptions",
    "BoundResult",
    "Compile",
    "CompileResult",
    "Compiled",
    "Cost",
    "CostParameters",
    "DeepSeekV4ProCompileRequest",
    "DemoGCompileRequest",
    "ExecutableConformanceTranscript",
    "Expectation",
    "ExpectedCost",
    "GPT2CompileRequest",
    "GreedyTextExecutionShape",
    "InklingCompileRequest",
    "KimiK3CompileRequest",
    "MatmulCompileRequest",
    "Optimization",
    "Optimize",
    "PolicyGrid",
    "ProtocolRun",
    "Unsupported",
    "VerificationCode",
    "VerificationLimits",
    "VerificationPolicy",
    "VerificationReport",
    "Verify",
    "build_executable_conformance_transcript",
    "make_verification_expectation",
    "run_protocol",
]
