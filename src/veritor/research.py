"""Paper-level research facade for compilation, verification, and bounds.

The facade keeps the repository's three artifact kinds distinct:

* DemoG compiles to an executable protocol circuit;
* GPT-2 compiles to indexed structural metadata; and
* Kimi-K3, DeepSeek-V4-Pro, and Inkling compile to aggregate bound models.

Only the first kind can be adapted to staged transcript semantics.  Static
capacity analysis remains available for all artifacts when their certified
preconditions hold.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict, Unpack, cast

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.models.gpt2_gate_classes import GPT2ClassGranularity
from veritor.analysis import (
    DEFAULT_COUNTED_SOLVER_LIMITS,
    AdditiveExpectedCost,
    CountedCapacitySchema,
    CountedReplayLayout,
    CountedSolverLimits,
    FixedPolicyBoundResult,
    PolicyGridOptimizationResult,
    RationalPolicyGrid,
    StructuralCircuitCapacityOracle,
    VerificationUnitCapacityOracle,
    branch_and_bound_finite_bound,
    counted_fixed_policy_bound,
    counted_schema_from_weighted,
    exhaustive_finite_bound,
    optimize_policy_grid,
)
from veritor.commitment import MERKLE_SHA256_V1, ValueCommitmentRegistry
from veritor.core import (
    Capability,
    ProbabilityInput,
    Unsupported,
    VerificationLimits,
    VerificationPolicy,
)
from veritor.plugins import (
    AggregateBoundArtifact,
    ArchitectureCompileRequest,
    ArchitectureId,
    CompileResult,
    DeepSeekV4ProCompileRequest,
    DemoGCompileRequest,
    GPT2CompileRequest,
    GreedyTextExecutionShape,
    IndexedStructureArtifact,
    InklingCompileRequest,
    KimiK3CompileRequest,
    MatmulCompileRequest,
    ProtocolCircuitArtifact,
    TraceBinding,
    compile_architecture,
)
from veritor.staged import (
    TRANSPARENT_LOCAL_CHECK_V1,
    InteractionError,
    InteractionPhase,
    InteractiveProtocolRun,
    InteractivePublicContext,
    InteractiveVerificationResult,
    ResolvedExecutableArtifact,
    SampleEvidenceRegistry,
    StagedProverSession,
    StagedVerifierSession,
    TrustedArtifactRegistry,
    TrustedVerificationContext,
    VerificationCode,
    VerificationExpectation,
    VerificationReport,
    VerificationStatus,
    build_transcript_bytes,
    run_interactive_protocol,
    verify_transcript_bytes,
)


class FiniteBoundSolver(StrEnum):
    """Finite backends available for an executable literal ``(C, R, V)``."""

    AUTO = "auto"
    EXHAUSTIVE = "exhaustive"
    BRANCH_AND_BOUND = "branch-and-bound"


@dataclass(frozen=True, slots=True, init=False)
class BoundOptions:
    """Resource and representation choices for :func:`Bound`.

    ``AUTO`` uses exhaustive analysis up to ``max_verification_units`` and
    branch-and-bound above it.  Counted artifacts use the certified
    adversarial mega-unit relaxation unless a separately justified replay
    layout is supplied.
    """

    solver: FiniteBoundSolver
    max_verification_units: int
    max_states: int
    max_capacity_queries: int
    granularity: GPT2ClassGranularity
    position_bands: int
    counted_limits: CountedSolverLimits
    replay_layout: CountedReplayLayout | None
    assumptions: tuple[str, ...]

    def __init__(
        self,
        *,
        solver: FiniteBoundSolver | str = FiniteBoundSolver.AUTO,
        max_verification_units: int = 20,
        max_states: int = 1_000_000,
        max_capacity_queries: int = 1_000_000,
        granularity: GPT2ClassGranularity | str = GPT2ClassGranularity.ROW,
        position_bands: int = 8,
        counted_limits: CountedSolverLimits = DEFAULT_COUNTED_SOLVER_LIMITS,
        replay_layout: CountedReplayLayout | None = None,
        assumptions: tuple[str, ...] = (),
    ) -> None:
        try:
            checked_solver = FiniteBoundSolver(solver)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "solver must be 'auto', 'exhaustive', or 'branch-and-bound'"
            ) from error
        try:
            checked_granularity = GPT2ClassGranularity(granularity)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "granularity must be 'row', 'row-layer', or 'row-layer-band'"
            ) from error
        for name, value in (
            ("max_verification_units", max_verification_units),
            ("max_states", max_states),
            ("max_capacity_queries", max_capacity_queries),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(position_bands) is not int or position_bands <= 0:
            raise ValueError("position_bands must be a positive integer")
        if not isinstance(counted_limits, CountedSolverLimits):
            raise TypeError("counted_limits must be CountedSolverLimits")
        if replay_layout is not None and not isinstance(
            replay_layout,
            CountedReplayLayout,
        ):
            raise TypeError("replay_layout must be CountedReplayLayout or None")
        if type(assumptions) is not tuple or any(
            type(item) is not str or not item for item in assumptions
        ):
            raise TypeError("assumptions must be a tuple of nonempty strings")
        object.__setattr__(self, "solver", checked_solver)
        object.__setattr__(
            self,
            "max_verification_units",
            max_verification_units,
        )
        object.__setattr__(self, "max_states", max_states)
        object.__setattr__(
            self,
            "max_capacity_queries",
            max_capacity_queries,
        )
        object.__setattr__(self, "granularity", checked_granularity)
        object.__setattr__(self, "position_bands", position_bands)
        object.__setattr__(self, "counted_limits", counted_limits)
        object.__setattr__(self, "replay_layout", replay_layout)
        object.__setattr__(self, "assumptions", assumptions)


class BoundOptionOverrides(TypedDict, total=False):
    """Typed keyword overrides accepted by :func:`Bound`."""

    solver: FiniteBoundSolver | str
    max_verification_units: int
    max_states: int
    max_capacity_queries: int
    granularity: GPT2ClassGranularity | str
    position_bands: int
    counted_limits: CountedSolverLimits
    replay_layout: CountedReplayLayout | None
    assumptions: tuple[str, ...]


type BoundOptionsInput = BoundOptions | Mapping[str, object] | None
type BoundResult = FixedPolicyBoundResult | Unsupported
type OptimizationResult = PolicyGridOptimizationResult | Unsupported
type ExecutableArtifactResult = ResolvedExecutableArtifact | Unsupported

DEFAULT_CONFORMANCE_POLICY = VerificationPolicy(1, 1, 0)
_DEFAULT_BOUND_OPTIONS = BoundOptions()
_BOUND_OPTION_NAMES = frozenset(BoundOptionOverrides.__optional_keys__)


def _coerce_bound_options(
    options: BoundOptionsInput,
    overrides: Mapping[str, object] | None = None,
) -> BoundOptions:
    if options is None:
        selected = _DEFAULT_BOUND_OPTIONS
        raw: dict[str, object] = {}
    elif isinstance(options, BoundOptions):
        selected = options
        raw = {}
    elif isinstance(options, Mapping):
        selected = _DEFAULT_BOUND_OPTIONS
        raw = dict(options)
    else:
        raise TypeError("options must be BoundOptions, a mapping, or None")
    if any(type(key) is not str for key in raw):
        raise TypeError("bound option names must be strings")
    if overrides:
        raw.update(overrides)
    unknown = set(raw).difference(_BOUND_OPTION_NAMES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown bound option(s): {names}")
    return BoundOptions(
        solver=cast(
            FiniteBoundSolver | str,
            raw.get("solver", selected.solver),
        ),
        max_verification_units=cast(
            int,
            raw.get(
                "max_verification_units",
                selected.max_verification_units,
            ),
        ),
        max_states=cast(int, raw.get("max_states", selected.max_states)),
        max_capacity_queries=cast(
            int,
            raw.get(
                "max_capacity_queries",
                selected.max_capacity_queries,
            ),
        ),
        granularity=cast(
            GPT2ClassGranularity | str,
            raw.get("granularity", selected.granularity),
        ),
        position_bands=cast(
            int,
            raw.get("position_bands", selected.position_bands),
        ),
        counted_limits=cast(
            CountedSolverLimits,
            raw.get("counted_limits", selected.counted_limits),
        ),
        replay_layout=cast(
            CountedReplayLayout | None,
            raw.get("replay_layout", selected.replay_layout),
        ),
        assumptions=cast(
            tuple[str, ...],
            raw.get("assumptions", selected.assumptions),
        ),
    )


def _ordered_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
    return tuple(ordered)


def _unsupported(
    artifact: CompileResult,
    capability: Capability,
    *,
    reason_code: str,
    detail: str,
) -> Unsupported:
    return Unsupported(
        capability=capability,
        plugin_id=artifact.plugin_id,
        reason_code=reason_code,
        detail=detail,
        artifact_kind=artifact.artifact_kind,
    )


def _unsupported_from_capability(
    artifact: CompileResult,
    capability: Capability,
) -> Unsupported:
    try:
        status = artifact.capabilities.status_for(capability)
    except KeyError:
        return _unsupported(
            artifact,
            capability,
            reason_code="UNREPORTED_CAPABILITY",
            detail=f"{artifact.plugin_id} did not report this capability",
        )
    return _unsupported(
        artifact,
        capability,
        reason_code=status.reason_code or "CAPABILITY_NOT_EXECUTABLE",
        detail=status.detail or "the artifact does not supply executable semantics",
    )


def Compile(
    architecture_id: ArchitectureId | str,
    request: ArchitectureCompileRequest | None = None,
) -> CompileResult:
    """Compile one registered architecture or executable workload."""

    return compile_architecture(architecture_id, request)


def Verify(
    transcript_bytes: bytes,
    expectation: VerificationExpectation,
    trust: TrustedVerificationContext,
    *,
    limits: VerificationLimits | None = None,
) -> VerificationReport:
    """Purely verify canonical staged transcript bytes against local trust.

    ``expectation`` and ``trust`` are mandatory verifier-local inputs.  In
    particular, both 32-byte phase seeds must already be present in the
    expectation; no transcript field is treated as their source of truth.
    """

    checked_limits = VerificationLimits() if limits is None else limits
    return verify_transcript_bytes(
        transcript_bytes,
        expectation,
        trust,
        checked_limits,
    )


def _finite_bound(
    artifact: ProtocolCircuitArtifact,
    policy: VerificationPolicy,
    options: BoundOptions,
) -> FixedPolicyBoundResult:
    position_oracle = StructuralCircuitCapacityOracle(artifact.circuit)
    unit_oracle = VerificationUnitCapacityOracle(
        position_oracle,
        artifact.verification_partition,
    )
    solver = options.solver
    if solver is FiniteBoundSolver.AUTO:
        solver = (
            FiniteBoundSolver.EXHAUSTIVE
            if artifact.verification_partition.unit_count
            <= options.max_verification_units
            else FiniteBoundSolver.BRANCH_AND_BOUND
        )
    if solver is FiniteBoundSolver.EXHAUSTIVE:
        return exhaustive_finite_bound(
            artifact.replay_partition,
            artifact.verification_partition,
            policy,
            unit_oracle,
            max_verification_units=options.max_verification_units,
            assumptions=options.assumptions,
        )
    return branch_and_bound_finite_bound(
        artifact.replay_partition,
        artifact.verification_partition,
        policy,
        unit_oracle,
        max_states=options.max_states,
        max_capacity_queries=options.max_capacity_queries,
        assumptions=options.assumptions,
    )


def _counted_schema(
    artifact: IndexedStructureArtifact | AggregateBoundArtifact,
    options: BoundOptions,
) -> CountedCapacitySchema | Unsupported:
    if isinstance(artifact, IndexedStructureArtifact):
        catalog = artifact.gate_class_catalog(
            granularity=options.granularity,
            position_bands=options.position_bands,
        )
        if isinstance(catalog, Unsupported):
            return _unsupported(
                artifact,
                Capability.STATIC_BOUND,
                reason_code=catalog.reason_code,
                detail=(
                    "a certified GPT-2 gate-class catalog is required for the "
                    f"counted bound: {catalog.detail}"
                ),
            )
        weighted = catalog.partition
    else:
        weighted = artifact.weighted_partition
    assumptions = _ordered_unique(
        artifact.assumption_texts,
        options.assumptions,
    )
    return counted_schema_from_weighted(
        weighted,
        assumptions=assumptions,
        provenance_identity=artifact.identity.digest,
    )


def Bound(
    artifact: CompileResult,
    policy: VerificationPolicy,
    *,
    options: BoundOptionsInput = None,
    **overrides: Unpack[BoundOptionOverrides],
) -> BoundResult:
    """Return a guarantee-carrying fixed-policy capacity bound.

    Executable artifacts use their literal replay and verification partitions.
    GPT-2 and aggregate profiles use certified counted schemas.  Without a
    concrete counted replay layout, that path reports the adversarial
    mega-unit upper relaxation rather than an exact protocol-layout result.
    """

    if not isinstance(
        artifact,
        (
            ProtocolCircuitArtifact,
            IndexedStructureArtifact,
            AggregateBoundArtifact,
        ),
    ):
        raise TypeError("artifact must be a Compile result")
    if not isinstance(policy, VerificationPolicy):
        raise TypeError("policy must be VerificationPolicy")
    selected = _coerce_bound_options(options, overrides)
    if isinstance(artifact, ProtocolCircuitArtifact):
        return _finite_bound(artifact, policy, selected)
    schema = _counted_schema(artifact, selected)
    if isinstance(schema, Unsupported):
        return schema
    return counted_fixed_policy_bound(
        schema,
        policy,
        replay_layout=selected.replay_layout,
        limits=selected.counted_limits,
    )


@dataclass(frozen=True, slots=True)
class _UnsupportedBound(Exception):
    outcome: Unsupported


def Optimize(
    artifact: CompileResult,
    grid: RationalPolicyGrid,
    cost_model: AdditiveExpectedCost,
    *,
    bound_options: BoundOptionsInput = None,
    capacity_limit: LogCardinality | None = None,
    maximum_expected_cost: ProbabilityInput | None = None,
) -> OptimizationResult:
    """Optimize exact rational policies on a finite grid using :func:`Bound`.

    An unsupported bound is returned directly.  Otherwise the generic grid
    optimizer receives each original certified result, so exact, bracketed,
    conditional, resource-limited, and relaxed statuses are not upgraded.
    """

    selected = _coerce_bound_options(bound_options)

    def evaluate(policy: VerificationPolicy) -> FixedPolicyBoundResult:
        outcome = Bound(artifact, policy, options=selected)
        if isinstance(outcome, Unsupported):
            raise _UnsupportedBound(outcome)
        return outcome

    try:
        return optimize_policy_grid(
            grid,
            cost_model,
            evaluate,
            capacity_limit=capacity_limit,
            maximum_expected_cost=maximum_expected_cost,
        )
    except _UnsupportedBound as error:
        return error.outcome


def adapt_protocol_artifact(
    artifact: CompileResult,
) -> ExecutableArtifactResult:
    """Attach only the executable services trusted by a protocol artifact."""

    if not isinstance(
        artifact,
        (
            ProtocolCircuitArtifact,
            IndexedStructureArtifact,
            AggregateBoundArtifact,
        ),
    ):
        raise TypeError("artifact must be a Compile result")
    if not isinstance(artifact, ProtocolCircuitArtifact):
        return _unsupported_from_capability(artifact, Capability.VERIFY)
    return ResolvedExecutableArtifact.from_uniform_circuit(
        artifact.circuit,
        artifact.replay_partition,
        artifact.verification_partition,
        codec=artifact.circuit.value_codec,
        relation_evaluator=artifact.circuit.relation_evaluator,
    )


def create_trusted_artifact_registry(
    artifact: CompileResult,
) -> TrustedArtifactRegistry | Unsupported:
    """Create a content-addressed verifier-local registry for one artifact."""

    resolved = adapt_protocol_artifact(artifact)
    if isinstance(resolved, Unsupported):
        return resolved
    return TrustedArtifactRegistry((resolved,))


def create_trusted_verification_context(
    artifact: CompileResult,
    *,
    value_commitment_backends: ValueCommitmentRegistry | None = None,
    sample_evidence_backends: SampleEvidenceRegistry | None = None,
) -> TrustedVerificationContext | Unsupported:
    """Create local artifact and backend trust roots for staged verification."""

    registry = create_trusted_artifact_registry(artifact)
    if isinstance(registry, Unsupported):
        return registry
    return TrustedVerificationContext(
        artifact_resolver=registry,
        value_commitment_backends=(
            ValueCommitmentRegistry.with_defaults()
            if value_commitment_backends is None
            else value_commitment_backends
        ),
        sample_evidence_backends=(
            SampleEvidenceRegistry.with_defaults()
            if sample_evidence_backends is None
            else sample_evidence_backends
        ),
    )


def make_verification_expectation(
    artifact: CompileResult,
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    public_inputs: Sequence[object] | None = None,
    claimed_outputs: Sequence[object] | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    value_commitment_backend_id: str = MERKLE_SHA256_V1,
    sample_evidence_backend_id: str = TRANSPARENT_LOCAL_CHECK_V1,
) -> VerificationExpectation | Unsupported:
    """Build a verifier-local expectation with fresh CSPRNG seeds by default."""

    if not isinstance(
        artifact,
        (
            ProtocolCircuitArtifact,
            IndexedStructureArtifact,
            AggregateBoundArtifact,
        ),
    ):
        raise TypeError("artifact must be a Compile result")
    if not isinstance(artifact, ProtocolCircuitArtifact):
        return _unsupported_from_capability(artifact, Capability.VERIFY)
    return VerificationExpectation(
        session_id=secrets.token_bytes(32) if session_id is None else session_id,
        compiled_result_digest=str(artifact.compiled_identity.digest),
        policy=policy,
        public_inputs=(
            tuple(artifact.public_inputs)
            if public_inputs is None
            else tuple(public_inputs)
        ),
        claimed_outputs=(
            tuple(artifact.expected_outputs)
            if claimed_outputs is None
            else tuple(claimed_outputs)
        ),
        q_seed=secrets.token_bytes(32) if q_seed is None else q_seed,
        s_seed=secrets.token_bytes(32) if s_seed is None else s_seed,
        value_commitment_backend_id=value_commitment_backend_id,
        sample_evidence_backend_id=sample_evidence_backend_id,
    )


@dataclass(frozen=True, slots=True)
class ExecutableConformanceTranscript:
    """One-shot fixture, not evidence of interactive phase ordering."""

    transcript_bytes: bytes
    expectation: VerificationExpectation
    trust: TrustedVerificationContext

    @property
    def data(self) -> bytes:
        """Concise alias for the canonical transcript bytes."""

        return self.transcript_bytes


DemoConformanceTranscript = ExecutableConformanceTranscript


def build_executable_conformance_transcript(
    artifact: CompileResult,
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    public_inputs: Sequence[int] | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
) -> ExecutableConformanceTranscript | Unsupported:
    """Build a complete honest executable transcript for local conformance.

    This helper evaluates the trusted tape, constructs transparent evidence,
    and serializes every phase in one process.  It is deliberately not a
    secure interaction and cannot prove that ``q`` was withheld until the
    boundary was fixed or that ``s`` was withheld until selected-unit roots
    were fixed.
    """

    if not isinstance(
        artifact,
        (
            ProtocolCircuitArtifact,
            IndexedStructureArtifact,
            AggregateBoundArtifact,
        ),
    ):
        raise TypeError("artifact must be a Compile result")
    if not isinstance(artifact, ProtocolCircuitArtifact):
        return _unsupported_from_capability(artifact, Capability.VERIFY)
    resolved = adapt_protocol_artifact(artifact)
    if isinstance(resolved, Unsupported):
        return resolved
    inputs = (
        tuple(artifact.public_inputs)
        if public_inputs is None
        else tuple(public_inputs)
    )
    tape = artifact.circuit.evaluate_tape(inputs)
    assignment = dict(enumerate(tape))
    outputs = tuple(
        assignment[int(port.position)] for port in artifact.circuit.output_ports
    )
    expectation = make_verification_expectation(
        artifact,
        policy,
        public_inputs=inputs,
        claimed_outputs=outputs,
        session_id=session_id,
        q_seed=q_seed,
        s_seed=s_seed,
    )
    if isinstance(expectation, Unsupported):
        return expectation
    trust = create_trusted_verification_context(artifact)
    if isinstance(trust, Unsupported):
        return trust
    checked_limits = VerificationLimits() if limits is None else limits
    data = build_transcript_bytes(
        resolved,
        expectation,
        assignment,
        limits=checked_limits,
    )
    return ExecutableConformanceTranscript(data, expectation, trust)


def build_demo_conformance_transcript(
    artifact: CompileResult | None = None,
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    public_inputs: Sequence[int] | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
) -> DemoConformanceTranscript | Unsupported:
    """Compatibility wrapper defaulting the generic builder to DemoG."""

    selected = Compile(ArchitectureId.DEMO_G) if artifact is None else artifact
    return build_executable_conformance_transcript(
        selected,
        policy,
        public_inputs=public_inputs,
        session_id=session_id,
        q_seed=q_seed,
        s_seed=s_seed,
        limits=limits,
    )


# Paper spellings and Python-style spellings are intentionally identical.
compile = Compile
verify = Verify
bound = Bound
optimize = Optimize
resolve_executable_artifact = adapt_protocol_artifact
create_verification_expectation = make_verification_expectation
build_conformance_transcript = build_demo_conformance_transcript


__all__ = [
    "DEFAULT_CONFORMANCE_POLICY",
    "AdditiveExpectedCost",
    "AggregateBoundArtifact",
    "ArchitectureCompileRequest",
    "ArchitectureId",
    "Bound",
    "BoundOptionOverrides",
    "BoundOptions",
    "BoundOptionsInput",
    "BoundResult",
    "Compile",
    "CompileResult",
    "DeepSeekV4ProCompileRequest",
    "DemoConformanceTranscript",
    "DemoGCompileRequest",
    "ExecutableArtifactResult",
    "ExecutableConformanceTranscript",
    "FiniteBoundSolver",
    "FixedPolicyBoundResult",
    "GPT2ClassGranularity",
    "GPT2CompileRequest",
    "GreedyTextExecutionShape",
    "IndexedStructureArtifact",
    "InklingCompileRequest",
    "InteractionError",
    "InteractionPhase",
    "InteractiveProtocolRun",
    "InteractivePublicContext",
    "InteractiveVerificationResult",
    "KimiK3CompileRequest",
    "MatmulCompileRequest",
    "OptimizationResult",
    "PolicyGridOptimizationResult",
    "ProtocolCircuitArtifact",
    "RationalPolicyGrid",
    "ResolvedExecutableArtifact",
    "StagedProverSession",
    "StagedVerifierSession",
    "TraceBinding",
    "TrustedArtifactRegistry",
    "TrustedVerificationContext",
    "Unsupported",
    "VerificationCode",
    "VerificationExpectation",
    "VerificationLimits",
    "VerificationPolicy",
    "VerificationReport",
    "VerificationStatus",
    "Verify",
    "adapt_protocol_artifact",
    "bound",
    "build_conformance_transcript",
    "build_demo_conformance_transcript",
    "build_executable_conformance_transcript",
    "compile",
    "create_trusted_artifact_registry",
    "create_trusted_verification_context",
    "create_verification_expectation",
    "make_verification_expectation",
    "optimize",
    "resolve_executable_artifact",
    "run_interactive_protocol",
    "verify",
]
