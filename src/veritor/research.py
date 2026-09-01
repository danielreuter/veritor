"""Paper-level research facade for compilation, verification, and bounds.

The facade keeps the repository's three artifact kinds distinct:

* DemoG and matmul compile to an executable protocol circuit;
* GPT-2 compiles to indexed structural metadata; and
* Kimi-K3, DeepSeek-V4-Pro, and Inkling compile to aggregate bound models.

Only the first kind carries a :class:`~veritor.core.CompiledArtifact` that
the protocol can verify.  Static capacity analysis remains available for all
artifacts when their certified preconditions hold.
"""

from __future__ import annotations

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
from veritor.core import (
    Capability,
    CompiledArtifact,
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
    expectation: Expectation,
    artifact: CompiledArtifact,
    *,
    limits: VerificationLimits | None = None,
) -> VerificationReport:
    """Purely verify canonical transcript bytes against a trusted artifact.

    ``expectation`` carries both 32-byte verifier seeds; no transcript field
    is ever treated as their source of truth.
    """

    return verify_transcript(transcript_bytes, expectation, artifact, limits)


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


def _executable(
    artifact: CompileResult,
) -> ProtocolCircuitArtifact | Unsupported:
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
    return artifact


def make_verification_expectation(
    artifact: CompileResult,
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    public_inputs: Sequence[object] | None = None,
    claimed_outputs: Sequence[object] | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
) -> Expectation | Unsupported:
    """Build a verifier-local expectation for an executable compile result.

    Public inputs and claimed outputs default to the values bound into the
    artifact; seeds are drawn from the CSPRNG unless given.
    """

    executable = _executable(artifact)
    if isinstance(executable, Unsupported):
        return executable
    return make_expectation(
        executable.compiled,
        policy,
        executable.public_inputs if public_inputs is None else public_inputs,
        executable.expected_outputs if claimed_outputs is None else claimed_outputs,
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
    policy: VerificationPolicy = DEFAULT_CONFORMANCE_POLICY,
    *,
    public_inputs: Sequence[int] | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
) -> ExecutableConformanceTranscript | Unsupported:
    """Run an honest prover against the verifier in one process.

    The trusted tape is evaluated, the claimed outputs are read from it, and
    both protocol parties run locally via :func:`run_protocol`.  This cannot
    demonstrate that either seed was withheld until the message it depends on
    was fixed; it is a conformance fixture for :func:`Verify`.
    """

    executable = _executable(artifact)
    if isinstance(executable, Unsupported):
        return executable
    compiled = executable.compiled
    inputs = (
        executable.public_inputs if public_inputs is None else tuple(public_inputs)
    )
    values = dict(enumerate(compiled.circuit.evaluate_tape(inputs)))
    outputs = tuple(values[int(port.position)] for port in compiled.circuit.output_ports)
    expectation = make_expectation(
        compiled,
        policy,
        inputs,
        outputs,
        session_id=session_id,
        q_seed=q_seed,
        s_seed=s_seed,
    )
    run = run_protocol(compiled, expectation, values, limits=limits)
    if run.transcript is None:
        raise RuntimeError(
            f"honest conformance run was rejected: {run.report.code.value}: "
            f"{run.report.detail}"
        )
    return ExecutableConformanceTranscript(encode_transcript(run.transcript), expectation)


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
    "CompiledArtifact",
    "DeepSeekV4ProCompileRequest",
    "DemoGCompileRequest",
    "ExecutableConformanceTranscript",
    "Expectation",
    "FiniteBoundSolver",
    "FixedPolicyBoundResult",
    "GPT2ClassGranularity",
    "GPT2CompileRequest",
    "GreedyTextExecutionShape",
    "IndexedStructureArtifact",
    "InklingCompileRequest",
    "KimiK3CompileRequest",
    "MatmulCompileRequest",
    "OptimizationResult",
    "Optimize",
    "PolicyGridOptimizationResult",
    "ProtocolCircuitArtifact",
    "ProtocolRun",
    "RationalPolicyGrid",
    "TraceBinding",
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
