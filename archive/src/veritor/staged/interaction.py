"""Phase-separated in-process execution of the staged protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from veritor.commitment import (
    CommitmentDomain,
    CommitmentError,
    CommitmentOwner,
    MerkleSha256Backend,
    ValueCommitmentBackend,
    ValueCommitmentProver,
)
from veritor.core import (
    ArtifactKind,
    VerificationLimits,
    VerificationPolicy,
    validate_circuit_contract,
    validate_compiled_result,
    validate_digest,
)

from .artifact import ResolvedExecutableArtifact
from .boundary import (
    CommitmentLayout,
    derive_commitment_ownership,
    validate_output_schemas,
)
from .builder import (
    CommitmentOpeningTrees,
    ExecutingReplayService,
    MappingValueSource,
    ReplayService,
    TranscriptBuildError,
    ValueSource,
    build_public_statement,
    canonical_encode_value,
    public_io_positions,
)
from .challenge import derive_q_challenge, derive_s_challenge
from .evidence import (
    SampleEvidenceBackend,
    SampleVerificationInstance,
    TransparentLocalCheckBackend,
)
from .model import (
    BoundaryMessage,
    OwnedValueCommitment,
    PublicStatement,
    ReplayChallengeMessage,
    SampleChallengeMessage,
    SampledUnitEvidence,
    SampleEvidenceMessage,
    SessionParameters,
    StagedProtocolError,
    StagedTranscript,
    UnitCommitmentsMessage,
    VerificationExpectation,
    VerificationReport,
)
from .phases import (
    derive_boundary_commitment_domain,
    derive_boundary_phase_digest,
    derive_initial_phase_digest,
    derive_q_phase_digest,
    derive_sample_phase_digest,
    derive_unit_commitment_domain,
    derive_unit_commitments_phase_digest,
)
from .trust import TrustedVerificationContext
from .verifier import verify_transcript_bytes
from .wire import encode_transcript


class InteractionError(StagedProtocolError):
    """An interactive message is invalid or arrives in the wrong phase."""


class InteractionPhase(StrEnum):
    INITIAL = "initial"
    BOUNDARY_COMMITTED = "boundary_committed"
    REPLAY_CHALLENGED = "replay_challenged"
    UNITS_COMMITTED = "units_committed"
    SAMPLE_CHALLENGED = "sample_challenged"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class InteractivePublicContext:
    """Everything public to the prover; deliberately contains no seeds."""

    session_id: bytes
    compiled_result_digest: str
    policy: VerificationPolicy
    public_inputs: tuple[object, ...]
    claimed_outputs: tuple[object, ...]
    value_commitment_backend_id: str
    sample_evidence_backend_id: str

    def __post_init__(self) -> None:
        if type(self.session_id) is not bytes or not self.session_id:
            raise InteractionError("session_id must be nonempty bytes")
        validate_digest(self.compiled_result_digest, "compiled_result_digest")
        if not isinstance(self.policy, VerificationPolicy):
            raise InteractionError("policy must be VerificationPolicy")
        if type(self.public_inputs) is not tuple:
            raise InteractionError("public_inputs must be a tuple")
        if type(self.claimed_outputs) is not tuple:
            raise InteractionError("claimed_outputs must be a tuple")
        for name in (
            "value_commitment_backend_id",
            "sample_evidence_backend_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise InteractionError(f"{name} must be a nonempty string")

    @classmethod
    def from_expectation(
        cls,
        expectation: VerificationExpectation,
    ) -> InteractivePublicContext:
        if not isinstance(expectation, VerificationExpectation):
            raise InteractionError("expectation has the wrong type")
        return cls(
            session_id=expectation.session_id,
            compiled_result_digest=expectation.compiled_result_digest,
            policy=expectation.policy,
            public_inputs=expectation.public_inputs,
            claimed_outputs=expectation.claimed_outputs,
            value_commitment_backend_id=(expectation.value_commitment_backend_id),
            sample_evidence_backend_id=(expectation.sample_evidence_backend_id),
        )


def _value_source(
    value: ValueSource | Mapping[int, object],
) -> ValueSource:
    if isinstance(value, ValueSource):
        return value
    if isinstance(value, Mapping):
        return MappingValueSource(value)
    raise InteractionError("value source has the wrong type")


def _canonical_encode(
    artifact: ResolvedExecutableArtifact,
    value_type: str,
    value: object,
) -> bytes:
    try:
        return canonical_encode_value(artifact, value_type, value)
    except TranscriptBuildError as error:
        raise InteractionError(str(error)) from error


def _statement(
    artifact: ResolvedExecutableArtifact,
    context: InteractivePublicContext,
) -> PublicStatement:
    try:
        return build_public_statement(
            artifact,
            context.public_inputs,
            context.claimed_outputs,
        )
    except TranscriptBuildError as error:
        raise InteractionError(str(error)) from error


def _session(context: InteractivePublicContext) -> SessionParameters:
    return SessionParameters(
        session_id=context.session_id,
        compiled_result_digest=context.compiled_result_digest,
        policy_digest=str(context.policy.digest),
        value_commitment_backend_id=context.value_commitment_backend_id,
        sample_evidence_backend_id=context.sample_evidence_backend_id,
    )


def _validate_artifact(
    artifact: ResolvedExecutableArtifact,
    context: InteractivePublicContext,
    limits: VerificationLimits,
) -> CommitmentLayout:
    if not isinstance(artifact, ResolvedExecutableArtifact):
        raise InteractionError("interactive protocol requires executable artifact")
    if artifact.circuit.identity.artifact_kind is not ArtifactKind.EXECUTABLE_CIRCUIT:
        raise InteractionError("interactive artifact is not executable")
    limits.enforce(
        "max_positions",
        len(artifact.circuit.input_ports) + artifact.circuit.computed_positions.count,
    )
    limits.enforce(
        "max_units",
        max(
            artifact.replay_partition.unit_count,
            artifact.verification_partition.unit_count,
        ),
    )
    compiled = validate_compiled_result(
        artifact.circuit,
        artifact.replay_partition,
        artifact.verification_partition,
    )
    if str(compiled.digest) != context.compiled_result_digest:
        raise InteractionError("public context names another compiled result")
    validate_circuit_contract(artifact.circuit, exhaustive=True)
    for replay_unit in artifact.replay_partition.units:
        limits.enforce("max_positions_per_unit", replay_unit.count)
    for verification_unit in artifact.verification_partition.units:
        limits.enforce("max_positions_per_unit", verification_unit.count)
    validate_output_schemas(artifact.circuit)
    return derive_commitment_ownership(
        artifact.circuit,
        artifact.replay_partition,
        limits,
    )


class StagedProverSession:
    """Prover state with public context but no verifier sampling seeds."""

    __slots__ = (
        "_artifact",
        "_boundary_domain",
        "_boundary_message",
        "_boundary_phase_digest",
        "_boundary_tree",
        "_boundary_values",
        "_commitment_backend",
        "_context",
        "_evidence_backend",
        "_evidence_message",
        "_initial_phase_digest",
        "_layout",
        "_limits",
        "_phase",
        "_q_message",
        "_replay_service",
        "_replayed_units",
        "_s_message",
        "_session",
        "_statement",
        "_trees",
        "_unit_commitments",
        "_unit_domains",
    )

    def __init__(
        self,
        artifact: ResolvedExecutableArtifact,
        context: InteractivePublicContext,
        value_source: ValueSource | Mapping[int, object],
        *,
        commitment_backend: ValueCommitmentBackend | None = None,
        evidence_backend: SampleEvidenceBackend | None = None,
        replay_service: ReplayService | None = None,
        limits: VerificationLimits | None = None,
    ) -> None:
        self._context = context
        self._limits = VerificationLimits() if limits is None else limits
        if not isinstance(self._limits, VerificationLimits):
            raise InteractionError("limits must be VerificationLimits")
        self._artifact = artifact
        self._layout = _validate_artifact(
            artifact,
            context,
            self._limits,
        )
        self._commitment_backend = (
            MerkleSha256Backend() if commitment_backend is None else commitment_backend
        )
        self._evidence_backend = (
            TransparentLocalCheckBackend()
            if evidence_backend is None
            else evidence_backend
        )
        self._replay_service = (
            ExecutingReplayService() if replay_service is None else replay_service
        )
        if not isinstance(self._commitment_backend, ValueCommitmentBackend):
            raise InteractionError("commitment backend has the wrong type")
        if not isinstance(self._evidence_backend, SampleEvidenceBackend):
            raise InteractionError("evidence backend has the wrong type")
        if not isinstance(self._replay_service, ReplayService):
            raise InteractionError("replay service has the wrong type")
        if self._commitment_backend.backend_id != context.value_commitment_backend_id:
            raise InteractionError("public context names another commitment backend")
        if self._evidence_backend.backend_id != context.sample_evidence_backend_id:
            raise InteractionError("public context names another evidence backend")

        self._statement = _statement(artifact, context)
        self._session = _session(context)
        self._initial_phase_digest = derive_initial_phase_digest(
            self._session,
            self._statement,
            context.policy,
        )
        self._boundary_domain = derive_boundary_commitment_domain(
            self._session,
            self._statement,
            context.policy,
            artifact.circuit,
            self._layout,
            self._initial_phase_digest,
        )
        source = _value_source(value_source)
        self._boundary_values = {
            int(position): source.value_at(int(position))
            for position in self._layout.boundary.items
        }
        self._phase = InteractionPhase.INITIAL
        self._boundary_tree: ValueCommitmentProver | None = None
        self._boundary_message: BoundaryMessage | None = None
        self._boundary_phase_digest: bytes | None = None
        self._q_message: ReplayChallengeMessage | None = None
        self._unit_commitments: UnitCommitmentsMessage | None = None
        self._s_message: SampleChallengeMessage | None = None
        self._evidence_message: SampleEvidenceMessage | None = None
        self._trees: dict[CommitmentOwner, ValueCommitmentProver] = {}
        self._unit_domains: dict[int, CommitmentDomain] = {}
        self._replayed_units: list[int] = []

    @property
    def phase(self) -> InteractionPhase:
        return self._phase

    @property
    def replayed_unit_indices(self) -> tuple[int, ...]:
        return tuple(self._replayed_units)

    @property
    def replayed_gate_count(self) -> int:
        return sum(
            self._artifact.replay_partition.unit_at(index).count
            for index in self._replayed_units
        )

    @property
    def replayed_cost(self) -> int:
        return sum(
            self._artifact.replay_partition.unit_at(index).replay_cost
            for index in self._replayed_units
        )

    def commit_boundary(self) -> BoundaryMessage:
        if self._phase is not InteractionPhase.INITIAL:
            raise InteractionError("boundary may only be committed once")
        encoded_values = {
            position: _canonical_encode(
                self._artifact,
                self._boundary_domain.schema_for(position),
                value,
            )
            for position, value in self._boundary_values.items()
        }
        tree = self._commitment_backend.commit(
            self._boundary_domain,
            encoded_values,
            self._limits,
        )
        required_public_io = public_io_positions(
            self._artifact,
            self._layout,
        )
        self._limits.enforce("max_openings", len(required_public_io))
        message = BoundaryMessage(
            tree.commitment,
            tuple(tree.open(position) for position in required_public_io),
        )
        self._boundary_tree = tree
        self._trees[CommitmentOwner.boundary()] = tree
        self._boundary_message = message
        self._boundary_phase_digest = derive_boundary_phase_digest(
            self._initial_phase_digest,
            message,
        )
        self._phase = InteractionPhase.BOUNDARY_COMMITTED
        return message

    def answer_replay_challenge(
        self,
        challenge: ReplayChallengeMessage,
    ) -> UnitCommitmentsMessage:
        if self._phase is not InteractionPhase.BOUNDARY_COMMITTED:
            raise InteractionError(
                "replay challenge requires a fixed boundary commitment"
            )
        if not isinstance(challenge, ReplayChallengeMessage):
            raise InteractionError("replay challenge has the wrong type")
        if self._boundary_phase_digest is None:
            raise RuntimeError("boundary phase digest is unavailable")
        expected_units = derive_q_challenge(
            challenge.seed,
            self._boundary_phase_digest,
            self._artifact.replay_partition,
            self._context.policy.q,
            self._limits,
        )
        expected_phase = derive_q_phase_digest(
            self._boundary_phase_digest,
            challenge.seed,
            expected_units,
        )
        if (
            challenge.boundary_phase_digest != self._boundary_phase_digest
            or challenge.selected_replay_units != expected_units
            or challenge.phase_digest != expected_phase
        ):
            raise InteractionError("replay challenge is not correctly derived")
        self._q_message = challenge
        self._phase = InteractionPhase.REPLAY_CHALLENGED

        commitments: list[OwnedValueCommitment] = []
        boundary_source = MappingValueSource(self._boundary_values)
        for replay_unit_index in expected_units:
            domain = derive_unit_commitment_domain(
                self._session,
                self._statement,
                self._context.policy,
                self._artifact.circuit,
                self._layout,
                expected_phase,
                replay_unit_index,
            )
            positions = tuple(int(item) for item in domain.positions.items)
            try:
                raw_values = self._replay_service.values_for_unit(
                    self._artifact,
                    replay_unit_index,
                    positions,
                    self._boundary_values,
                    boundary_source,
                )
            except InteractionError:
                raise
            except Exception as error:
                raise InteractionError(
                    f"replay failed for unit {replay_unit_index}: {error}"
                ) from error
            if set(raw_values) != set(positions):
                raise InteractionError(
                    "replay service did not return the exact unit interior"
                )
            encoded_values = {
                position: _canonical_encode(
                    self._artifact,
                    domain.schema_for(position),
                    raw_values[position],
                )
                for position in positions
            }
            tree = self._commitment_backend.commit(
                domain,
                encoded_values,
                self._limits,
            )
            owner = CommitmentOwner.replay_unit(replay_unit_index)
            self._trees[owner] = tree
            self._unit_domains[replay_unit_index] = domain
            self._replayed_units.append(replay_unit_index)
            commitments.append(OwnedValueCommitment(replay_unit_index, tree.commitment))
        commitment_tuple = tuple(commitments)
        phase_digest = derive_unit_commitments_phase_digest(
            expected_phase,
            commitment_tuple,
        )
        message = UnitCommitmentsMessage(
            expected_phase,
            commitment_tuple,
            phase_digest,
        )
        self._unit_commitments = message
        self._phase = InteractionPhase.UNITS_COMMITTED
        return message

    def answer_sample_challenge(
        self,
        challenge: SampleChallengeMessage,
    ) -> SampleEvidenceMessage:
        if self._phase is not InteractionPhase.UNITS_COMMITTED:
            raise InteractionError(
                "sample challenge requires fixed selected-unit commitments"
            )
        if not isinstance(challenge, SampleChallengeMessage):
            raise InteractionError("sample challenge has the wrong type")
        if (
            self._boundary_message is None
            or self._q_message is None
            or self._unit_commitments is None
        ):
            raise RuntimeError("selected-unit phase is unavailable")
        expected_units = derive_s_challenge(
            challenge.seed,
            self._unit_commitments.phase_digest,
            self._artifact.verification_partition,
            self._q_message.selected_replay_units,
            self._context.policy.s,
            self._limits,
        )
        expected_phase = derive_sample_phase_digest(
            self._unit_commitments.phase_digest,
            challenge.seed,
            expected_units,
        )
        if (
            challenge.unit_commitments_phase_digest
            != self._unit_commitments.phase_digest
            or challenge.selected_verification_units != expected_units
            or challenge.phase_digest != expected_phase
        ):
            raise InteractionError("sample challenge is not correctly derived")
        self._s_message = challenge
        self._phase = InteractionPhase.SAMPLE_CHALLENGED

        opening_trees = CommitmentOpeningTrees(self._trees)
        commitments = {
            item.replay_unit_index: item.commitment
            for item in self._unit_commitments.commitments
        }
        evidence: list[SampledUnitEvidence] = []
        for verification_unit_index in expected_units:
            verification_unit = self._artifact.verification_partition.unit_at(
                verification_unit_index
            )
            replay_unit_index = int(verification_unit.replay_unit)
            domain = self._unit_domains.get(replay_unit_index)
            if domain is None:
                raise InteractionError("sampled unit has no selected replay commitment")
            instance = SampleVerificationInstance(
                artifact=self._artifact,
                layout=self._layout,
                session=self._session,
                statement=self._statement,
                policy=self._context.policy,
                sample_phase_digest=expected_phase,
                verification_unit_index=verification_unit_index,
                boundary_domain=self._boundary_domain,
                boundary_commitment=self._boundary_message.commitment,
                unit_domain=domain,
                unit_commitment=commitments[replay_unit_index],
                value_backend=self._commitment_backend,
                sample_evidence_backend_id=self._evidence_backend.backend_id,
            )
            payload = self._evidence_backend.build_evidence(
                instance,
                opening_trees,
                self._limits,
            )
            evidence.append(
                SampledUnitEvidence(
                    verification_unit_index,
                    self._evidence_backend.backend_id,
                    payload,
                )
            )
        message = SampleEvidenceMessage(expected_phase, tuple(evidence))
        self._evidence_message = message
        self._phase = InteractionPhase.COMPLETE
        return message

    def transcript(self) -> StagedTranscript:
        if self._phase is not InteractionPhase.COMPLETE:
            raise InteractionError("transcript is unavailable before evidence")
        if (
            self._boundary_message is None
            or self._q_message is None
            or self._unit_commitments is None
            or self._s_message is None
            or self._evidence_message is None
        ):
            raise RuntimeError("complete prover state is inconsistent")
        return StagedTranscript(
            session=self._session,
            statement=self._statement,
            policy=self._context.policy,
            boundary=self._boundary_message,
            replay_challenge=self._q_message,
            unit_commitments=self._unit_commitments,
            sample_challenge=self._s_message,
            sample_evidence=self._evidence_message,
        )


class StagedVerifierSession:
    """Verifier state that privately owns and reveals both seeds in order."""

    __slots__ = (
        "_artifact",
        "_boundary_message",
        "_boundary_phase_digest",
        "_context",
        "_expectation",
        "_initial_phase_digest",
        "_layout",
        "_limits",
        "_phase",
        "_q_message",
        "_s_message",
        "_session",
        "_statement",
        "_trust",
        "_unit_commitments",
        "_value_backend",
    )

    def __init__(
        self,
        expectation: VerificationExpectation,
        trust: TrustedVerificationContext,
        *,
        limits: VerificationLimits | None = None,
    ) -> None:
        if not isinstance(expectation, VerificationExpectation):
            raise InteractionError("expectation has the wrong type")
        if not isinstance(trust, TrustedVerificationContext):
            raise InteractionError("trust has the wrong type")
        self._expectation = expectation
        self._trust = trust
        self._limits = VerificationLimits() if limits is None else limits
        self._context = InteractivePublicContext.from_expectation(expectation)
        try:
            resolved = trust.artifact_resolver.resolve(
                expectation.compiled_result_digest
            )
        except Exception as error:
            raise InteractionError("trusted artifact resolver failed") from error
        if not isinstance(resolved, ResolvedExecutableArtifact):
            raise InteractionError("trusted artifact is absent or not executable")
        self._artifact = resolved
        self._layout = _validate_artifact(
            resolved,
            self._context,
            self._limits,
        )
        value_backend = trust.value_commitment_backends.resolve(
            expectation.value_commitment_backend_id
        )
        if value_backend is None:
            raise InteractionError("commitment backend is not trusted")
        if (
            trust.sample_evidence_backends.resolve(
                expectation.sample_evidence_backend_id
            )
            is None
        ):
            raise InteractionError("evidence backend is not trusted")
        self._value_backend = value_backend
        self._statement = _statement(resolved, self._context)
        self._session = _session(self._context)
        self._initial_phase_digest = derive_initial_phase_digest(
            self._session,
            self._statement,
            expectation.policy,
        )
        self._phase = InteractionPhase.INITIAL
        self._boundary_message: BoundaryMessage | None = None
        self._boundary_phase_digest: bytes | None = None
        self._q_message: ReplayChallengeMessage | None = None
        self._unit_commitments: UnitCommitmentsMessage | None = None
        self._s_message: SampleChallengeMessage | None = None

    @property
    def public_context(self) -> InteractivePublicContext:
        return self._context

    @property
    def phase(self) -> InteractionPhase:
        return self._phase

    def receive_boundary(
        self,
        message: BoundaryMessage,
    ) -> ReplayChallengeMessage:
        if self._phase is not InteractionPhase.INITIAL:
            raise InteractionError("boundary may only be received once")
        if not isinstance(message, BoundaryMessage):
            raise InteractionError("boundary message has the wrong type")
        domain = derive_boundary_commitment_domain(
            self._session,
            self._statement,
            self._expectation.policy,
            self._artifact.circuit,
            self._layout,
            self._initial_phase_digest,
        )
        required = public_io_positions(self._artifact, self._layout)
        try:
            self._value_backend.validate_commitment(
                domain,
                message.commitment,
                self._limits,
            )
            opened = self._value_backend.verify_openings(
                domain,
                message.commitment,
                message.public_io_openings,
                required,
                self._limits,
            )
        except CommitmentError as error:
            raise InteractionError(
                "boundary commitment or openings are invalid"
            ) from error
        expected: dict[int, bytes] = {}
        for port, value in zip(
            self._artifact.circuit.input_ports,
            self._statement.public_inputs,
            strict=True,
        ):
            expected[int(port.position)] = value
        for port, value in zip(
            self._artifact.circuit.output_ports,
            self._statement.claimed_outputs,
            strict=True,
        ):
            position = int(port.position)
            previous = expected.get(position)
            if previous is not None and previous != value:
                raise InteractionError(
                    "one public position has inconsistent input/output values"
                )
            expected[position] = value
        if dict(opened) != expected:
            raise InteractionError(
                "boundary openings do not equal public inputs and claimed outputs"
            )
        boundary_digest = derive_boundary_phase_digest(
            self._initial_phase_digest,
            message,
        )
        selected = derive_q_challenge(
            self._expectation.q_seed,
            boundary_digest,
            self._artifact.replay_partition,
            self._expectation.policy.q,
            self._limits,
        )
        phase_digest = derive_q_phase_digest(
            boundary_digest,
            self._expectation.q_seed,
            selected,
        )
        challenge = ReplayChallengeMessage(
            self._expectation.q_seed,
            boundary_digest,
            selected,
            phase_digest,
        )
        self._boundary_message = message
        self._boundary_phase_digest = boundary_digest
        self._q_message = challenge
        self._phase = InteractionPhase.REPLAY_CHALLENGED
        return challenge

    def receive_unit_commitments(
        self,
        message: UnitCommitmentsMessage,
    ) -> SampleChallengeMessage:
        if self._phase is not InteractionPhase.REPLAY_CHALLENGED:
            raise InteractionError(
                "unit commitments require an issued replay challenge"
            )
        if not isinstance(message, UnitCommitmentsMessage):
            raise InteractionError("unit commitments have the wrong type")
        if self._q_message is None:
            raise RuntimeError("replay challenge is unavailable")
        expected_indices = self._q_message.selected_replay_units
        actual_indices = tuple(item.replay_unit_index for item in message.commitments)
        if (
            message.q_phase_digest != self._q_message.phase_digest
            or actual_indices != expected_indices
        ):
            raise InteractionError(
                "unit commitments do not exactly cover selected replay units"
            )
        for item in message.commitments:
            domain = derive_unit_commitment_domain(
                self._session,
                self._statement,
                self._expectation.policy,
                self._artifact.circuit,
                self._layout,
                self._q_message.phase_digest,
                item.replay_unit_index,
            )
            try:
                self._value_backend.validate_commitment(
                    domain,
                    item.commitment,
                    self._limits,
                )
            except CommitmentError as error:
                raise InteractionError(
                    f"replay-unit commitment {item.replay_unit_index} is invalid"
                ) from error
        expected_phase = derive_unit_commitments_phase_digest(
            self._q_message.phase_digest,
            message.commitments,
        )
        if message.phase_digest != expected_phase:
            raise InteractionError("unit-commitment phase digest is invalid")
        selected = derive_s_challenge(
            self._expectation.s_seed,
            expected_phase,
            self._artifact.verification_partition,
            expected_indices,
            self._expectation.policy.s,
            self._limits,
        )
        sample_phase = derive_sample_phase_digest(
            expected_phase,
            self._expectation.s_seed,
            selected,
        )
        challenge = SampleChallengeMessage(
            self._expectation.s_seed,
            expected_phase,
            selected,
            sample_phase,
        )
        self._unit_commitments = message
        self._s_message = challenge
        self._phase = InteractionPhase.SAMPLE_CHALLENGED
        return challenge

    def receive_sample_evidence(
        self,
        message: SampleEvidenceMessage,
    ) -> InteractiveVerificationResult:
        if self._phase is not InteractionPhase.SAMPLE_CHALLENGED:
            raise InteractionError(
                "sample evidence requires an issued sample challenge"
            )
        if not isinstance(message, SampleEvidenceMessage):
            raise InteractionError("sample evidence has the wrong type")
        if (
            self._boundary_message is None
            or self._q_message is None
            or self._unit_commitments is None
            or self._s_message is None
        ):
            raise RuntimeError("verifier phase state is incomplete")
        transcript = StagedTranscript(
            session=self._session,
            statement=self._statement,
            policy=self._expectation.policy,
            boundary=self._boundary_message,
            replay_challenge=self._q_message,
            unit_commitments=self._unit_commitments,
            sample_challenge=self._s_message,
            sample_evidence=message,
        )
        data = encode_transcript(transcript)
        self._limits.enforce("max_transcript_bytes", len(data))
        report = verify_transcript_bytes(
            data,
            self._expectation,
            self._trust,
            self._limits,
        )
        self._phase = InteractionPhase.COMPLETE
        return InteractiveVerificationResult(transcript, data, report)


@dataclass(frozen=True, slots=True)
class InteractiveVerificationResult:
    transcript: StagedTranscript
    transcript_bytes: bytes
    report: VerificationReport

    @property
    def accepted(self) -> bool:
        return self.report.accepted


@dataclass(frozen=True, slots=True)
class InteractiveProtocolRun:
    result: InteractiveVerificationResult
    replayed_unit_indices: tuple[int, ...]
    replayed_gate_count: int
    replayed_cost: int

    @property
    def transcript(self) -> StagedTranscript:
        return self.result.transcript

    @property
    def transcript_bytes(self) -> bytes:
        return self.result.transcript_bytes

    @property
    def report(self) -> VerificationReport:
        return self.result.report

    @property
    def accepted(self) -> bool:
        return self.result.accepted


def run_interactive_protocol(
    artifact: ResolvedExecutableArtifact,
    expectation: VerificationExpectation,
    trust: TrustedVerificationContext,
    value_source: ValueSource | Mapping[int, object],
    *,
    commitment_backend: ValueCommitmentBackend | None = None,
    evidence_backend: SampleEvidenceBackend | None = None,
    replay_service: ReplayService | None = None,
    limits: VerificationLimits | None = None,
) -> InteractiveProtocolRun:
    """Execute all message turns using separate seed-owning session objects."""

    checked_limits = VerificationLimits() if limits is None else limits
    verifier = StagedVerifierSession(
        expectation,
        trust,
        limits=checked_limits,
    )
    prover = StagedProverSession(
        artifact,
        verifier.public_context,
        value_source,
        commitment_backend=commitment_backend,
        evidence_backend=evidence_backend,
        replay_service=replay_service,
        limits=checked_limits,
    )
    boundary = prover.commit_boundary()
    replay_challenge = verifier.receive_boundary(boundary)
    unit_commitments = prover.answer_replay_challenge(replay_challenge)
    sample_challenge = verifier.receive_unit_commitments(unit_commitments)
    evidence = prover.answer_sample_challenge(sample_challenge)
    result = verifier.receive_sample_evidence(evidence)
    return InteractiveProtocolRun(
        result=result,
        replayed_unit_indices=prover.replayed_unit_indices,
        replayed_gate_count=prover.replayed_gate_count,
        replayed_cost=prover.replayed_cost,
    )


__all__ = [
    "InteractionError",
    "InteractionPhase",
    "InteractiveProtocolRun",
    "InteractivePublicContext",
    "InteractiveVerificationResult",
    "StagedProverSession",
    "StagedVerifierSession",
    "run_interactive_protocol",
]
