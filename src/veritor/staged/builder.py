"""Transcript builder for the boundary -> q -> roots -> s protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from veritor.commitment import (
    CommitmentOwner,
    MerkleSha256Backend,
    ValueCommitmentBackend,
    ValueCommitmentProver,
    ValueOpening,
)
from veritor.core import (
    ArtifactKind,
    VerificationLimits,
    iter_domain,
    validate_circuit_contract,
    validate_compiled_result,
)

from .artifact import ResolvedExecutableArtifact
from .boundary import (
    CommitmentLayout,
    derive_commitment_ownership,
    validate_output_schemas,
)
from .challenge import derive_q_challenge, derive_s_challenge
from .evidence import (
    EvidenceOpeningProvider,
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
from .wire import encode_transcript


class TranscriptBuildError(StagedProtocolError):
    """The supplied values or builder services cannot form a transcript."""


@runtime_checkable
class ValueSource(Protocol):
    """Client-side source of claimed values by global circuit position."""

    def value_at(self, global_position: int) -> object: ...


@dataclass(frozen=True, slots=True)
class MappingValueSource:
    """Assignment-backed value source for the current milestone."""

    assignment: Mapping[int, object]

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, Mapping):
            raise TranscriptBuildError("assignment must be a mapping")

    def value_at(self, global_position: int) -> object:
        try:
            return self.assignment[global_position]
        except KeyError as error:
            raise TranscriptBuildError(
                f"value source has no value for position {global_position}"
            ) from error


@runtime_checkable
class ReplayService(Protocol):
    """Untrusted client-side interface for later per-unit replay."""

    def values_for_unit(
        self,
        artifact: ResolvedExecutableArtifact,
        replay_unit_index: int,
        interior_positions: tuple[int, ...],
        boundary_values: Mapping[int, object],
        source: ValueSource,
    ) -> Mapping[int, object]: ...


@dataclass(frozen=True, slots=True)
class AssignmentReplayService:
    """Read selected interiors directly from a complete assignment."""

    def values_for_unit(
        self,
        artifact: ResolvedExecutableArtifact,
        replay_unit_index: int,
        interior_positions: tuple[int, ...],
        boundary_values: Mapping[int, object],
        source: ValueSource,
    ) -> Mapping[int, object]:
        del artifact, replay_unit_index, boundary_values
        return {position: source.value_at(position) for position in interior_positions}


@dataclass(frozen=True, slots=True)
class ExecutingReplayService:
    """Recompute one selected replay unit from its committed boundary."""

    def values_for_unit(
        self,
        artifact: ResolvedExecutableArtifact,
        replay_unit_index: int,
        interior_positions: tuple[int, ...],
        boundary_values: Mapping[int, object],
        source: ValueSource,
    ) -> Mapping[int, object]:
        del source
        try:
            unit = artifact.replay_partition.unit_at(replay_unit_index)
        except KeyError as error:
            raise TranscriptBuildError("replay unit index is out of range") from error
        evaluate = getattr(artifact.relation_service, "evaluate", None)
        if not callable(evaluate):
            evaluate = getattr(artifact.circuit, "evaluate_relation", None)
        if not callable(evaluate):
            raise TranscriptBuildError(
                "executable circuit does not expose client-side replay semantics"
            )
        known = dict(boundary_values)
        requested = set(interior_positions)
        replayed: dict[int, object] = {}
        for position_value in iter_domain(unit.members):
            position = int(position_value)
            gate = artifact.circuit.executable_gate_at(position_value)
            try:
                arguments = tuple(known[int(item)] for item in gate.arguments)
            except KeyError as error:
                raise TranscriptBuildError(
                    f"replay unit {replay_unit_index} is missing predecessor "
                    f"{int(error.args[0])}"
                ) from error
            try:
                output = evaluate(str(gate.relation_id), arguments)
            except Exception as error:
                raise TranscriptBuildError(
                    f"replay relation failed at position {position}"
                ) from error
            if position in boundary_values and boundary_values[position] != output:
                raise TranscriptBuildError(
                    f"replay unit {replay_unit_index} disagrees with committed "
                    f"boundary position {position}"
                )
            known[position] = output
            if position in requested:
                replayed[position] = output
        if set(replayed) != requested:
            raise TranscriptBuildError(
                "replay unit did not produce its exact committed interior"
            )
        return replayed


class CommitmentOpeningTrees(EvidenceOpeningProvider):
    __slots__ = ("_trees",)

    def __init__(
        self,
        trees: Mapping[CommitmentOwner, ValueCommitmentProver],
    ) -> None:
        self._trees = dict(trees)

    def open(
        self,
        owner: CommitmentOwner,
        global_position: int,
    ) -> ValueOpening:
        tree = self._trees.get(owner)
        if tree is None:
            raise TranscriptBuildError(
                f"no commitment material exists for owner {owner}"
            )
        return tree.open(global_position)


def _source(value: ValueSource | Mapping[int, object]) -> ValueSource:
    if isinstance(value, ValueSource):
        return value
    if isinstance(value, Mapping):
        return MappingValueSource(value)
    raise TranscriptBuildError("value_source has the wrong type")


def canonical_encode_value(
    artifact: ResolvedExecutableArtifact,
    value_type: str,
    value: object,
) -> bytes:
    try:
        payload = artifact.value_service.encode(value_type, value)
        decoded = artifact.value_service.decode(value_type, payload)
        reencoded = artifact.value_service.encode(value_type, decoded)
    except Exception as error:
        raise TranscriptBuildError(
            f"value is invalid for schema {value_type!r}"
        ) from error
    if type(payload) is not bytes or reencoded != payload:
        raise TranscriptBuildError(
            f"value service is not canonical for schema {value_type!r}"
        )
    return payload


def build_public_statement(
    artifact: ResolvedExecutableArtifact,
    public_inputs: Sequence[object],
    claimed_outputs: Sequence[object],
) -> PublicStatement:
    circuit = artifact.circuit
    if len(public_inputs) != len(circuit.input_ports):
        raise TranscriptBuildError("public input count does not match the circuit")
    if len(claimed_outputs) != len(circuit.output_ports):
        raise TranscriptBuildError(
            "claimed output count does not match the ordered output view"
        )
    return PublicStatement(
        tuple(
            canonical_encode_value(
                artifact,
                str(port.value_type),
                value,
            )
            for port, value in zip(
                circuit.input_ports,
                public_inputs,
                strict=True,
            )
        ),
        tuple(
            canonical_encode_value(
                artifact,
                str(port.value_type),
                value,
            )
            for port, value in zip(
                circuit.output_ports,
                claimed_outputs,
                strict=True,
            )
        ),
    )


def public_io_positions(
    artifact: ResolvedExecutableArtifact,
    layout: CommitmentLayout,
) -> tuple[int, ...]:
    required = {
        *(int(port.position) for port in artifact.circuit.input_ports),
        *(int(port.position) for port in artifact.circuit.output_ports),
    }
    return tuple(
        int(position) for position in layout.boundary.items if int(position) in required
    )


@dataclass(frozen=True, slots=True)
class StagedProtocolBuilder:
    """Construct honest conformance transcripts in one process.

    This helper receives both verifier seeds up front, so it does not enforce
    the temporal secrecy required against an adversarial prover.  A real
    transport must reveal ``q_seed`` only after the boundary message is fixed
    and ``s_seed`` only after all selected-unit commitments are fixed.
    """

    artifact: ResolvedExecutableArtifact
    commitment_backend: ValueCommitmentBackend
    evidence_backend: SampleEvidenceBackend
    limits: VerificationLimits = field(default_factory=VerificationLimits)
    replay_service: ReplayService = field(default_factory=AssignmentReplayService)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ResolvedExecutableArtifact):
            raise TranscriptBuildError(
                "builder requires a resolved executable artifact"
            )
        if not isinstance(self.commitment_backend, ValueCommitmentBackend):
            raise TranscriptBuildError("commitment backend has the wrong type")
        if not isinstance(self.evidence_backend, SampleEvidenceBackend):
            raise TranscriptBuildError("evidence backend has the wrong type")
        if not isinstance(self.limits, VerificationLimits):
            raise TranscriptBuildError("limits must be VerificationLimits")
        if not isinstance(self.replay_service, ReplayService):
            raise TranscriptBuildError("replay service has the wrong type")

    def build(
        self,
        expectation: VerificationExpectation,
        value_source: ValueSource | Mapping[int, object],
        *,
        q_seed: bytes | None = None,
        s_seed: bytes | None = None,
    ) -> StagedTranscript:
        if not isinstance(expectation, VerificationExpectation):
            raise TranscriptBuildError("expectation has the wrong type")
        q_seed = expectation.q_seed if q_seed is None else q_seed
        s_seed = expectation.s_seed if s_seed is None else s_seed
        if type(q_seed) is not bytes or len(q_seed) != 32:
            raise TranscriptBuildError("builder requires a 32-byte q seed")
        if type(s_seed) is not bytes or len(s_seed) != 32:
            raise TranscriptBuildError("builder requires a 32-byte s seed")
        if q_seed != expectation.q_seed:
            raise TranscriptBuildError("q seed differs from verifier expectation")
        if s_seed != expectation.s_seed:
            raise TranscriptBuildError("s seed differs from verifier expectation")
        if (
            expectation.value_commitment_backend_id
            != self.commitment_backend.backend_id
        ):
            raise TranscriptBuildError("expectation names another commitment backend")
        if expectation.sample_evidence_backend_id != self.evidence_backend.backend_id:
            raise TranscriptBuildError("expectation names another evidence backend")

        self.limits.enforce(
            "max_positions",
            len(self.artifact.circuit.input_ports)
            + self.artifact.circuit.computed_positions.count,
        )
        self.limits.enforce(
            "max_units",
            max(
                self.artifact.replay_partition.unit_count,
                self.artifact.verification_partition.unit_count,
            ),
        )
        compiled = validate_compiled_result(
            self.artifact.circuit,
            self.artifact.replay_partition,
            self.artifact.verification_partition,
        )
        if str(compiled.digest) != expectation.compiled_result_digest:
            raise TranscriptBuildError("expectation names another compiled result")
        if (
            self.artifact.circuit.identity.artifact_kind
            is not ArtifactKind.EXECUTABLE_CIRCUIT
        ):
            raise TranscriptBuildError("artifact is not executable")
        validate_circuit_contract(self.artifact.circuit, exhaustive=True)
        validate_output_schemas(self.artifact.circuit)
        layout = derive_commitment_ownership(
            self.artifact.circuit,
            self.artifact.replay_partition,
            self.limits,
        )
        source = _source(value_source)
        statement = build_public_statement(
            self.artifact,
            expectation.public_inputs,
            expectation.claimed_outputs,
        )
        session = SessionParameters(
            session_id=expectation.session_id,
            compiled_result_digest=expectation.compiled_result_digest,
            policy_digest=str(expectation.policy.digest),
            value_commitment_backend_id=(expectation.value_commitment_backend_id),
            sample_evidence_backend_id=(expectation.sample_evidence_backend_id),
        )
        initial_digest = derive_initial_phase_digest(
            session,
            statement,
            expectation.policy,
        )

        boundary_domain = derive_boundary_commitment_domain(
            session,
            statement,
            expectation.policy,
            self.artifact.circuit,
            layout,
            initial_digest,
        )
        raw_boundary_values = {
            int(position): source.value_at(int(position))
            for position in layout.boundary.items
        }
        encoded_boundary_values = {
            position: canonical_encode_value(
                self.artifact,
                boundary_domain.schema_for(position),
                value,
            )
            for position, value in raw_boundary_values.items()
        }
        boundary_tree = self.commitment_backend.commit(
            boundary_domain,
            encoded_boundary_values,
            self.limits,
        )
        required_public_io = public_io_positions(self.artifact, layout)
        self.limits.enforce("max_openings", len(required_public_io))
        boundary_message = BoundaryMessage(
            boundary_tree.commitment,
            tuple(boundary_tree.open(position) for position in required_public_io),
        )
        boundary_phase_digest = derive_boundary_phase_digest(
            initial_digest,
            boundary_message,
        )
        selected_replay_units = derive_q_challenge(
            q_seed,
            boundary_phase_digest,
            self.artifact.replay_partition,
            expectation.policy.q,
            self.limits,
        )
        q_phase_digest = derive_q_phase_digest(
            boundary_phase_digest,
            q_seed,
            selected_replay_units,
        )
        replay_challenge = ReplayChallengeMessage(
            q_seed,
            boundary_phase_digest,
            selected_replay_units,
            q_phase_digest,
        )

        trees: dict[CommitmentOwner, ValueCommitmentProver] = {
            CommitmentOwner.boundary(): boundary_tree
        }
        unit_domains = {}
        owned_commitments: list[OwnedValueCommitment] = []
        for replay_unit_index in selected_replay_units:
            domain = derive_unit_commitment_domain(
                session,
                statement,
                expectation.policy,
                self.artifact.circuit,
                layout,
                q_phase_digest,
                replay_unit_index,
            )
            positions = tuple(int(item) for item in domain.positions.items)
            raw_values = self.replay_service.values_for_unit(
                self.artifact,
                replay_unit_index,
                positions,
                raw_boundary_values,
                source,
            )
            if not isinstance(raw_values, Mapping):
                raise TranscriptBuildError(
                    "replay service did not return a value mapping"
                )
            if set(raw_values) != set(positions) or any(
                type(position) is not int for position in raw_values
            ):
                raise TranscriptBuildError(
                    "replay service values must exactly cover the unit interior"
                )
            encoded_values = {
                position: canonical_encode_value(
                    self.artifact,
                    domain.schema_for(position),
                    raw_values[position],
                )
                for position in positions
            }
            tree = self.commitment_backend.commit(
                domain,
                encoded_values,
                self.limits,
            )
            owner = CommitmentOwner.replay_unit(replay_unit_index)
            trees[owner] = tree
            unit_domains[replay_unit_index] = domain
            owned_commitments.append(
                OwnedValueCommitment(replay_unit_index, tree.commitment)
            )

        owned_tuple = tuple(owned_commitments)
        unit_phase_digest = derive_unit_commitments_phase_digest(
            q_phase_digest,
            owned_tuple,
        )
        unit_message = UnitCommitmentsMessage(
            q_phase_digest,
            owned_tuple,
            unit_phase_digest,
        )
        selected_verification_units = derive_s_challenge(
            s_seed,
            unit_phase_digest,
            self.artifact.verification_partition,
            selected_replay_units,
            expectation.policy.s,
            self.limits,
        )
        sample_phase_digest = derive_sample_phase_digest(
            unit_phase_digest,
            s_seed,
            selected_verification_units,
        )
        sample_challenge = SampleChallengeMessage(
            s_seed,
            unit_phase_digest,
            selected_verification_units,
            sample_phase_digest,
        )

        opening_trees = CommitmentOpeningTrees(trees)
        commitments_by_unit = {
            item.replay_unit_index: item.commitment for item in owned_tuple
        }
        evidence: list[SampledUnitEvidence] = []
        for verification_unit_index in selected_verification_units:
            verification_unit = self.artifact.verification_partition.unit_at(
                verification_unit_index
            )
            replay_unit_index = int(verification_unit.replay_unit)
            instance = SampleVerificationInstance(
                artifact=self.artifact,
                layout=layout,
                session=session,
                statement=statement,
                policy=expectation.policy,
                sample_phase_digest=sample_phase_digest,
                verification_unit_index=verification_unit_index,
                boundary_domain=boundary_domain,
                boundary_commitment=boundary_tree.commitment,
                unit_domain=unit_domains[replay_unit_index],
                unit_commitment=commitments_by_unit[replay_unit_index],
                value_backend=self.commitment_backend,
                sample_evidence_backend_id=self.evidence_backend.backend_id,
            )
            payload = self.evidence_backend.build_evidence(
                instance,
                opening_trees,
                self.limits,
            )
            evidence.append(
                SampledUnitEvidence(
                    verification_unit_index,
                    self.evidence_backend.backend_id,
                    payload,
                )
            )

        transcript = StagedTranscript(
            session=session,
            statement=statement,
            policy=expectation.policy,
            boundary=boundary_message,
            replay_challenge=replay_challenge,
            unit_commitments=unit_message,
            sample_challenge=sample_challenge,
            sample_evidence=SampleEvidenceMessage(
                sample_phase_digest,
                tuple(evidence),
            ),
        )
        self.limits.enforce(
            "max_transcript_bytes",
            len(encode_transcript(transcript)),
        )
        return transcript

    def build_bytes(
        self,
        expectation: VerificationExpectation,
        value_source: ValueSource | Mapping[int, object],
        *,
        q_seed: bytes | None = None,
        s_seed: bytes | None = None,
    ) -> bytes:
        data = encode_transcript(
            self.build(
                expectation,
                value_source,
                q_seed=q_seed,
                s_seed=s_seed,
            )
        )
        self.limits.enforce("max_transcript_bytes", len(data))
        return data


StagedProtocolOrchestrator = StagedProtocolBuilder


def build_transcript(
    artifact: ResolvedExecutableArtifact,
    expectation: VerificationExpectation,
    value_source: ValueSource | Mapping[int, object],
    *,
    commitment_backend: ValueCommitmentBackend | None = None,
    evidence_backend: SampleEvidenceBackend | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
    replay_service: ReplayService | None = None,
) -> StagedTranscript:
    """Build an honest conformance transcript; not an interactive transport."""

    builder = StagedProtocolBuilder(
        artifact,
        MerkleSha256Backend() if commitment_backend is None else commitment_backend,
        TransparentLocalCheckBackend()
        if evidence_backend is None
        else evidence_backend,
        VerificationLimits() if limits is None else limits,
        AssignmentReplayService() if replay_service is None else replay_service,
    )
    return builder.build(
        expectation,
        value_source,
        q_seed=q_seed,
        s_seed=s_seed,
    )


TranscriptBuilder = StagedProtocolBuilder


def build_transcript_bytes(
    artifact: ResolvedExecutableArtifact,
    expectation: VerificationExpectation,
    value_source: ValueSource | Mapping[int, object],
    *,
    commitment_backend: ValueCommitmentBackend | None = None,
    evidence_backend: SampleEvidenceBackend | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    limits: VerificationLimits | None = None,
    replay_service: ReplayService | None = None,
) -> bytes:
    """Build canonical honest-run bytes; not enforce challenge reveal timing."""

    checked_limits = VerificationLimits() if limits is None else limits
    builder = StagedProtocolBuilder(
        artifact,
        MerkleSha256Backend() if commitment_backend is None else commitment_backend,
        TransparentLocalCheckBackend()
        if evidence_backend is None
        else evidence_backend,
        checked_limits,
        AssignmentReplayService() if replay_service is None else replay_service,
    )
    return builder.build_bytes(
        expectation,
        value_source,
        q_seed=q_seed,
        s_seed=s_seed,
    )
