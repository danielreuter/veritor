"""Pure deterministic staged-transcript verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from veritor.commitment import CommitmentError
from veritor.core import (
    ArtifactKind,
    ExecutableCircuit,
    InvalidArtifact,
    ResourceLimit,
    VerificationLimits,
    iter_domain,
    validate_circuit_contract,
    validate_compiled_result,
)

from ._json import NonCanonicalWireError, WireCodecError
from .artifact import ResolvedExecutableArtifact
from .boundary import (
    CommitmentLayout,
    derive_commitment_ownership,
    validate_output_schemas,
)
from .challenge import derive_q_challenge, derive_s_challenge
from .evidence import (
    EvidenceError,
    RelationRejected,
    SampleVerificationInstance,
    TrustedServiceFailure,
)
from .model import (
    STAGED_TRANSCRIPT_VERSION,
    PublicStatement,
    StagedProtocolError,
    StagedTranscript,
    VerificationCode,
    VerificationExpectation,
    VerificationReport,
    unsupported_execution_artifact,
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
from .wire import decode_transcript


@dataclass(frozen=True, slots=True)
class _Reject(Exception):
    code: VerificationCode
    detail: str


def _reject(code: VerificationCode, detail: str) -> NoReturn:
    raise _Reject(code, detail)


def _canonical_encode(
    artifact: ResolvedExecutableArtifact,
    value_type: str,
    value: object,
) -> bytes:
    try:
        payload = artifact.value_service.encode(value_type, value)
        decoded = artifact.value_service.decode(value_type, payload)
        canonical = artifact.value_service.encode(value_type, decoded)
    except Exception:  # noqa: BLE001 - trusted plug-ins may raise any exception
        _reject(
            VerificationCode.TRUSTED_SERVICE_FAILURE,
            "trusted value service could not encode the expected statement",
        )
    if type(payload) is not bytes or payload != canonical:
        _reject(
            VerificationCode.TRUSTED_SERVICE_FAILURE,
            "trusted value service is not canonical",
        )
    return payload


def _expected_statement(
    artifact: ResolvedExecutableArtifact,
    expectation: VerificationExpectation,
) -> PublicStatement:
    circuit = artifact.circuit
    if len(expectation.public_inputs) != len(circuit.input_ports):
        _reject(
            VerificationCode.EXPECTATION_MISMATCH,
            "expected public input count does not match the circuit",
        )
    if len(expectation.claimed_outputs) != len(circuit.output_ports):
        _reject(
            VerificationCode.EXPECTATION_MISMATCH,
            "expected output count does not match the ordered circuit outputs",
        )
    return PublicStatement(
        tuple(
            _canonical_encode(artifact, str(port.value_type), value)
            for port, value in zip(
                circuit.input_ports,
                expectation.public_inputs,
                strict=True,
            )
        ),
        tuple(
            _canonical_encode(artifact, str(port.value_type), value)
            for port, value in zip(
                circuit.output_ports,
                expectation.claimed_outputs,
                strict=True,
            )
        ),
    )


def _public_io_positions(
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


def _validate_executable_artifact(
    artifact: ResolvedExecutableArtifact,
    expected_digest: str,
    limits: VerificationLimits,
) -> CommitmentLayout:
    try:
        limits.enforce(
            "max_positions",
            len(artifact.circuit.input_ports)
            + artifact.circuit.computed_positions.count,
        )
        limits.enforce(
            "max_units",
            max(
                artifact.replay_partition.unit_count,
                artifact.verification_partition.unit_count,
            ),
        )
        for replay_unit in artifact.replay_partition.units:
            limits.enforce("max_positions_per_unit", replay_unit.count)
        for verification_unit in artifact.verification_partition.units:
            limits.enforce("max_positions_per_unit", verification_unit.count)
        identity = validate_compiled_result(
            artifact.circuit,
            artifact.replay_partition,
            artifact.verification_partition,
        )
        if str(identity.digest) != expected_digest:
            _reject(
                VerificationCode.INVALID_COMPILED_RESULT,
                "resolved compiled tuple has another identity",
            )
        validate_circuit_contract(artifact.circuit, exhaustive=True)
        validate_output_schemas(artifact.circuit)
        for position in iter_domain(artifact.circuit.computed_positions):
            structural = artifact.circuit.gate_at(position)
            limits.enforce(
                "max_positions_per_unit",
                len(structural.predecessors),
            )
            executable = artifact.circuit.executable_gate_at(position)
            if (
                executable.position != position
                or executable.operation != structural.operation
                or executable.arguments != structural.predecessors
                or (
                    structural.value_type is not None
                    and executable.output_type != structural.value_type
                )
            ):
                _reject(
                    VerificationCode.INVALID_COMPILED_RESULT,
                    "resolved structural and executable gate views disagree",
                )
        return derive_commitment_ownership(
            artifact.circuit,
            artifact.replay_partition,
            limits,
        )
    except _Reject:
        raise
    except ResourceLimit:
        raise
    except (InvalidArtifact, AttributeError, TypeError, KeyError, IndexError):
        _reject(
            VerificationCode.INVALID_COMPILED_RESULT,
            "resolved compiled tuple violates the executable contract",
        )


def _verify(
    transcript: StagedTranscript,
    expectation: VerificationExpectation,
    trust: TrustedVerificationContext,
    limits: VerificationLimits,
    data: bytes,
) -> VerificationReport:
    session = transcript.session
    if (
        transcript.version != STAGED_TRANSCRIPT_VERSION
        or session.protocol_version != STAGED_TRANSCRIPT_VERSION
        or session.session_id != expectation.session_id
        or session.compiled_result_digest != expectation.compiled_result_digest
        or transcript.policy != expectation.policy
        or session.policy_digest != expectation.policy.digest
        or session.value_commitment_backend_id
        != expectation.value_commitment_backend_id
        or session.sample_evidence_backend_id != expectation.sample_evidence_backend_id
    ):
        _reject(
            VerificationCode.EXPECTATION_MISMATCH,
            "transcript session, identity, policy, or backend differs from expectation",
        )
    if (
        transcript.replay_challenge.seed != expectation.q_seed
        or transcript.sample_challenge.seed != expectation.s_seed
    ):
        _reject(
            VerificationCode.EXPECTATION_MISMATCH,
            "phase seed differs from verifier expectation",
        )

    try:
        resolved = trust.artifact_resolver.resolve(expectation.compiled_result_digest)
    except Exception:  # noqa: BLE001 - resolver is a trusted plug-in boundary
        _reject(
            VerificationCode.TRUSTED_SERVICE_FAILURE,
            "trusted artifact resolver failed",
        )
    if resolved is None:
        _reject(
            VerificationCode.ARTIFACT_NOT_FOUND,
            "no trusted artifact resolves the compiled result identity",
        )

    kind = resolved.circuit.identity.artifact_kind
    if (
        kind is not ArtifactKind.EXECUTABLE_CIRCUIT
        or not isinstance(resolved, ResolvedExecutableArtifact)
        or not isinstance(resolved.circuit, ExecutableCircuit)
    ):
        if kind is not ArtifactKind.CAPACITY_PROFILE:
            try:
                limits.enforce(
                    "max_positions",
                    len(resolved.circuit.input_ports)
                    + resolved.circuit.computed_positions.count,
                )
                limits.enforce(
                    "max_units",
                    max(
                        resolved.replay_partition.unit_count,
                        resolved.verification_partition.unit_count,
                    ),
                )
                resolved_identity = validate_compiled_result(
                    resolved.circuit,
                    resolved.replay_partition,
                    resolved.verification_partition,
                )
            except (InvalidArtifact, AttributeError, TypeError):
                _reject(
                    VerificationCode.INVALID_COMPILED_RESULT,
                    "resolved non-executable compiled tuple is invalid",
                )
            if str(resolved_identity.digest) != expectation.compiled_result_digest:
                _reject(
                    VerificationCode.INVALID_COMPILED_RESULT,
                    "resolved non-executable tuple has another identity",
                )
        outcome = unsupported_execution_artifact(
            artifact_kind=kind,
            detail=(
                "the resolved artifact does not supply trusted executable "
                "local relations"
            ),
        )
        return VerificationReport.unsupported_artifact(outcome, data)
    artifact = resolved

    layout = _validate_executable_artifact(
        artifact,
        expectation.compiled_result_digest,
        limits,
    )
    expected_statement = _expected_statement(artifact, expectation)
    if transcript.statement != expected_statement:
        _reject(
            VerificationCode.EXPECTATION_MISMATCH,
            "transcript public statement differs from expectation",
        )

    value_backend = trust.value_commitment_backends.resolve(
        session.value_commitment_backend_id
    )
    if value_backend is None:
        _reject(
            VerificationCode.UNKNOWN_COMMITMENT_BACKEND,
            "commitment backend is not locally trusted",
        )
    evidence_backend = trust.sample_evidence_backends.resolve(
        session.sample_evidence_backend_id
    )
    if evidence_backend is None:
        _reject(
            VerificationCode.UNKNOWN_EVIDENCE_BACKEND,
            "sample evidence backend is not locally trusted",
        )

    initial_digest = derive_initial_phase_digest(
        session,
        transcript.statement,
        transcript.policy,
    )
    boundary_domain = derive_boundary_commitment_domain(
        session,
        transcript.statement,
        transcript.policy,
        artifact.circuit,
        layout,
        initial_digest,
    )
    try:
        value_backend.validate_commitment(
            boundary_domain,
            transcript.boundary.commitment,
            limits,
        )
    except CommitmentError:
        _reject(
            VerificationCode.INVALID_COMMITMENT,
            "boundary commitment has invalid shape or binding",
        )

    public_io_positions = _public_io_positions(artifact, layout)
    try:
        opened_io = value_backend.verify_openings(
            boundary_domain,
            transcript.boundary.commitment,
            transcript.boundary.public_io_openings,
            public_io_positions,
            limits,
        )
    except CommitmentError:
        _reject(
            VerificationCode.INVALID_OPENING,
            "public-I/O openings are invalid, duplicated, missing, or extra",
        )
    for port, expected in zip(
        artifact.circuit.input_ports,
        transcript.statement.public_inputs,
        strict=True,
    ):
        if opened_io[int(port.position)] != expected:
            _reject(
                VerificationCode.PUBLIC_IO_MISMATCH,
                "an authenticated input differs from the public input",
            )
    for port, expected in zip(
        artifact.circuit.output_ports,
        transcript.statement.claimed_outputs,
        strict=True,
    ):
        # Iterating the ordered output view intentionally preserves duplicates.
        if opened_io[int(port.position)] != expected:
            _reject(
                VerificationCode.PUBLIC_IO_MISMATCH,
                "an authenticated output differs from the ordered claim",
            )

    boundary_phase_digest = derive_boundary_phase_digest(
        initial_digest,
        transcript.boundary,
    )
    replay_message = transcript.replay_challenge
    if replay_message.boundary_phase_digest != boundary_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "q challenge does not bind the boundary phase",
        )
    selected_replay_units = derive_q_challenge(
        replay_message.seed,
        boundary_phase_digest,
        artifact.replay_partition,
        transcript.policy.q,
        limits,
    )
    if replay_message.selected_replay_units != selected_replay_units:
        _reject(
            VerificationCode.CHALLENGE_MISMATCH,
            "selected replay units do not equal the exact q derivation",
        )
    q_phase_digest = derive_q_phase_digest(
        boundary_phase_digest,
        replay_message.seed,
        selected_replay_units,
    )
    if replay_message.phase_digest != q_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "q phase digest is invalid",
        )

    unit_message = transcript.unit_commitments
    if unit_message.q_phase_digest != q_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "unit commitments do not bind the q phase",
        )
    actual_unit_indices = tuple(
        item.replay_unit_index for item in unit_message.commitments
    )
    if actual_unit_indices != selected_replay_units:
        _reject(
            VerificationCode.COVERAGE_MISMATCH,
            "unit commitments do not exactly cover ordered J",
        )

    unit_domains = {}
    unit_commitments = {}
    for item in unit_message.commitments:
        domain = derive_unit_commitment_domain(
            session,
            transcript.statement,
            transcript.policy,
            artifact.circuit,
            layout,
            q_phase_digest,
            item.replay_unit_index,
        )
        try:
            value_backend.validate_commitment(
                domain,
                item.commitment,
                limits,
            )
        except CommitmentError:
            _reject(
                VerificationCode.INVALID_COMMITMENT,
                "a selected replay-unit commitment is invalid",
            )
        unit_domains[item.replay_unit_index] = domain
        unit_commitments[item.replay_unit_index] = item.commitment

    unit_phase_digest = derive_unit_commitments_phase_digest(
        q_phase_digest,
        unit_message.commitments,
    )
    if unit_message.phase_digest != unit_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "ordered selected-unit roots have an invalid phase digest",
        )
    sample_message = transcript.sample_challenge
    if sample_message.unit_commitments_phase_digest != unit_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "s challenge does not bind the ordered selected-unit roots",
        )
    selected_verification_units = derive_s_challenge(
        sample_message.seed,
        unit_phase_digest,
        artifact.verification_partition,
        selected_replay_units,
        transcript.policy.s,
        limits,
    )
    if sample_message.selected_verification_units != selected_verification_units:
        _reject(
            VerificationCode.CHALLENGE_MISMATCH,
            "sampled units do not equal the exact s derivation",
        )
    sample_phase_digest = derive_sample_phase_digest(
        unit_phase_digest,
        sample_message.seed,
        selected_verification_units,
    )
    if sample_message.phase_digest != sample_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "sample phase digest is invalid",
        )

    evidence_message = transcript.sample_evidence
    if evidence_message.sample_phase_digest != sample_phase_digest:
        _reject(
            VerificationCode.INVALID_PHASE,
            "sample evidence does not bind the sample phase",
        )
    evidence_indices = tuple(
        item.verification_unit_index for item in evidence_message.units
    )
    if evidence_indices != selected_verification_units:
        _reject(
            VerificationCode.COVERAGE_MISMATCH,
            "sample evidence does not exactly cover ordered T",
        )
    if any(
        item.backend_id != evidence_backend.backend_id
        for item in evidence_message.units
    ):
        _reject(
            VerificationCode.UNKNOWN_EVIDENCE_BACKEND,
            "sample evidence envelope names another backend",
        )

    for evidence in evidence_message.units:
        unit = artifact.verification_partition.unit_at(evidence.verification_unit_index)
        replay_unit_index = int(unit.replay_unit)
        # T can only contain units whose replay owner is in J.  Keeping this
        # explicit avoids any dictionary-key shortcut on malformed evidence.
        if (
            replay_unit_index not in unit_domains
            or replay_unit_index not in unit_commitments
        ):
            _reject(
                VerificationCode.COVERAGE_MISMATCH,
                "sampled unit has no selected replay-unit commitment",
            )
        instance = SampleVerificationInstance(
            artifact=artifact,
            layout=layout,
            session=session,
            statement=transcript.statement,
            policy=transcript.policy,
            sample_phase_digest=sample_phase_digest,
            verification_unit_index=evidence.verification_unit_index,
            boundary_domain=boundary_domain,
            boundary_commitment=transcript.boundary.commitment,
            unit_domain=unit_domains[replay_unit_index],
            unit_commitment=unit_commitments[replay_unit_index],
            value_backend=value_backend,
            sample_evidence_backend_id=evidence_backend.backend_id,
        )
        try:
            evidence_backend.verify_evidence(
                instance,
                evidence.payload,
                limits,
            )
        except RelationRejected:
            _reject(
                VerificationCode.RELATION_REJECTED,
                "an authenticated sampled local relation is false",
            )
        except TrustedServiceFailure:
            _reject(
                VerificationCode.TRUSTED_SERVICE_FAILURE,
                "trusted relation service failed",
            )
        except (EvidenceError, CommitmentError):
            _reject(
                VerificationCode.INVALID_EVIDENCE,
                "sample evidence is malformed, inauthentic, or incomplete",
            )
    return VerificationReport.accept(data)


def verify_transcript_bytes(
    data: bytes,
    expectation: VerificationExpectation,
    trust: TrustedVerificationContext,
    limits: VerificationLimits,
) -> VerificationReport:
    """Verify serialized evidence using only expectation and local trust.

    The function is stateless: repeated calls with the same bytes and trusted
    services produce the same report.  Untrusted transcript failures return
    stable reports rather than escaping exceptions.
    """

    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    if not isinstance(expectation, VerificationExpectation):
        raise TypeError("expectation must be VerificationExpectation")
    if not isinstance(trust, TrustedVerificationContext):
        raise TypeError("trust must be TrustedVerificationContext")
    if not isinstance(limits, VerificationLimits):
        raise TypeError("limits must be VerificationLimits")
    try:
        limits.enforce("max_transcript_bytes", len(data))
        transcript = decode_transcript(data, limits)
        return _verify(transcript, expectation, trust, limits, data)
    except NonCanonicalWireError:
        return VerificationReport.reject(
            VerificationCode.NONCANONICAL_TRANSCRIPT,
            "transcript is not in its unique canonical encoding",
            data,
        )
    except ResourceLimit:
        return VerificationReport.reject(
            VerificationCode.RESOURCE_LIMIT,
            "verification resource limit exceeded",
            data,
        )
    except _Reject as rejection:
        return VerificationReport.reject(
            rejection.code,
            rejection.detail,
            data,
        )
    except (WireCodecError, StagedProtocolError, CommitmentError, ValueError):
        return VerificationReport.reject(
            VerificationCode.MALFORMED_TRANSCRIPT,
            "transcript is malformed",
            data,
        )
    except Exception:  # noqa: BLE001 - pure API converts plug-in failures
        return VerificationReport.reject(
            VerificationCode.TRUSTED_SERVICE_FAILURE,
            "trusted verification service failed",
            data,
        )
