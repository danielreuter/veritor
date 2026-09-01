"""Interactive prover and verifier sessions for the two-stage protocol.

The verifier owns both seeds and releases each challenge only after the
message it depends on has been received and checked.  Every lookup the
verifier performs is against the trusted :class:`CompiledArtifact`; the prover
never tells the verifier where a value lives or which positions a unit has.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from secrets import token_bytes

from veritor.core import (
    CompiledArtifact,
    ExecutableCircuit,
    VerificationLimits,
    VerificationPolicy,
    iter_domain,
)

from .challenge import derive_replay_selection, derive_sample_selection
from .merkle import CommitmentDomain, MerkleTree, validate_commitment, verify_opening
from .messages import (
    BoundaryMessage,
    Commitment,
    EvidenceMessage,
    Header,
    InteriorMessage,
    Opening,
    ProtocolError,
    Reject,
    ReplayChallenge,
    SampleChallenge,
    Transcript,
    VerificationCode,
    VerificationReport,
)
from .phases import (
    BOUNDARY_OWNER,
    boundary_domain,
    boundary_phase,
    interior_domain,
    interior_phase,
    replay_phase,
    sample_phase,
)

type Values = Mapping[int, object]
type Replay = Callable[[int, Values], Values]
"""``replay(unit, boundary_values) -> interior values`` for one replay unit."""


@dataclass(frozen=True, slots=True)
class Expectation:
    """What the verifier expects and the randomness it owns.

    ``public_inputs`` and ``claimed_outputs`` are plain values; the verifier
    encodes them with the circuit's canonical codec.  Both seeds are mandatory
    so a verifier can never accidentally let the prover choose them.
    """

    session_id: bytes
    compiled_digest: str
    policy: VerificationPolicy
    public_inputs: tuple[object, ...]
    claimed_outputs: tuple[object, ...]
    q_seed: bytes
    s_seed: bytes

    def __post_init__(self) -> None:
        for name in ("q_seed", "s_seed"):
            value = getattr(self, name)
            if type(value) is not bytes or len(value) != 32:
                raise ProtocolError(f"expected {name.replace('_', ' ')} of 32 bytes")


def make_expectation(
    artifact: CompiledArtifact,
    policy: VerificationPolicy,
    public_inputs: Iterable[object],
    claimed_outputs: Iterable[object],
    *,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
) -> Expectation:
    """Build an expectation, drawing fresh seeds unless given."""

    return Expectation(
        session_id=token_bytes(16) if session_id is None else session_id,
        compiled_digest=artifact.identity.digest,
        policy=policy,
        public_inputs=tuple(public_inputs),
        claimed_outputs=tuple(claimed_outputs),
        q_seed=token_bytes(32) if q_seed is None else q_seed,
        s_seed=token_bytes(32) if s_seed is None else s_seed,
    )


class _Layout:
    """Position-level lookups shared by both sessions; all ``O(log n)``."""

    __slots__ = ("_input_types", "artifact", "circuit")

    def __init__(self, artifact: CompiledArtifact) -> None:
        if not isinstance(artifact, CompiledArtifact):
            raise ProtocolError("sessions require a CompiledArtifact")
        if not isinstance(artifact.circuit, ExecutableCircuit):
            raise ProtocolError("the protocol requires an executable circuit")
        self.artifact = artifact
        self.circuit: ExecutableCircuit = artifact.circuit
        self._input_types = {
            int(port.position): str(port.value_type)
            for port in self.circuit.input_ports
        }

    def schema(self, position: int) -> str:
        value_type = self._input_types.get(position)
        if value_type is None:
            return str(self.circuit.executable_gate_at(position).output_type)
        return value_type

    def io_positions(self) -> tuple[int, ...]:
        """Distinct public I/O positions in boundary rank order."""

        positions = {int(port.position) for port in self.circuit.input_ports}
        positions.update(int(port.position) for port in self.circuit.output_ports)
        return tuple(sorted(positions, key=self.artifact.boundary.rank))

    def required(self, unit: int) -> tuple[tuple[int, int], ...]:
        """``(owner, position)`` for every value a verification unit reads or writes."""

        replay_unit = int(self.artifact.verification.unit_at(unit).replay_unit)
        positions: set[int] = set()
        for member in iter_domain(self.artifact.verification.unit_at(unit).members):
            positions.add(int(member))
            positions.update(
                int(item) for item in self.circuit.executable_gate_at(member).arguments
            )
        result: list[tuple[int, int]] = []
        for position in sorted(positions):
            owner = self.artifact.value_owner(position)
            if owner not in (BOUNDARY_OWNER, replay_unit):
                raise Reject(
                    VerificationCode.INVALID_COMPILED_RESULT,
                    f"position {position} is read by unit {unit} but owned by "
                    f"replay unit {owner} and is not a boundary position",
                )
            result.append((owner, position))
        return tuple(result)


def replay_unit(
    artifact: CompiledArtifact, unit: int, boundary_values: Values
) -> dict[int, object]:
    """Honest replay: recompute ``Int(unit)`` from the boundary, in position order."""

    circuit = artifact.circuit
    if not isinstance(circuit, ExecutableCircuit):
        raise ProtocolError("replay requires an executable circuit")
    known: dict[int, object] = {}
    interior = artifact.interior(unit)
    for position in iter_domain(artifact.replay.unit_at(unit).members):
        if not interior.contains(position):
            continue
        gate = circuit.executable_gate_at(position)
        arguments = []
        for argument in gate.arguments:
            if argument in known:
                arguments.append(known[argument])
            else:
                try:
                    arguments.append(boundary_values[int(argument)])
                except KeyError as error:
                    raise ProtocolError(
                        f"replay of unit {unit} needs boundary value {argument}"
                    ) from error
        known[int(position)] = circuit.evaluate_relation(
            str(gate.relation_id), tuple(arguments)
        )
    return known


def assignment_replay(values: Values) -> Replay:
    """A replay that reads interiors from a fixed (possibly dishonest) assignment."""

    def replay(unit: int, boundary_values: Values) -> Values:
        del boundary_values
        return values

    return replay


class ProverSession:
    """The prover's side.  Call ``boundary``, ``interiors``, ``evidence`` in order."""

    def __init__(
        self,
        artifact: CompiledArtifact,
        header: Header,
        values: Values,
        *,
        replay: Replay | None = None,
        limits: VerificationLimits | None = None,
    ) -> None:
        self._layout = _Layout(artifact)
        if header.compiled_digest != artifact.identity.digest:
            raise ProtocolError("header names a different compiled artifact")
        self.header = header
        self._values = values
        self._replay = replay
        self._limits = VerificationLimits() if limits is None else limits
        self._trees: dict[int, MerkleTree] = {}
        self._phase = "boundary"
        self._boundary_phase = b""
        self._replay_phase = b""
        self._interior_phase = b""
        self.transcript_parts: list[object] = [header]

    def _expect(self, phase: str) -> None:
        if self._phase != phase:
            raise ProtocolError(f"prover is in phase {self._phase!r}, not {phase!r}")

    def _commit(self, domain: CommitmentDomain, values: Values) -> MerkleTree:
        layout = self._layout
        encoded: dict[int, bytes] = {}
        schemas: dict[int, str] = {}
        for position in iter_domain(domain.positions):
            try:
                value = values[int(position)]
            except KeyError as error:
                raise ProtocolError(
                    f"prover has no value for position {position}"
                ) from error
            schema = layout.schema(position)
            schemas[int(position)] = schema
            encoded[int(position)] = layout.circuit.encode_value(schema, value)
        tree = MerkleTree(domain, encoded, schemas.__getitem__)
        self._trees[domain.owner] = tree
        return tree

    def boundary(self) -> BoundaryMessage:
        self._expect("boundary")
        tree = self._commit(
            boundary_domain(self.header, self._layout.artifact), self._values
        )
        message = BoundaryMessage(
            tree.commitment, tuple(tree.open(p) for p in self._layout.io_positions())
        )
        self._boundary_phase = boundary_phase(self.header, message)
        self._phase = "interiors"
        self.transcript_parts.append(message)
        return message

    def interiors(self, challenge: ReplayChallenge) -> InteriorMessage:
        self._expect("interiors")
        self._replay_phase = replay_phase(self._boundary_phase, challenge)
        artifact = self._layout.artifact
        commitments: list[Commitment] = []
        for unit in challenge.selected:
            if unit >= artifact.replay.unit_count:
                raise ProtocolError(f"challenge names unknown replay unit {unit}")
            interior_values = (
                replay_unit(artifact, unit, self._values)
                if self._replay is None
                else self._replay(unit, self._values)
            )
            domain = interior_domain(self.header, self._replay_phase, artifact, unit)
            commitments.append(self._commit(domain, interior_values).commitment)
        message = InteriorMessage(tuple(commitments))
        self._interior_phase = interior_phase(self._replay_phase, message)
        self._phase = "evidence"
        self.transcript_parts.extend((challenge, message))
        return message

    def evidence(self, challenge: SampleChallenge) -> EvidenceMessage:
        self._expect("evidence")
        sample_phase(self._interior_phase, challenge)
        batches: list[tuple[Opening, ...]] = []
        for unit in challenge.selected:
            openings: list[Opening] = []
            for owner, position in self._layout.required(unit):
                tree = self._trees.get(owner)
                if tree is None:
                    raise ProtocolError(
                        f"sampled unit {unit} needs uncommitted replay unit {owner}"
                    )
                openings.append(tree.open(position))
            batches.append(tuple(openings))
        message = EvidenceMessage(tuple(batches))
        self._phase = "done"
        self.transcript_parts.extend((challenge, message))
        return message

    @property
    def transcript(self) -> Transcript:
        if self._phase != "done":
            raise ProtocolError("the protocol has not finished")
        return Transcript(*self.transcript_parts)  # type: ignore[arg-type]


class VerifierSession:
    """The verifier's side.  Feed messages in order; each returns the next challenge."""

    def __init__(
        self,
        expectation: Expectation,
        artifact: CompiledArtifact,
        *,
        limits: VerificationLimits | None = None,
    ) -> None:
        self._layout = _Layout(artifact)
        if expectation.compiled_digest != artifact.identity.digest:
            raise ProtocolError("expectation names a different compiled artifact")
        self._expectation = expectation
        self._limits = VerificationLimits() if limits is None else limits
        circuit = self._layout.circuit
        try:
            inputs = tuple(
                circuit.encode_value(str(port.value_type), value)
                for port, value in zip(
                    circuit.input_ports, expectation.public_inputs, strict=True
                )
            )
            outputs = tuple(
                circuit.encode_value(str(port.value_type), value)
                for port, value in zip(
                    circuit.output_ports, expectation.claimed_outputs, strict=True
                )
            )
        except Exception as error:
            raise ProtocolError(
                "expectation values do not encode canonically"
            ) from error
        self.header = Header(
            expectation.session_id,
            artifact.identity.digest,
            expectation.policy,
            inputs,
            outputs,
        )
        self._commitments: dict[int, tuple[CommitmentDomain, Commitment]] = {}
        self._phase = "boundary"
        self._boundary_phase = b""
        self._replay_phase = b""
        self._interior_phase = b""
        self.selected_replay_units: tuple[int, ...] = ()
        self.selected_verification_units: tuple[int, ...] = ()
        self.transcript_parts: list[object] = [self.header]

    def _expect(self, phase: str) -> None:
        if self._phase != phase:
            raise Reject(
                VerificationCode.INVALID_PHASE,
                f"verifier is in phase {self._phase!r}, not {phase!r}",
            )

    def _reject(self, code: VerificationCode, detail: str) -> Reject:
        self._phase = "rejected"
        return Reject(code, detail)

    def _accept_commitment(
        self, domain: CommitmentDomain, commitment: Commitment
    ) -> None:
        self._limits.enforce("max_positions", domain.count)
        if not validate_commitment(domain, commitment):
            raise self._reject(
                VerificationCode.INVALID_COMMITMENT,
                f"commitment for owner {domain.owner} does not fit its domain",
            )
        self._commitments[domain.owner] = (domain, commitment)

    def _open(self, owner: int, opening: Opening) -> bytes:
        try:
            domain, commitment = self._commitments[owner]
        except KeyError:
            raise self._reject(
                VerificationCode.INVALID_OPENING, f"no commitment for owner {owner}"
            ) from None
        if not verify_opening(
            domain,
            commitment,
            opening,
            self._layout.schema(opening.position),
            self._limits,
        ):
            raise self._reject(
                VerificationCode.INVALID_OPENING,
                f"opening of position {opening.position} failed under owner {owner}",
            )
        return opening.value

    def receive_boundary(self, message: BoundaryMessage) -> ReplayChallenge:
        self._expect("boundary")
        artifact = self._layout.artifact
        self._accept_commitment(
            boundary_domain(self.header, artifact), message.commitment
        )
        expected = self._layout.io_positions()
        if tuple(item.position for item in message.io_openings) != expected:
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                "public I/O openings must cover exactly the I/O positions in boundary order",
            )
        opened = {
            item.position: self._open(BOUNDARY_OWNER, item)
            for item in message.io_openings
        }
        circuit = self._layout.circuit
        for ports, values, label in (
            (circuit.input_ports, self.header.public_inputs, "input"),
            (circuit.output_ports, self.header.claimed_outputs, "output"),
        ):
            for port, value in zip(ports, values, strict=True):
                if opened[int(port.position)] != value:
                    raise self._reject(
                        VerificationCode.PUBLIC_IO_MISMATCH,
                        f"{label} {port.name!r} at position {port.position} differs",
                    )
        self._boundary_phase = boundary_phase(self.header, message)
        selected = derive_replay_selection(
            self._expectation.q_seed,
            self._boundary_phase,
            artifact,
            self.header.policy,
            self._limits,
        )
        challenge = ReplayChallenge(self._expectation.q_seed, selected)
        self._replay_phase = replay_phase(self._boundary_phase, challenge)
        self.selected_replay_units = selected
        self._phase = "interiors"
        self.transcript_parts.extend((message, challenge))
        return challenge

    def receive_interiors(self, message: InteriorMessage) -> SampleChallenge:
        self._expect("interiors")
        artifact = self._layout.artifact
        selected = self.selected_replay_units
        if len(message.commitments) != len(selected):
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                f"expected {len(selected)} interior commitments, got "
                f"{len(message.commitments)}",
            )
        for unit, commitment in zip(selected, message.commitments, strict=True):
            self._accept_commitment(
                interior_domain(self.header, self._replay_phase, artifact, unit),
                commitment,
            )
        self._interior_phase = interior_phase(self._replay_phase, message)
        sampled = derive_sample_selection(
            self._expectation.s_seed,
            self._interior_phase,
            artifact,
            selected,
            self.header.policy,
            self._limits,
        )
        challenge = SampleChallenge(self._expectation.s_seed, sampled)
        sample_phase(self._interior_phase, challenge)
        self.selected_verification_units = sampled
        self._phase = "evidence"
        self.transcript_parts.extend((message, challenge))
        return challenge

    def receive_evidence(self, message: EvidenceMessage) -> VerificationReport:
        self._expect("evidence")
        sampled = self.selected_verification_units
        if len(message.units) != len(sampled):
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                f"expected evidence for {len(sampled)} units, got {len(message.units)}",
            )
        layout = self._layout
        circuit = layout.circuit
        opened_total = 0
        for unit, batch in zip(sampled, message.units, strict=True):
            required = layout.required(unit)
            opened_total += len(batch)
            self._limits.enforce("max_openings", opened_total)
            if tuple(item.position for item in batch) != tuple(p for _, p in required):
                raise self._reject(
                    VerificationCode.COVERAGE_MISMATCH,
                    f"evidence for unit {unit} must open exactly its required positions",
                )
            values: dict[int, object] = {}
            for (owner, position), opening in zip(required, batch, strict=True):
                payload = self._open(owner, opening)
                try:
                    values[position] = circuit.decode_value(
                        layout.schema(position), payload
                    )
                except Exception as error:
                    raise self._reject(
                        VerificationCode.INVALID_VALUE,
                        f"value at position {position} is not canonical: {error}",
                    ) from error
            for member in iter_domain(
                layout.artifact.verification.unit_at(unit).members
            ):
                gate = circuit.executable_gate_at(member)
                try:
                    satisfied = circuit.check_relation(
                        str(gate.relation_id),
                        tuple(values[int(item)] for item in gate.arguments),
                        values[int(member)],
                    )
                except Exception as error:
                    raise self._reject(
                        VerificationCode.TRUSTED_SERVICE_FAILURE,
                        f"relation {gate.relation_id} raised at position {member}: {error}",
                    ) from error
                if not satisfied:
                    raise self._reject(
                        VerificationCode.RELATION_REJECTED,
                        f"gate at position {member} violates {gate.relation_id}",
                    )
        self._phase = "done"
        self.transcript_parts.append(message)
        return VerificationReport(
            VerificationCode.ACCEPTED,
            sampled_replay_units=self.selected_replay_units,
            sampled_verification_units=sampled,
        )

    @property
    def transcript(self) -> Transcript:
        if self._phase != "done":
            raise ProtocolError("the protocol has not finished")
        return Transcript(*self.transcript_parts)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProtocolRun:
    report: VerificationReport
    transcript: Transcript | None


def run_protocol(
    artifact: CompiledArtifact,
    expectation: Expectation,
    values: Values,
    *,
    replay: Replay | None = None,
    limits: VerificationLimits | None = None,
) -> ProtocolRun:
    """Run prover and verifier against each other in one process."""

    verifier = VerifierSession(expectation, artifact, limits=limits)
    prover = ProverSession(
        artifact, verifier.header, values, replay=replay, limits=limits
    )
    try:
        replay_challenge = verifier.receive_boundary(prover.boundary())
        sample_challenge = verifier.receive_interiors(
            prover.interiors(replay_challenge)
        )
        report = verifier.receive_evidence(prover.evidence(sample_challenge))
    except Reject as rejection:
        return ProtocolRun(
            VerificationReport(
                rejection.code,
                rejection.detail,
                verifier.selected_replay_units,
                verifier.selected_verification_units,
            ),
            None,
        )
    return ProtocolRun(report, verifier.transcript)
