"""Interactive prover and verifier sessions for the two-stage protocol.

The verifier owns both seeds and releases each challenge only after the
message it depends on has been received and checked.  Every lookup the
verifier performs is against the trusted :class:`Compiled` ``(C, I)``; the
prover never tells the verifier where a value lives or which addresses a unit
has.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from secrets import token_bytes

from veritor.core import Compiled, VerificationLimits, VerificationPolicy, iter_domain

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
    compiled: Compiled,
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
        compiled_digest=compiled.digest,
        policy=policy,
        public_inputs=tuple(public_inputs),
        claimed_outputs=tuple(claimed_outputs),
        q_seed=token_bytes(32) if q_seed is None else q_seed,
        s_seed=token_bytes(32) if s_seed is None else s_seed,
    )


class _Layout:
    """Address-level lookups shared by both sessions; all ``O(depth)``."""

    __slots__ = ("boundary", "circuit", "compiled", "index")

    def __init__(self, compiled: Compiled) -> None:
        if not isinstance(compiled, Compiled):
            raise ProtocolError("sessions require a Compiled circuit")
        self.compiled = compiled
        self.circuit = compiled.circuit
        self.index = compiled.index
        self.boundary = compiled.index.boundary()

    def schema(self, address: int) -> str:
        """The leaf schema of an address: its value width."""

        return f"u{self.circuit[address].width}"

    def owner(self, address: int) -> int:
        """``BOUNDARY_OWNER`` for boundary addresses, else the owning replay unit."""

        if self.boundary.contains(address):
            return BOUNDARY_OWNER
        return self.index.replay_units.owner(address)

    def io_addresses(self) -> tuple[int, ...]:
        """Distinct public I/O addresses in boundary rank order."""

        addresses = set(self.circuit.inputs)
        addresses.update(self.circuit.outputs)
        return tuple(sorted(addresses, key=self.boundary.rank))

    def required(self, unit: int) -> tuple[tuple[int, int], ...]:
        """``(owner, address)`` for every value a verification unit reads or writes."""

        node = self.index.verification_unit(unit)
        replay_unit = node.replay_unit
        addresses = set(node.interval)
        addresses.update(self.circuit.In(node))
        result: list[tuple[int, int]] = []
        for address in sorted(addresses):
            owner = self.owner(address)
            if owner not in (BOUNDARY_OWNER, replay_unit):
                raise Reject(
                    VerificationCode.INVALID_COMPILED_RESULT,
                    f"address {address} is read by unit {unit} but owned by "
                    f"replay unit {owner} and is not a boundary address",
                )
            result.append((owner, address))
        return tuple(result)


def replay_unit(compiled: Compiled, unit: int, boundary_values: Values) -> dict[int, object]:
    """Honest replay: recompute ``Int(unit)`` from the boundary, in address order."""

    circuit = compiled.circuit
    known: dict[int, object] = {}
    for address in iter_domain(compiled.index.interior(unit)):
        arguments = []
        for argument in circuit[address].args:
            if argument in known:
                arguments.append(known[argument])
            else:
                try:
                    arguments.append(boundary_values[argument])
                except KeyError as error:
                    raise ProtocolError(
                        f"replay of unit {unit} needs boundary value {argument}"
                    ) from error
        known[address] = circuit.evaluate_gate(address, arguments)  # type: ignore[arg-type]
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
        compiled: Compiled,
        header: Header,
        values: Values,
        *,
        replay: Replay | None = None,
        limits: VerificationLimits | None = None,
    ) -> None:
        self._layout = _Layout(compiled)
        if header.compiled_digest != compiled.digest:
            raise ProtocolError("header names a different compiled circuit")
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
        for address in iter_domain(domain.positions):
            try:
                value = values[int(address)]
            except KeyError as error:
                raise ProtocolError(f"prover has no value for address {address}") from error
            encoded[int(address)] = layout.circuit.encode(address, value)
        tree = MerkleTree(domain, encoded, layout.schema)
        self._trees[domain.owner] = tree
        return tree

    def boundary(self) -> BoundaryMessage:
        self._expect("boundary")
        tree = self._commit(boundary_domain(self.header, self._layout.compiled), self._values)
        message = BoundaryMessage(
            tree.commitment, tuple(tree.open(p) for p in self._layout.io_addresses())
        )
        self._boundary_phase = boundary_phase(self.header, message)
        self._phase = "interiors"
        self.transcript_parts.append(message)
        return message

    def interiors(self, challenge: ReplayChallenge) -> InteriorMessage:
        self._expect("interiors")
        self._replay_phase = replay_phase(self._boundary_phase, challenge)
        compiled = self._layout.compiled
        commitments: list[Commitment] = []
        for unit in challenge.selected:
            if unit >= compiled.index.replay_units.count:
                raise ProtocolError(f"challenge names unknown replay unit {unit}")
            interior_values = (
                replay_unit(compiled, unit, self._values)
                if self._replay is None
                else self._replay(unit, self._values)
            )
            domain = interior_domain(self.header, self._replay_phase, compiled, unit)
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
            for owner, address in self._layout.required(unit):
                tree = self._trees.get(owner)
                if tree is None:
                    raise ProtocolError(
                        f"sampled unit {unit} needs uncommitted replay unit {owner}"
                    )
                openings.append(tree.open(address))
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
        compiled: Compiled,
        *,
        limits: VerificationLimits | None = None,
    ) -> None:
        self._layout = _Layout(compiled)
        if expectation.compiled_digest != compiled.digest:
            raise ProtocolError("expectation names a different compiled circuit")
        self._expectation = expectation
        self._limits = VerificationLimits() if limits is None else limits
        circuit = self._layout.circuit
        try:
            inputs = tuple(
                circuit.encode(address, value)
                for address, value in zip(
                    circuit.inputs, expectation.public_inputs, strict=True
                )
            )
            outputs = tuple(
                circuit.encode(address, value)
                for address, value in zip(
                    circuit.outputs, expectation.claimed_outputs, strict=True
                )
            )
        except Exception as error:
            raise ProtocolError("expectation values do not encode canonically") from error
        self.header = Header(
            expectation.session_id,
            compiled.digest,
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

    def _accept_commitment(self, domain: CommitmentDomain, commitment: Commitment) -> None:
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
                f"opening of address {opening.position} failed under owner {owner}",
            )
        return opening.value

    def receive_boundary(self, message: BoundaryMessage) -> ReplayChallenge:
        self._expect("boundary")
        compiled = self._layout.compiled
        self._accept_commitment(boundary_domain(self.header, compiled), message.commitment)
        expected = self._layout.io_addresses()
        if tuple(item.position for item in message.io_openings) != expected:
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                "public I/O openings must cover exactly the I/O addresses in boundary order",
            )
        opened = {
            item.position: self._open(BOUNDARY_OWNER, item) for item in message.io_openings
        }
        circuit = self._layout.circuit
        for addresses, values, label in (
            (circuit.inputs, self.header.public_inputs, "input"),
            (circuit.outputs, self.header.claimed_outputs, "output"),
        ):
            for address, value in zip(addresses, values, strict=True):
                if opened[address] != value:
                    raise self._reject(
                        VerificationCode.PUBLIC_IO_MISMATCH,
                        f"{label} at address {address} differs",
                    )
        self._boundary_phase = boundary_phase(self.header, message)
        selected = derive_replay_selection(
            self._expectation.q_seed,
            self._boundary_phase,
            compiled,
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
        compiled = self._layout.compiled
        selected = self.selected_replay_units
        if len(message.commitments) != len(selected):
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                f"expected {len(selected)} interior commitments, got "
                f"{len(message.commitments)}",
            )
        for unit, commitment in zip(selected, message.commitments, strict=True):
            self._accept_commitment(
                interior_domain(self.header, self._replay_phase, compiled, unit),
                commitment,
            )
        self._interior_phase = interior_phase(self._replay_phase, message)
        sampled = derive_sample_selection(
            self._expectation.s_seed,
            self._interior_phase,
            compiled,
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
                    f"evidence for unit {unit} must open exactly its required addresses",
                )
            values: dict[int, int] = {}
            for (owner, address), opening in zip(required, batch, strict=True):
                payload = self._open(owner, opening)
                try:
                    values[address] = circuit.decode(address, payload)
                except Exception as error:
                    raise self._reject(
                        VerificationCode.INVALID_VALUE,
                        f"value at address {address} is not canonical: {error}",
                    ) from error
            for member in layout.index.verification_unit(unit).interval:
                gate = circuit[member]
                if gate.is_input:
                    continue
                try:
                    satisfied = circuit.check_gate(
                        member, tuple(values[item] for item in gate.args), values[member]
                    )
                except Exception as error:
                    raise self._reject(
                        VerificationCode.TRUSTED_SERVICE_FAILURE,
                        f"gate {gate.op} raised at address {member}: {error}",
                    ) from error
                if not satisfied:
                    raise self._reject(
                        VerificationCode.RELATION_REJECTED,
                        f"gate at address {member} violates {gate.op}",
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
    compiled: Compiled,
    expectation: Expectation,
    values: Values,
    *,
    replay: Replay | None = None,
    limits: VerificationLimits | None = None,
) -> ProtocolRun:
    """Run prover and verifier against each other in one process."""

    verifier = VerifierSession(expectation, compiled, limits=limits)
    prover = ProverSession(compiled, verifier.header, values, replay=replay, limits=limits)
    try:
        replay_challenge = verifier.receive_boundary(prover.boundary())
        sample_challenge = verifier.receive_interiors(prover.interiors(replay_challenge))
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
