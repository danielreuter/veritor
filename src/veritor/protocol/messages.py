"""Messages of the two-stage protocol, verifier outcomes, and errors.

The protocol has five messages after the public header::

    prover   -> verifier   BoundaryMessage    (commit beta over ∂ \\ W, open public I/O)
    verifier -> prover     ReplayChallenge    (q seed, J)
    prover   -> verifier   InteriorMessage    (commit alpha_r for r in J)
    verifier -> prover     SampleChallenge    (s seed, T)
    prover   -> verifier   EvidenceMessage    (openings for every sampled unit)

The header binds ``(C, I)``, the constructor ``G`` that produced it and the
advice ``a`` it was run on, the client's proposal ``theta`` and the
verifier's ``eta``, so the whole hash chain does.  It may also bind
:class:`Weights`: a per-model commitment ``kappa_W`` to the model's weight
vector, which the circuit's ``weight`` gates read by rank and the verifier
holds, so a run never carries the weights themselves.  A :class:`Transcript`
is the header plus these five messages in order.

The header also binds ``max_faults`` (``f_max``, default 0): how many VUs
the prover may *declare* incorrect in the interior message
(``InteriorMessage.declarations``).  A declared VU is committed like any
other -- its value is authenticated and every VU reading it is checked
against that value -- but its own relation check is skipped if it is
sampled, so an honest server that finds a hardware fault when it replays an
opened RU can own up to it instead of being rejected.  Declaring costs
capacity, not soundness: ``bound(..., max_faults=f_max)`` charges a prover
that declares after seeing ``J`` (:mod:`veritor.analysis.faults`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction

from veritor.core import (
    Digest,
    InvalidArtifact,
    JSONValue,
    VerificationPolicy,
    identity_digest,
    rational_manifest,
    validate_digest,
)

PROTOCOL_VERSION = "veritor/protocol/v6"
TRANSPARENT_BACKEND = "transparent"
"""The default proof backend: openings as the proof, relations recomputed by the verifier."""


class ProtocolError(InvalidArtifact):
    """A party used the protocol API incorrectly (not a verdict)."""


class VerificationCode(StrEnum):
    """Why the verifier accepted or rejected."""

    ACCEPTED = "accepted"
    EXPECTATION_MISMATCH = "expectation_mismatch"
    POLICY_REJECTED = "policy_rejected"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    INVALID_PHASE = "invalid_phase"
    INVALID_COMMITMENT = "invalid_commitment"
    INVALID_OPENING = "invalid_opening"
    INVALID_VALUE = "invalid_value"
    PUBLIC_IO_MISMATCH = "public_io_mismatch"
    CHALLENGE_MISMATCH = "challenge_mismatch"
    COVERAGE_MISMATCH = "coverage_mismatch"
    RELATION_REJECTED = "relation_rejected"
    INVALID_COMPILED_RESULT = "invalid_compiled_result"
    MALFORMED_TRANSCRIPT = "malformed_transcript"
    NONCANONICAL_TRANSCRIPT = "noncanonical_transcript"
    RESOURCE_LIMIT = "resource_limit"
    TRUSTED_SERVICE_FAILURE = "trusted_service_failure"
    PROOF_REJECTED = "proof_rejected"
    FAULTS_EXCEEDED = "faults_exceeded"
    FAULT_DECLARATION_INVALID = "fault_declaration_invalid"


class Reject(ProtocolError):
    """The verifier rejects; ``code`` says why."""

    def __init__(self, code: VerificationCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The verifier's verdict for one protocol run."""

    code: VerificationCode
    detail: str = ""
    sampled_replay_units: tuple[int, ...] = ()
    sampled_verification_units: tuple[int, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.code is VerificationCode.ACCEPTED


def _bytes32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ProtocolError(f"{name} must be exactly 32 bytes")
    return value


def _sorted_unique(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int or item < 0 for item in value):
        raise ProtocolError(f"{name} must be a tuple of nonnegative unit indices")
    if tuple(sorted(set(value))) != value:
        raise ProtocolError(f"{name} must be sorted and unique")
    return value


def _bytes_tuple(value: object, name: str) -> tuple[bytes, ...]:
    if type(value) is not tuple or any(type(item) is not bytes for item in value):
        raise ProtocolError(f"{name} must be a tuple of bytes")
    return value


def raw_digest(tag: str, manifest: JSONValue) -> bytes:
    """Tagged canonical-manifest SHA-256 as raw bytes."""

    return bytes.fromhex(identity_digest(tag, manifest))


@dataclass(frozen=True, slots=True)
class Commitment:
    """A Merkle root over one verifier-derived position domain."""

    root: bytes
    count: int

    def __post_init__(self) -> None:
        _bytes32(self.root, "commitment root")
        if type(self.count) is not int or self.count < 0:
            raise ProtocolError("commitment count must be a nonnegative integer")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"count": self.count, "root": self.root.hex()}


@dataclass(frozen=True, slots=True)
class Opening:
    """An authenticated value at one position of one commitment."""

    position: int
    value: bytes
    path: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ProtocolError("opening position must be a nonnegative integer")
        if type(self.value) is not bytes:
            raise ProtocolError("opening value must be bytes")
        if type(self.path) is not tuple or any(
            type(item) is not bytes or len(item) != 32 for item in self.path
        ):
            raise ProtocolError("opening path must be a tuple of 32-byte digests")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "path": [item.hex() for item in self.path],
            "position": self.position,
            "value": self.value.hex(),
        }


@dataclass(frozen=True, slots=True)
class Weights:
    """The root ``kappa_W`` over a model's weight vector of ``count`` values.

    Leaf ``k`` is the ``k``-th weight, read by the ``k``-th ``weight`` gate in
    address order of whichever circuit is verified, so the root is committed
    once per model and is the same for every description compiled from it.
    The verifier holds this and binds it into the header, and weight values
    are opened only where sampled, at their ranks.
    """

    count: int
    root: bytes

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ProtocolError("weight count must be a nonnegative integer")
        _bytes32(self.root, "weight root")

    @property
    def commitment(self) -> Commitment:
        return Commitment(self.root, self.count)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"count": self.count, "root": self.root.hex()}


@dataclass(frozen=True, slots=True)
class Header:
    """Public parameters both parties fix before any message is sent.

    ``compiled_digest`` names ``(C, I)``, ``constructor`` the digest of the
    ``G`` that produced it and ``advice`` the ``a`` it was run on, so a
    transcript is bound to one ``Compile(G, x, a)``.  ``policy`` is the
    client's ``theta = (q, s)`` and ``eta`` the verifier's acceptance
    threshold.      ``public_inputs`` are the encoded values of the circuit's
    ``in`` gates by rank (address order); the weight gates are under
    ``weights``.  ``backend`` names the proof backend the reveal step runs
    through (:mod:`veritor.protocol.proofs`); the default, ``"transparent"``,
    is the openings-as-proof protocol and leaves the header's manifest and
    digest exactly as they were before backends were pluggable.
    ``max_faults`` is the verifier's ``f_max``: how many VUs the interior
    message may declare incorrect; the default 0 likewise leaves manifest and
    digest untouched.
    """

    session_id: bytes
    compiled_digest: Digest
    constructor: Digest
    advice: bytes
    policy: VerificationPolicy
    eta: Fraction
    public_inputs: tuple[bytes, ...]
    claimed_outputs: tuple[bytes, ...]
    weights: Weights | None
    backend: str = TRANSPARENT_BACKEND
    max_faults: int = 0
    digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.session_id) is not bytes or not self.session_id:
            raise ProtocolError("session_id must be nonempty bytes")
        object.__setattr__(
            self, "compiled_digest", validate_digest(self.compiled_digest, "compiled digest")
        )
        object.__setattr__(
            self, "constructor", validate_digest(self.constructor, "constructor digest")
        )
        if type(self.advice) is not bytes:
            raise ProtocolError("advice must be bytes")
        if not isinstance(self.policy, VerificationPolicy):
            raise ProtocolError("policy must be a VerificationPolicy")
        if not isinstance(self.eta, Fraction) or not 0 <= self.eta < 1:
            raise ProtocolError("eta must be a Fraction in [0, 1)")
        _bytes_tuple(self.public_inputs, "public_inputs")
        _bytes_tuple(self.claimed_outputs, "claimed_outputs")
        if self.weights is not None and not isinstance(self.weights, Weights):
            raise ProtocolError("weights must be Weights or None")
        if type(self.backend) is not str or not self.backend:
            raise ProtocolError("backend must be a nonempty backend id")
        if type(self.max_faults) is not int or self.max_faults < 0:
            raise ProtocolError("max_faults must be a nonnegative integer")
        manifest: dict[str, JSONValue] = {
            "advice": self.advice.hex(),
            "claimed_outputs": [item.hex() for item in self.claimed_outputs],
            "compiled_digest": self.compiled_digest,
            "constructor": self.constructor,
            "eta": rational_manifest(self.eta),
            "policy": self.policy.manifest,
            "protocol_version": PROTOCOL_VERSION,
            "public_inputs": [item.hex() for item in self.public_inputs],
            "session_id": self.session_id.hex(),
            "weights": None if self.weights is None else self.weights.manifest,
        }
        if self.backend != TRANSPARENT_BACKEND:
            manifest["backend"] = self.backend
        if self.max_faults:
            manifest["max_faults"] = self.max_faults
        object.__setattr__(self, "digest", raw_digest("veritor/protocol/header/v6", manifest))


@dataclass(frozen=True, slots=True)
class BoundaryMessage:
    commitment: Commitment
    io_openings: tuple[Opening, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, Commitment):
            raise ProtocolError("boundary commitment has the wrong type")
        if type(self.io_openings) is not tuple or any(
            not isinstance(item, Opening) for item in self.io_openings
        ):
            raise ProtocolError("io_openings must be a tuple of openings")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "commitment": self.commitment.manifest,
            "io_openings": [item.manifest for item in self.io_openings],
        }


@dataclass(frozen=True, slots=True)
class ReplayChallenge:
    seed: bytes
    selected: tuple[int, ...]

    def __post_init__(self) -> None:
        _bytes32(self.seed, "q seed")
        _sorted_unique(self.selected, "selected replay units")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"seed": self.seed.hex(), "selected": list(self.selected)}


@dataclass(frozen=True, slots=True)
class InteriorMessage:
    """One interior commitment per selected replay unit, in ``J`` order, and
    the prover's fault declarations.

    ``declarations`` are global VU indices the prover declares incorrect
    ("this VU holds the committed value; I do not claim it is correct"):
    sorted, unique, each inside an opened RU and naming a VU with a relation,
    at most ``Header.max_faults`` of them -- the verifier checks all of this
    before deriving the s-challenge, whose seed binds them.  Empty (the
    default) leaves the manifest exactly as it was before declarations
    existed.
    """

    commitments: tuple[Commitment, ...]
    declarations: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.commitments) is not tuple or any(
            not isinstance(item, Commitment) for item in self.commitments
        ):
            raise ProtocolError("interior commitments must be a tuple of commitments")
        _sorted_unique(self.declarations, "fault declarations")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {
            "commitments": [item.manifest for item in self.commitments]
        }
        if self.declarations:
            manifest["declarations"] = list(self.declarations)
        return manifest


@dataclass(frozen=True, slots=True)
class SampleChallenge:
    seed: bytes
    selected: tuple[int, ...]

    def __post_init__(self) -> None:
        _bytes32(self.seed, "s seed")
        _sorted_unique(self.selected, "selected verification units")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"seed": self.seed.hex(), "selected": list(self.selected)}


@dataclass(frozen=True, slots=True)
class ProofMessage:
    """One proof from a non-transparent backend and which sampled VUs it covers.

    ``units`` are indices into ``T`` (the sample challenge's selection), sorted
    and unique; ``proof`` is the backend's opaque proof; ``foreign`` is the
    canonical statement of any other sessions' obligations proved in the same
    batch (empty when the batch is this session's alone).
    """

    units: tuple[int, ...]
    proof: bytes
    foreign: bytes = b""

    def __post_init__(self) -> None:
        _sorted_unique(self.units, "covered units")
        if not self.units:
            raise ProtocolError("a proof must cover at least one sampled unit")
        if type(self.proof) is not bytes or type(self.foreign) is not bytes:
            raise ProtocolError("proof and foreign statement must be bytes")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {"proof": self.proof.hex(), "units": list(self.units)}
        if self.foreign:
            manifest["foreign"] = self.foreign.hex()
        return manifest


@dataclass(frozen=True, slots=True)
class EvidenceMessage:
    """The reveal step: under the transparent backend, one opening batch per
    sampled verification unit in ``T`` order (``units``); under any other
    backend, the proofs covering ``T`` (``proofs``) and no openings.  Exactly
    one of the two is nonempty unless nothing was sampled.
    """

    units: tuple[tuple[Opening, ...], ...]
    proofs: tuple[ProofMessage, ...] = ()

    def __post_init__(self) -> None:
        if type(self.units) is not tuple or any(
            type(batch) is not tuple
            or any(not isinstance(item, Opening) for item in batch)
            for batch in self.units
        ):
            raise ProtocolError("evidence must be a tuple of opening tuples")
        if type(self.proofs) is not tuple or any(
            not isinstance(item, ProofMessage) for item in self.proofs
        ):
            raise ProtocolError("evidence proofs must be a tuple of proof messages")
        if self.units and self.proofs:
            raise ProtocolError("evidence carries either openings or proofs, not both")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {
            "units": [[item.manifest for item in batch] for batch in self.units]
        }
        if self.proofs:
            manifest["proofs"] = [item.manifest for item in self.proofs]
        return manifest


@dataclass(frozen=True, slots=True)
class Transcript:
    header: Header
    boundary: BoundaryMessage
    replay_challenge: ReplayChallenge
    interiors: InteriorMessage
    sample_challenge: SampleChallenge
    evidence: EvidenceMessage
