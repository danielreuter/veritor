"""Sample-evidence plug-ins and the transparent local-check backend."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from veritor.commitment import (
    CommitmentDomain,
    CommitmentOwner,
    ValueCommitment,
    ValueCommitmentBackend,
    ValueOpening,
    commitment_manifest,
    opening_manifest,
)
from veritor.core import (
    Position,
    VerificationLimits,
    VerificationPolicy,
    identity_digest,
    iter_domain,
)

from ._json import (
    NonCanonicalWireError,
    WireCodecError,
    array,
    canonical_bytes,
    exact_keys,
    hex_bytes,
    integer,
    load_strict_json,
    text,
)
from .artifact import ResolvedExecutableArtifact
from .boundary import (
    CommitmentLayout,
    required_positions_for_verification_unit,
)
from .model import (
    TRANSPARENT_LOCAL_CHECK_V1,
    PublicStatement,
    SessionParameters,
    StagedProtocolError,
)
from .phases import derive_statement_digest, session_manifest

TRANSPARENT_EVIDENCE_WIRE_VERSION = "veritor.staged.transparent-local-check-evidence/v1"


class EvidenceError(StagedProtocolError):
    """Sample evidence is malformed, inauthentic, or incomplete."""


class RelationRejected(EvidenceError):
    """An authenticated sampled local relation does not hold."""


class TrustedServiceFailure(EvidenceError):
    """A trusted semantic service failed instead of returning a decision."""


@runtime_checkable
class EvidenceOpeningProvider(Protocol):
    """Builder-side access to private commitment opening material."""

    def open(
        self,
        owner: CommitmentOwner,
        global_position: int,
    ) -> ValueOpening: ...


@dataclass(frozen=True, slots=True)
class SampleVerificationInstance:
    """Verifier-derived public instance for exactly one sampled unit."""

    artifact: ResolvedExecutableArtifact = field(repr=False, compare=False)
    layout: CommitmentLayout = field(repr=False, compare=False)
    session: SessionParameters
    statement: PublicStatement
    policy: VerificationPolicy
    sample_phase_digest: bytes
    verification_unit_index: int
    boundary_domain: CommitmentDomain
    boundary_commitment: ValueCommitment
    unit_domain: CommitmentDomain
    unit_commitment: ValueCommitment
    value_backend: ValueCommitmentBackend = field(repr=False, compare=False)
    sample_evidence_backend_id: str
    required_positions: tuple[int, ...] = field(init=False)
    instance_digest: bytes = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.sample_phase_digest) is not bytes
            or len(self.sample_phase_digest) != 32
        ):
            raise EvidenceError("sample_phase_digest must be 32 bytes")
        if (
            type(self.verification_unit_index) is not int
            or self.verification_unit_index < 0
            or self.verification_unit_index
            >= self.artifact.verification_partition.unit_count
        ):
            raise EvidenceError("verification unit index is out of range")
        if not isinstance(self.value_backend, ValueCommitmentBackend):
            raise EvidenceError("value backend has the wrong type")
        if (
            type(self.sample_evidence_backend_id) is not str
            or not self.sample_evidence_backend_id
        ):
            raise EvidenceError("sample evidence backend id must be nonempty")
        unit = self.artifact.verification_partition.unit_at(
            self.verification_unit_index
        )
        if self.unit_domain.owner != CommitmentOwner.replay_unit(int(unit.replay_unit)):
            raise EvidenceError("unit domain belongs to another replay unit")
        required = tuple(
            int(item)
            for item in required_positions_for_verification_unit(
                self.artifact.circuit,
                unit,
                self.layout,
            )
        )
        manifest = {
            "boundary_commitment": commitment_manifest(self.boundary_commitment),
            "boundary_domain_id": self.boundary_domain.domain_id.hex(),
            "compiled_result_digest": self.session.compiled_result_digest,
            "policy_digest": str(self.policy.digest),
            "required_positions": list(required),
            "sample_evidence_backend_id": self.sample_evidence_backend_id,
            "sample_phase_digest": self.sample_phase_digest.hex(),
            "session": session_manifest(self.session),
            "statement_digest": derive_statement_digest(self.statement).hex(),
            "unit_commitment": commitment_manifest(self.unit_commitment),
            "unit_domain_id": self.unit_domain.domain_id.hex(),
            "verification_unit_identity": str(unit.identity_digest),
            "verification_unit_index": self.verification_unit_index,
        }
        object.__setattr__(self, "required_positions", required)
        object.__setattr__(
            self,
            "instance_digest",
            bytes.fromhex(
                identity_digest(
                    "veritor/staged/sample-verification-instance/v1",
                    manifest,  # type: ignore[arg-type]
                )
            ),
        )

    def domain_and_commitment_for(
        self,
        global_position: int,
    ) -> tuple[CommitmentDomain, ValueCommitment]:
        owner = self.layout.owner_of(global_position)
        if owner == CommitmentOwner.boundary():
            return self.boundary_domain, self.boundary_commitment
        if owner != self.unit_domain.owner:
            raise EvidenceError(
                "sampled value belongs to another replay-unit commitment"
            )
        return self.unit_domain, self.unit_commitment


@runtime_checkable
class SampleEvidenceBackend(Protocol):
    """Backend-neutral sampled-unit evidence interface."""

    @property
    def backend_id(self) -> str: ...

    def build_evidence(
        self,
        instance: SampleVerificationInstance,
        openings: EvidenceOpeningProvider,
        limits: VerificationLimits,
    ) -> bytes: ...

    def verify_evidence(
        self,
        instance: SampleVerificationInstance,
        payload: bytes,
        limits: VerificationLimits,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TransparentLocalCheckEvidence:
    """Canonical transparent evidence: exactly one opening per required value."""

    instance_digest: bytes
    openings: tuple[ValueOpening, ...]

    def __post_init__(self) -> None:
        if type(self.instance_digest) is not bytes or len(self.instance_digest) != 32:
            raise EvidenceError("evidence instance_digest must be 32 bytes")
        if type(self.openings) is not tuple or any(
            not isinstance(opening, ValueOpening) for opening in self.openings
        ):
            raise EvidenceError("evidence openings have the wrong shape")


def _transparent_manifest(
    evidence: TransparentLocalCheckEvidence,
) -> dict[str, object]:
    return {
        "instance_digest": evidence.instance_digest.hex(),
        "openings": [opening_manifest(opening) for opening in evidence.openings],
        "version": TRANSPARENT_EVIDENCE_WIRE_VERSION,
    }


def encode_transparent_evidence(
    evidence: TransparentLocalCheckEvidence,
) -> bytes:
    return canonical_bytes(_transparent_manifest(evidence))


def decode_transparent_evidence(
    payload: bytes,
    limits: VerificationLimits,
) -> TransparentLocalCheckEvidence:
    document = load_strict_json(
        payload,
        limits,
        byte_limit=limits.max_proof_bytes,
    )
    root = exact_keys(
        document,
        frozenset({"instance_digest", "openings", "version"}),
        "transparent evidence",
    )
    if text(root["version"], "transparent evidence.version") != (
        TRANSPARENT_EVIDENCE_WIRE_VERSION
    ):
        raise WireCodecError("unknown transparent evidence version")
    raw_openings = array(root["openings"], "transparent evidence.openings")
    limits.enforce("max_openings", len(raw_openings))
    openings: list[ValueOpening] = []
    proof_bytes = 0
    for index, raw in enumerate(raw_openings):
        obj = exact_keys(
            raw,
            frozenset({"path", "position", "value"}),
            f"transparent evidence.openings[{index}]",
        )
        raw_path = array(
            obj["path"],
            f"transparent evidence.openings[{index}].path",
        )
        limits.enforce("max_openings", len(raw_path))
        path = tuple(
            hex_bytes(
                sibling,
                f"transparent evidence.openings[{index}].path[{path_index}]",
                allow_empty=True,
            )
            for path_index, sibling in enumerate(raw_path)
        )
        value = hex_bytes(
            obj["value"],
            f"transparent evidence.openings[{index}].value",
            allow_empty=True,
        )
        proof_bytes += len(value) + sum(len(item) for item in path)
        limits.enforce("max_proof_bytes", proof_bytes)
        openings.append(
            ValueOpening(
                Position(
                    integer(
                        obj["position"],
                        f"transparent evidence.openings[{index}].position",
                    )
                ),
                value,
                path,
            )
        )
    result = TransparentLocalCheckEvidence(
        hex_bytes(
            root["instance_digest"],
            "transparent evidence.instance_digest",
            length=32,
        ),
        tuple(openings),
    )
    if encode_transparent_evidence(result) != payload:
        raise NonCanonicalWireError("transparent evidence is not canonically encoded")
    return result


@dataclass(frozen=True, slots=True)
class TransparentLocalCheckBackend:
    """Authenticate exact values and call only trusted artifact relations."""

    backend_id: str = TRANSPARENT_LOCAL_CHECK_V1

    def build_evidence(
        self,
        instance: SampleVerificationInstance,
        openings: EvidenceOpeningProvider,
        limits: VerificationLimits,
    ) -> bytes:
        if not isinstance(instance, SampleVerificationInstance):
            raise EvidenceError("sample instance has the wrong type")
        values = tuple(
            openings.open(
                instance.layout.owner_of(position),
                position,
            )
            for position in instance.required_positions
        )
        limits.enforce("max_openings", len(values))
        payload = encode_transparent_evidence(
            TransparentLocalCheckEvidence(instance.instance_digest, values)
        )
        limits.enforce("max_proof_bytes", len(payload))
        return payload

    def verify_evidence(
        self,
        instance: SampleVerificationInstance,
        payload: bytes,
        limits: VerificationLimits,
    ) -> None:
        if not isinstance(instance, SampleVerificationInstance):
            raise EvidenceError("sample instance has the wrong type")
        if instance.sample_evidence_backend_id != self.backend_id:
            raise EvidenceError("sample instance names another evidence backend")
        if type(payload) is not bytes:
            raise EvidenceError("sample evidence payload must be bytes")
        limits.enforce("max_proof_bytes", len(payload))
        try:
            evidence = decode_transparent_evidence(payload, limits)
        except (WireCodecError, ValueError) as error:
            raise EvidenceError(
                "transparent evidence wire encoding is invalid"
            ) from error
        if evidence.instance_digest != instance.instance_digest:
            raise EvidenceError("sample evidence belongs to another instance")
        actual_positions = tuple(int(opening.position) for opening in evidence.openings)
        if actual_positions != instance.required_positions:
            raise EvidenceError(
                "sample evidence does not exactly cover required values"
            )
        if len(set(actual_positions)) != len(actual_positions):
            raise EvidenceError("sample evidence repeats an opening")

        decoded_values: dict[int, object] = {}
        for opening in evidence.openings:
            global_position = int(opening.position)
            domain, commitment = instance.domain_and_commitment_for(global_position)
            if not instance.value_backend.verify_opening(
                domain,
                commitment,
                opening,
                limits,
            ):
                raise EvidenceError(
                    f"value opening for position {global_position} is invalid"
                )
            schema = domain.schema_for(global_position)
            try:
                decoded = instance.artifact.value_service.decode(
                    schema,
                    opening.value,
                )
                canonical = instance.artifact.value_service.encode(
                    schema,
                    decoded,
                )
            except Exception as error:
                raise EvidenceError(
                    f"value at position {global_position} is not canonical "
                    f"for schema {schema!r}"
                ) from error
            if type(canonical) is not bytes or canonical != opening.value:
                raise EvidenceError(
                    f"value at position {global_position} is noncanonical"
                )
            decoded_values[global_position] = decoded

        unit = instance.artifact.verification_partition.unit_at(
            instance.verification_unit_index
        )
        for member in iter_domain(unit.members):
            structural = instance.artifact.circuit.gate_at(member)
            executable = instance.artifact.circuit.executable_gate_at(member)
            if (
                executable.position != member
                or executable.arguments != structural.predecessors
                or executable.operation != structural.operation
            ):
                raise TrustedServiceFailure(
                    "trusted executable and structural gate views disagree"
                )
            try:
                decision = instance.artifact.relation_service.check(
                    str(executable.relation_id),
                    tuple(
                        decoded_values[int(argument)]
                        for argument in executable.arguments
                    ),
                    decoded_values[int(member)],
                )
            except Exception as error:
                raise TrustedServiceFailure(
                    f"trusted relation service failed for position {member}"
                ) from error
            if type(decision) is not bool:
                raise TrustedServiceFailure(
                    "trusted relation service did not return bool"
                )
            if not decision:
                raise RelationRejected(f"sampled relation failed at position {member}")


class SampleEvidenceRegistry:
    """Immutable allowlist of locally trusted sample-evidence backends."""

    __slots__ = ("_backends",)

    def __init__(self, backends: Iterable[SampleEvidenceBackend]) -> None:
        by_id: dict[str, SampleEvidenceBackend] = {}
        for backend in backends:
            if not isinstance(backend, SampleEvidenceBackend):
                raise EvidenceError("sample evidence registry entry has the wrong type")
            if backend.backend_id in by_id:
                raise EvidenceError(
                    f"duplicate sample evidence backend {backend.backend_id!r}"
                )
            by_id[backend.backend_id] = backend
        self._backends = MappingProxyType(by_id)

    @classmethod
    def with_defaults(cls) -> SampleEvidenceRegistry:
        return cls((TransparentLocalCheckBackend(),))

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def resolve(self, backend_id: str) -> SampleEvidenceBackend | None:
        if type(backend_id) is not str:
            return None
        return self._backends.get(backend_id)

    def require(self, backend_id: str) -> SampleEvidenceBackend:
        backend = self.resolve(backend_id)
        if backend is None:
            raise EvidenceError(f"unknown sample evidence backend {backend_id!r}")
        return backend
