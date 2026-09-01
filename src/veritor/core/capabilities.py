"""Artifact kinds, capability reports, and non-exceptional outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .errors import InvalidArtifact, UnsupportedCapability


class ArtifactKind(StrEnum):
    """The strongest honest representation supplied by a static artifact."""

    EXECUTABLE_CIRCUIT = "executable_circuit"
    STRUCTURAL_CIRCUIT = "structural_circuit"
    CAPACITY_PROFILE = "capacity_profile"


class Capability(StrEnum):
    """Operations and properties that a plug-in may provide."""

    STATIC_COMPILE = "static_compile"
    COMPILE = "static_compile"
    STATIC_PARTITION = "static_partition"
    PARTITION = "static_partition"
    STATIC_BOUND = "static_bound"
    BOUND = "static_bound"
    EXECUTE = "execute"
    VERIFY = "verify"
    HIDDEN_STRUCTURE = "hidden_structure"


class SupportState(StrEnum):
    """Whether a capability is available under the reported contract."""

    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


# ``Availability`` is a readable spelling for plug-in APIs.
Availability = SupportState
SupportStatus = SupportState


class EvidenceStatus(StrEnum):
    """How a structural or analytical assertion was established."""

    NONE = "none"
    EXHAUSTIVE = "exhaustive"
    BY_CONSTRUCTION = "by_construction"
    CERTIFIED = "certified"
    ASSUMPTION_SCOPED = "assumption_scoped"
    HEURISTIC = "heuristic"


class ClaimStatus(StrEnum):
    """Strength of a claim returned by a later protocol or analysis layer."""

    EXACT = "exact"
    CERTIFIED_UPPER = "certified_upper"
    CONDITIONAL = "conditional"
    HEURISTIC = "heuristic"
    UNSUPPORTED = "unsupported"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def _nonempty_text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise InvalidArtifact(f"{field} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """One capability assertion for one artifact kind."""

    capability: Capability
    state: SupportState
    artifact_kind: ArtifactKind
    guarantee: str = ""
    reason_code: str | None = None
    detail: str = ""
    evidence: EvidenceStatus = EvidenceStatus.NONE

    def __post_init__(self) -> None:
        try:
            capability = Capability(self.capability)
            state = SupportState(self.state)
            artifact_kind = ArtifactKind(self.artifact_kind)
            evidence = EvidenceStatus(self.evidence)
        except (TypeError, ValueError) as error:
            raise InvalidArtifact(
                "capability status contains an unknown enum value"
            ) from error
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "artifact_kind", artifact_kind)
        object.__setattr__(self, "evidence", evidence)
        if type(self.guarantee) is not str:
            raise InvalidArtifact("capability guarantee must be a string")
        if type(self.detail) is not str:
            raise InvalidArtifact("capability detail must be a string")
        if self.reason_code is not None:
            _nonempty_text(self.reason_code, "reason_code")
        if state is SupportState.UNSUPPORTED and self.reason_code is None:
            raise InvalidArtifact("an unsupported capability needs a reason_code")


@dataclass(frozen=True, slots=True, init=False)
class CapabilityReport:
    """A deterministic, queryable collection of capability statuses."""

    plugin_id: str
    artifact_kind: ArtifactKind
    statuses: tuple[CapabilityStatus, ...]

    def __init__(
        self,
        plugin_id: str,
        artifact_kind: ArtifactKind,
        statuses: Iterable[CapabilityStatus],
    ) -> None:
        checked_plugin_id = _nonempty_text(plugin_id, "plugin_id")
        try:
            checked_kind = ArtifactKind(artifact_kind)
        except (TypeError, ValueError) as error:
            raise InvalidArtifact(
                "capability report has an unknown artifact kind"
            ) from error
        by_capability: dict[Capability, CapabilityStatus] = {}
        for status in statuses:
            if not isinstance(status, CapabilityStatus):
                raise InvalidArtifact(
                    "capability report statuses must be CapabilityStatus values"
                )
            if status.artifact_kind is not checked_kind:
                raise InvalidArtifact(
                    "capability status artifact kind does not match its report"
                )
            if status.capability in by_capability:
                raise InvalidArtifact(
                    f"duplicate capability status for {status.capability.value}"
                )
            by_capability[status.capability] = status
        ordered = tuple(by_capability[key] for key in sorted(by_capability, key=str))
        object.__setattr__(self, "plugin_id", checked_plugin_id)
        object.__setattr__(self, "artifact_kind", checked_kind)
        object.__setattr__(self, "statuses", ordered)

    def status_for(self, capability: Capability) -> CapabilityStatus:
        """Return the status for ``capability``, or raise ``KeyError``."""

        requested = Capability(capability)
        for status in self.statuses:
            if status.capability is requested:
                return status
        raise KeyError(requested)

    def supports(
        self,
        capability: Capability,
        *,
        allow_conditional: bool = False,
    ) -> bool:
        """Return whether the report supports the requested capability."""

        try:
            state = self.status_for(capability).state
        except KeyError:
            return False
        return state is SupportState.SUPPORTED or (
            allow_conditional and state is SupportState.CONDITIONAL
        )

    def require(self, capability: Capability) -> CapabilityStatus:
        """Return a supported status or raise ``UnsupportedCapability``."""

        requested = Capability(capability)
        try:
            status = self.status_for(requested)
        except KeyError as error:
            raise UnsupportedCapability(
                requested,
                reason_code="UNREPORTED_CAPABILITY",
                detail=f"{self.plugin_id} did not report this capability",
            ) from error
        if status.state is not SupportState.SUPPORTED:
            raise UnsupportedCapability(
                requested,
                reason_code=status.reason_code or "CONDITIONAL_CAPABILITY",
                detail=status.detail
                or "the capability is not unconditionally supported",
            )
        return status


@dataclass(frozen=True, slots=True)
class Unsupported:
    """A typed, non-exceptional result for an absent capability."""

    capability: Capability
    plugin_id: str
    reason_code: str
    detail: str
    artifact_kind: ArtifactKind | None = None

    def __post_init__(self) -> None:
        try:
            capability = Capability(self.capability)
            artifact_kind = (
                None if self.artifact_kind is None else ArtifactKind(self.artifact_kind)
            )
        except (TypeError, ValueError) as error:
            raise InvalidArtifact(
                "unsupported outcome contains an unknown enum"
            ) from error
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "artifact_kind", artifact_kind)
        _nonempty_text(self.plugin_id, "plugin_id")
        _nonempty_text(self.reason_code, "reason_code")
        _nonempty_text(self.detail, "detail")

    def as_error(self) -> UnsupportedCapability:
        """Convert this outcome to the corresponding exceptional form."""

        return UnsupportedCapability(
            self.capability,
            reason_code=self.reason_code,
            detail=self.detail,
        )
