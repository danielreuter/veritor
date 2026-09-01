import pytest

from veritor.core import (
    ArtifactKind,
    Capability,
    CapabilityReport,
    CapabilityStatus,
    ClaimStatus,
    EvidenceStatus,
    InvalidArtifact,
    ResourceLimit,
    SupportState,
    Unsupported,
    UnsupportedCapability,
    VerificationLimits,
)


def status(
    capability: Capability,
    state: SupportState,
    *,
    reason_code: str | None = None,
) -> CapabilityStatus:
    return CapabilityStatus(
        capability=capability,
        state=state,
        artifact_kind=ArtifactKind.STRUCTURAL_CIRCUIT,
        guarantee="exact declared structure",
        reason_code=reason_code,
        detail="test detail",
        evidence=EvidenceStatus.BY_CONSTRUCTION,
    )


def test_artifact_and_claim_vocabularies_are_stable_string_enums():
    assert ArtifactKind.EXECUTABLE_CIRCUIT.value == "executable_circuit"
    assert ArtifactKind.STRUCTURAL_CIRCUIT.value == "structural_circuit"
    assert ArtifactKind.CAPACITY_PROFILE.value == "capacity_profile"
    assert Capability.COMPILE is Capability.STATIC_COMPILE
    assert Capability.BOUND is Capability.STATIC_BOUND
    assert ClaimStatus.CERTIFIED_UPPER.value == "certified_upper"


def test_capability_report_is_deterministic_and_queryable():
    report = CapabilityReport(
        "tests.gpt2",
        ArtifactKind.STRUCTURAL_CIRCUIT,
        (
            status(
                Capability.VERIFY,
                SupportState.UNSUPPORTED,
                reason_code="NO_EXECUTABLE_RELATIONS",
            ),
            status(Capability.STATIC_COMPILE, SupportState.SUPPORTED),
            status(Capability.STATIC_BOUND, SupportState.CONDITIONAL),
        ),
    )

    assert tuple(item.capability.value for item in report.statuses) == (
        "static_bound",
        "static_compile",
        "verify",
    )
    assert report.supports(Capability.COMPILE)
    assert not report.supports(Capability.BOUND)
    assert report.supports(Capability.BOUND, allow_conditional=True)
    assert not report.supports(Capability.EXECUTE)
    assert report.require(Capability.COMPILE).state is SupportState.SUPPORTED
    with pytest.raises(UnsupportedCapability) as caught:
        report.require(Capability.VERIFY)
    assert caught.value.reason_code == "NO_EXECUTABLE_RELATIONS"


def test_capability_report_rejects_ambiguous_or_inconsistent_statuses():
    supported = status(Capability.STATIC_COMPILE, SupportState.SUPPORTED)

    with pytest.raises(InvalidArtifact, match="duplicate"):
        CapabilityReport(
            "plugin",
            ArtifactKind.STRUCTURAL_CIRCUIT,
            (supported, supported),
        )
    with pytest.raises(InvalidArtifact, match="does not match"):
        CapabilityReport(
            "plugin",
            ArtifactKind.CAPACITY_PROFILE,
            (supported,),
        )
    with pytest.raises(InvalidArtifact, match="reason_code"):
        status(Capability.VERIFY, SupportState.UNSUPPORTED)


def test_unsupported_is_a_typed_outcome_distinct_from_errors():
    outcome = Unsupported(
        capability=Capability.VERIFY,
        plugin_id="tests.profile",
        reason_code="NO_EXECUTABLE_RELATIONS",
        detail="the profile has no local relation",
        artifact_kind=ArtifactKind.CAPACITY_PROFILE,
    )

    error = outcome.as_error()

    assert not isinstance(outcome, BaseException)
    assert isinstance(error, UnsupportedCapability)
    assert error.capability is Capability.VERIFY
    assert InvalidArtifact is not ResourceLimit
    assert InvalidArtifact is not UnsupportedCapability
    assert ResourceLimit is not UnsupportedCapability


def test_verification_limits_are_frozen_validated_and_enforceable():
    limits = VerificationLimits(max_positions=3)

    limits.enforce("max_positions", 3)
    with pytest.raises(ResourceLimit) as caught:
        limits.enforce("max_positions", 4)
    assert caught.value.limit == 3
    assert caught.value.observed == 4
    with pytest.raises(InvalidArtifact):
        VerificationLimits(max_units=True)
    with pytest.raises(ValueError, match="unknown"):
        limits.enforce("max_unknown", 1)
