from __future__ import annotations

from dataclasses import replace

import pytest

import veritor
from veritor import (
    AggregateBoundArtifact,
    ArchitectureId,
    Compile,
    IndexedStructureArtifact,
    MatmulCompileRequest,
    ProtocolCircuitArtifact,
    Unsupported,
    VerificationPolicy,
    build_executable_conformance_transcript,
    make_verification_expectation,
)
from veritor.protocol import ProtocolError


@pytest.mark.parametrize(
    ("architecture_id", "expected_type"),
    (
        (ArchitectureId.DEMO_G, ProtocolCircuitArtifact),
        (ArchitectureId.MATMUL, ProtocolCircuitArtifact),
        (ArchitectureId.GPT2, IndexedStructureArtifact),
        (ArchitectureId.KIMI_K3, AggregateBoundArtifact),
        (ArchitectureId.DEEPSEEK_V4_PRO, AggregateBoundArtifact),
        (ArchitectureId.INKLING, AggregateBoundArtifact),
    ),
)
def test_compile_delegates_all_six_closed_registry_artifacts(
    architecture_id: ArchitectureId,
    expected_type: type[object],
) -> None:
    artifact = Compile(architecture_id)

    assert isinstance(artifact, expected_type)
    assert artifact.architecture_id is architecture_id
    assert Compile(architecture_id).identity == artifact.identity


def test_compile_rejects_unknown_architecture_without_fabricating_artifact() -> None:
    with pytest.raises(KeyError, match="unknown architecture"):
        Compile("not-an-architecture")


@pytest.mark.parametrize(
    "architecture_id",
    (
        ArchitectureId.GPT2,
        ArchitectureId.KIMI_K3,
        ArchitectureId.DEEPSEEK_V4_PRO,
        ArchitectureId.INKLING,
    ),
)
def test_non_executable_artifacts_have_typed_transcript_outcomes(
    architecture_id: ArchitectureId,
) -> None:
    artifact = Compile(architecture_id)

    outcomes = (
        make_verification_expectation(artifact, VerificationPolicy(1, 1, 0)),
        build_executable_conformance_transcript(artifact),
    )

    assert all(isinstance(outcome, Unsupported) for outcome in outcomes)
    assert all(outcome.capability.value == "verify" for outcome in outcomes)


def test_expectation_generates_mandatory_verifier_seeds() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    first = make_verification_expectation(artifact)
    second = make_verification_expectation(artifact)

    assert not isinstance(first, Unsupported)
    assert not isinstance(second, Unsupported)
    assert len(first.q_seed) == len(first.s_seed) == 32
    assert (first.q_seed, first.s_seed) != (second.q_seed, second.s_seed)
    assert first.session_id != second.session_id
    assert first.public_inputs == artifact.public_inputs
    assert first.claimed_outputs == artifact.expected_outputs
    assert first.compiled_digest == artifact.compiled.identity.digest
    with pytest.raises(ProtocolError, match="expected q seed"):
        replace(first, q_seed=None)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="expected s seed"):
        replace(first, s_seed=b"short")


def test_matmul_request_is_exported_and_compiles_through_public_facade() -> None:
    request = MatmulCompileRequest(
        ((1,),),
        (((2,),),),
    )

    artifact = Compile("matmul", request)

    assert isinstance(artifact, ProtocolCircuitArtifact)
    assert artifact.expected_outputs == (2,)


def test_paper_level_api_is_exported() -> None:
    names = {"Bound", "Compile", "Optimize", "Verify", "run_protocol"}

    assert names <= set(veritor.__all__)
    assert all(hasattr(veritor, name) for name in names)
    assert set(veritor.__all__) == set(veritor.research.__all__)
    assert all(hasattr(veritor.research, name) for name in veritor.research.__all__)
