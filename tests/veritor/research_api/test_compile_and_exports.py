from __future__ import annotations

import importlib
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
    adapt_protocol_artifact,
    build_demo_conformance_transcript,
    create_trusted_verification_context,
    make_verification_expectation,
)
from veritor.research import compile as research_compile
from veritor.staged import StagedProtocolError


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
def test_compile_delegates_all_five_closed_registry_artifacts(
    architecture_id: ArchitectureId,
    expected_type: type[object],
) -> None:
    artifact = Compile(architecture_id)

    assert isinstance(artifact, expected_type)
    assert artifact.architecture_id is architecture_id
    assert research_compile(architecture_id).identity == artifact.identity


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
        adapt_protocol_artifact(artifact),
        create_trusted_verification_context(artifact),
        make_verification_expectation(artifact, VerificationPolicy(1, 1, 0)),
        build_demo_conformance_transcript(artifact),
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
    assert first.public_inputs == artifact.public_inputs
    assert first.claimed_outputs == artifact.expected_outputs
    with pytest.raises(StagedProtocolError, match="expected q seed"):
        replace(first, q_seed=None)  # type: ignore[arg-type]
    with pytest.raises(StagedProtocolError, match="expected s seed"):
        replace(first, s_seed=b"short")


def test_matmul_request_is_exported_and_compiles_through_public_facade() -> None:
    request = MatmulCompileRequest(
        ((1,),),
        (((2,),),),
    )

    artifact = Compile("matmul", request)

    assert isinstance(artifact, ProtocolCircuitArtifact)
    assert artifact.expected_outputs == (2,)


def test_every_legacy_top_level_symbol_remains_exported() -> None:
    legacy = {
        "CELL_BYTES",
        "PRIMITIVES",
        "Commitment",
        "Instruction",
        "InstructionOpening",
        "LEAF_INSTRUCTION",
        "LEAF_VALUE",
        "MerkleTree",
        "Operand",
        "Primitive",
        "Program",
        "Prover",
        "TraceError",
        "Transcript",
        "ValueOpening",
        "Verifier",
        "apply_primitive",
        "decode_cell",
        "decode_instruction",
        "encode_cell",
        "encode_instruction",
        "execute",
        "primitive",
        "run_protocol",
        "trace",
        "verify_leaf",
    }
    modern = {
        "Bound",
        "Compile",
        "Optimize",
        "Verify",
        "bound",
        "optimize",
        "verify",
    }

    assert legacy | modern <= set(veritor.__all__)
    assert all(hasattr(veritor, name) for name in legacy | modern)
    assert veritor.compile is importlib.import_module("veritor.compile")
    assert research_compile is Compile
