from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from veritor.core import ArtifactKind, Capability, SupportState
from veritor.plugins import (
    AggregateBoundArtifact,
    ArchitectureId,
    ArchitecturePlugin,
    GreedyTextExecutionShape,
    IndexedStructureArtifact,
    ProtocolCircuitArtifact,
    architecture_registry,
    compile_architecture,
    get_architecture_plugin,
    list_architecture_plugins,
    list_architectures,
)


def test_registry_is_closed_ordered_and_immutable() -> None:
    expected = (
        ArchitectureId.DEMO_G,
        ArchitectureId.MATMUL,
        ArchitectureId.GPT2,
        ArchitectureId.KIMI_K3,
        ArchitectureId.DEEPSEEK_V4_PRO,
        ArchitectureId.INKLING,
    )
    assert list_architectures() == expected
    assert tuple(architecture_registry()) == expected
    assert (
        tuple(plugin.architecture_id for plugin in list_architecture_plugins())
        == expected
    )
    assert all(
        isinstance(plugin, ArchitecturePlugin) for plugin in list_architecture_plugins()
    )
    with pytest.raises(TypeError):
        architecture_registry()[ArchitectureId.DEMO_G] = get_architecture_plugin(  # type: ignore[index]
            ArchitectureId.GPT2
        )
    with pytest.raises(KeyError, match="unknown architecture"):
        get_architecture_plugin("not-an-architecture")


def test_fixed_greedy_shape_contract() -> None:
    shape = GreedyTextExecutionShape(prompt_tokens=3, generated_tokens=2)
    assert shape.processed_positions == 4
    assert shape.prediction_positions == (2, 3)
    assert shape.batch_size == 1
    assert shape.fixed_horizon
    assert not shape.final_generated_forward
    assert not shape.eos_termination
    with pytest.raises(ValueError, match="batch_size=1"):
        GreedyTextExecutionShape(batch_size=2)
    with pytest.raises(ValueError, match="final generated token"):
        GreedyTextExecutionShape(final_generated_forward=True)


@pytest.mark.parametrize("architecture_id", list_architectures())
def test_registry_compiles_have_deterministic_identity(
    architecture_id: ArchitectureId,
) -> None:
    plugin = get_architecture_plugin(architecture_id)
    first = compile_architecture(architecture_id, plugin.default_request())
    second = compile_architecture(architecture_id.value, plugin.default_request())
    assert first.identity == second.identity
    assert first.identity.request_digest == second.identity.request_digest
    assert first.identity.representation_digest == second.identity.representation_digest
    assert first.identity.digest == second.identity.digest
    assert first.architecture_id is architecture_id
    assert not first.runtime_validated
    assert first.assumptions
    assert first.evidence
    with pytest.raises(FrozenInstanceError):
        first.runtime_validated = True  # type: ignore[misc]


def test_compile_results_are_discriminated() -> None:
    assert isinstance(
        compile_architecture(ArchitectureId.DEMO_G),
        ProtocolCircuitArtifact,
    )
    assert isinstance(
        compile_architecture(ArchitectureId.MATMUL),
        ProtocolCircuitArtifact,
    )
    assert isinstance(
        compile_architecture(ArchitectureId.GPT2),
        IndexedStructureArtifact,
    )
    for architecture_id in (
        ArchitectureId.KIMI_K3,
        ArchitectureId.INKLING,
    ):
        assert isinstance(
            compile_architecture(architecture_id),
            AggregateBoundArtifact,
        )


def test_capability_matrix_is_honest() -> None:
    artifacts = {
        architecture_id: compile_architecture(architecture_id)
        for architecture_id in list_architectures()
    }
    expected_kinds = {
        ArchitectureId.DEMO_G: ArtifactKind.EXECUTABLE_CIRCUIT,
        ArchitectureId.MATMUL: ArtifactKind.EXECUTABLE_CIRCUIT,
        ArchitectureId.GPT2: ArtifactKind.STRUCTURAL_CIRCUIT,
        ArchitectureId.KIMI_K3: ArtifactKind.CAPACITY_PROFILE,
        ArchitectureId.DEEPSEEK_V4_PRO: ArtifactKind.CAPACITY_PROFILE,
        ArchitectureId.INKLING: ArtifactKind.CAPACITY_PROFILE,
    }
    for architecture_id, artifact in artifacts.items():
        assert artifact.artifact_kind is expected_kinds[architecture_id]
        assert artifact.capabilities.supports(Capability.STATIC_COMPILE)
        assert (
            artifact.capabilities.status_for(Capability.HIDDEN_STRUCTURE).state
            is SupportState.UNSUPPORTED
        )

    for architecture_id in (ArchitectureId.DEMO_G, ArchitectureId.MATMUL):
        executable = artifacts[architecture_id]
        for capability in (
            Capability.STATIC_PARTITION,
            Capability.STATIC_BOUND,
            Capability.EXECUTE,
            Capability.VERIFY,
        ):
            assert executable.capabilities.supports(capability)

    gpt2 = artifacts[ArchitectureId.GPT2]
    assert not gpt2.capabilities.supports(Capability.STATIC_PARTITION)
    assert gpt2.capabilities.supports(Capability.STATIC_BOUND)
    assert not gpt2.capabilities.supports(Capability.EXECUTE)
    assert not gpt2.capabilities.supports(Capability.VERIFY)

    for architecture_id in (
        ArchitectureId.KIMI_K3,
        ArchitectureId.INKLING,
    ):
        aggregate = artifacts[architecture_id]
        assert not aggregate.capabilities.supports(Capability.STATIC_PARTITION)
        assert aggregate.capabilities.supports(Capability.STATIC_BOUND)
        assert not aggregate.capabilities.supports(Capability.EXECUTE)
        assert not aggregate.capabilities.supports(Capability.VERIFY)

    deepseek = artifacts[ArchitectureId.DEEPSEEK_V4_PRO]
    assert not deepseek.capabilities.supports(Capability.STATIC_PARTITION)
    assert (
        deepseek.capabilities.status_for(Capability.STATIC_BOUND).state
        is SupportState.CONDITIONAL
    )
