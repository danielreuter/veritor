from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from circuit_cut_analysis.models.gpt2 import GPT2Config
from veritor.core import Capability, Compiled, InvalidArtifact, Unsupported
from veritor.plugins import (
    NO_CONSTRUCTOR,
    ArchitectureId,
    ArchitecturePlugin,
    ConfiguredPlugin,
    DeepSeekV4ProCompileRequest,
    GPT2CompileRequest,
    GreedyTextExecutionShape,
    InklingCompileRequest,
    KimiK3CompileRequest,
    architecture_registry,
    compile_architecture,
    get_architecture_plugin,
    list_architecture_plugins,
    list_architectures,
)

EXECUTABLE = (ArchitectureId.DEMO_G, ArchitectureId.MATMUL)
CONFIGURED = tuple(item for item in ArchitectureId if item not in EXECUTABLE)


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
    assert tuple(plugin.architecture_id for plugin in list_architecture_plugins()) == expected
    assert all(isinstance(plugin, ArchitecturePlugin) for plugin in list_architecture_plugins())
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


@pytest.mark.parametrize("architecture_id", EXECUTABLE)
def test_executable_compiles_are_deterministic(architecture_id: ArchitectureId) -> None:
    plugin = get_architecture_plugin(architecture_id)
    first = compile_architecture(architecture_id, plugin.default_request())
    second = compile_architecture(architecture_id.value, plugin.default_request())
    assert isinstance(first, Compiled)
    assert first.digest == second.digest
    assert first.index.digest == second.index.digest
    with pytest.raises(FrozenInstanceError):
        first.digest = "x"  # type: ignore[misc]


@pytest.mark.parametrize("architecture_id", CONFIGURED)
def test_configured_architectures_report_the_missing_constructor(
    architecture_id: ArchitectureId,
) -> None:
    plugin = get_architecture_plugin(architecture_id)
    assert isinstance(plugin, ConfiguredPlugin)
    request = plugin.default_request()
    assert request.architecture_id is architecture_id
    assert isinstance(request.execution_shape, GreedyTextExecutionShape)
    outcome = compile_architecture(architecture_id, request)
    assert isinstance(outcome, Unsupported)
    assert outcome.capability is Capability.STATIC_COMPILE
    assert outcome.reason_code == NO_CONSTRUCTOR
    assert outcome.plugin_id == plugin.plugin_id
    assert architecture_id.value in outcome.detail
    assert compile_architecture(architecture_id) == outcome
    with pytest.raises(TypeError, match="requires"):
        plugin.compile(object())
    with pytest.raises(ValueError, match="does not match"):
        compile_architecture(ArchitectureId.DEMO_G, request)


def test_configured_requests_validate_their_configuration() -> None:
    tiny = GPT2Config(layers=1, hidden_size=8, heads=2, intermediate_size=16, vocabulary_size=16)
    request = GPT2CompileRequest(config=tiny)
    assert request.config is tiny and request.architecture_id is ArchitectureId.GPT2
    with pytest.raises(TypeError, match="GPT2Config"):
        GPT2CompileRequest(config=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="execution_shape"):
        KimiK3CompileRequest(execution_shape=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numerical_profile_id"):
        InklingCompileRequest(numerical_profile_id=" ")
    with pytest.raises(InvalidArtifact, match="trace_digest"):
        DeepSeekV4ProCompileRequest(trace_digest="not a digest")
