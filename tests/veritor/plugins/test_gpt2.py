from __future__ import annotations

from circuit_cut_analysis.indexed import GateRef
from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.models.gpt2_capacity_oracle import (
    GPT2StructuralCapacityOracle,
)
from circuit_cut_analysis.models.gpt2_circuit import build_gpt2_indexed_circuit
from circuit_cut_analysis.models.gpt2_gate_classes import (
    GPT2ClassGranularity,
    GPT2GateClassCatalog,
    build_gpt2_gate_class_catalog,
)
from circuit_cut_analysis.profiles import VLLM_FP16_REFERENCE, ServingProfile
from veritor.core import Capability, Unsupported
from veritor.plugins import (
    CapacityClaimKind,
    GPT2CompileRequest,
    GreedyTextExecutionShape,
    IndexedStructureArtifact,
    compile_gpt2,
)


def _tiny_config() -> GPT2Config:
    return GPT2Config(
        model_id="gpt2-plugin-tiny",
        layers=1,
        hidden_size=2,
        heads=1,
        intermediate_size=3,
        vocabulary_size=5,
        max_context=4,
    )


def _catalog_config() -> GPT2Config:
    return GPT2Config(
        model_id="gpt2-plugin-catalog",
        layers=3,
        hidden_size=9,
        heads=3,
        intermediate_size=9,
        vocabulary_size=4,
        max_context=4,
    )


def _three_bit_profile() -> ServingProfile:
    return ServingProfile(
        id="all-three-bit-test",
        description="Small exact-width profile for lifted catalog tests.",
        weight_bits=3,
        activation_boundary_bits=3,
        kv_cache_bits=3,
        accumulator_bits=3,
        reduction_bits=3,
        nonlinear_internal_bits=3,
        probability_boundary_bits=3,
        residual_bits=3,
        logit_bits=3,
        assumptions=("Every represented structural boundary has width 3.",),
    )


def _tiny_request() -> GPT2CompileRequest:
    return GPT2CompileRequest(
        execution_shape=GreedyTextExecutionShape(
            prompt_tokens=2,
            generated_tokens=2,
        ),
        config=_tiny_config(),
        profile=VLLM_FP16_REFERENCE,
    )


def test_gpt2_matches_direct_indexed_builder_metadata_and_output_order() -> None:
    request = _tiny_request()
    artifact = compile_gpt2(request)
    direct = build_gpt2_indexed_circuit(
        request.shape.prompt_tokens,
        request.shape.generated_tokens,
        config=request.config,
        profile=request.profile,
    )
    assert isinstance(artifact, IndexedStructureArtifact)
    assert artifact.ordered_output_refs == direct.output_refs
    assert artifact.output_refs == direct.output_refs
    assert artifact.gate_count == direct.circuit.gate_count
    assert artifact.computed_gate_count == direct.circuit.computed_gate_count
    assert artifact.primitive_gate_count == direct.circuit.primitive_gate_count
    assert artifact.gate_family_count == len(direct.circuit.families)
    assert artifact.edge_rule_count == len(direct.circuit.edge_rules)
    assert artifact.gate_domain.count == direct.circuit.gate_count
    assert artifact.computed_gate_domain.count == direct.circuit.computed_gate_count
    assert tuple(artifact.indexed.circuit.families) == tuple(direct.circuit.families)
    assert tuple(
        (rule.name, rule.source_family, rule.target_family)
        for rule in artifact.indexed.circuit.edge_rules
    ) == tuple(
        (rule.name, rule.source_family, rule.target_family)
        for rule in direct.circuit.edge_rules
    )
    assert tuple(ref.index[-1] for ref in artifact.output_refs) == tuple(
        range(request.shape.generated_tokens)
    )


def test_gpt2_expanded_edge_count_is_available_only_on_demand() -> None:
    artifact = compile_gpt2(_tiny_request())
    max_gates = artifact.gate_count
    max_edges = 1_000_000
    plugin_count = artifact.count_expanded_edges(
        max_gates=max_gates,
        max_edges=max_edges,
    )
    direct_count = sum(
        1
        for _ in artifact.indexed.circuit.iter_edges(
            max_gates=max_gates,
            max_edges=max_edges,
        )
    )
    assert plugin_count == direct_count
    assert plugin_count > 0


def test_gpt2_structural_oracle_factory_preserves_claim_strength() -> None:
    artifact = compile_gpt2(_tiny_request())
    oracle = artifact.structural_oracle(
        max_exact_gates=artifact.gate_count,
        max_exact_edges=1_000_000,
    )
    assert isinstance(oracle, GPT2StructuralCapacityOracle)
    assert artifact.bound_provider.claim_kind is CapacityClaimKind.CERTIFIED_INTERVAL
    result = artifact.bound_provider.evaluate(
        {artifact.output_refs[0]},
        max_exact_gates=artifact.gate_count,
        max_exact_edges=1_000_000,
    )
    assert result.claim_kind in (
        CapacityClaimKind.EXACT,
        CapacityClaimKind.CERTIFIED_INTERVAL,
    )
    assert result.lower_bound <= result.upper_bound


def test_gpt2_gate_class_catalog_factory_matches_direct_builder() -> None:
    request = GPT2CompileRequest(
        execution_shape=GreedyTextExecutionShape(
            prompt_tokens=2,
            generated_tokens=2,
        ),
        config=_catalog_config(),
        profile=_three_bit_profile(),
    )
    artifact = compile_gpt2(request)
    assert artifact.bound_provider.catalog_available
    catalog = artifact.gate_class_catalog(
        granularity=GPT2ClassGranularity.ROW_LAYER_BAND,
        position_bands=2,
    )
    assert isinstance(catalog, GPT2GateClassCatalog)
    direct = build_gpt2_gate_class_catalog(
        artifact.indexed,
        granularity=GPT2ClassGranularity.ROW_LAYER_BAND,
        position_bands=2,
    )
    assert catalog.granularity is direct.granularity
    assert catalog.position_bands == direct.position_bands
    assert catalog.partition == direct.partition
    assert catalog.computed_gate_count == artifact.computed_gate_count


def test_gpt2_unavailable_catalog_and_protocol_features_are_typed() -> None:
    artifact = compile_gpt2(_tiny_request())
    catalog = artifact.gate_class_catalog()
    assert isinstance(catalog, Unsupported)
    assert catalog.capability is Capability.STATIC_PARTITION

    assert not hasattr(artifact, "gate_at")
    replay = artifact.replay()
    execution = artifact.execute()
    verification = artifact.verify()
    assert isinstance(replay, Unsupported)
    assert isinstance(execution, Unsupported)
    assert isinstance(verification, Unsupported)
    assert replay.capability is Capability.STATIC_PARTITION
    assert execution.capability is Capability.EXECUTE
    assert verification.capability is Capability.VERIFY


def test_gpt2_output_refs_are_concrete_gate_refs() -> None:
    artifact = compile_gpt2(_tiny_request())
    assert artifact.output_refs
    assert all(isinstance(ref, GateRef) for ref in artifact.output_refs)
    assert all(artifact.gate_domain.contains(ref) for ref in artifact.output_refs)


def test_gpt2_gate_ref_rank_unrank_is_exhaustive_on_tiny_config() -> None:
    artifact = compile_gpt2(_tiny_request())
    expected = tuple(
        GateRef(family.name, index)
        for family in artifact.indexed.circuit.families.values()
        for index in family.domain.iter_indices()
    )
    assert len(expected) == artifact.gate_domain.count
    for rank, ref in enumerate(expected):
        assert artifact.gate_domain.rank(ref) == rank
        assert artifact.gate_domain.unrank(rank) == ref
