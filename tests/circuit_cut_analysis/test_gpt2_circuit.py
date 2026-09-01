from __future__ import annotations

import pytest

from circuit_cut_analysis.indexed import ExpansionLimitExceeded
from circuit_cut_analysis.indexed_mincut import minimum_vertex_cut_indexed
from circuit_cut_analysis.mincut import minimum_vertex_cut
from circuit_cut_analysis.models.gpt2 import (
    GPT2_SMALL,
    GPT2Config,
    analyze_gpt2_execution,
)
from circuit_cut_analysis.models.gpt2_circuit import build_gpt2_indexed_circuit
from circuit_cut_analysis.models.gpt2_partition import (
    compute_gpt2_canonical_partition,
    lifted_downstream_cut,
)
from circuit_cut_analysis.partition import partition_gate_cuts
from circuit_cut_analysis.profiles import ServingProfile

TINY_GPT2 = GPT2Config(
    model_id="tiny-gpt2",
    layers=1,
    hidden_size=2,
    heads=1,
    intermediate_size=3,
    vocabulary_size=4,
    max_context=8,
)

SCALED_MIXED_PROFILE = ServingProfile(
    id="scaled-mixed",
    description="Small integer analogue of the default mixed-precision profile.",
    weight_bits=2,
    activation_boundary_bits=2,
    kv_cache_bits=2,
    accumulator_bits=4,
    reduction_bits=4,
    nonlinear_internal_bits=4,
    probability_boundary_bits=2,
    residual_bits=2,
    logit_bits=2,
    assumptions=(),
)


def test_tiny_indexed_circuit_matches_independent_accounting_formulas() -> None:
    indexed = build_gpt2_indexed_circuit(2, 2, config=TINY_GPT2)
    accounting = analyze_gpt2_execution(2, 2, config=TINY_GPT2)

    assert indexed.processed_positions == 3
    assert indexed.prediction_positions == (1, 2)
    assert indexed.circuit.primitive_gate_count == accounting.total_unit_gates == 506
    assert (
        dict(indexed.circuit.primitive_counts) == accounting.total_primitives.as_dict()
    )


def test_tiny_indexed_wiring_is_bidirectional_and_materializes_to_a_dag() -> None:
    indexed = build_gpt2_indexed_circuit(2, 2, config=TINY_GPT2)
    circuit = indexed.circuit

    assert circuit.validate_bidirectional(max_gates=1_000, max_edges=2_000) == 772
    expanded = circuit.materialize(max_gates=1_000, max_edges=2_000)
    assert len(expanded.gates) == circuit.gate_count == 571
    assert len(expanded.edges) == 772
    assert len(expanded.topological_order) == 571
    assert expanded.outputs == frozenset({"output/argmax[0]", "output/argmax[1]"})


def test_minimum_dimensions_keep_degenerate_reductions_exact() -> None:
    config = GPT2Config(
        model_id="minimum-gpt2",
        layers=1,
        hidden_size=1,
        heads=1,
        intermediate_size=1,
        vocabulary_size=2,
        max_context=2,
    )
    indexed = build_gpt2_indexed_circuit(1, 1, config=config)
    accounting = analyze_gpt2_execution(1, 1, config=config)

    assert indexed.circuit.primitive_gate_count == accounting.total_unit_gates == 61
    assert (
        indexed.circuit.validate_bidirectional(
            max_gates=100,
            max_edges=100,
        )
        == 86
    )
    expanded = indexed.circuit.materialize(max_gates=100, max_edges=100)
    assert len(expanded.gates) == 72
    assert len(expanded.edges) == 86
    partition = partition_gate_cuts(expanded)
    assert len(partition.groups) == 1
    assert partition.groups[0].cut == frozenset({"output/argmax[0]"})
    assert partition.groups[0].source_gates == expanded.computed_gates


def test_generated_token_feedback_and_persistent_kv_edges_are_queryable() -> None:
    circuit = build_gpt2_indexed_circuit(2, 2, config=TINY_GPT2).circuit
    argmax = circuit.families["output/argmax"].ref(0)
    key = circuit.families["blocks/attention/k_projection/write"].ref(0, 0, 0)
    value = circuit.families["blocks/attention/v_projection/write"].ref(0, 0, 0)

    assert circuit.successors(argmax) == frozenset(
        {
            circuit.families["embedding/generated_lookup"].ref(0, 0),
            circuit.families["embedding/generated_lookup"].ref(0, 1),
        }
    )
    assert {ref.index for ref in circuit.successors(key)} == {
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 2, 0, 0, 0),
    }
    assert {ref.index for ref in circuit.successors(value)} == {
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 2, 0, 0, 0),
    }


def test_layernorm_shared_mean_fans_out_to_every_coordinate() -> None:
    circuit = build_gpt2_indexed_circuit(2, 2, config=TINY_GPT2).circuit
    mean = circuit.families["blocks/ln1/mean"].ref(0, 1)

    assert circuit.successors(mean) == frozenset(
        {
            circuit.families["blocks/ln1/centered"].ref(0, 1, 0),
            circuit.families["blocks/ln1/centered"].ref(0, 1, 1),
        }
    )


def test_one_lazy_dot_product_corridor_has_the_explicit_write_out_cut() -> None:
    circuit = build_gpt2_indexed_circuit(2, 2, config=TINY_GPT2).circuit
    source = circuit.families["blocks/attention/q_projection/mul"].ref(0, 1, 0, 0)
    output = circuit.families["blocks/attention/q_projection/write"].ref(0, 1, 0)
    result = minimum_vertex_cut_indexed(
        circuit,
        {source},
        {output},
        max_gates=20,
        max_edges=30,
    )

    assert result.cut == frozenset({output})
    assert result.result.width_bits == 16
    assert result.expanded_gate_count == 4
    assert result.expanded_edge_count == 3


def test_tiny_power_two_partition_matches_oracle_gate_for_gate() -> None:
    config = GPT2Config(
        model_id="tiny-power-two-oracle",
        layers=1,
        hidden_size=1,
        heads=1,
        intermediate_size=1,
        vocabulary_size=2,
        max_context=4,
    )
    indexed = build_gpt2_indexed_circuit(2, 2, config=config)
    computed = compute_gpt2_canonical_partition(indexed)

    assert computed.explicit_partition is not None
    explicit = indexed.circuit.materialize(max_gates=500, max_edges=1_000)
    saw_dead = False
    for ref in indexed.circuit.iter_gate_refs(max_gates=500):
        if indexed.circuit.families[ref.family].op == "input":
            continue
        lifted = computed.singleton_result(ref)
        oracle = minimum_vertex_cut(explicit, {ref.id})
        assert lifted.source_most_cut == oracle.source_most_cut
        assert lifted.downstream_most_cut == oracle.downstream_most_cut
        assert lifted.cut == oracle.cut
        assert lifted.exact_capacity == oracle.exact_capacity
        saw_dead |= not oracle.cut
    assert saw_dead


def test_lifted_exceptional_frontiers_match_the_power_two_oracle() -> None:
    config = GPT2Config(
        model_id="lift-witness",
        layers=2,
        hidden_size=3,
        heads=1,
        intermediate_size=3,
        vocabulary_size=4,
        max_context=8,
    )
    indexed = build_gpt2_indexed_circuit(
        2,
        3,
        config=config,
        profile=SCALED_MIXED_PROFILE,
    )
    explicit = indexed.circuit.materialize(max_gates=5_000, max_edges=50_000)
    representatives = (
        indexed.circuit.families["blocks/attention/softmax/max"].ref(0, 1, 1, 0),
        indexed.circuit.families["blocks/attention/softmax/exp"].ref(0, 1, 0, 0),
        indexed.circuit.families["blocks/attention/softmax/reciprocal"].ref(0, 1, 0),
        indexed.circuit.families["blocks/attention/q_projection/mul"].ref(0, 0, 0, 0),
    )

    for ref in representatives:
        oracle = minimum_vertex_cut(explicit, {ref.id})
        expected = frozenset(
            indexed.circuit.ref_from_id(gate_id)
            for gate_id in oracle.downstream_most_cut
        )
        assert lifted_downstream_cut(indexed, ref) == expected


def test_lift_chooses_between_local_softmax_and_output_pair_frontiers() -> None:
    config = GPT2Config(
        model_id="lift-frontier-ordering",
        layers=2,
        hidden_size=3,
        heads=1,
        intermediate_size=3,
        vocabulary_size=3,
        max_context=8,
    )
    indexed = build_gpt2_indexed_circuit(
        2,
        2,
        config=config,
        profile=SCALED_MIXED_PROFILE,
    )
    explicit = indexed.circuit.materialize(max_gates=5_000, max_edges=50_000)
    representatives = (
        indexed.circuit.families["blocks/attention/softmax/shifted"].ref(0, 0, 0, 0),
        indexed.circuit.families["blocks/attention/softmax/exp"].ref(0, 0, 0, 0),
        indexed.circuit.families["blocks/attention/softmax/reciprocal"].ref(0, 0, 0),
        indexed.circuit.families["blocks/attention/softmax/max"].ref(0, 1, 1, 0),
        indexed.circuit.families["blocks/attention/softmax/shifted"].ref(0, 1, 0, 0),
        indexed.circuit.families["blocks/attention/softmax/exp"].ref(0, 1, 0, 0),
        indexed.circuit.families["blocks/attention/softmax/denominator"].ref(
            0, 1, 1, 0
        ),
        indexed.circuit.families["blocks/attention/softmax/reciprocal"].ref(0, 1, 0),
    )

    for ref in representatives:
        oracle = minimum_vertex_cut(explicit, {ref.id})
        expected = frozenset(
            indexed.circuit.ref_from_id(gate_id)
            for gate_id in oracle.downstream_most_cut
        )
        assert lifted_downstream_cut(indexed, ref) == expected


def test_full_case_is_a_small_descriptor_for_the_exact_42_billion_gate_graph() -> None:
    indexed = build_gpt2_indexed_circuit(config=GPT2_SMALL)
    circuit = indexed.circuit

    assert len(circuit.families) == 82
    assert len(circuit.edge_rules) == 119
    assert circuit.gate_count == 42_387_485_394
    assert circuit.primitive_gate_count == 42_361_101_422
    assert dict(circuit.primitive_counts) == {
        "add": 21_158_279_101,
        "argmax": 100,
        "exp": 2_865_600,
        "max": 2_836_944,
        "mul": 21_189_750_110,
        "reciprocal": 28_656,
        "rsqrt": 4_975,
        "tanh": 7_335_936,
    }
    assert circuit.cache_info().gate_entries == 0
    assert circuit.cache_info().predecessor_entries == 0
    assert circuit.cache_info().successor_entries == 0

    with pytest.raises(ExpansionLimitExceeded, match="42,387,485,394"):
        compute_gpt2_canonical_partition(indexed, force_explicit=True)
