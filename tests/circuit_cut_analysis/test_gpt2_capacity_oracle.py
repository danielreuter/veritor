from __future__ import annotations

import itertools
import random

import pytest

from circuit_cut_analysis.capacity_oracle import StructuralCapacityStatus
from circuit_cut_analysis.circuit import CircuitDAG
from circuit_cut_analysis.indexed import ExpansionLimitExceeded
from circuit_cut_analysis.mincut import minimum_vertex_cut
from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.models.gpt2_capacity_oracle import (
    GPT2StructuralCapacityOracle,
)
from circuit_cut_analysis.models.gpt2_circuit import (
    GPT2IndexedCircuit,
    build_gpt2_indexed_circuit,
)
from circuit_cut_analysis.models.gpt2_partition import lifted_certificate_reasons
from circuit_cut_analysis.profiles import ServingProfile

LIFT_WITNESS = GPT2Config(
    model_id="lift-witness",
    layers=2,
    hidden_size=3,
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


def _witness_setup() -> tuple[GPT2IndexedCircuit, CircuitDAG]:
    indexed = build_gpt2_indexed_circuit(
        2,
        3,
        config=LIFT_WITNESS,
        profile=SCALED_MIXED_PROFILE,
    )
    explicit = indexed.circuit.materialize(max_gates=5_000, max_edges=50_000)
    return indexed, explicit


def test_corridor_route_is_exact_for_small_supports() -> None:
    indexed, explicit = _witness_setup()
    oracle = GPT2StructuralCapacityOracle(indexed)
    families = indexed.circuit.families
    support = frozenset(
        (
            families["blocks/attention/softmax/exp"].ref(0, 1, 0, 0),
            families["blocks/attention/softmax/reciprocal"].ref(0, 1, 0),
        )
    )

    evaluation = oracle.evaluate(support)
    reference = minimum_vertex_cut(explicit, {ref.id for ref in support})

    assert evaluation.status is StructuralCapacityStatus.EXACT
    assert evaluation.exact_capacity == reference.exact_capacity
    assert evaluation.cut_gate_ids == reference.cut
    assert evaluation.method == "indexed-corridor-max-flow"


def test_interval_route_brackets_the_explicit_oracle_on_random_supports() -> None:
    indexed, explicit = _witness_setup()
    assert lifted_certificate_reasons(indexed) == ()
    constrained = GPT2StructuralCapacityOracle(
        indexed,
        max_exact_gates=10,
        max_exact_edges=10,
    )
    computed_refs = tuple(
        ref
        for ref in indexed.circuit.iter_gate_refs(max_gates=5_000)
        if indexed.circuit.families[ref.family].op != "input"
    )
    rng = random.Random(20260829)

    bounded_seen = 0
    for _ in range(40):
        support = frozenset(rng.sample(computed_refs, rng.randint(1, 6)))
        reference = minimum_vertex_cut(explicit, {ref.id for ref in support})
        assert reference.exact_capacity is not None
        evaluation = constrained.evaluate(support)

        assert evaluation.lower_bound <= reference.exact_capacity
        assert evaluation.upper_bound >= reference.exact_capacity
        assert evaluation.method == "lifted-region-certified-interval"
        bounded_seen += evaluation.status is StructuralCapacityStatus.BOUNDED
    assert bounded_seen > 0


def test_single_region_support_closes_the_interval_exactly() -> None:
    indexed, explicit = _witness_setup()
    constrained = GPT2StructuralCapacityOracle(
        indexed,
        max_exact_gates=10,
        max_exact_edges=10,
    )
    families = indexed.circuit.families
    mul = families["blocks/attention/q_projection/mul"]
    support = frozenset(
        (
            mul.ref(1, 2, 0, 0),
            mul.ref(1, 2, 0, 1),
            mul.ref(1, 2, 0, 2),
        )
    )

    evaluation = constrained.evaluate(support)
    reference = minimum_vertex_cut(explicit, {ref.id for ref in support})

    assert evaluation.status is StructuralCapacityStatus.EXACT
    assert evaluation.exact_capacity == reference.exact_capacity
    assert evaluation.method == "lifted-region-certified-interval"


def test_empty_and_input_supports_are_handled_explicitly() -> None:
    indexed, _ = _witness_setup()
    oracle = GPT2StructuralCapacityOracle(indexed)

    empty = oracle.evaluate(frozenset())
    assert empty.status is StructuralCapacityStatus.EXACT
    assert empty.exact_capacity is not None
    assert empty.exact_capacity.width_bits == 0

    input_ref = next(
        ref
        for ref in indexed.circuit.iter_gate_refs(max_gates=5_000)
        if indexed.circuit.families[ref.family].op == "input"
    )
    with pytest.raises(ValueError, match="computed gates"):
        oracle.evaluate(frozenset((input_ref,)))


def test_unsupported_lifted_profile_refuses_to_fabricate_an_interval() -> None:
    config = GPT2Config(
        model_id="unsupported-tiny",
        layers=1,
        hidden_size=1,
        heads=1,
        intermediate_size=1,
        vocabulary_size=2,
        max_context=2,
    )
    indexed = build_gpt2_indexed_circuit(1, 1, config=config)
    assert lifted_certificate_reasons(indexed) != ()
    constrained = GPT2StructuralCapacityOracle(
        indexed,
        max_exact_gates=2,
        max_exact_edges=2,
    )
    computed = next(
        ref
        for ref in indexed.circuit.iter_gate_refs(max_gates=100)
        if indexed.circuit.families[ref.family].op != "input"
    )

    with pytest.raises(ExpansionLimitExceeded, match="lifted certificates"):
        constrained.evaluate(frozenset((computed,)))


def test_exhaustive_pairs_within_one_layer_head_are_bracketed() -> None:
    indexed, explicit = _witness_setup()
    constrained = GPT2StructuralCapacityOracle(
        indexed,
        max_exact_gates=10,
        max_exact_edges=10,
    )
    families = indexed.circuit.families
    probes = (
        families["blocks/attention/softmax/probability"].ref(0, 1, 0, 0),
        families["blocks/attention/softmax/probability"].ref(0, 1, 1, 0),
        families["blocks/attention/softmax/reciprocal"].ref(0, 1, 0),
        families["blocks/attention/value_reduction/write"].ref(0, 1, 0),
        families["output/argmax"].ref(0),
        families["output/argmax"].ref(2),
    )

    for left, right in itertools.combinations(probes, 2):
        support = frozenset((left, right))
        reference = minimum_vertex_cut(explicit, {ref.id for ref in support})
        assert reference.exact_capacity is not None
        evaluation = constrained.evaluate(support)
        assert evaluation.lower_bound <= reference.exact_capacity
        assert evaluation.upper_bound >= reference.exact_capacity
