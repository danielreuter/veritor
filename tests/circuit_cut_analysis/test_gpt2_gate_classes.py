from __future__ import annotations

import itertools
from collections import Counter

from circuit_cut_analysis.mincut import minimum_vertex_cut
from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.models.gpt2_circuit import (
    GPT2IndexedCircuit,
    build_gpt2_indexed_circuit,
)
from circuit_cut_analysis.models.gpt2_gate_classes import (
    GPT2ClassGranularity,
    build_gpt2_gate_class_catalog,
)
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


def _witness_indexed() -> GPT2IndexedCircuit:
    return build_gpt2_indexed_circuit(
        2,
        3,
        config=LIFT_WITNESS,
        profile=SCALED_MIXED_PROFILE,
    )


def test_catalog_classes_cover_every_computed_scalar_gate() -> None:
    indexed = _witness_indexed()
    for granularity in GPT2ClassGranularity:
        catalog = build_gpt2_gate_class_catalog(
            indexed,
            granularity=granularity,
        )
        observed: Counter[str] = Counter()
        for ref in indexed.circuit.iter_gate_refs(max_gates=5_000):
            if indexed.circuit.families[ref.family].op != "input":
                observed[catalog.class_id_for(ref)] += 1
        declared = {
            gate_class.id: gate_class.gate_count
            for gate_class in catalog.partition.classes
        }
        assert dict(observed) == declared
        assert catalog.computed_gate_count == indexed.circuit.computed_gate_count


def test_count_capacity_envelope_dominates_actual_tiny_attacks() -> None:
    indexed = _witness_indexed()
    explicit = indexed.circuit.materialize(max_gates=5_000, max_edges=50_000)
    catalog = build_gpt2_gate_class_catalog(
        indexed,
        granularity=GPT2ClassGranularity.ROW_LAYER,
    )
    class_index = {
        gate_class.id: index
        for index, gate_class in enumerate(catalog.partition.classes)
    }

    representatives = []
    represented_classes: set[str] = set()
    for ref in indexed.circuit.iter_gate_refs(max_gates=5_000):
        if indexed.circuit.families[ref.family].op == "input":
            continue
        class_id = catalog.class_id_for(ref)
        if class_id not in represented_classes:
            represented_classes.add(class_id)
            representatives.append(ref)
        if len(representatives) == 12:
            break

    attacks = [
        subset
        for size in range(1, 4)
        for subset in itertools.combinations(representatives, size)
    ]
    attacks.append(tuple(representatives))
    for attack in attacks:
        counts = [0] * catalog.class_count
        for ref in attack:
            counts[class_index[catalog.class_id_for(ref)]] += 1
        certified = catalog.capacity_upper_bound(tuple(counts))
        exact = minimum_vertex_cut(
            explicit,
            (ref.id for ref in attack),
        ).exact_capacity
        assert exact is not None
        assert exact <= certified


def test_capacity_coalescing_preserves_gate_count_and_frontier() -> None:
    indexed = _witness_indexed()
    catalog = build_gpt2_gate_class_catalog(indexed)
    coalesced = catalog.coalesced_for_linear_game()

    assert coalesced.total_gate_count == catalog.computed_gate_count
    assert coalesced.output_frontier == catalog.partition.output_frontier
    assert len(coalesced.classes) <= catalog.class_count
