from __future__ import annotations

import math
from pathlib import Path

from circuit_cut_analysis.mincut import minimum_vertex_cut
from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.models.gpt2_circuit import (
    GPT2IndexedCircuit,
    build_gpt2_indexed_circuit,
)
from circuit_cut_analysis.models.gpt2_partition import lifted_downstream_cut
from circuit_cut_analysis.models.gpt2_sampling_study import (
    build_gpt2_region_units,
    certified_adversary_bits,
    expected_checked_gates,
    greedy_protection_probabilities,
    output_frontier_bits,
    run_gpt2_sampling_study,
    write_gpt2_sampling_study,
)
from circuit_cut_analysis.profiles import ServingProfile
from circuit_cut_analysis.sampling_study import RegionUnit

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


def test_region_units_cover_every_computed_gate_disjointly() -> None:
    indexed = _witness_indexed()
    for granularity in ("row", "row-layer", "row-layer-band"):
        units = build_gpt2_region_units(indexed, granularity=granularity)
        assert sum(unit.checked_gate_count for unit in units) == (
            indexed.circuit.computed_gate_count
        )
        assert len({unit.id for unit in units}) == len(units)


def test_unit_capacity_upper_bounds_dominate_the_explicit_joint_oracle() -> None:
    indexed = _witness_indexed()
    explicit = indexed.circuit.materialize(max_gates=5_000, max_edges=50_000)
    frontier = output_frontier_bits(indexed)
    units = build_gpt2_region_units(indexed, granularity="row-layer")

    ref_ids_by_family: dict[str, list[str]] = {}
    for ref in indexed.circuit.iter_gate_refs(max_gates=5_000):
        if indexed.circuit.families[ref.family].op != "input":
            ref_ids_by_family.setdefault(ref.family, []).append(ref.id)

    row_of_gate: dict[str, str] = {}
    for ref in indexed.circuit.iter_gate_refs(max_gates=5_000):
        family = indexed.circuit.families[ref.family]
        if family.op == "input":
            continue
        cut = lifted_downstream_cut(indexed, ref)
        row_of_gate[ref.id] = ",".join(sorted(gate.id for gate in cut))

    for unit in units:
        member_gates = frozenset(
            gate_id
            for family, gate_ids in ref_ids_by_family.items()
            for gate_id in gate_ids
            if _unit_owns(indexed, unit, family, gate_id)
        )
        if not member_gates:
            continue
        joint = minimum_vertex_cut(explicit, member_gates)
        assert joint.exact_capacity is not None
        exact_bits = float(joint.exact_capacity.width_bits)
        certified = min(unit.capacity_upper_bits, frontier)
        assert exact_bits <= certified + 1e-9, (
            f"unit {unit.id}: exact {exact_bits} exceeds certified {certified}"
        )


def _unit_owns(
    indexed: GPT2IndexedCircuit,
    unit: RegionUnit,
    family_name: str,
    gate_id: str,
) -> bool:
    from circuit_cut_analysis.models.gpt2_partition import (
        _assignment_row,
        _owner_plan,
        _ref_schedule,
    )

    ref = indexed.circuit.ref_from_id(gate_id)
    plan = _owner_plan(family_name)
    layer, position = _ref_schedule(indexed, ref)
    row_id = _assignment_row(indexed, plan, position=position, layer=layer)
    expected = (row_id, layer)
    return unit.id == "/".join(str(part) for part in expected)


def test_certified_adversary_math_is_exact_for_two_units() -> None:
    indexed = _witness_indexed()
    units = build_gpt2_region_units(indexed, granularity="row")[:2]
    frontier = output_frontier_bits(indexed)

    fully_checked = {unit.id: 1.0 for unit in units}
    certified, witness, residual = certified_adversary_bits(
        units,
        fully_checked,
        detection_threshold=0.99,
        frontier_bits=frontier,
    )
    assert certified == witness == residual == 0.0

    unchecked = {unit.id: 0.0 for unit in units}
    certified, witness, _ = certified_adversary_bits(
        units,
        unchecked,
        detection_threshold=0.99,
        frontier_bits=frontier,
    )
    expected = min(sum(unit.capacity_upper_bits for unit in units), frontier)
    assert math.isclose(certified, expected)
    assert math.isclose(witness, expected)


def test_greedy_protection_never_exceeds_budget_and_beats_nothing() -> None:
    indexed = _witness_indexed()
    units = build_gpt2_region_units(indexed, granularity="row")
    frontier = output_frontier_bits(indexed)
    total = sum(unit.checked_gate_count for unit in units)
    budget = 0.5 * total

    probabilities, protected_count = greedy_protection_probabilities(
        units,
        budget,
        detection_threshold=0.9,
        frontier_bits=frontier,
    )

    assert expected_checked_gates(units, probabilities) <= budget + 1e-6
    assert 0 <= protected_count <= len(units)
    protected_bound, _, _ = certified_adversary_bits(
        units,
        probabilities,
        detection_threshold=0.9,
        frontier_bits=frontier,
    )
    idle_bound, _, _ = certified_adversary_bits(
        units,
        {unit.id: 0.0 for unit in units},
        detection_threshold=0.9,
        frontier_bits=frontier,
    )
    assert protected_bound <= idle_bound


def test_study_runs_end_to_end_and_writes_artifacts(tmp_path: Path) -> None:
    indexed = _witness_indexed()
    study = run_gpt2_sampling_study(
        indexed,
        granularities=("row",),
        budget_fractions=(1e-2, 1e-1),
        detection_thresholds=(0.9,),
    )

    assert study.outcomes
    strategies = {outcome.strategy for outcome in study.outcomes}
    assert "uniform-independent" in strategies
    assert "greedy-protection" in strategies
    for outcome in study.outcomes:
        assert outcome.certified_upper_bits <= study.output_frontier_bits + 1e-9
        assert outcome.greedy_attack_witness_bits <= (
            outcome.certified_upper_bits + 1e-9
        )

    json_path, markdown_path = write_gpt2_sampling_study(study, tmp_path)
    assert json_path.exists()
    assert markdown_path.exists()
    assert "certified" in markdown_path.read_text().lower()
