from __future__ import annotations

from collections.abc import Iterable

import pytest

from circuit_cut_analysis.circuit import CircuitDAG, Gate
from circuit_cut_analysis.indexed import (
    CausalPairsDomain,
    CausalReductionStepsDomain,
    EdgeRule,
    ExpansionLimitExceeded,
    GateCapacity,
    GateFamily,
    GateRef,
    IndexedCircuit,
    ProductDomain,
    RectangularDomain,
)
from circuit_cut_analysis.mincut import minimum_vertex_cut


def _row_reduction_circuit(
    *,
    cache_capacity: int = 4_096,
    max_cached_adjacency: int = 4_096,
) -> IndexedCircuit:
    row = RectangularDomain((2,))
    row_term = RectangularDomain((2, 3))
    scalar = RectangularDomain(())
    families = (
        GateFamily("input", row, ("row",), GateCapacity.bits(16), "input"),
        GateFamily(
            "project/mul",
            row_term,
            ("row", "term"),
            GateCapacity.bits(32),
            "mul",
            "mul",
        ),
        GateFamily(
            "project/sum",
            row,
            ("row",),
            GateCapacity.bits(32),
            "add",
            "add",
        ),
        GateFamily(
            "project/write",
            row,
            ("row",),
            GateCapacity.bits(16),
            "write",
            tags=("boundary",),
        ),
        GateFamily(
            "merge",
            scalar,
            (),
            GateCapacity.bits(30),
            "add",
            "add",
        ),
    )
    rules = (
        EdgeRule(
            "input-to-products",
            "input",
            "project/mul",
            lambda target: ((target[0],),),
            lambda source: ((source[0], term) for term in range(3)),
        ),
        EdgeRule(
            "products-to-sums",
            "project/mul",
            "project/sum",
            lambda target: ((target[0], term) for term in range(3)),
            lambda source: ((source[0],),),
        ),
        EdgeRule(
            "sums-to-writes",
            "project/sum",
            "project/write",
            lambda target: (target,),
            lambda source: (source,),
        ),
        EdgeRule(
            "writes-to-merge",
            "project/write",
            "merge",
            lambda _target: ((0,), (1,)),
            lambda _source: ((),),
        ),
    )
    return IndexedCircuit(
        families,
        rules,
        (GateRef("merge"),),
        cache_capacity=cache_capacity,
        max_cached_adjacency=max_cached_adjacency,
    )


def _explicit_row_reduction_circuit() -> CircuitDAG:
    gates = [
        *(Gate(f"input[{row}]", 16, "input") for row in range(2)),
        *(
            Gate(f"project/mul[{row},{term}]", 32, "mul")
            for row in range(2)
            for term in range(3)
        ),
        *(Gate(f"project/sum[{row}]", 32, "add") for row in range(2)),
        *(Gate(f"project/write[{row}]", 16, "write") for row in range(2)),
        Gate("merge", 30, "add"),
    ]
    edges = [
        *(
            (f"input[{row}]", f"project/mul[{row},{term}]")
            for row in range(2)
            for term in range(3)
        ),
        *(
            (f"project/mul[{row},{term}]", f"project/sum[{row}]")
            for row in range(2)
            for term in range(3)
        ),
        *((f"project/sum[{row}]", f"project/write[{row}]") for row in range(2)),
        *((f"project/write[{row}]", "merge") for row in range(2)),
    ]
    return CircuitDAG(gates, edges, {"merge"})


def test_index_domains_are_finite_and_deterministic() -> None:
    causal = CausalPairsDomain(3)
    strict = CausalPairsDomain(3, strict=True)
    reductions = CausalReductionStepsDomain(3)
    product = ProductDomain((RectangularDomain((2,)), causal))

    assert causal.count == 6
    assert tuple(causal.iter_indices()) == (
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    )
    assert strict.count == 3
    assert tuple(strict.iter_indices()) == ((1, 0), (2, 0), (2, 1))
    assert reductions.count == 3
    assert tuple(reductions.iter_indices()) == ((1, 1), (2, 1), (2, 2))
    assert product.arity == 3
    assert product.count == 12
    assert product.contains((1, 2, 1))
    assert not product.contains((1, 1, 2))


def test_lazy_queries_are_exact_bidirectional_and_memoized() -> None:
    circuit = _row_reduction_circuit()
    product = GateRef("project/mul", (1, 2))
    row_sum = GateRef("project/sum", (1,))

    assert circuit.gate_count == 13
    assert circuit.computed_gate_count == 11
    assert dict(circuit.primitive_counts) == {"add": 3, "mul": 6}
    assert circuit.predecessors(product) == frozenset({GateRef("input", (1,))})
    assert circuit.successors(product) == frozenset({row_sum})
    assert circuit.predecessors(row_sum) == frozenset(
        GateRef("project/mul", (1, term)) for term in range(3)
    )

    circuit.predecessors(product)
    circuit.successors(product)
    circuit.gate(product)
    circuit.gate(product)
    cache = circuit.cache_info()
    assert cache.predecessor_hits == 1
    assert cache.successor_hits == 1
    assert cache.gate_hits == 1
    assert circuit.validate_bidirectional(max_gates=20, max_edges=30) == 16


def test_lru_cache_is_bounded_and_oversized_adjacency_is_not_retained() -> None:
    circuit = _row_reduction_circuit(
        cache_capacity=1,
        max_cached_adjacency=2,
    )
    first = GateRef("project/mul", (0, 0))
    second = GateRef("project/mul", (0, 1))
    row_sum = GateRef("project/sum", (0,))

    circuit.gate(first)
    circuit.gate(second)
    circuit.predecessors(first)
    circuit.predecessors(second)
    assert len(circuit.predecessors(row_sum)) == 3

    cache = circuit.cache_info()
    assert cache.capacity == 1
    assert cache.max_cached_adjacency == 2
    assert cache.gate_entries == 1
    assert cache.predecessor_entries == 1
    assert cache.gate_evictions == 1
    assert cache.predecessor_evictions == 1
    assert cache.oversized_predecessors_skipped == 1


def test_complete_lazy_expansion_equals_independent_explicit_graph() -> None:
    lazy = _row_reduction_circuit().materialize(max_gates=20, max_edges=30)
    explicit = _explicit_row_reduction_circuit()

    assert lazy.gates == explicit.gates
    assert lazy.edges == explicit.edges
    assert lazy.outputs == explicit.outputs
    assert lazy.topological_order == explicit.topological_order

    lazy_cut = minimum_vertex_cut(lazy, {"project/mul[0,0]"})
    explicit_cut = minimum_vertex_cut(explicit, {"project/mul[0,0]"})
    assert lazy_cut == explicit_cut
    assert lazy_cut.cut == frozenset({"project/write[0]"})
    assert lazy_cut.width_bits == 16


def test_corridor_expansion_omits_unreachable_parallel_row() -> None:
    circuit = _row_reduction_circuit()
    source = GateRef("project/mul", (0, 0))
    row_output = GateRef("project/write", (0,))

    corridor = circuit.materialize_corridor(
        {source},
        {row_output},
        max_gates=20,
        max_edges=30,
    )

    assert set(corridor.gates) == {
        "project/mul[0,0]",
        "project/sum[0]",
        "project/write[0]",
    }
    assert corridor.edges == frozenset(
        {
            ("project/mul[0,0]", "project/sum[0]"),
            ("project/sum[0]", "project/write[0]"),
        }
    )


def test_expansion_limits_fail_before_complete_materialization() -> None:
    circuit = _row_reduction_circuit()

    with pytest.raises(ExpansionLimitExceeded, match="13 gates"):
        circuit.materialize(max_gates=12, max_edges=30)
    with pytest.raises(ExpansionLimitExceeded, match="edge"):
        circuit.materialize(max_gates=20, max_edges=10)


def test_non_power_of_two_capacity_stays_exact_through_materialization() -> None:
    token = GateFamily(
        "token",
        RectangularDomain(()),
        (),
        GateCapacity.values(50_257),
        "argmax",
        "argmax",
    )
    circuit = IndexedCircuit((token,), (), (GateRef("token"),))

    assert circuit.gate(GateRef("token")).capacity.cardinality == 50_257
    assert circuit.gate(GateRef("token")).capacity.display == "log2(50257)"
    materialized = circuit.materialize(max_gates=1, max_edges=0)
    result = minimum_vertex_cut(materialized, {"token"})

    assert materialized.gates["token"].capacity.cardinality == 50_257
    assert result.exact_capacity is not None
    assert result.exact_capacity.multiplier == 50_257
    assert result.width_bits == pytest.approx(15.617036934288085)
    assert result.cut == frozenset({"token"})


def test_bidirectional_validation_detects_an_inconsistent_inverse() -> None:
    family = GateFamily(
        "gate",
        RectangularDomain((2,)),
        ("index",),
        GateCapacity.bits(4),
        "gate",
    )
    broken = EdgeRule(
        "broken",
        "gate",
        "gate",
        lambda target: ((0,),) if target == (1,) else (),
        lambda _source: (),
    )
    circuit = IndexedCircuit((family,), (broken,), (GateRef("gate", (1,)),))

    with pytest.raises(ValueError, match="disagree"):
        circuit.validate_bidirectional(max_gates=2, max_edges=2)


def test_invalid_rule_output_names_the_rule_and_gate() -> None:
    family = GateFamily(
        "gate",
        RectangularDomain((1,)),
        ("index",),
        GateCapacity.bits(4),
        "gate",
    )

    def invalid_source(_target: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
        return ((99,),)

    broken = EdgeRule(
        "invalid-source",
        "gate",
        "gate",
        invalid_source,
        lambda _source: (),
    )
    circuit = IndexedCircuit((family,), (broken,), (GateRef("gate", (0,)),))

    with pytest.raises(ValueError, match="invalid-source"):
        circuit.predecessors(GateRef("gate", (0,)))
