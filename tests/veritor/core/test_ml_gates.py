"""The structural ML gate set: declared widths and costs, no semantics, and nothing needs them.

:func:`make_ml_gate_set` is what GPT-2's structure is written over.  Its
gates carry an output width (activations at ``width``, accumulators and
statistics at ``acc_width``, comparisons one bit) and replay/proof costs;
every evaluator is a stub that raises.  The compile, index and analysis path
-- ``Compile``, ``Index.kinds``, ``Bound``, ``Cost`` -- never evaluates a
gate, so a mixed-width circuit over this set compiles, is indexed and is
priced while any attempt to run it fails loudly.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.analysis.bound import BoundOptions, bound
from veritor.analysis.cost import cost
from veritor.compile import Compiler
from veritor.constructors import Tracer
from veritor.core import (
    Compiled,
    DescriptionCircuit,
    VerificationPolicy,
    make_isa_gate_set,
    make_ml_gate_set,
)
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.ml_gates import (
    ML_GATE_SET_NAME,
    STRUCTURAL_MESSAGE,
    TRANSCENDENTAL_COST,
    structural_stub,
)

WIDTH, ACC = 16, 32
ACTIVATION_GATES = {"add": 2, "sub": 2, "mul": 2, "max": 2, "select": 3, "narrow": 1}
COMPARISON_GATES = {"lt": 2, "eq": 2}
ACCUMULATOR_GATES = {"acc_add": 2, "acc_sub": 2, "acc_mul": 2, "acc_max": 2}
TRANSCENDENTALS = {"exp": 1, "recip": 1, "rsqrt": 1, "tanh": 1}


def test_the_gates_have_the_declared_widths_arities_and_costs() -> None:
    gates = make_ml_gate_set()

    assert gates.name == ML_GATE_SET_NAME
    assert {gate.name for gate in gates} == (
        set(ACTIVATION_GATES) | set(COMPARISON_GATES) | set(ACCUMULATOR_GATES) | set(TRANSCENDENTALS) | {"in", "weight"}
    )
    for name, arity in ACTIVATION_GATES.items():
        assert (gates[name].arity, gates[name].width) == (arity, WIDTH)
    for name, arity in COMPARISON_GATES.items():
        assert (gates[name].arity, gates[name].width) == (arity, 1)
    for name, arity in ACCUMULATOR_GATES.items():
        assert (gates[name].arity, gates[name].width) == (arity, ACC)
    for name, arity in TRANSCENDENTALS.items():
        assert (gates[name].arity, gates[name].width) == (arity, ACC)
        assert gates[name].replay_cost == gates[name].proof_cost == TRANSCENDENTAL_COST
    assert gates["mul"].replay_cost == 2 * gates["add"].replay_cost
    assert gates["acc_mul"].replay_cost == 2 * gates["mul"].replay_cost
    assert gates.input_gates == ("in",) and gates.weight_gates == ("weight",)
    assert gates["in"].width == gates["weight"].width == WIDTH
    # the widths are parameters
    custom = make_ml_gate_set(width=8, acc_width=24)
    assert (custom["add"].width, custom["acc_add"].width, custom["lt"].width) == (8, 24, 1)
    assert custom.id == gates.id and custom.digest != gates.digest  # the manifest carries the widths


def test_the_set_is_named_by_its_widths_and_validated() -> None:
    assert make_ml_gate_set().manifest == make_ml_gate_set(16, 32).manifest
    assert make_ml_gate_set().digest == make_ml_gate_set(16, 32).digest
    with pytest.raises(ValueError, match="acc_width must be at least width"):
        make_ml_gate_set(width=32, acc_width=16)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive bit count"):
            make_ml_gate_set(width=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive bit count"):
            make_ml_gate_set(acc_width=bad)  # type: ignore[arg-type]


def test_every_gate_refuses_to_evaluate_or_check() -> None:
    """The semantics are structural stubs: ``Gate.evaluate`` and ``Gate.check`` raise."""

    gates = make_ml_gate_set()
    for gate in gates:
        if gate.source is not None:
            continue
        with pytest.raises(RuntimeError, match=f"gate {gate.name} evaluator raised") as raised:
            gate.evaluate([0] * gate.arity)
        assert isinstance(raised.value.__cause__, NotImplementedError)
        assert str(raised.value.__cause__) == f"{STRUCTURAL_MESSAGE} ({gate.name})"
        with pytest.raises(RuntimeError):
            gate.check([0] * gate.arity, 0)
    with pytest.raises(NotImplementedError, match=STRUCTURAL_MESSAGE):
        structural_stub("anything")((1, 2))


def dot_cell_circuit(k: int, copies: int) -> tuple[Compiled, Tracer]:
    """``copies`` biased dot products of ``k`` inputs against ``k`` weights, one replay unit each.

    A dot accumulates at ``acc_width`` (``acc_mul``, ``acc_add``), adds a bias
    and rounds with ``narrow``: a mixed-width verification unit of ``2k + 1``
    gates.  Its replay unit owns ``k`` input gates; the weights are one unit.
    """

    gates = make_ml_gate_set()
    tracer = Tracer(gates)
    acc_mul, acc_add, narrow = tracer.gate("acc_mul"), tracer.gate("acc_add"), tracer.gate("narrow")

    @tracer.definition(input_count=2 * k + 1, key=("dot", k), role=VERIFICATION)
    def dot(v):
        acc = acc_mul(v[0], v[k])
        for i in range(1, k):
            acc = acc_add(acc, acc_mul(v[i], v[k + i]))
        return narrow(acc_add(acc, v[2 * k]))

    @tracer.definition(input_count=0, key="weights", role=REPLAY)
    def weights(_v):
        return tracer.weights(k * copies + copies)

    @tracer.definition(input_count=k + 1, key="unit", role=REPLAY)
    def unit(w):
        x = tracer.inputs(k)
        return dot(x, w[:k], w[k])

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        w = weights()
        return [unit(w[c * k : (c + 1) * k], w[k * copies + c]) for c in range(copies)]

    description = tracer.serialize(root)
    return Compiler(gates).compile(description, [1] * (k * copies)), tracer


def test_a_mixed_width_circuit_compiles_indexes_and_is_priced_without_evaluating() -> None:
    """Nothing on the compile / index / analysis path touches a gate's semantics.

    Every stub raises, so completing without an exception *is* the proof
    that no gate was evaluated.  The kind table sees the mixed widths: the
    dot's interface is its ``width``-bit ``narrow``; the circuit has no
    common width.
    """

    k, copies = 4, 3
    compiled, tracer = dot_cell_circuit(k, copies)
    circuit, index = compiled.circuit, compiled.index

    assert isinstance(circuit, DescriptionCircuit) and circuit.width is None
    assert circuit.n == copies * (k + 2 * k + 1) + k * copies + copies
    assert circuit.input_count == k * copies and circuit.weight_count == k * copies + copies
    assert index.replay_units.count == copies + 1
    rows = {row.kind: row for row in index.kinds()}
    dot = rows[tracer._by_key[("dot", k)].digest]
    assert (dot.copies, dot.size, dot.out_bits, dot.role) == (copies, 2 * k + 1, WIDTH, VERIFICATION)
    assert dot.replay_cost == k * 4 + k * 2 + 1  # k acc_mul at 4, k acc_add at 2, one narrow
    unit = rows[tracer._by_key["unit"].digest]
    assert unit.closed and unit.out_bits == WIDTH and unit.reach_bits == WIDTH
    root = rows[compiled.index.root.kind]
    assert root.out_bits == copies * WIDTH
    # per-gate widths are what the runs of ``Out`` and the domains read
    widths = sorted({circuit[address].width for address in range(circuit.n)})
    assert widths == [WIDTH, ACC]
    assert circuit.Cost(index.replay_units.unit(1)) == dot.replay_cost
    # the analysis folds read the table alone
    policy = VerificationPolicy(Fraction(1, 2), 1)
    result = bound(compiled, policy, Fraction(1, 2**20), BoundOptions(max_buckets=64))
    assert 0 <= result.bits <= root.out_bits
    expected = cost(compiled, policy)
    sources = circuit.input_count + circuit.weight_count  # one-gate source cells, proof cost 1 each
    assert expected.proof == Fraction(1, 2) * (copies * dot.proof_cost + sources)
    assert expected.total > 0
    assert compiled.digest == dot_cell_circuit(k, copies)[0].digest


def test_running_a_circuit_over_the_structural_set_fails_loudly() -> None:
    compiled, _ = dot_cell_circuit(2, 1)
    circuit = compiled.circuit

    with pytest.raises(RuntimeError, match="evaluator raised") as raised:
        circuit.evaluate([1, 2], [3, 4, 5])
    assert isinstance(raised.value.__cause__, NotImplementedError)
    first_gate = next(a for a in range(circuit.n) if not circuit[a].is_source)
    with pytest.raises(RuntimeError, match="evaluator raised"):
        circuit.evaluate_gate(first_gate, [1, 2])
    with pytest.raises(RuntimeError, match="evaluator raised"):
        circuit.check_gate(first_gate, [1, 2], 3)


def test_the_toy_isa_set_is_untouched_and_keeps_one_width() -> None:
    isa = make_isa_gate_set(16)
    assert {gate.width for gate in isa} == {16}
    assert isa["add"].evaluate([1, 2]) == 3
    assert not {"acc_add", "narrow", "exp", "rsqrt"} & {gate.name for gate in isa}
