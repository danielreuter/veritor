"""Component 8: tiling and refinement (core/index.py, session._Layout).

Every gate is in exactly one replay unit and one verification unit; the
verification units refine the replay units; cross-unit reads go only through
declared outputs, so ``Out(R_r)`` is the cut.  The compiler's mark rules are
also covered by ``tests/veritor/compile/test_index.py`` (``test_gate_step_
above_the_replay_cut_is_named``, ``test_gate_step_inside_a_replay_unit_must_
be_in_a_verification_unit``, ``test_marks_may_not_nest_and_verification_
needs_a_replay_unit``); here they are exercised as a client's attacks.
"""

from __future__ import annotations

import pytest

from veritor.compile import CompileError, Compiler
from veritor.constructors import Tracer
from veritor.core import Compiled, FlatCircuit, GateRef, iter_domain, make_word_gate_set
from veritor.protocol import Reject, VerificationCode, VerifierSession
from veritor.protocol.session import _Layout


def circuits(sec):
    yield sec.Model(2, 2).compiled
    yield sec.Model(2, 2, wide_units=True).compiled
    for seed in range(4):
        for marking in range(3):
            yield sec.random_marked_compiled(seed, marking)


def test_every_gate_is_in_exactly_one_replay_unit_and_one_verification_unit(sec):
    for compiled in circuits(sec):
        index, n = compiled.index, compiled.circuit.n
        replay_intervals = [
            index.replay_units.unit(r).interval for r in range(index.replay_units.count)
        ]
        assert sorted(a for interval in replay_intervals for a in interval) == list(
            range(n)
        )
        verification_intervals = [
            index.verification_unit(u).interval
            for u in range(index.verification_unit_count)
        ]
        assert sorted(
            a for interval in verification_intervals for a in interval
        ) == list(range(n))
        for address in range(n):
            r = index.replay_units.owner(address)
            assert address in replay_intervals[r]
            block = index.verification_units(r)
            u = block.first + block.owner(address)
            assert address in verification_intervals[u]


def test_verification_units_refine_replay_units(sec):
    for compiled in circuits(sec):
        index = compiled.index
        for r in range(index.replay_units.count):
            outer = index.replay_units.unit(r).interval
            block = index.verification_units(r)
            inner = [
                index.verification_unit(block.first + k).interval
                for k in range(block.count)
            ]
            assert all(
                index.verification_unit(block.first + k).replay_unit == r
                for k in range(block.count)
            )
            assert all(set(i) <= set(outer) for i in inner)
            assert sorted(a for i in inner for a in i) == list(outer)


def test_cross_unit_reads_go_only_through_declared_outputs(sec):
    """A gate reading outside its replay unit reads a boundary address or a weight."""

    for compiled in circuits(sec):
        circuit, index = compiled.circuit, compiled.index
        boundary = set(iter_domain(index.boundary()))
        weights = set(circuit.weights)
        declared = set(circuit.inputs)
        for r in range(index.replay_units.count):
            node = index.replay_units.unit(r)
            declared.update(circuit.Out(node))
            # Out(R_r) as the circuit computes it is inside the index's boundary
            assert set(circuit.Out(node)) <= boundary
        for address in range(circuit.n):
            r = index.replay_units.owner(address)
            for argument in circuit[address].args:
                if index.replay_units.owner(argument) != r:
                    assert argument in boundary or argument in weights, (
                        address,
                        argument,
                    )
        assert (
            boundary == declared
        )  # and the boundary is nothing but In and the Out(R_r)
        layout = _Layout(compiled)
        for unit in range(index.verification_unit_count):
            layout.required(unit)  # never rejects a compiler-produced (C, I)


def test_marks_leaving_a_gate_uncovered_are_a_compile_error():
    gate_set = make_word_gate_set(8)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")
    cell = tracer.definition(input_count=1, key="cell", role="verification")(
        lambda v: add(v[0], v[0])
    )

    @tracer.definition(input_count=0, key="leaky-replay", role="replay")
    def leaky(_v):
        x = tracer.inputs(1)
        return add(cell(x[0]), x[0])  # a gate directly inside the replay unit

    with pytest.raises(CompileError, match="verification"):
        Compiler(gate_set).compile(tracer.serialize(leaky), [1])

    @tracer.definition(input_count=0, key="sources", role="replay")
    def sources(_v):
        return tracer.inputs(2)

    @tracer.definition(input_count=0, key="gate-above-cut")
    def root(_v):
        x = sources()
        return add(x[0], x[1])  # a gate above every replay unit

    with pytest.raises(CompileError, match="replay"):
        Compiler(gate_set).compile(tracer.serialize(root), [1, 1])


def test_nested_or_straddling_marks_are_a_compile_error():
    gate_set = make_word_gate_set(8)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")
    cell = tracer.definition(input_count=1, key="cell", role="verification")(
        lambda v: add(v[0], v[0])
    )

    @tracer.definition(input_count=0, key="replay-in-replay", role="replay")
    def inner(_v):
        return cell(tracer.inputs(1)[0])

    @tracer.definition(input_count=0, key="outer-replay", role="replay")
    def outer(_v):
        return cell(inner())

    with pytest.raises(CompileError, match="contains a replay mark"):
        Compiler(gate_set).compile(tracer.serialize(outer), [1])

    @tracer.definition(
        input_count=1, key="verification-in-verification", role="verification"
    )
    def nested(v):
        return add(cell(v[0]), v[0])

    @tracer.definition(input_count=0, key="holder", role="replay")
    def holder(_v):
        return nested(tracer.inputs(1)[0])

    with pytest.raises(CompileError, match="contains a verification mark"):
        Compiler(gate_set).compile(tracer.serialize(holder), [1])

    # a verification unit "straddling" replay units: it would have to contain a replay mark
    @tracer.definition(input_count=1, key="straddle", role="verification")
    def straddle(v):
        return add(inner(), v[0])

    @tracer.definition(input_count=0, key="straddle-root", role="replay")
    def root(_v):
        return straddle(tracer.inputs(1)[0])

    with pytest.raises(
        CompileError, match="marked verification and contains a replay mark"
    ):
        Compiler(gate_set).compile(tracer.serialize(root), [1, 1])


def test_layout_rejects_a_circuit_that_reads_across_the_cut(model, sec):
    """Defense in depth: a forged ``(C, I)`` whose ``C`` reads another unit's interior.

    The compiler cannot produce one (a call reads only its caller's wires, which
    are declared outputs), so ``C`` is rebuilt as an explicit gate list with one
    argument rewired from a hidden boundary address to the interior ``mul`` of
    the same cell; ``I`` and the digest are the honest ones.
    """

    circuit = model.circuit
    stage0_mul, stage0_add = model.cell_addresses(0, 0)
    stage1_mul, _ = model.cell_addresses(1, 0)
    assert (
        stage0_add in circuit[stage1_mul].args
        and stage0_add in model.hidden_boundary_addresses
    )
    gates = []
    for address in range(circuit.n):
        ref = circuit[address]
        args = (
            tuple(stage0_mul if a == stage0_add else a for a in ref.args)
            if address == stage1_mul
            else ref.args
        )
        gates.append(GateRef(ref.op, args, ref.width, ref.source))
    forged = Compiled(
        FlatCircuit(gates, circuit.outputs, circuit.gate_set),
        model.index,
        model.compiled.digest,
    )
    layout = _Layout(forged)
    unit = model.unit_of(stage1_mul)
    with pytest.raises(Reject) as rejection:
        layout.required(unit)
    assert rejection.value.code == VerificationCode.INVALID_COMPILED_RESULT
    # a verifier holding the forged pair rejects the honest evidence with that code
    expectation = model.expectation()
    honest = model.run(expectation, model.values).transcript
    verifier = VerifierSession(expectation, forged)
    verifier.receive_boundary(honest.boundary)
    verifier.receive_interiors(honest.interiors)
    with pytest.raises(Reject) as verdict:
        verifier.receive_evidence(honest.evidence)
    assert verdict.value.code == VerificationCode.INVALID_COMPILED_RESULT
