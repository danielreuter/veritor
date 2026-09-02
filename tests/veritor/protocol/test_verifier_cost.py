"""The verifier's fixed work must not scale with the number of gates."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from veritor.compile import Compiler
from veritor.constructors import (
    MatmulG,
    MatmulWorkload,
    Tracer,
    expected_matmul_outputs,
)
from veritor.core import (
    Compiled,
    DescriptionCircuit,
    VerificationPolicy,
    make_word_gate_set,
)
from veritor.protocol import (
    ProverSession,
    VerifierSession,
    commit_weights,
    make_expectation,
)

POLICY = VerificationPolicy(1, 1, 0)
SEEDS = {"q_seed": b"Q" * 32, "s_seed": b"S" * 32, "session_id": b"cost"}
GATE_SET = make_word_gate_set(8)


def matmul_workload(n: int, rows: int = 4) -> MatmulWorkload:
    weights = tuple(tuple((i * j + 1) % 256 for j in range(n)) for i in range(n))
    activation = tuple(tuple((r + 3 * c) % 256 for c in range(n)) for r in range(rows))
    return MatmulWorkload(weights, (activation,))


def compile_matmul(workload: MatmulWorkload) -> Compiled:
    return Compiler(GATE_SET).compile(MatmulG(8)(workload, b""), workload.public_inputs)


def chain_compiled(blocks: int, width: int = 16) -> Compiled:
    """One input, one output, ``blocks * width`` gates in one replay unit.

    The blocks are one ``repeat`` step, so the description (what the verifier
    compiles and prices) has the same size for every ``blocks``.
    """

    tracer = Tracer(GATE_SET)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key="block", role="verification")
    def block(v):
        accumulator = v[0]
        for _ in range(width):
            accumulator = add(accumulator, v[0])
        return accumulator

    @tracer.definition(input_count=1, key=("root", blocks), role="replay")
    def root(v):
        return tracer.repeat(blocks, block, v[0])[-1]

    return Compiler(GATE_SET).compile(tracer.serialize(root), (3,))


def fastest(action: Callable[[], object], repetitions: int = 20) -> float:
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("weights_committed", [False, True])
@pytest.mark.parametrize("n", [16, 128])
def test_verifier_setup_and_boundary_phase_never_touch_interior_gates(
    monkeypatch, n, weights_committed
) -> None:
    workload = matmul_workload(n)
    compiled = compile_matmul(workload)
    circuit = compiled.circuit
    outputs = expected_matmul_outputs(workload)
    boundary_values: dict[int, object] = dict(enumerate(workload.public_inputs))
    boundary_values.update(zip(circuit.outputs, outputs, strict=True))
    weight_count = n * n
    if weights_committed:
        weights, tree = commit_weights(compiled, 0, weight_count, boundary_values)
        public_inputs = workload.public_inputs[weight_count:]
        io = set(range(weight_count, circuit.input_count)) | set(circuit.outputs)
    else:
        weights, tree = None, None
        public_inputs = workload.public_inputs
        io = set(circuit.inputs) | set(circuit.outputs)
    expectation = make_expectation(
        compiled, POLICY, public_inputs, outputs, weights=weights, **SEEDS
    )
    header = VerifierSession(expectation, compiled).header
    boundary = ProverSession(compiled, header, boundary_values, weight_tree=tree).boundary()

    looked_up: list[int] = []
    original_getitem = DescriptionCircuit.__getitem__

    def counting_getitem(self, address):
        looked_up.append(address)
        return original_getitem(self, address)

    def refuse_evaluate(*_args, **_kwargs):
        raise AssertionError("the verifier must never evaluate the circuit")

    monkeypatch.setattr(DescriptionCircuit, "__getitem__", counting_getitem)
    monkeypatch.setattr(DescriptionCircuit, "evaluate", refuse_evaluate)

    verifier = VerifierSession(expectation, compiled)
    verifier.receive_boundary(boundary)

    assert set(looked_up) <= io
    assert len(looked_up) <= 2 * len(io)
    assert circuit.n - len(io) > 4 * len(io)
    if weights_committed:
        assert set(looked_up).isdisjoint(range(weight_count))
        assert len(io) < weight_count


def test_verifier_construction_time_is_flat_in_gate_count() -> None:
    small = chain_compiled(128)
    large = chain_compiled(8192)
    assert large.circuit.n - 1 == 64 * (small.circuit.n - 1)
    assert small.circuit.input_count == large.circuit.input_count == 1

    def construction(compiled: Compiled) -> Callable[[], object]:
        outputs = tuple(compiled.circuit.evaluate((3,))[o] for o in compiled.circuit.outputs)
        expectation = make_expectation(compiled, POLICY, (3,), outputs, **SEEDS)
        return lambda: VerifierSession(expectation, compiled)

    small_time = fastest(construction(small))
    large_time = fastest(construction(large))

    assert large_time < 20 * small_time
