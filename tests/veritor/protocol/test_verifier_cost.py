"""The verifier's fixed work must not scale with the number of gates."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from veritor.compile import (
    CallDagCircuit,
    Kernel,
    MatmulWorkload,
    PartitionPolicy,
    Producer,
    compile_call_dag,
    compile_matmul_workload,
    expected_matmul_outputs,
    make_word_kernel,
)
from veritor.core import CompiledArtifact, VerificationPolicy
from veritor.protocol import ProverSession, VerifierSession, make_expectation

POLICY = VerificationPolicy(1, 1, 0)
SEEDS = {"q_seed": b"Q" * 32, "s_seed": b"S" * 32, "session_id": b"cost"}


def matmul_workload(n: int, rows: int = 4) -> MatmulWorkload:
    weights = tuple(tuple((i * j + 1) % 256 for j in range(n)) for i in range(n))
    activation = tuple(tuple((r + 3 * c) % 256 for c in range(n)) for r in range(rows))
    return MatmulWorkload(weights, (activation,))


def chain_artifact(blocks: int, width: int = 16) -> CompiledArtifact:
    """One input, one output, ``blocks * width`` gates in one replay unit."""

    producer = Producer(8)

    @producer.gate(name="add")
    def add(left: int, right: int) -> int:
        return left + right

    @producer.circuit(key="block", input_count=1)
    def block(value):
        accumulator = value
        for _ in range(width):
            accumulator = add(accumulator, value)
        return accumulator

    @producer.circuit(key=("root", blocks), input_count=1)
    def root(value):
        accumulator = value
        for _ in range(blocks):
            accumulator = block(accumulator)
        return accumulator

    return compile_call_dag(
        make_word_kernel(8),
        lambda _x, _a: producer.serialize(root),
        None,
        b"",
        input_cells=(3,),
        advice_bound_bits=0,
        replay_policy=PartitionPolicy.WHOLE_ROOT,
        verification_policy=PartitionPolicy.POSITIVE_TOP_LEVEL_OCCURRENCES,
    )


def fastest(action: Callable[[], object], repetitions: int = 20) -> float:
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("n", [16, 128])
def test_verifier_setup_and_boundary_phase_never_touch_interior_gates(monkeypatch, n) -> None:
    workload = matmul_workload(n)
    artifact = compile_matmul_workload(workload)
    outputs = expected_matmul_outputs(workload)
    expectation = make_expectation(artifact, POLICY, workload.public_inputs, outputs, **SEEDS)
    boundary_values: dict[int, object] = dict(enumerate(workload.public_inputs))
    boundary_values.update(
        (int(port.position), value)
        for port, value in zip(artifact.circuit.output_ports, outputs, strict=True)
    )
    header = VerifierSession(expectation, artifact).header
    boundary = ProverSession(artifact, header, boundary_values).boundary()

    lookups = {"gates": 0}
    original_gate_at = CallDagCircuit.gate_at
    original_executable_gate_at = CallDagCircuit.executable_gate_at

    def counting_gate_at(self, position):
        lookups["gates"] += 1
        return original_gate_at(self, position)

    def counting_executable_gate_at(self, position):
        lookups["gates"] += 1
        return original_executable_gate_at(self, position)

    def refuse_flatten(*_args, **_kwargs):
        raise AssertionError("the verifier must never flatten the circuit")

    monkeypatch.setattr(CallDagCircuit, "gate_at", counting_gate_at)
    monkeypatch.setattr(CallDagCircuit, "executable_gate_at", counting_executable_gate_at)
    monkeypatch.setattr(Kernel, "flatten", refuse_flatten)

    verifier = VerifierSession(expectation, artifact)
    assert lookups["gates"] == 0

    verifier.receive_boundary(boundary)
    assert lookups["gates"] <= len(artifact.circuit.output_ports)
    assert artifact.circuit.gate_count > 8 * len(artifact.circuit.output_ports)


def test_verifier_construction_time_is_flat_in_gate_count() -> None:
    small = chain_artifact(128)
    large = chain_artifact(8192)
    assert large.circuit.gate_count == 64 * small.circuit.gate_count
    assert small.circuit.input_count == large.circuit.input_count == 1

    def construction(artifact: CompiledArtifact) -> Callable[[], object]:
        expectation = make_expectation(
            artifact, POLICY, (3,), artifact.circuit.evaluate((3,)), **SEEDS
        )
        return lambda: VerifierSession(expectation, artifact)

    small_time = fastest(construction(small))
    large_time = fastest(construction(large))

    assert large_time < 20 * small_time
