from __future__ import annotations

import time

import pytest

from veritor.compile import CompileError, Compiler
from veritor.core import (
    CompilationLimits,
    Compiled,
    DescriptionCircuit,
    Index,
    make_word_gate_set,
)

GATES = make_word_gate_set(8)
IN, LOC = "input", "local"


def test_compile_returns_the_circuit_the_index_and_a_bound_digest(helpers):
    payload = helpers.matmul_payload(4, 3, 2)
    inputs = tuple(range(2 * 4 + 4 * 3))
    compiled = Compiler(GATES).compile(payload, inputs)

    assert isinstance(compiled, Compiled)
    assert isinstance(compiled.circuit, DescriptionCircuit)
    assert isinstance(compiled.index, Index)
    assert compiled.circuit.n == 20 + 2 * 3 * 7
    assert compiled.index.replay_units.count == 2
    assert compiled.digest == Compiler(GATES).compile(payload, inputs).digest
    assert compiled.digest != Compiler(make_word_gate_set(16)).compile(payload, inputs).digest
    assert compiled.digest != Compiler(GATES).compile(helpers.matmul_payload(4, 3, 1), inputs[:16]).digest


def test_compile_checks_inputs_advice_and_marks(helpers):
    h = helpers
    payload = h.matmul_payload(4, 2, 1)
    compiler = Compiler(GATES)

    with pytest.raises(CompileError, match="expects 12 inputs, got 2"):
        compiler.compile(payload, (1, 2))
    with pytest.raises(CompileError, match="advice exceeds the public bit bound"):
        compiler.compile(payload, (0,) * 12, b"hint", advice_bound_bits=16)
    assert compiler.compile(payload, (0,) * 12, b"hint", advice_bound_bits=32).digest
    with pytest.raises(CompileError, match="advice must be bytes"):
        compiler.compile(payload, (0,) * 12, "hint")  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="not inside a replay unit"):
        compiler.compile(h.single(h.body(1, [h.gate("add", h.rng(IN, 0, 2, 0))], [h.rng(LOC, 0)])), (1,))
    with pytest.raises(CompileError, match="proof cost 11; the limit is 10"):
        Compiler(GATES, CompilationLimits(max_verification_unit_proof_cost=10)).compile(payload, (0,) * 12)
    with pytest.raises(TypeError):
        Compiler(object())  # type: ignore[arg-type]


def test_compile_and_index_are_sublinear_in_the_gate_count(helpers):
    """A constant-size description of ``d x d`` matmul compiles in the same time for any ``d``."""

    def timed(d: int) -> tuple[float, Compiled]:
        payload = helpers.matmul_payload(d, d, d)
        inputs = bytes(2 * d * d)  # zero inputs; a bytes object is a cheap Sequence[int]
        compiler = Compiler(GATES)
        started = time.perf_counter()
        compiled = compiler.compile(payload, inputs)
        index = compiled.index
        # exercise the lazy structures at the far end of the address space
        last = index.replay_units.count - 1
        unit = index.replay_units.unit(last)
        assert index.replay_units.owner(unit.interval.stop - 1) == last
        boundary = index.boundary()
        assert boundary.unrank(boundary.count - 1) == unit.interval.stop - 1
        assert boundary.rank(unit.interval.stop - 1) == boundary.count - 1
        vunit = index.verification_units(last).unit(d - 1)
        assert compiled.circuit[vunit.interval.stop - 1].op == "add"
        return time.perf_counter() - started, compiled

    small_time, small = timed(64)
    large_time, large = timed(1024)
    small_payload = helpers.matmul_payload(64, 64, 64)
    large_payload = helpers.matmul_payload(1024, 1024, 1024)

    assert small.circuit.n == 2 * 64**2 + 64**2 * (2 * 64 - 1)
    assert large.circuit.n == 2 * 1024**2 + 1024**2 * (2 * 1024 - 1)
    assert large.circuit.n > 2 * 10**9
    # the description grows only by its reduction depth (four more repeat steps)
    assert len(large_payload) - len(small_payload) < 800
    assert large_time < 0.5, large_time
    assert large_time < 20 * small_time + 0.1
