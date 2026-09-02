from __future__ import annotations

from dataclasses import replace

import pytest

import veritor
from veritor import (
    Compile,
    Compiled,
    DemoGCompileRequest,
    MatmulCompileRequest,
    VerificationPolicy,
    compile_demo_g,
    compile_matmul,
    make_verification_expectation,
    make_word_gate_set,
)
from veritor.compile import CompileError
from veritor.constructors import DemoG, MatmulG
from veritor.protocol import ProtocolError


def test_compile_is_the_trusted_half_of_the_constructors() -> None:
    """``Compile`` on ``G``'s bytes is exactly what the convenience wrappers do."""

    request = MatmulCompileRequest()
    description = MatmulG(request.width)(request.workload, b"")
    gate_set = make_word_gate_set(request.width)

    compiled = Compile(description, request.public_inputs, gate_set)

    assert isinstance(compiled, Compiled)
    assert compiled.digest == compile_matmul(request).digest
    assert Compile(description, request.public_inputs, gate_set).digest == compiled.digest


def test_compile_checks_the_input_count_and_the_advice_bound() -> None:
    request = DemoGCompileRequest()
    description = DemoG(request.width)(request.batch, b"")
    gate_set = make_word_gate_set(request.width)

    with pytest.raises(CompileError, match="inputs"):
        Compile(description, request.public_inputs[:-1], gate_set)
    with pytest.raises(CompileError, match="advice"):
        Compile(description, request.public_inputs, gate_set, advice=b"x", advice_bound_bits=4)
    assert isinstance(compile_demo_g(request), Compiled)


def test_paper_functions_reject_anything_but_a_compiled() -> None:
    with pytest.raises(TypeError, match="Compiled"):
        veritor.Bound(object(), VerificationPolicy(1, 1, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compiled"):
        veritor.Cost(object(), VerificationPolicy(1, 1, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compiled"):
        make_verification_expectation(object(), VerificationPolicy(1, 1, 0), (), ())  # type: ignore[arg-type]


def test_expectation_generates_mandatory_verifier_seeds() -> None:
    request = DemoGCompileRequest()
    compiled = compile_demo_g(request)
    policy = VerificationPolicy(1, 1, 0)
    inputs, outputs = request.public_inputs, request.expected_outputs
    first = make_verification_expectation(compiled, policy, inputs, outputs)
    second = make_verification_expectation(compiled, policy, inputs, outputs)

    assert len(first.q_seed) == len(first.s_seed) == 32
    assert (first.q_seed, first.s_seed) != (second.q_seed, second.s_seed)
    assert first.session_id != second.session_id
    assert first.public_inputs == inputs
    assert first.claimed_outputs == outputs
    assert first.compiled_digest == compiled.digest
    with pytest.raises(ProtocolError, match="expected q seed"):
        replace(first, q_seed=None)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="expected s seed"):
        replace(first, s_seed=b"short")


def test_matmul_request_compiles_through_the_top_level_package() -> None:
    request = MatmulCompileRequest(
        ((1,),),
        (((2,),),),
    )

    compiled = compile_matmul(request)

    assert isinstance(compiled, Compiled)
    assert request.expected_outputs == (2,)
    assert compiled.circuit.evaluate(request.public_inputs)[compiled.circuit.outputs[0]] == 2


def test_paper_level_api_is_exported() -> None:
    names = {"Bound", "Compile", "Cost", "Optimize", "Verify", "run_protocol"}

    assert names <= set(veritor.__all__)
    assert all(hasattr(veritor, name) for name in names)
    assert set(veritor.research.__all__) <= set(veritor.__all__)
    assert all(hasattr(veritor.research, name) for name in veritor.research.__all__)
    assert all(hasattr(veritor, name) for name in veritor.__all__)
