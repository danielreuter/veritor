from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

import veritor
from veritor import (
    Bound,
    Capacity,
    Compilation,
    Compile,
    Compiled,
    Constructor,
    DemoGCompileRequest,
    MatmulCompileRequest,
    VerificationPolicy,
    compile_demo_g,
    compile_matmul,
    make_verification_expectation,
    make_word_gate_set,
)
from veritor.compile import CompileError, constructor_digest
from veritor.constructors import DemoG, MatmulG, TracerError
from veritor.protocol import ProtocolError

GATE_SET = make_word_gate_set(8)


def same(first: Compilation, second: Compilation) -> bool:
    """The same record of ``Compile(G, x, a)``: ``(C, I)`` by digest, the rest by value."""

    return (first.compiled.digest, first.constructor, first.inputs, first.advice) == (
        second.compiled.digest,
        second.constructor,
        second.inputs,
        second.advice,
    )


def test_compile_runs_g_and_records_what_it_ran_on() -> None:
    """``Compile(G, x, a)`` is exactly what the convenience wrappers do."""

    request = MatmulCompileRequest()
    G = MatmulG(request.width)

    compilation = Compile(G, request.workload, b"", GATE_SET)

    assert isinstance(compilation, Compilation) and isinstance(G, Constructor)
    assert isinstance(compilation.compiled, Compiled)
    assert compilation.constructor == G.digest == MatmulG(8).digest
    assert compilation.inputs == request.public_inputs
    assert compilation.advice == b"" and compilation.advice_bits == 0
    assert same(compilation, compile_matmul(request))
    assert same(compilation, Compile(G, request.workload, b"", GATE_SET))
    with pytest.raises(TypeError, match="Constructor"):
        Compile(object(), request.workload, b"", GATE_SET)  # type: ignore[arg-type]


def test_compile_charges_the_advice_g_consumes() -> None:
    request = DemoGCompileRequest(advice=b"hint", max_advice_bits=32)

    compilation = Compile(DemoG(8), request.batch, b"hint", GATE_SET, max_advice_bits=32)

    assert compilation.advice == b"hint" and compilation.advice_bits == 32
    assert compilation.constructor == DemoG(8).digest != MatmulG(8).digest
    assert compilation.inputs == request.public_inputs
    assert same(compilation, compile_demo_g(request))
    # DemoG ignores its advice: the same circuit, only the charged advice differs
    without = Compile(DemoG(8), request.batch, b"", GATE_SET)
    assert without.compiled.digest == compilation.compiled.digest and without.advice_bits == 0


def test_compile_rejects_advice_over_the_bound() -> None:
    request = DemoGCompileRequest()

    with pytest.raises(CompileError, match="advice exceeds the public bit bound"):
        Compile(DemoG(8), request.batch, b"x", GATE_SET, max_advice_bits=7)
    with pytest.raises(CompileError, match="advice exceeds the public bit bound"):
        Compile(DemoG(8), request.batch, b"x", GATE_SET)
    with pytest.raises(CompileError, match="advice must be bytes"):
        Compile(DemoG(8), request.batch, "x", GATE_SET, max_advice_bits=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_advice_bits"):
        Compile(DemoG(8), request.batch, b"", GATE_SET, max_advice_bits=-1)
    assert Compile(DemoG(8), request.batch, b"x", GATE_SET, max_advice_bits=8).advice_bits == 8


class FailingG:
    digest = constructor_digest("FailingG", "1", {})

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        raise RuntimeError("boom")


class MalformedG(FailingG):
    def __init__(self, produced: object) -> None:
        self.produced = produced

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        return self.produced  # type: ignore[return-value]


def test_compile_wraps_a_failing_constructor_into_a_rejection() -> None:
    request = MatmulCompileRequest()

    # MatmulG refuses advice with a TracerError; the verifier sees a CompileError
    with pytest.raises(TracerError, match="advice"):
        MatmulG(8)(request.workload, b"x")
    with pytest.raises(CompileError, match="constructor failed") as failure:
        Compile(MatmulG(8), request.workload, b"x", GATE_SET, max_advice_bits=8)
    assert isinstance(failure.value.__cause__, TracerError)
    with pytest.raises(CompileError, match="constructor failed: boom"):
        Compile(FailingG(), request.workload, b"", GATE_SET)
    with pytest.raises(CompileError, match=r"returns \(description, inputs\)"):
        Compile(MalformedG(b"just bytes"), None, b"", GATE_SET)
    with pytest.raises(CompileError, match="description must be bytes"):
        Compile(MalformedG(("text", ())), None, b"", GATE_SET)
    with pytest.raises(CompileError, match="tuple of integers"):
        Compile(MalformedG((b"", [1])), None, b"", GATE_SET)
    with pytest.raises(CompileError, match="tuple of integers"):
        Compile(MalformedG((b"", (1.5,))), None, b"", GATE_SET)
    with pytest.raises(CompileError):
        Compile(MalformedG((b"not a description", ())), None, b"", GATE_SET)


def test_compile_checks_the_constructors_digest() -> None:
    class BadDigestG(FailingG):
        digest = "not-a-digest"

    with pytest.raises(CompileError, match="constructor digest"):
        Compile(BadDigestG(), None, b"", GATE_SET)


def test_paper_functions_reject_anything_but_what_compile_produced() -> None:
    with pytest.raises(TypeError, match="Compiled"):
        veritor.Bound(object(), VerificationPolicy(1, 1), 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compiled"):
        veritor.Cost(object(), VerificationPolicy(1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compilation"):
        Capacity(compile_demo_g().compiled, VerificationPolicy(1, 1), 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compilation"):
        make_verification_expectation(object(), VerificationPolicy(1, 1), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compilation"):
        make_verification_expectation(compile_demo_g().compiled, VerificationPolicy(1, 1), ())  # type: ignore[arg-type]


def test_expectation_generates_mandatory_verifier_seeds() -> None:
    request = DemoGCompileRequest(advice=b"a", max_advice_bits=8)
    compilation = compile_demo_g(request)
    policy = VerificationPolicy(1, 1)
    outputs = request.expected_outputs
    first = make_verification_expectation(compilation, policy, outputs)
    second = make_verification_expectation(compilation, policy, outputs)

    assert len(first.q_seed) == len(first.s_seed) == 32
    assert (first.q_seed, first.s_seed) != (second.q_seed, second.s_seed)
    assert first.session_id != second.session_id
    assert first.public_inputs == request.public_inputs == compilation.inputs
    assert first.claimed_outputs == outputs
    assert first.compiled_digest == compilation.compiled.digest
    assert first.constructor == DemoG(8).digest
    assert first.advice == b"a"
    with pytest.raises(ProtocolError, match="expected q seed"):
        replace(first, q_seed=None)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="expected s seed"):
        replace(first, s_seed=b"short")
    with pytest.raises(ProtocolError, match="advice must be bytes"):
        replace(first, advice="a")  # type: ignore[arg-type]


def test_capacity_adds_exactly_the_advice_bits() -> None:
    request = DemoGCompileRequest()
    without = compile_demo_g(request)
    with_advice = compile_demo_g(replace(request, advice=b"hint!", max_advice_bits=40))
    assert with_advice.compiled.digest == without.compiled.digest  # DemoG ignores its advice

    for theta, eta in [
        (VerificationPolicy(1, 1), 0),
        (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
        (VerificationPolicy(0, 1), Fraction(1, 2)),
    ]:
        bits = Bound(without.compiled, theta, eta).bits
        assert Capacity(without, theta, eta) == bits
        assert Capacity(with_advice, theta, eta) == bits + 40
    assert Capacity(without, VerificationPolicy(1, 1), 0) == 0.0


def test_matmul_request_compiles_through_the_top_level_package() -> None:
    request = MatmulCompileRequest(
        ((1,),),
        (((2,),),),
    )

    compilation = compile_matmul(request)

    assert isinstance(compilation, Compilation)
    assert request.expected_outputs == (2,)
    circuit = compilation.compiled.circuit
    tape = circuit.evaluate(compilation.inputs, request.weight_values)
    assert tape[circuit.outputs[0]] == 2


def test_paper_level_api_is_exported() -> None:
    names = {
        "Bound",
        "Capacity",
        "Compilation",
        "Compile",
        "Constructor",
        "Cost",
        "Optimize",
        "Verify",
        "run_protocol",
    }

    assert names <= set(veritor.__all__)
    assert all(hasattr(veritor, name) for name in names)
    assert set(veritor.research.__all__) <= set(veritor.__all__)
    assert all(hasattr(veritor.research, name) for name in veritor.research.__all__)
    assert all(hasattr(veritor, name) for name in veritor.__all__)
