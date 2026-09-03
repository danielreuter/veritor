"""The verifier enforces check outputs and charges the advice at its declared, canonical bit length."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from veritor.compile import Compilation, CompileError, declared_advice_bits
from veritor.constructors import Tracer
from veritor.core import Compiled, VerificationPolicy, make_word_gate_set
from veritor.protocol import (
    Expectation,
    Header,
    ProtocolError,
    VerificationCode,
    VerifierParameters,
    VerifierSession,
    decode_transcript,
    encode_transcript,
    make_expectation,
    run_protocol,
    verify_transcript,
)
from veritor.protocol.session import assignment_replay
from veritor.research import Compile

GATES = make_word_gate_set(8)
CHECK_EVERYTHING = VerificationPolicy(1, 1)
SEEDS = {"session_id": b"checks", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}
NO_CAPACITY = VerifierParameters(max_capacity=None)


class CheckedG:
    """``(x0 + x0, x0 * x0)`` in one replay unit, the sum marked as a check equal to ``10``."""

    digest = "cd" * 32

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        tracer = Tracer(GATES)
        add, mul = tracer.gate("add"), tracer.gate("mul")
        pair = tracer.definition(input_count=1, key="pair", role="verification")(
            lambda v: [add(v[0], v[0]), mul(v[0], v[0])]
        )

        @tracer.definition(input_count=0, key="unit", role="replay")
        def unit(_v):
            return pair(tracer.inputs(1))

        @tracer.definition(input_count=0, key="root")
        def root(_v):
            outputs = unit()
            tracer.check(outputs[0], 10)
            return outputs

        return tracer.serialize(root), (5,)


@pytest.fixture(scope="module")
def compilation() -> Compilation:
    return Compile(CheckedG(), None, b"", GATES)


@pytest.fixture(scope="module")
def compiled(compilation: Compilation) -> Compiled:
    return compilation.compiled


def expect(
    compilation: Compilation, claimed: tuple[int, ...], **changes
) -> Expectation:
    return make_expectation(
        compilation,
        CHECK_EVERYTHING,
        claimed,
        parameters=NO_CAPACITY,
        **SEEDS,
        **changes,
    )


def test_the_header_carries_the_check_constants_and_an_honest_run_is_accepted(
    compilation, compiled
) -> None:
    assert list(compiled.check_values()) == [(0, 10)]
    values = dict(enumerate(compiled.circuit.evaluate((5,))))
    run = run_protocol(compiled, expect(compilation, (10, 25)), values)
    assert run.report.accepted and run.transcript is not None
    header = run.transcript.header
    assert header.claimed_outputs == (bytes([10]), bytes([25]))
    data = encode_transcript(run.transcript)
    assert decode_transcript(data) == run.transcript
    assert verify_transcript(data, expect(compilation, (10, 25)), compiled).accepted


def test_claiming_another_value_at_a_check_position_is_rejected_at_admission(
    compilation, compiled
) -> None:
    with pytest.raises(Exception, match="check_mismatch") as info:
        VerifierSession(expect(compilation, (11, 25)), compiled)
    assert info.value.code is VerificationCode.CHECK_MISMATCH  # type: ignore[attr-defined]
    values = dict(enumerate(compiled.circuit.evaluate((5,))))
    run = run_protocol(compiled, expect(compilation, (11, 25)), values)
    assert run.report.code is VerificationCode.CHECK_MISMATCH and run.transcript is None
    # an ordinary output may be claimed wrongly: that is the public-I/O check's business
    run = run_protocol(compiled, expect(compilation, (10, 26)), values)
    assert run.report.code is VerificationCode.PUBLIC_IO_MISMATCH


def test_a_prover_whose_check_output_moved_is_rejected_at_the_boundary(
    compilation, compiled
) -> None:
    values = dict(enumerate(compiled.circuit.evaluate((5,))))
    check_address = compiled.circuit.outputs[0]
    values[check_address] = 11
    run = run_protocol(
        compiled,
        expect(compilation, (10, 25)),
        values,
        replay=assignment_replay(values),
    )
    assert run.report.code is VerificationCode.CHECK_MISMATCH
    assert f"check output 0 at address {check_address} is not 10" in run.report.detail
    # the transcript of an honest run, with the opening at the check position altered
    honest = run_protocol(
        compiled,
        expect(compilation, (10, 25)),
        dict(enumerate(compiled.circuit.evaluate((5,)))),
    )
    assert honest.transcript is not None
    document = json.loads(encode_transcript(honest.transcript))
    for opening in document["boundary"]["io_openings"]:
        if opening["position"] == check_address:
            opening["value"] = "0b"
    altered = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    report = verify_transcript(altered, expect(compilation, (10, 25)), compiled)
    assert report.code in (
        VerificationCode.CHECK_MISMATCH,
        VerificationCode.INVALID_OPENING,
    )


# -- exact advice bits -----------------------------------------------------------------------


class AdvisedG:
    """A constructor whose advice is ``bits`` bits, canonically zero-padded to bytes."""

    digest = "ad" * 32

    def __init__(self, bits: int) -> None:
        self.bits = bits

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        tracer = Tracer(GATES)
        add = tracer.gate("add")
        double = tracer.definition(input_count=1, key="double", role="verification")(
            lambda v: add(v[0], v[0])
        )

        @tracer.definition(input_count=0, key="unit", role="replay")
        def unit(_v):
            return double(tracer.inputs(1))

        root = tracer.definition(input_count=0, key="root")(lambda _v: unit())
        return tracer.serialize(root), (3,)

    def advice_bits(self, x: object, a: bytes) -> int:
        return self.bits


def test_the_compiler_charges_the_declared_bits_of_a_canonical_advice() -> None:
    compilation = Compile(
        AdvisedG(20), None, b"\xab\xcd\xe0", GATES, max_advice_bits=20
    )
    assert compilation.advice_bits == 20 and compilation.advice == b"\xab\xcd\xe0"
    assert declared_advice_bits(AdvisedG(20), None, b"\xab\xcd\xe0") == 20
    assert (
        declared_advice_bits(CheckedG(), None, b"\xab\xcd\xef") == 24
    )  # no declaration: every byte counts
    with pytest.raises(CompileError, match="padding bits are not zero"):
        Compile(AdvisedG(20), None, b"\xab\xcd\xef", GATES, max_advice_bits=24)
    with pytest.raises(CompileError, match="declares 20 bits, which take 3"):
        Compile(AdvisedG(20), None, b"\xab\xcd", GATES, max_advice_bits=24)
    with pytest.raises(CompileError, match="declares 20 bits, which take 3"):
        Compile(AdvisedG(20), None, b"\xab\xcd\xe0\x00", GATES, max_advice_bits=32)
    with pytest.raises(CompileError, match="advice exceeds the public bit bound"):
        Compile(AdvisedG(20), None, b"\xab\xcd\xe0", GATES, max_advice_bits=19)
    with pytest.raises(CompileError, match="nonnegative integer"):
        Compile(AdvisedG(-1), None, b"", GATES)
    with pytest.raises(CompileError, match="nonnegative integer"):
        Compile(AdvisedG("x"), None, b"", GATES)  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="nonnegative integer"):
        Compile(AdvisedG(True), None, b"", GATES)  # type: ignore[arg-type]

    class Failing(AdvisedG):
        def advice_bits(self, x: object, a: bytes) -> int:
            raise RuntimeError("no")

    with pytest.raises(CompileError, match="failed to declare"):
        Compile(Failing(0), None, b"", GATES)
    assert Compile(AdvisedG(0), None, b"", GATES).advice_bits == 0
    assert (
        Compile(AdvisedG(8), None, b"\xff", GATES, max_advice_bits=8).advice_bits == 8
    )
    assert (
        Compile(AdvisedG(1), None, b"\x80", GATES, max_advice_bits=1).advice_bits == 1
    )


def test_compilation_expectation_and_header_reject_a_noncanonical_advice(
    compiled,
) -> None:
    compilation = Compilation(compiled, "ad" * 32, (3,), b"\xab\xc0", 12)
    assert compilation.advice_bits == 12
    with pytest.raises(CompileError, match="padding bits are not zero"):
        Compilation(compiled, "ad" * 32, (3,), b"\xab\xcd", 12)
    with pytest.raises(CompileError, match="declares 12 bits, which take 2"):
        Compilation(compiled, "ad" * 32, (3,), b"\xab\xc0\x00", 12)
    with pytest.raises(CompileError, match="declares 0 bits"):
        Compilation(
            compiled, "ad" * 32, (3,), b"\xab"
        )  # the default 0 fits empty advice only
    advised = VerifierParameters(max_advice_bits=12, max_capacity=None)
    expectation = make_expectation(
        compilation, CHECK_EVERYTHING, (10, 25), parameters=advised, **SEEDS
    )
    assert expectation.advice_bits == 12
    with pytest.raises(ProtocolError, match="padding bits are not zero"):
        replace(expectation, advice=b"\xab\xcd")
    with pytest.raises(ProtocolError, match="declares 12 bits, which take 2"):
        replace(expectation, advice=b"\xab")
    header = VerifierSession(expectation, compiled).header
    assert header.advice_bits == 12 and header.advice == b"\xab\xc0"
    with pytest.raises(ProtocolError, match="padding bits are not zero"):
        replace(header, advice=b"\xab\xcd")
    with pytest.raises(ProtocolError, match="nonnegative integer"):
        replace(header, advice_bits=-1)
    assert (
        Header(
            *[
                getattr(header, f)
                for f in (
                    "session_id",
                    "compiled_digest",
                    "constructor",
                    "advice",
                    "policy",
                    "eta",
                    "public_inputs",
                    "claimed_outputs",
                    "weights",
                    "backend",
                    "max_faults",
                    "advice_bits",
                )
            ]
        )
        == header
    )
    assert replace(header, advice_bits=16).digest != header.digest


def test_the_verifier_admits_the_advice_against_its_declared_bits(compiled) -> None:
    compilation = Compilation(compiled, "ad" * 32, (3,), b"\xab\xc0", 12)
    tight = VerifierParameters(max_advice_bits=12, max_capacity=None)
    expectation = make_expectation(
        compilation, CHECK_EVERYTHING, (10, 25), parameters=tight, **SEEDS
    )
    assert (
        VerifierSession(expectation, compiled).header.advice_bits == 12
    )  # 16 bits of bytes, 12 charged
    looser = replace(
        expectation,
        parameters=VerifierParameters(max_advice_bits=11, max_capacity=None),
    )
    with pytest.raises(Exception) as info:
        VerifierSession(looser, compiled)
    assert info.value.code is VerificationCode.POLICY_REJECTED  # type: ignore[attr-defined]
    assert "12 bits, exceeding max_advice_bits 11" in str(info.value)
