from __future__ import annotations

import json

import pytest

from veritor.constructors import DemoG, DemoGCompileRequest, compile_demo_g
from veritor.protocol import (
    PROTOCOL_VERSION,
    MalformedTranscript,
    NoncanonicalTranscript,
    VerificationCode,
    VerifierParameters,
    decode_transcript,
    encode_transcript,
    run_protocol,
    verify_transcript,
)
from veritor.research import build_executable_conformance_transcript

ADVICE = b"\xab\xcd\xef"


@pytest.fixture
def recorded(compiled, honest_values, expect):
    expectation = expect()
    run = run_protocol(compiled, expectation, honest_values)
    assert run.transcript is not None
    return encode_transcript(run.transcript), run.transcript, expectation


@pytest.fixture(scope="module")
def recorded_with_advice():
    """A DemoG run under three bytes of advice: bytes, compiled circuit, expectation."""

    compilation = compile_demo_g(DemoGCompileRequest(advice=ADVICE, max_advice_bits=24))
    run = build_executable_conformance_transcript(
        compilation,
        parameters=VerifierParameters(max_advice_bits=24, max_capacity=None),
        session_id=b"wire/advice",
        q_seed=b"Q" * 32,
        s_seed=b"S" * 32,
    )
    return run.transcript_bytes, compilation.compiled, run.expectation


def canonical(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def test_encoding_is_canonical_json_and_round_trips(recorded) -> None:
    data, transcript, _ = recorded

    assert data == canonical(json.loads(data))
    assert decode_transcript(data) == transcript
    assert encode_transcript(decode_transcript(data)) == data
    assert PROTOCOL_VERSION == "veritor/protocol/v8" == json.loads(data)["version"]


def test_the_header_carries_the_constructor_and_the_advice_as_hex(
    recorded_with_advice,
) -> None:
    data, compiled, expectation = recorded_with_advice
    header = json.loads(data)["header"]

    assert header["advice"] == "abcdef"
    assert header["constructor"] == DemoG(8).digest == expectation.constructor
    transcript = decode_transcript(data)
    assert transcript.header.advice == ADVICE == expectation.advice
    assert transcript.header.constructor == DemoG(8).digest
    assert encode_transcript(transcript) == data
    assert verify_transcript(data, expectation, compiled).accepted


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("advice", "ABCDEF", id="uppercase-advice"),
        pytest.param("constructor", "UPPER", id="uppercase-constructor"),
    ],
)
def test_noncanonical_header_fields_are_rejected(
    recorded_with_advice, field, value
) -> None:
    data, compiled, expectation = recorded_with_advice
    document = json.loads(data)
    original = document["header"][field]
    document["header"][field] = original.upper() if value == "UPPER" else value
    assert document["header"][field] != original
    altered = canonical(document)

    with pytest.raises((NoncanonicalTranscript, MalformedTranscript)):
        decode_transcript(altered)
    report = verify_transcript(altered, expectation, compiled)
    assert report.code in {
        VerificationCode.NONCANONICAL_TRANSCRIPT,
        VerificationCode.MALFORMED_TRANSCRIPT,
    }
    # uppercase advice is well-formed hex that does not re-encode to the same bytes
    if field == "advice":
        with pytest.raises(NoncanonicalTranscript):
            decode_transcript(altered)
        assert report.code is VerificationCode.NONCANONICAL_TRANSCRIPT


@pytest.mark.parametrize(
    "field, value, detail",
    [
        pytest.param(
            "advice", 5, "header.advice must be a hex string", id="advice-int"
        ),
        pytest.param(
            "advice", "abc", "header.advice is not hexadecimal", id="advice-odd"
        ),
        pytest.param(
            "advice", "xyz1", "header.advice is not hexadecimal", id="advice-nonhex"
        ),
        pytest.param(
            "constructor", 5, "header.constructor must be a string", id="ctor-int"
        ),
        pytest.param("constructor", "ab" * 31, "constructor digest", id="ctor-short"),
        pytest.param("constructor", "zz" * 32, "constructor digest", id="ctor-nonhex"),
    ],
)
def test_malformed_header_fields_are_rejected(
    recorded_with_advice, field, value, detail
) -> None:
    data, compiled, expectation = recorded_with_advice
    document = json.loads(data)
    document["header"][field] = value
    altered = canonical(document)

    with pytest.raises(MalformedTranscript, match=detail):
        decode_transcript(altered)
    report = verify_transcript(altered, expectation, compiled)
    assert report.code is VerificationCode.MALFORMED_TRANSCRIPT


def test_a_transcript_under_other_advice_is_a_mismatch_not_a_decode_error(
    recorded_with_advice,
) -> None:
    data, compiled, expectation = recorded_with_advice
    document = json.loads(data)
    document["header"]["advice"] = "01abcd"
    altered = canonical(document)

    assert decode_transcript(altered).header.advice == b"\x01\xab\xcd"
    report = verify_transcript(altered, expectation, compiled)
    assert report.code is VerificationCode.EXPECTATION_MISMATCH
    # The advice is charged at the header's ``advice_bits``, of which it must be the
    # canonical encoding: another length, or padding bits set, is malformed.
    document["header"]["advice"] = "abcd"
    with pytest.raises(MalformedTranscript, match="declares 24 bits"):
        decode_transcript(canonical(document))
    document["header"]["advice"] = "abcdef"
    document["header"]["advice_bits"] = 20
    with pytest.raises(MalformedTranscript, match="padding bits are not zero"):
        decode_transcript(canonical(document))
    document["header"]["advice"] = "abcde0"
    assert decode_transcript(canonical(document)).header.advice_bits == 20


def uppercase_root(document: dict) -> dict:
    root = document["boundary"]["commitment"]["root"]
    if root == root.upper():
        pytest.skip("digest happens to contain no hex letters")
    document["boundary"]["commitment"]["root"] = root.upper()
    return document


@pytest.mark.parametrize(
    "rewrite",
    [
        pytest.param(lambda doc: json.dumps(doc, indent=1).encode(), id="whitespace"),
        pytest.param(
            lambda doc: json.dumps(
                dict(reversed(list(doc.items()))), separators=(",", ":")
            ).encode(),
            id="key-order",
        ),
        pytest.param(lambda doc: canonical(uppercase_root(doc)), id="uppercase-hex"),
    ],
)
def test_noncanonical_bytes_are_rejected(compiled, recorded, rewrite) -> None:
    data, _, expectation = recorded
    altered = rewrite(json.loads(data))
    assert altered != data

    with pytest.raises(NoncanonicalTranscript):
        decode_transcript(altered)
    report = verify_transcript(altered, expectation, compiled)
    assert report.code is VerificationCode.NONCANONICAL_TRANSCRIPT


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda data: data[:-1], id="truncated"),
        pytest.param(lambda data: b"[]", id="not-an-object"),
        pytest.param(
            lambda data: data.replace(b'"version"', b'"verzion"'), id="unknown-key"
        ),
        pytest.param(
            lambda data: data.replace(b"veritor/protocol/v8", b"veritor/protocol/v7"),
            id="version",
        ),
        pytest.param(
            lambda data: data.replace(b'"count":', b'"count":1.0,"c":', 1), id="float"
        ),
        pytest.param(
            lambda data: data.replace(b'"count":', b'"count":1,"count":', 1),
            id="duplicate-key",
        ),
    ],
)
def test_malformed_bytes_are_rejected(compiled, recorded, corrupt) -> None:
    data, _, expectation = recorded
    altered = corrupt(data)
    assert altered != data

    with pytest.raises(MalformedTranscript):
        decode_transcript(altered)
    report = verify_transcript(altered, expectation, compiled)
    assert report.code is VerificationCode.MALFORMED_TRANSCRIPT


def test_decode_rejects_non_bytes() -> None:
    with pytest.raises(MalformedTranscript):
        decode_transcript("{}")  # type: ignore[arg-type]
