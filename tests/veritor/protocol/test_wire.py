from __future__ import annotations

import json

import pytest

from veritor.protocol import (
    MalformedTranscript,
    NoncanonicalTranscript,
    VerificationCode,
    decode_transcript,
    encode_transcript,
    run_protocol,
    verify_transcript,
)


@pytest.fixture
def recorded(compiled, honest_values, expect):
    expectation = expect()
    run = run_protocol(compiled, expectation, honest_values)
    assert run.transcript is not None
    return encode_transcript(run.transcript), run.transcript, expectation


def canonical(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def test_encoding_is_canonical_json_and_round_trips(recorded) -> None:
    data, transcript, _ = recorded

    assert data == canonical(json.loads(data))
    assert decode_transcript(data) == transcript
    assert encode_transcript(decode_transcript(data)) == data


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
        pytest.param(lambda data: data.replace(b'"version"', b'"verzion"'), id="unknown-key"),
        pytest.param(
            lambda data: data.replace(b"veritor/protocol/v3", b"veritor/protocol/v2"),
            id="version",
        ),
        pytest.param(lambda data: data.replace(b'"count":', b'"count":1.0,"c":', 1), id="float"),
        pytest.param(
            lambda data: data.replace(b'"count":', b'"count":1,"count":', 1), id="duplicate-key"
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
