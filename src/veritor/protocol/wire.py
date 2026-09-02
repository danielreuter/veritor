"""Strict canonical JSON encoding of a :class:`Transcript`.

Decoding accepts exactly the bytes that encoding produces: sorted keys, no
whitespace, lowercase hex, reduced fractions, no floats, no unknown keys.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction

from veritor.core import (
    JSONValue,
    VerificationLimits,
    VerificationPolicy,
    canonical_json_bytes,
)

from .messages import (
    PROTOCOL_VERSION,
    BoundaryMessage,
    Commitment,
    EvidenceMessage,
    Header,
    InteriorMessage,
    Opening,
    ProtocolError,
    ReplayChallenge,
    SampleChallenge,
    Transcript,
    Weights,
)


class MalformedTranscript(ProtocolError):
    """The bytes are not a well-formed transcript document."""


class NoncanonicalTranscript(ProtocolError):
    """The document is well formed but not in canonical byte form."""


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def encode_transcript(transcript: Transcript) -> bytes:
    header = transcript.header
    document: dict[str, JSONValue] = {
        "version": PROTOCOL_VERSION,
        "header": {
            "claimed_outputs": [item.hex() for item in header.claimed_outputs],
            "compiled_digest": header.compiled_digest,
            "policy": {
                "eta": _pair(header.policy.eta),
                "q": _pair(header.policy.q),
                "s": _pair(header.policy.s),
            },
            "public_inputs": [item.hex() for item in header.public_inputs],
            "session_id": header.session_id.hex(),
            "weights": None if header.weights is None else header.weights.manifest,
        },
        "boundary": transcript.boundary.manifest,
        "replay_challenge": transcript.replay_challenge.manifest,
        "interiors": transcript.interiors.manifest,
        "sample_challenge": transcript.sample_challenge.manifest,
        "evidence": transcript.evidence.manifest,
    }
    return canonical_json_bytes(document)


def _object(value: object, keys: set[str], where: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise MalformedTranscript(f"{where} must be an object with keys {sorted(keys)}")
    return value


def _list[T](value: object, item: Callable[[object, str], T], where: str) -> tuple[T, ...]:
    if type(value) is not list:
        raise MalformedTranscript(f"{where} must be a list")
    return tuple(item(element, f"{where}[{index}]") for index, element in enumerate(value))


def _int(value: object, where: str) -> int:
    if type(value) is not int or value < 0:
        raise MalformedTranscript(f"{where} must be a nonnegative integer")
    return value


def _hex(value: object, where: str) -> bytes:
    if type(value) is not str:
        raise MalformedTranscript(f"{where} must be a hex string")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise MalformedTranscript(f"{where} is not hexadecimal") from error


def _fraction(value: object, where: str, limits: VerificationLimits) -> Fraction:
    if type(value) is not list or len(value) != 2:
        raise MalformedTranscript(f"{where} must be [numerator, denominator]")
    numerator, denominator = (_int(item, where) for item in value)
    if denominator == 0:
        raise MalformedTranscript(f"{where} has a zero denominator")
    limits.enforce("max_probability_denominator_bits", denominator.bit_length())
    return Fraction(numerator, denominator)


def _commitment(value: object, where: str) -> Commitment:
    fields = _object(value, {"count", "root"}, where)
    return Commitment(_hex(fields["root"], f"{where}.root"), _int(fields["count"], f"{where}.count"))


def _opening(value: object, where: str) -> Opening:
    fields = _object(value, {"path", "position", "value"}, where)
    return Opening(
        _int(fields["position"], f"{where}.position"),
        _hex(fields["value"], f"{where}.value"),
        _list(fields["path"], _hex, f"{where}.path"),
    )


def _openings(value: object, where: str) -> tuple[Opening, ...]:
    return _list(value, _opening, where)


def _weights(value: object, where: str) -> Weights | None:
    if value is None:
        return None
    fields = _object(value, {"root", "start", "stop"}, where)
    return Weights(
        _int(fields["start"], f"{where}.start"),
        _int(fields["stop"], f"{where}.stop"),
        _hex(fields["root"], f"{where}.root"),
    )


def _challenge[T](
    factory: Callable[[bytes, tuple[int, ...]], T], value: object, where: str
) -> T:
    fields = _object(value, {"seed", "selected"}, where)
    return factory(
        _hex(fields["seed"], f"{where}.seed"),
        _list(fields["selected"], _int, f"{where}.selected"),
    )


def _duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedTranscript(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_float(text: str) -> object:
    raise MalformedTranscript(f"floating point value {text!r} is not allowed")


def decode_transcript(data: bytes, limits: VerificationLimits | None = None) -> Transcript:
    checked = VerificationLimits() if limits is None else limits
    if type(data) is not bytes:
        raise MalformedTranscript("transcript must be bytes")
    checked.enforce("max_transcript_bytes", len(data))
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_key_guard,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except MalformedTranscript:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise MalformedTranscript("transcript is not valid JSON") from error
    top = _object(
        document,
        {
            "boundary",
            "evidence",
            "header",
            "interiors",
            "replay_challenge",
            "sample_challenge",
            "version",
        },
        "transcript",
    )
    if top["version"] != PROTOCOL_VERSION:
        raise MalformedTranscript("unsupported transcript version")
    header_fields = _object(
        top["header"],
        {
            "claimed_outputs",
            "compiled_digest",
            "policy",
            "public_inputs",
            "session_id",
            "weights",
        },
        "header",
    )
    policy_fields = _object(header_fields["policy"], {"eta", "q", "s"}, "header.policy")
    compiled_digest = header_fields["compiled_digest"]
    if type(compiled_digest) is not str:
        raise MalformedTranscript("header.compiled_digest must be a string")
    try:
        policy = VerificationPolicy(
            _fraction(policy_fields["q"], "header.policy.q", checked),
            _fraction(policy_fields["s"], "header.policy.s", checked),
            _fraction(policy_fields["eta"], "header.policy.eta", checked),
        )
        header = Header(
            _hex(header_fields["session_id"], "header.session_id"),
            compiled_digest,  # type: ignore[arg-type]
            policy,
            _list(header_fields["public_inputs"], _hex, "header.public_inputs"),
            _list(header_fields["claimed_outputs"], _hex, "header.claimed_outputs"),
            _weights(header_fields["weights"], "header.weights"),
        )
        boundary_fields = _object(top["boundary"], {"commitment", "io_openings"}, "boundary")
        transcript = Transcript(
            header,
            BoundaryMessage(
                _commitment(boundary_fields["commitment"], "boundary.commitment"),
                _openings(boundary_fields["io_openings"], "boundary.io_openings"),
            ),
            _challenge(ReplayChallenge, top["replay_challenge"], "replay_challenge"),
            InteriorMessage(
                _list(
                    _object(top["interiors"], {"commitments"}, "interiors")["commitments"],
                    _commitment,
                    "interiors.commitments",
                )
            ),
            _challenge(SampleChallenge, top["sample_challenge"], "sample_challenge"),
            EvidenceMessage(
                _list(
                    _object(top["evidence"], {"units"}, "evidence")["units"],
                    _openings,
                    "evidence.units",
                )
            ),
        )
    except MalformedTranscript:
        raise
    except (ProtocolError, ValueError, TypeError) as error:
        raise MalformedTranscript(str(error)) from error
    if encode_transcript(transcript) != data:
        raise NoncanonicalTranscript("transcript bytes are not in canonical form")
    return transcript
