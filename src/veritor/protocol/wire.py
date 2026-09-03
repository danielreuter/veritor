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
    TRANSPARENT_BACKEND,
    BoundaryMessage,
    Commitment,
    EvidenceMessage,
    Header,
    InteriorMessage,
    Opening,
    ProofMessage,
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
    header_document: dict[str, JSONValue] = {
        "advice": header.advice.hex(),
        "advice_bits": header.advice_bits,
        "claimed_outputs": [item.hex() for item in header.claimed_outputs],
        "compiled_digest": header.compiled_digest,
        "constructor": header.constructor,
        "eta": _pair(header.eta),
        "policy": {
            "q": _pair(header.policy.q),
            "s": _pair(header.policy.s),
        },
        "public_inputs": [item.hex() for item in header.public_inputs],
        "session_id": header.session_id.hex(),
        "weights": None if header.weights is None else header.weights.manifest,
    }
    if header.backend != TRANSPARENT_BACKEND:
        # The default backend is left implicit so transparent transcripts are
        # byte-identical to those written before backends were pluggable.
        header_document["backend"] = header.backend
    if header.max_faults:
        # Likewise f_max = 0 (no declarations admitted) is left implicit.
        header_document["max_faults"] = header.max_faults
    document: dict[str, JSONValue] = {
        "version": PROTOCOL_VERSION,
        "header": header_document,
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
    fields = _object(value, {"count", "root"}, where)
    return Weights(_int(fields["count"], f"{where}.count"), _hex(fields["root"], f"{where}.root"))


def _optional_key(value: object, keys: set[str], optional: str, where: str) -> dict[str, object]:
    """An object with ``keys``, plus ``optional`` if present (its absence is the default)."""

    return _optional_keys(value, keys, {optional}, where)


def _optional_keys(
    value: object, keys: set[str], optional: set[str], where: str
) -> dict[str, object]:
    """An object with ``keys``, plus any of ``optional`` (each absent one is its default)."""

    if type(value) is not dict or not keys <= set(value) <= keys | optional:
        raise MalformedTranscript(f"{where} must be an object with keys {sorted(keys)}")
    return value


def _proof(value: object, where: str) -> ProofMessage:
    fields = _optional_key(value, {"proof", "units"}, "foreign", where)
    foreign = _hex(fields["foreign"], f"{where}.foreign") if "foreign" in fields else b""
    if "foreign" in fields and not foreign:
        raise NoncanonicalTranscript(f"{where}.foreign is empty; it must be omitted")
    return ProofMessage(
        _list(fields["units"], _int, f"{where}.units"),
        _hex(fields["proof"], f"{where}.proof"),
        foreign,
    )


def _evidence(value: object, where: str) -> EvidenceMessage:
    fields = _optional_key(value, {"units"}, "proofs", where)
    proofs = _list(fields["proofs"], _proof, f"{where}.proofs") if "proofs" in fields else ()
    if "proofs" in fields and not proofs:
        raise NoncanonicalTranscript(f"{where}.proofs is empty; it must be omitted")
    return EvidenceMessage(_list(fields["units"], _openings, f"{where}.units"), proofs)


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
    header_fields = _optional_keys(
        top["header"],
        {
            "advice",
            "advice_bits",
            "claimed_outputs",
            "compiled_digest",
            "constructor",
            "eta",
            "policy",
            "public_inputs",
            "session_id",
            "weights",
        },
        {"backend", "max_faults"},
        "header",
    )
    policy_fields = _object(header_fields["policy"], {"q", "s"}, "header.policy")
    for name in ("compiled_digest", "constructor"):
        if type(header_fields[name]) is not str:
            raise MalformedTranscript(f"header.{name} must be a string")
    backend = header_fields.get("backend", TRANSPARENT_BACKEND)
    if type(backend) is not str or not backend:
        raise MalformedTranscript("header.backend must be a nonempty string")
    if backend == TRANSPARENT_BACKEND and "backend" in header_fields:
        raise NoncanonicalTranscript("header.backend names the default; it must be omitted")
    max_faults = _int(header_fields.get("max_faults", 0), "header.max_faults")
    if max_faults == 0 and "max_faults" in header_fields:
        raise NoncanonicalTranscript("header.max_faults is zero; it must be omitted")
    interior_fields = _optional_key(top["interiors"], {"commitments"}, "declarations", "interiors")
    declarations = _list(
        interior_fields.get("declarations", []), _int, "interiors.declarations"
    )
    if not declarations and "declarations" in interior_fields:
        raise NoncanonicalTranscript("interiors.declarations is empty; it must be omitted")
    try:
        policy = VerificationPolicy(
            _fraction(policy_fields["q"], "header.policy.q", checked),
            _fraction(policy_fields["s"], "header.policy.s", checked),
        )
        header = Header(
            _hex(header_fields["session_id"], "header.session_id"),
            header_fields["compiled_digest"],  # type: ignore[arg-type]
            header_fields["constructor"],  # type: ignore[arg-type]
            _hex(header_fields["advice"], "header.advice"),
            policy,
            _fraction(header_fields["eta"], "header.eta", checked),
            _list(header_fields["public_inputs"], _hex, "header.public_inputs"),
            _list(header_fields["claimed_outputs"], _hex, "header.claimed_outputs"),
            _weights(header_fields["weights"], "header.weights"),
            backend,
            max_faults,
            _int(header_fields["advice_bits"], "header.advice_bits"),
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
                _list(interior_fields["commitments"], _commitment, "interiors.commitments"),
                declarations,
            ),
            _challenge(SampleChallenge, top["sample_challenge"], "sample_challenge"),
            _evidence(top["evidence"], "evidence"),
        )
    except (MalformedTranscript, NoncanonicalTranscript):
        raise
    except (ProtocolError, ValueError, TypeError) as error:
        raise MalformedTranscript(str(error)) from error
    if encode_transcript(transcript) != data:
        raise NoncanonicalTranscript("transcript bytes are not in canonical form")
    return transcript
