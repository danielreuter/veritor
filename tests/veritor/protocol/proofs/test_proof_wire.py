"""The canonical statement/witness codec: round trips, strictness, digests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from veritor.protocol import ProtocolError
from veritor.protocol.proofs import (
    STATEMENT_MAGIC,
    WITNESS_MAGIC,
    CommitmentRef,
    GateOp,
    KindProgram,
    Obligation,
    PositionRef,
    Statement,
    Witness,
    decode_statement,
    decode_witness,
    encode_statement,
    encode_witness,
    make_statement,
    statement_digest,
)


def digest(tag: str) -> bytes:
    return hashlib.sha256(tag.encode()).digest()


def program(kind: str = "kind-a") -> KindProgram:
    return KindProgram(
        digest(kind),
        3,
        (0, 2),
        (
            GateOp("in", ()),
            GateOp("mul", (("port", 0), ("local", 0))),
            GateOp("add", (("local", 1), ("port", 1))),
        ),
    )


def obligation(
    unit: int = 4, kind: str = "kind-a", session: str = "session"
) -> Obligation:
    return Obligation(
        digest(session),
        digest("compiled"),
        unit,
        1,
        digest(kind),
        (
            CommitmentRef(-1, digest("boundary"), digest("root-b"), 12),
            CommitmentRef(1, digest("interior"), digest("root-i"), 7),
        ),
        (
            PositionRef(0, 3, 30, "u16", b"\x00\x2a"),
            PositionRef(0, 4, 31, "u16"),
            PositionRef(1, 0, 40, "u16"),
            PositionRef(1, 1, 41, "u16"),
            PositionRef(1, 2, 42, "u16"),
        ),
        (0, 1),
        (2, 3, 4),
    )


def statement() -> Statement:
    return make_statement(
        "veritor.toy-isa@1", digest("gate-set"), 16, [program()], [obligation()]
    )


def test_statement_round_trips_and_is_canonical() -> None:
    encoded = encode_statement(statement())
    assert encoded.startswith(STATEMENT_MAGIC)
    decoded = decode_statement(encoded)
    assert decoded == statement()
    assert encode_statement(decoded) == encoded
    assert (
        statement_digest(statement())
        == hashlib.sha256(encoded).digest()
        == statement_digest(encoded)
    )


def test_witness_round_trips() -> None:
    witness = Witness(
        (((b"\x00\x2a", (digest("s0"), digest("s1"))), (b"\x01\x02", ())),)
    )
    encoded = encode_witness(witness)
    assert encoded.startswith(WITNESS_MAGIC)
    assert decode_witness(encoded) == witness
    witness.for_statement(
        make_statement(
            "veritor.toy-isa@1",
            digest("g"),
            16,
            [
                KindProgram(
                    digest("k"), 1, (0,), (GateOp("add", (("port", 0), ("port", 0))),)
                )
            ],
            [
                Obligation(
                    digest("s"),
                    digest("c"),
                    0,
                    0,
                    digest("k"),
                    (CommitmentRef(-1, digest("d"), digest("r"), 2),),
                    (PositionRef(0, 0, 0, "u16"), PositionRef(0, 1, 1, "u16")),
                    (0,),
                    (1,),
                )
            ],
        )
    )


def test_make_statement_sorts_kinds_and_obligations_and_rejects_duplicates() -> None:
    a, b = program("kind-a"), program("kind-b")
    early, late = obligation(2, "kind-b"), obligation(9, "kind-a")
    built = make_statement(
        "veritor.toy-isa@1", digest("g"), 16, [b, a, b], [late, early]
    )
    assert [item.kind for item in built.kinds] == sorted([a.kind, b.kind])
    assert [item.unit for item in built.obligations] == [2, 9]
    with pytest.raises(ProtocolError, match="distinct"):
        make_statement("veritor.toy-isa@1", digest("g"), 16, [a], [late, late])
    other = KindProgram(a.kind, 1, (), (GateOp("in", ()),))
    with pytest.raises(ProtocolError, match="two different programs"):
        make_statement("veritor.toy-isa@1", digest("g"), 16, [a, other], [])


def test_statement_validation_is_strict() -> None:
    with pytest.raises(ProtocolError, match="lacks"):
        make_statement("veritor.toy-isa@1", digest("g"), 16, [], [obligation()])
    with pytest.raises(ProtocolError, match="ports"):
        replace(obligation(), inputs=(0,)).check_program(program())
    with pytest.raises(ProtocolError, match="not earlier"):
        KindProgram(digest("k"), 1, (), (GateOp("add", (("local", 0), ("local", 0))),))
    with pytest.raises(ProtocolError, match="port index"):
        KindProgram(digest("k"), 1, (), (GateOp("add", (("port", 0), ("port", 0))),))
    with pytest.raises(ProtocolError, match="commitment the obligation lacks"):
        replace(
            obligation(), positions=(PositionRef(5, 0, 0, "u16"),), inputs=(), gates=()
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data + b"\0",
        lambda data: data[:-1],
        lambda data: b"x" + data[1:],
        lambda data: (
            data[: len(STATEMENT_MAGIC)]
            + b"\xff\xff\xff\xff"
            + data[len(STATEMENT_MAGIC) + 4 :]
        ),
    ],
    ids=["trailing", "truncated", "magic", "huge-length"],
)
def test_decode_statement_rejects_malformed_bytes(mutate) -> None:
    with pytest.raises(ProtocolError):
        decode_statement(mutate(encode_statement(statement())))


def test_decode_statement_rejects_noncanonical_order() -> None:
    a, b = program("kind-a"), program("kind-b")
    good = make_statement("veritor.toy-isa@1", digest("g"), 16, [a, b], [])
    swapped = Statement.__new__(Statement)
    for name, value in (
        ("gate_set_id", good.gate_set_id),
        ("gate_set_digest", good.gate_set_digest),
        ("width", good.width),
        ("kinds", (good.kinds[1], good.kinds[0])),
        ("obligations", ()),
    ):
        object.__setattr__(swapped, name, value)
    with pytest.raises(ProtocolError, match="increasing"):
        decode_statement(encode_statement(swapped))


def test_decode_witness_rejects_trailing_bytes() -> None:
    encoded = encode_witness(Witness((((b"\x01", ()),),)))
    with pytest.raises(ProtocolError, match="trailing"):
        decode_witness(encoded + b"\0")
