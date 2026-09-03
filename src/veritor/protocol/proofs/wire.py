"""Canonical bytes for statements and witnesses.

The batch statement is what a proof is *about*: a guest program commits
``sha256(encode_statement(statement))`` as a public value, so both parties
must produce the same bytes from the same demands.  The encoding is a strict,
length-prefixed binary framing (big-endian integers; ``u32`` length prefixes
for byte strings and lists; raw 32-byte digests) that a zkVM guest parses in a
few cycles per field, and it is *canonical*: :func:`decode_statement` accepts
exactly the outputs of :func:`encode_statement` (every index range-checked,
kinds and obligations sorted, no trailing bytes) and re-encoding a decoded
statement reproduces the input bytes.

The Rust mirror is ``zk/sp1/common/src/codec.rs``; the layout is::

    Statement   = MAGIC str gate_set_id digest gate_set_digest u32 width
                  list<KindProgram> kinds  list<Obligation> obligations
    KindProgram = digest kind u32 size list<u32> ports list<GateOp> gates
    GateOp      = str op list<Arg> args ;  Arg = u8 space (0 port, 1 local) u32 value
    Obligation  = digest session digest compiled u64 unit u64 replay_unit digest kind
                  list<CommitmentRef> list<PositionRef> list<u32> inputs list<u32> gates
    CommitmentRef = u64 owner+2 digest domain_id digest root u64 count
    PositionRef   = u32 commitment u64 rank u64 position str schema
                    u8 has_expected [bytes expected]
    Witness     = MAGIC list< list< bytes value list<digest> path > >
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from veritor.protocol.messages import ProtocolError

from .statement import (
    LOCAL,
    PORT,
    Arg,
    CommitmentRef,
    GateOp,
    KindProgram,
    Obligation,
    PositionRef,
    Statement,
    Witness,
)

STATEMENT_MAGIC = b"veritor/proofs/statement/v1\0"
WITNESS_MAGIC = b"veritor/proofs/witness/v1\0"

_SPACE_CODE = {PORT: 0, LOCAL: 1}
_SPACE_NAME = {0: PORT, 1: LOCAL}


class _Writer:
    __slots__ = ("_parts",)

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def raw(self, data: bytes) -> None:
        self._parts.append(data)

    def u8(self, value: int) -> None:
        self._parts.append(value.to_bytes(1, "big"))

    def u32(self, value: int) -> None:
        self._parts.append(value.to_bytes(4, "big"))

    def u64(self, value: int) -> None:
        self._parts.append(value.to_bytes(8, "big"))

    def digest(self, value: bytes) -> None:
        if len(value) != 32:
            raise ProtocolError("digests are 32 bytes")
        self._parts.append(value)

    def bytes(self, value: bytes) -> None:
        self.u32(len(value))
        self._parts.append(value)

    def string(self, value: str) -> None:
        self.bytes(value.encode("utf-8"))

    def getvalue(self) -> bytes:
        return b"".join(self._parts)


class _Reader:
    __slots__ = ("_data", "_offset")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, count: int, what: str) -> bytes:
        end = self._offset + count
        if end > len(self._data):
            raise ProtocolError(f"proof input truncated while reading {what}")
        chunk = self._data[self._offset : end]
        self._offset = end
        return chunk

    def magic(self, expected: bytes) -> None:
        if self.take(len(expected), "magic") != expected:
            raise ProtocolError("proof input has the wrong magic")

    def u8(self, what: str) -> int:
        return self.take(1, what)[0]

    def u32(self, what: str) -> int:
        return int.from_bytes(self.take(4, what), "big")

    def u64(self, what: str) -> int:
        return int.from_bytes(self.take(8, what), "big")

    def digest(self, what: str) -> bytes:
        return self.take(32, what)

    def bytes(self, what: str) -> bytes:
        return self.take(self.u32(what), what)

    def string(self, what: str) -> str:
        try:
            return self.bytes(what).decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError(f"{what} is not UTF-8") from None

    def count(self, what: str) -> int:
        count = self.u32(what)
        if count > len(self._data) - self._offset:
            raise ProtocolError(f"{what} count {count} exceeds the remaining bytes")
        return count

    def finish(self, what: str) -> None:
        if self._offset != len(self._data):
            raise ProtocolError(
                f"{len(self._data) - self._offset} trailing bytes after the {what}"
            )


# -- statements ---------------------------------------------------------------


def _write_program(writer: _Writer, program: KindProgram) -> None:
    writer.digest(program.kind)
    writer.u32(program.size)
    writer.u32(len(program.ports))
    for ordinal in program.ports:
        writer.u32(ordinal)
    writer.u32(len(program.gates))
    for gate in program.gates:
        writer.string(gate.op)
        writer.u32(len(gate.args))
        for space, value in gate.args:
            writer.u8(_SPACE_CODE[space])
            writer.u32(value)


def _write_obligation(writer: _Writer, obligation: Obligation) -> None:
    writer.digest(obligation.session)
    writer.digest(obligation.compiled)
    writer.u64(obligation.unit)
    writer.u64(obligation.replay_unit)
    writer.digest(obligation.kind)
    writer.u32(len(obligation.commitments))
    for commitment in obligation.commitments:
        writer.u64(commitment.owner + 2)
        writer.digest(commitment.domain_id)
        writer.digest(commitment.root)
        writer.u64(commitment.count)
    writer.u32(len(obligation.positions))
    for position in obligation.positions:
        writer.u32(position.commitment)
        writer.u64(position.rank)
        writer.u64(position.position)
        writer.string(position.schema)
        if position.expected is None:
            writer.u8(0)
        else:
            writer.u8(1)
            writer.bytes(position.expected)
    for slots in (obligation.inputs, obligation.gates):
        writer.u32(len(slots))
        for slot in slots:
            writer.u32(slot)


def encode_obligation(obligation: Obligation) -> bytes:
    """The canonical bytes of one obligation (a component of the statement)."""

    writer = _Writer()
    _write_obligation(writer, obligation)
    return writer.getvalue()


def encode_statement(statement: Statement) -> bytes:
    """The canonical bytes of a batch statement."""

    if not isinstance(statement, Statement):
        raise ProtocolError("encode_statement requires a Statement")
    writer = _Writer()
    writer.raw(STATEMENT_MAGIC)
    writer.string(statement.gate_set_id)
    writer.digest(statement.gate_set_digest)
    writer.u32(statement.width)
    writer.u32(len(statement.kinds))
    for program in statement.kinds:
        _write_program(writer, program)
    writer.u32(len(statement.obligations))
    for obligation in statement.obligations:
        _write_obligation(writer, obligation)
    return writer.getvalue()


def statement_digest(statement: Statement | bytes) -> bytes:
    """``sha256`` of the canonical statement bytes: the public value a proof commits."""

    data = statement if isinstance(statement, bytes) else encode_statement(statement)
    return hashlib.sha256(data).digest()


def _read_program(reader: _Reader) -> KindProgram:
    kind = reader.digest("kind digest")
    size = reader.u32("kind size")
    ports = tuple(reader.u32("port ordinal") for _ in range(reader.count("ports")))
    gate_count = reader.count("gates")
    gates: list[GateOp] = []
    for _ in range(gate_count):
        op = reader.string("gate op")
        args: list[Arg] = []
        for _ in range(reader.count("gate args")):
            space = reader.u8("arg space")
            value = reader.u32("arg value")
            if space not in _SPACE_NAME:
                raise ProtocolError(f"unknown argument space {space}")
            args.append((_SPACE_NAME[space], value))
        gates.append(GateOp(op, tuple(args)))
    return KindProgram(kind, size, ports, tuple(gates))


def _read_obligation(reader: _Reader) -> Obligation:
    session = reader.digest("session")
    compiled = reader.digest("compiled digest")
    unit = reader.u64("unit")
    replay_unit = reader.u64("replay unit")
    kind = reader.digest("obligation kind")
    commitments: list[CommitmentRef] = []
    for _ in range(reader.count("commitments")):
        owner_plus_two = reader.u64("owner")
        commitments.append(
            CommitmentRef(
                owner_plus_two - 2,
                reader.digest("domain id"),
                reader.digest("root"),
                reader.u64("commitment count"),
            )
        )
    positions: list[PositionRef] = []
    for _ in range(reader.count("positions")):
        commitment = reader.u32("position commitment")
        rank = reader.u64("rank")
        position = reader.u64("position")
        schema = reader.string("schema")
        flag = reader.u8("expected flag")
        if flag == 0:
            expected = None
        elif flag == 1:
            expected = reader.bytes("expected value")
        else:
            raise ProtocolError(f"bad expected flag {flag}")
        positions.append(PositionRef(commitment, rank, position, schema, expected))
    inputs = tuple(reader.u32("inputs") for _ in range(reader.count("inputs")))
    gates = tuple(reader.u32("gates") for _ in range(reader.count("gates")))
    return Obligation(
        session,
        compiled,
        unit,
        replay_unit,
        kind,
        tuple(commitments),
        tuple(positions),
        inputs,
        gates,
    )


def decode_statement(data: bytes) -> Statement:
    """Parse canonical statement bytes strictly; ``encode_statement`` inverts it."""

    if type(data) is not bytes:
        raise ProtocolError("decode_statement requires bytes")
    reader = _Reader(data)
    reader.magic(STATEMENT_MAGIC)
    gate_set_id = reader.string("gate set id")
    gate_set_digest = reader.digest("gate set digest")
    width = reader.u32("width")
    kinds = tuple(_read_program(reader) for _ in range(reader.count("kinds")))
    obligations = tuple(
        _read_obligation(reader) for _ in range(reader.count("obligations"))
    )
    reader.finish("statement")
    statement = Statement(gate_set_id, gate_set_digest, width, kinds, obligations)
    if encode_statement(statement) != data:
        raise ProtocolError("statement bytes are not canonical")
    return statement


# -- witnesses ----------------------------------------------------------------


def encode_witness(witness: Witness) -> bytes:
    """The canonical bytes of a witness (the guest's private input)."""

    if not isinstance(witness, Witness):
        raise ProtocolError("encode_witness requires a Witness")
    writer = _Writer()
    writer.raw(WITNESS_MAGIC)
    writer.u32(len(witness.obligations))
    for openings in witness.obligations:
        writer.u32(len(openings))
        for value, path in openings:
            writer.bytes(value)
            writer.u32(len(path))
            for digest in path:
                writer.digest(digest)
    return writer.getvalue()


def decode_witness(data: bytes) -> Witness:
    """Parse canonical witness bytes strictly."""

    if type(data) is not bytes:
        raise ProtocolError("decode_witness requires bytes")
    reader = _Reader(data)
    reader.magic(WITNESS_MAGIC)
    obligations: list[tuple[tuple[bytes, tuple[bytes, ...]], ...]] = []
    for _ in range(reader.count("witness obligations")):
        openings: list[tuple[bytes, tuple[bytes, ...]]] = []
        for _ in range(reader.count("openings")):
            value = reader.bytes("opening value")
            path = tuple(
                reader.digest("path digest") for _ in range(reader.count("path"))
            )
            openings.append((value, path))
        obligations.append(tuple(openings))
    reader.finish("witness")
    return Witness(tuple(obligations))


def encode_obligations(obligations: Sequence[Obligation]) -> bytes:
    """Concatenated canonical obligations, for logging and vectors (not a statement)."""

    return b"".join(encode_obligation(item) for item in obligations)
