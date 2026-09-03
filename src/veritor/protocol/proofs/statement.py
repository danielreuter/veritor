"""The public statements of the proof layer: obligations, kind programs, batches.

An :class:`Obligation` is what the verifier demands for one sampled
verification unit (VU): *this* copy of *this* kind, under *these* commitment
roots, has every input and output value it touches authenticated at *these*
``(rank, position, schema)`` coordinates, and the kind's relation holds over
them.  It names no value: the values are the witness.  A :class:`KindProgram`
is the relation of one kind in coordinates relative to the copy (its ports and
its own gate offsets), so one program serves every copy of the kind.  A
:class:`Statement` is a batch: one gate set, the programs of every kind that
occurs, and the obligations, all sorted so the encoding of a set is unique.

Everything here is plain data with strict validation; the canonical bytes are
in :mod:`veritor.protocol.proofs.wire`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from veritor.protocol.messages import ProtocolError

PORT: Literal["port"] = "port"
LOCAL: Literal["local"] = "local"

type ArgSpace = Literal["port", "local"]
type Arg = tuple[ArgSpace, int]
"""A gate argument: ``("port", k)`` reads the copy's ``k``-th port as listed in
:attr:`KindProgram.ports`; ``("local", j)`` reads the copy's own gate at offset ``j``."""

_U32 = 1 << 32
_U64 = 1 << 64


def _check_index(value: object, name: str, bound: int = _U32) -> int:
    if type(value) is not int or not 0 <= value < bound:
        raise ProtocolError(f"{name} must be an integer in [0, {bound})")
    return value


def _check_digest(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ProtocolError(f"{name} must be 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class GateOp:
    """One gate of a kind: its op and where it reads."""

    op: str
    args: tuple[Arg, ...]

    def __post_init__(self) -> None:
        if type(self.op) is not str or not self.op:
            raise ProtocolError("gate op must be a nonempty string")
        if type(self.args) is not tuple:
            raise ProtocolError("gate args must be a tuple")
        for arg in self.args:
            if (
                type(arg) is not tuple
                or len(arg) != 2
                or arg[0] not in (PORT, LOCAL)
                or type(arg[1]) is not int
                or not 0 <= arg[1] < _U32
            ):
                raise ProtocolError("gate args must be ('port' | 'local', index) pairs")


@dataclass(frozen=True, slots=True)
class KindProgram:
    """The relation of one kind, in coordinates relative to a copy.

    ``ports`` lists (ascending) the port ordinals of the definition that its
    gates read, transitively; a ``("port", k)`` argument means the port
    ``ports[k]``.  ``gates[j]`` is the copy's gate at offset ``j``, and every
    ``("local", i)`` argument of it has ``i < j``: the kind is a program that
    reads only what it has already produced.
    """

    kind: bytes
    size: int
    ports: tuple[int, ...]
    gates: tuple[GateOp, ...]

    def __post_init__(self) -> None:
        _check_digest(self.kind, "kind digest")
        _check_index(self.size, "kind size")
        if type(self.ports) is not tuple or any(
            type(item) is not int or not 0 <= item < _U32 for item in self.ports
        ):
            raise ProtocolError("kind ports must be a tuple of ordinals")
        if tuple(sorted(set(self.ports))) != self.ports:
            raise ProtocolError("kind ports must be strictly increasing")
        if type(self.gates) is not tuple or len(self.gates) != self.size:
            raise ProtocolError(
                f"kind declares {self.size} gates but lists {len(self.gates)}"
            )
        for offset, gate in enumerate(self.gates):
            if not isinstance(gate, GateOp):
                raise ProtocolError("kind gates must be GateOp values")
            for space, value in gate.args:
                if space == PORT and value >= len(self.ports):
                    raise ProtocolError(
                        f"gate {offset} reads port index {value} of {len(self.ports)}"
                    )
                if space == LOCAL and value >= offset:
                    raise ProtocolError(
                        f"gate {offset} reads local offset {value}, not earlier"
                    )


@dataclass(frozen=True, slots=True)
class CommitmentRef:
    """One commitment an obligation opens against: its owner, domain and root."""

    owner: int
    domain_id: bytes
    root: bytes
    count: int

    def __post_init__(self) -> None:
        if type(self.owner) is not int or not -2 <= self.owner < _U64 - 2:
            raise ProtocolError(
                "commitment owner must be -2, -1 or a replay unit index"
            )
        _check_digest(self.domain_id, "domain id")
        _check_digest(self.root, "commitment root")
        _check_index(self.count, "commitment count", _U64)


@dataclass(frozen=True, slots=True)
class PositionRef:
    """One authenticated coordinate: which commitment, at which rank and position.

    ``expected`` pins the value the position must hold (an ``in`` gate holds
    the public input of its rank); it is part of the public statement.
    """

    commitment: int
    rank: int
    position: int
    schema: str
    expected: bytes | None = None

    def __post_init__(self) -> None:
        _check_index(self.commitment, "position commitment")
        _check_index(self.rank, "rank", _U64)
        _check_index(self.position, "position", _U64)
        if type(self.schema) is not str or not self.schema:
            raise ProtocolError("leaf schema must be a nonempty string")
        if self.expected is not None and type(self.expected) is not bytes:
            raise ProtocolError("expected value must be bytes or None")


@dataclass(frozen=True, slots=True)
class Obligation:
    """The public statement for one sampled VU.

    ``session`` is the header digest of the run (it binds the session id, the
    compiled circuit, the policy, the public I/O and ``kappa_W``);
    ``compiled`` repeats ``H(C, I)`` explicitly.  ``unit`` is the VU index and
    ``replay_unit`` the RU it lies in.  ``positions`` are every coordinate the
    relation touches, in ascending address order; ``inputs[k]`` is the slot
    of the ``k``-th read port of the kind and ``gates[j]`` the slot of the
    copy's gate at offset ``j``.
    """

    session: bytes
    compiled: bytes
    unit: int
    replay_unit: int
    kind: bytes
    commitments: tuple[CommitmentRef, ...]
    positions: tuple[PositionRef, ...]
    inputs: tuple[int, ...]
    gates: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_digest(self.session, "session digest")
        _check_digest(self.compiled, "compiled digest")
        _check_index(self.unit, "unit", _U64)
        _check_index(self.replay_unit, "replay unit", _U64)
        _check_digest(self.kind, "kind digest")
        if type(self.commitments) is not tuple or not all(
            isinstance(item, CommitmentRef) for item in self.commitments
        ):
            raise ProtocolError("obligation commitments must be CommitmentRef values")
        if type(self.positions) is not tuple or not all(
            isinstance(item, PositionRef) for item in self.positions
        ):
            raise ProtocolError("obligation positions must be PositionRef values")
        if len(self.commitments) >= _U32 or len(self.positions) >= _U32:
            raise ProtocolError("obligation is too large")
        for item in self.positions:
            if item.commitment >= len(self.commitments):
                raise ProtocolError("position names a commitment the obligation lacks")
        for name, slots in (("inputs", self.inputs), ("gates", self.gates)):
            if type(slots) is not tuple or any(
                type(slot) is not int or not 0 <= slot < len(self.positions)
                for slot in slots
            ):
                raise ProtocolError(f"obligation {name} must index its positions")

    @property
    def key(self) -> tuple[bytes, bytes, int]:
        """The sort key of the canonical batch order."""

        return (self.session, self.compiled, self.unit)

    def check_program(self, program: KindProgram) -> None:
        """Raise unless this obligation is shaped for ``program``."""

        if program.kind != self.kind:
            raise ProtocolError("obligation and program name different kinds")
        if len(self.inputs) != len(program.ports):
            raise ProtocolError(
                f"obligation binds {len(self.inputs)} inputs but its kind reads "
                f"{len(program.ports)} ports"
            )
        if len(self.gates) != program.size:
            raise ProtocolError(
                f"obligation binds {len(self.gates)} gates but its kind has {program.size}"
            )


@dataclass(frozen=True, slots=True)
class Statement:
    """One batch: a gate set, the programs of its kinds and its obligations.

    Kinds are sorted by digest and obligations by :attr:`Obligation.key`, so
    a batch is a set and its encoding is unique.  Use :func:`make_statement`
    to build one from unordered inputs.
    """

    gate_set_id: str
    gate_set_digest: bytes
    width: int
    kinds: tuple[KindProgram, ...]
    obligations: tuple[Obligation, ...]

    def __post_init__(self) -> None:
        if type(self.gate_set_id) is not str or not self.gate_set_id:
            raise ProtocolError("gate set id must be a nonempty string")
        _check_digest(self.gate_set_digest, "gate set digest")
        _check_index(self.width, "width")
        if type(self.kinds) is not tuple or not all(
            isinstance(item, KindProgram) for item in self.kinds
        ):
            raise ProtocolError("statement kinds must be KindProgram values")
        digests = [item.kind for item in self.kinds]
        if sorted(set(digests)) != digests:
            raise ProtocolError("statement kinds must be strictly increasing by digest")
        if type(self.obligations) is not tuple or not all(
            isinstance(item, Obligation) for item in self.obligations
        ):
            raise ProtocolError("statement obligations must be Obligation values")
        keys = [item.key for item in self.obligations]
        if sorted(set(keys)) != keys:
            raise ProtocolError("statement obligations must be distinct and sorted")
        programs = {item.kind: item for item in self.kinds}
        for obligation in self.obligations:
            try:
                program = programs[obligation.kind]
            except KeyError:
                raise ProtocolError(
                    "obligation names a kind the statement lacks"
                ) from None
            obligation.check_program(program)

    def program(self, kind: bytes) -> KindProgram:
        for item in self.kinds:
            if item.kind == kind:
                return item
        raise KeyError(kind.hex())


def make_statement(
    gate_set_id: str,
    gate_set_digest: bytes,
    width: int,
    kinds: Iterable[KindProgram],
    obligations: Iterable[Obligation],
) -> Statement:
    """Build the canonical statement of an unordered batch.

    Duplicate kinds must be identical programs; duplicate obligations are an
    error (a batch is a set of demands).
    """

    programs: dict[bytes, KindProgram] = {}
    for program in kinds:
        previous = programs.setdefault(program.kind, program)
        if previous != program:
            raise ProtocolError(
                f"kind {program.kind.hex()[:12]} has two different programs"
            )
    ordered = sorted(obligations, key=lambda item: item.key)
    return Statement(
        gate_set_id,
        gate_set_digest,
        width,
        tuple(programs[kind] for kind in sorted(programs)),
        tuple(ordered),
    )


@dataclass(frozen=True, slots=True)
class Witness:
    """The secret side of a statement: per obligation, per position, the opened value and path."""

    obligations: tuple[tuple[tuple[bytes, tuple[bytes, ...]], ...], ...]

    def __post_init__(self) -> None:
        if type(self.obligations) is not tuple:
            raise ProtocolError("witness must be a tuple of per-obligation openings")
        for openings in self.obligations:
            if type(openings) is not tuple:
                raise ProtocolError("witness openings must be tuples")
            for item in openings:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not bytes
                    or type(item[1]) is not tuple
                    or any(
                        type(digest) is not bytes or len(digest) != 32
                        for digest in item[1]
                    )
                ):
                    raise ProtocolError(
                        "a witness opening is (value, path of 32-byte digests)"
                    )

    def for_statement(self, statement: Statement) -> None:
        """Raise unless the witness has the statement's shape."""

        if len(self.obligations) != len(statement.obligations):
            raise ProtocolError(
                f"witness covers {len(self.obligations)} obligations, the statement "
                f"{len(statement.obligations)}"
            )
        for obligation, openings in zip(
            statement.obligations, self.obligations, strict=True
        ):
            if len(openings) != len(obligation.positions):
                raise ProtocolError("witness opens a different number of positions")


def select(
    statement: Statement, witness: Witness, obligations: Sequence[Obligation]
) -> Witness:
    """The sub-witness for ``obligations`` (a subset of the statement's), in canonical order."""

    lookup = {
        item.key: openings
        for item, openings in zip(
            statement.obligations, witness.obligations, strict=True
        )
    }
    return Witness(
        tuple(lookup[item.key] for item in sorted(obligations, key=lambda o: o.key))
    )
