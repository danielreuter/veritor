"""The machine model: a finite primitive library and an append-only value tape.

There is no mutable memory, no program counter, and no control flow. A
program is a straight-line sequence of instructions. The value tape starts
with the M input cells; instruction k appends exactly one new cell at
position M + k. An instruction may read any *earlier* cell, never a later
one; program constants enter the tape through `const` instructions, the only
place inline literals may appear. This append-only (SSA) discipline is what makes
a *local* check meaningful: to verify one instruction you only need its own
descriptor and the handful of committed cells it touches.

Every cell is exactly 32 bits: the big-endian IEEE-754 encoding of a
float32. Fixed width matters for the security story -- an instruction that
escapes checking can inject at most 32 unexplained bits into the tape.

The primitive library is a registry: functions decorated with @primitive.
"Opcode" here just means "which function from this agreed-upon library". The
verifier re-executes sampled instructions with the *same* `apply_primitive`
the executor used, so correct cells match bit-for-bit. Both sides must load
the same registry -- the library is part of the protocol's public parameters.
"""

from __future__ import annotations

import inspect
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, overload

import numpy as np

# ---------------------------------------------------------------------------
# Cells: fixed-width 32-bit values.
# ---------------------------------------------------------------------------

CELL_BYTES = 4


def encode_cell(value: np.float32) -> bytes:
    """Canonical 4-byte big-endian IEEE-754 encoding of a float32 cell."""
    return struct.pack(">f", float(np.float32(value)))


def decode_cell(payload: bytes) -> np.float32:
    assert len(payload) == CELL_BYTES
    return np.float32(struct.unpack(">f", payload)[0])


# ---------------------------------------------------------------------------
# The primitive library (the "instruction set").
# ---------------------------------------------------------------------------

# The instruction encoding has two fixed-size operand slots, so no primitive
# may take more than MAX_ARITY arguments. This is a wire-format constant, not
# a registry property: growing it would change INSTRUCTION_BYTES.
MAX_ARITY = 2


@dataclass(frozen=True)
class Primitive:
    """One entry of the shared library: a name (matching the JAX primitive
    the tracer sees) and the reference implementation of its semantics."""

    name: str
    arity: int
    fn: Callable[..., np.float32]


# name -> Primitive. The tracer only accepts programs whose jaxpr equations
# all use primitives registered here.
PRIMITIVES: dict[str, Primitive] = {}

PrimitiveFunction = Callable[..., np.float32]
PrimitiveFunctionT = TypeVar("PrimitiveFunctionT", bound=PrimitiveFunction)


@overload
def primitive(  # noqa: UP047
    fn: PrimitiveFunctionT,
    *,
    name: str | None = None,
) -> PrimitiveFunctionT: ...


@overload
def primitive(
    fn: None = None,
    *,
    name: str | None = None,
) -> Callable[[PrimitiveFunctionT], PrimitiveFunctionT]: ...


def primitive(  # noqa: UP047
    fn: PrimitiveFunctionT | None = None,
    *,
    name: str | None = None,
) -> PrimitiveFunctionT | Callable[[PrimitiveFunctionT], PrimitiveFunctionT]:
    """Register a function as a primitive: @primitive or @primitive(name=...).

    The primitive's name defaults to the function's name and must match the
    JAX primitive name that appears in jaxprs. Arity is read off the
    signature.
    """

    def register(candidate: PrimitiveFunctionT) -> PrimitiveFunctionT:
        prim_name = name or candidate.__name__
        arity = len(inspect.signature(candidate).parameters)
        assert 1 <= arity <= MAX_ARITY, f"{prim_name}: arity {arity} exceeds MAX_ARITY"
        assert prim_name not in PRIMITIVES, f"duplicate primitive: {prim_name}"
        PRIMITIVES[prim_name] = Primitive(
            name=prim_name,
            arity=arity,
            fn=candidate,
        )
        return candidate

    return register(fn) if fn is not None else register


def primitive_id(name: str) -> int:
    """Stable numeric id for the canonical instruction encoding. Derived from
    the sorted registry, so it is independent of definition order."""
    return sorted(PRIMITIVES).index(name)


def primitive_name(pid: int) -> str:
    return sorted(PRIMITIVES)[pid]


def apply_primitive(prim: str, args: Sequence[np.float32]) -> np.float32:
    """Reference semantics, shared by the executor and the verifier.

    numpy float32 arithmetic is deterministic on a given machine, and both
    sides call this exact function, so an honest cell always re-verifies
    bit-for-bit.
    """
    entry = PRIMITIVES.get(prim)
    if entry is None:
        raise ValueError(f"unknown primitive: {prim!r}")
    assert len(args) == entry.arity, f"{prim}: expected {entry.arity} args"
    return np.float32(entry.fn(*(np.float32(x) for x in args)))


# -- the library itself -----------------------------------------------------


@primitive
def const(a: np.float32) -> np.float32:
    # const is a source node: it materializes a program constant onto the
    # value tape (like MLIR's arith.constant or HLO's constant), so all other
    # instructions can use uniform cell references and every value cell has a
    # producing instruction. Its single operand is always an inline literal.
    return a


@primitive
def add(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a + b)


@primitive
def sub(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a - b)


@primitive
def mul(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(a * b)


@primitive
def neg(a: np.float32) -> np.float32:
    return np.float32(-a)


@primitive
def exp(a: np.float32) -> np.float32:
    return np.float32(np.exp(a, dtype=np.float32))


@primitive
def tanh(a: np.float32) -> np.float32:
    return np.float32(np.tanh(a, dtype=np.float32))


@primitive(name="max")
def max_(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(np.maximum(a, b))


# ---------------------------------------------------------------------------
# Instructions and their canonical byte encoding.
# ---------------------------------------------------------------------------

_OPERAND_INDEX = 0  # operand payload is a tape index (u32)
_OPERAND_LITERAL = 1  # operand payload is an inline float32 (an "immediate")
_OPERAND_UNUSED = 2  # padding slot for unary primitives


@dataclass(frozen=True)
class Operand:
    """One input of an instruction: an earlier tape cell, or a literal."""

    is_literal: bool
    index: int = 0  # meaningful when is_literal is False
    literal: float = 0.0  # meaningful when is_literal is True (a float32)

    @staticmethod
    def cell(index: int) -> Operand:
        return Operand(is_literal=False, index=index)

    @staticmethod
    def imm(value: float) -> Operand:
        return Operand(is_literal=True, literal=float(np.float32(value)))

    def describe(self) -> str:
        return f"imm({self.literal})" if self.is_literal else f"cell[{self.index}]"


@dataclass(frozen=True)
class Instruction:
    """One tape instruction: a primitive applied to some operands.

    Instruction k of a program implicitly writes cell M + k, so the write
    index is not stored here.
    """

    prim: str
    operands: tuple[Operand, ...]

    def describe(self) -> str:
        return f"{self.prim}({', '.join(op.describe() for op in self.operands)})"


# Encoding layout (11 bytes, fixed size):
#   byte 0        primitive id
#   bytes 1..5    operand slot 0: kind byte + 4-byte payload
#   bytes 6..10   operand slot 1: kind byte + 4-byte payload
# Index payloads are big-endian u32; literal payloads are big-endian IEEE-754.
INSTRUCTION_BYTES = 1 + MAX_ARITY * 5


def encode_instruction(instr: Instruction) -> bytes:
    out = bytes([primitive_id(instr.prim)])
    for slot in range(MAX_ARITY):
        if slot < len(instr.operands):
            op = instr.operands[slot]
            if op.is_literal:
                out += bytes([_OPERAND_LITERAL]) + struct.pack(">f", op.literal)
            else:
                out += bytes([_OPERAND_INDEX]) + struct.pack(">I", op.index)
        else:
            out += bytes([_OPERAND_UNUSED]) + bytes(4)
    return out


def decode_instruction(payload: bytes) -> Instruction:
    assert len(payload) == INSTRUCTION_BYTES
    prim = primitive_name(payload[0])
    operands = []
    for slot in range(MAX_ARITY):
        chunk = payload[1 + slot * 5 : 1 + (slot + 1) * 5]
        kind, body = chunk[0], chunk[1:]
        if kind == _OPERAND_INDEX:
            operands.append(Operand.cell(struct.unpack(">I", body)[0]))
        elif kind == _OPERAND_LITERAL:
            operands.append(Operand.imm(struct.unpack(">f", body)[0]))
        else:
            assert kind == _OPERAND_UNUSED and body == bytes(4)
    instr = Instruction(prim=prim, operands=tuple(operands))
    assert len(instr.operands) == PRIMITIVES[prim].arity, "arity mismatch in encoding"
    # Literal operands may appear ONLY as the sole operand of `const`; every
    # other primitive reads earlier tape cells. The verifier decodes committed
    # instructions with this function, so the invariant holds for everything
    # it checks.
    if prim == "const":
        assert instr.operands[0].is_literal, "const requires a literal operand"
    else:
        assert all(
            not op.is_literal for op in instr.operands
        ), f"{prim}: literal operands are allowed only on const"
    return instr


# ---------------------------------------------------------------------------
# Programs and the executor.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Program:
    """The instruction tape plus its input/output views.

    The value tape layout is implied: cells 0..num_inputs-1 hold the inputs,
    and instruction k writes cell num_inputs + k. `output_indices` says
    which cells constitute the program's output (the "output view").
    """

    num_inputs: int
    instructions: tuple[Instruction, ...]
    output_indices: tuple[int, ...]

    @property
    def num_cells(self) -> int:
        return self.num_inputs + len(self.instructions)

    def describe(self) -> str:
        lines = [f"inputs: cells 0..{self.num_inputs - 1}"]
        for k, instr in enumerate(self.instructions):
            lines.append(f"cell[{self.num_inputs + k}] = {instr.describe()}")
        lines.append(f"outputs: {[f'cell[{i}]' for i in self.output_indices]}")
        return "\n".join(lines)


def execute(
    program: Program,
    inputs: Sequence[float | np.float32],
    overrides: Mapping[int, float | np.float32] | None = None,
) -> list[np.float32]:
    """Run the program, returning the full value tape.

    `overrides` models a *dishonest* executor: at each overridden cell index
    the computed value is replaced, and later instructions honestly consume
    the corrupted value. This is the canonical adversary of the counting
    argument -- each override makes exactly one instruction locally
    inconsistent, and downstream cells remain locally consistent.
    """
    assert len(inputs) == program.num_inputs
    overrides = overrides or {}
    cells: list[np.float32] = [np.float32(x) for x in inputs]
    for k, instr in enumerate(program.instructions):
        args = []
        for op in instr.operands:
            if op.is_literal:
                args.append(np.float32(op.literal))
            else:
                assert op.index < len(cells), "instruction reads a future cell"
                args.append(cells[op.index])
        value = apply_primitive(instr.prim, args)
        write_index = program.num_inputs + k
        if write_index in overrides:
            value = np.float32(overrides[write_index])
        cells.append(value)
    return cells
