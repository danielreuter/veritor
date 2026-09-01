"""Tracer: jaxprs flatten to the expected tapes, and the whitelist holds."""

import jax.numpy as jnp
import numpy as np
import pytest

from veritor.tracer import TraceError, trace


def test_single_add_with_literal():
    # x + 1.0: the literal is materialized by a `const` instruction (cell 1),
    # and the add reads that cell instead of carrying an immediate.
    program = trace(lambda x: x + 1.0, np.float32(2.0))
    assert program.num_inputs == 1
    assert len(program.instructions) == 2
    const, add = program.instructions
    assert const.prim == "const"
    (imm,) = const.operands
    assert imm.is_literal and imm.literal == 1.0
    assert add.prim == "add"
    a, b = add.operands
    assert not a.is_literal and a.index == 0
    assert not b.is_literal and b.index == 1  # the const's cell
    assert program.output_indices == (2,)


def test_two_inputs_dataflow():
    # f(x, y) = x*y + x: instruction 0 writes cell 2, instruction 1 reads it.
    program = trace(lambda x, y: x * y + x, np.float32(1.0), np.float32(2.0))
    assert program.num_inputs == 2
    mul, add = program.instructions
    assert mul.prim == "mul"
    assert [op.index for op in mul.operands] == [0, 1]
    assert add.prim == "add"
    assert [op.index for op in add.operands] == [2, 0]
    assert program.output_indices == (3,)


def test_exp_is_unary():
    program = trace(lambda x: jnp.exp(x), np.float32(0.5))
    (instr,) = program.instructions
    assert instr.prim == "exp"
    assert len(instr.operands) == 1


def test_const_materialization_first_use_order():
    # f(x) = exp(x*(x+1)) + 2: each const appears right before its first
    # consumer, so the tape is const 1.0, add, mul, exp, const 2.0, add.
    program = trace(lambda x: jnp.exp(x * (x + 1.0)) + 2.0, np.float32(0.7))
    assert [i.prim for i in program.instructions] == [
        "const", "add", "mul", "exp", "const", "add",
    ]
    for instr in program.instructions:
        if instr.prim == "const":
            assert len(instr.operands) == 1 and instr.operands[0].is_literal
        else:
            assert all(not op.is_literal for op in instr.operands)


def test_literal_dedup_single_const():
    # 1.0 appears twice but is materialized once: dedup by bit pattern means
    # both adds read the same const cell.
    program = trace(lambda x: (x + 1.0) + 1.0, np.float32(0.0))
    consts = [i for i in program.instructions if i.prim == "const"]
    assert len(consts) == 1
    assert [i.prim for i in program.instructions] == ["const", "add", "add"]
    _, add1, add2 = program.instructions
    assert add1.operands[1].index == 1  # the const's cell
    assert add2.operands[1].index == 1  # same cell, not a second const


def test_division_rejected():
    with pytest.raises(TraceError, match="div"):
        trace(lambda x: x / 2.0, np.float32(1.0))


def test_integer_pow_rejected():
    with pytest.raises(TraceError, match="integer_pow"):
        trace(lambda x: x**2, np.float32(1.0))


def test_vector_input_rejected():
    with pytest.raises(TraceError, match="shape"):
        trace(lambda x: x + 1.0, np.ones(3, dtype=np.float32))


def test_trivial_program_rejected():
    with pytest.raises(TraceError):
        trace(lambda x: x, np.float32(1.0))
