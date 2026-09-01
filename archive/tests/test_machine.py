"""Machine: canonical encodings roundtrip; the interpreter matches JAX."""

import jax.numpy as jnp
import numpy as np
import pytest

from veritor.machine import (
    Instruction,
    Operand,
    Program,
    decode_cell,
    decode_instruction,
    encode_cell,
    encode_instruction,
    execute,
)
from veritor.tracer import trace


@pytest.mark.parametrize("value", [0.0, 1.0, -1.5, 3.14159, 1e30, -1e-30, float("inf")])
def test_cell_roundtrip(value):
    payload = encode_cell(np.float32(value))
    assert len(payload) == 4
    assert encode_cell(decode_cell(payload)) == payload


@pytest.mark.parametrize(
    "instr",
    [
        Instruction("add", (Operand.cell(0), Operand.cell(7))),
        Instruction("mul", (Operand.cell(3), Operand.cell(2))),
        Instruction("const", (Operand.imm(2.5),)),
        Instruction("exp", (Operand.cell(11),)),
    ],
)
def test_instruction_roundtrip(instr):
    payload = encode_instruction(instr)
    assert len(payload) == 11
    assert decode_instruction(payload) == instr


def test_literal_on_non_const_rejected_on_decode():
    # Literals may appear only as the sole operand of `const`. A hand-built
    # instruction smuggling an immediate into add fails to decode -- and
    # decode is what the verifier trusts.
    payload = encode_instruction(Instruction("add", (Operand.cell(0), Operand.imm(1.0))))
    with pytest.raises(AssertionError, match="literal"):
        decode_instruction(payload)


def test_const_with_cell_operand_rejected_on_decode():
    payload = encode_instruction(Instruction("const", (Operand.cell(0),)))
    with pytest.raises(AssertionError, match="const"):
        decode_instruction(payload)


def test_interpreter_matches_jax():
    def f(x):
        y = x * (x + 1.0)
        return jnp.exp(y) + 2.0

    x = np.float32(0.7)
    program = trace(f, x)
    cells = execute(program, [x])
    ours = cells[program.output_indices[0]]
    jax_result = np.float32(f(x))
    # add/sub/mul are bit-exact IEEE-754; libm exp may differ from XLA's by
    # an ulp, hence allclose rather than equality. Protocol soundness never
    # relies on matching JAX -- executor and verifier share apply_primitive.
    assert np.allclose(ours, jax_result, rtol=1e-6)


def test_reading_future_cell_asserts():
    program = Program(
        num_inputs=1,
        instructions=(Instruction("add", (Operand.cell(5), Operand.cell(0))),),
        output_indices=(1,),
    )
    with pytest.raises(AssertionError, match="future"):
        execute(program, [np.float32(1.0)])


def test_overrides_propagate_downstream():
    # const 1.0 -> cell1; x+cell1 -> cell2; cell2+cell1 -> cell3. Forging
    # cell2 changes cell3 honestly.
    def f(x):
        return (x + 1.0) + 1.0

    x = np.float32(0.0)
    program = trace(f, x)
    honest = execute(program, [x])
    forged = execute(program, [x], overrides={2: np.float32(10.0)})
    assert honest[2] == np.float32(1.0) and honest[3] == np.float32(2.0)
    assert forged[2] == np.float32(10.0) and forged[3] == np.float32(11.0)
