"""The compiler: JAX tracing turns a Python function into an instruction tape.

We do not write a compiler at all. `jax.make_jaxpr(f)` traces `f` and returns
a jaxpr -- a typed, SSA intermediate representation that is *already* the
object our protocol wants:

  - `invars` are the inputs (our input view),
  - each equation applies one primitive from a finite library and writes a
    fresh variable (our instructions, one write each, append-only),
  - `outvars` name the results (our output view).

This module just flattens a jaxpr into the canonical `Program` form: variables
become tape indices, each distinct scalar literal is materialized once onto
the tape by a `const` instruction (constant materialization, as in MLIR's
arith.constant or HLO's constant -- consumers then reference its cell), and
anything outside the whitelisted primitive library is rejected.

v0 restriction: scalar float32 programs only, so every equation writes exactly
one 32-bit cell. Tensor primitives (dot_general, ...) are deliberately
excluded -- in the full design they are the higher-order instructions that
*refine* into many fixed-width leaves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from veritor.machine import PRIMITIVES, Instruction, Operand, Program, encode_cell

try:  # JAX moved core types; support both locations.
    from jax.extend.core import Literal
except ImportError:  # pragma: no cover
    from jax.core import Literal  # type: ignore


class TraceError(ValueError):
    """The traced function does not fit the v0 machine model."""


def _check_scalar_f32(aval: Any, what: str) -> None:
    if getattr(aval, "shape", None) != ():
        raise TraceError(
            f"{what} has shape {aval.shape}; v0 supports scalars only "
            "(tensor ops are the v1 'higher-order instruction' story)"
        )
    if np.dtype(getattr(aval, "dtype", None)) != np.float32:
        raise TraceError(f"{what} has dtype {aval.dtype}; v0 supports float32 only")


def _literal_value(lit: Literal) -> np.float32:
    val = np.asarray(lit.val)
    if val.shape != ():
        raise TraceError(f"literal {lit} is not a scalar")
    return np.float32(val)


def trace(
    f: Callable[..., object],
    *example_inputs: Any,
) -> Program:
    """Trace `f` at float32 scalar inputs and return the canonical Program.

    Both prover and verifier call this on the *public* function. In v0 the
    verifier re-traces and compares instruction-tape Merkle roots -- the
    "verifier recompilation" baseline for instruction provenance.
    """
    xs = [np.float32(x) for x in example_inputs]
    if not xs:
        raise TraceError(
            "the program needs at least one input: Python folds pure-literal "
            "arithmetic before JAX ever sees it"
        )
    closed = jax.make_jaxpr(f)(*xs)
    jaxpr = closed.jaxpr

    if jaxpr.constvars:
        raise TraceError(
            "traced function closes over arrays (constvars); close over plain "
            "Python floats instead -- those become literals, materialized as consts"
        )

    # Inputs occupy tape cells 0..M-1.
    var_to_index: dict[Any, int] = {}
    for i, var in enumerate(jaxpr.invars):
        _check_scalar_f32(var.aval, f"input {i}")
        var_to_index[var] = i
    num_inputs = len(jaxpr.invars)

    # Each equation becomes one instruction; when an equation input is a
    # literal, a `const` instruction is first emitted to materialize it onto
    # the tape (writing its own cell), and the consumer references that cell.
    instructions: list[Instruction] = []
    # Constant pool: float32 bit pattern (the 4-byte cell encoding) -> tape
    # cell of the `const` that materialized it. Dedup by exact bits, in
    # first-use order, keeps the flattening deterministic -- the verifier
    # re-traces and must arrive at an identical instruction tape.
    const_pool: dict[bytes, int] = {}
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name not in PRIMITIVES:
            raise TraceError(
                f"primitive '{name}' is not in the instruction library "
                f"({sorted(PRIMITIVES)}); rewrite the function to avoid it"
            )
        # Scalar add/sub/mul/exp carry no meaningful parameters. Some JAX
        # versions attach e.g. {'accuracy': None} to exp; only all-None
        # parameters are accepted.
        if any(v is not None for v in eqn.params.values()):
            raise TraceError(f"primitive '{name}' has parameters {eqn.params}")
        if len(eqn.outvars) != 1:
            raise TraceError(f"primitive '{name}' has {len(eqn.outvars)} outputs")
        if len(eqn.invars) != PRIMITIVES[name].arity:
            raise TraceError(f"primitive '{name}' has unexpected arity {len(eqn.invars)}")

        operands = []
        for iv in eqn.invars:
            if isinstance(iv, Literal):
                value = _literal_value(iv)
                key = encode_cell(value)
                if key not in const_pool:
                    const_pool[key] = num_inputs + len(instructions)
                    instructions.append(
                        Instruction(prim="const", operands=(Operand.imm(float(value)),))
                    )
                operands.append(Operand.cell(const_pool[key]))
            else:
                if iv not in var_to_index:
                    raise TraceError(
                        f"instruction {len(instructions)} reads an unknown variable {iv}"
                    )
                operands.append(Operand.cell(var_to_index[iv]))

        out = eqn.outvars[0]
        _check_scalar_f32(out.aval, f"output of instruction {len(instructions)}")
        var_to_index[out] = num_inputs + len(instructions)
        instructions.append(Instruction(prim=name, operands=tuple(operands)))

    if not instructions:
        raise TraceError("the traced function performs no operations")

    # The output view: which tape cells the caller receives.
    output_indices = []
    for ov in jaxpr.outvars:
        if isinstance(ov, Literal) or ov not in var_to_index:
            raise TraceError("outputs must be computed cells or inputs, not constants")
        output_indices.append(var_to_index[ov])

    return Program(
        num_inputs=num_inputs,
        instructions=tuple(instructions),
        output_indices=tuple(output_indices),
    )
