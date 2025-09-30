# SPDX-License-Identifier: Apache-2.0
"""
Simplified Veritor StableHLO → Teacher-Forcing transformer.

This is a simpler version that doesn't rely on IREE's transform dialect,
but directly walks and analyzes the MLIR operations.
"""

from __future__ import annotations
import dataclasses
from typing import Optional

# IREE embedded MLIR
try:
    from iree.compiler import ir as mlir_ir
    from iree.compiler import passmanager as mlir_pm
    IREE_AVAILABLE = True
except ImportError:
    IREE_AVAILABLE = False
    mlir_ir = None
    mlir_pm = None


# =============================================================================
# Errors
# =============================================================================

class TransformError(RuntimeError):
    pass


# =============================================================================
# Simple utilities
# =============================================================================

def _i64_attr(v: int):
    return mlir_ir.IntegerAttr.get(mlir_ir.IntegerType.get_signless(64), v)

def _sattr(ctx: mlir_ir.Context, s: str):
    return mlir_ir.StringAttr.get(s, ctx)

def _array_i64(vals: list[int]):
    return mlir_ir.ArrayAttr.get([_i64_attr(v) for v in vals])

def _f32_attr(v: float):
    return mlir_ir.FloatAttr.get(mlir_ir.F32Type.get(), v)

def _ranked_tensor(elt, dims: list[int]):
    return mlir_ir.RankedTensorType.get(list(dims), elt)


# =============================================================================
# Analysis
# =============================================================================

@dataclasses.dataclass
class ARLoopInfo:
    while_op: mlir_ir.Operation
    seq_len: int = 5
    vocab: int = 10
    hidden: int = 8


def find_and_analyze_ar_loops(module: mlir_ir.Module) -> list[ARLoopInfo]:
    """Find and analyze AR while loops in the module."""
    loops = []

    def visitor(op):
        if op.name == "stablehlo.while":
            # Simple heuristic: assume it's AR if it has while
            # In reality we'd do more analysis here
            info = ARLoopInfo(
                while_op=op,
                seq_len=5,  # Would extract from analysis
                vocab=10,   # Would extract from analysis
                hidden=8    # Would extract from analysis
            )
            loops.append(info)
        return mlir_ir.WalkResult.ADVANCE

    module.operation.walk(visitor)
    return loops


# =============================================================================
# Rewriting
# =============================================================================

def emit_tf_function(module: mlir_ir.Module, info: ARLoopInfo) -> str:
    """
    Emit a teacher-forcing function.
    Returns the function name.
    """
    ctx = module.context
    f32 = mlir_ir.F32Type.get()
    i32 = mlir_ir.IntegerType.get_signless(32)

    S = info.seq_len
    V = info.vocab
    H = info.hidden

    tok_ty = _ranked_tensor(i32, [S])
    emb_ty = _ranked_tensor(f32, [V, H])
    out_ty = _ranked_tensor(f32, [H, V])
    logits_ty = _ranked_tensor(f32, [S - 1, V])

    func_name = "main_veritor_tf"

    # Create function
    ftype = mlir_ir.FunctionType.get([tok_ty, emb_ty, out_ty], [logits_ty])
    fn_op = mlir_ir.Operation.create(
        "func.func",
        regions=1,
        attributes={
            "sym_name": _sattr(ctx, func_name),
            "function_type": mlir_ir.TypeAttr.get(ftype),
            "sym_visibility": _sattr(ctx, "private"),
        },
    )

    # Add to module
    module.operation.regions[0].blocks[0].append(fn_op)

    # Create entry block
    region = fn_op.regions[0]
    entry = mlir_ir.Block.create_at_start(region, [tok_ty, emb_ty, out_ty])

    with mlir_ir.InsertionPoint(entry):
        tok_arg = entry.arguments[0]
        emb_arg = entry.arguments[1]
        out_arg = entry.arguments[2]

        # Simple placeholder: just return zeros for now
        # Real implementation would do the unrolled computation
        zero_attr = mlir_ir.DenseElementsAttr.get_splat(
            logits_ty, _f32_attr(0.0)
        )
        zeros = mlir_ir.Operation.create(
            "stablehlo.constant",
            attributes={"value": zero_attr},
            results=[logits_ty]
        )
        entry.append(zeros)

        # Return
        ret = mlir_ir.Operation.create("func.return", operands=[zeros.results[0]])
        entry.append(ret)

    return func_name


# =============================================================================
# Main transformation
# =============================================================================

def apply_teacher_forcing_transform_simple(
    stablehlo_text: str,
    emit_bind_wrapper: bool = True,
    mode: str = "add_func",
) -> str:
    """
    Simplified transformation that directly analyzes and rewrites.
    """
    if not IREE_AVAILABLE:
        raise ImportError("IREE compiler not available. Install with: pip install iree-compiler")

    with mlir_ir.Context() as ctx, mlir_ir.Location.unknown():
        ctx.allow_unregistered_dialects = True

        # Remove sdy dialect ops if present (newer JAX feature)
        if 'sdy.mesh' in stablehlo_text:
            lines = stablehlo_text.split('\n')
            filtered_lines = [line for line in lines if 'sdy.mesh' not in line]
            stablehlo_text = '\n'.join(filtered_lines)

        module = mlir_ir.Module.parse(stablehlo_text)

        # Find AR loops
        loops = find_and_analyze_ar_loops(module)

        # Add TF functions for each
        for info in loops:
            emit_tf_function(module, info)

        # Clean up with simple passes
        pm = mlir_pm.PassManager()
        pm.add("builtin.module(canonicalize,cse)")
        pm.run(module.operation)

        return str(module)