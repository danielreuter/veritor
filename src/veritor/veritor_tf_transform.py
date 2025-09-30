# SPDX-License-Identifier: Apache-2.0
"""
Veritor StableHLO → Teacher-Forcing transformer (beefy/extensible, IREE Transform API).

Key ideas
---------
1) Use the Transform dialect (with PDL patterns) to tag "autoregressive decode" while-loops.
2) Analyze the tagged loops to *prove AR-ness* and extract the structural bits we need.
3) Emit a small teacher-forcing (prefill) function the verifier can run quickly and deterministically.
4) Keep it extensible: new decode variants register a recipe with a matcher + analyzer + rewriter.

Public API
----------
apply_teacher_forcing_transform(
    stablehlo_text: str,
    recipe: str = "jax_scan_v1",
    emit_bind_wrapper: bool = True,
    mode: str = "add_func",              # "add_func" (default) or "rewrite_in_place"
) -> str

Design choices
--------------
- "Additive by default": we **add** a TF function instead of mutating decode. This avoids meddling
  with the original while-loop and makes verification straightforward.
- If you really want to, set mode="rewrite_in_place" to replace the while with TF logits at the site.
- Weights:
  - Primary TF function takes `%tokens, %embed, %output` as args — simple + decoupled.
  - If we can resolve weights as constants, we also emit a `*_bind` wrapper that bakes them in.
- Recipes make it easy to add a new AR variant with a different loop shape or fused body.

Requirements
------------
pip install iree-compiler
"""

from __future__ import annotations
import dataclasses
import io
import os
import tempfile
from typing import Callable, Optional, Sequence

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
# Errors, logging, small utils
# =============================================================================

class TransformError(RuntimeError):
    pass

class MatchError(RuntimeError):
    pass

class RewriteError(RuntimeError):
    pass

def _tmp(text: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with io.open(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def _i64_attr(v: int):
    return mlir_ir.IntegerAttr.get(mlir_ir.IntegerType.get_signless(64), v)

def _sattr(ctx: mlir_ir.Context, s: str):
    return mlir_ir.StringAttr.get(s, ctx)

def _array_i64(vals: Sequence[int]):
    return mlir_ir.ArrayAttr.get([_i64_attr(v) for v in vals])

def _f32_attr(v: float):
    return mlir_ir.FloatAttr.get(mlir_ir.F32Type.get(), v)

def _ranked_tensor(elt, dims: Sequence[int]):
    return mlir_ir.RankedTensorType.get(list(dims), elt)

def _type_is_i32_scalar(t: mlir_ir.Type) -> bool:
    return isinstance(t, mlir_ir.RankedTensorType) and t.shape == [] and isinstance(t.element_type, mlir_ir.IntegerType) and t.element_type.width == 32

def _type_is_i32_1d(t: mlir_ir.Type) -> bool:
    return isinstance(t, mlir_ir.RankedTensorType) and len(t.shape) == 1 and isinstance(t.element_type, mlir_ir.IntegerType) and t.element_type.width == 32

def _type_is_f32_2d(t: mlir_ir.Type) -> bool:
    return isinstance(t, mlir_ir.RankedTensorType) and len(t.shape) == 2 and isinstance(t.element_type, mlir_ir.F32Type)

def _find_parent_func(op: mlir_ir.Operation) -> Optional[mlir_ir.Operation]:
    parent = op.parent
    while parent is not None:
        if parent.name == "func.func":
            return parent
        parent = parent.parent
    return None

def _printable_shape(t: mlir_ir.Type) -> str:
    if isinstance(t, mlir_ir.RankedTensorType):
        return "x".join(str(d) for d in t.shape)
    return str(t)


# =============================================================================
# Transform scripts (PDL+Transform) and recipe plumbing
# =============================================================================

@dataclasses.dataclass
class TransformScript:
    name: str
    transform_ir: str  # transform.module { ... } with @__transform_main

@dataclasses.dataclass
class TFOptions:
    # Strategy for emitting TF graph
    mode: str = "add_func"              # "add_func" (default) | "rewrite_in_place"
    emit_bind_wrapper: bool = True      # Also emit a wrapper that bakes in weights if available
    annotate: bool = True               # Annotate new function/op with veritor.* metadata

@dataclasses.dataclass
class Recipe:
    key: str
    description: str
    transform: TransformScript
    analyzer: Callable[[mlir_ir.Operation], Optional["ARLoopInfo"]]
    rewriter: Callable[[mlir_ir.Module, "ARLoopInfo", TFOptions], None]


# -----------------------------------------------------------------------------
# Default Transform IR (tighter than v0): tag while-loops that look AR-like
# -----------------------------------------------------------------------------
#
# - Uses PDL to select `stablehlo.while`.
# - Relies on canonicalization so the body/cond are normalized.
# - Tag matches with `veritor.ar.loop` (and optionally kind string).
#
DEFAULT_TRANSFORM_IR = r"""
transform.module {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %m = transform.cast %root : !transform.any_op to !transform.op<"builtin.module">

    transform.with_pdl_patterns {
    ^p(%p0: !pdl.operation):
      // Conservatively match while ops (details refined by Python analyzer).
      pdl.pattern @match_ar_decode_v1 : benefit(1) {
        %w = pdl.operation "stablehlo.while"
        pdl.rewrite %w { pdl.replace %w with %w }
      }

      transform.sequence failures(propagate) {
      ^s(%scope: !transform.any_op):
        %hits = transform.pdl_match @match_ar_decode_v1 in %m
        transform.annotate %hits { veritor.ar.loop, veritor.ar.kind = "decode_v1" } : !transform.any_op
        // Enable simple canonical forms to help the analyzer.
        transform.apply_patterns.canonicalization to %m
        transform.apply_cse to %m
        transform.apply_dce to %m
        transform.yield
      }
    }

    transform.yield
  }
}
""".strip()


# =============================================================================
# Analysis: Prove AR-ness and extract shapes/weights
# =============================================================================

@dataclasses.dataclass
class ARLoopInfo:
    while_op: mlir_ir.Operation
    func_op: mlir_ir.Operation
    # Loop structure
    cond_region: mlir_ir.Region
    body_region: mlir_ir.Region
    # Key values and shapes
    tokens_arg: Optional[mlir_ir.Value]   # loop-carried tokens buffer (tensor<Sxi32>)
    pos_arg: Optional[mlir_ir.Value]      # loop-carried position (i32 scalar)
    seq_len: int
    vocab: int
    hidden: int
    # Weights
    embed_const: Optional[mlir_ir.Operation]   # tensor<VxHxf32> constant if resolvable
    output_const: Optional[mlir_ir.Operation]  # tensor<HxVxf32> constant if resolvable
    # Body details
    has_dynamic_update_slice: bool
    has_gather: bool
    has_reduce: bool
    has_dot: bool
    has_dot_general: bool

def _tensor_dims(t: mlir_ir.Type) -> Optional[tuple[int, ...]]:
    if isinstance(t, mlir_ir.RankedTensorType):
        shape = tuple(int(d) for d in t.shape)
        if all(d >= 0 for d in shape):
            return shape
    return None

def _first_op_of_type(region: mlir_ir.Region, opname: str) -> Optional[mlir_ir.Operation]:
    for b in region.blocks:
        for op in b.operations:
            if op.name == opname:
                return op
    return None

def _walk_ops(region: mlir_ir.Region) -> list[mlir_ir.Operation]:
    out = []
    for b in region.blocks:
        for op in b.operations:
            out.append(op)
    return out

def _analyze_ar_while_decode_v1(while_op: mlir_ir.Operation) -> Optional[ARLoopInfo]:
    """Best-effort structural analysis for AR decode while-loop."""
    if while_op.name != "stablehlo.while":
        return None
    func_op = _find_parent_func(while_op)
    if func_op is None:
        return None

    regs = while_op.regions
    if len(regs) < 2:
        return None
    cond = regs[0]
    body = regs[1]

    # Heuristic checks in body: dynamic_update_slice, gather, reduce, dot/dot_general.
    has_dus = _first_op_of_type(body, "stablehlo.dynamic_update_slice") is not None
    has_gather = _first_op_of_type(body, "stablehlo.gather") is not None
    has_reduce = _first_op_of_type(body, "stablehlo.reduce") is not None
    has_dot = _first_op_of_type(body, "stablehlo.dot") is not None
    has_dotg = _first_op_of_type(body, "stablehlo.dot_general") is not None
    if not (has_dus and has_gather and (has_dot or has_dotg)):
        return None

    # Identify loop-carried tokens and pos from body block args by type.
    # While body entry block args mirror iter_args; scan argument types.
    tokens_arg = None
    pos_arg = None
    for bb in body.blocks:
        for barg in bb.arguments:
            t = barg.type
            if _type_is_i32_1d(t) and tokens_arg is None:
                tokens_arg = barg
            if _type_is_i32_scalar(t) and pos_arg is None:
                pos_arg = barg
        break

    if tokens_arg is None:
        # Try to guess tokens from any 1D i32 value used by gather
        for op in _walk_ops(body):
            if op.name == "stablehlo.gather":
                for v in op.operands:
                    if _type_is_i32_1d(v.type):
                        tokens_arg = v
                        break
    if tokens_arg is None:
        return None

    # seq_len from tokens type
    tok_dims = _tensor_dims(tokens_arg.type)
    if not tok_dims or len(tok_dims) != 1:
        return None
    seq_len = tok_dims[0]

    # Grab embed constant (operand 0 of gather likely)
    embed_const = None
    vocab = None
    hidden = None
    for op in _walk_ops(body):
        if op.name == "stablehlo.gather":
            # Heuristic: operand[0] == embedding table
            if len(op.operands) >= 1:
                emb_src = op.operands[0]
                # Try to find its defining op (constant)
                try:
                    defop = emb_src.owner
                except Exception:
                    defop = None
                if defop and defop.name == "stablehlo.constant":
                    embed_const = defop
                    dims = _tensor_dims(embed_const.results[0].type)
                    if dims and len(dims) == 2:
                        vocab, hidden = dims[0], dims[1]
            break

    # Find output projection weights from dot/dot_general
    output_const = None
    for op in _walk_ops(body):
        if op.name in ("stablehlo.dot", "stablehlo.dot_general"):
            # Search constant operand
            for v in op.operands:
                try:
                    d = v.owner
                except Exception:
                    d = None
                if d and d.name == "stablehlo.constant":
                    dims = _tensor_dims(d.results[0].type)
                    if dims and len(dims) == 2:
                        # Expect HxV
                        if hidden is None:
                            hidden = dims[0]
                        if vocab is None:
                            vocab = dims[1]
                        output_const = d
                        break
        if output_const:
            break

    # Validate dims
    if seq_len is None or vocab is None or hidden is None:
        return None

    return ARLoopInfo(
        while_op=while_op,
        func_op=func_op,
        cond_region=cond,
        body_region=body,
        tokens_arg=tokens_arg,
        pos_arg=pos_arg,
        seq_len=seq_len,
        vocab=vocab,
        hidden=hidden,
        embed_const=embed_const,
        output_const=output_const,
        has_dynamic_update_slice=has_dus,
        has_gather=has_gather,
        has_reduce=has_reduce,
        has_dot=has_dot,
        has_dot_general=has_dotg,
    )


# =============================================================================
# Rewriters
# =============================================================================

def _emit_tf_function(module: mlir_ir.Module, info: ARLoopInfo, opts: TFOptions) -> tuple[str, mlir_ir.Operation]:
    """
    Emit a *new* teacher-forcing function:

      func.func @{base}_veritor_tf(%tokens: tensor<Sxi32>, %embed: tensor<VxHxf32>, %output: tensor<HxVxf32>)
        -> tensor<(S-1)xVxf32>

    Returns (new_func_name, func_op).
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

    base = "veritor_tf"
    # Base on parent func symbol name if present
    if "sym_name" in info.func_op.attributes:
        fn_name_attr = info.func_op.attributes["sym_name"]
        base = f"{str(fn_name_attr.value)}_veritor_tf"
    new_fn = base
    # Ensure symbol uniqueness
    existing = set()
    for op in module.operation.regions[0].blocks[0].operations:
        if op.name == "func.func":
            if "sym_name" in op.attributes:
                a = op.attributes["sym_name"]
                existing.add(str(a.value))
    if new_fn in existing:
        idx = 1
        while f"{new_fn}_{idx}" in existing:
            idx += 1
        new_fn = f"{new_fn}_{idx}"

    ftype = mlir_ir.FunctionType.get([tok_ty, emb_ty, out_ty], [logits_ty])
    fn_op = mlir_ir.Operation.create(
        "func.func",
        regions=1,
        attributes={
            "sym_name": _sattr(ctx, new_fn),
            "function_type": mlir_ir.TypeAttr.get(ftype),
            "sym_visibility": _sattr(ctx, "private"),
        },
    )
    module.operation.regions[0].blocks[0].append(fn_op)
    region = fn_op.regions[0]
    entry = mlir_ir.Block.create_at_start(region, [tok_ty, emb_ty, out_ty])

    with mlir_ir.InsertionPoint(entry):
        # Arguments
        tok_arg = entry.arguments[0]
        emb_arg = entry.arguments[1]
        out_arg = entry.arguments[2]

        # %iota : tensor<Sxi32> with iota_dimension=0
        iota = mlir_ir.Operation.create(
            "stablehlo.iota",
            results=[tok_ty],
            attributes={"iota_dimension": _i64_attr(0)},
        )
        entry.append(iota)

        # %zero : f32 scalar constant
        zero_attr = mlir_ir.DenseElementsAttr.get_splat(
            _ranked_tensor(f32, []), _f32_attr(0.0)
        )
        zero_cst = mlir_ir.Operation.create(
            "stablehlo.constant", attributes={"value": zero_attr}, results=[_ranked_tensor(f32, [])]
        )
        entry.append(zero_cst)

        logits_1xV = []
        # Gather once: (V,H) + (S) -> (S,H)
        # dimension_numbers + slice_sizes as attributes
        dn = mlir_ir.Attribute.parse(
            ctx,
            '#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>'
        )
        slice_sizes = mlir_ir.Attribute.parse(ctx, f'array<i64: 1, {H}>')
        gather_res_ty = _ranked_tensor(f32, [S, H])
        gather = mlir_ir.Operation.create(
            "stablehlo.gather",
            operands=[emb_arg, tok_arg],
            results=[gather_res_ty],
            attributes={
                "dimension_numbers": dn,
                "slice_sizes": slice_sizes,
            },
        )
        entry.append(gather)

        # Build a small "add" reducer region for reduce(sum over dim 0)
        def _make_add_region(elem_ty: mlir_ir.Type) -> mlir_ir.Region:
            reg = mlir_ir.Region.create()
            blk = mlir_ir.Block.create_at_start(reg, [elem_ty, elem_ty])
            with mlir_ir.InsertionPoint(blk):
                add = mlir_ir.Operation.create(
                    "stablehlo.add",
                    operands=[blk.arguments[0], blk.arguments[1]],
                    results=[elem_ty],
                )
                blk.append(add)
                ret = mlir_ir.Operation.create(
                    "stablehlo.return", operands=[add.results[0]]
                )
                blk.append(ret)
            return reg

        for pos in range(S - 1):
            # pos constant (i32 scalar)
            pos_attr = mlir_ir.DenseElementsAttr.get_splat(
                _ranked_tensor(i32, []),
                mlir_ir.IntegerAttr.get(i32, pos)
            )
            pos_cst = mlir_ir.Operation.create(
                "stablehlo.constant",
                attributes={"value": pos_attr},
                results=[_ranked_tensor(i32, [])],
            )
            entry.append(pos_cst)

            # Broadcast pos to shape (S)
            pos_b = mlir_ir.Operation.create(
                "stablehlo.broadcast_in_dim",
                operands=[pos_cst.results[0]],
                results=[tok_ty],
                attributes={"broadcast_dimensions": _array_i64([])},
            )
            entry.append(pos_b)

            # cmp = compare LE (iota <= pos_b) : (Sxi32, Sxi32) -> (Sxi1)
            cmp = mlir_ir.Operation.create(
                "stablehlo.compare",
                operands=[iota.results[0], pos_b.results[0]],
                results=[_ranked_tensor(mlir_ir.IntegerType.get_signless(1), [S])],
                attributes={"comparison_direction": _sattr(ctx, "LE")},
            )
            entry.append(cmp)

            # convert i1 -> f32 : Sxf32
            mask_f = mlir_ir.Operation.create(
                "stablehlo.convert",
                operands=[cmp.results[0]],
                results=[_ranked_tensor(f32, [S])],
            )
            entry.append(mask_f)

            # broadcast mask to (S,H)
            mask_b = mlir_ir.Operation.create(
                "stablehlo.broadcast_in_dim",
                operands=[mask_f.results[0]],
                results=[_ranked_tensor(f32, [S, H])],
                attributes={"broadcast_dimensions": _array_i64([0])},
            )
            entry.append(mask_b)

            # masked = gather * mask_b : (S,H)
            mul = mlir_ir.Operation.create(
                "stablehlo.multiply",
                operands=[gather.results[0], mask_b.results[0]],
                results=[_ranked_tensor(f32, [S, H])],
            )
            entry.append(mul)

            # reduce across dim 0 → (H)
            red = mlir_ir.Operation.create(
                "stablehlo.reduce",
                operands=[mul.results[0], zero_cst.results[0]],
                results=[_ranked_tensor(f32, [H])],
                attributes={"dimensions": _array_i64([0])},
                regions=1,
            )
            red.regions[0] = _make_add_region(f32)
            entry.append(red)

            # logits: (H) · (H,V) -> (V)
            if info.has_dot or not info.has_dot_general:
                dot = mlir_ir.Operation.create(
                    "stablehlo.dot",
                    operands=[red.results[0], out_arg],
                    results=[_ranked_tensor(f32, [V])],
                )
            else:
                # dot_general variant if needed
                ddn = mlir_ir.Attribute.parse(
                    ctx,
                    "#stablehlo.dot_dimension_numbers<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>"
                )
                dot = mlir_ir.Operation.create(
                    "stablehlo.dot_general",
                    operands=[red.results[0], out_arg],
                    results=[_ranked_tensor(f32, [V])],
                    attributes={"dot_dimension_numbers": ddn},
                )
            entry.append(dot)

            # reshape (V) -> (1,V) for concat
            one_v = mlir_ir.Operation.create(
                "stablehlo.reshape",
                operands=[dot.results[0]],
                results=[_ranked_tensor(f32, [1, V])],
            )
            entry.append(one_v)
            logits_1xV.append(one_v.results[0])

        # concatenate along dim 0: (S-1,V)
        concat = mlir_ir.Operation.create(
            "stablehlo.concatenate",
            operands=logits_1xV,
            results=[logits_ty],
            attributes={"dimension": _i64_attr(0)},
        )
        entry.append(concat)

        # return
        ret = mlir_ir.Operation.create("func.return", operands=[concat.results[0]])
        entry.append(ret)

        # Annotate
        if opts.annotate:
            attrs = dict(fn_op.attributes)
            attrs["veritor.teacher_forcing"] = mlir_ir.UnitAttr.get(ctx)
            func_name = "unknown"
            if "sym_name" in info.func_op.attributes:
                func_name = str(info.func_op.attributes["sym_name"].value)
            attrs["veritor.source.func"] = _sattr(ctx, func_name)
            attrs["veritor.seq_len"] = _i64_attr(S)
            attrs["veritor.vocab"] = _i64_attr(V)
            attrs["veritor.hidden"] = _i64_attr(H)
            fn_op.attributes = attrs

    return new_fn, fn_op


def _emit_bind_wrapper_if_possible(module: mlir_ir.Module, info: ARLoopInfo, tf_name: str, opts: TFOptions) -> Optional[mlir_ir.Operation]:
    """Emit @{tf_name}_bind(%tokens) that bakes in constants if we can recover them."""
    if not (info.embed_const and info.output_const):
        return None

    ctx = module.context
    f32 = mlir_ir.F32Type.get()
    i32 = mlir_ir.IntegerType.get_signless(32)

    S, V, H = info.seq_len, info.vocab, info.hidden
    tok_ty = _ranked_tensor(i32, [S])
    emb_ty = _ranked_tensor(f32, [V, H])
    out_ty = _ranked_tensor(f32, [H, V])
    logits_ty = _ranked_tensor(f32, [S - 1, V])

    bind_name = f"{tf_name}_bind"
    # Create func
    ftype = mlir_ir.FunctionType.get([tok_ty], [logits_ty])
    fn_op = mlir_ir.Operation.create(
        "func.func",
        regions=1,
        attributes={
            "sym_name": _sattr(ctx, bind_name),
            "function_type": mlir_ir.TypeAttr.get(ftype),
            "sym_visibility": _sattr(ctx, "private"),
        },
    )
    module.operation.regions[0].blocks[0].append(fn_op)
    region = fn_op.regions[0]
    entry = mlir_ir.Block.create_at_start(region, [tok_ty])

    with mlir_ir.InsertionPoint(entry):
        tokens = entry.arguments[0]

        # Clone constants' values
        emb_attr = info.embed_const.attributes["value"] if "value" in info.embed_const.attributes else None
        out_attr = info.output_const.attributes["value"] if "value" in info.output_const.attributes else None
        if emb_attr is None or out_attr is None:
            return None

        emb_c = mlir_ir.Operation.create(
            "stablehlo.constant", attributes={"value": emb_attr}, results=[emb_ty]
        )
        out_c = mlir_ir.Operation.create(
            "stablehlo.constant", attributes={"value": out_attr}, results=[out_ty]
        )
        entry.append(emb_c)
        entry.append(out_c)

        # Call the param version
        callee = mlir_ir.FlatSymbolRefAttr.get(tf_name)
        call = mlir_ir.Operation.create(
            "func.call",
            operands=[tokens, emb_c.results[0], out_c.results[0]],
            results=[logits_ty],
            attributes={"callee": callee},
        )
        entry.append(call)

        # Return
        ret = mlir_ir.Operation.create("func.return", operands=[call.results[0]])
        entry.append(ret)

        if opts.annotate:
            attrs = dict(fn_op.attributes)
            attrs["veritor.teacher_forcing.bind_weights"] = mlir_ir.UnitAttr.get(ctx)
            fn_op.attributes = attrs

    return fn_op


def _rewrite_in_place(module: mlir_ir.Module, info: ARLoopInfo, opts: TFOptions) -> None:
    """
    Best-effort in-place rewrite: replace the while's logits result with TF logits.
    This keeps weights from the same function and avoids cross-function capture.
    """
    ctx = module.context
    # Emit a local TF subgraph just before the while op
    S, V, H = info.seq_len, info.vocab, info.hidden
    f32 = mlir_ir.F32Type.get()
    i32 = mlir_ir.IntegerType.get_signless(32)
    tok_ty = _ranked_tensor(i32, [S])
    out_logits_ty = _ranked_tensor(f32, [S - 1, V])

    # Try to find a tokens value in the parent block that matches tok_ty
    parent_block = info.while_op.parent
    if parent_block is None:
        raise RewriteError("While op has no parent block")

    # Heuristic: the tokens loop-carried value often originates outside the while; if not,
    # we fall back to info.tokens_arg (but it is local to body). We therefore default to
    # creating TF function instead if we cannot find a usable tokens outside.
    tokens_val = None
    for op in parent_block.operations:
        for res in op.results:
            if res.type == tok_ty:
                tokens_val = res
                break
        if tokens_val:
            break
    if tokens_val is None:
        # Fallback: emit a separate TF function instead of in-place rewrite.
        _emit_tf_function(module, info, opts)
        return

    # Find weights inside the same function
    embed_val = info.embed_const.results[0] if info.embed_const else None
    out_val = info.output_const.results[0] if info.output_const else None
    if embed_val is None or out_val is None:
        # Fallback to TF function emission
        _emit_tf_function(module, info, opts)
        return

    # Build TF subgraph adjacent to the while
    with mlir_ir.InsertionPoint.before(info.while_op):
        # Reuse the helper to emit a compact TF graph via a private function, then call it.
        tf_name, _ = _emit_tf_function(module, info, TFOptions(annotate=opts.annotate))
        callee = mlir_ir.FlatSymbolRefAttr.get(tf_name)
        call = mlir_ir.Operation.create(
            "func.call",
            operands=[tokens_val, embed_val, out_val],
            results=[out_logits_ty],
            attributes={"callee": callee},
        )
        parent_block.append(call)

        # Replace while result that matches logits_ty
        replaced = False
        for res in info.while_op.results:
            if res.type == out_logits_ty:
                res.replace_all_uses_with(call.results[0])
                replaced = True
                break
        # Erase while if no results are needed anymore
        if replaced:
            info.while_op.erase()


def _rewriter_add_func(module: mlir_ir.Module, info: ARLoopInfo, opts: TFOptions) -> None:
    tf_name, _ = _emit_tf_function(module, info, opts)
    if opts.emit_bind_wrapper:
        _emit_bind_wrapper_if_possible(module, info, tf_name, opts)


# =============================================================================
# Recipe registry
# =============================================================================

_RECIPE_REGISTRY: dict[str, Recipe] = {}

def register_recipe(recipe: Recipe) -> None:
    _RECIPE_REGISTRY[recipe.key] = recipe

def _default_recipe() -> Recipe:
    return Recipe(
        key="jax_scan_v1",
        description="Decode while-loop (JAX/XLA style): tokens dynamic_update_slice, gather, reduce, dot/dot_general.",
        transform=TransformScript("jax_scan_v1", DEFAULT_TRANSFORM_IR),
        analyzer=_analyze_ar_while_decode_v1,
        rewriter=lambda module, info, opts: (
            _rewriter_add_func(module, info, opts)
            if opts.mode == "add_func"
            else _rewrite_in_place(module, info, opts)
        ),
    )

if IREE_AVAILABLE:
    register_recipe(_default_recipe())


# =============================================================================
# Running the transform pipeline
# =============================================================================

def _run_transform(module: mlir_ir.Module, transform_ir: str) -> None:
    """Run Transform interpreter with @__transform_main from a temp file."""
    tf_path = _tmp(transform_ir, ".td.mlir")
    try:
        pm = mlir_pm.PassManager()
        pm.add(f'builtin.module(iree-preprocessing-transform-interpreter{{transform-spec-path="{tf_path}"}})')
        pm.run(module.operation)
    finally:
        try:
            os.remove(tf_path)
        except Exception:
            pass

def _cleanup(module: mlir_ir.Module) -> None:
    pm = mlir_pm.PassManager()
    # Tidy up regardless of mode.
    pm.add("builtin.module(canonicalize,cse,canonicalize,symbol-dce)")
    pm.run(module.operation)


def apply_teacher_forcing_transform(
    stablehlo_text: str,
    recipe: str = "jax_scan_v1",
    emit_bind_wrapper: bool = True,
    mode: str = "add_func",  # or "rewrite_in_place"
) -> str:
    """
    Main entry point.

    - Runs the recipe's Transform script to tag while-loops.
    - Analyzes each tagged while; if it proves AR decode shape, rewrites.
    - By default, *adds* @<func>_veritor_tf (and maybe @<func>_veritor_tf_bind).
    - Returns updated module text.

    Notes:
    - If no matches are proven, returns the input (post-canonicalization).
    - For safety, failures in one loop don't abort others.
    """
    if not IREE_AVAILABLE:
        raise ImportError("IREE compiler not available. Install with: pip install iree-compiler")

    if recipe not in _RECIPE_REGISTRY:
        raise KeyError(f"Unknown recipe '{recipe}'. Available: {list(_RECIPE_REGISTRY.keys())}")

    r = _RECIPE_REGISTRY[recipe]
    opts = TFOptions(mode=mode, emit_bind_wrapper=emit_bind_wrapper, annotate=True)

    with mlir_ir.Context() as ctx, mlir_ir.Location.unknown():
        # IREE contexts typically have StableHLO registered. We also allow unregistered to avoid hard failures.
        ctx.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(stablehlo_text)

        # 1) Tag candidates with Transform
        _run_transform(module, r.transform.transform_ir)

        # 2) Collect tagged while ops
        hits: list[mlir_ir.Operation] = []
        def collect_tagged(op):
            try:
                if op.name == "stablehlo.while":
                    # Check if tagged with veritor.ar.loop
                    try:
                        if "veritor.ar.loop" in op.attributes:
                            hits.append(op)
                    except:
                        pass
            except Exception:
                pass
            return mlir_ir.WalkResult.ADVANCE
        module.operation.walk(collect_tagged)

        # 3) Analyze and rewrite
        for w in hits:
            try:
                info = r.analyzer(w)
                if not info:
                    continue
                r.rewriter(module, info, opts)
            except Exception as e:
                # Keep going; record minimal context in the IR for debugging.
                attrs = dict(w.attributes)
                attrs["veritor.analysis_error"] = mlir_ir.StringAttr.get(repr(e), ctx)
                w.attributes = attrs

        # 4) Cleanup
        _cleanup(module)
        return str(module)


# =============================================================================
# Extending for new AR variants
# =============================================================================

ALT_TRANSFORM_SKELETON = r"""
transform.module {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %m = transform.cast %root : !transform.any_op to !transform.op<"builtin.module">
    transform.with_pdl_patterns {
    ^p(%p0: !pdl.operation):
      // Define one or more patterns tailored to your lowering variant.
      pdl.pattern @match_ar_variant_X : benefit(1) {
        %w = pdl.operation "stablehlo.while"
        pdl.rewrite %w { pdl.replace %w with %w }
      }
      transform.sequence failures(propagate) {
      ^s(%scope: !transform.any_op):
        %hits = transform.pdl_match @match_ar_variant_X in %m
        transform.annotate %hits { veritor.ar.loop, veritor.ar.kind = "variant_X" } : !transform.any_op
        transform.apply_patterns.canonicalization to %m
        transform.apply_cse to %m
        transform.apply_dce to %m
        transform.yield
      }
    }
    transform.yield
  }
}
""".strip()

def make_recipe(
    key: str,
    transform_ir: str,
    analyzer: Callable[[mlir_ir.Operation], Optional[ARLoopInfo]],
    rewriter: Callable[[mlir_ir.Module, ARLoopInfo, TFOptions], None],
    description: str = "",
) -> Recipe:
    return Recipe(
        key=key,
        description=description or key,
        transform=TransformScript(key, transform_ir),
        analyzer=analyzer,
        rewriter=rewriter,
    )

# Example: register a stricter variant (copy and customize the analyzer).
# register_recipe(make_recipe("torch_xla_v0", ALT_TRANSFORM_SKELETON, _analyze_ar_while_decode_v1, _rewriter_add_func))


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".mlir"):
        print("Usage: python veritor_tf_transform.py <input_stablehlo.mlir> [mode]")
        print("  mode: add_func (default) | rewrite_in_place")
        sys.exit(1)
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "add_func"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    out = apply_teacher_forcing_transform(text, recipe="jax_scan_v1", emit_bind_wrapper=True, mode=mode)
    print(out)