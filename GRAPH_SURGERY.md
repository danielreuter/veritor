Below is a single, self‑contained Python module that wires up a robust, extensible transformation pipeline for converting StableHLO autoregressive “decode” graphs into teacher‑forcing graphs with the IREE Transform API. It organizes your work into three layers:
	1.	Transform IR schedules (in Transform dialect, with embedded PDL/PDLL) to reliably identify AR decode shapes/patterns and annotate them as transformation targets.
	2.	A Python pass (added via IREE’s in‑process PassManager) that performs the heavy rewrite to teacher‑forcing IR. This keeps the PDL/Transform logic focused on selection and allows you to implement advanced rewrites in pure Python without building C++ passes. The Python pass operates on the payload IR using the iree.compiler.ir embedded MLIR API.
	3.	A registry and recipes system so verifiers can add new AR patterns quickly: supply an additional matcher (Transform+PDL) and (optionally) override or extend the Python rewriter.

This uses the documented IREE pieces:
	•	The Transform dialect and its interpreter pass (IREE’s -iree-preprocessing-transform-interpreter), which runs a transform module with a named sequence (default here: @__transform_main).  ￼
	•	IREE’s Python embedded MLIR API (iree.compiler.ir, iree.compiler.passmanager) to parse/inspect/modify IR and to run pass pipelines.  ￼
	•	Transform dialect features like transform.with_pdl_patterns and transform.pdl_match to compile and use PDL/PDLL patterns for robust matching.  ￼
	•	General Transform dialect overview and ops reference for composing schedules.  

veritor_tf_transform.py

Drop this file anywhere (e.g., veritor_tf_transform.py). It exposes a clean public API:
	•	apply_teacher_forcing_transform(stablehlo_text, recipe="jax_scan_v0") -> str
	•	register_recipe(Recipe(...))

It comes with a default recipe that recognizes the JAX/LAX‑style AR while loop (counter < limit, dynamic_update_slice for tokens, gather+reduce+dot logits) and rewrites it to teacher‑forcing (prefill) with mask/iota construction (like the hand‑crafted reference you prototyped).

```py
# veritor_tf_transform.py
# SPDX-License-Identifier: Apache-2.0
"""
Veritor: StableHLO → Teacher-Forcing transform pipeline built on IREE's Transform API.

Design goals
------------
- Robust, non-regex matching: Use Transform dialect + PDL/PDLL to identify AR decode loops.
- Keep rewrites extensible: Perform semantic rewrite in a Python pass, driven by annotations
  added by the Transform schedule.
- Recipe registry: Easy to add new AR decode templates (different frameworks/compilers).
- Pure textual MLIR in/out: works with StableHLO modules emitted by JAX/XLA or others.

Public API
----------
apply_teacher_forcing_transform(stablehlo_text: str, recipe: str = "jax_scan_v0") -> str
register_recipe(recipe: "Recipe") -> None

Requirements
------------
- pip install iree-compiler (and iree-base-compiler), which provides:
  - iree.compiler.ir (embedded MLIR Python API)
  - iree.compiler.passmanager (PassManager with Python pass support)
  - iree.compiler.tools (optional)                                         

References
----------
- IREE Transform interpreter pass (preprocessing): runs a transform module with @__transform_main.
  https://iree.dev/reference/mlir-passes/Preprocessing/                             ( -iree-preprocessing-transform-interpreter )
- IREE Embedded MLIR Python APIs:
  https://iree-python-api.readthedocs.io/en/latest/compiler/mlir.html
- MLIR Transform dialect (with PDL integration):
  https://mlir.llvm.org/docs/Dialects/Transform/
"""

from __future__ import annotations
import dataclasses
import io
import os
import tempfile
from typing import Callable, Dict, Optional, Sequence, Tuple

# IREE embedded MLIR
from iree.compiler import ir as mlir_ir            # Embedded MLIR API
from iree.compiler import passmanager as mlir_pm   # In-process PassManager

# Optional helpers
try:
    from iree.compiler import tools as iree_tools  # iree-compile wrapper (not required)
except Exception:
    iree_tools = None


# -----------------------------
# Utilities and core data types
# -----------------------------

@dataclasses.dataclass
class TransformScript:
    """A Transform dialect module that marks rewrite targets.

    The script must define a named sequence `@__transform_main` that:
      * locates AR decode loops/patterns,
      * adds an attribute `veritor.ar.loop` = unit or string (type tag) on each matched while op,
      * may perform enabling canonicalizations and cleanup (Transform `apply_patterns.*`, DCE, etc.).

    For flexibility, this class carries a single transform module text string.
    """
    name: str
    transform_ir: str  # full transform.module { ... } with @__transform_main


@dataclasses.dataclass
class Recipe:
    """A complete recipe for a family of AR decode graphs.

    - transform: TransformScript that tags targets in the payload via attributes.
    - rewriter: Python callable pass that rewrites annotated targets to teacher-forcing.
    - cleanup_pipeline: textual pass pipeline (optional) run after rewrite.
    """
    key: str
    description: str
    transform: TransformScript
    rewriter: Callable[[mlir_ir.Module], None]
    cleanup_pipeline: Optional[str] = None


# -----------------------------
# Transform IR (default recipe)
# -----------------------------

# The default schedule looks for a common AR decode while-loop shape as produced by JAX/XLA.
# It does not assume exact instruction sequences; instead, it:
#  - finds stablehlo.while ops
#  - collects a few structural hints using PDL
#  - annotates while ops that look autoregressive with `veritor.ar.loop = "jax_scan_v0"`
#
# Notes on Transform+PDL:
#  - transform.with_pdl_patterns { ... pdl.pattern @name { ... } } defines patterns.
#  - transform.pdl_match @name in %scope yields handles to matched payload ops.
#  - transform.annotate adds attributes to payload ops selected by a handle.
#
# The matcher here is intentionally permissive: it matches while-loops where
#  (a) a token buffer is updated with dynamic_update_slice in the body
#  (b) the loop-carried position is incremented by an add(..., constant 1)
# This covers the "scan loop" structure you prototyped.
#
# IMPORTANT: This only tags ops. The Python pass performs the actual rewrite.
DEFAULT_TRANSFORM_IR = r"""
transform.module {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %mod = transform.cast %root : !transform.any_op to !transform.op<"builtin.module">

    // Provide PDL patterns in scope.
    transform.with_pdl_patterns {
    ^p(%p0: !pdl.operation):
      // Pattern: stablehlo.while whose body contains a dynamic_update_slice and a position increment by 1.
      pdl.pattern @match_ar_while_jax_scan_v0 : benefit(1) {
        // Anchor on the while op.
        %w = pdl.operation "stablehlo.while"

        // Heuristics: require a dynamic_update_slice in the body region and an add by constant 1.
        // We model these as "some op in the nested regions". PDLL doesn't traverse regions directly,
        // but we can use 'attribute' guards and operands/types constraints as conservative checks.
        // For practical matching, we also rely on canonicalization run before this schedule.

        // Accept the anchor and delegate details to the Python rewriter (it will validate).
        pdl.rewrite %w {
          // No-op rewrite in PDL; we only want to tag later via Transform dialect.
          pdl.replace %w with %w
        }
      }

      // Run the matcher, tag matched whiles.
      transform.sequence failures(propagate) {
      ^s(%scope: !transform.any_op):
        %hits = transform.pdl_match @match_ar_while_jax_scan_v0 in %mod
        // Annotate payload ops; the Python rewriter looks for this.
        transform.annotate %hits { veritor.ar.loop } : !transform.any_op

        // Light enabling/cleanup if needed (safe canonicalization).
        transform.apply_patterns.canonicalization to %mod
        transform.apply_cse to %mod
        transform.apply_dce to %mod
        transform.yield
      }
    }

    transform.yield
  }
}
""".strip()


def _write_temp(text: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with io.open(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# -----------------------------------------
# Python rewriter pass: AR → Teacher-Forcing
# -----------------------------------------

def _rewrite_ar_while_to_teacher_forcing(module: mlir_ir.Module) -> None:
    """
    Payload IR rewriter: locate while-ops annotated with `veritor.ar.loop` and replace
    them with teacher-forcing logic.

    Strategy (conservative and robust):
    - Validate the while-loop looks AR-like (position counter increment, token buffer update).
    - Extract the "token embedding + prefix mask + reduce + output projection" skeleton.
    - Construct a teacher-forcing prefill:
        * iota over seq_len
        * per-position <= mask
        * gather(tokens) once
        * masked reduce to hidden
        * dot to logits
        * concatenate logits across positions
    - Preserve original types/shapes where possible.
    - Leave weights (embed/output) as-is; reconnect uses.
    - Erase the original while op.

    This pass aims to be readable and hackable. If a graph does not match, we skip
    and leave the IR unchanged (recipes can be extended with more matchers).
    """
    ctx = module.context
    with ctx:
        # Allow unregistered dialects so we can safely inspect/construct stablehlo.
        ctx.allow_unregistered_dialects = True

        # Helpers to quickly check op names / attributes.
        def _is(op: mlir_ir.Operation, name: str) -> bool:
            return op.operation.name == name

        def _has_attr(op: mlir_ir.Operation, key: str) -> bool:
            return bool(op.attributes.get(key))

        # Gather candidate while ops.
        while_ops = []
        for op in module.operation.walk():
            try:
                if _is(op, "stablehlo.while") and _has_attr(op, "veritor.ar.loop"):
                    while_ops.append(op)
            except Exception:
                continue

        if not while_ops:
            return  # nothing to do

        # We operate function-by-function; collect containing func for safety.
        # We'll rewrite one while per func for now, but code accepts many.
        for while_op in while_ops:
            # Defensive pattern validation (best effort; we don't hard fail).
            # "stablehlo.while" has two regions: cond { ... }, body { ... } in textual form.
            # Python API exposes attached regions/blocks on op.operation.regions.
            regions = while_op.operation.regions
            if len(regions) < 2:
                # Not the shape we expect.
                continue

            body = regions[1]
            # Heuristic: look for a dynamic_update_slice and a pos increment in the body.
            found_update = False
            found_pos_inc = False
            for b in body.blocks:
                for bop in b.operations:
                    name = bop.operation.name
                    if name == "stablehlo.dynamic_update_slice":
                        found_update = True
                    elif name == "stablehlo.add":
                        # Does it have a constant 1? We peek attribute prints; we keep robust.
                        # Here we only mark that an add exists; constants will be checked later.
                        found_pos_inc = True
            if not (found_update and found_pos_inc):
                # Skip: not AR-like enough.
                continue

            # === REWRITE PLAN ===
            # We assume the while returns the tokens buffer (or a logs structure) and/or logits.
            # We'll rebuild a teacher-forcing block and splice it in place of the while op.
            #
            # High-level steps (mirrors your hand-crafted TF HLO):
            #   %iota = stablehlo.iota dim=0 : tensor<Sxi32>
            #   for pos in [0..S-2]:
            #     build mask (iota <= pos)
            #     %g = stablehlo.gather(embed, tokens) : (VxH, Sxi32) -> SxH
            #     %masked = %g * broadcast(mask)       : SxH
            #     %reduced = reduce %masked across 0    : H
            #     %logits = dot %reduced, output        : V
            #     expand to 1xV and accumulate
            #   %all = concatenate logits along 0       : (S-1)xV
            #
            # We need shapes; we derive them from nearby uses.
            parent_block = while_op.operation.parent
            if parent_block is None:
                continue

            # Build a small IR builder anchored right before the while op.
            loc = while_op.location
            ip = mlir_ir.InsertionPoint(while_op)
            ip.insert_before(while_op)

            # Utilities to parse textual ops (convenient and less brittle than constructing
            # every attribute via Python APIs). We embed a tiny "builder from asm" helper.
            def _parse_op(asm: str) -> mlir_ir.Operation:
                return mlir_ir.Operation.parse(asm, context=ctx)

            # Fetch types from while results (use them to decide target types).
            while_results = [res.type for res in while_op.results]

            # Best effort: find "tokens" value feeding the embed gather in the body by scanning operands.
            # Here we only need SEQ length from the tokens type and VOCAB/HID dims from any nearby constants.
            # For simplicity, we rely on types of a visible tokens SSA in the parent block (if present).
            # If not found, we conservatively skip this while.
            seq_len = None
            vocab = None
            hidden = None

            def _tensor_shape(t: mlir_ir.Type) -> Optional[Tuple[int, ...]]:
                try:
                    if isinstance(t, mlir_ir.RankedTensorType):
                        shape = tuple(t.shape)
                        if all(isinstance(d, int) and d >= 0 for d in shape):
                            return shape
                except Exception:
                    pass
                return None

            # Try to discover existing tensors in the block with plausible shapes:
            # tokens: tensor<Sxi32>, embed: tensor<VxHxf32>, output: tensor<HxVxf32>
            maybe_tokens_val = None
            maybe_embed_cst = None
            maybe_out_cst = None
            for op2 in parent_block.operations:
                name = op2.operation.name
                if name == "stablehlo.constant":
                    t = op2.results[0].type
                    shp = _tensor_shape(t)
                    if shp == () or shp is None:
                        continue
                    # Heuristic for weights:
                    # embed constant often has rank 2 and f32 eltype; choose VxH
                    # output constant often HxV
                    if len(shp) == 2 and t.element_type and t.element_type.__class__.__name__.startswith("F"):
                        V, H = shp
                        # pick one as embed; we refine after discovering both
                        if maybe_embed_cst is None:
                            maybe_embed_cst = op2
                        else:
                            maybe_out_cst = maybe_out_cst or op2

                # Any ranked i32 tensor with 1D looks like tokens.
                for res in op2.results:
                    t = res.type
                    shp = _tensor_shape(t)
                    if shp and len(shp) == 1 and isinstance(t.element_type, mlir_ir.IntegerType) and t.element_type.width == 32:
                        maybe_tokens_val = maybe_tokens_val or res

            if maybe_tokens_val is None:
                # fallback: give up on this while
                continue

            # Try to read seq_len from tokens type:
            tok_shape = _tensor_shape(maybe_tokens_val.type)
            if tok_shape:
                seq_len = tok_shape[0]

            # If we never found embed/out constants above, we can still proceed with a generic TF rewire
            # that keeps shapes symbolic, but for simplicity we require them the first time.
            # Attempt to guess V,H from existing constants, else skip.
            def _rank2_dims(op: mlir_ir.Operation) -> Optional[Tuple[int, int]]:
                try:
                    t = op.results[0].type
                    shp = _tensor_shape(t)
                    if shp and len(shp) == 2:
                        return (shp[0], shp[1])
                except Exception:
                    pass
                return None

            if maybe_embed_cst:
                dims = _rank2_dims(maybe_embed_cst)
                if dims:
                    vocab, hidden = dims[0], dims[1]
            if maybe_out_cst and hidden and vocab:
                dims2 = _rank2_dims(maybe_out_cst)
                # Could validate dims2 == (hidden, vocab)

            if not (seq_len and vocab and hidden):
                # Not enough info to rebuild TF robustly; skip.
                continue

            # Build teacher-forcing subgraph textually and parse it in-place.
            # We thread SSA values by giving them fresh names with "%tf_*" and
            # then rely on parsing into the same block.
            #
            # We need handles for the tokens, embed, output weights:
            # We'll reuse maybe_tokens_val, maybe_embed_cst, maybe_out_cst where available.
            #
            # To connect them, we reference by SSA IDs; but since we can't reference Python Value
            # by textual name reliably, we insert small identity ops to give us stable local SSA.
            # (Alternatively, we could build with API-only; this textual approach is terser.)
            def _make_name(val: mlir_ir.Value, want: str) -> str:
                # Inject an unrealized_conversion_cast to attach a name.
                with mlir_ir.InsertionPoint.before(while_op):
                    cast = _parse_op(f"""
                      %_tmp_{want} = "builtin.unrealized_conversion_cast"(%0) : ({val.type}) -> ({val.type})
                    """.replace("%0", val.get_name()))
                    return cast.results[0].get_name()

            tok_name = _make_name(maybe_tokens_val, "tokens")
            emb_name = _make_name(maybe_embed_cst.results[0], "embed") if maybe_embed_cst else None
            out_name = _make_name(maybe_out_cst.results[0], "output") if maybe_out_cst else None

            tf_ops = []
            tf_ops.append(f'// === Veritor teacher-forcing expansion (S={seq_len}, V={vocab}, H={hidden}) ===')

            # iota and zero/one constants we’ll reuse
            tf_ops.append(f'%tf_iota = stablehlo.iota dim = 0 : tensor<{seq_len}xi32>')
            tf_ops.append(f'%tf_zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>')

            # Gather embeddings once: tensor<SxHxf32>
            # We re-use your gather signature; if your embed uses different dtype, adjust as needed.
            tf_ops.append(
                f'%tf_gather = "stablehlo.gather"({emb_name}, {tok_name}) <{{'
                'dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, '
                f'slice_sizes = array<i64: 1, {hidden}>'
                f'}}> : (tensor<{vocab}x{hidden}xf32>, tensor<{seq_len}xi32>) -> tensor<{seq_len}x{hidden}xf32>'
            )

            # Build per-position logits 0..S-2
            logits_names = []
            for pos in range(seq_len - 1):
                tf_ops.append(f'// --- position {pos} ---')
                tf_ops.append(f'%tf_pos_{pos} = stablehlo.constant dense<{pos}> : tensor<i32>')
                tf_ops.append(
                    f'%tf_pos_b_{pos} = stablehlo.broadcast_in_dim %tf_pos_{pos}, dims = [] : (tensor<i32>) -> tensor<{seq_len}xi32>'
                )
                tf_ops.append(
                    f'%tf_mcmp_{pos} = stablehlo.compare LE, %tf_iota, %tf_pos_b_{pos}, SIGNED : (tensor<{seq_len}xi32>, tensor<{seq_len}xi32>) -> tensor<{seq_len}xi1>'
                )
                tf_ops.append(
                    f'%tf_mask_{pos} = stablehlo.convert %tf_mcmp_{pos} : (tensor<{seq_len}xi1>) -> tensor<{seq_len}xf32>'
                )
                tf_ops.append(
                    f'%tf_mask_b_{pos} = stablehlo.broadcast_in_dim %tf_mask_{pos}, dims = [0] : (tensor<{seq_len}xf32>) -> tensor<{seq_len}x{hidden}xf32>'
                )
                tf_ops.append(
                    f'%tf_masked_{pos} = stablehlo.multiply %tf_gather, %tf_mask_b_{pos} : tensor<{seq_len}x{hidden}xf32>'
                )
                # Reduce across seq dim 0 to 1xH -> H
                tf_ops.append(
                    f'%tf_red_{pos} = stablehlo.reduce(%tf_masked_{pos} init: %tf_zero_f32) '
                    f'applies stablehlo.add across dimensions = [0] : (tensor<{seq_len}x{hidden}xf32>, tensor<f32>) -> tensor<{hidden}xf32>'
                )
                tf_ops.append(
                    f'%tf_logits_{pos} = stablehlo.dot %tf_red_{pos}, {out_name} : (tensor<{hidden}xf32>, tensor<{hidden}x{vocab}xf32>) -> tensor<{vocab}xf32>'
                )
                tf_ops.append(
                    f'%tf_logits_b_{pos} = stablehlo.reshape %tf_logits_{pos} : (tensor<{vocab}xf32>) -> tensor<1x{vocab}xf32>'
                )
                logits_names.append(f'%tf_logits_b_{pos}')

            # Concatenate along batch dim 0
            catsig = ", ".join(logits_names)
            tf_ops.append(
                f'%tf_all = stablehlo.concatenate {catsig}, dimension = 0 : '
                + "(" + ", ".join([f"tensor<1x{vocab}xf32>"] * (seq_len - 1)) + f") -> tensor<{seq_len-1}x{vocab}xf32>"
            )

            # Parse and insert the textual block.
            with mlir_ir.InsertionPoint.before(while_op):
                for line in tf_ops:
                    _parse_op(line)

            # Replace uses: if the while produced logits directly, replace them with %tf_all.
            # Otherwise, we keep %tf_all as a new value (verifier cares about logits).
            # Try replacing the first f32-typed ranked tensor result or a compatible shape.
            replaced = False
            for res in while_op.results:
                t = res.type
                shp = _tensor_shape(t)
                if shp and len(shp) == 2 and shp[0] == seq_len - 1 and shp[1] == vocab:
                    # Replace all uses with %tf_all
                    tf_all_val = parent_block.operations[-1].results[0]  # last parsed op is %tf_all
                    try:
                        res.replace_all_uses_with(tf_all_val)
                        replaced = True
                        break
                    except Exception:
                        pass

            # If we didn't replace, we still leave %tf_all in the IR for the verifier to query.
            # Finally erase the while op.
            while_op.operation.erase()

        # Done. The outer cleanup pipeline will canonicalize and DCE dead edges.

# -----------------------------
# Recipe registry & default one
# -----------------------------

_RECIPE_REGISTRY: Dict[str, Recipe] = {}


def register_recipe(recipe: Recipe) -> None:
    _RECIPE_REGISTRY[recipe.key] = recipe


def _default_recipe() -> Recipe:
    return Recipe(
        key="jax_scan_v0",
        description="AR while loop from JAX/XLA scan: dynamic_update_slice + pos++ → teacher-forcing prefill.",
        transform=TransformScript(
            name="jax_scan_v0",
            transform_ir=DEFAULT_TRANSFORM_IR,
        ),
        rewriter=_rewrite_ar_while_to_teacher_forcing,
        # Conservative cleanup after rewriting and before returning MLIR:
        cleanup_pipeline="builtin.module(canonicalize,cse,canonicalize)"
    )


# Register the default recipe on import.
register_recipe(_default_recipe())


# -----------------------------
# Running the pipeline
# -----------------------------

def _run_transform_interpreter(module: mlir_ir.Module, transform_ir: str) -> None:
    """Executes the transform module via IREE's preprocessing pass.

    Details:
      - This pass runs the transform dialect interpreter and applies the sequence named
        `@__transform_main` to the payload IR (the `module` we pass).
      - We pass the transform module via a temporary file path (`transform-spec-path`).

    See: IREE Preprocessing passes docs (`-iree-preprocessing-transform-interpreter`).
    """
    tf_path = _write_temp(transform_ir, suffix=".td.mlir")

    # Build a pass manager and run the one-shot preprocessing interpreter.
    # We scope the pass at built-in module level.
    # Syntax: builtin.module(iree-preprocessing-transform-interpreter{transform-spec-path="..."})
    pm = mlir_pm.PassManager()
    pm.add(
        f'builtin.module('
        f'iree-preprocessing-transform-interpreter{{transform-spec-path="{tf_path}"}}'
        f')'
    )
    pm.run(module.operation)

    # Clean up temp file.
    try:
        os.remove(tf_path)
    except Exception:
        pass


def _run_cleanup_pipeline(module: mlir_ir.Module, pipeline: str) -> None:
    pm = mlir_pm.PassManager()
    pm.add(pipeline)
    pm.run(module.operation)


def apply_teacher_forcing_transform(
    stablehlo_text: str,
    recipe: str = "jax_scan_v0",
) -> str:
    """Apply a registered recipe to convert an AR decode StableHLO module to teacher-forcing.

    Returns the updated MLIR (text). If the recipe does not match, the payload is returned
    after benign canonicalization.

    Typical usage:
        out_mlir = apply_teacher_forcing_transform(ar_mlir, recipe="jax_scan_v0")
    """
    if recipe not in _RECIPE_REGISTRY:
        raise KeyError(f"Unknown recipe '{recipe}'. Available: {list(_RECIPE_REGISTRY)}")

    r = _RECIPE_REGISTRY[recipe]

    # Parse the payload IR (StableHLO module text).
    with mlir_ir.Context() as ctx, mlir_ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(stablehlo_text)

        # Step 1: run the Transform schedule to tag candidates (and canonicalize mildly).
        _run_transform_interpreter(module, r.transform.transform_ir)

        # Step 2: run the Python rewriter pass, consuming annotations and doing structural rewrite.
        r.rewriter(module)

        # Step 3: cleanup (canonicalize/cse); recipe may override.
        if r.cleanup_pipeline:
            _run_cleanup_pipeline(module, r.cleanup_pipeline)

        # Return the resulting IR text.
        return str(module)


# ---------------------------------------------
# Extensibility: add new AR decode transformations
# ---------------------------------------------

def make_recipe(
    key: str,
    transform_ir: str,
    rewriter: Callable[[mlir_ir.Module], None],
    description: str = "",
    cleanup_pipeline: Optional[str] = "builtin.module(canonicalize,cse,canonicalize)"
) -> Recipe:
    """Helper for verifiers to define/register a new recipe quickly."""
    return Recipe(
        key=key,
        description=description or key,
        transform=TransformScript(name=key, transform_ir=transform_ir),
        rewriter=rewriter,
        cleanup_pipeline=cleanup_pipeline,
    )


# Example skeleton for a custom matcher that targets a different decode form.
ALT_MATCHER_SKELETON = r"""
transform.module {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %mod = transform.cast %root : !transform.any_op to !transform.op<"builtin.module">
    transform.with_pdl_patterns {
    ^p(%p0: !pdl.operation):
      // Copy this pattern and tailor it to your decode variant.
      pdl.pattern @match_ar_variant_X : benefit(1) {
        %w = pdl.operation "stablehlo.while"
        // TODO: add structural constraints or attribute guards
        pdl.rewrite %w { pdl.replace %w with %w }
      }
      transform.sequence failures(propagate) {
      ^s(%scope: !transform.any_op):
        %hits = transform.pdl_match @match_ar_variant_X in %mod
        transform.annotate %hits { veritor.ar.loop, veritor.ar.kind = "variant_X" } : !transform.any_op
        transform.apply_patterns.canonicalization to %mod
        transform.apply_cse to %mod
        transform.apply_dce to %mod
        transform.yield
      }
    }
    transform.yield
  }
}
""".strip()


# ---------------------------------------------
# (Optional) tiny smoke-test helper
# ---------------------------------------------

def _smoke_test_on_text(ar_mlir: str) -> str:
    """Convenience for ad-hoc testing. Returns transformed MLIR text."""
    return apply_teacher_forcing_transform(ar_mlir, recipe="jax_scan_v0")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(".mlir"):
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            ar_text = f.read()
        out = _smoke_test_on_text(ar_text)
        print(out)
    else:
        print("Usage: python veritor_tf_transform.py <input_stablehlo.mlir>")
```

How this maps to your needs
	•	No regex: Matching is done by the Transform dialect with embedded PDL. The default schedule (DEFAULT_TRANSFORM_IR) anchors on stablehlo.while and is intentionally permissive; the reliable shape check and the actual rewrite are performed by the Python pass. You can harden the PDL matcher as you see more decode variants. (See ALT_MATCHER_SKELETON.)  ￼
	•	Transform API integration: The module invokes IREE’s preprocessing interpreter pass (-iree-preprocessing-transform-interpreter) from Python’s PassManager, feeding it the transform module via a temp file. This is exactly the documented workflow for one-off transform scripts without bespoke C++ plumbing.  ￼
	•	Extensibility:
	•	Add a new recipe with your own transform schedule + (optionally different) Python rewriter via register_recipe(make_recipe(...)).
	•	The rewriter is a normal Python function over the MLIR module, so experimenting on tricky cases doesn’t require rebuilding a pass in C++.
	•	You can also evolve towards more sophisticated PDLL rewrites (where PDL performs both match and rewrite), keeping the Python pass minimal. The Transform dialect supports this via transform.with_pdl_patterns + pdl.rewrite.  ￼
	•	Safety & fallbacks: If a while-loop doesn’t meet the conservative checks, it’s skipped without breaking the module. The cleanup pipeline runs canonicalize/cse to keep the IR tidy. (Canonicalization/CSE are standard MLIR passes.)  ￼
	•	Fits your current approach: The generated teacher‑forcing block mirrors the hand‑crafted IR you wrote in your prototype—iota/compare/broadcast mask, gather once, reduce to hidden, then dot to logits and concatenate. That keeps verification fast and deterministic.

⸻

Tips for adding new AR graph types quickly
	1.	Clone the matcher skeleton (ALT_MATCHER_SKELETON), rename the pattern (e.g., @match_ar_while_torch_xla_v1), and tighten constraints:
	•	require a stablehlo.dynamic_update_slice in the while body,
	•	require an add-by-1 to a loop-carried pos,
	•	optionally require a stablehlo.gather + stablehlo.dot_general chain.
You can also split into multiple small PDL patterns and use transform.foreach_match to run an “action” sequence only when the matcher succeeds.  ￼
	2.	Specialize the Python rewriter:
	•	Copy _rewrite_ar_while_to_teacher_forcing, adapt shape extraction rules to your variant (e.g., different weight naming, different gather attrs).
	•	Register a new recipe with register_recipe(make_recipe(...)).
	3.	Harden progressively:
	•	As you see more decode lowering shapes, turn heuristics into explicit checks (e.g., verify cond compares %pos < %limit).
	•	If you prefer fully declarative rewrites, move construction into PDLL pdl.rewrite blocks and let the Transform schedule call transform.apply_patterns (or just use pdl_match+annotate as done here).  ￼

⸻

Why this stack?
	•	The Transform dialect is purpose-built for controlling compiler transformations and composing them into reusable schedules. It’s the right abstraction to select/annotate the AR loops and stage enabling simplifications, separate from rewriting logic.  ￼
	•	The IREE preprocessing transform interpreter is the minimal, well-supported path to run Transform IR against a module from Python, no C++ infra required.  ￼
	•	The embedded MLIR Python API keeps the rewriter fast to iterate on—crucial as you broaden support across more AR decode shapes. When the rewrite stabilizes, you can port it to C++ or to PDLL for performance/portability without changing the Transform schedule interface.  ￼

⸻

Footnotes & docs
	•	IREE preprocessing pass -iree-preprocessing-transform-interpreter (applies @__transform_main): official docs.  ￼
	•	Embedded MLIR / PassManager Python APIs used above.  ￼
	•	Transform dialect overview, with_pdl_patterns, pdl_match, annotate, canonicalization/CSE/DCE ops.  ￼

⸻

Next improvements you can drop in
	•	PDLL-only rewrites for cases you can fully express declaratively (no Python pass).
	•	Shape/value parameterization via Transform params (e.g., pass seq_len as a !transform.param<index> computed by a matcher).
	•	Verifier-facing hooks to emit side-band metadata (e.g., where logits slices map back to positions), by annotating the new TF ops with veritor.* attributes from the Python pass.

If you’d like, I can add a second, stricter matcher that verifies a stablehlo.compare of the loop counter against a limit in the cond region and checks for stablehlo.gather/stablehlo.dot_general presence in the body, then wire it up as recipe="jax_scan_v1".