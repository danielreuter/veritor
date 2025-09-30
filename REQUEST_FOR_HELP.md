# Request for Help: AR→TF Graph Surgery with JAX Function Outlining

## Problem Statement

We need to transform autoregressive (AR) inference graphs into teacher-forcing (TF) graphs for efficient verification in the STAMP protocol. The transformation must work on real JAX-generated StableHLO, but JAX optimizes the code in a way that breaks our pattern matching.

## The Core Issue

**JAX outlines while loop bodies into separate functions**, making it impossible to detect AR patterns using direct graph analysis.

### What We Expected
```mlir
stablehlo.while(...) {
  // Condition region
}, {
  // Body region with these operations directly inside:
  - stablehlo.gather (embedding lookup)
  - stablehlo.dot (projection)
  - stablehlo.dynamic_update_slice (token buffer update)
  - stablehlo.argmax (next token selection)
}
```

### What JAX Actually Generates
```mlir
stablehlo.while(...) {
  // Condition region
}, {
  // Body just calls an outlined function:
  func.call @outlined_function_xyz(...)
  stablehlo.dynamic_update_slice(...)  // Only this remains
}

// Elsewhere in the module:
func.func private @outlined_function_xyz(...) {
  // The actual AR operations are here:
  stablehlo.gather(...)
  stablehlo.dot(...)
  // etc.
}
```

## Current Implementation Status

### What Works
- `/src/veritor/veritor_tf_transform_simple.py` - Simplified version that blindly assumes any while loop is AR (not robust)
- `/tests/test_graph_surgery.py` - All 9 tests pass with the simplified version

### What Doesn't Work
- `/src/veritor/veritor_tf_transform.py` - Full implementation using IREE Transform dialect
  - Lines 226-348: `_analyze_ar_while_decode_v1()` looks for gather/dot ops in while body
  - Fails because these ops are in the outlined function, not the while body
  - The Transform dialect patterns correctly tag while loops, but analysis fails

### Evidence of the Problem
When analyzing real JAX output (`/tmp/test_ar.mlir`):
```python
# From debugging output:
Found while loop #1
  Regions: 2
  ✗ Analysis failed - checking why...
    Operations in body: {'stablehlo.return', 'stablehlo.dynamic_update_slice',
                         'stablehlo.constant', 'stablehlo.add',
                         'func.call', 'stablehlo.broadcast_in_dim'}
    Has dynamic_update_slice: True
    Has gather: False  # ← Problem: it's in the called function
    Has dot/dot_general: False  # ← Problem: it's in the called function
```

## Specific Technical Requirements

### Input
- StableHLO module text with JAX-outlined functions
- Example at `/tmp/test_ar.mlir` (generate with test code in problem description)

### Required Analysis Algorithm
1. Find `stablehlo.while` operations
2. When while body contains `func.call`:
   - Extract the called function name
   - Look up that function in the module
   - Analyze the called function for AR patterns
3. Extract shapes and weights from the called function
4. Validate it's actually an AR pattern (has gather, dot, dynamic_update_slice)

### Output
- Teacher-forcing function that unrolls the computation
- Must work with IREE's MLIR infrastructure
- Should integrate with existing `ARLoopInfo` dataclass (line 187-209)

## Key Files to Understand

1. **Main Implementation**: `/src/veritor/veritor_tf_transform.py`
   - `_analyze_ar_while_decode_v1()` (lines 226-348) - Needs to follow func.call
   - `ARLoopInfo` dataclass (lines 187-209) - Stores analysis results
   - `_emit_tf_function()` (lines 351-587) - Generates TF function (works fine)

2. **Test File**: `/tests/test_graph_surgery.py`
   - `create_ar_model()` (lines 44-79) - Generates test AR model
   - `test_ar_to_tf_transformation()` (lines 81-115) - Main test

3. **Reference Design**: `/Users/danielreuter/projects/veritor/GRAPH_SURGERY.md`
   - Complete specification from original AI
   - Assumes direct access to ops (doesn't handle outlining)

## Solution Approach Needed

### Option 1: Enhanced Analysis (Preferred)
Modify `_analyze_ar_while_decode_v1()` to:
```python
def _analyze_ar_while_decode_v1(while_op):
    # ... existing checks ...

    # NEW: If body has func.call, analyze the called function
    for op in _walk_ops(body):
        if op.name == "func.call":
            callee = op.attributes["callee"]  # Get function name
            called_func = _find_function_in_module(module, callee)
            if called_func:
                # Analyze called_func for AR patterns
                # Extract embed_const, output_const from there
                # ...
```

### Option 2: Inline Functions First
Pre-process the module to inline outlined functions back into while bodies before analysis. This might be cleaner but requires MLIR pass manipulation.

### Option 3: Different Pattern Matching
Use IREE Transform dialect patterns that can match across function boundaries (if possible).

## Test Case to Validate Solution

```python
# Generate test AR model
import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(42)
embed = random.normal(key, (10, 8)) * 0.1
output = random.normal(random.split(key)[1], (8, 10)) * 0.1

def ar_generate(start_token):
    tokens = jnp.zeros(5, dtype=jnp.int32)
    tokens = tokens.at[0].set(start_token)

    def loop_body(carry, _):
        tokens, pos = carry
        mask = (jnp.arange(5) <= pos).astype(jnp.float32)
        x = embed[tokens]
        x = x * mask[:, None]
        x = jnp.sum(x, axis=0)
        logits = jnp.dot(x, output)
        next_token = jnp.argmax(logits)
        new_tokens = tokens.at[pos + 1].set(next_token)
        return (new_tokens, pos + 1), logits

    (final_tokens, _), all_logits = jax.lax.scan(
        loop_body, (tokens, 0), xs=None, length=4
    )
    return final_tokens, all_logits

# Lower to StableHLO
jitted = jax.jit(ar_generate)
lowered = jitted.lower(3)
ar_stablehlo = lowered.as_text()

# This generates StableHLO with outlined functions
# Save to /tmp/test_ar.mlir for testing
```

## Success Criteria

1. `_analyze_ar_while_decode_v1()` successfully analyzes JAX-generated AR models
2. Correctly extracts seq_len, vocab, hidden dimensions from outlined functions
3. Finds embedding and output weight constants even when in called functions
4. All tests in `test_graph_surgery.py` pass with the full implementation
5. Can transform real JAX AR models, not just hand-crafted ones

## Context About the Project

This is part of Veritor, implementing the STAMP (Scalable Trustless Attestation of Model Production) protocol for verifying ML workloads. The AR→TF transformation is critical for efficient verification of autoregressive inference, converting sequential token generation into parallel teacher-forcing for ~100x speedup.

## Why This Matters

Without this transformation working on real JAX output:
- Can't verify autoregressive model inference efficiently
- Verification would take 100x longer (sequential vs parallel)
- The STAMP protocol's inference verification is incomplete

The solution needs to be robust enough to handle various JAX optimization patterns while maintaining the correctness guarantees needed for cryptographic verification.