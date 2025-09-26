#!/usr/bin/env python3
"""
Test proper autoregressive to teacher-forcing transformation.

This demonstrates the correct approach for STAMP protocol:
1. Store the AR step function as StableHLO
2. Transform it at verification time to teacher-forcing
3. Execute both and compare
"""

import jax
import jax.numpy as jnp
from jax import random

from veritor.db.api import WorkloadDatabase
from veritor.db.models import Graph, DataBundle, TensorData
from veritor.db.ir_store import IRFormat, IRRole
from veritor.verifier.ar_transformer import (
    create_runtime_teacher_forcing_executor,
    extract_step_function_from_loop,
)


def test_proper_ar_to_tf_transformation():
    """Test the proper AR->TF transformation approach."""
    print("=" * 80)
    print("PROPER AR->TF TRANSFORMATION TEST")
    print("=" * 80)

    # Create database
    db = WorkloadDatabase()

    # Simple model for testing
    class SimpleARModel:
        def __init__(self):
            key = random.PRNGKey(42)
            self.embed = random.normal(key, (10, 8)) * 0.1
            self.output = random.normal(random.split(key)[1], (8, 10)) * 0.1

        def step(self, tokens, mask):
            """Single AR step."""
            x = self.embed[tokens]
            x = x * mask[:, None]
            x = jnp.sum(x, axis=0)
            return jnp.dot(x, self.output)

    model = SimpleARModel()
    seq_len = 5

    # === PHASE 1: Store AR Step Function ===
    print("\n1. Storing AR step function as StableHLO...")

    # JIT compile the step function
    jitted_step = jax.jit(model.step)

    # Get StableHLO for the step
    example_tokens = jnp.zeros(seq_len, dtype=jnp.int32)
    example_mask = jnp.ones(seq_len)
    lowered = jitted_step.lower(example_tokens, example_mask)
    ar_step_hlo = lowered.as_text()

    print(f"   AR step HLO size: {len(ar_step_hlo)} bytes")

    # Store as graph
    ar_graph = Graph(
        id="ar_step_graph",
        metadata={
            "type": "autoregressive_step",
            "sequence_length": seq_len,
            "vocab_size": 10,
        }
    )
    ar_graph_id = db.store_graph(ar_graph)

    # Store the HLO
    db.ir_store.attach_ir(
        ar_graph_id,
        IRRole.LOGICAL,
        ar_step_hlo,
        IRFormat.STABLEHLO,
        {"mode": "autoregressive_step"}
    )

    # === PHASE 2: Run AR Generation ===
    print("\n2. Running AR generation...")

    tokens = [3]  # Start token
    all_logits = []

    for pos in range(seq_len - 1):
        token_array = jnp.array(tokens + [0] * (seq_len - len(tokens)), dtype=jnp.int32)
        mask = (jnp.arange(seq_len) <= pos).astype(jnp.float32)

        logits = jitted_step(token_array, mask)
        all_logits.append(logits)

        next_token = int(jnp.argmax(logits))
        tokens.append(next_token)

    print(f"   Generated tokens: {tokens}")

    # Store AR results
    ar_data = DataBundle(
        id="ar_execution",
        graph_id=ar_graph_id,
        inputs={"start_token": TensorData.from_array(jnp.array([3]))},
        outputs={
            "tokens": TensorData.from_array(jnp.array(tokens)),
            **{f"logits_{i}": TensorData.from_array(logits) for i, logits in enumerate(all_logits)}
        },
        weights={},
        activations={},
        metadata={"mode": "autoregressive"}
    )
    db.store_data_bundle(ar_data)

    # === PHASE 3: Transform to Teacher-Forcing at Verification Time ===
    print("\n3. Transforming to teacher-forcing at verification time...")

    # Analyze the stored HLO
    metadata = extract_step_function_from_loop(ar_step_hlo)
    print(f"   HLO metadata: {metadata}")

    # Create runtime TF executor
    tf_executor = create_runtime_teacher_forcing_executor(jitted_step, seq_len)

    # === PHASE 4: Execute Teacher-Forcing ===
    print("\n4. Executing teacher-forcing with same tokens...")

    tf_tokens = jnp.array(tokens, dtype=jnp.int32)
    tf_logits = tf_executor(tf_tokens)

    print(f"   TF logits shape: {tf_logits.shape}")

    # === PHASE 5: Compare Results ===
    print("\n5. Comparing AR and TF results...")

    all_match = True
    for i in range(len(all_logits)):
        ar_logit = all_logits[i]
        tf_logit = tf_logits[i]

        match = jnp.allclose(ar_logit, tf_logit, rtol=1e-5)
        max_diff = jnp.max(jnp.abs(ar_logit - tf_logit))

        status = "✓" if match else "✗"
        print(f"   Position {i}: {status} (max diff: {max_diff:.6f})")

        if not match:
            all_match = False

    # === PHASE 6: Demonstrate the Key Insight ===
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)

    print("""
The proper AR->TF transformation for STAMP protocol:

1. STORAGE PHASE (Prover):
   - Store the AR step function as StableHLO
   - Store execution traces with generated tokens and logits
   - No need to store a while loop - just the step function

2. VERIFICATION PHASE (Verifier):
   - Retrieve the AR step function HLO
   - Create a TF executor at runtime that calls the step function
   - Use teacher-forcing with the prover's tokens
   - Compare logits position by position

3. WHY THIS WORKS:
   - The step function is deterministic given inputs
   - TF just calls it with different masking patterns
   - No need for complex HLO graph surgery
   - Runtime transformation is simpler and more reliable

4. WHAT WE AVOID:
   - Complex while loop unrolling in HLO
   - Graph rewriting at the IR level
   - Dealing with control flow transformation
   - Maintaining two separate model implementations
""")

    if all_match:
        print("✅ VERIFICATION SUCCESS: AR and TF produce identical logits!")
    else:
        print("❌ VERIFICATION FAILED: Logits don't match")

    return all_match


if __name__ == "__main__":
    success = test_proper_ar_to_tf_transformation()
    exit(0 if success else 1)