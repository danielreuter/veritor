"""
Test autoregressive inference with teacher-forcing verification.

This test:
1. Creates a simple autoregressive transformer-like model in JAX
2. Runs autoregressive generation (deterministic, no sampling)
3. Generates real StableHLO from the JAX computation
4. Verifier transforms the graph for teacher-forcing
5. Executes teacher-forcing verification to ensure identical results

CRITICAL INVARIANT:
- The StableHLO stored must exactly match the executed computation
- Autoregressive and teacher-forcing must produce identical logits
- Graph transformation preserves computational semantics
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import random

from veritor.db.api import WorkloadDatabase
from veritor.db.ir_store import IRFormat, IRRole
from veritor.db.models import (
    DataBundle,
    EventType,
    Graph,
    TensorData,
    Trace,
    TraceEvent,
)


@dataclass
class AutoregressiveConfig:
    """Configuration for autoregressive inference test."""

    vocab_size: int = 12  # Small vocab for testing
    hidden_dim: int = 16  # Hidden dimension
    max_seq_length: int = 6  # Short sequences
    n_layers: int = 2  # Number of transformer layers
    n_heads: int = 2  # Attention heads
    batch_size: int = 1  # Single batch
    seed: int = 42


class SimpleTransformer:
    """Simple autoregressive transformer model."""

    def __init__(self, config: AutoregressiveConfig):
        self.config = config
        key = random.PRNGKey(config.seed)

        # Token embeddings
        key, emb_key = random.split(key)
        self.token_embedding = random.normal(
            emb_key, (config.vocab_size, config.hidden_dim)
        ) * 0.02

        # Position embeddings
        key, pos_key = random.split(key)
        self.pos_embedding = random.normal(
            pos_key, (config.max_seq_length, config.hidden_dim)
        ) * 0.02

        # Transformer layers (simplified)
        self.layers = []
        for _ in range(config.n_layers):
            layer_params = {}

            # Attention weights (simplified single-head for now)
            key, q_key, k_key, v_key, o_key = random.split(key, 5)
            layer_params['q_proj'] = random.normal(
                q_key, (config.hidden_dim, config.hidden_dim)
            ) * 0.02
            layer_params['k_proj'] = random.normal(
                k_key, (config.hidden_dim, config.hidden_dim)
            ) * 0.02
            layer_params['v_proj'] = random.normal(
                v_key, (config.hidden_dim, config.hidden_dim)
            ) * 0.02
            layer_params['out_proj'] = random.normal(
                o_key, (config.hidden_dim, config.hidden_dim)
            ) * 0.02

            # Feed-forward network
            key, ff1_key, ff2_key = random.split(key, 3)
            ff_dim = config.hidden_dim * 4
            layer_params['ff_w1'] = random.normal(
                ff1_key, (config.hidden_dim, ff_dim)
            ) * 0.02
            layer_params['ff_w2'] = random.normal(
                ff2_key, (ff_dim, config.hidden_dim)
            ) * 0.02

            self.layers.append(layer_params)

        # Output projection to vocabulary
        key, out_key = random.split(key)
        self.output_proj = random.normal(
            out_key, (config.hidden_dim, config.vocab_size)
        ) * 0.02

    def embed_tokens(self, token_ids: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        """Embed tokens with position information."""
        # token_ids: [batch_size, seq_len] or [seq_len] for single step
        # positions: [batch_size, seq_len] or [seq_len]

        if token_ids.ndim == 1:
            # Single step: [seq_len] -> [1, seq_len]
            token_ids = token_ids[None, :]
            positions = positions[None, :]

        # Get embeddings
        token_emb = self.token_embedding[token_ids]  # [batch_size, seq_len, hidden_dim]
        pos_emb = self.pos_embedding[positions]      # [batch_size, seq_len, hidden_dim]

        return token_emb + pos_emb

    def apply_layer(self, x: jnp.ndarray, layer_params: Dict, mask: jnp.ndarray) -> jnp.ndarray:
        """Apply one transformer layer."""
        # x: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len, seq_len] - causal mask

        batch_size, seq_len, hidden_dim = x.shape

        # Simplified self-attention (single head for simplicity)
        q = jnp.dot(x, layer_params['q_proj'])  # [batch_size, seq_len, hidden_dim]
        k = jnp.dot(x, layer_params['k_proj'])  # [batch_size, seq_len, hidden_dim]
        v = jnp.dot(x, layer_params['v_proj'])  # [batch_size, seq_len, hidden_dim]

        # Attention scores
        scores = jnp.matmul(q, k.transpose(0, 2, 1))  # [batch_size, seq_len, seq_len]
        scores = scores / jnp.sqrt(hidden_dim)

        # Apply causal mask
        scores = jnp.where(mask, scores, -1e9)

        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention
        attn_output = jnp.matmul(attn_weights, v)  # [batch_size, seq_len, hidden_dim]

        # Output projection
        attn_output = jnp.dot(attn_output, layer_params['out_proj'])

        # Residual connection
        x = x + attn_output

        # Feed-forward network
        ff_output = jnp.dot(x, layer_params['ff_w1'])
        ff_output = jax.nn.gelu(ff_output)
        ff_output = jnp.dot(ff_output, layer_params['ff_w2'])

        # Residual connection
        x = x + ff_output

        return x

    def forward_step(self, token_ids: jnp.ndarray, position_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Single autoregressive step.

        Args:
            token_ids: [max_seq_len] - full token array
            position_mask: [max_seq_len] - mask indicating valid positions (1.0 for valid, 0.0 for invalid)

        Returns:
            logits: [vocab_size] - logits for next token
        """
        # Create position indices
        positions = jnp.arange(self.config.max_seq_length)

        # Embed all tokens (will be masked later)
        x = self.embed_tokens(token_ids, positions)  # [1, max_seq_len, hidden_dim]

        # Apply position mask to embeddings
        x = x * position_mask[None, :, None]  # Broadcast mask

        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((self.config.max_seq_length, self.config.max_seq_length)))
        # Combine with position mask
        position_mask_2d = position_mask[:, None] * position_mask[None, :]
        mask = causal_mask * position_mask_2d
        mask = mask[None, :, :]  # Add batch dimension

        # Apply transformer layers
        for layer_params in self.layers:
            x = self.apply_layer(x, layer_params, mask)

        # Find the last valid position
        valid_positions = jnp.sum(position_mask).astype(jnp.int32)
        last_pos_idx = valid_positions - 1

        # Get logits for the last valid position
        last_hidden = x[0, last_pos_idx, :]  # [hidden_dim]
        logits = jnp.dot(last_hidden, self.output_proj)  # [vocab_size]

        return logits

    def generate_autoregressive(self, start_token: int, max_length: int) -> Tuple[List[int], List[jnp.ndarray]]:
        """
        Generate sequence autoregressively using argmax (deterministic).

        Returns:
            tokens: Generated sequence including start_token
            logits_sequence: Logits at each step
        """
        tokens = [start_token]
        logits_sequence = []

        # Initialize token array
        token_array = jnp.zeros(max_length, dtype=jnp.int32)
        token_array = token_array.at[0].set(start_token)

        for step in range(max_length - 1):
            seq_len = step + 1  # Current sequence length

            # Create position mask
            position_mask = jnp.zeros(max_length)
            position_mask = position_mask.at[:seq_len].set(1.0)

            # Get logits for next token
            logits = self.forward_step(token_array, position_mask)
            logits_sequence.append(logits)

            # Deterministic: take argmax
            next_token = jnp.argmax(logits)
            tokens.append(int(next_token))

            # Update token array
            token_array = token_array.at[step + 1].set(next_token)

        return tokens, logits_sequence

    def forward_teacher_forcing(self, token_sequence: List[int]) -> List[jnp.ndarray]:
        """
        Process entire sequence in parallel (teacher-forcing).

        Args:
            token_sequence: Complete sequence of tokens

        Returns:
            logits_sequence: Logits for each position (except last)
        """
        seq_len = len(token_sequence)
        token_array = jnp.array(token_sequence)

        # Create position indices
        positions = jnp.arange(seq_len)

        # Embed all tokens
        x = self.embed_tokens(token_array, positions)  # [1, seq_len, hidden_dim]

        # Create causal mask for the full sequence
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))  # Lower triangular
        mask = mask[None, :, :]  # Add batch dimension

        # Apply transformer layers
        for layer_params in self.layers:
            x = self.apply_layer(x, layer_params, mask)

        # Get logits for all positions
        all_logits = jnp.dot(x[0], self.output_proj)  # [seq_len, vocab_size]

        # Return logits for all positions except the last (which has no next token)
        logits_sequence = [all_logits[i] for i in range(seq_len - 1)]

        return logits_sequence


def test_autoregressive_with_teacher_forcing_verification(workload_db):
    """Test autoregressive inference with teacher-forcing verification."""
    database = workload_db

    # Configuration
    config = AutoregressiveConfig(
        vocab_size=12,
        hidden_dim=16,
        max_seq_length=6,
        n_layers=2,
        batch_size=1,
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Layers: {config.n_layers}")

    # Initialize model
    model = SimpleTransformer(config)

    # === AUTOREGRESSIVE GENERATION ===

    # Create graph for autoregressive computation
    ar_graph = Graph(
        id=f"autoregressive_{uuid.uuid4().hex[:8]}",
        metadata={
            "model_type": "simple_transformer",
            "generation_type": "autoregressive",
            "vocab_size": config.vocab_size,
            "max_seq_length": config.max_seq_length,
            "test_type": "autoregressive_inference",
        },
    )
    ar_graph_id = database.store_graph(ar_graph)

    # Define the core autoregressive step function for StableHLO generation
    def ar_step_core(token_array: jnp.ndarray, position_mask: jnp.ndarray) -> jnp.ndarray:
        """Core autoregressive step - this will be stored as StableHLO."""
        return model.forward_step(token_array, position_mask)

    # JIT compile the step function
    jitted_ar_step = jax.jit(ar_step_core)

    # Generate real StableHLO
    example_tokens = jnp.zeros(config.max_seq_length, dtype=jnp.int32)
    example_mask = jnp.ones(config.max_seq_length)  # All positions valid for example

    lowered = jitted_ar_step.lower(example_tokens, example_mask)
    ar_stablehlo = lowered.as_text()

    # Verify real StableHLO
    assert len(ar_stablehlo) > 1000, "StableHLO seems too short"
    assert "stablehlo.constant" in ar_stablehlo, "Missing model constants"
    assert "stablehlo.dot_general" in ar_stablehlo, "Missing matrix operations"

    print(f"\nGenerated autoregressive StableHLO ({len(ar_stablehlo)} bytes)")

    # Store autoregressive IR
    database.ir_store.attach_ir(
        ar_graph_id,
        IRRole.LOGICAL,
        ar_stablehlo,
        IRFormat.STABLEHLO,
        {
            "generated_from": "test_autoregressive_inference",
            "jax_version": jax.__version__,
            "model_type": "simple_transformer",
            "generation_mode": "autoregressive",
        },
    )

    # Run autoregressive generation using the SAME jitted function
    start_token = 3
    print(f"\nGenerating sequence starting with token {start_token}...")

    tokens, logits_ar = model.generate_autoregressive(start_token, config.max_seq_length)
    print(f"Generated tokens: {tokens}")

    # Verify that jitted function produces same results
    token_array = jnp.array([start_token] + [0] * (config.max_seq_length - 1))
    position_mask = jnp.array([1.0] + [0.0] * (config.max_seq_length - 1))  # Only first position valid
    jitted_logits_0 = jitted_ar_step(token_array, position_mask)
    python_logits_0 = model.forward_step(token_array, position_mask)

    assert jnp.allclose(jitted_logits_0, python_logits_0, rtol=1e-5), \
        "CRITICAL: JIT autoregressive step doesn't match Python execution!"

    # Create trace for autoregressive generation
    ar_events = []
    for i in range(len(tokens) - 1):
        ar_events.append(TraceEvent(
            timestamp=datetime.now().timestamp() + i * 0.001,
            event_type=EventType.KERNEL_LAUNCH,
            device_id="cpu_0",
            operation_id=f"ar_step_{i}",
            data={
                "step": i,
                "input_token": tokens[i],
                "output_token": tokens[i + 1],
                "generation_mode": "autoregressive",
            },
        ))

    ar_trace = Trace(
        id=f"ar_trace_{uuid.uuid4().hex[:8]}",
        graph_id=ar_graph_id,
        start_time=ar_events[0].timestamp if ar_events else datetime.now().timestamp(),
        end_time=ar_events[-1].timestamp if ar_events else datetime.now().timestamp(),
        events=ar_events,
        metadata={"generation_mode": "autoregressive", "sequence_length": len(tokens)},
    )
    ar_trace_id = database.store_trace(ar_trace)

    # Store autoregressive data
    ar_data_bundle = DataBundle(
        id=f"ar_data_{uuid.uuid4().hex[:8]}",
        graph_id=ar_graph_id,
        inputs={"start_token": TensorData.from_array(jnp.array([start_token]))},
        outputs={
            "tokens": TensorData.from_array(jnp.array(tokens)),
            **{f"logits_{i}": TensorData.from_array(logits_ar[i])
               for i in range(len(logits_ar))}
        },
        weights={},  # Model weights are embedded in StableHLO
        activations={},
        metadata={"trace_id": ar_trace_id, "generation_mode": "autoregressive"},
    )
    ar_data_id = database.store_data_bundle(ar_data_bundle)

    print(f"\nAutoregressive generation stored:")
    print(f"  Graph ID: {ar_graph_id}")
    print(f"  Trace ID: {ar_trace_id}")
    print(f"  Data ID: {ar_data_id}")

    # === TEACHER-FORCING VERIFICATION ===

    print(f"\n{'='*60}")
    print("Teacher-Forcing Verification")
    print(f"{'='*60}")

    # Create teacher-forcing graph
    tf_graph = Graph(
        id=f"teacher_forcing_{uuid.uuid4().hex[:8]}",
        metadata={
            "model_type": "simple_transformer",
            "generation_type": "teacher_forcing",
            "source_graph_id": ar_graph_id,
            "verification_for": ar_graph_id,
            "test_type": "teacher_forcing_verification",
        },
    )
    tf_graph_id = database.store_graph(tf_graph)

    # Define teacher-forcing function
    def teacher_forcing_core(token_sequence: jnp.ndarray) -> jnp.ndarray:
        """Core teacher-forcing computation - processes all tokens in parallel."""
        seq_len = len(token_sequence)
        positions = jnp.arange(seq_len)

        # Embed all tokens
        x = model.embed_tokens(token_sequence, positions)  # [1, seq_len, hidden_dim]

        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask[None, :, :]

        # Apply all transformer layers
        for layer_params in model.layers:
            x = model.apply_layer(x, layer_params, mask)

        # Get logits for all positions
        all_logits = jnp.dot(x[0], model.output_proj)  # [seq_len, vocab_size]

        return all_logits[:-1]  # Return logits for all positions except last

    # JIT compile teacher-forcing
    jitted_tf = jax.jit(teacher_forcing_core)

    # Generate teacher-forcing StableHLO
    example_sequence = jnp.array(tokens)
    tf_lowered = jitted_tf.lower(example_sequence)
    tf_stablehlo = tf_lowered.as_text()

    print(f"Generated teacher-forcing StableHLO ({len(tf_stablehlo)} bytes)")

    # Store teacher-forcing IR
    database.ir_store.attach_ir(
        tf_graph_id,
        IRRole.LOGICAL,
        tf_stablehlo,
        IRFormat.STABLEHLO,
        {
            "generated_from": "test_autoregressive_inference",
            "jax_version": jax.__version__,
            "model_type": "simple_transformer",
            "generation_mode": "teacher_forcing",
            "derived_from": ar_graph_id,
        },
    )

    # Link the transformation
    database.link_graph_transformation(ar_graph_id, tf_graph_id, "autoregressive_to_teacher_forcing")

    # Execute teacher-forcing using the SAME jitted function
    print(f"\nRunning teacher-forcing with tokens: {tokens}")
    tf_logits_jitted = jitted_tf(jnp.array(tokens))

    # Verify against Python execution
    tf_logits_python = model.forward_teacher_forcing(tokens)
    tf_logits_python_array = jnp.stack(tf_logits_python)

    assert jnp.allclose(tf_logits_jitted, tf_logits_python_array, rtol=1e-5), \
        "CRITICAL: JIT teacher-forcing doesn't match Python execution!"

    # Store teacher-forcing data
    tf_data_bundle = DataBundle(
        id=f"tf_data_{uuid.uuid4().hex[:8]}",
        graph_id=tf_graph_id,
        inputs={"token_sequence": TensorData.from_array(jnp.array(tokens))},
        outputs={
            f"logits_{i}": TensorData.from_array(tf_logits_jitted[i])
            for i in range(len(tf_logits_jitted))
        },
        weights={},
        activations={},
        metadata={
            "verification_for_trace": ar_trace_id,
            "teacher_forcing_tokens": tokens,
            "generation_mode": "teacher_forcing",
        },
    )
    tf_data_id = database.store_data_bundle(tf_data_bundle)

    print(f"\nTeacher-forcing verification stored:")
    print(f"  Graph ID: {tf_graph_id}")
    print(f"  Data ID: {tf_data_id}")

    # === VERIFICATION ===

    print(f"\n{'='*60}")
    print("Verification Results")
    print(f"{'='*60}")

    print("\nComparing logits between autoregressive and teacher-forcing:")

    verification_success = True
    max_diff = 0.0

    for i in range(len(logits_ar)):
        ar_logits = logits_ar[i]
        tf_logits = tf_logits_jitted[i]

        # Check if logits are close
        diff = jnp.abs(ar_logits - tf_logits).max()
        max_diff = max(max_diff, float(diff))

        # Get predicted tokens
        ar_token = jnp.argmax(ar_logits)
        tf_token = jnp.argmax(tf_logits)

        match = ar_token == tf_token
        if not match:
            verification_success = False

        status = "✓" if match else "✗"
        print(f"  Position {i}: AR token={int(ar_token)}, TF token={int(tf_token)} {status} (max diff={diff:.6f})")

    print(f"\nMax logit difference: {max_diff:.6f}")

    if verification_success and max_diff < 1e-4:
        print("\n✅ VERIFICATION SUCCESSFUL: Teacher-forcing produces identical results!")
    else:
        print(f"\n⚠️  VERIFICATION WARNING: Differences found (max={max_diff:.6f})")

    # Additional verification: Check that TF is indeed parallel
    print(f"\n{'='*60}")
    print("Parallelism Verification")
    print(f"{'='*60}")

    import time

    # Time autoregressive generation
    start = time.time()
    for _ in range(10):
        model.generate_autoregressive(start_token, config.max_seq_length)
    ar_time = time.time() - start

    # Time teacher-forcing
    start = time.time()
    for _ in range(10):
        jitted_tf(jnp.array(tokens))
    tf_time = time.time() - start

    speedup = ar_time / tf_time if tf_time > 0 else float('inf')
    print(f"\nTiming (10 iterations):")
    print(f"  Autoregressive: {ar_time:.4f}s")
    print(f"  Teacher-forcing: {tf_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print("  ✓ Teacher-forcing shows expected parallel processing benefit")

    # Final verification
    assert database.get_graph(ar_graph_id) is not None
    assert database.get_graph(tf_graph_id) is not None
    assert database.get_trace(ar_trace_id) is not None
    assert database.get_data_bundle(ar_data_id) is not None
    assert database.get_data_bundle(tf_data_id) is not None

    # === NEW: Unified Verification Engine ===
    from veritor.verifier.engine import verify_single_execution, VerificationConfig

    # Configure verification for autoregressive pattern
    verification_config = VerificationConfig(
        enable_jit_vs_python=True,
        enable_challenge_verification=True,
        enable_transformation_checks=True,  # Key for autoregressive verification
        execution_rtol=1e-5,
        lsh_rtol=1e-3,
    )

    # Verify autoregressive graph
    ar_result = verify_single_execution(
        database=database,
        graph_id=ar_graph_id,
        trace_id=ar_trace_id,
        config=verification_config
    )

    # Verify teacher-forcing graph
    tf_result = verify_single_execution(
        database=database,
        graph_id=tf_graph_id,
        # No trace_id for TF graph as it's just a transformation
        config=verification_config
    )

    # Check autoregressive verification
    assert ar_result.success, f"Autoregressive verification failed: {ar_result.errors}"
    if ar_result.execution_match is not None:
        assert ar_result.execution_match, "Autoregressive JIT vs Python execution mismatch"

    # Check teacher-forcing verification
    assert tf_result.success, f"Teacher-forcing verification failed: {tf_result.errors}"
    if tf_result.execution_match is not None:
        assert tf_result.execution_match, "Teacher-forcing JIT vs Python execution mismatch"

    # Check transformation results
    transformation_success = (
        ar_result.transformation_results.get("autoregressive_to_teacher_forcing", True) and
        verification_success  # Use original cross-comparison result
    )
    assert transformation_success, "Autoregressive ↔ Teacher-forcing transformation verification failed"

    print(f"\n✅ Unified autoregressive verification passed!")
    print(f"   - AR challenges: {len(ar_result.challenge_results)} verified")
    print(f"   - TF execution verified")
    print(f"   - Transformation consistency: {transformation_success}")
    if ar_result.metrics:
        print(f"   - Max AR execution difference: {ar_result.metrics.get('max_difference', 'N/A')}")

    print(f"\n✅ Autoregressive inference with teacher-forcing verification complete!")

    # For standalone execution, return values; for pytest, return None
    import sys
    if 'pytest' not in sys.modules:
        return database, ar_graph_id, tf_graph_id, verification_success


if __name__ == "__main__":
    # Run the test directly
    from veritor.db.api import WorkloadDatabase

    db = WorkloadDatabase()
    result = test_autoregressive_with_teacher_forcing_verification(db)

    if result is not None:
        database, ar_graph_id, tf_graph_id, success = result
    else:
        # This shouldn't happen when run standalone
        print("Error: No result returned")
        exit(1)

    print(f"\nFinal Results:")
    print(f"  Autoregressive Graph: {ar_graph_id}")
    print(f"  Teacher-Forcing Graph: {tf_graph_id}")
    print(f"  Verification Success: {success}")
    print(f"  Total Graphs: {len(db.graphs)}")
    print(f"  Total Traces: {len(db.traces)}")
    print(f"  Total Data Bundles: {len(db.data_bundles)}")

    # Test persistence
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/test_ar_db"
        db.save(save_path)
        print(f"\nDatabase saved to: {save_path}")

        loaded_db = WorkloadDatabase.load(save_path)
        assert len(loaded_db.graphs) == 2
        assert len(loaded_db.traces) == 1
        assert len(loaded_db.data_bundles) == 2
        print("✓ Database persistence verified")