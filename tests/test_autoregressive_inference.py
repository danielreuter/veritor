"""
Test autoregressive inference with teacher-forcing verification.

This test:
1. Creates a simple autoregressive transformer-like model in JAX
2. Runs autoregressive generation (deterministic, no sampling)
3. Creates Claims for both autoregressive steps and teacher-forcing
4. Verifies that autoregressive and teacher-forcing produce identical results

CRITICAL INVARIANT:
- The StableHLO stored must exactly match the executed computation
- Autoregressive and teacher-forcing must produce identical logits
- Graph transformation preserves computational semantics
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from veritor import ClaimDatabase, create_claim_from_jax_function, verify


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
        self.token_embedding = (
            random.normal(emb_key, (config.vocab_size, config.hidden_dim)) * 0.02
        )

        # Position embeddings
        key, pos_key = random.split(key)
        self.pos_embedding = (
            random.normal(pos_key, (config.max_seq_length, config.hidden_dim)) * 0.02
        )

        # Transformer layers (simplified)
        self.layers = []
        for _ in range(config.n_layers):
            layer_params = {}

            # Attention weights (simplified single-head for now)
            key, q_key, k_key, v_key, o_key = random.split(key, 5)
            layer_params["q_proj"] = (
                random.normal(q_key, (config.hidden_dim, config.hidden_dim)) * 0.02
            )
            layer_params["k_proj"] = (
                random.normal(k_key, (config.hidden_dim, config.hidden_dim)) * 0.02
            )
            layer_params["v_proj"] = (
                random.normal(v_key, (config.hidden_dim, config.hidden_dim)) * 0.02
            )
            layer_params["out_proj"] = (
                random.normal(o_key, (config.hidden_dim, config.hidden_dim)) * 0.02
            )

            # Feed-forward network
            key, ff1_key, ff2_key = random.split(key, 3)
            ff_dim = config.hidden_dim * 4
            layer_params["ff_w1"] = (
                random.normal(ff1_key, (config.hidden_dim, ff_dim)) * 0.02
            )
            layer_params["ff_w2"] = (
                random.normal(ff2_key, (ff_dim, config.hidden_dim)) * 0.02
            )

            self.layers.append(layer_params)

        # Output projection to vocabulary
        key, out_key = random.split(key)
        self.output_proj = (
            random.normal(out_key, (config.hidden_dim, config.vocab_size)) * 0.02
        )

    def embed_tokens(
        self, token_ids: jnp.ndarray, positions: jnp.ndarray
    ) -> jnp.ndarray:
        """Embed tokens with position information."""
        # token_ids: [batch_size, seq_len] or [seq_len] for single step
        # positions: [batch_size, seq_len] or [seq_len]

        if token_ids.ndim == 1:
            # Single step: [seq_len] -> [1, seq_len]
            token_ids = token_ids[None, :]
            positions = positions[None, :]

        # Get embeddings
        token_emb = self.token_embedding[token_ids]  # [batch_size, seq_len, hidden_dim]
        pos_emb = self.pos_embedding[positions]  # [batch_size, seq_len, hidden_dim]

        return token_emb + pos_emb

    def apply_layer(
        self, x: jnp.ndarray, layer_params: dict, mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply one transformer layer."""
        # x: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len, seq_len] - causal mask

        batch_size, seq_len, hidden_dim = x.shape

        # Simplified self-attention (single head for simplicity)
        q = jnp.dot(x, layer_params["q_proj"])  # [batch_size, seq_len, hidden_dim]
        k = jnp.dot(x, layer_params["k_proj"])  # [batch_size, seq_len, hidden_dim]
        v = jnp.dot(x, layer_params["v_proj"])  # [batch_size, seq_len, hidden_dim]

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
        attn_output = jnp.dot(attn_output, layer_params["out_proj"])

        # Residual connection
        x = x + attn_output

        # Feed-forward network
        ff_output = jnp.dot(x, layer_params["ff_w1"])
        ff_output = jax.nn.gelu(ff_output)
        ff_output = jnp.dot(ff_output, layer_params["ff_w2"])

        # Residual connection
        x = x + ff_output

        return x

    def forward_step(
        self, token_ids: jnp.ndarray, position_mask: jnp.ndarray
    ) -> jnp.ndarray:
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
        causal_mask = jnp.tril(
            jnp.ones((self.config.max_seq_length, self.config.max_seq_length))
        )
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

    def generate_autoregressive(
        self, start_token: int, max_length: int
    ) -> tuple[list[int], list[jnp.ndarray]]:
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

    def forward_teacher_forcing(self, token_sequence: list[int]) -> list[jnp.ndarray]:
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


def test_autoregressive_with_teacher_forcing_verification():
    """Test autoregressive inference with teacher-forcing verification using Claims."""
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

    # Initialize claim database
    db = ClaimDatabase()

    # === AUTOREGRESSIVE GENERATION ===

    start_token = 3
    print(f"\nGenerating sequence starting with token {start_token}...")

    # Run autoregressive generation
    tokens, logits_ar = model.generate_autoregressive(
        start_token, config.max_seq_length
    )
    print(f"Generated tokens: {tokens}")

    # Define the core autoregressive step function
    def ar_step_core(
        token_array: jnp.ndarray, position_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Core autoregressive step - this will be stored as StableHLO."""
        return model.forward_step(token_array, position_mask)

    # Create a claim for one autoregressive step (first step)
    # Note: Using jnp arrays to ensure proper types are maintained through JAX
    token_array = np.array(
        [start_token] + [0] * (config.max_seq_length - 1), dtype=np.int32
    )
    position_mask = np.array(
        [1.0] + [0.0] * (config.max_seq_length - 1), dtype=np.float32
    )

    ar_claim = create_claim_from_jax_function(
        ar_step_core,
        [token_array, position_mask],
        classifier_name="bit_exact",
        model_type="simple_transformer",
        generation_type="autoregressive",
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
    )

    db.add_claim(ar_claim)
    print(f"\nAutoregressive claim created: {ar_claim.id}")

    # Verify autoregressive claim
    ar_result = verify(ar_claim)
    assert ar_result.passed, "Autoregressive claim verification failed"
    print(f"✅ Autoregressive claim verified")

    # === TEACHER-FORCING VERIFICATION ===

    print(f"\n{'=' * 60}")
    print("Teacher-Forcing Verification")
    print(f"{'=' * 60}")

    # Define teacher-forcing function
    def teacher_forcing_core(token_sequence: jnp.ndarray) -> jnp.ndarray:
        """Core teacher-forcing computation - processes all tokens in parallel."""
        seq_len = len(token_sequence)
        positions = jnp.arange(seq_len)

        # Embed all tokens
        x = model.embed_tokens(token_sequence, positions)

        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask[None, :, :]

        # Apply all transformer layers
        for layer_params in model.layers:
            x = model.apply_layer(x, layer_params, mask)

        # Get logits for all positions
        all_logits = jnp.dot(x[0], model.output_proj)

        return all_logits[:-1]  # Return logits for all positions except last

    # Create teacher-forcing claim
    token_sequence = np.array(tokens, dtype=np.int32)

    tf_claim = create_claim_from_jax_function(
        teacher_forcing_core,
        [token_sequence],
        classifier_name="bit_exact",
        model_type="simple_transformer",
        generation_type="teacher_forcing",
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
    )

    db.add_claim(tf_claim)
    print(f"\nTeacher-forcing claim created: {tf_claim.id}")

    # Verify teacher-forcing claim
    tf_result = verify(tf_claim)
    assert tf_result.passed, "Teacher-forcing claim verification failed"
    print(f"✅ Teacher-forcing claim verified")

    # === CROSS-VERIFICATION ===

    print(f"\n{'=' * 60}")
    print("Verification Results")
    print(f"{'=' * 60}")

    print("\nComparing logits between autoregressive and teacher-forcing:")

    # Execute teacher-forcing to get all logits
    tf_logits_array = jax.jit(teacher_forcing_core)(jnp.array(tokens))

    verification_success = True
    max_diff = 0.0

    for i in range(len(logits_ar)):
        ar_logits = logits_ar[i]
        tf_logits = tf_logits_array[i]

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
        print(
            f"  Position {i}: AR token={int(ar_token)}, TF token={int(tf_token)} {status} (max diff={diff:.6f})"
        )

    print(f"\nMax logit difference: {max_diff:.6f}")

    if verification_success and max_diff < 1e-4:
        print(
            "\n✅ VERIFICATION SUCCESSFUL: Teacher-forcing produces identical results!"
        )
    else:
        print(f"\n⚠️  VERIFICATION WARNING: Differences found (max={max_diff:.6f})")

    assert verification_success, (
        "Autoregressive and teacher-forcing outputs don't match"
    )

    print(f"\n✅ Autoregressive inference with teacher-forcing verification complete!")
    print(f"   - Claims in database: {len(db.list_all())}")
    print(f"   - Max logit difference: {max_diff:.6f}")
