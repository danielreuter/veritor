#!/usr/bin/env python3
"""
Debug script to understand what autoregressive StableHLO actually looks like.
"""

import jax
import jax.numpy as jnp
from jax import random

# Simple model for testing
class SimpleModel:
    def __init__(self):
        key = random.PRNGKey(42)
        self.embed = random.normal(key, (10, 8)) * 0.1  # vocab_size=10, hidden=8
        self.output = random.normal(random.split(key)[1], (8, 10)) * 0.1

    def forward_step(self, tokens, mask):
        """Single autoregressive step."""
        # tokens: [seq_len]
        # mask: [seq_len] - which positions are valid

        # Embed
        x = self.embed[tokens]  # [seq_len, hidden]

        # Apply mask
        x = x * mask[:, None]

        # Simple reduction (sum over sequence)
        x = jnp.sum(x, axis=0)  # [hidden]

        # Output logits
        logits = jnp.dot(x, self.output)  # [vocab_size]

        return logits

# Create model
model = SimpleModel()

# Create autoregressive loop
def autoregressive_generate(start_token, max_len=5):
    """Full autoregressive generation with explicit loop."""
    tokens = jnp.zeros(max_len, dtype=jnp.int32)
    tokens = tokens.at[0].set(start_token)

    def step(carry, _):
        tokens, pos = carry

        # Create mask for current position
        mask = jnp.arange(max_len) <= pos

        # Get logits
        logits = model.forward_step(tokens, mask.astype(jnp.float32))

        # Sample next token (argmax for determinism)
        next_token = jnp.argmax(logits)

        # Update tokens
        new_tokens = tokens.at[pos + 1].set(next_token)

        return (new_tokens, pos + 1), logits

    # Run the loop
    (final_tokens, _), all_logits = jax.lax.scan(
        step,
        (tokens, 0),
        xs=None,
        length=max_len-1
    )

    return final_tokens, all_logits

# JIT compile and get StableHLO
jitted_generate = jax.jit(autoregressive_generate)

print("Generating StableHLO for full autoregressive loop...")
lowered = jitted_generate.lower(3)  # start_token=3
full_loop_hlo = lowered.as_text()

print(f"Full loop HLO size: {len(full_loop_hlo)} bytes")
print("\n=== Key patterns in the HLO ===")

# Look for key patterns
if "while" in full_loop_hlo:
    print("✓ Contains 'while' loop")
    # Find while loop structure
    lines = full_loop_hlo.split('\n')
    for i, line in enumerate(lines):
        if 'while' in line.lower():
            print(f"  Line {i}: {line[:100]}...")
            if i < len(lines) - 1:
                print(f"  Line {i+1}: {lines[i+1][:100]}...")
            break

if "stablehlo.dynamic_update_slice" in full_loop_hlo:
    print("✓ Contains dynamic_update_slice (for token updates)")

if "stablehlo.dynamic_slice" in full_loop_hlo:
    print("✓ Contains dynamic_slice (for masking/selection)")

print("\n=== Now let's look at just the step function ===")

# Just the step function
def just_step(tokens, pos):
    mask = (jnp.arange(5) <= pos).astype(jnp.float32)
    logits = model.forward_step(tokens, mask)
    next_token = jnp.argmax(logits)
    new_tokens = tokens.at[pos + 1].set(next_token)
    return new_tokens, logits

jitted_step = jax.jit(just_step)
example_tokens = jnp.array([3, 0, 0, 0, 0], dtype=jnp.int32)
lowered_step = jitted_step.lower(example_tokens, 0)
step_hlo = lowered_step.as_text()

print(f"Step HLO size: {len(step_hlo)} bytes")

# Save both for analysis
with open('/tmp/ar_full_loop.hlo', 'w') as f:
    f.write(full_loop_hlo)
print("Saved full loop HLO to /tmp/ar_full_loop.hlo")

with open('/tmp/ar_step.hlo', 'w') as f:
    f.write(step_hlo)
print("Saved step HLO to /tmp/ar_step.hlo")

print("\n=== What would teacher-forcing look like? ===")

def teacher_forcing(tokens):
    """Process all tokens in parallel."""
    all_logits = []

    for pos in range(len(tokens) - 1):
        # Create mask for this position
        mask = (jnp.arange(len(tokens)) <= pos).astype(jnp.float32)

        # Get logits for this position
        logits = model.forward_step(tokens, mask)
        all_logits.append(logits)

    return jnp.stack(all_logits)

jitted_tf = jax.jit(teacher_forcing)
example_seq = jnp.array([3, 5, 2, 7, 1], dtype=jnp.int32)
lowered_tf = jitted_tf.lower(example_seq)
tf_hlo = lowered_tf.as_text()

print(f"Teacher-forcing HLO size: {len(tf_hlo)} bytes")

with open('/tmp/ar_teacher_forcing.hlo', 'w') as f:
    f.write(tf_hlo)
print("Saved teacher-forcing HLO to /tmp/ar_teacher_forcing.hlo")

print("\n=== Key insight ===")
print("The autoregressive HLO has a while loop with:")
print("  1. Loop carry state (tokens, position)")
print("  2. Dynamic updates to the token array")
print("  3. Sequential dependencies")
print("\nThe teacher-forcing version needs to:")
print("  1. Unroll the loop")
print("  2. Use the provided tokens instead of generated ones")
print("  3. Process positions in parallel (or at least without token dependencies)")