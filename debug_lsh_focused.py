#!/usr/bin/env python3
"""
Focused debugging of LSH computation - create challenges and immediately verify them.
"""

import jax.numpy as jnp
from jax import random
from veritor.db.api import WorkloadDatabase
from veritor.db.models import ChallengeRecord
from veritor.verifier.engine import DirectExecutionStrategy, VerificationConfig, GraphExecutionData, Graph

def debug_lsh_step_by_step():
    """Create and verify LSH challenges step by step."""

    print("=== STEP-BY-STEP LSH DEBUGGING ===\n")

    # Parameters from the failing test
    config_lsh_dim = 4
    config_hidden_dim = 8
    config_batch_size = 3

    # Test different scenarios
    scenarios = [
        {"pass_idx": 0, "batch_idx": 0, "layer_idx": 1},  # This should work
        {"pass_idx": 1, "batch_idx": 2, "layer_idx": 1},  # This fails
        {"pass_idx": 2, "batch_idx": 1, "layer_idx": 1},  # This fails
    ]

    for scenario in scenarios:
        print(f"\n--- SCENARIO: {scenario} ---")
        debug_single_scenario(**scenario, lsh_dim=config_lsh_dim, hidden_dim=config_hidden_dim)

def debug_single_scenario(pass_idx, batch_idx, layer_idx, lsh_dim, hidden_dim):
    """Debug a single LSH challenge scenario."""

    # 1. Generate seed like the original test
    seed = pass_idx * 10000 + batch_idx * 100 + layer_idx
    print(f"1. Seed calculation: {pass_idx} * 10000 + {batch_idx} * 100 + {layer_idx} = {seed}")

    # 2. Create mock activation tensor (single batch element)
    # Use deterministic data based on scenario for reproducibility
    activation_seed = 1000 + pass_idx * 100 + batch_idx * 10 + layer_idx
    activation_key = random.PRNGKey(activation_seed)
    activation = random.normal(activation_key, (hidden_dim,))
    print(f"2. Mock activation (seed={activation_seed}): {activation}")

    # 3. Original prover computation (exact replica)
    print(f"3. Original prover computation:")
    original_lsh = compute_original_lsh(activation, seed, lsh_dim)
    print(f"   Result: {original_lsh}")

    # 4. Verifier computation
    print(f"4. Verifier computation:")
    strategy = DirectExecutionStrategy(VerificationConfig())
    verifier_lsh = strategy._compute_lsh_projection(activation, seed, lsh_dim)
    print(f"   Result: {verifier_lsh}")

    # 5. Compare
    matches = jnp.allclose(original_lsh, verifier_lsh, rtol=1e-10, atol=1e-10)
    print(f"5. Exact match: {matches}")

    if not matches:
        diff = jnp.abs(original_lsh - verifier_lsh)
        max_diff = jnp.max(diff)
        print(f"   Max difference: {max_diff}")
        print(f"   Difference vector: {diff}")

        # Debug the computation step by step
        debug_computation_details(activation, seed, lsh_dim)
    else:
        print(f"   ✅ This scenario works correctly!")

    # 6. Create and verify a challenge
    print(f"6. Challenge verification test:")
    test_challenge_verification(activation, original_lsh, seed, lsh_dim, pass_idx, batch_idx, layer_idx)

def compute_original_lsh(activation, seed, lsh_dim):
    """Exact replica of original test computation."""
    key = random.PRNGKey(seed)
    act_dim = activation.shape[-1]
    proj_matrix = random.normal(key, (act_dim, lsh_dim))
    proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)
    return jnp.dot(activation, proj_matrix)

def debug_computation_details(activation, seed, lsh_dim):
    """Debug the LSH computation step by step."""
    print(f"   DEBUG: Step-by-step computation")

    # Original method
    print(f"   Original method:")
    key1 = random.PRNGKey(seed)
    act_dim = activation.shape[-1]
    proj_matrix1 = random.normal(key1, (act_dim, lsh_dim))
    print(f"     Raw projection matrix shape: {proj_matrix1.shape}")
    print(f"     Raw projection matrix[0]: {proj_matrix1[0]}")

    norms1 = jnp.linalg.norm(proj_matrix1, axis=1, keepdims=True)
    print(f"     Norms shape: {norms1.shape}")
    print(f"     Norms[0]: {norms1[0]}")

    normalized1 = proj_matrix1 / norms1
    print(f"     Normalized[0]: {normalized1[0]}")

    result1 = jnp.dot(activation, normalized1)
    print(f"     Final result: {result1}")

    # Verifier method (should be identical)
    print(f"   Verifier method:")
    key2 = random.PRNGKey(seed)
    proj_matrix2 = random.normal(key2, (act_dim, lsh_dim))
    print(f"     Raw projection matrix[0]: {proj_matrix2[0]}")

    # Check if raw matrices match
    raw_match = jnp.allclose(proj_matrix1, proj_matrix2, rtol=1e-10, atol=1e-10)
    print(f"     Raw matrices match: {raw_match}")

def test_challenge_verification(activation, expected_lsh, seed, lsh_dim, pass_idx, batch_idx, layer_idx):
    """Test the actual challenge verification process."""

    # Create a challenge record
    challenge = ChallengeRecord(
        id=f"challenge_p{pass_idx}_b{batch_idx}_l{layer_idx}",
        challenge_type="lsh_dynamic",
        timestamp=1234567.0,
        target_operation_id=f"layer_{layer_idx}",
        seed=seed,
        projection_dim=lsh_dim,
        response_value=expected_lsh.tolist(),
        metadata={
            "pass_idx": pass_idx,
            "batch_idx": batch_idx,
            "layer_idx": layer_idx,
            "activation_shape": activation.shape,
            "trace_id": "test_trace",
        }
    )

    # Create mock execution data
    execution_data = GraphExecutionData(
        graph=Graph(id="test_graph", metadata={}),
        activations={f"layer_{layer_idx}": activation}  # Single activation (already extracted)
    )

    # Run verification
    strategy = DirectExecutionStrategy(VerificationConfig())

    # Test extraction
    extracted = strategy._extract_challenge_activation(challenge, execution_data)
    print(f"   Extracted activation: {extracted}")
    print(f"   Extraction matches: {jnp.allclose(extracted, activation) if extracted is not None else False}")

    # Test verification
    if extracted is not None:
        result = strategy._verify_lsh_challenge(challenge, execution_data)
        print(f"   Challenge verification result: {result}")
    else:
        print(f"   ❌ Could not extract activation!")

def debug_batch_vs_single_element():
    """Debug the difference between batch tensors and single elements."""
    print(f"\n=== BATCH VS SINGLE ELEMENT DEBUG ===")

    batch_size = 3
    hidden_dim = 8
    lsh_dim = 4
    batch_idx = 1
    layer_idx = 1
    seed = 12345

    # Create a full batch activation (what's stored in original test)
    key = random.PRNGKey(100)
    batch_activation = random.normal(key, (batch_size, hidden_dim))
    print(f"Batch activation shape: {batch_activation.shape}")

    # Extract single element (what challenge uses)
    single_activation = batch_activation[batch_idx]
    print(f"Single activation shape: {single_activation.shape}")
    print(f"Single activation: {single_activation}")

    # Compute LSH
    lsh = compute_original_lsh(single_activation, seed, lsh_dim)
    print(f"LSH result: {lsh}")

    # Now test if verifier can extract the same single element
    print(f"\nSimulating verifier extraction:")

    # Mock execution data with the batch
    execution_data = GraphExecutionData(
        graph=Graph(id="test_graph", metadata={}),
        activations={f"layer_{layer_idx}": batch_activation}  # Store full batch
    )

    # Mock challenge
    challenge = ChallengeRecord(
        id=f"test_challenge",
        challenge_type="lsh_dynamic",
        timestamp=1234567.0,
        target_operation_id=f"layer_{layer_idx}",
        seed=seed,
        projection_dim=lsh_dim,
        response_value=lsh.tolist(),
        metadata={
            "layer_idx": layer_idx,
            "batch_idx": batch_idx,
            "trace_id": "test_trace",
        }
    )

    strategy = DirectExecutionStrategy(VerificationConfig())
    extracted = strategy._extract_challenge_activation(challenge, execution_data)

    if extracted is not None:
        print(f"Verifier extracted shape: {extracted.shape}")
        print(f"Verifier extracted: {extracted}")
        print(f"Matches original single: {jnp.allclose(extracted, single_activation)}")

        # Test verification
        result = strategy._verify_lsh_challenge(challenge, execution_data)
        print(f"Verification result: {result}")
    else:
        print(f"❌ Verifier could not extract activation!")

if __name__ == "__main__":
    debug_lsh_step_by_step()
    debug_batch_vs_single_element()