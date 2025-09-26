#!/usr/bin/env python3
"""
Systematic debugging of LSH computation differences between prover and verifier.
Focus on RNG coupling and seed generation.
"""

import jax.numpy as jnp
from jax import random
from datetime import datetime
from veritor.db.api import WorkloadDatabase
from veritor.db.models import Graph, ChallengeRecord
from veritor.verifier.engine import verify_single_execution, VerificationConfig

def debug_challenge_creation_vs_verification():
    """Debug the exact process of challenge creation vs verification."""

    print("=== SYSTEMATIC LSH DEBUGGING ===\n")

    # Create database and run a minimal test to generate real challenges
    db = WorkloadDatabase()

    # Import the test and run it to generate challenges
    from test_simple_inference import test_simple_inference_with_real_stablehlo
    graph_id, trace_id = test_simple_inference_with_real_stablehlo(db)

    print(f"Generated graph: {graph_id}, trace: {trace_id}")
    print(f"Total challenges in DB: {len(db.challenges)}")

    # Get the failing challenges
    failing_challenges = [c for c in db.challenges if 'p1_b2_l1' in c.id or 'p2_b1_l1' in c.id]
    print(f"Found {len(failing_challenges)} challenges to debug")

    if not failing_challenges:
        print("No failing challenges found - need to run test with challenge verification enabled")
        return

    # Debug each failing challenge
    for i, challenge in enumerate(failing_challenges[:2]):  # Debug first 2
        print(f"\n--- CHALLENGE {i+1}: {challenge.id} ---")
        debug_single_challenge(challenge, db, graph_id, trace_id)

def debug_single_challenge(challenge, db, graph_id, trace_id):
    """Debug a single challenge in detail."""

    print(f"Challenge ID: {challenge.id}")
    print(f"Challenge type: {challenge.challenge_type}")
    print(f"Challenge seed: {challenge.seed}")
    print(f"Challenge metadata: {challenge.metadata}")
    print(f"Stored response: {challenge.response_value[:4]}...")  # First 4 values

    # Extract metadata
    pass_idx = challenge.metadata.get('pass_idx')
    batch_idx = challenge.metadata.get('batch_idx')
    layer_idx = challenge.metadata.get('layer_idx')

    print(f"Context: pass={pass_idx}, batch={batch_idx}, layer={layer_idx}")

    # 1. VERIFY SEED GENERATION
    expected_seed = pass_idx * 10000 + batch_idx * 100 + layer_idx
    print(f"Expected seed formula: {pass_idx} * 10000 + {batch_idx} * 100 + {layer_idx} = {expected_seed}")
    print(f"Stored seed: {challenge.seed}")
    print(f"Seed matches: {expected_seed == challenge.seed}")

    # 2. GET VERIFIER'S ACTIVATION EXTRACTION
    print(f"\n2. Verifier activation extraction:")

    # Load execution data like the verifier does
    from veritor.verifier.engine import UnifiedVerificationEngine
    engine = UnifiedVerificationEngine(db)
    execution_data = engine._load_execution_data(graph_id, trace_id)

    print(f"Available activation keys: {list(execution_data.activations.keys())}")

    # Try to extract activation like verifier does
    from veritor.verifier.engine import DirectExecutionStrategy, VerificationConfig
    strategy = DirectExecutionStrategy(VerificationConfig())

    extracted_activation = strategy._extract_challenge_activation(challenge, execution_data)

    if extracted_activation is not None:
        print(f"Extracted activation shape: {extracted_activation.shape}")
        print(f"Extracted activation preview: {extracted_activation[:4]}")

        # 3. REPRODUCE ORIGINAL COMPUTATION
        print(f"\n3. Original prover computation:")
        original_lsh = reproduce_original_computation(extracted_activation, challenge.seed, 4)
        print(f"Original LSH: {original_lsh}")

        # 4. REPRODUCE VERIFIER COMPUTATION
        print(f"\n4. Verifier computation:")
        verifier_lsh = strategy._compute_lsh_projection(extracted_activation, challenge.seed, 4)
        print(f"Verifier LSH: {verifier_lsh}")

        # 5. COMPARE
        stored_lsh = jnp.array(challenge.response_value)
        print(f"Stored LSH: {stored_lsh}")

        original_matches_stored = jnp.allclose(original_lsh, stored_lsh, rtol=1e-6, atol=1e-6)
        verifier_matches_stored = jnp.allclose(verifier_lsh, stored_lsh, rtol=1e-6, atol=1e-6)
        original_matches_verifier = jnp.allclose(original_lsh, verifier_lsh, rtol=1e-6, atol=1e-6)

        print(f"\nComparisons:")
        print(f"  Original matches stored: {original_matches_stored}")
        print(f"  Verifier matches stored: {verifier_matches_stored}")
        print(f"  Original matches verifier: {original_matches_verifier}")

        if not original_matches_stored:
            print(f"  Original vs stored diff: {jnp.abs(original_lsh - stored_lsh)}")
            print(f"  Max diff: {jnp.max(jnp.abs(original_lsh - stored_lsh))}")

        # 6. CHECK IF WE HAVE THE RIGHT ACTIVATION
        print(f"\n6. Activation debugging:")
        print(f"  Original activation used for challenge is UNKNOWN (not stored)")
        print(f"  We need to reproduce the exact activation from the forward pass")

        # 7. REPRODUCE THE FORWARD PASS THAT GENERATED THIS ACTIVATION
        debug_activation_reproduction(db, graph_id, pass_idx, layer_idx, batch_idx)

    else:
        print("❌ Could not extract activation from execution data")

def reproduce_original_computation(activation, seed, lsh_dim):
    """Reproduce the exact original LSH computation."""
    key = random.PRNGKey(seed)
    act_dim = activation.shape[-1]
    proj_matrix = random.normal(key, (act_dim, lsh_dim))
    proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)
    return jnp.dot(activation, proj_matrix)

def debug_activation_reproduction(db, graph_id, pass_idx, layer_idx, batch_idx):
    """Try to reproduce the exact activation tensor that was used for the challenge."""
    print(f"\n7. Reproducing forward pass activation:")

    # Get the input that was used for this pass
    data_bundles = db.get_data_for_graph(graph_id)
    if not data_bundles:
        print("❌ No data bundles found")
        return

    bundle = data_bundles[0]  # Should be the main data bundle

    # Get the input for this pass
    input_key = f"input_pass_{pass_idx}"
    if input_key not in bundle.inputs:
        print(f"❌ Input key {input_key} not found in bundle")
        print(f"Available input keys: {list(bundle.inputs.keys())}")
        return

    input_tensor = bundle.inputs[input_key].to_array()
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor preview: {input_tensor.flatten()[:4]}")

    # Get the model weights
    weights = []
    weight_keys = [k for k in bundle.weights.keys() if k.startswith('weight_')]
    weight_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort by layer index

    for key in weight_keys:
        w = bundle.weights[key].to_array()
        # Find corresponding bias - this is trickier, need to infer structure
        weights.append(w)  # Just weights for now, need to handle bias properly

    print(f"Found {len(weights)} weight matrices")

    # Try to reproduce the forward pass
    # This is where we need to be very careful about the exact computation
    print("⚠️  Need to reproduce exact forward pass computation...")
    print("⚠️  This requires knowing the exact model structure from the test")

def debug_rng_determinism():
    """Test RNG determinism with same seeds."""
    print("\n=== RNG DETERMINISM TEST ===")

    seed = 12345

    # Test 1: Same seed, same computation
    key1 = random.PRNGKey(seed)
    matrix1 = random.normal(key1, (8, 4))

    key2 = random.PRNGKey(seed)
    matrix2 = random.normal(key2, (8, 4))

    print(f"Same seed produces same matrix: {jnp.allclose(matrix1, matrix2)}")

    # Test 2: Different order of operations
    key3 = random.PRNGKey(seed)
    # Do some other random operation first
    dummy = random.normal(key3, (2, 2))
    matrix3 = random.normal(key3, (8, 4))

    print(f"Order matters: {jnp.allclose(matrix1, matrix3)}")  # Should be False

    # Test 3: Key splitting
    key4 = random.PRNGKey(seed)
    subkey4, _ = random.split(key4)
    matrix4 = random.normal(subkey4, (8, 4))

    print(f"Key splitting changes output: {jnp.allclose(matrix1, matrix4)}")  # Should be False

if __name__ == "__main__":
    debug_rng_determinism()
    debug_challenge_creation_vs_verification()