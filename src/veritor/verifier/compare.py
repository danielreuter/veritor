"""
Comparison utilities for verification - tolerances, metrics, distance functions.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp


def compare_tensors(
    computed: jnp.ndarray, expected: jnp.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> Dict[str, Any]:
    """
    Compare two tensors with specified tolerances.

    Args:
        computed: Computed tensor from re-execution
        expected: Expected tensor from prover's claim
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with comparison results
    """
    # Check shapes match
    if computed.shape != expected.shape:
        return {
            "match": False,
            "error": "shape_mismatch",
            "computed_shape": computed.shape,
            "expected_shape": expected.shape,
        }

    # Compute various distance metrics
    abs_diff = jnp.abs(computed - expected)
    max_abs_diff = float(jnp.max(abs_diff))
    mean_abs_diff = float(jnp.mean(abs_diff))

    # Relative difference (avoiding division by zero)
    expected_abs = jnp.abs(expected)
    mask = expected_abs > atol
    rel_diff = jnp.where(mask, abs_diff / expected_abs, 0)
    max_rel_diff = float(jnp.max(rel_diff))

    # Check if within tolerance
    is_close = jnp.allclose(computed, expected, rtol=rtol, atol=atol)

    return {
        "match": is_close,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "rtol_used": rtol,
        "atol_used": atol,
    }


def compare_lsh_projections(
    computed: jnp.ndarray, expected: jnp.ndarray, tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Compare LSH projections with noise tolerance.

    Args:
        computed: Computed LSH projection
        expected: Expected LSH projection from prover
        tolerance: L2 distance tolerance

    Returns:
        Dictionary with comparison results
    """
    if computed.shape != expected.shape:
        return {
            "match": False,
            "error": "shape_mismatch",
            "computed_shape": computed.shape,
            "expected_shape": expected.shape,
        }

    # Compute L2 distance
    l2_distance = float(jnp.linalg.norm(computed - expected))

    # Compute cosine similarity
    computed_norm = jnp.linalg.norm(computed)
    expected_norm = jnp.linalg.norm(expected)

    if computed_norm > 0 and expected_norm > 0:
        cosine_sim = float(
            jnp.dot(computed, expected) / (computed_norm * expected_norm)
        )
    else:
        cosine_sim = 0.0

    is_match = l2_distance < tolerance

    return {
        "match": is_match,
        "l2_distance": l2_distance,
        "cosine_similarity": cosine_sim,
        "tolerance_used": tolerance,
    }


def compare_sampling_distributions(
    logits: jnp.ndarray,
    sampled_token: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify that a sampled token is consistent with the claimed distribution.

    Args:
        logits: Logits from model
        sampled_token: Token that was allegedly sampled
        temperature: Temperature used for sampling
        top_k: Top-k filtering parameter

    Returns:
        Dictionary with verification results
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # Apply top-k if specified
    if top_k is not None and top_k > 0:
        top_k_indices = jnp.argsort(scaled_logits)[-top_k:]
        mask = jnp.zeros_like(scaled_logits, dtype=bool)
        mask = mask.at[top_k_indices].set(True)
        scaled_logits = jnp.where(mask, scaled_logits, -jnp.inf)

    # Compute probabilities
    probs = jax.nn.softmax(scaled_logits)

    # Get probability of sampled token
    token_prob = float(probs[sampled_token])

    # Compute surprise (negative log likelihood)
    if token_prob > 0:
        surprise = -float(jnp.log(token_prob))
    else:
        surprise = float("inf")

    # Check if token was in top-k
    if top_k is not None:
        sorted_indices = jnp.argsort(logits)[::-1]
        token_rank = int(jnp.where(sorted_indices == sampled_token)[0][0]) + 1
        is_valid = token_rank <= top_k
    else:
        token_rank = None
        is_valid = token_prob > 0

    return {
        "valid": is_valid,
        "token_probability": token_prob,
        "surprise": surprise,
        "token_rank": token_rank,
        "temperature": temperature,
        "top_k": top_k,
    }


def compute_activation_fingerprint(
    activations: jnp.ndarray,
    n_projections: int = 16,
    seed: int = 42,
) -> jnp.ndarray:
    """
    Compute a compact fingerprint of activations using random projections.

    Args:
        activations: Activation tensor
        n_projections: Number of projection dimensions
        seed: Random seed for projection matrix

    Returns:
        Fingerprint vector
    """
    import jax

    # Generate random projection matrix
    key = jax.random.PRNGKey(seed)
    proj_shape = (activations.size, n_projections)
    projection_matrix = jax.random.normal(key, proj_shape) / jnp.sqrt(activations.size)

    # Project and quantize
    flat_activations = activations.flatten()
    fingerprint = jnp.matmul(flat_activations, projection_matrix)

    # Quantize to reduce noise sensitivity
    fingerprint = jnp.round(fingerprint * 100) / 100

    return fingerprint
