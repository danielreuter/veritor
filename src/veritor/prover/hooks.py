"""
Hook system for the prover to handle challenges and LSH projections.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class HookSystem:
    """
    System for managing runtime hooks including LSH projections and challenges.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.challenge_responses = []
        self.challenge_schedule: Dict[Tuple[int, int, int], bool] = {}
        self.projections_cache = {}

    def set_challenge_schedule(self, schedule: Dict[Tuple[int, int, int], bool]):
        """Set the challenge schedule for execution"""
        self.challenge_schedule = schedule

    def should_challenge(self, step: int, layer: int, batch_idx: int) -> bool:
        """Check if we should challenge at this point"""
        return self.challenge_schedule.get((step, layer, batch_idx), False)

    def challenge_hook(
        self,
        state: jnp.ndarray,
        step: int,
        layer: int,
        batch_idx: int,
        projection_seed: int,
    ) -> jnp.ndarray:
        """
        Hook called during execution to generate LSH projection.

        This is called via io_callback from JAX.
        """
        projection = self.compute_lsh_projection(state, projection_seed)

        response = {
            "step": step,
            "layer": layer,
            "batch_idx": batch_idx,
            "timestamp": datetime.now().timestamp(),
            "projection": projection.tolist(),
            "state_shape": state.shape,
        }

        self.challenge_responses.append(response)

        # Print for debugging
        print(
            f"  Challenge: S{step}/L{layer}/B{batch_idx} → "
            f"LSH[{projection_seed}]={projection[:3]}..."
        )

        return state  # Pass through

    def compute_lsh_projection(
        self, state: jnp.ndarray, seed: int, n_projections: int = 16
    ) -> jnp.ndarray:
        """
        Compute locality-sensitive hash projection of state.

        Args:
            state: Activation tensor to project
            seed: Random seed for projection matrix
            n_projections: Number of projection dimensions

        Returns:
            Projected vector of shape (n_projections,)
        """
        cache_key = (seed, state.shape, n_projections)

        if cache_key not in self.projections_cache:
            # Generate random projection matrix
            key = jax.random.PRNGKey(seed)
            proj_shape = (state.size, n_projections)
            projection_matrix = jax.random.normal(key, proj_shape) / np.sqrt(state.size)
            self.projections_cache[cache_key] = projection_matrix
        else:
            projection_matrix = self.projections_cache[cache_key]

        # Flatten and project
        flat_state = state.flatten()
        projection = jnp.matmul(flat_state, projection_matrix)

        # Quantize to reduce noise sensitivity
        projection = jnp.round(projection * 100) / 100

        return projection

    def verify_lsh_projection(
        self,
        state: jnp.ndarray,
        projection: jnp.ndarray,
        seed: int,
        tolerance: float = 0.01,
    ) -> bool:
        """
        Verify that a projection matches the expected value for a state.

        Args:
            state: Original activation tensor
            projection: Claimed projection
            seed: Seed used for projection
            tolerance: Tolerance for comparison

        Returns:
            True if projection is valid within tolerance
        """
        expected = self.compute_lsh_projection(state, seed, len(projection))
        distance = jnp.linalg.norm(expected - projection)
        return distance < tolerance

    def get_challenge_responses(self) -> List[Dict[str, Any]]:
        """Get all challenge responses recorded"""
        return self.challenge_responses

    def clear_responses(self):
        """Clear recorded challenge responses"""
        self.challenge_responses = []

    def generate_challenge_nonce(self, trace_id: str, timestamp: float) -> int:
        """
        Generate a deterministic nonce for challenges.

        Args:
            trace_id: ID of the trace being challenged
            timestamp: Timestamp of the challenge

        Returns:
            Deterministic nonce for LSH seeding
        """
        # Create deterministic nonce from trace ID and timestamp
        nonce_str = f"{trace_id}_{timestamp}"
        nonce_hash = hashlib.sha256(nonce_str.encode()).digest()
        nonce = int.from_bytes(nonce_hash[:4], byteorder="big")
        return nonce
