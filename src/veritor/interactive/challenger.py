"""
Simple challenger for interactive verification.

Just randomly samples at hook points - no complex schedule generation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback


@dataclass
class Challenger:
    """
    Simple challenger that makes random decisions at each hook point.

    No complex StableHLO parsing - just flip a coin at each layer.
    """

    challenge_probability: float = 0.3
    seed: Optional[int] = None
    lsh_dim: int = 4

    def __post_init__(self):
        """Initialize random state."""
        self.rng = np.random.RandomState(self.seed)
        self.challenge_count = 0
        self.query_count = 0

    def should_challenge(self) -> Tuple[bool, int, int]:
        """
        Decide whether to challenge at this hook point.

        Returns:
            (should_challenge, seed_for_lsh, projection_dim)
        """
        self.query_count += 1

        # Roll the dice
        if self.rng.random() < self.challenge_probability:
            self.challenge_count += 1
            # Generate seed for this specific challenge
            lsh_seed = self.rng.randint(0, 2**31)
            return True, lsh_seed, self.lsh_dim

        return False, 0, 0

    def get_stats(self) -> Dict[str, Any]:
        """Get challenge statistics."""
        return {
            "queries": self.query_count,
            "challenges": self.challenge_count,
            "rate": self.challenge_count / max(1, self.query_count),
        }


# Global challenger for io_callback
_global_challenger: Optional[Challenger] = None


def set_global_challenger(challenger: Challenger):
    """Set global challenger for io_callback use."""
    global _global_challenger
    _global_challenger = challenger


def create_challenge_hook(
    challenger: Optional[Challenger] = None,
    challenge_probability: float = 0.3,
    seed: Optional[int] = None,
) -> callable:
    """
    Create a challenge hook for use in JAX functions.

    Just randomly decides at each hook point whether to compute LSH.

    Args:
        challenger: Optional pre-configured challenger
        challenge_probability: Probability of challenge at each hook
        seed: Random seed for reproducibility

    Returns:
        Hook function that can be called during forward pass
    """
    # Only set global challenger if one was explicitly provided
    if challenger is not None:
        set_global_challenger(challenger)
    elif _global_challenger is None:
        # Only create a new one if there's no global challenger
        challenger = Challenger(challenge_probability=challenge_probability, seed=seed)
        set_global_challenger(challenger)

    def challenge_hook(activation: jnp.ndarray, name: str = "layer") -> jnp.ndarray:
        """
        Hook that maybe computes LSH projection.

        Args:
            activation: Current activation tensor
            name: Name of this hook point (for debugging)

        Returns:
            LSH projection or zeros
        """

        def query_challenger(_):
            """Query the global challenger."""
            if _global_challenger is None:
                return jnp.array([0.0, 0.0, 4.0])

            should_challenge, seed, dim = _global_challenger.should_challenge()
            return jnp.array([float(should_challenge), float(seed), float(dim)])

        # Use io_callback to query challenger
        decision = io_callback(query_challenger, jnp.zeros(3), jnp.array(0.0))

        should_challenge = decision[0] > 0.5
        seed = jnp.int32(decision[1])

        # Fixed projection dim for simplicity
        projection_dim = 4

        # Compute LSH if challenged
        def compute_lsh():
            key = jax.random.PRNGKey(seed)
            flat = activation.flatten()

            # Generate random projection matrix
            proj = jax.random.normal(key, (projection_dim, flat.shape[0]))
            proj = proj / jnp.linalg.norm(proj, axis=1, keepdims=True)

            # Project and scale
            result = proj @ flat
            scale = jnp.sqrt(flat.shape[0] / projection_dim)
            return result * scale

        def no_challenge():
            return jnp.zeros(projection_dim)

        lsh = jax.lax.cond(should_challenge, compute_lsh, no_challenge)

        return lsh

    return challenge_hook


def compute_lsh_projection(
    tensor: np.ndarray, seed: int = 42, projection_dim: int = 4
) -> np.ndarray:
    """
    Compute LSH projection of a tensor (numpy version).

    Args:
        tensor: Input tensor
        seed: Random seed
        projection_dim: Target dimension

    Returns:
        LSH projection
    """
    rng = np.random.RandomState(seed)
    flat = tensor.flatten()

    # Random projection - ensure we maintain dtype
    proj = rng.randn(projection_dim, len(flat)).astype(tensor.dtype)
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)

    # Project and scale
    result = proj @ flat
    scale = np.sqrt(len(flat) / projection_dim)
    return (result * scale).astype(tensor.dtype)


class ChallengeContext:
    """
    Context manager for challenge-enabled execution.

    Example:
        ```python
        challenger = Challenger(challenge_probability=0.3, seed=42)

        with ChallengeContext(challenger):
            # JAX functions can now use io_callbacks to query challenger
            output = jitted_fn(input)
        ```
    """

    def __init__(self, challenger: Challenger):
        self.challenger = challenger
        self._previous_challenger = None

    def __enter__(self):
        global _global_challenger
        self._previous_challenger = _global_challenger
        set_global_challenger(self.challenger)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_global_challenger(self._previous_challenger)
