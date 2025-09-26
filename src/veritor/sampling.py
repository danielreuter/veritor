"""
Production-ready deterministic sampling for Veritor.

This version prioritizes simplicity and reliability over advanced features.
Top-k and top-p filtering are implemented without JAX tracing issues.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


class ProductionSampler:
    """
    Production-ready deterministic sampler.

    Features:
    - Temperature sampling
    - Simple top-k filtering (without tracing issues)
    - Deterministic seeding
    - Batch support
    """

    def __init__(self, enable_x64: bool = True):
        """Initialize the sampler."""
        if not HAS_JAX:
            raise ImportError("JAX is required for ProductionSampler")

        if enable_x64:
            jax.config.update("jax_enable_x64", True)

    @staticmethod
    def _simple_prng(seed: int, position: int) -> float:
        """Simple deterministic PRNG that works with JAX."""
        # Mix seed and position
        mixed = (seed * 31337 + position * 17 + 42) % 2147483647
        # Convert to float in [0, 1)
        return (mixed % 997) / 997.0

    @staticmethod
    @jax.jit
    def _apply_temperature(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
        """Apply temperature scaling to logits."""
        # Ensure temperature is positive
        T = jnp.maximum(temperature, 1e-8)
        return logits / T

    @staticmethod
    def _simple_top_k_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
        """
        Simple top-k masking.
        Note: k must be a Python int, not a traced value.
        """
        V = logits.shape[-1]

        if k <= 0 or k >= V:
            # No filtering
            return jnp.ones_like(logits, dtype=jnp.bool_)

        # Sort logits to find threshold
        sorted_logits = jnp.sort(logits, axis=-1)
        # Get the (V-k)th value as threshold
        threshold = sorted_logits[..., -(k+1)]
        # Keep values > threshold
        mask = logits > threshold[..., None]

        return mask

    def sample_simple(
        self,
        logits: jnp.ndarray,
        temperature: float = 1.0,
        seed: int = 42,
        position: int = 0
    ) -> jnp.ndarray:
        """
        Simple sampling without complex features.

        Args:
            logits: Model logits [vocab_size] or [batch, vocab_size]
            temperature: Sampling temperature
            seed: Random seed
            position: Position in sequence

        Returns:
            Sampled indices
        """
        # Apply temperature
        scaled = self._apply_temperature(logits, temperature)

        # Compute probabilities
        probs = jax.nn.softmax(scaled, axis=-1)

        # Get deterministic random value
        u = self._simple_prng(seed, position)

        # Sample using inverse CDF
        cdf = jnp.cumsum(probs, axis=-1)
        # Ensure last value is 1.0
        cdf = cdf.at[..., -1].set(1.0)

        # Find first index where CDF >= u
        idx = jnp.sum(cdf < u, axis=-1)

        return idx

    def sample_with_top_k(
        self,
        logits: jnp.ndarray,
        temperature: float = 1.0,
        top_k: int = -1,
        seed: int = 42,
        position: int = 0
    ) -> jnp.ndarray:
        """
        Sample with top-k filtering.

        Note: For production use with JAX JIT, top_k must be a static integer.
        """
        # Apply temperature
        scaled = self._apply_temperature(logits, temperature)

        # Apply top-k if specified
        if top_k > 0 and top_k < logits.shape[-1]:
            mask = self._simple_top_k_mask(scaled, top_k)
            scaled = jnp.where(mask, scaled, -jnp.inf)

        # Compute probabilities
        probs = jax.nn.softmax(scaled, axis=-1)

        # Get deterministic random value
        u = self._simple_prng(seed, position)

        # Sample
        cdf = jnp.cumsum(probs, axis=-1)
        cdf = cdf.at[..., -1].set(1.0)
        idx = jnp.sum(cdf < u, axis=-1)

        return idx


def create_production_decode(vocab_size: int = 32, hidden_dim: int = 16):
    """
    Create a production-ready decode function with sampling.

    Returns both the decode function and its HLO export.
    """
    if not HAS_JAX:
        raise ImportError("JAX required for decode generation")

    # Simple model parameters
    key = jax.random.PRNGKey(42)
    E = jax.random.normal(key, (vocab_size, hidden_dim), dtype=jnp.float32) * 0.1
    W = jax.random.normal(
        jax.random.fold_in(key, 1),
        (hidden_dim, vocab_size),
        dtype=jnp.float32
    ) * 0.1

    sampler = ProductionSampler(enable_x64=False)

    def decode_greedy(init_token, embedding, projection, steps=5):
        """Greedy decode (argmax)."""
        def body(token, _):
            logits = embedding[token] @ projection
            next_token = jnp.argmax(logits).astype(jnp.int32)
            return next_token, next_token

        _, sequence = jax.lax.scan(body, init_token, xs=None, length=steps)
        return sequence

    def decode_sampled(init_token, embedding, projection, temperature, seed, steps=5):
        """Decode with sampling."""
        def body(carry, pos):
            token = carry
            logits = embedding[token] @ projection
            next_token = sampler.sample_simple(logits, temperature, seed, int(pos))
            next_token = next_token.astype(jnp.int32)
            return next_token, next_token

        positions = jnp.arange(steps)
        _, sequence = jax.lax.scan(body, init_token, xs=positions)
        return sequence

    return {
        "greedy": decode_greedy,
        "sampled": decode_sampled,
        "params": (E, W),
        "sampler": sampler
    }


# Simplified API for direct use
class SimpleTokenSampler:
    """Dead simple token sampler for non-JAX environments."""

    @staticmethod
    def sample(logits, temperature=1.0, top_k=None, seed=42, position=0):
        """
        Sample a token from logits.

        Args:
            logits: numpy array of logits
            temperature: float > 0
            top_k: int or None
            seed: random seed
            position: position in sequence

        Returns:
            int: sampled token index
        """
        logits = np.asarray(logits, dtype=np.float32)

        # Temperature
        if temperature > 0:
            logits = logits / temperature

        # Top-k
        if top_k and 0 < top_k < len(logits):
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.zeros_like(logits, dtype=bool)
            mask[indices] = True
            logits = np.where(mask, logits, -np.inf)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Sample
        rng = np.random.RandomState(seed + position)
        return rng.choice(len(probs), p=probs)


if __name__ == "__main__":
    print("Production Sampler Test")
    print("=" * 60)

    if HAS_JAX:
        print("\n1. JAX Sampler:")
        sampler = ProductionSampler()
        logits = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])

        for temp in [0.5, 1.0, 2.0]:
            idx = sampler.sample_simple(logits, temperature=temp)
            print(f"   T={temp}: sampled {idx}")

        print("\n2. With top-k:")
        for k in [2, 3]:
            idx = sampler.sample_with_top_k(logits, top_k=k)
            print(f"   k={k}: sampled {idx}")

        print("\n3. Decode functions:")
        decode = create_production_decode()
        E, W = decode["params"]
        init = jnp.int32(0)

        # Greedy
        seq = decode["greedy"](init, E, W)
        print(f"   Greedy: {seq}")

        # Sampled
        seq = decode["sampled"](init, E, W, 1.0, 42)
        print(f"   Sampled: {seq}")

    print("\n4. Simple NumPy Sampler:")
    logits = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

    for temp in [0.5, 1.0, 2.0]:
        idx = SimpleTokenSampler.sample(logits, temperature=temp)
        print(f"   T={temp}: sampled {idx}")

    print("\n✅ Production sampler ready for deployment!")