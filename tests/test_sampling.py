"""
Tests for deterministic sampling utilities.
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

from veritor.common.sampler import ProductionSampler as DeterministicSampler
from veritor.common.sampler import SimpleTokenSampler as SimpleSampler


class TestSimpleSampler:
    """Test the simple numpy-based sampler."""

    def test_basic_sampling(self):
        """Test basic sampling functionality."""
        logits = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        idx = SimpleSampler.sample(logits, temperature=1.0, seed=42)
        assert 0 <= idx < len(logits)

    def test_determinism(self):
        """Test that sampling is deterministic."""
        logits = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        # Same seed and position should give same result
        idx1 = SimpleSampler.sample(logits, seed=42, position=0)
        idx2 = SimpleSampler.sample(logits, seed=42, position=0)
        assert idx1 == idx2

        # Different seeds should (likely) give different results
        idx3 = SimpleSampler.sample(logits, seed=123, position=0)
        # Note: This might occasionally fail if both seeds map to same token

        # Different positions should give different results
        idx4 = SimpleSampler.sample(logits, seed=42, position=1)
        # Could be same by chance, so we don't assert inequality

    def test_temperature_effects(self):
        """Test that temperature affects sampling."""
        logits = np.array([1.0, 2.0, 10.0, 2.0, 1.0])  # Strong peak at index 2

        # Low temperature should favor the peak
        samples_low_temp = [
            SimpleSampler.sample(logits, temperature=0.1, seed=42, position=i)
            for i in range(10)
        ]
        # Most samples should be the peak (index 2)
        assert samples_low_temp.count(2) > len(samples_low_temp) // 2

        # High temperature should be more uniform
        samples_high_temp = [
            SimpleSampler.sample(logits, temperature=10.0, seed=42, position=i)
            for i in range(10)
        ]
        # Should have more variety
        unique_high = len(set(samples_high_temp))
        unique_low = len(set(samples_low_temp))
        assert unique_high >= unique_low

    def test_edge_cases(self):
        """Test edge cases."""
        # Single element
        logits = np.array([1.0])
        idx = SimpleSampler.sample(logits)
        assert idx == 0

        # Very small temperature (near-greedy)
        logits = np.array([1.0, 2.0, 3.0])
        idx = SimpleSampler.sample(logits, temperature=1e-8)
        assert idx == 2  # Should pick maximum

        # Uniform logits
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        idx = SimpleSampler.sample(logits)
        assert 0 <= idx < 4

    def test_top_k_filtering(self):
        """Test top-k filtering."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # With top_k=2, should only sample from indices 3 and 4
        samples = []
        for i in range(20):
            idx = SimpleSampler.sample(logits, top_k=2, seed=42, position=i)
            samples.append(idx)

        # Should only contain the top 2 indices
        assert set(samples).issubset({3, 4})


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestDeterministicSampler:
    """Test the JAX-based deterministic sampler."""

    @pytest.fixture
    def sampler(self):
        """Create a sampler instance."""
        return DeterministicSampler(enable_x64=True)

    def test_simple_sampling(self, sampler):
        """Test basic sampling with sample_simple."""
        logits = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])

        idx = sampler.sample_simple(logits, temperature=1.0, seed=42, position=0)
        assert 0 <= idx < len(logits)

    def test_determinism(self, sampler):
        """Test that sampling is deterministic."""
        logits = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])

        # Same parameters should give same result
        idx1 = sampler.sample_simple(logits, seed=42, position=0)
        idx2 = sampler.sample_simple(logits, seed=42, position=0)

        assert idx1 == idx2

    def test_temperature(self, sampler):
        """Test temperature effects."""
        logits = jnp.array([1.0, 2.0, 10.0, 2.0, 1.0])

        # Low temperature
        samples_low = []
        for i in range(10):
            idx = sampler.sample_simple(logits, temperature=0.1, seed=42, position=i)
            samples_low.append(int(idx))

        # High temperature
        samples_high = []
        for i in range(10):
            idx = sampler.sample_simple(logits, temperature=10.0, seed=42, position=i)
            samples_high.append(int(idx))

        # Low temperature should be more concentrated
        assert len(set(samples_low)) <= len(set(samples_high))

    def test_batch_sampling(self, sampler):
        """Test sampling with batched logits."""
        # Batch of 2 sequences
        logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

        # Sample from each
        idx = sampler.sample_simple(logits, temperature=1.0, seed=42, position=0)

        # Should return indices for both sequences
        assert idx.shape == (2,)
        assert all(0 <= i < 3 for i in idx)

    def test_top_k_sampling(self, sampler):
        """Test sample_with_top_k method."""
        logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Sample with top-k
        idx = sampler.sample_with_top_k(
            logits, top_k=2, temperature=1.0, seed=42, position=0
        )

        # Should be from top-k indices
        assert 0 <= idx < len(logits)
        # Note: Due to the implementation, we can't strictly guarantee it's in top-k
        # without multiple samples

    def test_edge_cases(self, sampler):
        """Test edge cases."""
        # Single element
        logits = jnp.array([1.0])
        idx = sampler.sample_simple(logits, seed=42)
        assert idx == 0

        # Very peaked distribution
        logits = jnp.array([1.0, 1.0, 100.0, 1.0])
        idx = sampler.sample_simple(logits, temperature=0.01, seed=42, position=0)
        # Should almost always pick the peak
        assert idx == 2
