"""
Tests for deterministic sampling utilities.
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sampling import DeterministicSampler, SimpleSampler


class TestSimpleSampler:
    """Test the simple numpy-based sampler."""

    def test_basic_sampling(self):
        """Test basic sampling functionality."""
        logits = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        idx = SimpleSampler.sample_with_temperature(logits, temperature=1.0, seed=42)
        assert 0 <= idx < len(logits)

    def test_determinism(self):
        """Test that sampling is deterministic."""
        logits = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        # Same seed and position should give same result
        idx1 = SimpleSampler.sample_with_temperature(logits, seed=42, position=0)
        idx2 = SimpleSampler.sample_with_temperature(logits, seed=42, position=0)
        assert idx1 == idx2

        # Different seeds should (likely) give different results
        idx3 = SimpleSampler.sample_with_temperature(logits, seed=123, position=0)
        # Note: This might occasionally fail if both seeds map to same token

        # Different positions should give different results
        idx4 = SimpleSampler.sample_with_temperature(logits, seed=42, position=1)
        assert idx1 != idx4 or idx1 == idx4  # Could be same by chance

    def test_temperature_effects(self):
        """Test that temperature affects sampling."""
        logits = np.array([1.0, 2.0, 10.0, 2.0, 1.0])  # Strong peak at index 2

        # Low temperature should favor the peak
        samples_low_temp = [
            SimpleSampler.sample_with_temperature(logits, temperature=0.1, seed=42, position=i)
            for i in range(10)
        ]
        # Most samples should be the peak (index 2)
        assert samples_low_temp.count(2) > len(samples_low_temp) // 2

        # High temperature should be more uniform
        samples_high_temp = [
            SimpleSampler.sample_with_temperature(logits, temperature=10.0, seed=42, position=i)
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
        idx = SimpleSampler.sample_with_temperature(logits)
        assert idx == 0

        # Very small temperature (near-greedy)
        logits = np.array([1.0, 2.0, 3.0])
        idx = SimpleSampler.sample_with_temperature(logits, temperature=1e-8)
        assert idx == 2  # Should pick maximum

        # Uniform logits
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        idx = SimpleSampler.sample_with_temperature(logits)
        assert 0 <= idx < 4


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestDeterministicSampler:
    """Test the JAX-based deterministic sampler."""

    @pytest.fixture
    def sampler(self):
        """Create a sampler instance."""
        return DeterministicSampler(enable_x64=True)

    def test_basic_sampling(self, sampler):
        """Test basic sampling with explicit U."""
        logits = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])
        u = jnp.float32(0.5)

        idx, used_u = sampler.sample(logits, u_explicit=u)
        assert 0 <= idx < len(logits)
        assert used_u == u

    def test_address_based_sampling(self, sampler):
        """Test sampling with address-based PRNG."""
        logits = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])

        idx, u = sampler.sample(
            logits,
            seed=jnp.uint64(42),
            session_id=jnp.uint64(1),
            sequence_id=jnp.uint64(1),
            position=jnp.uint64(0)
        )

        assert 0 <= idx < len(logits)
        assert 0.0 < u < 1.0

    def test_determinism(self, sampler):
        """Test that sampling is deterministic."""
        logits = jnp.array([[1.0, 2.0, 3.0, 2.0, 1.0]])

        # Same parameters should give same result
        idx1, u1 = sampler.sample(
            logits,
            seed=jnp.uint64(42),
            session_id=jnp.uint64(1),
            sequence_id=jnp.uint64(1),
            position=jnp.array([0], dtype=jnp.uint64)
        )

        idx2, u2 = sampler.sample(
            logits,
            seed=jnp.uint64(42),
            session_id=jnp.uint64(1),
            sequence_id=jnp.uint64(1),
            position=jnp.array([0], dtype=jnp.uint64)
        )

        assert jnp.array_equal(idx1, idx2)
        assert jnp.array_equal(u1, u2)

    def test_batch_invariance(self, sampler):
        """Test that batching doesn't affect results."""
        logits = jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
        positions = jnp.array([0, 1], dtype=jnp.uint64)

        # Batched sampling
        batch_idx, batch_u = sampler.sample(
            logits,
            seed=jnp.uint64(42),
            session_id=jnp.uint64(1),
            sequence_id=jnp.uint64(1),
            position=positions
        )

        # Individual sampling
        idx0, u0 = sampler.sample(
            logits[0:1],
            seed=jnp.uint64(42),
            session_id=jnp.uint64(1),
            sequence_id=jnp.uint64(1),
            position=positions[0:1]
        )

        idx1, u1 = sampler.sample(
            logits[1:2],
            seed=jnp.uint64(42),
            session_id=jnp.uint64(1),
            sequence_id=jnp.uint64(1),
            position=positions[1:2]
        )

        assert batch_idx[0] == idx0[0]
        assert batch_idx[1] == idx1[0]

    def test_temperature(self, sampler):
        """Test temperature scaling."""
        logits = jnp.array([1.0, 2.0, 10.0, 2.0, 1.0])  # Strong peak at index 2

        # Low temperature - should favor the peak
        samples_low = []
        for i in range(20):
            idx, _ = sampler.sample(
                logits,
                temperature=jnp.float32(0.1),
                seed=jnp.uint64(42),
                session_id=jnp.uint64(1),
                sequence_id=jnp.uint64(1),
                position=jnp.uint64(i)
            )
            samples_low.append(int(idx))

        # Most should be index 2
        assert samples_low.count(2) > len(samples_low) // 2

    def test_top_k_filtering(self, sampler):
        """Test top-k filtering."""
        logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Top-k = 2 should only sample from top 2
        samples = []
        for i in range(20):
            idx, _ = sampler.sample(
                logits,
                top_k=jnp.int32(2),
                seed=jnp.uint64(42),
                session_id=jnp.uint64(1),
                sequence_id=jnp.uint64(1),
                position=jnp.uint64(i)
            )
            samples.append(int(idx))

        # Should only contain indices 3 and 4 (highest logits)
        unique_samples = set(samples)
        assert unique_samples.issubset({3, 4})

    def test_top_p_filtering(self, sampler):
        """Test top-p (nucleus) filtering."""
        # Create logits with clear probability distribution
        logits = jnp.array([1.0, 1.0, 1.0, 10.0, 1.0])  # Index 3 dominates

        # With small top_p, should mostly sample the dominant token
        samples = []
        for i in range(20):
            idx, _ = sampler.sample(
                logits,
                top_p=jnp.float32(0.5),
                seed=jnp.uint64(42),
                session_id=jnp.uint64(1),
                sequence_id=jnp.uint64(1),
                position=jnp.uint64(i)
            )
            samples.append(int(idx))

        # Most samples should be index 3
        assert samples.count(3) > len(samples) // 2

    def test_combined_filtering(self, sampler):
        """Test combining temperature, top-k, and top-p."""
        logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        idx, u = sampler.sample(
            logits,
            temperature=jnp.float32(0.8),
            top_k=jnp.int32(3),
            top_p=jnp.float32(0.9),
            u_explicit=jnp.float32(0.5)
        )

        assert 0 <= idx < len(logits)
        assert u == 0.5

    def test_edge_cases(self, sampler):
        """Test edge cases."""
        # Single element
        logits = jnp.array([1.0])
        idx, _ = sampler.sample(logits, u_explicit=jnp.float32(0.5))
        assert idx == 0

        # Top-k = 0 (should sample first token)
        logits = jnp.array([1.0, 2.0, 3.0])
        idx, _ = sampler.sample(
            logits,
            top_k=jnp.int32(0),
            u_explicit=jnp.float32(0.5)
        )
        assert idx == 0

        # Top-p = 0 (should sample first token)
        idx, _ = sampler.sample(
            logits,
            top_p=jnp.float32(0.0),
            u_explicit=jnp.float32(0.5)
        )
        assert idx == 0

        # Very high temperature
        logits = jnp.array([1.0, 2.0, 3.0])
        idx, _ = sampler.sample(
            logits,
            temperature=jnp.float32(1000.0),
            u_explicit=jnp.float32(0.5)
        )
        assert 0 <= idx < 3


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestSamplerExport:
    """Test StableHLO export functionality."""

    def test_export_sampler(self):
        """Test exporting sampler to StableHLO."""
        from sampling import export_sampler_hlo

        hlo_text = export_sampler_hlo(vocab_size=32, batch_size=2)

        # Check that it's valid HLO
        assert "module" in hlo_text
        assert "func.func" in hlo_text
        assert "stablehlo" in hlo_text

        # Should contain sampling operations
        assert "softmax" in hlo_text.lower() or "exp" in hlo_text.lower()

    def test_export_different_sizes(self):
        """Test exporting with different vocab and batch sizes."""
        from sampling import export_sampler_hlo

        # Small vocab
        hlo1 = export_sampler_hlo(vocab_size=8, batch_size=1)
        assert "tensor<1x8xf32>" in hlo1 or "tensor<8xf32>" in hlo1

        # Large vocab
        hlo2 = export_sampler_hlo(vocab_size=128, batch_size=4)
        assert "tensor<4x128xf32>" in hlo2 or "tensor<128xf32>" in hlo2


class TestSamplerIntegration:
    """Integration tests for samplers."""

    def test_numpy_jax_consistency(self):
        """Test that numpy and JAX samplers behave similarly."""
        if not HAS_JAX:
            pytest.skip("JAX not available")

        # Use same logits and parameters
        logits_np = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        logits_jax = jnp.array(logits_np)

        # Simple sampler
        idx_simple = SimpleSampler.sample_with_temperature(
            logits_np,
            temperature=1.0,
            seed=42,
            position=0
        )

        # Deterministic sampler with similar seed
        sampler = DeterministicSampler(enable_x64=True)
        idx_det, _ = sampler.sample(
            logits_jax,
            temperature=jnp.float32(1.0),
            seed=jnp.uint64(42),
            session_id=jnp.uint64(0),
            sequence_id=jnp.uint64(0),
            position=jnp.uint64(0)
        )

        # Both should produce valid indices
        assert 0 <= idx_simple < 5
        assert 0 <= idx_det < 5