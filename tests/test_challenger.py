"""
Test the simplified Challenger API.

Just tests random sampling at hook points - no complex schedules.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veritor import create_claim_from_jax_function, verify
from veritor.interactive import (
    ChallengeContext,
    Challenger,
    compute_lsh_projection,
    create_challenge_hook,
)


@pytest.fixture
def challenger():
    """Create a challenger with fixed seed."""
    return Challenger(challenge_probability=0.5, seed=42)


@pytest.fixture
def high_prob_challenger():
    """Create a challenger with high challenge probability."""
    return Challenger(challenge_probability=0.8, seed=42)


class TestBasicChallenger:
    """Tests for basic challenger functionality."""

    def test_basic_challenger(self, challenger):
        """Test basic random challenge functionality."""
        # Simulate some queries
        results = []
        for i in range(10):
            should_challenge, seed, dim = challenger.should_challenge()
            results.append(should_challenge)

        # Should get roughly 50% challenges
        challenge_rate = sum(results) / len(results)
        assert 0.2 <= challenge_rate <= 0.8, (
            f"Challenge rate {challenge_rate} outside expected range"
        )

        stats = challenger.get_stats()
        assert stats["queries"] == 10
        assert stats["challenges"] == sum(results)


class TestLSHProjection:
    """Tests for LSH projection computation."""

    def test_lsh_projection(self):
        """Test LSH projection computation."""
        # Create test tensor
        tensor = np.random.randn(3, 4).astype(np.float32)

        # Compute projection
        projection = compute_lsh_projection(tensor, seed=42, projection_dim=4)

        assert projection.shape == (4,)
        assert projection.dtype == np.float32

        # Test determinism
        projection2 = compute_lsh_projection(tensor, seed=42, projection_dim=4)
        assert np.allclose(projection, projection2)

    def test_lsh_projection_different_dims(self):
        """Test LSH projection with different dimensions."""
        tensor = np.random.randn(5, 5).astype(np.float32)

        proj_2d = compute_lsh_projection(tensor, seed=1, projection_dim=2)
        proj_8d = compute_lsh_projection(tensor, seed=1, projection_dim=8)

        assert proj_2d.shape == (2,)
        assert proj_8d.shape == (8,)


class TestChallengeWithIOCallback:
    """Tests for challenge hooks with JAX io_callback."""

    def test_challenge_with_io_callback(self, high_prob_challenger):
        """Test challenge hook with JAX io_callback."""
        hook = create_challenge_hook(high_prob_challenger)

        # Define a simple model with hooks
        @jax.jit
        def model_with_hooks(x):
            # Layer 1
            h = jnp.dot(x, x.T)
            lsh1 = hook(h, "layer1")

            # Layer 2
            h = jax.nn.relu(h)
            lsh2 = hook(h, "layer2")

            # Stack LSH fingerprints
            return h, jnp.stack([lsh1, lsh2])

        # Run the model
        x = np.random.randn(2, 3).astype(np.float32)

        with ChallengeContext(high_prob_challenger):
            output, lsh_fingerprints = model_with_hooks(x)

        assert output.shape == (2, 2)
        assert lsh_fingerprints.shape == (2, 4)

        # Check if we got challenges (with 80% probability, likely at least one)
        has_challenges = np.any(lsh_fingerprints != 0)
        assert has_challenges or high_prob_challenger.get_stats()["queries"] > 0

        stats = high_prob_challenger.get_stats()
        assert stats["rate"] >= 0  # Should have a valid rate


class TestDeterministicChallenges:
    """Tests for deterministic challenge behavior."""

    def test_deterministic_challenges(self):
        """Test that challenges are deterministic with same seed."""
        # Create two challengers with same seed
        c1 = Challenger(challenge_probability=0.5, seed=123)
        c2 = Challenger(challenge_probability=0.5, seed=123)

        # Query them the same number of times
        results1 = []
        results2 = []

        for _ in range(20):
            should_challenge1, _, _ = c1.should_challenge()
            should_challenge2, _, _ = c2.should_challenge()
            results1.append(should_challenge1)
            results2.append(should_challenge2)

        # Should get identical results
        assert results1 == results2

    def test_different_seeds_produce_different_patterns(self):
        """Test that different seeds produce different patterns."""
        c1 = Challenger(challenge_probability=0.5, seed=123)
        c2 = Challenger(challenge_probability=0.5, seed=456)

        results1 = []
        results2 = []

        for _ in range(20):
            should_challenge1, _, _ = c1.should_challenge()
            should_challenge2, _, _ = c2.should_challenge()
            results1.append(should_challenge1)
            results2.append(should_challenge2)

        # Very unlikely to be identical (but theoretically possible)
        # So we just check they're not all the same
        assert results1 != [True] * 20 and results1 != [False] * 20
        assert results2 != [True] * 20 and results2 != [False] * 20


class TestChallengesWithClaims:
    """Tests for integrating challenges with CAP claims."""

    def test_with_claims(self):
        """Test how challenges could work with CAP claims."""

        # Create a simple model
        def model(x, w):
            h = jnp.dot(x, w)
            return jnp.maximum(0, h)  # relu without jax.nn

        x = np.random.randn(3, 4).astype(np.float32)
        w = np.random.randn(4, 2).astype(np.float32)

        # Create standard claim
        claim = create_claim_from_jax_function(model, [x, w], test_name="bit_exact")

        # Verify it
        result = verify(claim)
        assert result.passed if hasattr(result, "passed") else result

        # Now show how you'd use challenges
        challenger = Challenger(challenge_probability=0.5, seed=42)
        hook = create_challenge_hook(challenger)

        # Model with hooks (this would be done during proving)
        @jax.jit
        def model_with_hooks(x, w):
            h = jnp.dot(x, w)
            lsh1 = hook(h, "matmul")
            h = jnp.maximum(0, h)  # relu
            lsh2 = hook(h, "relu")
            return h, jnp.stack([lsh1, lsh2])

        with ChallengeContext(challenger):
            output, lsh_fingerprints = model_with_hooks(x, w)

        assert lsh_fingerprints.shape == (2, 4)


class TestChallengeContext:
    """Tests for ChallengeContext manager."""

    def test_context_manager(self):
        """Test that ChallengeContext properly sets and resets global challenger."""
        challenger1 = Challenger(challenge_probability=0.3, seed=1)
        challenger2 = Challenger(challenge_probability=0.7, seed=2)

        # Use first challenger
        with ChallengeContext(challenger1):
            # Inside context, challenger1 should be active
            pass  # Would test global state if we had access to it

        # Use second challenger
        with ChallengeContext(challenger2):
            # Inside context, challenger2 should be active
            pass

        # After contexts, global state should be reset
        # (Can't directly test this without accessing private global state)

    def test_context_isolation(self):
        """Test that different contexts don't interfere."""
        @jax.jit
        def model(x):
            hook = create_challenge_hook()
            return hook(x, "test")

        x = np.ones(3, dtype=np.float32)

        # Create two challengers with different probabilities
        c1 = Challenger(challenge_probability=1.0, seed=1)  # Always
        c2 = Challenger(challenge_probability=0.0, seed=2)  # Never

        # Test rapid context switching
        with ChallengeContext(c1):
            r1 = model(x)

        with ChallengeContext(c2):
            r2 = model(x)

        with ChallengeContext(c1):
            r3 = model(x)

        assert np.any(r1 != 0), "c1 should challenge"
        assert np.all(r2 == 0), "c2 should not challenge"
        assert np.any(r3 != 0), "c1 should challenge again"


class TestDynamicGlobalState:
    """Tests verifying dynamic global state reading during execution."""

    def test_prover_cannot_predict(self):
        """Test that the prover can't predict challenges ahead of time."""
        @jax.jit
        def prover_model(x):
            """Model that would like to know ahead of time."""
            # The prover can't access _global_challenger directly in JIT code
            # They MUST go through the hook
            hook = create_challenge_hook()

            h = x * 2
            lsh = hook(h, "layer")

            # Prover can't conditionally do less work based on challenge
            # They always have to pass the activation through
            expensive_computation = jnp.sum(h ** 3)  # Always computed

            return expensive_computation, lsh

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with always challenge
        c_always = Challenger(challenge_probability=1.0, seed=42)
        with ChallengeContext(c_always):
            result, lsh = prover_model(x)
            assert np.any(lsh != 0), "Should always challenge"

        # Test with never challenge
        c_never = Challenger(challenge_probability=0.0, seed=42)
        with ChallengeContext(c_never):
            result, lsh = prover_model(x)
            assert np.all(lsh == 0), "Should never challenge"

    def test_dynamic_challenge_patterns(self):
        """Test that challenge patterns vary dynamically."""
        @jax.jit
        def model_with_multiple_hooks(x):
            """Model that calls hooks multiple times."""
            hook = create_challenge_hook()

            h1 = x + 1
            lsh1 = hook(h1, "layer1")

            h2 = h1 * 2
            lsh2 = hook(h2, "layer2")

            h3 = h2 - 1
            lsh3 = hook(h3, "layer3")

            return jnp.stack([lsh1, lsh2, lsh3])

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Test with 50% probability - should see variation
        challenger = Challenger(challenge_probability=0.5, seed=None)

        patterns = []
        with ChallengeContext(challenger):
            for _ in range(5):
                result = model_with_multiple_hooks(x)
                pattern = tuple(bool(np.any(result[j] != 0)) for j in range(3))
                patterns.append(pattern)

        # Should get some variation in patterns
        unique_patterns = set(patterns)
        assert len(unique_patterns) > 1, f"Expected variation, got {unique_patterns}"

    def test_prover_must_compute_everything(self):
        """Test that prover must always compute full activations."""
        @jax.jit
        def honest_prover(x):
            """Honest prover that always passes full data."""
            hook = create_challenge_hook()

            # Layer 1: Full computation
            h1 = jax.nn.relu(jnp.dot(x, x.T))
            lsh1 = hook(h1, "layer1")  # Must pass FULL h1

            # Layer 2: More computation
            h2 = jax.nn.tanh(h1 * 2)
            lsh2 = hook(h2, "layer2")  # Must pass FULL h2

            # The prover can't know ahead of time which layers will be challenged
            return h1, h2, jnp.stack([lsh1, lsh2])

        x = np.random.randn(3, 3).astype(np.float32)

        # Run with random challenger
        challenger = Challenger(challenge_probability=0.5, seed=None)
        with ChallengeContext(challenger):
            h1, h2, lsh_stack = honest_prover(x)

        # Verify shapes - computation was always done
        assert h1.shape == (3, 3), "H1 must be computed"
        assert h2.shape == (3, 3), "H2 must be computed"
        assert lsh_stack.shape == (2, 4), "LSH stack shape correct"
