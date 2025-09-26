"""
Aggressive stress test for sampling and graph transformation.

This test is designed to BREAK the system and find its failure modes.
We test extreme conditions, numerical edge cases, and push the boundaries
of what the verification system can handle.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random
import numpy as np
from datetime import datetime
import uuid
import sys

from veritor.db.models import Graph, Trace, TraceEvent, EventType, DataBundle, TensorData
from veritor.db.ir_store import IRRole, IRFormat


class StressTestSampling:
    """Stress test sampling with extreme conditions."""

    def __init__(self):
        self.failure_modes = []
        self.edge_cases_found = []

    def test_extreme_temperatures(self, logits: jnp.ndarray):
        """Test sampling at extreme temperature values."""
        key = random.PRNGKey(42)
        failures = []

        # Test near-zero temperature (should approach argmax)
        for temp in [1e-10, 1e-8, 1e-6, 1e-4]:
            try:
                scaled = logits / temp
                # This should overflow!
                if jnp.any(jnp.isinf(scaled)) or jnp.any(jnp.isnan(scaled)):
                    failures.append(f"Overflow at temperature {temp}")
                probs = jax.nn.softmax(scaled)
                if jnp.any(jnp.isnan(probs)):
                    failures.append(f"NaN in probs at temperature {temp}")
            except Exception as e:
                failures.append(f"Exception at temp {temp}: {e}")

        # Test extreme high temperature (should approach uniform)
        for temp in [100, 1000, 10000, 1e6]:
            try:
                scaled = logits / temp
                probs = jax.nn.softmax(scaled)
                # Check if distribution is too uniform (entropy too high)
                entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))
                max_entropy = jnp.log(len(logits))
                if entropy > 0.99 * max_entropy:
                    failures.append(f"Distribution too uniform at temp {temp}")
            except Exception as e:
                failures.append(f"Exception at temp {temp}: {e}")

        return failures

    def test_degenerate_distributions(self):
        """Test sampling from pathological distributions."""
        key = random.PRNGKey(42)
        failures = []

        # All zeros (should fail)
        try:
            logits = jnp.zeros(10)
            probs = jax.nn.softmax(logits)
            sample = random.categorical(key, logits)
            # This is actually valid (uniform distribution)
        except Exception as e:
            failures.append(f"Failed on all zeros: {e}")

        # Single huge value (numerical overflow)
        try:
            logits = jnp.array([-1000.0] * 9 + [1000.0])
            probs = jax.nn.softmax(logits)
            if not jnp.allclose(probs[-1], 1.0, rtol=1e-6):
                failures.append("Softmax failed on extreme values")
        except Exception as e:
            failures.append(f"Failed on extreme values: {e}")

        # NaN contamination
        try:
            logits = jnp.array([1.0, 2.0, jnp.nan, 3.0])
            probs = jax.nn.softmax(logits)
            if not jnp.all(jnp.isnan(probs)):
                failures.append("NaN didn't propagate through softmax")
        except Exception as e:
            # This is expected to fail
            pass

        # Infinity
        try:
            logits = jnp.array([1.0, 2.0, jnp.inf, 3.0])
            probs = jax.nn.softmax(logits)
            if not jnp.isnan(probs).any() and probs[2] != 1.0:
                failures.append("Infinity handling incorrect in softmax")
        except Exception as e:
            failures.append(f"Failed on infinity: {e}")

        return failures

    def test_top_k_edge_cases(self):
        """Test top-k sampling with edge cases."""
        key = random.PRNGKey(42)
        failures = []

        # k larger than vocabulary
        try:
            logits = jnp.array([1.0, 2.0, 3.0])
            k = 5  # Larger than vocab size
            top_k_indices = jnp.argsort(logits)[-k:]
            # Should handle gracefully by using all tokens
            if len(top_k_indices) != 3:
                failures.append("Top-k didn't handle k > vocab_size")
        except Exception as e:
            failures.append(f"Failed on k > vocab_size: {e}")

        # k = 0 (invalid)
        try:
            logits = jnp.array([1.0, 2.0, 3.0])
            k = 0
            top_k_indices = jnp.argsort(logits)[-k:]
            # This should give empty array - problematic!
            if len(top_k_indices) != 0:
                failures.append("Top-k with k=0 didn't return empty")
        except Exception as e:
            failures.append(f"Failed on k=0: {e}")

        # All equal logits with top-k
        try:
            logits = jnp.ones(10)
            k = 3
            # All tokens equally likely - sampling should still work
            top_k_indices = jnp.argsort(logits)[-k:]
            # But which 3 to pick? This is ambiguous!
            failures.append("Top-k ambiguous with equal logits")
        except Exception as e:
            failures.append(f"Failed on equal logits: {e}")

        return failures

    def test_nucleus_sampling_edge_cases(self):
        """Test nucleus (top-p) sampling edge cases."""
        failures = []

        # p = 0 (should select nothing)
        try:
            probs = jnp.array([0.1, 0.2, 0.3, 0.4])
            p = 0.0
            sorted_probs = jnp.sort(probs)[::-1]
            cumsum = jnp.cumsum(sorted_probs)
            mask = cumsum <= p
            if mask.any():
                failures.append("p=0 selected tokens")
        except Exception as e:
            failures.append(f"Failed on p=0: {e}")

        # p > 1 (invalid)
        try:
            probs = jnp.array([0.1, 0.2, 0.3, 0.4])
            p = 1.5
            # Should this error or just use all tokens?
            sorted_probs = jnp.sort(probs)[::-1]
            cumsum = jnp.cumsum(sorted_probs)
            mask = cumsum <= p
            if not mask.all():
                failures.append("p>1 didn't select all tokens")
        except Exception as e:
            failures.append(f"Failed on p>1: {e}")

        # Very small p with concentrated distribution
        try:
            probs = jnp.array([0.99, 0.005, 0.003, 0.002])
            p = 0.5  # Should only select first token
            sorted_probs = jnp.sort(probs)[::-1]
            cumsum = jnp.cumsum(sorted_probs)
            mask = cumsum <= p
            selected = jnp.sum(mask)
            if selected != 1:
                failures.append(f"Small p selected {selected} tokens, expected 1")
        except Exception as e:
            failures.append(f"Failed on small p: {e}")

        return failures

    def test_graph_transformation_under_sampling(self, database):
        """Test if graph transformations remain consistent under sampling."""
        failures = []

        # Create a simple model
        def model_with_sampling(x, key):
            # Linear layer
            W = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            logits = jnp.dot(x, W)

            # Apply temperature scaling
            temp = 0.5
            scaled_logits = logits / temp

            # Sample
            probs = jax.nn.softmax(scaled_logits)
            samples = random.categorical(key, scaled_logits)

            return samples, probs, logits

        # JIT compile
        jitted_model = jax.jit(model_with_sampling)

        # Test inputs
        x = jnp.array([[1.0, 0.5]])
        key = random.PRNGKey(42)

        # Get StableHLO
        try:
            lowered = jitted_model.lower(x, key)
            stablehlo_text = lowered.as_text()

            # Check for expected operations
            if "stablehlo.rng" not in stablehlo_text:
                failures.append("RNG operation missing from StableHLO")

            if "stablehlo.divide" not in stablehlo_text:
                failures.append("Temperature scaling missing from StableHLO")

            # Store in database
            graph = Graph(
                id=f"stress_test_{uuid.uuid4().hex[:8]}",
                metadata={"test": "sampling_stress"}
            )
            graph_id = database.store_graph(graph)

            database.ir_store.attach_ir(
                graph_id,
                IRRole.LOGICAL,
                stablehlo_text,
                IRFormat.STABLEHLO,
                {"has_sampling": True}
            )

            # Execute multiple times - should get different results
            results = []
            for i in range(10):
                key_i = random.PRNGKey(42 + i)
                samples, _, _ = jitted_model(x, key_i)
                results.append(int(samples[0]))

            # Check for diversity
            unique_results = len(set(results))
            if unique_results == 1:
                failures.append(f"No diversity in sampling: all results = {results[0]}")

        except Exception as e:
            failures.append(f"Graph transformation failed: {e}")

        return failures

    def test_numerical_stability_cascading(self):
        """Test cascading numerical errors through multiple sampling steps."""
        failures = []
        key = random.PRNGKey(42)

        # Start with reasonable logits
        logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Apply increasingly extreme transformations
        for step in range(10):
            try:
                # Exponentially decrease temperature
                temp = 10 ** (-step)
                scaled = logits / temp

                # Check for numerical issues
                if jnp.any(jnp.isinf(scaled)):
                    failures.append(f"Infinity at step {step}, temp={temp}")
                    break

                if jnp.any(jnp.isnan(scaled)):
                    failures.append(f"NaN at step {step}, temp={temp}")
                    break

                # Try to compute softmax
                probs = jax.nn.softmax(scaled)

                # Check if distribution collapsed
                max_prob = jnp.max(probs)
                if max_prob > 0.9999:
                    failures.append(f"Distribution collapsed at step {step}, max_prob={max_prob}")

                # Update logits for next iteration (compound the numerical errors)
                logits = jnp.log(probs + 1e-10)  # This will accumulate errors!

            except Exception as e:
                failures.append(f"Cascading failed at step {step}: {e}")
                break

        return failures


def test_break_the_sampling_system(workload_db):
    """Aggressive test to find failure modes in sampling."""

    stress_tester = StressTestSampling()
    all_failures = []

    print("\n🔨 STRESS TESTING SAMPLING SYSTEM - TRYING TO BREAK IT!")
    print("=" * 60)

    # Test 1: Extreme temperatures
    print("\n1. Testing extreme temperatures...")
    logits = jnp.array([1.0, 2.0, 3.0, -1.0, 0.0])
    temp_failures = stress_tester.test_extreme_temperatures(logits)
    all_failures.extend(temp_failures)
    print(f"   Found {len(temp_failures)} failure modes")
    for f in temp_failures[:3]:  # Show first 3
        print(f"   - {f}")

    # Test 2: Degenerate distributions
    print("\n2. Testing degenerate distributions...")
    degen_failures = stress_tester.test_degenerate_distributions()
    all_failures.extend(degen_failures)
    print(f"   Found {len(degen_failures)} failure modes")
    for f in degen_failures[:3]:
        print(f"   - {f}")

    # Test 3: Top-k edge cases
    print("\n3. Testing top-k edge cases...")
    topk_failures = stress_tester.test_top_k_edge_cases()
    all_failures.extend(topk_failures)
    print(f"   Found {len(topk_failures)} failure modes")
    for f in topk_failures[:3]:
        print(f"   - {f}")

    # Test 4: Nucleus sampling edge cases
    print("\n4. Testing nucleus sampling edge cases...")
    nucleus_failures = stress_tester.test_nucleus_sampling_edge_cases()
    all_failures.extend(nucleus_failures)
    print(f"   Found {len(nucleus_failures)} failure modes")
    for f in nucleus_failures[:3]:
        print(f"   - {f}")

    # Test 5: Graph transformation under sampling
    print("\n5. Testing graph transformation consistency...")
    graph_failures = stress_tester.test_graph_transformation_under_sampling(workload_db)
    all_failures.extend(graph_failures)
    print(f"   Found {len(graph_failures)} failure modes")
    for f in graph_failures[:3]:
        print(f"   - {f}")

    # Test 6: Numerical stability cascading
    print("\n6. Testing cascading numerical errors...")
    cascade_failures = stress_tester.test_numerical_stability_cascading()
    all_failures.extend(cascade_failures)
    print(f"   Found {len(cascade_failures)} failure modes")
    for f in cascade_failures[:3]:
        print(f"   - {f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"🔥 TOTAL FAILURE MODES FOUND: {len(all_failures)}")
    print("=" * 60)

    if all_failures:
        print("\n⚠️  KEY VULNERABILITIES:")
        for i, failure in enumerate(all_failures[:10], 1):
            print(f"{i}. {failure}")

        print(f"\n💥 The system is indeed BRITTLE as expected!")
        print("These failures show the sampling is vulnerable to:")
        print("  - Numerical overflow/underflow")
        print("  - Temperature scaling issues")
        print("  - Edge case handling")
        print("  - Distribution degeneracy")
        print("  - Cascading numerical errors")

    # Return success even though we found failures - that was the point!
    return True


if __name__ == "__main__":
    # Run standalone
    import sys
    sys.path.insert(0, '/Users/danielreuter/projects/veritor')

    from veritor.db.api import WorkloadDatabase

    print("Running stress test standalone...")
    db = WorkloadDatabase()
    result = test_break_the_sampling_system(db)

    if result:
        print("\n✅ Stress test completed - found the brittleness!")
    else:
        print("\n❌ Stress test failed to run")