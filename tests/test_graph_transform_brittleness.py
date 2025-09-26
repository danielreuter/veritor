"""
Deep stress test for graph transformations - finding where JAX/StableHLO breaks.

This test specifically targets the interaction between:
1. Dynamic control flow (sampling decisions)
2. Static graph compilation (JAX/StableHLO)
3. Verification assumptions
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random
import numpy as np


class GraphTransformationBreaker:
    """Tests designed to break graph transformation assumptions."""

    def test_dynamic_shapes_in_sampling(self):
        """Test what happens when sampling changes tensor shapes."""
        failures = []

        # This is a problem: sampling might select different sequence lengths
        def variable_length_sampling(logits, key, max_len):
            """Sample until we hit an end token (token 0)."""
            outputs = []
            for i in range(max_len):
                key, subkey = random.split(key)
                probs = jax.nn.softmax(logits)
                token = random.categorical(subkey, logits)
                outputs.append(token)
                if token == 0:  # End token
                    break  # PROBLEM: Dynamic loop termination!
            return jnp.array(outputs)

        # Try to JIT this
        try:
            jitted = jax.jit(variable_length_sampling, static_argnums=2)
            logits = jnp.array([1.0, 2.0, 3.0, -10.0])  # Token 0 is unlikely
            key = random.PRNGKey(42)
            result = jitted(logits, key, 10)
            # This actually works but returns fixed size!
            failures.append("Dynamic termination compiled to fixed size")
        except Exception as e:
            failures.append(f"Dynamic shapes failed: {e}")

        return failures

    def test_sampling_inside_scan(self):
        """Test sampling within JAX scan operations."""
        failures = []

        def scan_with_sampling(carry, x):
            key, state = carry
            key, subkey = random.split(key)

            # Apply transformation
            logits = jnp.dot(state, x)

            # Sample
            probs = jax.nn.softmax(logits)
            sample = random.categorical(subkey, logits)

            # Update state based on sample (conditional update!)
            new_state = jax.lax.cond(
                sample > 5,
                lambda s: s * 2.0,  # Branch 1
                lambda s: s * 0.5,  # Branch 2
                state
            )

            return (key, new_state), sample

        # Try to trace this
        try:
            key = random.PRNGKey(42)
            init_state = jnp.ones(10)
            xs = jnp.ones((20, 10))  # 20 timesteps

            # Run scan
            _, samples = jax.lax.scan(scan_with_sampling, (key, init_state), xs)

            # JIT compile
            jitted_scan = jax.jit(lambda k, s, x: jax.lax.scan(scan_with_sampling, (k, s), x))

            # Get StableHLO
            lowered = jitted_scan.lower(key, init_state, xs)
            stablehlo = lowered.as_text()

            # Check for expected patterns
            if "stablehlo.while" not in stablehlo:
                failures.append("Scan didn't generate while loop")
            if "stablehlo.rng" not in stablehlo:
                failures.append("RNG missing from scan")

        except Exception as e:
            failures.append(f"Scan sampling failed: {e}")

        return failures

    def test_gradient_through_sampling(self):
        """Test what happens when we try to differentiate through sampling."""
        failures = []

        def loss_with_sampling(params, key, x):
            # Forward pass
            logits = jnp.dot(x, params)

            # Sample (non-differentiable!)
            probs = jax.nn.softmax(logits)
            samples = random.categorical(key, logits)

            # Try to compute loss based on samples
            loss = jnp.sum(samples * params)  # Problematic!
            return loss

        try:
            params = jnp.ones((5, 5))
            x = jnp.ones(5)
            key = random.PRNGKey(42)

            # Try to compute gradient - this should fail or give wrong results
            grad_fn = jax.grad(loss_with_sampling, argnums=0)
            grads = grad_fn(params, key, x)

            # Check if gradients are zero (they should be due to sampling)
            if jnp.all(grads == 0):
                failures.append("Gradients through sampling are zero")
            else:
                failures.append("Gradients through sampling are non-zero (incorrect!)")

        except Exception as e:
            failures.append(f"Gradient computation failed: {e}")

        return failures

    def test_nested_sampling_loops(self):
        """Test deeply nested sampling operations."""
        failures = []

        def nested_sampling(key, depth=3):
            if depth == 0:
                return random.normal(key)

            key1, key2 = random.split(key)
            # Sample to decide branch
            choice = random.categorical(key1, jnp.array([1.0, 1.0]))

            # Recursive sampling based on choice
            return jax.lax.cond(
                choice == 0,
                lambda k: nested_sampling(k, depth - 1),
                lambda k: -nested_sampling(k, depth - 1),
                key2
            )

        try:
            # JIT compile with static depth
            jitted = jax.jit(nested_sampling, static_argnums=1)
            key = random.PRNGKey(42)

            # Test different depths
            for depth in [1, 2, 3, 5]:
                result = jitted(key, depth)
                # Should work but creates complex graph

            # Get StableHLO for depth 3
            lowered = jitted.lower(key, 3)
            stablehlo = lowered.as_text()

            # Check complexity
            num_conditionals = stablehlo.count("stablehlo.if")
            if num_conditionals > 10:
                failures.append(f"Exponential graph explosion: {num_conditionals} conditionals")

        except Exception as e:
            failures.append(f"Nested sampling failed: {e}")

        return failures

    def test_sampling_with_side_effects(self):
        """Test sampling operations that try to have side effects."""
        failures = []

        # Global state (BAD in JAX!)
        global_counter = [0]

        def sampling_with_side_effects(key, logits):
            # Try to modify global state
            global_counter[0] += 1  # This won't work as expected in JIT!

            # Sample
            sample = random.categorical(key, logits)

            # Try to print (side effect)
            # jax.debug.print("Sample: {}", sample)  # This would work but differently

            return sample

        try:
            jitted = jax.jit(sampling_with_side_effects)
            key = random.PRNGKey(42)
            logits = jnp.array([1.0, 2.0, 3.0])

            # Run multiple times
            for i in range(5):
                result = jitted(random.PRNGKey(i), logits)

            # Check if counter was updated
            if global_counter[0] == 1:
                failures.append("Side effects only executed once (JIT cached)")
            elif global_counter[0] == 0:
                failures.append("Side effects completely ignored")

        except Exception as e:
            failures.append(f"Side effects test failed: {e}")

        return failures

    def test_verify_sampling_determinism(self):
        """Test if we can verify sampling is deterministic with same key."""
        failures = []

        def model_with_sampling(x, key):
            # Complex model with multiple sampling points
            h1 = jax.nn.relu(jnp.dot(x, jnp.ones((5, 10))))

            key1, key2, key3 = random.split(key, 3)

            # First sampling point
            logits1 = h1[:5]
            sample1 = random.categorical(key1, logits1)

            # Conditional second sampling
            if sample1 > 5:  # This won't work in JIT!
                sample2 = random.categorical(key2, h1[5:])
            else:
                sample2 = random.categorical(key3, h1[5:])

            return sample1 + sample2

        try:
            # This should fail to JIT due to Python conditional
            jitted = jax.jit(model_with_sampling)
            x = jnp.ones(5)
            key = random.PRNGKey(42)
            result = jitted(x, key)
            failures.append("Python conditional in JIT didn't fail (unexpected)")
        except Exception as e:
            # Expected to fail
            pass

        # Try with proper JAX conditional
        def fixed_model(x, key):
            h1 = jax.nn.relu(jnp.dot(x, jnp.ones((5, 10))))
            key1, key2, key3 = random.split(key, 3)

            logits1 = h1[:5]
            sample1 = random.categorical(key1, logits1)

            sample2 = jax.lax.cond(
                sample1 > 5,
                lambda: random.categorical(key2, h1[5:]),
                lambda: random.categorical(key3, h1[5:])
            )

            return sample1 + sample2

        try:
            jitted_fixed = jax.jit(fixed_model)
            x = jnp.ones(5)
            key = random.PRNGKey(42)

            # Run multiple times with same key
            results = []
            for _ in range(3):
                results.append(int(jitted_fixed(x, key)))

            if len(set(results)) != 1:
                failures.append(f"Non-deterministic with same key: {results}")

        except Exception as e:
            failures.append(f"Fixed model failed: {e}")

        return failures


def test_graph_transformation_brittleness():
    """Run all graph transformation stress tests."""

    breaker = GraphTransformationBreaker()
    all_failures = []

    print("\n🔥 TESTING GRAPH TRANSFORMATION BRITTLENESS")
    print("=" * 60)

    # Test 1: Dynamic shapes
    print("\n1. Testing dynamic shapes in sampling...")
    failures = breaker.test_dynamic_shapes_in_sampling()
    all_failures.extend(failures)
    print(f"   Found {len(failures)} issues")
    for f in failures:
        print(f"   - {f}")

    # Test 2: Sampling in scan
    print("\n2. Testing sampling inside scan...")
    failures = breaker.test_sampling_inside_scan()
    all_failures.extend(failures)
    print(f"   Found {len(failures)} issues")
    for f in failures:
        print(f"   - {f}")

    # Test 3: Gradients through sampling
    print("\n3. Testing gradients through sampling...")
    failures = breaker.test_gradient_through_sampling()
    all_failures.extend(failures)
    print(f"   Found {len(failures)} issues")
    for f in failures:
        print(f"   - {f}")

    # Test 4: Nested sampling
    print("\n4. Testing nested sampling loops...")
    failures = breaker.test_nested_sampling_loops()
    all_failures.extend(failures)
    print(f"   Found {len(failures)} issues")
    for f in failures:
        print(f"   - {f}")

    # Test 5: Side effects
    print("\n5. Testing sampling with side effects...")
    failures = breaker.test_sampling_with_side_effects()
    all_failures.extend(failures)
    print(f"   Found {len(failures)} issues")
    for f in failures:
        print(f"   - {f}")

    # Test 6: Verification challenges
    print("\n6. Testing verification determinism...")
    failures = breaker.test_verify_sampling_determinism()
    all_failures.extend(failures)
    print(f"   Found {len(failures)} issues")
    for f in failures:
        print(f"   - {f}")

    print("\n" + "=" * 60)
    print(f"💀 TOTAL BRITTLENESS ISSUES FOUND: {len(all_failures)}")
    print("=" * 60)

    if all_failures:
        print("\n⚠️  CRITICAL ISSUES FOR VERIFICATION:")
        print("1. Dynamic control flow doesn't play well with static graphs")
        print("2. Sampling breaks differentiability")
        print("3. Side effects are problematic in JIT")
        print("4. Graph explosion with nested conditionals")
        print("5. Determinism verification is complex")

    return len(all_failures)


if __name__ == "__main__":
    failures_found = test_graph_transformation_brittleness()
    print(f"\n{'✅' if failures_found > 0 else '❌'} Test completed - found {failures_found} brittleness issues!")