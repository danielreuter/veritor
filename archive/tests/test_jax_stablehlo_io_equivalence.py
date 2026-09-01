"""
Test that JAX functions and their StableHLO exports have equivalent I/O interfaces.

This is a fundamental assumption: when we export a JAX function to StableHLO
and re-execute it via IREE, the inputs/outputs should match exactly.

We test various edge cases:
- Different numbers of inputs/outputs
- Different tensor shapes and dtypes
- Scalar inputs/outputs
- Multiple return values
- Functions with different computational patterns
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veritor.accounting import create_claim_from_jax_function
from veritor.verify import verify


class TestSingleInputOutput:
    """Tests for single input/output functions."""

    def test_single_input_single_output(self):
        """Test basic case: one input, one output."""

        def square(x):
            return x**2

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Create claim using our machinery
        claim = create_claim_from_jax_function(square, [x])

        # Verify should pass (outputs match what JAX computed)
        assert verify(claim)

    def test_single_input_multiple_outputs(self):
        """Test one input, multiple outputs."""

        def split_ops(x):
            return x + 1, x * 2, x**2

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        claim = create_claim_from_jax_function(split_ops, [x])
        assert verify(claim)


class TestMultipleInputs:
    """Tests for multiple input functions."""

    def test_multiple_inputs_single_output(self):
        """Test multiple inputs, one output."""

        def add_three(a, b, c):
            return a + b + c

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = np.array([5.0, 6.0], dtype=np.float32)

        claim = create_claim_from_jax_function(add_three, [a, b, c])
        assert verify(claim)

    def test_multiple_inputs_multiple_outputs(self):
        """Test multiple inputs, multiple outputs."""

        def compute_pair(a, b):
            return a + b, a * b

        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        claim = create_claim_from_jax_function(compute_pair, [a, b])
        assert verify(claim)


class TestScalarOperations:
    """Tests for scalar inputs and outputs."""

    def test_scalar_inputs_outputs(self):
        """Test scalar inputs and outputs."""

        def scalar_ops(x, y):
            return x + y, x * y

        # Use 0-d arrays for scalars
        x = np.array(3.0, dtype=np.float32)
        y = np.array(4.0, dtype=np.float32)

        claim = create_claim_from_jax_function(scalar_ops, [x, y])
        assert verify(claim)


class TestTensorShapes:
    """Tests for various tensor shapes and operations."""

    def test_different_shapes(self):
        """Test various tensor shapes."""

        def matmul_add(a, b, c):
            return jnp.matmul(a, b) + c

        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        c = np.random.randn(3, 5).astype(np.float32)

        claim = create_claim_from_jax_function(matmul_add, [a, b, c])
        assert verify(claim)

    def test_broadcasting(self):
        """Test that broadcasting works the same way."""

        def broadcast_add(a, b):
            return a + b

        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape (1, 3)
        b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # Shape (3, 1)

        claim = create_claim_from_jax_function(broadcast_add, [a, b])
        assert verify(claim)

    def test_reshaping_operations(self):
        """Test that reshape operations preserve I/O equivalence."""

        def reshape_ops(x):
            flat = jnp.reshape(x, (-1,))
            transposed = jnp.transpose(x)
            return flat, transposed

        x = np.random.randn(3, 4).astype(np.float32)

        claim = create_claim_from_jax_function(reshape_ops, [x])
        assert verify(claim)


class TestReductionOperations:
    """Tests for reduction operations."""

    def test_reduction_operations(self):
        """Test reduction operations that change output shape."""

        def reduce_sum(x):
            return jnp.sum(x), jnp.sum(x, axis=0), jnp.sum(x, axis=1)

        x = np.random.randn(4, 5).astype(np.float32)

        claim = create_claim_from_jax_function(reduce_sum, [x])
        assert verify(claim)


class TestNonlinearOperations:
    """Tests for nonlinear operations."""

    def test_nonlinear_activations(self):
        """Test nonlinear functions like relu, tanh, sigmoid."""

        def activations(x):
            relu = jnp.maximum(0, x)
            tanh = jnp.tanh(x)
            sigmoid = jax.nn.sigmoid(x)
            return relu, tanh, sigmoid

        x = np.random.randn(10).astype(np.float32)

        claim = create_claim_from_jax_function(activations, [x])
        assert verify(claim)


class TestIndexingOperations:
    """Tests for indexing and slicing."""

    def test_indexing_and_slicing(self):
        """Test indexing and slicing operations."""

        def slice_ops(x):
            first_row = x[0]
            middle_cols = x[:, 1:3]
            diagonal = jnp.diag(x)
            return first_row, middle_cols, diagonal

        x = np.random.randn(4, 4).astype(np.float32)

        claim = create_claim_from_jax_function(slice_ops, [x])
        assert verify(claim)


class TestComplexComputations:
    """Tests for complex computations."""

    def test_complex_computation(self):
        """Test a more complex computation combining multiple operations."""

        def mlp_layer(x, w1, b1, w2, b2):
            """Simple 2-layer MLP."""
            h = jnp.maximum(0, jnp.matmul(x, w1) + b1)  # ReLU activation
            out = jnp.matmul(h, w2) + b2
            return out, h  # Return both output and hidden state

        # Random weights and biases
        x = np.random.randn(2, 10).astype(np.float32)  # Batch of 2
        w1 = np.random.randn(10, 20).astype(np.float32)
        b1 = np.random.randn(20).astype(np.float32)
        w2 = np.random.randn(20, 5).astype(np.float32)
        b2 = np.random.randn(5).astype(np.float32)

        claim = create_claim_from_jax_function(mlp_layer, [x, w1, b1, w2, b2])
        assert verify(claim)

    def test_constants_in_function(self):
        """Test that functions with internal constants work correctly."""

        def with_constants(x):
            """Function that uses constants internally."""
            scale = 2.0
            offset = jnp.array([1.0, 2.0, 3.0])
            return x * scale + offset

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        claim = create_claim_from_jax_function(with_constants, [x])
        assert verify(claim)


class TestBatchedOperations:
    """Tests for batched operations."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batched_operations(self, batch_size):
        """Test batched operations with different batch sizes."""

        def batch_norm(x):
            """Simple batch normalization."""
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)
            normalized = (x - mean) / jnp.sqrt(var + 1e-5)
            return normalized, mean, var

        x = np.random.randn(batch_size, 10).astype(np.float32)
        claim = create_claim_from_jax_function(batch_norm, [x])
        assert verify(claim)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_numerical_stability(self):
        """Test that numerically sensitive operations remain stable."""

        def softmax(x):
            """Numerically stable softmax."""
            exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
            return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

        x = np.random.randn(5, 10).astype(np.float32)

        claim = create_claim_from_jax_function(softmax, [x])
        assert verify(claim)
