"""Simple random model generator for testing execution backend."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random


def create_simple_mlp(
    input_dim: int = 10,
    hidden_dim: int = 20,
    output_dim: int = 5,
    num_layers: int = 2,
    seed: int = 42,
) -> Tuple[callable, Dict[str, jnp.ndarray]]:
    """Create a simple MLP model with random weights.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        seed: Random seed for weight initialization

    Returns:
        (forward_fn, params) where forward_fn takes (params, x) and returns output
    """
    key = random.PRNGKey(seed)
    params = {}

    # Initialize weights
    dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]

    for i in range(len(dims) - 1):
        key, w_key = random.split(key, 2)
        # Xavier/Glorot initialization
        scale = jnp.sqrt(2.0 / (dims[i] + dims[i + 1]))
        params[f"w{i}"] = random.normal(w_key, (dims[i], dims[i + 1])) * scale
        params[f"b{i}"] = jnp.zeros(dims[i + 1])

    def forward(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through MLP."""
        h = x
        num_layers = len([k for k in params.keys() if k.startswith("w")])

        for i in range(num_layers):
            h = jnp.dot(h, params[f"w{i}"]) + params[f"b{i}"]
            # ReLU activation except for last layer
            if i < num_layers - 1:
                h = jax.nn.relu(h)
        return h

    return forward, params


def create_simple_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),  # (H, W, C)
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[callable, Dict[str, jnp.ndarray]]:
    """Create a simple CNN model with random weights.

    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        seed: Random seed

    Returns:
        (forward_fn, params) where forward_fn takes (params, x) and returns output
    """
    key = random.PRNGKey(seed)
    params = {}

    # Conv layer 1: 32 filters of size 3x3
    key, subkey = random.split(key)
    params["conv1_w"] = random.normal(subkey, (3, 3, input_shape[2], 32)) * 0.1
    params["conv1_b"] = jnp.zeros(32)

    # Conv layer 2: 64 filters of size 3x3
    key, subkey = random.split(key)
    params["conv2_w"] = random.normal(subkey, (3, 3, 32, 64)) * 0.1
    params["conv2_b"] = jnp.zeros(64)

    # Calculate flattened dimension after convolutions and pooling
    # After conv1 + pool: (28-2)/2 = 13
    # After conv2 + pool: (13-2)/2 = 5 (approx)
    h, w = input_shape[0], input_shape[1]
    h = (h - 2) // 2  # After conv1 + maxpool
    w = (w - 2) // 2
    h = (h - 2) // 2  # After conv2 + maxpool
    w = (w - 2) // 2
    flat_dim = h * w * 64

    # Dense layers
    key, subkey = random.split(key)
    params["fc1_w"] = random.normal(subkey, (flat_dim, 128)) * 0.1
    params["fc1_b"] = jnp.zeros(128)

    key, subkey = random.split(key)
    params["fc2_w"] = random.normal(subkey, (128, num_classes)) * 0.1
    params["fc2_b"] = jnp.zeros(num_classes)

    def forward(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through CNN.

        Args:
            params: Model parameters
            x: Input of shape (batch_size, height, width, channels) or (height, width, channels)

        Returns:
            Logits of shape (batch_size, num_classes) or (num_classes,)
        """
        # Handle both batched and unbatched inputs
        if x.ndim == 3:
            x = x[None, ...]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        # Conv block 1
        h = jax.lax.conv_general_dilated(
            x,
            params["conv1_w"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        h = h + params["conv1_b"]
        h = jax.nn.relu(h)
        h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (2, 2), (2, 2), "VALID")

        # Conv block 2
        h = jax.lax.conv_general_dilated(
            h,
            params["conv2_w"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        h = h + params["conv2_b"]
        h = jax.nn.relu(h)
        h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (2, 2), (2, 2), "VALID")

        # Flatten
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)

        # Dense layers
        h = jnp.dot(h, params["fc1_w"]) + params["fc1_b"]
        h = jax.nn.relu(h)
        h = jnp.dot(h, params["fc2_w"]) + params["fc2_b"]

        if squeeze_output:
            h = h[0]

        return h

    return forward, params


def create_random_input(shape: Tuple[int, ...], seed: int = 123) -> jnp.ndarray:
    """Create random input data for testing.

    Args:
        shape: Shape of input tensor
        seed: Random seed

    Returns:
        Random input tensor
    """
    key = random.PRNGKey(seed)
    return random.normal(key, shape)


def export_to_stablehlo(
    forward_fn: callable,
    params: Dict[str, jnp.ndarray],
    input_shape: Tuple[int, ...],
    use_vmap: bool = False,
) -> str:
    """Export a JAX model to StableHLO text format.

    Args:
        forward_fn: Forward function taking (params, x)
        params: Model parameters
        input_shape: Shape of single input (without batch)
        use_vmap: Whether to vmap the function for batched execution

    Returns:
        StableHLO module as text
    """
    # Create the function to export
    if use_vmap:
        # vmap over the batch dimension (first axis of input)
        batched_fn = jax.vmap(forward_fn, in_axes=(None, 0))
        # For vmapped version, we need a batch dimension
        example_input = jnp.zeros((2,) + input_shape)  # batch size of 2 for example
        fn_to_export = lambda x: batched_fn(params, x)
    else:
        example_input = jnp.zeros(input_shape)
        fn_to_export = lambda x: forward_fn(params, x)

    # Lower to StableHLO
    lowered = jax.jit(fn_to_export).lower(example_input)

    # Get the StableHLO module as text
    stablehlo_module = lowered.as_text()

    return stablehlo_module
