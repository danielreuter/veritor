"""
Simple inference Prover implementation.

This module demonstrates a concrete Prover subclass that implements
a simple multi-layer neural network inference workload.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
from jax import random

from veritor.prover.base import BaseProver, ProverConfig, challengeable


@dataclass
class SimpleInferenceConfig(ProverConfig):
    """Configuration for simple inference Prover."""

    n_layers: int = 4
    input_dim: int = 2
    hidden_dim: int = 8
    output_dim: int = 2
    batch_size: int = 3


class SimpleInferenceProver(BaseProver):
    """
    Prover implementation for simple neural network inference.

    This demonstrates how to extend the base Prover class with
    a specific workload while using the challengeable decorator.
    """

    def __init__(self, config: SimpleInferenceConfig, challenger=None):
        """
        Initialize the simple inference Prover.

        Args:
            config: Configuration for the inference workload
            challenger: Optional Challenger for three-party mode
        """
        super().__init__(config, challenger)
        self.config: SimpleInferenceConfig = config
        self.model_params = self._initialize_model()

    def _initialize_model(self) -> Dict[str, jnp.ndarray]:
        """Initialize model parameters."""
        key = random.PRNGKey(self.config.seed)
        params = {}

        dims = (
            [self.config.input_dim]
            + [self.config.hidden_dim] * (self.config.n_layers - 1)
            + [self.config.output_dim]
        )

        for i in range(len(dims) - 1):
            key, w_key, b_key = random.split(key, 3)
            params[f"w_{i}"] = random.normal(w_key, (dims[i], dims[i + 1])) * 0.1
            params[f"b_{i}"] = random.normal(b_key, (dims[i + 1],)) * 0.01

        return params

    def build_workload(self) -> Callable:
        """
        Build the inference workload with challengeable operations.

        Returns:
            Callable that performs inference
        """
        # Pre-register all operations
        for i in range(self.config.n_layers):
            self.register_operation(f"layer_{i}_linear")
            if i < self.config.n_layers - 1:
                self.register_operation(f"layer_{i}_activation")

        def inference_workload(x: jnp.ndarray) -> jnp.ndarray:
            """Forward pass through the network with challenge hooks."""
            h = x

            for i in range(self.config.n_layers):
                # Linear transformation with challenge hook
                w = self.model_params[f"w_{i}"]
                b = self.model_params[f"b_{i}"]

                # Apply linear and make it challengeable
                linear_op_id = self.op_mapper.get_operation_id(f"layer_{i}_linear")
                h = jnp.dot(h, w) + b

                # Challenge hook for linear output
                decision = self.query_challenge(h, linear_op_id)
                should_challenge = decision[0] > 0.5
                seed = jnp.int32(decision[1])
                proj_dim = jnp.int32(decision[2])

                challenge_response = jax.lax.cond(
                    should_challenge,
                    lambda: self.compute_challenge_response(h, seed, proj_dim),
                    lambda: jnp.zeros((h.shape[0], self.config.fixed_projection_dim)),
                )

                self.outfeed(challenge_response, linear_op_id, "challenge_response")

                # Apply activation (except last layer)
                if i < self.config.n_layers - 1:
                    h = jax.nn.relu(h)

                    # Challenge hook for activation
                    activation_op_id = self.op_mapper.get_operation_id(f"layer_{i}_activation")
                    decision = self.query_challenge(h, activation_op_id)
                    should_challenge = decision[0] > 0.5
                    seed = jnp.int32(decision[1])
                    proj_dim = jnp.int32(decision[2])

                    challenge_response = jax.lax.cond(
                        should_challenge,
                        lambda: self.compute_challenge_response(h, seed, proj_dim),
                        lambda: jnp.zeros((h.shape[0], self.config.fixed_projection_dim)),
                    )

                    self.outfeed(challenge_response, activation_op_id, "challenge_response")

            # Outfeed final output
            self.outfeed(h, "final_output", "workload_output")

            return h

        return inference_workload

    def get_example_input(self) -> jnp.ndarray:
        """Get example input for StableHLO generation."""
        return jnp.zeros((self.config.batch_size, self.config.input_dim))

    def get_workload_metadata(self) -> Dict[str, Any]:
        """Get metadata about the inference workload."""
        return {
            "workload_type": "simple_inference",
            "n_layers": self.config.n_layers,
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "output_dim": self.config.output_dim,
            "model_parameters": len(self.model_params)
        }