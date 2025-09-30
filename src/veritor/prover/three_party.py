"""
Three-party architecture support for the Prover.

This module provides Prover functionality for the three-party verification
architecture with proper separation between Prover, Challenger, and Verifier.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import io_callback

from veritor.challenger import Challenger, ChallengeResponse
from veritor.common.operation_mapping import OperationIDMapper
from veritor.db.models import ChallengeRecord


@dataclass
class ProverConfig:
    """Configuration for the three-party Prover."""

    n_layers: int = 4
    input_dim: int = 2
    hidden_dim: int = 8
    output_dim: int = 2
    batch_size: int = 3
    n_forward_passes: int = 5
    seed: int = 42
    fixed_projection_dim: int = 4  # Fixed for simplicity in JAX


class ThreePartyProver:
    """
    Prover implementation for the three-party architecture.

    The Prover:
    - Compiles workloads with embedded challenge hooks
    - Maintains operation ID mapping
    - Queries Challenger at hook points via io_callback
    - Uses outfeed for all outputs
    - Has no knowledge of the challenge schedule
    """

    def __init__(self, config: ProverConfig, challenger: Challenger):
        """
        Initialize the Prover.

        Args:
            config: Prover configuration
            challenger: The Challenger entity to query
        """
        self.config = config
        self.challenger = challenger
        self.op_mapper = OperationIDMapper()
        self.model_params: Dict[str, jnp.ndarray] = {}
        self.challenge_responses: List[ChallengeRecord] = []
        self.outfeed_buffer: List[Dict[str, Any]] = []
        self.jitted_workload: Optional[Callable] = None

    def initialize_model(self) -> Dict[str, jnp.ndarray]:
        """
        Initialize model parameters.

        Returns:
            Dictionary of model parameters
        """
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

        self.model_params = params
        return params

    def _challenge_decision_hook(
        self, activation: jnp.ndarray, operation_id: str
    ) -> jnp.ndarray:
        """
        IMPURE: Query the Challenger for a decision via io_callback.

        Args:
            activation: The activation tensor at this point
            operation_id: The operation ID being executed

        Returns:
            JAX array [should_challenge, seed, projection_dim]
        """

        def query_challenger(_):
            """External call to Challenger."""
            response: ChallengeResponse = self.challenger.query_challenge(operation_id)

            should_challenge = float(response.should_challenge)
            seed = float(response.seed if response.seed is not None else 0)
            projection_dim = float(
                response.projection_dim if response.projection_dim is not None else self.config.fixed_projection_dim
            )

            return jnp.array([should_challenge, seed, projection_dim])

        # Use io_callback for the impure operation
        decision = io_callback(
            query_challenger,
            jnp.zeros(3),  # result shape
            jnp.array(0.0),  # dummy input
        )

        return decision

    def _compute_challenge(
        self, activation: jnp.ndarray, seed: jnp.ndarray, projection_dim: jnp.ndarray
    ) -> jnp.ndarray:
        """
        PURE: Compute the challenge response given parameters.

        This computation is pure and will be visible in the StableHLO.

        Args:
            activation: The activation to challenge
            seed: Random seed for projection
            projection_dim: Dimension for projection (unused, fixed for simplicity)

        Returns:
            The challenge response (LSH projection)
        """
        # LSH projection with fixed dimension
        key = random.PRNGKey(seed)
        flat_dim = activation.shape[-1]
        proj_dim = self.config.fixed_projection_dim

        proj_matrix = random.normal(key, (flat_dim, proj_dim))
        proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)

        # Flatten activation if needed
        if activation.ndim > 2:
            activation = activation.reshape(activation.shape[0], -1)

        projection = jnp.dot(activation, proj_matrix)
        return projection

    def _outfeed_operation(self, data: jnp.ndarray, operation_id: str, data_type: str):
        """
        SIDE-EFFECT: Write output via outfeed.

        In production, this would use XLA outfeed. For now, we simulate
        by storing to buffer via io_callback.

        Args:
            data: The data to outfeed
            operation_id: The operation producing this output
            data_type: Type of data (challenge_response, workload_output, etc.)
        """

        def store_outfeed(data_array):
            """Store to outfeed buffer."""
            self.outfeed_buffer.append({
                "operation_id": operation_id,
                "data_type": data_type,
                "data": data_array,
                "timestamp": datetime.now().timestamp(),
            })
            return data_array

        # Use io_callback to simulate outfeed
        io_callback(store_outfeed, data, data)

    def compile_workload(self) -> Tuple[str, Dict[str, str], Callable]:
        """
        Compile the full workload with embedded challenge hooks.

        Returns:
            Tuple of (stablehlo_text, operation_mapping, jitted_function)
        """
        print("\n📝 Compiling workload with challenge hooks...")

        # Initialize model if not already done
        if not self.model_params:
            self.initialize_model()

        # Register all operations that will have hooks
        for i in range(self.config.n_layers):
            self.op_mapper.register_operation(f"layer_{i}_linear")
            if i < self.config.n_layers - 1:
                self.op_mapper.register_operation(f"layer_{i}_activation")

        # Define the workload computation
        def workload_computation(x: jnp.ndarray) -> jnp.ndarray:
            """The main workload with embedded hooks."""
            h = x

            for i in range(self.config.n_layers):
                # Linear transformation
                w = self.model_params[f"w_{i}"]
                b = self.model_params[f"b_{i}"]
                h = jnp.dot(h, w) + b

                # Challenge hook for linear output
                linear_op_id = self.op_mapper.get_operation_id(f"layer_{i}_linear")
                decision = self._challenge_decision_hook(h, linear_op_id)
                should_challenge = decision[0] > 0.5
                seed = jnp.int32(decision[1])
                proj_dim = jnp.int32(decision[2])

                # Conditional challenge computation
                challenge_response = jax.lax.cond(
                    should_challenge,
                    lambda: self._compute_challenge(h, seed, proj_dim),
                    lambda: jnp.zeros((h.shape[0], self.config.fixed_projection_dim)),
                )

                # Outfeed challenge response
                self._outfeed_operation(
                    challenge_response, linear_op_id, "challenge_response"
                )

                # Apply activation (except last layer)
                if i < self.config.n_layers - 1:
                    h = jax.nn.relu(h)

                    # Challenge hook for activation
                    activation_op_id = self.op_mapper.get_operation_id(
                        f"layer_{i}_activation"
                    )
                    decision = self._challenge_decision_hook(h, activation_op_id)
                    should_challenge = decision[0] > 0.5
                    seed = jnp.int32(decision[1])
                    proj_dim = jnp.int32(decision[2])

                    challenge_response = jax.lax.cond(
                        should_challenge,
                        lambda: self._compute_challenge(h, seed, proj_dim),
                        lambda: jnp.zeros((h.shape[0], self.config.fixed_projection_dim)),
                    )

                    self._outfeed_operation(
                        challenge_response, activation_op_id, "challenge_response"
                    )

            # Outfeed final output
            self._outfeed_operation(h, "final_output", "workload_output")

            return h

        # JIT compile the workload
        self.jitted_workload = jax.jit(workload_computation)

        # Generate StableHLO
        example_input = jnp.zeros((self.config.batch_size, self.config.input_dim))
        lowered = self.jitted_workload.lower(example_input)
        stablehlo_text = lowered.as_text(dialect="stablehlo")

        print(f"  Generated StableHLO: {len(stablehlo_text)} bytes")
        print(f"  Registered {len(self.op_mapper.operation_registry)} operations")

        return stablehlo_text, self.op_mapper.get_registry(), self.jitted_workload

    def execute_forward_pass(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Execute a single forward pass with challenge injection.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.jitted_workload is None:
            raise RuntimeError("Workload not compiled. Call compile_workload() first.")

        # Clear outfeed buffer for this pass
        self.outfeed_buffer.clear()

        # Execute with hooks
        output = self.jitted_workload(x)

        # Process outfeed buffer to extract challenges
        for outfeed in self.outfeed_buffer:
            if outfeed["data_type"] == "challenge_response":
                # Check if this is a real challenge (non-zero response)
                if jnp.any(outfeed["data"] != 0):
                    challenge = ChallengeRecord(
                        id=f"challenge_{outfeed['operation_id']}_{len(self.challenge_responses)}",
                        challenge_type="lsh_dynamic",
                        timestamp=outfeed["timestamp"],
                        target_operation_id=outfeed["operation_id"],
                        seed=0,  # Will be filled from schedule later
                        projection_dim=self.config.fixed_projection_dim,
                        response_value=outfeed["data"].tolist(),
                        metadata={}
                    )
                    self.challenge_responses.append(challenge)

        return output

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about prover execution."""
        return {
            "operations_registered": len(self.op_mapper.operation_registry),
            "challenges_responded": len(self.challenge_responses),
            "outfeed_operations": len(self.outfeed_buffer),
            "model_parameters": len(self.model_params)
        }