"""
Base Prover class for the three-party verification architecture.

This module provides the base Prover abstraction that handles:
- Workload data storage
- Graph compilation and registration
- Challenger interaction orchestration
- Execution management
- Database population

Subclasses should implement their specific workload logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import io_callback

from veritor.challenger import Challenger, ChallengeResponse
from veritor.common.operation_mapping import OperationIDMapper
from veritor.db.models import ChallengeRecord


@dataclass
class ProverConfig:
    """Base configuration for Prover instances."""

    seed: int = 42
    fixed_projection_dim: int = 4  # Fixed for JAX compilation


class BaseProver(ABC):
    """
    Base Prover class that handles common functionality.

    Subclasses implement specific workloads while this base class handles:
    - Challenge hook infrastructure
    - Outfeed management
    - Database interaction
    - Challenger communication
    """

    def __init__(self, config: ProverConfig, challenger: Optional[Challenger] = None):
        """
        Initialize the base Prover.

        Args:
            config: Prover configuration
            challenger: Optional Challenger entity (required for three-party mode)
        """
        self.config = config
        self.challenger = challenger
        self.op_mapper = OperationIDMapper()
        self.challenge_responses: List[ChallengeRecord] = []
        self.outfeed_buffer: List[Dict[str, Any]] = []
        self.jitted_workload: Optional[Callable] = None
        self._challenge_enabled = challenger is not None

    @abstractmethod
    def build_workload(self) -> Callable:
        """
        Build the workload computation.

        Subclasses implement this to define their specific workload.
        Should return a function that takes inputs and returns outputs.

        Returns:
            Callable that implements the workload
        """
        pass

    @abstractmethod
    def get_workload_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the workload.

        Returns:
            Dictionary with workload-specific metadata
        """
        pass

    def register_operation(self, operation_name: str) -> str:
        """
        Register an operation for potential challenging.

        Args:
            operation_name: Name of the operation

        Returns:
            The operation ID
        """
        return self.op_mapper.register_operation(operation_name)

    def query_challenge(self, activation: jnp.ndarray, operation_id: str) -> jnp.ndarray:
        """
        Query the Challenger for a challenge decision.

        This is an IMPURE operation that uses io_callback.

        Args:
            activation: The activation tensor at this point
            operation_id: The operation ID being executed

        Returns:
            JAX array [should_challenge, seed, projection_dim]
        """
        if not self._challenge_enabled:
            return jnp.zeros(3)

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

    def compute_challenge_response(
        self, activation: jnp.ndarray, seed: jnp.ndarray, projection_dim: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the challenge response (LSH projection).

        This is a PURE computation that will be visible in the StableHLO.

        Args:
            activation: The activation to challenge
            seed: Random seed for projection
            projection_dim: Dimension for projection (unused, fixed for simplicity)

        Returns:
            The challenge response
        """
        from jax import random

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

    def outfeed(self, data: jnp.ndarray, operation_id: str, data_type: str):
        """
        Write output via outfeed (simulated with io_callback).

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
        Compile the workload with embedded challenge hooks.

        Returns:
            Tuple of (stablehlo_text, operation_mapping, jitted_function)
        """
        print("\n📝 Compiling workload...")

        # Build the workload
        workload_fn = self.build_workload()

        # JIT compile
        self.jitted_workload = jax.jit(workload_fn)

        # Generate StableHLO using example input
        example_input = self.get_example_input()
        lowered = self.jitted_workload.lower(example_input)
        stablehlo_text = lowered.as_text(dialect="stablehlo")

        print(f"  Generated StableHLO: {len(stablehlo_text)} bytes")
        print(f"  Registered {len(self.op_mapper.operation_registry)} operations")

        return stablehlo_text, self.op_mapper.get_registry(), self.jitted_workload

    @abstractmethod
    def get_example_input(self) -> Any:
        """
        Get an example input for StableHLO generation.

        Returns:
            Example input tensor(s)
        """
        pass

    def execute(self, *inputs) -> Any:
        """
        Execute the compiled workload.

        Args:
            inputs: Input tensors

        Returns:
            Output from the workload
        """
        if self.jitted_workload is None:
            raise RuntimeError("Workload not compiled. Call compile_workload() first.")

        # Clear outfeed buffer for this execution
        self.outfeed_buffer.clear()

        # Execute
        output = self.jitted_workload(*inputs)

        # Process outfeed buffer to extract challenges
        self._process_outfeed()

        return output

    def _process_outfeed(self):
        """Process the outfeed buffer to extract challenge responses."""
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about prover execution."""
        stats = {
            "operations_registered": len(self.op_mapper.operation_registry),
            "challenges_responded": len(self.challenge_responses),
            "outfeed_operations": len(self.outfeed_buffer),
            "challenge_enabled": self._challenge_enabled
        }

        # Add workload-specific metadata
        stats.update(self.get_workload_metadata())

        return stats


def challengeable(prover: BaseProver, operation_name: str):
    """
    Decorator/helper to make an operation challengeable.

    This wraps an operation with challenge hooks that:
    1. Query the challenger for a decision (impure)
    2. Conditionally compute a challenge response (pure)
    3. Outfeed the response (side-effect)

    Usage:
        @challengeable(prover, "layer_0_linear")
        def linear_op(h):
            return jnp.dot(h, w) + b

    Args:
        prover: The Prover instance
        operation_name: Name of the operation

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapped(activation, *args, **kwargs):
            # Execute the operation
            result = func(activation, *args, **kwargs)

            # Register and get operation ID
            operation_id = prover.register_operation(operation_name)

            # Query challenger
            decision = prover.query_challenge(result, operation_id)
            should_challenge = decision[0] > 0.5
            seed = jnp.int32(decision[1])
            proj_dim = jnp.int32(decision[2])

            # Conditional challenge computation
            challenge_response = jax.lax.cond(
                should_challenge,
                lambda: prover.compute_challenge_response(result, seed, proj_dim),
                lambda: jnp.zeros((result.shape[0], prover.config.fixed_projection_dim)),
            )

            # Outfeed challenge response
            prover.outfeed(challenge_response, operation_id, "challenge_response")

            return result

        return wrapped

    return decorator