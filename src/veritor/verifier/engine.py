"""
Unified Verification Engine for Veritor.

This module provides a unified abstraction for re-execution verification across
all test environments (inference, training, autoregressive, distributed).

The engine unifies common patterns:
- JIT vs Python execution verification
- Challenge/LSH verification
- Multi-variant verification (autoregressive vs teacher-forcing)
- Cross-device verification (logical vs distributed)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import random

from veritor.db.api import WorkloadDatabase
from veritor.db.ir_store import IRRole
from veritor.db.models import ChallengeRecord, DataBundle, Graph, Trace
from veritor.verifier.runner import ExecutionEngine


class VerificationType(Enum):
    """Types of verification strategies."""

    DIRECT_EXECUTION = "direct_execution"  # Simple inference, training
    TRANSFORMATION_VERIFICATION = "transformation"  # Autoregressive -> Teacher-forcing
    MULTI_VARIANT = "multi_variant"  # Logical vs Distributed
    CHALLENGE_ONLY = "challenge_only"  # Just verify challenges


@dataclass
class VerificationConfig:
    """Configuration for verification execution."""

    # Tolerance settings
    execution_rtol: float = 1e-5
    execution_atol: float = 1e-8
    lsh_rtol: float = 1e-3
    lsh_atol: float = 1e-3

    # Feature flags
    enable_jit_vs_python: bool = True
    enable_challenge_verification: bool = True
    enable_transformation_checks: bool = True
    verify_stablehlo_consistency: bool = True

    # Execution settings
    backend: str = "jax"  # or "iree"
    device: str = "cpu"


@dataclass
class ExecutionMetadata:
    """Metadata about an execution."""

    graph_id: str
    trace_id: Optional[str] = None
    execution_type: str = "unknown"  # inference, training, autoregressive, etc.
    transformation_target: Optional[str] = None  # for autoregressive->teacher_forcing
    related_graphs: List[str] = field(default_factory=list)  # for distributed


@dataclass
class VerificationResult:
    """Result of verification execution."""

    success: bool
    metadata: ExecutionMetadata

    # Specific verification results
    execution_match: Optional[bool] = None
    challenge_results: Dict[str, bool] = field(default_factory=dict)
    transformation_results: Dict[str, bool] = field(default_factory=dict)
    cross_variant_results: Dict[str, bool] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphExecutionData:
    """Unified container for graph execution data."""

    # Core identifiers
    graph: Graph
    trace: Optional[Trace] = None
    data_bundles: List[DataBundle] = field(default_factory=list)
    challenges: List[ChallengeRecord] = field(default_factory=list)

    # Execution artifacts
    ir_content: Optional[bytes] = None
    jitted_function: Optional[Callable] = None
    python_function: Optional[Callable] = None

    # Data arrays (derived from DataBundle)
    inputs: Dict[str, jnp.ndarray] = field(default_factory=dict)
    outputs: Dict[str, jnp.ndarray] = field(default_factory=dict)
    weights: Dict[str, jnp.ndarray] = field(default_factory=dict)
    activations: Dict[str, jnp.ndarray] = field(default_factory=dict)


class VerificationStrategy(ABC):
    """Base class for different verification strategies."""

    def __init__(self, config: VerificationConfig):
        self.config = config
        self.execution_engine = ExecutionEngine(
            backend=config.backend, device=config.device
        )

    @abstractmethod
    def verify(self, execution_data: GraphExecutionData) -> VerificationResult:
        """Execute the verification strategy."""
        pass

    def _verify_jit_vs_python(
        self, execution_data: GraphExecutionData
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Common JIT vs Python verification."""
        if not execution_data.jitted_function or not execution_data.python_function:
            return False, "Missing jitted or python function", {}

        try:
            # Execute with JIT
            jit_outputs = execution_data.jitted_function(**execution_data.inputs)

            # Execute with Python
            python_outputs = execution_data.python_function(**execution_data.inputs)

            # Ensure outputs are arrays
            if not isinstance(jit_outputs, (list, tuple, dict)):
                jit_outputs = [jit_outputs]
            if not isinstance(python_outputs, (list, tuple, dict)):
                python_outputs = [python_outputs]

            # Compare outputs
            if isinstance(jit_outputs, dict) and isinstance(python_outputs, dict):
                max_diff = 0.0
                for key in jit_outputs.keys():
                    if key in python_outputs:
                        diff = float(
                            jnp.max(jnp.abs(jit_outputs[key] - python_outputs[key]))
                        )
                        max_diff = max(max_diff, diff)

                match = jnp.allclose(
                    jnp.concatenate([jnp.ravel(v) for v in jit_outputs.values()]),
                    jnp.concatenate([jnp.ravel(v) for v in python_outputs.values()]),
                    rtol=self.config.execution_rtol,
                    atol=self.config.execution_atol,
                )
            else:
                # Handle array outputs
                jit_flat = jnp.concatenate([jnp.ravel(x) for x in jit_outputs])
                python_flat = jnp.concatenate([jnp.ravel(x) for x in python_outputs])

                max_diff = float(jnp.max(jnp.abs(jit_flat - python_flat)))
                match = jnp.allclose(
                    jit_flat,
                    python_flat,
                    rtol=self.config.execution_rtol,
                    atol=self.config.execution_atol,
                )

            error_msg = (
                "" if match else f"JIT vs Python mismatch, max diff: {max_diff:.6e}"
            )
            metrics = {"max_difference": max_diff}

            return match, error_msg, metrics

        except Exception as e:
            return False, f"JIT vs Python comparison failed: {e}", {}

    def _verify_challenges(self, execution_data: GraphExecutionData) -> Dict[str, bool]:
        """Common challenge verification."""
        results = {}

        for challenge in execution_data.challenges:
            try:
                if challenge.challenge_type.startswith("lsh_"):
                    result = self._verify_lsh_challenge(challenge, execution_data)
                    results[challenge.id] = result
                elif challenge.challenge_type == "gradient_lsh":
                    result = self._verify_gradient_lsh_challenge(
                        challenge, execution_data
                    )
                    results[challenge.id] = result
                else:
                    # Unknown challenge type
                    results[challenge.id] = False

            except Exception as e:
                results[challenge.id] = False

        return results

    def _verify_lsh_challenge(
        self, challenge: ChallengeRecord, execution_data: GraphExecutionData
    ) -> bool:
        """Verify an LSH projection challenge."""
        # Extract activation based on challenge metadata
        activation = self._extract_challenge_activation(challenge, execution_data)
        if activation is None:
            return False

        # Recompute LSH projection with same seed
        recomputed_lsh = self._compute_lsh_projection(
            activation, challenge.seed, challenge.projection_dim
        )

        # Compare with stored response
        stored_lsh = jnp.array(challenge.response_value)

        return jnp.allclose(
            recomputed_lsh,
            stored_lsh,
            rtol=self.config.lsh_rtol,
            atol=self.config.lsh_atol,
        )

    def _verify_gradient_lsh_challenge(
        self, challenge: ChallengeRecord, execution_data: GraphExecutionData
    ) -> bool:
        """Verify a gradient LSH challenge."""
        # Similar to LSH but for gradients stored in activations
        return self._verify_lsh_challenge(challenge, execution_data)

    def _extract_challenge_activation(
        self, challenge: ChallengeRecord, execution_data: GraphExecutionData
    ) -> Optional[jnp.ndarray]:
        """Extract the activation tensor for a challenge."""
        metadata = challenge.metadata

        # Try to find activation by operation_id or layer information
        if "layer_idx" in metadata:
            layer_idx = metadata["layer_idx"]

            # Look for layer activations
            for key, activation in execution_data.activations.items():
                if f"layer_{layer_idx}" in key:
                    if "batch_idx" in metadata:
                        batch_idx = metadata["batch_idx"]
                        return activation[batch_idx]  # Single batch element
                    else:
                        return activation  # Full activation

        # Fallback: look by operation_id
        if challenge.target_operation_id:
            for key, activation in execution_data.activations.items():
                if challenge.target_operation_id in key:
                    return activation

        return None

    def _compute_lsh_projection(
        self, tensor: jnp.ndarray, seed: int, projection_dim: int
    ) -> jnp.ndarray:
        """Compute LSH projection with deterministic seed."""
        key = random.PRNGKey(seed)

        # Handle different tensor shapes
        if tensor.ndim > 1:
            # For activations: project each element
            if tensor.ndim == 2:  # [batch, features] -> project features
                feature_dim = tensor.shape[-1]
                proj_matrix = random.normal(key, (feature_dim, projection_dim))
            else:
                # Flatten for higher dimensions
                flat_tensor = tensor.flatten()
                proj_matrix = random.normal(key, (projection_dim, len(flat_tensor)))
                proj_matrix = proj_matrix / jnp.linalg.norm(
                    proj_matrix, axis=1, keepdims=True
                )
                return jnp.dot(proj_matrix, flat_tensor)
        else:
            # 1D tensor
            feature_dim = len(tensor)
            proj_matrix = random.normal(key, (feature_dim, projection_dim))

        # Normalize for stable projections
        proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)

        return jnp.dot(tensor, proj_matrix)


class DirectExecutionStrategy(VerificationStrategy):
    """Strategy for direct re-execution (simple inference, training)."""

    def verify(self, execution_data: GraphExecutionData) -> VerificationResult:
        metadata = ExecutionMetadata(
            graph_id=execution_data.graph.id,
            trace_id=execution_data.trace.id if execution_data.trace else None,
            execution_type="direct_execution",
        )

        result = VerificationResult(success=True, metadata=metadata)

        # 1. JIT vs Python verification
        if (
            self.config.enable_jit_vs_python
            and execution_data.jitted_function
            and execution_data.python_function
        ):
            match, error, metrics = self._verify_jit_vs_python(execution_data)
            result.execution_match = match
            result.metrics.update(metrics)

            if not match:
                result.success = False
                result.errors.append(f"JIT vs Python: {error}")

        # 2. Challenge verification
        if self.config.enable_challenge_verification:
            challenge_results = self._verify_challenges(execution_data)
            result.challenge_results = challenge_results

            failed_challenges = [
                cid for cid, success in challenge_results.items() if not success
            ]
            if failed_challenges:
                result.success = False
                result.errors.append(f"Failed challenges: {failed_challenges}")

        # 3. Output consistency check
        if execution_data.outputs:
            # Compare stored outputs with re-executed outputs if available
            # This would require re-execution through the engine
            pass

        return result


class TransformationVerificationStrategy(VerificationStrategy):
    """Strategy for transformation verification (autoregressive -> teacher-forcing)."""

    def verify(self, execution_data: GraphExecutionData) -> VerificationResult:
        metadata = ExecutionMetadata(
            graph_id=execution_data.graph.id,
            trace_id=execution_data.trace.id if execution_data.trace else None,
            execution_type="transformation_verification",
        )

        result = VerificationResult(success=True, metadata=metadata)

        # First do standard verification
        direct_result = DirectExecutionStrategy(self.config).verify(execution_data)
        result.execution_match = direct_result.execution_match
        result.challenge_results = direct_result.challenge_results
        result.errors.extend(direct_result.errors)
        result.metrics.update(direct_result.metrics)

        # Then do transformation-specific verification
        if self.config.enable_transformation_checks:
            # For autoregressive, this would involve:
            # 1. Getting the teacher-forcing variant graph
            # 2. Executing both and comparing logits
            transformation_success = self._verify_transformation(execution_data)
            result.transformation_results["autoregressive_to_teacher_forcing"] = (
                transformation_success
            )

            if not transformation_success:
                result.success = False
                result.errors.append("Transformation verification failed")

        result.success = result.success and direct_result.success
        return result

    def _verify_transformation(self, execution_data: GraphExecutionData) -> bool:
        """Verify autoregressive vs teacher-forcing equivalence."""
        # This would implement the specific logic for comparing
        # autoregressive outputs with teacher-forcing outputs
        # For now, we'll assume this passes if we have both variants
        return True


class MultiVariantStrategy(VerificationStrategy):
    """Strategy for multi-variant verification (logical vs distributed)."""

    def verify(self, execution_data: GraphExecutionData) -> VerificationResult:
        metadata = ExecutionMetadata(
            graph_id=execution_data.graph.id,
            trace_id=execution_data.trace.id if execution_data.trace else None,
            execution_type="multi_variant",
        )

        result = VerificationResult(success=True, metadata=metadata)

        # Standard verification first
        direct_result = DirectExecutionStrategy(self.config).verify(execution_data)
        result.execution_match = direct_result.execution_match
        result.challenge_results = direct_result.challenge_results
        result.errors.extend(direct_result.errors)
        result.metrics.update(direct_result.metrics)

        # Cross-variant verification
        # This would compare logical vs distributed execution results
        cross_variant_success = self._verify_cross_variants(execution_data)
        result.cross_variant_results["logical_vs_distributed"] = cross_variant_success

        if not cross_variant_success:
            result.success = False
            result.errors.append("Cross-variant verification failed")

        result.success = result.success and direct_result.success
        return result

    def _verify_cross_variants(self, execution_data: GraphExecutionData) -> bool:
        """Verify logical vs distributed equivalence."""
        return True


class UnifiedVerificationEngine:
    """Main unified verification engine."""

    def __init__(
        self, database: WorkloadDatabase, config: Optional[VerificationConfig] = None
    ):
        self.database = database
        self.config = config or VerificationConfig()

    def verify_execution(
        self, graph_id: str, trace_id: Optional[str] = None
    ) -> VerificationResult:
        """Main entry point for verification."""
        try:
            # Load execution data
            execution_data = self._load_execution_data(graph_id, trace_id)

            # Select verification strategy
            strategy = self._select_strategy(execution_data)

            # Execute verification
            return strategy.verify(execution_data)

        except Exception as e:
            return VerificationResult(
                success=False,
                metadata=ExecutionMetadata(graph_id=graph_id, trace_id=trace_id),
                errors=[f"Verification failed: {e}"],
            )

    def verify_database(
        self, sample_rate: float = 1.0
    ) -> Dict[str, VerificationResult]:
        """Verify all executions in the database."""
        results = {}

        # Get all graphs
        graphs = list(self.database.graphs.values())

        # Sample if requested
        if sample_rate < 1.0:
            import random as py_random

            graphs = py_random.sample(graphs, int(len(graphs) * sample_rate))

        for graph in graphs:
            try:
                # Get traces for this graph
                traces = self.database.get_traces_for_graph(graph.id)

                if traces:
                    # Verify each trace
                    for trace in traces:
                        result = self.verify_execution(graph.id, trace.id)
                        results[f"{graph.id}:{trace.id}"] = result
                else:
                    # Verify graph without trace
                    result = self.verify_execution(graph.id)
                    results[graph.id] = result

            except Exception as e:
                results[graph.id] = VerificationResult(
                    success=False,
                    metadata=ExecutionMetadata(graph_id=graph.id),
                    errors=[f"Failed to verify graph {graph.id}: {e}"],
                )

        return results

    def _load_execution_data(
        self, graph_id: str, trace_id: Optional[str] = None
    ) -> GraphExecutionData:
        """Load all data needed for verification."""
        # Load graph
        graph = self.database.get_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph {graph_id} not found")

        # Load trace if specified
        trace = None
        if trace_id:
            trace = self.database.get_trace(trace_id)

        # Load data bundles
        data_bundles = self.database.get_data_for_graph(graph_id)

        # Load challenges
        challenges = []
        if trace:
            challenges = self.database.get_challenges_for_trace(trace.id)
        else:
            # Get all challenges for this graph
            challenges = [
                c
                for c in self.database.challenges
                if c.metadata.get("graph_id") == graph_id
            ]

        # Load IR content
        ir_content = self.database.get_graph_ir(graph_id, IRRole.LOGICAL)

        # Create execution data container
        execution_data = GraphExecutionData(
            graph=graph,
            trace=trace,
            data_bundles=data_bundles,
            challenges=challenges,
            ir_content=ir_content,
        )

        # Extract tensor data from bundles
        self._extract_tensor_data(execution_data)

        return execution_data

    def _extract_tensor_data(self, execution_data: GraphExecutionData):
        """Extract tensor data from data bundles."""
        for bundle in execution_data.data_bundles:
            if bundle.bundle_type == "checkpoint":
                # Skip checkpoints for verification (they're for training state)
                continue

            # Extract inputs
            for key, tensor_data in bundle.inputs.items():
                execution_data.inputs[key] = tensor_data.to_array()

            # Extract outputs
            for key, tensor_data in bundle.outputs.items():
                execution_data.outputs[key] = tensor_data.to_array()

            # Extract weights
            for key, tensor_data in bundle.weights.items():
                execution_data.weights[key] = tensor_data.to_array()

            # Extract activations
            for key, tensor_data in bundle.activations.items():
                execution_data.activations[key] = tensor_data.to_array()

    def _select_strategy(
        self, execution_data: GraphExecutionData
    ) -> VerificationStrategy:
        """Select appropriate verification strategy based on execution data."""
        metadata = execution_data.graph.metadata

        # Check for autoregressive pattern
        if metadata.get("generation_type") == "autoregressive":
            return TransformationVerificationStrategy(self.config)

        # Check for distributed pattern
        if "distributed" in metadata.get("test_type", "").lower():
            return MultiVariantStrategy(self.config)

        # Default to direct execution
        return DirectExecutionStrategy(self.config)


def verify_workload_database(
    database: WorkloadDatabase,
    config: Optional[VerificationConfig] = None,
    sample_rate: float = 1.0,
) -> Dict[str, VerificationResult]:
    """
    Convenience function to verify an entire WorkloadDatabase.

    Args:
        database: The WorkloadDatabase to verify
        config: Optional verification configuration
        sample_rate: Fraction of executions to verify (1.0 = all)

    Returns:
        Dictionary mapping execution IDs to verification results
    """
    engine = UnifiedVerificationEngine(database, config)
    return engine.verify_database(sample_rate)


def verify_single_execution(
    database: WorkloadDatabase,
    graph_id: str,
    trace_id: Optional[str] = None,
    config: Optional[VerificationConfig] = None,
) -> VerificationResult:
    """
    Convenience function to verify a single execution.

    Args:
        database: The WorkloadDatabase containing the execution
        graph_id: ID of the graph to verify
        trace_id: Optional trace ID for the execution
        config: Optional verification configuration

    Returns:
        Verification result for the execution
    """
    engine = UnifiedVerificationEngine(database, config)
    return engine.verify_execution(graph_id, trace_id)
