"""
Prover implementation using Veritor data models.

This module implements the prover that executes computations and produces
standardized proof bundles containing graphs, traces, and data.
"""

import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.profiler
from jax import random
from jax.experimental import io_callback

from ..db.api import WorkloadDatabase
from ..db.models import (
    ChallengeRecord,
    DataBundle,
    EventType,
    Graph,
    TensorData,
    Trace,
    TraceEvent,
)


@dataclass
class ProverConfig:
    """Configuration for the prover"""

    enable_profiling: bool = True
    enable_challenges: bool = True
    challenge_probability: float = 0.3
    projection_dim: int = 16


class HookSystem:
    """Hook system for capturing challenges during execution"""

    def __init__(self, config: ProverConfig):
        self.config = config
        self.challenge_schedule = {}
        self.recorded_challenges = []

    def set_challenge_schedule(self, schedule: Dict[str, bool]):
        """Set which operations should be challenged"""
        self.challenge_schedule = schedule

    def should_challenge(self, step: int, layer: int, batch_idx: int) -> bool:
        """Check if this operation should be challenged"""
        op_id = f"{step}_{layer}_{batch_idx}"
        return self.challenge_schedule.get(op_id, False)

    def challenge_hook(
        self,
        state: jnp.ndarray,
        step: int,
        layer: int,
        batch_idx: int,
        projection_seed: int,
    ):
        """Hook called during forward pass to record challenges"""
        # Compute the projection
        key = random.PRNGKey(projection_seed)
        proj_matrix = self._create_projection_matrix(
            key, state.shape[0], self.config.projection_dim
        )
        projection = self._apply_projection(state, proj_matrix)

        # Record the challenge
        challenge = ChallengeRecord(
            id=f"challenge_S{step}_L{layer}_B{batch_idx}",
            challenge_type="activation_lsh",
            timestamp=datetime.now().timestamp(),
            seed=projection_seed,
            projection_dim=self.config.projection_dim,
            target_operation_id=f"layer_{layer}_step_{step}",
            response_value=projection,
            metadata={"step": step, "layer": layer, "batch_idx": batch_idx},
        )
        self.recorded_challenges.append(challenge)

        return state  # Pass through unchanged

    def _create_projection_matrix(self, key, original_dim, reduced_dim):
        """Create orthogonal projection matrix"""
        proj_matrix = random.normal(key, (reduced_dim, original_dim))
        q, _ = jnp.linalg.qr(proj_matrix.T)
        proj_matrix = q[:, :reduced_dim].T
        return proj_matrix

    def _apply_projection(self, tensor, projection_matrix):
        """Apply projection with Johnson-Lindenstrauss scaling"""
        flat_tensor = tensor.flatten()
        projected = jnp.dot(projection_matrix, flat_tensor)

        original_dim = flat_tensor.shape[0]
        reduced_dim = projection_matrix.shape[0]
        scale = jnp.sqrt(original_dim / reduced_dim)

        return projected * scale


class ModelExecutor:
    """Executes a model and captures its computational graph"""

    def __init__(self, model, config: ProverConfig):
        self.model = model
        self.config = config
        self.hook_system = HookSystem(config)

    def build_graph(self, input_shape, n_layers: int) -> Graph:
        """Build the computational graph for the model"""
        # TODO: This method needs to be refactored to use IR-based graphs
        # For now, returning a simple Graph reference
        return Graph(
            id=f"model_graph_{uuid.uuid4().hex[:8]}",
            metadata={"model_type": "feedforward", "n_layers": n_layers},
        )

    def execute_with_trace(
        self, x: jnp.ndarray, step: int, batch_idx: int
    ) -> Tuple[jnp.ndarray, List[TraceEvent]]:
        """Execute model and capture trace events"""
        events = []
        start_time = datetime.now().timestamp()

        # Record execution start
        events.append(
            TraceEvent(
                timestamp=start_time,
                event_type=EventType.KERNEL_LAUNCH,
                device_id="device_0",
                operation_id=f"exec_S{step}_B{batch_idx}",
                data={"step": step, "batch_idx": batch_idx},
            )
        )

        # Execute through layers with hooks
        for layer_idx, (w, b) in enumerate(self.model.hidden_weights):
            # Layer computation
            x = jnp.matmul(x, w) + b
            x = jax.nn.relu(x)

            # Record layer execution
            events.append(
                TraceEvent(
                    timestamp=datetime.now().timestamp(),
                    event_type=EventType.KERNEL_LAUNCH,
                    device_id="device_0",
                    operation_id=f"layer_{layer_idx}",
                    data={"layer": layer_idx, "step": step},
                )
            )

            # Check if we should challenge this layer
            if self.hook_system.should_challenge(step, layer_idx, batch_idx):
                projection_seed = step * 1000 + layer_idx * 100 + batch_idx

                # Wrapper for io_callback
                def hook_wrapper(state):
                    return self.hook_system.challenge_hook(
                        state, step, layer_idx, batch_idx, projection_seed
                    )

                x = io_callback(hook_wrapper, x, x)

                # Record challenge event
                events.append(
                    TraceEvent(
                        timestamp=datetime.now().timestamp(),
                        event_type=EventType.CHALLENGE,
                        device_id="device_0",
                        operation_id=f"layer_{layer_idx}",
                        challenge_data={
                            "type": "activation_lsh",
                            "seed": projection_seed,
                        },
                    )
                )

        # Output layer
        x = jnp.matmul(x, self.model.output_weight) + self.model.output_bias

        return x, events


class Prover:
    """
    The prover runs computations and writes to the WorkloadDatabase.
    """

    def __init__(self, model, database: WorkloadDatabase, config: ProverConfig = None):
        self.model = model
        self.database = database  # Use shared database
        self.config = config or ProverConfig()
        self.executor = ModelExecutor(model, self.config)

    def prove(self, input_data: jnp.ndarray, n_steps: int = 2) -> Dict[str, str]:
        """
        Run the proving protocol and write results to the database.

        Returns IDs of stored graph, trace, and data bundle.
        """
        batch_size = input_data.shape[0]

        print(f"\n{'=' * 60}")
        print(f"Prover running")
        print(f"{'=' * 60}")

        # Generate challenge schedule
        schedule = self._generate_schedule(
            n_steps, self.model.n_layers, batch_size, self.config.challenge_probability
        )
        print(
            f"Challenge schedule: {len([v for v in schedule.values() if v])} challenges"
        )
        self.executor.hook_system.set_challenge_schedule(schedule)

        # Build computational graph
        graph = self.executor.build_graph(
            input_shape=input_data[0].shape, n_layers=self.model.n_layers
        )
        graph_id = self.database.store_graph(graph)

        # Set up profiling
        profile_dir = tempfile.mkdtemp(prefix="prover_")
        if self.config.enable_profiling:
            jax.profiler.start_trace(profile_dir)

        # Collect execution data
        all_events = []
        input_tensors = {}
        output_tensors = {}
        activation_tensors = {}

        trace_start = datetime.now()

        try:
            for step in range(n_steps):
                for batch_idx in range(batch_size):
                    # Get input
                    x = input_data[batch_idx]
                    input_id = f"input_S{step}_B{batch_idx}"
                    input_tensors[input_id] = TensorData.from_array(x)

                    # Execute and capture trace
                    output, events = self.executor.execute_with_trace(
                        x, step, batch_idx
                    )
                    all_events.extend(events)

                    # Store output
                    output_id = f"output_S{step}_B{batch_idx}"
                    output_tensors[output_id] = TensorData.from_array(output)

        finally:
            if self.config.enable_profiling:
                jax.profiler.stop_trace()

        trace_end = datetime.now()

        # Create and store trace
        trace = Trace(
            id=f"trace_{uuid.uuid4().hex[:8]}",
            graph_id=graph_id,
            start_time=trace_start,
            end_time=trace_end,
            events=all_events,
            metadata={
                "n_steps": n_steps,
                "batch_size": batch_size,
                "profile_dir": profile_dir,
            },
        )
        trace_id = self.database.store_trace(trace)

        # Store challenges
        for challenge in self.executor.hook_system.recorded_challenges:
            self.database.store_challenge(challenge)

        # Create and store data bundle
        data_bundle = DataBundle(
            id=f"data_{uuid.uuid4().hex[:8]}",
            graph_id=graph_id,
            inputs=input_tensors,
            outputs=output_tensors,
            weights={
                f"weight_{i}": TensorData.from_array(w)
                for i, (w, _) in enumerate(self.model.hidden_weights)
            },
            activations=activation_tensors,
            metadata={"trace_id": trace_id},
        )
        data_id = self.database.store_data_bundle(data_bundle)

        print(f"✓ Created proof bundle")
        print(f"  Graph ID: {graph_id}")
        print(f"  Trace ID: {trace_id}")
        print(f"  Data ID: {data_id}")
        print(f"  Total events: {len(all_events)}")
        print(
            f"  Total challenges: {len(self.executor.hook_system.recorded_challenges)}"
        )

        if self.config.enable_profiling:
            print(f"  Profile saved to: {profile_dir}")

        return {"graph_id": graph_id, "trace_id": trace_id, "data_id": data_id}

    def _generate_schedule(
        self, n_steps: int, n_layers: int, batch_size: int, challenge_prob: float
    ) -> Dict[str, bool]:
        """Generate deterministic challenge schedule"""
        key = random.PRNGKey(42)
        schedule = {}

        for step in range(n_steps):
            for layer in range(n_layers):
                for batch_idx in range(batch_size):
                    key, subkey = random.split(key)
                    if random.bernoulli(subkey, p=challenge_prob):
                        op_id = f"{step}_{layer}_{batch_idx}"
                        schedule[op_id] = True

        return schedule
