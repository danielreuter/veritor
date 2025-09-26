"""
Test simple non-autoregressive inference with LSH projections and challenges.

This test:
1. Creates a simple K-layer model with 2D input/output
2. Runs N forward passes with batch size B
3. Statically computes LSH at second-to-last layer every time
4. Dynamically challenges random layers via hooks
5. Stores everything in the database for verification

CRITICAL INVARIANT:
- The StableHLO stored in the database MUST exactly match what's executed
- Core computation (JIT-compiled) is separate from verification hooks
- LSH projections and challenges happen OUTSIDE the verified graph
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
from jax import random

from veritor.db.ir_store import IRFormat, IRRole
from veritor.db.models import (
    ChallengeRecord,
    DataBundle,
    EventType,
    Graph,
    TensorData,
    Trace,
    TraceEvent,
)


@dataclass
class SimpleInferenceConfig:
    """Configuration for simple inference test."""

    n_forward_passes: int = 5  # Number of forward passes
    n_layers: int = 4  # Number of layers (K)
    batch_size: int = 3  # Batch size (B)
    input_dim: int = 2  # Input dimension
    hidden_dim: int = 8  # Hidden dimension
    output_dim: int = 2  # Output dimension
    lsh_dim: int = 4  # LSH projection dimension
    challenge_prob: float = 0.3  # Probability of challenging a layer
    seed: int = 42


class SimpleModel:
    """Simple K-layer feedforward model."""

    def __init__(self, config: SimpleInferenceConfig):
        self.config = config
        key = random.PRNGKey(config.seed)

        # Initialize weights for K layers
        self.weights = []
        dims = (
            [config.input_dim]
            + [config.hidden_dim] * (config.n_layers - 1)
            + [config.output_dim]
        )

        for i in range(len(dims) - 1):
            key, w_key, b_key = random.split(key, 3)
            w = random.normal(w_key, (dims[i], dims[i + 1])) * 0.1
            b = random.normal(b_key, (dims[i + 1],)) * 0.01
            self.weights.append((w, b))

        # LSH projection matrix for second-to-last layer
        key, lsh_key = random.split(key)
        self.lsh_matrix = random.normal(lsh_key, (config.hidden_dim, config.lsh_dim))
        # Normalize rows for stable projections
        self.lsh_matrix = self.lsh_matrix / jnp.linalg.norm(
            self.lsh_matrix, axis=1, keepdims=True
        )

    def forward(
        self, x: jnp.ndarray, capture_activations: bool = True
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Forward pass through the model.

        Returns:
            output: Final output [batch_size, output_dim]
            activations: Dict of intermediate activations if capture_activations=True
        """
        activations = {}
        h = x

        for i, (w, b) in enumerate(self.weights):
            h = jnp.dot(h, w) + b

            # Apply ReLU except for last layer
            if i < len(self.weights) - 1:
                h = jax.nn.relu(h)

            if capture_activations:
                activations[f"layer_{i}"] = h

            # Static LSH at second-to-last layer
            if i == len(self.weights) - 2:
                lsh_projection = jnp.dot(h, self.lsh_matrix)
                activations["lsh_static"] = lsh_projection

        return h, activations


class ChallengeHookSystem:
    """Hook system for dynamic challenges during inference."""

    def __init__(self, config: SimpleInferenceConfig):
        self.config = config
        self.challenges = []
        self.challenge_schedule = {}

    def generate_challenge_schedule(self, n_passes: int) -> dict[str, bool]:
        """Generate random challenge schedule."""
        key = random.PRNGKey(self.config.seed + 1000)
        schedule = {}

        for pass_idx in range(n_passes):
            for batch_idx in range(self.config.batch_size):
                # Randomly select layers to challenge (not the last layer)
                for layer_idx in range(self.config.n_layers - 1):
                    key, subkey = random.split(key)
                    if random.bernoulli(subkey, p=self.config.challenge_prob):
                        challenge_id = (
                            f"pass_{pass_idx}_batch_{batch_idx}_layer_{layer_idx}"
                        )
                        schedule[challenge_id] = True

        self.challenge_schedule = schedule
        return schedule

    def compute_lsh_challenge(
        self, activation: jnp.ndarray, pass_idx: int, batch_idx: int, layer_idx: int
    ) -> jnp.ndarray:
        """Compute LSH projection for a challenge."""
        # Generate deterministic projection matrix based on indices
        seed = pass_idx * 10000 + batch_idx * 100 + layer_idx
        key = random.PRNGKey(seed)

        # Create projection matrix
        act_dim = activation.shape[-1]
        proj_matrix = random.normal(key, (act_dim, self.config.lsh_dim))
        proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)

        # Project
        projection = jnp.dot(activation, proj_matrix)

        # Record challenge
        challenge = ChallengeRecord(
            id=f"challenge_p{pass_idx}_b{batch_idx}_l{layer_idx}",
            challenge_type="lsh_dynamic",
            timestamp=datetime.now().timestamp(),
            target_operation_id=f"layer_{layer_idx}",
            seed=seed,
            projection_dim=self.config.lsh_dim,
            response_value=projection.tolist(),
            metadata={
                "pass_idx": pass_idx,
                "batch_idx": batch_idx,
                "layer_idx": layer_idx,
                "activation_shape": activation.shape,
                "trace_id": "",  # Will be updated later
            },
        )
        self.challenges.append(challenge)

        return projection


def test_simple_inference_with_real_stablehlo(workload_db):
    """Test simple inference with real JAX-generated StableHLO."""
    database = workload_db

    # Configuration
    config = SimpleInferenceConfig(
        n_forward_passes=5,
        n_layers=4,
        batch_size=3,
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        lsh_dim=4,
    )

    # Initialize
    model = SimpleModel(config)
    hook_system = ChallengeHookSystem(config)

    # Generate challenge schedule
    schedule = hook_system.generate_challenge_schedule(config.n_forward_passes)
    n_challenges = len([v for v in schedule.values() if v])
    assert n_challenges > 0, "Should have some challenges planned"

    # Create graph metadata
    graph = Graph(
        id=f"simple_inference_{uuid.uuid4().hex[:8]}",
        metadata={
            "model_type": "simple_feedforward",
            "n_layers": config.n_layers,
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "test_type": "simple_inference",
        },
    )
    graph_id = database.store_graph(graph)

    # Generate real StableHLO from the JAX model
    # This function MUST match exactly what we execute during forward passes
    def model_core_forward(x):
        """
        Core forward pass - this is the EXACT computation we verify.
        Returns only the final output, no intermediate activations.
        """
        h = x
        for i, (w, b) in enumerate(model.weights):
            h = jnp.dot(h, w) + b
            # Apply ReLU except for last layer
            if i < len(model.weights) - 1:
                h = jax.nn.relu(h)
        return h

    # This is the function we'll actually JIT and execute
    jitted_forward = jax.jit(model_core_forward)

    # Create example input for shape inference
    example_input = jnp.zeros((config.batch_size, config.input_dim))

    # Lower to StableHLO - this is our source of truth
    lowered = jitted_forward.lower(example_input)
    stablehlo_text = lowered.as_text()

    # Store the jitted function for later use
    model.jitted_forward = jitted_forward

    # Verify we generated real StableHLO
    assert len(stablehlo_text) > 1000, "StableHLO seems too short"
    assert "stablehlo.constant" in stablehlo_text, "Missing weight constants"
    assert "stablehlo.dot_general" in stablehlo_text, "Missing matrix operations"
    assert "func.func public @main" in stablehlo_text, "Missing main function"

    # Store the real StableHLO IR
    database.ir_store.attach_ir(
        graph_id,
        IRRole.LOGICAL,
        stablehlo_text,
        IRFormat.STABLEHLO,
        {
            "generated_from": "test_simple_inference",
            "jax_version": jax.__version__,
            "model_type": "simple_feedforward",
            "n_layers": config.n_layers,
            "weights_embedded": True,  # Weights are constants in the graph
        },
    )

    # Storage for results
    all_inputs = {}
    all_outputs = {}
    all_activations = {}
    all_lsh_static = {}
    all_events = []

    # Run N forward passes
    for pass_idx in range(config.n_forward_passes):
        # Generate random input batch
        key = random.PRNGKey(config.seed + pass_idx)
        x = random.normal(key, (config.batch_size, config.input_dim))

        # Store input
        input_id = f"input_pass_{pass_idx}"
        all_inputs[input_id] = TensorData.from_array(x)

        # Execute using the SAME JIT-compiled function that we stored as StableHLO
        output = model.jitted_forward(x)

        # Store output
        output_id = f"output_pass_{pass_idx}"
        all_outputs[output_id] = TensorData.from_array(output)

        # Now compute activations and LSH OUTSIDE the verified computation
        # These are for challenges/verification, not part of the core graph
        activations = {}
        h = x
        for i, (w, b) in enumerate(model.weights):
            h = jnp.dot(h, w) + b
            if i < len(model.weights) - 1:
                h = jax.nn.relu(h)
            activations[f"layer_{i}"] = h

            # Static LSH at second-to-last layer (computed separately)
            if i == len(model.weights) - 2:
                lsh_projection = jnp.dot(h, model.lsh_matrix)
                activations["lsh_static"] = lsh_projection

        # Store static LSH projection
        lsh_id = f"lsh_static_pass_{pass_idx}"
        all_lsh_static[lsh_id] = TensorData.from_array(activations["lsh_static"])

        # Process dynamic challenges
        for batch_idx in range(config.batch_size):
            for layer_idx in range(config.n_layers - 1):
                challenge_id = f"pass_{pass_idx}_batch_{batch_idx}_layer_{layer_idx}"

                if schedule.get(challenge_id, False):
                    # Extract activation for this batch element
                    layer_act = activations[f"layer_{layer_idx}"][batch_idx]

                    # Compute LSH challenge
                    lsh_proj = hook_system.compute_lsh_challenge(
                        layer_act, pass_idx, batch_idx, layer_idx
                    )

                    # Record event
                    event = TraceEvent(
                        timestamp=datetime.now().timestamp(),
                        event_type=EventType.CHALLENGE,
                        device_id="cpu_0",
                        operation_id=f"layer_{layer_idx}",
                        data={
                            "pass_idx": pass_idx,
                            "batch_idx": batch_idx,
                            "challenge_type": "lsh_dynamic",
                        },
                    )
                    all_events.append(event)

        # Store activations for this pass
        for name, act in activations.items():
            act_id = f"{name}_pass_{pass_idx}"
            all_activations[act_id] = TensorData.from_array(act)

    # Create and store trace
    trace = Trace(
        id=f"trace_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        start_time=all_events[0].timestamp
        if all_events
        else datetime.now().timestamp(),
        end_time=all_events[-1].timestamp if all_events else datetime.now().timestamp(),
        events=all_events,
        metadata={
            "n_passes": config.n_forward_passes,
            "n_challenges": len(hook_system.challenges),
        },
    )
    trace_id = database.store_trace(trace)

    # Update challenge metadata with trace ID and store
    for challenge in hook_system.challenges:
        challenge.metadata["trace_id"] = trace_id
        database.store_challenge(challenge)

    # Create data bundle
    data_bundle = DataBundle(
        id=f"data_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        inputs=all_inputs,
        outputs=all_outputs,
        weights={
            f"weight_{i}": TensorData.from_array(w)
            for i, (w, _) in enumerate(model.weights)
        },
        activations=all_activations,
        metadata={
            "trace_id": trace_id,
            "lsh_static_projections": [lsh_id for lsh_id in all_lsh_static.keys()],
        },
    )

    # Add static LSH projections to activations
    data_bundle.activations.update(all_lsh_static)

    data_id = database.store_data_bundle(data_bundle)

    # CRITICAL: Verify that stored StableHLO matches execution
    test_input = random.normal(
        random.PRNGKey(999), (config.batch_size, config.input_dim)
    )

    # Execute with the jitted function
    jitted_output = model.jitted_forward(test_input)

    # Execute with Python (should match)
    h = test_input
    for i, (w, b) in enumerate(model.weights):
        h = jnp.dot(h, w) + b
        if i < len(model.weights) - 1:
            h = jax.nn.relu(h)
    python_output = h

    # Check they match
    assert jnp.allclose(jitted_output, python_output, rtol=1e-5), (
        "CRITICAL: JIT output doesn't match Python execution!"
    )

    # Check that static LSH was computed for each pass
    for pass_idx in range(config.n_forward_passes):
        lsh_key = f"lsh_static_pass_{pass_idx}"
        assert lsh_key in data_bundle.activations, (
            f"Missing static LSH for pass {pass_idx}"
        )
        lsh_data = data_bundle.activations[lsh_key]
        assert lsh_data.shape == (config.batch_size, config.lsh_dim), (
            f"Wrong LSH shape: {lsh_data.shape}"
        )

    # Check challenges
    assert len(hook_system.challenges) == n_challenges, (
        f"Challenge count mismatch: {len(hook_system.challenges)} != {n_challenges}"
    )

    # Check outputs
    for pass_idx in range(config.n_forward_passes):
        output_key = f"output_pass_{pass_idx}"
        assert output_key in all_outputs, f"Missing output for pass {pass_idx}"
        output_data = all_outputs[output_key]
        assert output_data.shape == (config.batch_size, config.output_dim), (
            f"Wrong output shape: {output_data.shape}"
        )

    # Verify database contents
    assert database.get_graph(graph_id) is not None
    assert database.get_trace(trace_id) is not None
    assert database.get_data_bundle(data_id) is not None

    # Verify challenges were stored
    assert len(database.challenges) > 0
    # Verify our challenges have the trace_id in metadata
    for challenge in database.challenges:
        assert challenge.metadata.get("trace_id") == trace_id

    # === NEW: Unified Verification Engine ===
    from veritor.verifier.engine import VerificationConfig, verify_single_execution

    # Configure verification
    verification_config = VerificationConfig(
        enable_jit_vs_python=True,
        enable_challenge_verification=True,
        execution_rtol=1e-5,
        lsh_rtol=1e-3,
    )

    # Run unified verification
    result = verify_single_execution(
        database=database,
        graph_id=graph_id,
        trace_id=trace_id,
        config=verification_config,
    )

    # Verify the verification succeeded
    assert result.success, f"Verification failed: {result.errors}"

    # Check specific verification results
    if result.execution_match is not None:
        assert result.execution_match, "JIT vs Python execution mismatch"

    # Check challenge verification results
    failed_challenges = [
        cid for cid, success in result.challenge_results.items() if not success
    ]
    assert len(failed_challenges) == 0, f"Failed challenges: {failed_challenges}"

    print(f"✅ Unified verification passed for {graph_id}")
    print(f"   - Challenge results: {len(result.challenge_results)} verified")
    if result.metrics:
        print(
            f"   - Max execution difference: {result.metrics.get('max_difference', 'N/A')}"
        )
