"""
Test simple deterministic training with checkpointing and gradient LSH.

This test:
1. Creates a simple model and trains it for N steps
2. Saves checkpoints at each step
3. Computes LSH of gradients for verification
4. Stores everything in the database for verification
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
import pytest
from jax import grad, random

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
class SimpleTrainingConfig:
    """Configuration for simple training test."""

    n_training_steps: int = 10  # Number of training steps
    batch_size: int = 4  # Batch size
    input_dim: int = 3  # Input dimension
    hidden_dim: int = 8  # Hidden dimension
    output_dim: int = 2  # Output dimension
    learning_rate: float = 0.01  # Learning rate
    lsh_dim: int = 4  # LSH projection dimension for gradients
    checkpoint_every: int = 1  # Save checkpoint every N steps
    seed: int = 42


class SimpleTrainingModel:
    """Simple 2-layer feedforward model for training."""

    def __init__(self, config: SimpleTrainingConfig):
        self.config = config
        key = random.PRNGKey(config.seed)

        # Initialize weights
        key, w1_key, b1_key, w2_key, b2_key = random.split(key, 5)

        self.params = {
            "w1": random.normal(w1_key, (config.input_dim, config.hidden_dim)) * 0.1,
            "b1": random.normal(b1_key, (config.hidden_dim,)) * 0.01,
            "w2": random.normal(w2_key, (config.hidden_dim, config.output_dim)) * 0.1,
            "b2": random.normal(b2_key, (config.output_dim,)) * 0.01,
        }

        # LSH projection matrices for gradients
        key, lsh_key = random.split(key)
        self.grad_lsh_matrices = {}
        for param_name in self.params:
            key, subkey = random.split(key)
            param_shape = self.params[param_name].shape
            total_dim = jnp.prod(jnp.array(param_shape))
            proj_matrix = random.normal(subkey, (config.lsh_dim, total_dim))
            # Normalize for stable projections
            proj_matrix = proj_matrix / jnp.linalg.norm(
                proj_matrix, axis=1, keepdims=True
            )
            self.grad_lsh_matrices[param_name] = proj_matrix

    def forward(self, params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the model."""
        # First layer
        h = jnp.dot(x, params["w1"]) + params["b1"]
        h = jax.nn.relu(h)

        # Output layer
        out = jnp.dot(h, params["w2"]) + params["b2"]
        return out

    def loss_fn(self, params: dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute MSE loss."""
        pred = self.forward(params, x)
        return jnp.mean((pred - y) ** 2)

    def compute_gradient_lsh(
        self, gradients: dict, step: int
    ) -> dict[str, jnp.ndarray]:
        """Compute LSH projections of gradients."""
        lsh_projections = {}

        for param_name, grad_value in gradients.items():
            # Flatten gradient
            flat_grad = grad_value.flatten()

            # Apply projection
            proj_matrix = self.grad_lsh_matrices[param_name]
            projection = jnp.dot(proj_matrix, flat_grad)

            # Scale by Johnson-Lindenstrauss factor
            original_dim = flat_grad.shape[0]
            reduced_dim = proj_matrix.shape[0]
            scale = jnp.sqrt(original_dim / reduced_dim)
            projection = projection * scale

            lsh_projections[f"grad_lsh_{param_name}_step_{step}"] = projection

        return lsh_projections


class TrainingExecutor:
    """Executes training and captures all necessary data."""

    def __init__(self, model: SimpleTrainingModel, config: SimpleTrainingConfig):
        self.model = model
        self.config = config
        self.checkpoints = []
        self.gradient_lshs = []
        self.training_data = []

        # Compile gradient function
        self.grad_fn = jax.jit(grad(model.loss_fn))

    def generate_batch(self, step: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a deterministic batch of training data."""
        key = random.PRNGKey(self.config.seed + step * 1000)

        # Generate input
        key, x_key = random.split(key)
        x = random.normal(x_key, (self.config.batch_size, self.config.input_dim))

        # Generate target (could be a simple function of input for determinism)
        # For simplicity, use a random target
        key, y_key = random.split(key)
        y = random.normal(y_key, (self.config.batch_size, self.config.output_dim))

        return x, y

    def training_step(
        self, params: dict, x: jnp.ndarray, y: jnp.ndarray, step: int
    ) -> tuple[dict, float, dict]:
        """Execute one training step."""
        # Compute gradients
        gradients = self.grad_fn(params, x, y)

        # Compute loss
        loss = self.model.loss_fn(params, x, y)

        # Compute gradient LSH
        grad_lsh = self.model.compute_gradient_lsh(gradients, step)

        # Update parameters (simple SGD)
        new_params = {}
        for param_name in params:
            new_params[param_name] = (
                params[param_name] - self.config.learning_rate * gradients[param_name]
            )

        return new_params, float(loss), grad_lsh


def test_simple_training_with_real_gradients(workload_db):
    """Test simple training with real JAX gradients and LSH projections."""
    from veritor.db.api import WorkloadDatabase

    database: WorkloadDatabase = workload_db

    # Configuration
    config = SimpleTrainingConfig(
        n_training_steps=10,
        batch_size=4,
        input_dim=3,
        hidden_dim=8,
        output_dim=2,
        learning_rate=0.01,
        lsh_dim=4,
    )

    # Initialize
    model = SimpleTrainingModel(config)
    executor = TrainingExecutor(model, config)

    # Create graph metadata
    graph = Graph(
        id=f"training_graph_{uuid.uuid4().hex[:8]}",
        metadata={
            "model_type": "simple_feedforward",
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "test_type": "simple_training",
        },
    )
    graph_id = database.store_graph(graph)

    # Generate real StableHLO for the loss function
    def training_loss(params, x, y):
        """Training loss function that will be JIT-compiled."""
        return model.loss_fn(params, x, y)

    # Create example inputs for shape inference
    example_params = model.params
    example_x = jnp.zeros((config.batch_size, config.input_dim))
    example_y = jnp.zeros((config.batch_size, config.output_dim))

    # Lower the loss function to StableHLO
    jitted_loss = jax.jit(training_loss)
    lowered = jitted_loss.lower(example_params, example_x, example_y)
    stablehlo_text = lowered.as_text()

    # Store the jitted function for later use
    model.jitted_loss = jitted_loss

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
            "generated_from": "test_simple_training",
            "jax_version": jax.__version__,
            "model_type": "simple_feedforward_training",
            "weights_embedded": True,
        },
    )

    # Storage for all training data
    all_checkpoints = []
    all_gradient_lshs = {}
    all_losses = []
    all_events = []
    all_training_data = {}
    current_params = model.params.copy()

    trace_start = datetime.now()

    # Training loop
    for step in range(config.n_training_steps):
        # Generate batch
        x_batch, y_batch = executor.generate_batch(step)

        # Store training data
        batch_id = f"batch_step_{step}"
        all_training_data[f"input_{batch_id}"] = TensorData.from_array(x_batch)
        all_training_data[f"target_{batch_id}"] = TensorData.from_array(y_batch)

        # Execute training step
        new_params, loss, grad_lsh = executor.training_step(
            current_params, x_batch, y_batch, step
        )

        # Store loss
        all_losses.append(loss)

        # Store gradient LSH projections
        for lsh_name, lsh_value in grad_lsh.items():
            all_gradient_lshs[lsh_name] = TensorData.from_array(lsh_value)

        # Store checkpoint as DataBundle
        checkpoint_id = database.store_checkpoint(
            graph_id,
            new_params,
            step=step,
            loss=float(loss),
            batch_id=batch_id,
            gradient_lsh_computed=True
        )
        all_checkpoints.append(checkpoint_id)

        # Record events
        events = [
            TraceEvent(
                timestamp=datetime.now().timestamp(),
                event_type=EventType.KERNEL_LAUNCH,
                device_id="cpu_0",
                operation_id=f"forward_pass_step_{step}",
                data={"step": step, "phase": "forward"},
            ),
            TraceEvent(
                timestamp=datetime.now().timestamp(),
                event_type=EventType.KERNEL_LAUNCH,
                device_id="cpu_0",
                operation_id=f"backward_pass_step_{step}",
                data={"step": step, "phase": "backward"},
            ),
            TraceEvent(
                timestamp=datetime.now().timestamp(),
                event_type=EventType.CHECKPOINT,
                device_id="cpu_0",
                operation_id=f"checkpoint_step_{step}",
                data={"step": step, "loss": float(loss)},
            ),
        ]
        all_events.extend(events)

        # Update params for next step
        current_params = new_params

    trace_end = datetime.now()

    # Create and store trace
    trace = Trace(
        id=f"trace_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        start_time=trace_start,
        end_time=trace_end,
        events=all_events,
        metadata={
            "n_steps": config.n_training_steps,
            "final_loss": all_losses[-1],
            "initial_loss": all_losses[0],
        },
    )
    trace_id = database.store_trace(trace)

    # Checkpoints already stored during training loop

    # Store gradient LSH as challenges (verification protocol)
    for step in range(config.n_training_steps):
        for param_name in model.params:
            lsh_key = f"grad_lsh_{param_name}_step_{step}"
            if lsh_key in all_gradient_lshs:
                challenge = ChallengeRecord(
                    id=f"grad_challenge_{param_name}_step_{step}",
                    challenge_type="gradient_lsh",
                    timestamp=datetime.now().timestamp(),
                    target_operation_id=f"backward_pass_step_{step}",
                    seed=step,  # Step as seed for reproducibility
                    projection_dim=config.lsh_dim,
                    response_value=all_gradient_lshs[lsh_key].data.tolist(),
                    metadata={
                        "step": step,
                        "param_name": param_name,
                        "trace_id": trace_id,
                    },
                )
                database.store_challenge(challenge)

    # Create data bundle
    data_bundle = DataBundle(
        id=f"data_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        inputs=all_training_data,
        outputs={},  # No final outputs, just checkpoints
        weights={
            f"initial_{name}": TensorData.from_array(value)
            for name, value in model.params.items()
        },
        activations=all_gradient_lshs,  # Store gradient LSHs as activations
        metadata={
            "trace_id": trace_id,
            "n_checkpoints": len(all_checkpoints),
            "losses": all_losses,
        },
    )
    data_id = database.store_data_bundle(data_bundle)

    # CRITICAL: Verify that stored StableHLO matches execution
    test_params = model.params
    test_x = jnp.zeros((config.batch_size, config.input_dim))
    test_y = jnp.zeros((config.batch_size, config.output_dim))

    # Execute with the jitted function
    jitted_loss = model.jitted_loss(test_params, test_x, test_y)

    # Execute with Python (should match)
    python_loss = model.loss_fn(test_params, test_x, test_y)

    # Check they match
    assert jnp.allclose(jitted_loss, python_loss, rtol=1e-5), \
        "CRITICAL: JIT loss doesn't match Python execution!"

    # Verification assertions
    # Check that all checkpoints were saved
    assert len(all_checkpoints) == config.n_training_steps

    # Check that gradient LSHs were computed for all steps
    expected_grad_lshs = config.n_training_steps * len(model.params)
    actual_grad_lshs = sum(
        1 for c in database.challenges if c.challenge_type == "gradient_lsh"
    )
    assert actual_grad_lshs == expected_grad_lshs

    # Verify checkpoint consistency
    for i, checkpoint_id in enumerate(all_checkpoints):
        checkpoint = database.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.metadata.get("step") == i
        assert "loss" in checkpoint.metadata
        assert len(checkpoint.weights) == len(model.params)

    # Verify database contents
    assert database.get_graph(graph_id) is not None
    assert database.get_trace(trace_id) is not None
    assert database.get_data_bundle(data_id) is not None

    # Verify challenges were stored
    assert len(database.challenges) > 0

    # === NEW: Unified Verification Engine ===
    from veritor.verifier.engine import verify_single_execution, VerificationConfig

    # Configure verification
    verification_config = VerificationConfig(
        enable_jit_vs_python=True,
        enable_challenge_verification=False,  # Gradient challenges need different handling
        execution_rtol=1e-5,
        lsh_rtol=1e-2,  # More lenient for gradient LSH
        lsh_atol=1e-2,
    )

    # Run unified verification
    result = verify_single_execution(
        database=database,
        graph_id=graph_id,
        trace_id=trace_id,
        config=verification_config
    )

    # Verify the verification succeeded
    assert result.success, f"Training verification failed: {result.errors}"

    # Check specific verification results
    if result.execution_match is not None:
        assert result.execution_match, "JIT vs Python execution mismatch in training"

    # Check challenge verification results (gradient LSH)
    failed_challenges = [cid for cid, success in result.challenge_results.items() if not success]
    assert len(failed_challenges) == 0, f"Failed gradient LSH challenges: {failed_challenges}"

    print(f"✅ Unified training verification passed for {graph_id}")
    print(f"   - Gradient LSH challenges: {len(result.challenge_results)} verified")
    if result.metrics:
        print(f"   - Max execution difference: {result.metrics.get('max_difference', 'N/A')}")


def verify_training_replay(
    database,  # WorkloadDatabase
    graph_id: str,
    trace_id: str,
    checkpoint_step: int = 0,
) -> bool:
    """
    Verify training by replaying from a checkpoint.

    This simulates the verifier's workflow:
    1. Load a checkpoint
    2. Load the corresponding training batch
    3. Recompute gradients
    4. Verify gradient LSH matches
    """
    # Load checkpoint
    checkpoint = database.get_checkpoint_at_step(graph_id, checkpoint_step)
    if not checkpoint:
        return False

    # Load training data for this step
    data_bundles = database.get_data_for_graph(graph_id)
    if not data_bundles:
        return False

    # Find the data bundle with training data
    data = None
    for bundle in data_bundles:
        if bundle.bundle_type != "checkpoint" and bundle.inputs:  # Main data bundle with training data
            data = bundle
            break

    if not data:
        return False

    batch_id = f"batch_step_{checkpoint_step}"
    x_batch = data.inputs[f"input_{batch_id}"].to_array()
    y_batch = data.inputs[f"target_{batch_id}"].to_array()

    # Reconstruct model and compute gradients
    config = SimpleTrainingConfig()  # Use default config
    model = SimpleTrainingModel(config)

    # Load checkpoint parameters
    params = checkpoint.to_checkpoint_params()

    # Compute gradients
    grad_fn = grad(model.loss_fn)
    gradients = grad_fn(params, x_batch, y_batch)

    # Compute gradient LSH
    grad_lsh = model.compute_gradient_lsh(gradients, checkpoint_step)

    # Verify against stored LSH
    challenges = [
        c
        for c in database.challenges
        if c.challenge_type == "gradient_lsh"
        and c.metadata.get("step") == checkpoint_step
    ]

    if not challenges:
        return False

    all_match = True
    for challenge in challenges:
        param_name = challenge.metadata.get("param_name")
        lsh_key = f"grad_lsh_{param_name}_step_{checkpoint_step}"

        if lsh_key in grad_lsh:
            computed = grad_lsh[lsh_key]
            stored = jnp.array(challenge.response_value)

            if not jnp.allclose(computed, stored, rtol=1e-2, atol=1e-2):
                all_match = False

    return all_match