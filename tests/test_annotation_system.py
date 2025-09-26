"""
Test the new annotation system with examples from existing patterns.
"""

import pytest
import jax.numpy as jnp
from jax import random

from veritor.db.api import WorkloadDatabase
from veritor.db.models import Graph
from veritor.prover.annotations import (
    AnnotationManager,
    AnnotationContext,
    AnnotationScope,
    ViewType,
)


def test_lsh_annotation_pattern(workload_db):
    """Test the LSH annotation pattern from test_simple_inference.py."""
    database = workload_db
    manager = AnnotationManager(database)

    # Create a graph
    graph = Graph(
        id="test_inference_graph",
        metadata={"test_type": "lsh_annotation"}
    )
    graph_id = database.store_graph(graph)

    # Simulate inference with LSH annotations
    context = AnnotationContext(
        graph_id=graph_id,
        trace_id="test_trace",
        device_id="cpu_0"
    )

    # Generate some activation data
    key = random.PRNGKey(42)
    activations = random.normal(key, (3, 8))  # batch_size=3, hidden_dim=8

    with AnnotationScope(manager, context) as ann:
        # Static LSH at layer
        static_lsh = ann.annotate_lsh(
            activations,
            projection_dim=4,
            projection_type="activation",
            context=AnnotationContext(**context.to_dict(), layer_idx=2, operation_id="layer_2")
        )

        # Dynamic challenge LSH
        for batch_idx in range(3):
            challenge_lsh = ann.annotate_lsh(
                activations[batch_idx],
                projection_dim=4,
                projection_type="challenge",
                context=AnnotationContext(
                    **context.to_dict(),
                    batch_idx=batch_idx,
                    layer_idx=1,
                    operation_id="layer_1"
                )
            )

    # Verify annotations were stored
    assert len(manager.annotations) == 4  # 1 static + 3 challenge
    assert len(database.challenges) == 4

    # Check that LSH projections are deterministic
    lsh_annotations = manager.get_annotations()
    first_lsh = lsh_annotations[0]

    # Re-compute with same seed should give same result
    new_lsh = manager.annotate_lsh(
        activations,
        projection_dim=4,
        projection_type="activation",
        context=AnnotationContext(**context.to_dict(), layer_idx=2, operation_id="layer_2")
    )

    assert jnp.allclose(first_lsh.projection, new_lsh.projection)


def test_checkpoint_annotation_pattern(workload_db):
    """Test the checkpoint annotation pattern from test_simple_training.py."""
    database = workload_db
    manager = AnnotationManager(database)

    # Create a graph
    graph = Graph(
        id="test_training_graph",
        metadata={"test_type": "checkpoint_annotation"}
    )
    graph_id = database.store_graph(graph)

    # Simulate training with checkpoint annotations
    key = random.PRNGKey(42)
    initial_params = {
        "w1": random.normal(key, (3, 8)),
        "b1": random.normal(random.split(key)[1], (8,)),
        "w2": random.normal(random.split(key)[0], (8, 2)),
        "b2": random.normal(random.split(key)[1], (2,))
    }

    context = AnnotationContext(
        graph_id=graph_id,
        trace_id="training_trace"
    )

    # Training loop with annotations
    current_params = initial_params.copy()

    with AnnotationScope(manager, context) as ann:
        for step in range(3):
            step_context = AnnotationContext(**context.to_dict(), step=step)

            # Simulate parameter update
            for param_name in current_params:
                current_params[param_name] += random.normal(key, current_params[param_name].shape) * 0.01

            # Store checkpoint
            loss = 1.0 / (step + 1)  # Fake decreasing loss
            checkpoint = ann.annotate_checkpoint(
                current_params.copy(),
                loss=loss,
                context=step_context,
                batch_id=f"batch_{step}"
            )

            # Store gradient LSH for verification
            for param_name, param_value in current_params.items():
                fake_gradient = random.normal(key, param_value.shape) * 0.1
                grad_lsh = ann.annotate_lsh(
                    fake_gradient,
                    projection_dim=4,
                    projection_type="gradient",
                    context=AnnotationContext(**step_context.to_dict(), operation_id=f"grad_{param_name}")
                )

    # Verify checkpoints were stored
    checkpoints = database.get_checkpoints_for_graph(graph_id)
    assert len(checkpoints) == 3

    # Verify gradient LSH annotations
    grad_lsh_annotations = manager.get_annotations(
        context_filter={"graph_id": graph_id}
    )
    # 3 checkpoints + 12 gradient LSH (3 steps * 4 params)
    assert len(grad_lsh_annotations) == 15


def test_multi_view_annotation_pattern(workload_db):
    """Test multi-view pattern from test_distributed_inference.py."""
    database = workload_db
    manager = AnnotationManager(database)

    # Create logical and distributed graphs
    logical_graph = Graph(
        id="logical_graph",
        metadata={"view": "logical"}
    )
    distributed_graph = Graph(
        id="distributed_graph",
        metadata={"view": "distributed"}
    )

    logical_id = database.store_graph(logical_graph)
    distributed_id = database.store_graph(distributed_graph)

    context = AnnotationContext(
        graph_id=logical_id,
        trace_id="multi_view_trace"
    )

    # Generate some data
    key = random.PRNGKey(42)
    logical_data = {
        "input": random.normal(key, (8, 64)),    # Full batch
        "output": random.normal(key, (8, 10))   # Full output
    }

    # Simulate sharding across 2 devices
    distributed_data = {
        "device_0": {
            "input": logical_data["input"][:4],   # First 4 samples
            "output": logical_data["output"][:4]
        },
        "device_1": {
            "input": logical_data["input"][4:],   # Last 4 samples
            "output": logical_data["output"][4:]
        }
    }

    with AnnotationScope(manager, context) as ann:
        # Create multi-view data bundles
        from veritor.db.models import TensorData

        views = {
            ViewType.LOGICAL: {
                "inputs": {"batch_input": TensorData.from_array(logical_data["input"])},
                "outputs": {"batch_output": TensorData.from_array(logical_data["output"])}
            },
            ViewType.DISTRIBUTED: {
                "inputs": {
                    f"device_{i}_input": TensorData.from_array(data["input"])
                    for i, data in enumerate([distributed_data["device_0"], distributed_data["device_1"]])
                },
                "outputs": {
                    f"device_{i}_output": TensorData.from_array(data["output"])
                    for i, data in enumerate([distributed_data["device_0"], distributed_data["device_1"]])
                }
            }
        }

        bundle_ids = ann.create_multi_view_bundle(
            base_data={"id": "base_computation"},
            views=views,
            context=context
        )

        # Link the logical and distributed graphs
        ann.annotate_cross_reference(
            "graph_transformation",
            logical_id,
            distributed_id,
            "logical_to_distributed",
            context=context
        )

    # Verify multi-view bundles were created
    assert ViewType.LOGICAL in bundle_ids
    assert ViewType.DISTRIBUTED in bundle_ids

    # Verify cross-references
    cross_refs = manager.get_annotations(
        context_filter={"graph_id": logical_id}
    )

    # Should have view equivalence cross-refs + graph transformation
    assert len([a for a in cross_refs if "cross_reference" in a.annotation_type.value]) >= 1


def test_annotation_filtering():
    """Test annotation filtering and retrieval."""
    from veritor.db.api import WorkloadDatabase
    database = WorkloadDatabase()
    manager = AnnotationManager(database)

    # Create test graph
    graph = Graph(id="filter_test", metadata={})
    graph_id = database.store_graph(graph)

    # Create annotations with different contexts
    contexts = [
        AnnotationContext(graph_id=graph_id, step=0, layer_idx=1),
        AnnotationContext(graph_id=graph_id, step=0, layer_idx=2),
        AnnotationContext(graph_id=graph_id, step=1, layer_idx=1),
        AnnotationContext(graph_id=graph_id, step=1, layer_idx=2),
    ]

    key = random.PRNGKey(42)
    test_tensor = random.normal(key, (4, 8))

    for ctx in contexts:
        manager.annotate_lsh(test_tensor, 4, context=ctx)

    # Filter by step
    step_0_annotations = manager.get_annotations(
        context_filter={"step": 0}
    )
    assert len(step_0_annotations) == 2

    # Filter by layer
    layer_1_annotations = manager.get_annotations(
        context_filter={"layer_idx": 1}
    )
    assert len(layer_1_annotations) == 2

    # Filter by both
    step_1_layer_2 = manager.get_annotations(
        context_filter={"step": 1, "layer_idx": 2}
    )
    assert len(step_1_layer_2) == 1


