"""
Example: Refactoring test_simple_inference.py to use the new annotation system.

This shows how the annotation system simplifies the original test while
making the verification metadata more explicit and reusable.
"""

import uuid
from datetime import datetime

import jax
import jax.numpy as jnp
from jax import random

from veritor.db.api import WorkloadDatabase
from veritor.db.ir_store import IRFormat, IRRole
from veritor.db.models import Graph, TraceEvent, EventType, Trace
from veritor.prover.annotations import (
    AnnotationManager,
    AnnotationContext,
    AnnotationScope,
    ViewType,
)


def run_inference_with_annotations():
    """
    Run simple inference using the new annotation system.

    Compare this to the original test_simple_inference.py to see how
    the annotation system makes verification metadata more explicit.
    """
    print("🚀 Running inference with annotation system...")

    # Setup
    database = WorkloadDatabase()
    manager = AnnotationManager(database)

    # Model configuration
    config = {
        "n_forward_passes": 3,
        "n_layers": 4,
        "batch_size": 2,
        "input_dim": 3,
        "hidden_dim": 6,
        "output_dim": 2,
        "lsh_dim": 4,
        "seed": 42,
    }

    # Initialize model weights
    key = random.PRNGKey(config["seed"])
    dims = ([config["input_dim"]] +
            [config["hidden_dim"]] * (config["n_layers"] - 1) +
            [config["output_dim"]])

    weights = []
    for i in range(len(dims) - 1):
        key, w_key, b_key = random.split(key, 3)
        w = random.normal(w_key, (dims[i], dims[i + 1])) * 0.1
        b = random.normal(b_key, (dims[i + 1],)) * 0.01
        weights.append((w, b))

    # LSH matrix for verification
    key, lsh_key = random.split(key)
    lsh_matrix = random.normal(lsh_key, (config["hidden_dim"], config["lsh_dim"]))
    lsh_matrix = lsh_matrix / jnp.linalg.norm(lsh_matrix, axis=1, keepdims=True)

    # Core computation function (this gets stored as StableHLO)
    def model_forward(x):
        h = x
        for i, (w, b) in enumerate(weights):
            h = jnp.dot(h, w) + b
            if i < len(weights) - 1:
                h = jax.nn.relu(h)
        return h

    # Create graph and store real StableHLO
    graph = Graph(
        id=f"annotated_inference_{uuid.uuid4().hex[:8]}",
        metadata={
            "model_type": "simple_feedforward",
            "annotation_system": "v1",
            **config
        }
    )
    graph_id = database.store_graph(graph)

    # Generate and store real StableHLO
    jitted_forward = jax.jit(model_forward)
    example_input = jnp.zeros((config["batch_size"], config["input_dim"]))
    lowered = jitted_forward.lower(example_input)
    stablehlo_text = lowered.as_text()

    database.ir_store.attach_ir(
        graph_id,
        IRRole.LOGICAL,
        stablehlo_text,
        IRFormat.STABLEHLO,
        {"generated_with_annotation_system": True}
    )

    print(f"📊 Created graph {graph_id} with real StableHLO ({len(stablehlo_text)} bytes)")

    # Create base annotation context
    base_context = AnnotationContext(
        graph_id=graph_id,
        trace_id=f"trace_{uuid.uuid4().hex[:8]}",
        device_id="cpu_0"
    )

    # Storage for execution results
    all_inputs = {}
    all_outputs = {}
    all_events = []

    # Main annotation scope for the entire inference session
    with AnnotationScope(manager, base_context) as ann:

        print("🔄 Running forward passes with annotations...")

        # Run forward passes
        for pass_idx in range(config["n_forward_passes"]):
            pass_context = AnnotationContext(
                **base_context.to_dict(),
                pass_idx=pass_idx,
                operation_id=f"forward_pass_{pass_idx}"
            )

            # Generate input
            key = random.PRNGKey(config["seed"] + pass_idx)
            x = random.normal(key, (config["batch_size"], config["input_dim"]))

            # Execute using JIT (matches stored StableHLO)
            output = jitted_forward(x)

            # Store I/O data
            input_id = f"input_pass_{pass_idx}"
            output_id = f"output_pass_{pass_idx}"
            all_inputs[input_id] = x
            all_outputs[output_id] = output

            # Record execution event
            event = TraceEvent(
                timestamp=datetime.now().timestamp(),
                event_type=EventType.KERNEL_LAUNCH,
                device_id=base_context.device_id,
                operation_id=pass_context.operation_id,
                data={"pass_idx": pass_idx}
            )
            all_events.append(event)

            # Compute activations for verification (separate from core computation)
            activations = {}
            h = x
            for layer_idx, (w, b) in enumerate(weights):
                h = jnp.dot(h, w) + b
                if layer_idx < len(weights) - 1:
                    h = jax.nn.relu(h)
                activations[f"layer_{layer_idx}"] = h

                # Static LSH at second-to-last layer
                if layer_idx == len(weights) - 2:
                    lsh_projection = jnp.dot(h, lsh_matrix)
                    activations["lsh_static"] = lsh_projection

                    # ANNOTATION: Store static LSH with clean API
                    static_context = AnnotationContext(
                        graph_id=pass_context.graph_id,
                        trace_id=pass_context.trace_id,
                        device_id=pass_context.device_id,
                        pass_idx=pass_context.pass_idx,
                        layer_idx=layer_idx,
                        operation_id=f"lsh_static_layer_{layer_idx}"
                    )
                    ann.annotate_lsh(
                        lsh_projection,
                        config["lsh_dim"],
                        projection_type="static_verification",
                        context=static_context
                    )

            # Dynamic challenges with probability
            challenge_key = random.PRNGKey(config["seed"] + 1000 + pass_idx)
            for batch_idx in range(config["batch_size"]):
                for layer_idx in range(config["n_layers"] - 1):
                    challenge_key, subkey = random.split(challenge_key)

                    if random.bernoulli(subkey, p=0.3):  # 30% challenge probability
                        activation = activations[f"layer_{layer_idx}"][batch_idx]

                        # ANNOTATION: Dynamic challenge LSH with automatic seed generation
                        challenge_context = AnnotationContext(
                            graph_id=pass_context.graph_id,
                            trace_id=pass_context.trace_id,
                            device_id=pass_context.device_id,
                            pass_idx=pass_context.pass_idx,
                            batch_idx=batch_idx,
                            layer_idx=layer_idx,
                            operation_id=f"challenge_layer_{layer_idx}_batch_{batch_idx}"
                        )

                        challenge_lsh = ann.annotate_lsh(
                            activation,
                            config["lsh_dim"],
                            projection_type="dynamic_challenge",
                            context=challenge_context
                        )

                        # Record challenge event
                        challenge_event = TraceEvent(
                            timestamp=datetime.now().timestamp(),
                            event_type=EventType.CHALLENGE,
                            device_id=base_context.device_id,
                            operation_id=challenge_context.operation_id,
                            data={
                                "challenge_type": "lsh_dynamic",
                                "pass_idx": pass_idx,
                                "batch_idx": batch_idx,
                                "layer_idx": layer_idx,
                            }
                        )
                        all_events.append(challenge_event)

        print(f"✅ Generated {len(manager.annotations)} annotations")

        # Store trace
        trace = Trace(
            id=base_context.trace_id,
            graph_id=graph_id,
            start_time=all_events[0].timestamp if all_events else datetime.now().timestamp(),
            end_time=all_events[-1].timestamp if all_events else datetime.now().timestamp(),
            events=all_events,
            metadata={"annotation_system_version": "v1"}
        )
        database.store_trace(trace)

        # Create multi-view data storage
        from veritor.db.models import TensorData

        logical_view = {
            "inputs": {k: TensorData.from_array(v) for k, v in all_inputs.items()},
            "outputs": {k: TensorData.from_array(v) for k, v in all_outputs.items()},
            "weights": {f"weight_{i}": TensorData.from_array(w) for i, (w, _) in enumerate(weights)}
        }

        # In a real distributed setting, this would be actual device-local data
        device_view = {
            "inputs": {f"device_local_{k}": v for k, v in logical_view["inputs"].items()},
            "outputs": {f"device_local_{k}": v for k, v in logical_view["outputs"].items()},
        }

        # ANNOTATION: Multi-view storage with automatic cross-referencing
        bundle_ids = ann.create_multi_view_bundle(
            base_data={"computation_type": "simple_inference"},
            views={
                ViewType.LOGICAL: logical_view,
                ViewType.DEVICE_LOCAL: device_view
            },
            context=base_context
        )

        print(f"💾 Created multi-view bundles: {list(bundle_ids.keys())}")

    # Verification phase
    print("\n🔍 Verification capabilities:")

    # 1. Check deterministic LSH
    lsh_annotations = manager.get_annotations(
        context_filter={"graph_id": graph_id}
    )
    static_lsh = [a for a in lsh_annotations if "static" in getattr(a, 'projection_type', '')]
    dynamic_lsh = [a for a in lsh_annotations if "dynamic" in getattr(a, 'projection_type', '')]

    print(f"   - {len(static_lsh)} static LSH projections")
    print(f"   - {len(dynamic_lsh)} dynamic challenge LSH projections")

    # 2. Verify StableHLO consistency
    test_input = random.normal(random.PRNGKey(999), (config["batch_size"], config["input_dim"]))
    jit_output = jitted_forward(test_input)

    # Python execution (should match)
    h = test_input
    for w, b in weights:
        h = jnp.dot(h, w) + b
        if w.shape[1] != config["output_dim"]:  # Not last layer
            h = jax.nn.relu(h)
    python_output = h

    assert jnp.allclose(jit_output, python_output, rtol=1e-5)
    print("   ✅ StableHLO matches Python execution")

    # 3. Cross-reference verification
    cross_refs = [a for a in manager.annotations
                  if hasattr(a, 'annotation_type') and 'cross_reference' in a.annotation_type.value]
    print(f"   - {len(cross_refs)} cross-references created")

    # 4. Database integrity
    assert database.get_graph(graph_id) is not None
    assert database.get_trace(base_context.trace_id) is not None
    # All LSH annotations create challenges (static + dynamic)
    total_lsh = len(static_lsh) + len(dynamic_lsh)
    assert len(database.challenges) == total_lsh
    print("   ✅ Database integrity verified")

    print(f"\n🎉 Inference with annotations completed successfully!")
    print(f"   Graph: {graph_id}")
    print(f"   Trace: {base_context.trace_id}")
    print(f"   Annotations: {len(manager.annotations)}")
    print(f"   Challenges: {len(database.challenges)}")

    return database, manager, graph_id


def demonstrate_annotation_benefits():
    """Show the key benefits of the annotation system."""
    print("\n" + "="*60)
    print("🎯 ANNOTATION SYSTEM BENEFITS")
    print("="*60)

    db, manager, graph_id = run_inference_with_annotations()

    print("\n1. 🔍 EASY FILTERING:")
    # Get all LSH annotations from layer 2
    layer_2_lsh = manager.get_annotations(
        context_filter={"layer_idx": 2}
    )
    print(f"   Found {len(layer_2_lsh)} LSH annotations from layer 2")

    print("\n2. 🎯 DETERMINISTIC CHALLENGES:")
    # Show that same context produces same LSH
    context = AnnotationContext(graph_id=graph_id, layer_idx=1, batch_idx=0)
    key = random.PRNGKey(42)
    test_activation = random.normal(key, (8,))

    lsh1 = manager.annotate_lsh(test_activation, 4, context=context)
    lsh2 = manager.annotate_lsh(test_activation, 4, context=context)

    assert jnp.allclose(lsh1.projection, lsh2.projection)
    print("   ✅ Same context → same LSH projection")

    print("\n3. 🔗 AUTOMATIC CROSS-REFERENCING:")
    cross_refs = [a for a in manager.annotations
                  if hasattr(a, 'annotation_type') and 'cross_reference' in a.annotation_type.value]
    for ref in cross_refs[:2]:  # Show first 2
        print(f"   {ref.source_id[:12]}... → {ref.target_id[:12]}... ({ref.relationship})")

    print("\n4. 📊 RICH METADATA:")
    for annotation in manager.annotations[:3]:  # Show first 3
        metadata = annotation.to_metadata()
        relevant_keys = [k for k in metadata.keys()
                        if k in ['annotation_type', 'layer_idx', 'projection_type']]
        subset = {k: metadata[k] for k in relevant_keys if k in metadata}
        print(f"   {annotation.id[:16]}...: {subset}")

    print("\n5. 🚀 CLEAN API:")
    print("   Before: Complex manual challenge creation and storage")
    print("   After: ann.annotate_lsh(tensor, dim, context=ctx)")

    print("\n" + "="*60)


if __name__ == "__main__":
    demonstrate_annotation_benefits()