# Veritor Annotation System

## Overview

The Veritor Annotation System provides a clean, extensible abstraction for Provers to annotate their execution data with verification metadata. This system addresses the common pattern where Provers need to add verification hints, challenges, cross-references, and multi-view data storage while maintaining clean separation between core computation and verification metadata.

## Key Benefits

### 1. **Clean API**
```python
# Before: Manual challenge creation and storage
challenge = ChallengeRecord(
    id=f"challenge_p{pass_idx}_b{batch_idx}_l{layer_idx}",
    challenge_type="lsh_dynamic",
    timestamp=datetime.now().timestamp(),
    target_operation_id=f"layer_{layer_idx}",
    seed=seed,
    projection_dim=lsh_dim,
    response_value=projection.tolist(),
    metadata={...}
)
database.store_challenge(challenge)

# After: Simple annotation with automatic metadata
ann.annotate_lsh(activation, lsh_dim, context=context)
```

### 2. **Deterministic Challenges**
The system automatically generates deterministic seeds from context, ensuring reproducible verification:

```python
context = AnnotationContext(graph_id="model", step=0, layer_idx=2, batch_idx=1)
lsh1 = ann.annotate_lsh(tensor, 4, context=context)
lsh2 = ann.annotate_lsh(tensor, 4, context=context)
# lsh1.projection == lsh2.projection (guaranteed)
```

### 3. **Automatic Cross-Referencing**
Multi-view data bundles are automatically cross-referenced:

```python
bundle_ids = ann.create_multi_view_bundle(
    base_data={"computation": "inference"},
    views={
        ViewType.LOGICAL: logical_data,
        ViewType.DISTRIBUTED: distributed_data
    }
)
# Cross-references between views are created automatically
```

### 4. **Rich Metadata and Filtering**
Easy retrieval and filtering of annotations:

```python
# Get all LSH projections from layer 2
layer_2_lsh = manager.get_annotations(
    context_filter={"layer_idx": 2}
)

# Get all gradient LSH from training step 5
grad_lsh = manager.get_annotations(
    context_filter={"step": 5, "projection_type": "gradient"}
)
```

## Core Components

### AnnotationContext
Captures the execution context for creating annotations:

```python
context = AnnotationContext(
    graph_id="training_model_123",
    trace_id="trace_456",
    device_id="gpu_0",
    step=10,
    layer_idx=2,
    batch_idx=1,
    view_type=ViewType.LOGICAL
)
```

### AnnotationManager
Central manager for creating and storing annotations:

```python
manager = AnnotationManager(database)

# Use context scopes for cleaner code
with AnnotationScope(manager, context) as ann:
    # All annotations in this scope inherit the context
    lsh = ann.annotate_lsh(tensor, projection_dim=4)
    checkpoint = ann.annotate_checkpoint(params, loss=0.1)
```

### Annotation Types

#### LSHProjectionAnnotation
For compact verification of tensors using Locality-Sensitive Hashing:

```python
# Static LSH (computed every time)
static_lsh = ann.annotate_lsh(
    activation,
    projection_dim=4,
    projection_type="static_verification"
)

# Dynamic LSH (for challenges)
challenge_lsh = ann.annotate_lsh(
    activation,
    projection_dim=4,
    projection_type="dynamic_challenge"
)

# Gradient LSH (for training verification)
grad_lsh = ann.annotate_lsh(
    gradient,
    projection_dim=4,
    projection_type="gradient"
)
```

#### CheckpointAnnotation
For model checkpoints during training:

```python
checkpoint = ann.annotate_checkpoint(
    params=model_params,
    loss=current_loss,
    context=AnnotationContext(**base_context, step=training_step)
)
```

#### CrossReferenceAnnotation
For linking related data across different views:

```python
# Link logical and distributed graphs
ann.annotate_cross_reference(
    "graph_transformation",
    logical_graph_id,
    distributed_graph_id,
    "logical_to_distributed"
)
```

## Usage Patterns

### 1. Simple Inference with LSH Challenges

```python
manager = AnnotationManager(database)
context = AnnotationContext(graph_id="inference_model", trace_id="trace_1")

with AnnotationScope(manager, context) as ann:
    for pass_idx in range(n_passes):
        pass_context = AnnotationContext(**context.to_dict(), pass_idx=pass_idx)

        # Execute core computation
        output = model(input_batch)

        # Add verification annotations
        for layer_idx, activation in enumerate(activations):
            # Static LSH at key layers
            if layer_idx == n_layers - 2:
                ann.annotate_lsh(
                    activation,
                    lsh_dim,
                    projection_type="static_verification",
                    context=AnnotationContext(**pass_context, layer_idx=layer_idx)
                )

            # Dynamic challenges with probability
            if should_challenge(layer_idx):
                ann.annotate_lsh(
                    activation,
                    lsh_dim,
                    projection_type="dynamic_challenge",
                    context=AnnotationContext(**pass_context, layer_idx=layer_idx)
                )
```

### 2. Training with Checkpoint and Gradient Verification

```python
with AnnotationScope(manager, base_context) as ann:
    for step in range(training_steps):
        step_context = AnnotationContext(**base_context, step=step)

        # Training step
        new_params, loss, gradients = training_step(params, batch)

        # Store checkpoint
        ann.annotate_checkpoint(
            new_params,
            loss=loss,
            context=step_context
        )

        # Store gradient LSH for verification
        for param_name, grad in gradients.items():
            ann.annotate_lsh(
                grad,
                lsh_dim,
                projection_type="gradient",
                context=AnnotationContext(**step_context, operation_id=f"grad_{param_name}")
            )
```

### 3. Multi-View Storage for Distributed Execution

```python
# Logical view: Full computation
logical_view = {
    "inputs": {"batch": full_input_batch},
    "outputs": {"result": full_output_batch}
}

# Distributed view: Sharded across devices
distributed_view = {
    "inputs": {f"device_{i}_input": shard for i, shard in enumerate(input_shards)},
    "outputs": {f"device_{i}_output": shard for i, shard in enumerate(output_shards)}
}

# Create multi-view storage with automatic cross-referencing
bundle_ids = ann.create_multi_view_bundle(
    base_data={"computation_type": "distributed_inference"},
    views={
        ViewType.LOGICAL: logical_view,
        ViewType.DISTRIBUTED: distributed_view
    }
)
```

## Integration with Existing Tests

The annotation system is designed to be compatible with existing test patterns:

### Before (test_simple_inference.py)
```python
# Manual challenge creation
challenge = ChallengeRecord(
    id=f"challenge_p{pass_idx}_b{batch_idx}_l{layer_idx}",
    challenge_type="lsh_dynamic",
    timestamp=datetime.now().timestamp(),
    target_operation_id=f"layer_{layer_idx}",
    seed=seed,
    projection_dim=lsh_dim,
    response_value=projection.tolist(),
    metadata={
        "pass_idx": pass_idx,
        "batch_idx": batch_idx,
        "layer_idx": layer_idx,
        "trace_id": trace_id,
    },
)
self.challenges.append(challenge)
database.store_challenge(challenge)
```

### After (with annotation system)
```python
# Clean annotation with automatic metadata
challenge_context = AnnotationContext(
    graph_id=graph_id,
    trace_id=trace_id,
    pass_idx=pass_idx,
    batch_idx=batch_idx,
    layer_idx=layer_idx
)
ann.annotate_lsh(
    activation,
    lsh_dim,
    projection_type="dynamic_challenge",
    context=challenge_context
)
```

## Design Principles

### 1. **Separation of Concerns**
- Core computation remains unchanged
- Verification metadata is added separately
- Clear distinction between what's verified and what's for verification

### 2. **Deterministic Reproduction**
- Same context always produces same annotations
- Automatic seed generation from context
- Reproducible challenge schedules

### 3. **Extensibility**
- Easy to add new annotation types
- Pluggable storage backends
- Flexible filtering and retrieval

### 4. **Performance**
- Minimal overhead on core computation
- Lazy evaluation where possible
- Efficient storage and retrieval

## Critical Invariants

### 1. **StableHLO Consistency**
The stored IR must exactly match what's executed:
```python
# Core computation (gets stored as StableHLO)
def model_forward(x):
    return core_computation(x)

# JIT and store
jitted = jax.jit(model_forward)
stablehlo = jitted.lower(example_input).as_text()
database.store_ir(graph_id, stablehlo)

# Execute the SAME function
output = jitted(actual_input)
```

### 2. **Challenge Determinism**
Same context must produce identical challenge responses:
```python
context = AnnotationContext(graph_id="test", layer_idx=1, batch_idx=0)
lsh1 = ann.annotate_lsh(tensor, 4, context=context)
lsh2 = ann.annotate_lsh(tensor, 4, context=context)
assert jnp.allclose(lsh1.projection, lsh2.projection)  # Must pass
```

### 3. **Cross-Reference Integrity**
Related data must be properly linked:
```python
# Multi-view bundles automatically create cross-references
bundle_ids = ann.create_multi_view_bundle(...)
# Verification can trace between logical ↔ distributed views
```

## Future Extensions

The annotation system is designed to support future verification patterns:

1. **Hierarchical Challenges**: Nested challenge contexts
2. **Probabilistic Verification**: Configurable challenge probabilities
3. **Adaptive LSH**: Dynamic projection dimension selection
4. **Temporal Correlations**: Cross-time-step verification
5. **Privacy-Preserving Annotations**: Zero-knowledge proofs of computation

## Example Files

- `/examples/inference_with_annotations.py`: Complete inference example
- `/tests/test_annotation_system.py`: Comprehensive test suite
- `/src/veritor/prover/annotations.py`: Core implementation

This annotation system provides a robust foundation for verification in the Veritor framework while maintaining clean APIs and extensibility for future verification protocols.