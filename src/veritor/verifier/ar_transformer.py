"""
Autoregressive to Teacher-Forcing Graph Transformation.

This module implements the core transformation needed for STAMP protocol:
Converting autoregressive inference graphs into teacher-forcing verification graphs.

The key insight: Instead of trying to manipulate StableHLO text directly,
we use JAX's compilation infrastructure to generate the appropriate graphs.
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.interpreters import mlir


def extract_step_function_from_loop(ar_hlo_text: str) -> Dict[str, Any]:
    """
    Extract the step function logic from an autoregressive while loop.

    Returns metadata about the loop structure that can guide transformation.
    """
    metadata = {
        'has_while_loop': 'stablehlo.while' in ar_hlo_text,
        'has_dynamic_update': 'dynamic_update_slice' in ar_hlo_text,
        'has_gather': 'gather' in ar_hlo_text,
        'has_argmax': 'argmax' in ar_hlo_text,
        'input_shape': None,
        'output_shape': None
    }

    # Extract shapes from function signature
    import re
    sig_match = re.search(r'@main\((.*?)\) -> (.*?) \{', ar_hlo_text)
    if sig_match:
        input_part = sig_match.group(1)
        output_part = sig_match.group(2)

        # Parse input shape
        input_shape_match = re.search(r'tensor<(.*?)>', input_part)
        if input_shape_match:
            metadata['input_shape'] = input_shape_match.group(1)

        # Parse output shape
        output_shape_match = re.search(r'tensor<(.*?)>', output_part)
        if output_shape_match:
            metadata['output_shape'] = output_shape_match.group(1)

    return metadata


def create_teacher_forcing_from_step(
    step_fn: Callable,
    sequence_length: int,
    vocab_size: int,
    hidden_dim: int
) -> Callable:
    """
    Create a teacher-forcing function from an autoregressive step function.

    Args:
        step_fn: The autoregressive step function (tokens, mask) -> logits
        sequence_length: Maximum sequence length
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension

    Returns:
        Teacher-forcing function that processes all positions
    """

    def teacher_forcing_fn(tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Process entire sequence with teacher-forcing.

        Args:
            tokens: [sequence_length] token ids

        Returns:
            logits: [sequence_length-1, vocab_size] logits for each position
        """
        all_logits = []

        # Process each position
        for pos in range(sequence_length - 1):
            # Create mask for positions up to and including current
            mask = jnp.zeros(sequence_length)
            mask = mask.at[:pos + 1].set(1.0)

            # Get logits for current position
            logits = step_fn(tokens, mask)
            all_logits.append(logits)

        # Stack all logits
        return jnp.stack(all_logits)

    return teacher_forcing_fn


def transform_with_jax_recompilation(
    ar_step_fn: Callable,
    example_tokens: jnp.ndarray,
    example_mask: jnp.ndarray
) -> Tuple[Callable, str]:
    """
    Transform autoregressive step to teacher-forcing using JAX recompilation.

    This approach:
    1. Takes the original step function
    2. Creates a teacher-forcing wrapper
    3. JIT compiles it to get StableHLO

    Args:
        ar_step_fn: Original autoregressive step function
        example_tokens: Example token array for shape inference
        example_mask: Example mask for shape inference

    Returns:
        (teacher_forcing_fn, stablehlo_text)
    """
    sequence_length = len(example_tokens)

    def teacher_forcing_wrapper(teacher_tokens: jnp.ndarray) -> jnp.ndarray:
        """Wrapper that calls step function for each position."""
        def compute_position(pos):
            # Create mask up to current position
            mask = (jnp.arange(sequence_length) <= pos).astype(jnp.float32)
            # Compute logits using original step function
            return ar_step_fn(teacher_tokens, mask)

        # Use vmap to process all positions in parallel
        positions = jnp.arange(sequence_length - 1)
        all_logits = jax.vmap(compute_position)(positions)

        return all_logits

    # JIT compile the teacher-forcing function
    jitted_tf = jax.jit(teacher_forcing_wrapper)

    # Get StableHLO
    lowered = jitted_tf.lower(example_tokens)
    tf_stablehlo = lowered.as_text()

    return jitted_tf, tf_stablehlo


class UnifiedARTransformer:
    """
    Unified approach to AR->TF transformation that works with the verifier.
    """

    def __init__(self, database, graph_id: str):
        """
        Initialize with a graph from the database.

        Args:
            database: WorkloadDatabase instance
            graph_id: ID of the autoregressive graph
        """
        self.database = database
        self.graph_id = graph_id
        self.graph = database.get_graph(graph_id)

        if not self.graph:
            raise ValueError(f"Graph {graph_id} not found")

        # Get the StableHLO IR
        ir_entries = database.ir_store.get_ir_for_graph(graph_id)
        self.ar_hlo = None
        for entry in ir_entries:
            if entry.format == "STABLEHLO":
                self.ar_hlo = entry.content
                break

        if not self.ar_hlo:
            raise ValueError(f"No StableHLO found for graph {graph_id}")

    def transform_to_teacher_forcing(self) -> str:
        """
        Transform the autoregressive graph to teacher-forcing.

        Returns:
            Graph ID of the teacher-forcing graph
        """
        # Extract metadata from AR graph
        metadata = extract_step_function_from_loop(self.ar_hlo)

        # For now, we'll store a marker that this needs transformation
        # The actual transformation would happen at execution time
        tf_graph_id = f"{self.graph_id}_teacher_forcing"

        # Create a new graph entry
        from veritor.db.models import Graph
        tf_graph = Graph(
            id=tf_graph_id,
            metadata={
                **self.graph.metadata,
                "transformation_type": "teacher_forcing",
                "source_graph": self.graph_id,
                "requires_runtime_transform": True,
                "loop_metadata": metadata
            }
        )

        # Store the graph
        self.database.store_graph(tf_graph)

        # Store a placeholder IR that indicates transformation is needed
        from veritor.db.ir_store import IRFormat, IRRole
        self.database.ir_store.attach_ir(
            tf_graph_id,
            IRRole.VERIFICATION,
            f"# Teacher-forcing transformation of {self.graph_id}\n"
            f"# Original has while loop: {metadata['has_while_loop']}\n"
            f"# To be transformed at runtime\n",
            IRFormat.STABLEHLO,
            {"needs_transformation": True}
        )

        return tf_graph_id


def create_runtime_teacher_forcing_executor(
    ar_step_fn: Callable,
    sequence_length: int
) -> Callable:
    """
    Create a runtime executor for teacher-forcing that doesn't require
    pre-transformation of the graph.

    This is the practical approach: instead of transforming StableHLO,
    we execute the step function appropriately at runtime.
    """

    @jax.jit
    def execute_teacher_forcing(tokens: jnp.ndarray) -> jnp.ndarray:
        """Execute teacher-forcing using the AR step function."""

        def scan_fn(carry, token_pos):
            token_idx, pos = token_pos

            # Create mask for this position
            mask = (jnp.arange(sequence_length) <= pos).astype(jnp.float32)

            # Run step function
            logits = ar_step_fn(tokens, mask)

            return carry, logits

        # Process all positions except last
        positions = jnp.arange(sequence_length - 1)
        token_indices = jnp.arange(sequence_length - 1)

        _, all_logits = lax.scan(
            scan_fn,
            init=None,
            xs=(token_indices, positions)
        )

        return all_logits

    return execute_teacher_forcing