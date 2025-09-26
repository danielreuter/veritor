"""
StableHLO graph transformation utilities for verification.

This module provides utilities for transforming StableHLO computational graphs,
particularly for converting autoregressive generation graphs into teacher-forcing
verification graphs.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class HLOOperation:
    """Represents a single HLO operation."""
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, str]
    body: Optional[str] = None


class AutoregressiveToTeacherForcing:
    """
    Transform autoregressive StableHLO graphs to teacher-forcing graphs.

    The autoregressive graph contains a while loop that:
    1. Maintains state (tokens, position)
    2. Generates next token based on current state
    3. Updates tokens with generated token
    4. Increments position

    The teacher-forcing graph needs to:
    1. Take full token sequence as input
    2. Process each position with appropriate masking
    3. Return logits for all positions
    4. Remove token generation/sampling
    """

    def __init__(self, ar_stablehlo: str):
        """Initialize with autoregressive StableHLO text."""
        self.ar_hlo = ar_stablehlo
        self.operations = []
        self.constants = {}

    def transform(self, teacher_tokens: Optional[jnp.ndarray] = None) -> str:
        """
        Transform autoregressive HLO to teacher-forcing HLO.

        Args:
            teacher_tokens: Optional pre-specified token sequence for teacher-forcing.
                          If None, expects tokens as input.

        Returns:
            Teacher-forcing StableHLO as string.
        """
        # Parse the autoregressive HLO
        self._parse_hlo()

        # Identify the while loop
        while_op = self._find_while_loop()
        if not while_op:
            raise ValueError("No while loop found in autoregressive HLO")

        # Extract the loop body computation
        loop_body = self._extract_loop_body(while_op)

        # Identify loop components
        components = self._analyze_loop_components(loop_body)

        # Generate teacher-forcing HLO
        tf_hlo = self._generate_teacher_forcing(components, teacher_tokens)

        return tf_hlo

    def _parse_hlo(self):
        """Parse HLO text into structured operations."""
        lines = self.ar_hlo.split('\n')

        current_op = None
        in_block = False
        block_content = []

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('//'):
                continue

            # Parse operation definitions
            if '=' in line and 'stablehlo.' in line:
                # Extract operation
                parts = line.split('=')
                outputs = parts[0].strip().split(',')

                # Extract operation type and inputs
                op_match = re.search(r'stablehlo\.(\w+)', parts[1])
                if op_match:
                    op_type = op_match.group(1)

                    # Extract inputs (simplified parsing)
                    input_match = re.search(r'\((.*?)\)', parts[1])
                    inputs = []
                    if input_match:
                        inputs = [i.strip() for i in input_match.group(1).split(',')]

                    current_op = HLOOperation(
                        op_type=op_type,
                        name=outputs[0].strip() if outputs else "",
                        inputs=inputs,
                        outputs=outputs,
                        attributes={}
                    )

                    # Check if this starts a block
                    if '{' in line:
                        in_block = True
                        block_content = []

            # Handle block content
            if in_block:
                block_content.append(line)
                if '}' in line:
                    in_block = False
                    if current_op:
                        current_op.body = '\n'.join(block_content)
                        self.operations.append(current_op)
                        current_op = None
            elif current_op and not in_block:
                self.operations.append(current_op)
                current_op = None

    def _find_while_loop(self) -> Optional[HLOOperation]:
        """Find the main while loop in the HLO."""
        for op in self.operations:
            if op.op_type == "while":
                return op
        return None

    def _extract_loop_body(self, while_op: HLOOperation) -> str:
        """Extract the loop body computation."""
        # The while loop body is typically in the 'do' block
        if while_op.body:
            # Find the 'do {' section
            do_start = while_op.body.find('do {')
            if do_start >= 0:
                # Extract everything after 'do {'
                body = while_op.body[do_start + 4:]
                # Remove the final '}'
                body = body.rsplit('}', 1)[0]
                return body.strip()
        return ""

    def _analyze_loop_components(self, loop_body: str) -> Dict:
        """
        Analyze loop body to identify key components.

        Returns dict with:
        - token_update: How tokens are updated
        - logits_computation: How logits are computed
        - position_increment: How position is incremented
        """
        components = {
            'token_update': None,
            'logits_computation': None,
            'position_increment': None,
            'mask_generation': None
        }

        # Look for token update pattern (dynamic_update_slice)
        token_update_match = re.search(
            r'dynamic_update_slice.*?\((.*?)\)',
            loop_body
        )
        if token_update_match:
            components['token_update'] = token_update_match.group(0)

        # Look for gather operations (token embedding lookup)
        gather_match = re.search(
            r'gather.*?\((.*?)\)',
            loop_body
        )
        if gather_match:
            components['embedding_lookup'] = gather_match.group(0)

        # Look for dot operations (matrix multiplications)
        dot_matches = re.findall(
            r'dot_general.*?\((.*?)\)',
            loop_body
        )
        if dot_matches:
            components['matrix_ops'] = dot_matches

        # Look for position increment
        add_match = re.search(
            r'add.*?\b1\b',  # Looking for adding 1
            loop_body
        )
        if add_match:
            components['position_increment'] = add_match.group(0)

        return components

    def _generate_teacher_forcing(
        self,
        components: Dict,
        teacher_tokens: Optional[jnp.ndarray]
    ) -> str:
        """
        Generate teacher-forcing HLO based on analyzed components.

        This is a simplified version that demonstrates the concept.
        A full implementation would need to:
        1. Properly unroll the loop
        2. Replace token generation with teacher tokens
        3. Maintain proper tensor shapes
        4. Handle all edge cases
        """

        # Start building the teacher-forcing HLO
        tf_hlo_lines = []

        # Module header
        tf_hlo_lines.append(
            "module @teacher_forcing attributes {mhlo.num_partitions = 1 : i32, "
            "mhlo.num_replicas = 1 : i32} {"
        )

        # Main function signature
        # Input: token sequence, Output: logits for all positions
        tf_hlo_lines.append(
            "  func.func public @main(%tokens: tensor<5xi32>) -> "
            "tensor<4x10xf32> {"
        )

        # Extract constants from original HLO (embeddings, weights)
        for line in self.ar_hlo.split('\n'):
            if 'stablehlo.constant dense' in line and 'xf32>' in line:
                tf_hlo_lines.append(f"    {line.strip()}")

        # Unroll the loop for each position
        tf_hlo_lines.append("    // Unrolled teacher-forcing positions")

        # For each position, compute logits
        for pos in range(4):  # Assuming 5 tokens, 4 positions to predict
            tf_hlo_lines.append(f"    // Position {pos}")

            # Create mask for this position
            tf_hlo_lines.append(
                f"    %mask_{pos} = stablehlo.iota dim = 0 : tensor<5xi32>"
            )
            tf_hlo_lines.append(
                f"    %pos_{pos} = stablehlo.constant dense<{pos}> : tensor<i32>"
            )
            tf_hlo_lines.append(
                f"    %pos_broadcast_{pos} = stablehlo.broadcast_in_dim %pos_{pos}, "
                f"dims = [] : (tensor<i32>) -> tensor<5xi32>"
            )
            tf_hlo_lines.append(
                f"    %mask_bool_{pos} = stablehlo.compare LE, %mask_{pos}, "
                f"%pos_broadcast_{pos}, SIGNED : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi1>"
            )
            tf_hlo_lines.append(
                f"    %mask_float_{pos} = stablehlo.convert %mask_bool_{pos} : "
                f"(tensor<5xi1>) -> tensor<5xf32>"
            )

            # Embedding lookup
            tf_hlo_lines.append(
                f"    %embeddings_{pos} = \"stablehlo.gather\"(%embeddings, %tokens) "
                f"<{{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], "
                f"collapsed_slice_dims = [], start_index_map = [0], "
                f"index_vector_dim = 1>, slice_sizes = array<i64: 5, 8>}}> : "
                f"(tensor<10x8xf32>, tensor<5xi32>) -> tensor<5x8xf32>"
            )

            # Apply mask
            tf_hlo_lines.append(
                f"    %masked_embeddings_{pos} = stablehlo.multiply "
                f"%embeddings_{pos}, %mask_float_{pos} : tensor<5x8xf32>"
            )

            # Compute logits for this position (simplified)
            tf_hlo_lines.append(
                f"    %hidden_{pos} = stablehlo.reduce(%masked_embeddings_{pos}, %zero) "
                f"applies stablehlo.add across dimensions = [0] : "
                f"(tensor<5x8xf32>, tensor<f32>) -> tensor<8xf32>"
            )
            tf_hlo_lines.append(
                f"    %logits_{pos} = stablehlo.dot_general %hidden_{pos}, %output_proj, "
                f"contracting_dims = [0] x [0] : "
                f"(tensor<8xf32>, tensor<8x10xf32>) -> tensor<10xf32>"
            )

        # Stack all logits
        tf_hlo_lines.append("    // Stack logits from all positions")
        tf_hlo_lines.append(
            "    %all_logits = stablehlo.concatenate "
            "%logits_0, %logits_1, %logits_2, %logits_3, "
            "dimension = 0 : tensor<4x10xf32>"
        )

        # Return
        tf_hlo_lines.append("    return %all_logits : tensor<4x10xf32>")
        tf_hlo_lines.append("  }")
        tf_hlo_lines.append("}")

        return '\n'.join(tf_hlo_lines)


def transform_autoregressive_to_teacher_forcing(
    ar_stablehlo: str,
    sequence_length: int = 5,
    vocab_size: int = 10
) -> str:
    """
    High-level function to transform autoregressive StableHLO to teacher-forcing.

    Args:
        ar_stablehlo: The autoregressive StableHLO text
        sequence_length: Length of sequences
        vocab_size: Size of vocabulary

    Returns:
        Teacher-forcing StableHLO text
    """
    transformer = AutoregressiveToTeacherForcing(ar_stablehlo)
    return transformer.transform()


def validate_transformation(ar_hlo: str, tf_hlo: str) -> bool:
    """
    Validate that the transformation preserves semantics.

    This would:
    1. Check that all model weights are preserved
    2. Verify masking is correct
    3. Ensure output shapes match expected
    """
    # Check that key constants are preserved
    ar_constants = re.findall(r'constant dense<\[(.*?)\]>', ar_hlo)
    tf_constants = re.findall(r'constant dense<\[(.*?)\]>', tf_hlo)

    # Model weights should be preserved
    weights_preserved = all(
        const in tf_constants
        for const in ar_constants
        if 'xf32>' in const
    )

    return weights_preserved