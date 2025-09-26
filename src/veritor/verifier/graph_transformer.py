"""
Graph transformation utilities for verification.

This module provides utilities for transforming computational graphs
for verification purposes.
"""

from typing import List

from ..db.models import Graph


class GraphTransformer:
    """Utilities for transforming computational graphs"""

    @staticmethod
    def extract_subgraph(
        graph: Graph, start_ops: List[str], end_ops: List[str]
    ) -> Graph:
        """
        Extract a subgraph containing paths from start_ops to end_ops.

        Uses BFS to find all operations on paths between start and end operations.
        """
        # Find all ops on paths from start to end
        relevant_ops = set()
        relevant_edges = set()

        # BFS from start ops
        queue = list(start_ops)
        visited = set()

        while queue:
            op_id = queue.pop(0)
            if op_id in visited or op_id not in graph.operations:
                continue

            visited.add(op_id)
            op = graph.operations[op_id]

            # Check if this op can reach any end op
            if GraphTransformer._can_reach_any(graph, op_id, end_ops):
                relevant_ops.add(op_id)
                relevant_edges.update(op.inputs)
                relevant_edges.update(op.outputs)

                # Add successors to queue
                for edge_id in op.outputs:
                    if edge_id in graph.edges:
                        edge = graph.edges[edge_id]
                        if edge.target_op:
                            queue.append(edge.target_op)

        # Build subgraph
        sub_operations = {op_id: graph.operations[op_id] for op_id in relevant_ops}
        sub_edges = {
            edge_id: graph.edges[edge_id]
            for edge_id in relevant_edges
            if edge_id in graph.edges
        }

        # Determine new inputs/outputs
        sub_inputs = [
            e_id
            for e_id in relevant_edges
            if e_id in graph.edges and graph.edges[e_id].source_op not in relevant_ops
        ]
        sub_outputs = [
            e_id
            for e_id in relevant_edges
            if e_id in graph.edges and graph.edges[e_id].target_op not in relevant_ops
        ]

        return Graph(
            id=f"{graph.id}_subgraph",
            graph_type=graph.graph_type,
            operations=sub_operations,
            edges=sub_edges,
            inputs=sub_inputs,
            outputs=sub_outputs,
            metadata={**graph.metadata, "source_graph": graph.id},
        )

    @staticmethod
    def _can_reach_any(graph: Graph, start_op: str, target_ops: List[str]) -> bool:
        """Check if start_op can reach any of the target_ops"""
        if start_op in target_ops:
            return True

        visited = set()
        queue = [start_op]

        while queue:
            op_id = queue.pop(0)
            if op_id in visited:
                continue
            visited.add(op_id)

            if op_id in target_ops:
                return True

            if op_id in graph.operations:
                op = graph.operations[op_id]
                for edge_id in op.outputs:
                    if edge_id in graph.edges:
                        edge = graph.edges[edge_id]
                        if edge.target_op:
                            queue.append(edge.target_op)

        return False

    @staticmethod
    def slice_batch(graph: Graph, batch_index: int) -> Graph:
        """
        Extract computation for a single batch element.

        This would modify the graph to process only the specified batch index.
        """
        # This is a simplified version - real implementation would need to:
        # 1. Identify batch dimensions in tensors
        # 2. Add slicing operations
        # 3. Update shapes throughout the graph

        sliced_graph = Graph(
            id=f"{graph.id}_batch{batch_index}",
            graph_type=graph.graph_type,
            operations={**graph.operations},  # Copy operations
            edges={**graph.edges},  # Copy edges
            inputs=graph.inputs,
            outputs=graph.outputs,
            metadata={**graph.metadata, "batch_index": batch_index},
        )

        # TODO: Implement actual batch slicing logic
        return sliced_graph

    @staticmethod
    def convert_to_verification_graph(graph: Graph, verification_type: str) -> Graph:
        """
        Convert a graph to a verification-friendly form.

        E.g., autoregressive -> teacher-forcing for inference verification.
        """
        if verification_type == "teacher_forcing":
            # Convert autoregressive graph to teacher-forcing
            # This would:
            # 1. Remove sampling operations
            # 2. Convert sequential processing to parallel
            # 3. Add teacher-forcing connections

            verification_graph = Graph(
                id=f"{graph.id}_verification",
                graph_type=graph.graph_type,
                operations={**graph.operations},  # Start with copy
                edges={**graph.edges},
                inputs=graph.inputs,
                outputs=graph.outputs,
                metadata={
                    **graph.metadata,
                    "source_graph_id": graph.id,
                    "transformation_type": verification_type,
                },
            )

            # TODO: Implement actual transformation logic
            return verification_graph

        else:
            raise ValueError(f"Unknown verification type: {verification_type}")
