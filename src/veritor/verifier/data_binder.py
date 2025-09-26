"""
Data binding utilities for verification.

This module provides utilities for binding concrete data to computational graphs
for execution and verification.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from veritor.db.models import DataBundle, Graph, TensorData


@dataclass
class BoundGraph:
    """A graph bound with concrete data for execution"""

    graph: Graph
    data: DataBundle

    # Binding configuration
    partial_binding: bool = False  # If True, not all edges need data

    # Validation results
    validation_errors: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate that data is compatible with graph"""
        self.validation_errors = []

        # Check input compatibility
        for input_id in self.graph.inputs:
            if input_id not in self.data.inputs:
                if not self.partial_binding:
                    self.validation_errors.append(
                        f"Missing input data for edge {input_id}"
                    )
            elif hasattr(self.graph, "edges"):
                # Check shape/dtype compatibility only if graph has edges
                edge = self.graph.edges[input_id]
                tensor = self.data.inputs[input_id]
                if edge.shape != tensor.shape:
                    self.validation_errors.append(
                        f"Shape mismatch for {input_id}: expected {edge.shape}, got {tensor.shape}"
                    )
                if edge.dtype != tensor.dtype:
                    self.validation_errors.append(
                        f"Dtype mismatch for {input_id}: expected {edge.dtype}, got {tensor.dtype}"
                    )

        return len(self.validation_errors) == 0

    def get_missing_data(self) -> List[str]:
        """Get list of edge IDs that need data"""
        missing = []
        all_tensors = self.data.get_all_tensors()

        if hasattr(self.graph, "edges"):
            for edge_id in self.graph.edges:
                if edge_id not in all_tensors:
                    missing.append(edge_id)

        return missing


class DataBinder:
    """Utilities for binding data to graphs"""

    @staticmethod
    def bind_full(graph: Graph, data: DataBundle) -> BoundGraph:
        """Bind complete data to all graph edges"""
        bound = BoundGraph(graph=graph, data=data, partial_binding=False)
        bound.validate()
        return bound

    @staticmethod
    def bind_io_only(
        graph: Graph, inputs: Dict[str, TensorData], outputs: Dict[str, TensorData]
    ) -> BoundGraph:
        """Bind only inputs and outputs for black-box verification"""
        data = DataBundle(
            id=f"io_bundle_{graph.id}",
            graph_id=graph.id,
            inputs=inputs,
            outputs=outputs,
            weights={},
            activations={},
        )

        bound = BoundGraph(graph=graph, data=data, partial_binding=True)
        bound.validate()
        return bound

    @staticmethod
    def bind_with_weights(
        graph: Graph, inputs: Dict[str, TensorData], weights: Dict[str, TensorData]
    ) -> BoundGraph:
        """Bind inputs and model weights for execution"""
        data = DataBundle(
            id=f"exec_bundle_{graph.id}",
            graph_id=graph.id,
            inputs=inputs,
            outputs={},
            weights=weights,
            activations={},
        )

        bound = BoundGraph(graph=graph, data=data, partial_binding=True)
        bound.validate()
        return bound
