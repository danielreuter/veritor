"""
Veritor API for querying, transforming, and verifying workloads.

This module provides the core API capabilities organized around the data types:
- Query: Retrieve graphs, traces, and data
- Transform: Modify and slice graphs
- Bind: Combine graphs with data
- Execute: Run and verify computations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
import jax
import jax.numpy as jnp
from datetime import datetime

from .data_models import (
    Graph,
    Trace, TraceEvent, EventType,
    DataBundle, TensorData,
    DeviceTopology, DeviceSpec,
    ChallengeRecord, CheckpointRecord
)
from .ir_store import IRStore, IRRole, IRFormat


# -----------------------------------------------------------------------------
# Query API
# -----------------------------------------------------------------------------

@dataclass
class QueryFilter:
    """Filter criteria for queries"""
    time_range: Optional[Tuple[datetime, datetime]] = None
    device_ids: Optional[List[str]] = None
    # graph_types: Optional[List[str]] = None  # TODO: Add GraphType enum if needed
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


class WorkloadDatabase:
    """
    Central database for workload data, accessible to both Prover and Verifier.

    This is the trusted store that contains all execution data, graphs, traces, etc.
    Both the prover writes to it and the verifier reads from it.

    The actual computational graphs are stored as IR blobs in the IRStore.
    """

    def __init__(self):
        # IR storage (source of truth for computation)
        self.ir_store = IRStore()

        # Graph metadata (lightweight references to IR)
        self.graphs: Dict[str, Graph] = {}

        # Execution data
        self.traces: Dict[str, Trace] = {}
        self.data_bundles: Dict[str, DataBundle] = {}
        self.device_topology: Optional[DeviceTopology] = None
        self.challenges: List[ChallengeRecord] = []
        self.checkpoints: List[CheckpointRecord] = []

        # Metadata about the database itself
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now(),
            'version': '0.2.0'  # Bumped for IR store integration
        }

    # Graph and IR management
    def store_graph_with_ir(self, graph_id: str, ir_content: str | bytes,
                           role: IRRole, format: IRFormat,
                           metadata: Optional[Dict] = None) -> str:
        """Store a graph with its IR representation"""
        # Store the IR blob
        blob_id = self.ir_store.attach_ir(graph_id, role, ir_content, format, metadata)

        # Create lightweight graph reference
        graph = Graph(id=graph_id, ir_blob_id=blob_id, metadata=metadata or {})
        self.graphs[graph_id] = graph

        return graph_id

    def store_graph(self, graph: Graph) -> str:
        """Store a graph and return its ID (backward compat)"""
        self.graphs[graph.id] = graph
        return graph.id

    def get_graph(self, graph_id: str) -> Optional[Graph]:
        """Retrieve a graph by ID"""
        return self.graphs.get(graph_id)

    def get_graph_ir(self, graph_id: str, role: IRRole) -> Optional[bytes]:
        """Get the IR content for a graph in a specific role"""
        blob = self.ir_store.get_ir(graph_id, role)
        return blob.content if blob else None

    def link_graph_transformation(self, source_id: str, target_id: str,
                                 transformation_type: str):
        """Record a transformation between graphs"""
        self.ir_store.link_derivation(source_id, target_id, transformation_type)

    def query_graphs(self, filter: Optional[QueryFilter] = None) -> List[Graph]:
        """Query graphs with optional filtering"""
        results = list(self.graphs.values())

        if filter and filter.graph_types:
            results = [g for g in results if g.graph_type in filter.graph_types]

        if filter and filter.metadata_filters:
            results = [g for g in results
                      if all(g.metadata.get(k) == v
                            for k, v in filter.metadata_filters.items())]

        return results

    # Trace queries
    def store_trace(self, trace: Trace) -> str:
        """Store a trace and return its ID"""
        self.traces[trace.id] = trace
        return trace.id

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a trace by ID"""
        return self.traces.get(trace_id)

    def get_traces_for_graph(self, graph_id: str) -> List[Trace]:
        """Get all traces for a specific graph"""
        return [t for t in self.traces.values() if t.graph_id == graph_id]

    def query_traces(self, filter: Optional[QueryFilter] = None) -> List[Trace]:
        """Query traces with optional filtering"""
        results = list(self.traces.values())

        if filter and filter.time_range:
            start, end = filter.time_range
            results = [t for t in results
                      if start <= t.start_time <= end]

        if filter and filter.device_ids:
            device_set = set(filter.device_ids)
            results = [t for t in results
                      if any(e.device_id in device_set for e in t.events)]

        return results

    # Data queries
    def store_data_bundle(self, bundle: DataBundle) -> str:
        """Store a data bundle and return its ID"""
        self.data_bundles[bundle.id] = bundle
        return bundle.id

    def get_data_bundle(self, bundle_id: str) -> Optional[DataBundle]:
        """Retrieve a data bundle by ID"""
        return self.data_bundles.get(bundle_id)

    def get_data_for_graph(self, graph_id: str) -> List[DataBundle]:
        """Get all data bundles for a specific graph"""
        return [d for d in self.data_bundles.values() if d.graph_id == graph_id]

    # Device queries
    def set_device_topology(self, topology: DeviceTopology):
        """Set the device topology"""
        self.device_topology = topology

    def get_device_topology(self) -> Optional[DeviceTopology]:
        """Get the device topology"""
        return self.device_topology

    # Challenge/checkpoint queries
    def store_challenge(self, challenge: ChallengeRecord):
        """Store a challenge record"""
        self.challenges.append(challenge)

    def get_challenges(self, challenge_type: Optional[str] = None) -> List[ChallengeRecord]:
        """Get challenge records, optionally filtered by type"""
        if challenge_type:
            return [c for c in self.challenges if c.challenge_type == challenge_type]
        return self.challenges

    def store_checkpoint(self, checkpoint: CheckpointRecord):
        """Store a checkpoint record"""
        self.checkpoints.append(checkpoint)

    def get_checkpoints(self, step_range: Optional[Tuple[int, int]] = None) -> List[CheckpointRecord]:
        """Get checkpoint records, optionally filtered by step range"""
        if step_range:
            start, end = step_range
            return [c for c in self.checkpoints if start <= c.step <= end]
        return self.checkpoints

    # Serialization/Deserialization
    def save(self, path: str):
        """
        Serialize the database to disk.

        Uses a simple format for now - in production would use proper DB.
        """
        import pickle
        import json
        from pathlib import Path

        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)

        # Save metadata (convert datetime objects to strings)
        metadata_to_save = {}
        for k, v in self.metadata.items():
            if isinstance(v, datetime):
                metadata_to_save[k] = v.isoformat()
            else:
                metadata_to_save[k] = v

        with open(db_path / 'metadata.json', 'w') as f:
            json.dump({
                **metadata_to_save,
                'saved_at': datetime.now().isoformat(),
                'num_graphs': len(self.graphs),
                'num_traces': len(self.traces),
                'num_data_bundles': len(self.data_bundles),
                'num_challenges': len(self.challenges),
                'num_checkpoints': len(self.checkpoints)
            }, f, indent=2)

        # Save IR store
        self.ir_store.save(db_path / 'ir_store')

        # Save graphs (lightweight metadata)
        with open(db_path / 'graphs.pkl', 'wb') as f:
            pickle.dump(self.graphs, f)

        # Save traces
        with open(db_path / 'traces.pkl', 'wb') as f:
            pickle.dump(self.traces, f)

        # Save data bundles (may be large)
        with open(db_path / 'data_bundles.pkl', 'wb') as f:
            pickle.dump(self.data_bundles, f)

        # Save device topology
        if self.device_topology:
            with open(db_path / 'device_topology.pkl', 'wb') as f:
                pickle.dump(self.device_topology, f)

        # Save challenges
        with open(db_path / 'challenges.pkl', 'wb') as f:
            pickle.dump(self.challenges, f)

        # Save checkpoints
        with open(db_path / 'checkpoints.pkl', 'wb') as f:
            pickle.dump(self.checkpoints, f)

        print(f"✓ Database saved to {db_path}")

    @classmethod
    def load(cls, path: str) -> 'WorkloadDatabase':
        """
        Load a database from disk.
        """
        import pickle
        import json
        from pathlib import Path

        db_path = Path(path)
        if not db_path.exists():
            raise ValueError(f"Database path {db_path} does not exist")

        db = cls()

        # Load metadata
        with open(db_path / 'metadata.json', 'r') as f:
            saved_metadata = json.load(f)
            db.metadata.update(saved_metadata)

        # Load IR store
        if (db_path / 'ir_store').exists():
            db.ir_store = IRStore.load(db_path / 'ir_store')

        # Load graphs
        if (db_path / 'graphs.pkl').exists():
            with open(db_path / 'graphs.pkl', 'rb') as f:
                db.graphs = pickle.load(f)

        # Load traces
        if (db_path / 'traces.pkl').exists():
            with open(db_path / 'traces.pkl', 'rb') as f:
                db.traces = pickle.load(f)

        # Load data bundles
        if (db_path / 'data_bundles.pkl').exists():
            with open(db_path / 'data_bundles.pkl', 'rb') as f:
                db.data_bundles = pickle.load(f)

        # Load device topology
        if (db_path / 'device_topology.pkl').exists():
            with open(db_path / 'device_topology.pkl', 'rb') as f:
                db.device_topology = pickle.load(f)

        # Load challenges
        if (db_path / 'challenges.pkl').exists():
            with open(db_path / 'challenges.pkl', 'rb') as f:
                db.challenges = pickle.load(f)

        # Load checkpoints
        if (db_path / 'checkpoints.pkl').exists():
            with open(db_path / 'checkpoints.pkl', 'rb') as f:
                db.checkpoints = pickle.load(f)

        print(f"✓ Database loaded from {db_path}")
        print(f"  - {len(db.graphs)} graphs")
        print(f"  - {len(db.traces)} traces")
        print(f"  - {len(db.data_bundles)} data bundles")
        print(f"  - {len(db.challenges)} challenges")

        return db

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the database contents"""
        return {
            'num_graphs': len(self.graphs),
            'num_traces': len(self.traces),
            'num_data_bundles': len(self.data_bundles),
            'num_challenges': len(self.challenges),
            'num_checkpoints': len(self.checkpoints),
            'has_device_topology': self.device_topology is not None,
            'metadata': self.metadata
        }


# -----------------------------------------------------------------------------
# Graph Transformation API
# -----------------------------------------------------------------------------

class GraphTransformer:
    """Utilities for transforming computational graphs"""

    @staticmethod
    def extract_subgraph(graph: Graph,
                        start_ops: List[str],
                        end_ops: List[str]) -> Graph:
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
        sub_operations = {op_id: graph.operations[op_id]
                         for op_id in relevant_ops}
        sub_edges = {edge_id: graph.edges[edge_id]
                    for edge_id in relevant_edges if edge_id in graph.edges}

        # Determine new inputs/outputs
        sub_inputs = [e_id for e_id in relevant_edges
                     if e_id in graph.edges and graph.edges[e_id].source_op not in relevant_ops]
        sub_outputs = [e_id for e_id in relevant_edges
                      if e_id in graph.edges and graph.edges[e_id].target_op not in relevant_ops]

        return Graph(
            id=f"{graph.id}_subgraph",
            graph_type=graph.graph_type,
            operations=sub_operations,
            edges=sub_edges,
            inputs=sub_inputs,
            outputs=sub_outputs,
            metadata={**graph.metadata, 'source_graph': graph.id}
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
            metadata={**graph.metadata, 'batch_index': batch_index}
        )

        # TODO: Implement actual batch slicing logic
        return sliced_graph

    @staticmethod
    def convert_to_verification_graph(graph: Graph,
                                     verification_type: str) -> Graph:
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
                # graph_type="verification",  # TODO: Add GraphType if needed
                operations={**graph.operations},  # Start with copy
                edges={**graph.edges},
                inputs=graph.inputs,
                outputs=graph.outputs,
                metadata={**graph.metadata},
                source_graph_id=graph.id,
                transformation_type=verification_type
            )

            # TODO: Implement actual transformation logic
            return verification_graph

        else:
            raise ValueError(f"Unknown verification type: {verification_type}")


# -----------------------------------------------------------------------------
# Data Binding API
# -----------------------------------------------------------------------------

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
                    self.validation_errors.append(f"Missing input data for edge {input_id}")
            else:
                # Check shape/dtype compatibility
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
    def bind_io_only(graph: Graph, inputs: Dict[str, TensorData],
                     outputs: Dict[str, TensorData]) -> BoundGraph:
        """Bind only inputs and outputs for black-box verification"""
        data = DataBundle(
            id=f"io_bundle_{graph.id}",
            graph_id=graph.id,
            inputs=inputs,
            outputs=outputs,
            weights={},
            activations={}
        )

        bound = BoundGraph(graph=graph, data=data, partial_binding=True)
        bound.validate()
        return bound

    @staticmethod
    def bind_with_weights(graph: Graph, inputs: Dict[str, TensorData],
                         weights: Dict[str, TensorData]) -> BoundGraph:
        """Bind inputs and model weights for execution"""
        data = DataBundle(
            id=f"exec_bundle_{graph.id}",
            graph_id=graph.id,
            inputs=inputs,
            outputs={},
            weights=weights,
            activations={}
        )

        bound = BoundGraph(graph=graph, data=data, partial_binding=True)
        bound.validate()
        return bound


# -----------------------------------------------------------------------------
# Interpretation/Execution API
# -----------------------------------------------------------------------------

class ExecutionEngine:
    """Engine for executing bound graphs"""

    def __init__(self, backend: str = "jax"):
        """
        Initialize execution engine.

        In production, this would use IREE or similar for graph execution.
        For now, we'll use JAX directly.
        """
        self.backend = backend

    def execute(self, bound_graph: BoundGraph) -> Dict[str, TensorData]:
        """
        Execute a bound graph and return outputs.

        This is a placeholder - real implementation would:
        1. Compile the graph to executable format
        2. Load input/weight data
        3. Execute on available hardware
        4. Return output tensors
        """
        if not bound_graph.validate():
            raise ValueError(f"Invalid binding: {bound_graph.validation_errors}")

        # Placeholder execution
        outputs = {}

        # For now, just return a dummy output
        for output_id in bound_graph.graph.outputs:
            edge = bound_graph.graph.edges[output_id]
            dummy_data = jnp.zeros(edge.shape)
            outputs[output_id] = TensorData.from_array(dummy_data, output_id)

        return outputs

    def execute_with_hooks(self, bound_graph: BoundGraph,
                          hooks: List[Callable]) -> Dict[str, TensorData]:
        """Execute with verification hooks (e.g., for challenges)"""
        # This would integrate with the hook system from experiment1.py
        return self.execute(bound_graph)


class Verifier:
    """High-level verification interface"""

    def __init__(self, database: WorkloadDatabase):
        self.database = database
        self.engine = ExecutionEngine()
        self.transformer = GraphTransformer()
        self.binder = DataBinder()

    def verify_execution(self, trace_id: str,
                        tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        Verify a claimed execution by replaying it.

        Returns verification results including any discrepancies found.
        """
        # Get trace and associated data
        trace = self.database.get_trace(trace_id)
        if not trace:
            return {'error': 'Trace not found'}

        graph = self.database.get_graph(trace.graph_id)
        if not graph:
            return {'error': 'Graph not found'}

        data_bundles = self.database.get_data_for_graph(graph.id)
        if not data_bundles:
            return {'error': 'No data found for graph'}

        data = data_bundles[0]  # Use first bundle for now

        # Bind and execute
        bound = self.binder.bind_full(graph, data)
        if not bound.validate():
            return {'error': 'Binding validation failed', 'errors': bound.validation_errors}

        computed_outputs = self.engine.execute(bound)

        # Compare outputs
        discrepancies = []
        for output_id, computed in computed_outputs.items():
            if output_id in data.outputs:
                claimed = data.outputs[output_id]
                computed_array = computed.to_array()
                claimed_array = claimed.to_array()

                if not jnp.allclose(computed_array, claimed_array, rtol=tolerance):
                    max_diff = float(jnp.max(jnp.abs(computed_array - claimed_array)))
                    discrepancies.append({
                        'output_id': output_id,
                        'max_diff': max_diff
                    })

        return {
            'trace_id': trace_id,
            'graph_id': graph.id,
            'verified': len(discrepancies) == 0,
            'discrepancies': discrepancies
        }

    def verify_challenges(self, trace_id: str) -> Dict[str, Any]:
        """Verify challenge responses in a trace"""
        trace = self.database.get_trace(trace_id)
        if not trace:
            return {'error': 'Trace not found'}

        challenges = [e for e in trace.events if e.event_type == EventType.CHALLENGE]

        # Verify each challenge
        failures = []
        for challenge_event in challenges:
            # This would re-execute the relevant portion and check the challenge response
            # For now, just placeholder
            pass

        return {
            'trace_id': trace_id,
            'total_challenges': len(challenges),
            'failures': failures
        }