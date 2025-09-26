"""
Database API for Veritor.

Provides the WorkloadDatabase facade for managing execution data, graphs, and traces.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from veritor.db.ir_store import IRFormat, IRRole, IRStore
from veritor.db.models import (
    ChallengeRecord,
    DataBundle,
    DeviceTopology,
    Graph,
    Trace,
)


@dataclass
class QueryFilter:
    """Filter criteria for queries"""

    time_range: Optional[Tuple[datetime, datetime]] = None
    device_ids: Optional[List[str]] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


class WorkloadDatabase:
    """
    Central database for workload data, accessible to both Prover and Verifier.

    This is the trusted store that contains all execution data, graphs, traces, etc.
    The prover writes to it and the verifier reads from it.

    The actual computational graphs are stored as IR blobs in the IRStore.
    """

    def __init__(self):
        # Core IR storage
        self.ir_store = IRStore()

        # Lightweight graph metadata (points to IR blobs)
        self.graphs: Dict[str, Graph] = {}
        self.traces: Dict[str, Trace] = {}
        self.data_bundles: Dict[str, DataBundle] = {}
        self.device_topology: Optional[DeviceTopology] = None
        self.challenges: List[ChallengeRecord] = []

        # Metadata about the database itself
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now(),
            "version": "0.2.0",  # Bumped for IR store integration
        }

    # Graph and IR management
    def store_graph_with_ir(
        self,
        graph_id: str,
        ir_content: str | bytes,
        role: IRRole,
        format: IRFormat,
        metadata: Optional[Dict] = None,
    ) -> str:
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

    def link_graph_transformation(
        self, source_id: str, target_id: str, transformation_type: str
    ):
        """Record a transformation between graphs"""
        self.ir_store.link_derivation(source_id, target_id, transformation_type)

    def query_graphs(self, filter: Optional[QueryFilter] = None) -> List[Graph]:
        """Query graphs with optional filtering"""
        results = list(self.graphs.values())

        if filter and filter.metadata_filters:
            results = [
                g
                for g in results
                if all(
                    g.metadata.get(k) == v for k, v in filter.metadata_filters.items()
                )
            ]

        return results

    # Trace queries
    def store_trace(self, trace: Trace) -> str:
        """Store an execution trace"""
        self.traces[trace.id] = trace
        return trace.id

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a specific trace"""
        return self.traces.get(trace_id)

    def get_traces_for_graph(self, graph_id: str) -> List[Trace]:
        """Get all traces for a specific graph"""
        return [t for t in self.traces.values() if t.graph_id == graph_id]

    def query_traces(self, filter: Optional[QueryFilter] = None) -> List[Trace]:
        """Query traces with optional filtering"""
        results = list(self.traces.values())

        if filter:
            if filter.time_range:
                start, end = filter.time_range
                results = [t for t in results if start <= t.start_time <= end]

            if filter.device_ids:
                results = [
                    t
                    for t in results
                    if any(e.device_id in filter.device_ids for e in t.events)
                ]

        return results

    # Data queries
    def store_data_bundle(self, bundle: DataBundle) -> str:
        """Store a data bundle"""
        self.data_bundles[bundle.id] = bundle
        return bundle.id

    def get_data_bundle(self, bundle_id: str) -> Optional[DataBundle]:
        """Get a specific data bundle"""
        return self.data_bundles.get(bundle_id)

    def get_data_for_graph(self, graph_id: str) -> List[DataBundle]:
        """Get all data bundles for a specific graph"""
        return [d for d in self.data_bundles.values() if d.graph_id == graph_id]

    # Device topology
    def set_device_topology(self, topology: DeviceTopology):
        """Set the device topology"""
        self.device_topology = topology

    def get_device_topology(self) -> Optional[DeviceTopology]:
        """Get the device topology"""
        return self.device_topology

    # Challenge records
    def store_challenge(self, challenge: ChallengeRecord):
        """Store a challenge record"""
        self.challenges.append(challenge)

    def get_challenges_for_trace(self, trace_id: str) -> List[ChallengeRecord]:
        """Get challenges associated with a trace"""
        return [c for c in self.challenges if c.trace_id == trace_id]

    # Checkpoint operations (using DataBundle)
    def store_checkpoint(self, graph_id: str, params: Dict[str, Any], step: Optional[int] = None, **metadata) -> str:
        """Store a model checkpoint as a DataBundle."""
        bundle = DataBundle.from_checkpoint(graph_id, params, step=step, **metadata)
        return self.store_data_bundle(bundle)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[DataBundle]:
        """Get a checkpoint bundle."""
        bundle = self.get_data_bundle(checkpoint_id)
        if bundle and bundle.bundle_type == "checkpoint":
            return bundle
        return None

    def get_checkpoints_for_graph(self, graph_id: str) -> List[DataBundle]:
        """Get all checkpoint bundles for a graph."""
        return [
            b for b in self.data_bundles.values()
            if b.graph_id == graph_id and b.bundle_type == "checkpoint"
        ]

    def get_checkpoint_at_step(self, graph_id: str, step: int) -> Optional[DataBundle]:
        """Get checkpoint at a specific training step."""
        for bundle in self.data_bundles.values():
            if (bundle.graph_id == graph_id and
                bundle.bundle_type == "checkpoint" and
                bundle.metadata.get("step") == step):
                return bundle
        return None

    # Persistence
    def save(self, path: str):
        """
        Save the database to disk.

        The IR store is saved separately as a content-addressable store.
        """
        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_to_save = {}
        for k, v in self.metadata.items():
            if isinstance(v, datetime):
                metadata_to_save[k] = v.isoformat()
            else:
                metadata_to_save[k] = v

        with open(db_path / "metadata.json", "w") as f:
            json.dump(
                {
                    **metadata_to_save,
                    "saved_at": datetime.now().isoformat(),
                    "num_graphs": len(self.graphs),
                    "num_traces": len(self.traces),
                    "num_data_bundles": len(self.data_bundles),
                    "num_challenges": len(self.challenges),
                },
                f,
                indent=2,
            )

        # Save IR store
        self.ir_store.save(db_path / "ir_store")

        # Save graphs (lightweight metadata)
        with open(db_path / "graphs.pkl", "wb") as f:
            pickle.dump(self.graphs, f)

        # Save traces
        with open(db_path / "traces.pkl", "wb") as f:
            pickle.dump(self.traces, f)

        # Save data bundles (may be large)
        with open(db_path / "data_bundles.pkl", "wb") as f:
            pickle.dump(self.data_bundles, f)

        # Save device topology
        if self.device_topology:
            with open(db_path / "device_topology.pkl", "wb") as f:
                pickle.dump(self.device_topology, f)

        # Save challenges
        with open(db_path / "challenges.pkl", "wb") as f:
            pickle.dump(self.challenges, f)

        print(f"✓ Database saved to {db_path}")

    @classmethod
    def load(cls, path: str) -> "WorkloadDatabase":
        """
        Load a database from disk.

        The IR store is loaded separately.
        """
        db_path = Path(path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database path {db_path} does not exist")

        db = cls()

        # Load metadata
        with open(db_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            print(f"✓ Database loaded from {db_path}")
            for k, v in metadata.items():
                if k != "saved_at" and not k.startswith("num_"):
                    if k == "created_at":
                        db.metadata[k] = datetime.fromisoformat(v)
                    else:
                        db.metadata[k] = v

        # Load IR store
        db.ir_store = IRStore.load(db_path / "ir_store")

        # Load graphs
        with open(db_path / "graphs.pkl", "rb") as f:
            db.graphs = pickle.load(f)

        # Load traces
        with open(db_path / "traces.pkl", "rb") as f:
            db.traces = pickle.load(f)

        # Load data bundles
        with open(db_path / "data_bundles.pkl", "rb") as f:
            db.data_bundles = pickle.load(f)

        # Load device topology if exists
        topo_path = db_path / "device_topology.pkl"
        if topo_path.exists():
            with open(topo_path, "rb") as f:
                db.device_topology = pickle.load(f)

        # Load challenges
        with open(db_path / "challenges.pkl", "rb") as f:
            db.challenges = pickle.load(f)

        print(f"  - {len(db.graphs)} graphs")
        print(f"  - {len(db.traces)} traces")
        print(f"  - {len(db.data_bundles)} data bundles")
        print(f"  - {len(db.challenges)} challenges")

        return db
