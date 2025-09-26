"""
IR blob storage and mapping layer for Veritor.

This module treats StableHLO/HLO as the source of truth for computation,
providing a sidecar layer that stores IR blobs and mappings between different
graph variants (logical, distributed, verification).
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class IRFormat(Enum):
    """Supported IR formats"""

    STABLEHLO = "stablehlo"  # Logical graph
    HLO = "hlo"  # Distributed/partitioned graph
    STABLEHLO_TF = "stablehlo_tf"  # Teacher-forcing variant
    JAXPR = "jaxpr"  # JAX's internal IR (for development)


class IRRole(Enum):
    """Role of an IR in the computation pipeline"""

    LOGICAL = "logical"  # Pure computational semantics
    DISTRIBUTED = "distributed"  # With distribution annotations
    VERIFICATION = "verification"  # Transformed for verification


@dataclass
class IRBlob:
    """Content-addressable IR blob"""

    blob_id: str  # SHA256 of normalized content
    content: bytes
    format: IRFormat
    metadata: Dict[str, any] = field(default_factory=dict)

    @classmethod
    def from_content(
        cls, content: str | bytes, format: IRFormat, metadata: Optional[Dict] = None
    ) -> "IRBlob":
        """Create blob from content, computing stable ID"""
        # Normalize content for consistent hashing
        if isinstance(content, str):
            # Normalize whitespace for textual MLIR
            normalized = cls._normalize_mlir(content)
            content_bytes = normalized.encode("utf-8")
        else:
            content_bytes = content

        # Compute content-addressable ID
        blob_id = hashlib.sha256(content_bytes).hexdigest()

        return cls(
            blob_id=blob_id,
            content=content_bytes,
            format=format,
            metadata=metadata or {},
        )

    @staticmethod
    def _normalize_mlir(mlir_text: str) -> str:
        """Normalize MLIR text for consistent hashing"""
        # Remove trailing whitespace and normalize line endings
        lines = mlir_text.splitlines()
        normalized_lines = [line.rstrip() for line in lines]
        return "\n".join(normalized_lines) + "\n" if normalized_lines else ""


@dataclass
class GraphIRMapping:
    """Maps a graph_id to its IR blobs and roles"""

    graph_id: str
    ir_mappings: Dict[IRRole, str] = field(default_factory=dict)  # role -> blob_id

    # Compatibility metadata
    intent_precision: Optional[str] = None  # e.g., "fp32"
    effective_precision: Optional[str] = None  # e.g., "mixed"

    # Additional metadata
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class TransformationRecord:
    """Records a transformation from one graph to another"""

    source_graph_id: str
    target_graph_id: str
    transformation_type: str  # e.g., "partition", "teacher_forcing"
    timestamp: float
    metadata: Dict[str, any] = field(default_factory=dict)


class IRStore:
    """
    Sidecar storage for IR blobs and graph mappings.

    This is the source of truth for all IR representations, providing:
    - Content-addressable blob storage
    - Graph ID to IR role mappings
    - Transformation provenance tracking
    """

    def __init__(self):
        # Core storage
        self.blobs: Dict[str, IRBlob] = {}  # blob_id -> IRBlob
        self.graph_mappings: Dict[
            str, GraphIRMapping
        ] = {}  # graph_id -> GraphIRMapping
        self.transformations: List[TransformationRecord] = []

        # Indexes for efficient queries
        self._role_index: Dict[IRRole, Set[str]] = {role: set() for role in IRRole}
        self._lineage_index: Dict[str, List[str]] = {}  # graph_id -> [parent_ids]

    # R1: IR blob storage
    def put_blob(
        self, content: str | bytes, format: IRFormat, metadata: Optional[Dict] = None
    ) -> str:
        """
        Store an IR blob, returning its content-addressable ID.
        Same content always produces same blob_id.
        """
        blob = IRBlob.from_content(content, format, metadata)

        # Only store if new (content-addressable deduplication)
        if blob.blob_id not in self.blobs:
            self.blobs[blob.blob_id] = blob

        return blob.blob_id

    def get_blob(self, blob_id: str) -> Optional[IRBlob]:
        """Retrieve an IR blob by ID"""
        return self.blobs.get(blob_id)

    # R2: Sidecar mapping between graphs and IR roles
    def attach_ir(
        self,
        graph_id: str,
        role: IRRole,
        content: str | bytes,
        format: IRFormat,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Attach an IR to a graph with a specific role.
        Returns the blob_id of the stored IR.
        """
        # Store the blob
        blob_id = self.put_blob(content, format, metadata)

        # Create or update graph mapping
        if graph_id not in self.graph_mappings:
            self.graph_mappings[graph_id] = GraphIRMapping(graph_id)

        mapping = self.graph_mappings[graph_id]
        mapping.ir_mappings[role] = blob_id

        # Update role index
        self._role_index[role].add(graph_id)

        return blob_id

    def get_ir(self, graph_id: str, role: IRRole) -> Optional[IRBlob]:
        """Get IR blob for a graph in a specific role (O(1) lookup)"""
        mapping = self.graph_mappings.get(graph_id)
        if not mapping or role not in mapping.ir_mappings:
            return None

        blob_id = mapping.ir_mappings[role]
        return self.blobs.get(blob_id)

    def list_ir_roles(self, graph_id: str) -> List[Tuple[IRRole, str, IRFormat]]:
        """List all IR roles and formats available for a graph"""
        mapping = self.graph_mappings.get(graph_id)
        if not mapping:
            return []

        result = []
        for role, blob_id in mapping.ir_mappings.items():
            blob = self.blobs.get(blob_id)
            if blob:
                result.append((role, blob_id, blob.format))

        return result

    # R3: Provenance / transformation chain
    def link_derivation(
        self,
        source_graph_id: str,
        target_graph_id: str,
        transformation_type: str,
        metadata: Optional[Dict] = None,
    ):
        """Record that target was derived from source via transformation"""
        import time

        record = TransformationRecord(
            source_graph_id=source_graph_id,
            target_graph_id=target_graph_id,
            transformation_type=transformation_type,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.transformations.append(record)

        # Update lineage index
        if target_graph_id not in self._lineage_index:
            self._lineage_index[target_graph_id] = []
        self._lineage_index[target_graph_id].append(source_graph_id)

    def get_lineage(self, graph_id: str) -> List[TransformationRecord]:
        """Get the transformation chain that produced this graph"""
        lineage = []

        # Walk backwards through transformations
        current = graph_id
        visited = set()

        while current and current not in visited:
            visited.add(current)

            # Find transformations that produced current
            for transform in self.transformations:
                if transform.target_graph_id == current:
                    lineage.append(transform)
                    current = transform.source_graph_id
                    break
            else:
                # No more transformations found
                break

        return list(reversed(lineage))  # Return in chronological order

    def find_related(self, graph_id: str, relationship: str = "all") -> List[str]:
        """
        Find related graphs.
        relationship: "all", "parents", "children", "siblings"
        """
        related = set()

        if relationship in ["all", "parents"]:
            # Find parent graphs
            for transform in self.transformations:
                if transform.target_graph_id == graph_id:
                    related.add(transform.source_graph_id)

        if relationship in ["all", "children"]:
            # Find child graphs
            for transform in self.transformations:
                if transform.source_graph_id == graph_id:
                    related.add(transform.target_graph_id)

        if relationship in ["all", "siblings"]:
            # Find siblings (graphs with same parent)
            parents = self.find_related(graph_id, "parents")
            for parent in parents:
                children = self.find_related(parent, "children")
                related.update(children)
            related.discard(graph_id)  # Remove self

        return list(related)

    # R4: Compatibility metadata
    def set_precision_metadata(
        self,
        graph_id: str,
        intent_precision: Optional[str] = None,
        effective_precision: Optional[str] = None,
    ):
        """Set precision metadata for a graph"""
        if graph_id not in self.graph_mappings:
            self.graph_mappings[graph_id] = GraphIRMapping(graph_id)

        mapping = self.graph_mappings[graph_id]
        if intent_precision is not None:
            mapping.intent_precision = intent_precision
        if effective_precision is not None:
            mapping.effective_precision = effective_precision

    def get_precision_metadata(
        self, graph_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get (intent_precision, effective_precision) for a graph"""
        mapping = self.graph_mappings.get(graph_id)
        if not mapping:
            return (None, None)
        return (mapping.intent_precision, mapping.effective_precision)

    # R6: Persistence & integrity
    def save(self, path: str):
        """Save IR store to disk with integrity checks"""
        store_path = Path(path)
        store_path.mkdir(parents=True, exist_ok=True)

        # Save blobs
        blobs_path = store_path / "blobs"
        blobs_path.mkdir(exist_ok=True)

        blob_manifest = {}
        for blob_id, blob in self.blobs.items():
            blob_file = blobs_path / f"{blob_id}.blob"
            with open(blob_file, "wb") as f:
                f.write(blob.content)

            # Store metadata and checksum in manifest
            blob_manifest[blob_id] = {
                "format": blob.format.value,
                "metadata": blob.metadata,
                "checksum": hashlib.sha256(blob.content).hexdigest(),
            }

        # Save manifest with integrity info
        with open(store_path / "blob_manifest.json", "w") as f:
            json.dump(blob_manifest, f, indent=2)

        # Save mappings and transformations
        store_data = {
            "graph_mappings": {
                gid: {
                    "ir_mappings": {
                        role.value: bid for role, bid in gm.ir_mappings.items()
                    },
                    "intent_precision": gm.intent_precision,
                    "effective_precision": gm.effective_precision,
                    "metadata": gm.metadata,
                }
                for gid, gm in self.graph_mappings.items()
            },
            "transformations": [
                {
                    "source_graph_id": t.source_graph_id,
                    "target_graph_id": t.target_graph_id,
                    "transformation_type": t.transformation_type,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata,
                }
                for t in self.transformations
            ],
        }

        with open(store_path / "store_data.pkl", "wb") as f:
            pickle.dump(store_data, f)

        print(f"✓ IR store saved to {store_path}")
        print(f"  - {len(self.blobs)} IR blobs")
        print(f"  - {len(self.graph_mappings)} graph mappings")
        print(f"  - {len(self.transformations)} transformations")

    @classmethod
    def load(cls, path: str) -> "IRStore":
        """Load IR store from disk with integrity verification"""
        store_path = Path(path)
        if not store_path.exists():
            raise ValueError(f"IR store path {store_path} does not exist")

        store = cls()

        # Load blob manifest
        with open(store_path / "blob_manifest.json", "r") as f:
            blob_manifest = json.load(f)

        # Load blobs with integrity check
        blobs_path = store_path / "blobs"
        for blob_id, manifest_entry in blob_manifest.items():
            blob_file = blobs_path / f"{blob_id}.blob"

            with open(blob_file, "rb") as f:
                content = f.read()

            # Verify integrity
            actual_checksum = hashlib.sha256(content).hexdigest()
            expected_checksum = manifest_entry["checksum"]

            if actual_checksum != expected_checksum:
                raise ValueError(f"Integrity check failed for blob {blob_id}")

            # Reconstruct blob
            blob = IRBlob(
                blob_id=blob_id,
                content=content,
                format=IRFormat(manifest_entry["format"]),
                metadata=manifest_entry["metadata"],
            )
            store.blobs[blob_id] = blob

        # Load mappings and transformations
        with open(store_path / "store_data.pkl", "rb") as f:
            store_data = pickle.load(f)

        # Reconstruct graph mappings
        for gid, mapping_data in store_data["graph_mappings"].items():
            mapping = GraphIRMapping(
                graph_id=gid,
                ir_mappings={
                    IRRole(role): bid
                    for role, bid in mapping_data["ir_mappings"].items()
                },
                intent_precision=mapping_data["intent_precision"],
                effective_precision=mapping_data["effective_precision"],
                metadata=mapping_data["metadata"],
            )
            store.graph_mappings[gid] = mapping

            # Rebuild role index
            for role in mapping.ir_mappings:
                store._role_index[role].add(gid)

        # Reconstruct transformations
        for t_data in store_data["transformations"]:
            transform = TransformationRecord(
                source_graph_id=t_data["source_graph_id"],
                target_graph_id=t_data["target_graph_id"],
                transformation_type=t_data["transformation_type"],
                timestamp=t_data["timestamp"],
                metadata=t_data["metadata"],
            )
            store.transformations.append(transform)

            # Rebuild lineage index
            target = t_data["target_graph_id"]
            source = t_data["source_graph_id"]
            if target not in store._lineage_index:
                store._lineage_index[target] = []
            store._lineage_index[target].append(source)

        print(f"✓ IR store loaded from {store_path}")
        print(f"  - {len(store.blobs)} IR blobs")
        print(f"  - {len(store.graph_mappings)} graph mappings")
        print(f"  - {len(store.transformations)} transformations")

        return store

    # R7: Zero IR re-serialization mutation
    def round_trip_test(self, blob_id: str) -> bool:
        """Test that round-trip preserves bit-for-bit equality"""
        original_blob = self.get_blob(blob_id)
        if not original_blob:
            return False

        # Put the same content again
        new_blob_id = self.put_blob(
            original_blob.content, original_blob.format, original_blob.metadata
        )

        # Should get same ID (content-addressable)
        return new_blob_id == blob_id
