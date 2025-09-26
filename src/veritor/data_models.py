"""
Core data models for Veritor workload verification system.

This module defines the fundamental data structures for capturing and verifying
ML workload execution: Graphs, Traces, Data, and Device specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import jax.numpy as jnp
from datetime import datetime
import hashlib
import json


# -----------------------------------------------------------------------------
# Graph Models
# -----------------------------------------------------------------------------

@dataclass
class Graph:
    """
    Lightweight reference to a computational graph stored as IR.

    The actual computation is stored in the IRStore as StableHLO/HLO.
    This class just provides metadata and references.
    """
    id: str

    # Reference to IR blob (actual computation)
    ir_blob_id: Optional[str] = None  # Points to actual IR in IRStore

    # Metadata about this graph instance
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy fields for compatibility (will be removed)
    # These are now stored in the IR itself
    inputs: List[str] = field(default_factory=list)  # For backward compat
    outputs: List[str] = field(default_factory=list)  # For backward compat


# -----------------------------------------------------------------------------
# Trace Models
# -----------------------------------------------------------------------------

class EventType(Enum):
    """Type of runtime event"""
    KERNEL_LAUNCH = "kernel_launch"
    MEMORY_TRANSFER = "memory_transfer"
    COLLECTIVE_OP = "collective_op"
    SYNCHRONIZATION = "synchronization"
    CHALLENGE = "challenge"  # For verification protocols
    CHECKPOINT = "checkpoint"  # For training verification
    SAMPLE = "sample"  # For inference verification


@dataclass
class TraceEvent:
    """A single event in an execution trace"""
    timestamp: float  # Unix timestamp with microsecond precision
    event_type: EventType
    device_id: str
    operation_id: Optional[str]  # Reference to operation in graph

    # Event-specific data
    data: Dict[str, Any] = field(default_factory=dict)

    # For challenge events
    challenge_data: Optional[Dict[str, Any]] = None

    # For memory events
    memory_usage: Optional[int] = None  # bytes

    @property
    def event_id(self) -> str:
        """Generate unique event identifier"""
        return f"{self.event_type.value}_{self.device_id}_{self.timestamp}"


@dataclass
class Trace:
    """
    Detailed log of actual execution.

    Always references a distributed graph because device information
    is necessary to interpret the events.
    """
    id: str
    graph_id: str  # Must be a distributed graph
    start_time: datetime
    end_time: datetime
    events: List[TraceEvent]

    # Metadata about the execution
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Challenge responses for verification protocols
    challenge_responses: List[Dict[str, Any]] = field(default_factory=list)

    def get_events_for_device(self, device_id: str) -> List[TraceEvent]:
        """Get all events for a specific device"""
        return [e for e in self.events if e.device_id == device_id]

    def get_events_for_operation(self, op_id: str) -> List[TraceEvent]:
        """Get all events for a specific operation"""
        return [e for e in self.events if e.operation_id == op_id]

    def get_events_in_window(self, start: float, end: float) -> List[TraceEvent]:
        """Get events within a time window"""
        return [e for e in self.events
                if start <= e.timestamp <= end]


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class TensorData:
    """Concrete tensor data"""
    id: str
    shape: Tuple[int, ...]
    dtype: str
    data: Union[jnp.ndarray, bytes]  # Can be serialized
    hash: str  # Cryptographic hash for verification

    # Metadata
    edge_id: Optional[str] = None  # Which edge in graph this corresponds to
    timestamp: Optional[float] = None
    device_id: Optional[str] = None

    @classmethod
    def from_array(cls, array: jnp.ndarray, edge_id: Optional[str] = None) -> 'TensorData':
        """Create TensorData from JAX array"""
        data_bytes = array.tobytes()
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        return cls(
            id=f"tensor_{data_hash[:16]}",
            shape=array.shape,
            dtype=str(array.dtype),
            data=array,
            hash=data_hash,
            edge_id=edge_id
        )

    def to_array(self) -> jnp.ndarray:
        """Convert to JAX array"""
        if isinstance(self.data, jnp.ndarray):
            return self.data
        else:
            import numpy as np
            return jnp.array(np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape))


@dataclass
class DataBundle:
    """
    Collection of concrete data from an execution.

    Can be bound to a graph for verification.
    """
    id: str
    graph_id: str  # Graph this data is for

    # Different categories of data
    inputs: Dict[str, TensorData]  # edge_id -> data
    outputs: Dict[str, TensorData]  # edge_id -> data
    weights: Dict[str, TensorData]  # param_name -> data
    activations: Dict[str, TensorData]  # edge_id -> data (optional, for challenges)

    # RNG states for reproducibility
    rng_seeds: Dict[str, int] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_tensors(self) -> Dict[str, TensorData]:
        """Get all tensors in the bundle"""
        all_tensors = {}
        all_tensors.update(self.inputs)
        all_tensors.update(self.outputs)
        all_tensors.update(self.weights)
        all_tensors.update(self.activations)
        return all_tensors


# -----------------------------------------------------------------------------
# Device Models
# -----------------------------------------------------------------------------

class DeviceType(Enum):
    """Type of compute device"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    ACCELERATOR = "accelerator"


@dataclass
class DeviceSpec:
    """Specification of a single compute device"""
    id: str
    device_type: DeviceType

    # Resource capacities
    memory_bytes: int
    compute_flops: float
    bandwidth_bytes_per_sec: float

    # Hardware details
    model: str  # e.g., "NVIDIA A100"
    location: str  # Physical or logical location

    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkLink:
    """Network connection between devices"""
    source_device_id: str
    target_device_id: str
    bandwidth_bytes_per_sec: float
    latency_seconds: float

    # Additional properties (e.g., RDMA support)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceTopology:
    """
    Hardware inventory and topology of compute cluster.

    Tracks devices and their interconnections.
    """
    devices: Dict[str, DeviceSpec]  # device_id -> DeviceSpec
    network_links: List[NetworkLink]

    # Metadata about the cluster
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_device(self, device_id: str) -> Optional[DeviceSpec]:
        """Get device by ID"""
        return self.devices.get(device_id)

    def get_connected_devices(self, device_id: str) -> List[str]:
        """Get IDs of devices connected to given device"""
        connected = []
        for link in self.network_links:
            if link.source_device_id == device_id:
                connected.append(link.target_device_id)
            elif link.target_device_id == device_id:
                connected.append(link.source_device_id)
        return connected

    def get_link(self, source: str, target: str) -> Optional[NetworkLink]:
        """Get network link between two devices"""
        for link in self.network_links:
            if (link.source_device_id == source and
                link.target_device_id == target):
                return link
            if (link.source_device_id == target and
                link.target_device_id == source):
                return link
        return None


# -----------------------------------------------------------------------------
# Challenge Models (for verification protocols)
# -----------------------------------------------------------------------------

@dataclass
class ChallengeRecord:
    """Record of a verification challenge"""
    id: str
    challenge_type: str  # e.g., "memory_spot_check", "activation_lsh"
    timestamp: float

    # Challenge parameters
    seed: int  # For LSH projection
    projection_dim: int
    target_device_id: Optional[str] = None
    target_operation_id: Optional[str] = None

    # Response
    response_value: Optional[jnp.ndarray] = None
    response_time: Optional[float] = None  # Time to respond

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointRecord:
    """Record of a training checkpoint for verification"""
    id: str
    step: int
    timestamp: float

    # Model state
    model_params: Dict[str, TensorData]
    optimizer_state: Optional[Dict[str, Any]] = None

    # Training data commitments
    batch_commitments: List[str] = field(default_factory=list)
    gradient_hashes: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)