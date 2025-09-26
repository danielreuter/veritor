"""
Database module for Veritor.

Provides data models, IR storage, and the main WorkloadDatabase API.
"""

from .api import (
    QueryFilter,
    WorkloadDatabase,
)
from .ir_store import (
    GraphIRMapping,
    IRBlob,
    IRFormat,
    IRRole,
    IRStore,
    TransformationRecord,
)
from .models import (
    ChallengeRecord,
    CheckpointRecord,
    DataBundle,
    DeviceSpec,
    DeviceTopology,
    DeviceType,
    EventType,
    Graph,
    TensorData,
    Trace,
    TraceEvent,
)

__all__ = [
    # Models
    "Graph",
    "Trace",
    "TraceEvent",
    "EventType",
    "DataBundle",
    "TensorData",
    "DeviceTopology",
    "DeviceSpec",
    "DeviceType",
    "ChallengeRecord",
    "CheckpointRecord",
    # IR Store
    "IRStore",
    "IRBlob",
    "IRFormat",
    "IRRole",
    "GraphIRMapping",
    "TransformationRecord",
    # API
    "WorkloadDatabase",
    "QueryFilter",
]
