"""
Veritor: Workload verification for ML computations with IR store.
"""

from .data_models import (
    Graph,
    Trace, TraceEvent, EventType,
    DataBundle, TensorData,
    DeviceTopology, DeviceSpec, DeviceType, NetworkLink,
    ChallengeRecord, CheckpointRecord
)

from .api import (
    WorkloadDatabase,
    QueryFilter,
    GraphTransformer,
    DataBinder, BoundGraph,
    ExecutionEngine,
    Verifier
)

from .ir_store import (
    IRStore,
    IRRole,
    IRFormat,
    IRBlob,
    GraphIRMapping,
    TransformationRecord
)

# Production sampling and transformation modules
from .sampling import (
    ProductionSampler,
    SimpleTokenSampler
)

from .transformation import (
    rewrite_decode_to_teacher_forcing,
    extract_function,
    list_functions
)

__version__ = "0.2.0"

__all__ = [
    # Data models
    "Graph",
    "Trace", "TraceEvent", "EventType",
    "DataBundle", "TensorData",
    "DeviceTopology", "DeviceSpec", "DeviceType", "NetworkLink",
    "ChallengeRecord", "CheckpointRecord",

    # API
    "WorkloadDatabase",
    "QueryFilter",
    "GraphTransformer",
    "DataBinder", "BoundGraph",
    "ExecutionEngine",
    "Verifier",

    # IR Store
    "IRStore",
    "IRRole",
    "IRFormat",
    "IRBlob",
    "GraphIRMapping",
    "TransformationRecord",

    # Production sampling
    "ProductionSampler",
    "SimpleTokenSampler",

    # Transformation utilities
    "rewrite_decode_to_teacher_forcing",
    "extract_function",
    "list_functions",
]