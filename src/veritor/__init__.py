"""
Veritor: Workload verification for ML computations with IR store.
"""

from .db.models import (
    Graph,
    Trace, TraceEvent, EventType,
    DataBundle, TensorData,
    DeviceTopology, DeviceSpec, DeviceType, NetworkLink,
    ChallengeRecord, CheckpointRecord
)

from .db.api import (
    WorkloadDatabase,
    QueryFilter
)

from .verifier.graph_transformer import GraphTransformer
from .verifier.data_binder import DataBinder, BoundGraph
from .verifier.runner import ExecutionEngine, Verifier

from .db.ir_store import (
    IRStore,
    IRRole,
    IRFormat,
    IRBlob,
    GraphIRMapping,
    TransformationRecord
)

# Production sampling and transformation modules
from .common.sampler import (
    ProductionSampler,
    SimpleTokenSampler
)

from .verifier.ir_transformation import (
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