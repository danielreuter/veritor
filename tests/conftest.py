"""
Shared pytest fixtures and configuration.
"""

import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from veritor.db.api import WorkloadDatabase
from veritor.db.ir_store import IRFormat, IRRole, IRStore
from veritor.db.models import EventType, Trace, TraceEvent


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    temp = tempfile.mkdtemp(prefix="veritor_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@contextmanager
def temp_workload_db():
    """Create a temporary WorkloadDatabase with auto-cleanup."""
    db = WorkloadDatabase()
    temp_dir = tempfile.mkdtemp(prefix="veritor_db_")

    try:
        yield db, Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def ir_store():
    """Provide a fresh IRStore instance."""
    return IRStore()


@pytest.fixture
def workload_db():
    """Provide a fresh WorkloadDatabase instance."""
    return WorkloadDatabase()


@pytest.fixture
def temp_db():
    """Provide a temporary database that auto-cleans."""
    with temp_workload_db() as (db, path):
        yield db, path


@pytest.fixture
def populated_db(temp_db):
    """Provide a database pre-populated with test data."""
    db, path = temp_db

    # Add sample graph
    graph_id = db.store_graph_with_ir(
        f"test_model_{uuid.uuid4().hex[:8]}",
        "module @test { func.func @main() { return } }",
        IRRole.LOGICAL,
        IRFormat.STABLEHLO,
        metadata={"test_data": True},
    )

    # Add sample trace
    trace = Trace(
        id=f"trace_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        start_time=datetime.now(),
        end_time=datetime.now(),
        events=[
            TraceEvent(
                id="event_1",
                type=EventType.KERNEL_LAUNCH,
                timestamp=datetime.now().timestamp(),
                device_id="device_0",
                operation_name="test_op",
                metadata={},
            )
        ],
        metadata={"test_trace": True},
    )
    db.store_trace(trace)

    return db, path


@pytest.fixture
def sample_stablehlo():
    """Provide sample StableHLO IR."""
    return """
    module @test {
        func.func @main(%arg0: tensor<32x64xf32>) -> tensor<32x10xf32> {
            %0 = stablehlo.constant dense<0.0> : tensor<64x10xf32>
            %1 = stablehlo.dot_general %arg0, %0,
                contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64x10xf32>) -> tensor<32x10xf32>
            return %1 : tensor<32x10xf32>
        }
    }
    """


@pytest.fixture
def sample_hlo():
    """Provide sample HLO IR (distributed)."""
    return """
    HloModule test, entry_computation_layout={(f32[32,64]{1,0})->f32[32,10]{1,0}}

    ENTRY main {
        %arg0 = f32[32,64]{1,0} parameter(0), sharding={devices=[2,1]0,1}
        %const = f32[64,10]{1,0} constant({...}), sharding={replicated}
        ROOT %dot = f32[32,10]{1,0} dot(%arg0, %const),
            lhs_contracting_dims={1}, rhs_contracting_dims={0},
            sharding={devices=[2,1]0,1}
    }
    """
