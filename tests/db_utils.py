"""
Database utilities for testing and experiments.

Provides helper functions for managing WorkloadDatabase lifecycle
in tests and experiments.
"""

from contextlib import contextmanager
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import uuid


@contextmanager
def temp_workload_db():
    """
    Create a temporary WorkloadDatabase with auto-cleanup.

    Usage:
        with temp_workload_db() as (db, temp_path):
            # Use db here
            db.save(temp_path / "checkpoint")
        # Auto-cleanup on exit
    """
    from src.veritor.db.api import WorkloadDatabase

    db = WorkloadDatabase()
    temp_dir = tempfile.mkdtemp(prefix="veritor_db_")

    try:
        yield db, Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class ExperimentHarness:
    """Manage persistent databases for longer experiments."""

    def __init__(self, experiment_name: str):
        self.exp_dir = Path("experiments") / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.exp_dir / "workload_db"

    def get_or_create_db(self):
        """Load existing or create new database."""
        from src.veritor.db.api import WorkloadDatabase

        if self.db_path.exists():
            print(f"Loading existing database from {self.db_path}")
            return WorkloadDatabase.load(str(self.db_path))
        else:
            print(f"Creating new database at {self.db_path}")
            return WorkloadDatabase()

    def save_db(self, db):
        """Save database to experiment directory."""
        db.save(str(self.db_path))

    def cleanup(self):
        """Remove experiment directory."""
        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)


def create_sample_graph_and_trace(db, graph_id: Optional[str] = None):
    """
    Helper to quickly populate a database with sample data.

    Returns: (graph_id, trace_id)
    """
    from src.veritor.db.ir_store import IRRole, IRFormat
    from src.veritor.db.models import Trace, TraceEvent, EventType

    if graph_id is None:
        graph_id = f"test_model_{uuid.uuid4().hex[:8]}"

    # Store graph with IR
    actual_graph_id = db.store_graph_with_ir(
        graph_id,
        "module @test { func.func @main() { return } }",
        IRRole.LOGICAL,
        IRFormat.STABLEHLO,
        metadata={'created_at': datetime.now().isoformat()}
    )

    # Create and store trace
    trace = Trace(
        id=f"trace_{uuid.uuid4().hex[:8]}",
        graph_id=actual_graph_id,
        start_time=datetime.now(),
        end_time=datetime.now(),
        events=[
            TraceEvent(
                id="event_1",
                type=EventType.KERNEL_LAUNCH,
                timestamp=datetime.now().timestamp(),
                device_id="device_0",
                operation_name="test_op",
                metadata={}
            )
        ],
        metadata={'test_trace': True}
    )
    trace_id = db.store_trace(trace)

    return actual_graph_id, trace_id


class DatabaseSandbox:
    """
    Isolate database operations in a sandbox.

    Useful for testing operations without affecting the base database.
    """

    def __init__(self, base_db=None):
        from src.veritor.db.api import WorkloadDatabase

        self.base_db = base_db
        self.sandbox_db = WorkloadDatabase()

        # Copy base database if provided
        if base_db:
            self._copy_from(base_db)

    def _copy_from(self, source_db):
        """Copy data from source database."""
        self.sandbox_db.graphs = source_db.graphs.copy()
        self.sandbox_db.traces = source_db.traces.copy()
        self.sandbox_db.data_bundles = source_db.data_bundles.copy()
        self.sandbox_db.challenges = source_db.challenges.copy()
        self.sandbox_db.checkpoints = source_db.checkpoints.copy()
        # Share IR store reference (not copying blobs)
        self.sandbox_db.ir_store = source_db.ir_store

    def commit(self):
        """Commit sandbox changes to base database."""
        if self.base_db:
            self.base_db.graphs.update(self.sandbox_db.graphs)
            self.base_db.traces.update(self.sandbox_db.traces)
            self.base_db.data_bundles.update(self.sandbox_db.data_bundles)
            self.base_db.challenges.extend(self.sandbox_db.challenges)
            self.base_db.checkpoints.extend(self.sandbox_db.checkpoints)

    def rollback(self):
        """Discard sandbox changes."""
        from src.veritor.db.api import WorkloadDatabase

        self.sandbox_db = WorkloadDatabase()
        if self.base_db:
            self._copy_from(self.base_db)

    @contextmanager
    def transaction(self):
        """Run operations in a transaction."""
        try:
            yield self.sandbox_db
            self.commit()
        except Exception:
            self.rollback()
            raise


def inspect_database(db_path: str):
    """
    Quick inspection of a saved database.

    Usage:
        python -c "from tests.db_utils import inspect_database; inspect_database('path/to/db')"
    """
    from src.veritor.db.api import WorkloadDatabase

    db = WorkloadDatabase.load(db_path)

    print(f"\nDatabase at: {db_path}")
    print(f"  Graphs: {len(db.graphs)}")
    print(f"  Traces: {len(db.traces)}")
    print(f"  Data Bundles: {len(db.data_bundles)}")
    print(f"  Challenges: {len(db.challenges)}")
    print(f"  Checkpoints: {len(db.checkpoints)}")

    # List graphs with their IR roles
    if db.graphs:
        print("\nGraphs:")
        for graph_id, graph in db.graphs.items():
            roles = db.ir_store.list_ir_roles(graph_id)
            role_names = [r[0].value for r in roles] if roles else ["no IR"]
            print(f"  {graph_id}: {role_names}")

    return db