"""
Core accounting components for the Compute Accounting Protocol.

This module contains the Claim dataclass, ClaimDatabase, and utilities
for creating claims from computational workloads.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Claim:
    """
    A self-contained, verifiable assertion about a computation.

    The graph in a Claim is a reference implementation optimized for
    verification, not necessarily the actual production code.
    """

    graph: str  # StableHLO MLIR text
    inputs: list[np.ndarray]  # Input tensors (positional)
    outputs: list[np.ndarray]  # Claimed output tensors (positional)
    test_spec: dict[str, Any]  # {"name": "bit_exact", "config": {...}}
    entry_point: str = "main"  # Function name to invoke
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> bool:
        """Basic validation of claim structure."""
        # Check graph is non-empty
        if not self.graph or not self.graph.strip():
            return False

        # Check test spec is provided
        if not self.test_spec or "name" not in self.test_spec:
            return False

        # Check inputs/outputs are provided
        if not self.inputs or not self.outputs:
            return False

        return True


class ClaimDatabase:
    """
    A queryable collection of Claims.

    This is a simple in-memory implementation. Production deployments
    could use SQLite, PostgreSQL, or any other backend.
    """

    def __init__(self):
        self.claims: dict[str, Claim] = {}

    def add_claim(self, claim: Claim) -> str:
        """Add a claim and return its ID."""
        if not claim.validate():
            raise ValueError("Invalid claim structure")
        self.claims[claim.id] = claim
        return claim.id

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        return self.claims.get(claim_id)

    def query_by_strategy(self, strategy: str) -> list[Claim]:
        """Get all claims with a specific strategy."""
        return [c for c in self.claims.values() if c.strategy == strategy]

    def query_by_time_range(self, start: datetime, end: datetime) -> list[Claim]:
        """Get claims created within a time range."""
        return [c for c in self.claims.values() if start <= c.created_at <= end]

    def query_by_metadata(self, key: str, value: Any) -> list[Claim]:
        """Get claims with specific metadata."""
        return [c for c in self.claims.values() if c.metadata.get(key) == value]

    def list_all(self) -> list[Claim]:
        """Get all claims."""
        return list(self.claims.values())

    def clear(self):
        """Clear all claims."""
        self.claims.clear()

    def __len__(self) -> int:
        """Number of claims in database."""
        return len(self.claims)

    def __repr__(self) -> str:
        return f"ClaimDatabase({len(self)} claims)"


def create_claim_from_jax_function(
    fn: Callable,
    inputs: list[np.ndarray],
    test_name: str = "bit_exact",
    test_config: dict[str, Any] | None = None,
    entry_point: str = "main",
    **metadata,
) -> Claim:
    """
    Create a Claim from a JAX function by exporting to StableHLO.

    This is what the prover uses to create verifiable claims from
    their production workloads.

    Args:
        fn: JAX-compatible function to capture
        inputs: Input tensors for the function
        test_name: Name of verification test to use
        test_config: Optional config for the test (uses defaults if not provided)
        entry_point: Entry point name in the generated module
        **metadata: Additional metadata to include in the claim

    Returns:
        A Claim object ready for verification
    """
    # Convert inputs to JAX arrays, preserving their dtypes
    jax_inputs = [jnp.array(x) for x in inputs]

    # Execute the function to get outputs
    outputs = fn(*jax_inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs = [np.array(o) for o in outputs]

    # Export to StableHLO
    lowered = jax.jit(fn).lower(*jax_inputs)
    graph = lowered.as_text()

    # Build test spec
    test_spec = {"name": test_name}
    if test_config is not None:
        test_spec["config"] = test_config

    return Claim(
        graph=graph,
        inputs=[np.array(x) for x in inputs],  # Preserve original dtypes
        outputs=outputs,
        test_spec=test_spec,
        entry_point=entry_point,
        metadata=metadata,
    )
