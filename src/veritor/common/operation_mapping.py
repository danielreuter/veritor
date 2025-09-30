"""
Operation ID mapping for correlating Python execution with StableHLO operations.

This module provides deterministic mapping between Python execution contexts
and StableHLO operation IDs, which is critical for the three-party architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class OperationExecution:
    """Record of an operation execution."""

    op_id: str
    """The operation ID."""

    timestamp: float
    """When the operation was executed."""

    context: Dict[str, Any] = field(default_factory=dict)
    """Execution context."""


class OperationIDMapper:
    """
    Maps between Python execution context and StableHLO operation IDs.

    This mapping is critical for:
    - Prover to know which operation it's executing
    - Challenger to issue challenges to specific operations
    - Verifier to correlate challenges with operations
    """

    def __init__(self):
        """Initialize the mapper."""
        self.operation_counter = 0
        self.operation_registry: Dict[str, str] = {}
        self.execution_trace: List[OperationExecution] = []

    def register_operation(self, python_context: str) -> str:
        """
        Register a Python operation and get its StableHLO ID.

        Args:
            python_context: Description of the Python operation (e.g., "layer_0_linear")

        Returns:
            The StableHLO operation ID
        """
        op_id = f"op_{self.operation_counter:04d}_{python_context}"
        self.operation_registry[python_context] = op_id
        self.operation_counter += 1
        return op_id

    def get_operation_id(self, python_context: str) -> str:
        """
        Get the StableHLO operation ID for a Python context.

        If the operation hasn't been registered, it will be registered.

        Args:
            python_context: Description of the Python operation

        Returns:
            The StableHLO operation ID
        """
        if python_context not in self.operation_registry:
            return self.register_operation(python_context)
        return self.operation_registry[python_context]

    def record_execution(self, op_id: str, context: Optional[Dict[str, Any]] = None):
        """
        Record that an operation was executed.

        Args:
            op_id: The operation ID that was executed
            context: Optional execution context
        """
        execution = OperationExecution(
            op_id=op_id,
            timestamp=datetime.now().timestamp(),
            context=context or {}
        )
        self.execution_trace.append(execution)

    def get_registry(self) -> Dict[str, str]:
        """
        Get the complete operation registry.

        Returns:
            Mapping from Python contexts to operation IDs
        """
        return self.operation_registry.copy()

    def get_execution_trace(self) -> List[OperationExecution]:
        """
        Get the execution trace.

        Returns:
            List of operation executions in order
        """
        return self.execution_trace.copy()

    def clear(self):
        """Clear all mappings and trace."""
        self.operation_counter = 0
        self.operation_registry.clear()
        self.execution_trace.clear()


class HookPoint:
    """
    Represents a hook point in the computation graph.

    Hook points are locations where challenges can be injected.
    """

    def __init__(self, operation_id: str, hook_type: str = "challenge"):
        """
        Initialize a hook point.

        Args:
            operation_id: The operation ID for this hook
            hook_type: Type of hook (challenge, outfeed, etc.)
        """
        self.operation_id = operation_id
        self.hook_type = hook_type
        self.metadata: Dict[str, Any] = {}

    def set_metadata(self, key: str, value: Any):
        """Set metadata for this hook point."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata for this hook point."""
        return self.metadata.get(key, default)