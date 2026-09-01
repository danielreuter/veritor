"""
Utility functions for the Compute Accounting Protocol.

This module contains helper functions for working with StableHLO,
tensors, and other common operations.
"""

import hashlib
import re

import numpy as np


# TODO: this should be a down-projection, not a sha256 hash
def compute_tensor_hash(tensor: np.ndarray) -> str:
    """
    Compute a cryptographic hash of a tensor.


    Args:
        tensor: NumPy array to hash

    Returns:
        SHA256 hash as hex string
    """
    if isinstance(tensor, np.ndarray):
        data_bytes = tensor.tobytes()
    else:
        data_bytes = bytes(tensor)
    return hashlib.sha256(data_bytes).hexdigest()


def compute_graph_hash(graph: str) -> str:
    """
    Compute a normalized hash of a StableHLO graph.

    Normalizes whitespace for consistent hashing.

    Args:
        graph: StableHLO MLIR text

    Returns:
        SHA256 hash as hex string
    """
    # Normalize whitespace for consistent hashing
    normalized = "\n".join(line.strip() for line in graph.splitlines() if line.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_module_info(graph: str) -> dict[str, any]:
    """
    Extract metadata from a StableHLO module.

    Args:
        graph: StableHLO MLIR text

    Returns:
        Dictionary with module_name, entry_points, etc.
    """
    info = {"module_name": None, "entry_points": [], "has_private_functions": False}

    # Extract module name
    module_match = re.search(r"module @(\w+)", graph)
    if module_match:
        info["module_name"] = module_match.group(1)

    # Find all functions
    func_pattern = r"func\.func\s+(@\w+)"
    private_pattern = r"func\.func\s+private\s+(@\w+)"

    # Check for private functions
    if re.search(private_pattern, graph):
        info["has_private_functions"] = True

    # Extract all public entry points
    for match in re.finditer(func_pattern, graph):
        func_name = match.group(1).lstrip("@")
        # Skip if this was a private function
        if not re.search(f"private.*{re.escape(match.group(1))}", graph):
            info["entry_points"].append(func_name)

    return info


def tensor_stats(tensor: np.ndarray) -> dict:
    """
    Compute basic statistics for a tensor.

    Useful for debugging and verification diagnostics.

    Args:
        tensor: NumPy array

    Returns:
        Dictionary with min, max, mean, std, shape, dtype
    """
    return {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "min": float(np.min(tensor)),
        "max": float(np.max(tensor)),
        "mean": float(np.mean(tensor)),
        "std": float(np.std(tensor)),
        "hash": compute_tensor_hash(tensor)[:8],  # First 8 chars for brevity
    }


def format_claim_summary(claim) -> str:
    """
    Format a human-readable summary of a claim.

    Args:
        claim: A Claim object

    Returns:
        Formatted string summary
    """
    lines = [
        f"Claim ID: {claim.id[:8]}...",
        f"Strategy: {claim.strategy}",
        f"Created: {claim.created_at.isoformat()}",
        f"Entry point: {claim.entry_point}",
        f"Inputs: {len(claim.inputs)} tensors",
        f"Outputs: {len(claim.outputs)} tensors",
    ]

    # Add metadata if present
    if claim.metadata:
        lines.append("Metadata:")
        for key, value in claim.metadata.items():
            lines.append(f"  {key}: {value}")

    # Add graph hash
    lines.append(f"Graph hash: {compute_graph_hash(claim.graph)[:16]}...")

    return "\n".join(lines)
