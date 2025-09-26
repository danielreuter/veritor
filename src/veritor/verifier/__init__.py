"""
Verifier module for Veritor.

Handles re-execution, comparison, and verification of prover claims.
"""

from .runner import (
    ExecutionEngine,
    Verifier,
)

from .compare import (
    compare_tensors,
    compare_lsh_projections,
    compare_sampling_distributions,
    compute_activation_fingerprint,
)

__all__ = [
    'ExecutionEngine',
    'Verifier',
    'compare_tensors',
    'compare_lsh_projections',
    'compare_sampling_distributions',
    'compute_activation_fingerprint',
]