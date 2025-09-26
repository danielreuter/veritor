"""
Prover module for Veritor.

Handles computation execution, challenge responses, and writing to the database.
"""

from .runner import (
    Prover,
    ProverConfig,
    ModelExecutor,
)

from .hooks import (
    HookSystem,
)

__all__ = [
    'Prover',
    'ProverConfig',
    'ModelExecutor',
    'HookSystem',
]