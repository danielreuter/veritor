"""
Veritor - Compute Accounting Protocol (CAP) implementation.

A minimal, claim-based system for verifiable ML workload accounting.
"""

from veritor.accounting import Claim, ClaimDatabase, create_claim_from_jax_function
from veritor.verify import (
    VerificationResult,
    VerificationTest,
    BitExactTest,
    StampTest,
    TrainingReplayTest,
    get_test,
    run_test,
    TEST_REGISTRY,
    verify,
)

__version__ = "0.1.0"

__all__ = [
    # Core accounting
    "Claim",
    "ClaimDatabase",
    "create_claim_from_jax_function",
    # Verification
    "verify",
    "VerificationResult",
    "VerificationTest",
    "BitExactTest",
    "StampTest",
    "TrainingReplayTest",
    "get_test",
    "run_test",
    "TEST_REGISTRY",
]