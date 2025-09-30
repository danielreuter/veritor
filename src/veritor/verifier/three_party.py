"""
Three-party architecture support for the Verifier.

This module provides Verifier functionality for the three-party verification
architecture with post-execution validation.
"""

from typing import Any, Dict, List, Optional

from veritor.challenger import ChallengeSchedule
from veritor.db.api import WorkloadDatabase
from veritor.verifier.base import BaseVerifier
from veritor.verifier.engine import (
    VerificationConfig,
    VerificationResult,
)


class ThreePartyVerifier(BaseVerifier):
    """
    Verifier implementation for the three-party architecture.

    The Verifier:
    - Reads database after execution
    - Reconstructs and validates execution
    - Verifies schedule adherence (schedule comes from Challenger)
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize the Verifier.

        Args:
            config: Optional verification configuration
        """
        super().__init__(config)

    def receive_schedule(self, schedule: ChallengeSchedule):
        """
        Receive the challenge schedule from the Challenger.

        The Verifier uses this to verify schedule adherence.

        Args:
            schedule: The challenge schedule from Challenger
        """
        print("\n📋 Verifier received schedule from Challenger")
        self.set_schedule(schedule)
        print(f"  Schedule contains {len(schedule.operation_challenges)} challenges")
        if schedule.operation_challenges:
            print(f"  Challenge targets: {list(schedule.operation_challenges.keys())[:3]}...")

    def verify_custom(
        self,
        database: WorkloadDatabase,
        graph_id: str,
        trace_id: str,
        result: VerificationResult
    ) -> VerificationResult:
        """
        Three-party specific verification.

        Currently just returns the result as-is, but can be extended
        for three-party specific checks.

        Args:
            database: The workload database
            graph_id: ID of the graph
            trace_id: ID of the trace
            result: Current verification result

        Returns:
            Updated verification result
        """
        # Could add three-party specific checks here
        return result


    def verify_post_execution(
        self,
        database: WorkloadDatabase,
        graph_id: str,
        trace_id: str,
        expected_challenges: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Simplified post-execution verification.

        Args:
            database: The workload database
            graph_id: ID of the graph
            trace_id: ID of the trace
            expected_challenges: Optional list of expected challenge operation IDs

        Returns:
            Dictionary with verification results
        """
        # Get trace and challenges
        trace = database.get_trace(trace_id)
        challenges = database.get_challenges_for_trace(trace_id)

        if not trace:
            return {"success": False, "error": "Trace not found"}

        # Basic verification
        verification_results = {}
        for challenge in challenges:
            op_id = challenge.target_operation_id
            has_response = len(challenge.response_value) > 0
            is_non_zero = self._is_non_zero_challenge(challenge)

            verification_results[challenge.id] = {
                "success": has_response,
                "operation_id": op_id,
                "is_non_zero": is_non_zero
            }

        # Summary
        n_verified = sum(1 for r in verification_results.values() if r["success"])
        n_total = len(verification_results)

        return {
            "success": n_verified == n_total,
            "challenges_verified": n_verified,
            "challenges_total": n_total,
            "details": verification_results
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about verifier activity."""
        # Use base class statistics
        return super().get_statistics()