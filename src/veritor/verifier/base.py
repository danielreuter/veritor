"""
Base Verifier class for the three-party verification architecture.

This module provides the base Verifier abstraction that handles:
- Reading execution data from database
- Reconstructing computation
- Validating challenge responses
- Schedule adherence checking
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from veritor.challenger import ChallengeSchedule
from veritor.db.api import WorkloadDatabase
from veritor.db.models import ChallengeRecord
from veritor.verifier.engine import (
    UnifiedVerificationEngine,
    VerificationConfig,
    VerificationResult,
)


class BaseVerifier(ABC):
    """
    Base Verifier class that handles common verification functionality.

    The Verifier:
    - Reads execution data from database after execution
    - Reconstructs computation from traces and challenges
    - Validates that challenges were correctly computed
    - Checks schedule adherence

    Subclasses can implement specific verification strategies.
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize the base Verifier.

        Args:
            config: Optional verification configuration
        """
        self.config = config or VerificationConfig()
        self.verification_engine: Optional[UnifiedVerificationEngine] = None
        self._schedule: Optional[ChallengeSchedule] = None

    def set_schedule(self, schedule: ChallengeSchedule):
        """
        Set the challenge schedule for verification.

        The Verifier receives this from the Challenger to verify
        that the execution followed the schedule.

        Args:
            schedule: The challenge schedule that was used
        """
        self._schedule = schedule

    def verify_execution(
        self,
        database: WorkloadDatabase,
        graph_id: str,
        trace_id: str
    ) -> VerificationResult:
        """
        Verify the execution by reconstructing computation.

        Args:
            database: The workload database
            graph_id: ID of the graph to verify
            trace_id: ID of the execution trace

        Returns:
            VerificationResult with detailed results
        """
        print("\n🔐 Verifying execution...")

        # Initialize verification engine if needed
        if self.verification_engine is None:
            self.verification_engine = UnifiedVerificationEngine(database, self.config)

        # Run core verification using the engine
        result = self.verification_engine.verify_execution(graph_id, trace_id)

        # Add schedule adherence verification if we have a schedule
        if self._schedule:
            result = self._verify_schedule_adherence(database, graph_id, trace_id, result)

        # Allow subclasses to add custom verification
        result = self.verify_custom(database, graph_id, trace_id, result)

        return result

    def _verify_schedule_adherence(
        self,
        database: WorkloadDatabase,
        graph_id: str,
        trace_id: str,
        result: VerificationResult
    ) -> VerificationResult:
        """
        Verify that the challenge schedule was followed.

        Args:
            database: The workload database
            graph_id: ID of the graph
            trace_id: ID of the trace
            result: Current verification result

        Returns:
            Updated verification result
        """
        if not self._schedule:
            return result

        # Get challenges from database
        challenges = database.get_challenges_for_trace(trace_id)

        # Get scheduled operations
        scheduled_ops = set(self._schedule.operation_challenges.keys())

        # Get challenged operations (non-zero responses)
        challenged_ops = {
            c.target_operation_id
            for c in challenges
            if self._is_non_zero_challenge(c)
        }

        # Check schedule adherence
        overlap = scheduled_ops & challenged_ops
        schedule_followed = len(overlap) > 0 if scheduled_ops else True

        # Update result
        if not schedule_followed:
            result.errors.append(
                f"Schedule not followed: scheduled {len(scheduled_ops)} ops, "
                f"challenged {len(challenged_ops)} ops, overlap {len(overlap)}"
            )
            result.success = False

        # Add metrics
        result.metrics["scheduled_operations"] = len(scheduled_ops)
        result.metrics["challenged_operations"] = len(challenged_ops)
        result.metrics["schedule_overlap"] = len(overlap)
        result.metrics["schedule_adherence"] = schedule_followed

        print(f"  Schedule adherence: {schedule_followed}")
        print(f"    Scheduled: {len(scheduled_ops)} operations")
        print(f"    Challenged: {len(challenged_ops)} operations")
        print(f"    Overlap: {len(overlap)} operations")

        return result

    def _is_non_zero_challenge(self, challenge: ChallengeRecord) -> bool:
        """
        Check if a challenge has a non-zero response.

        Args:
            challenge: The challenge record

        Returns:
            True if the challenge has a non-zero response
        """
        try:
            response = challenge.response_value
            if isinstance(response, list) and len(response) > 0:
                # Check if any value is non-zero
                if isinstance(response[0], list):
                    return any(v != 0 for v in response[0])
                else:
                    return any(v != 0 for v in response)
            return False
        except:
            return False

    @abstractmethod
    def verify_custom(
        self,
        database: WorkloadDatabase,
        graph_id: str,
        trace_id: str,
        result: VerificationResult
    ) -> VerificationResult:
        """
        Perform custom verification specific to the subclass.

        Subclasses can override this to add their own verification logic.

        Args:
            database: The workload database
            graph_id: ID of the graph
            trace_id: ID of the trace
            result: Current verification result

        Returns:
            Updated verification result
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about verifier activity."""
        stats = {
            "has_schedule": self._schedule is not None,
            "config_enabled": {
                "execution_rtol": self.config.execution_rtol,
                "lsh_rtol": self.config.lsh_rtol,
                "enable_jit_vs_python": self.config.enable_jit_vs_python,
                "enable_challenge_verification": self.config.enable_challenge_verification
            }
        }

        if self._schedule:
            stats["scheduled_challenges"] = len(self._schedule.operation_challenges)

        return stats