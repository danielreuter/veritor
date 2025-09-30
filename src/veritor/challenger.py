"""
Challenger module for the three-party verification architecture.

The Challenger is an external entity that:
- Generates secret challenge schedules based on graph structure
- Responds to queries from Prover during execution
- Has no knowledge of computational state
- Maintains cryptographic unpredictability
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChallengeSchedule:
    """Schedule of challenges to be issued during execution."""

    operation_challenges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Mapping from operation ID to challenge parameters."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the schedule."""


@dataclass
class ChallengeQuery:
    """Query from Prover to Challenger."""

    operation_id: str
    """The operation ID being executed."""

    timestamp: float
    """When the query was made."""

    context: Dict[str, Any] = field(default_factory=dict)
    """Additional execution context."""


@dataclass
class ChallengeResponse:
    """Response from Challenger to Prover."""

    should_challenge: bool
    """Whether to execute a challenge at this point."""

    challenge_type: Optional[str] = None
    """Type of challenge (lsh_dynamic, hash, etc.)."""

    seed: Optional[int] = None
    """Random seed for challenge computation."""

    projection_dim: Optional[int] = None
    """Dimension for LSH projection."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    """Additional challenge parameters."""


class Challenger:
    """
    External challenger entity in the three-party architecture.

    The Challenger:
    - Generates challenge schedules from graph structure
    - Maintains complete isolation from Prover's computational state
    - Responds to queries based on the secret schedule
    - Ensures cryptographic unpredictability
    """

    def __init__(self):
        """Initialize the Challenger."""
        self.schedule: Optional[ChallengeSchedule] = None
        self.queries_received: List[ChallengeQuery] = []
        self.responses_sent: List[ChallengeResponse] = []

    def generate_schedule(
        self,
        stablehlo: str,
        operation_mapping: Dict[str, str],
        challenge_probability: float = 0.3,
        seed: Optional[int] = None
    ) -> ChallengeSchedule:
        """
        Generate a challenge schedule from a graph.

        This is the Challenger's core responsibility - analyzing the graph
        structure and deciding which operations to challenge.

        Args:
            stablehlo: The StableHLO representation
            operation_mapping: Mapping of Python contexts to operation IDs
            challenge_probability: Probability of challenging each operation
            seed: Optional seed for deterministic schedule

        Returns:
            ChallengeSchedule for this execution
        """
        print(f"\n🎲 Challenger generating schedule from graph...")

        # Use hash of StableHLO as entropy source if no seed provided
        if seed is None:
            graph_hash = hashlib.sha256(stablehlo.encode()).hexdigest()
        else:
            graph_hash = hashlib.sha256(f"{seed}".encode()).hexdigest()

        schedule = ChallengeSchedule(
            metadata={
                "graph_hash": graph_hash[:16],
                "challenge_probability": challenge_probability,
                "created_at": datetime.now().isoformat(),
                "total_operations": len(operation_mapping)
            }
        )

        # Decide which operations to challenge
        for python_context, op_id in operation_mapping.items():
            # Use graph hash + op_id for deterministic decision
            decision_hash = hashlib.md5(f"{graph_hash}_{op_id}".encode()).hexdigest()
            decision_value = int(decision_hash[:8], 16) / (2**32)

            if decision_value < challenge_probability:
                # Generate challenge parameters
                seed_value = int(decision_hash[8:16], 16) % (2**31)

                schedule.operation_challenges[op_id] = {
                    "type": "lsh_dynamic",
                    "seed": seed_value,
                    "projection_dim": 4,
                    "python_context": python_context
                }

        # Store the schedule internally
        self.schedule = schedule

        print(f"  Generated schedule with {len(schedule.operation_challenges)} challenges")
        print(f"  Total operations: {len(operation_mapping)}")
        if schedule.operation_challenges:
            print(f"  Challenge targets: {list(schedule.operation_challenges.keys())[:3]}...")

        return schedule

    def query_challenge(self, operation_id: str, context: Optional[Dict[str, Any]] = None) -> ChallengeResponse:
        """
        Respond to a challenge query from the Prover.

        Args:
            operation_id: The operation being executed
            context: Optional execution context

        Returns:
            ChallengeResponse indicating whether and how to challenge
        """
        # Record the query
        query = ChallengeQuery(
            operation_id=operation_id,
            timestamp=datetime.now().timestamp(),
            context=context or {}
        )
        self.queries_received.append(query)

        # Check if we have a schedule
        if not self.schedule:
            response = ChallengeResponse(should_challenge=False)
            self.responses_sent.append(response)
            return response

        # Check if this operation should be challenged
        if operation_id in self.schedule.operation_challenges:
            challenge_params = self.schedule.operation_challenges[operation_id]

            response = ChallengeResponse(
                should_challenge=True,
                challenge_type=challenge_params.get("type", "lsh_dynamic"),
                seed=challenge_params.get("seed"),
                projection_dim=challenge_params.get("projection_dim", 4),
                parameters=challenge_params
            )

            print(f"  💥 Challenge issued for {operation_id}")
        else:
            response = ChallengeResponse(should_challenge=False)

        self.responses_sent.append(response)
        return response

    def get_schedule(self) -> Optional[ChallengeSchedule]:
        """
        Get the current schedule (for Verifier to validate against).

        Returns:
            The current challenge schedule, if any
        """
        return self.schedule

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about challenger activity."""
        return {
            "queries_received": len(self.queries_received),
            "responses_sent": len(self.responses_sent),
            "challenges_issued": sum(1 for r in self.responses_sent if r.should_challenge),
            "scheduled_operations": len(self.schedule.operation_challenges) if self.schedule else 0,
            "total_operations": self.schedule.metadata.get("total_operations", 0) if self.schedule else 0
        }


# Backward compatibility
ChallengeScheduleBuilder = Challenger  # Deprecated, use Challenger.generate_schedule directly