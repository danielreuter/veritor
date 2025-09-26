"""
Runtime monitoring and ping protocols for interactive verification.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class Challenge:
    """A verification challenge"""

    id: str
    trace_id: str
    timestamp: float
    challenge_type: str  # 'memory', 'activation', 'checkpoint'
    target: Dict[str, Any]  # What to challenge (device, layer, etc.)
    nonce: int
    response_deadline: float  # Maximum response time
    metadata: Dict[str, Any]


@dataclass
class ChallengeResponse:
    """Response to a challenge"""

    challenge_id: str
    timestamp: float
    response_data: Dict[str, Any]
    response_time: float  # Time taken to respond


class ChallengeProtocol:
    """
    Protocol for issuing and verifying runtime challenges.
    """

    def __init__(self, max_response_time: float = 1.0):
        """
        Initialize challenge protocol.

        Args:
            max_response_time: Maximum allowed response time in seconds
        """
        self.max_response_time = max_response_time
        self.pending_challenges: Dict[str, Challenge] = {}
        self.responses: Dict[str, ChallengeResponse] = {}

    def issue_challenge(
        self, trace_id: str, challenge_type: str, target: Dict[str, Any]
    ) -> Challenge:
        """
        Issue a new challenge.

        Args:
            trace_id: ID of the trace being challenged
            challenge_type: Type of challenge
            target: What to challenge (e.g., device_id, layer_id)

        Returns:
            The issued challenge
        """
        timestamp = datetime.now().timestamp()

        # Generate challenge ID and nonce
        challenge_id = self._generate_challenge_id(trace_id, timestamp)
        nonce = self._generate_nonce(challenge_id, timestamp)

        challenge = Challenge(
            id=challenge_id,
            trace_id=trace_id,
            timestamp=timestamp,
            challenge_type=challenge_type,
            target=target,
            nonce=nonce,
            response_deadline=timestamp + self.max_response_time,
            metadata={},
        )

        self.pending_challenges[challenge_id] = challenge
        return challenge

    def submit_response(
        self, challenge_id: str, response_data: Dict[str, Any]
    ) -> ChallengeResponse:
        """
        Submit a response to a challenge.

        Args:
            challenge_id: ID of the challenge
            response_data: Response data (e.g., LSH projection)

        Returns:
            The challenge response

        Raises:
            ValueError: If challenge not found or deadline exceeded
        """
        if challenge_id not in self.pending_challenges:
            raise ValueError(f"Challenge {challenge_id} not found")

        challenge = self.pending_challenges[challenge_id]
        response_timestamp = datetime.now().timestamp()

        if response_timestamp > challenge.response_deadline:
            raise ValueError(f"Response deadline exceeded for challenge {challenge_id}")

        response = ChallengeResponse(
            challenge_id=challenge_id,
            timestamp=response_timestamp,
            response_data=response_data,
            response_time=response_timestamp - challenge.timestamp,
        )

        self.responses[challenge_id] = response
        del self.pending_challenges[challenge_id]

        return response

    def verify_response(self, challenge_id: str) -> Dict[str, Any]:
        """
        Verify a challenge response.

        Args:
            challenge_id: ID of the challenge

        Returns:
            Verification results
        """
        if challenge_id not in self.responses:
            return {"verified": False, "error": "No response found"}

        response = self.responses[challenge_id]

        # Check response time
        if response.response_time > self.max_response_time:
            return {
                "verified": False,
                "error": "Response time exceeded",
                "response_time": response.response_time,
                "max_allowed": self.max_response_time,
            }

        # Additional verification would depend on challenge type
        # For now, just check that response was timely
        return {"verified": True, "response_time": response.response_time}

    def _generate_challenge_id(self, trace_id: str, timestamp: float) -> str:
        """Generate a unique challenge ID"""
        data = f"{trace_id}_{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def _generate_nonce(self, challenge_id: str, timestamp: float) -> int:
        """Generate a deterministic nonce for the challenge"""
        data = f"{challenge_id}_{timestamp}".encode()
        hash_bytes = hashlib.sha256(data).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")


class MemoryChallenge:
    """
    Memory spot-check challenge protocol.
    """

    def __init__(
        self,
        memory_limit: int = 1024 * 1024 * 1024,  # 1GB
        latency_bound: float = 0.1,
    ):  # 100ms
        """
        Initialize memory challenge protocol.

        Args:
            memory_limit: Maximum free memory allowed (bytes)
            latency_bound: Maximum response latency (seconds)
        """
        self.memory_limit = memory_limit
        self.latency_bound = latency_bound

    def generate_challenge(self, device_id: str, timestamp: float) -> Dict[str, Any]:
        """
        Generate a memory challenge for a device.

        Args:
            device_id: Device to challenge
            timestamp: Current timestamp

        Returns:
            Challenge parameters
        """
        # Generate seed for LSH projection
        seed_str = f"{device_id}_{timestamp}"
        seed_hash = hashlib.sha256(seed_str.encode()).digest()
        seed = int.from_bytes(seed_hash[:4], byteorder="big")

        return {
            "device_id": device_id,
            "challenge_type": "memory_lsh",
            "seed": seed,
            "projection_dims": 16,
            "deadline": timestamp + self.latency_bound,
        }

    def verify_response(
        self, challenge: Dict[str, Any], response: Dict[str, Any]
    ) -> bool:
        """
        Verify a memory challenge response.

        Args:
            challenge: Challenge parameters
            response: Response data including LSH projection

        Returns:
            True if response is valid
        """
        # Check response time
        if "timestamp" in response:
            response_time = response["timestamp"] - challenge.get("timestamp", 0)
            if response_time > self.latency_bound:
                return False

        # Check projection dimensions
        if "projection" in response:
            projection = response["projection"]
            expected_dims = challenge.get("projection_dims", 16)
            if len(projection) != expected_dims:
                return False

        # Additional checks would verify the projection is correct
        # for the claimed memory state
        return True
