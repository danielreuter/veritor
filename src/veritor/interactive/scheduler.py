"""
Challenge schedulers and nonce generation for interactive verification.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ChallengeSchedule:
    """Schedule of when to issue challenges"""

    steps: List[int]
    layers: List[int]
    batch_indices: List[int]
    challenge_map: Dict[Tuple[int, int, int], bool]


class ChallengeScheduler:
    """
    Generates challenge schedules for verification protocols.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize scheduler.

        Args:
            seed: Random seed for reproducible schedules
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_schedule(
        self,
        n_steps: int,
        n_layers: int,
        batch_size: int,
        challenge_probability: float = 0.1,
    ) -> ChallengeSchedule:
        """
        Generate a random challenge schedule.

        Args:
            n_steps: Number of execution steps
            n_layers: Number of model layers
            batch_size: Batch size
            challenge_probability: Probability of challenging at each point

        Returns:
            Challenge schedule
        """
        steps = []
        layers = []
        batch_indices = []
        challenge_map = {}

        for step in range(n_steps):
            for layer in range(n_layers):
                for batch_idx in range(batch_size):
                    if self.rng.random() < challenge_probability:
                        steps.append(step)
                        layers.append(layer)
                        batch_indices.append(batch_idx)
                        challenge_map[(step, layer, batch_idx)] = True
                    else:
                        challenge_map[(step, layer, batch_idx)] = False

        return ChallengeSchedule(
            steps=steps,
            layers=layers,
            batch_indices=batch_indices,
            challenge_map=challenge_map,
        )

    def generate_adaptive_schedule(
        self,
        n_steps: int,
        n_layers: int,
        batch_size: int,
        focus_layers: List[int] = None,
        focus_probability: float = 0.3,
        baseline_probability: float = 0.05,
    ) -> ChallengeSchedule:
        """
        Generate an adaptive challenge schedule that focuses on specific layers.

        Args:
            n_steps: Number of execution steps
            n_layers: Number of model layers
            batch_size: Batch size
            focus_layers: Layers to challenge more frequently
            focus_probability: Challenge probability for focus layers
            baseline_probability: Challenge probability for other layers

        Returns:
            Challenge schedule
        """
        if focus_layers is None:
            focus_layers = []

        steps = []
        layers = []
        batch_indices = []
        challenge_map = {}

        for step in range(n_steps):
            for layer in range(n_layers):
                prob = (
                    focus_probability if layer in focus_layers else baseline_probability
                )

                for batch_idx in range(batch_size):
                    if self.rng.random() < prob:
                        steps.append(step)
                        layers.append(layer)
                        batch_indices.append(batch_idx)
                        challenge_map[(step, layer, batch_idx)] = True
                    else:
                        challenge_map[(step, layer, batch_idx)] = False

        return ChallengeSchedule(
            steps=steps,
            layers=layers,
            batch_indices=batch_indices,
            challenge_map=challenge_map,
        )


class NonceGenerator:
    """
    Generates cryptographically secure nonces for challenges.
    """

    def __init__(self, master_seed: bytes = None):
        """
        Initialize nonce generator.

        Args:
            master_seed: Master seed for nonce derivation
        """
        if master_seed is None:
            master_seed = random.randbytes(32)
        self.master_seed = master_seed

    def generate_nonce(
        self, trace_id: str, step: int, layer: int, batch_idx: int
    ) -> int:
        """
        Generate a deterministic nonce for a specific challenge point.

        Args:
            trace_id: Trace identifier
            step: Execution step
            layer: Layer index
            batch_idx: Batch index

        Returns:
            32-bit nonce
        """
        # Create unique identifier for this challenge point
        data = f"{trace_id}_{step}_{layer}_{batch_idx}".encode()

        # Derive nonce using HMAC
        import hmac

        h = hmac.new(self.master_seed, data, hashlib.sha256)
        nonce_bytes = h.digest()[:4]
        nonce = int.from_bytes(nonce_bytes, byteorder="big")

        return nonce

    def generate_session_nonce(self, session_id: str, counter: int) -> int:
        """
        Generate a nonce for a verification session.

        Args:
            session_id: Session identifier
            counter: Challenge counter within session

        Returns:
            32-bit nonce
        """
        data = f"{session_id}_{counter}".encode()
        h = hashlib.sha256(self.master_seed + data)
        nonce_bytes = h.digest()[:4]
        nonce = int.from_bytes(nonce_bytes, byteorder="big")

        return nonce


class CheckpointScheduler:
    """
    Schedules checkpoints for training verification.
    """

    def __init__(self, buffer_size: int = 10, seed: int = 42):
        """
        Initialize checkpoint scheduler.

        Args:
            buffer_size: Size of checkpoint buffer
            seed: Random seed
        """
        self.buffer_size = buffer_size
        self.rng = random.Random(seed)

    def should_checkpoint(self, step: int, checkpoint_probability: float = 0.1) -> bool:
        """
        Determine if a checkpoint should be saved at this step.

        Args:
            step: Current training step
            checkpoint_probability: Probability of checkpointing

        Returns:
            True if should checkpoint
        """
        # Use step as additional seed for determinism
        step_rng = random.Random(self.rng.random() + step)
        return step_rng.random() < checkpoint_probability

    def select_checkpoint_to_evict(self, buffer_occupancy: List[int]) -> int:
        """
        Select which checkpoint to evict from buffer.

        Args:
            buffer_occupancy: List of checkpoint step numbers in buffer

        Returns:
            Index of checkpoint to evict
        """
        if len(buffer_occupancy) < self.buffer_size:
            return -1  # No eviction needed

        # Random eviction for now
        # Could implement more sophisticated strategies (LRU, etc.)
        return self.rng.randint(0, len(buffer_occupancy) - 1)

    def request_checkpoint(self, available_checkpoints: List[int]) -> Optional[int]:
        """
        Request a specific checkpoint from available ones.

        Args:
            available_checkpoints: List of available checkpoint steps

        Returns:
            Step number of requested checkpoint, or None
        """
        if not available_checkpoints:
            return None

        # Randomly select a checkpoint
        return self.rng.choice(available_checkpoints)
