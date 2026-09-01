"""
Infolock: Hardware-enforced confidentiality for verification protocols.

This module defines the abstract interfaces for running interactive protocols
where the verifier can see prover secrets but cannot exfiltrate them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


# =============================================================================
# TYPE VARIABLES
# =============================================================================

S = TypeVar("S")  # Statement payload type
A = TypeVar("A")  # Attestation payload type
X = TypeVar("X")  # Prover secret type


# =============================================================================
# PROTOCOL PRIMITIVES
# =============================================================================


@dataclass
class Statement(Generic[S]):
    """
    A statement issued by the prover.

    The statement contains some payload that the prover wants attested.
    The verifier will see this in plaintext inside the infolock.
    """
    payload: S


@dataclass
class Attestation(Generic[A]):
    """
    An attestation produced by the verifier.

    The attestation is bound to a particular statement and contains
    the verifier's output (e.g., pass/fail, or more complex results).
    """
    payload: A


@dataclass
class Round(Generic[S, A]):
    """A single round of the protocol: statement in, attestation out."""
    statement: Statement[S]
    attestation: Attestation[A]


# =============================================================================
# PARTIES: ALGORITHMS AND ENTITIES
# =============================================================================


class ProverAlgorithm(ABC, Generic[X, S]):
    """
    The algorithm that runs on the prover's device.

    The prover algorithm takes secrets and produces statements.
    It also implements the abort rule to detect verifier misbehavior.
    """

    @abstractmethod
    def generate_statement(self, secret: X) -> Statement[S]:
        """Generate a statement from prover secrets."""
        pass

    @abstractmethod
    def abort_rule(self, secret: X, statement: Statement[S], attestation: Attestation) -> bool:
        """
        Evaluate whether to abort based on the attestation.

        Returns True if the protocol should continue, False to abort.
        """
        pass


class VerifierAlgorithm(ABC, Generic[S, A]):
    """
    The algorithm that runs inside the infolock.

    The verifier algorithm sees the statement in plaintext and
    produces an attestation. This is the V(x) -> y computation.
    """

    @abstractmethod
    def attest(self, statement: Statement[S]) -> Attestation[A]:
        """Compute an attestation for a statement."""
        pass

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """
        Whether this verifier produces deterministic outputs.

        If True, the prover can independently compute V(x) and
        verify that the attestation matches exactly (n = 0).
        """
        pass


class RelyingParty(ABC, Generic[A]):
    """
    The entity that receives and acts on attestations.

    The relying party trusts the infolock to run the verifier correctly,
    and uses the attestations to make decisions.
    """

    @abstractmethod
    def receive_attestation(self, attestation: Attestation[A]) -> None:
        """Receive an attestation from the infolock."""
        pass

    @abstractmethod
    def verify_attestation(self, attestation: Attestation[A]) -> bool:
        """Verify that an attestation is valid (signatures, etc.)."""
        pass


# =============================================================================
# DEVICE: THE INFOLOCK
# =============================================================================


@dataclass
class ChannelConfig:
    """Configuration for the output channel."""
    max_bits_per_round: int      # m in the security model
    max_rounds_per_period: int
    period_seconds: float


class OutputChannel(ABC, Generic[A]):
    """
    The channel through which attestations leave the infolock.

    Observable by both prover and relying party.
    Throttled to bound information leakage.
    """

    @abstractmethod
    def emit(self, attestation: Attestation[A]) -> None:
        """Emit an attestation to observers."""
        pass

    @abstractmethod
    def get_config(self) -> ChannelConfig:
        """Get the channel's capacity constraints."""
        pass


class InputChannel(ABC, Generic[S]):
    """The channel through which statements enter the infolock."""

    @abstractmethod
    def receive(self) -> Statement[S]:
        """Receive the next statement from the prover."""
        pass


class Infolock(ABC, Generic[S, A]):
    """
    The infolock device.

    This is the physical box that:
    - Receives statements from the prover
    - Runs the verifier algorithm
    - Emits attestations through the output channel

    Trust assumptions:
    1. Process integrity: V runs correctly inside
    2. Output integrity: Relying party receives all outputs
    3. No side channels: Only the output channel emits information
    4. Output transparency: Prover can observe all outputs
    5. Known capacity: Output channel has bounded capacity
    """

    @abstractmethod
    def load_verifier(self, verifier: VerifierAlgorithm[S, A]) -> None:
        """Load a verifier algorithm into the infolock."""
        pass

    @abstractmethod
    def run_round(self, statement: Statement[S]) -> Attestation[A]:
        """Run one round: receive statement, compute attestation, emit it."""
        pass

    @abstractmethod
    def get_output_channel(self) -> OutputChannel[A]:
        """Get the output channel for observers."""
        pass

    @abstractmethod
    def get_input_channel(self) -> InputChannel[S]:
        """Get the input channel for the prover."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the infolock state."""
        pass

    @abstractmethod
    def attest_integrity(self) -> bytes:
        """Produce a hardware attestation of the infolock's integrity."""
        pass


# =============================================================================
# SESSION: COMPLETE PROTOCOL EXECUTION
# =============================================================================


@dataclass
class SessionConfig:
    """Configuration for a protocol session."""
    max_rounds: int | None = None
    continue_after_abort: bool = False


@dataclass
class SessionResult(Generic[S, A]):
    """The result of a completed session."""
    rounds: list[Round[S, A]] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str | None = None


class Session(Generic[X, S, A]):
    """
    A protocol session between prover and verifier.

    Lifecycle:
    1. Setup: Load algorithms, establish keys
    2. Run: Prover issues statements, verifier attests, prover checks abort
    3. Finalize: Clean up, rotate keys
    """

    def __init__(
        self,
        prover: ProverAlgorithm[X, S],
        verifier: VerifierAlgorithm[S, A],
        infolock: Infolock[S, A],
        relying_party: RelyingParty[A],
        config: SessionConfig | None = None,
    ):
        self.prover = prover
        self.verifier = verifier
        self.infolock = infolock
        self.relying_party = relying_party
        self.config = config or SessionConfig()

        self._rounds: list[Round[S, A]] = []
        self._aborted = False

    def setup(self) -> None:
        """Initialize the session. Loads the verifier into the infolock."""
        self.infolock.load_verifier(self.verifier)

    def submit(self, secret: X) -> Attestation[A] | None:
        """
        Submit a secret and receive an attestation.

        Returns None if the session has been aborted or at limit.
        """
        if self._aborted:
            return None
        if self.config.max_rounds and len(self._rounds) >= self.config.max_rounds:
            return None

        # 1. Prover generates statement from secret
        statement = self.prover.generate_statement(secret)

        # 2. Infolock runs verifier, emits attestation
        attestation = self.infolock.run_round(statement)

        # 3. Relying party receives attestation
        self.relying_party.receive_attestation(attestation)

        # 4. Record the round
        self._rounds.append(Round(statement, attestation))

        # 5. Prover checks abort rule
        should_continue = self.prover.abort_rule(secret, statement, attestation)
        if not should_continue:
            self._aborted = True

        return attestation

    def is_active(self) -> bool:
        """Whether the session is still active."""
        if self._aborted:
            return False
        if self.config.max_rounds and len(self._rounds) >= self.config.max_rounds:
            return False
        return True

    def finalize(self) -> SessionResult[S, A]:
        """Finalize the session and return results."""
        self.infolock.reset()
        return SessionResult(
            rounds=self._rounds,
            aborted=self._aborted,
            abort_reason="Abort rule triggered" if self._aborted else None,
        )
