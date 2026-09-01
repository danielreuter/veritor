"""The two-stage sampled verification protocol.

``Verify`` in the paper is :func:`run_protocol` (interactive) or
:func:`verify_transcript` (pure, over a recorded transcript).  Both consume a
trusted :class:`~veritor.core.CompiledArtifact` and a verifier-owned
:class:`Expectation`; everything else arrives from the prover as messages.
"""

from .challenge import derive_replay_selection, derive_sample_selection, uniform_below
from .merkle import CommitmentDomain, MerkleTree, validate_commitment, verify_opening
from .messages import (
    PROTOCOL_VERSION,
    BoundaryMessage,
    Commitment,
    EvidenceMessage,
    Header,
    InteriorMessage,
    Opening,
    ProtocolError,
    Reject,
    ReplayChallenge,
    SampleChallenge,
    Transcript,
    VerificationCode,
    VerificationReport,
)
from .phases import boundary_domain, interior_domain
from .session import (
    Expectation,
    ProtocolRun,
    ProverSession,
    Replay,
    Values,
    VerifierSession,
    assignment_replay,
    make_expectation,
    replay_unit,
    run_protocol,
)
from .verify import verify_transcript
from .wire import (
    MalformedTranscript,
    NoncanonicalTranscript,
    decode_transcript,
    encode_transcript,
)

__all__ = [
    "PROTOCOL_VERSION",
    "BoundaryMessage",
    "Commitment",
    "CommitmentDomain",
    "EvidenceMessage",
    "Expectation",
    "Header",
    "InteriorMessage",
    "MalformedTranscript",
    "MerkleTree",
    "NoncanonicalTranscript",
    "Opening",
    "ProtocolError",
    "ProtocolRun",
    "ProverSession",
    "Reject",
    "Replay",
    "ReplayChallenge",
    "SampleChallenge",
    "Transcript",
    "Values",
    "VerificationCode",
    "VerificationReport",
    "VerifierSession",
    "assignment_replay",
    "boundary_domain",
    "decode_transcript",
    "derive_replay_selection",
    "derive_sample_selection",
    "encode_transcript",
    "interior_domain",
    "make_expectation",
    "replay_unit",
    "run_protocol",
    "uniform_below",
    "validate_commitment",
    "verify_opening",
    "verify_transcript",
]
