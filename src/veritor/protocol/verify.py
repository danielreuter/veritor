"""Pure verification of a recorded transcript.

The verifier replays its own side of the interaction from the transcript: it
derives every challenge from its expectation and the messages so far, and
rejects if the recorded challenges differ.  No state persists between calls.
"""

from __future__ import annotations

from veritor.core import CompiledArtifact, ResourceLimit, VerificationLimits

from .messages import Reject, VerificationCode, VerificationReport
from .session import Expectation, VerifierSession
from .wire import MalformedTranscript, NoncanonicalTranscript, decode_transcript


def verify_transcript(
    data: bytes,
    expectation: Expectation,
    artifact: CompiledArtifact,
    limits: VerificationLimits | None = None,
) -> VerificationReport:
    checked = VerificationLimits() if limits is None else limits
    try:
        transcript = decode_transcript(data, checked)
    except NoncanonicalTranscript as error:
        return VerificationReport(VerificationCode.NONCANONICAL_TRANSCRIPT, str(error))
    except MalformedTranscript as error:
        return VerificationReport(VerificationCode.MALFORMED_TRANSCRIPT, str(error))
    except ResourceLimit as error:
        return VerificationReport(VerificationCode.RESOURCE_LIMIT, str(error))

    session = VerifierSession(expectation, artifact, limits=checked)
    if transcript.header != session.header:
        return VerificationReport(
            VerificationCode.EXPECTATION_MISMATCH,
            "transcript header differs from the verifier's expectation",
        )
    try:
        replay_challenge = session.receive_boundary(transcript.boundary)
        if replay_challenge.seed != transcript.replay_challenge.seed:
            raise Reject(VerificationCode.EXPECTATION_MISMATCH, "q seed differs")
        if replay_challenge != transcript.replay_challenge:
            raise Reject(VerificationCode.CHALLENGE_MISMATCH, "replay selection differs")
        sample_challenge = session.receive_interiors(transcript.interiors)
        if sample_challenge.seed != transcript.sample_challenge.seed:
            raise Reject(VerificationCode.EXPECTATION_MISMATCH, "s seed differs")
        if sample_challenge != transcript.sample_challenge:
            raise Reject(VerificationCode.CHALLENGE_MISMATCH, "sample selection differs")
        return session.receive_evidence(transcript.evidence)
    except Reject as rejection:
        return VerificationReport(
            rejection.code,
            rejection.detail,
            session.selected_replay_units,
            session.selected_verification_units,
        )
    except ResourceLimit as error:
        return VerificationReport(VerificationCode.RESOURCE_LIMIT, str(error))
