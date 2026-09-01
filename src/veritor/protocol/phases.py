"""The hash chain that orders the protocol's phases.

Each phase digest covers the previous one and the message just received, so a
commitment domain or challenge bound to a phase digest is bound to the entire
prefix of the interaction.
"""

from __future__ import annotations

from veritor.core import CompiledArtifact

from .merkle import CommitmentDomain
from .messages import (
    BoundaryMessage,
    Header,
    InteriorMessage,
    ReplayChallenge,
    SampleChallenge,
    raw_digest,
)

BOUNDARY_OWNER = -1


def boundary_phase(header: Header, boundary: BoundaryMessage) -> bytes:
    return raw_digest(
        "veritor/protocol/phase/boundary/v2",
        {"boundary": boundary.manifest, "header": header.digest.hex()},
    )


def replay_phase(boundary_phase_digest: bytes, challenge: ReplayChallenge) -> bytes:
    return raw_digest(
        "veritor/protocol/phase/replay/v2",
        {"challenge": challenge.manifest, "previous": boundary_phase_digest.hex()},
    )


def interior_phase(replay_phase_digest: bytes, interiors: InteriorMessage) -> bytes:
    return raw_digest(
        "veritor/protocol/phase/interior/v2",
        {"interiors": interiors.manifest, "previous": replay_phase_digest.hex()},
    )


def sample_phase(interior_phase_digest: bytes, challenge: SampleChallenge) -> bytes:
    return raw_digest(
        "veritor/protocol/phase/sample/v2",
        {"challenge": challenge.manifest, "previous": interior_phase_digest.hex()},
    )


def boundary_domain(header: Header, artifact: CompiledArtifact) -> CommitmentDomain:
    """The boundary commitment covers ``∂`` and is bound to the header alone."""

    return CommitmentDomain(header.digest, header.digest, BOUNDARY_OWNER, artifact.boundary)


def interior_domain(
    header: Header,
    replay_phase_digest: bytes,
    artifact: CompiledArtifact,
    replay_unit: int,
) -> CommitmentDomain:
    """The interior commitment of ``r`` covers ``Int(r)`` and is bound to ``J``."""

    return CommitmentDomain(
        header.digest, replay_phase_digest, replay_unit, artifact.interior(replay_unit)
    )
