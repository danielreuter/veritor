"""The hash chain that orders the protocol's phases.

Each phase digest covers the previous one and the message just received, so a
commitment domain or challenge bound to a phase digest is bound to the entire
prefix of the interaction.
"""

from __future__ import annotations

from .messages import (
    BoundaryMessage,
    Header,
    InteriorMessage,
    ReplayChallenge,
    SampleChallenge,
    raw_digest,
)


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
