"""
Interactive verification with simple random challenge sampling.

No complex schedule generation - just probabilistic decisions at hook points.
"""

from veritor.interactive.challenger import (
    Challenger,
    ChallengeContext,
    create_challenge_hook,
    compute_lsh_projection,
    set_global_challenger,
)

__all__ = [
    'Challenger',
    'ChallengeContext',
    'create_challenge_hook',
    'compute_lsh_projection',
    'set_global_challenger',
]