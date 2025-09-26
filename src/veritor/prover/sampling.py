"""
Prover-specific sampling helpers.

This module provides prover-specific sampling utilities that wrap
the common sampler while adding prover-specific functionality.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from veritor.common.sampler import (
    sample_bernoulli_logit,
    sample_categorical_logit,
    sample_gaussian_logit,
)
from veritor.prover.hooks import HookSystem


def prover_sample_with_hook(
    logits: jax.Array,
    key: jax.random.PRNGKey,
    hook_system: Optional[HookSystem] = None,
    sample_type: str = "categorical",
    **kwargs,
) -> Tuple[jax.Array, Optional[dict]]:
    """
    Wrapper for sampling that integrates with the hook system.

    Args:
        logits: Input logits
        key: PRNG key
        hook_system: Optional hook system for challenge integration
        sample_type: Type of sampling ("categorical", "bernoulli", "gaussian")
        **kwargs: Additional arguments for specific samplers

    Returns:
        Tuple of (samples, challenge_data if hook triggered else None)
    """
    challenge_data = None

    if hook_system and hook_system.should_trigger_sampling_challenge():
        challenge_data = hook_system.capture_sampling_state(
            logits=logits, key=key, sample_type=sample_type
        )

    if sample_type == "categorical":
        samples = sample_categorical_logit(logits, key, **kwargs)
    elif sample_type == "bernoulli":
        samples = sample_bernoulli_logit(logits, key, **kwargs)
    elif sample_type == "gaussian":
        samples = sample_gaussian_logit(logits, key, **kwargs)
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    return samples, challenge_data


def batch_sample_with_hooks(
    logits_batch: jax.Array,
    key: jax.random.PRNGKey,
    hook_system: Optional[HookSystem] = None,
    sample_type: str = "categorical",
    **kwargs,
) -> Tuple[jax.Array, list]:
    """
    Batch sampling with hook system integration.

    Args:
        logits_batch: Batch of logits [batch_size, ...]
        key: PRNG key
        hook_system: Optional hook system
        sample_type: Type of sampling
        **kwargs: Additional sampler arguments

    Returns:
        Tuple of (samples, list of challenge data per batch element)
    """
    batch_size = logits_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    samples = []
    challenges = []

    for i in range(batch_size):
        sample, challenge = prover_sample_with_hook(
            logits_batch[i], keys[i], hook_system, sample_type, **kwargs
        )
        samples.append(sample)
        challenges.append(challenge)

    return jnp.stack(samples), challenges
