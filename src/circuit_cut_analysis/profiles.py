"""Declared numerical semantics for structural circuit analyses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServingProfile:
    """Logical widths used by the symbolic circuit contract.

    These widths describe analysis boundaries.  They are not an assertion
    about the intermediates materialized by a particular fused kernel.
    """

    id: str
    description: str
    weight_bits: int
    activation_boundary_bits: int
    kv_cache_bits: int
    accumulator_bits: int
    reduction_bits: int
    nonlinear_internal_bits: int
    probability_boundary_bits: int
    residual_bits: int
    logit_bits: int
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.description:
            raise ValueError("profile id and description must be non-empty")
        widths = (
            self.weight_bits,
            self.activation_boundary_bits,
            self.kv_cache_bits,
            self.accumulator_bits,
            self.reduction_bits,
            self.nonlinear_internal_bits,
            self.probability_boundary_bits,
            self.residual_bits,
            self.logit_bits,
        )
        if any(width <= 0 for width in widths):
            raise ValueError("all profile widths must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "widths_bits": {
                "weights": self.weight_bits,
                "activation_boundaries": self.activation_boundary_bits,
                "kv_cache": self.kv_cache_bits,
                "dot_product_accumulators": self.accumulator_bits,
                "reductions": self.reduction_bits,
                "nonlinear_internals": self.nonlinear_internal_bits,
                "probability_boundaries": self.probability_boundary_bits,
                "residual_boundaries": self.residual_bits,
                "logits": self.logit_bits,
            },
            "assumptions": list(self.assumptions),
        }


VLLM_FP16_REFERENCE = ServingProfile(
    id="vllm-fp16-reference",
    description=(
        "Declared logical mixed-precision contract anchored to vLLM's auto "
        "dtype choices for an FP32 GPT-2 checkpoint."
    ),
    weight_bits=16,
    activation_boundary_bits=16,
    kv_cache_bits=16,
    accumulator_bits=32,
    reduction_bits=32,
    nonlinear_internal_bits=32,
    probability_boundary_bits=16,
    residual_bits=16,
    logit_bits=16,
    assumptions=(
        "Weights, ordinary activation boundaries, KV entries, and the complete logical logits boundary are FP16.",
        "vLLM dtype=auto selects FP16 for FP32/FP16 models, and kv-cache-dtype=auto selects the model dtype.",
        "Dot products conceptually accumulate in FP32 and round at their output boundary.",
        "LayerNorm, softmax reductions, exp, reciprocal, rsqrt, and GELU internals are FP32.",
        "Softmax probabilities round to FP16 before the probability-value contraction.",
        "Residual and embedding additions produce FP16 outputs.",
        "The accumulator, reduction, nonlinear, probability, and logit choices are declared analysis semantics, not vLLM guarantees.",
        "This profile is not an exact description of intermediates materialized by fused kernels.",
    ),
)


ALL_FP32_REFERENCE = ServingProfile(
    id="all-fp32-reference",
    description="Sensitivity profile in which every arithmetic value is FP32.",
    weight_bits=32,
    activation_boundary_bits=32,
    kv_cache_bits=32,
    accumulator_bits=32,
    reduction_bits=32,
    nonlinear_internal_bits=32,
    probability_boundary_bits=32,
    residual_bits=32,
    logit_bits=32,
    assumptions=(
        "All weights, activations, reductions, nonlinear intermediates, cache entries, and logits are FP32.",
        "This profile is a width sensitivity check, not the recommended serving configuration.",
    ),
)


PROFILES: dict[str, ServingProfile] = {
    profile.id: profile
    for profile in (
        VLLM_FP16_REFERENCE,
        ALL_FP32_REFERENCE,
    )
}


def get_profile(profile_id: str) -> ServingProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as error:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown profile {profile_id!r}; choose one of: {choices}"
        ) from error
