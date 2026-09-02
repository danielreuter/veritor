"""GPT-2: model configuration awaiting a description constructor."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.gpt2 import GPT2_SMALL, GPT2Config
from circuit_cut_analysis.profiles import VLLM_FP16_REFERENCE, ServingProfile

from ..api import ArchitectureId, ConfiguredPlugin, GreedyTextExecutionShape

PLUGIN_ID = "veritor.plugins.builtin.gpt2"
PLUGIN_VERSION = "1"
GPT2_ARCHITECTURE_ID = ArchitectureId.GPT2


@dataclass(frozen=True, slots=True)
class GPT2CompileRequest:
    """Shape, architecture dimensions, and numerical boundary profile."""

    execution_shape: GreedyTextExecutionShape = field(default_factory=GreedyTextExecutionShape)
    config: GPT2Config = GPT2_SMALL
    profile: ServingProfile = VLLM_FP16_REFERENCE
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.GPT2)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        if not isinstance(self.profile, ServingProfile):
            raise TypeError("profile must be ServingProfile")


GPT2_PLUGIN = ConfiguredPlugin(ArchitectureId.GPT2, PLUGIN_ID, PLUGIN_VERSION, GPT2CompileRequest)
