"""Kimi-K3: model configuration awaiting a description constructor."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.kimi_k3 import KIMI_K3, KimiK3Config

from ..api import ArchitectureId, ConfiguredPlugin, GreedyTextExecutionShape

PLUGIN_ID = "veritor.plugins.builtin.kimi-k3"
PLUGIN_VERSION = "1"
KIMI_K3_ARCHITECTURE_ID = ArchitectureId.KIMI_K3
KIMI_K3_NUMERICAL_PROFILE_ID = "kimi-k3-semantic-mixed"


@dataclass(frozen=True, slots=True)
class KimiK3CompileRequest:
    execution_shape: GreedyTextExecutionShape = field(default_factory=GreedyTextExecutionShape)
    config: KimiK3Config = KIMI_K3
    numerical_profile_id: str = KIMI_K3_NUMERICAL_PROFILE_ID
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.KIMI_K3)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, KimiK3Config):
            raise TypeError("config must be KimiK3Config")
        if type(self.numerical_profile_id) is not str or not self.numerical_profile_id.strip():
            raise ValueError("numerical_profile_id must be a nonempty string")


KIMI_K3_PLUGIN = ConfiguredPlugin(
    ArchitectureId.KIMI_K3, PLUGIN_ID, PLUGIN_VERSION, KimiK3CompileRequest
)
