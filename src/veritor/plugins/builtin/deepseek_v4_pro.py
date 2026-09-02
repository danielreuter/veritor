"""DeepSeek-V4-Pro: model configuration awaiting a description constructor."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.deepseek_v4_pro import (
    DEEPSEEK_V4_PRO,
    DeepSeekV4ProConfig,
)
from veritor.core import validate_digest

from ..api import ArchitectureId, ConfiguredPlugin, GreedyTextExecutionShape

PLUGIN_ID = "veritor.plugins.builtin.deepseek-v4-pro"
PLUGIN_VERSION = "1"
DEEPSEEK_V4_PRO_ARCHITECTURE_ID = ArchitectureId.DEEPSEEK_V4_PRO
DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID = "deepseek-v4-pro-bundled-reference"


@dataclass(frozen=True, slots=True)
class DeepSeekV4ProCompileRequest:
    """``trace_digest`` names the routing trace a data-dependent MoE run follows."""

    execution_shape: GreedyTextExecutionShape = field(default_factory=GreedyTextExecutionShape)
    config: DeepSeekV4ProConfig = DEEPSEEK_V4_PRO
    numerical_profile_id: str = DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID
    trace_digest: str | None = None
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.DEEPSEEK_V4_PRO)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, DeepSeekV4ProConfig):
            raise TypeError("config must be DeepSeekV4ProConfig")
        if type(self.numerical_profile_id) is not str or not self.numerical_profile_id.strip():
            raise ValueError("numerical_profile_id must be a nonempty string")
        if self.trace_digest is not None:
            object.__setattr__(
                self, "trace_digest", validate_digest(self.trace_digest, "trace_digest")
            )


DEEPSEEK_V4_PRO_PLUGIN = ConfiguredPlugin(
    ArchitectureId.DEEPSEEK_V4_PRO, PLUGIN_ID, PLUGIN_VERSION, DeepSeekV4ProCompileRequest
)
