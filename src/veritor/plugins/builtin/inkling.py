"""Inkling: model configuration awaiting a description constructor."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.inkling import INKLING, InklingConfig

from ..api import ArchitectureId, ConfiguredPlugin, GreedyTextExecutionShape

PLUGIN_ID = "veritor.plugins.builtin.inkling"
PLUGIN_VERSION = "1"
INKLING_ARCHITECTURE_ID = ArchitectureId.INKLING
INKLING_NUMERICAL_PROFILE_ID = "inkling-bf16-reference"


@dataclass(frozen=True, slots=True)
class InklingCompileRequest:
    execution_shape: GreedyTextExecutionShape = field(default_factory=GreedyTextExecutionShape)
    config: InklingConfig = INKLING
    numerical_profile_id: str = INKLING_NUMERICAL_PROFILE_ID
    architecture_id: ArchitectureId = field(init=False, default=ArchitectureId.INKLING)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, InklingConfig):
            raise TypeError("config must be InklingConfig")
        if type(self.numerical_profile_id) is not str or not self.numerical_profile_id.strip():
            raise ValueError("numerical_profile_id must be a nonempty string")


INKLING_PLUGIN = ConfiguredPlugin(
    ArchitectureId.INKLING, PLUGIN_ID, PLUGIN_VERSION, InklingCompileRequest
)
