"""Trace-conditional aggregate plug-in for DeepSeek-V4-Pro."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.deepseek_v4_pro import (
    DEEPSEEK_V4_PRO,
    DeepSeekV4ProConfig,
    build_deepseek_v4_pro_capacity_profile,
)
from veritor.core import Digest, validate_digest

from ..api import (
    AggregateBoundArtifact,
    ArchitectureId,
    GreedyTextExecutionShape,
)
from ._capacity_profile import build_aggregate_artifact

PLUGIN_ID = "veritor.plugins.builtin.deepseek-v4-pro"
PLUGIN_VERSION = "1"
DEEPSEEK_V4_PRO_ARCHITECTURE_ID = ArchitectureId.DEEPSEEK_V4_PRO
DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID = "deepseek-v4-pro-bundled-reference"


@dataclass(frozen=True, slots=True)
class DeepSeekV4ProCompileRequest:
    execution_shape: GreedyTextExecutionShape = field(
        default_factory=GreedyTextExecutionShape
    )
    config: DeepSeekV4ProConfig = DEEPSEEK_V4_PRO
    numerical_profile_id: str = DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID
    trace_digest: str | None = None
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.DEEPSEEK_V4_PRO,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, DeepSeekV4ProConfig):
            raise TypeError("config must be DeepSeekV4ProConfig")
        if type(self.numerical_profile_id) is not str:
            raise TypeError("numerical_profile_id must be a string")
        if not self.numerical_profile_id.strip():
            raise ValueError("numerical_profile_id must be a nonempty string")
        if self.trace_digest is not None:
            object.__setattr__(
                self,
                "trace_digest",
                validate_digest(self.trace_digest, "trace_digest"),
            )

    @property
    def shape(self) -> GreedyTextExecutionShape:
        return self.execution_shape


def compile_deepseek_v4_pro(
    request: DeepSeekV4ProCompileRequest | None = None,
) -> AggregateBoundArtifact:
    selected = DeepSeekV4ProCompileRequest() if request is None else request
    if not isinstance(selected, DeepSeekV4ProCompileRequest):
        raise TypeError("DeepSeek-V4-Pro requires DeepSeekV4ProCompileRequest")
    shape = selected.execution_shape
    profile = build_deepseek_v4_pro_capacity_profile(
        shape.prompt_tokens,
        shape.generated_tokens,
        config=selected.config,
        numerical_profile_id=selected.numerical_profile_id,
    )
    checked_trace: Digest | None = (
        None
        if selected.trace_digest is None
        else validate_digest(selected.trace_digest, "trace_digest")
    )
    return build_aggregate_artifact(
        architecture_id=ArchitectureId.DEEPSEEK_V4_PRO,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        execution_shape=shape,
        configuration=selected.config,
        numerical_profile_id=selected.numerical_profile_id,
        profile=profile,
        source="circuit_cut_analysis.models.deepseek_v4_pro",
        trace_conditional=True,
        trace_digest=checked_trace,
    )


compile_deepseek_v4_pro_bound_model = compile_deepseek_v4_pro


@dataclass(frozen=True, slots=True)
class DeepSeekV4ProPlugin:
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.DEEPSEEK_V4_PRO,
    )
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> DeepSeekV4ProCompileRequest:
        return DeepSeekV4ProCompileRequest()

    def compile(self, request: object | None = None) -> AggregateBoundArtifact:
        if request is not None and not isinstance(
            request,
            DeepSeekV4ProCompileRequest,
        ):
            raise TypeError("DeepSeek-V4-Pro requires DeepSeekV4ProCompileRequest")
        return compile_deepseek_v4_pro(request)


DEEPSEEK_V4_PRO_PLUGIN = DeepSeekV4ProPlugin()
