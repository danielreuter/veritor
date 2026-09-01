"""Aggregate capacity-profile plug-in for Kimi-K3."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.kimi_k3 import (
    KIMI_K3,
    KimiK3Config,
    build_kimi_k3_capacity_profile,
)

from ..api import (
    AggregateBoundArtifact,
    ArchitectureId,
    GreedyTextExecutionShape,
)
from ._capacity_profile import build_aggregate_artifact

PLUGIN_ID = "veritor.plugins.builtin.kimi-k3"
PLUGIN_VERSION = "1"
KIMI_K3_ARCHITECTURE_ID = ArchitectureId.KIMI_K3
KIMI_K3_NUMERICAL_PROFILE_ID = "kimi-k3-semantic-mixed"


@dataclass(frozen=True, slots=True)
class KimiK3CompileRequest:
    execution_shape: GreedyTextExecutionShape = field(
        default_factory=GreedyTextExecutionShape
    )
    config: KimiK3Config = KIMI_K3
    numerical_profile_id: str = KIMI_K3_NUMERICAL_PROFILE_ID
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.KIMI_K3,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, KimiK3Config):
            raise TypeError("config must be KimiK3Config")
        if type(self.numerical_profile_id) is not str:
            raise TypeError("numerical_profile_id must be a string")
        if not self.numerical_profile_id.strip():
            raise ValueError("numerical_profile_id must be a nonempty string")

    @property
    def shape(self) -> GreedyTextExecutionShape:
        return self.execution_shape


def compile_kimi_k3(
    request: KimiK3CompileRequest | None = None,
) -> AggregateBoundArtifact:
    selected = KimiK3CompileRequest() if request is None else request
    if not isinstance(selected, KimiK3CompileRequest):
        raise TypeError("Kimi-K3 requires KimiK3CompileRequest")
    shape = selected.execution_shape
    profile = build_kimi_k3_capacity_profile(
        shape.prompt_tokens,
        shape.generated_tokens,
        config=selected.config,
        numerical_profile_id=selected.numerical_profile_id,
    )
    return build_aggregate_artifact(
        architecture_id=ArchitectureId.KIMI_K3,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        execution_shape=shape,
        configuration=selected.config,
        numerical_profile_id=selected.numerical_profile_id,
        profile=profile,
        source="circuit_cut_analysis.models.kimi_k3",
    )


compile_kimi_k3_bound_model = compile_kimi_k3


@dataclass(frozen=True, slots=True)
class KimiK3Plugin:
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.KIMI_K3,
    )
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> KimiK3CompileRequest:
        return KimiK3CompileRequest()

    def compile(self, request: object | None = None) -> AggregateBoundArtifact:
        if request is not None and not isinstance(request, KimiK3CompileRequest):
            raise TypeError("Kimi-K3 requires KimiK3CompileRequest")
        return compile_kimi_k3(request)


KIMI_K3_PLUGIN = KimiK3Plugin()
