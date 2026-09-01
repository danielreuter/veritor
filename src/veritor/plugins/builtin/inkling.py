"""Aggregate capacity-profile plug-in for Inkling."""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_cut_analysis.models.inkling import (
    INKLING,
    InklingConfig,
    build_inkling_capacity_profile,
)

from ..api import (
    AggregateBoundArtifact,
    ArchitectureId,
    GreedyTextExecutionShape,
)
from ._capacity_profile import build_aggregate_artifact

PLUGIN_ID = "veritor.plugins.builtin.inkling"
PLUGIN_VERSION = "1"
INKLING_ARCHITECTURE_ID = ArchitectureId.INKLING
INKLING_NUMERICAL_PROFILE_ID = "inkling-bf16-reference"


@dataclass(frozen=True, slots=True)
class InklingCompileRequest:
    execution_shape: GreedyTextExecutionShape = field(
        default_factory=GreedyTextExecutionShape
    )
    config: InklingConfig = INKLING
    numerical_profile_id: str = INKLING_NUMERICAL_PROFILE_ID
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.INKLING,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.execution_shape, GreedyTextExecutionShape):
            raise TypeError("execution_shape must be GreedyTextExecutionShape")
        if not isinstance(self.config, InklingConfig):
            raise TypeError("config must be InklingConfig")
        if type(self.numerical_profile_id) is not str:
            raise TypeError("numerical_profile_id must be a string")
        if not self.numerical_profile_id.strip():
            raise ValueError("numerical_profile_id must be a nonempty string")

    @property
    def shape(self) -> GreedyTextExecutionShape:
        return self.execution_shape


def compile_inkling(
    request: InklingCompileRequest | None = None,
) -> AggregateBoundArtifact:
    selected = InklingCompileRequest() if request is None else request
    if not isinstance(selected, InklingCompileRequest):
        raise TypeError("Inkling requires InklingCompileRequest")
    shape = selected.execution_shape
    profile = build_inkling_capacity_profile(
        shape.prompt_tokens,
        shape.generated_tokens,
        config=selected.config,
        numerical_profile_id=selected.numerical_profile_id,
    )
    return build_aggregate_artifact(
        architecture_id=ArchitectureId.INKLING,
        plugin_id=PLUGIN_ID,
        plugin_version=PLUGIN_VERSION,
        execution_shape=shape,
        configuration=selected.config,
        numerical_profile_id=selected.numerical_profile_id,
        profile=profile,
        source="circuit_cut_analysis.models.inkling",
    )


compile_inkling_bound_model = compile_inkling


@dataclass(frozen=True, slots=True)
class InklingPlugin:
    architecture_id: ArchitectureId = field(
        init=False,
        default=ArchitectureId.INKLING,
    )
    plugin_id: str = field(init=False, default=PLUGIN_ID)
    plugin_version: str = field(init=False, default=PLUGIN_VERSION)

    def default_request(self) -> InklingCompileRequest:
        return InklingCompileRequest()

    def compile(self, request: object | None = None) -> AggregateBoundArtifact:
        if request is not None and not isinstance(request, InklingCompileRequest):
            raise TypeError("Inkling requires InklingCompileRequest")
        return compile_inkling(request)


INKLING_PLUGIN = InklingPlugin()
