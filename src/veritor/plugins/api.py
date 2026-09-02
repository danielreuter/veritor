"""Stable contracts shared by architecture plug-ins.

A plug-in compiles to a :class:`~veritor.core.Compiled` -- the executable
``(C, I)`` the protocol verifies and the analysis bounds -- or reports
:class:`~veritor.core.Unsupported`.  The LLM plug-ins currently hold model
configurations only: they keep the dimensions constructors will be written
against and compile to ``Unsupported`` until then.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from veritor.core import Capability, Compiled, JSONValue, Unsupported


class ArchitectureId(StrEnum):
    """Stable identifiers for the built-in architecture families."""

    DEMO_G = "demo-g"
    MATMUL = "matmul"
    GPT2 = "gpt2"
    KIMI_K3 = "kimi-k3"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"
    INKLING = "inkling"


class DecodingMode(StrEnum):
    """The only text-generation topology currently represented."""

    GREEDY_ARGMAX = "greedy-argmax"


@dataclass(frozen=True, slots=True)
class GreedyTextExecutionShape:
    """Fixed batch-one, fixed-horizon greedy text execution shape."""

    prompt_tokens: int = 100
    generated_tokens: int = 100
    batch_size: int = 1
    decoding_mode: DecodingMode = DecodingMode.GREEDY_ARGMAX
    fixed_horizon: bool = True
    final_generated_forward: bool = False
    eos_termination: bool = False
    text_only: bool = True

    def __post_init__(self) -> None:
        if type(self.prompt_tokens) is not int or self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer")
        if type(self.generated_tokens) is not int or self.generated_tokens <= 0:
            raise ValueError("generated_tokens must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("architecture profiles support batch_size=1 only")
        if self.decoding_mode is not DecodingMode.GREEDY_ARGMAX:
            raise ValueError("architecture profiles support greedy argmax only")
        for field_name in (
            "fixed_horizon",
            "final_generated_forward",
            "eos_termination",
            "text_only",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if not self.fixed_horizon:
            raise ValueError("architecture profiles require a fixed output horizon")
        if self.final_generated_forward:
            raise ValueError("the final generated token is not forwarded")
        if self.eos_termination:
            raise ValueError("EOS termination must be disabled in the fixed topology")
        if not self.text_only:
            raise ValueError("architecture profiles currently cover text only")

    @property
    def processed_positions(self) -> int:
        return self.prompt_tokens + self.generated_tokens - 1

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        first = self.prompt_tokens - 1
        return tuple(range(first, first + self.generated_tokens))

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "batch_size": self.batch_size,
            "decoding_mode": self.decoding_mode.value,
            "eos_termination": self.eos_termination,
            "final_generated_forward": self.final_generated_forward,
            "fixed_horizon": self.fixed_horizon,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens": self.prompt_tokens,
            "text_only": self.text_only,
        }


type CompileResult = Compiled | Unsupported


@runtime_checkable
class ArchitecturePlugin(Protocol):
    """Common registry-facing plug-in protocol."""

    @property
    def architecture_id(self) -> ArchitectureId: ...

    @property
    def plugin_id(self) -> str: ...

    @property
    def plugin_version(self) -> str: ...

    def default_request(self) -> object: ...

    def compile(self, request: object | None = None) -> CompileResult: ...


type ArchitectureRegistry = Mapping[ArchitectureId, ArchitecturePlugin]

NO_CONSTRUCTOR = "NO_CONSTRUCTOR"


@dataclass(frozen=True, slots=True)
class ConfiguredPlugin[RequestT]:
    """A plug-in that holds a model configuration but cannot compile it yet.

    ``compile`` validates the request and returns ``Unsupported``; the
    request type is where the dimensions live for a future constructor.
    """

    architecture_id: ArchitectureId
    plugin_id: str
    plugin_version: str
    request_type: type[RequestT]

    def default_request(self) -> RequestT:
        return self.request_type()

    def compile(self, request: object | None = None) -> Unsupported:
        if request is not None and not isinstance(request, self.request_type):
            raise TypeError(
                f"{self.architecture_id.value} requires {self.request_type.__name__}"
            )
        return Unsupported(
            capability=Capability.STATIC_COMPILE,
            plugin_id=self.plugin_id,
            reason_code=NO_CONSTRUCTOR,
            detail=(
                f"{self.architecture_id.value} has a model configuration but no "
                "description constructor yet"
            ),
        )


__all__ = [
    "NO_CONSTRUCTOR",
    "ArchitectureId",
    "ArchitecturePlugin",
    "ArchitectureRegistry",
    "CompileResult",
    "ConfiguredPlugin",
    "DecodingMode",
    "GreedyTextExecutionShape",
]
