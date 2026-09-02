"""Closed registry for the six built-in architecture plug-ins."""

from __future__ import annotations

from types import MappingProxyType

from .api import (
    ArchitectureId,
    ArchitecturePlugin,
    ArchitectureRegistry,
    CompileResult,
)
from .builtin import (
    DEEPSEEK_V4_PRO_PLUGIN,
    DEMO_G_PLUGIN,
    GPT2_PLUGIN,
    INKLING_PLUGIN,
    KIMI_K3_PLUGIN,
    MATMUL_PLUGIN,
)

_ORDERED_PLUGINS: tuple[ArchitecturePlugin, ...] = (
    DEMO_G_PLUGIN,
    MATMUL_PLUGIN,
    GPT2_PLUGIN,
    KIMI_K3_PLUGIN,
    DEEPSEEK_V4_PRO_PLUGIN,
    INKLING_PLUGIN,
)

ARCHITECTURE_PLUGINS: ArchitectureRegistry = MappingProxyType(
    {plugin.architecture_id: plugin for plugin in _ORDERED_PLUGINS}
)


def architecture_registry() -> ArchitectureRegistry:
    """Return the immutable built-in registry."""

    return ARCHITECTURE_PLUGINS


def list_architectures() -> tuple[ArchitectureId, ...]:
    """Return stable IDs in stable display order."""

    return tuple(plugin.architecture_id for plugin in _ORDERED_PLUGINS)


def list_architecture_plugins() -> tuple[ArchitecturePlugin, ...]:
    return _ORDERED_PLUGINS


def get_architecture_plugin(
    architecture_id: ArchitectureId | str,
) -> ArchitecturePlugin:
    """Resolve one exact stable architecture ID."""

    try:
        resolved = ArchitectureId(architecture_id)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in list_architectures())
        raise KeyError(
            f"unknown architecture {architecture_id!r}; choose one of: {choices}"
        ) from error
    return ARCHITECTURE_PLUGINS[resolved]


def compile_architecture(
    architecture_id: ArchitectureId | str,
    request: object | None = None,
) -> CompileResult:
    """Compile through the registered plug-in for ``architecture_id``."""

    plugin = get_architecture_plugin(architecture_id)
    if request is not None:
        request_architecture = getattr(request, "architecture_id", None)
        if (
            request_architecture is not None
            and request_architecture != plugin.architecture_id
        ):
            raise ValueError(
                "compile request architecture does not match registry selection"
            )
    return plugin.compile(request)
