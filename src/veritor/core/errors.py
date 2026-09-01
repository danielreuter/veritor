"""Errors raised at the protocol core trust boundary."""

from __future__ import annotations


class CoreContractError(Exception):
    """Base class for errors originating in the core contract layer."""


class InvalidArtifact(CoreContractError, ValueError):
    """A supplied artifact is malformed or violates a core invariant."""


class ResourceLimit(CoreContractError, RuntimeError):
    """A supported operation exceeded an explicit resource limit."""

    def __init__(
        self,
        resource: str,
        *,
        limit: int | None = None,
        observed: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.resource = resource
        self.limit = limit
        self.observed = observed
        if detail is not None:
            message = detail
        elif limit is None:
            message = f"resource limit exceeded for {resource}"
        elif observed is None:
            message = f"{resource} exceeds limit {limit}"
        else:
            message = f"{resource} is {observed}, exceeding limit {limit}"
        super().__init__(message)


class UnsupportedCapability(CoreContractError, RuntimeError):
    """An artifact or plug-in does not implement a requested capability."""

    def __init__(
        self,
        capability: object,
        *,
        reason_code: str,
        detail: str,
    ) -> None:
        self.capability = capability
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{capability!s} is unsupported ({reason_code}): {detail}")


class BackendUnavailable(CoreContractError, RuntimeError):
    """An optional implementation for a supported operation is unavailable."""


# Longer spellings are retained as convenience names for downstream layers.
ResourceLimitExceeded = ResourceLimit
