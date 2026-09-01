"""Trusted registries for value-commitment backends."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from .merkle import MerkleSha256Backend
from .types import CommitmentError, ValueCommitmentBackend


class ValueCommitmentRegistry:
    """An immutable allowlist of locally trusted commitment backends."""

    __slots__ = ("_backends",)

    def __init__(self, backends: Iterable[ValueCommitmentBackend]) -> None:
        by_id: dict[str, ValueCommitmentBackend] = {}
        for backend in backends:
            if not isinstance(backend, ValueCommitmentBackend):
                raise CommitmentError(
                    "commitment registry entries must satisfy ValueCommitmentBackend"
                )
            backend_id = backend.backend_id
            if type(backend_id) is not str or not backend_id.strip():
                raise CommitmentError("commitment backend_id must be nonempty")
            if backend_id in by_id:
                raise CommitmentError(f"duplicate commitment backend id {backend_id!r}")
            by_id[backend_id] = backend
        self._backends = MappingProxyType(by_id)

    @classmethod
    def with_defaults(cls) -> ValueCommitmentRegistry:
        return cls((MerkleSha256Backend(),))

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def resolve(self, backend_id: str) -> ValueCommitmentBackend | None:
        if type(backend_id) is not str:
            return None
        return self._backends.get(backend_id)

    def require(self, backend_id: str) -> ValueCommitmentBackend:
        backend = self.resolve(backend_id)
        if backend is None:
            raise CommitmentError(f"unknown commitment backend {backend_id!r}")
        return backend
