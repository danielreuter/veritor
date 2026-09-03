"""Several weight sets in one run (W1, W3) and per-request adapters (W2), without a protocol change.

The protocol has one weight root per session (``commit_weights``,
``Header.weights``): the Merkle root over every ``weight`` gate of the
circuit in rank order.  That is enough to serve more than one model, as long
as the *description* says which weights each request reads:

* :class:`ModelsG` serves ``models`` weight sets of one shape -- two versions
  of a model across a hot swap (W1), or several fine-tunes behind a router
  (W3).  Its ``weights`` replay unit holds every set in order, and each
  request's kind is called on the slice of its model, so a request of model
  ``k`` is exactly the ``RequestsG`` request kind wired to ``k``'s weights.
  Which model a request ran on is public input (``routing="input"``: the
  client asked for it, M2, 0 bits) or advice (``routing="advice"``: the
  server chose, M4, ``ceil(log2 models)`` bits per request).  The one root
  commits to the concatenation of the sets.

* :class:`AdaptedRequestsG` gives each request its own copy of one weight
  matrix -- a merged LoRA adapter ``W + A B``, the per-tenant part of a
  multi-tenant deployment -- as ``in`` gates of a small ``adapter`` replay
  unit: public inputs, committed in the boundary like the prompt, read by the
  request through the ports the base matrix would have filled.  No weight
  root is involved; the adapter is part of ``x``.

What this does not give, and a header with a *tuple* of weight roots would:
a per-model root that is stable across sessions (the joint root changes with
the set of models served), a request that names its root publicly rather
than through the description's wiring, and adapters as weights (private to
the server, committed once) rather than as public inputs.  Models of
*different shapes* on one cluster need one description with several
``ToyLM`` kind families; the tracer builds one family per shape today.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue
from veritor.core.description import REPLAY

from .lm import LMShape, Matrix, Parameters, wires
from .requests import RequestsG
from .schedule import Request
from .tracer import TracedDefinition, TracerError
from .truncation import field_width, pack_fields, unpack_fields

ROUTINGS = ("input", "advice")


class ModelsG:
    """``models`` weight sets of one shape under one root; each request names its model."""

    VERSION = "1"

    def __init__(self, shape: LMShape, models: int, routing: str = "input") -> None:
        if type(models) is not int or models < 1:
            raise TypeError("models must be a positive integer")
        if routing not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}")
        self.inner = RequestsG(shape)
        self.shape = shape
        self.models = models
        self.routing = routing
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"models": self.models, "routing": self.routing, "shape": self.shape.manifest}

    # -- weights --------------------------------------------------------------------

    def flatten_weights(self, parameters: Sequence[Parameters]) -> tuple[int, ...]:
        """The values of the ``weight`` gates: every model's :meth:`Parameters.flatten` in order."""

        if len(parameters) != self.models:
            raise ValueError(f"expected {self.models} parameter sets")
        return tuple(value for model in parameters for value in model.flatten())

    def weights_unit(self) -> TracedDefinition:
        tracer, count = self.inner.lm.tracer, self.models * self.shape.weight_count
        return tracer.definition(input_count=0, key=("weights", self.models), role=REPLAY)(
            lambda _v: tracer.weights(count)
        )

    # -- inputs and advice ----------------------------------------------------------

    def advice_bits(self, x: object) -> int:
        """``ceil(log2 models)`` per request when the server routes; ``0`` when the client does."""

        if self.routing == "input":
            return 0
        return len(self.requests(x)) * field_width(self.models)

    def advice(self, x: object, models: Sequence[int]) -> bytes:
        """The advice naming ``models[i]`` for request ``i`` (``routing="advice"`` only)."""

        if self.routing != "advice":
            raise ValueError("the model choice is public input under this routing")
        requests = self.requests(x)
        if len(models) != len(requests) or any(
            type(m) is not int or not 0 <= m < self.models for m in models
        ):
            raise ValueError(f"one model in 0..{self.models - 1} per request")
        return pack_fields(models, [field_width(self.models)] * len(requests))

    def _pairs(self, x: object) -> tuple[tuple[int, Request], ...]:
        """``x`` under input routing: validated ``(model, Request)`` pairs."""

        if type(x) is not tuple or not x or any(type(item) is not tuple or len(item) != 2 for item in x):
            raise TracerError("ModelsG expects a nonempty tuple of (model, Request) pairs")
        pairs: list[tuple[int, Request]] = []
        for model, request in x:
            if type(model) is not int or not 0 <= model < self.models:
                raise TracerError(f"model must lie in 0..{self.models - 1}")
            pairs.append((model, request))
        self.inner.requests(tuple(request for _model, request in pairs))
        return tuple(pairs)

    def requests(self, x: object) -> tuple[Request, ...]:
        if self.routing == "input":
            return tuple(request for _model, request in self._pairs(x))
        return self.inner.requests(x)

    def assignments(self, x: object, a: bytes) -> tuple[tuple[int, Request], ...]:
        """``(model, request)`` for every request, from ``x`` or from the advice."""

        if self.routing == "input":
            if a:
                raise TracerError("ModelsG with input routing takes no advice")
            return self._pairs(x)
        requests = self.requests(x)
        try:
            models = unpack_fields(a, [field_width(self.models)] * len(requests))
        except ValueError as error:
            raise TracerError(f"malformed routing advice: {error}") from error
        if any(model >= self.models for model in models):
            raise TracerError("routing advice names a model that is not served")
        return tuple(zip(models, requests, strict=True))

    def output_layout(self, x: object, a: bytes) -> tuple[tuple[int, int], ...]:
        return self.inner.output_layout(tuple(r for _m, r in self.assignments(x, a)))

    def flatten_inputs(self, x: object, a: bytes) -> tuple[int, ...]:
        return self.inner.flatten_inputs(tuple(r for _m, r in self.assignments(x, a)))

    # -- the description --------------------------------------------------------------

    def root(self, assignments: Sequence[tuple[int, Request]]) -> TracedDefinition:
        n = self.shape.weight_count

        @self.inner.lm.tracer.definition(input_count=0)
        def root(_v: object) -> object:
            w = wires(self.weights_unit()())
            return [
                self.inner.request(len(request.prompt), request.max_new)(w[model * n : (model + 1) * n])
                for model, request in assignments
            ]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        assignments = self.assignments(x, a)
        return self.inner.lm.tracer.serialize(self.root(assignments)), self.flatten_inputs(x, a)


MATRICES = ("w_q", "w_k", "w_v", "w_o", "w_1", "w_2")


class AdaptedRequestsG:
    """Every request brings its own copy of one weight matrix as public inputs (a merged adapter)."""

    VERSION = "1"

    def __init__(self, shape: LMShape, layer: int = 0, matrix: str = "w_1") -> None:
        if type(layer) is not int or not 0 <= layer < shape.layers:
            raise ValueError(f"layer must lie in 0..{shape.layers - 1}")
        if matrix not in MATRICES:
            raise ValueError(f"matrix must be one of {MATRICES}")
        self.inner = RequestsG(shape)
        self.shape = shape
        self.layer = layer
        self.matrix = matrix
        d, hidden = shape.d_model, shape.hidden
        sizes = {"w_q": d * d, "w_k": d * d, "w_v": d * d, "w_o": d * d, "w_1": d * hidden, "w_2": hidden * d}
        first = shape.vocab * d + layer * (4 * d * d + 2 * d * hidden)
        for name in MATRICES:
            if name == matrix:
                break
            first += sizes[name]
        self.first = first
        """The rank of the adapted matrix's first word in :meth:`Parameters.flatten`."""
        self.count = sizes[matrix]
        """The words of the adapted matrix: the ``in`` gates each request adds."""
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"layer": self.layer, "matrix": self.matrix, "shape": self.shape.manifest}

    def merged(self, base: Parameters, adapter: Matrix) -> Parameters:
        """``base`` with the adapted matrix replaced by ``adapter``: the reference for one tenant."""

        layers = list(base.layers)
        layers[self.layer] = replace(layers[self.layer], **{self.matrix: adapter})
        return replace(base, layers=tuple(layers))

    def requests(self, x: object) -> tuple[tuple[tuple[int, ...], Request], ...]:
        """``(adapter, request)`` pairs: the adapted matrix row-major, then the request."""

        if type(x) is not tuple or not x or any(type(item) is not tuple or len(item) != 2 for item in x):
            raise TracerError("AdaptedRequestsG expects a nonempty tuple of (adapter, Request) pairs")
        limit = 1 << self.shape.width
        pairs: list[tuple[tuple[int, ...], Request]] = []
        for adapter, request in x:
            if (
                type(adapter) is not tuple
                or len(adapter) != self.count
                or any(type(v) is not int or not 0 <= v < limit for v in adapter)
            ):
                raise TracerError(f"an adapter is {self.count} words of the model's width")
            pairs.append((adapter, request))
        self.inner.requests(tuple(request for _adapter, request in pairs))
        return tuple(pairs)

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        return self.inner.output_layout(tuple(r for _a, r in self.requests(x)))

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """Per request: the adapter's words, then the prompt tokens, then the random words."""

        return tuple(
            value
            for adapter, request in self.requests(x)
            for value in (*adapter, *request.prompt, *request.randomness)
        )

    def adapter_unit(self) -> TracedDefinition:
        tracer, count = self.inner.lm.tracer, self.count
        return tracer.definition(input_count=0, key=("adapter", count), role=REPLAY)(
            lambda _v: tracer.inputs(count)
        )

    def root(self, requests: Sequence[tuple[tuple[int, ...], Request]]) -> TracedDefinition:
        first, stop = self.first, self.first + self.count

        @self.inner.lm.tracer.definition(input_count=0)
        def root(_v: object) -> object:
            w = wires(self.inner.lm.weights_unit()())
            results = []
            for _adapter, request in requests:
                adapter = wires(self.adapter_unit()())
                results.append(
                    self.inner.request(len(request.prompt), request.max_new)(w[:first], adapter, w[stop:])
                )
            return results

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        if a:
            raise TracerError("AdaptedRequestsG takes no advice")
        requests = self.requests(x)
        return self.inner.lm.tracer.serialize(self.root(requests)), self.flatten_inputs(x)


__all__ = ["MATRICES", "ROUTINGS", "AdaptedRequestsG", "ModelsG"]
