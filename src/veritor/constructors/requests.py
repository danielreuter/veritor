"""The per-request constructor: one replay unit per request, no advice.

``RequestsG(shape)`` traces the toy decoder (:mod:`veritor.constructors.lm`)
serving each request of ``x`` on its own: a request is a replay unit holding
its prefill and every decode step, wired through its own KV cache, which
never leaves the unit.  Nothing about the circuit is left to the client once
``x`` is fixed, so the advice is empty (``a = b""``) and the schedule of
:class:`~veritor.constructors.cluster.ClusterG` -- which request occupies
which slot of which pod at which step -- is not part of the statement at
all: batching is the server's business, invisible to the verifier.

Structure.  The root has no ports.  It calls the ``weights`` unit once, then
one ``request`` per element of ``x``, in order.  A request's ports are the
weights; its prompt tokens are ``in`` gates inside it; its outputs are the
tokens it generates.  Two requests with the same prompt length and
``max_new`` are the same kind.

Marks.  ``request`` is the replay unit (with ``weights``, the one unit of
source gates); the verification units are the row-sized kinds of
:class:`~veritor.constructors.lm.ToyLM`.  Compared with ``step`` units this
puts the whole KV cache inside the unit: the boundary is the prompts, the
tokens and the weights, and a unit's declared outputs are exactly the
circuit outputs it produces.
"""

from __future__ import annotations

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue
from veritor.core.description import REPLAY

from .lm import LMShape, ToyLM, wires
from .schedule import Request
from .tracer import TracedDefinition, TracerError, Wire, Wires


class RequestsG:
    """The toy decoder serving each request in its own replay unit."""

    VERSION = "1"

    def __init__(self, shape: LMShape) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        self.shape = shape
        self.lm = ToyLM(shape)
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"shape": self.shape.manifest}

    # -- validation -----------------------------------------------------------------

    def requests(self, x: object) -> tuple[Request, ...]:
        if type(x) is not tuple or not x or any(type(item) is not Request for item in x):
            raise TracerError("RequestsG expects a nonempty tuple of Request")
        for index, request in enumerate(x):
            if any(token >= self.shape.vocab for token in request.prompt):
                raise TracerError(f"request {index} has a prompt token outside the vocabulary")
            if len(request.prompt) + request.max_new > self.shape.context:
                raise TracerError(
                    f"request {index} needs {len(request.prompt) + request.max_new} positions; "
                    f"the context is {self.shape.context}"
                )
        return x

    # -- layouts ---------------------------------------------------------------------

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order."""

        requests = self.requests(x)
        return tuple((r, g) for r, request in enumerate(requests) for g in range(request.max_new))

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """The prompt tokens in ``in``-gate address order: request by request."""

        return tuple(token for request in self.requests(x) for token in request.prompt)

    # -- kinds -----------------------------------------------------------------------

    def request(self, prompt: int, max_new: int) -> TracedDefinition:
        """One request: its prefill, then a decode step per further token, over its own cache.

        Ports: the weights.  Outputs: the ``max_new`` generated tokens.
        """

        shape, layers, d = self.shape, self.shape.layers, self.shape.d_model

        @self.lm.tracer.definition(input_count=shape.weight_count, key=("request", prompt, max_new), role=REPLAY)
        def request(w: Wires) -> object:
            keys: list[list[Wires]] = [[] for _ in range(layers)]
            values: list[list[Wires]] = [[] for _ in range(layers)]

            def remember(block: Wires, positions: int) -> Wire:
                for layer in range(layers):
                    start = 2 * layer * positions * d
                    keys[layer].append(block[start : start + positions * d])
                    values[layer].append(block[start + positions * d : start + 2 * positions * d])
                return block[-1]

            token = remember(wires(self.lm.prefill(prompt)(w)), prompt)
            tokens = [token]
            for step in range(1, max_new):
                args: list[Wire | Wires] = [w, token]
                for layer in range(layers):
                    args.extend(keys[layer])
                    args.extend(values[layer])
                token = remember(wires(self.lm.decode(prompt + step)(*args)), 1)
                tokens.append(token)
            return tokens

        return request

    def root(self, requests: tuple[Request, ...]) -> TracedDefinition:
        @self.lm.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = wires(self.lm.weights_unit()())
            return [self.request(len(request.prompt), request.max_new)(w) for request in requests]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        if a:
            raise TracerError("RequestsG takes no advice")
        requests = self.requests(x)
        return self.lm.tracer.serialize(self.root(requests)), self.flatten_inputs(requests)


__all__ = ["RequestsG"]
