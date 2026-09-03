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

Mixture of experts.  For a shape with ``experts > 0`` the ``routing`` picks
how the data-dependent route enters the circuit.  ``"padded"`` (the
default) keeps ``a`` empty: every position runs every expert and the route
is computed and applied in-circuit.  ``"advice"`` takes the routes as ``a``
(:mod:`veritor.constructors.moe`), runs only the chosen experts, and makes
each request output one more word first: ``ok``, the product of every
position's ``route_check``, which the verifier requires to be ``1``.  A
request's kind then depends on its routes, so the description grows with
the advice.
"""

from __future__ import annotations

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue
from veritor.core.description import REPLAY

from .lm import ADVICE, PADDED, LMShape, Parameters, Routes, ToyLM, wires
from .moe import RequestRoutes, decode_routes, encode_routes, reference_routes
from .schedule import Request
from .tracer import TracedDefinition, TracerError, Wire, Wires


class RequestsG:
    """The toy decoder serving each request in its own replay unit."""

    VERSION = "1"

    def __init__(self, shape: LMShape, routing: str = PADDED) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        if routing not in (PADDED, ADVICE):
            raise ValueError(f"routing must be {PADDED!r} or {ADVICE!r}")
        if routing == ADVICE and not shape.experts:
            raise ValueError("a dense shape has no route to advise")
        self.shape = shape
        self.routing = routing
        self.lm = ToyLM(shape)
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def advised(self) -> bool:
        return self.routing == ADVICE

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {"shape": self.shape.manifest}
        if self.shape.experts:
            manifest["routing"] = self.routing
        return manifest

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
            try:
                self.shape.check_randomness(request)
            except ValueError as error:
                raise TracerError(f"request {index}: {error}") from error
        return x

    # -- layouts ---------------------------------------------------------------------

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order.

        With advised routes a request's first output is its ``ok`` word, laid
        out at position ``-1``.
        """

        requests = self.requests(x)
        checks = (-1,) if self.advised else ()
        return tuple((r, g) for r, request in enumerate(requests) for g in (*checks, *range(request.max_new)))

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """The public inputs in ``in``-gate address order: request by request, the prompt
        tokens then (for a sampling shape) the random word of every generated position."""

        return tuple(
            value for request in self.requests(x) for value in (*request.prompt, *request.randomness)
        )

    def advice(self, x: object, parameters: Parameters) -> bytes:
        """The honest advice for ``x``: the routes the reference decoder takes, encoded; empty unless advised."""

        requests = self.requests(x)
        if not self.advised:
            return b""
        return encode_routes(self.shape, reference_routes(self.shape, parameters, requests))

    # -- kinds -----------------------------------------------------------------------

    def request(self, prompt: int, max_new: int, routes: RequestRoutes | None = None) -> TracedDefinition:
        """One request: its prefill, then a decode step per further token, over its own cache.

        Ports: the weights.  Outputs: the ``max_new`` generated tokens; with
        ``routes`` (the request's routes, step by step) the ``ok`` word comes
        first.
        """

        shape, layers, d = self.shape, self.shape.layers, self.shape.d_model
        if (routes is None) == self.advised:
            raise TracerError("advised routing needs the request's routes; padded routing takes none")
        if routes is not None and len(routes) != max_new:
            raise TracerError(f"a request of {max_new} tokens has {max_new} steps of routes")
        key = ("request", prompt, max_new) if routes is None else ("request", prompt, max_new, routes)

        @self.lm.tracer.definition(input_count=shape.weight_count, key=key, role=REPLAY)
        def request(w: Wires) -> object:
            keys: list[list[Wires]] = [[] for _ in range(layers)]
            values: list[list[Wires]] = [[] for _ in range(layers)]
            ok: Wire | None = self.lm.ports(w).constants[1] if routes is not None else None

            def remember(block: Wires, positions: int) -> Wire:
                nonlocal ok
                for layer in range(layers):
                    start = 2 * layer * positions * d
                    keys[layer].append(block[start : start + positions * d])
                    values[layer].append(block[start + positions * d : start + 2 * positions * d])
                if routes is None:
                    return block[-1]
                ok = block[-1]
                return block[-2]

            def step_routes(step: int) -> Routes | None:
                return None if routes is None else routes[step]

            check: list[Wire] = [ok] if ok is not None else []
            token = remember(wires(self.lm.prefill(prompt, step_routes(0))(w, *check)), prompt)
            tokens = [token]
            for step in range(1, max_new):
                args: list[Wire | Wires] = [w, token]
                for layer in range(layers):
                    args.extend(keys[layer])
                    args.extend(values[layer])
                if ok is not None:
                    args.append(ok)
                token = remember(wires(self.lm.decode(prompt + step, step_routes(step))(*args)), 1)
                tokens.append(token)
            return [ok, *tokens] if ok is not None else tokens

        return request

    def root(self, requests: tuple[Request, ...], routes: tuple[RequestRoutes, ...] | None) -> TracedDefinition:
        if (routes is None) == self.advised:
            raise TracerError("advised routing needs every request's routes; padded routing takes none")

        @self.lm.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = wires(self.lm.weights_unit()())
            return [
                self.request(len(request.prompt), request.max_new, None if routes is None else routes[index])(w)
                for index, request in enumerate(requests)
            ]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        requests = self.requests(x)
        if not self.advised:
            if a:
                raise TracerError("RequestsG takes no advice unless the routes are advised")
            return self.lm.tracer.serialize(self.root(requests, None)), self.flatten_inputs(requests)
        routes = decode_routes(self.shape, requests, a)
        return self.lm.tracer.serialize(self.root(requests, routes)), self.flatten_inputs(requests)


__all__ = ["RequestsG"]
