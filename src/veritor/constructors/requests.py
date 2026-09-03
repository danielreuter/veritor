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
the requests grouped by kind: requests with the same prompt length,
``max_new`` and banned-list length are the same kind, and every group is one
``repeat`` of that kind (a single call for a group of one), in the order the
kinds first appear in ``x``.  A request's tokens are the last gates of its
steps, so each generated position is its own run of the root's ``Out``; a
``repeat`` of ``n`` copies makes each of them one run of ``n`` elements, so
the root has ``sum over kinds of max_new`` output runs however many
requests there are -- the verifier's ``max_output_runs`` bounds the
generated positions of the distinct *shapes*, not of the requests -- and
:meth:`output_layout` says which request each output belongs to.  A
request's ports are the weights; its prompt tokens (and its banned tokens,
and for a sampling shape its random words) are ``in`` gates inside it; its
outputs are the tokens it generates.

Marks.  ``request`` is the replay unit (with ``weights``, the one unit of
source gates); the verification units are the row-sized kinds of
:class:`~veritor.constructors.lm.ToyLM`.  Compared with ``step`` units this
puts the whole KV cache inside the unit: the boundary is the prompts, the
tokens and the weights, and a unit's declared outputs are exactly the
circuit outputs it produces.

Constrained decoding.  A request with a nonempty :attr:`Request.banned`
computes its ``vocab`` allowed flags once (``allowed_row`` units over the
banned ids, public ``in`` gates) and its steps decide with the masked head.
Tensor parallelism (``tensor_parallel``) changes the model's dot kinds and
nothing else; see :class:`~veritor.constructors.lm.ToyLM`.

Mixture of experts.  For a shape with ``experts > 0`` the ``routing`` picks
how the data-dependent route enters the circuit.  ``"padded"`` (the
default) keeps ``a`` empty: every position runs every expert and the route
is computed and applied in-circuit.  ``"advice"`` takes the routes as ``a``
(:mod:`veritor.constructors.moe`), runs only the chosen experts, and makes
each request output one more word first: ``ok``, the product of every
position's ``route_check``, which the verifier requires to be ``1``.  A
request's kind then depends on its routes, so the description grows with
the advice, and the requests of a group are separate calls (in the group's
order still, which depends on ``x`` alone).
"""

from __future__ import annotations

from collections.abc import Hashable

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue
from veritor.core.description import REPLAY

from .lm import ADVICE, PADDED, LMShape, Parameters, Routes, ToyLM, wires
from .moe import RequestRoutes, decode_routes, encode_routes, reference_routes
from .schedule import Request
from .tracer import TracedDefinition, TracerError, Wire, Wires

RequestKind = tuple[int, int, int]
"""``(prompt length, max_new, banned length)``: what decides a request's kind, from ``x`` alone."""


class RequestsG:
    """The toy decoder serving each request in its own replay unit."""

    VERSION = "2"

    def __init__(self, shape: LMShape, routing: str = PADDED, *, tensor_parallel: int = 1) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        if routing not in (PADDED, ADVICE):
            raise ValueError(f"routing must be {PADDED!r} or {ADVICE!r}")
        if routing == ADVICE and not shape.experts:
            raise ValueError("a dense shape has no route to advise")
        self.shape = shape
        self.routing = routing
        self.lm = ToyLM(shape, tensor_parallel=tensor_parallel)
        self.gate_set = self.lm.tracer.gate_set
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def advised(self) -> bool:
        return self.routing == ADVICE

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {"shape": self.shape.manifest, **self.lm.manifest}
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
                self.shape.check_banned(request)
            except ValueError as error:
                raise TracerError(f"request {index}: {error}") from error
        return x

    # -- layouts ---------------------------------------------------------------------

    @staticmethod
    def kind_of(request: Request) -> RequestKind:
        return (len(request.prompt), request.max_new, len(request.banned))

    def groups(self, x: object) -> tuple[tuple[RequestKind, tuple[int, ...]], ...]:
        """The requests' indices grouped by kind, kinds in order of first appearance."""

        groups: dict[RequestKind, list[int]] = {}
        for index, request in enumerate(self.requests(x)):
            groups.setdefault(self.kind_of(request), []).append(index)
        return tuple((kind, tuple(members)) for kind, members in groups.items())

    def order(self, x: object) -> tuple[int, ...]:
        """The requests' indices in circuit order: group by group."""

        return tuple(index for _, members in self.groups(x) for index in members)

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order.

        With advised routes a request's first output is its ``ok`` word, laid
        out at position ``-1``.
        """

        requests = self.requests(x)
        checks = (-1,) if self.advised else ()
        return tuple((r, g) for r in self.order(requests) for g in (*checks, *range(requests[r].max_new)))

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """The public inputs in ``in``-gate address order: request by request in circuit
        order, the banned tokens, the prompt tokens, then (for a sampling shape) the
        random word of every generated position."""

        requests = self.requests(x)
        return tuple(
            value
            for r in self.order(requests)
            for value in (*requests[r].banned, *requests[r].prompt, *requests[r].randomness)
        )

    def advice(self, x: object, parameters: Parameters) -> bytes:
        """The honest advice for ``x``: the routes the reference decoder takes, encoded; empty unless advised."""

        requests = self.requests(x)
        if not self.advised:
            return b""
        return encode_routes(self.shape, reference_routes(self.shape, parameters, requests))

    # -- kinds -----------------------------------------------------------------------

    def request(
        self, prompt: int, max_new: int, banned: int = 0, routes: RequestRoutes | None = None
    ) -> TracedDefinition:
        """One request: its prefill, then a decode step per further token, over its own cache.

        Ports: the weights.  Outputs: the ``max_new`` generated tokens; with
        ``routes`` (the request's routes, step by step) the ``ok`` word comes
        first.  With ``banned > 0`` the request's allowed flags are computed
        first and every step decides with the masked head.
        """

        shape, layers, d = self.shape, self.shape.layers, self.shape.d_model
        if (routes is None) == self.advised:
            raise TracerError("advised routing needs the request's routes; padded routing takes none")
        if routes is not None and len(routes) != max_new:
            raise TracerError(f"a request of {max_new} tokens has {max_new} steps of routes")
        key: tuple[Hashable, ...] = ("request", prompt, max_new)
        if banned:
            key = (*key, banned)
        if routes is not None:
            key = (*key, routes)
        masked = bool(banned)

        @self.lm.tracer.definition(input_count=shape.weight_count, key=key, role=REPLAY)
        def request(w: Wires) -> object:
            keys: list[list[Wires]] = [[] for _ in range(layers)]
            values: list[list[Wires]] = [[] for _ in range(layers)]
            ports = self.lm.ports(w)
            ok: Wire | None = ports.constants[1] if routes is not None else None

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

            mask: tuple[Wires, ...] = ()
            if banned:
                ids = self.lm.tracer.inputs(banned)
                mask = (wires(self.lm.allowed(banned)(ports.constants, ids, ports.constants[1])),)
            check: list[Wire] = [ok] if ok is not None else []
            prefill = self.lm.prefill(prompt, step_routes(0), masked=masked)
            token = remember(wires(prefill(w, *mask, *check)), prompt)
            tokens = [token]
            for step in range(1, max_new):
                args: list[Wire | Wires] = [w, token]
                for layer in range(layers):
                    args.extend(keys[layer])
                    args.extend(values[layer])
                args.extend(mask)
                if ok is not None:
                    args.append(ok)
                decode = self.lm.decode(prompt + step, step_routes(step), masked=masked)
                token = remember(wires(decode(*args)), 1)
                tokens.append(token)
            return [ok, *tokens] if ok is not None else tokens

        return request

    def root(self, requests: tuple[Request, ...], routes: tuple[RequestRoutes, ...] | None) -> TracedDefinition:
        if (routes is None) == self.advised:
            raise TracerError("advised routing needs every request's routes; padded routing takes none")

        @self.lm.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = wires(self.lm.weights_unit()())
            outputs: list[Wire | Wires] = []
            for kind, members in self.groups(requests):
                if routes is not None:  # a kind per request: its routes are its own
                    outputs.extend(self.request(*kind, routes[index])(w) for index in members)
                elif len(members) == 1:
                    outputs.append(self.request(*kind)(w))
                else:
                    outputs.append(self.lm.tracer.repeat(len(members), self.request(*kind), w))
            return outputs

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


__all__ = ["RequestKind", "RequestsG"]
