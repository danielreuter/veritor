"""Prefix caching with the shared prefix as its own replay unit (route A of C6).

``PrefixG(shape)`` serves requests that share a system prompt.  The shared
prefix is the longest common prefix of the prompts (public: a function of
``x``), computed once by a ``prefix`` replay unit whose declared outputs are
its KV blocks -- ``state_size(p)`` words, the widest interface in the
circuit -- and read through ports by one ``suffix`` replay unit per request,
which prefills the request's own tokens over that cache and decodes as
:class:`~veritor.constructors.requests.RequestsG` does.  Route B, the
alternative, is ``RequestsG`` itself: every request recomputes the prefix
inside its own unit (mechanism M1) and nothing crosses a unit boundary but
the tokens.

Structure.  The root calls ``weights``, then ``prefix_p``, then the suffix
requests grouped by kind ``(suffix length, max_new)`` as ``RequestsG``
groups them.  Inputs in address order: the prefix tokens, then per request
in circuit order its suffix tokens and (for a sampling shape) its random
words.  Every prompt must be strictly longer than the prefix (a request's
prefill has at least one token of its own); the prefix has at least one.
"""

from __future__ import annotations

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue
from veritor.core.description import REPLAY

from .lm import LMShape, ToyLM, wires
from .schedule import Request
from .tracer import TracedDefinition, TracerError, Wire, Wires

SuffixKind = tuple[int, int]
"""``(suffix length, max_new)``: what decides a suffix request's kind."""


def shared_prefix(requests: tuple[Request, ...]) -> int:
    """The length of the longest common prefix of the prompts, leaving every prompt a token of its own."""

    first = requests[0].prompt
    length = min(len(request.prompt) for request in requests) - 1
    for request in requests:
        for index in range(length):
            if request.prompt[index] != first[index]:
                length = index
                break
    return length


class PrefixG:
    """The toy decoder over a shared prefix unit and one suffix unit per request."""

    VERSION = "1"

    def __init__(self, shape: LMShape) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        self.shape = shape
        self.lm = ToyLM(shape)
        self.gate_set = self.lm.tracer.gate_set
        self.digest: Digest = constructor_digest(
            type(self).__name__, self.VERSION, self.manifest
        )

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"shape": self.shape.manifest}

    # -- validation -----------------------------------------------------------------

    def requests(self, x: object) -> tuple[Request, ...]:
        if (
            type(x) is not tuple
            or not x
            or any(type(item) is not Request for item in x)
        ):
            raise TracerError("PrefixG expects a nonempty tuple of Request")
        for index, request in enumerate(x):
            if any(token >= self.shape.vocab for token in request.prompt):
                raise TracerError(
                    f"request {index} has a prompt token outside the vocabulary"
                )
            if len(request.prompt) + request.max_new > self.shape.context:
                raise TracerError(
                    f"request {index} needs {len(request.prompt) + request.max_new} positions; "
                    f"the context is {self.shape.context}"
                )
            if request.banned:
                raise TracerError(
                    f"request {index} bans tokens: PrefixG serves unconstrained requests"
                )
            try:
                self.shape.check_randomness(request)
            except ValueError as error:
                raise TracerError(f"request {index}: {error}") from error
        if shared_prefix(x) < 1:
            raise TracerError(
                "PrefixG needs prompts that share at least one token and each have one more"
            )
        return x

    # -- layouts ---------------------------------------------------------------------

    def groups(self, x: object) -> tuple[tuple[SuffixKind, tuple[int, ...]], ...]:
        """The requests' indices grouped by suffix kind, kinds in order of first appearance."""

        requests = self.requests(x)
        prefix = shared_prefix(requests)
        groups: dict[SuffixKind, list[int]] = {}
        for index, request in enumerate(requests):
            groups.setdefault(
                (len(request.prompt) - prefix, request.max_new), []
            ).append(index)
        return tuple((kind, tuple(members)) for kind, members in groups.items())

    def order(self, x: object) -> tuple[int, ...]:
        return tuple(index for _, members in self.groups(x) for index in members)

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order."""

        requests = self.requests(x)
        return tuple(
            (r, g) for r in self.order(requests) for g in range(requests[r].max_new)
        )

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        requests = self.requests(x)
        prefix = shared_prefix(requests)
        values = list(requests[0].prompt[:prefix])
        for r in self.order(requests):
            values.extend(requests[r].prompt[prefix:])
            values.extend(requests[r].randomness)
        return tuple(values)

    # -- kinds -----------------------------------------------------------------------

    def prefix_unit(self, prefix: int) -> TracedDefinition:
        """The shared prefix: ports the weights; outputs its KV blocks (``state_size(prefix)``)."""

        @self.lm.tracer.definition(
            input_count=self.shape.weight_count, key=("prefix", prefix), role=REPLAY
        )
        def prefix_unit(w: Wires) -> object:
            return self.lm.chunk(prefix, 0)(w)

        return prefix_unit

    def suffix(self, prefix: int, length: int, max_new: int) -> TracedDefinition:
        """One request over the shared cache: ports the weights, then the prefix's KV blocks.

        Outputs: the ``max_new`` generated tokens.
        """

        shape, layers, d = self.shape, self.shape.layers, self.shape.d_model
        weights, cache = shape.weight_count, shape.state_size(prefix)

        @self.lm.tracer.definition(
            input_count=weights + cache,
            key=("suffix", prefix, length, max_new),
            role=REPLAY,
        )
        def suffix(v: Wires) -> object:
            w, kv = v[:weights], v[weights:]
            block = prefix * d
            keys: list[list[Wires]] = [
                [kv[2 * layer * block : 2 * layer * block + block]]
                for layer in range(layers)
            ]
            values: list[list[Wires]] = [
                [kv[2 * layer * block + block : 2 * (layer + 1) * block]]
                for layer in range(layers)
            ]

            def remember(produced: Wires, positions: int) -> Wire:
                for layer in range(layers):
                    start = 2 * layer * positions * d
                    keys[layer].append(produced[start : start + positions * d])
                    values[layer].append(
                        produced[start + positions * d : start + 2 * positions * d]
                    )
                return produced[-1]

            token = remember(
                wires(self.lm.prefill(length, cached=prefix)(w, kv)), length
            )
            tokens = [token]
            for step in range(1, max_new):
                args: list[Wire | Wires] = [w, token]
                for layer in range(layers):
                    args.extend(keys[layer])
                    args.extend(values[layer])
                token = remember(
                    wires(self.lm.decode(prefix + length + step)(*args)), 1
                )
                tokens.append(token)
            return tokens

        return suffix

    def root(self, requests: tuple[Request, ...]) -> TracedDefinition:
        prefix = shared_prefix(requests)

        @self.lm.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = wires(self.lm.weights_unit()())
            kv = wires(self.prefix_unit(prefix)(w))
            outputs: list[Wire | Wires] = []
            for (length, max_new), members in self.groups(requests):
                definition = self.suffix(prefix, length, max_new)
                if len(members) == 1:
                    outputs.append(definition(w, kv))
                else:
                    outputs.append(
                        self.lm.tracer.repeat(len(members), definition, w, kv)
                    )
            return outputs

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        if a:
            raise TracerError("PrefixG takes no advice")
        requests = self.requests(x)
        return self.lm.tracer.serialize(self.root(requests)), self.flatten_inputs(
            requests
        )


__all__ = ["PrefixG", "SuffixKind", "shared_prefix"]
