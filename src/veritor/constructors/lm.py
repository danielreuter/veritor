"""A toy decoder-only transformer over the toy ISA: shape, parameters, reference, kinds.

Toy numerics, shape-faithful structure.  Every value is a ``width``-bit word
and every operation is a gate of :func:`veritor.core.make_isa_gate_set`:
matrix products are modular dot products, the softmax is the polynomial
``w_j = s_j * s_j`` (there is no division), there is no layer norm, and the
attention output is shifted right by a constant so its magnitude stays
tame.  Nothing here approximates a real model; what it keeps is the *shape*
of decoding -- token embedding, per-layer attention over a KV cache that is
the cross-step state, an MLP, an LM head with an argmax or a sampler -- so
that a cluster running it has the structure a verifier has to deal with.

Three things live here:

* :class:`LMShape` and :class:`Parameters`: the model's dimensions and its
  weights, in the exact address order of the ``weight`` gates
  (:meth:`Parameters.flatten`).  Constants the circuit needs (the token
  table ``0, 1, ..., vocab - 1`` for the one-hot and the argmax, the
  shift, and the sampler's shift and bit count) are weights too: the
  grammar has no immediates, and model constants pinned under the weight
  commitment are exactly what they are.
* :func:`reference_generate`: the semantic oracle, written the ordinary
  sequential way, one request at a time (:class:`Decoder` is its
  incremental form, what a simulated server runs).  A cluster circuit's
  outputs must equal it for every schedule.
* :class:`ToyLM`: the traced definitions (the *kinds*) a cluster run is
  assembled from, with their marks.  ``dot_k``, ``onehot``,
  ``attend_head_c``, ``argmax`` (or ``sample``) and the ``add``/``square``
  cells are the verification units; the replay units (``step``,
  ``weights``) belong to :mod:`veritor.constructors.cluster`.

Sampling.  With :attr:`LMShape.sampling` the LM head draws the token
instead of taking the argmax, from a public random word ``r`` per generated
position (an ``in`` gate: the server publishes its randomness, so it is
part of ``x``).  The sampler is division-free and stays inside the word:
scores ``s_j = l_j >> score_shift`` have ``score_bits`` bits, weights
``w_j = s_j * s_j + 1`` (never zero, so the CDF is strictly increasing),
``cdf_j`` their prefix sums, ``t = (r * total) >> random_bits`` with ``r <
2**random_bits``, and the token is the first ``j`` with ``cdf_j > t``,
counted as ``sum_j [cdf_j < t + 1]``.  The bit budget ``vocab_bits + 2 *
score_bits + random_bits <= width`` keeps every value below ``2**width``.

Mixture of experts.  With :attr:`LMShape.experts` ``= E > 0`` each layer's
MLP is replaced by a router matvec (``d_model x E``) and ``E`` expert MLPs
of the dense MLP's shape; a position is routed to its ``top_k`` experts
(the largest router logits, ties to the lower index; :func:`top_k_route`)
and their outputs are summed into the residual.  The route is a
data-dependent *structural* choice, and the circuit has two ways to take
it:

* **padded** (no advice): every position runs every expert; the ``k``-hot
  route is computed in-circuit by the ``router_topk`` VU (a rank count by
  ``lt`` chains) and the ``E`` expert outputs are combined through
  ``masked_sum`` cells.  Every gate is in the circuit whatever the route.
* **advice**: the route is given to the constructor (``ceil(log2 E)`` bits
  per chosen expert per position per layer, charged as advice), the
  circuit runs only the chosen experts, and a ``route_check`` VU recomputes
  the chosen experts' ranks from the router logits and multiplies the
  result into a running ``ok`` word.  ``ok`` is an output the verifier
  requires to be ``1``: a route that is not the router's top-``k`` either
  shows as ``ok = 0`` or forces the client to break the check's relation,
  which the sampled checks catch like any other incorrect gate.

The default ``experts = 0`` is the dense decoder, byte for byte.

Constrained decoding.  A request may ban tokens (:attr:`Request.banned`,
public: part of ``x``).  The circuit computes ``allowed[k] = prod_j (1 -
eq(k, banned_j))`` once per request (``allowed_row`` VUs over ``in`` gates)
and the head takes the flags as extra ports: ``masked_argmax`` is the first
maximum among the allowed tokens, ``masked_sample`` gives a banned token the
weight ``0``.  Nothing about the constraint is advice.

Tensor parallelism and fleets.  ``ToyLM(shape, tensor_parallel=t)`` splits
every marked ``dot_k`` into ``t`` partial dots and a fixed-order reduction:
other kinds, the same values.  ``ToyLM(shape, tracer=..., namespace=...)``
builds the model on one member of a :func:`~veritor.core.gates.union_gate_set`
so that pods of different architectures share one description.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from veritor.core import JSONValue, make_isa_gate_set
from veritor.core.description import VERIFICATION
from veritor.core.gates import namespaced

from .schedule import Request
from .tracer import TracedDefinition, Tracer, TracerError, TracerGate, Wire, Wires

Matrix = tuple[tuple[int, ...], ...]
"""A row-major matrix in the ``x @ W`` orientation: rows are inputs, columns outputs."""

Route = tuple[int, ...]
"""The experts one position of one layer is routed to: ``top_k`` distinct ids, ascending."""

Routes = tuple[tuple[Route, ...], ...]
"""The routes of one step's positions, layer by layer: ``routes[l][p]`` is position ``p``'s in layer ``l``."""

PADDED = "padded"
"""A data-dependent structural choice made in-circuit, the structure padded to its maximum."""

ADVICE = "advice"
"""A data-dependent structural choice given to the constructor as charged advice."""


@dataclass(frozen=True, slots=True)
class LMShape:
    """The dimensions of the toy decoder.

    ``d_head = d_model // heads`` and ``hidden = 2 * d_model``.  ``context``
    is the longest sequence (prompt plus generated tokens) a request may
    occupy; ``width`` is the word size of every value.  Token ids are words,
    so ``vocab <= 2**width``; the argmax needs at least two candidates.
    With ``sampling`` the LM head samples from a public random word per
    generated position instead of taking the argmax (see the module
    docstring); the sampler's bit budget needs ``width >= vocab_bits + 3``.
    With ``experts = E > 0`` every layer's MLP is a mixture of ``E`` experts
    routed ``top_k`` at a time (see the module docstring); ``top_k`` must be
    below ``vocab`` because the constant table is where the circuit finds
    it.
    """

    vocab: int
    d_model: int
    heads: int
    layers: int
    context: int
    width: int
    sampling: bool = False
    experts: int = 0
    top_k: int = 1

    def __post_init__(self) -> None:
        for name in ("vocab", "d_model", "heads", "layers", "context", "width"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.sampling) is not bool:
            raise ValueError("sampling must be a bool")
        if self.vocab < 2:
            raise ValueError("vocab must be at least 2")
        if self.d_model % self.heads:
            raise ValueError("d_model must be a multiple of heads")
        if self.vocab > 1 << self.width:
            raise ValueError("token ids must be words: vocab <= 2**width")
        if self.sampling and self.width < self.vocab_bits + 3:
            raise ValueError("sampling needs width >= vocab_bits + 3")
        if type(self.experts) is not int or self.experts < 0:
            raise ValueError("experts must be a nonnegative integer")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.experts == 0 and self.top_k != 1:
            raise ValueError("top_k needs experts")
        if self.experts and self.top_k > self.experts:
            raise ValueError("top_k must be at most experts")
        if self.experts and self.top_k >= self.vocab:
            raise ValueError("top_k must be below vocab: the constant table holds it")

    @property
    def d_head(self) -> int:
        return self.d_model // self.heads

    @property
    def hidden(self) -> int:
        return 2 * self.d_model

    @property
    def route_bits(self) -> int:
        """Bits naming one expert: ``ceil(log2 experts)``; ``0`` for a dense shape."""

        return (self.experts - 1).bit_length() if self.experts else 0

    def route_advice_bits(self, positions: int) -> int:
        """The description length of the routes of ``positions`` positions: ``top_k`` ids per layer each."""

        return positions * self.layers * self.top_k * self.route_bits

    @property
    def vocab_bits(self) -> int:
        """Bits needed for a token id: ``ceil(log2 vocab)``."""

        return (self.vocab - 1).bit_length()

    @property
    def score_bits(self) -> int:
        """Bits of a sampler score ``s_j = l_j >> score_shift``."""

        return max(1, (self.width - self.vocab_bits) // 3)

    @property
    def score_shift(self) -> int:
        return self.width - self.score_bits

    @property
    def random_bits(self) -> int:
        """Bits of the public random word ``r`` a sampled position consumes."""

        return self.width - self.vocab_bits - 2 * self.score_bits

    @property
    def sampler_constants(self) -> tuple[int, ...]:
        """The sampler's ``weight`` gates: ``score_shift`` and ``random_bits``; empty for the argmax."""

        return (self.score_shift, self.random_bits) if self.sampling else ()

    @property
    def ffn_weights(self) -> int:
        """Weights of one layer's feed-forward block: the dense pair, or the router and the experts."""

        d, hidden = self.d_model, self.hidden
        if self.experts:
            return d * self.experts + self.experts * 2 * d * hidden
        return 2 * d * hidden

    @property
    def weight_count(self) -> int:
        """The number of ``weight`` gates: every matrix, the constant table, the shift, the sampler's."""

        d, vocab = self.d_model, self.vocab
        matrices = vocab * d + self.layers * (4 * d * d + self.ffn_weights) + d * vocab
        return matrices + vocab + 1 + len(self.sampler_constants)

    def state_size(self, positions: int) -> int:
        """The KV-cache entries ``positions`` new positions add: ``k`` and ``v`` per layer."""

        return 2 * self.layers * positions * self.d_model

    def check_randomness(self, request: Request) -> None:
        """A sampled model needs a ``random_bits``-bit word per generated position; the argmax none."""

        if not self.sampling:
            if request.randomness:
                raise ValueError("an argmax model takes no randomness")
            return
        if len(request.randomness) != request.max_new:
            raise ValueError("a sampled request needs one random word per generated position")
        if any(word >= 1 << self.random_bits for word in request.randomness):
            raise ValueError(f"random words must have at most {self.random_bits} bits")

    def check_banned(self, request: Request) -> None:
        """A banned-token list names distinct tokens of the vocabulary and leaves one allowed."""

        if any(token >= self.vocab for token in request.banned):
            raise ValueError("banned tokens must be below vocab")
        if len(set(request.banned)) != len(request.banned):
            raise ValueError("banned tokens must be distinct")
        if len(request.banned) >= self.vocab:
            raise ValueError("constrained decoding needs at least one allowed token")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {
            "context": self.context,
            "d_model": self.d_model,
            "heads": self.heads,
            "layers": self.layers,
            "vocab": self.vocab,
            "width": self.width,
        }
        if self.sampling:  # argmax shapes keep the manifest (and the digests) they had
            manifest["sampling"] = True
        if self.experts:  # dense shapes too
            manifest["experts"] = self.experts
            manifest["top_k"] = self.top_k
        return manifest


def _check_matrix(value: object, rows: int, columns: int, width: int, name: str) -> Matrix:
    if type(value) is not tuple or len(value) != rows:
        raise ValueError(f"{name} must be a tuple of {rows} rows")
    limit = 1 << width
    for index, row in enumerate(value):
        if type(row) is not tuple or len(row) != columns:
            raise ValueError(f"{name}[{index}] must be a tuple of {columns} values")
        for column, item in enumerate(row):
            if type(item) is not int or not 0 <= item < limit:
                raise ValueError(f"{name}[{index}][{column}] must be a {width}-bit value")
    return value


@dataclass(frozen=True, slots=True)
class ExpertParameters:
    """One expert's MLP: ``w_1`` up (``d_model x hidden``) and ``w_2`` down (``hidden x d_model``)."""

    w_1: Matrix
    w_2: Matrix


@dataclass(frozen=True, slots=True)
class LayerParameters:
    """One layer's matrices: ``w_q, w_k, w_v, w_o`` square, then the feed-forward block.

    Dense: ``w_1`` up and ``w_2`` down.  Mixture of experts (a shape with
    ``experts > 0``): the router ``w_r`` (``d_model x experts``) and one
    :class:`ExpertParameters` per expert; ``w_1`` and ``w_2`` are then
    ``None``.
    """

    w_q: Matrix
    w_k: Matrix
    w_v: Matrix
    w_o: Matrix
    w_1: Matrix | None
    w_2: Matrix | None
    w_r: Matrix | None = None
    experts: tuple[ExpertParameters, ...] = ()

    def matrices(self) -> tuple[Matrix, ...]:
        """The layer's matrices in weight-gate order (see :meth:`Parameters.flatten`)."""

        attention = (self.w_q, self.w_k, self.w_v, self.w_o)
        if self.w_r is None:
            assert self.w_1 is not None and self.w_2 is not None
            return (*attention, self.w_1, self.w_2)
        experts = tuple(m for expert in self.experts for m in (expert.w_1, expert.w_2))
        return (*attention, self.w_r, *experts)


@dataclass(frozen=True, slots=True)
class Parameters:
    """All weights of one model, in the address order of the ``weight`` gates.

    :meth:`flatten` is ``embedding`` (``vocab x d_model``), then per layer
    ``w_q, w_k, w_v, w_o`` (``d_model x d_model``), ``w_1``
    (``d_model x hidden``), ``w_2`` (``hidden x d_model``), then
    ``unembedding`` (``d_model x vocab``), the constant table
    ``0, 1, ..., vocab - 1``, ``shift`` and, for a sampling shape, the
    sampler's ``score_shift`` and ``random_bits``, every matrix row-major.
    For a mixture-of-experts shape a layer's ``w_1, w_2`` are replaced by
    the router ``w_r`` (``d_model x experts``) and then ``w_1, w_2`` of
    every expert in order.  The constants are fixed by the shape and not
    stored; ``shift`` is the amount each attention output is shifted right
    (``0`` leaves it alone).
    """

    shape: LMShape
    embedding: Matrix
    layers: tuple[LayerParameters, ...]
    unembedding: Matrix
    shift: int

    def __post_init__(self) -> None:
        shape = self.shape
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        d, hidden, vocab, width = shape.d_model, shape.hidden, shape.vocab, shape.width
        _check_matrix(self.embedding, vocab, d, width, "embedding")
        if type(self.layers) is not tuple or len(self.layers) != shape.layers:
            raise ValueError(f"layers must be a tuple of {shape.layers} LayerParameters")
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, LayerParameters):
                raise TypeError(f"layers[{index}] must be LayerParameters")
            for name in ("w_q", "w_k", "w_v", "w_o"):
                _check_matrix(getattr(layer, name), d, d, width, f"layers[{index}].{name}")
            if shape.experts:
                if layer.w_1 is not None or layer.w_2 is not None:
                    raise ValueError(f"layers[{index}] of a mixture-of-experts shape has no dense w_1, w_2")
                _check_matrix(layer.w_r, d, shape.experts, width, f"layers[{index}].w_r")
                if type(layer.experts) is not tuple or len(layer.experts) != shape.experts:
                    raise ValueError(
                        f"layers[{index}].experts must be a tuple of {shape.experts} ExpertParameters"
                    )
                for e, expert in enumerate(layer.experts):
                    if not isinstance(expert, ExpertParameters):
                        raise TypeError(f"layers[{index}].experts[{e}] must be ExpertParameters")
                    _check_matrix(
                        expert.w_1,
                        d,
                        hidden,
                        width,
                        f"layers[{index}].experts[{e}].w_1",
                    )
                    _check_matrix(
                        expert.w_2,
                        hidden,
                        d,
                        width,
                        f"layers[{index}].experts[{e}].w_2",
                    )
            else:
                if layer.w_r is not None or layer.experts:
                    raise ValueError(f"layers[{index}] of a dense shape has no router or experts")
                _check_matrix(layer.w_1, d, hidden, width, f"layers[{index}].w_1")
                _check_matrix(layer.w_2, hidden, d, width, f"layers[{index}].w_2")
        _check_matrix(self.unembedding, d, vocab, width, "unembedding")
        if type(self.shift) is not int or not 0 <= self.shift < 1 << width:
            raise ValueError(f"shift must be a {width}-bit value")

    @property
    def constants(self) -> tuple[int, ...]:
        """The token table ``0, 1, ..., vocab - 1``, a weight like any other."""

        return tuple(range(self.shape.vocab))

    def flatten(self) -> tuple[int, ...]:
        """The values of the ``weight`` gates by rank: the layout of the ``weights`` unit."""

        matrices: list[Matrix] = [self.embedding]
        for layer in self.layers:
            matrices.extend(layer.matrices())
        matrices.append(self.unembedding)
        flat = [item for matrix in matrices for row in matrix for item in row]
        flat.extend(self.constants)
        flat.append(self.shift)
        flat.extend(self.shape.sampler_constants)
        assert len(flat) == self.shape.weight_count
        return tuple(flat)


def random_parameters(shape: LMShape, seed: int) -> Parameters:
    """Uniformly random words for every matrix; the shift is ``width // 4``."""

    if not isinstance(shape, LMShape):
        raise TypeError("shape must be an LMShape")
    rng = random.Random(seed)
    limit = 1 << shape.width
    d, hidden, vocab = shape.d_model, shape.hidden, shape.vocab

    def matrix(rows: int, columns: int) -> Matrix:
        return tuple(tuple(rng.randrange(limit) for _ in range(columns)) for _ in range(rows))

    def layer() -> LayerParameters:
        attention = (matrix(d, d), matrix(d, d), matrix(d, d), matrix(d, d))
        if not shape.experts:
            return LayerParameters(*attention, matrix(d, hidden), matrix(hidden, d))
        router = matrix(d, shape.experts)
        experts = tuple(ExpertParameters(matrix(d, hidden), matrix(hidden, d)) for _ in range(shape.experts))
        return LayerParameters(*attention, None, None, router, experts)

    return Parameters(
        shape,
        matrix(vocab, d),
        tuple(layer() for _ in range(shape.layers)),
        matrix(d, vocab),
        shape.width // 4,
    )


# -- the reference: ordinary sequential decoding ---------------------------------


def _matvec(x: Sequence[int], matrix: Matrix, mask: int) -> list[int]:
    return [sum(x[i] * matrix[i][o] for i in range(len(x))) & mask for o in range(len(matrix[0]))]


def allowed_mask(vocab: int, banned: Sequence[int]) -> tuple[bool, ...] | None:
    """``allowed[k] = k not in banned``, or ``None`` when nothing is banned."""

    if not banned:
        return None
    return tuple(k not in banned for k in range(vocab))


def argmax_token(logits: Sequence[int], allowed: Sequence[bool] | None = None) -> int:
    """The first maximum of the logits, as the ``argmax`` unit computes it.

    With ``allowed`` (constrained decoding) the first maximum among the
    allowed tokens, as the ``masked_argmax`` unit computes it; at least one
    token must be allowed.
    """

    if allowed is None:
        best, index = logits[0], 0
        for candidate in range(1, len(logits)):
            if best < logits[candidate]:  # ties keep the first maximum
                best, index = logits[candidate], candidate
        return index
    if not any(allowed):
        raise ValueError("constrained decoding needs at least one allowed token")
    best, index = 0, 0
    for candidate in reversed(range(len(logits))):
        if allowed[candidate] and best <= logits[candidate]:  # scanning down: ties keep the first
            best, index = logits[candidate], candidate
    return index


def expert_ranks(logits: Sequence[int]) -> list[int]:
    """``rank_e``: the experts that beat ``e`` -- a larger logit, or an equal one at a lower index.

    This is what the ``router_topk`` and ``route_check`` VUs count with
    ``lt`` gates: ``rank_e = sum_{f < e} [g_f >= g_e] + sum_{f > e} [g_f > g_e]``.
    """

    return [
        sum(1 for f in range(e) if logits[f] >= logits[e])
        + sum(1 for f in range(e + 1, len(logits)) if logits[f] > logits[e])
        for e in range(len(logits))
    ]


def top_k_route(logits: Sequence[int], k: int) -> Route:
    """The ``k`` experts of rank below ``k``: the largest logits, ties to the lower index, ascending."""

    if not 1 <= k <= len(logits):
        raise ValueError("k must lie in 1..experts")
    return tuple(e for e, rank in enumerate(expert_ranks(logits)) if rank < k)


def sample_token(shape: LMShape, logits: Sequence[int], r: int, allowed: Sequence[bool] | None = None) -> int:
    """The token the ``sample`` unit draws from ``logits`` with the public word ``r``.

    Scores are the logits' top ``score_bits``, weights their squares plus
    one, the threshold ``(r * total) >> random_bits`` lies in
    ``[0, total)``, and the token is the number of CDF entries at most the
    threshold, i.e. the first ``j`` with ``cdf_j > t``.  No value exceeds
    ``2**width``, so the modular circuit computes exactly this.  With
    ``allowed`` a banned token's weight is zero, so it is never drawn (the
    ``masked_sample`` unit).
    """

    if not 0 <= r < 1 << shape.random_bits:
        raise ValueError(f"r must be a {shape.random_bits}-bit word")
    weights = [(logit >> shape.score_shift) ** 2 + 1 for logit in logits]
    if allowed is not None:
        if not any(allowed):
            raise ValueError("constrained decoding needs at least one allowed token")
        weights = [weight * flag for weight, flag in zip(weights, allowed, strict=True)]
    total = sum(weights)
    threshold = (r * total) >> shape.random_bits
    assert total < 1 << shape.width and threshold < total
    cdf, count = 0, 0
    for weight in weights:
        cdf += weight
        count += cdf < threshold + 1
    return count


class Decoder:
    """One request's sequential decoder: a KV cache and the forward pass of one token.

    This is the reference in incremental form -- what a server runs, one
    token at a time -- so a simulated cluster can stop a request at an
    end-of-sequence token or restart it after a failure and know exactly
    what its circuit will produce.
    """

    def __init__(self, parameters: Parameters) -> None:
        self.p = parameters
        self.shape = parameters.shape
        self.mask = (1 << self.shape.width) - 1
        self.keys: list[list[list[int]]] = [[] for _ in range(self.shape.layers)]
        self.values: list[list[list[int]]] = [[] for _ in range(self.shape.layers)]
        self.routes: list[list[Route]] = [[] for _ in range(self.shape.layers)]
        """Per layer, the route of every position fed so far (empty lists for a dense shape)."""

    @property
    def positions(self) -> int:
        """The positions fed so far: the length of the KV cache."""

        return len(self.keys[0])

    def truncate(self, keep: int) -> None:
        """Forget every position after the first ``keep``: the rollback of speculative decoding."""

        if not 0 <= keep <= self.positions:
            raise ValueError("keep must lie in 0..positions")
        for layer in range(self.shape.layers):
            del self.keys[layer][keep:]
            del self.values[layer][keep:]
            del self.routes[layer][keep:]

    def forward(self, token: int, r: int | None = None, allowed: Sequence[bool] | None = None) -> int:
        """Feed ``token`` at the next position; return the next token (argmax, or sampled
        with ``r``), among the ``allowed`` tokens when a mask is given."""

        logits = self.logits(token)
        if not self.shape.sampling:
            if r is not None:
                raise ValueError("an argmax model takes no randomness")
            return argmax_token(logits, allowed)
        if r is None:
            raise ValueError("a sampling model needs a random word")
        return sample_token(self.shape, logits, r, allowed)

    def logits(self, token: int) -> list[int]:
        """Feed ``token`` at the next position; return the logits of the position after it."""

        shape, p, mask = self.shape, self.p, self.mask
        dh = shape.d_head
        x = list(p.embedding[token])
        for index, layer in enumerate(p.layers):
            q, k, v = (_matvec(x, w, mask) for w in (layer.w_q, layer.w_k, layer.w_v))
            keys, values = self.keys[index], self.values[index]
            keys.append(k)
            values.append(v)
            attention: list[int] = []
            for head in range(shape.heads):
                low, high = head * dh, (head + 1) * dh
                scores = [sum(q[i] * key[i] for i in range(low, high)) & mask for key in keys]
                weights = [(s * s) & mask for s in scores]
                for i in range(low, high):
                    mixed = sum(w * value[i] for w, value in zip(weights, values)) & mask
                    attention.append(mixed >> p.shift if p.shift < shape.width else 0)
            x = [(a + b) & mask for a, b in zip(x, _matvec(attention, layer.w_o, mask))]
            if layer.w_r is None:
                assert layer.w_1 is not None and layer.w_2 is not None
                hidden = [(h * h) & mask for h in _matvec(x, layer.w_1, mask)]
                x = [(a + b) & mask for a, b in zip(x, _matvec(hidden, layer.w_2, mask))]
            else:
                route = top_k_route(_matvec(x, layer.w_r, mask), shape.top_k)
                self.routes[index].append(route)
                mixture = [0] * shape.d_model
                for e in route:
                    expert = layer.experts[e]
                    hidden = [(h * h) & mask for h in _matvec(x, expert.w_1, mask)]
                    mixture = [(a + b) & mask for a, b in zip(mixture, _matvec(hidden, expert.w_2, mask))]
                x = [(a + b) & mask for a, b in zip(x, mixture)]
        return _matvec(x, p.unembedding, mask)


def reference_generate(
    shape: LMShape, parameters: Parameters, requests: Sequence[Request]
) -> tuple[tuple[int, ...], ...]:
    """The generated token ids of every request, decoded one request at a time.

    The prompt is fed position by position (each attending to itself and the
    positions before it, exactly the causal prefill), the last prompt
    position's decision (argmax, or a sample with the request's first random
    word) is the first generated token, and every generated token is fed
    back for the next, ``max_new`` tokens in all.  A schedule that cuts a
    request short produces a prefix of this.
    """

    if not isinstance(parameters, Parameters) or parameters.shape != shape:
        raise ValueError("parameters must be Parameters of the given shape")
    generated: list[tuple[int, ...]] = []
    for request in requests:
        if type(request) is not Request:
            raise TypeError("requests must be Request instances")
        if any(token >= shape.vocab for token in request.prompt):
            raise ValueError("prompt tokens must be below vocab")
        shape.check_randomness(request)
        shape.check_banned(request)
        allowed = allowed_mask(shape.vocab, request.banned)
        decoder = Decoder(parameters)
        randomness = request.randomness if shape.sampling else (None,) * request.max_new
        token = 0
        for prompt_token in request.prompt[:-1]:
            decoder.logits(prompt_token)
        token = decoder.forward(request.prompt[-1], randomness[0], allowed)
        tokens = [token]
        for position in range(1, request.max_new):
            token = decoder.forward(token, randomness[position], allowed)
            tokens.append(token)
        generated.append(tuple(tokens))
    return tuple(generated)


# -- the traced definitions ---------------------------------------------------------


def wires(value: Wires) -> Wires:
    """Identity: a one-value range is a :class:`Wire` already (kept for :mod:`.gpt2`)."""

    return value


def concat(parts: Sequence[Wires]) -> Wires:
    """Consecutive results as one range: each part must start where the previous ends.

    Results of consecutive steps occupy consecutive slots, so the outputs of
    ``n`` calls emitted one after another are one range; the tracer has no
    primitive for saying so, hence this check.
    """

    first = parts[0]
    total = 0
    for part in parts:
        if (
            part.trace is not first.trace
            or part.space != first.space
            or (part.count > 1 and part.stride != 1)
            or part.jstride
            or part.start != first.start + total
        ):
            raise TracerError("concatenated ranges must be consecutive")
        total += part.count
    return Wires(first.trace, first.space, first.start, total, 1)


@dataclass(frozen=True, slots=True)
class _LayerPorts:
    w_q: Wires
    w_k: Wires
    w_v: Wires
    w_o: Wires
    w_1: Wires | None
    w_2: Wires | None
    w_r: Wires | None = None
    experts: Wires | None = None
    """Every expert's ``w_1`` then ``w_2``, consecutive: ``experts * 2 * d_model * hidden`` ports."""

    def expert(self, e: int, d: int, hidden: int) -> tuple[Wires, Wires]:
        """Expert ``e``'s ``w_1`` and ``w_2`` ports."""

        assert self.experts is not None
        return expert_ports(self.experts, e, d, hidden)


def expert_ports(experts: Wires, e: int, d: int, hidden: int) -> tuple[Wires, Wires]:
    """Expert ``e``'s ``w_1`` (``d x hidden``) and ``w_2`` (``hidden x d``) in a block of expert weights."""

    start = e * 2 * d * hidden
    return experts[start : start + d * hidden], experts[start + d * hidden : start + 2 * d * hidden]


@dataclass(frozen=True, slots=True)
class _WeightPorts:
    """The weight ports of a definition sliced in the layout of :meth:`Parameters.flatten`."""

    embedding: Wires
    layers: tuple[_LayerPorts, ...]
    unembedding: Wires
    constants: Wires
    shift: Wire
    sampler: Wires | None
    """``score_shift, random_bits`` for a sampling shape."""


class ToyLM:
    """The kinds of the toy decoder over one :class:`Tracer`, and their marks.

    Verification units, each described once whatever the number of copies:

    * ``dot_k`` -- one output of a matvec, ``k`` products and a sum tree
      (``k`` is ``vocab``, ``d_model`` or ``hidden``);
    * ``onehot`` -- the ``vocab`` equality tests of a token embedding;
    * ``attend_head_c`` -- one head over ``c`` positions: ``c`` scores, their
      squares, the mix of the values and the shift.  Its dots are the same
      body as ``dot_k`` without the mark, so no mark nests in another;
    * ``argmax`` -- the chain over ``vocab`` logits; or, for a sampling
      shape, ``sample`` -- the CDF over the squared scores and the count of
      entries below the threshold drawn from the public random word;
    * ``add`` and ``square`` cells -- the residual sums and the MLP squares;
    * the tracer's one-gate ``in`` cell for prompt tokens and random words.

    Unmarked kinds compose them: ``matvec_{k,m}`` (``m`` dots), ``embed_row``
    (one-hot then matvec), ``prefill_n`` (``n`` prompt positions with causal
    attention) and ``decode_c`` (one position over a cache of ``c - 1``).  The
    ``weights`` replay unit holds every ``weight`` gate, all declared.

    A mixture-of-experts shape adds the verification units ``router_topk``
    (the ``k``-hot route of one position from its ``E`` router logits),
    ``route_check`` (the ranks of an advised route, folded into a running
    ``ok`` word), ``masked_sum`` (one output of the ``E`` masked experts) and
    the one-gate ``mul_cell``; see the module docstring for the padded and
    the advice route.

    Constrained decoding (``masked`` steps): ``allowed_row_b`` computes
    ``allowed[k] = prod_j (1 - eq(k, banned_j))`` for one token from ``b``
    banned ids (``in`` gates: the list is public), and ``masked_argmax`` /
    ``masked_sample`` take the ``vocab`` flags as extra ports.

    Tensor parallelism: with ``tensor_parallel = t > 1`` every marked
    ``dot_k`` is ``t`` partial dots over ``k / t`` (unmarked) and a
    fixed-order chain of ``t - 1`` sums, so the kinds differ while every
    value is the same.

    Two models in one description share a ``tracer``.  A draft and a target
    for speculative decoding each qualify their cache keys with a ``prefix``
    so that shape-dependent kinds of the two models stay apart while
    identical bodies still hash-cons to one definition; the pods of a fleet
    each name a member of a :func:`~veritor.core.gates.union_gate_set` as
    their ``namespace`` and use its gates (``add@namespace`` ...), sharing
    the source gates and the ``weights`` unit.
    """

    def __init__(
        self,
        shape: LMShape,
        *,
        tracer: Tracer | None = None,
        prefix: str | None = None,
        namespace: str | None = None,
        tensor_parallel: int = 1,
    ) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        if type(tensor_parallel) is not int or tensor_parallel < 1:
            raise ValueError("tensor_parallel must be a positive integer")
        for name in ("vocab", "d_model", "hidden"):
            if getattr(shape, name) % tensor_parallel:
                raise ValueError(f"tensor_parallel must divide {name}")
        if namespace is not None and (type(namespace) is not str or not namespace):
            raise ValueError("namespace must be None or a nonempty string")
        self.shape = shape
        self.prefix = prefix
        self.namespace = namespace
        self.tensor_parallel = tensor_parallel
        qualifiers = (prefix, namespace, tensor_parallel)
        self._tag: tuple[Hashable, ...] | None = None if qualifiers == (None, None, 1) else qualifiers

        def gate_name(name: str) -> str:
            return name if namespace is None else namespaced(name, namespace)

        if tracer is None:
            tracer = Tracer(make_isa_gate_set(shape.width))
        elif tracer.gate_set[gate_name("add")].width != shape.width:
            raise TracerError("the shared tracer's gate set has another word width")
        self.tracer = tracer

        def gate(name: str) -> TracerGate:
            return self.tracer.gate(gate_name(name))

        add, mul, sub, lt, eq, shr = (gate(name) for name in ("add", "mul", "sub", "lt", "eq", "shr"))
        self.add, self.mul, self.sub, self.lt, self.eq, self.shr = (
            add,
            mul,
            sub,
            lt,
            eq,
            shr,
        )
        define = self.define
        self.mul_pair = define(input_count=2, key="mul")(lambda v: mul(v[0], v[1]))
        self.add_pair = define(input_count=2, key="add")(lambda v: add(v[0], v[1]))
        self.eq_pair = define(input_count=2, key="eq")(lambda v: eq(v[0], v[1]))
        self.shr_pair = define(input_count=2, key="shr")(lambda v: shr(v[0], v[1]))
        self.square = define(input_count=1, key="square")(lambda v: mul(v[0], v[0]))
        self.square_cell = define(input_count=1, key="square_cell", role=VERIFICATION)(
            lambda v: mul(v[0], v[0])
        )
        self.add_cell = define(input_count=2, key="add_cell", role=VERIFICATION)(lambda v: add(v[0], v[1]))
        self.mul_cell = define(input_count=2, key="mul_cell", role=VERIFICATION)(lambda v: mul(v[0], v[1]))

    def key(self, *parts: Hashable) -> Hashable:
        """A tracer cache key: as given for a lone model; qualified by the prefix, the namespace and
        the TP degree when any is set, so models sharing a tracer keep their kinds apart."""

        if self._tag is not None:
            return (*self._tag, *parts)
        return parts[0] if len(parts) == 1 else parts

    def define(
        self, *, input_count: int, key: Hashable, role: str | None = None
    ) -> Callable[[Callable[[Wires], object]], TracedDefinition]:
        """:meth:`Tracer.definition` under this model's :meth:`key`."""

        parts = key if isinstance(key, tuple) else (key,)
        return self.tracer.definition(input_count=input_count, key=self.key(*parts), role=role)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        """What a constructor built on this model adds to its manifest (empty for the defaults)."""

        manifest: dict[str, JSONValue] = {}
        if self.namespace is not None:
            manifest["namespace"] = self.namespace
        if self.tensor_parallel != 1:
            manifest["tensor_parallel"] = self.tensor_parallel
        return manifest

    # -- ports --------------------------------------------------------------------

    def ports(self, v: Wires) -> _WeightPorts:
        """Slice ``v[:weight_count]`` in the order of :meth:`Parameters.flatten`."""

        shape = self.shape
        d, hidden, vocab = shape.d_model, shape.hidden, shape.vocab
        cursor = 0

        def take(count: int) -> Wires:
            nonlocal cursor
            piece = v[cursor : cursor + count]
            cursor += count
            return piece

        def layer() -> _LayerPorts:
            attention = (take(d * d), take(d * d), take(d * d), take(d * d))
            if not shape.experts:
                return _LayerPorts(*attention, take(d * hidden), take(hidden * d))
            return _LayerPorts(
                *attention,
                None,
                None,
                take(d * shape.experts),
                take(shape.ffn_weights - d * shape.experts),
            )

        embedding = take(vocab * d)
        layers = tuple(layer() for _ in range(shape.layers))
        unembedding = take(d * vocab)
        constants = take(vocab)
        shift = take(1)[0]
        sampler = take(2) if shape.sampling else None
        assert cursor == shape.weight_count
        return _WeightPorts(embedding, layers, unembedding, constants, shift, sampler)

    # -- verification units ----------------------------------------------------------

    def dot(self, k: int, *, marked: bool = True) -> TracedDefinition:
        """``x . w`` over ``k``-vectors: a ``repeat`` of products, then a tree of pairwise sums.

        Marked it is the verification unit of every matvec output; unmarked
        (the same body, no role) it is a building block of ``attend_head``.
        """

        if type(k) is not int or k <= 0:
            raise TracerError("dot length must be positive")
        role = VERIFICATION if marked else None
        shards = self.tensor_parallel if marked else 1
        if k % shards:
            raise TracerError("tensor_parallel must divide the length of every marked dot")

        @self.define(input_count=2 * k, key=("dot", k, role), role=role)
        def dot(v: Wires) -> object:
            x, w = v[:k], v[k:]
            if shards > 1:  # tensor parallel: a partial dot per shard, then a fixed-order reduction
                part = k // shards
                partial = self.dot(part, marked=False)
                partials = [partial(x[i * part : (i + 1) * part], w[i * part : (i + 1) * part]) for i in range(shards)]
                total = partials[0]
                for term in partials[1:]:
                    total = self.add(total, term)
                return total
            level = self.tracer.repeat(k, self.mul_pair, x[0].by(1), w[0].by(1))
            carried = []
            while len(level) > 1:
                if len(level) % 2:
                    carried.append(level[-1])
                level = self.tracer.repeat(len(level) // 2, self.add_pair, level[0:2].by(2))
            result = level[0]
            for carry in carried:
                result = self.add(result, carry)
            return result

        return dot

    def onehot(self) -> TracedDefinition:
        """``eq(t, k)`` for every token ``k`` of the constant table."""

        vocab = self.shape.vocab

        @self.define(input_count=1 + vocab, key="onehot", role=VERIFICATION)
        def onehot(v: Wires) -> object:
            return self.tracer.repeat(vocab, self.eq_pair, v[0], v[1].by(1))

        return onehot

    def allowed_row(self, b: int) -> TracedDefinition:
        """``prod_j (1 - eq(k, banned_j))`` for one token ``k``: ports ``k``, the ``b`` banned ids, ``1``."""

        if type(b) is not int or b <= 0:
            raise TracerError("a banned list must be nonempty")

        @self.define(input_count=b + 2, key=("allowed_row", b), role=VERIFICATION)
        def allowed_row(v: Wires) -> object:
            token, banned, one = v[0], v[1 : 1 + b], v[b + 1]
            flag = self.sub(one, self.eq(token, banned[0]))
            for j in range(1, b):
                flag = self.mul(flag, self.sub(one, self.eq(token, banned[j])))
            return flag

        return allowed_row

    def allowed(self, b: int) -> TracedDefinition:
        """The ``vocab`` allowed flags of a request with ``b`` banned tokens.

        Ports: the constant table, then the ``b`` banned ids, then the
        constant ``1``; one ``allowed_row`` unit per token.
        """

        vocab = self.shape.vocab

        @self.define(input_count=vocab + b + 1, key=("allowed", b))
        def allowed(v: Wires) -> object:
            constants, banned, one = v[:vocab], v[vocab : vocab + b], v[vocab + b]
            return self.tracer.repeat(vocab, self.allowed_row(b), constants[0].by(1), banned, one)

        return allowed

    def attend_head(self, c: int) -> TracedDefinition:
        """One head over ``c`` positions: ports ``q, k_0..k_{c-1}, v_0..v_{c-1}, shift``.

        ``s_j = q . k_j``, ``w_j = s_j * s_j``, ``out_i = (sum_j w_j v_j[i]) >> shift``.
        """

        if type(c) is not int or c <= 0:
            raise TracerError("attention context must be positive")
        dh = self.shape.d_head
        repeat = self.tracer.repeat

        @self.define(input_count=dh + 2 * c * dh + 1, key=("attend_head", c), role=VERIFICATION)
        def attend_head(v: Wires) -> object:
            q, keys, values, shift = (
                v[:dh],
                v[dh : dh + c * dh],
                v[dh + c * dh : dh + 2 * c * dh],
                v[-1],
            )
            scores = repeat(c, self.dot(dh, marked=False), q, keys[0:dh].by(dh))
            weights = repeat(c, self.square, scores[0].by(1))
            mixed = repeat(dh, self.dot(c, marked=False), weights, values[0 : c * dh : dh].by(1))
            return repeat(dh, self.shr_pair, mixed[0].by(1), shift)

        return attend_head

    def argmax(self) -> TracedDefinition:
        """The first maximum of ``vocab`` logits as a token id, by a chain of ``lt`` selects."""

        vocab = self.shape.vocab

        @self.define(input_count=2 * vocab, key="argmax", role=VERIFICATION)
        def argmax(v: Wires) -> object:
            logits, constants = v[:vocab], v[vocab:]
            best, index = logits[0], constants[0]
            for k in range(1, vocab):
                better = self.lt(best, logits[k])
                best = self.add(best, self.mul(better, self.sub(logits[k], best)))
                index = self.add(index, self.mul(better, self.sub(constants[k], index)))
            return index

        return argmax

    def masked_argmax(self) -> TracedDefinition:
        """The first maximum among the allowed tokens: ports the logits, the ``vocab`` allowed
        flags and the constant table.

        The chain scans the tokens downwards from a virtual candidate of
        value ``0`` and takes token ``k`` when it is allowed and ``best <=
        l_k``, so the result is the smallest index attaining the maximum
        over the allowed tokens (:func:`argmax_token` with a mask).  Nine
        gates per token against the plain argmax's seven.
        """

        vocab = self.shape.vocab

        @self.define(input_count=3 * vocab, key="masked_argmax", role=VERIFICATION)
        def masked_argmax(v: Wires) -> object:
            logits, allowed, constants = v[:vocab], v[vocab : 2 * vocab], v[2 * vocab :]
            one = constants[1]
            best, index = constants[0], constants[0]
            for k in reversed(range(vocab)):
                better = self.mul(allowed[k], self.sub(one, self.lt(logits[k], best)))
                best = self.add(best, self.mul(better, self.sub(logits[k], best)))
                index = self.add(index, self.mul(better, self.sub(constants[k], index)))
            return index

        return masked_argmax

    def _sampler(self, logits: Wires, allowed: Wires | None, ports: Wires) -> Wire:
        """The sampler's body over ``logits`` with ports ``r, 1, score_shift, random_bits``."""

        add, mul, lt, shr = self.add, self.mul, self.lt, self.shr
        r, one, score_shift, random_bits = ports[0], ports[1], ports[2], ports[3]
        weights = []
        for k, logit in enumerate(logits):
            score = shr(logit, score_shift)
            weight = add(mul(score, score), one)
            weights.append(weight if allowed is None else mul(allowed[k], weight))
        cdf = [weights[0]]
        for weight in weights[1:]:
            cdf.append(add(cdf[-1], weight))
        bound = add(shr(mul(r, cdf[-1]), random_bits), one)  # t + 1 with t = (r * total) >> random_bits
        below = [lt(entry, bound) for entry in cdf]  # cdf_j <= t
        index = below[0]
        for flag in below[1:]:
            index = add(index, flag)
        return index

    def sample(self) -> TracedDefinition:
        """A token drawn from ``vocab`` logits with the public word ``r``: see the module docstring.

        Ports: the logits, ``r``, the constant ``1``, ``score_shift`` and
        ``random_bits``.  Every value stays below ``2**width`` by the shape's
        bit budget, so the modular gates compute :func:`sample_token` exactly.
        """

        vocab = self.shape.vocab

        @self.define(input_count=vocab + 4, key="sample", role=VERIFICATION)
        def sample(v: Wires) -> object:
            return self._sampler(v[:vocab], None, v[vocab:])

        return sample

    def masked_sample(self) -> TracedDefinition:
        """:meth:`sample` with the ``vocab`` allowed flags after the logits: a banned token's
        weight is ``0``, so it is never drawn.  One gate more per token."""

        vocab = self.shape.vocab

        @self.define(input_count=2 * vocab + 4, key="masked_sample", role=VERIFICATION)
        def masked_sample(v: Wires) -> object:
            return self._sampler(v[:vocab], v[vocab : 2 * vocab], v[2 * vocab :])

        return masked_sample

    # -- mixture of experts ----------------------------------------------------------

    def _ranks(self, logits: Wires, one: Wire, chosen: Sequence[int]) -> dict[int, Wire]:
        """``rank_e`` of every expert in ``chosen`` (see :func:`expert_ranks`), by ``lt`` gates.

        A pair ``f < e`` shares one ``lt(g_f, g_e)``: it counts towards
        ``rank_f`` as is and towards ``rank_e`` as ``1 - lt``.
        """

        add, sub, lt = self.add, self.sub, self.lt
        wanted = set(chosen)
        beaten: dict[tuple[int, int], Wire] = {}
        ranks: dict[int, Wire] = {}
        for e in sorted(wanted):
            terms: list[Wire] = []
            for f in range(len(logits)):
                if f == e:
                    continue
                low, high = min(e, f), max(e, f)
                if (low, high) not in beaten:
                    beaten[(low, high)] = lt(logits[low], logits[high])  # g_high beats g_low
                terms.append(beaten[(low, high)] if f > e else sub(one, beaten[(low, high)]))
            rank = terms[0]
            for term in terms[1:]:
                rank = add(rank, term)
            ranks[e] = rank
        return ranks

    def router_topk(self) -> TracedDefinition:
        """The ``k``-hot route of one position: ``[rank_e < k]`` for every expert ``e``.

        Ports: the ``E`` router logits, the constants ``1`` and ``k``.  The
        padded route's decision unit: every expert's flag is a gate the
        sampled checks can catch, and the flags mask the experts' outputs.
        """

        experts, k = self.shape.experts, self.shape.top_k
        if experts < 2:
            raise TracerError("routing needs at least two experts")

        @self.define(input_count=experts + 2, key=("router_topk", experts, k), role=VERIFICATION)
        def router_topk(v: Wires) -> object:
            logits, one, kconst = v[:experts], v[experts], v[experts + 1]
            ranks = self._ranks(logits, one, range(experts))
            return [self.lt(ranks[e], kconst) for e in range(experts)]

        return router_topk

    def route_check(self, route: Route) -> TracedDefinition:
        """``ok_in * prod_{e in route} [rank_e < k]``: the advised route against the router's logits.

        Ports: the ``E`` router logits, the constants ``1`` and ``k``, the
        running ``ok``.  Only the advised experts' ranks are computed, so the
        unit costs ``k (E - 1)`` comparisons instead of the padded route's
        ``E (E - 1) / 2``.  Its output is folded into the request's ``ok``
        output, which the verifier requires to be ``1``.
        """

        experts, k = self.shape.experts, self.shape.top_k
        self.check_route(route)

        @self.define(
            input_count=experts + 3,
            key=("route_check", experts, k, route),
            role=VERIFICATION,
        )
        def route_check(v: Wires) -> object:
            logits, one, kconst, ok = (
                v[:experts],
                v[experts],
                v[experts + 1],
                v[experts + 2],
            )
            ranks = self._ranks(logits, one, route)
            for e in route:
                ok = self.mul(ok, self.lt(ranks[e], kconst))
            return ok

        return route_check

    def check_route(self, route: object) -> Route:
        """A route is ``top_k`` distinct expert ids below ``experts``, ascending."""

        experts, k = self.shape.experts, self.shape.top_k
        if type(route) is not tuple or len(route) != k:
            raise TracerError(f"a route names exactly {k} experts")
        if any(type(e) is not int or not 0 <= e < experts for e in route):
            raise TracerError(f"route {route!r} names an expert outside 0..{experts - 1}")
        if any(a >= b for a, b in pairwise(route)):
            raise TracerError(f"route {route!r} must list distinct experts in ascending order")
        return route

    def masked_sum(self) -> TracedDefinition:
        """``sum_e flag_e * y_e`` over the ``E`` experts: one output of the padded mixture."""

        experts = self.shape.experts

        @self.define(input_count=2 * experts, key=("masked_sum", experts), role=VERIFICATION)
        def masked_sum(v: Wires) -> object:
            flags, values = v[:experts], v[experts:]
            total = self.mul(flags[0], values[0])
            for e in range(1, experts):
                total = self.add(total, self.mul(flags[e], values[e]))
            return total

        return masked_sum

    def combine(self) -> TracedDefinition:
        """The ``d_model`` masked sums of one position: ports ``E`` flags then ``E`` expert outputs."""

        experts, d = self.shape.experts, self.shape.d_model

        @self.define(input_count=experts + experts * d, key=("combine", experts, d))
        def combine(v: Wires) -> object:
            flags, values = v[:experts], v[experts:]
            return self.tracer.repeat(d, self.masked_sum(), flags, values[0 : experts * d : d].by(1))

        return combine

    def expert_mlp(self, w_1: Wires, w_2: Wires, x: Wires, positions: int) -> Wires:
        """One expert (or the dense MLP) over ``positions`` positions: up, square, down."""

        d, hidden = self.shape.d_model, self.shape.hidden
        repeat = self.tracer.repeat
        h = repeat(positions, self.matvec(d, hidden), x[0:d].by(d), w_1)
        h = repeat(positions * hidden, self.square_cell, h[0].by(1))
        return repeat(positions, self.matvec(hidden, d), h[0:hidden].by(hidden), w_2)

    def moe_padded(self, layer: _LayerPorts, x: Wires, positions: int, constants: Wires) -> Wires:
        """The padded mixture: every position through every expert, combined by the in-circuit route."""

        shape = self.shape
        experts, k, d = shape.experts, shape.top_k, shape.d_model
        repeat = self.tracer.repeat
        assert layer.w_r is not None
        logits = repeat(positions, self.matvec(d, experts), x[0:d].by(d), layer.w_r)
        flags = repeat(
            positions,
            self.router_topk(),
            logits[0:experts].by(experts),
            constants[1],
            constants[k],
        )
        outputs = [self.expert_mlp(*layer.expert(e, d, shape.hidden), x, positions) for e in range(experts)]
        mixture = repeat(
            positions,
            self.combine(),
            flags[0:experts].by(experts),
            *(y[0:d].by(d) for y in outputs),
        )
        return repeat(positions * d, self.add_cell, x[0].by(1), mixture[0].by(1))

    def moe_block(self, routes: tuple[Route, ...]) -> TracedDefinition:
        """The advised mixture of one layer over ``len(routes)`` positions, route ``routes[p]`` at ``p``.

        Ports: the positions' activations (``positions * d_model``), the
        router, every expert's weights, the constants ``1`` and ``k``, the
        running ``ok``.  Outputs: the new activations, then ``ok``.  Only the
        advised experts run; each position's ``route_check`` folds its
        verdict into ``ok``.  A definition per distinct route pattern: the
        description grows with the advice, as it must.
        """

        shape = self.shape
        experts, k, d, hidden = shape.experts, shape.top_k, shape.d_model, shape.hidden
        positions = len(routes)
        if positions < 1:
            raise TracerError("a mixture block needs at least one position")
        for route in routes:
            self.check_route(route)
        repeat = self.tracer.repeat
        width = positions * d + d * experts + shape.ffn_weights - d * experts + 3

        @self.define(input_count=width, key=("moe_block", experts, k, routes))
        def moe_block(v: Wires) -> object:
            x = v[: positions * d]
            w_r = v[positions * d : positions * d + d * experts]
            weights = v[positions * d + d * experts : width - 3]
            one, kconst, ok = v[width - 3], v[width - 2], v[width - 1]
            new: list[Wires] = []
            for p, route in enumerate(routes):
                x_p = x[p * d : (p + 1) * d]
                logits = self.matvec(d, experts)(x_p, w_r)
                ok = self.route_check(route)(logits, one, kconst, ok)[0]
                mixture: Wires | None = None
                for e in route:
                    y = self.expert_mlp(*expert_ports(weights, e, d, hidden), x_p, 1)
                    mixture = y if mixture is None else repeat(d, self.add_cell, mixture[0].by(1), y[0].by(1))
                assert mixture is not None
                new.append(repeat(d, self.add_cell, x_p[0].by(1), mixture[0].by(1)))
            return [*new, ok]

        return moe_block

    def moe_advice(
        self,
        layer: _LayerPorts,
        x: Wires,
        positions: int,
        constants: Wires,
        routes: tuple[Route, ...],
        ok: Wire,
    ) -> tuple[Wires, Wire]:
        """The advised mixture over ``positions`` positions: the new activations and the running ``ok``."""

        shape = self.shape
        assert layer.w_r is not None and layer.experts is not None and len(routes) == positions
        result = self.moe_block(routes)(
            x, layer.w_r, layer.experts, constants[1], constants[shape.top_k], ok
        )
        return result[: positions * shape.d_model], result[positions * shape.d_model]

    def head(
        self, logits: Wire | Wires, ports: _WeightPorts, r: Wire | None, allowed: Wires | None = None
    ) -> Wire | Wires:
        """The LM head's decision: the argmax, or a sample with the position's random word;
        among the ``allowed`` tokens when a request is constrained."""

        if not self.shape.sampling:
            assert r is None
            if allowed is None:
                return self.argmax()(logits, ports.constants)
            return self.masked_argmax()(logits, allowed, ports.constants)
        assert r is not None and ports.sampler is not None
        if allowed is None:
            return self.sample()(logits, r, ports.constants[1], ports.sampler)
        return self.masked_sample()(logits, allowed, r, ports.constants[1], ports.sampler)

    def randomness(self) -> Wire | None:
        """The ``in`` gate of a position's public random word, for a sampling shape."""

        if not self.shape.sampling:
            return None
        wire = self.tracer.inputs(1)[0]
        assert isinstance(wire, Wire)
        return wire

    # -- unmarked composites -------------------------------------------------------

    def matvec(self, k: int, m: int) -> TracedDefinition:
        """``x W`` for a ``k``-vector and a row-major ``k x m`` matrix: ``m`` dot units."""

        @self.define(input_count=k + k * m, key=("matvec", k, m))
        def matvec(v: Wires) -> object:
            x, w = v[:k], v[k:]
            return self.tracer.repeat(m, self.dot(k), x, w[0 : k * m : m].by(1))

        return matvec

    def embed_row(self) -> TracedDefinition:
        """A token id to its embedding: one-hot against the constant table, then ``E``."""

        vocab, d = self.shape.vocab, self.shape.d_model

        @self.define(input_count=1 + vocab + vocab * d, key="embed_row")
        def embed_row(v: Wires) -> object:
            token, constants, embedding = v[0], v[1 : 1 + vocab], v[1 + vocab :]
            return self.matvec(vocab, d)(self.onehot()(token, constants), embedding)

        return embed_row

    def forward(
        self,
        ports: _WeightPorts,
        x: Wires,
        positions: int,
        caches: Sequence[tuple[Wires, Wires] | None],
        routes: Sequence[tuple[Route, ...]] | None = None,
        ok: Wire | None = None,
    ) -> tuple[list[Wires], Wires, Wire | None]:
        """The layers over ``positions`` new positions of one sequence.

        ``x`` holds the new positions' embeddings; ``caches[l]`` the layer's
        cached ``(K, V)`` for the earlier positions, position-major, or
        ``None``.  New position ``p`` attends to the cache and to new positions
        ``0..p``.  Returns the new ``K`` and ``V`` of every layer (position-major
        blocks, the cache entries later steps read), the final activations
        and the running ``ok`` word.

        A mixture-of-experts shape takes the padded route when ``routes`` is
        ``None`` and the advised route otherwise: ``routes[l][p]`` is position
        ``p``'s route in layer ``l`` and ``ok`` the word the route checks fold
        into (``None`` for a dense or a padded pass).
        """

        shape = self.shape
        d, dh, heads = shape.d_model, shape.d_head, shape.heads
        if routes is not None and (ok is None or not shape.experts):
            raise TracerError("advised routes need a mixture-of-experts shape and an ok word")
        if routes is not None and (len(routes) != shape.layers or any(len(r) != positions for r in routes)):
            raise TracerError(f"routes must give {shape.layers} layers of {positions} positions")
        repeat = self.tracer.repeat
        project = self.matvec(d, d)

        def head_slices(blocks: Sequence[Wires], count: int) -> list[Wires]:
            """Head ``0``'s ``dh`` values at each of ``count`` positions, shifting by head."""

            return [block[j * d : j * d + dh].by(dh) for block in blocks for j in range(count)]

        state: list[Wires] = []
        for index, (layer, cache) in enumerate(zip(ports.layers, caches, strict=True)):
            q = repeat(positions, project, x[0:d].by(d), layer.w_q)
            k = repeat(positions, project, x[0:d].by(d), layer.w_k)
            v = repeat(positions, project, x[0:d].by(d), layer.w_v)
            cached = 0 if cache is None else len(cache[0]) // d
            attended = []
            for p in range(positions):
                keys = head_slices(() if cache is None else (cache[0],), cached)
                values = head_slices(() if cache is None else (cache[1],), cached)
                attended.append(
                    repeat(
                        heads,
                        self.attend_head(cached + p + 1),
                        q[p * d : p * d + dh].by(dh),
                        *keys,
                        *head_slices((k,), p + 1),
                        *values,
                        *head_slices((v,), p + 1),
                        ports.shift,
                    )
                )
            o = repeat(positions, project, concat(attended)[0:d].by(d), layer.w_o)
            x = repeat(positions * d, self.add_cell, x[0].by(1), o[0].by(1))
            if layer.w_r is None:
                assert layer.w_1 is not None and layer.w_2 is not None
                m = self.expert_mlp(layer.w_1, layer.w_2, x, positions)
                x = repeat(positions * d, self.add_cell, x[0].by(1), m[0].by(1))
            elif routes is None:
                x = self.moe_padded(layer, x, positions, ports.constants)
            else:
                assert ok is not None
                x, ok = self.moe_advice(layer, x, positions, ports.constants, routes[index], ok)
            state += [k, v]
        return state, x, ok

    def _step(
        self,
        *,
        cached: int,
        new: int,
        inside: bool,
        heads: int,
        routes: Routes | None,
        key: tuple[Hashable, ...],
        masked: bool = False,
    ) -> TracedDefinition:
        """The body shared by :meth:`prefill`, :meth:`chunk`, :meth:`decode` and :meth:`extend`.

        ``new`` positions after ``cached`` cached ones; the tokens are ``in``
        gates inside the definition (``inside``) or ports after the weights;
        the head decides the last ``heads`` positions (``0``: none, a chunk
        of a prompt that only extends the cache).  Ports: the weights, the
        tokens (unless inside), per layer the cached ``K`` then ``V``, for a
        ``masked`` head the request's ``vocab`` allowed flags, and for advised
        routes the incoming ``ok``.  Outputs: per layer the new ``K`` then
        ``V`` (position-major), the decided tokens, and for advised routes
        the outgoing ``ok``.
        """

        shape = self.shape
        d, vocab, weights = shape.d_model, shape.vocab, shape.weight_count
        if heads not in (0, 1, new):
            raise TracerError("the head decides no position, the last position or every position")
        if heads > 1 and shape.sampling:
            raise TracerError("a sampling shape decides one position per step")
        if masked and heads != 1:
            raise TracerError("a masked step decides exactly one position")
        advised = routes is not None
        if routes is not None:
            if not shape.experts:
                raise TracerError("routes need a mixture-of-experts shape")
            if len(routes) != shape.layers or any(len(r) != new for r in routes):
                raise TracerError(f"routes must give {shape.layers} layers of {new} positions")
            for layer_routes in routes:
                for route in layer_routes:
                    self.check_route(route)
            key = (*key, routes)
        token_ports = 0 if inside else new
        cache = cached * d
        mask_start = weights + token_ports + shape.layers * 2 * cache
        input_count = mask_start + (vocab if masked else 0) + (1 if advised else 0)

        @self.define(input_count=input_count, key=key)
        def step(v: Wires) -> object:
            ports = self.ports(v)
            tokens = self.tracer.inputs(new) if inside else v[weights : weights + new]
            caches: list[tuple[Wires, Wires] | None] = []
            for layer in range(shape.layers):
                if not cached:
                    caches.append(None)
                    continue
                start = weights + token_ports + layer * 2 * cache
                caches.append((v[start : start + cache], v[start + cache : start + 2 * cache]))
            allowed = v[mask_start : mask_start + vocab] if masked else None
            ok = v[input_count - 1] if advised else None
            r = self.randomness() if heads == 1 else None
            embed = self.embed_row()
            if not inside and new == 1:
                x = embed(tokens[0], ports.constants, ports.embedding)  # a lone token is a call
            else:
                x = self.tracer.repeat(new, embed, tokens[0].by(1), ports.constants, ports.embedding)
            state, x, ok = self.forward(ports, x, new, caches, routes, ok)
            unembed = self.matvec(d, vocab)
            decided: list[Wire | Wires] = []
            if heads == 1:
                decided.append(self.head(unembed(x[(new - 1) * d : new * d], ports.unembedding), ports, r, allowed))
            elif heads:
                logits = self.tracer.repeat(new, unembed, x[0:d].by(d), ports.unembedding)
                decided.append(self.tracer.repeat(new, self.argmax(), logits[0:vocab].by(vocab), ports.constants))
            return [*state, *decided, *([ok] if advised else [])]

        return step

    def prefill(
        self, n: int, routes: Routes | None = None, *, cached: int = 0, masked: bool = False
    ) -> TracedDefinition:
        """An ``n``-token prompt: ports are the weights; the tokens are ``in`` gates inside.

        Outputs: per layer ``K`` then ``V`` for the ``n`` positions
        (``state_size(n)`` values, position-major), then the first generated
        token, the head's decision on the last position's logits.  For a
        sampling shape the position's random word is one more ``in`` gate,
        after the prompt tokens.

        For a mixture-of-experts shape ``routes[l][p]`` (layer ``l``, prompt
        position ``p``) takes the advice route: one more port, the incoming
        ``ok``, and one more output, the outgoing ``ok``.  Without ``routes``
        the mixture is padded.

        With ``cached > 0`` the ``n`` tokens end a prompt whose first
        ``cached`` positions earlier :meth:`chunk` steps processed: per layer
        their ``K`` and ``V`` are ports after the weights.  A ``masked`` step
        takes the request's ``vocab`` allowed flags as further ports and
        decides among them.
        """

        if type(n) is not int or n <= 0:
            raise TracerError("prompt length must be positive")
        if type(cached) is not int or cached < 0:
            raise TracerError("cached positions must be nonnegative")
        key: tuple[Hashable, ...] = ("prefill", n, *((cached,) if cached else ()), *(("masked",) if masked else ()))
        return self._step(cached=cached, new=n, inside=True, heads=1, routes=routes, key=key, masked=masked)

    def chunk(self, n: int, cached: int) -> TracedDefinition:
        """``n`` prompt tokens that do not end the prompt (chunked prefill): :meth:`prefill`
        without the head, so no token and no random word.

        Outputs: per layer ``K`` then ``V`` for the ``n`` positions.
        """

        if type(n) is not int or n <= 0:
            raise TracerError("chunk length must be positive")
        if type(cached) is not int or cached < 0:
            raise TracerError("cached positions must be nonnegative")
        return self._step(cached=cached, new=n, inside=True, heads=0, routes=None, key=("chunk", n, cached))

    def prefill_ports(self, n: int) -> TracedDefinition:
        """:meth:`prefill` with the ``n`` prompt tokens as ports after the weights instead of ``in`` gates.

        Two models reading one prompt (speculative decoding's draft and
        target) take it this way from ``in`` gates the caller emits once.
        """

        if type(n) is not int or n <= 0:
            raise TracerError("prompt length must be positive")
        return self._step(cached=0, new=n, inside=False, heads=1, routes=None, key=("prefill_ports", n))

    def decode(self, c: int, routes: Routes | None = None, *, masked: bool = False) -> TracedDefinition:
        """One token at context ``c``: ports are the weights, the token, then per layer
        the cached ``K`` and ``V`` of the ``c - 1`` earlier positions (then, when
        ``masked``, the request's ``vocab`` allowed flags).

        Outputs: per layer the new ``k`` then ``v`` (``state_size(1)`` values), then
        the next token.  For a sampling shape the position's random word is
        an ``in`` gate inside the step.  ``routes`` as in :meth:`prefill`,
        one position per layer.
        """

        if type(c) is not int or c < 2:
            raise TracerError("a decode step needs at least one cached position")
        key: tuple[Hashable, ...] = ("decode", c, *(("masked",) if masked else ()))
        return self._step(cached=c - 1, new=1, inside=False, heads=1, routes=routes, key=key, masked=masked)

    def extend(self, cached: int, new: int, routes: Routes | None = None) -> TracedDefinition:
        """``new`` tokens after ``cached`` cached positions, the head deciding every one of them.

        Ports: the weights, the ``new`` tokens, per layer the cached ``K`` then
        ``V``.  Outputs: per layer the new ``K`` then ``V`` (``state_size(new)``
        values), then the ``new`` argmax tokens, position by position: token
        ``p`` is the model's greedy continuation of the sequence through new
        position ``p``.  This is the target model's verification pass of
        speculative decoding; a sampling shape has no such pass.
        """

        if type(cached) is not int or cached < 0 or type(new) is not int or new < 1:
            raise TracerError("extend needs a nonnegative cache and at least one new token")
        return self._step(
            cached=cached,
            new=new,
            inside=False,
            heads=new,
            routes=routes,
            key=("extend", cached, new),
        )

    def weights_unit(self) -> TracedDefinition:
        """The replay unit holding every ``weight`` gate, all declared.

        Models of one shape on a shared tracer (a fleet's architectures, any
        TP degree) share it: the weights are one environment.  A ``prefix``
        (another model, another shape) keeps its own.
        """

        key: Hashable = "weights" if self.prefix is None else (self.prefix, "weights")
        return self.tracer.definition(input_count=0, key=key, role="replay")(
            lambda _v: self.tracer.weights(self.shape.weight_count)
        )


__all__ = [
    "Decoder",
    "LMShape",
    "LayerParameters",
    "Matrix",
    "Parameters",
    "ToyLM",
    "allowed_mask",
    "argmax_token",
    "concat",
    "random_parameters",
    "reference_generate",
    "sample_token",
    "wires",
]
