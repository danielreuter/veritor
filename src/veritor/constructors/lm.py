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
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from veritor.core import JSONValue, make_isa_gate_set
from veritor.core.description import VERIFICATION

from .schedule import Request
from .tracer import TracedDefinition, Tracer, TracerError, Wire, Wires

Matrix = tuple[tuple[int, ...], ...]
"""A row-major matrix in the ``x @ W`` orientation: rows are inputs, columns outputs."""


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
    """

    vocab: int
    d_model: int
    heads: int
    layers: int
    context: int
    width: int
    sampling: bool = False

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

    @property
    def d_head(self) -> int:
        return self.d_model // self.heads

    @property
    def hidden(self) -> int:
        return 2 * self.d_model

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
    def weight_count(self) -> int:
        """The number of ``weight`` gates: every matrix, the constant table, the shift, the sampler's."""

        d, hidden, vocab = self.d_model, self.hidden, self.vocab
        matrices = vocab * d + self.layers * (4 * d * d + 2 * d * hidden) + d * vocab
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
class LayerParameters:
    """One layer's matrices: ``w_q, w_k, w_v, w_o`` square, ``w_1`` up and ``w_2`` down."""

    w_q: Matrix
    w_k: Matrix
    w_v: Matrix
    w_o: Matrix
    w_1: Matrix
    w_2: Matrix


@dataclass(frozen=True, slots=True)
class Parameters:
    """All weights of one model, in the address order of the ``weight`` gates.

    :meth:`flatten` is ``embedding`` (``vocab x d_model``), then per layer
    ``w_q, w_k, w_v, w_o`` (``d_model x d_model``), ``w_1``
    (``d_model x hidden``), ``w_2`` (``hidden x d_model``), then
    ``unembedding`` (``d_model x vocab``), the constant table
    ``0, 1, ..., vocab - 1``, ``shift`` and, for a sampling shape, the
    sampler's ``score_shift`` and ``random_bits``, every matrix row-major.
    The constants are fixed by the shape and not stored; ``shift`` is the
    amount each attention output is shifted right (``0`` leaves it alone).
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
            matrices.extend((layer.w_q, layer.w_k, layer.w_v, layer.w_o, layer.w_1, layer.w_2))
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

    return Parameters(
        shape,
        matrix(vocab, d),
        tuple(
            LayerParameters(
                matrix(d, d), matrix(d, d), matrix(d, d), matrix(d, d), matrix(d, hidden), matrix(hidden, d)
            )
            for _ in range(shape.layers)
        ),
        matrix(d, vocab),
        shape.width // 4,
    )


# -- the reference: ordinary sequential decoding ---------------------------------


def _matvec(x: Sequence[int], matrix: Matrix, mask: int) -> list[int]:
    return [sum(x[i] * matrix[i][o] for i in range(len(x))) & mask for o in range(len(matrix[0]))]


def argmax_token(logits: Sequence[int]) -> int:
    """The first maximum of the logits, as the ``argmax`` unit computes it."""

    best, index = logits[0], 0
    for candidate in range(1, len(logits)):
        if best < logits[candidate]:  # ties keep the first maximum
            best, index = logits[candidate], candidate
    return index


def sample_token(shape: LMShape, logits: Sequence[int], r: int) -> int:
    """The token the ``sample`` unit draws from ``logits`` with the public word ``r``.

    Scores are the logits' top ``score_bits``, weights their squares plus
    one, the threshold ``(r * total) >> random_bits`` lies in
    ``[0, total)``, and the token is the number of CDF entries at most the
    threshold, i.e. the first ``j`` with ``cdf_j > t``.  No value exceeds
    ``2**width``, so the modular circuit computes exactly this.
    """

    if not 0 <= r < 1 << shape.random_bits:
        raise ValueError(f"r must be a {shape.random_bits}-bit word")
    weights = [(logit >> shape.score_shift) ** 2 + 1 for logit in logits]
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

    def forward(self, token: int, r: int | None = None) -> int:
        """Feed ``token`` at the next position; return the next token (argmax, or sampled with ``r``)."""

        logits = self.logits(token)
        if not self.shape.sampling:
            if r is not None:
                raise ValueError("an argmax model takes no randomness")
            return argmax_token(logits)
        if r is None:
            raise ValueError("a sampling model needs a random word")
        return sample_token(self.shape, logits, r)

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
                scores = [
                    sum(q[i] * key[i] for i in range(low, high)) & mask for key in keys
                ]
                weights = [(s * s) & mask for s in scores]
                for i in range(low, high):
                    mixed = sum(w * value[i] for w, value in zip(weights, values)) & mask
                    attention.append(mixed >> p.shift if p.shift < shape.width else 0)
            x = [(a + b) & mask for a, b in zip(x, _matvec(attention, layer.w_o, mask))]
            hidden = [(h * h) & mask for h in _matvec(x, layer.w_1, mask)]
            x = [(a + b) & mask for a, b in zip(x, _matvec(hidden, layer.w_2, mask))]
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
        decoder = Decoder(parameters)
        randomness = request.randomness if shape.sampling else (None,) * request.max_new
        token = 0
        for prompt_token in request.prompt[:-1]:
            decoder.logits(prompt_token)
        token = decoder.forward(request.prompt[-1], randomness[0])
        tokens = [token]
        for position in range(1, request.max_new):
            token = decoder.forward(token, randomness[position])
            tokens.append(token)
        generated.append(tuple(tokens))
    return tuple(generated)


# -- the traced definitions ---------------------------------------------------------


def wires(value: Wire | Wires) -> Wires:
    """A call's result as a range (a one-output call returns a single wire)."""

    if isinstance(value, Wire):
        return Wires(value.trace, value.space, value.index, 1, 0)
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
            or part.stride != 1
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
    w_1: Wires
    w_2: Wires


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
    """

    def __init__(self, shape: LMShape) -> None:
        if not isinstance(shape, LMShape):
            raise TypeError("shape must be an LMShape")
        self.shape = shape
        self.tracer = Tracer(make_isa_gate_set(shape.width))
        gate = self.tracer.gate
        add, mul, sub, lt, eq, shr = (gate(name) for name in ("add", "mul", "sub", "lt", "eq", "shr"))
        self.add, self.mul, self.sub, self.lt, self.shr = add, mul, sub, lt, shr
        define = self.tracer.definition
        self.mul_pair = define(input_count=2, key="mul")(lambda v: mul(v[0], v[1]))
        self.add_pair = define(input_count=2, key="add")(lambda v: add(v[0], v[1]))
        self.eq_pair = define(input_count=2, key="eq")(lambda v: eq(v[0], v[1]))
        self.shr_pair = define(input_count=2, key="shr")(lambda v: shr(v[0], v[1]))
        self.square = define(input_count=1, key="square")(lambda v: mul(v[0], v[0]))
        self.square_cell = define(input_count=1, key="square_cell", role=VERIFICATION)(
            lambda v: mul(v[0], v[0])
        )
        self.add_cell = define(input_count=2, key="add_cell", role=VERIFICATION)(
            lambda v: add(v[0], v[1])
        )

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

        embedding = take(vocab * d)
        layers = tuple(
            _LayerPorts(take(d * d), take(d * d), take(d * d), take(d * d), take(d * hidden), take(hidden * d))
            for _ in range(shape.layers)
        )
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

        @self.tracer.definition(input_count=2 * k, key=("dot", k, role), role=role)
        def dot(v: Wires) -> object:
            x, w = v[:k], v[k:]
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

        @self.tracer.definition(input_count=1 + vocab, key="onehot", role=VERIFICATION)
        def onehot(v: Wires) -> object:
            return self.tracer.repeat(vocab, self.eq_pair, v[0], v[1].by(1))

        return onehot

    def attend_head(self, c: int) -> TracedDefinition:
        """One head over ``c`` positions: ports ``q, k_0..k_{c-1}, v_0..v_{c-1}, shift``.

        ``s_j = q . k_j``, ``w_j = s_j * s_j``, ``out_i = (sum_j w_j v_j[i]) >> shift``.
        """

        if type(c) is not int or c <= 0:
            raise TracerError("attention context must be positive")
        dh = self.shape.d_head
        repeat = self.tracer.repeat

        @self.tracer.definition(input_count=dh + 2 * c * dh + 1, key=("attend_head", c), role=VERIFICATION)
        def attend_head(v: Wires) -> object:
            q, keys, values, shift = v[:dh], v[dh : dh + c * dh], v[dh + c * dh : dh + 2 * c * dh], v[-1]
            scores = repeat(c, self.dot(dh, marked=False), q, keys[0:dh].by(dh))
            weights = repeat(c, self.square, scores[0].by(1))
            mixed = repeat(dh, self.dot(c, marked=False), weights, values[0 : c * dh : dh].by(1))
            return repeat(dh, self.shr_pair, mixed[0].by(1), shift)

        return attend_head

    def argmax(self) -> TracedDefinition:
        """The first maximum of ``vocab`` logits as a token id, by a chain of ``lt`` selects."""

        vocab = self.shape.vocab

        @self.tracer.definition(input_count=2 * vocab, key="argmax", role=VERIFICATION)
        def argmax(v: Wires) -> object:
            logits, constants = v[:vocab], v[vocab:]
            best, index = logits[0], constants[0]
            for k in range(1, vocab):
                better = self.lt(best, logits[k])
                best = self.add(best, self.mul(better, self.sub(logits[k], best)))
                index = self.add(index, self.mul(better, self.sub(constants[k], index)))
            return index

        return argmax

    def sample(self) -> TracedDefinition:
        """A token drawn from ``vocab`` logits with the public word ``r``: see the module docstring.

        Ports: the logits, ``r``, the constant ``1``, ``score_shift`` and
        ``random_bits``.  Every value stays below ``2**width`` by the shape's
        bit budget, so the modular gates compute :func:`sample_token` exactly.
        """

        vocab = self.shape.vocab
        add, mul, lt, shr = self.add, self.mul, self.lt, self.shr

        @self.tracer.definition(input_count=vocab + 4, key="sample", role=VERIFICATION)
        def sample(v: Wires) -> object:
            logits = wires(v[:vocab])
            r, one, score_shift, random_bits = v[vocab], v[vocab + 1], v[vocab + 2], v[vocab + 3]
            weights = []
            for logit in logits:
                score = shr(logit, score_shift)
                weights.append(add(mul(score, score), one))
            cdf = [weights[0]]
            for weight in weights[1:]:
                cdf.append(add(cdf[-1], weight))
            bound = add(shr(mul(r, cdf[-1]), random_bits), one)  # t + 1 with t = (r * total) >> random_bits
            below = [lt(entry, bound) for entry in cdf]  # cdf_j <= t
            index = below[0]
            for flag in below[1:]:
                index = add(index, flag)
            return index

        return sample

    def head(self, logits: Wire | Wires, ports: _WeightPorts, r: Wire | None) -> Wire | Wires:
        """The LM head's decision: the argmax, or a sample with the position's random word."""

        if not self.shape.sampling:
            assert r is None
            return self.argmax()(logits, ports.constants)
        assert r is not None and ports.sampler is not None
        return self.sample()(logits, r, ports.constants[1], ports.sampler)

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

        @self.tracer.definition(input_count=k + k * m, key=("matvec", k, m))
        def matvec(v: Wires) -> object:
            x, w = v[:k], v[k:]
            return self.tracer.repeat(m, self.dot(k), x, w[0 : k * m : m].by(1))

        return matvec

    def embed_row(self) -> TracedDefinition:
        """A token id to its embedding: one-hot against the constant table, then ``E``."""

        vocab, d = self.shape.vocab, self.shape.d_model

        @self.tracer.definition(input_count=1 + vocab + vocab * d, key="embed_row")
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
    ) -> tuple[list[Wires], Wires]:
        """The layers over ``positions`` new positions of one sequence.

        ``x`` holds the new positions' embeddings; ``caches[l]`` the layer's
        cached ``(K, V)`` for the earlier positions, position-major, or
        ``None``.  New position ``p`` attends to the cache and to new positions
        ``0..p``.  Returns the new ``K`` and ``V`` of every layer (position-major
        blocks, the cache entries later steps read) and the final activations.
        """

        shape = self.shape
        d, dh, heads, hidden = shape.d_model, shape.d_head, shape.heads, shape.hidden
        repeat = self.tracer.repeat
        project = self.matvec(d, d)

        def head_slices(blocks: Sequence[Wires], count: int) -> list[Wires]:
            """Head ``0``'s ``dh`` values at each of ``count`` positions, shifting by head."""

            return [block[j * d : j * d + dh].by(dh) for block in blocks for j in range(count)]

        state: list[Wires] = []
        for layer, cache in zip(ports.layers, caches, strict=True):
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
            h = repeat(positions, self.matvec(d, hidden), x[0:d].by(d), layer.w_1)
            h = repeat(positions * hidden, self.square_cell, h[0].by(1))
            m = repeat(positions, self.matvec(hidden, d), h[0:hidden].by(hidden), layer.w_2)
            x = repeat(positions * d, self.add_cell, x[0].by(1), m[0].by(1))
            state += [k, v]
        return state, x

    def prefill(self, n: int) -> TracedDefinition:
        """An ``n``-token prompt: ports are the weights; the tokens are ``in`` gates inside.

        Outputs: per layer ``K`` then ``V`` for the ``n`` positions
        (``state_size(n)`` values, position-major), then the first generated
        token, the head's decision on the last position's logits.  For a
        sampling shape the position's random word is one more ``in`` gate,
        after the prompt tokens.
        """

        if type(n) is not int or n <= 0:
            raise TracerError("prompt length must be positive")
        shape = self.shape
        d, vocab = shape.d_model, shape.vocab

        @self.tracer.definition(input_count=shape.weight_count, key=("prefill", n))
        def prefill(v: Wires) -> object:
            ports = self.ports(v)
            tokens = self.tracer.inputs(n)
            r = self.randomness()
            x = self.tracer.repeat(n, self.embed_row(), tokens[0].by(1), ports.constants, ports.embedding)
            state, x = self.forward(ports, x, n, [None] * shape.layers)
            logits = self.matvec(d, vocab)(x[(n - 1) * d : n * d], ports.unembedding)
            return [*state, self.head(logits, ports, r)]

        return prefill

    def decode(self, c: int) -> TracedDefinition:
        """One token at context ``c``: ports are the weights, the token, then per layer
        the cached ``K`` and ``V`` of the ``c - 1`` earlier positions.

        Outputs: per layer the new ``k`` then ``v`` (``state_size(1)`` values), then
        the next token.  For a sampling shape the position's random word is
        an ``in`` gate inside the step.
        """

        if type(c) is not int or c < 2:
            raise TracerError("a decode step needs at least one cached position")
        shape = self.shape
        d, vocab, weights = shape.d_model, shape.vocab, shape.weight_count
        cached = (c - 1) * d

        @self.tracer.definition(input_count=weights + 1 + shape.layers * 2 * cached, key=("decode", c))
        def decode(v: Wires) -> object:
            ports = self.ports(v)
            token = v[weights]
            caches = []
            for layer in range(shape.layers):
                start = weights + 1 + layer * 2 * cached
                caches.append((v[start : start + cached], v[start + cached : start + 2 * cached]))
            r = self.randomness()
            x = wires(self.embed_row()(token, ports.constants, ports.embedding))
            state, x = self.forward(ports, x, 1, caches)
            logits = self.matvec(d, vocab)(x, ports.unembedding)
            return [*state, self.head(logits, ports, r)]

        return decode

    def weights_unit(self) -> TracedDefinition:
        """The replay unit holding every ``weight`` gate, all declared."""

        return self.tracer.definition(input_count=0, key="weights", role="replay")(
            lambda _v: self.tracer.weights(self.shape.weight_count)
        )


__all__ = [
    "Decoder",
    "LMShape",
    "LayerParameters",
    "Matrix",
    "Parameters",
    "ToyLM",
    "argmax_token",
    "concat",
    "random_parameters",
    "reference_generate",
    "sample_token",
    "wires",
]
