"""GPT-2's structure in the ontology: shape, kinds and marks, one replay unit per request.

Structure only.  Every value is a gate of the *structural* gate set
:func:`veritor.core.make_ml_gate_set` (declared widths and costs, no
executable semantics), so a circuit traced here can be compiled, indexed and
priced -- ``Compile``, ``Index.kinds``, ``Bound``, ``Cost`` -- but never run.
What it keeps is GPT-2's *shape*, gate for gate: token and position
embedding, per layer a LayerNorm, ``q``/``k``/``v`` projections with biases,
causal attention over a KV cache with a max-shifted softmax (``exp``, a
summed denominator, one reciprocal), the output projection, a residual sum,
a second LayerNorm, the ``gelu_new`` MLP (tanh approximation) and a second
residual sum; a final LayerNorm, an LM head tied to the token embedding and
a greedy argmax.  Dot products accumulate at ``acc_width`` and are rounded
to ``width`` by an explicit ``narrow`` gate, LayerNorm and softmax
statistics are ``acc_width`` values, activations, KV-cache entries,
probabilities, residuals and logits are ``width`` values, as in the
``vllm-fp16-reference`` profile of the legacy explicit-DAG analysis
(:mod:`circuit_cut_analysis.models.gpt2_circuit`) this mirrors.

Two departures from that analysis, both forced by the grammar (no
immediates, no gather): the token embedding is a one-hot against a constant
table (``vocab`` equality tests) followed by ``d_model`` dot products of
length ``vocab``, where the legacy DAG has a zero-work lookup; and the argmax
is a tournament of ``vocab - 1`` compare-and-select nodes (three gates each)
where the legacy DAG has one atomic gate.  Two choices of what to trace: the
final LayerNorm is applied only at positions that predict a token (the
legacy DAG normalises every position), and the tied LM head reads the
embedding rows directly.  Model constants (``1/d_model``, ``eps``, the score
scale ``1/sqrt(d_head)``, the GELU constants, ``1`` and ``1/2``, and the
token table ``0, 1, ..., vocab - 1``) are ``weight`` gates: the grammar has
no immediates and constants pinned under the weight commitment are what they
are.

Marks.  Every dot product is a verification unit (VU): ``dot_k`` (one
output of a biased projection or of the LM head, ``k`` products, a sum tree,
the bias, the ``narrow``), ``score_dh`` (one scaled query-key score) and the
value mix (a ``dot_c`` over the ``c`` probabilities).  Every nonlinearity
cell is a VU: the LayerNorm mean and inverse standard deviation
(``ln_mean_d``, ``ln_var_d``, one each per normalised vector), the centred
and the scaled-and-shifted coordinates (``ln_center``, ``ln_out``), the
softmax maximum and reciprocal denominator (``softmax_max_c``,
``softmax_denominator_c``, one each per query and head), the shifted
exponential and the probability (``exp_cell``, ``prob_cell``), the
``gelu_cell``, the residual and embedding ``add_cell``, each one-bit
equality of the one-hot (``eq_cell``); the argmax is one VU; the source
gates are the tracer's one-gate cells.  No VU of the model has an interface
wider than ``acc_width`` (32 bits); the widest is the ``argmax`` at ``width``
bits out of ``vocab`` logits.
The replay units (RUs) are the ``weights`` unit of every ``weight`` gate and
one ``request`` per request -- its prefill and every decode step, wired
through its own KV cache, which never leaves the unit -- exactly as
:class:`~veritor.constructors.requests.RequestsG` does for the toy.  Nothing
marked nests in anything marked.

Kinds are shared as far as the grammar allows: one ``layer`` kind per
``(positions, cached)`` serves every layer (the weights are ports), one
``attend_head`` kind per ``(cached, fresh)`` serves every head and every
layer (its key and value ports are *windows* of the position-major cache
block, one range each, so a call names four ranges whatever the context),
and two requests of the same prompt length and ``max_new`` are one kind.

Weights.  :meth:`GPT2Shape.layout` is the flat order of the ``weight``
gates: ``wte`` (``vocab x d_model``), ``wpe`` (``context x d_model``), per
layer ``ln1_g, ln1_b, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, ln2_g, ln2_b,
w_fc, b_fc, w_proj, b_proj`` (matrices row-major in the ``x @ W``
orientation), ``lnf_g, lnf_b``, the token table, then the seven scalars
``inv_d, eps, scale, gelu_c3, gelu_k, one, half``.
"""

from __future__ import annotations

from dataclasses import dataclass

from veritor.compile import constructor_digest
from veritor.core import Digest, GateSet, JSONValue, make_ml_gate_set
from veritor.core.description import REPLAY, VERIFICATION

from .lm import concat, wires
from .schedule import Request
from .tracer import TracedDefinition, Tracer, TracerError, TracerGate, Wire, Wires

SCALARS = ("inv_d", "eps", "scale", "gelu_c3", "gelu_k", "one", "half")
"""The scalar constants, in the order they follow the token table among the weights."""

LAYER_FIELDS = (
    "ln1_g",
    "ln1_b",
    "w_q",
    "b_q",
    "w_k",
    "b_k",
    "w_v",
    "b_v",
    "w_o",
    "b_o",
    "ln2_g",
    "ln2_b",
    "w_fc",
    "b_fc",
    "w_proj",
    "b_proj",
)
"""One layer's weights in flat order."""


@dataclass(frozen=True, slots=True)
class GPT2Shape:
    """The dimensions of a GPT-2 model and the two widths of its values.

    ``d_head = d_model // heads``; ``context`` is the longest sequence
    (prompt plus generated tokens) a request may occupy; ``width`` is the
    activation width and ``acc_width`` the accumulator width of
    :func:`~veritor.core.make_ml_gate_set`.  Token ids are ``width``-bit
    words, so ``vocab <= 2**width``.
    """

    layers: int
    d_model: int
    heads: int
    d_ff: int
    vocab: int
    context: int
    width: int = 16
    acc_width: int = 32

    def __post_init__(self) -> None:
        for name in ("layers", "d_model", "heads", "d_ff", "vocab", "context", "width", "acc_width"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab < 2:
            raise ValueError("vocab must be at least 2")
        if self.d_model % self.heads:
            raise ValueError("d_model must be a multiple of heads")
        if self.vocab > 1 << self.width:
            raise ValueError("token ids must be words: vocab <= 2**width")
        if self.acc_width < self.width:
            raise ValueError("acc_width must be at least width")

    @classmethod
    def small(cls) -> GPT2Shape:
        """GPT-2 Small: 12 layers, ``d_model`` 768, 12 heads, ``d_ff`` 3072, vocabulary 50257, context 1024."""

        return cls(12, 768, 12, 3072, 50257, 1024)

    @property
    def d_head(self) -> int:
        return self.d_model // self.heads

    @property
    def layer_weights(self) -> int:
        """One layer's ``weight`` gates: four square matrices, the MLP, five bias vectors, two LayerNorms."""

        d, f = self.d_model, self.d_ff
        return 4 * d * d + 2 * d * f + f + 9 * d

    def layer_layout(self) -> tuple[tuple[str, int], ...]:
        d, f = self.d_model, self.d_ff
        sizes = {
            "ln1_g": d,
            "ln1_b": d,
            "w_q": d * d,
            "b_q": d,
            "w_k": d * d,
            "b_k": d,
            "w_v": d * d,
            "b_v": d,
            "w_o": d * d,
            "b_o": d,
            "ln2_g": d,
            "ln2_b": d,
            "w_fc": d * f,
            "b_fc": f,
            "w_proj": f * d,
            "b_proj": d,
        }
        return tuple((name, sizes[name]) for name in LAYER_FIELDS)

    def layout(self) -> tuple[tuple[str, int], ...]:
        """The flat order of the ``weight`` gates as ``(name, count)`` blocks (see the module docstring)."""

        d = self.d_model
        blocks: list[tuple[str, int]] = [("wte", self.vocab * d), ("wpe", self.context * d)]
        for layer in range(self.layers):
            blocks.extend((f"layer{layer}.{name}", count) for name, count in self.layer_layout())
        blocks.extend((("lnf_g", d), ("lnf_b", d), ("tokens", self.vocab)))
        blocks.extend((name, 1) for name in SCALARS)
        return tuple(blocks)

    @property
    def weight_count(self) -> int:
        """The number of ``weight`` gates: every matrix and vector, the token table and the scalars."""

        d = self.d_model
        return (
            self.vocab * d
            + self.context * d
            + self.layers * self.layer_weights
            + 2 * d
            + self.vocab
            + len(SCALARS)
        )

    def state_size(self, positions: int) -> int:
        """The KV-cache entries ``positions`` new positions add: ``k`` and ``v`` per layer."""

        return 2 * self.layers * positions * self.d_model

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "acc_width": self.acc_width,
            "context": self.context,
            "d_ff": self.d_ff,
            "d_model": self.d_model,
            "heads": self.heads,
            "layers": self.layers,
            "vocab": self.vocab,
            "width": self.width,
        }


# -- gate budget --------------------------------------------------------------------


def gate_budget(shape: GPT2Shape, prompt: int, max_new: int) -> dict[str, int]:
    """The computed (non-source) gates of one request by component, in closed form.

    The request processes ``prompt + max_new - 1`` positions (the prompt,
    then one decode forward per further token; the last prompt position
    predicts the first token) and predicts ``max_new`` tokens.  Position
    ``p`` (zero-based) attends over ``c = p + 1`` keys.  Components:

    * ``embedding``: the one-hot (``vocab`` gates), the ``d_model`` dots of
      length ``vocab`` (``2 vocab`` gates each) and the position add;
    * ``layer_norm``: ``7 d_model + 2`` per normalised vector, two per
      layer per position and one per prediction;
    * ``attention``: the four biased projections (``2 d_model + 1`` per
      output), the scores (``2 d_head + 1`` each) and the value mix
      (``2 c`` per output coordinate);
    * ``softmax``: ``5 c - 1`` per query and head;
    * ``mlp``: two biased projections and ``9`` gates per GELU;
    * ``residual``: two adds per coordinate per layer;
    * ``lm_head``: ``vocab`` dots of length ``d_model`` (``2 d_model`` each);
    * ``argmax``: ``3 (vocab - 1)``.
    """

    d, f, dh, heads, vocab, layers = (
        shape.d_model,
        shape.d_ff,
        shape.d_head,
        shape.heads,
        shape.vocab,
        shape.layers,
    )
    positions = prompt + max_new - 1
    keys = positions * (positions + 1) // 2  # sum of c over the processed positions
    budget = {
        "embedding": positions * (vocab + 2 * vocab * d + d),
        "layer_norm": (2 * layers * positions + max_new) * (7 * d + 2),
        "attention": layers * (positions * 4 * d * (2 * d + 1) + keys * (heads * (2 * dh + 1) + 2 * d)),
        "softmax": layers * heads * (5 * keys - positions),
        "mlp": layers * positions * (f * (2 * d + 1) + 9 * f + d * (2 * f + 1)),
        "residual": layers * positions * 2 * d,
        "lm_head": max_new * vocab * 2 * d,
        "argmax": max_new * 3 * (vocab - 1),
    }
    budget["total"] = sum(budget.values())
    return budget


# -- ports ----------------------------------------------------------------------------


def cut(v: Wires, start: int, stop: int, step: int = 1) -> Wires:
    """``v[start:stop:step]`` as a range (a slice of a range is always a range)."""

    piece = v[start:stop:step]
    assert isinstance(piece, Wires)
    return piece


def at(v: Wires, index: int) -> Wire:
    """``v[index]`` as a single wire."""

    wire = v[index]
    assert isinstance(wire, Wire)
    return wire


@dataclass(frozen=True, slots=True)
class _Scalars:
    inv_d: Wire
    eps: Wire
    scale: Wire
    gelu_c3: Wire
    gelu_k: Wire
    one: Wire
    half: Wire


@dataclass(frozen=True, slots=True)
class _LayerPorts:
    ln1_g: Wires
    ln1_b: Wires
    w_q: Wires
    b_q: Wires
    w_k: Wires
    b_k: Wires
    w_v: Wires
    b_v: Wires
    w_o: Wires
    b_o: Wires
    ln2_g: Wires
    ln2_b: Wires
    w_fc: Wires
    b_fc: Wires
    w_proj: Wires
    b_proj: Wires


@dataclass(frozen=True, slots=True)
class _Ports:
    """The weight ports of a definition sliced in the order of :meth:`GPT2Shape.layout`."""

    wte: Wires
    wpe: Wires
    layers: tuple[Wires, ...]
    lnf_g: Wires
    lnf_b: Wires
    tokens: Wires
    consts: Wires
    scalars: _Scalars


class _Cursor:
    def __init__(self, v: Wires) -> None:
        self.v = v
        self.at = 0

    def take(self, count: int) -> Wires:
        piece = cut(self.v, self.at, self.at + count)
        self.at += count
        return piece


# -- the traced definitions ----------------------------------------------------------


class GPT2:
    """The kinds of GPT-2 over one :class:`Tracer`, and their marks (see the module docstring)."""

    def __init__(self, shape: GPT2Shape) -> None:
        if not isinstance(shape, GPT2Shape):
            raise TypeError("shape must be a GPT2Shape")
        self.shape = shape
        self.tracer = Tracer(make_ml_gate_set(shape.width, shape.acc_width))
        gate = self.tracer.gate
        self.add, self.mul, self.narrow = gate("add"), gate("mul"), gate("narrow")
        self.lt, self.eq, self.select = gate("lt"), gate("eq"), gate("select")
        self.acc_add, self.acc_sub, self.acc_mul, self.acc_max = (
            gate(name) for name in ("acc_add", "acc_sub", "acc_mul", "acc_max")
        )
        self.exp, self.recip, self.rsqrt, self.tanh = (gate(name) for name in ("exp", "recip", "rsqrt", "tanh"))
        define = self.tracer.definition
        # unmarked one-gate building blocks (the copies of a ``repeat``)
        self.acc_mul_pair = define(input_count=2, key="acc_mul")(lambda v: self.acc_mul(v[0], v[1]))
        self.acc_add_pair = define(input_count=2, key="acc_add")(lambda v: self.acc_add(v[0], v[1]))
        self.acc_max_pair = define(input_count=2, key="acc_max")(lambda v: self.acc_max(v[0], v[1]))
        self.acc_square = define(input_count=1, key="acc_square")(lambda v: self.acc_mul(v[0], v[0]))
        # one-gate and few-gate verification units
        self.eq_cell = define(input_count=2, key="eq_cell", role=VERIFICATION)(lambda v: self.eq(v[0], v[1]))
        self.add_cell = define(input_count=2, key="add_cell", role=VERIFICATION)(lambda v: self.add(v[0], v[1]))
        self.ln_center = define(input_count=2, key="ln_center", role=VERIFICATION)(
            lambda v: self.acc_sub(v[0], v[1])
        )
        self.ln_out = define(input_count=4, key="ln_out", role=VERIFICATION)(
            lambda v: self.add(self.acc_mul(self.acc_mul(v[0], v[1]), v[2]), v[3])
        )
        self.exp_cell = define(input_count=2, key="exp_cell", role=VERIFICATION)(
            lambda v: self.exp(self.acc_sub(v[0], v[1]))
        )
        self.prob_cell = define(input_count=2, key="prob_cell", role=VERIFICATION)(lambda v: self.mul(v[0], v[1]))
        self.gelu_cell = define(input_count=5, key="gelu_cell", role=VERIFICATION)(self._gelu)

    def kind_names(self) -> dict[str, str]:
        """Kind digest -> the readable key it was traced under (``dot(8,True)``, ``layer(4,0)``).

        Only kinds traced so far are named; the anonymous one-gate copies of
        a ``repeat`` carry their gate name as key.  Two keys can share a
        digest (hash-consing): the first traced wins.
        """

        names: dict[str, str] = {}
        for key, definition in self.tracer._by_key.items():
            if isinstance(key, tuple):
                name = f"{key[0]}({','.join(str(part) for part in key[1:])})"
            else:
                name = str(key)
            names.setdefault(definition.digest, name)
        return names

    # -- ports ---------------------------------------------------------------------------

    def ports(self, v: Wires) -> _Ports:
        """Slice ``v[:weight_count]`` in the order of :meth:`GPT2Shape.layout`."""

        shape = self.shape
        d = shape.d_model
        cursor = _Cursor(v)
        wte, wpe = cursor.take(shape.vocab * d), cursor.take(shape.context * d)
        layers = tuple(cursor.take(shape.layer_weights) for _ in range(shape.layers))
        lnf_g, lnf_b, tokens = cursor.take(d), cursor.take(d), cursor.take(shape.vocab)
        consts = cursor.take(len(SCALARS))
        assert cursor.at == shape.weight_count
        return _Ports(wte, wpe, layers, lnf_g, lnf_b, tokens, consts, _Scalars(*consts))

    def layer_ports(self, lw: Wires) -> _LayerPorts:
        cursor = _Cursor(lw)
        pieces = [cursor.take(count) for _, count in self.shape.layer_layout()]
        assert cursor.at == self.shape.layer_weights
        return _LayerPorts(*pieces)

    # -- reductions ------------------------------------------------------------------------

    def reduce(self, level: Wires, pair: TracedDefinition, combine: TracerGate) -> Wire:
        """Fold ``level`` pairwise with ``pair`` (a two-input cell) level by level; carries at the end.

        ``len(level) - 1`` gates, whatever the length; a single value is
        returned as it is.
        """

        carried: list[Wire] = []
        while len(level) > 1:
            if len(level) % 2:
                carried.append(at(level, -1))
            level = self.tracer.repeat(len(level) // 2, pair, cut(level, 0, 2).by(2))
        result = at(level, 0)
        for carry in carried:
            result = combine(result, carry)
        return result

    # -- verification units ------------------------------------------------------------------

    def dot(self, k: int, *, biased: bool) -> TracedDefinition:
        """``narrow(x . w [+ b])`` over ``k``-vectors: products and a sum tree at ``acc_width``, rounded once."""

        if type(k) is not int or k <= 0:
            raise TracerError("dot length must be positive")

        @self.tracer.definition(input_count=2 * k + int(biased), key=("dot", k, biased), role=VERIFICATION)
        def dot(v: Wires) -> object:
            products = self.tracer.repeat(k, self.acc_mul_pair, v[0].by(1), v[k].by(1))
            acc = self.reduce(products, self.acc_add_pair, self.acc_add)
            if biased:
                acc = self.acc_add(acc, v[2 * k])
            return self.narrow(acc)

        return dot

    def score(self) -> TracedDefinition:
        """One scaled query-key score: ``narrow((q . k) * scale)``."""

        dh = self.shape.d_head

        @self.tracer.definition(input_count=2 * dh + 1, key="score", role=VERIFICATION)
        def score(v: Wires) -> object:
            products = self.tracer.repeat(dh, self.acc_mul_pair, v[0].by(1), v[dh].by(1))
            total = self.reduce(products, self.acc_add_pair, self.acc_add)
            return self.narrow(self.acc_mul(total, v[2 * dh]))

        return score

    def ln_mean(self) -> TracedDefinition:
        """The mean of ``d_model`` values: a sum tree and one multiply by ``1/d_model``."""

        d = self.shape.d_model

        @self.tracer.definition(input_count=d + 1, key="ln_mean", role=VERIFICATION)
        def ln_mean(v: Wires) -> object:
            return self.acc_mul(self.reduce(cut(v, 0, d), self.acc_add_pair, self.acc_add), v[d])

        return ln_mean

    def ln_var(self) -> TracedDefinition:
        """The inverse standard deviation of ``d_model`` centred values: squares, a sum tree, ``1/d``, ``eps``, ``rsqrt``."""

        d = self.shape.d_model

        @self.tracer.definition(input_count=d + 2, key="ln_var", role=VERIFICATION)
        def ln_var(v: Wires) -> object:
            squares = self.tracer.repeat(d, self.acc_square, v[0].by(1))
            variance = self.acc_mul(self.reduce(squares, self.acc_add_pair, self.acc_add), v[d])
            return self.rsqrt(self.acc_add(variance, v[d + 1]))

        return ln_var

    def softmax_max(self, c: int) -> TracedDefinition:
        """The maximum of ``c >= 2`` scores: a tree of ``c - 1`` ``acc_max`` gates."""

        if type(c) is not int or c < 2:
            raise TracerError("a softmax maximum needs at least two scores")

        @self.tracer.definition(input_count=c, key=("softmax_max", c), role=VERIFICATION)
        def softmax_max(v: Wires) -> object:
            return self.reduce(cut(v, 0, c), self.acc_max_pair, self.acc_max)

        return softmax_max

    def softmax_denominator(self, c: int) -> TracedDefinition:
        """The reciprocal of the sum of ``c`` exponentials."""

        if type(c) is not int or c <= 0:
            raise TracerError("a softmax denominator needs at least one exponential")

        @self.tracer.definition(input_count=c, key=("softmax_denominator", c), role=VERIFICATION)
        def softmax_denominator(v: Wires) -> object:
            return self.recip(self.reduce(cut(v, 0, c), self.acc_add_pair, self.acc_add))

        return softmax_denominator

    def _gelu(self, v: Wires) -> object:
        """``gelu_new``: ``0.5 x (1 + tanh(k (x + c3 x^3)))``, ports ``x, c3, k, one, half``."""

        x, c3, k, one, half = v[0], v[1], v[2], v[3], v[4]
        x2 = self.acc_mul(x, x)
        x3 = self.acc_mul(x2, x)
        inner = self.acc_add(x, self.acc_mul(x3, c3))
        gate = self.acc_add(self.tanh(self.acc_mul(inner, k)), one)
        return self.mul(self.acc_mul(x, gate), half)

    def onehot(self) -> TracedDefinition:
        """``eq(t, k)`` for every token ``k`` of the constant table: ``vocab`` one-bit cells.

        Unmarked: each comparison is its own one-bit ``eq_cell`` verification
        unit, so no verification unit of the model has an interface wider
        than ``acc_width`` (a marked ``onehot`` would have a ``vocab``-bit one).
        """

        vocab = self.shape.vocab

        @self.tracer.definition(input_count=1 + vocab, key="onehot")
        def onehot(v: Wires) -> object:
            return self.tracer.repeat(vocab, self.eq_cell, v[0], v[1].by(1))

        return onehot

    def argmax2(self) -> TracedDefinition:
        """One tournament node: ports ``l_a, l_b, i_a, i_b``; outputs the larger logit and its index.

        ``lt(l_a, l_b)`` selects ``b`` only when it is strictly larger, so
        ties keep the earlier index.
        """

        @self.tracer.definition(input_count=4, key="argmax2")
        def argmax2(v: Wires) -> object:
            better = self.lt(v[0], v[1])
            return [self.select(better, v[0], v[1]), self.select(better, v[2], v[3])]

        return argmax2

    def argmax(self) -> TracedDefinition:
        """The first maximum of ``vocab`` logits as a token id: a tournament of ``vocab - 1`` nodes.

        Ports: the logits, then the token table.  Each level pairs
        consecutive ``(logit, index)`` entries with one ``repeat``; an odd
        entry is carried and folded in at the end, so the description is
        ``O(log vocab)`` for ``3 (vocab - 1)`` gates.
        """

        vocab = self.shape.vocab
        node = self.argmax2()

        @self.tracer.definition(input_count=2 * vocab, key="argmax", role=VERIFICATION)
        def argmax(v: Wires) -> object:
            logits, table = cut(v, 0, vocab), cut(v, vocab, 2 * vocab)
            carried: list[tuple[Wire, Wire]] = []
            if vocab % 2:
                carried.append((at(logits, -1), at(table, -1)))
            level = self.tracer.repeat(vocab // 2, node, cut(logits, 0, 2).by(2), cut(table, 0, 2).by(2))
            while len(level) > 2:
                pairs = len(level) // 2
                if pairs % 2:
                    carried.append((at(level, -2), at(level, -1)))
                level = self.tracer.repeat(pairs // 2, node, cut(level, 0, 4, 2).by(4), cut(level, 1, 4, 2).by(4))
            best, index = at(level, 0), at(level, 1)
            for logit, token in carried:
                folded = wires(node(best, logit, index, token))
                best, index = at(folded, 0), at(folded, 1)
            return index

        return argmax

    # -- unmarked composites -----------------------------------------------------------------

    def matvec(self, k: int, m: int, *, biased: bool, rows: bool = False) -> TracedDefinition:
        """``m`` dots of length ``k``: ``x W [+ b]`` for a row-major ``k x m`` matrix, or against the rows of an ``m x k`` one.

        Ports: ``x`` (``k``), the matrix (``k m``), then ``b`` (``m``) when
        biased.
        """

        @self.tracer.definition(input_count=k + k * m + (m if biased else 0), key=("matvec", k, m, biased, rows))
        def matvec(v: Wires) -> object:
            x, w = cut(v, 0, k), cut(v, k, k + k * m)
            column = cut(w, 0, k).by(k) if rows else cut(w, 0, k * m, m).by(1)
            if biased:
                return self.tracer.repeat(m, self.dot(k, biased=True), x, column, v[k + k * m].by(1))
            return self.tracer.repeat(m, self.dot(k, biased=False), x, column)

        return matvec

    def layer_norm(self) -> TracedDefinition:
        """LayerNorm of ``d_model`` values: ports ``x, gamma, beta, inv_d, eps``; four kinds of VU inside."""

        d = self.shape.d_model
        mean, var = self.ln_mean(), self.ln_var()

        @self.tracer.definition(input_count=3 * d + 2, key="layer_norm")
        def layer_norm(v: Wires) -> object:
            x, inv_d, eps = cut(v, 0, d), v[3 * d], v[3 * d + 1]
            mu = mean(x, inv_d)
            centred = self.tracer.repeat(d, self.ln_center, v[0].by(1), mu)
            inv_std = var(centred, inv_d, eps)
            return self.tracer.repeat(d, self.ln_out, centred[0].by(1), inv_std, v[d].by(1), v[2 * d].by(1))

        return layer_norm

    def embed(self) -> TracedDefinition:
        """A token id and a position row to the residual stream: one-hot, ``E``, plus the position embedding.

        Ports: the token, the token table, ``wte``, the position's row of ``wpe``.
        """

        vocab, d = self.shape.vocab, self.shape.d_model
        onehot, project = self.onehot(), self.matvec(vocab, d, biased=False)

        @self.tracer.definition(input_count=1 + vocab + vocab * d + d, key="embed")
        def embed(v: Wires) -> object:
            table, wte = cut(v, 1, 1 + vocab), cut(v, 1 + vocab, 1 + vocab + vocab * d)
            e = wires(project(onehot(v[0], table), wte))
            return self.tracer.repeat(d, self.add_cell, e[0].by(1), v[1 + vocab + vocab * d].by(1))

        return embed

    def attend_head(self, cached: int, fresh: int) -> TracedDefinition:
        """One head over ``cached`` cache positions and ``fresh`` new ones (``c = cached + fresh``).

        Ports: ``q`` (``d_head``); a *window* of the position-major cache
        block ``K`` starting at this head's first coordinate and ending at
        its last one at the last cached position (``(cached - 1) d_model +
        d_head`` values, absent when ``cached == 0``); the same window of
        the new keys; the two windows of the values; the score scale.  A
        window is one range, so key ``j`` of the head is ``window[j
        d_model : j d_model + d_head]`` and a ``repeat`` over positions
        shifts by ``d_model``.  Scores, the maximum (when ``c >= 2``), the
        shifted exponentials, the reciprocal denominator, the probabilities
        and the ``d_head`` value mixes are the VUs inside.
        """

        if type(cached) is not int or cached < 0 or type(fresh) is not int or fresh < 1:
            raise TracerError("attention needs a nonnegative cache and at least one new position")
        shape = self.shape
        d, dh = shape.d_model, shape.d_head
        c = cached + fresh
        cache_window = (cached - 1) * d + dh if cached else 0
        fresh_window = (fresh - 1) * d + dh
        repeat = self.tracer.repeat
        score, mix = self.score(), self.dot(c, biased=False)
        maximum = self.softmax_max(c) if c >= 2 else None
        denominator = self.softmax_denominator(c)

        @self.tracer.definition(
            input_count=dh + 2 * (cache_window + fresh_window) + 1, key=("attend_head", cached, fresh)
        )
        def attend_head(v: Wires) -> object:
            cursor = _Cursor(v)
            q = cursor.take(dh)
            k_cache = cursor.take(cache_window) if cached else None
            k_new = cursor.take(fresh_window)
            v_cache = cursor.take(cache_window) if cached else None
            v_new = cursor.take(fresh_window)
            scale = v[-1]
            parts: list[Wires] = []
            if k_cache is not None:
                parts.append(repeat(cached, score, q, cut(k_cache, 0, dh).by(d), scale))
            parts.append(repeat(fresh, score, q, cut(k_new, 0, dh).by(d), scale))
            scores = concat(parts)
            m = scores[0] if maximum is None else maximum(scores)
            e = repeat(c, self.exp_cell, scores[0].by(1), m)
            r = denominator(e)
            p = repeat(c, self.prob_cell, e[0].by(1), r)
            values: list[Wire | Wires] = [p]
            if v_cache is not None:
                values.append(cut(v_cache, 0, (cached - 1) * d + 1, d).by(1))
            values.append(cut(v_new, 0, (fresh - 1) * d + 1, d).by(1))
            return repeat(dh, mix, *values)

        return attend_head

    def layer(self, positions: int, cached: int) -> TracedDefinition:
        """One transformer block over ``positions`` new positions attending to ``cached`` earlier ones.

        Ports: the layer's weights (one block in the order of
        :data:`LAYER_FIELDS`), ``x`` (``positions x d_model``, position-major),
        the cached ``K`` and ``V`` (``cached x d_model`` each, absent when
        ``cached == 0``), the seven scalars.  Outputs: the new ``k`` and
        ``v`` (``positions x d_model`` each, the cache entries later steps
        read) and the new residual stream.  One kind for every layer.
        """

        if type(positions) is not int or positions < 1 or type(cached) is not int or cached < 0:
            raise TracerError("a layer needs at least one new position and a nonnegative cache")
        shape = self.shape
        d, dh, heads, f = shape.d_model, shape.d_head, shape.heads, shape.d_ff
        repeat = self.tracer.repeat
        norm, project = self.layer_norm(), self.matvec(d, d, biased=True)
        up, down = self.matvec(d, f, biased=True), self.matvec(f, d, biased=True)
        heads_at = [self.attend_head(cached, p + 1) for p in range(positions)]
        cache = cached * d

        @self.tracer.definition(
            input_count=shape.layer_weights + positions * d + 2 * cache + len(SCALARS),
            key=("layer", positions, cached),
        )
        def layer(v: Wires) -> object:
            cursor = _Cursor(v)
            w = self.layer_ports(cursor.take(shape.layer_weights))
            x = cursor.take(positions * d)
            k_cache = cursor.take(cache) if cached else None
            v_cache = cursor.take(cache) if cached else None
            s = _Scalars(*cursor.take(len(SCALARS)))
            h = repeat(positions, norm, cut(x, 0, d).by(d), w.ln1_g, w.ln1_b, s.inv_d, s.eps)
            q = repeat(positions, project, cut(h, 0, d).by(d), w.w_q, w.b_q)
            k = repeat(positions, project, cut(h, 0, d).by(d), w.w_k, w.b_k)
            vv = repeat(positions, project, cut(h, 0, d).by(d), w.w_v, w.b_v)
            attended: list[Wires] = []
            for p in range(positions):
                args: list[Wire | Wires] = [cut(q, p * d, p * d + dh).by(dh)]
                if k_cache is not None:
                    args.append(cut(k_cache, 0, (cached - 1) * d + dh).by(dh))
                args.append(cut(k, 0, p * d + dh).by(dh))
                if v_cache is not None:
                    args.append(cut(v_cache, 0, (cached - 1) * d + dh).by(dh))
                args.append(cut(vv, 0, p * d + dh).by(dh))
                args.append(s.scale)
                attended.append(repeat(heads, heads_at[p], *args))
            o = repeat(positions, project, cut(concat(attended), 0, d).by(d), w.w_o, w.b_o)
            x1 = repeat(positions * d, self.add_cell, x[0].by(1), o[0].by(1))
            h2 = repeat(positions, norm, cut(x1, 0, d).by(d), w.ln2_g, w.ln2_b, s.inv_d, s.eps)
            hidden = repeat(positions, up, cut(h2, 0, d).by(d), w.w_fc, w.b_fc)
            act = repeat(positions * f, self.gelu_cell, hidden[0].by(1), s.gelu_c3, s.gelu_k, s.one, s.half)
            m = repeat(positions, down, cut(act, 0, f).by(f), w.w_proj, w.b_proj)
            x2 = repeat(positions * d, self.add_cell, x1[0].by(1), m[0].by(1))
            return [k, vv, x2]

        return layer

    def head(self, ports: _Ports, x: Wires) -> Wire:
        """The final LayerNorm, the tied LM head and the argmax of one position's residual stream."""

        shape = self.shape
        h = self.layer_norm()(x, ports.lnf_g, ports.lnf_b, ports.scalars.inv_d, ports.scalars.eps)
        logits = self.matvec(shape.d_model, shape.vocab, biased=False, rows=True)(wires(h), ports.wte)
        token = self.argmax()(wires(logits), ports.tokens)
        assert isinstance(token, Wire)
        return token

    def prefill(self, n: int) -> TracedDefinition:
        """An ``n``-token prompt: ports are the weights; the tokens are ``in`` gates inside.

        Outputs: per layer ``K`` then ``V`` for the ``n`` positions
        (``state_size(n)`` values, position-major), then the first generated
        token, predicted at the last position.
        """

        if type(n) is not int or n <= 0:
            raise TracerError("prompt length must be positive")
        shape = self.shape
        d = shape.d_model
        embed, layer = self.embed(), self.layer(n, 0)

        @self.tracer.definition(input_count=shape.weight_count, key=("prefill", n))
        def prefill(v: Wires) -> object:
            ports = self.ports(v)
            tokens = self.tracer.inputs(n)
            positions = cut(ports.wpe, 0, d).by(d)
            x = self.tracer.repeat(n, embed, tokens[0].by(1), ports.tokens, ports.wte, positions)
            state: list[Wires] = []
            for block in ports.layers:
                out = wires(layer(block, x, ports.consts))
                state += [cut(out, 0, n * d), cut(out, n * d, 2 * n * d)]
                x = cut(out, 2 * n * d, 3 * n * d)
            return [*state, self.head(ports, cut(x, (n - 1) * d, n * d))]

        return prefill

    def decode(self, c: int) -> TracedDefinition:
        """One token at context ``c``: ports are the weights, the token, then per layer the cached ``K`` and ``V``.

        Outputs: per layer the new ``k`` then ``v`` (``state_size(1)``
        values), then the next token.
        """

        if type(c) is not int or c < 2:
            raise TracerError("a decode step needs at least one cached position")
        shape = self.shape
        d, weights, cached = shape.d_model, shape.weight_count, c - 1
        embed, layer = self.embed(), self.layer(1, cached)
        cache = cached * d

        @self.tracer.definition(input_count=weights + 1 + shape.layers * 2 * cache, key=("decode", c))
        def decode(v: Wires) -> object:
            ports = self.ports(v)
            position = cut(ports.wpe, cached * d, c * d)
            x = wires(embed(v[weights], ports.tokens, ports.wte, position))
            state: list[Wires] = []
            for index, block in enumerate(ports.layers):
                start = weights + 1 + index * 2 * cache
                k_cache, v_cache = cut(v, start, start + cache), cut(v, start + cache, start + 2 * cache)
                out = wires(layer(block, x, k_cache, v_cache, ports.consts))
                state += [cut(out, 0, d), cut(out, d, 2 * d)]
                x = cut(out, 2 * d, 3 * d)
            return [*state, self.head(ports, x)]

        return decode

    def weights_unit(self) -> TracedDefinition:
        """The replay unit holding every ``weight`` gate, all declared."""

        return self.tracer.definition(input_count=0, key="weights", role=REPLAY)(
            lambda _v: self.tracer.weights(self.shape.weight_count)
        )

    def request(self, prompt: int, max_new: int) -> TracedDefinition:
        """One request: its prefill, then a decode step per further token, over its own cache.

        The replay unit.  Ports: the weights.  Outputs: the ``max_new``
        generated tokens.
        """

        if type(prompt) is not int or prompt < 1 or type(max_new) is not int or max_new < 1:
            raise TracerError("a request needs a nonempty prompt and at least one token")
        shape = self.shape
        layers, d = shape.layers, shape.d_model
        prefill = self.prefill(prompt)
        decodes = [self.decode(prompt + step) for step in range(1, max_new)]

        @self.tracer.definition(input_count=shape.weight_count, key=("request", prompt, max_new), role=REPLAY)
        def request(w: Wires) -> object:
            keys: list[list[Wires]] = [[] for _ in range(layers)]
            values: list[list[Wires]] = [[] for _ in range(layers)]

            def remember(block: Wires, positions: int) -> Wire:
                for layer in range(layers):
                    start = 2 * layer * positions * d
                    keys[layer].append(cut(block, start, start + positions * d))
                    values[layer].append(cut(block, start + positions * d, start + 2 * positions * d))
                return at(block, -1)

            token = remember(wires(prefill(w)), prompt)
            tokens = [token]
            for decode in decodes:
                args: list[Wire | Wires] = [w, token]
                for layer in range(layers):
                    args.extend(keys[layer])
                    args.extend(values[layer])
                token = remember(wires(decode(*args)), 1)
                tokens.append(token)
            return tokens

        return request


# -- the constructor ------------------------------------------------------------------------


class GPT2G:
    """GPT-2 serving each request of ``x`` in its own replay unit; no advice.

    ``x`` is a nonempty tuple of :class:`~veritor.constructors.schedule.Request`;
    the ``in`` gates are the prompt tokens, request by request; the circuit
    outputs are the generated tokens in the same order
    (:meth:`output_layout`).  Structure only: the circuit cannot be
    evaluated.
    """

    VERSION = "1"

    def __init__(self, shape: GPT2Shape) -> None:
        if not isinstance(shape, GPT2Shape):
            raise TypeError("shape must be a GPT2Shape")
        self.shape = shape
        self.model = GPT2(shape)
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"shape": self.shape.manifest}

    @property
    def gate_set(self) -> GateSet:
        """The structural gate set the description is written over."""

        return self.model.tracer.gate_set

    # -- validation -------------------------------------------------------------------------

    def requests(self, x: object) -> tuple[Request, ...]:
        if type(x) is not tuple or not x or any(type(item) is not Request for item in x):
            raise TracerError("GPT2G expects a nonempty tuple of Request")
        for index, request in enumerate(x):
            if any(token >= self.shape.vocab for token in request.prompt):
                raise TracerError(f"request {index} has a prompt token outside the vocabulary")
            if len(request.prompt) + request.max_new > self.shape.context:
                raise TracerError(
                    f"request {index} needs {len(request.prompt) + request.max_new} positions; "
                    f"the context is {self.shape.context}"
                )
        return x

    # -- layouts -----------------------------------------------------------------------------

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order."""

        return tuple((r, g) for r, request in enumerate(self.requests(x)) for g in range(request.max_new))

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """The prompt tokens in ``in``-gate address order: request by request."""

        return tuple(token for request in self.requests(x) for token in request.prompt)

    def gate_budget(self, x: object) -> dict[str, int]:
        """:func:`gate_budget` summed over the requests of ``x``."""

        total: dict[str, int] = {}
        for request in self.requests(x):
            for component, count in gate_budget(self.shape, len(request.prompt), request.max_new).items():
                total[component] = total.get(component, 0) + count
        return total

    # -- the run --------------------------------------------------------------------------------

    def root(self, requests: tuple[Request, ...]) -> TracedDefinition:
        """The run: the weights unit, then every request in order, consecutive requests of one shape as a ``repeat``.

        A ``repeat`` lays the copies' tokens out as one run of the root's
        outputs (request-major, as :meth:`output_layout` says) where one
        call per request would spend an output run each; a run of many
        identical requests stays within ``max_output_runs``.
        """

        model = self.model
        groups: list[tuple[TracedDefinition, int]] = []
        for request in requests:
            unit = model.request(len(request.prompt), request.max_new)
            if groups and groups[-1][0] is unit:
                groups[-1] = (unit, groups[-1][1] + 1)
            else:
                groups.append((unit, 1))

        @model.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w = wires(model.weights_unit()())
            return [unit(w) if count == 1 else model.tracer.repeat(count, unit, w) for unit, count in groups]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        if a:
            raise TracerError("GPT2G takes no advice")
        requests = self.requests(x)
        return self.model.tracer.serialize(self.root(requests)), self.flatten_inputs(requests)


__all__ = ["GPT2", "GPT2G", "LAYER_FIELDS", "SCALARS", "GPT2Shape", "at", "cut", "gate_budget"]
