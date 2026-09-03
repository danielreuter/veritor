"""GPT-2 as a circuit of *pinned* gates: one replay unit per request, every value re-executable.

Every gate is one of :func:`veritor.core.make_pinned_gate_set`: a tensor-core
step (``tc_dot16``, Ada's ``mma.sync m16n8k16`` reproduced bit-exactly by
:mod:`veritor.core.silicon`) or an explicit IEEE binary32 operation sequence
on the CUDA cores.  A circuit traced here is therefore not only compiled,
indexed and priced but *run*: ``Circuit.evaluate`` on it is the model, and a
GPU that executes the same fixed-order kernels (``gpu/gpt2/``) produces the
same words at every address.  ``docs/gpt2-silicon.md`` is the report.

Dataflow (HF ``gpt2`` semantics up to the pinned numerics).  Weights are the
fp32 checkpoint rounded once to BF16, ``weight`` gates.  The residual stream
is fp32.  Every matmul takes BF16 operands and accumulates in fp32 through a
``K/16``-step ``tc_dot16`` chain (``K = 768``: 48 steps; the MLP down
projection: 192; a score: ``d_head / 16 = 4``; the value mix: ``ceil(c/16)``
over the ``c`` attended keys, zero-padded) whose first step is either
``tc_dot16`` with the widened BF16 bias as the incoming accumulator (the
biased projections) or ``tc_dot16_0`` (scores, the mix, the tied LM head, the
one-hot embedding).  Rounding to BF16 (``f32_to_bf16``, nearest-even) happens
exactly where a value becomes a matmul operand: the LayerNorm outputs, ``q``,
``k``, ``v``, the softmax probabilities, the attention mix and the GELU
output.  LayerNorm statistics, softmax and GELU are fp32: fixed pairwise
reduction trees (:meth:`GPT2.reduce`), ``mean = sum / n``, ``var = sum(c^2)
/ n``, ``rstd = 1 / sqrt(var + 1e-5)`` (``ln_rstd``), ``y = bf16((c * rstd)
* g + b)``; scores ``* 1/sqrt(d_head)`` (``f32_mul`` by the BF16 constant
``0.125``, exact), max-shifted ``f32_exp``, ``p = bf16(e / sum)``;
``gelu_tanh`` is the ``gelu_new`` sequence.  The embedding is a one-hot
(``token_eq`` against the token table, BF16 ``1.0``/``0``) times ``wte`` as
a ``tc_dot16_0`` chain -- provably the gather -- plus the widened position row
in fp32.  Logits stay fp32; the argmax is a tournament of ``f32_max`` /
``argmax_select`` nodes over blocks of ``argmax_block`` logits, ties to the
earlier token.  Model constants that are exact in BF16 (``n = d_model``,
``scale = 0.125``, ``zero``) and the token table are ``weight`` gates; the
LayerNorm ``eps`` and the GELU constants are part of the gate semantics.

Marks.  Every dot product is a verification unit (VU): ``dot(k, biased,
rounded)`` (the widened bias, the ``k/16`` steps, the rounding).  Every
nonlinearity cell is a VU: ``ln_mean``, ``ln_var`` (the reductions), the
centring ``sub_cell`` and the ``ln_out`` cell, ``softmax_max``,
``softmax_sum``, the ``scale_cell``, ``exp_cell``, ``prob_cell``, the
``gelu_cell``, the residual and embedding ``add_cell``, the ``widen_cell``,
each ``eq_cell`` of the one-hot, the argmax blocks and their top tournament;
the source gates are the tracer's one-gate cells.  No VU has an interface
wider than 32 bits except the argmax blocks (``32 + 16`` out).  The replay
units (RUs) are the ``weights`` unit and one ``request`` per request (its
prefill and decode steps over its own KV cache), as before.

Weights.  :meth:`GPT2Shape.layout` is the flat order of the ``weight``
gates: ``wte`` (``vocab x d_model``), ``wpe`` (``context x d_model``), per
layer ``ln1_g, ln1_b, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, ln2_g, ln2_b,
w_fc, b_fc, w_proj, b_proj`` (matrices row-major in the ``x @ W``
orientation), ``lnf_g, lnf_b``, the token table ``0 .. vocab - 1`` (16-bit
integers), then the BF16 scalars ``n, scale, zero``.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from veritor.compile import constructor_digest
from veritor.core import Digest, GateSet, JSONValue, make_pinned_gate_set
from veritor.core.description import REPLAY, VERIFICATION

from .lm import concat, wires
from .schedule import Request
from .tracer import TracedDefinition, Tracer, TracerError, TracerGate, Wire, Wires

STEP = 16
"""Operand pairs per tensor-core step: every chain length is a multiple of it."""

SCALARS = ("n", "scale", "zero")
"""The BF16 constants, in the order they follow the token table among the weights."""

SCALAR_WORDS = {"n": None, "scale": 0x3E00, "zero": 0x0000}
"""The BF16 words of the scalars (``n`` is ``float(d_model)``, see :meth:`GPT2Shape.scalar_words`)."""

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


def padded(k: int) -> int:
    """``k`` rounded up to a multiple of :data:`STEP`."""

    return -(-k // STEP) * STEP


def bf16_word(value: float) -> int:
    """The BF16 word of a float that BF16 represents exactly (a constant of the model)."""

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0xFFFF:
        raise ValueError(f"{value!r} is not exact in BF16")
    return bits >> 16


@dataclass(frozen=True, slots=True)
class GPT2Shape:
    """The dimensions of a GPT-2 model.

    ``d_head = d_model // heads``; ``context`` is the longest sequence
    (prompt plus generated tokens) a request may occupy; ``argmax_block`` is
    the number of logits one argmax block tournament covers.  ``d_model``,
    ``d_head`` and ``d_ff`` are chain lengths and must be multiples of 16;
    token ids are 16-bit words, so ``vocab <= 2**16``.
    """

    layers: int
    d_model: int
    heads: int
    d_ff: int
    vocab: int
    context: int
    argmax_block: int = 64

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "d_model",
            "heads",
            "d_ff",
            "vocab",
            "context",
            "argmax_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab < 2 or self.argmax_block < 2:
            raise ValueError("vocab and argmax_block must be at least 2")
        if self.d_model % self.heads:
            raise ValueError("d_model must be a multiple of heads")
        if self.d_model % STEP or self.d_head % STEP or self.d_ff % STEP:
            raise ValueError(f"d_model, d_head and d_ff must be multiples of {STEP}")
        if self.vocab > 1 << self.width:
            raise ValueError("token ids must be words: vocab <= 2**16")
        bf16_word(
            float(self.d_model)
        )  # ``n`` is a BF16 weight: d_model must be exact in it

    @classmethod
    def small(cls) -> GPT2Shape:
        """GPT-2 Small: 12 layers, ``d_model`` 768, 12 heads, ``d_ff`` 3072, vocabulary 50257, context 1024."""

        return cls(12, 768, 12, 3072, 50257, 1024)

    @property
    def width(self) -> int:
        """The word width of tokens, BF16 activations and weights."""

        return 16

    @property
    def acc_width(self) -> int:
        """The width of fp32 words: accumulators, residuals, statistics, logits."""

        return 32

    @property
    def d_head(self) -> int:
        return self.d_model // self.heads

    @property
    def vocab_padded(self) -> int:
        """The one-hot chain length: ``vocab`` rounded up to a multiple of 16."""

        return padded(self.vocab)

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
        blocks: list[tuple[str, int]] = [
            ("wte", self.vocab * d),
            ("wpe", self.context * d),
        ]
        for layer in range(self.layers):
            blocks.extend(
                (f"layer{layer}.{name}", count) for name, count in self.layer_layout()
            )
        blocks.extend((("lnf_g", d), ("lnf_b", d), ("tokens", self.vocab)))
        blocks.extend((name, 1) for name in SCALARS)
        return tuple(blocks)

    def scalar_words(self) -> dict[str, int]:
        """The 16-bit words of the scalar weights: ``n`` (``float(d_model)``), ``scale`` (``0.125``), ``zero``."""

        words = dict(SCALAR_WORDS)
        words["n"] = bf16_word(float(self.d_model))
        return {name: int(words[name]) for name in SCALARS}  # type: ignore[arg-type]

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
            "argmax_block": self.argmax_block,
            "context": self.context,
            "d_ff": self.d_ff,
            "d_model": self.d_model,
            "heads": self.heads,
            "layers": self.layers,
            "vocab": self.vocab,
        }


# -- gate budget --------------------------------------------------------------------


def gate_budget(shape: GPT2Shape, prompt: int, max_new: int) -> dict[str, int]:
    """The computed (non-source) gates of one request by component, in closed form.

    The request processes ``prompt + max_new - 1`` positions (the prompt,
    then one decode forward per further token) in ``max_new`` forwards and
    predicts ``max_new`` tokens.  Position ``p`` (zero-based) attends over
    ``c = p + 1`` keys.  Components:

    * ``embedding``: the one-hot (``vocab`` gates), ``d_model`` chains of
      ``vocab_padded / 16`` steps, the widened position row and the add;
    * ``constants``: the two widened scalars of every forward;
    * ``layer_norm``: ``10 d_model + 1`` per normalised vector, two per
      layer per position and one per prediction;
    * ``attention``: the biased ``q``/``k``/``v`` projections (``d/16 + 2``
      per output: bias, chain, rounding) and the output projection (``d/16
      + 1``); per query and head the scores (``d_head/16`` each), the
      scaling, the max tree, the shifted exponentials (2 each), the sum tree,
      the probabilities (2 each) and the mix (``ceil(c/16) + 1`` per
      coordinate);
    * ``mlp``: the up projection (``d/16 + 1`` per output), the GELU cells
      (2 each) and the down projection (``d_ff/16 + 1``);
    * ``residual``: two adds per coordinate per layer;
    * ``lm_head``: ``vocab`` chains of ``d/16`` steps per prediction;
    * ``argmax``: ``2 (vocab - 1)`` per prediction, whatever the blocking.
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
    attention_per_layer = positions * (3 * d * (d // STEP + 2) + d * (d // STEP + 1))
    for p in range(positions):
        c = p + 1
        tree = 2 * (c - 1)  # the max tree and the sum tree
        attention_per_layer += heads * (
            c * (dh // STEP) + c + tree + 2 * c + 2 * c + dh * (padded(c) // STEP + 1)
        )
    budget = {
        "embedding": positions * (vocab + d * (shape.vocab_padded // STEP) + 2 * d),
        "constants": 2 * max_new,
        "layer_norm": (2 * layers * positions + max_new) * (10 * d + 1),
        "attention": layers * attention_per_layer,
        "mlp": layers * positions * (f * (d // STEP + 1) + 2 * f + d * (f // STEP + 1)),
        "residual": layers * positions * 2 * d,
        "lm_head": max_new * vocab * (d // STEP),
        "argmax": max_new * 2 * (vocab - 1),
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


def broadcast(wire: Wire, count: int) -> Wires:
    """The same wire ``count`` times: a stride-``0`` range (zero padding of a chain)."""

    if count < 1:
        raise TracerError("a broadcast needs at least one copy")
    return Wires(wire.trace, wire.space, wire.index, count, 0)


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
    n: Wire
    scale: Wire
    zero: Wire


@dataclass(frozen=True, slots=True)
class _Consts:
    """The per-forward constants a layer reads: the widened ``n`` and ``scale``, the raw ``zero`` word."""

    n32: Wire
    scale32: Wire
    zero: Wire

    def wires(self) -> tuple[Wire, Wire, Wire]:
        return (self.n32, self.scale32, self.zero)


class _Cursor:
    def __init__(self, v: Wires) -> None:
        self.v = v
        self.at = 0

    def take(self, count: int) -> Wires:
        piece = cut(self.v, self.at, self.at + count)
        self.at += count
        return piece

    def one(self) -> Wire:
        wire = at(self.v, self.at)
        self.at += 1
        return wire


# -- the traced definitions ----------------------------------------------------------


class GPT2:
    """The kinds of GPT-2 over one :class:`Tracer`, and their marks (see the module docstring)."""

    def __init__(self, shape: GPT2Shape, gate_set: GateSet | None = None) -> None:
        if not isinstance(shape, GPT2Shape):
            raise TypeError("shape must be a GPT2Shape")
        self.shape = shape
        self.tracer = Tracer(make_pinned_gate_set() if gate_set is None else gate_set)
        gate = self.tracer.gate
        self.tc, self.tc0 = gate("tc_dot16"), gate("tc_dot16_0")
        self.widen, self.round = gate("bf16_to_f32"), gate("f32_to_bf16")
        self.add, self.sub, self.mul, self.div, self.max = (
            gate(name)
            for name in ("f32_add", "f32_sub", "f32_mul", "f32_div", "f32_max")
        )
        self.exp, self.gelu, self.rstd = (
            gate("f32_exp"),
            gate("gelu_tanh"),
            gate("ln_rstd"),
        )
        self.select, self.eq = gate("argmax_select"), gate("token_eq")
        define = self.tracer.definition
        # unmarked one- and two-gate building blocks (the copies of a ``repeat``)
        self.add_pair = define(input_count=2, key="add_pair")(
            lambda v: self.add(v[0], v[1])
        )
        self.max_pair = define(input_count=2, key="max_pair")(
            lambda v: self.max(v[0], v[1])
        )
        self.square = define(input_count=1, key="square")(
            lambda v: self.mul(v[0], v[0])
        )
        self.node = define(input_count=4, key="argmax_node")(
            lambda v: [self.max(v[0], v[1]), self.select(v[0], v[1], v[2], v[3])]
        )

        # one- and two-gate verification units
        def cell(
            name: str, n: int
        ) -> Callable[[Callable[[Wires], object]], TracedDefinition]:
            return define(input_count=n, key=name, role=VERIFICATION)

        self.widen_cell = cell("widen_cell", 1)(lambda v: self.widen(v[0]))
        self.add_cell = cell("add_cell", 2)(lambda v: self.add(v[0], v[1]))
        self.eq_cell = cell("eq_cell", 2)(lambda v: self.eq(v[0], v[1]))
        self.sub_cell = cell("sub_cell", 2)(lambda v: self.sub(v[0], v[1]))
        self.scale_cell = cell("scale_cell", 2)(lambda v: self.mul(v[0], v[1]))
        self.exp_cell = cell("exp_cell", 2)(lambda v: self.exp(self.sub(v[0], v[1])))
        self.prob_cell = cell("prob_cell", 2)(
            lambda v: self.round(self.div(v[0], v[1]))
        )
        self.gelu_cell = cell("gelu_cell", 1)(lambda v: self.round(self.gelu(v[0])))
        self.ln_out = cell("ln_out", 4)(
            lambda v: self.round(
                self.add(
                    self.mul(self.mul(v[0], v[1]), self.widen(v[2])), self.widen(v[3])
                )
            )
        )

    def kind_names(self) -> dict[str, str]:
        """Kind digest -> the readable key it was traced under (``dot(768,True,True)``, ``layer(4,0)``).

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

    def definition(self, key: Hashable) -> TracedDefinition:
        """The traced definition of ``key`` (``("dot", 768, True, True)``, ``"ln_out"``, ...)."""

        try:
            return self.tracer._by_key[key]
        except KeyError:
            raise KeyError(f"no definition traced under {key!r}") from None

    # -- ports ---------------------------------------------------------------------------

    def ports(self, v: Wires) -> _Ports:
        """Slice ``v[:weight_count]`` in the order of :meth:`GPT2Shape.layout`."""

        shape = self.shape
        d = shape.d_model
        cursor = _Cursor(v)
        wte, wpe = cursor.take(shape.vocab * d), cursor.take(shape.context * d)
        layers = tuple(cursor.take(shape.layer_weights) for _ in range(shape.layers))
        lnf_g, lnf_b, tokens = cursor.take(d), cursor.take(d), cursor.take(shape.vocab)
        n, scale, zero = cursor.one(), cursor.one(), cursor.one()
        assert cursor.at == shape.weight_count
        return _Ports(wte, wpe, layers, lnf_g, lnf_b, tokens, n, scale, zero)

    def layer_ports(self, lw: Wires) -> _LayerPorts:
        cursor = _Cursor(lw)
        pieces = [cursor.take(count) for _, count in self.shape.layer_layout()]
        assert cursor.at == self.shape.layer_weights
        return _LayerPorts(*pieces)

    def consts(self, ports: _Ports) -> _Consts:
        """Widen ``n`` and ``scale`` once per forward (two ``widen_cell`` VUs)."""

        n32, scale32 = self.widen_cell(ports.n), self.widen_cell(ports.scale)
        assert isinstance(n32, Wire) and isinstance(scale32, Wire)
        return _Consts(n32, scale32, ports.zero)

    # -- reductions ------------------------------------------------------------------------

    def reduce(self, level: Wires, pair: TracedDefinition, combine: TracerGate) -> Wire:
        """Fold ``level`` pairwise with ``pair`` (a two-input cell) level by level; carries at the end.

        Level ``i`` pairs consecutive elements ``(2j, 2j + 1)``; an odd last
        element is carried, and the carries are folded into the result in
        the order they arose.  ``len(level) - 1`` gates, whatever the length;
        a single value is returned as it is.  ``gpu/gpt2/`` and the reference
        forward spell out the same order.
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

    def dot(self, k: int, *, biased: bool, rounded: bool) -> TracedDefinition:
        """A ``k``-long dot product as a ``k/16``-step tensor-core chain: ports ``x`` (``k``), ``w`` (``k``), ``b``.

        Biased: the chain starts at the widened BF16 bias; otherwise its first
        step is ``tc_dot16_0``.  ``rounded``: the fp32 result is rounded to
        BF16 (an operand of the next matmul).
        """

        if type(k) is not int or k <= 0 or k % STEP:
            raise TracerError(f"dot length must be a positive multiple of {STEP}")

        @self.tracer.definition(
            input_count=2 * k + int(biased),
            key=("dot", k, biased, rounded),
            role=VERIFICATION,
        )
        def dot(v: Wires) -> object:
            x, w = cut(v, 0, k), cut(v, k, 2 * k)
            if biased:
                acc, start = self.widen(v[2 * k]), 0
            else:
                acc, start = self.tc0(cut(x, 0, STEP), cut(w, 0, STEP)), STEP
            for s in range(start, k, STEP):
                acc = self.tc(acc, cut(x, s, s + STEP), cut(w, s, s + STEP))
            return self.round(acc) if rounded else acc

        return dot

    def ln_mean(self) -> TracedDefinition:
        """The mean of ``d_model`` fp32 values: the sum tree, divided by ``n``.  Ports ``x``, ``n32``."""

        d = self.shape.d_model

        @self.tracer.definition(input_count=d + 1, key="ln_mean", role=VERIFICATION)
        def ln_mean(v: Wires) -> object:
            return self.div(self.reduce(cut(v, 0, d), self.add_pair, self.add), v[d])

        return ln_mean

    def ln_var(self) -> TracedDefinition:
        """The reciprocal standard deviation of ``d_model`` centred values: squares, the sum tree, ``/ n``, ``ln_rstd``."""

        d = self.shape.d_model

        @self.tracer.definition(input_count=d + 1, key="ln_var", role=VERIFICATION)
        def ln_var(v: Wires) -> object:
            squares = self.tracer.repeat(d, self.square, v[0].by(1))
            variance = self.div(self.reduce(squares, self.add_pair, self.add), v[d])
            return self.rstd(variance)

        return ln_var

    def softmax_max(self, c: int) -> TracedDefinition:
        """The maximum of ``c >= 2`` scaled scores: a tree of ``c - 1`` ``f32_max`` gates."""

        if type(c) is not int or c < 2:
            raise TracerError("a softmax maximum needs at least two scores")

        @self.tracer.definition(
            input_count=c, key=("softmax_max", c), role=VERIFICATION
        )
        def softmax_max(v: Wires) -> object:
            return self.reduce(cut(v, 0, c), self.max_pair, self.max)

        return softmax_max

    def softmax_sum(self, c: int) -> TracedDefinition:
        """The sum of ``c >= 2`` exponentials: a tree of ``c - 1`` ``f32_add`` gates."""

        if type(c) is not int or c < 2:
            raise TracerError("a softmax sum needs at least two exponentials")

        @self.tracer.definition(
            input_count=c, key=("softmax_sum", c), role=VERIFICATION
        )
        def softmax_sum(v: Wires) -> object:
            return self.reduce(cut(v, 0, c), self.add_pair, self.add)

        return softmax_sum

    def tournament(self, logits: Wires, table: Wires) -> tuple[Wire, Wire]:
        """The first maximum of ``(logit, index)`` pairs: levels of ``argmax_node`` repeats, carries folded at the end.

        Returns the winning logit and its index (``2 (n - 1)`` gates).
        """

        n = len(logits)
        if n == 1:
            return at(logits, 0), at(table, 0)
        carried: list[tuple[Wire, Wire]] = []
        if n % 2:
            carried.append((at(logits, -1), at(table, -1)))
        level = self.tracer.repeat(
            n // 2, self.node, cut(logits, 0, 2).by(2), cut(table, 0, 2).by(2)
        )
        while len(level) > 2:
            pairs = len(level) // 2
            if pairs % 2:
                carried.append((at(level, -2), at(level, -1)))
            level = self.tracer.repeat(
                pairs // 2,
                self.node,
                cut(level, 0, 4, 2).by(4),
                cut(level, 1, 4, 2).by(4),
            )
        best, index = at(level, 0), at(level, 1)
        for logit, token in carried:
            folded = wires(self.node(best, logit, index, token))
            best, index = at(folded, 0), at(folded, 1)
        return best, index

    def argmax_block(self, n: int) -> TracedDefinition:
        """The tournament over ``n >= 2`` logits: ports the logits then their token ids; outputs ``(best, index)``."""

        if type(n) is not int or n < 2:
            raise TracerError("an argmax block needs at least two logits")

        @self.tracer.definition(
            input_count=2 * n, key=("argmax_block", n), role=VERIFICATION
        )
        def argmax_block(v: Wires) -> object:
            return list(self.tournament(cut(v, 0, n), cut(v, n, 2 * n)))

        return argmax_block

    def argmax_top(self, m: int) -> TracedDefinition:
        """The tournament over ``m >= 2`` block winners: ports the logits then the indices; outputs the index."""

        if type(m) is not int or m < 2:
            raise TracerError("an argmax top needs at least two blocks")

        @self.tracer.definition(
            input_count=2 * m, key=("argmax_top", m), role=VERIFICATION
        )
        def argmax_top(v: Wires) -> object:
            return self.tournament(cut(v, 0, m), cut(v, m, 2 * m))[1]

        return argmax_top

    # -- unmarked composites -----------------------------------------------------------------

    def matvec(
        self, k: int, m: int, *, biased: bool, rounded: bool, rows: bool = False
    ) -> TracedDefinition:
        """``m`` dots of length ``k``: ``x W [+ b]`` for a row-major ``k x m`` matrix, or against the rows of an ``m x k`` one.

        Ports: ``x`` (``k``), the matrix (``k m``), then ``b`` (``m``) when
        biased.
        """

        key = ("matvec", k, m, biased, rounded, rows)

        @self.tracer.definition(input_count=k + k * m + (m if biased else 0), key=key)
        def matvec(v: Wires) -> object:
            x, w = cut(v, 0, k), cut(v, k, k + k * m)
            column = cut(w, 0, k).by(k) if rows else cut(w, 0, k * m, m).by(1)
            dot = self.dot(k, biased=biased, rounded=rounded)
            if biased:
                return self.tracer.repeat(m, dot, x, column, v[k + k * m].by(1))
            return self.tracer.repeat(m, dot, x, column)

        return matvec

    def layer_norm(self) -> TracedDefinition:
        """LayerNorm of ``d_model`` fp32 values to BF16: ports ``x, gamma, beta, n32``; four kinds of VU inside."""

        d = self.shape.d_model
        mean, var = self.ln_mean(), self.ln_var()

        @self.tracer.definition(input_count=3 * d + 1, key="layer_norm")
        def layer_norm(v: Wires) -> object:
            x, n32 = cut(v, 0, d), v[3 * d]
            mu = mean(x, n32)
            centred = self.tracer.repeat(d, self.sub_cell, v[0].by(1), mu)
            inv_std = var(centred, n32)
            return self.tracer.repeat(
                d, self.ln_out, centred[0].by(1), inv_std, v[d].by(1), v[2 * d].by(1)
            )

        return layer_norm

    def embed(self) -> TracedDefinition:
        """A token id and a position row to the residual stream: the one-hot, ``E``, plus the widened position row.

        Ports: the token, the token table, ``wte``, the position's row of
        ``wpe``, ``zero``.
        """

        vocab, d = self.shape.vocab, self.shape.d_model
        vpad = self.shape.vocab_padded
        project = self.dot(vpad, biased=False, rounded=False)

        @self.tracer.definition(input_count=2 + vocab + vocab * d + d, key="embed")
        def embed(v: Wires) -> object:
            cursor = _Cursor(v)
            token, table, wte, row = (
                cursor.one(),
                cursor.take(vocab),
                cursor.take(vocab * d),
                cursor.take(d),
            )
            zero = cursor.one()
            onehot = self.tracer.repeat(vocab, self.eq_cell, token, table[0].by(1))
            pads = [broadcast(zero, vpad - vocab)] if vpad > vocab else []
            e = self.tracer.repeat(
                d, project, onehot, *pads, cut(wte, 0, vocab * d, d).by(1), *pads
            )
            row32 = self.tracer.repeat(d, self.widen_cell, row[0].by(1))
            return self.tracer.repeat(d, self.add_cell, e[0].by(1), row32[0].by(1))

        return embed

    def attend_head(self, cached: int, fresh: int) -> TracedDefinition:
        """One head over ``cached`` cache positions and ``fresh`` new ones (``c = cached + fresh``).

        Ports: ``q`` (``d_head`` BF16 words); a *window* of the position-major
        cache block ``K`` starting at this head's first coordinate and ending
        at its last one at the last cached position (``(cached - 1) d_model +
        d_head`` values, absent when ``cached == 0``); the same window of the
        new keys; the two windows of the values; ``scale32``; ``zero``.  A
        window is one range, so key ``j`` of the head is ``window[j d_model :
        j d_model + d_head]`` and a ``repeat`` over positions shifts by
        ``d_model``.  Scores, the scaling, the maximum (when ``c >= 2``), the
        shifted exponentials, the sum, the probabilities and the ``d_head``
        value mixes (``ceil(c/16)`` steps, zero-padded) are the VUs inside.
        """

        if type(cached) is not int or cached < 0 or type(fresh) is not int or fresh < 1:
            raise TracerError(
                "attention needs a nonnegative cache and at least one new position"
            )
        shape = self.shape
        d, dh = shape.d_model, shape.d_head
        c = cached + fresh
        cache_window = (cached - 1) * d + dh if cached else 0
        fresh_window = (fresh - 1) * d + dh
        repeat = self.tracer.repeat
        score = self.dot(dh, biased=False, rounded=False)
        mix = self.dot(padded(c), biased=False, rounded=True)
        maximum = self.softmax_max(c) if c >= 2 else None
        total = self.softmax_sum(c) if c >= 2 else None

        @self.tracer.definition(
            input_count=dh + 2 * (cache_window + fresh_window) + 2,
            key=("attend_head", cached, fresh),
        )
        def attend_head(v: Wires) -> object:
            cursor = _Cursor(v)
            q = cursor.take(dh)
            k_cache = cursor.take(cache_window) if cached else None
            k_new = cursor.take(fresh_window)
            v_cache = cursor.take(cache_window) if cached else None
            v_new = cursor.take(fresh_window)
            scale32, zero = cursor.one(), cursor.one()
            parts: list[Wires] = []
            if k_cache is not None:
                parts.append(repeat(cached, score, q, cut(k_cache, 0, dh).by(d)))
            parts.append(repeat(fresh, score, q, cut(k_new, 0, dh).by(d)))
            scores = concat(parts)
            u = repeat(c, self.scale_cell, scores[0].by(1), scale32)
            m = u[0] if maximum is None else maximum(u)
            e = repeat(c, self.exp_cell, u[0].by(1), m)
            s = e[0] if total is None else total(e)
            p = repeat(c, self.prob_cell, e[0].by(1), s)
            pads = [broadcast(zero, padded(c) - c)] if padded(c) > c else []
            values: list[Wire | Wires] = [p, *pads]
            if v_cache is not None:
                values.append(cut(v_cache, 0, (cached - 1) * d + 1, d).by(1))
            values.append(cut(v_new, 0, (fresh - 1) * d + 1, d).by(1))
            values.extend(pads)
            return repeat(dh, mix, *values)

        return attend_head

    def layer(self, positions: int, cached: int) -> TracedDefinition:
        """One transformer block over ``positions`` new positions attending to ``cached`` earlier ones.

        Ports: the layer's weights (one block in the order of
        :data:`LAYER_FIELDS`), ``x`` (``positions x d_model`` fp32,
        position-major), the cached ``K`` and ``V`` (``cached x d_model``
        BF16 each, absent when ``cached == 0``), the constants ``n32,
        scale32, zero``.  Outputs: the new ``k`` and ``v`` (``positions x
        d_model`` each, the cache entries later steps read) and the new
        residual stream.  One kind for every layer.
        """

        if (
            type(positions) is not int
            or positions < 1
            or type(cached) is not int
            or cached < 0
        ):
            raise TracerError(
                "a layer needs at least one new position and a nonnegative cache"
            )
        shape = self.shape
        d, dh, heads, f = shape.d_model, shape.d_head, shape.heads, shape.d_ff
        repeat = self.tracer.repeat
        norm = self.layer_norm()
        project_r = self.matvec(d, d, biased=True, rounded=True)
        project = self.matvec(d, d, biased=True, rounded=False)
        up, down = (
            self.matvec(d, f, biased=True, rounded=False),
            self.matvec(f, d, biased=True, rounded=False),
        )
        heads_at = [self.attend_head(cached, p + 1) for p in range(positions)]
        cache = cached * d

        @self.tracer.definition(
            input_count=shape.layer_weights + positions * d + 2 * cache + 3,
            key=("layer", positions, cached),
        )
        def layer(v: Wires) -> object:
            cursor = _Cursor(v)
            w = self.layer_ports(cursor.take(shape.layer_weights))
            x = cursor.take(positions * d)
            k_cache = cursor.take(cache) if cached else None
            v_cache = cursor.take(cache) if cached else None
            n32, scale32, zero = cursor.one(), cursor.one(), cursor.one()
            h = repeat(positions, norm, cut(x, 0, d).by(d), w.ln1_g, w.ln1_b, n32)
            q = repeat(positions, project_r, cut(h, 0, d).by(d), w.w_q, w.b_q)
            k = repeat(positions, project_r, cut(h, 0, d).by(d), w.w_k, w.b_k)
            vv = repeat(positions, project_r, cut(h, 0, d).by(d), w.w_v, w.b_v)
            attended: list[Wires] = []
            for p in range(positions):
                args: list[Wire | Wires] = [cut(q, p * d, p * d + dh).by(dh)]
                if k_cache is not None:
                    args.append(cut(k_cache, 0, (cached - 1) * d + dh).by(dh))
                args.append(cut(k, 0, p * d + dh).by(dh))
                if v_cache is not None:
                    args.append(cut(v_cache, 0, (cached - 1) * d + dh).by(dh))
                args.append(cut(vv, 0, p * d + dh).by(dh))
                args.extend((scale32, zero))
                attended.append(repeat(heads, heads_at[p], *args))
            o = repeat(
                positions, project, cut(concat(attended), 0, d).by(d), w.w_o, w.b_o
            )
            x1 = repeat(positions * d, self.add_cell, x[0].by(1), o[0].by(1))
            h2 = repeat(positions, norm, cut(x1, 0, d).by(d), w.ln2_g, w.ln2_b, n32)
            hidden = repeat(positions, up, cut(h2, 0, d).by(d), w.w_fc, w.b_fc)
            act = repeat(positions * f, self.gelu_cell, hidden[0].by(1))
            m = repeat(positions, down, cut(act, 0, f).by(f), w.w_proj, w.b_proj)
            x2 = repeat(positions * d, self.add_cell, x1[0].by(1), m[0].by(1))
            return [k, vv, x2]

        return layer

    def argmax(self, logits: Wires, table: Wires) -> Wire:
        """The first maximum of the logits as a token id: block tournaments, then one over the block winners."""

        shape = self.shape
        vocab, block = shape.vocab, shape.argmax_block
        full, rest = divmod(vocab, block)
        if full == 0:
            token = self.argmax_block(vocab)(logits, table)
            return at(wires(token), 1)
        level = self.tracer.repeat(
            full,
            self.argmax_block(block),
            cut(logits, 0, block).by(block),
            cut(table, 0, block).by(block),
        )
        bests: list[Wire | Wires] = [cut(level, 0, 2 * full, 2)]
        indices: list[Wire | Wires] = [cut(level, 1, 2 * full, 2)]
        if rest == 1:
            bests.append(at(logits, -1))
            indices.append(at(table, -1))
        elif rest > 1:
            tail = wires(
                self.argmax_block(rest)(
                    cut(logits, full * block, vocab), cut(table, full * block, vocab)
                )
            )
            bests.append(at(tail, 0))
            indices.append(at(tail, 1))
        blocks = full + int(rest > 0)
        if blocks == 1:
            return at(level, 1)
        token = self.argmax_top(blocks)(*bests, *indices)
        assert isinstance(token, Wire)
        return token

    def head(self, ports: _Ports, consts: _Consts, x: Wires) -> Wire:
        """The final LayerNorm, the tied LM head and the argmax of one position's residual stream."""

        shape = self.shape
        h = self.layer_norm()(x, ports.lnf_g, ports.lnf_b, consts.n32)
        logits = self.matvec(
            shape.d_model, shape.vocab, biased=False, rounded=False, rows=True
        )(wires(h), ports.wte)
        return self.argmax(wires(logits), ports.tokens)

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
            consts = self.consts(ports)
            rows = cut(ports.wpe, 0, d).by(d)
            x = self.tracer.repeat(
                n, embed, tokens[0].by(1), ports.tokens, ports.wte, rows, ports.zero
            )
            state: list[Wires] = []
            for block in ports.layers:
                out = wires(layer(block, x, *consts.wires()))
                state += [cut(out, 0, n * d), cut(out, n * d, 2 * n * d)]
                x = cut(out, 2 * n * d, 3 * n * d)
            return [*state, self.head(ports, consts, cut(x, (n - 1) * d, n * d))]

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

        @self.tracer.definition(
            input_count=weights + 1 + shape.layers * 2 * cache, key=("decode", c)
        )
        def decode(v: Wires) -> object:
            ports = self.ports(v)
            consts = self.consts(ports)
            row = cut(ports.wpe, cached * d, c * d)
            x = wires(embed(v[weights], ports.tokens, ports.wte, row, ports.zero))
            state: list[Wires] = []
            for index, block in enumerate(ports.layers):
                start = weights + 1 + index * 2 * cache
                k_cache, v_cache = (
                    cut(v, start, start + cache),
                    cut(v, start + cache, start + 2 * cache),
                )
                out = wires(layer(block, x, k_cache, v_cache, *consts.wires()))
                state += [cut(out, 0, d), cut(out, d, 2 * d)]
                x = cut(out, 2 * d, 3 * d)
            return [*state, self.head(ports, consts, x)]

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

        if (
            type(prompt) is not int
            or prompt < 1
            or type(max_new) is not int
            or max_new < 1
        ):
            raise TracerError(
                "a request needs a nonempty prompt and at least one token"
            )
        shape = self.shape
        layers, d = shape.layers, shape.d_model
        prefill = self.prefill(prompt)
        decodes = [self.decode(prompt + step) for step in range(1, max_new)]

        @self.tracer.definition(
            input_count=shape.weight_count,
            key=("request", prompt, max_new),
            role=REPLAY,
        )
        def request(w: Wires) -> object:
            keys: list[list[Wires]] = [[] for _ in range(layers)]
            values: list[list[Wires]] = [[] for _ in range(layers)]

            def remember(block: Wires, positions: int) -> Wire:
                for layer in range(layers):
                    start = 2 * layer * positions * d
                    keys[layer].append(cut(block, start, start + positions * d))
                    values[layer].append(
                        cut(block, start + positions * d, start + 2 * positions * d)
                    )
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
    (:meth:`output_layout`).  The weights are the model's BF16 words in the
    order of :meth:`GPT2Shape.layout`.
    """

    VERSION = "2"

    def __init__(self, shape: GPT2Shape) -> None:
        if not isinstance(shape, GPT2Shape):
            raise TypeError("shape must be a GPT2Shape")
        self.shape = shape
        self.model = GPT2(shape)
        self.digest: Digest = constructor_digest(
            type(self).__name__, self.VERSION, self.manifest
        )

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"gate_set": self.gate_set.digest, "shape": self.shape.manifest}

    @property
    def gate_set(self) -> GateSet:
        """The pinned gate set the description is written over."""

        return self.model.tracer.gate_set

    # -- validation -------------------------------------------------------------------------

    def requests(self, x: object) -> tuple[Request, ...]:
        if (
            type(x) is not tuple
            or not x
            or any(type(item) is not Request for item in x)
        ):
            raise TracerError("GPT2G expects a nonempty tuple of Request")
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
        return x

    # -- layouts -----------------------------------------------------------------------------

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, generated position)`` of every circuit output, in output order."""

        return tuple(
            (r, g)
            for r, request in enumerate(self.requests(x))
            for g in range(request.max_new)
        )

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """The prompt tokens in ``in``-gate address order: request by request."""

        return tuple(token for request in self.requests(x) for token in request.prompt)

    def gate_budget(self, x: object) -> dict[str, int]:
        """:func:`gate_budget` summed over the requests of ``x``."""

        total: dict[str, int] = {}
        for request in self.requests(x):
            for component, count in gate_budget(
                self.shape, len(request.prompt), request.max_new
            ).items():
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
            return [
                unit(w) if count == 1 else model.tracer.repeat(count, unit, w)
                for unit, count in groups
            ]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        if a:
            raise TracerError("GPT2G takes no advice")
        requests = self.requests(x)
        return self.model.tracer.serialize(self.root(requests)), self.flatten_inputs(
            requests
        )


__all__ = [
    "GPT2",
    "GPT2G",
    "LAYER_FIELDS",
    "SCALARS",
    "STEP",
    "GPT2Shape",
    "at",
    "bf16_word",
    "broadcast",
    "cut",
    "gate_budget",
    "padded",
]
