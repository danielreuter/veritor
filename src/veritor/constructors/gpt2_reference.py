"""The executable twin of :class:`~veritor.constructors.gpt2.GPT2G`: the pinned forward pass over numpy words.

:func:`forward` runs GPT-2 greedy decoding with exactly the semantics the
constructor's gates pin -- the same tensor-core chains (through a
:class:`GemmBackend`: the pure-Python :func:`~veritor.core.silicon.tc_dot_chain`,
the C++ reference ``ref_chain.cpp``, or the GPU kernels of ``gpu/gpt2/``) and
the same fp32 sequences and reduction orders on the CUDA-core side (the numpy
functions of :mod:`veritor.core.silicon`, vectorised).  It records every
intermediate tensor a verification unit (VU) reads or writes in a
:class:`Capture`; :func:`address_map` says at which circuit address each
recorded word lives, so a capture can be compared word for word with
``Circuit.evaluate`` (tiny shapes, ``tests/veritor/constructors/test_gpt2.py``)
or with a GPU run of GPT-2 Small (``gpu/gpt2/``, ``docs/gpt2-silicon.md``).

Words: BF16 tensors are ``uint16`` arrays, fp32 tensors ``float32`` arrays,
token ids ``uint16``.  Nothing here is trusted; it is the client's model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from veritor.core import Compiled, IndexNode, Pipeline, pipeline_for, tc_dot_chain
from veritor.core.description import CallStep, Frame
from veritor.core.silicon import (
    bf16_to_f32,
    f32_exp,
    f32_max,
    f32_to_bf16,
    gelu_tanh,
    ln_rstd,
)

from .gpt2 import GPT2, LAYER_FIELDS, SCALARS, STEP, GPT2Shape, padded

U16 = np.ndarray[tuple[int, ...], np.dtype[np.uint16]]
F32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
I64 = np.ndarray[tuple[int, ...], np.dtype[np.int64]]


# -- weights ------------------------------------------------------------------------------


def as_bf16_words(values: np.ndarray) -> U16:
    """Round a float32 (or float64) array to BF16 words, nearest-even (the model's single rounding)."""

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return f32_to_bf16(bits)


def words_as_f32(words: np.ndarray) -> F32:
    """BF16 words as float32 values (exact)."""

    return bf16_to_f32(np.asarray(words, dtype=np.uint16)).view(np.float32)


@dataclass(frozen=True, slots=True)
class LayerWeights:
    """One layer's BF16 words, matrices in the ``x @ W`` orientation (``[in, out]``)."""

    ln1_g: U16
    ln1_b: U16
    w_q: U16
    b_q: U16
    w_k: U16
    b_k: U16
    w_v: U16
    b_v: U16
    w_o: U16
    b_o: U16
    ln2_g: U16
    ln2_b: U16
    w_fc: U16
    b_fc: U16
    w_proj: U16
    b_proj: U16


@dataclass(frozen=True, slots=True)
class GPT2Weights:
    """The model's ``weight`` words: :meth:`GPT2Shape.layout` as arrays."""

    shape: GPT2Shape
    wte: U16  # [vocab, d_model]
    wpe: U16  # [context, d_model]
    layers: tuple[LayerWeights, ...]
    lnf_g: U16
    lnf_b: U16

    @property
    def tokens(self) -> U16:
        return np.arange(self.shape.vocab, dtype=np.uint16)

    @property
    def scalars(self) -> dict[str, int]:
        return self.shape.scalar_words()

    def blocks(self) -> Iterator[tuple[str, U16]]:
        """The weight blocks in layout order, each flattened."""

        yield "wte", self.wte.reshape(-1)
        yield "wpe", self.wpe.reshape(-1)
        for index, layer in enumerate(self.layers):
            for name in LAYER_FIELDS:
                yield f"layer{index}.{name}", getattr(layer, name).reshape(-1)
        yield "lnf_g", self.lnf_g
        yield "lnf_b", self.lnf_b
        yield "tokens", self.tokens
        for name in SCALARS:
            yield name, np.array([self.scalars[name]], dtype=np.uint16)

    def flat(self) -> U16:
        """Every ``weight`` word by rank."""

        flat = np.concatenate([block for _, block in self.blocks()]).astype(np.uint16)
        assert flat.shape == (self.shape.weight_count,)
        return flat

    @classmethod
    def from_flat(cls, shape: GPT2Shape, flat: np.ndarray) -> GPT2Weights:
        flat = np.asarray(flat, dtype=np.uint16)
        if flat.shape != (shape.weight_count,):
            raise ValueError(
                f"expected {shape.weight_count} weight words, got {flat.shape}"
            )
        d, f = shape.d_model, shape.d_ff
        cursor = 0

        def take(count: int, *dims: int) -> U16:
            nonlocal cursor
            piece = flat[cursor : cursor + count]
            cursor += count
            return piece.reshape(*dims) if dims else piece

        wte, wpe = (
            take(shape.vocab * d, shape.vocab, d),
            take(shape.context * d, shape.context, d),
        )
        layers = []
        for _ in range(shape.layers):
            fields = {}
            for name, count in shape.layer_layout():
                if name in ("w_q", "w_k", "w_v", "w_o"):
                    fields[name] = take(count, d, d)
                elif name == "w_fc":
                    fields[name] = take(count, d, f)
                elif name == "w_proj":
                    fields[name] = take(count, f, d)
                else:
                    fields[name] = take(count)
            layers.append(LayerWeights(**fields))
        lnf_g, lnf_b = take(d), take(d)
        tokens = take(shape.vocab)
        scalars = {name: int(take(1)[0]) for name in SCALARS}
        if cursor != shape.weight_count:
            raise ValueError("weight layout mismatch")
        if not np.array_equal(tokens, np.arange(shape.vocab, dtype=np.uint16)):
            raise ValueError("the token table must be 0 .. vocab - 1")
        if scalars != shape.scalar_words():
            raise ValueError(f"the scalar words must be {shape.scalar_words()}")
        return cls(shape, wte, wpe, tuple(layers), lnf_g, lnf_b)

    @classmethod
    def random(cls, shape: GPT2Shape, seed: int = 0, scale: float = 0.2) -> GPT2Weights:
        """Small random weights (LayerNorm gains near 1) for tests at tiny shapes."""

        rng = np.random.default_rng(seed)
        d, f = shape.d_model, shape.d_ff

        def normal(*dims: int, sd: float = scale) -> U16:
            return as_bf16_words(rng.normal(0.0, sd, size=dims).astype(np.float32))

        def gains(n: int) -> U16:
            return as_bf16_words(
                (1.0 + rng.normal(0.0, 0.1, size=n)).astype(np.float32)
            )

        layers = tuple(
            LayerWeights(
                ln1_g=gains(d),
                ln1_b=normal(d, sd=0.1),
                w_q=normal(d, d),
                b_q=normal(d, sd=0.1),
                w_k=normal(d, d),
                b_k=normal(d, sd=0.1),
                w_v=normal(d, d),
                b_v=normal(d, sd=0.1),
                w_o=normal(d, d),
                b_o=normal(d, sd=0.1),
                ln2_g=gains(d),
                ln2_b=normal(d, sd=0.1),
                w_fc=normal(d, f),
                b_fc=normal(f, sd=0.1),
                w_proj=normal(f, d),
                b_proj=normal(d, sd=0.1),
            )
            for _ in range(shape.layers)
        )
        return cls(
            shape,
            normal(shape.vocab, d, sd=1.0),
            normal(shape.context, d, sd=0.3),
            layers,
            gains(d),
            normal(d, sd=0.1),
        )


# -- GEMM backends -------------------------------------------------------------------------


class GemmBackend(Protocol):
    """A fixed-order tensor-core GEMM: ``D[i][j] = chain(C[i][j], A[i][:], Bt[j][:])`` in ``k`` order.

    ``A`` is ``[M][K]`` and ``Bt`` ``[N][K]`` BF16 words, ``C`` an optional
    ``[M][N]`` fp32 incoming accumulator (``+0`` when ``None``); ``K`` is a
    multiple of 16.  The result is ``[M][N]`` fp32.
    """

    def gemm(self, a: U16, bt: U16, c: F32 | None) -> F32: ...


class PythonGemm:
    """The pure-Python chain, element by element: for tiny shapes and spot checks."""

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline_for("sm_89", "bf16") if pipeline is None else pipeline

    def gemm(self, a: U16, bt: U16, c: F32 | None) -> F32:
        m, k = a.shape
        n, k2 = bt.shape
        if k != k2 or k % STEP:
            raise ValueError("operands need one K, a multiple of 16")
        acc = (
            np.zeros((m, n), dtype=np.uint32)
            if c is None
            else np.ascontiguousarray(c, dtype=np.float32).view(np.uint32)
        )
        out = np.empty((m, n), dtype=np.uint32)
        rows = [[int(w) for w in a[i]] for i in range(m)]
        cols = [[int(w) for w in bt[j]] for j in range(n)]
        for i in range(m):
            for j in range(n):
                out[i, j] = tc_dot_chain(
                    self.pipeline, int(acc[i, j]), rows[i], cols[j]
                )
        return out.view(np.float32)


# -- the forward pass -----------------------------------------------------------------------

Capture = dict[str, np.ndarray]
"""Recorded tensors by name (:func:`forward` and :func:`address_map` share the names and shapes)."""


def tree_reduce(
    x: np.ndarray, op: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """Reduce the last axis in the fixed pairwise order of :meth:`GPT2.reduce`.

    Level ``i`` combines elements ``(2j, 2j + 1)``; an odd last element is
    carried and the carries are folded into the result in order.
    """

    level = x
    carried = []
    while level.shape[-1] > 1:
        n = level.shape[-1]
        if n % 2:
            carried.append(level[..., -1])
        level = op(level[..., 0 : 2 * (n // 2) : 2], level[..., 1 : 2 * (n // 2) : 2])
    result = level[..., 0]
    for carry in carried:
        result = op(result, carry)
    return result


def f32_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.add(a, b, dtype=np.float32)


def tournament(vals: F32, idxs: U16) -> tuple[np.float32, np.uint16]:
    """The first maximum of ``(logit, index)`` pairs in the order of :meth:`GPT2.tournament`."""

    def node(
        la: np.ndarray, lb: np.ndarray, ia: np.ndarray, ib: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        better = lb > la
        return np.where(better, lb, la).astype(np.float32), np.where(
            better, ib, ia
        ).astype(np.uint16)

    n = vals.shape[0]
    if n == 1:
        return np.float32(vals[0]), np.uint16(idxs[0])
    carried: list[tuple[np.ndarray, np.ndarray]] = []
    if n % 2:
        carried.append((vals[-1], idxs[-1]))
    half = n // 2
    lv, li = node(
        vals[0 : 2 * half : 2],
        vals[1 : 2 * half : 2],
        idxs[0 : 2 * half : 2],
        idxs[1 : 2 * half : 2],
    )
    while lv.shape[0] > 1:
        entries = lv.shape[0]
        if entries % 2:
            carried.append((lv[-1], li[-1]))
        half = entries // 2
        lv, li = node(
            lv[0 : 2 * half : 2],
            lv[1 : 2 * half : 2],
            li[0 : 2 * half : 2],
            li[1 : 2 * half : 2],
        )
    best, index = lv[0], li[0]
    for logit, token in carried:
        best, index = node(best, logit, index, token)
    return np.float32(best), np.uint16(index)


@dataclass(slots=True)
class Run:
    """The result of :func:`forward`: the generated tokens and every recorded tensor."""

    shape: GPT2Shape
    prompt: tuple[int, ...]
    max_new: int
    tokens: tuple[int, ...]
    capture: Capture = field(default_factory=dict)

    @property
    def positions(self) -> int:
        return len(self.prompt) + self.max_new - 1


def argmax_blocks(shape: GPT2Shape) -> list[int]:
    """The block sizes of the argmax tournament whose ``(best, index)`` outputs are gates."""

    full, rest = divmod(shape.vocab, shape.argmax_block)
    if full == 0:
        return [shape.vocab]
    return [shape.argmax_block] * full + ([rest] if rest > 1 else [])


def forward(
    weights: GPT2Weights, prompt: Sequence[int], max_new: int, backend: GemmBackend
) -> Run:
    """Greedy decoding of ``prompt`` for ``max_new`` tokens with the pinned semantics; records everything."""

    shape = weights.shape
    d, dh, heads, f, vocab = (
        shape.d_model,
        shape.d_head,
        shape.heads,
        shape.d_ff,
        shape.vocab,
    )
    prompt = tuple(int(t) for t in prompt)
    if not prompt or max_new < 1 or len(prompt) + max_new > shape.context:
        raise ValueError("a request needs a nonempty prompt and fits the context")
    positions = len(prompt) + max_new - 1
    n32 = np.float32(words_as_f32(np.array([weights.scalars["n"]], dtype=np.uint16))[0])
    scale32 = np.float32(
        words_as_f32(np.array([weights.scalars["scale"]], dtype=np.uint16))[0]
    )
    blocks = argmax_blocks(shape)
    cap: Capture = {}

    def new(name: str, *dims: int, dtype: type = np.float32) -> np.ndarray:
        array: np.ndarray = np.zeros(dims, dtype=dtype)
        cap[name] = array
        return array

    cap["tokens"] = np.zeros(positions, dtype=np.uint16)
    cap["n32"] = np.full(max_new, n32, dtype=np.float32)
    cap["scale32"] = np.full(max_new, scale32, dtype=np.float32)
    onehot_all = new("onehot", positions, vocab, dtype=np.uint16)
    emb = new("emb", positions, d)
    wpe32 = new("wpe32", positions, d)
    x0 = new("x0", positions, d)
    for layer in range(shape.layers):
        prefix = f"L{layer}."
        for ln in ("ln1", "ln2"):
            new(f"{prefix}{ln}.mean", positions)
            new(f"{prefix}{ln}.center", positions, d)
            new(f"{prefix}{ln}.rstd", positions)
            new(f"{prefix}{ln}.out", positions, d, dtype=np.uint16)
        for name in ("q", "k", "v"):
            new(f"{prefix}{name}32", positions, d)
            new(f"{prefix}{name}", positions, d, dtype=np.uint16)
        new(f"{prefix}scores", heads, positions, positions)
        new(f"{prefix}u", heads, positions, positions)
        new(f"{prefix}m", heads, positions)
        new(f"{prefix}e", heads, positions, positions)
        new(f"{prefix}S", heads, positions)
        new(f"{prefix}p", heads, positions, positions, dtype=np.uint16)
        new(f"{prefix}mix32", positions, d)
        new(f"{prefix}mix", positions, d, dtype=np.uint16)
        new(f"{prefix}proj", positions, d)
        new(f"{prefix}x1", positions, d)
        new(f"{prefix}fc", positions, f)
        new(f"{prefix}gelu32", positions, f)
        new(f"{prefix}gelu", positions, f, dtype=np.uint16)
        new(f"{prefix}mlp", positions, d)
        new(f"{prefix}x2", positions, d)
    new("lnf.mean", max_new)
    new("lnf.center", max_new, d)
    new("lnf.rstd", max_new)
    new("lnf.out", max_new, d, dtype=np.uint16)
    logits_all = new("logits", max_new, vocab)
    best_all = new("argmax.best", max_new, len(blocks))
    idx_all = new("argmax.idx", max_new, len(blocks), dtype=np.uint16)
    token_all = new("argmax.token", max_new, dtype=np.uint16)

    wte_t = np.zeros((d, shape.vocab_padded), dtype=np.uint16)
    wte_t[:, :vocab] = weights.wte.T
    transposed = [
        {
            name: np.ascontiguousarray(getattr(lw, name).T)
            for name in ("w_q", "w_k", "w_v", "w_o", "w_fc", "w_proj")
        }
        for lw in weights.layers
    ]
    k_cache = [np.zeros((0, d), dtype=np.uint16) for _ in range(shape.layers)]
    v_cache = [np.zeros((0, d), dtype=np.uint16) for _ in range(shape.layers)]

    def bias_rows(bias: U16, m: int) -> F32:
        return np.repeat(words_as_f32(bias)[None, :], m, axis=0)

    def layer_norm(
        x: F32, g: U16, b: U16, out: Capture, prefix: str, rows: slice
    ) -> U16:
        total = tree_reduce(x, f32_add)
        with np.errstate(all="ignore"):
            mean = (total / n32).astype(np.float32)
            center = (x - mean[:, None]).astype(np.float32)
            var = (
                tree_reduce((center * center).astype(np.float32), f32_add) / n32
            ).astype(np.float32)
            rstd = ln_rstd(var)
            y = (
                (center * rstd[:, None]).astype(np.float32) * words_as_f32(g)[None, :]
            ).astype(np.float32)
            y = (y + words_as_f32(b)[None, :]).astype(np.float32)
        y16 = f32_to_bf16(y.view(np.uint32))
        out[f"{prefix}.mean"][rows] = mean
        out[f"{prefix}.center"][rows] = center
        out[f"{prefix}.rstd"][rows] = rstd
        out[f"{prefix}.out"][rows] = y16
        return y16

    generated: list[int] = []
    p0 = 0
    for step in range(max_new):
        step_tokens = prompt if step == 0 else (generated[-1],)
        m = len(step_tokens)
        rows = slice(p0, p0 + m)
        cap["tokens"][rows] = np.array(step_tokens, dtype=np.uint16)
        # embedding: one-hot times wte as a chain, plus the widened position row
        onehot = np.zeros((m, shape.vocab_padded), dtype=np.uint16)
        for i, token in enumerate(step_tokens):
            onehot[i, token] = 0x3F80
        onehot_all[rows] = onehot[:, :vocab]
        e = backend.gemm(onehot, wte_t, None)
        emb[rows] = e
        row32 = words_as_f32(weights.wpe[p0 : p0 + m])
        wpe32[rows] = row32
        x = f32_add(e, row32)
        x0[rows] = x
        for layer in range(shape.layers):
            lw, tw = weights.layers[layer], transposed[layer]
            prefix = f"L{layer}."
            h = layer_norm(x, lw.ln1_g, lw.ln1_b, cap, f"{prefix}ln1", rows)
            qkv: dict[str, U16] = {}
            for name in ("q", "k", "v"):
                value32 = backend.gemm(
                    h, tw[f"w_{name}"], bias_rows(getattr(lw, f"b_{name}"), m)
                )
                cap[f"{prefix}{name}32"][rows] = value32
                qkv[name] = f32_to_bf16(value32.view(np.uint32))
                cap[f"{prefix}{name}"][rows] = qkv[name]
            k_cache[layer] = np.concatenate([k_cache[layer], qkv["k"]])
            v_cache[layer] = np.concatenate([v_cache[layer], qkv["v"]])
            keys = k_cache[layer].shape[0]
            mix32 = np.zeros((m, d), dtype=np.float32)
            for head in range(heads):
                cols = slice(head * dh, (head + 1) * dh)
                q_h = np.ascontiguousarray(qkv["q"][:, cols])
                k_h = np.ascontiguousarray(k_cache[layer][:, cols])
                v_h_t = np.ascontiguousarray(v_cache[layer][:, cols].T)  # [dh, keys]
                scores = backend.gemm(q_h, k_h, None)  # [m, keys]
                probs = np.zeros((m, padded(keys)), dtype=np.uint16)
                for i in range(m):
                    c = p0 + i + 1
                    s = scores[i, :c]
                    with np.errstate(all="ignore"):
                        u = (s * scale32).astype(np.float32)
                    mx = tree_reduce(u, f32_max) if c >= 2 else u[0]
                    with np.errstate(all="ignore"):
                        ex = f32_exp((u - mx).astype(np.float32))
                    total = tree_reduce(ex, f32_add) if c >= 2 else ex[0]
                    with np.errstate(all="ignore"):
                        p = f32_to_bf16((ex / total).astype(np.float32).view(np.uint32))
                    probs[i, :c] = p
                    pos = p0 + i
                    cap[f"{prefix}scores"][head, pos, :c] = s
                    cap[f"{prefix}u"][head, pos, :c] = u
                    cap[f"{prefix}m"][head, pos] = mx
                    cap[f"{prefix}e"][head, pos, :c] = ex
                    cap[f"{prefix}S"][head, pos] = total
                    cap[f"{prefix}p"][head, pos, :c] = p
                v_pad = np.zeros((dh, padded(keys)), dtype=np.uint16)
                v_pad[:, :keys] = v_h_t
                mix32[:, cols] = backend.gemm(probs, v_pad, None)
            cap[f"{prefix}mix32"][rows] = mix32
            mix = f32_to_bf16(mix32.view(np.uint32))
            cap[f"{prefix}mix"][rows] = mix
            proj = backend.gemm(mix, tw["w_o"], bias_rows(lw.b_o, m))
            cap[f"{prefix}proj"][rows] = proj
            x1 = f32_add(x, proj)
            cap[f"{prefix}x1"][rows] = x1
            h2 = layer_norm(x1, lw.ln2_g, lw.ln2_b, cap, f"{prefix}ln2", rows)
            fc = backend.gemm(h2, tw["w_fc"], bias_rows(lw.b_fc, m))
            cap[f"{prefix}fc"][rows] = fc
            g32 = gelu_tanh(fc)
            cap[f"{prefix}gelu32"][rows] = g32
            act = f32_to_bf16(g32.view(np.uint32))
            cap[f"{prefix}gelu"][rows] = act
            mlp = backend.gemm(act, tw["w_proj"], bias_rows(lw.b_proj, m))
            cap[f"{prefix}mlp"][rows] = mlp
            x = f32_add(x1, mlp)
            cap[f"{prefix}x2"][rows] = x
        # the head at the last position of this forward
        last = x[m - 1 : m]
        hf = layer_norm(
            last, weights.lnf_g, weights.lnf_b, cap, "lnf", slice(step, step + 1)
        )
        logits = backend.gemm(hf, weights.wte, None)[0]
        logits_all[step] = logits
        table = weights.tokens
        bests: list[np.float32] = []
        indices: list[np.uint16] = []
        start = 0
        for size in blocks:
            best, index = tournament(
                logits[start : start + size], table[start : start + size]
            )
            bests.append(best)
            indices.append(index)
            start += size
        best_all[step] = np.array(bests, dtype=np.float32)
        idx_all[step] = np.array(indices, dtype=np.uint16)
        if start < vocab:  # a single trailing logit joins the top tournament unblocked
            bests.append(np.float32(logits[start]))
            indices.append(np.uint16(table[start]))
        if len(bests) == 1:
            token = int(indices[0])
        else:
            token = int(
                tournament(
                    np.array(bests, dtype=np.float32),
                    np.array(indices, dtype=np.uint16),
                )[1]
            )
        token_all[step] = token
        generated.append(token)
        p0 += m
    return Run(shape, prompt, max_new, tuple(generated), cap)


# -- addresses -------------------------------------------------------------------------------


def _copies(frame: Frame, digest: str) -> list[Frame]:
    """The child frames of ``frame`` whose definition is ``digest``, in layout order."""

    found: list[Frame] = []
    for index, step in enumerate(frame.definition.steps):
        if isinstance(step, CallStep) and step.child.digest == digest:
            found.extend(frame.child(index, copy) for copy in range(step.count))
    return found


def _outputs(frame: Frame, digest: str, back: int = 1) -> I64:
    """The address of the ``back``-th gate from the end of every copy of ``digest`` called by ``frame``.

    For a one-output cell the last gate is its output; ``back = 2`` names the
    gate before it (the fp32 value a rounded dot rounds, the winning logit
    of an argmax block).
    """

    pieces: list[np.ndarray] = []
    for index, step in enumerate(frame.definition.steps):
        if isinstance(step, CallStep) and step.child.digest == digest:
            size = step.child.size
            first = frame.base + frame.definition.step_address[index] + size - back
            pieces.append(first + size * np.arange(step.count, dtype=np.int64))
    if not pieces:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(pieces)


def request_frames(compiled: Compiled) -> list[Frame]:
    """The frame of every ``request`` replay unit, in order (the ``weights`` unit is replay unit 0)."""

    units = compiled.index.replay_units
    return [units.unit(index).frame for index in range(1, len(units))]


def address_map(
    compiled: Compiled, model: GPT2, request: Frame, prompt: int, max_new: int
) -> Capture:
    """The circuit address of every word :func:`forward` records, by the same names and shapes.

    ``-1`` marks entries that are no gate (the upper triangle of the
    attention tensors, the softmax statistics of a one-key query).
    """

    shape = model.shape
    d, dh, heads, f, vocab = (
        shape.d_model,
        shape.d_head,
        shape.heads,
        shape.d_ff,
        shape.vocab,
    )
    positions = prompt + max_new - 1
    defs = model.definition
    dig = lambda key: defs(key).digest
    blocks = argmax_blocks(shape)
    out: Capture = {}

    def new(name: str, *dims: int) -> I64:
        array = np.full(dims, -1, dtype=np.int64)
        out[name] = array
        return array

    tokens = new("tokens", positions)
    n32, scale32 = new("n32", max_new), new("scale32", max_new)
    onehot = new("onehot", positions, vocab)
    emb, wpe32, x0 = (
        new("emb", positions, d),
        new("wpe32", positions, d),
        new("x0", positions, d),
    )
    per_layer: dict[str, I64] = {}
    for layer in range(shape.layers):
        prefix = f"L{layer}."
        for ln in ("ln1", "ln2"):
            new(f"{prefix}{ln}.mean", positions)
            new(f"{prefix}{ln}.center", positions, d)
            new(f"{prefix}{ln}.rstd", positions)
            new(f"{prefix}{ln}.out", positions, d)
        for name in ("q", "k", "v"):
            new(f"{prefix}{name}32", positions, d)
            new(f"{prefix}{name}", positions, d)
        new(f"{prefix}scores", heads, positions, positions)
        new(f"{prefix}u", heads, positions, positions)
        new(f"{prefix}m", heads, positions)
        new(f"{prefix}e", heads, positions, positions)
        new(f"{prefix}S", heads, positions)
        new(f"{prefix}p", heads, positions, positions)
        for name in ("mix32", "mix", "proj", "x1"):
            new(f"{prefix}{name}", positions, d)
        for name in ("fc", "gelu32", "gelu"):
            new(f"{prefix}{name}", positions, f)
        new(f"{prefix}mlp", positions, d)
        new(f"{prefix}x2", positions, d)
    del per_layer
    new("lnf.mean", max_new)
    new("lnf.center", max_new, d)
    new("lnf.rstd", max_new)
    new("lnf.out", max_new, d)
    logits = new("logits", max_new, vocab)
    best, idx = (
        new("argmax.best", max_new, len(blocks)),
        new("argmax.idx", max_new, len(blocks)),
    )
    token = new("argmax.token", max_new)

    layer_norm = dig("layer_norm")
    ln_mean, ln_var, sub_cell, ln_out = (
        dig("ln_mean"),
        dig("ln_var"),
        dig("sub_cell"),
        dig("ln_out"),
    )
    widen_cell, add_cell, eq_cell = dig("widen_cell"), dig("add_cell"), dig("eq_cell")
    scale_cell, exp_cell, prob_cell, gelu_cell = (
        dig("scale_cell"),
        dig("exp_cell"),
        dig("prob_cell"),
        dig("gelu_cell"),
    )
    embed = dig("embed")
    project_r = dig(("matvec", d, d, True, True, False))
    project = dig(("matvec", d, d, True, False, False))
    up, down = (
        dig(("matvec", d, f, True, False, False)),
        dig(("matvec", f, d, True, False, False)),
    )
    lm = dig(("matvec", d, vocab, False, False, True))
    dot_rounded, dot_biased = dig(("dot", d, True, True)), dig(("dot", d, True, False))
    dot_down, dot_lm = dig(("dot", f, True, False)), dig(("dot", d, False, False))
    dot_embed = dig(("dot", shape.vocab_padded, False, False))
    score = dig(("dot", dh, False, False))

    def norm_addresses(frame: Frame, prefix: str, rows: slice | int) -> None:
        out[f"{prefix}.mean"][rows] = _outputs(frame, ln_mean)[0]
        out[f"{prefix}.center"][rows] = _outputs(frame, sub_cell)
        out[f"{prefix}.rstd"][rows] = _outputs(frame, ln_var)[0]
        out[f"{prefix}.out"][rows] = _outputs(frame, ln_out)

    p0 = 0
    for step in range(max_new):
        forward_frame = request.child(step, 0)
        m = prompt if step == 0 else 1
        rows = slice(p0, p0 + m)
        widened = _outputs(forward_frame, widen_cell)
        n32[step], scale32[step] = widened[0], widened[1]
        if step == 0:
            # the prompt tokens are the ``in`` gates: one repeat of the input cell; a
            # decode step's token is the previous forward's ``argmax.token`` gate
            inputs = compiled.index.inputs()
            tokens[rows] = [
                inputs.unrank(forward_frame.input_before + i) for i in range(m)
            ]
        for i, embed_frame in enumerate(_copies(forward_frame, embed)):
            onehot[p0 + i] = _outputs(embed_frame, eq_cell)
            emb[p0 + i] = _outputs(embed_frame, dot_embed)
            wpe32[p0 + i] = _outputs(embed_frame, widen_cell)
            x0[p0 + i] = _outputs(embed_frame, add_cell)
        layer_key = ("layer", m, p0)
        layer_frames = _copies(forward_frame, dig(layer_key))
        assert len(layer_frames) == shape.layers
        for layer, layer_frame in enumerate(layer_frames):
            prefix = f"L{layer}."
            norms = _copies(layer_frame, layer_norm)
            for i in range(m):
                norm_addresses(norms[i], f"{prefix}ln1", p0 + i)
                norm_addresses(norms[m + i], f"{prefix}ln2", p0 + i)
            projections = _copies(layer_frame, project_r)
            for which, name in enumerate(("q", "k", "v")):
                for i in range(m):
                    out[f"{prefix}{name}32"][p0 + i] = _outputs(
                        projections[which * m + i], dot_rounded, back=2
                    )
                    out[f"{prefix}{name}"][p0 + i] = _outputs(
                        projections[which * m + i], dot_rounded
                    )
            for i in range(m):
                pos = p0 + i
                c = pos + 1
                mix_dot = dig(("dot", padded(c), False, True))
                head_frames = _copies(layer_frame, dig(("attend_head", p0, i + 1)))
                assert len(head_frames) == heads
                for head, head_frame in enumerate(head_frames):
                    out[f"{prefix}scores"][head, pos, :c] = _outputs(head_frame, score)
                    out[f"{prefix}u"][head, pos, :c] = _outputs(head_frame, scale_cell)
                    if c >= 2:
                        out[f"{prefix}m"][head, pos] = _outputs(
                            head_frame, dig(("softmax_max", c))
                        )[0]
                        out[f"{prefix}S"][head, pos] = _outputs(
                            head_frame, dig(("softmax_sum", c))
                        )[0]
                    out[f"{prefix}e"][head, pos, :c] = _outputs(head_frame, exp_cell)
                    out[f"{prefix}p"][head, pos, :c] = _outputs(head_frame, prob_cell)
                    cols = slice(head * dh, (head + 1) * dh)
                    out[f"{prefix}mix32"][pos, cols] = _outputs(
                        head_frame, mix_dot, back=2
                    )
                    out[f"{prefix}mix"][pos, cols] = _outputs(head_frame, mix_dot)
            for i, proj_frame in enumerate(_copies(layer_frame, project)):
                out[f"{prefix}proj"][p0 + i] = _outputs(proj_frame, dot_biased)
            adds = _outputs(layer_frame, add_cell)
            out[f"{prefix}x1"][rows] = adds[: m * d].reshape(m, d)
            out[f"{prefix}x2"][rows] = adds[m * d :].reshape(m, d)
            for i, up_frame in enumerate(_copies(layer_frame, up)):
                out[f"{prefix}fc"][p0 + i] = _outputs(up_frame, dot_biased)
            gelus = _outputs(layer_frame, gelu_cell)
            out[f"{prefix}gelu32"][rows] = (gelus - 1).reshape(m, f)
            out[f"{prefix}gelu"][rows] = gelus.reshape(m, f)
            for i, down_frame in enumerate(_copies(layer_frame, down)):
                out[f"{prefix}mlp"][p0 + i] = _outputs(down_frame, dot_down)
        final_norm = _copies(forward_frame, layer_norm)
        assert len(final_norm) == 1
        norm_addresses(final_norm[0], "lnf", step)
        lm_frames = _copies(forward_frame, lm)
        assert len(lm_frames) == 1
        logits[step] = _outputs(lm_frames[0], dot_lm)
        block_frames: list[Frame] = []
        for size in dict.fromkeys(blocks):
            block_frames.extend(_copies(forward_frame, dig(("argmax_block", size))))
        assert len(block_frames) == len(blocks)
        for b, block_frame in enumerate(block_frames):
            best[step, b] = block_frame.base + block_frame.definition.size - 2
            idx[step, b] = block_frame.base + block_frame.definition.size - 1
        if len(blocks) == 1 and shape.vocab <= shape.argmax_block:
            token[step] = idx[step, 0]
        else:
            tops = _copies(
                forward_frame,
                dig(("argmax_top", len(blocks) + int(sum(blocks) < vocab))),
            )
            assert len(tops) == 1
            token[step] = tops[0].base + tops[0].definition.size - 1
        p0 += m
    return out


# -- values by address ------------------------------------------------------------------------


class SparseValues(Mapping[int, int]):
    """Values at a sparse set of addresses (the recorded words), plus every weight by rank and the inputs.

    Lookup is a binary search in the sorted recorded addresses, then the
    weight interval (rank = address - first weight address, the ``weights``
    replay unit being one contiguous run of ``weight`` cells), then the
    input gates.  Absent addresses raise ``KeyError`` so that
    :func:`veritor.protocol.replay_unit` can tell a boundary value from an
    interior one.
    """

    def __init__(
        self,
        compiled: Compiled,
        recorded: dict[str, np.ndarray],
        addresses: dict[str, np.ndarray],
        weights: np.ndarray,
        inputs: Sequence[int],
    ) -> None:
        pairs_a: list[np.ndarray] = []
        pairs_v: list[np.ndarray] = []
        for name, address in addresses.items():
            value = recorded[name]
            if value.shape != address.shape:
                raise ValueError(
                    f"{name}: recorded shape {value.shape} != address shape {address.shape}"
                )
            mask = address >= 0
            pairs_a.append(address[mask].reshape(-1))
            words = value.view(np.uint32) if value.dtype == np.float32 else value
            pairs_v.append(words[mask].reshape(-1).astype(np.int64))
        flat_a = np.concatenate(pairs_a) if pairs_a else np.zeros(0, dtype=np.int64)
        flat_v = np.concatenate(pairs_v) if pairs_v else np.zeros(0, dtype=np.int64)
        order = np.argsort(flat_a, kind="stable")
        self._addresses = flat_a[order]
        self._values = flat_v[order]
        duplicates = np.flatnonzero(np.diff(self._addresses) == 0)
        if duplicates.size:
            raise ValueError(
                f"address {int(self._addresses[duplicates[0]])} recorded twice"
            )
        weight_domain = compiled.index.weights()
        self._weight_start = weight_domain.unrank(0) if weight_domain.count else 0
        self._weights = np.asarray(weights, dtype=np.uint16)
        if self._weights.shape != (compiled.index.weight_count,):
            raise ValueError("one word per weight gate is required")
        input_domain = compiled.index.inputs()
        self._inputs = {
            input_domain.unrank(rank): int(value) for rank, value in enumerate(inputs)
        }
        self.compiled = compiled

    def __getitem__(self, address: int) -> int:
        start = self._weight_start
        if start <= address < start + self._weights.shape[0]:
            return int(self._weights[address - start])
        position = int(np.searchsorted(self._addresses, address))
        if position < self._addresses.shape[0] and self._addresses[position] == address:
            return int(self._values[position])
        try:
            return self._inputs[address]
        except KeyError:
            raise KeyError(address) from None

    def __contains__(self, address: object) -> bool:
        if type(address) is not int:
            return False
        try:
            self[address]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield from self._inputs
        yield from (int(a) for a in self._addresses)
        yield from range(
            self._weight_start, self._weight_start + self._weights.shape[0]
        )

    def __len__(self) -> int:
        return len(self._inputs) + self._addresses.shape[0] + self._weights.shape[0]

    @property
    def recorded_count(self) -> int:
        return int(self._addresses.shape[0])


def evaluate_unit(
    compiled: Compiled, node: IndexNode, values: Mapping[int, int]
) -> dict[int, int]:
    """Re-execute a unit: evaluate its non-source gates in address order from ``values`` (its inputs)."""

    circuit = compiled.circuit
    known: dict[int, int] = {}
    for address in node.interval:
        ref = circuit[address]
        if ref.is_source:
            continue
        args = [known[a] if a in known else values[a] for a in ref.args]
        known[address] = circuit.evaluate_gate(address, args)
    return known


def check_unit(
    compiled: Compiled, node: IndexNode, values: Mapping[int, int]
) -> tuple[int, int]:
    """Check every non-source gate of a unit against recorded values where they exist.

    Returns ``(checked, agreeing)``: a gate is checked when ``values`` holds
    its output (recorded words) or when it was re-executed from recorded
    inputs; it agrees when the re-executed value equals the recorded one and
    the gate relation holds on the recorded words.
    """

    circuit = compiled.circuit
    known: dict[int, int] = {}
    checked = agreeing = 0
    for address in node.interval:
        ref = circuit[address]
        if ref.is_source:
            continue
        args = [known[a] if a in known else values[a] for a in ref.args]
        value = circuit.evaluate_gate(address, args)
        known[address] = value
        if address in values:
            checked += 1
            recorded = values[address]
            if recorded == value and circuit.check_gate(address, args, recorded):
                agreeing += 1
    return checked, agreeing


__all__ = [
    "Capture",
    "GPT2Weights",
    "GemmBackend",
    "LayerWeights",
    "PythonGemm",
    "Run",
    "SparseValues",
    "address_map",
    "argmax_blocks",
    "as_bf16_words",
    "check_unit",
    "evaluate_unit",
    "f32_add",
    "forward",
    "request_frames",
    "tournament",
    "tree_reduce",
    "words_as_f32",
]
