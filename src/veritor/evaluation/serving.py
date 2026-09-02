"""An honest inference server as a table of kinds, at any dimensions.

``Bound``, ``Cost`` and ``expected_work`` fold over the per-kind table of an
index and nothing else, so a serving run can be priced without being traced:
this module writes the :class:`~veritor.core.KindTable` the compiler would
profile for the toy decoder of :mod:`veritor.constructors.lm` serving a batch
of identical requests, at dimensions of the caller's choice and under each of
the partitions an honest server might mark.  At toy dimensions the tables
coincide with the compiled ones (``tests/veritor/evaluation``); at frontier
dimensions (``d_model = 8192``, contexts in the thousands) the description
would not fit, which is what the table is for.

The structure is the toy's, gate for gate: ``dot_k`` is ``k`` products and a
sum tree, ``attend_head_c`` is ``c`` scores, their squares, the mix and the
shift, a token's layer is four projections, the heads, two residual sums and
the square MLP, a request is a prefill over its prompt then one decode step
per generated token, and the run is the weights unit followed by the
requests.  What varies is where the marks go:

* the *replay level* -- ``request`` (one unit per request), ``step`` (one
  unit per synchronous decode step of a batch, the layout of
  :class:`~veritor.constructors.cluster.ClusterG`), ``layer`` (one token's
  layer), ``matvec`` (one matrix-vector product, or one attention head),
  ``row`` (one dot product, one attention head, one one-hot, one argmax, one
  block of residual or square cells) or ``cell`` (every unit has one
  output: a dot product or a single gate, so no unit's cut exceeds a word);
* the *verification level* -- ``layer``, ``row`` (the toy's marks: dots,
  heads, the one-hot, the argmax, the residual and square cells) or ``gate``
  (every gate its own unit).

Every gate lies in exactly one unit of each level, nothing marked nests in a
verification unit, and a kind that would sit above one mark and below the
other (a token's layer under ``request``, say) is defined only when one of
the two levels needs it, so the ``request`` and ``step`` tables with ``row``
units are the toy's own hierarchy.  Source gates (the prompt tokens and the
weights) are always verification units, as the tracer marks them; the
weights are one replay unit, and the prompt tokens sit in the request or
step that reads them or, below those levels, in a ``prompt`` unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from veritor.core import Digest, KindSummary, KindTable, identity_digest
from veritor.core.description import REPLAY, VERIFICATION

ReplayLevel = Literal["request", "step", "layer", "matvec", "row", "cell"]
VerificationLevel = Literal["layer", "row", "gate"]

REPLAY_LEVELS: tuple[ReplayLevel, ...] = ("request", "step", "layer", "matvec", "row", "cell")
VERIFICATION_LEVELS: tuple[VerificationLevel, ...] = ("layer", "row", "gate")

_COARSENESS = {"gate": 0, "cell": 1, "row": 2, "matvec": 3, "layer": 4, "step": 5, "request": 5}

_REPLAY_COST = {"add": 1, "sub": 1, "mul": 2, "lt": 1, "eq": 1, "shr": 1, "in": 0, "weight": 0}
_PROOF_COST = {"add": 1, "sub": 1, "mul": 2, "lt": 1, "eq": 1, "shr": 1, "in": 1, "weight": 1}
_ARITY = {"add": 2, "sub": 2, "mul": 2, "lt": 2, "eq": 2, "shr": 2, "square": 1}
_OP = {name: name for name in _ARITY} | {"square": "mul"}  # ``square`` is ``mul(x, x)``

# where a kind is being defined: outside every unit, inside a replay unit, inside a verification unit
_FREE, _IN_REPLAY, _IN_VERIFICATION = "free", "replay", "verification"

TABLE_DIGEST_TAG = "veritor/evaluation/serving-table/v1"


@dataclass(frozen=True, slots=True)
class ServingShape:
    """A run of the toy decoder: its dimensions and a batch of identical requests.

    ``hidden = hidden_multiplier * d_model`` (the toy uses ``2``).  Every
    request has ``prompt`` tokens and generates ``generated`` tokens, the
    first from its prefill; ``batch`` requests share a decode step at the
    ``step`` level, so ``requests`` must be a multiple of it.
    """

    vocab: int
    d_model: int
    heads: int
    layers: int
    prompt: int
    generated: int
    requests: int
    batch: int = 1
    width: int = 16
    hidden_multiplier: int = 2

    def __post_init__(self) -> None:
        for name in (
            "vocab",
            "d_model",
            "heads",
            "layers",
            "prompt",
            "generated",
            "requests",
            "batch",
            "width",
            "hidden_multiplier",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab < 2:
            raise ValueError("vocab must be at least 2")
        if self.d_model % self.heads:
            raise ValueError("d_model must be a multiple of heads")
        if self.requests % self.batch:
            raise ValueError("requests must be a multiple of batch")

    @property
    def d_head(self) -> int:
        return self.d_model // self.heads

    @property
    def hidden(self) -> int:
        return self.hidden_multiplier * self.d_model

    @property
    def context(self) -> int:
        return self.prompt + self.generated

    @property
    def layer_weights(self) -> int:
        d = self.d_model
        return 4 * d * d + 2 * d * self.hidden

    @property
    def weight_count(self) -> int:
        """Every matrix, the constant table and the shift, as in ``LMShape.weight_count``."""

        d, vocab = self.d_model, self.vocab
        return vocab * d + self.layers * self.layer_weights + d * vocab + vocab + 1

    def state_size(self, positions: int) -> int:
        return 2 * self.layers * positions * self.d_model

    @property
    def input_count(self) -> int:
        return self.requests * self.prompt

    @property
    def output_count(self) -> int:
        return self.requests * self.generated

    @property
    def manifest(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "d_model": self.d_model,
            "generated": self.generated,
            "heads": self.heads,
            "hidden_multiplier": self.hidden_multiplier,
            "layers": self.layers,
            "prompt": self.prompt,
            "requests": self.requests,
            "vocab": self.vocab,
            "width": self.width,
        }


@dataclass(slots=True)
class _Row:
    """One kind under construction: what :class:`KindSummary` needs, before copies."""

    key: str
    role: str | None
    input_count: int
    out_count: int
    size: int = 0
    replay_cost: int = 0
    proof_cost: int = 0
    source_inputs: int = 0
    source_weights: int = 0
    children: dict[str, int] = field(default_factory=dict)
    verification_units: int = 0
    verification_kinds: dict[str, int] = field(default_factory=dict)


class _Builder:
    """The kinds of one run under one pair of levels, defined once each by key."""

    def __init__(self, shape: ServingShape, replay: ReplayLevel, verification: VerificationLevel) -> None:
        self.shape = shape
        self.ru = replay
        self.vu = verification
        self.rows: dict[str, _Row] = {}

    # -- roles -----------------------------------------------------------------------

    def role(self, levels: tuple[str, ...], ctx: str) -> str | None:
        """The mark of a kind at ``levels`` defined in context ``ctx``."""

        if ctx == _IN_VERIFICATION:
            return None
        if ctx == _FREE and self.ru in levels:
            return REPLAY
        if self.vu in levels:
            return VERIFICATION
        return None

    @staticmethod
    def within(role: str | None, ctx: str) -> str:
        """The context of the children of a kind with ``role`` defined in ``ctx``."""

        if role == VERIFICATION:
            return _IN_VERIFICATION
        if role == REPLAY:
            return _IN_REPLAY
        return ctx

    # -- definitions -----------------------------------------------------------------

    def define(
        self,
        key: tuple[object, ...],
        role: str | None,
        *,
        ports: int,
        outputs: int,
        gates: dict[str, int] | None = None,
        calls: dict[str, int] | None = None,
    ) -> str:
        """Define (or reuse) the kind ``key`` with its own ``gates`` and its ``calls``."""

        name = repr(key)
        if name in self.rows:
            return name
        row = _Row(name, role, ports, outputs)
        for gate, count in (gates or {}).items():
            row.size += count
            row.replay_cost += count * _REPLAY_COST[gate]
            row.proof_cost += count * _PROOF_COST[gate]
            if gate == "in":
                row.source_inputs += count
            elif gate == "weight":
                row.source_weights += count
        for child, count in (calls or {}).items():
            if count <= 0:
                continue
            sub = self.rows[child]
            row.size += count * sub.size
            row.replay_cost += count * sub.replay_cost
            row.proof_cost += count * sub.proof_cost
            row.source_inputs += count * sub.source_inputs
            row.source_weights += count * sub.source_weights
            row.children[child] = row.children.get(child, 0) + count
            if role != VERIFICATION:
                row.verification_units += count * sub.verification_units
                for kind, inner in sub.verification_kinds.items():
                    row.verification_kinds[kind] = row.verification_kinds.get(kind, 0) + count * inner
        if role == VERIFICATION:
            row.verification_units = 1
            row.verification_kinds = {name: 1}
        self.rows[name] = row
        return name

    @staticmethod
    def merge(*parts: dict[str, int]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for part in parts:
            for kind, count in part.items():
                merged[kind] = merged.get(kind, 0) + count
        return merged

    # -- cells -----------------------------------------------------------------------

    def pair(self, gate: str, ctx: str) -> str:
        """The toy's one-gate building block: unmarked inside a unit, else a unit of one gate.

        Outside every unit (the ``cell`` level) it is a one-gate replay unit
        holding its one-gate verification unit.
        """

        if ctx == _FREE:
            assert self.ru == "cell", "a bare gate outside every unit"
            inner = self.pair(gate, _IN_REPLAY)
            return self.define(("cell_unit", gate), REPLAY, ports=_ARITY[gate], outputs=1, calls={inner: 1})
        role = None if ctx == _IN_VERIFICATION else VERIFICATION
        return self.define(("pair", gate, role), role, ports=_ARITY[gate], outputs=1, gates={_OP[gate]: 1})

    def source(self, gate: str, ctx: str) -> str:
        """An ``in`` or ``weight`` cell: always a verification unit, its one output pinned."""

        assert ctx == _IN_REPLAY, "source cells lie directly in a replay unit"
        return self.define(("source", gate), VERIFICATION, ports=0, outputs=0, gates={gate: 1})

    def cells(self, gate: str, count: int, ctx: str) -> dict[str, int]:
        """``count`` residual ``add_cell`` or MLP ``square_cell`` gates.

        The toy's marked one-gate kinds inside a replay unit; unmarked inside
        a verification unit; where the layer is not a unit and nothing above
        them is, one block unit of them all (``row``) or a unit each (``cell``).
        """

        if ctx == _FREE and self.ru == "cell":
            ((cell, _),) = self.cells(gate, 1, _IN_REPLAY).items()
            unit = self.define((gate + "_cell_unit",), REPLAY, ports=_ARITY[gate], outputs=1, calls={cell: 1})
            return {unit: count}
        if ctx == _FREE:
            cell = self.cells(gate, count, _IN_REPLAY)
            block = self.define(
                (gate + "_block", count), REPLAY, ports=_ARITY[gate] * count, outputs=count, calls=cell
            )
            return {block: 1}
        role = None if ctx == _IN_VERIFICATION else VERIFICATION
        cell = self.define((gate + "_cell", role), role, ports=_ARITY[gate], outputs=1, gates={_OP[gate]: 1})
        return {cell: count}

    # -- rows ------------------------------------------------------------------------

    def dot(self, k: int, ctx: str) -> str:
        """``k`` products and a pairwise sum tree; the carries are the definition's own gates."""

        role = self.role(("row", "cell"), ctx)
        sub = self.within(role, ctx)
        mul, add = self.pair("mul", sub), self.pair("add", sub)
        calls = {mul: k, add: 0}
        level, carried = k, 0
        while level > 1:
            if level % 2:
                carried += 1
            calls[add] += level // 2
            level //= 2
        gates: dict[str, int] = {}
        if sub == _IN_VERIFICATION:
            gates["add"] = carried  # the toy adds the carries directly
        else:
            calls[add] += carried  # every gate must lie in a verification unit
        return self.define(("dot", k, role), role, ports=2 * k, outputs=1, gates=gates, calls=calls)

    def onehot(self, ctx: str) -> str:
        role = self.role(("row",), ctx)
        eq = self.pair("eq", self.within(role, ctx))
        vocab = self.shape.vocab
        return self.define(("onehot", role), role, ports=1 + vocab, outputs=vocab, calls={eq: vocab})

    def attend_head(self, c: int, ctx: str) -> str:
        """One head over ``c`` positions: ``c`` scores, ``c`` squares, ``dh`` mixes, ``dh`` shifts."""

        role = self.role(("row", "matvec"), ctx)
        sub = self.within(role, ctx)
        dh = self.shape.d_head
        calls = self.merge(  # the score and mix dots are one kind when ``c == dh``
            {self.dot(dh, sub): c},
            {self.pair("square", sub): c},
            {self.dot(c, sub): dh},
            {self.pair("shr", sub): dh},
        )
        return self.define(
            ("attend_head", c, role), role, ports=dh + 2 * c * dh + 1, outputs=dh, calls=calls
        )

    def argmax(self, ctx: str) -> str:
        """The chain of ``vocab - 1`` selects, seven gates each: the toy's own gates when marked."""

        role = self.role(("row",), ctx)
        sub = self.within(role, ctx)
        vocab = self.shape.vocab
        counts = {"lt": vocab - 1, "sub": 2 * (vocab - 1), "mul": 2 * (vocab - 1), "add": 2 * (vocab - 1)}
        if sub == _IN_VERIFICATION:
            gates, calls = counts, {}
        else:
            gates, calls = {}, {self.pair(gate, sub): count for gate, count in counts.items()}
        return self.define(("argmax", role), role, ports=2 * vocab, outputs=1, gates=gates, calls=calls)

    # -- matvec-level -----------------------------------------------------------------

    def matvec(self, k: int, m: int, ctx: str) -> str:
        role = self.role(("matvec",), ctx)
        dot = self.dot(k, self.within(role, ctx))
        return self.define(("matvec", k, m, role), role, ports=k + k * m, outputs=m, calls={dot: m})

    def embed_row(self, ctx: str) -> str:
        """A token to its embedding: one-hot then ``E``; a unit at the ``matvec`` and ``layer`` levels."""

        role = self.role(("matvec", "layer"), ctx)
        sub = self.within(role, ctx)
        vocab, d = self.shape.vocab, self.shape.d_model
        calls = {self.onehot(sub): 1, self.matvec(vocab, d, sub): 1}
        return self.define(("embed_row", role), role, ports=1 + vocab + vocab * d, outputs=d, calls=calls)

    # -- layers -----------------------------------------------------------------------

    def layer_calls(self, positions: int, cached: int, ctx: str) -> dict[str, int]:
        """One layer over ``positions`` new positions attending to ``cached`` earlier ones."""

        shape = self.shape
        d, hidden, heads = shape.d_model, shape.hidden, shape.heads
        return self.merge(
            {self.matvec(d, d, ctx): 3 * positions},
            *({self.attend_head(cached + p + 1, ctx): heads} for p in range(positions)),
            {self.matvec(d, d, ctx): positions},
            self.cells("add", positions * d, ctx),
            {self.matvec(d, hidden, ctx): positions},
            self.cells("square", positions * hidden, ctx),
            {self.matvec(hidden, d, ctx): positions},
            self.cells("add", positions * d, ctx),
        )

    def layers(self, positions: int, cached: int, ctx: str) -> dict[str, int]:
        """All layers: inlined calls as in the toy, or ``layers`` calls of a layer kind."""

        shape = self.shape
        if "layer" not in (self.ru, self.vu):
            calls = self.layer_calls(positions, cached, ctx)
            return {kind: shape.layers * count for kind, count in calls.items()}
        role = self.role(("layer",), ctx)
        d = shape.d_model
        layer = self.define(
            ("layer", positions, cached, role),
            role,
            ports=positions * d + 2 * cached * d + shape.layer_weights,
            outputs=3 * positions * d,
            calls=self.layer_calls(positions, cached, self.within(role, ctx)),
        )
        return {layer: shape.layers}

    def head(self, ctx: str) -> dict[str, int]:
        """The unembedding and the argmax: inlined as in the toy, or one ``lm_head`` unit."""

        shape = self.shape
        d, vocab = shape.d_model, shape.vocab
        if self.ru not in ("matvec", "layer") and self.vu != "layer":
            return {self.matvec(d, vocab, ctx): 1, self.argmax(ctx): 1}
        role = self.role(("matvec", "layer"), ctx)
        sub = self.within(role, ctx)
        head = self.define(
            ("lm_head", role),
            role,
            ports=d + d * vocab + vocab,
            outputs=1,
            calls={self.matvec(d, vocab, sub): 1, self.argmax(sub): 1},
        )
        return {head: 1}

    def prompt(self, ctx: str) -> dict[str, int]:
        """The prompt tokens: ``in`` cells in the request or step, else a ``prompt`` unit."""

        n = self.shape.prompt
        if ctx == _IN_REPLAY:
            return {self.source("in", ctx): n}
        cell = self.source("in", _IN_REPLAY)
        return {self.define(("prompt", n), REPLAY, ports=0, outputs=0, calls={cell: n}): 1}

    def prefill(self, ctx: str) -> str:
        """The prompt: its tokens, their embeddings, the layers, the first generated token."""

        shape = self.shape
        n = shape.prompt
        calls = self.merge(
            self.prompt(ctx),
            {self.embed_row(ctx): n},
            self.layers(n, 0, ctx),
            self.head(ctx),
        )
        return self.define(
            ("prefill", n), None, ports=shape.weight_count, outputs=shape.state_size(n) + 1, calls=calls
        )

    def decode(self, c: int, ctx: str) -> str:
        """One token at context ``c`` over a cache of ``c - 1``: embedding, layers, next token."""

        shape = self.shape
        calls = self.merge({self.embed_row(ctx): 1}, self.layers(1, c - 1, ctx), self.head(ctx))
        return self.define(
            ("decode", c),
            None,
            ports=shape.weight_count + 1 + shape.state_size(c - 1),
            outputs=shape.state_size(1) + 1,
            calls=calls,
        )

    # -- the run ------------------------------------------------------------------------

    def weights(self) -> str:
        cell = self.source("weight", _IN_REPLAY)
        return self.define(("weights",), REPLAY, ports=0, outputs=0, calls={cell: self.shape.weight_count})

    def request(self) -> str:
        """A request: its prefill then a decode step per further token; the unit at ``request``."""

        shape = self.shape
        role = REPLAY if self.ru == "request" else None
        ctx = self.within(role, _FREE)
        calls = {self.prefill(ctx): 1}
        for c in range(shape.prompt + 1, shape.prompt + shape.generated):
            calls[self.decode(c, ctx)] = 1
        return self.define(
            ("request", shape.prompt, shape.generated),
            role,
            ports=shape.weight_count,
            outputs=shape.generated,
            calls=calls,
        )

    def steps(self) -> dict[str, int]:
        """The ``step`` layout: a prefill step then a decode step per token, per wave of ``batch`` requests."""

        shape = self.shape
        b, waves = shape.batch, shape.requests // shape.batch
        prefill = self.define(
            ("prefill_step", b),
            REPLAY,
            ports=shape.weight_count,
            outputs=b * (shape.state_size(shape.prompt) + 1),
            calls={self.prefill(_IN_REPLAY): b},
        )
        calls = {prefill: waves}
        for c in range(shape.prompt + 1, shape.prompt + shape.generated):
            step = self.define(
                ("decode_step", c, b),
                REPLAY,
                ports=shape.weight_count + b * (1 + shape.state_size(c - 1)),
                outputs=b * (shape.state_size(1) + 1),
                calls={self.decode(c, _IN_REPLAY): b},
            )
            calls[step] = waves
        return calls

    def root(self) -> str:
        shape = self.shape
        calls = {self.weights(): 1}
        if self.ru == "step":
            calls.update(self.steps())
        else:
            calls[self.request()] = shape.requests
        return self.define(("root",), None, ports=0, outputs=shape.output_count, calls=calls)

    # -- the table -----------------------------------------------------------------------

    def table(self) -> KindTable:
        root = self.root()
        copies = {root: 1}
        min_depth = {root: 0}
        max_depth = {root: 0}
        for name in self.parents_first(root):
            row = self.rows[name]
            for child, count in row.children.items():
                copies[child] = copies.get(child, 0) + copies[name] * count
                depth = min_depth[name] + 1
                min_depth[child] = min(min_depth.get(child, depth), depth)
                depth = max_depth[name] + 1
                max_depth[child] = max(max_depth.get(child, depth), depth)
        width = self.shape.width
        rows = tuple(
            KindSummary(
                kind=row.key,
                role=row.role,
                copies=copies[row.key],
                size=row.size,
                replay_cost=row.replay_cost,
                proof_cost=row.proof_cost,
                input_count=row.input_count,
                out_count=row.out_count,
                out_bits=row.out_count * width,
                source_inputs=row.source_inputs,
                source_weights=row.source_weights,
                min_depth=min_depth[row.key],
                max_depth=max_depth[row.key],
                children=tuple(row.children.items()),
                verification_units=row.verification_units,
                verification_kinds=tuple(row.verification_kinds.items()),
            )
            for row in (self.rows[name] for name in self.preorder(root))
        )
        top = self.rows[root]
        table = KindTable(
            rows=rows,
            root=root,
            n=top.size,
            input_count=top.source_inputs,
            weight_count=top.source_weights,
            replay_unit_count=sum(row.copies for row in rows if row.role == REPLAY),
            digest=Digest(
                identity_digest(
                    TABLE_DIGEST_TAG,
                    {"shape": self.shape.manifest, "replay": self.ru, "verification": self.vu},
                )
            ),
        )
        _check_tiling(table)
        return table

    def preorder(self, root: str) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()
        pending = [root]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            order.append(name)
            pending.extend(reversed(list(self.rows[name].children)))
        return order

    def parents_first(self, root: str) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            for child in self.rows[name].children:
                visit(child)
            order.append(name)

        visit(root)
        return order[::-1]


def _check_tiling(table: KindTable) -> None:
    """Every gate in exactly one replay unit and one verification unit, by counts."""

    for role in (REPLAY, VERIFICATION):
        covered = sum(row.copies * row.size for row in table.rows if row.role == role)
        if covered != table.n:
            raise ValueError(f"{role} units cover {covered} of {table.n} gates")
    inside = sum(row.copies * row.verification_units for row in table.rows if row.role == REPLAY)
    units = sum(row.copies for row in table.rows if row.role == VERIFICATION)
    if inside != units:
        raise ValueError("verification units do not refine the replay units")


def serving_table(
    shape: ServingShape, replay: ReplayLevel = "request", verification: VerificationLevel = "row"
) -> KindTable:
    """The table of ``shape`` served with replay units at ``replay`` and verification units at ``verification``.

    The verification level must be strictly finer than the replay level.
    """

    if replay not in REPLAY_LEVELS:
        raise ValueError(f"replay level must be one of {REPLAY_LEVELS}")
    if verification not in VERIFICATION_LEVELS:
        raise ValueError(f"verification level must be one of {VERIFICATION_LEVELS}")
    if _COARSENESS[verification] >= _COARSENESS[replay]:
        raise ValueError("verification units must be strictly finer than replay units")
    return _Builder(shape, replay, verification).table()


def partitions() -> tuple[tuple[ReplayLevel, VerificationLevel], ...]:
    """Every admissible pair of levels, coarsest replay first."""

    return tuple(
        (replay, verification)
        for replay in REPLAY_LEVELS
        for verification in VERIFICATION_LEVELS
        if _COARSENESS[verification] < _COARSENESS[replay]
    )


__all__ = [
    "REPLAY_LEVELS",
    "VERIFICATION_LEVELS",
    "ReplayLevel",
    "ServingShape",
    "VerificationLevel",
    "partitions",
    "serving_table",
]
