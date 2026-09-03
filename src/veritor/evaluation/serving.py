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
  heads, the one-hot, the argmax, the residual and square cells), ``cell``
  (what ``cell`` denotes for replay: a dot product or a single gate per
  unit) or ``gate`` (every gate its own unit).

The verification level must be strictly finer than the replay level
(``gate < cell < row < matvec < layer < step = request``), so a ``cell``
replay unit takes ``gate`` verification units.  Every gate lies in exactly
one unit of each level, nothing marked nests in a verification unit, and a
kind that would sit above one mark and below the other (a token's layer
under ``request``, say) is defined only when one of the two levels needs
it, so the ``request`` and ``step`` tables with ``row`` units are the toy's
own hierarchy.  Source gates (the prompt tokens and the weights) are always
verification units, as the tracer marks them; the weights are one replay
unit, and the prompt tokens sit in the request or step that reads them or,
below those levels, in a ``prompt`` unit.

Each kind also records whether it is *closed* (every port fed a source gate
at every call site, see :attr:`~veritor.core.KindSummary.closed`), derived
from the builder's own wiring: a kind's ports are the weights it is handed
and the activations it reads, and every call site says where each group
comes from.  The weights, the root, a request, a prefill and a prefill step
are closed; anything reading a token, an activation or the cache is not.

Each kind also records its *reach* (:attr:`~veritor.core.KindSummary.reach_bits`):
the width of the circuit outputs a copy can influence, the third downstream
cut ``Bound`` may charge a replay unit (RU) or a verification unit (VU) in
place of its interface.  The builder
reproduces what :meth:`~veritor.core.Index.kinds` computes for the compiled
toy at *step granularity* (a call is one node: any input to it may reach
any of its outputs), from the dataflow of the run.  At the ``request``
level the root's outputs are the requests' tokens and requests never read
each other, so a request reaches exactly its own ``generated`` tokens, and
so does everything inside it -- a decode step, a layer, a matvec or a dot
whose interface is far wider.  At the ``step`` level the steps of one wave
of ``batch`` requests are chained through the tokens and the cache: a step
reaches the tokens of every step from itself to the end of its wave, the
prefill step the whole wave's, and the next wave's prefill step reads only
the weights, so nothing reaches across waves.  The weights RU is read by
everything and reaches the whole output.  Tracking dataflow per slot of a
step rather than per step would be a further refinement; in this wave model
the ``batch`` slots of a step advance together, so the two coincide.

Each kind also records the narrowest interface enclosing a copy of it
(:attr:`~veritor.core.KindSummary.ancestor_bits`), the third cut of the
bottleneck ``Bound`` charges, from the same hierarchy: a kind called from a
caller is enclosed by what encloses the caller, narrowed by the caller's own
interface, and keeps the widest such value over its call sites, exactly as
:meth:`~veritor.core.Index.kinds` computes it for the compiled toy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from veritor.core import Digest, KindSummary, KindTable, identity_digest
from veritor.core.description import REPLAY, VERIFICATION

ReplayLevel = Literal["request", "step", "layer", "matvec", "row", "cell"]
VerificationLevel = Literal["layer", "row", "cell", "gate"]

REPLAY_LEVELS: tuple[ReplayLevel, ...] = (
    "request",
    "step",
    "layer",
    "matvec",
    "row",
    "cell",
)
VERIFICATION_LEVELS: tuple[VerificationLevel, ...] = ("layer", "row", "cell", "gate")

_COARSENESS = {
    "gate": 0,
    "cell": 1,
    "row": 2,
    "matvec": 3,
    "layer": 4,
    "step": 5,
    "request": 5,
}

_REPLAY_COST = {
    "add": 1,
    "sub": 1,
    "mul": 2,
    "lt": 1,
    "eq": 1,
    "shr": 1,
    "in": 0,
    "weight": 0,
}
_PROOF_COST = {
    "add": 1,
    "sub": 1,
    "mul": 2,
    "lt": 1,
    "eq": 1,
    "shr": 1,
    "in": 1,
    "weight": 1,
}
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


# A kind's ports come in two groups, the weights it is handed (``W``) and the
# activations, tokens and cache entries it reads (``A``); a call site feeds each
# group of the child from the caller's own ``W`` or ``A`` ports, from source
# gates the caller holds (``SOURCE``: the weight cells or the prompt tokens) or
# from values computed in the caller (``COMPUTED``).  That is enough wiring to
# derive ``KindSummary.closed`` the way ``Index.kinds`` does: a group is retained
# iff every call site feeds it source gates or retained ports of the caller,
# and a kind is closed iff both its groups are retained (an empty group is).
W, A, SOURCE, COMPUTED = "w", "a", "source", "computed"
type _Feed = dict[str, frozenset[str]]


def _feed(
    w: str | tuple[str, ...] | None = None, a: str | tuple[str, ...] | None = None
) -> _Feed:
    """What a call site feeds the child's ``W`` and ``A`` groups (``None``: the child has none)."""

    feed: _Feed = {}
    for group, sources in ((W, w), (A, a)):
        if sources is not None:
            feed[group] = frozenset((sources,) if isinstance(sources, str) else sources)
    return feed


@dataclass(slots=True)
class _Row:
    """One kind under construction: what :class:`KindSummary` needs, before copies.

    ``ports`` counts the ``W`` and ``A`` groups; ``feeds`` records, per child
    kind, what this kind feeds each group of the child (united over the call
    sites inside one copy).  ``reaches`` records, per child kind, the bits of
    the circuit output the child's copies here can reach when that is less
    than what this kind reaches; a child not named inherits this kind's
    reach (a call is one node at step granularity).
    """

    key: str
    role: str | None
    ports: dict[str, int]
    out_count: int
    size: int = 0
    replay_cost: int = 0
    proof_cost: int = 0
    source_inputs: int = 0
    source_weights: int = 0
    children: dict[str, int] = field(default_factory=dict)
    feeds: dict[str, _Feed] = field(default_factory=dict)
    reaches: dict[str, int] = field(default_factory=dict)
    verification_units: int = 0
    verification_kinds: dict[str, int] = field(default_factory=dict)
    vu_outputs: int = 0
    """Declared outputs of the verification units inside one copy (its own, for a verification kind)."""
    interior_count: int = 0
    """Interior positions inside one copy: per replay unit inside, its ``vu_outputs`` less its ``out_count``."""

    @property
    def input_count(self) -> int:
        return sum(self.ports.values())


class _Builder:
    """The kinds of one run under one pair of levels, defined once each by key."""

    def __init__(
        self, shape: ServingShape, replay: ReplayLevel, verification: VerificationLevel
    ) -> None:
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
        w: int = 0,
        a: int = 0,
        outputs: int,
        gates: dict[str, int] | None = None,
        calls: dict[str, int] | None = None,
        feeds: dict[str, _Feed] | None = None,
        reaches: dict[str, int] | None = None,
    ) -> str:
        """Define (or reuse) the kind ``key`` with its own ``gates`` and its ``calls``.

        ``w`` and ``a`` are the sizes of its port groups; ``feeds`` says, per
        called kind, what this kind passes to each group of the child (see
        :func:`_feed`) and must name exactly the child's nonempty groups.
        ``reaches`` says, per called kind, how many bits of the circuit
        output the child's outputs here can reach when that is less than
        this kind's own reach (see :class:`_Row`).
        """

        name = repr(key)
        if name in self.rows:
            return name
        row = _Row(name, role, {W: w, A: a}, outputs, reaches=dict(reaches or {}))
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
            groups = {group for group, size in sub.ports.items() if size}
            feed = (feeds or {}).get(child, {})
            if set(feed) != groups:
                raise ValueError(
                    f"{name} feeds {sorted(feed)} of {child}, which has the groups {sorted(groups)}"
                )
            row.size += count * sub.size
            row.replay_cost += count * sub.replay_cost
            row.proof_cost += count * sub.proof_cost
            row.source_inputs += count * sub.source_inputs
            row.source_weights += count * sub.source_weights
            row.children[child] = row.children.get(child, 0) + count
            merged = row.feeds.setdefault(child, {})
            for group, sources in feed.items():
                merged[group] = merged.get(group, frozenset()) | sources
            if role != VERIFICATION:
                row.verification_units += count * sub.verification_units
                for kind, inner in sub.verification_kinds.items():
                    row.verification_kinds[kind] = (
                        row.verification_kinds.get(kind, 0) + count * inner
                    )
                row.vu_outputs += count * sub.vu_outputs
                row.interior_count += count * sub.interior_count
        if role == VERIFICATION:
            row.verification_units = 1
            row.verification_kinds = {name: 1}
            row.vu_outputs = outputs
            row.interior_count = 0
        elif role == REPLAY:
            if row.vu_outputs < outputs:
                raise ValueError(
                    f"{name} declares {outputs} outputs but its verification units declare {row.vu_outputs}"
                )
            row.interior_count = row.vu_outputs - outputs
        self.rows[name] = row
        return name

    @staticmethod
    def feeds(*parts: dict[str, _Feed]) -> dict[str, _Feed]:
        """Unite per-child feeds from several call sites of one kind."""

        merged: dict[str, _Feed] = {}
        for part in parts:
            for child, feed in part.items():
                target = merged.setdefault(child, {})
                for group, sources in feed.items():
                    target[group] = target.get(group, frozenset()) | sources
        return merged

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
            return self.define(
                ("cell_unit", gate),
                REPLAY,
                a=_ARITY[gate],
                outputs=1,
                calls={inner: 1},
                feeds={inner: _feed(a=A)},
            )
        role = None if ctx == _IN_VERIFICATION else VERIFICATION
        return self.define(
            ("pair", gate, role), role, a=_ARITY[gate], outputs=1, gates={_OP[gate]: 1}
        )

    def source(self, gate: str, ctx: str) -> str:
        """An ``in`` or ``weight`` cell: always a verification unit, its one output pinned."""

        assert ctx == _IN_REPLAY, "source cells lie directly in a replay unit"
        return self.define(("source", gate), VERIFICATION, outputs=0, gates={gate: 1})

    def cells(self, gate: str, count: int, ctx: str) -> dict[str, int]:
        """``count`` residual ``add_cell`` or MLP ``square_cell`` gates.

        The toy's marked one-gate kinds inside a replay unit; unmarked inside
        a verification unit; where the layer is not a unit and nothing above
        them is, one block unit of them all (``row``) or a unit each (``cell``).
        Every port is an activation: the caller says where they come from.
        """

        if ctx == _FREE and self.ru == "cell":
            ((cell, _),) = self.cells(gate, 1, _IN_REPLAY).items()
            unit = self.define(
                (gate + "_cell_unit",),
                REPLAY,
                a=_ARITY[gate],
                outputs=1,
                calls={cell: 1},
                feeds={cell: _feed(a=A)},
            )
            return {unit: count}
        if ctx == _FREE:
            cells = self.cells(gate, count, _IN_REPLAY)
            block = self.define(
                (gate + "_block", count),
                REPLAY,
                a=_ARITY[gate] * count,
                outputs=count,
                calls=cells,
                feeds={kind: _feed(a=A) for kind in cells},
            )
            return {block: 1}
        role = None if ctx == _IN_VERIFICATION else VERIFICATION
        cell = self.define(
            (gate + "_cell", role),
            role,
            a=_ARITY[gate],
            outputs=1,
            gates={_OP[gate]: 1},
        )
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
        # a product reads one activation and one weight of the dot; the sums read products
        feeds = {mul: _feed(a=(A, W)), add: _feed(a=COMPUTED)}
        return self.define(
            ("dot", k, role),
            role,
            w=k,
            a=k,
            outputs=1,
            gates=gates,
            calls=calls,
            feeds=feeds,
        )

    def onehot(self, ctx: str) -> str:
        """The token against the constant table: one ``eq`` per vocabulary entry."""

        role = self.role(("row",), ctx)
        eq = self.pair("eq", self.within(role, ctx))
        vocab = self.shape.vocab
        return self.define(
            ("onehot", role),
            role,
            w=vocab,
            a=1,
            outputs=vocab,
            calls={eq: vocab},
            feeds={eq: _feed(a=(A, W))},
        )

    def attend_head(self, c: int, ctx: str) -> str:
        """One head over ``c`` positions: ``c`` scores, ``c`` squares, ``dh`` mixes, ``dh`` shifts.

        The ports are the query, the keys and the values (activations) and
        the shift (a weight); the score dots read the query against the keys,
        the mix dots the squared scores against the values.
        """

        role = self.role(("row", "matvec"), ctx)
        sub = self.within(role, ctx)
        dh = self.shape.d_head
        score, square = self.dot(dh, sub), self.pair("square", sub)
        mix, shift = self.dot(c, sub), self.pair("shr", sub)
        calls = self.merge(
            {score: c}, {square: c}, {mix: dh}, {shift: dh}
        )  # one dot kind when ``c == dh``
        feeds = self.feeds(
            {score: _feed(w=A, a=A)},
            {square: _feed(a=COMPUTED)},
            {mix: _feed(w=A, a=COMPUTED)},
            {shift: _feed(a=(COMPUTED, W))},
        )
        return self.define(
            ("attend_head", c, role),
            role,
            w=1,
            a=dh + 2 * c * dh,
            outputs=dh,
            calls=calls,
            feeds=feeds,
        )

    def argmax(self, ctx: str) -> str:
        """The chain of ``vocab - 1`` selects, seven gates each: the toy's own gates when marked."""

        role = self.role(("row",), ctx)
        sub = self.within(role, ctx)
        vocab = self.shape.vocab
        counts = {
            "lt": vocab - 1,
            "sub": 2 * (vocab - 1),
            "mul": 2 * (vocab - 1),
            "add": 2 * (vocab - 1),
        }
        # the selects compare and mix the logits (activations), the running best and index
        # (computed) and the constant table (weights)
        reads = {
            "lt": (A, COMPUTED),
            "sub": (A, W, COMPUTED),
            "mul": (COMPUTED,),
            "add": (A, W, COMPUTED),
        }
        if sub == _IN_VERIFICATION:
            gates, calls, feeds = counts, {}, {}
        else:
            gates, calls = (
                {},
                {self.pair(gate, sub): count for gate, count in counts.items()},
            )
            feeds = {self.pair(gate, sub): _feed(a=reads[gate]) for gate in counts}
        return self.define(
            ("argmax", role),
            role,
            w=vocab,
            a=vocab,
            outputs=1,
            gates=gates,
            calls=calls,
            feeds=feeds,
        )

    # -- matvec-level -----------------------------------------------------------------

    def matvec(self, k: int, m: int, ctx: str) -> str:
        role = self.role(("matvec",), ctx)
        dot = self.dot(k, self.within(role, ctx))
        return self.define(
            ("matvec", k, m, role),
            role,
            w=k * m,
            a=k,
            outputs=m,
            calls={dot: m},
            feeds={dot: _feed(w=W, a=A)},
        )

    def embed_row(self, ctx: str) -> str:
        """A token to its embedding: one-hot then ``E``; a unit at the ``matvec`` and ``layer`` levels."""

        role = self.role(("matvec", "layer"), ctx)
        sub = self.within(role, ctx)
        vocab, d = self.shape.vocab, self.shape.d_model
        onehot, embed = self.onehot(sub), self.matvec(vocab, d, sub)
        feeds = {onehot: _feed(w=W, a=A), embed: _feed(w=W, a=COMPUTED)}
        return self.define(
            ("embed_row", role),
            role,
            w=vocab + vocab * d,
            a=1,
            outputs=d,
            calls={onehot: 1, embed: 1},
            feeds=feeds,
        )

    # -- layers -----------------------------------------------------------------------

    def layer_calls(
        self, positions: int, cached: int, ctx: str, x: str
    ) -> tuple[dict[str, int], dict[str, dict[str, frozenset[str]]]]:
        """One layer over ``positions`` new positions attending to ``cached`` earlier ones.

        ``x`` says where the layer's input activations come from in the kind
        holding these calls: its own ``A`` ports (a ``layer`` kind) or values
        computed before them (the layers inlined in a prefill or decode); a
        cache, when there is one, is always read from that kind's ``A`` ports.
        """

        shape = self.shape
        d, hidden, heads = shape.d_model, shape.hidden, shape.heads
        project = self.matvec(d, d, ctx)
        attend = {
            self.attend_head(cached + p + 1, ctx): heads for p in range(positions)
        }
        residual = self.cells("add", positions * d, ctx)
        up = self.matvec(d, hidden, ctx)
        square = self.cells("square", positions * hidden, ctx)
        down = self.matvec(hidden, d, ctx)
        calls = self.merge(
            {project: 3 * positions},
            attend,
            {project: positions},
            residual,
            {up: positions},
            square,
            {down: positions},
            residual,
        )
        attended = (COMPUTED, A) if cached else (COMPUTED,)
        feeds = self.feeds(
            {project: _feed(w=W, a=(x, COMPUTED))},
            {head: _feed(w=W, a=attended) for head in attend},
            {kind: _feed(a=(x, COMPUTED)) for kind in residual},
            {up: _feed(w=W, a=COMPUTED)},
            {kind: _feed(a=COMPUTED) for kind in square},
            {down: _feed(w=W, a=COMPUTED)},
        )
        return calls, feeds

    def layers(
        self, positions: int, cached: int, ctx: str
    ) -> tuple[dict[str, int], dict[str, dict[str, frozenset[str]]]]:
        """All layers: inlined calls as in the toy, or ``layers`` calls of a layer kind."""

        shape = self.shape
        if "layer" not in (self.ru, self.vu):
            calls, feeds = self.layer_calls(positions, cached, ctx, COMPUTED)
            return {kind: shape.layers * count for kind, count in calls.items()}, feeds
        role = self.role(("layer",), ctx)
        d = shape.d_model
        calls, feeds = self.layer_calls(positions, cached, self.within(role, ctx), A)
        layer = self.define(
            ("layer", positions, cached, role),
            role,
            w=shape.layer_weights,
            a=positions * d + 2 * cached * d,
            outputs=3 * positions * d,
            calls=calls,
            feeds=feeds,
        )
        return {layer: shape.layers}, {
            layer: _feed(w=W, a=(COMPUTED, A) if cached else COMPUTED)
        }

    def head(
        self, ctx: str
    ) -> tuple[dict[str, int], dict[str, dict[str, frozenset[str]]]]:
        """The unembedding and the argmax: inlined as in the toy, or one ``lm_head`` unit."""

        shape = self.shape
        d, vocab = shape.d_model, shape.vocab
        if self.ru not in ("matvec", "layer") and self.vu != "layer":
            unembed, argmax = self.matvec(d, vocab, ctx), self.argmax(ctx)
            return {unembed: 1, argmax: 1}, {
                unembed: _feed(w=W, a=COMPUTED),
                argmax: _feed(w=W, a=COMPUTED),
            }
        role = self.role(("matvec", "layer"), ctx)
        sub = self.within(role, ctx)
        unembed, argmax = self.matvec(d, vocab, sub), self.argmax(sub)
        head = self.define(
            ("lm_head", role),
            role,
            w=d * vocab + vocab,
            a=d,
            outputs=1,
            calls={unembed: 1, argmax: 1},
            feeds={unembed: _feed(w=W, a=A), argmax: _feed(w=W, a=COMPUTED)},
        )
        return {head: 1}, {head: _feed(w=W, a=COMPUTED)}

    def prompt(self, ctx: str) -> dict[str, int]:
        """The prompt tokens: ``in`` cells in the request or step, else a ``prompt`` unit."""

        n = self.shape.prompt
        if ctx == _IN_REPLAY:
            return {self.source("in", ctx): n}
        cell = self.source("in", _IN_REPLAY)
        return {self.define(("prompt", n), REPLAY, outputs=0, calls={cell: n}): 1}

    def prefill(self, ctx: str) -> str:
        """The prompt: its tokens, their embeddings, the layers, the first generated token.

        Its ports are the weights; the tokens the embeddings read are the
        ``in`` gates inside it, so an embedding row here is fed source gates.
        """

        shape = self.shape
        n = shape.prompt
        embed = self.embed_row(ctx)
        layer_calls, layer_feeds = self.layers(n, 0, ctx)
        head_calls, head_feeds = self.head(ctx)
        calls = self.merge(self.prompt(ctx), {embed: n}, layer_calls, head_calls)
        feeds = self.feeds({embed: _feed(w=W, a=SOURCE)}, layer_feeds, head_feeds)
        return self.define(
            ("prefill", n),
            None,
            w=shape.weight_count,
            outputs=shape.state_size(n) + 1,
            calls=calls,
            feeds=feeds,
        )

    def decode(self, c: int, ctx: str) -> str:
        """One token at context ``c`` over a cache of ``c - 1``: embedding, layers, next token.

        Its ports are the weights, the token and the cache: the embedding
        row reads the token port, the layers the cache ports.
        """

        shape = self.shape
        embed = self.embed_row(ctx)
        layer_calls, layer_feeds = self.layers(1, c - 1, ctx)
        head_calls, head_feeds = self.head(ctx)
        calls = self.merge({embed: 1}, layer_calls, head_calls)
        feeds = self.feeds({embed: _feed(w=W, a=A)}, layer_feeds, head_feeds)
        return self.define(
            ("decode", c),
            None,
            w=shape.weight_count,
            a=1 + shape.state_size(c - 1),
            outputs=shape.state_size(1) + 1,
            calls=calls,
            feeds=feeds,
        )

    # -- the run ------------------------------------------------------------------------

    def weights(self) -> str:
        cell = self.source("weight", _IN_REPLAY)
        return self.define(
            ("weights",), REPLAY, outputs=0, calls={cell: self.shape.weight_count}
        )

    def request(self) -> str:
        """A request: its prefill then a decode step per further token; the unit at ``request``.

        Its ports are the weights, handed on to the prefill and the decodes;
        each decode's token and cache are outputs of the calls before it.
        """

        shape = self.shape
        role = REPLAY if self.ru == "request" else None
        ctx = self.within(role, _FREE)
        prefill = self.prefill(ctx)
        calls, feeds = {prefill: 1}, {prefill: _feed(w=W)}
        for c in range(shape.prompt + 1, shape.prompt + shape.generated):
            decode = self.decode(c, ctx)
            calls[decode] = 1
            feeds[decode] = _feed(w=W, a=COMPUTED)
        return self.define(
            ("request", shape.prompt, shape.generated),
            role,
            w=shape.weight_count,
            outputs=shape.generated,
            calls=calls,
            feeds=feeds,
        )

    def steps(self) -> tuple[dict[str, int], dict[str, _Feed], dict[str, int]]:
        """The ``step`` layout: a prefill step then a decode step per token, per wave of ``batch`` requests.

        A prefill step's ports are the weights; a decode step's are the
        weights and each occupant's token and cache, produced by the earlier
        steps of the run.  Returns the calls, their feeds and their reaches:
        the steps of a wave are chained, each reading the tokens and cache of
        those before it, so the step at context ``c`` reaches the ``batch``
        tokens of every step from ``c`` to the end of the wave and the
        prefill step the whole wave's; the next wave starts from the weights
        alone, so no step reaches another wave's tokens.
        """

        shape = self.shape
        b, waves, width = shape.batch, shape.requests // shape.batch, shape.width
        prefill = self.prefill(_IN_REPLAY)
        step = self.define(
            ("prefill_step", b),
            REPLAY,
            w=shape.weight_count,
            outputs=b * (shape.state_size(shape.prompt) + 1),
            calls={prefill: b},
            feeds={prefill: _feed(w=W)},
        )
        calls, feeds = {step: waves}, {step: _feed(w=SOURCE)}
        reaches = {step: b * shape.generated * width}
        for c in range(shape.prompt + 1, shape.prompt + shape.generated):
            decode = self.decode(c, _IN_REPLAY)
            step = self.define(
                ("decode_step", c, b),
                REPLAY,
                w=shape.weight_count,
                a=b * (1 + shape.state_size(c - 1)),
                outputs=b * (shape.state_size(1) + 1),
                calls={decode: b},
                feeds={decode: _feed(w=W, a=A)},
            )
            calls[step] = waves
            feeds[step] = _feed(w=SOURCE, a=COMPUTED)
            reaches[step] = b * (shape.prompt + shape.generated - c) * width
        return calls, feeds, reaches

    def root(self) -> str:
        """The run: the weights RU, then the requests or the steps, handed the weight cells.

        The weights are read by every request or step and reach the whole
        output; a request reaches its own ``generated`` tokens, requests
        never reading each other; a step reaches what :meth:`steps` says.
        """

        shape = self.shape
        calls: dict[str, int] = {self.weights(): 1}
        feeds: dict[str, _Feed] = {}
        reaches: dict[str, int] = {}
        if self.ru == "step":
            step_calls, feeds, reaches = self.steps()
            calls.update(step_calls)
        else:
            request = self.request()
            calls[request] = shape.requests
            feeds[request] = _feed(w=SOURCE)
            reaches[request] = shape.generated * shape.width
        return self.define(
            ("root",),
            None,
            outputs=shape.output_count,
            calls=calls,
            feeds=feeds,
            reaches=reaches,
        )

    # -- the table -----------------------------------------------------------------------

    def table(self) -> KindTable:
        root = self.root()
        width = self.shape.width
        copies = {root: 1}
        min_depth = {root: 0}
        max_depth = {root: 0}
        # a port group is retained iff every caller feeds it source gates or its own
        # retained groups: callers are settled before their callees, so one pass
        retained: dict[str, dict[str, bool]] = {root: {}}
        # a kind reaches the most any call site lets it: what the caller reaches, unless the
        # caller's dataflow says the child's outputs reach less (the root's requests and steps)
        reach: dict[str, int] = {root: self.rows[root].out_count * width}
        # the narrowest interface enclosing a copy, widest over the copies: what the caller
        # is enclosed by, narrowed by the caller's own interface (the root by nothing)
        ancestor: dict[str, int] = {root: self.rows[root].out_count * width}
        for name in self.parents_first(root):
            row = self.rows[name]
            own = retained.setdefault(name, {})
            enclosing = min(ancestor[name], row.out_count * width)
            for child, count in row.children.items():
                copies[child] = copies.get(child, 0) + copies[name] * count
                depth = min_depth[name] + 1
                min_depth[child] = min(min_depth.get(child, depth), depth)
                depth = max_depth[name] + 1
                max_depth[child] = max(max_depth.get(child, depth), depth)
                target = retained.setdefault(child, {})
                for group, sources in row.feeds[child].items():
                    fed = all(
                        source == SOURCE or own.get(source, False) for source in sources
                    )
                    target[group] = target.get(group, True) and fed
                site = min(reach[name], row.reaches.get(child, reach[name]))
                reach[child] = max(reach.get(child, 0), site)
                ancestor[child] = max(ancestor.get(child, 0), enclosing)
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
                interior_count=row.interior_count,
                reach_bits=reach[row.key],
                ancestor_bits=ancestor[row.key],
                source_inputs=row.source_inputs,
                source_weights=row.source_weights,
                min_depth=min_depth[row.key],
                max_depth=max_depth[row.key],
                children=tuple(row.children.items()),
                verification_units=row.verification_units,
                verification_kinds=tuple(row.verification_kinds.items()),
                closed=all(retained[row.key].values()),
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
                    {
                        "shape": self.shape.manifest,
                        "replay": self.ru,
                        "verification": self.vu,
                    },
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
    inside = sum(
        row.copies * row.verification_units for row in table.rows if row.role == REPLAY
    )
    units = sum(row.copies for row in table.rows if row.role == VERIFICATION)
    if inside != units:
        raise ValueError("verification units do not refine the replay units")


def serving_table(
    shape: ServingShape,
    replay: ReplayLevel = "request",
    verification: VerificationLevel = "row",
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
