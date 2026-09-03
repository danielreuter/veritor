"""Speculative decoding: a draft model proposes, the target model verifies.

Each step a small *draft* model decodes ``gamma`` tokens one after another
from the last emitted token ``y``, and the *target* model runs one forward
pass over the ``gamma + 1`` positions ``y, d_1, ..., d_gamma``, giving its
greedy continuation ``z_0, ..., z_gamma`` at each.  Draft token ``d_i`` is
accepted when ``d_1..d_{i-1}`` were and ``d_i == z_{i-1}``; with ``m``
accepted the step emits ``d_1, ..., d_m, z_m`` and continues from ``z_m``.
Greedy acceptance is exact: the emitted tokens are the target's own greedy
sequence, whatever the draft proposes.  ``m`` is data-dependent -- in
``0..gamma`` every step -- and so are the number of steps to ``max_new``
tokens and which KV entries survive a step.  The circuit takes it one of two
ways.

**Padded** (``acceptance="padded"``, no advice).  Every request runs
``max_new - 1`` steps (the prefill emits the first token; each step emits at
least one more).  Every step drafts ``gamma`` tokens plus one more draft
decode that only fills the draft's cache for ``d_gamma``, verifies
``gamma + 1`` positions, and computes the acceptance in-circuit
(:meth:`SpeculativeG.acceptance_unit`: ``eq`` per position, a prefix
product, and the emitted slots and the next token as masked sums).  The
step emits ``gamma + 1`` output slots, ``vocab`` (never a token) filling
the slots past ``z_m``; the rejected positions' cached ``V`` entries are
multiplied by their acceptance flag so the fixed-shape attention (an
unnormalised weighted sum, so a zero ``V`` contributes nothing whatever its
``K``) reproduces the un-padded computation exactly.  The caches grow by
``gamma + 1`` positions per step whatever was accepted.  The count of real
tokens in a step is output-determined -- the blanks show it -- which is the
uncharged third route to ``m`` (not taken here).

**Advice** (``acceptance="advice"``).  The per-step ``m`` is advice,
``ceil(log2(gamma + 2))`` bits each, self-delimiting: steps follow until the
emitted count reaches ``max_new``.  A step with advice ``m`` drafts ``m + 1``
tokens (``d_{m+1}`` is the rejected one, or the cache fill when ``m =
gamma``), verifies ``m + 1`` positions and emits ``d_1..d_m, z_m``.  Its
:meth:`SpeculativeG.acceptance_check` unit multiplies into a running ``ok``
word ``[d_i == z_{i-1}]`` for ``i <= m`` and, for ``m < gamma``, ``1 - [d_{m+1}
== z_m]``: the accepted prefix is exactly the agreeing prefix.  ``ok`` is the
request's first output and the verifier requires ``1``.  The circuit holds
only the accepted work: the same target positions as plain decoding, plus
the draft's, plus the checks.

Structure.  The root calls the target's ``weights`` unit, then the draft's,
then one ``request`` replay unit per request whose ports are both weight
blocks; the prompt tokens are ``in`` gates inside it, read once by
:meth:`~veritor.constructors.lm.ToyLM.prefill_ports` of each model.
Outputs per request: (advice) ``ok`` then ``max_new`` tokens; (padded) the
prefill's token then ``(max_new - 1) * (gamma + 1)`` slots.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue, make_isa_gate_set
from veritor.core.description import REPLAY, VERIFICATION

from .lm import ADVICE, PADDED, Decoder, LMShape, Parameters, ToyLM, wire, wires
from .schedule import Request
from .tracer import TracedDefinition, Tracer, TracerError, Wire, Wires

Acceptances = tuple[int, ...]
"""One request's accepted draft counts, step by step."""


@dataclass(frozen=True, slots=True)
class SpeculativeTrace:
    """The reference run of one request: every emitted token (the prefill's first) and each step's ``m``."""

    tokens: tuple[int, ...]
    acceptances: Acceptances


def acceptance_bits(gamma: int) -> int:
    """``ceil(log2(gamma + 2))``: the description length of one step's ``m`` in ``0..gamma``."""

    return (gamma + 1).bit_length()


def reference_speculative(
    target: Parameters, draft: Parameters, gamma: int, requests: Sequence[Request]
) -> tuple[SpeculativeTrace, ...]:
    """Speculative decoding in Python: the emitted tokens and acceptances of every request.

    Steps follow until ``max_new`` tokens have been emitted; the last step
    may overshoot, and ``tokens`` keeps the overshoot (the circuit emits the
    first ``max_new``).
    """

    traces: list[SpeculativeTrace] = []
    for request in requests:
        t, d = Decoder(target), Decoder(draft)
        for token in request.prompt[:-1]:
            t.logits(token)
            d.logits(token)
        y = t.forward(request.prompt[-1])
        d.logits(request.prompt[-1])
        tokens, acceptances = [y], []
        while len(tokens) < request.max_new:
            cached = t.positions
            drafts = [y]
            for _ in range(gamma):
                drafts.append(d.forward(drafts[-1]))
            d.forward(drafts[-1])  # fills the draft's cache for d_gamma
            z = [t.forward(token) for token in drafts]  # positions y, d_1, ..., d_gamma
            m = 0
            while m < gamma and drafts[m + 1] == z[m]:
                m += 1
            tokens.extend([*drafts[1 : m + 1], z[m]])
            acceptances.append(m)
            y = z[m]
            t.truncate(cached + m + 1)
            d.truncate(cached + m + 1)
        traces.append(SpeculativeTrace(tuple(tokens), tuple(acceptances)))
    return tuple(traces)


def encode_acceptances(gamma: int, traces: Sequence[Sequence[int]]) -> bytes:
    """Every request's ``m`` sequence, ``acceptance_bits(gamma)`` bits each, packed MSB first."""

    width = acceptance_bits(gamma)
    text = "".join(format(m, f"0{width}b") for steps in traces for m in steps)
    text += "0" * (-len(text) % 8)
    return bytes(int(text[i : i + 8], 2) for i in range(0, len(text), 8))


def decode_acceptances(gamma: int, requests: Sequence[Request], a: bytes) -> tuple[Acceptances, ...]:
    """The advice back into per-request acceptances, each self-delimited by reaching ``max_new``."""

    width = acceptance_bits(gamma)
    text = "".join(format(byte, "08b") for byte in a)
    cursor = 0
    result: list[Acceptances] = []
    for request in requests:
        emitted, steps = 1, []
        while emitted < request.max_new:
            if cursor + width > len(text):
                raise TracerError("the acceptance advice ends before the requests do")
            m = int(text[cursor : cursor + width], 2)
            cursor += width
            if m > gamma:
                raise TracerError(f"an acceptance of {m} exceeds gamma = {gamma}")
            steps.append(m)
            emitted += m + 1
        result.append(tuple(steps))
    if len(a) != (cursor + 7) // 8 or any(bit != "0" for bit in text[cursor:]):
        raise TracerError("the acceptance advice has trailing bits")
    return tuple(result)


class SpeculativeG:
    """Speculative decoding of every request in its own replay unit, padded or with advised acceptances."""

    VERSION = "1"

    def __init__(self, target: LMShape, draft: LMShape, gamma: int, acceptance: str = PADDED) -> None:
        if not isinstance(target, LMShape) or not isinstance(draft, LMShape):
            raise TypeError("target and draft must be LMShapes")
        if target.sampling or draft.sampling:
            raise ValueError("greedy speculative decoding needs argmax models")
        if target.vocab != draft.vocab or target.width != draft.width:
            raise ValueError("the draft must share the target's vocabulary and word width")
        if type(gamma) is not int or gamma < 1:
            raise ValueError("gamma must be a positive integer")
        if gamma + 1 >= target.vocab:
            raise ValueError("gamma + 1 must be below vocab: the constant table holds the sentinel's parts")
        if acceptance not in (PADDED, ADVICE):
            raise ValueError(f"acceptance must be {PADDED!r} or {ADVICE!r}")
        self.target_shape, self.draft_shape, self.gamma, self.acceptance = target, draft, gamma, acceptance
        self.tracer = Tracer(make_isa_gate_set(target.width))
        self.target = ToyLM(target, tracer=self.tracer, prefix="target")
        self.draft = ToyLM(draft, tracer=self.tracer, prefix="draft")
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def advised(self) -> bool:
        return self.acceptance == ADVICE

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "target": self.target_shape.manifest,
            "draft": self.draft_shape.manifest,
            "gamma": self.gamma,
            "acceptance": self.acceptance,
        }

    @property
    def weight_count(self) -> int:
        return self.target_shape.weight_count + self.draft_shape.weight_count

    def flatten_weights(self, target: Parameters, draft: Parameters) -> tuple[int, ...]:
        """Both models' weights in ``weight``-gate order: the target's, then the draft's."""

        if target.shape != self.target_shape or draft.shape != self.draft_shape:
            raise ValueError("parameters must match the constructor's shapes")
        return (*target.flatten(), *draft.flatten())

    # -- validation -----------------------------------------------------------------

    def positions(self, request: Request) -> int:
        """The cache positions a request can reach: padded steps pad the cache, advised ones do not."""

        n, gamma = len(request.prompt), self.gamma
        if self.advised:
            return n + request.max_new + gamma  # the last step may overshoot by at most gamma
        return n + (request.max_new - 1) * (gamma + 1) + 1

    def requests(self, x: object) -> tuple[Request, ...]:
        if type(x) is not tuple or not x or any(type(item) is not Request for item in x):
            raise TracerError("SpeculativeG expects a nonempty tuple of Request")
        for index, request in enumerate(x):
            if any(token >= self.target_shape.vocab for token in request.prompt):
                raise TracerError(f"request {index} has a prompt token outside the vocabulary")
            if request.randomness:
                raise TracerError(f"request {index} carries randomness; speculative decoding is greedy")
            needed = self.positions(request)
            if needed > min(self.target_shape.context, self.draft_shape.context):
                raise TracerError(f"request {index} reaches {needed} positions, beyond a model's context")
        return x

    # -- layouts ---------------------------------------------------------------------

    def slots(self, request: Request) -> int:
        """How many words a request outputs: ``ok`` and its tokens, or the prefill's token and every slot."""

        if self.advised:
            return 1 + request.max_new
        return 1 + (request.max_new - 1) * (self.gamma + 1)

    def output_layout(self, x: object) -> tuple[tuple[int, int], ...]:
        """``(request, position)`` of every output: ``-1`` for an ``ok`` word, else the slot index."""

        layout: list[tuple[int, int]] = []
        for r, request in enumerate(self.requests(x)):
            first = -1 if self.advised else 0
            layout.extend((r, p) for p in range(first, first + self.slots(request)))
        return tuple(layout)

    def flatten_inputs(self, x: object) -> tuple[int, ...]:
        """The prompt tokens, request by request: each read once, by both models."""

        return tuple(token for request in self.requests(x) for token in request.prompt)

    def advice(self, x: object, target: Parameters, draft: Parameters) -> bytes:
        """The honest advice: the reference run's acceptances; empty when padded."""

        requests = self.requests(x)
        if not self.advised:
            return b""
        traces = reference_speculative(target, draft, self.gamma, requests)
        return encode_acceptances(self.gamma, [trace.acceptances for trace in traces])

    def tokens(self, outputs: Sequence[int], x: object) -> tuple[tuple[int, ...], ...]:
        """What the client reads: each request's tokens, blanks dropped and cut at ``max_new``."""

        requests = self.requests(x)
        by_request: list[list[int]] = [[] for _ in requests]
        for value, (r, position) in zip(outputs, self.output_layout(x), strict=True):
            if position >= 0 and value != self.target_shape.vocab:
                by_request[r].append(value)
        return tuple(tuple(tokens[: request.max_new]) for tokens, request in zip(by_request, requests, strict=True))

    def checks(self, outputs: Sequence[int], x: object) -> tuple[int, ...]:
        """The ``ok`` words, one per request (none when padded)."""

        return tuple(value for value, (_r, position) in zip(outputs, self.output_layout(x), strict=True) if position < 0)

    # -- verification units ------------------------------------------------------------

    def mask_row(self, d: int) -> TracedDefinition:
        """``flag * v_i`` over ``d`` values: a cached ``V`` entry kept or zeroed by its acceptance flag."""

        mul = self.tracer.gate("mul")

        @self.tracer.definition(input_count=1 + d, key=("speculative", "mask_row", d), role=VERIFICATION)
        def mask_row(v: Wires) -> object:
            return [mul(v[0], v[1 + i]) for i in range(d)]

        return mask_row

    def acceptance_unit(self) -> TracedDefinition:
        """The padded step's decision.  Ports: ``d_1..d_gamma``, ``z_0..z_gamma``, the constants ``1`` and ``vocab - 1``.

        Outputs: ``acc_1..acc_gamma`` (the prefix product of ``[d_i == z_{i-1}]``),
        the ``gamma + 1`` emitted slots (``d_i`` while accepted, then ``z_m``,
        then ``vocab`` blanks) and the next token ``z_m``.
        """

        gamma = self.gamma
        add, mul, sub, eq = (self.tracer.gate(name) for name in ("add", "mul", "sub", "eq"))

        @self.tracer.definition(input_count=2 * gamma + 3, key=("speculative", "accept", gamma), role=VERIFICATION)
        def accept(v: Wires) -> object:
            drafts, z = v[:gamma], v[gamma : 2 * gamma + 1]
            one, blank = v[2 * gamma + 1], add(v[2 * gamma + 2], v[2 * gamma + 1])  # blank = vocab
            acc: list[Wire] = []
            for i in range(gamma):
                agree = eq(drafts[i], z[i])
                acc.append(agree if i == 0 else mul(acc[-1], agree))
            before = [one, *acc]  # acc_0 = 1
            select = [sub(before[i], acc[i]) if i < gamma else acc[-1] for i in range(gamma + 1)]  # [m == i]
            nxt = mul(select[0], z[0])
            for i in range(1, gamma + 1):
                nxt = add(nxt, mul(select[i], z[i]))
            slots: list[Wire] = []
            for i in range(1, gamma + 1):  # slot i: d_i if accepted, z_{i-1} if first rejected, else blank
                slot = add(mul(acc[i - 1], drafts[i - 1]), mul(select[i - 1], z[i - 1]))
                if i > 1:
                    slot = add(slot, mul(sub(one, before[i - 1]), blank))
                slots.append(slot)
            slots.append(add(mul(acc[-1], z[gamma]), mul(sub(one, acc[-1]), blank)))
            return [*acc, *slots, nxt]

        return accept

    def acceptance_check(self, m: int) -> TracedDefinition:
        """The advised step's check.  Ports: ``d_1..d_{m+1}``, ``z_0..z_m``, ``1``, the incoming ``ok``.

        Output: ``ok * prod_{i<=m} [d_i == z_{i-1}] * (1 - [d_{m+1} == z_m])``, the
        last factor only when ``m < gamma``: exactly ``m`` drafts agree.
        """

        gamma = self.gamma
        if not 0 <= m <= gamma:
            raise TracerError(f"an acceptance lies in 0..{gamma}")
        mul, sub, eq = (self.tracer.gate(name) for name in ("mul", "sub", "eq"))
        drafted = m + 1 if m < gamma else gamma

        @self.tracer.definition(
            input_count=drafted + m + 3, key=("speculative", "check", gamma, m), role=VERIFICATION
        )
        def check(v: Wires) -> object:
            drafts, z, one, ok = v[:drafted], v[drafted : drafted + m + 1], v[drafted + m + 1], v[drafted + m + 2]
            for i in range(m):
                ok = mul(ok, eq(drafts[i], z[i]))
            if m < gamma:
                ok = mul(ok, sub(one, eq(drafts[m], z[m])))
            return ok

        return check

    # -- the request replay unit ---------------------------------------------------------

    def request(self, prompt: int, max_new: int, acceptances: Acceptances | None = None) -> TracedDefinition:
        """One request: both prefills, then the speculative steps, over both caches.

        Ports: the target's weights, then the draft's.  Outputs as in the
        module docstring.  Advised requests take their acceptances, which
        become part of the kind.
        """

        if (acceptances is None) == self.advised:
            raise TracerError("advised acceptance needs the request's acceptances; padded takes none")
        gamma, wt, wd = self.gamma, self.target_shape.weight_count, self.draft_shape.weight_count
        dt, dd = self.target_shape.d_model, self.draft_shape.d_model
        lt, ld = self.target_shape.layers, self.draft_shape.layers
        key: tuple[object, ...] = ("speculative", "request", prompt, max_new)
        if acceptances is not None:
            if sum(m + 1 for m in acceptances) + 1 < max_new or any(not 0 <= m <= gamma for m in acceptances):
                raise TracerError("the acceptances must reach max_new with counts in 0..gamma")
            if sum(m + 1 for m in acceptances[:-1]) + 1 >= max_new:
                raise TracerError("the acceptances run past max_new")
            key = (*key, acceptances)
        repeat = self.tracer.repeat

        @self.tracer.definition(input_count=wt + wd, key=key, role=REPLAY)
        def request(w: Wires) -> object:
            w_target, w_draft = w[:wt], w[wt:]
            constants = self.target.ports(w_target).constants
            one, top = constants[1], constants[self.target_shape.vocab - 1]
            tokens = self.tracer.inputs(prompt)
            keys_t: list[list[Wires]] = [[] for _ in range(lt)]
            values_t: list[list[Wires]] = [[] for _ in range(lt)]
            keys_d: list[list[Wires]] = [[] for _ in range(ld)]
            values_d: list[list[Wires]] = [[] for _ in range(ld)]

            def remember(
                block: Wires, positions: int, d: int, keys: list[list[Wires]], values: list[list[Wires]]
            ) -> list[Wires]:
                """File a step's ``K`` blocks and return its ``V`` blocks, position-major, for masking."""

                out: list[Wires] = []
                for layer in range(len(keys)):
                    start = 2 * layer * positions * d
                    keys[layer].append(block[start : start + positions * d])
                    out.append(block[start + positions * d : start + 2 * positions * d])
                return out

            def file_values(blocks: Sequence[Wires], values: list[list[Wires]]) -> None:
                for layer, block in enumerate(blocks):
                    values[layer].append(block)

            def cache_args(keys: list[list[Wires]], values: list[list[Wires]]) -> list[Wire | Wires]:
                args: list[Wire | Wires] = []
                for layer in range(len(keys)):
                    args.extend(keys[layer])
                    args.extend(values[layer])
                return args

            block = wires(self.target.prefill_ports(prompt)(w_target, tokens))
            file_values(remember(block, prompt, dt, keys_t, values_t), values_t)
            y: Wire = block[-1]
            block = wires(self.draft.prefill_ports(prompt)(w_draft, tokens))
            file_values(remember(block, prompt, dd, keys_d, values_d), values_d)  # its token is unused
            outputs: list[Wire | Wires] = [y]
            ok: Wire | None = one if acceptances is not None else None
            cached, emitted = prompt, 1
            steps: Sequence[int | None] = acceptances if acceptances is not None else (None,) * (max_new - 1)
            for m in steps:
                # The draft decodes y, d_1, ..., producing d_1, d_2, ...: gamma + 1 times when padded
                # (gamma proposals, then a decode of d_gamma that only fills the cache), m + 1 times
                # when advised (d_{m+1} is the rejected draft, or the cache fill when m = gamma).
                drafted = gamma + 1 if m is None else m + 1
                verified = gamma + 1 if m is None else m + 1  # positions the target looks at: y and drafts
                drafts: list[Wire] = []
                token = y
                for i in range(drafted):
                    step = wires(self.draft.decode(cached + i + 1)(w_draft, token, *cache_args(keys_d, values_d)))
                    file_values(remember(step, 1, dd, keys_d, values_d), values_d)
                    token = step[-1]
                    drafts.append(token)
                block = wires(
                    self.target.extend(cached, verified)(
                        w_target, y, *drafts[: verified - 1], *cache_args(keys_t, values_t)
                    )
                )
                target_values = remember(block, verified, dt, keys_t, values_t)
                z = block[-verified:]
                if m is None:
                    decision = wires(self.acceptance_unit()(*drafts[:gamma], z, one, top))
                    acc, slots, y = decision[:gamma], decision[gamma : 2 * gamma + 1], decision[-1]
                    outputs.append(slots)
                    # Position 0 of the step is y, always kept; positions 1..gamma keep their V
                    # entries only if accepted.  K entries need no mask: attention is a weighted
                    # sum of V, and a zero V contributes nothing whatever its score.
                    for layer, values in enumerate(target_values):
                        values_t[layer].append(values[0:dt])
                        values_t[layer].append(
                            repeat(gamma, self.mask_row(dt), acc[0].by(1), values[dt : 2 * dt].by(dt))
                        )
                    for layer in range(ld):
                        filed = values_d[layer][-drafted:]
                        del values_d[layer][-drafted:]
                        values_d[layer].append(filed[0])
                        values_d[layer].extend(
                            wires(self.mask_row(dd)(acc[i - 1], filed[i])) for i in range(1, drafted)
                        )
                    cached += gamma + 1
                else:
                    assert ok is not None
                    checked = drafts if m < gamma else drafts[:gamma]
                    ok = wire(self.acceptance_check(m)(*checked, z, one, ok))
                    for layer, values in enumerate(target_values):
                        values_t[layer].append(values)
                    new_tokens: list[Wire] = [*drafts[:m], z[m]]
                    outputs.extend(new_tokens[: max_new - emitted])
                    emitted += m + 1
                    y = z[m]
                    cached += m + 1
            return [ok, *outputs] if ok is not None else outputs

        return request

    def root(self, requests: tuple[Request, ...], acceptances: Sequence[Acceptances] | None) -> TracedDefinition:
        @self.tracer.definition(input_count=0)
        def root(_v: Wires) -> object:
            w_target = wires(self.target.weights_unit()())
            w_draft = wires(self.draft.weights_unit()())
            return [
                self.request(len(r.prompt), r.max_new, None if acceptances is None else acceptances[i])(
                    w_target, w_draft
                )
                for i, r in enumerate(requests)
            ]

        return root

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        requests = self.requests(x)
        if not self.advised:
            if a:
                raise TracerError("padded speculative decoding takes no advice")
            return self.tracer.serialize(self.root(requests, None)), self.flatten_inputs(requests)
        acceptances = decode_acceptances(self.gamma, requests, a)
        return self.tracer.serialize(self.root(requests, acceptances)), self.flatten_inputs(requests)


__all__ = [
    "Acceptances",
    "SpeculativeG",
    "SpeculativeTrace",
    "acceptance_bits",
    "decode_acceptances",
    "encode_acceptances",
    "reference_speculative",
]
