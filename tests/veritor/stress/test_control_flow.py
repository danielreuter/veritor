"""Control-flow stress tests: padding against advice (``docs/stress-tests.md`` C1, C2, C3).

A data-dependent structural choice -- how long a request runs, which expert
a position goes to, how many draft tokens were accepted -- can enter the
circuit two ways.  *Padded*, the circuit decides in-circuit with ``lt``/``eq``
and masks and does the work of every outcome, so no advice is charged and
compute is wasted.  *Advised*, the choice is bits of ``a``: the circuit does
only the chosen work, checks the choice in-circuit, and ``Bound + |a|`` pays
for the bits.  A third route, output-determined advice (choices the public
outputs pin down, charged nothing), is noted where it applies and not built.

Every scenario builds both circuits over :class:`~veritor.constructors.requests.RequestsG`-style
request RUs, checks them against the Python oracle, prices them (``Bound`` at
``eta = 2^-40`` under the datacenter policy, ``Cost``, gates, description
bytes) and appends rows to ``docs/data/stress-control-flow.json`` -- a JSON
object keyed by scenario ID, merged by ID so other recorders' rows survive.
One MoE and one speculative configuration run the three-message protocol
honestly and against a dishonest server.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from veritor.analysis.cost import cost
from veritor.compile import Compilation, Constructor
from veritor.constructors import (
    ADVICE,
    PADDED,
    ClusterG,
    Join,
    LMShape,
    Parameters,
    Request,
    RequestsG,
    Schedule,
    random_parameters,
    reference_generate,
)
from veritor.constructors.moe import advice_bits as route_advice_bits
from veritor.constructors.moe import decode_routes, encode_routes, reference_routes
from veritor.constructors.schedule import gamma as gamma_code
from veritor.constructors.speculative import (
    SpeculativeG,
    acceptance_bits,
    decode_acceptances,
    encode_acceptances,
    reference_speculative,
)
from veritor.core import Compiled, VerificationPolicy, as_kind_table, make_isa_gate_set
from veritor.protocol import (
    VerificationCode,
    VerificationReport,
    VerifierParameters,
    assignment_replay,
    commit_weights,
    make_expectation,
    run_protocol,
)
from veritor.research import Bound, Compile
from veritor.simulation import adversary
from veritor.simulation.datacenter import POLICY

ETA = Fraction(1, 2**40)
FULL = VerificationPolicy(1, 1)
"""Every RU replayed, every VU checked: a dishonest relation is caught with certainty."""

DATA = (
    Path(__file__).resolve().parents[3] / "docs" / "data" / "stress-control-flow.json"
)
WIDTH = 16


# -- rows -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One priced scenario of ``docs/data/stress-control-flow.json``."""

    id: str
    what: str
    mechanism: str
    advice_bits: int
    capacity_bits: int
    overhead: float
    description_bytes: int
    gates: int
    verdict: str
    notes: str

    def body(self) -> dict[str, object]:
        body = asdict(self)
        del body["id"]
        body["overhead"] = round(self.overhead, 6)
        return body


def record(rows: Iterable[Row], path: Path = DATA) -> dict[str, dict[str, object]]:
    """Merge ``rows`` by ID into the JSON object at ``path`` (one row per line, IDs sorted) and return it."""

    merged: dict[str, dict[str, object]] = {}
    if path.exists() and path.read_text(encoding="utf-8").strip():
        merged = dict(json.loads(path.read_text(encoding="utf-8")))
    for row in rows:
        merged[row.id] = row.body()

    def key(identifier: str) -> tuple[str, int, str]:
        digits = "".join(ch for ch in identifier[1:] if ch.isdigit())
        return identifier[0], int(digits or 0), identifier[1 + len(digits) :]

    ordered = sorted(merged, key=key)
    lines = [
        f' "{identifier}": {json.dumps(merged[identifier], sort_keys=True)}{"," if i + 1 < len(ordered) else ""}'
        for i, identifier in enumerate(ordered)
    ]
    text = "{\n" + "\n".join(lines) + "\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(temporary, path)
    return {identifier: merged[identifier] for identifier in ordered}


# -- measurement ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Measured:
    """A compiled scenario, evaluated and priced."""

    compilation: Compilation
    values: dict[int, int]
    outputs: tuple[int, ...]
    description_bytes: int
    bound_bits: float
    capped: bool
    out_bits: int
    overhead: float
    honest_cost: int

    @property
    def compiled(self) -> Compiled:
        return self.compilation.compiled

    @property
    def inputs(self) -> tuple[int, ...]:
        return self.compilation.inputs

    @property
    def gates(self) -> int:
        return self.compiled.circuit.n

    @property
    def advice_bits(self) -> int:
        """``|a|`` as charged: the bits the constructor declares (``8 * len(a)`` if it declares none)."""

        return self.compilation.advice_bits

    @property
    def capacity(self) -> float:
        """``Bound + |a|``, the paper's per-request charge."""

        return self.bound_bits + self.advice_bits


def measure(
    constructor: Constructor,
    x: object,
    advice: bytes,
    weights: Sequence[int],
    policy: VerificationPolicy = POLICY,
) -> Measured:
    """``Compile``, evaluate, ``Bound`` at ``eta`` and ``Cost`` under ``policy``."""

    compilation = Compile(
        constructor,
        x,
        advice,
        make_isa_gate_set(WIDTH),
        max_advice_bits=8 * len(advice),
    )
    compiled = compilation.compiled
    circuit = compiled.circuit
    values = dict(enumerate(circuit.evaluate(compilation.inputs, weights)))
    table = as_kind_table(compiled)
    honest = next(row.replay_cost for row in table.rows if row.kind == table.root)
    bound = Bound(compiled, policy, ETA)
    expected = cost(compiled, policy)
    return Measured(
        compilation=compilation,
        values=values,
        outputs=tuple(values[address] for address in circuit.outputs),
        description_bytes=len(constructor(x, advice)[0]),
        bound_bits=bound.bits,
        capped=bound.capped,
        out_bits=bound.out_bits,
        overhead=float(expected.total / honest),
        honest_cost=honest,
    )


def protocol(
    measured: Measured,
    weights: Sequence[int],
    policy: VerificationPolicy,
    *,
    outputs: Sequence[int] | None = None,
    values: Mapping[int, int] | None = None,
    dishonest: bool = False,
) -> VerificationReport:
    """The three-message protocol over ``measured``: honest by default, or committing ``values`` as given."""

    kappa, tree = commit_weights(make_isa_gate_set(WIDTH), weights)
    parameters = VerifierParameters(
        ETA,
        max_capacity=math.ceil(measured.bound_bits),
        max_advice_bits=measured.advice_bits,
    )
    expectation = make_expectation(
        measured.compilation,
        policy,
        tuple(measured.outputs if outputs is None else outputs),
        parameters=parameters,
        weights=kappa,
        session_id=b"control-flow-000",
        q_seed=b"q" * 32,
        s_seed=b"s" * 32,
    )
    assignment = dict(measured.values if values is None else values)
    run = run_protocol(
        measured.compiled,
        expectation,
        assignment,
        replay=assignment_replay(assignment) if dishonest else None,
        weight_tree=tree,
    )
    return run.report


def check_address(
    measured: Measured, layout: Sequence[tuple[int, int]], request: int
) -> int:
    """The output address of ``request``'s ``ok`` word (laid out at position ``-1``)."""

    return next(
        a
        for a, (r, g) in zip(measured.compiled.circuit.outputs, layout, strict=True)
        if r == request and g < 0
    )


def per_token(value: float, tokens: int) -> float:
    return round(value / tokens, 1)


# -- C1: variable-length generation ---------------------------------------------------

DENSE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=32, width=WIDTH)
EOS = DENSE.vocab - 1


def stopped(
    reference: Sequence[Sequence[int]], requests: Sequence[Request]
) -> tuple[int, ...]:
    """Each request's streamed length: through its first EOS, else ``max_new``."""

    lengths = []
    for tokens, request in zip(reference, requests, strict=True):
        lengths.append(tokens.index(EOS) + 1 if EOS in tokens else request.max_new)
    return tuple(lengths)


def test_c1_variable_length_generation() -> None:
    """Different ``max_new`` and EOS stops: request RUs need no advice, step RUs take the schedule."""

    parameters = random_parameters(DENSE, 7)
    prompts = ((1, 2, 3), (4, 5), (6, 1, 2, 3), (2,), (3, 4), (5, 6, 0))
    requests = tuple(Request(prompt, 3 + 2 * i) for i, prompt in enumerate(prompts))
    reference = reference_generate(DENSE, parameters, requests)
    lengths = stopped(reference, requests)
    assert len(set(lengths)) >= 3, lengths
    eos_stops = sum(
        length < request.max_new
        for length, request in zip(lengths, requests, strict=True)
    )
    assert eos_stops >= 2, lengths
    streamed = tuple(
        tokens[:length] for tokens, length in zip(reference, lengths, strict=True)
    )
    weights = parameters.flatten()
    tokens = sum(lengths)

    # (a) request RUs: the streamed length is the request's shape; nothing is left to the client.
    shaped = tuple(
        Request(r.prompt, length) for r, length in zip(requests, lengths, strict=True)
    )
    requests_g = RequestsG(DENSE)
    by_request = measure(requests_g, shaped, b"", weights)
    assert by_request.outputs == tuple(t for response in streamed for t in response)
    assert by_request.advice_bits == 0

    # (b) step RUs: one pod of two slots; each request's join length (its EOS stop) is advice.
    joins, free = [], [0, 0]
    for index, length in enumerate(lengths):
        slot = min(range(2), key=free.__getitem__)
        joins.append(Join(0, free[slot], slot, index, length))
        free[slot] += length
    steps = max(free)
    schedule = Schedule(1, 2, steps, tuple(sorted(joins)))
    schedule.validate(requests)
    cluster_g = ClusterG(DENSE, 1, 2, steps)
    a = schedule.encode()
    by_step = measure(cluster_g, requests, a, weights)
    layout = cluster_g.output_layout(requests, schedule)
    got = [[0] * length for length in lengths]
    for value, (r, g) in zip(by_step.outputs, layout, strict=True):
        got[r][g] = value
    assert tuple(tuple(g) for g in got) == streamed
    # Schedule v4 is bit-packed and charged exactly: here 1 pod, 2 slots, so a join is step, length-1
    # (ceil(log2 steps) bits each), slot (1), resume (1), gamma(1 + request), gamma(1) = 1 bit for chunk 0.
    assert by_step.advice_bits == schedule.bit_length() < 8 * len(a) + 8
    step_bits = (steps - 1).bit_length()
    assert by_step.advice_bits == sum(
        2 * step_bits + 1 + 1 + len(gamma_code(1 + join.request)) + 1 for join in joins
    ) + sum(len(gamma_code(v)) for v in (1, 2, steps, 1 + len(joins)))

    record(
        [
            Row(
                id="C1a",
                what="variable-length generation, request RUs: 6 requests, max_new 3..13, EOS stops",
                mechanism="M5 in-shape: each request's circuit is its streamed length; no advice",
                advice_bits=0,
                capacity_bits=math.ceil(by_request.capacity),
                overhead=by_request.overhead,
                description_bytes=by_request.description_bytes,
                gates=by_request.gates,
                verdict="no advice: lengths are public (the client sees the stream end), route (c)",
                notes=(
                    f"{tokens} tokens streamed of {sum(r.max_new for r in requests)} asked; lengths {lengths} "
                    f"({eos_stops} EOS stops); "
                    f"U = {by_request.bound_bits:.0f} bits{' (interface-capped)' if by_request.capped else ''}; "
                    f"{per_token(by_request.gates, tokens)} gates/token. The lengths are in x, so the circuit "
                    "has no absent slots to blank (S7 is the case where the length is the server's and is "
                    "advice); a presence mask would carry the same choice uncharged."
                ),
            ),
            Row(
                id="C1b",
                what="variable-length generation, step RUs (ClusterG, 1 pod x 2 slots): the same requests",
                mechanism="M4 charged advice: the schedule (Schedule.encode() v4, bit-packed, exact bits)",
                advice_bits=by_step.advice_bits,
                capacity_bits=math.ceil(by_step.capacity),
                overhead=by_step.overhead,
                description_bytes=by_step.description_bytes,
                gates=by_step.gates,
                verdict=(
                    f"advice = the schedule; each request's length rides in its join ({by_step.advice_bits} bits "
                    f"for {len(joins)} joins, charged exactly; {8 * len(a)} bits on the wire)"
                ),
                notes=(
                    f"U = {by_step.bound_bits:.0f} bits{' (interface-capped)' if by_step.capped else ''} + "
                    f"{by_step.advice_bits} advice bits for {len(joins)} joins over {steps} steps (a join is "
                    f"step and length in {step_bits} bits each, slot and resume in 1, the request gamma-coded); "
                    f"{per_token(by_step.gates, tokens)} gates/token (same work, cut into steps); overhead "
                    f"{by_step.overhead:.2f} vs {by_request.overhead:.2f} with request RUs: the KV cache crosses "
                    "every step boundary. Every output is a streamed token (a join's length is its request's "
                    "output count), so there are no absent slots to blank; the lengths are charged in the joins."
                ),
            ),
        ]
    )


# -- C2: mixture-of-experts routing ------------------------------------------------------


def moe_shape(experts: int, top_k: int, layers: int = 1, vocab: int = 8) -> LMShape:
    return LMShape(
        vocab=vocab,
        d_model=4,
        heads=1,
        layers=layers,
        context=32,
        width=WIDTH,
        experts=experts,
        top_k=top_k,
    )


MOE_REQUESTS = (Request((1, 2, 3), 4), Request((5, 6), 3))


@dataclass(frozen=True, slots=True)
class MoEPair:
    shape: LMShape
    parameters: Parameters
    padded: Measured
    advised: Measured
    exact_advice_bits: int


def moe_pair(
    shape: LMShape,
    requests: tuple[Request, ...],
    seed: int = 1,
    policy: VerificationPolicy = POLICY,
) -> MoEPair:
    """Both routes of one MoE shape over ``requests``, checked against the reference decoder."""

    parameters = random_parameters(shape, seed)
    weights = parameters.flatten()
    reference = tuple(
        t
        for response in reference_generate(shape, parameters, requests)
        for t in response
    )
    padded = measure(RequestsG(shape, PADDED), requests, b"", weights, policy)
    assert padded.outputs == reference
    advised_g = RequestsG(shape, ADVICE)
    a = advised_g.advice(requests, parameters)
    advised = measure(advised_g, requests, a, weights, policy)
    layout = advised_g.output_layout(requests)
    assert (
        tuple(v for v, (_r, g) in zip(advised.outputs, layout, strict=True) if g >= 0)
        == reference
    )
    assert all(
        v == 1 for v, (_r, g) in zip(advised.outputs, layout, strict=True) if g < 0
    ), "a route check failed"
    return MoEPair(
        shape, parameters, padded, advised, route_advice_bits(shape, requests)
    )


def test_c2_route_codec_round_trips_and_rejects_bad_routes() -> None:
    shape = moe_shape(4, 2)
    parameters = random_parameters(shape, 1)
    routes = reference_routes(shape, parameters, MOE_REQUESTS)
    a = encode_routes(shape, routes)
    assert len(a) == (route_advice_bits(shape, MOE_REQUESTS) + 7) // 8
    assert decode_routes(shape, MOE_REQUESTS, a) == routes
    for route_list in routes:
        for step in route_list:
            for layer in step:
                for route in layer:
                    assert len(route) == 2 and route[0] < route[1] < 4
    with pytest.raises(ValueError):
        decode_routes(shape, MOE_REQUESTS, a + b"\0")
    repeated = bytes([0b00_00_0000]) + a[1:]  # the first route names expert 0 twice
    with pytest.raises(ValueError):
        decode_routes(shape, MOE_REQUESTS, repeated)


def test_c2_moe_routing_padded_and_advised() -> None:
    """E = 4, k = 1: both routes compute the reference; padding costs gates, advice costs bits."""

    pair = moe_pair(moe_shape(4, 1), MOE_REQUESTS)
    padded, advised = pair.padded, pair.advised
    tokens = sum(r.max_new for r in MOE_REQUESTS)
    assert advised.gates < padded.gates
    fed = sum(
        len(r.prompt) + r.max_new - 1 for r in MOE_REQUESTS
    )  # the last token is never fed back
    assert pair.exact_advice_bits == fed * pair.shape.route_bits == 10 * 2
    assert advised.advice_bits == pair.exact_advice_bits, (
        "the routes are charged at their exact bit length"
    )
    assert len(advised.compilation.advice) == 3  # ...though they occupy three bytes
    # the ok word is one more output per request but a check output: the verifier fixes it at 1,
    # so it carries no bits and lifts no kind's reach
    layout = RequestsG(pair.shape, ADVICE).output_layout(MOE_REQUESTS)
    assert list(advised.compiled.check_values()) == [
        (i, 1) for i, (_r, g) in enumerate(layout) if g < 0
    ]
    assert advised.out_bits == padded.out_bits == WIDTH * tokens
    assert padded.capped and advised.capped, "at this scale Bound is the interface"
    assert padded.bound_bits == advised.bound_bits == WIDTH * tokens
    # the step bodies are one definition per (context, positions) whatever the route: the advised
    # description is now the smaller one (only the chosen experts' work is described)
    assert advised.description_bytes < padded.description_bytes
    ratio = padded.gates / advised.gates
    record(
        [
            Row(
                id="C2a",
                what="MoE routing, padded: E=4 experts, top-1, 1 layer, d_model 4; every position runs every expert",
                mechanism="M5 in-circuit decision: router_topk VU (rank by lt chains) masks the experts' outputs",
                advice_bits=0,
                capacity_bits=math.ceil(padded.capacity),
                overhead=padded.overhead,
                description_bytes=padded.description_bytes,
                gates=padded.gates,
                verdict="no advice, E/k times the expert work; lowest U at equal theta, and at equal cost while E/k <= 4 (C2c)",
                notes=(
                    f"{tokens} tokens; {per_token(padded.gates, tokens)} gates/token vs {per_token(advised.gates, tokens)} advised "
                    f"({ratio:.2f}x); U = {padded.bound_bits:.0f} bits = the interface (capped); route check 0 bits."
                ),
            ),
            Row(
                id="C2b",
                what="MoE routing, advised: the same shape and requests; the route is advice, only chosen experts run",
                mechanism=(
                    "M4 charged advice: ceil(log2 E) bits per chosen expert per position per layer, charged exactly; "
                    "route_check VU folds into ok, a check output (0 bits)"
                ),
                advice_bits=advised.advice_bits,
                capacity_bits=math.ceil(advised.capacity),
                overhead=advised.overhead,
                description_bytes=advised.description_bytes,
                gates=advised.gates,
                verdict=(
                    "the honest server's pick once E/k is large (from E=16, k=1 under the VU-output interior, C2c): k/E of the expert compute "
                    "for k*log2(E) bits per position, spent on a stronger theta"
                ),
                notes=(
                    f"route description {pair.exact_advice_bits} bits, charged exactly ({len(advised.compilation.advice)} "
                    f"bytes on the wire, the padding checked zero); the ok word is a check output the verifier requires "
                    f"to be 1, so U = {advised.bound_bits:.0f} bits, the interface, as padded; description "
                    f"{advised.description_bytes} vs {padded.description_bytes} bytes at {tokens} tokens: the step bodies "
                    "are one definition per (context, positions) whatever the route -- the route enters at the call site as "
                    "the ranges passed for the router's columns and the chosen experts' weights -- so only the request "
                    "bodies (their call sites) are per request (see C2c at 128 tokens)."
                ),
            ),
        ]
    )


CROSSOVER_REQUESTS = tuple(
    Request(((i * 3) % 7 + 1, (i * 5) % 7 + 1), 4) for i in range(32)
)
"""32 requests of 4 tokens: enough interface that a strong policy takes Bound below it."""

GRID = tuple(
    VerificationPolicy(q, s)
    for q in (Fraction(1, 2), Fraction(1))
    for s in (
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(7, 8),
    )
)


@dataclass(frozen=True, slots=True)
class Priced:
    """One circuit under one policy: ``Bound + |a|``, the relative overhead and the absolute expected cost."""

    capacity: float
    capped: bool
    overhead: float
    absolute: float


def price(compiled: Compiled, advice_bits: int, policy: VerificationPolicy) -> Priced:
    table = as_kind_table(compiled)
    honest = next(row.replay_cost for row in table.rows if row.kind == table.root)
    expected = cost(compiled, policy).total
    bound = Bound(compiled, policy, ETA)
    return Priced(
        bound.bits + advice_bits,
        bound.capped,
        float(expected / honest),
        float(expected),
    )


def test_c2_crossover_in_experts_and_top_k() -> None:
    """Where padding beats advice: at equal policy always; at equal absolute prover cost only for small E/k.

    Compiles and prices without evaluating (the outputs are checked at the
    smaller scale above): the crossover needs enough interface that a strong
    policy takes ``Bound`` below it.
    """

    findings: list[str] = []
    winners: dict[tuple[int, int], str] = {}
    gate_set = make_isa_gate_set(WIDTH)
    tokens = sum(r.max_new for r in CROSSOVER_REQUESTS)
    for experts, top_k in ((2, 1), (4, 1), (8, 1), (8, 2), (16, 1)):
        # vocab 16 throughout: advised routing names experts by the constant table, so E <= vocab
        shape = moe_shape(experts, top_k, vocab=16)
        parameters = random_parameters(shape, 3)
        padded_g, advised_g = RequestsG(shape, PADDED), RequestsG(shape, ADVICE)
        padded = Compile(padded_g, CROSSOVER_REQUESTS, b"", gate_set).compiled
        a = advised_g.advice(CROSSOVER_REQUESTS, parameters)
        compilation = Compile(
            advised_g, CROSSOVER_REQUESTS, a, gate_set, max_advice_bits=8 * len(a)
        )
        advised, advice_bits = compilation.compiled, compilation.advice_bits
        assert advice_bits == route_advice_bits(shape, CROSSOVER_REQUESTS) <= 8 * len(a)
        descriptions = (
            len(padded_g(CROSSOVER_REQUESTS, b"")[0]),
            len(advised_g(CROSSOVER_REQUESTS, a)[0]),
        )
        grid_p = {policy: price(padded, 0, policy) for policy in GRID}
        grid_a = {policy: price(advised, advice_bits, policy) for policy in GRID}
        # (i) equal policy (hence equal relative overhead): padding wins by the advice (the ok words are free)
        assert all(grid_p[p].capacity < grid_a[p].capacity for p in GRID)
        assert all(abs(grid_p[p].overhead - grid_a[p].overhead) < 0.02 for p in GRID), (
            "relative overhead is theta's"
        )
        assert not grid_p[GRID[-1]].capped, (
            "the grid never leaves the interface cap; the crossover needs the knapsack regime"
        )
        # (ii) equal absolute prover cost: the padded server at the base policy sets the budget,
        # which the advised server, whose honest work is E/k times smaller, spends on a stronger theta
        budget = grid_p[POLICY].absolute
        best_padded = min(p.capacity for p in grid_p.values() if p.absolute <= budget)
        best_advised = min(
            (p.capacity for p in grid_a.values() if p.absolute <= budget),
            default=math.inf,
        )
        winner = "advice" if best_advised < best_padded else "padding"
        winners[(experts, top_k)] = winner
        findings.append(
            f"E={experts},k={top_k}: gates padded/advised {padded.circuit.n / advised.circuit.n:.2f}x "
            f"({padded.circuit.n} vs {advised.circuit.n}), description {descriptions[0]} vs {descriptions[1]} bytes, "
            f"|a| {advice_bits} b over {tokens} tokens; "
            f"at theta=(1/2,1/8) capacity {grid_p[POLICY].capacity:.0f} vs {grid_a[POLICY].capacity:.0f} "
            f"(overhead {grid_p[POLICY].overhead:.3f} vs {grid_a[POLICY].overhead:.3f}); "
            f"at equal absolute cost {budget:.0f}: padding {best_padded:.0f} vs advice {best_advised:.0f} -> {winner}"
        )
    assert winners[(2, 1)] == "padding" and winners[(16, 1)] == "advice", winners
    crossover = next((e, k) for (e, k), w in sorted(winners.items()) if w == "advice")
    record(
        [
            Row(
                id="C2c",
                what="MoE crossover sweep: (E, k) in {(2,1), (4,1), (8,1), (8,2), (16,1)}, vocab 16, 32 requests x 4 tokens, theta grid q in {1/2, 1}, s in {1/8..7/8}",
                mechanism="M5 vs M4 compared at equal theta (equal relative overhead) and at equal absolute prover cost",
                advice_bits=0,
                capacity_bits=0,
                overhead=0.0,
                description_bytes=0,
                gates=0,
                verdict=(
                    "at equal theta padding beats advice in U at every (E, k), by exactly |a| (the ok words are check "
                    f"outputs, 0 bits); at equal absolute prover cost advice wins from E={crossover[0]}, k={crossover[1]} on ("
                    + ", ".join(
                        f"E={e},k={k}:{w}" for (e, k), w in sorted(winners.items())
                    )
                    + ")"
                ),
                notes=" | ".join(findings),
            )
        ]
    )


# -- C3: speculative decoding -------------------------------------------------------------

TARGET = LMShape(vocab=8, d_model=4, heads=1, layers=1, context=64, width=WIDTH)
DRAFT = LMShape(vocab=8, d_model=2, heads=1, layers=1, context=64, width=WIDTH)
SPEC_REQUESTS = (Request((1, 2, 3), 6), Request((5, 6), 5))


@dataclass(frozen=True, slots=True)
class SpecPair:
    gamma: int
    target: Parameters
    draft: Parameters
    padded: Measured
    advised: Measured
    acceptances: tuple[tuple[int, ...], ...]


def spec_pair(
    gamma: int,
    requests: tuple[Request, ...],
    draft: Parameters | None = None,
    policy: VerificationPolicy = POLICY,
) -> SpecPair:
    """Both acceptances of one (target, draft, gamma), checked against the target decoding alone."""

    target = random_parameters(TARGET, 1)
    draft = random_parameters(DRAFT, 2) if draft is None else draft
    weights = (*target.flatten(), *draft.flatten())
    reference = reference_generate(TARGET, target, requests)
    traces = reference_speculative(target, draft, gamma, requests)
    for trace, expected, request in zip(traces, reference, requests, strict=True):
        assert trace.tokens[: request.max_new] == expected  # greedy acceptance is exact
    padded_g = SpeculativeG(TARGET, draft.shape, gamma, PADDED)
    padded = measure(padded_g, requests, b"", weights, policy)
    assert padded_g.tokens(padded.outputs, requests) == reference
    advised_g = SpeculativeG(TARGET, draft.shape, gamma, ADVICE)
    a = advised_g.advice(requests, target, draft)
    advised = measure(advised_g, requests, a, weights, policy)
    assert advised_g.tokens(advised.outputs, requests) == reference
    assert advised_g.checks(advised.outputs, requests) == (1,) * len(requests)
    return SpecPair(
        gamma, target, draft, padded, advised, tuple(t.acceptances for t in traces)
    )


def test_c3_acceptance_codec_round_trips() -> None:
    target, draft = random_parameters(TARGET, 1), random_parameters(DRAFT, 2)
    for gamma in (2, 4):
        traces = reference_speculative(target, draft, gamma, SPEC_REQUESTS)
        acceptances = tuple(t.acceptances for t in traces)
        a = encode_acceptances(gamma, acceptances)
        assert decode_acceptances(gamma, SPEC_REQUESTS, a) == acceptances
        assert acceptance_bits(gamma) == math.ceil(math.log2(gamma + 2))
        with pytest.raises(ValueError):
            decode_acceptances(gamma, SPEC_REQUESTS, a + b"\0")


def test_c3_speculative_decoding_padded_and_advised() -> None:
    """gamma = 2 with a random draft and a perfect one: padding multiplies both the work and the interface."""

    tokens = sum(r.max_new for r in SPEC_REQUESTS)
    poor = spec_pair(2, SPEC_REQUESTS)
    perfect = spec_pair(
        2, SPEC_REQUESTS, draft=random_parameters(TARGET, 1)
    )  # the draft is the target
    assert all(m == 2 for steps in perfect.acceptances for m in steps)
    assert perfect.advised.advice_bits < poor.advised.advice_bits
    assert poor.padded.gates > 2 * poor.advised.gates
    for pair in (poor, perfect):
        assert pair.padded.gates > pair.advised.gates
        assert pair.padded.out_bits == WIDTH * sum(
            1 + (r.max_new - 1) * 3 for r in SPEC_REQUESTS
        )
        # the ok words are check outputs: one more output per request, 0 bits
        assert pair.advised.out_bits == WIDTH * tokens
        assert len(list(pair.advised.compiled.check_values())) == len(SPEC_REQUESTS)
        assert pair.advised.advice_bits == sum(
            len(s) for s in pair.acceptances
        ) * acceptance_bits(2)
    wide = spec_pair(4, SPEC_REQUESTS)
    exact_bits = {
        p.gamma: sum(len(s) for s in p.acceptances) * acceptance_bits(p.gamma)
        for p in (poor, wide)
    }
    assert wide.advised.advice_bits == exact_bits[4]
    record(
        [
            Row(
                id="C3a",
                what="speculative decoding, padded: gamma=2, target d_model 4, draft d_model 2 (random weights), 2 requests",
                mechanism="M5 in-circuit acceptance: eq per position, prefix product, masked slots, V entries masked by the flags",
                advice_bits=0,
                capacity_bits=math.ceil(poor.padded.capacity),
                overhead=poor.padded.overhead,
                description_bytes=poor.padded.description_bytes,
                gates=poor.padded.gates,
                verdict="never the honest server's pick: (gamma+1)x the target positions and the interface, max_new-1 steps",
                notes=(
                    f"{tokens} tokens; {per_token(poor.padded.gates, tokens)} gates/token vs {per_token(poor.advised.gates, tokens)} advised; "
                    f"outputs {poor.padded.out_bits // WIDTH} slots (blank = vocab) vs {tokens} tokens; U = {poor.padded.bound_bits:.0f} bits (capped); "
                    f"gamma=4: {wide.padded.gates} gates, U {wide.padded.bound_bits:.0f}. The blanks make each step's m output-determined (route c, not taken)."
                ),
            ),
            Row(
                id="C3b",
                what="speculative decoding, advised: the same models and requests; m per step is advice",
                mechanism=(
                    "M4 charged advice: ceil(log2(gamma+2)) bits per step, charged exactly; acceptance_check VU folds "
                    "'exactly m agree' into ok, a check output (0 bits)"
                ),
                advice_bits=poor.advised.advice_bits,
                capacity_bits=math.ceil(poor.advised.capacity),
                overhead=poor.advised.overhead,
                description_bytes=poor.advised.description_bytes,
                gates=poor.advised.gates,
                verdict="the honest server's pick: the target does exactly plain decoding's positions plus the draft's",
                notes=(
                    f"acceptances {poor.acceptances}: {poor.advised.advice_bits} bits, charged exactly "
                    f"({len(poor.advised.compilation.advice)} bytes on the wire); "
                    f"perfect draft (= target): acceptances {perfect.acceptances}, {perfect.advised.advice_bits} advice bits, {perfect.advised.gates} gates; "
                    f"gamma=4 random draft: {wide.advised.advice_bits} advice bits, {wide.advised.gates} gates; "
                    f"the ok word is a check output the verifier requires to be 1: 0 bits, U = {poor.advised.bound_bits:.0f} = "
                    f"the {tokens} tokens' interface. The token count is output-determined; each m is not."
                ),
            ),
        ]
    )


# -- the protocol: honest runs and dishonest servers ---------------------------------------


def test_protocol_moe_honest_and_dishonest() -> None:
    """E = 4, k = 1: both routes accepted honestly; a wrong route shows ok = 0 or a broken relation."""

    pair = moe_pair(moe_shape(4, 1), MOE_REQUESTS)
    weights = pair.parameters.flatten()
    for measured in (pair.padded, pair.advised):
        report = protocol(measured, weights, POLICY)
        assert report.accepted, report
    # the standard adversary: one request's last token forced in its head VU, caught under FULL
    advised_g = RequestsG(pair.shape, ADVICE)
    layout = advised_g.output_layout(MOE_REQUESTS)
    (carrier,) = adversary.carriers(layout, 1)
    assert layout[carrier][1] == MOE_REQUESTS[layout[carrier][0]].max_new - 1, (
        "the carrier is a token, not ok"
    )
    secret = format(
        (pair.advised.outputs[carrier] + 1) % pair.shape.vocab,
        f"0{pair.shape.vocab_bits}b",
    )
    attack = adversary.plan_attack(
        pair.advised.compiled,
        pair.advised.inputs,
        weights,
        layout,
        secret,
        pair.shape.vocab_bits,
    )
    assert attack.carriers == (carrier,) and len(attack.corrupted) == 1
    report = protocol(
        pair.advised,
        weights,
        FULL,
        outputs=attack.outputs,
        values=attack.values,
        dishonest=True,
    )
    assert not report.accepted and report.code is VerificationCode.RELATION_REJECTED
    assert set(attack.verification_units) & set(report.sampled_verification_units)
    # a dishonest route in the advice, computed honestly: the check word comes out 0
    routes = reference_routes(pair.shape, pair.parameters, MOE_REQUESTS)
    (honest_route,) = routes[0][1][
        0
    ]  # request 0, first decode step, layer 0, its one position
    wrong = tuple((e + 1) % pair.shape.experts for e in honest_route)
    lying = ((routes[0][0], ((wrong,),), *routes[0][2:]), *routes[1:])
    a = encode_routes(pair.shape, lying)
    lied = measure(advised_g, MOE_REQUESTS, a, weights)
    checks = tuple(
        v
        for v, (_r, g) in zip(
            lied.outputs, advised_g.output_layout(MOE_REQUESTS), strict=True
        )
        if g < 0
    )
    assert checks == (0, 1), "the lied-about request's ok word must be 0"
    # ok is a check output: a run that reports it as computed is rejected before anything is opened...
    report = protocol(lied, weights, POLICY)
    assert not report.accepted and report.code is VerificationCode.CHECK_MISMATCH
    # ...and a server that claims ok = 1 instead breaks route_check's relation, which FULL catches
    ok_address = check_address(lied, advised_g.output_layout(MOE_REQUESTS), 0)
    forced = adversary.evaluate_with_overrides(
        lied.compiled, lied.inputs, weights, {ok_address: 1}
    )
    outputs = tuple(forced[address] for address in lied.compiled.circuit.outputs)
    report = protocol(
        lied, weights, FULL, outputs=outputs, values=forced, dishonest=True
    )
    assert not report.accepted and report.code is VerificationCode.RELATION_REJECTED


def test_protocol_speculative_honest_and_dishonest() -> None:
    """gamma = 2 advised: accepted honestly; an overstated acceptance shows ok = 0, and forcing ok breaks a relation."""

    pair = spec_pair(2, SPEC_REQUESTS)
    weights = (*pair.target.flatten(), *pair.draft.flatten())
    for measured in (pair.padded, pair.advised):
        report = protocol(measured, weights, POLICY)
        assert report.accepted, report
    advised_g = SpeculativeG(TARGET, DRAFT, 2, ADVICE)
    honest = pair.acceptances
    steps = list(honest[0])
    index = next(i for i, m in enumerate(steps) if m < 2)
    steps[index] += 1  # claim one more draft token than the target agreed with
    kept, emitted = (
        [],
        1,
    )  # the advice is self-delimiting: keep the steps that start before max_new
    for m in steps:
        if emitted >= SPEC_REQUESTS[0].max_new:
            break
        kept.append(m)
        emitted += m + 1
    lying = (tuple(kept), *honest[1:])
    a = encode_acceptances(2, lying)
    lied = measure(advised_g, SPEC_REQUESTS, a, weights)
    assert advised_g.checks(lied.outputs, SPEC_REQUESTS)[0] == 0
    report = protocol(
        lied, weights, POLICY
    )  # reported as computed: the check output is not 1
    assert not report.accepted and report.code is VerificationCode.CHECK_MISMATCH
    ok_address = check_address(lied, advised_g.output_layout(SPEC_REQUESTS), 0)
    forced = adversary.evaluate_with_overrides(
        lied.compiled, lied.inputs, weights, {ok_address: 1}
    )
    outputs = tuple(forced[address] for address in lied.compiled.circuit.outputs)
    report = protocol(
        lied, weights, FULL, outputs=outputs, values=forced, dishonest=True
    )
    assert not report.accepted and report.code is VerificationCode.RELATION_REJECTED
