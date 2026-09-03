"""Stress scenarios priced through the protocol: N4, N5 (M6 fault declarations), S7, C1, W1-W3.

Each test builds a real run, drives the real protocol and asserts a real
verdict, then records one row of ``docs/data/stress-protocol.json`` (merged
by id, catalogue order, the same shape as every ``docs/data/stress*.json``
the report tool reads).  The M6 scenarios use the small simulation of
:mod:`veritor.simulation.datacenter`: two pods of two slots, sixteen steps,
a dozen requests, the schedule as advice.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from veritor.analysis.bound import bound
from veritor.analysis.cost import cost
from veritor.analysis.faults import unit_fault_bits
from veritor.compile import Compilation, Constructor
from veritor.constructors.cluster import ClusterG
from veritor.constructors.lm import (
    LMShape,
    Parameters,
    random_parameters,
    reference_generate,
)
from veritor.constructors.requests import RequestsG
from veritor.constructors.schedule import Request
from veritor.constructors.tenancy import AdaptedRequestsG, ModelsG
from veritor.constructors.truncation import TruncatedRequestsG, field_width
from veritor.core import Compiled, VerificationPolicy, as_kind_table, make_isa_gate_set
from veritor.protocol import (
    Declare,
    MerkleTree,
    ProtocolRun,
    VerificationCode,
    VerifierParameters,
    Weights,
    assignment_replay,
    commit_weights,
    honest_declare,
    make_expectation,
    run_protocol,
    self_check,
)
from veritor.research import Compile
from veritor.simulation import adversary
from veritor.simulation.datacenter import ETA, POLICY, small_config
from veritor.simulation.faults import (
    SDC_RATE_PER_DEVICE_HOUR,
    Fault,
    FaultInjector,
    expected_faults,
    fault_budget,
    is_dot_unit,
)
from veritor.simulation.workload import simulate

# -- the rows -------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[3] / "docs" / "data" / "stress-protocol.json"
_IDENTIFIER = re.compile(r"([A-Z])(\d+)([a-z]*)")
_SECTIONS = "SCNWE"
FULL = VerificationPolicy(1, 1)
"""Every RU replayed, every VU sampled: the verdict is deterministic."""


@dataclass(frozen=True, slots=True)
class Row:
    id: str
    what: str
    mechanism: str
    advice_bits: int
    capacity_bits: int
    overhead: float
    description_bytes: int
    verdict: str
    notes: str = ""


def _row_key(identifier: str) -> tuple[int, int, str]:
    match = _IDENTIFIER.fullmatch(identifier)
    assert match is not None, identifier
    letter, number, suffix = match.groups()
    return (_SECTIONS.index(letter), int(number), suffix)


def record(*rows: Row, path: Path = DATA) -> dict[str, dict[str, Any]]:
    """Merge ``rows`` by id into the JSON file (one row per line, catalogue order) and return it."""

    stored: dict[str, dict[str, Any]] = {}
    if path.exists() and path.read_text(encoding="utf-8").strip():
        stored = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        body = asdict(row)
        del body["id"]
        body["overhead"] = round(float(body["overhead"]), 6)
        stored[row.id] = body
    ordered = sorted(stored, key=_row_key)
    lines = ["{"]
    for position, identifier in enumerate(ordered):
        body_text = json.dumps(stored[identifier], sort_keys=True, ensure_ascii=False)
        lines.append(f' "{identifier}": {body_text}{"," if position + 1 < len(ordered) else ""}')
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stored


# -- one priced run -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Priced:
    """A compiled run with everything a row and a protocol run need."""

    compilation: Compilation
    weights: tuple[int, ...]
    kappa: Weights
    weight_tree: MerkleTree
    values: dict[int, int]
    outputs: tuple[int, ...]
    description_bytes: int
    policy: VerificationPolicy

    @property
    def compiled(self) -> Compiled:
        return self.compilation.compiled

    @property
    def advice_bits(self) -> int:
        return self.compilation.advice_bits

    @property
    def honest_cost(self) -> Fraction:
        table = as_kind_table(self.compiled)
        return next(row.replay_cost for row in table.rows if row.kind == table.root)

    @property
    def overhead(self) -> float:
        return float(cost(self.compiled, self.policy).total / self.honest_cost)

    def capacity(self, max_faults: int = 0) -> int:
        """``Bound + |a|`` at the row's policy and ``eta = 2^-40``, in whole bits."""

        return math.ceil(bound(self.compiled, self.policy, ETA, max_faults=max_faults).bits) + self.advice_bits

    def run(
        self,
        values: Mapping[int, int] | None = None,
        outputs: Sequence[int] | None = None,
        *,
        policy: VerificationPolicy | None = None,
        max_faults: int = 0,
        declare: bool | Declare = False,
        label: str = "",
    ) -> ProtocolRun:
        """Prover commits ``values`` as computed (the honest assignment by default).

        ``declare=True`` is the honest server's policy; a callable is the prover's own.
        """

        values = self.values if values is None else values
        outputs = self.outputs if outputs is None else outputs
        seed = f"veritor/stress/protocol/{label}".encode().ljust(32, b"\0")[:32]
        expectation = make_expectation(
            self.compilation,
            self.policy if policy is None else policy,
            tuple(outputs),
            parameters=VerifierParameters(
                ETA, max_capacity=1 << 20, max_advice_bits=self.advice_bits, max_faults=max_faults
            ),
            weights=self.kappa,
            session_id=seed[:16],
            q_seed=seed,
            s_seed=bytes(reversed(seed)),
        )
        policy_of_prover: Declare | None
        if declare is True:
            policy_of_prover = honest_declare(self.compiled)
        elif declare is False:
            policy_of_prover = None
        else:
            policy_of_prover = declare
        return run_protocol(
            self.compiled,
            expectation,
            values,
            replay=assignment_replay(values),
            weight_tree=self.weight_tree,
            declare=policy_of_prover,
        )


def price(
    constructor: Constructor,
    x: object,
    advice: bytes,
    shape: LMShape,
    weights: Sequence[int],
    *,
    policy: VerificationPolicy = POLICY,
    limits: Any = None,
) -> Priced:
    gate_set = make_isa_gate_set(shape.width)
    compilation = Compile(constructor, x, advice, gate_set, limits=limits, max_advice_bits=8 * len(advice))
    circuit = compilation.compiled.circuit
    values = dict(enumerate(circuit.evaluate(compilation.inputs, weights)))
    kappa, tree = commit_weights(gate_set, weights)
    return Priced(
        compilation=compilation,
        weights=tuple(weights),
        kappa=kappa,
        weight_tree=tree,
        values=values,
        outputs=tuple(values[a] for a in circuit.outputs),
        description_bytes=len(constructor(x, advice)[0]),
        policy=policy,
    )


def rejected_at(run: ProtocolRun, unit: int) -> bool:
    """The run was rejected by the relation check of a sample that contains VU ``unit``."""

    return (
        run.report.code is VerificationCode.RELATION_REJECTED
        and unit in run.report.sampled_verification_units
    )


def served_with(
    priced: Priced, *, inputs: Sequence[int] | None = None, weights: Sequence[int] | None = None
) -> dict[int, int]:
    """The assignment of a server that computed with other inputs or weights than the statement's.

    Its boundary (the ``in`` gates) and its weights are the public ones -- the
    verifier checks those against ``x`` and the weight root -- so the gates
    that read what was substituted are the ones that disobey.
    """

    circuit = priced.compiled.circuit
    values = dict(
        enumerate(
            circuit.evaluate(
                priced.compilation.inputs if inputs is None else tuple(inputs),
                priced.weights if weights is None else tuple(weights),
            )
        )
    )
    for address, value in zip(circuit.inputs, priced.compilation.inputs, strict=True):
        values[address] = value
    for address, value in zip(circuit.weights, priced.weights, strict=True):
        values[address] = value
    return values


# -- the small simulation -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Small:
    priced: Priced
    shape: LMShape
    parameters: Parameters
    requests: tuple[Request, ...]
    pods: int
    steps: int
    tokens: int
    injector: FaultInjector


@pytest.fixture(scope="module")
def small() -> Small:
    config = small_config()
    shape = config.shape
    parameters = random_parameters(shape, config.parameters_seed)
    simulation = simulate(config.workload, shape, parameters)
    advice = simulation.schedule.encode()
    constructor = ClusterG(shape, config.workload.pods, config.workload.slots, config.workload.steps)
    priced = price(
        constructor,
        simulation.requests,
        advice,
        shape,
        parameters.flatten(),
        limits=config.compilation_limits,
    )
    streamed = tuple(token for response in simulation.streamed for token in response)
    assert priced.outputs == streamed, "the circuit's outputs are the streamed tokens"
    return Small(
        priced,
        shape,
        parameters,
        simulation.requests,
        config.workload.pods,
        config.workload.steps,
        len(streamed),
        FaultInjector(priced.compiled, priced.compilation.inputs, priced.weights),
    )


def faults(small: Small, units: Sequence[int], bit: int) -> Iterator[Fault]:
    for unit in units:
        yield small.injector.inject(unit, bit)


def owning_unit(compiled: Compiled, address: int) -> int:
    """The global index of the VU holding gate ``address``."""

    index = compiled.index
    block = index.verification_units(index.replay_units.owner(address))
    return block.first + block.owner(address)


def dot_candidates(compiled: Compiled, units: Sequence[int]) -> list[int]:
    """The dot product VUs among ``units``, in order."""

    return [u for u in units if is_dot_unit(compiled, compiled.index.verification_unit(u))]


def units_of(compiled: Compiled, replay_unit: int) -> range:
    """The global VU indices inside ``replay_unit``."""

    block = compiled.index.verification_units(replay_unit)
    return range(block.first, block.first + block.count)


def logit_dots(compiled: Compiled) -> list[int]:
    """The dot VUs whose output a head VU reads: the logits, one per streamed token and vocab entry."""

    circuit = compiled.circuit
    found: list[int] = []
    for output in circuit.outputs:
        head = compiled.index.verification_unit(owning_unit(compiled, output))
        inside = set(head.interval)
        for address in head.interval:
            for argument in circuit[address].args:
                if argument in inside or circuit[argument].is_source:
                    continue
                unit = owning_unit(compiled, argument)
                if unit not in found:
                    found.append(unit)
    return dot_candidates(compiled, found)


def device_hours_note(small: Small) -> tuple[int, str]:
    """``f_max`` from the brief's SDC rate and the run's device-hours, with the arithmetic."""

    step_seconds = 0.05  # one decode step of a production server, an assumption stated in the row
    run_hours = small.pods * small.steps * step_seconds / 3600
    run_mean = expected_faults(run_hours)
    f_run = fault_budget(run_mean)
    hour_mean = expected_faults(16_384 * 1.0)
    day_mean = expected_faults(16_384 * 24.0)
    note = (
        f"f_max from the Llama-3 SDC rate {SDC_RATE_PER_DEVICE_HOUR:.2e}/device-hour (6 events, 54 days, "
        f"16,384 GPUs): this run is {small.pods} pods x {small.steps} steps at {step_seconds:.2f} s/step = "
        f"{run_hours * 3600:.1f} device-seconds = {run_hours:.1e} device-hours, mean {run_mean:.1e} faults, "
        f"f_max = {f_run} (the floor: admitting declarations at all admits one). A 16,384-GPU fleet for an hour "
        f"has mean {hour_mean:.2e} -> f_max = {fault_budget(hour_mean)} at tail 1e-6; for a day mean "
        f"{day_mean:.3f} -> f_max = {fault_budget(day_mean)}."
    )
    return f_run, note


# -- N4: silent data corruption in a sampled VU ----------------------------------------


def test_n4_silent_corruption_rejected_without_declaration_accepted_with(small: Small) -> None:
    priced = small.priced
    f_max, arithmetic = device_hours_note(small)
    assert f_max == 1
    # The q-challenge depends on the header and the boundary only (max_faults is in the header, so
    # the two headers open different RUs); the s-challenge also on the committed interiors, so which
    # VUs a faulty run gets sampled is known only by running it.
    honest = {f: priced.run(max_faults=f, label=f"n4/{f}") for f in (0, f_max)}
    assert all(run.report.accepted for run in honest.values())
    opened = sorted(
        set(honest[0].report.sampled_replay_units) & set(honest[f_max].report.sampled_replay_units)
    )
    assert opened, "some RU is opened under both headers"
    candidates = [
        unit for replay_unit in opened for unit in dot_candidates(priced.compiled, units_of(priced.compiled, replay_unit))
    ][::7]  # every seventh, to spread the attempts over the opened RUs
    escaped = 0
    for fault in faults(small, candidates, bit=0):
        if fault.changed_outputs:
            continue  # the boundary changed with the tokens, so did J: not the run we are studying
        assert self_check(priced.compiled, fault.replay_unit, fault.values) == (fault.verification_unit,)
        without = priced.run(fault.values, fault.outputs, max_faults=0, label="n4/0")
        if rejected_at(without, fault.verification_unit):
            break
        assert without.report.accepted, without.report  # the s-challenge missed it: completeness 1 - s
        escaped += 1
    else:  # pragma: no cover - each attempt is sampled with probability s = 1/8
        pytest.fail("no candidate fault was sampled")

    # The same faulty run under a header that admits one declaration: the server declares it.
    with_declaration = priced.run(fault.values, fault.outputs, max_faults=f_max, declare=True, label=f"n4/{f_max}")
    assert with_declaration.report.accepted, with_declaration.report
    assert with_declaration.transcript is not None
    assert fault.replay_unit in with_declaration.report.sampled_replay_units
    assert with_declaration.transcript.interiors.declarations == (fault.verification_unit,)
    where = (
        "declared and sampled: openings authenticated, relation skipped"
        if fault.verification_unit in with_declaration.report.sampled_verification_units
        else "declared; the s-challenge did not sample it"
    )
    # A dishonest server cannot pardon a second, undeclared corruption with the same budget.
    second = next(unit for unit in candidates if unit != fault.verification_unit)
    second_address = priced.compiled.index.verification_unit(second).interval[-1]
    two = small.injector.propagate({fault.address: 1, second_address: 1})
    declare_first: Declare = lambda unit, _values: (fault.verification_unit,) if unit == fault.replay_unit else ()
    hidden = priced.run(
        two,
        tuple(two[a] for a in priced.compiled.circuit.outputs),
        policy=FULL,
        max_faults=f_max,
        declare=declare_first,
        label="n4/hidden",
    )
    assert rejected_at(hidden, second), hidden.report

    u1 = unit_fault_bits(priced.compiled)
    at_policy = bound(priced.compiled, POLICY, ETA)
    with_faults = bound(priced.compiled, POLICY, ETA, max_faults=f_max)
    full_0 = bound(priced.compiled, FULL, ETA).bits
    full_f = bound(priced.compiled, FULL, ETA, max_faults=f_max).bits
    assert full_f - full_0 == pytest.approx(f_max * u1)
    table = as_kind_table(priced.compiled)
    units = sum(row.copies for row in table.rows if row.role == "verification")
    record(
        Row(
            id="N4",
            what=(
                f"silent data corruption: bit {fault.bit} of one dot product's output word flipped in RU "
                f"{fault.replay_unit} of the small simulation ({small.pods} pods x {small.steps} steps, "
                f"{len(small.requests)} requests, {small.tokens} streamed tokens, {priced.compiled.circuit.n} gates, "
                f"{units} VUs); the server streamed the consequences ({fault.changed_outputs} tokens changed) "
                f"and finds the VU when it replays the opened RU"
            ),
            mechanism="M6",
            advice_bits=priced.advice_bits,
            capacity_bits=priced.capacity(f_max),
            overhead=priced.overhead,
            description_bytes=priced.description_bytes,
            verdict=(
                f"f_max = 0: RELATION_REJECTED at the faulty VU once the s-challenge sampled it (it escaped "
                f"{escaped} earlier challenge{'s' if escaped != 1 else ''} at s = 1/8); f_max = {f_max}: ACCEPTED, "
                f"{where}; a second undeclared corruption under the same budget, everything sampled: RELATION_REJECTED"
            ),
            notes=(
                f"u(1) = log2(1 + |S| 2^W_V) = {u1:.2f} bits (|S| = {units}, W_V = {round(u1 - math.log2(units))}); "
                f"Bound at theta = (1/2, 1/8): {at_policy.bits:.0f} -> {with_faults.bits:.0f} bits, both capped at "
                f"out_bits = {at_policy.out_bits} (a {small.tokens}-token run); uncapped, at theta = (1, 1): "
                f"{full_0:.0f} -> {full_f:.2f} = f_max * u(1). {arithmetic} Adaptive caveat: the prover declares "
                f"after seeing J, so f * u(1) prices a D fixed before the q-challenge; see docs/stress-tests.md M6."
            ),
        )
    )


# -- N5: a wrong token that was streamed ---------------------------------------------


def test_n5_wrong_token_streamed_unopened_and_unsampled(small: Small) -> None:
    priced = small.priced
    f_max = 1
    candidates = logit_dots(priced.compiled)
    assert len(candidates) >= small.tokens, "one logit dot per token at least"
    unopened: tuple[Fault, ProtocolRun] | None = None
    unsampled: tuple[Fault, ProtocolRun] | None = None
    top_bit = small.shape.width - 1
    for fault in faults(small, candidates, bit=top_bit):
        if not fault.changed_outputs:
            continue  # a flipped top bit of a logit that did not move the token: not this scenario
        run = priced.run(fault.values, fault.outputs, max_faults=f_max, declare=True, label="n5")
        assert run.report.accepted, run.report
        assert run.transcript is not None
        declared = run.transcript.interiors.declarations
        if fault.replay_unit not in run.report.sampled_replay_units:
            assert declared == (), "nothing to declare: the faulty RU was not opened"
            unopened = unopened or (fault, run)
        elif fault.verification_unit not in run.report.sampled_verification_units:
            assert declared == (fault.verification_unit,), "declared although the s-challenge missed it"
            unsampled = unsampled or (fault, run)
        else:
            assert declared == (fault.verification_unit,)
        if unopened and unsampled:
            break
    assert unopened is not None and unsampled is not None
    fault_a, run_a = unopened
    fault_b, _run_b = unsampled
    # Without the mechanism the opened fault is a coin flip on the s-challenge; the unopened one is invisible.
    assert priced.run(fault_a.values, fault_a.outputs, max_faults=0, label="n5").report.accepted
    record(
        Row(
            id="N5",
            what=(
                f"a GPU produces a wrong token that was streamed: the top bit of a logit dot's output word "
                f"flips, the sampler draws a different token and the request continues from it "
                f"({fault_a.changed_outputs} and {fault_b.changed_outputs} tokens changed in the two runs); "
                f"outputs stand as streamed"
            ),
            mechanism="M6",
            advice_bits=priced.advice_bits,
            capacity_bits=priced.capacity(f_max),
            overhead=priced.overhead,
            description_bytes=priced.description_bytes,
            verdict=(
                f"fault in RU {fault_a.replay_unit}, not opened by the q-challenge: nothing declared, ACCEPTED "
                f"({len(run_a.report.sampled_replay_units)} of {priced.compiled.index.replay_units.count} RUs opened); "
                f"fault in RU {fault_b.replay_unit}, opened but VU {fault_b.verification_unit} not sampled: declared "
                f"anyway (the server cannot know the s-challenge), ACCEPTED"
            ),
            notes=(
                "The honest server self-checks every opened RU against the values it streamed and declares what "
                "disagrees, before the s-challenge exists; a declaration the sample misses costs nothing beyond the "
                "f_max * u(1) already in Bound. At f_max = 0 the unopened fault is accepted too (the verifier never "
                "sees that RU) and the opened one is rejected iff sampled: completeness 1 - s per opened fault."
            ),
        )
    )


# -- S7: client disconnect / abort ------------------------------------------------------


def _requests(shape: LMShape, seed: int, count: int, max_new: int) -> tuple[Request, ...]:
    import random

    rng = random.Random(seed)
    return tuple(
        Request(
            tuple(rng.randrange(shape.vocab) for _ in range(rng.randint(2, 4))),
            max_new,
            tuple(rng.randrange(1 << shape.random_bits) for _ in range(max_new)) if shape.sampling else (),
        )
        for _ in range(count)
    )


def test_s7_client_disconnect_length_as_advice() -> None:
    shape = small_config().shape
    parameters = random_parameters(shape, 0)
    weights = parameters.flatten()
    max_new = 8
    x = _requests(shape, 7, 6, max_new)
    lengths = (8, 3, 5, 1, 6, 2)  # what each client received before it went away
    constructor = TruncatedRequestsG(shape)
    advice = constructor.advice(x, lengths)
    assert constructor.lengths(x, advice) == lengths
    assert constructor.advice_bits(x) == 6 * field_width(max_new) == 18
    assert 8 * len(advice) == 24  # charged: padded to whole bytes
    truncated = price(constructor, x, advice, shape, weights)
    # The truncated run streams exactly the prefixes of the full generations.
    full_reference = reference_generate(shape, parameters, x)
    assert truncated.outputs == tuple(
        token for tokens, t in zip(full_reference, lengths, strict=True) for token in tokens[:t]
    )
    assert truncated.run(label="s7").report.accepted
    # The same circuit as a run that asked for t tokens: the length advice adds no kind and no capacity.
    asked = price(RequestsG(shape), constructor.truncated(x, advice), b"", shape, weights)
    assert asked.compiled.digest == truncated.compiled.digest
    assert truncated.capacity() == asked.capacity() + 8 * len(advice)
    # Advice that lies about a length names a different circuit; its outputs are not what was streamed.
    lying = constructor.advice(x, (8, 4, 5, 1, 6, 2))
    assert Compile(constructor, x, lying, make_isa_gate_set(shape.width), max_advice_bits=24).compiled.digest != truncated.compiled.digest
    # The alternative: pad every request to max_new and let the verifier ignore the tail.
    padded = price(RequestsG(shape), x, b"", shape, weights)
    extra = float(padded.honest_cost / truncated.honest_cost) - 1
    generated, asked_for = sum(lengths), 6 * max_new
    record(
        Row(
            id="S7",
            what=(
                f"client disconnect / abort: 6 requests asked for max_new = {max_new}, the clients received "
                f"{lengths} tokens ({generated} of {asked_for}); request RUs, the generated length as advice"
            ),
            mechanism="M4 (charged as advice; M3 open)",
            advice_bits=8 * len(advice),
            capacity_bits=truncated.capacity(),
            overhead=truncated.overhead,
            description_bytes=truncated.description_bytes,
            verdict="ACCEPTED; the circuit is byte-identical to RequestsG over requests that asked for t tokens",
            notes=(
                f"t as advice: ceil(log2 max_new) = {field_width(max_new)} bits per request, {constructor.advice_bits(x)} "
                f"bits, charged {8 * len(advice)} after padding to bytes. Alternative, padding to max_new with the tail "
                f"masked: 0 advice bits, {extra:.0%} more prover compute for this batch ({padded.honest_cost:.0f} vs "
                f"{truncated.honest_cost:.0f} honest gate-cost units) and {padded.description_bytes} description bytes. "
                "Whether output-determined shape (len(y) = t, no EOS rule to check) may be charged 0 like EOS "
                "termination is an open theory question for the architect; here it is charged."
            ),
        )
    )


# -- C1: nondeterministic sampling over published randomness -------------------------


def test_c1_sampling_over_public_randomness_biased_token_is_caught(small: Small) -> None:
    shape = small.shape
    weights = small.parameters.flatten()
    requests = small.requests[:5]  # enough tokens to carry a secret, small enough to sample everything
    sampling = price(RequestsG(shape), requests, b"", shape, weights)
    argmax_shape = LMShape(shape.vocab, shape.d_model, shape.heads, shape.layers, shape.context, shape.width)
    argmax_requests = tuple(Request(r.prompt, r.max_new) for r in requests)
    argmax = price(RequestsG(argmax_shape), argmax_requests, b"", argmax_shape, random_parameters(argmax_shape, 0).flatten())
    tokens = len(sampling.outputs)  # every request run to max_new (no EOS or run end here)
    assert tokens == sum(request.max_new for request in requests)
    sample_size = len(
        sampling.compiled.index.verification_unit(owning_unit(sampling.compiled, sampling.compiled.circuit.outputs[0])).interval
    )
    argmax_size = len(
        argmax.compiled.index.verification_unit(owning_unit(argmax.compiled, argmax.compiled.circuit.outputs[0])).interval
    )
    # Per token: the head's extra gates plus the in gate of the random word; per run: the sampler's two constants.
    extra = sampling.compiled.circuit.n - argmax.compiled.circuit.n
    assert extra == tokens * (sample_size - argmax_size + 1) + len(shape.sampler_constants)
    per_token = sample_size - argmax_size + 1
    assert sampling.run(label="c1").report.accepted
    # The adversary decides tokens instead of sampling them: one head VU per carrier disobeys.
    layout = RequestsG(shape).output_layout(requests)
    secret = adversary.random_secret(4 * shape.vocab_bits, "c1")
    attack = adversary.plan_attack(sampling.compiled, sampling.compilation.inputs, weights, layout, secret, shape.vocab_bits)
    assert attack.corrupted, "at least one carrier had to change"
    caught = sampling.run(attack.values, attack.outputs, policy=FULL, label="c1")
    assert caught.report.code is VerificationCode.RELATION_REJECTED
    assert set(attack.verification_units) & set(caught.report.sampled_verification_units)
    predicted = adversary.predicted_survival(POLICY, attack)
    trials = 400
    escaped = adversary.survival_trials(sampling.compiled, POLICY, attack, trials, label="c1")
    p = float(predicted)
    sigma = math.sqrt(p * (1 - p) / trials)
    assert abs(escaped / trials - p) <= 4 * sigma
    record(
        Row(
            id="C1",
            what=(
                f"nondeterministic sampling: the LM head draws each token from a public {shape.random_bits}-bit "
                f"random word in x (sample VU, {sample_size} gates over vocab {shape.vocab}, vs {argmax_size} for the "
                f"argmax); {len(requests)} requests, {tokens} tokens, request RUs"
            ),
            mechanism="M7 + M5",
            advice_bits=0,
            capacity_bits=sampling.capacity(),
            overhead=sampling.overhead,
            description_bytes=sampling.description_bytes,
            verdict=(
                f"honest run ACCEPTED; a server that biases {len(attack.corrupted)} sampled tokens is "
                f"RELATION_REJECTED at a head VU under theta = (1, 1), and escapes theta = (1/2, 1/8) in "
                f"{escaped}/{trials} fresh challenges against sigma(E) = {p:.3f} predicted"
            ),
            notes=(
                f"The sampling head adds {per_token} gate(s) per token over the argmax head: the sample VU has "
                f"{sample_size} gates vs {argmax_size} for the argmax at vocab {shape.vocab}, plus the in gate of the random "
                f"word ({sampling.compiled.circuit.n} vs {argmax.compiled.circuit.n} gates for the same requests, "
                f"{len(shape.sampler_constants)} sampler constants as weights); the randomness is {shape.random_bits} public "
                "bits per token, advice 0. Biasing a token is a relation violation of the sample VU, priced by Bound like "
                "any corruption: the server never chooses its randomness. (Catalogue row C5; recorded as C1 per the brief.)"
            ),
        )
    )


# -- W1-W3: weights, versions, adapters -----------------------------------------------


def test_w1_hot_swap_two_versions_under_one_root() -> None:
    shape = small_config().shape
    versions = (random_parameters(shape, 0), random_parameters(shape, 1))
    constructor = ModelsG(shape, 2, "input")
    requests = _requests(shape, 11, 6, 4)
    x = tuple((i % 2, request) for i, request in enumerate(requests))  # the swap: odd requests on version 1
    weights = constructor.flatten_weights(versions)
    priced = price(constructor, x, b"", shape, weights)
    expected = tuple(
        token for model, request in x for token in reference_generate(shape, versions[model], (request,))[0]
    )
    assert priced.outputs == expected
    assert priced.run(label="w1").report.accepted
    # A server that quietly served version 1 to every request (the swap it did not announce): its
    # interiors are computed from version 1's weights while the committed weights are the real ones.
    served = served_with(priced, weights=constructor.flatten_weights((versions[1], versions[1])))
    outputs = tuple(served[a] for a in priced.compiled.circuit.outputs)
    assert outputs != priced.outputs
    swapped = priced.run(served, outputs, policy=FULL, label="w1")
    assert swapped.report.code is VerificationCode.RELATION_REJECTED, swapped.report
    single = price(RequestsG(shape), requests, b"", shape, versions[0].flatten())
    record(
        Row(
            id="W1",
            what=(
                "model version hot swap: two weight sets of one shape in one run, six requests alternating "
                "versions; one weights RU holds both, each request's kind is wired to its version's slice"
            ),
            mechanism="M2 (one root over both versions; version per request public in x)",
            advice_bits=0,
            capacity_bits=priced.capacity(),
            overhead=priced.overhead,
            description_bytes=priced.description_bytes,
            verdict=(
                f"ACCEPTED, outputs equal each version's reference; a server that served version 1 to a version-0 "
                f"request is {swapped.report.code.name} under theta = (1, 1)"
            ),
            notes=(
                f"No protocol change: Header.weights is the root over the concatenation ({len(weights)} weight gates, "
                f"vs {len(versions[0].flatten())} for one version; description {priced.description_bytes} vs "
                f"{single.description_bytes} bytes). Gap vs M8: the joint root changes whenever the served set "
                "changes, so a version has no root of its own across sessions, and the request names its version "
                "through the description's wiring, not by a root id; the fix is Header.weights as a tuple of roots "
                "each covering a declared rank range of the weight gates."
            ),
        )
    )


def test_w2_per_request_adapter_as_boundary_inputs() -> None:
    shape = small_config().shape
    base = random_parameters(shape, 0)
    constructor = AdaptedRequestsG(shape, layer=0, matrix="w_1")
    d, hidden, mask = shape.d_model, shape.hidden, (1 << shape.width) - 1
    import random

    def merged(seed: int) -> tuple[tuple[int, ...], ...]:
        rng = random.Random(seed)
        a = [rng.randrange(1 << 4) for _ in range(d)]  # rank-1 LoRA: A is d x 1, B is 1 x hidden
        b = [rng.randrange(1 << 4) for _ in range(hidden)]
        w1 = base.layers[0].w_1
        return tuple(tuple((w1[i][j] + a[i] * b[j]) & mask for j in range(hidden)) for i in range(d))

    tenants = (merged(1), merged(2), merged(3))
    requests = _requests(shape, 13, 6, 4)
    x = tuple((tuple(v for row in tenants[i % 3] for v in row), request) for i, request in enumerate(requests))
    weights = base.flatten()
    priced = price(constructor, x, b"", shape, weights)
    expected = tuple(
        token
        for i, request in enumerate(requests)
        for token in reference_generate(shape, constructor.merged(base, tenants[i % 3]), (request,))[0]
    )
    assert priced.outputs == expected
    assert priced.run(label="w2").report.accepted
    # A server that ignored the tenants' adapters and ran the base matrix: the boundary holds the
    # published adapters, the dots that read them were computed from W_1.
    base_words = tuple(v for row in base.layers[0].w_1 for v in row)
    ignored_inputs = tuple(
        value
        for _adapter, request in x
        for value in (*base_words, *request.prompt, *request.randomness)
    )
    served = served_with(priced, inputs=ignored_inputs)
    outputs = tuple(served[a] for a in priced.compiled.circuit.outputs)
    plain = price(RequestsG(shape), requests, b"", shape, weights)
    assert outputs == plain.outputs != priced.outputs
    ignored = priced.run(served, outputs, policy=FULL, label="w2")
    assert ignored.report.code is VerificationCode.RELATION_REJECTED, ignored.report
    record(
        Row(
            id="W2",
            what=(
                f"LoRA adapters per tenant: each request carries its tenant's merged matrix W_1 + A B "
                f"({constructor.count} words, layer 0) as in gates of an adapter RU; three tenants, six requests"
            ),
            mechanism="M1 (adapter as public input, committed in the boundary)",
            advice_bits=0,
            capacity_bits=priced.capacity(),
            overhead=priced.overhead,
            description_bytes=priced.description_bytes,
            verdict=(
                f"ACCEPTED, outputs equal each tenant's reference; a server that ran the base matrix for a tenant "
                f"is {ignored.report.code.name} under theta = (1, 1)"
            ),
            notes=(
                f"No protocol change: {constructor.count} more boundary positions per request "
                f"({len(priced.compilation.inputs)} public inputs vs {len(plain.compilation.inputs)}), the request kind "
                "is unchanged (its ports are the same width; some are now in gates). Bound "
                f"{plain.capacity()} -> {priced.capacity()} bits. What this is not: the adapter is public and per "
                "run, not a server-held weight under its own root (M8), and the toy LM has no low-rank path, so "
                "the merged d x hidden matrix stands in for A and B."
            ),
        )
    )


def test_w3_router_picks_a_model_public_or_advised() -> None:
    shape = small_config().shape
    models = (random_parameters(shape, 0), random_parameters(shape, 1), random_parameters(shape, 2))
    requests = _requests(shape, 17, 6, 4)
    choice = (0, 2, 1, 1, 0, 2)
    public = ModelsG(shape, 3, "input")
    advised = ModelsG(shape, 3, "advice")
    weights = public.flatten_weights(models)
    by_input = price(public, tuple(zip(choice, requests, strict=True)), b"", shape, weights)
    advice = advised.advice(requests, choice)
    by_advice = price(advised, requests, advice, shape, weights)
    assert advised.advice_bits(requests) == 6 * field_width(3) == 12 and 8 * len(advice) == 16
    assert by_input.compiled.digest == by_advice.compiled.digest
    assert by_input.outputs == by_advice.outputs
    assert by_input.run(label="w3").report.accepted and by_advice.run(label="w3").report.accepted
    assert by_advice.capacity() == by_input.capacity() + 8 * len(advice)
    record(
        Row(
            id="W3",
            what="several models on one cluster: three weight sets of one shape, a router assigns six requests",
            mechanism="M2 (client chose, model in x: 0 bits) / M4 (server chose: advice)",
            advice_bits=8 * len(advice),
            capacity_bits=by_advice.capacity(),
            overhead=by_advice.overhead,
            description_bytes=by_advice.description_bytes,
            verdict="both ACCEPTED on the same circuit (same digest); the advised route costs exactly its bits",
            notes=(
                f"Server-chosen routing: ceil(log2 3) = {field_width(3)} bits per request, "
                f"{advised.advice_bits(requests)} bits, charged {8 * len(advice)} after padding; Bound "
                f"{by_input.capacity()} -> {by_advice.capacity()}. Gap: models of different shapes need one "
                "description with several kind families (the tracer builds one ToyLM family per shape) and, "
                "as for W1, a root per model rather than one root over the concatenation."
            ),
        )
    )
