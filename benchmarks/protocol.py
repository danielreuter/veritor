"""Benchmark 7 -- the protocol end to end on ``ClusterG`` and ``RequestsG``.

``ProverSession`` and ``VerifierSession`` are driven message by message so
each phase is timed around the public API: the prover's boundary commitment,
its replay of the selected replay units (RUs) and the interior commitments
(split by timing the ``replay`` callback), its openings; the verifier's
admission, its boundary check and challenge derivation, its interior check,
and its evidence check, split into Merkle openings (``verify_opening`` is
wrapped for the duration of the run) and gate recomputation.  Message sizes
are the canonical JSON of each message's manifest and of the whole transcript.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from veritor import Compile
from veritor.core import Compiled, VerificationPolicy, canonical_json_bytes
from veritor.protocol import (
    ProverSession,
    VerifierParameters,
    VerifierSession,
    commit_weights,
    encode_transcript,
    make_expectation,
)
from veritor.protocol import session as session_module
from veritor.protocol.merkle import MerkleTree
from veritor.protocol.session import replay_unit

from ._harness import Benchmark, Point, Scale, Series, measure
from ._synthetic import CLUSTER_LADDER, ISA, DecoderCase, decoder_case


def seeds(index: int) -> dict[str, bytes]:
    """Deterministic session seeds; the selections `J`, `T` are pseudorandom in them."""

    tag = index.to_bytes(4, "big")
    return {
        "session_id": b"benchmark-run" + tag[1:],
        "q_seed": (b"Q" * 28) + tag,
        "s_seed": (b"S" * 28) + tag,
    }


PARAMETERS = VerifierParameters(max_advice_bits=1 << 24, max_capacity=None)
Q_VALUES = (Fraction(1, 32), Fraction(1, 8), Fraction(1, 2), Fraction(1))
S_VALUES = (Fraction(1, 64), Fraction(1, 16), Fraction(1, 4), Fraction(1))
DEFAULT_POLICY = VerificationPolicy(Fraction(1, 2), Fraction(1, 4))
PROTOCOL_LADDER: tuple[DecoderCase, ...] = (
    *CLUSTER_LADDER[:3],
    decoder_case(
        "d32-L2-r4", d_model=32, layers=2, prompt=4, max_new=4, requests=4, slots=4
    ),
)
"""Sizes the whole protocol runs at within a couple of minutes each; `d32-L2-r4` is about 10^6 gates."""


@dataclass(slots=True)
class Scenario:
    """A compiled workload with the honest prover's values and the model's ``kappa_W``."""

    label: str
    constructor: str
    compilation: object
    compiled: Compiled
    values: dict[int, int]
    outputs: tuple[int, ...]
    weights: object
    weight_tree: MerkleTree
    evaluate_s: float
    kappa_w_s: float
    description_bytes: int


def prepare(case: DecoderCase, constructor: str, scale: Scale) -> Scenario:
    G = case.cluster if constructor == "cluster" else case.per_request
    advice = case.advice if constructor == "cluster" else b""
    description, _ = G(case.requests, advice)
    compilation = Compile(G, case.requests, advice, ISA, max_advice_bits=1 << 24)
    compiled = compilation.compiled
    flat = case.weights()
    evaluate = measure(
        lambda: compiled.circuit.evaluate(compilation.inputs, flat),
        scale,
        repeats=1,
        memory=False,
    )
    values = dict(enumerate(evaluate.result))
    outputs = tuple(values[a] for a in compiled.circuit.outputs)
    kappa = measure(lambda: commit_weights(ISA, flat), scale, repeats=1, memory=False)
    weights, tree = kappa.result
    return Scenario(
        case.label,
        constructor,
        compilation,
        compiled,
        values,
        outputs,
        weights,
        tree,
        evaluate.time_s,
        kappa.time_s,
        len(description),
    )


def _run_once(
    scenario: Scenario, policy: VerificationPolicy, seed: int
) -> dict[str, float]:
    compiled = scenario.compiled
    timings: dict[str, float] = {}
    replay_time = 0.0
    merkle_time = 0.0
    merkle_calls = 0

    def timed_replay(unit: int, boundary_values):
        nonlocal replay_time
        start = time.perf_counter()
        result = replay_unit(compiled, unit, boundary_values)
        replay_time += time.perf_counter() - start
        return result

    original = session_module.verify_opening

    def timed_verify(*args, **kwargs):
        nonlocal merkle_time, merkle_calls
        start = time.perf_counter()
        result = original(*args, **kwargs)
        merkle_time += time.perf_counter() - start
        merkle_calls += 1
        return result

    def clock(name: str, fn: Callable[[], object]) -> object:
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    expectation = make_expectation(
        scenario.compilation,
        policy,
        scenario.outputs,
        parameters=PARAMETERS,
        weights=scenario.weights,
        **seeds(seed),
    )
    session_module.verify_opening = timed_verify
    try:
        verifier = clock(
            "verifier_admit_s", lambda: VerifierSession(expectation, compiled)
        )
        prover = clock(
            "prover_setup_s",
            lambda: ProverSession(
                compiled,
                verifier.header,
                scenario.values,
                replay=timed_replay,
                weight_tree=scenario.weight_tree,
            ),
        )
        boundary = clock("prover_commit_boundary_s", prover.boundary)
        merkle_before = merkle_time
        replay_challenge = clock(
            "verifier_boundary_s", lambda: verifier.receive_boundary(boundary)
        )
        timings["verifier_boundary_merkle_s"] = merkle_time - merkle_before
        interiors = clock(
            "prover_interiors_s", lambda: prover.interiors(replay_challenge)
        )
        timings["prover_replay_s"] = replay_time
        timings["prover_commit_interior_s"] = (
            timings["prover_interiors_s"] - replay_time
        )
        sample_challenge = clock(
            "verifier_interiors_s", lambda: verifier.receive_interiors(interiors)
        )
        evidence = clock("prover_prove_s", lambda: prover.evidence(sample_challenge))
        merkle_before, calls_before = merkle_time, merkle_calls
        report = clock(
            "verifier_evidence_s", lambda: verifier.receive_evidence(evidence)
        )
        timings["verifier_merkle_s"] = merkle_time - merkle_before
        timings["verifier_recompute_s"] = (
            timings["verifier_evidence_s"] - timings["verifier_merkle_s"]
        )
        timings["openings_verified"] = merkle_calls - calls_before
    finally:
        session_module.verify_opening = original
    if not report.accepted:
        raise RuntimeError(f"honest run rejected: {report.code}: {report.detail}")
    timings["prover_total_s"] = (
        timings["prover_setup_s"]
        + timings["prover_commit_boundary_s"]
        + timings["prover_interiors_s"]
        + timings["prover_prove_s"]
    )
    timings["verifier_total_s"] = (
        timings["verifier_admit_s"]
        + timings["verifier_boundary_s"]
        + timings["verifier_interiors_s"]
        + timings["verifier_evidence_s"]
    )
    index = compiled.index
    timings["boundary_count"] = index.boundary().count
    timings["selected_replay_units"] = len(replay_challenge.selected)
    timings["interior_positions"] = sum(
        index.interior(u).count for u in replay_challenge.selected
    )
    kinds = {kind.kind: kind for kind in index.kinds()}
    timings["replayed_gates"] = sum(
        kinds[node.kind].size
        - kinds[node.kind].source_inputs
        - kinds[node.kind].source_weights
        for node in (index.replay_units.unit(u) for u in replay_challenge.selected)
    )
    timings["sampled_verification_units"] = len(sample_challenge.selected)
    timings["openings"] = sum(len(batch) for batch in evidence.units)
    timings["boundary_bytes"] = len(canonical_json_bytes(boundary.manifest))
    timings["interiors_bytes"] = len(canonical_json_bytes(interiors.manifest))
    timings["evidence_bytes"] = len(canonical_json_bytes(evidence.manifest))
    timings["transcript_bytes"] = len(encode_transcript(verifier.transcript))
    return timings


def run_policy(
    scenario: Scenario, policy: VerificationPolicy, scale: Scale, *, seed_count: int = 1
) -> dict[str, float]:
    """Mean over session seeds of every phase (the selections are random in the seed), one run per seed
    unless the first run is well inside the budget, in which case each seed's run is repeated and its median taken."""

    per_seed: list[dict[str, float]] = []
    for seed in range(seed_count):
        runs = [_run_once(scenario, policy, seed)]
        wanted = 2 if scale.quick else scale.repeats
        while (
            len(runs) < wanted
            and seed_count
            * sum(r["prover_total_s"] + r["verifier_total_s"] for r in runs)
            < scale.budget_s
        ):
            runs.append(_run_once(scenario, policy, seed))
        per_seed.append(
            {key: statistics.median(r[key] for r in runs) for key in runs[0]}
        )
        per_seed[-1]["repeats"] = len(runs)
    result = {key: statistics.fmean(r[key] for r in per_seed) for key in per_seed[0]}
    result["seeds"] = seed_count
    return result


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _point(
    scenario: Scenario,
    policy: VerificationPolicy,
    x: float,
    scale: Scale,
    *,
    seed_count: int = 1,
) -> Point:
    timings = run_policy(scenario, policy, scale, seed_count=seed_count)
    compiled = scenario.compiled
    repeats = int(timings.pop("repeats"))
    positions = timings["interior_positions"]
    openings = timings["openings"]
    constants = {
        "replay_us_per_gate": _rate(
            1e6 * timings["prover_replay_s"], timings["replayed_gates"]
        ),
        "replay_us_per_position": _rate(1e6 * timings["prover_replay_s"], positions),
        "commit_us_per_position": _rate(
            1e6 * timings["prover_commit_interior_s"], positions
        ),
        "prove_us_per_opening": _rate(1e6 * timings["prover_prove_s"], openings),
        "verify_merkle_us_per_opening": _rate(
            1e6 * timings["verifier_merkle_s"], openings
        ),
        "verify_recompute_us_per_opening": _rate(
            1e6 * timings["verifier_recompute_s"], openings
        ),
        "evidence_bytes_per_opening": _rate(timings["evidence_bytes"], openings),
    }
    return Point(
        x,
        time_s=timings["prover_total_s"],
        repeats=repeats,
        extra={
            "case": scenario.label,
            "q": str(policy.q),
            "s": str(policy.s),
            "n": compiled.circuit.n,
            "replay_units": compiled.index.replay_units.count,
            "verification_units": compiled.index.verification_unit_count,
            "weights": compiled.index.weight_count,
            "description_bytes": scenario.description_bytes,
            "evaluate_s": scenario.evaluate_s,
            "gates_per_s": compiled.circuit.n / scenario.evaluate_s,
            "kappa_w_s": scenario.kappa_w_s,
            **timings,
            **constants,
        },
    )


PHASES = (
    "time_s",
    "verifier_total_s",
    "prover_commit_boundary_s",
    "prover_replay_s",
    "prover_commit_interior_s",
    "prover_prove_s",
    "verifier_boundary_s",
    "verifier_interiors_s",
    "verifier_recompute_s",
    "verifier_merkle_s",
    "transcript_bytes",
    "evaluate_s",
)


def run(scale: Scale) -> Benchmark:
    started = time.perf_counter()
    bench = Benchmark(
        "protocol",
        "The protocol end to end: prover and verifier phases, message bytes",
        "`time_s` is the prover's total (setup, boundary commitment, replay + interior commitments, openings); "
        "`verifier_total_s` the verifier's (admission, boundary check and `J`, interior check and `T`, "
        "evidence check).  `evaluate_s` is the honest computation itself through the lazy circuit "
        "(`circuit.evaluate`, `gates_per_s`), `kappa_w_s` the one-off weight commitment.  "
        "Sizes: `boundary_count = |∂|`, `replayed_gates` recomputed (every non-source gate of the selected RUs), "
        "`interior_positions` committed (the VU outputs among them that are not RU outputs), `openings` sent, "
        "message bytes as canonical JSON.",
    )

    ladder = PROTOCOL_LADDER[: 2 if scale.quick else 4]
    for constructor in ("cluster", "requests"):
        series = Series(
            f"{constructor}_vs_n",
            "n (gates)",
            fit_columns=PHASES,
            note=f"`{'ClusterG' if constructor == 'cluster' else 'RequestsG'}` at growing model width, "
            f"`q = {DEFAULT_POLICY.q}, s = {DEFAULT_POLICY.s}`.  "
            + (
                "Steps are the RUs: the boundary is the KV cache and the tokens, so `|∂|` grows with `n`."
                if constructor == "cluster"
                else "Requests are the RUs: the boundary is prompts and tokens only, the interiors hold everything else."
            ),
        )
        for case in ladder:
            scenario = prepare(case, constructor, scale)
            series.points.append(
                _point(scenario, DEFAULT_POLICY, scenario.compiled.circuit.n, scale)
            )
        bench.series.append(series)

    case = (
        decoder_case(
            "d8-r8", d_model=8, layers=1, prompt=3, max_new=3, requests=8, slots=2
        )
        if scale.quick
        else decoder_case(
            "d8-r32", d_model=8, layers=1, prompt=3, max_new=3, requests=32, slots=2
        )
    )
    scenario = prepare(case, "cluster", scale)
    seed_count = 1 if scale.quick else 2
    series = Series(
        "cluster_vs_q",
        "q",
        fit_columns=PHASES,
        note=f"`ClusterG` `{case.label}` ({scenario.compiled.index.replay_units.count} RUs), `s = 1/4`, mean over "
        f"{seed_count} session seed(s): the replay phase and the interior commitments scale with the selected RUs, "
        "`q * #RU`, so the realized counts (`selected_replay_units`, `interior_positions`) are the honest x-axis.",
    )
    for q in Q_VALUES:
        series.points.append(
            _point(
                scenario,
                VerificationPolicy(q, Fraction(1, 4)),
                float(q),
                scale,
                seed_count=seed_count,
            )
        )
    bench.series.append(series)

    series = Series(
        "cluster_vs_s",
        "s",
        fit_columns=PHASES,
        note=f"`ClusterG` `{case.label}`, `q = 1/2`, mean over {seed_count} session seed(s): the openings and the "
        "verifier's evidence check scale with the sampled VUs, `q s * #VU`; the replay phase depends only on `J`, "
        "which is redrawn per policy (the policy is in the header digest), hence its scatter.",
    )
    for s in S_VALUES:
        series.points.append(
            _point(
                scenario,
                VerificationPolicy(Fraction(1, 2), s),
                float(s),
                scale,
                seed_count=seed_count,
            )
        )
    bench.series.append(series)

    bench.seconds = time.perf_counter() - started
    return bench
