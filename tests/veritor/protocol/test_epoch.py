"""The epoch layer on the matmul fixture: rounds of runs sealed before they are challenged."""

from __future__ import annotations

import hashlib
import math
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction

import pytest

from veritor.analysis import bound, union
from veritor.analysis.probability import survival
from veritor.constructors import expected_matmul_outputs
from veritor.core import Compiled, VerificationPolicy
from veritor.protocol import (
    BoundaryMessage,
    Claim,
    Header,
    Opening,
    ProtocolError,
    ProverSession,
    Reject,
    VerificationCode,
    VerifierParameters,
    VerifierSession,
    assignment_replay,
    encode_transcript,
    honest_declare,
    run_protocol,
    verify_transcript,
)
from veritor.protocol.epoch import (
    EpochParameters,
    EpochProver,
    EpochVerifier,
    RoundChallenge,
    Run,
    derive_run_seed,
    run_epoch,
    stream_link,
)
from veritor.protocol.phases import boundary_phase
from veritor.simulation.datacenter import TOLERANCE_SIGMAS

from .conftest import Q_SEED, S_SEED

ETA = Fraction(1, 8)
HALF = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
EVERYTHING = VerificationPolicy(1, 1)
NOTHING = VerificationPolicy(1, 0)
POLICIES = [
    EVERYTHING,
    NOTHING,
    HALF,
    VerificationPolicy(Fraction(1, 3), Fraction(3, 4)),
]


def seed(label: str, trial: int = 0) -> bytes:
    return hashlib.sha256(
        f"tests/veritor/protocol/epoch/{label}/{trial}".encode()
    ).digest()


def parameters(
    policy: VerificationPolicy = EVERYTHING, **kwargs: object
) -> EpochParameters:
    kwargs.setdefault("max_capacity", None)
    return EpochParameters(ETA, policy, **kwargs)  # type: ignore[arg-type]


def relation_units(compiled: Compiled) -> list[int]:
    """One VU with a relation per replay unit that has one, in RU order."""

    index, circuit = compiled.index, compiled.circuit
    units: list[int] = []
    for replay_unit in range(index.replay_units.count):
        block = index.verification_units(replay_unit)
        for offset in range(block.count):
            node = block.unit(offset)
            if not all(circuit[address].is_source for address in node.interval):
                units.append(block.first + offset)
                break
    return units


def flipped(
    compiled: Compiled, honest: dict[int, object], unit: int
) -> tuple[dict[int, object], tuple[int, ...]]:
    """An assignment with bit 0 of VU ``unit``'s output word flipped, and the outputs it streams.

    The matmul VUs' output words are circuit outputs: exactly one relation fails.
    """

    values = dict(honest)
    address = compiled.index.verification_unit(unit).interval[-1]
    values[address] = values[address] ^ 1  # type: ignore[operator]
    return values, tuple(values[a] for a in compiled.circuit.outputs)  # type: ignore[misc]


@pytest.fixture
def make_run(compilation, workload, model_weights, honest_values):
    def build(
        values=None, outputs=None, *, session_id: bytes = b"epoch-run", **kwargs
    ) -> Run:
        return Run(
            compilation,
            honest_values if values is None else values,
            expected_matmul_outputs(workload) if outputs is None else outputs,
            weights=model_weights[0],
            weight_tree=model_weights[1],
            session_id=session_id,
            **kwargs,
        )

    return build


def committed(
    verifier: EpochVerifier, prover: EpochProver, run: Run, label: str
) -> Header:
    """Admit ``run`` into the open round and send its boundary."""

    header = verifier.admit(
        run.compilation,
        run.claimed_outputs,
        weights=run.weights,
        session_id=seed(label)[:16],
    )
    message = prover.boundary(
        run.compilation.compiled,
        header,
        run.values,
        replay=run.replay,
        weight_tree=run.weight_tree,
        declare=run.declare,
    )
    verifier.receive_boundary(header, message)
    return header


def answer(
    verifier: EpochVerifier, prover: EpochProver, challenge: RoundChallenge
) -> None:
    for header, replay_challenge in challenge.challenges:
        try:
            sample = verifier.receive_interiors(
                header, prover.interiors(header, replay_challenge)
            )
            verifier.receive_evidence(header, prover.evidence(header, sample))
        except Reject:
            continue


# -- one run, one round: the protocol as it was ----------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_an_epoch_of_one_run_is_the_protocol_under_the_derived_seeds(
    compiled, compilation, honest_values, model_weights, expect, make_run, policy
) -> None:
    """Same run, same seeds: the epoch's verdict and sample are ``run_protocol``'s."""

    values, outputs = flipped(compiled, honest_values, relation_units(compiled)[0])
    for run in (
        make_run(),
        make_run(values, outputs, replay=assignment_replay(values)),
    ):
        verifier = EpochVerifier(parameters(policy))
        prover = EpochProver()
        header = committed(verifier, prover, run, "one")
        assert header.eta == ETA and header.policy == policy and header.max_faults == 0
        round_seed = seed("one/round")
        challenge = verifier.close_round(round_seed)
        answer(verifier, prover, challenge)
        report = verifier.report()

        expectation = expect(
            policy,
            parameters=VerifierParameters(ETA, max_capacity=None),
            claimed_outputs=run.claimed_outputs,
            session_id=header.session_id,
            q_seed=derive_run_seed(round_seed, challenge.seal, 0, 0, b"q"),
            s_seed=derive_run_seed(round_seed, challenge.seal, 0, 0, b"s"),
        )
        assert VerifierSession(expectation, compiled).header == header
        single = run_protocol(compiled, expectation, run.values, replay=run.replay)
        assert report.accepted == single.report.accepted
        assert report.rounds[0].runs[0].report == single.report
        assert report.capacity_bits == bound(compiled, policy, ETA).bits
        assert report.run_count == 1 and len(report.rounds) == 1
        if single.report.accepted:
            assert single.transcript is not None
            data = encode_transcript(verifier.transcript(header))
            assert verify_transcript(data, expectation, compiled) == single.report
            assert (
                prover.transcript(header)
                == verifier.transcript(header)
                == single.transcript
            )
        else:
            assert report.code is single.report.code
            assert single.report.detail in report.detail


def test_run_epoch_drives_the_same_thing(compiled, make_run) -> None:
    report = run_epoch(parameters(HALF), [make_run()], [[0]], [seed("drive")])
    assert (
        report.accepted
        and report.code is VerificationCode.ACCEPTED
        and report.detail == ""
    )
    assert (
        report.run_count == 1 and report.rounds[0].closed and report.rounds[0].accepted
    )
    assert report.capacity_bits == bound(compiled, HALF, ETA).bits
    assert report.rounds[0].seal is not None and len(report.rounds[0].seal) == 32
    assert report.rounds[0].declarations == 0 and report.rounds[0].refused == ()


def test_the_header_is_the_sessions_header_and_binds_no_seed(
    compiled, expect, make_run
) -> None:
    run = make_run()
    verifier = EpochVerifier(
        parameters(HALF, rounds=4, max_faults=3, max_advice_bits=16)
    )
    header = verifier.admit(
        run.compilation, run.claimed_outputs, weights=run.weights, session_id=b"same"
    )
    expectation = expect(
        HALF,
        parameters=VerifierParameters(
            ETA / 4, max_capacity=None, max_advice_bits=16, max_faults=3
        ),
        session_id=b"same",
    )
    assert VerifierSession(expectation, compiled).header == header
    assert (
        VerifierSession(
            replace(expectation, q_seed=S_SEED, s_seed=Q_SEED), compiled
        ).header
        == header
    )
    assert header.eta == ETA / 4 and header.max_faults == 3
    with pytest.raises(ProtocolError, match="already admitted"):
        verifier.admit(
            run.compilation,
            run.claimed_outputs,
            weights=run.weights,
            session_id=b"same",
        )


# -- the stream and its seal -------------------------------------------------------------


def test_the_seal_is_the_chain_over_headers_and_boundaries(compiled, make_run) -> None:
    runs = [make_run(session_id=f"run{i}".encode()) for i in range(3)]
    verifier = EpochVerifier(parameters(HALF))
    prover = EpochProver()
    genesis = verifier.link
    assert (
        genesis != EpochVerifier(parameters(EVERYTHING)).link
    )  # the parameters are in the chain
    assert genesis == EpochVerifier(parameters(HALF)).link
    link = genesis
    for index, run in enumerate(runs):
        header = committed(verifier, prover, run, f"chain/{index}")
        message = verifier.stream[-1]
        assert isinstance(message, BoundaryMessage) and verifier.stream[-2] == header
        link = stream_link(link, header, boundary_phase(header, message))
        assert verifier.link == link
    challenge = verifier.close_round(seed("chain/round"))
    assert challenge.seal == link and challenge.round == 0
    assert [h for h, _ in challenge.challenges] == [
        s for s in verifier.stream if isinstance(s, Header)
    ]
    for index, (header, replay_challenge) in enumerate(challenge.challenges):
        assert replay_challenge.seed == derive_run_seed(
            seed("chain/round"), link, 0, index, b"q"
        )


def test_altering_the_stream_changes_the_seal_and_the_challenges(
    compiled, honest_values, make_run
) -> None:
    """Drop, reorder, insert or alter a boundary: a different seal, hence different seeds."""

    values, outputs = flipped(compiled, honest_values, relation_units(compiled)[0])
    runs = [
        make_run(),
        make_run(),
        make_run(values, outputs, replay=assignment_replay(values)),
    ]
    round_seed = seed("alter/round")

    def sealed(*members: tuple[int, str]) -> RoundChallenge:
        """A round of the given runs, each admitted under the session id of its label."""

        verifier, prover = EpochVerifier(parameters(HALF)), EpochProver()
        for index, label in members:
            committed(verifier, prover, runs[index], f"alter/{label}")
        return verifier.close_round(round_seed)

    reference = sealed((0, "a"), (1, "b"))
    variants = {
        "dropped": sealed((0, "a")),
        "reordered": sealed((1, "b"), (0, "a")),
        "inserted": sealed((0, "a"), (1, "b"), (1, "c")),
        "altered boundary (another run's values)": sealed((0, "a"), (2, "b")),
        "altered header (another session id)": sealed((0, "a"), (1, "c")),
    }
    for name, variant in variants.items():
        assert variant.seal != reference.seal, name
        theirs = {c.seed for _, c in variant.challenges}
        assert not theirs & {c.seed for _, c in reference.challenges}, name
    assert sealed((0, "a"), (1, "b")).seal == reference.seal  # deterministic


def test_a_tampered_boundary_is_rejected_and_does_not_enter_the_stream(
    compiled, make_run
) -> None:
    run = make_run()
    verifier, prover = EpochVerifier(parameters(HALF)), EpochProver()
    header = verifier.admit(run.compilation, run.claimed_outputs, weights=run.weights)
    message = prover.boundary(compiled, header, run.values, weight_tree=run.weight_tree)
    before = verifier.link
    first = message.io_openings[0]
    forged = replace(
        message,
        io_openings=(
            Opening(
                first.position,
                bytes([first.value[0] ^ 1]) + first.value[1:],
                first.path,
            ),
        )
        + message.io_openings[1:],
    )
    with pytest.raises(Reject) as rejection:
        verifier.receive_boundary(header, forged)
    assert rejection.value.code is VerificationCode.INVALID_OPENING
    assert verifier.link == before and verifier.stream == [header]
    with pytest.raises(
        Reject
    ) as second:  # the run's verdict stands; the honest boundary is too late
        verifier.receive_boundary(header, message)
    assert second.value.code is VerificationCode.INVALID_PHASE
    verifier.close_round(seed("tamper"))
    report = verifier.report()
    assert (
        report.code is VerificationCode.INVALID_OPENING and "opening" in report.detail
    )
    assert not report.accepted and report.rounds[0].runs[0].report is not None


def test_a_boundary_is_taken_once_from_an_admitted_run_in_its_own_round(
    compiled, make_run
) -> None:
    run = make_run()
    verifier, prover = EpochVerifier(parameters(HALF, rounds=2)), EpochProver()
    stranger = EpochVerifier(parameters(HALF, rounds=2))
    header = verifier.admit(
        run.compilation, run.claimed_outputs, weights=run.weights, session_id=b"a"
    )
    foreign = stranger.admit(
        run.compilation, run.claimed_outputs, weights=run.weights, session_id=b"b"
    )
    message = prover.boundary(compiled, header, run.values, weight_tree=run.weight_tree)
    with pytest.raises(Reject) as unknown:
        verifier.receive_boundary(foreign, message)
    assert unknown.value.code is VerificationCode.EXPECTATION_MISMATCH
    verifier.receive_boundary(header, message)
    with pytest.raises(Reject) as again:
        verifier.receive_boundary(header, message)
    assert again.value.code is VerificationCode.INVALID_PHASE
    assert "already received" in str(again.value)

    late = verifier.admit(
        run.compilation, run.claimed_outputs, weights=run.weights, session_id=b"late"
    )
    late_message = EpochProver().boundary(
        compiled, late, run.values, weight_tree=run.weight_tree
    )
    challenge = verifier.close_round(seed("once/0"))
    assert [h for h, _ in challenge.challenges] == [
        header
    ]  # the late run has no challenge
    with pytest.raises(Reject) as closed:
        verifier.receive_boundary(late, late_message)
    assert closed.value.code is VerificationCode.INVALID_PHASE and "closed" in str(
        closed.value
    )
    answer(verifier, prover, challenge)
    verifier.close_round(seed("once/1"))
    report = verifier.report()
    assert not report.accepted and report.code is VerificationCode.INVALID_PHASE
    assert "never arrived" in report.detail and b"late".hex() in report.detail
    assert report.rounds[0].runs[0].accepted and not report.rounds[0].runs[1].accepted
    assert report.run_count == 2 and report.rounds[1].runs == ()


# -- no challenge before the seal ------------------------------------------------------


def _bytes_in(obj: object, seen: set[int], depth: int = 0) -> set[bytes]:
    """Every ``bytes`` reachable from ``obj`` through attributes and containers."""

    if id(obj) in seen or depth > 8:
        return set()
    seen.add(id(obj))
    if isinstance(obj, bytes):
        return {obj}
    found: set[bytes] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            found |= _bytes_in(key, seen, depth + 1) | _bytes_in(value, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            found |= _bytes_in(item, seen, depth + 1)
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            found |= _bytes_in(getattr(obj, f.name), seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        found |= _bytes_in(vars(obj), seen, depth + 1)
    for name in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, name):
            found |= _bytes_in(getattr(obj, name), seen, depth + 1)
    return found


def test_no_seed_exists_before_the_round_closes(compiled, make_run) -> None:
    runs = [make_run(session_id=f"seedless{i}".encode()) for i in range(2)]
    verifier, prover = EpochVerifier(parameters(HALF)), EpochProver()
    headers = [
        committed(verifier, prover, run, f"seedless/{i}") for i, run in enumerate(runs)
    ]
    round_seed = seed("seedless/round")
    seal = verifier.link
    would_be = {
        derive_run_seed(round_seed, seal, 0, index, label)
        for index in range(2)
        for label in (b"q", b"s")
    }
    assert len(would_be) == 4
    # nothing the verifier holds is a seed, and no method derives one
    assert not would_be & _bytes_in(verifier, set())
    for run in verifier._runs.values():
        assert not run.session.released
        with pytest.raises(ProtocolError, match="not been released"):
            run.session.challenge_replay()
    assert not hasattr(RoundChallenge, "round_seed") and "round_seed" not in {
        f.name for f in fields(RoundChallenge)
    }

    challenge = verifier.close_round(round_seed)
    assert challenge.seal == seal
    revealed = {c.seed for _, c in challenge.challenges}
    assert revealed == {derive_run_seed(round_seed, seal, 0, i, b"q") for i in range(2)}
    # the s seeds stay the verifier's until each run's interiors are in
    assert not {
        derive_run_seed(round_seed, seal, 0, i, b"s") for i in range(2)
    } & _bytes_in(challenge, set())
    for index, (header, replay_challenge) in enumerate(challenge.challenges):
        assert header == headers[index]
        sample = verifier.receive_interiors(
            header, prover.interiors(header, replay_challenge)
        )
        assert sample.seed == derive_run_seed(round_seed, seal, 0, index, b"s")
        assert verifier.receive_evidence(
            header, prover.evidence(header, sample)
        ).accepted
    assert verifier.report().accepted
    with pytest.raises(ProtocolError, match="released once"):
        next(iter(verifier._runs.values())).session.release(Q_SEED, S_SEED)


def test_a_session_admitted_from_a_claim_is_challenged_once_released(
    compiled, expect, honest_values, model_weights
) -> None:
    """The restructured session: boundary first, seeds later, the same challenge."""

    expectation = expect(HALF)
    claim = expectation.claim
    assert (
        isinstance(claim, Claim)
        and Claim(**{f.name: getattr(claim, f.name) for f in fields(Claim)}) == claim
    )
    assert {f.name for f in fields(Claim)} == {f.name for f in fields(expectation)} - {
        "q_seed",
        "s_seed",
    }
    with pytest.raises(ProtocolError, match="Expectation or a Claim"):
        VerifierSession("nope", compiled)  # type: ignore[arg-type]

    reference = VerifierSession(expectation, compiled)
    deferred = VerifierSession(claim, compiled)
    assert (
        deferred.header == reference.header
        and reference.released
        and not deferred.released
    )
    prover = ProverSession(
        compiled, deferred.header, honest_values, weight_tree=model_weights[1]
    )
    message = prover.boundary()
    assert deferred.boundary_phase == b""
    with pytest.raises(ProtocolError, match="not been released"):
        deferred.receive_boundary(message)  # accepted the boundary, could not challenge
    assert deferred.boundary_phase == boundary_phase(deferred.header, message)
    with pytest.raises(Reject) as twice:
        deferred.accept_boundary(message)
    assert twice.value.code is VerificationCode.INVALID_PHASE
    deferred.release(Q_SEED, S_SEED)
    with pytest.raises(ProtocolError, match="released once"):
        deferred.release(Q_SEED, S_SEED)
    with pytest.raises(ProtocolError, match="32 bytes"):
        VerifierSession(claim, compiled).release(b"short", S_SEED)

    challenge = deferred.challenge_replay()
    assert challenge == reference.receive_boundary(message)
    interiors = prover.interiors(challenge)
    sample = deferred.receive_interiors(interiors)
    assert sample == reference.receive_interiors(interiors)
    evidence = prover.evidence(sample)
    assert deferred.receive_evidence(evidence) == reference.receive_evidence(evidence)
    assert deferred.transcript == reference.transcript == prover.transcript
    assert (
        run_protocol(compiled, expectation, honest_values).transcript
        == deferred.transcript
    )


# -- rounds --------------------------------------------------------------------------------


def test_rounds_sum_their_bounds_at_eta_over_rounds(compiled, make_run) -> None:
    table = compiled.kind_table()
    for rounds, schedule in (
        (1, [[0, 1, 2]]),
        (2, [[0, 1], [2]]),
        (3, [[0], [1], [2]]),
    ):
        runs = [make_run(session_id=f"r{i}".encode()) for i in range(3)]
        seeds = [seed(f"rounds/{rounds}", r) for r in range(rounds)]
        report = run_epoch(parameters(HALF, rounds=rounds), runs, schedule, seeds)
        assert (
            report.accepted and report.run_count == 3 and len(report.rounds) == rounds
        )
        expected = [
            bound(union([table] * len(members)), HALF, ETA / rounds).bits
            for members in schedule
        ]
        assert [r.capacity_bits for r in report.rounds] == pytest.approx(expected)
        assert report.capacity_bits == pytest.approx(sum(expected))
        seals = [r.seal for r in report.rounds]
        assert len(set(seals)) == rounds and all(s is not None for s in seals)
        for r, members in zip(report.rounds, schedule, strict=True):
            assert len(r.runs) == len(members) and r.accepted
            assert all(run.header.eta == ETA / rounds for run in r.runs)
    # one round prices the three runs together at eta; three rounds price each at eta / 3
    runs = [make_run(session_id=f"o{i}".encode()) for i in range(3)]
    one = run_epoch(parameters(HALF), runs, [[0, 1, 2]], [seed("rounds/one")])
    three = run_epoch(
        parameters(HALF, rounds=3),
        runs,
        [[0], [1], [2]],
        [seed("rounds/three", r) for r in range(3)],
    )
    assert one.capacity_bits == pytest.approx(bound(union([table] * 3), HALF, ETA).bits)
    assert three.capacity_bits == pytest.approx(3 * bound(table, HALF, ETA / 3).bits)


def test_a_rejected_run_in_any_round_rejects_the_epoch(
    compiled, honest_values, make_run
) -> None:
    values, outputs = flipped(compiled, honest_values, relation_units(compiled)[0])
    bad = make_run(values, outputs, replay=assignment_replay(values), session_id=b"bad")
    runs = [make_run(session_id=b"good0"), make_run(session_id=b"good1"), bad]
    for schedule in ([[0, 1], [2]], [[2], [0, 1]], [[0, 2], [1]]):
        report = run_epoch(
            parameters(EVERYTHING, rounds=2),
            runs,
            schedule,
            [seed("reject", 0), seed("reject", 1)],
        )
        assert not report.accepted and report.code is VerificationCode.RELATION_REJECTED
        assert b"bad".hex() in report.detail
        assert report.run_count == 3
        for r, members in zip(report.rounds, schedule, strict=True):
            assert r.closed and r.accepted == (2 not in members)
        assert report.capacity_bits == pytest.approx(
            sum(
                bound(union([compiled.kind_table()] * len(m)), EVERYTHING, ETA / 2).bits
                for m in schedule
            )
        )
    # at (1, 0) nothing is sampled and the corrupted run survives: the epoch accepts
    lenient = run_epoch(
        parameters(NOTHING, rounds=2),
        runs,
        [[0, 1], [2]],
        [seed("lenient", 0), seed("lenient", 1)],
    )
    assert lenient.accepted


def test_the_fault_budget_is_the_rounds_to_share(
    compiled, honest_values, make_run
) -> None:
    units = relation_units(compiled)
    first, first_outputs = flipped(compiled, honest_values, units[0])
    second, second_outputs = flipped(compiled, honest_values, units[1])
    declaring = [
        make_run(
            first,
            first_outputs,
            replay=assignment_replay(first),
            declare=honest_declare(compiled),
            session_id=b"f1",
        ),
        make_run(
            second,
            second_outputs,
            replay=assignment_replay(second),
            declare=honest_declare(compiled),
            session_id=b"f2",
        ),
    ]
    table = compiled.kind_table()

    # one budget for the round: the second declaration is one too many
    report = run_epoch(
        parameters(EVERYTHING, max_faults=1), declaring, [[0, 1]], [seed("budget")]
    )
    assert not report.accepted and report.code is VerificationCode.FAULTS_EXCEEDED
    assert "exceed its budget of 1" in report.detail and b"f2".hex() in report.detail
    assert report.rounds[0].runs[0].accepted and report.rounds[0].declarations == 1
    assert all(run.header.max_faults == 1 for run in report.rounds[0].runs)
    # charged once, over the union
    assert (
        report.capacity_bits
        == bound(union([table, table]), EVERYTHING, ETA, max_faults=1).bits
    )
    assert report.capacity_bits > bound(union([table, table]), EVERYTHING, ETA).bits

    # a budget of two lets both through; two rounds give each run its own budget of one
    both = run_epoch(
        parameters(EVERYTHING, max_faults=2), declaring, [[0, 1]], [seed("budget/2")]
    )
    assert both.accepted and both.rounds[0].declarations == 2
    split = run_epoch(
        parameters(EVERYTHING, rounds=2, max_faults=1),
        declaring,
        [[0], [1]],
        [seed("budget/split", r) for r in range(2)],
    )
    assert split.accepted and [r.declarations for r in split.rounds] == [1, 1]

    # any one run may use the whole budget
    twice, twice_outputs = flipped(compiled, first, units[1])
    hog = make_run(
        twice,
        twice_outputs,
        replay=assignment_replay(twice),
        declare=honest_declare(compiled),
        session_id=b"hog",
    )
    greedy = run_epoch(
        parameters(EVERYTHING, max_faults=2),
        [hog, make_run(session_id=b"quiet")],
        [[0, 1]],
        [seed("budget/hog")],
    )
    assert greedy.accepted and greedy.rounds[0].declarations == 2
    # undeclared, the same faults are relation failures
    silent = run_epoch(
        parameters(EVERYTHING, max_faults=2),
        [replace(hog, declare=None)],
        [[0]],
        [seed("budget/silent")],
    )
    assert silent.code is VerificationCode.RELATION_REJECTED


def test_admission_enforces_u_max_against_the_running_union(compiled, make_run) -> None:
    table = compiled.kind_table()
    two = bound(union([table, table]), HALF, ETA).bits
    three = bound(union([table] * 3), HALF, ETA).bits
    assert (
        two <= math.ceil(two) < three
    )  # a U_max that admits two of these runs, not three
    cap = parameters(HALF, max_capacity=math.ceil(two))
    runs = [make_run(session_id=f"cap{i}".encode()) for i in range(3)]

    verifier, prover = EpochVerifier(cap), EpochProver()
    committed(verifier, prover, runs[0], "cap/0")
    committed(verifier, prover, runs[1], "cap/1")
    with pytest.raises(Reject) as refused:
        verifier.admit(
            runs[2].compilation, runs[2].claimed_outputs, weights=runs[2].weights
        )
    assert refused.value.code is VerificationCode.POLICY_REJECTED and "U_max" in str(
        refused.value
    )
    assert len(verifier.stream) == 4  # nothing of the refused run is in the stream
    answer(verifier, prover, verifier.close_round(seed("cap")))
    report = verifier.report()
    assert report.accepted and report.run_count == 2
    assert [r.code for r in report.rounds[0].refused] == [
        VerificationCode.POLICY_REJECTED
    ]
    assert report.capacity_bits == pytest.approx(two)

    driven = run_epoch(cap, runs, [[0, 1, 2]], [seed("cap/driven")])
    assert (
        driven.accepted and driven.run_count == 2 and len(driven.rounds[0].refused) == 1
    )
    # the same three runs are admitted without the cap, or with a cap that fits
    assert (
        run_epoch(parameters(HALF), runs, [[0, 1, 2]], [seed("cap/free")]).run_count
        == 3
    )
    roomy = parameters(HALF, max_capacity=math.ceil(three))
    assert run_epoch(roomy, runs, [[0, 1, 2]], [seed("cap/roomy")]).run_count == 3


def test_the_report_says_what_is_missing(compiled, make_run) -> None:
    run = make_run()
    verifier, prover = EpochVerifier(parameters(HALF, rounds=2)), EpochProver()
    assert verifier.round == 0
    open_report = verifier.report()
    assert not open_report.accepted and "round 0 of 2 is open" in open_report.detail
    assert open_report.rounds[0].seal is None and not open_report.rounds[0].closed
    header = committed(verifier, prover, run, "missing")
    challenge = verifier.close_round(seed("missing/0"))
    assert verifier.round == 1
    unanswered = verifier.report()
    assert (
        unanswered.code is VerificationCode.INVALID_PHASE
        and "was not answered" in unanswered.detail
    )
    answer(verifier, prover, challenge)
    assert "round 1 of 2 is open" in verifier.report().detail
    verifier.close_round(seed("missing/1"))
    assert verifier.round == 2 and verifier.report().accepted
    assert (
        verifier.report().rounds[1].runs == ()
        and verifier.report().rounds[1].capacity_bits == 0.0
    )
    with pytest.raises(ProtocolError, match="all closed"):
        verifier.close_round(seed("missing/2"))
    with pytest.raises(ProtocolError, match="all closed"):
        verifier.admit(run.compilation, run.claimed_outputs, weights=run.weights)
    assert verifier.transcript(header).header == header


def test_parameters_and_schedules_are_checked(make_run) -> None:
    with pytest.raises(ProtocolError, match="rounds"):
        EpochParameters(ETA, HALF, max_capacity=None, rounds=0)
    with pytest.raises(ProtocolError, match="VerificationPolicy"):
        EpochParameters(ETA, (1, 1), max_capacity=None)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="eta"):
        EpochParameters(1, HALF, max_capacity=None)
    with pytest.raises(ProtocolError, match="max_faults"):
        EpochParameters(ETA, HALF, max_capacity=None, max_faults=-1)
    with pytest.raises(TypeError):
        EpochParameters(ETA, HALF)  # type: ignore[call-arg]
    p = parameters(HALF, rounds=4, max_faults=2)
    assert p.round_eta == ETA / 4 and p.run_parameters == VerifierParameters(
        ETA / 4, max_capacity=None, max_faults=2
    )
    with pytest.raises(ProtocolError, match="one schedule entry and one seed"):
        run_epoch(p, [make_run()], [[0]], [seed("s")])
    with pytest.raises(ProtocolError, match="one schedule entry and one seed"):
        run_epoch(parameters(HALF), [make_run()], [[0]], [])
    with pytest.raises(ProtocolError, match="EpochParameters"):
        EpochVerifier(VerifierParameters(max_capacity=None))  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="32 bytes"):
        EpochVerifier(p).close_round(b"short")
    with pytest.raises(ProtocolError, match="32 bytes"):
        derive_run_seed(seed("x"), b"seal", 0, 0, b"q")


# -- the adversary spread over a round ----------------------------------------------------


def test_an_adversary_spread_over_the_runs_of_a_round_survives_as_predicted(
    compiled, honest_values, make_run
) -> None:
    """Three runs, one corrupted VU each, one round: the union's survival, the union's Bound.

    Sampling is Bernoulli per unit under per-run seeds derived from one seal, so
    an erroneous unit survives with the single-run probability ``1 - q s`` and
    the round survives with the product over its runs -- ``sigma(E)`` over the
    union's error multiset -- within :data:`TOLERANCE_SIGMAS` of the observed
    rate, as the datacenter simulation checks.
    """

    policy = VerificationPolicy(Fraction(1, 2), Fraction(3, 4))
    units = relation_units(compiled)[:3]
    assert len(units) == 3
    runs = []
    for index, unit in enumerate(units):
        values, outputs = flipped(compiled, honest_values, unit)
        runs.append(
            make_run(
                values,
                outputs,
                replay=assignment_replay(values),
                session_id=f"adv{index}".encode(),
            )
        )
    errors_per_replay_unit = (1,) * len(
        units
    )  # one erroneous VU in each of three distinct RUs
    predicted = float(survival(policy, errors_per_replay_unit))
    assert predicted == pytest.approx(float((1 - policy.q * policy.s) ** 3))

    trials = 400
    accepted = 0
    survived = [0] * len(units)
    capacities = set()
    for trial in range(trials):
        report = run_epoch(
            parameters(policy), runs, [[0, 1, 2]], [seed("adversary", trial)]
        )
        capacities.add(report.capacity_bits)
        accepted += report.accepted
        if not report.accepted:
            assert report.code is VerificationCode.RELATION_REJECTED
        # every run is answered; a rejection is the relation check of a sampled corrupted VU
        for index, (run, unit) in enumerate(
            zip(report.rounds[0].runs, units, strict=True)
        ):
            assert run.report is not None
            if run.report.accepted:
                survived[index] += 1
                assert unit not in run.report.sampled_verification_units
            else:
                assert run.report.code is VerificationCode.RELATION_REJECTED
                assert unit in run.report.sampled_verification_units
    observed = accepted / trials
    sigma = math.sqrt(predicted * (1 - predicted) / trials)
    assert abs(observed - predicted) <= TOLERANCE_SIGMAS * sigma, (
        observed,
        predicted,
        sigma,
    )
    # each erroneous unit survives at the single-run rate 1 - q s
    per_unit = float(1 - policy.q * policy.s)
    unit_sigma = math.sqrt(per_unit * (1 - per_unit) / trials)
    for count in survived:
        assert abs(count / trials - per_unit) <= TOLERANCE_SIGMAS * unit_sigma, (
            count,
            per_unit,
        )
    table = compiled.kind_table()
    assert capacities == {bound(union([table] * 3), policy, ETA).bits}
