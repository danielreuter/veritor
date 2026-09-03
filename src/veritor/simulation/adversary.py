"""A dishonest server that exfiltrates a secret through the token decisions.

The only values the users receive are tokens, and every token is the
output of one head verification unit (VU) -- ``argmax`` or ``sample`` --
whose result is a single ``width``-bit word.  A server that wants to leak a
secret (a stand-in for its weights) has no better channel than to *decide*
some tokens instead of computing them: it overrides the head VU's output
with the next ``vocab_bits`` bits of the secret and computes everything
downstream honestly from the corrupted token.  Exactly one VU per carrier
token then violates its relation; every other gate of the circuit is
consistent with what it reads, so the transcript is rejected iff the
verifier samples a corrupted VU.

That is the tightness claim of the paper made concrete.  ``Bound`` charges
a head VU ``kappa = min(out_bits, reach_bits)`` bits -- the width of the
word it decides -- and an adversary who corrupts ``k`` of them realizes
``k * vocab_bits`` bits of that budget while escaping with probability
``sigma(E) = prod_r (1 - q + q (1 - s)^(l_r))`` over the replay units (RUs)
``r`` holding ``l_r`` corrupted VUs.  A carrier whose honest token already
equals its chunk costs nothing (its gate is not incorrect), so the error
set ``E`` is the carriers that had to change.  :func:`survival_trials` measures that
escape rate by deriving the verifier's challenges under fresh seeds, and
:func:`protocol_trials` runs the full protocol to confirm that every
rejection is the relation check of a sampled corrupted VU.

The carriers are the last tokens of the responses, round-robin over the
requests from the tail: the users receive the honest responses except for
their final few tokens, which spell the secret.  A carrier only influences
later positions of its own request, and those are carriers too, so no
honest token changes.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from veritor.analysis.probability import survival
from veritor.compile import Compilation
from veritor.core import Compiled, VerificationLimits, VerificationPolicy
from veritor.protocol import (
    MerkleTree,
    VerificationCode,
    VerificationReport,
    VerifierParameters,
    Weights,
    assignment_replay,
    derive_replay_selection,
    derive_sample_selection,
    make_expectation,
    run_protocol,
)


@dataclass(frozen=True, slots=True)
class Attack:
    """One exfiltration: the secret, where it went and the transcript that carries it."""

    secret: str
    """The secret as a bit string, ``vocab_bits`` bits per carrier."""
    carriers: tuple[int, ...]
    """Indices into the circuit outputs (the streamed tokens) that spell the secret."""
    addresses: tuple[int, ...]
    """The forced gates: the token gate of each carrier's head VU."""
    corrupted: tuple[int, ...]
    """The forced gates whose relation fails: a carrier whose honest token already spelled its
    chunk needs no corruption (one in ``vocab`` of them, on average)."""
    verification_units: tuple[int, ...]
    """The corrupted VUs (global indices), one per corrupted gate."""
    replay_units: tuple[int, ...]
    """The RUs touched, one entry per corrupted gate (with repeats)."""
    values: Mapping[int, int]
    """The dishonest assignment: the carriers forced, everything else computed from what it reads."""
    outputs: tuple[int, ...]
    """The claimed ``y*``: the tokens the users received."""

    @property
    def bits(self) -> int:
        return len(self.secret)

    @property
    def errors_per_replay_unit(self) -> tuple[int, ...]:
        return tuple(sorted(Counter(self.replay_units).values(), reverse=True))


def carriers(layout: Sequence[tuple[int, int]], count: int) -> tuple[int, ...]:
    """The output indices that carry ``count`` secret chunks: last tokens first, round-robin over requests."""

    if not 0 <= count <= len(layout):
        raise ValueError(f"the run streamed {len(layout)} tokens, cannot carry {count}")
    by_request: dict[int, list[int]] = {}
    for index, (request, _position) in enumerate(layout):
        by_request.setdefault(request, []).append(index)
    chosen: list[int] = []
    depth = 1
    while len(chosen) < count:
        for request in sorted(by_request):
            positions = by_request[request]
            if len(positions) >= depth and len(chosen) < count:
                chosen.append(positions[-depth])
        depth += 1
    return tuple(sorted(chosen))


def random_secret(bits: int, seed: object) -> str:
    rng = random.Random(f"veritor/simulation/secret/{seed}")
    return "".join(rng.choice("01") for _ in range(bits))


def evaluate_with_overrides(
    compiled: Compiled,
    inputs: Iterable[int],
    weights: Iterable[int],
    overrides: Mapping[int, int],
) -> dict[int, int]:
    """Every value of the circuit with ``overrides`` forced and propagated downstream.

    The gates at the overridden addresses are the only ones whose relation
    fails; everything downstream is computed from what it reads.
    """

    circuit = compiled.circuit
    given = {"input": iter(tuple(inputs)), "weight": iter(tuple(weights))}
    values: dict[int, int] = {}
    for address in range(circuit.n):
        ref = circuit[address]
        if ref.is_source:
            value = next(given[ref.source])  # type: ignore[index]
        else:
            value = circuit.evaluate_gate(address, tuple(values[a] for a in ref.args))
        values[address] = overrides.get(address, value)
    return values


def plan_attack(
    compiled: Compiled,
    inputs: Sequence[int],
    weights: Sequence[int],
    layout: Sequence[tuple[int, int]],
    secret: str,
    vocab_bits: int,
) -> Attack:
    """Encode ``secret`` (``vocab_bits`` bits per token) into the last tokens of the responses."""

    if len(secret) % vocab_bits or set(secret) - {"0", "1"}:
        raise ValueError(f"the secret must be a bit string of a multiple of {vocab_bits} bits")
    chosen = carriers(layout, len(secret) // vocab_bits)
    circuit, index = compiled.circuit, compiled.index
    addresses = tuple(circuit.outputs[c] for c in chosen)
    overrides = {
        address: int(secret[rank * vocab_bits : (rank + 1) * vocab_bits], 2)
        for rank, address in enumerate(addresses)
    }
    values = evaluate_with_overrides(compiled, inputs, weights, overrides)
    corrupted: list[int] = []
    units: list[int] = []
    replay_units: list[int] = []
    for address in addresses:
        honest = circuit.evaluate_gate(address, tuple(values[a] for a in circuit[address].args))
        if values[address] == honest:
            continue
        corrupted.append(address)
        replay_unit = index.replay_units.owner(address)
        block = index.verification_units(replay_unit)
        units.append(block.first + block.owner(address))
        replay_units.append(replay_unit)
    return Attack(
        secret=secret,
        carriers=chosen,
        addresses=addresses,
        corrupted=tuple(corrupted),
        verification_units=tuple(units),
        replay_units=tuple(replay_units),
        values=values,
        outputs=tuple(values[address] for address in circuit.outputs),
    )


def decode_secret(outputs: Sequence[int], chosen: Sequence[int], vocab_bits: int) -> str:
    """What the accomplice reads off the streamed tokens: the carriers, ``vocab_bits`` bits each."""

    return "".join(format(outputs[c], f"0{vocab_bits}b") for c in chosen)


def predicted_survival(policy: VerificationPolicy, attack: Attack) -> Fraction:
    """``sigma(E)``: the exact chance the verifier samples none of the corrupted VUs."""

    return survival(policy, attack.errors_per_replay_unit)


def _seed(label: str, trial: int) -> bytes:
    return hashlib.sha256(f"veritor/simulation/{label}/{trial}".encode()).digest()


def survival_trials(
    compiled: Compiled,
    policy: VerificationPolicy,
    attack: Attack,
    trials: int,
    *,
    label: str = "",
    limits: VerificationLimits | None = None,
) -> int:
    """How many of ``trials`` fresh challenge pairs miss every corrupted VU.

    This is the verifier's own challenge derivation (the same functions
    :class:`~veritor.protocol.VerifierSession` calls) under fresh ``q`` and
    ``s`` seeds; the boundary phase is fixed because the prover's messages
    do not depend on the seeds.  A transcript with this error set is accepted
    exactly when the sample misses it, which :func:`protocol_trials` confirms.
    """

    limits = VerificationLimits() if limits is None else limits
    errors = set(attack.verification_units)
    phase = hashlib.sha256(b"veritor/simulation/phase").digest()
    escaped = 0
    for trial in range(trials):
        selected = derive_replay_selection(
            _seed(f"{label}/q", trial), phase, compiled, policy, limits
        )
        sampled = derive_sample_selection(
            _seed(f"{label}/s", trial), phase, compiled, selected, policy, limits
        )
        if not errors & set(sampled):
            escaped += 1
    return escaped


def protocol_trials(
    compilation: Compilation,
    policy: VerificationPolicy,
    parameters: VerifierParameters,
    weights: Weights,
    weight_tree: MerkleTree,
    attack: Attack,
    trials: int,
    *,
    label: str = "",
    limits: VerificationLimits | None = None,
) -> tuple[VerificationReport, ...]:
    """Run the full protocol against the dishonest server ``trials`` times with fresh seeds.

    The prover commits the dishonest assignment (its interiors are replayed
    from it, not recomputed) and claims the corrupted tokens; every
    rejection must be a relation check of a sampled corrupted VU.
    """

    reports: list[VerificationReport] = []
    errors = set(attack.verification_units)
    for trial in range(trials):
        expectation = make_expectation(
            compilation,
            policy,
            attack.outputs,
            parameters=parameters,
            weights=weights,
            session_id=_seed(f"{label}/session", trial)[:16],
            q_seed=_seed(f"{label}/protocol/q", trial),
            s_seed=_seed(f"{label}/protocol/s", trial),
        )
        run = run_protocol(
            compilation.compiled,
            expectation,
            attack.values,
            replay=assignment_replay(attack.values),
            limits=limits,
            weight_tree=weight_tree,
        )
        report = run.report
        if not report.accepted and (
            report.code is not VerificationCode.RELATION_REJECTED
            or not errors & set(report.sampled_verification_units)
        ):
            raise AssertionError(f"unexpected verdict against the dishonest server: {report}")
        reports.append(report)
    return tuple(reports)


__all__ = [
    "Attack",
    "carriers",
    "decode_secret",
    "evaluate_with_overrides",
    "plan_attack",
    "predicted_survival",
    "protocol_trials",
    "random_secret",
    "survival_trials",
]
