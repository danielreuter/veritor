import random
import time
from dataclasses import dataclass


def interpret(program: bytes, inputs: list, novel: bytes, deadline: float) -> bytes:
    """The one universal interpreter. `deadline` plays the role of C."""
    t0 = time.monotonic()
    env = {}
    exec(program, env)
    output = env["f"](inputs, novel)
    if time.monotonic() - t0 > deadline:
        raise TimeoutError("recomputation blew the compute bound C")
    return output


EXTERNAL, EMITTED = "external", "emitted"


@dataclass(frozen=True)
class Event:
    """What a tap attests when an artifact crosses a boundary."""

    kind: str  # EXTERNAL: inbound at the perimeter | EMITTED: outbound from a pod
    hash: str  # content hash of the artifact that crossed


@dataclass(frozen=True)
class Derivation:
    """A claim: this emitted artifact = program(earlier artifacts, novel bits)."""

    program: str  # hash of a baseline artifact
    inputs: tuple  # record indices, all strictly earlier
    novel: bytes  # literal asserted bits -- the only unvouched ingredient


# -------------------------------------------------------------- the protocol


def generate_claims(baseline, generator, record, artifacts, sigma):
    """Run the committed generator online over the record. At each EMITTED
    event it must produce the Derivation *before* the artifact is revealed,
    spending sigma wherever its committed expectations don't pin the answer.
    (g re-reads sigma deterministically each call; sigma is its only
    post-hoc input, so its length is the whole meter.)"""
    env = {"Derivation": Derivation}
    exec(baseline[generator], env)
    g = env["g"]
    claims, prefix = {}, []
    for i, event in enumerate(record):
        if event.kind == EMITTED:
            claim = g(baseline, list(prefix), sigma)  # emit before reading
            assert claim.program in baseline, f"event {i}: cites uncommitted program"
            assert all(j < i for j in claim.inputs), (
                f"event {i}: cites a non-earlier event"
            )
            claims[i] = claim
        artifact = artifacts[event.hash]  # developer opens its commitment
        assert hash(artifact) == event.hash, f"event {i}: opening doesn't match the tap"
        prefix.append((event.kind, artifact))
    return claims


def challenge(claims, k, beacon):
    """Drawn only after claims are sealed. (Real version: size-weighted over
    emitted bits; toy: uniform over emitted events.)"""
    return random.Random(beacon).sample(sorted(claims), min(k, len(claims)))


def verify(baseline, record, claims, artifacts, sample, C):
    """Recompute each sampled derivation; demand bit-exactness within C.
    (Openings were already hash-checked during the walk.)"""
    for i in sample:
        claim = claims[i]
        inputs = [artifacts[record[j].hash] for j in claim.inputs]
        output = interpret(baseline[claim.program], inputs, claim.novel, deadline=C)
        assert hash(output) == record[i].hash, (
            f"event {i}: replay mismatch -- artifact unexplained"
        )


def audit(baseline, generator, record, artifacts, sigma, S, C, k, beacon):
    """The whole protocol: budget, claims, challenge, verification, one bit."""
    assert len(sigma) <= S, (
        f"novel information |sigma|={len(sigma)} exceeds budget S={S}"
    )
    claims = generate_claims(baseline, generator, record, artifacts, sigma)
    sample = challenge(claims, k, beacon)
    verify(baseline, record, claims, artifacts, sample, C)
    return {
        "verdict": "PASS",
        "novel_bytes": len(sigma),
        "claims": len(claims),
        "replayed": len(sample),
    }


# --------------------------------------------------------------------- demo

if __name__ == "__main__":
    # Two committed programs and a committed generator.
    SERVE = b"def f(inputs, novel): return inputs[0].upper()"  # 'inference'
    IDENTITY = b"def f(inputs, novel): return novel"  # pay-per-bit escape hatch
    GENERATOR = (
        """
def g(baseline, prefix, sigma):
    # Committed expectation: every emitted artifact is SERVE applied to the
    # latest external artifact. Costs zero novel bits while true; anything
    # else must ride through sigma, at face value, via IDENTITY.
    if sigma:
        return Derivation(program=%r, inputs=(), novel=sigma)
    last = max(i for i, (kind, a) in enumerate(prefix) if kind == "external")
    return Derivation(program=%r, inputs=(last,), novel=b"")
"""
        % (hash(IDENTITY), hash(SERVE))
    ).encode()

    baseline = {hash(p): p for p in (SERVE, IDENTITY, GENERATOR)}
    prompt = b"hello pod"

    def run(name, record, artifacts, sigma, S=4):
        try:
            print(
                name,
                "->",
                audit(
                    baseline,
                    hash(GENERATOR),
                    record,
                    artifacts,
                    sigma,
                    S=S,
                    C=1.0,
                    k=10,
                    beacon=2026,
                ),
            )
        except AssertionError as e:
            print(name, "-> FAIL:", e)

    # 1) Honest inference: emitted = SERVE(external). Zero novel bits.
    reply = prompt.upper()
    run(
        "honest   ",
        [Event(EXTERNAL, hash(prompt)), Event(EMITTED, hash(reply))],
        {hash(prompt): prompt, hash(reply): reply},
        sigma=b"",
    )

    # 2) Cheat, horn one: emit a deep artifact, claim it's inference.
    #    Replay recomputes SERVE(prompt) != secret -> caught at verification.
    secret = b"weights: a month of GPUs"
    run(
        "cheat / C",
        [Event(EXTERNAL, hash(prompt)), Event(EMITTED, hash(secret))],
        {hash(prompt): prompt, hash(secret): secret},
        sigma=b"",
    )

    # 3) Cheat, horn two: confess the bits through sigma instead.
    #    Replay would now succeed bit-exactly -- but |sigma| blows S.
    #    (With S large enough this PASSES: novelty is allowed, just metered.)
    run(
        "cheat / S",
        [Event(EXTERNAL, hash(prompt)), Event(EMITTED, hash(secret))],
        {hash(prompt): prompt, hash(secret): secret},
        sigma=secret,
    )
