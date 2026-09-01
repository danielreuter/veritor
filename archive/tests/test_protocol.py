"""Protocol: honest runs accept; every cheating strategy we model is caught
by the right check (or slips through exactly when sampling misses it)."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from veritor.machine import Instruction, Operand, Program
from veritor.protocol import Prover, Verifier, run_protocol
from veritor.tracer import trace


def f(x):
    y = x * (x + 1.0)
    return jnp.exp(y) + 2.0


X = np.float32(0.7)


@pytest.fixture()
def program():
    return trace(f, X)


def test_honest_run_accepts_every_challenge(program):
    prover = Prover(program, [X])
    verifier = Verifier(program, [X])
    n = len(program.instructions)
    transcript = run_protocol(prover, verifier, num_samples=0, challenges=list(range(n)))
    assert transcript.accepted


def test_forged_cell_caught_iff_sampled(program):
    m = program.num_inputs
    k = 1  # forge the write of instruction 1 (the x+1.0 add, a computed cell)
    cheater = Prover(program, [X], overrides={m + k: np.float32(3.25)})
    verifier = Verifier(program, [X])

    # Challenge exactly the forged instruction: caught.
    t = run_protocol(cheater, verifier, num_samples=0, challenges=[k])
    assert not t.accepted
    assert any(not c.ok and c.name == f"instruction {k}" for c in t.checks)

    # Challenge every OTHER instruction: the forgery survives sampling
    # (downstream cells were honestly recomputed from the forged cell), but
    # the boundary check still pins the claimed output to the committed tape.
    others = [i for i in range(len(program.instructions)) if i != k]
    t = run_protocol(cheater, verifier, num_samples=0, challenges=others)
    assert t.accepted  # this is the residual leakage the design quantifies


def test_forged_const_cell_caught_when_sampled(program):
    # Instruction 0 is `const 1.0`, writing cell 1. Forging that cell makes
    # the const locally inconsistent: its check re-applies the identity to
    # the immediate and compares against the committed write bit-for-bit.
    assert program.instructions[0].prim == "const"
    m = program.num_inputs
    cheater = Prover(program, [X], overrides={m + 0: np.float32(99.0)})
    verifier = Verifier(program, [X])
    t = run_protocol(cheater, verifier, num_samples=0, challenges=[0])
    assert not t.accepted
    assert any(not c.ok and c.name == "instruction 0" for c in t.checks)


def test_smuggled_literal_operand_rejected():
    # A hand-built tape where a non-const instruction carries a literal
    # operand. Both parties hold the same doctored program, so provenance
    # passes -- but the local check rejects the instruction while decoding
    # it, regardless of the value it wrote.
    bad = Program(
        num_inputs=1,
        instructions=(Instruction("add", (Operand.cell(0), Operand.imm(1.0))),),
        output_indices=(1,),
    )
    prover = Prover(bad, [X])
    verifier = Verifier(bad, [X])
    t = run_protocol(prover, verifier, num_samples=0, challenges=[0])
    assert not t.accepted
    assert any("literal" in c.detail for c in t.failures())


def test_forging_the_output_cell_fails_boundary_check(program):
    # Forge the LAST cell (the output view). The claimed output then matches
    # the committed tape, but the final instruction is locally inconsistent.
    m = program.num_inputs
    last = len(program.instructions) - 1
    cheater = Prover(program, [X], overrides={m + last: np.float32(99.0)})
    verifier = Verifier(program, [X])
    assert cheater.commit().claimed_outputs == (99.0,)
    t = run_protocol(cheater, verifier, num_samples=0, challenges=[last])
    assert not t.accepted


def test_lying_about_output_without_forging_tape(program):
    # Commit honestly but CLAIM a different output. Boundary check catches
    # this deterministically -- no sampling required. run_protocol would
    # re-commit honestly, so drive the verifier directly: deliver the
    # doctored commitment as message 1, then run the exact checks.
    prover = Prover(program, [X])
    honest = prover.commit()
    lied = dataclasses.replace(honest, claimed_outputs=(honest.claimed_outputs[0] + 1.0,))
    verifier = Verifier(program, [X])
    verifier.receive_commitment(lied)

    assert verifier.check_provenance().ok  # instruction tape is still honest
    boundary_indices = list(range(lied.num_inputs)) + list(program.output_indices)
    openings = {i: prover.open_value(i) for i in boundary_indices}
    results = verifier.check_boundaries(openings)
    assert any("output cell" in c.name and not c.ok for c in results)


def test_wrong_program_fails_provenance(program):
    # The prover runs g != f. The verifier re-traces f and the committed
    # instruction root does not match.
    def g(x):
        y = x * (x + 1.0)
        return jnp.exp(y) + 3.0  # one literal differs

    cheater = Prover(trace(g, X), [X])
    verifier = Verifier(program, [X])
    t = run_protocol(cheater, verifier, num_samples=0, challenges=[])
    assert not t.accepted
    assert any(c.name == "instruction provenance" and not c.ok for c in t.checks)


def test_wrong_input_fails_boundary(program):
    prover = Prover(program, [X])
    verifier = Verifier(program, [np.float32(0.9)])
    t = run_protocol(prover, verifier, num_samples=0, challenges=[])
    assert not t.accepted
    assert any("input cell" in c.name and not c.ok for c in t.checks)


def test_detection_rate_matches_analytic_bound():
    # 514-instruction chain (2 consts + 512 ops), L=4 forged cells, s=32
    # samples: empirical catch rate should sit near 1-(1-L/N)^s.
    import random

    def chain(x):
        for _ in range(256):
            x = 0.5 * x + 0.1
        return x

    x0 = np.float32(3.0)
    program = trace(chain, x0)
    n = len(program.instructions)
    rng = random.Random(7)
    forged = rng.sample(range(program.num_inputs, program.num_inputs + n), 4)
    cheater = Prover(program, [x0], overrides={i: np.float32(0.5) for i in forged})
    verifier = Verifier(program, [x0])

    trials, s = 300, 32
    caught = sum(
        0 if run_protocol(cheater, verifier, num_samples=s, rng=rng).accepted else 1
        for _ in range(trials)
    )
    analytic = 1.0 - (1.0 - 4 / n) ** s
    assert abs(caught / trials - analytic) < 0.1
