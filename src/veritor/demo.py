"""End-to-end run, honest and cheating: python -m veritor.demo"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from veritor.protocol import Prover, Verifier, run_protocol
from veritor.tracer import trace


def f(x: Any) -> Any:
    return jnp.exp(x * (x + 1.0)) + 2.0


def main() -> None:
    x = np.float32(0.7)
    program = trace(f, x)
    print(program.describe())

    prover = Prover(program, [x])
    verifier = Verifier(trace(f, x), [x])
    honest = run_protocol(prover, verifier, num_samples=3)
    print(f"honest:   accepted={honest.accepted}, challenges={honest.challenges}")

    # Forge instruction 1's write (cell 2, the x+1.0 add -- a computed
    # mid-tape cell, not a const); downstream cells recomputed honestly.
    cheater = Prover(program, [x], overrides={2: np.float32(3.25)})
    cheat_verifier = Verifier(trace(f, x), [x])
    cheat = run_protocol(cheater, cheat_verifier, num_samples=3)
    print(f"cheating: accepted={cheat.accepted}, challenges={cheat.challenges}")
    for c in cheat.failures():
        print(f"  {c.name}: {c.detail}")


if __name__ == "__main__":
    main()
