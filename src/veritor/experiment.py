"""Detection rate vs the analytic bound 1-(1-L/N)^s: python -m veritor.experiment"""

import random
from typing import Any

import numpy as np

from veritor.protocol import Prover, Verifier, run_protocol
from veritor.tracer import trace


def chain(x: Any) -> Any:
    """256 iterations of x <- 0.5*x + 0.1, tracing to N = 514 instructions
    (two consts for the deduped literals 0.5 and 0.1, then 512 ops)."""
    for _ in range(256):
        x = 0.5 * x + 0.1
    return x


def main() -> None:
    x = np.float32(3.0)
    rng = random.Random(1234)
    program = trace(chain, x)
    m, n = program.num_inputs, len(program.instructions)
    trials = 400

    print(f"N={n}; catch rate over {trials} trials vs analytic 1-(1-L/N)^s")
    for num_forged in (1, 4, 16):
        forged = rng.sample(range(m, m + n), num_forged)
        overrides = {i: np.float32(rng.uniform(-1.0, 1.0)) for i in forged}
        cheater = Prover(program, [x], overrides=overrides)
        # One verifier reused across trials: run_protocol re-runs the full
        # interaction each time, and the prover's commit is deterministic.
        verifier = Verifier(program, [x])
        for s in (1, 4, 16, 64, 128):
            caught = sum(
                not run_protocol(cheater, verifier, num_samples=s, rng=rng).accepted
                for _ in range(trials)
            )
            analytic = 1.0 - (1.0 - num_forged / n) ** s
            print(
                f"  L={num_forged:>2} s={s:>3}  "
                f"empirical={caught / trials:.3f}  analytic={analytic:.3f}"
            )


if __name__ == "__main__":
    main()
