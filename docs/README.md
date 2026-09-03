# Documents

What each file is, what it is generated from, and where to start.

## Start here

- `security-argument.md` — component by component: what the protocol claims, why the code provides it, which test in `tests/veritor/security/` shows the failure mode caught, and what is only achieved by convention.
- `epoch.md` — the epoch layer (`veritor.protocol.epoch`): a year of runs as one circuit; commitment stream, sealed rounds, verifier-private round seed, union-of-runs `Bound`.
- `global-estimate.md` — the headline: covert capacity of a year of the world's inference compute under the protocol at a 1 % prover budget. Regenerate with `python -m veritor.evaluation.global_estimate`.
- `overnight-report-2026-09-02.md` — the latest report to the architect: what landed, decisions taken, measurements, flags for the paper, what is outstanding. `overnight-report.md` is the previous night's.

## Measurements

- `benchmarks.md` — how each component scales, from `data/benchmarks.json`. Measure with `python -m benchmarks.run`, render with `python -m benchmarks.report`.
- `cost-model.md` — the paper's §6.5: what the protocol commits, replays, proves and retains, priced at GPT-2 Small and the 70B serving shape (`alpha`, `beta`, `gamma`, the corrected leaf count at VU-output granularity); hash costs measured into `data/hash-cost.json` by `uv run --with blake3 python -m benchmarks.hash_cost`; a Provenance table maps every number to its source.
- `stress-tests.md` — the catalogue of datacenter realities (mechanisms M1–M8, scenarios S/C/N/W/E) and the table of recorded rows. Rows are recorded by the tests in `tests/veritor/stress/` into `data/stress*.json`; render §4 with `python -m veritor.stress.report` (`--check` verifies the rendering is current).
- `honest-prover.md` — the honest prover under faults: fault classes, secure late advice, what the declarations cost and how much the prover must re-execute to know what to declare; `H` rows from `tests/veritor/stress/` into `data/stress-honest.json`, rendered by the same report. The theory is `notes/late-advice.md`.
- `frontier-report.md` — the honest-server frontier for the 70B serving shape, from `data/frontier-70b.json` (produced by `python -m veritor.evaluation.sweep docs/data/frontier-70b.json --workers 10`).
- `zk-backend.md` — the proof layer (`veritor.protocol.proofs`): obligations, statements, batches, the transparent and SP1 backends, measured cycle costs and α. `sp1-benchmark-plan.md` is the plan that preceded it.
- `hardware-semantics.md` — tensor-core `mma.sync` semantics recovered bit-exactly on an RTX 4090 (`veritor.core.silicon`); kernels and results under `gpu/tensor-core-semantics/`.
- `gpt2-structure.md`, `gpt2-silicon.md` — GPT-2 Small as a description with marks, then run through fixed-order GPU kernels so the run is a circuit of pinned gates; kernels and captures under `gpu/gpt2/`.

## Notes and drafts

- `notes/datacenter-realities.md` — literature brief: twelve topics on what production inference datacenters actually do, with checked sources.
- `notes/declaration-kinds.md` — design note, nothing implemented: source-position, port and RU-scope pardons as declaration kinds beyond the VU; syntax, what the verifier checks instead, scope, the price each must carry in `Bound` (the ladder of `notes/late-advice.md`), the wire and verifier changes each would need, and the open pricing questions. Motivated by the systematic-fault rows `H4a`-`H4d` of `honest-prover.md`.
- `notes/late-advice.md` — which prover statements are admissible at which stage (before `J`, after `J`, after the VU sample) and at what price: the stage ladder for VU, RU-scope, source-cell and port pardons and for late lowering, the epoch accounting, the conservation law; numbers pinned by `tests/veritor/analysis/test_late_advice.py`.
- `paper/` — drafts of paper sections (§5 covert capacity, §7 secure circuit compilation) and their context notes.
- `compute-accounting/` — a separate track: compute-accounted bounded accumulation, combining the sampled verification with proofs of useful work.
- `lean-collaborator-handoff.md` — brief for a Lean collaborator on private optimized macros over public instructions.

## Data

`data/` holds the recorded measurements the documents above are rendered from: `benchmarks.json`, `frontier-70b.json`, `global-estimate.json`, `hash-cost.json`, `stress.json`, `stress-control-flow.json`, `stress-protocol.json`, `stress-honest.json`. They are inputs to the renderers, not hand-edited.
