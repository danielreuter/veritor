# The epoch layer: a year of runs as one circuit

`veritor.protocol.epoch`, tested in `tests/veritor/protocol/test_epoch.py` and
`tests/veritor/analysis/test_union.py`. The headline estimate
(`docs/global-estimate.md`) prices a year of serving as *one* circuit under one
`eta`; the protocol as implemented ran one session per request. This layer
makes the headline literal: runs are committed as they happen, challenged only
when a round closes, and priced by one `Bound` over their union.

Terminology. Here an **epoch** is the unit of the guarantee (think: one year),
a **round** is the interval between two challenge times, and a **run** is what
`docs/stress-tests.md` and the paper call an epoch: one `Compile -> commit ->
challenge -> replay -> challenge -> evidence -> verdict`. The paper's word is
overloaded by this layer; see the flags.

## 1. Parameters

`EpochParameters(eta, policy, *, max_capacity, rounds=1, max_advice_bits=0,
max_work=W_max, max_faults=0)`:

- `eta` -- the epoch's acceptance threshold; the capacity the epoch certifies
  holds at `eta` over the whole epoch.
- `rounds` -- challenge rounds per epoch. Each round is bounded at `eta /
  rounds` and the epoch's capacity is the sum over rounds (a union bound over
  the rounds' bad events). `rounds = 1` is the single end-of-epoch challenge
  the headline estimate assumes; `rounds = N` for `N` runs recovers per-run
  challenges.
- `policy` -- `theta = (q, s)`, fixed by the verifier for every run of the
  epoch. The client proposes nothing.
- `max_advice_bits`, `max_work` -- the per-run caps `A` and `W_max` of
  `VerifierParameters`, checked at each admission exactly as today.
- `max_capacity` -- `U_max`, a cap on the **round**: a run is admitted only if
  `Bound` over the union of the round's tables *with this run's* stays at or
  below `U_max` (the running union; `None` waives the check and must be written
  out, as for `VerifierParameters`).
- `max_faults` -- the round's fault budget `f_max`. Every header of the round
  carries all of it (any one run may use it); the round rejects when its runs'
  declarations together exceed it; the union bound is charged with it once.

Each run's header binds `eta / rounds`, `theta`, the budget, and everything a
session header binds today -- and no seed. `EpochParameters.run_parameters` is
the `VerifierParameters` the per-run session sees: the round's `eta`, the
per-run caps, `max_capacity=None` (the cap is the round's).

## 2. Message flow

~~~
run time (round r open)
  client   -> verifier   Compile(G, x, a), claimed y*, kappa_W
  verifier -> prover     Header_i           = admit(...)        [stream += Header_i]
  prover   -> verifier   BoundaryMessage_i  = boundary()
                         receive_boundary   checks as VerifierSession.accept_boundary,
                                            no selection derived
                                            link <- H(link || Header_i.digest || boundary_phase_i)
                                                                [stream += Boundary_i]
round close
  verifier               seal = link;  round_seed = 32 fresh bytes (verifier-private)
                         q_i = HMAC(round_seed, domain || seal || r || i || "q")
                         s_i = HMAC(round_seed, domain || seal || r || i || "s")
  verifier -> prover     RoundChallenge(r, seal, [(Header_i, ReplayChallenge_i)])
  per run i, as today:
  prover   -> verifier   InteriorMessage_i  (replay of the opened RUs, declarations)
  verifier -> prover     SampleChallenge_i  (s_i, released only now)
  prover   -> verifier   EvidenceMessage_i
  verifier               VerificationReport_i
  round r + 1 opens; its chain continues from the seal
epoch end
  verifier               EpochReport: per-round seal, runs, capacity, declarations; the verdict
~~~

The stream's first link is a digest of the epoch's parameters, so a seal
commits to `eta`, `theta`, `rounds` and the caps as well as to every header and
boundary before it. `derive_run_seed` is the one place the seed derivation
lives (`SEED_DOMAIN = b"veritor/protocol/epoch/seed/v1"`); `stream_link` is
the one place the chain lives.

The **verdict**: an epoch is accepted only if every round is closed and every
committed run was challenged and accepted. A run admitted whose boundary never
arrived, whose interiors or evidence never came, or that was rejected anywhere
rejects the epoch, and the report names the run, the round and the reason. A
run *refused at admission* (over `U_max`, over `A`, over `W_max`) was never
committed: it is listed under the round's `refused` and does not touch the
verdict. `run_epoch(parameters, runs, schedule, seeds)` drives all of this in
process, as `run_protocol` does for one run.

## 3. Why the union-of-runs bound is sound

`Bound(C, I, theta)` at `eta` counts the outputs a prover can be accepted on
with probability at least `eta` when the sampling is Bernoulli per unit and
unpredictable at the time the boundary and the interiors are committed. Three
facts carry it from one run to a round:

1. **All boundaries of the round are fixed before any seed is known.** The
   round seed is drawn (or becomes public) only after the seal, and each run's
   seeds are HMACs of the seal; a run's boundary commitment is bound into the
   seal through its boundary-phase digest. Altering, dropping, inserting or
   reordering any boundary changes the seal and hence every seed of the round
   (tested).
2. **Within a run, interiors are committed after its `J` and before its `T`**,
   exactly as today: `challenge_replay` is derived from `q_i` over the sealed
   boundary phase, `receive_interiors` derives `T` from `s_i` over the interior
   phase. The `s` seeds are never in the `RoundChallenge`; they are derived
   from the verifier-private round seed and appear only in each run's
   `SampleChallenge`.
3. **Per-unit Bernoulli sampling makes the union's sample the product of the
   runs' samples.** Selecting each RU of the union independently with
   probability `q` and each VU of a selected RU with probability `s` is the
   same distribution whether one seed drives the whole union or each run has
   its own seed; HMAC-derived seeds are independent of one another for anyone
   without the round seed. So the survival of an error set spread over the
   runs of a round is `sigma(E) = prod_r f(l_r)` over all RUs of all runs, the
   quantity `Bound` integrates.

Hence the round is one circuit -- the disjoint union of its runs' circuits --
whose boundary was committed before its challenge, and `Bound` over the
union's kind table at `eta / rounds` applies verbatim. The adversary test
spreads one corrupted VU over each of three runs of a round and observes the
epoch's acceptance rate at `(1 - q s)^3` within the datacenter simulation's
tolerance (`TOLERANCE_SIGMAS = 4`), each unit surviving at the single-run
rate; the epoch's capacity is the union bound.

## 4. The union table

`veritor.analysis.union(tables) -> KindTable` is the kind table of the
disjoint union of circuits. A fresh root with no ports calls every constituent
root once (identical tables are counted, so `union([T] * N)` is `O(rows + N)`);
its `out_bits = reach_bits = ancestor_bits` are the summed interface. Rows
sharing a kind digest merge: `copies` sum, `reach_bits` and `ancestor_bits`
take the max over copies (sound: `cut_bits` is a min), depths shift by one,
`closed` is the conjunction; rows of one kind must agree on everything the kind
digest determines, else `ValueError`. A constituent root keeps its own
`out_bits` and is enclosed by the union's interface, so its cut is its own
interface -- exactly the row a copy of a request has in the headline
estimate's single table. Tests: `union([T, T])` doubles every count and the
root's `out_bits`; `bound(union([T] * N))` equals the bound of the table with
copies scaled by `N` (the headline method) for several shapes, policies,
`eta`s and fault budgets; nested unions price like the flat union; `cost` is
additive; and **the bound is monotone under adding runs** on random mixtures
of shapes with and without a fault budget -- no counterexample was found, so
admission's running check against `U_max` is sound: a run refused now would
be refused in any superset of the round.

The round's capacity is `bound(union(round tables), theta, eta / rounds,
max_faults=budget).bits`; the epoch's is the sum over rounds. Note the two
are different numbers: three runs in one round are priced together at `eta`,
three rounds of one run are priced separately at `eta / 3` and summed (both
tested against their formulas).

## 5. Storage and replay

Round close is when the prover replays. `EpochProver` holds a `ProverSession`
per run with the same `(compiled, values, weight_tree, replay, declare,
backend, plan)` a session holds today; in a real system the values are not in
memory for a year, and `replay(unit, boundary_values)` is deterministic
recomputation from stored boundary data. The **replay unit is the unit of
recompute**: for a request RU the boundary is the prompt and the weights (the
response is public), for a step RU it is a KV-cache state. The storage a round
costs is therefore the boundary data of its RUs -- for request RUs, the
prompts, which the server keeps anyway -- plus the boundary commitments, and
it is released when the round's evidence is accepted. Nothing here builds that
storage; `Run.values` stands in for it.

## 6. Judgment calls and flags for the architect

- **The round seed is verifier-private, not a beacon.** The design allowed
  "fresh verifier randomness or a public beacon output". Because the `s`
  seeds derive from the same round seed, a public round seed would let the
  prover compute every `s_i` before committing interiors and grind its
  interior commitments against the sample -- the attack the two-stage
  structure exists to prevent. `RoundChallenge` therefore carries the seal and
  the per-run `ReplayChallenge`s (each with its `q_i`), never the round seed.
  A beacon can serve the `q` side only if the `s` seeds come from a second
  secret (verifier randomness, or a second beacon output after all interiors
  of the round are committed). Decide before any deployment uses a beacon.
- **`Claim` instead of optional seeds.** Making `Expectation`'s seeds optional
  contradicts an existing test that `replace(expectation, q_seed=None)`
  raises. `Claim` is the expectation without seeds (the header's preimage);
  `Expectation.claim` projects to it; `VerifierSession` admits either and
  `release(q_seed, s_seed)` supplies seeds later. `receive_boundary` is now
  `accept_boundary` + `challenge_replay`. Header bytes, transcripts and all
  existing tests are unchanged.
- **Which run index enters the seed.** `run_index` is the run's admission
  index within its round (not its position among received boundaries), so a
  run's seed does not move when an earlier run's boundary never arrives.
- **Out-of-phase messages are refused, not judged.** A second boundary for a
  run, a boundary for a run whose round has closed, interiors before the round
  closes, or any message to a run that already has its verdict raises
  `Reject(INVALID_PHASE)` without judging the run -- the semantics of
  `VerifierSession._expect` (which leaves the session's phase alone). A
  message that fails its checks (bad opening, I/O mismatch, a relation, the
  round's budget) judges the run, as today, and the verdict is final: a
  smaller declaration list cannot revive a run rejected for the budget
  (tested). A run admitted whose boundary never arrives is judged at round
  close ("the boundary never arrived") and rejects the epoch.
- **The header enters the chain with its boundary.** `link_i` covers
  `Header_i.digest` and `boundary_phase_i` as specified; admission alone does
  not move the chain (the header is in `stream` for inspection). An admitted
  run without a boundary is not in the seal but rejects the round regardless.
- **Refused admissions are not committed runs.** Over-`U_max` (or `A`,
  `W_max`) refusals are recorded per round and do not affect the verdict; the
  epoch guarantee covers what was served.
- **Terminology collision.** The paper (and `docs/stress-tests.md`) use
  *epoch* for one protocol execution and *session* for the epochs between
  decisive rejections. This layer uses *epoch* for the year and *run* for the
  paper's epoch. One of the two vocabularies has to give.
- **What does not compose across runs.** `max_work` and the resource limits are
  per run: the verifier's expected work for a round is the sum over its runs,
  and nothing caps that sum. `max_advice_bits` is per run too, so the epoch's
  capacity is `sum_rounds Bound + sum_runs |a|`, not `Bound + A`. The union
  table treats the runs' circuits as disjoint: a model's weights are committed
  once under `kappa_W` and referenced by every run, which is fine for the
  sampling argument (`Bound` reads no weight total; a weight opening is
  authenticated per run) but the union's `weight_count` sums per run, so
  `cost(union)`'s weight-commitment term counts the model once per run.
- **`eta / rounds` in every header.** A run's header binds the round's
  threshold, not the epoch's. Transcripts of an epoch run verify under an
  `Expectation` with the derived seeds and `VerifierParameters(eta / rounds,
  ...)` (tested); a reader of a single transcript sees a per-round `eta`.
- **Not built.** Storage of boundary data, deterministic replay from a KV
  state, a wire format for `RoundChallenge`/`EpochReport`, and a beacon
  binding. The datacenter simulation runs through the epoch layer in
  `veritor.simulation.epochs` (a simulation cut into rounds of `ClusterG`
  runs, one per fleet or per pod, through `run_epoch`; the `H6` rows of
  `docs/honest-prover.md`, section 8), but a request that spans two rounds
  still cannot read the KV its earlier steps declared in the previous round's
  run: the cross-run read, whose four pieces section 8 lists (a `Join` that
  names the run and request it resumes, a `reads` field of `Claim` and
  `Header` with `foreign_openings` in the `BoundaryMessage` against the
  foreign `Commitment`, a `Bound` that counts the read rows' declaring units
  as reaching the reading round, and retention of the read rows past their
  round). Nor is there a straggler path: an admitted run whose boundary
  misses the seal is judged, not refused or carried into the next round
  (section 8 specifies both alternatives), so the honest prover admits a run
  only with its boundary in hand. The adversary test is dedicated and uses
  the simulation's tolerance.
