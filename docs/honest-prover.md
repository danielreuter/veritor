# The honest prover under faults: fault classes, secure late advice, no full replay

An honest server's stack is deterministic but not perfect: silent data
corruption, a pod that reads a corrupted weight for an hour, a KV block that
rots between steps, a crash mid-request, a straggler at round close. This
programme plays that developer against the protocol as built and records, per
fault class and per strategy, what the prover had to declare, what the
declarations cost in capacity, and how much of its own computation it had to
re-execute to know what to declare. The end product is the table in the last
section and the decision memo in section 1.

Standing assumption: production kernels are deterministic and pinned, so an
honest replay is bit-exact with the description; nondeterministic kernels are
the hardware track's problem and are out of scope here.

Terminology: *epoch*, *round*, *run* as in `docs/epoch.md`; RU = replay unit,
VU = verification unit; `J` is the q-challenge (which RUs are opened), `s` the
VU sample; `u(1) = W_V + log2 |S|` is the pre-challenge price of one
declaration (`docs/stress-tests.md`, M6).

## 1. Findings

*Filled in by the decision memo once every section below has its rows.*

## 2. Where advice may enter, and at what price

*Summary of `docs/notes/late-advice.md`; the note carries the adversary model
and the proofs.*

## 3. Honest replay: what the prover holds and how it replays

*What a real server records (the recording policy), how it reconstructs an
opened RU from the recorded boundary with the recorded values pinned, and
which VUs that leaves it to declare.*

## 4. Fault classes

*The matrix: landing site, magnitude, persistence, scope, detectability before
streaming, RU choice, recording policy; per cell the declarations needed,
whether a VU declaration can express the fault at all, the cheapest admissible
declaration kind, and the prover's diagnosis cost.*

## 5. Random silent data corruption at fleet scale

*Strategies P0 (no declarations), P1 (post-J declarations as built), P2
(signal-assisted pre-J pardons plus P1), P3 (100% pre-J replay); the phase
diagram over fault density, `q` and `s`.*

## 6. Systematic faults: a pod is wrong for an hour

*Corrupted weight cell, stale weight version, wrong kernel path, fleet-wide
update mid-round; what each costs under P1 and under the priced alternatives.*

## 7. Detection before the challenge

*Hardware signals, pre-streaming value checks with S7 truncation, partial
re-execution on idle capacity; the expected charge per fault against the cost.*

## 8. Round-close logistics

These are faults of logistics, not of computation: a pod that dies, a pod
whose values are late at the seal, a KV transfer that never arrives, a
request that outlives the round it started in. Each is driven through the
datacenter simulation (`veritor.simulation.workload`) cut into rounds of
`ClusterG` runs (`veritor.simulation.epochs`: a round is a window of fleet
time, RU = step, the window's schedule as advice, one run per round for the
fleet or one per pod) and through `run_epoch` with an honest prover;
`tests/veritor/stress/test_honest_logistics.py` records the rows. Two pods of
two slots, sixteen steps in two rounds of eight (H6e: three pods, six
rounds), the datacenter demo's small shape. Bits are the schedule's advice
bits against the same arrivals undisturbed, cost is the honest replay cost
against that baseline (`Cost(...).total` over it stays between 1.04 and 1.12
in every row), and the close delay is what the verifier waits past the
nominal boundary.

| Row | Scenario | Verdict today | Decl. | Bits | Cost | Close delay |
|---|---|---|---|---|---|---|
| H6a | pod 0 dies at step 4 with 2 requests in flight; both restart on pod 1 inside the round | both rounds ACCEPTED | 0 | 228 (+24: two re-joins) | +14.4% (the prefix recomputed) | 0 |
| H6at | the same crash, the clients give up: 2 requests keep the 3 tokens they had of 6 | ACCEPTED | 0 | 204 (+0; 27 outputs for 30) | -6.8% (steps not run) | 0 |
| H6bw | straggler, wait: pod 1's last 3 steps arrive 3 steps late; one run per round | ACCEPTED | 0 | 204 (+0; the runs are identical) | 0 | 3 steps (0.15 s) per close |
| H6bd | straggler, defer: per-pod runs, pod 1's run admitted one round late | all 3 rounds ACCEPTED | 0 | 200 (+0 over per-pod runs on time) | 0 | 0 at the close; pod 1's tokens one round (0.40 s) late, a trailing round, `eta / 3` for `eta / 2` |
| H6bt | straggler, truncate: pod 1 commits 3 steps early; its 2 cut requests continue next round with the prefix in `x` | ACCEPTED on time | 0 | 236 (+36: the continuation joins) | +6.2% (prefills over prompt + prefix) | 0 |
| H6c | KV transfer lost: a request prefilled on pod 0 never reaches pod 1, which re-prefills it | ACCEPTED | 0 | 53 (+0: a fresh join costs what a resume did) | +23.4% (one prefill) | 0 |
| H6dh | request longer than a round, HOLD: committed whole in the round it completes in | ACCEPTED | 0 | 180 (baseline) | 0 | 1 step here; 3.0 steps mean, 5 max (62% of a round) over 99 closes |
| H6ds | the same, SPLIT: cut at the boundary, re-joined afresh in a slot of its own | ACCEPTED | 0 | 190 (+10) | +10.6%; the 4-token prefix output twice | 0 |
| H6dc | the same, CONTINUE: the rest is a new request, prompt + prefix | ACCEPTED | 0 | 185 (+5) | +9.1% (a 7-position prefill for 3) | 0 |
| H6e | fleet churn: 3 pods, 6 rounds, 3 failures, 3 restarts; crossings continue | all 6 rounds ACCEPTED | 0 | 1470 (+108; 36 per restart) | +4.8% | 0 |

Nothing here is a declaration. A crash is not silent: the schedule says
where the pod stopped and where the request re-joined (M4), the recomputed
prefix is gates, and with RU = step a truncated request's length is its
join's length field, so H6at pads nothing (S7's blank check outputs are the
request-RU case); a round that holds the crash needs nothing from the epoch
layer. The straggler is the one scenario the layer as built gets wrong for an
honest prover: `EpochVerifier.close_round` records an admitted run whose
boundary is not in hand as `INVALID_PHASE` ("the boundary never arrived
before the round closed"), `receive_boundary` refuses the boundary when it
does arrive ("the run was admitted in round 0, which is closed"), the run's
table has been counted in the round's `Bound`, and re-admitting the same
compilation under a fresh session id is accepted in the next round while the
epoch's verdict stays the first failure (`test_h6b_verifier_today`). The
three honest moves that avoid it cost wall-clock (wait: 3 steps per close,
the runs unchanged), a round of commitment latency and a round of `eta`
(defer: per-pod runs, the late pod's run admitted one round later with the
boundary it then has, `eta / 3` per round for `eta / 2` and a trailing
round), or 36 advice bits and a longer prefill (truncate: the late pod
commits what it has and its cut requests continue as new requests). A
network partition during the seal is the straggler for every pod on the far
side and is priced by H6b; its data-plane consequence, a KV transfer that
never arrives, is H6c: pod 1 re-prefills, the orphan prefill stays in the
circuit with its KV read by nobody, and nothing distinguishes it from a pod
that died after its first step. With per-pod runs the *successful* transfer
is itself a cross-run read and is not expressible today.

A request that crosses the boundary reads, in round `r + 1`, the KV cache
its own steps declared in round `r`, under another run's boundary
commitment, which no port of the later run can name. HOLD costs nothing in
bits or compute and pays in latency: the earlier round's close waits for the
request (for lengths uniform on 3-6 steps and rounds of 8 steps, 99% of the
99 boundaries of an 800-step run have a request in flight, 2.9 of 4 slots on
average, and the wait is 3.0 steps on average and 5 at most) or the tokens
it streamed before the boundary are committed a round late, which in a fleet
whose requests are longer than a round is every request. SPLIT closes on
time and re-joins the request afresh, recomputing the prefix one step per
position in a slot the original schedule did not have; `Schedule` has no
field for positions streamed in another run, so the prefix is output twice
(47 outputs for 43 tokens) and the re-join costs +10 bits and +10.6% replay.
CONTINUE closes on time too: the remainder is a new request whose prompt is
the original prompt plus the streamed prefix, prefilled in one step in the
original slot, +5 bits and +9.1% replay (one prefill over 7 positions for
3), nothing output twice; that the continuation's prompt ends in round `r`'s
claimed outputs for the same request is checkable from the two runs' public
claims, and no rule checks it today. The cross-run read would make all three
zero: the resume join's bits, no recompute, no wait.

What the epoch layer must add for the cross-run read (a resumed join reading
the KV rows another run declared), four pieces, none built:

- `Schedule`: a `Join` with `resume` names the request's latest attempt *in
  this schedule*. A cross-run resume must name the earlier run (its position
  in the commitment stream) and the request's index in that run's `x`, two
  gamma-coded fields on the join charged as advice, and the position it
  continues from, which `Span.start` today derives from a previous attempt
  the schedule does not have.
- `Claim` and `Header`: a `reads` field, per foreign run its `Header.digest`
  and the boundary positions read, bound into the header digest (a
  `PROTOCOL_VERSION` bump), so the run's statement names what it read. The
  foreign run must be committed before the reading run is admitted (a closed
  round, or earlier in the same round); the epoch's verdict, which requires
  every run accepted, then covers the read. Opening rule: the rows enter the
  reading run as `in` gates whose values are not in `x`; `BoundaryMessage`
  carries `foreign_openings`, openings of the read positions against the
  foreign run's `Commitment` (in `EpochVerifier.stream`), checked at
  `receive_boundary` as `io_openings` are today, after which every opening
  of the reading run is its own.
- `Bound` accounting: `union` prices a round's runs as disjoint circuits
  (`docs/epoch.md`, section 4). A read joins two rounds' circuits: the units
  that declared the rows in round `r` now reach outputs of round `r + 1`,
  which round `r`'s bound did not count. Either the reading round carries
  those units (a row per foreign declaring kind with `copies` = the rows
  read and `reach_bits` = the reading run's outputs reachable from them,
  `cut_bits = min(out_bits, reach_bits, ancestor_bits)` as for any unit) or
  the rows are charged as source values at their width in bits (the M4
  price, one KV row per layer per position). The first is the cheap one and
  needs the union bound over rounds re-proved for connected rounds.
- Storage: a round's boundary data is released when its evidence is accepted
  (`docs/epoch.md`, section 5). Rows a later run reads must be retained by
  the prover until the reading round is accepted, and the verifier keeps the
  foreign `Commitment`, which the stream already holds. For a live request
  this is the KV cache the server holds anyway; the extension is the rows of
  requests that completed in round `r + 1` while round `r` is still being
  answered.

For stragglers (a run admitted whose boundary misses the seal), one of two
changes to `EpochVerifier`, neither built:

- `close_round` lists an admitted run without a boundary under
  `RoundReport.refused` instead of judging it: the header never entered the
  chain (`stream_link` covers a header and its boundary together), so
  nothing the seal commits changes; the run's table leaves the round's
  `Bound`, and the prover re-admits it later with no rejection standing.
- Or the run is *carried*: its boundary is accepted in round `r + 1`,
  `_Run.round` and `_Run.index` become that round's (the seeds are HMACs of
  round `r + 1`'s seal and the run's index there), `RoundReport` names it in
  both rounds, its table is counted where it is sealed, and an
  `EpochParameters.max_carry` (rounds a run may wait; 1) bounds the wait. The
  header binds `eta / rounds` and no round index, so no re-admission is
  needed.

What the honest prover should do today. Admit a run only with its boundary
in hand: compute the run's values, then `admit` and `receive_boundary` back
to back; the seal binds a header only together with its boundary, so nothing
is lost by admitting late. A pod whose values are not in hand at the close is
deferred a round (H6bd, per-pod runs) or truncated at the last step it has
(H6bt), never admitted early; with one run for the fleet the verifier waits
(H6bw). A request that will cross the boundary is continued (H6dc), the
cheapest on-time policy in bits and in compute with nothing output twice;
held (H6dh) when the close can wait and bits matter; never split. Crashes,
restarts, lost transfers and abandoned requests are schedule, never
declarations (H6a, H6at, H6c, H6e). Per-pod runs isolate a straggler to its
pod and cost 4 bits less than the fleet run here, but make every cross-pod
resume (S6) a cross-run read; a fleet run needs every pod's values before it
can be admitted.

## 9. Recording policies

*Tokens only, KV boundary, all VU outputs, kernel-path logs: which strategies
and declaration kinds each policy enables and what it costs.*

## 10. Row identifiers

Rows are the `H` section of the stress catalogue, recorded through the
`honest` fixture of `tests/veritor/stress/conftest.py` into
`docs/data/stress-honest.json`: `H2*` fault classes (section 4), `H3*` random
SDC at fleet scale (section 5), `H4*` systematic faults (section 6), `H5*`
detection (section 7), `H6*` round-close logistics (section 8: `H6a` a crash
mid-request and `H6at` the same with the clients gone; `H6bw`, `H6bd`, `H6bt`
a straggler waited for, deferred, truncated; `H6c` a lost KV transfer; `H6dh`,
`H6ds`, `H6dc` a request longer than a round held, split, continued; `H6e`
fleet churn). Each row carries `declarations`, `charge_bits` and `recompute`
besides the catalogue's fields; the `H6` rows add `rounds`, `runs`, `outputs`,
`check_outputs` and `honest_cost`.

## Results

Generated by `python -m veritor.stress.report` from `docs/data/stress-honest.json`; declarations are what the honest prover had to declare for the run to be accepted, the charge is what those declarations add to `U` under the price the mechanism's stage admits (`docs/notes/late-advice.md`), recompute is the fraction of the production computation the prover re-executed to know what to declare, `U` is `Bound` at `eta = 2^-40` including the charge.

| ID | What happened | Mechanism | Declarations | Charge (bits) | Recompute | U (λ = 40) | Verdict |
|---|---|---|---|---|---|---|---|
| H6a | crash mid-request, ClusterG through the epoch layer: pod 0 dies at step 4 of round 0 with 2 requests in flight; both restart from the prefill on pod 1 (Schedule v3 re-join) and finish inside the round; 2 pods x 2 slots, 2 rounds of 8 steps, one run per round, 12 requests | M4 (the failed attempt and the re-join are two joins of the schedule) | 0 | 0 | 0 | 480 | both rounds ACCEPTED, 0 declarations: a crash is not silent, the schedule says where the pod stopped; 228 advice bits vs 204 over the same arrivals without the crash (+24: 2 re-joins, the queue reshuffled); outputs = the 30 streamed tokens |
| H6at | the same crash, the clients give up: the 2 requests stay truncated at the 3 tokens streamed before step 4 (of 6 wanted); no restart | M4 (the join's length is the truncated request's t; no blank check outputs with RU = step) | 0 | 0 | 0 | 432 | both rounds ACCEPTED, 0 declarations, 0 check outputs: with RU = step the generated length is the join's length field, already in the schedule (S7 pays it as its own advice and pads with blank check outputs under RU = request); 204 advice bits vs 228 with the restarts; 27 outputs vs 30 |
| H6bd | straggler, defer: one run per pod per round; pod 1's run of round r is admitted in round r + 1 with the boundary it then has; the epoch closes 2 rounds on time and needs a trailing round 2 for pod 1's last run | none (admission is the prover's move; the run is the same, admitted later) | 0 | 0 | 0 | 480 | all 3 rounds ACCEPTED, 0 declarations; runs per round [1, 2, 1]; the same 200 advice bits as the per-pod fleet on time (200); pod 1's tokens are committed one round (8 steps (0.40 s)) late, and each round is bounded at eta / 3 instead of eta / 2 (U 480 vs 480 bits, both capped at \|Out\|) |
| H6bt | straggler, truncate: one run per pod per round; pod 1's round-0 run ends 3 steps early (step 5) with the values it has, and its 2 request(s) across step 5 continue in round 1 as new requests whose prompt is the original prompt plus the 2 tokens already streamed | M4 + x (the continuation is a request of the next run; its prefix is public input) | 0 | 0 | 0 | 480 | both rounds ACCEPTED on time, 0 declarations, 0 check outputs; 30 outputs = the fleet's (30), nothing streamed twice; 236 advice bits vs 200 for the per-pod fleet (+36: the continuation joins, and the cut run's shorter fields); the 2 prefix tokens enter round 1's x as prompt |
| H6bw | straggler, wait: pod 1's boundary values for the last 3 steps of each round arrive 3 steps after the verifier wanted to close; the verifier waits; one run per round for the fleet, as H6a | none (the close is the verifier's move; the runs do not change) | 0 | 0 | 0 | 480 | both rounds ACCEPTED, 0 declarations; the runs are byte-identical to the fleet without a straggler (204 advice bits); the cost is 3 steps (0.15 s) of delay per close on the challenges and on the next round's opening, nothing on U |
| H6c | KV transfer lost, ClusterG: request 0 prefilled on pod 0 (3 prompt positions, first token streamed), its KV bound for pod 1 never arrives; pod 1 re-prefills it as a fresh attempt one step later and decodes the remaining 4 positions; one round, 2 pods x 2 slots, 8 steps, an unrelated request alongside | M4 (a fresh join in place of the resumed one) + M1 (the re-prefill recomputes) | 0 | 0 | 0 | 128 | ACCEPTED, 0 declarations, 0 check outputs; outputs = the same 8 tokens as the disaggregated run whose transfer succeeded (position 0 recomputed, not re-streamed); 53 advice bits, exactly the successful transfer's (53): the resume flag is a bit either way; honest replay cost 9178 vs 7436 (+23.4%, one prefill of 3 positions) |
| H6dc | request longer than a round, continue: round 0 commits the 4 positions; round 1 holds a new request whose prompt is the original prompt plus those 4 tokens, prefilled in one step in the original slot, then 1 more decode step | M4 + x (the prefix is public input of the next run) | 0 | 0 | 0 | 688 | both rounds ACCEPTED, 0 declarations, 0 check outputs; 43 outputs = the streamed tokens, nothing twice; 185 advice bits (+5 over hold), 573 description bytes for the 4 prefix tokens and the longer prefill; honest replay cost 41174 vs 37748 (+9.1%: one prefill over 7 instead of 3 positions) |
| H6dh | request longer than a round, hold: request 4 joins at step 4 for 5 steps, crossing the boundary at step 8 (4 tokens before, 1 after); its whole attempt goes into round 1's run, whose window opens at step 4; 10 requests, lengths 3-6 on 2 pods x 2 slots | none (the run is compiled when the request completes; the schedule is unchanged) | 0 | 0 | 0 | 688 | both rounds ACCEPTED, 0 declarations; 43 outputs = the streamed tokens, 180 advice bits (the baseline for H6ds, H6dc); the 4 tokens streamed in round 0 are committed one round late, or round 0's close waits 1 step (0.05 s) |
| H6ds | request longer than a round, split: round 0 commits the 4 positions streamed by step 8; in round 1 the request re-joins as a fresh attempt in a slot of its own, recomputing the 4 positions one per step (a restart's semantics) before 1 new one | M4 (a second join) + M1 (the prefix recomputed) | 0 | 0 | 0 | 752 | both rounds ACCEPTED, 0 declarations; 47 outputs: the prefix is output twice (4 tokens, in both runs), since Schedule has no field for positions streamed in another run; 190 advice bits (+10 over hold: the re-join and a third slot); honest replay cost 41758 vs 37748 (+10.6%) |
| H6e | fleet churn, ClusterG through 6 rounds of 8 steps on 3 pods x 2 slots: pods fail at 2% per step, 3 failures (2 with occupants), 3 restarts over 60 requests; spanning requests continue (H6dc) | M4 (every restart is a join; every crossing is a continuation in x) | 0 | 0 | 0 | 2,608 | all 6 rounds ACCEPTED, 0 declarations; 2 of 6 rounds hold a restart; 1470 advice bits vs 1362 for the same arrivals without failures (+108, 36 per restart); honest replay cost 192382 vs 183604 |

Notes:

- **H6a**: U capped at \|Out\| = 480 bits summed over 2 rounds at eta / 2 (uncapped 1621 bits); honest replay cost 38220 vs 33396 without the crash (the recomputed positions are gates); a run whose window holds the crash needs nothing from the epoch layer
- **H6at**: U capped at \|Out\| = 432 bits summed over 2 rounds at eta / 2 (uncapped 1478 bits); honest replay cost 31128 vs 33396 uninterrupted: the truncated requests' remaining steps are not in the circuit
- **H6bd**: U capped at \|Out\| = 480 bits summed over 3 rounds at eta / 3 (uncapped 1655 bits); per-pod runs take 200 advice bits against the fleet run's 204: a schedule header per run, no pod field per join; a request restarted on the other pod would be in both pods' runs (its recomputed prefix output twice)
- **H6bt**: U capped at \|Out\| = 480 bits summed over 2 rounds at eta / 2 (uncapped 1648 bits); honest replay cost 35472 vs 33396: the continuations' prefills re-read the 2 prefix tokens the earlier run computed (M1 recompute of the prefill, not of the decodes); the verifier can check a continuation's prompt against round 0's claimed outputs but no rule does so today (section 8 gap list)
- **H6bw**: U capped at \|Out\| = 480 bits summed over 2 rounds at eta / 2 (uncapped 1648 bits); EpochVerifier today: a run admitted but without a boundary at close_round is recorded INVALID_PHASE ('the boundary never arrived before the round closed'), its late boundary is refused, and the epoch's verdict is that run's -- so an admitted straggler fails the epoch; the honest prover admits a run only with its boundary in hand (test_h6b_verifier_today)
- **H6c**: U capped at \|Out\| = 128 bits (uncapped 508 bits); the circuit holds the orphan prefill step (its KV declared, read by nobody) and the second prefill; nothing distinguishes a lost transfer from a pod that died after its first step (H6a); with per-pod runs the successful transfer itself is a cross-run read (the resume would name a KV row of another run's boundary) and is not expressible today
- **H6dc**: U capped at \|Out\| = 688 bits summed over 2 rounds at eta / 2 (uncapped 2451 bits); the verifier sees two requests where the client saw one; that the continuation's prompt ends in round 0's claimed outputs for the same request is checkable from the two runs' public claims and is not checked by any rule today; the cross-run read (a resume join naming round 0's KV rows through the stream) would cost the resume join's bits and no recompute, and needs the four pieces listed in docs/honest-prover.md section 8
- **H6dh**: U capped at \|Out\| = 688 bits summed over 2 rounds at eta / 2 (uncapped 3047 bits); over 99 closes of an 800-step run of this workload, 99% of the boundaries have a request in flight (2.9 on average of 4 slots); holding the close costs 3.0 steps on average and 5 at most (62% of a round) for lengths uniform on 3-6; the cross-run read would make this 0
- **H6ds**: U capped at \|Out\| = 752 bits summed over 2 rounds at eta / 2 (uncapped 2675 bits); the re-join needs 5 steps of slot time in round 1 where the original attempt had 1 left, so it does not fit an existing slot (the cluster gets one more); a real server would re-prefill prompt + prefix in one step, which is the continuation's circuit (H6dc) with the original request in x: a join shape Schedule lacks
- **H6e**: U capped at \|Out\| = 2608 bits summed over 6 rounds at eta / 6 (uncapped 7599 bits); at the Llama 3 405B rate (419 unexpected interruptions in 54 days on 16,384 H100s: 4.7e-04 per GPU-day, 3.8e-03 per 8-GPU pod-day) a 1,000-pod fleet sees 3.8 pod failures a day, 2.6e-03 per one-minute round: 0.26% of rounds hold a restart, each a few joins of advice and no declaration; the simulation's rate is inflated to see several in 6 rounds
