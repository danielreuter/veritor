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

A fault is classified by where it lands (a VU's output word in an RU's
interior, a recorded boundary word, a weight cell, an input cell, a kernel),
by how much of the word it moves (one low bit, one high bit, the whole word),
by how long it lasts (once; for a pod-hour; until a rollout is fixed) and by
how far it reaches (one VU; every reader of one cell on one pod; every VU of
one kind on one pod; every pod). Two protocol choices then set what the honest
prover has to declare: the RU (request under `RequestsG`, step under
`ClusterG`) and the recording policy of section 3.

The declaration count is the pinned set of the honest model's replay
(`replay_pinned` in `veritor.simulation.honest`, honest-sim branch; the same
rule is `pinned_units` in `veritor.simulation.systematic`, checked against it
in `tests/veritor/stress/test_honest_systematic.py`). Under `VU_OUTPUTS`
recording it is every VU whose recorded output disagrees with its
recomputation from its recorded inputs: exactly the VUs that computed wrongly,
one per corrupted kernel, whatever happened downstream. Under `BOUNDARY`
recording (tokens under request RUs; KV entries and tokens under step RUs) it
is every VU owning a recorded word that disagrees with the recomputation of the
RU from the recorded boundary: the committed words the fault changed, however
many or few VUs were wrong on the way. A fault that changed no recorded word
costs nothing under `BOUNDARY`; a fault that corrupted one VU early in an RU
costs every KV entry and token downstream of it. The two counts are the
readers' cut and the outputs' cut of the same fault and neither bounds the
other: a weight read fault costs 18 declarations under `VU_OUTPUTS` and 7
under `BOUNDARY` on the toy (36 gate readers, every streamed token flipped),
a corrupted constant costs 2 and 17.

The matrix. "Signal" is what can tell the prover about the fault before it
streams and before `J`, which is what a pre-J price needs (section 2).
Toy numbers are the honest model's `H2` rows (two requests, `RequestsG` unless
said) and this document's `H4` rows (section 6); `u_post(1)` is the post-J
price of one VU declaration, `145.6` bits at the simulation policy on the
note's fixture (`142.6` on the cluster fixture), `6.12e9` at the headline.

| Class | Site, magnitude, persistence, scope | Signal before streaming | M6 declarations, `VU_OUTPUTS` | M6 declarations, `BOUNDARY` | Cheapest admissible kind and price | Diagnosis: what the prover recomputes to know what to declare |
|---|---|---|---|---|---|---|
| Single-bit flip in a VU output, low bit | interior word; one bit; once; one VU | none (silent data corruption in compute) | 1, the VU (toy: 1) | 0 when no token or KV word moved (toy: 0), else the recorded words it moved | M6 post-J, `u_post(1)`; nothing cheaper is needed | none beyond the replay of the opened RU: the pin falls out of pinned replay |
| Single-bit flip in a VU output, high bit | interior word; one bit that moves the value by half its range (sign or exponent in floating point; the modular toy's top bit); once; one VU | none | 1 (toy: 1) | the recorded words it moved: 0 (toy, request RUs: 0, the flip was absorbed) up to every later token of the request | M6 post-J | none beyond the replay |
| Flipped token | the streamed token, a boundary word; the argmax moved; once; one position, and every later position of the request reads the wrong token | a recompute of the head before streaming | 1, the head's VU | 1, the token; the later positions were computed from the recorded token and are consistent with it | M6 post-J | none beyond the replay |
| Catastrophic value (NaN/Inf-like, saturated word) | interior word; every bit; once; one VU, and everything downstream of it in the RU is garbage | a range check on the interior (S7 truncation) or on the logits before streaming | 1 (toy: 1) | every recorded word downstream in the RU: the request's tail of tokens, or the position's KV entries and token under step RUs | under `VU_OUTPUTS` M6 post-J for the one VU; under `BOUNDARY` the tail is many declarations against `f_max`, so re-serve the request before streaming (the value check makes it detectable) | the value check names the RU; the pin falls out of the replay |
| Weight read fault (one cell misread by every reader of one run; `kappa_W` right) | weight cell as read; one bit; the run; every reader of the cell in the run (one VU per position for a layer cell) | the ECC/scrub log when the misread is a memory error; none for a corruption in the load path | every reader whose output moved (toy, the most-read cell, top bit: 18 VUs pinned, 36 gate readers) | the tokens that flipped (toy: 7 of 7) and, under step RUs, the KV entries the readers of a `W_k`/`W_v` cell produce | run-wide source-position pardon, `W_cell + log2 n_W` bits pre-J (`25.7` toy, `51.9` headline); post-J the note's bound is `d u_post(1)` and the forced-consistency price is open (`docs/notes/declaration-kinds.md`) | the ECC log names the cell; else, under `VU_OUTPUTS`, the first disagreeing VU in address order is a reader and its fan-in (`d_model` cells) holds `w`, one relation solves `v'`; under `BOUNDARY` the first disagreement is a KV entry or token far downstream, so re-execute one opened RU on the faulty pod with VU logging and diff |
| Weight cell corrupted at rest for a pod-hour | weight cell in the pod's HBM; one bit; an hour; every reader on the pod (one per position for a layer cell) | the ECC scrub log, when the corruption is detectable | `positions per pod-hour x readers per position`: 144,000 at the toy shape, `H4a` | `tokens per pod-hour x phi`, `phi` the argmax-flip fraction under the corruption (toy: 1 for the low bit of `W_q`, 0 to 2/7 for a top bit; 112,000 at the toy shape), plus the readers themselves for a `W_k`/`W_v` cell under step RUs | pod-scoped source-position pardon, one message of `W_cell + log2 (n_pods n_W)` bits pre-J (`25.7` toy); post-J see section 6 | as above; the pardon needs `(pod, w, v')` |
| Stale weight version on a pod | every cell of the pod's copy; whole words; until the rollout is fixed; every weight-reading VU on the pod | the rollout log | every dot on the pod (toy: 344 of 526 VUs; 5.5 M per pod-hour, `H4b`) | every recorded word of the pod (toy: 79 of 79; 1.26 M per pod-hour) | a per-pod public weight root (M2/M8), 0 bits: the version is a public object; source pardons cost one message per changed cell and scale with `|W|` | none: the rollout log names the version |
| Wrong kernel path on a pod | every output of one gate kind on the pod; a relation change, not a value; until fixed; every VU of the kind | the kernel-path log | every VU of the kind (toy: 344 dots, `H4c`) | every recorded word of the pod (toy: 79 of 79) | a per-pod gate set in the constructor (M8), 0 bits; a late lowering costs one bit per RU and is admissible only under the note's conditions (`1.55 U_0` per bit per RU at the headline) | the kernel-path log; else replay one opened RU under each candidate lowering |
| KV/boundary value corrupted at rest | a KV word committed right by its producing step, read wrong by the later steps of the request on the pod (step RUs only; under request RUs the KV cache is interior and this is a VU-output read fault); one bit or a word; until eviction; every attention VU of the later steps (toy: 1 to 3 per KV word) | ECC of the KV store | every consumer whose output moved (toy: 0, the attention absorbed the flip) | the consumers' recorded words that moved (toy: 0) | port pardon, `W_port + log2 (n_RU n_ports)` bits pre-J, one per consuming RU; or a run-wide source-position pardon of the boundary commitment's cell | the consuming RU's first disagreeing attention VU names the KV word among its keys |
| Input read fault | an input token as read by the embedding; another token id; once; the one embedding VU and the position's whole tail | none | 1 (toy: 1) | the tokens and KV words the misread moved: 0 up to the request's tail | M6 post-J for the one VU under `VU_OUTPUTS`; under `BOUNDARY` a source-position pardon of the input commitment's cell, `W_in + log2 n_in` bits pre-J | the first disagreeing VU is the embedding; the cell is its one input |

Reading the matrix. M6 can express every class in the table: a declaration
skips a VU's relation, and every fault is, at the recorded granularity, some
set of VUs whose relation fails. What it cannot do is express a class
cheaply. For the one-VU classes the two policies differ by at most the length
of a tail, and M6 at `u_post(1)` is the right kind. For the systematic
classes the count is a wiring count (readers per position times positions in
the window) or an output count (flipped words in the window), both far above
any `f_max` (section 6), and the cheap kinds are all statements about a
public object of a few dozen bits: a cell's value, a version, an architecture.
Detectability decides which price applies: the ladder of section 2 prices a
pre-J statement at its message and prohibits or charges `d u_post(1)` for the
same statement after `J`, so a hardware signal (ECC scrub, rollout, kernel
path) is what makes the systematic classes affordable, and the value check
that makes a catastrophic value detectable is what keeps its tail off the
declaration count under `BOUNDARY` recording.

The magnitude axis is about the value's range, not the bit index. The toy's
word is a 16-bit modular integer: flipping the top bit of a weight adds
`2^15 x` to a dot product, which is `0 mod 2^16` for an even activation `x`, so
the toy's top-bit flips are the silent ones (3 to 6 of 9 readers moved) and
its low-bit flips move every reader. In floating point the sign and exponent bits
are the catastrophic ones and a low mantissa bit the silent one.

## 5. Random silent data corruption at fleet scale

*Strategies P0 (no declarations), P1 (post-J declarations as built), P2
(signal-assisted pre-J pardons plus P1), P3 (100% pre-J replay); the phase
diagram over fault density, `q` and `s`.*

## 6. Systematic faults: a pod is wrong for an hour

A random flip corrupts one VU; a systematic fault corrupts a *cell* -- a
weight in a pod's HBM, the pod's copy of the weights, a kernel -- and every VU
that reads the cell is wrong for as long as the fault lasts. What M6 would
have to declare is then a property of the wiring and of the recording policy,
not of the fault: how many VUs on the pod read the cell in the window
(`VU_OUTPUTS`), or how many committed words those readers moved (`BOUNDARY`).
`veritor.simulation.systematic` computes the reader counts from the
description (`readers`: for each definition, how many VUs read each of its
ports, propagated through the call graph as arithmetic progressions; GPT-2
Small's 1.9 G gates resolve in about a second, and on the toy the result
agrees exactly with a walk of every gate) and measures both declaration
counts on the toy (`perturbed_run`: the production assignment of a run whose
RUs on one pod misread a cell or store what a wrong kernel computes;
`pinned_units`: the honest model's pinned replay over what the server
recorded). Rows `H4a`-`H4d` are `tests/veritor/stress/test_honest_systematic.py`.

**Structural counts.** On the toy (`ClusterG`, one layer, `d_model = 4`, two
heads) a layer matrix or bias cell is read by exactly one dot per position,
the attention shift by two (one per head), a head cell by one per prediction,
the embedding by one per position and the head's readers besides. On GPT-2
Small every one of the 85,054,464 layer parameters is read by exactly one VU
per position processed, `wte` by one per position plus one per prediction, a
`wpe` row by one VU per request at that position. So for the modal cell the
readers' cut of a pod-hour is the positions the pod processes in the hour, and
the outputs' cut is the tokens among those positions whose argmax the
corruption moved.

**The fixture and its scaling.** The simulated datacenter is `ClusterG` with
two pods of two slots, five requests over six occupied synchronous steps
(23 positions, 16 predictions; `small_config`). Pod 0 runs four steps (RUs 1,
3, 5, 7): 9 positions, 526 VUs of which 344 are dots, 79 recorded computed
words under `BOUNDARY` (72 KV entries and 7 tokens), every VU's output word
under `VU_OUTPUTS`. A pod-hour is 72,000 steps of 0.05 s and 144,000 slot
positions (`pod_scope`); the pod-hour counts below are the measured pod
scaled by positions (a factor of 16,000, keeping the fixture's mix: 58.4 VUs,
38.2 dots, 8.8 recorded words and 0.78 tokens per position -- the window
holds two prefills, in steady decode tokens approach positions). P1 declares
the pins that lie in opened RUs, `q` of them at the simulation policy
`theta = (1/2, 1/8)`; `f_max = 1` for two pods for an hour and `2` for
16,384 devices (`fault_budget(expected_faults(...))`); `u_post(1) = 142.6`
bits on this fixture (`rho = 740`). The interface of the toy run is 256 bits,
so every M6 charge below is notional: what the fold would add before the cap.

| Row | Fault on pod 0 | Pinned on the measured pod, `BOUNDARY` / `VU_OUTPUTS` | Faulty in the pod-hour, words / VUs | P1 declarations at `q = 1/2` | M6 charge at `u_post(1)`, bits | Cheapest mechanism and its charge |
|---|---|---|---|---|---|---|
| H4a | one weight cell (`W_q[0][0]`, low bit) read wrong by every reader | 7 / 9 | 112,000 / 144,000 | 56,000 / 72,000 against `f_max = 1` | 8.0 M / 10.3 M | pod-scoped source-position pardon: one message, 26 bits |
| H4b | stale weight version (192 of 201 cells differ) | 79 / 344 | 1.26 M / 5.50 M | 632,000 / 2,752,000 | 90 M / 392 M | per-pod public weight root (M2/M8): 0 bits |
| H4c | wrong kernel path: every dot stores a perturbed word | 79 / 344 | 1.26 M / 5.50 M | 632,000 / 2,752,000 | 90 M / 392 M | per-pod gate set (M8): 0 bits; a late lowering 72,000 bits (one per RU), under the note's conditions only |
| H4d | fleet-wide weight update mid-round (steps 3-5 on both pods) | 35 of 36 / 160 of 240 VUs | half-round: 1.23 M / 5.50 M | 614,445 / 2,752,000 | 88 M / 392 M | round close at the update (M2): 0 bits |

**A corrupted weight cell (H4a), cell by cell.** The grid over the layer's six
matrices at `[0][0]`, low bit and top bit, pinned `BOUNDARY` / `VU_OUTPUTS`
on the pod's 9 readers and 7 tokens:

| Cell | Low bit | Top bit | Why |
|---|---|---|---|
| `W_q` | 7 / 9 | 0 / 3 | the query feeds attention and the token; every token flipped under the low bit, none under the top bit |
| `W_k`, `W_v` | 9 / 9 | 3 / 3 | the readers' outputs are the KV entries a step RU commits: recorded under both policies, so the two counts agree |
| `W_o` | 4 / 9 | 2 / 3 | four, then two, of the seven tokens flipped |
| `W_1` | 7 / 9 | 0 / 6 | |
| `W_2` | 6 / 9 | 2 / 3 | |
| constant table cell | 17 / 2 | 19 / 3 | read once per position and once per prediction (16 readers on the pod) of which 2 or 3 move, early in the position: the outputs' cut exceeds the readers' cut |

Every low-bit reader moves (a modular dot product changes by the activation);
a top-bit reader is silent when its activation is even (section 4). Under
`BOUNDARY` the count is the tokens that flipped, `phi` of the pod's tokens
with `phi` between 0 and 1 on the toy's 8-token vocabulary, except for the KV
projections, whose readers' outputs are themselves recorded words. The row
scales the worst low-bit case (`phi = 1`): 144,000 readers, 112,000 flipped
tokens, 72,000 and 56,000 declarations in the opened RUs against `f_max = 1`.
Under both policies M6 rejects the run; the honest prover with M6 alone
re-serves the pod-hour on another pod before streaming (half the fleet-hour
in this fixture) or is rejected.

The fleet-scale flipped-output count is `tokens per pod-hour x phi`, `phi`
the probability that a corruption of one 16-bit cell moves the argmax at a
position of the large model. The toy measures `phi` from 0 to 1 depending on
the cell and bit; the 70B `phi` is an open measurement, and it decides whether
tokens-only recording brings a systematic fault within M6 at all: with request
RUs of 512 generated tokens, the one opened request of a pod-hour needs
`512 phi` declarations against `f_max = 2`, admissible only if `phi <= 0.4 %`,
where its readers' cut is 1024 declarations whatever `phi`.

**The alternatives, priced as the ladder of section 2 allows.** Toy prices
are at `theta = (1/2, 1/8)` on the fixture; headline prices are the note's
(`q = 1.57e-8`, `s = 8.91e-3`, `rho = 4.74e11`, `U_0 = 1.90e13`, `n_RU =
2.93e13` requests of 1024 tokens, `n_W = 7e10`, `W_cell = 16`, `W_R = 8192`,
`d = 1024` readers of a layer cell per request).

- *M6 as built* (post-J VU declarations, `u_post(1) = rho log2 (1 / (1 - s))`;
  142.6 toy, `6.12e9` headline): the counts above, times `u_post(1)`.
- *Run-wide or pod-scoped source-position pardon* ("cell `w` read as `v'`
  throughout the pod's RUs of the run"; `docs/notes/declaration-kinds.md`).
  Its message is `W_cell + log2 (n_pods n_W)` bits, 25.7 on the toy (the pod,
  the cell among 413 committed positions, `v'`) and about 52 at the headline.
  That is its established price before `J`. After `J` the note bounds it by
  `d u_post(1)`, the readers of the cell in the opened RU declared one by one
  (`0.33 U_0` at the headline, exactly the M6 charge under `VU_OUTPUTS`), and
  the argument that the run-wide scope forces every opened reader to answer to
  one value and so keeps the message price is stated in the design note and
  flagged there as open.
- *RU-scoped source-position pardon*: priced like `declared_bits`, the
  smaller of the readers it removes from the opened RU at `u_post(1)` each and
  the union over the pardons it could have been. On the fixture that is 285
  bits per opened RU, 10.3 M for the pod-hour -- the M6 charge under
  `VU_OUTPUTS` to the bit -- and at the headline `d u_post(1) = 0.33 U_0`
  again: an RU-scoped pardon after `J` buys nothing over declaring the readers.
  Before `J` it costs `W_cell + log2 (n_W n_RU)`, 96.6 bits per affected
  request at the headline, about `5e6` bits for a pod-hour of `5e4` requests
  against 52 for the run-wide one.
- *RU-scope pardon* (the RU's whole interior exempt): `W_R + log2 n_RU` per
  RU before `J`, 337 bits per step on the fixture (24.3 M for the pod-hour's
  72,000 steps), 8,237 at the headline (`4e8` for a pod-hour of requests);
  the pod's outputs are then unverified. After `J` the note bounds one pardon
  at `0.47 U_0` and prohibits the kind.
- *A public per-pod configuration* (M2/M8): a per-pod weight root or gate set
  in the constructor costs nothing in `Bound` because the version and the
  architecture are public objects, and the verifier checks the pod's RUs
  against them. A late lowering costs one bit per RU (`1.55 U_0` at the
  headline for one bit per request) and is admissible only under the note's
  conditions.
- *Re-serve*: recompute the pod-hour on a healthy pod before the round closes;
  `1 / n_pods` of the fleet-hour in the fixture, a pod-hour in a fleet
  otherwise, and no declaration.

**The fleet-scale comparison, on message entropy.** The outputs of the
pardoned readers are a function of `(w, v')` given honest inputs, so what a
source-position pardon can carry is the entropy of its message, about 52 bits
at the headline, not the cut of the readers. Take a pod-hour on an 8-GPU pod
serving the 70B shape at a production decode rate of order `1.5e4` tokens per
second per node (`docs/notes/datacenter-realities.md`, DeepSeek): about `5e7`
positions, `5e7` readers of any layer cell, `5e4` requests. At `q = 1.57e-8`
the challenge opens one of them with probability about `8e-4`; when it does,
M6 needs 1024 declarations under `VU_OUTPUTS` (or `512 phi` under
`BOUNDARY`) against `f_max = 2`, and would charge `1024 x 6.12e9 = 6.3e12`
bits, `0.33 U_0`, if it could admit them; the run-wide source-position pardon
says the same thing in 52 bits, `1.2e11` times less, and says it for the
`4.99e4` requests that were not opened as well. The comparison holds only
for a pardon fixed before `J`: after `J` the note's established price for the
pardon is the same `0.33 U_0` as M6 (no saving), and the message price rests
on the forced-consistency argument listed under open questions in the design
note.

**What this says.** Systematic faults are outside M6's competence at every
scale: their declaration count is a wiring count or an output count, both far
above `f_max`, under either recording policy; `BOUNDARY` recording lowers the
count by the flip fraction `phi` and by the words-per-VU ratio, not by orders
of magnitude on the toy, and whether it does at 70B is the open measurement
above. The honest prover as built re-serves the pod-hour or is rejected. Every
cheap alternative is a statement about a public object worth tens of bits (a
cell's value, a version, an architecture, a round boundary), and every one of
them is cheap only before `J`, which is why the detection signals of section 7
(ECC scrub logs, rollout logs, kernel-path logs) are worth `0.33 U_0` per
opened RU here. Diagnosis is free when the signal names the object; without
it the prover replays its opened RUs (which it does anyway), and under
`BOUNDARY` recording it must re-execute one opened RU on the faulty pod with
VU logging to find which cell to name.

## 7. Detection before the challenge

*Hardware signals, pre-streaming value checks with S7 truncation, partial
re-execution on idle capacity; the expected charge per fault against the cost.*

## 8. Round-close logistics

*Crashes, stragglers, partitions, requests longer than a round; what the epoch
layer lacks.*

## 9. Recording policies

*Tokens only, KV boundary, all VU outputs, kernel-path logs: which strategies
and declaration kinds each policy enables and what it costs.*

## 10. Row identifiers

Rows are the `H` section of the stress catalogue, recorded through the
`honest` fixture of `tests/veritor/stress/conftest.py` into
`docs/data/stress-honest.json`: `H2*` fault classes (section 4), `H3*` random
SDC at fleet scale (section 5), `H4*` systematic faults (section 6), `H5*`
detection (section 7), `H6*` round-close logistics (section 8). Each row
carries `declarations`, `charge_bits` and `recompute` besides the catalogue's
fields. An `H4` row's `declarations` is the P1 count under `BOUNDARY`
recording for the pod-hour; it also carries `declarations_vu_outputs` (the
same under `VU_OUTPUTS`), `faulty_boundary` and `faulty_vu_outputs` (the
pinned words and VUs of the pod-hour before the `q`-cut), `toy_boundary` and
`toy_vu_outputs` (the pins measured on the fixture's pod) and
`m6_bits_boundary`, `m6_bits_vu_outputs` (the M6 charge under each policy at
`u_post(1)`); `charge_bits` is the cheapest mechanism's.

## Results

Generated by `python -m veritor.stress.report` from `docs/data/stress-honest.json`; declarations are what the honest prover had to declare for the run to be accepted, the charge is what those declarations add to `U` under the price the mechanism's stage admits (`docs/notes/late-advice.md`), recompute is the fraction of the production computation the prover re-executed to know what to declare, `U` is `Bound` at `eta = 2^-40` including the charge.

| ID | What happened | Mechanism | Declarations | Charge (bits) | Recompute | U (λ = 40) | Verdict |
|---|---|---|---|---|---|---|---|
| H4a | corrupted weight cell on one pod, one hour | source-position pardon (pod scope) | 56,000 | 26 | 0.25 | 282 | one pardon, 26 bits, pre-J from the ECC log (post-J at the same price only under the forced-consistency argument, open); P1 needs 56,000 declarations under BOUNDARY recording or 72,000 under VU_OUTPUTS against f_max = 1: rejected |
| H4b | stale weight version on one pod, one hour | per-pod public weight root (M2/M8) | 632,000 | 0 | 0 | 256 | per-pod kappa_W in the constructor: 0 bits; else re-serve the pod-hour; P1 needs 632,000 (BOUNDARY) or 2,752,000 (VU_OUTPUTS) declarations: rejected |
| H4c | wrong kernel path on one pod, one hour | per-pod architecture (M8) | 632,000 | 0 | 0 | 256 | per-pod gate set in the constructor (ClusterG arches): 0 bits; a late lowering would cost 72,000 bits (1 per RU) and only under the note's conditions; P1 needs 632,000 (BOUNDARY) or 2,752,000 (VU_OUTPUTS) declarations: rejected |
| H4d | fleet-wide weight update mid-round | round close at the update (M2) | 614,445 | 0 | 0 | 256 | close the round and start a run under the new kappa_W: 0 bits; else re-serve the half-round; P1 needs 614,445 (BOUNDARY) or 2,752,000 (VU_OUTPUTS) declarations: rejected |

Notes:

- **H4a**: a layer matrix cell is read by 1 VU per position; measured on pod 0 of the fixture (4 steps, 9 positions, 526 VUs, 79 recorded words, 7 tokens), pinned VUs BOUNDARY/VU_OUTPUTS per cell: low bit w_q 7/9, w_k 9/9, w_v 9/9, w_o 4/9, w_1 7/9, w_2 6/9; top bit w_q 0/3, w_k 3/3, w_v 3/3, w_o 2/3, w_1 0/6, w_2 2/3 (the toy's word is modular, so the top bit of a weight is silent for an even activation; W_k and W_v pin their readers under both policies because a step RU commits its KV entries; the others pin the tokens that flipped, between 0 and 1 of the pod's tokens on an 8-token vocabulary); a constant-table cell (16 readers on the pod) pins 17/2 low bit and 19/3 top bit (few readers move, early in the position, so the outputs' cut exceeds the readers' cut); the row scales W_q low bit: 144,000 readers and 112,000 flipped tokens in the pod-hour, 72,000 / 56,000 in opened RUs at q = 1/2; M6 would charge 10,266,597 / 7,985,131 bits at u_post(1) = 142.6; an RU-scoped source pardon per opened RU costs the same as declaring the RU's readers (10,266,597 bits); RU-scope pardons 24,273,772 bits pre-J and prohibited post-J; headline f_max for 16,384 devices is 2; diagnosis: the ECC scrub log names the cell, else replay the pod's opened RUs (q / pods of the run) and solve one opened VU for the cell
- **H4b**: 192 of 201 cells differ between the versions; measured on pod 0: every recorded word is pinned under BOUNDARY (79 of 79), every dot under VU_OUTPUTS (344 of 526 VUs); pod-hour 1,264,000 words and 5,504,000 dots; one run-wide source-position pardon per changed cell would cost 4,933 bits here and scales with \|W\| (GPT-2 Small 124,490,068 cells); the version is a public object known from the rollout log, so the constructor can carry a per-pod root and the verifier checks the pod's RUs against it
- **H4c**: every dot of the pod (344 of 526 VUs) stores a perturbed word; measured on pod 0: every recorded word pinned under BOUNDARY (79), every dot under VU_OUTPUTS (344); pod-hour 1,264,000 words and 5,504,000 dots; a source-position pardon cannot express a relation change; diagnosis without a kernel-path log: replay one opened RU under each candidate lowering
- **H4d**: measured with the update at step 3 of 6: RUs (7, 8, 9, 10) read the new version, 35 of their 36 recorded words are pinned under BOUNDARY (a token can agree with the honest run by chance on an 8-token vocabulary), every dot (160 of 240 VUs) under VU_OUTPUTS, and nothing before the update; the half-round after the update on 2 pods is 1,228,889 words and 5,504,000 dots; the header binds one kappa_W per run, so the update is a run boundary, not a pardon; RU-scope pardons for the half-round would cost 24,273,772 bits pre-J
