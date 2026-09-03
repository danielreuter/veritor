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

The protocol's `ProverSession` was written as if the server kept every gate
of the production run (`assignment_replay`) and found its faults by
`self_check` over that assignment. A real server keeps the boundary it has to
commit plus whatever it chooses to log, and reconstructs an opened RU's
interior when the q-challenge asks for it. `veritor.simulation.honest` models
that server; `tests/veritor/stress/test_honest_replay.py` records its rows.

*Recording policies.* `RecordingPolicy` names what the server keeps of a run
(`record` restricts the omniscient assignment to it). `BOUNDARY` is the
inputs, the weights and every RU's declared outputs `Out(R)`: what the
boundary commitment and the weight tree need and nothing more. With RU =
request (`RequestsG`) the recorded computed values of a run are its streamed
tokens, 7 of the toy run's 3,943 computed gates (0.18%); with RU = step
(`ClusterG`) the KV words and tokens that cross steps as well, 79 (2.0%).
`VU_OUTPUTS` is the boundary plus every VU's declared output word, the
interior positions of every RU: 603 (15.3%), the server logging every
kernel's output. A VU's internal gates are recorded under no policy; they are
never committed and never read from outside the VU. Kernel-path logs and
other policies are section 9's.

*Pinned replay.* `replay_pinned` is honest replay from a recording. The gates
of RU `R` are recomputed in address order from the recorded inputs, weights
and boundary, as `replay_unit` does, with one rule. At an address the server
*recorded*, the recorded value is pinned: it is the value the interior
commits at that address and the value every later gate of the replay reads
there; the recomputed value is compared with it and, when the two differ, the
VU owning the address is added to the pinned set. At an address the server
did not record, the recomputed value is stored and read downstream. The
interior this yields satisfies every VU's relation except at the pinned VUs,
whose relations fail against their own inputs; a reader of a pinned value was
recomputed from that value, so its relation holds against it. The pinned VUs
are therefore exactly what the server must declare (M6) for the run to be
accepted whatever the s-challenge samples, and `self_check` over the
committed interior finds the same set (asserted for every H1 fault under both
policies and both RU choices). The replay consults nothing the policy did
not record: the recording is its only source of values, and a `BOUNDARY`
recording alone, 5% of the toy assignment, drives `ProverSession` through
`HonestReplay` to an accepted transcript with every RU opened.

Two consequences shape every declaration count below. A fault that changed no
recorded value costs no declaration: the replay recomputes the correct
interior, it agrees with the recording, and the committed interior is the
correct one; the run's tokens were right. A fault that changed recorded values
costs one declaration per recorded value it changed, whether or not the VU
that produced that value is the one that faulted: with tokens-only recording
the declarations name the tokens that came out wrong, not the kernel that went
wrong, and a wrong token is a declared VU whose downstream is then recomputed
from the wrong token as the server did. The declarations are the recorded
values at the frontier of the fault's cone: recording the corrupted word
itself pins the cone at its root, one declaration; recording only the
boundary pins it where it crosses the boundary, its cascade; and for a read
fault, whose wrong copy is stored nowhere, recording the consumers' outputs
pins every consumer. Which policy declares less is a property of the fault,
which is section 9's trade.

*One fault of each class (H1).* Rows H1a-h inject one fault of each of
section 4's classes into the toy's production run
(`veritor.simulation.faults.FaultInjector`: stored corruptions are flips of
an output word after it was computed, read faults are `misreads`, the stored
word and the commitments right while the readers computed from a wrong copy),
record under both policies, pin every RU and run the tokens-only prover
through the protocol with every RU opened (`theta = (1, 1/8)`, so the
declarations are the run's whole count) and a budget of exactly its pins; the
`c` rows also run it at one less and are refused with `FAULTS_EXCEEDED`.
Declarations under `BOUNDARY` / `VU_OUTPUTS`, and the streamed tokens (of 7)
the fault changed:

| Class | RU = request | Tokens | RU = step | Tokens |
|---|---|---|---|---|
| a. interior low bit (H1a) | 0 / 1 | 0 | 0 / 1 | 0 |
| b. interior high bit (H1b) | 0 / 1 | 0 | 3 / 1 | 0 |
| c. token flip (H1c) | 2 / 1 | 3 | 4 / 1 | 2 |
| d. catastrophic word (H1d) | 3 / 1 | 3 | 8 / 1 | 2 |
| e. weight cell read fault, 36 readers in 18 VUs (H1e) | 7 / 18 | 5 | 7 / 18 | 5 |
| f. input token read fault, 8 readers in 1 VU (H1f) | 2 / 1 | 3 | 8 / 1 | 3 |
| g. VU-output read fault, 13 readers in 13 VUs (H1g) | 0 / 5 | 0 | 3 / 5 | 0 |
| h. KV word at rest, 1 reader (H1h) | n/a | n/a | 1 / 1 | 2 |

A stored corruption (a-d) pins exactly the faulty VU under `VU_OUTPUTS`, one
declaration however far its consequences travel, because every downstream VU
is recomputed from the recorded wrong word it actually read. Under `BOUNDARY`
it pins the recorded values it changed: none when it reached no token (a, b
with RU = request), and its cascade when it did. The token flip of H1cr
changed three of seven tokens and pins two; the third wrong token is
recomputed from the two pinned ones and comes out as streamed. With RU = step
the KV words that cross steps are recorded too, so the same flips pin their
corrupted KV words as well (3, 4, 8 against 0, 2, 3), and the high-bit flip
that changed no token now costs three declarations. A read fault (e, f, g)
inverts the order: the stored word is right and every policy holds it, the
consumers computed from a wrong copy, and `VU_OUTPUTS` pins every consumer
VU whose recorded output disagrees with its recomputation from the right word
(all 18 for the most-read weight cell, 1 for the input token whose eight
readers are one embedding VU, 5 of the 13 for the VU-output read: the other
eight multiply the flipped top bit by an even weight, which annihilates it
modulo 2^16, and their output words are unchanged), where `BOUNDARY` pins
only what reached the recording (7, 2 or 8, 0 or 3). Under `BOUNDARY` a read
fault and a stored corruption of its consumers' outputs are the same
declarations; the recording cannot tell them apart. Neither policy needs
fewer declarations in general: tokens-only wins for silent faults and for
read faults with many consumers, every-VU-output wins for a stored corruption
that reached the tokens, and over the random flips of H3 (10 flips, 6 runs)
tokens-only pins 7 to every-VU-output's 10, because most flips are silent and
the ones that reach a token cascade.

Class h is the boundary at rest: a KV word committed right by the step that
produced it, read corrupted by the step that consumes it. Not every bit of a
word is significant for the gate that reads it. `significant_bits` flips
each bit of the word in the readers' copy and asks whether any value a
policy records changes. For the KV key word of H1hs bits 14 and 15 are inert:
the bit enters the attention score linearly through one `mul` and the toy's
polynomial softmax squares the score, and modulo 2^16 `(s + 2^15)^2 = s^2`,
as does `(s + 2^14)^2` for even `s`. A flip there is a fault no recording and
no replay can see under any policy, so the row flips bit 13, the most
significant live one, and pins one VU under either policy. Which bits are
inert is a property of the landing site, not of the fault: over the 34 words
step 1 reads from step 0, the inert set runs from nothing (most value words,
and the tokens) to bits 6-15 (two value words whose one reader multiplies
them by a softmax weight that is a multiple of 2^10), and the same top-bit
flip in a word read by an `add` alone is always live.

The prover's diagnosis cost is the replay itself: to know what to declare in
an opened RU it replays that RU, which the protocol requires of it anyway, so
the tokens-only prover's recompute is the opened share `q` of the production
replay cost and nothing extra; the H1 rows show 1 because every RU was
opened. Knowing what the *unopened* RUs would need costs replaying them, which
is what separates the strategies of section 5.

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
Toy numbers are the honest model's `H1` rows (two requests, `RequestsG` unless
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

Four strategies for a server whose production run holds silent faults it has
not noticed (`veritor.simulation.honest.Strategy`, priced by `account`). P0
declares nothing: the opened RUs are replayed pinned as the protocol requires,
and a pinned VU the s-challenge samples is a rejection. P1 is the protocol as
built: the opened RUs are replayed pinned after `J` and their pins declared
(M6), each at the post-J price `u_post(1) = rho log2(1 / (1 - s))`, the slope
of the fold times the threshold one adaptive declaration lowers
(`docs/notes/late-advice.md`; `(u(1) + 1) / q` at the scattered channel). P2
declares before `J`, at `u(1)`, the pins of the RUs a signal flagged before
streaming (a range check that catches a garbage word, a hardware counter;
section 7), replaying those RUs to know them, and the rest as P1. P3 replays
every RU before the boundary commitment and declares every pin at `u(1)`.
The protocol has no pre-J declaration message, so P2 and P3 are
counterfactual accounting: their charge is what a protocol with that message
would bill, their verdict the built protocol's under P1's declarations.
Every strategy's recompute is the share of the production replay cost it
re-executes: the opened share `q` for P0 and P1, which the protocol demands
whether or not anything is declared; `q` plus the flagged RUs for P2; 1 for
P3, the production run twice. A post-J declaration is what the header's
budget `f_max` admits and `bound(..., max_faults = f_max)` prices, whether or
not the budget is used; the rows below price P1's declarations one by one on
a single run (H2) and its round budget on a fleet (H3, the phase diagram).

*One run, four faults (H2).* Four requests under RU = request, one fault
each: a token flip in RU 1, a catastrophic word in RU 2, a low-bit interior
flip in RU 3, a misread prompt token in RU 4. Tokens-only recording pins 8
VUs (2, 4, 0, 2: the low-bit flip changed no token; every-VU-output recording
would pin 4, one per fault). `u(1) = 74.2` bits, `u_post(1) = 132.4` bits at
`theta = (1/2, 1/8)` (`rho = 687`). P0 and P1 run under one header with
`f_max = 8`, every pin of the run, because the header enters the derivation
of `J`: one header, one challenge, and the two strategies differ only in
what the prover declares. The first seed at which P0 is rejected opens the
request RUs 2 and 3.

| Row | Strategy | Declarations | Charge (bits) | Recompute | Verdict |
|---|---|---|---|---|---|
| H2a | P0 | 0 | 0 | 0.49 | RELATION_REJECTED: a pinned VU of RU 2 sampled |
| H2b | P1 | 4 post-J | 530 | 0.49 | ACCEPTED; the 4 pins of the unopened RUs 1 and 4 never declared |
| H2c | P2, RU 2 flagged | 4 pre-J, 0 post-J | 297 | 0.49 | ACCEPTED (under P1's declarations) |
| H2d | P3 | 8 pre-J | 594 | 1 | ACCEPTED (under P1's declarations) |

P1 undercuts P3 here, 530 to 594 bits, because it declares only the opened
half of the pins and `u_post(1) / u(1) = 1.78` is below `1 / q = 2`: P1 costs
`q u_post(1)` per pin in expectation against P3's `u(1)`. At the headline
policy `u_post(1) = (u(1) + 1) / q` exceeds `u(1) / q` and the order reverses,
and reverses harder once P1 pays a budget rather than its declarations. P2's
saving is the fraction of the pins a signal finds before streaming, priced at
`u(1)` in place of `q u_post(1)`; its recompute is the flagged RUs' share on
top of `q` (here RU 2 was opened anyway). The toy's fold is capped at the
run's interface, `|Out| = 192` bits, at every `f_max`, so `U` reads 192 for
all four rows and the charge column carries what `U` would lose; the prices
are the closed-form slope's, which the last paragraph of this section checks
against the fold on the headline table.

*A fleet through the epoch layer (H3).* Nine runs of the two-request
workload in three rounds of three through `run_epoch` at `theta = (1/2,
1/8)`, each run taking Poisson(1) bit flips at random dot words and bits, a
billion times the Llama-3 rate for a run this size so that faults occur at
all in a fleet we can afford to run; the mechanism, not the rate, is what the
rows check. The first fleet seed at which the silent prover loses a round has
10 flips in 6 runs, 3 of which changed a streamed token; tokens-only recording
pins 7 VUs over the fleet, every-VU-output recording 10. The round budget is
`fault_budget(q x pins / rounds, tail 1e-3)`, carried by every header of the
round and charged once per round at `u_post(1) = 135.7` bits (`rho = 704` for
the round's union of three tables at `eta / 3`; `u(1) = 75.1`).

| Row | Strategy, recording | `f_max` | Declared | Charge (bits, epoch) | Recompute | Verdict |
|---|---|---|---|---|---|---|
| H3a | P0 | 0 | 0 of 7 | 0 | 0.55 | epoch RELATION_REJECTED: 1 of 9 runs, in round 1 |
| H3b | P1, tokens only | 6 | 4 of 7 | 2,443 (3 x 6 x 135.7) | 0.44 | epoch ACCEPTED, rounds declared 2, 2, 0 |
| H3c | P1, every VU output | 7 | 5 of 10 | 2,850 (3 x 7 x 135.7) | 0.65 | epoch ACCEPTED, rounds declared 1, 2, 2 |

One undeclared pinned VU, opened and sampled, rejects its run and with it the
epoch (`EpochVerifier` accepts only if every committed run was accepted). The
declaring prover is accepted under either policy; it declares the pins of the
RUs the rounds opened, about half of the fleet's, and pays for the budget
that covers a round with probability `1 - 1e-3`: six per round against the
2, 2 and 0 it used. Every-VU-output recording pins more here (10 to 7) and so
needs the larger budget; the two headers differ, hence the two `J` and the
two recompute shares.

*The phase diagram.* At fleet scale the counts are Poisson and the question
is analytic (`phase_diagram`, asserted in
`test_phase_diagram_at_the_headline`; nothing of size is simulated). Let `D`
be the mean number of pinned VUs per round had every RU been opened. Each
lies in an opened RU with probability `q`, so P1 declares Poisson(`q D`) per
round and carries the budget `f_max = fault_budget(q D, 1e-6)`, which is 1
unless `q D` exceeds about `sqrt(2e-6) = 1.4e-3`; its charge is `f_max
u_post(1)` per round, paid whether or not a fault occurred; the round exceeds
the budget with probability `P[Poisson(q D) > f_max]`. P0 is rejected when a
pinned VU is opened and sampled, probability `1 - exp(-q s D)`. P3 pardons
every fault before `J`, `fault_budget(D, 1e-6)` of them at `u(1)`, and
re-executes the round. The headline operating point of
`docs/global-estimate.md` is `q = 1.57e-8`, `s = 8.9e-3`, `rho = 4.74e11`,
`U_0 = rho lambda + log2 e = 1.90e13` bits at `lambda = 40`, `u(1) = 94.7`
bits and `u_post(1) = 6.12e9` bits, `3.2e-4` of `U_0`; the alternates are
ten times `q` (`U_0 = 1.90e12`) and ten times `s` (`U_0 = 1.46e13`,
`u_post(1) = 4.9e10`). Densities are faults per round, labelled by the
fleet-window that produces them at the Llama-3 rate (`2.8e-7` per
device-hour), each fault taken to pin one VU.

| Faults per round `D` | `q D` | `f_max` | `P[> f_max]` | P1 charge, bits (share of `U_0`) | P0 `P[rejected]` per round | P3 budget, charge (bits) |
|---|---|---|---|---|---|---|
| 16,384 GPUs for an hour, 4.6e-3 | 7.3e-11 | 1 | 2.6e-21 | 6.1e9 (3.2e-4) | 6.5e-13 | 2, 189 |
| 16,384 GPUs for a day, 0.11 | 1.7e-9 | 1 | 1.5e-18 | 6.1e9 (3.2e-4) | 1.6e-11 | 4, 379 |
| a million GPUs for a day, 6.8 | 1.1e-7 | 1 | 5.7e-15 | 6.1e9 (3.2e-4) | 9.5e-10 | 22, 2.1e3 |
| a million GPUs for a year, 2.5e3 | 3.9e-5 | 1 | 7.6e-10 | 6.1e9 (3.2e-4) | 3.5e-7 | 2,715, 2.6e5 |
| a billion GPUs for a year, 2.5e6 | 0.039 | 3 | 9.2e-8 | 1.8e10 (9.7e-4) | 3.5e-4 | 2.5e6, 2.4e8 |

The two alternates at the two largest densities:

| Policy | `D` | `q D` | `f_max` | `P[> f_max]` | P1 share of `U_0` | P0 `P[rejected]` |
|---|---|---|---|---|---|---|
| `q x 10` | 2.5e3 | 3.9e-4 | 1 | 7.6e-8 | 3.2e-4 | 3.5e-6 |
| `q x 10` | 2.5e6 | 0.39 | 6 | 1.9e-7 | 1.9e-3 | 3.5e-3 |
| `s x 10` | 2.5e3 | 3.9e-5 | 1 | 7.6e-10 | 3.4e-3 | 3.5e-6 |
| `s x 10` | 2.5e6 | 0.039 | 3 | 9.2e-8 | 1.0e-2 | 3.5e-3 |

Three things follow. First, at every realistic density the post-J price is
the floor: one declaration's budget, `u_post(1) / U_0 = log2(1 / (1 - s)) /
lambda` to within `log2 e / U_0`, `3.2e-4` of `U_0` per round at the
headline `s`, ten times that at ten times `s`, the same at ten times `q`.
It is the price of admitting M6 at all, paid for the possibility of a fault,
and no signal (P2) lowers it: removing the flagged fraction of the faults
from the post-J stream leaves `f_max = 1`. The floor is left only past `q D
= 1.4e-3`, `9.0e4` faults per round at the headline `q`, and P1's charge
reaches 1% of `U_0` (31 declarations) only near `7e8` faults per round.
Second, P3 undercuts P1 by four orders of magnitude in bits wherever the
floor holds (189 to 2.6e5 bits against 6.1e9) at the cost of re-executing
production in full, `1 / q = 6e7` times P1's recompute at the headline; its
charge grows with `D` and would meet the floor near `6e7` faults per round.
Third, where P0 beats P1: P0 loses, in expectation, `P[rejected]` times what a
rejection forfeits, `L` rounds' capacity, against P1's share, so at the floor
P0 wins when `(1 - exp(-q s D)) L < log2(1 / (1 - s)) / lambda`, which for
small arguments is `q D L < log2(e) / lambda = 0.036`; `s` cancels, and the
`s x 10` rows return the headline's verdicts. At the headline `q` that is
`D L < 2.3e6`. If a rejected round is retried alone (`L = 1`) P0 wins at every
density in the table, the last row included since `f_max = 3` triples P1's
charge there. Under the epoch layer as built one rejected run rejects the
epoch, so `L` is the epoch's rounds: at `L = 1000` P0 still wins up to a
million GPUs for a day and loses from a million GPUs for a year on (`q D L =
0.039`). In expected certified bits, then, a server at the headline density
does better declaring nothing and retrying the rare rejected round than
carrying the one-declaration budget every round; the case for M6 is
completeness, that an honest server is never rejected, and rests on what a
rejection costs beyond the round's bits.

*Prices.* The phase diagram and the rows price a post-J declaration with the
closed-form slope, `rho` from `veritor.analysis.rate` as the headline
estimate does, because the fold of `bound` is loose at the headline policy:
`bound(table, theta, 2^-40)` on the headline table is `2.4e17` bits, within
0.1% of the output interface and `12,600` times the closed form's `U_0`, and
its price of one declaration, `bound(max_faults = 1) - bound()`, is `7.5e10`
bits, 12 times the closed form's `6.1e9`
(`test_fold_and_closed_form_price_the_headline_declaration`). On the toy the
fold is capped at the run's outputs at every `f_max`, so no toy row can read
its charge off `U`; the charge column is `rho log2(1 / (1 - s))` per
declaration with the fold's own `rho` for the toy tables.

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

M6 lets the prover declare a faulty VU after `J` is revealed, at the adaptive
price `u_post(1)` that `bound(..., max_faults=f)` charges. A prover that found
the fault before `J` could declare it at the fixed price `u(1)` (the protocol
has no such message; this section prices the extension), or, if it found it
before the garbage was streamed, end the request there and pay the truncation's
advice (S7). This section asks what the honest prover can know before `J`
without algebraic checks (Freivalds-style checks are out of scope: production
matmuls are floating point), what knowing it costs, and what it saves. The
result first. At the headline operating point the expected charge of an
undetected fault, `q u_post(1)`, equals the pre-J price `u(1)` within 1.5%, and
this is an identity of the scattered channel rather than a coincidence of the
operating point. Detection before the challenge therefore buys no expected
capacity: at `p = 1` it saves 1.4 bits per fault and no `p` halves the charge.
What it buys is a charge that is certain instead of a lottery, and headroom
under the per-round cap `f_max`. The module is
`src/veritor/simulation/detection.py`; every number below is pinned in
`tests/veritor/stress/test_honest_detection.py`.

### 7.1 The fault mixture and the detector menu

The fault mixture is the GPU part of Llama-3 405B pre-training's unexpected
interruptions ([Dubey et al. 2024](https://arxiv.org/abs/2407.21783), table 5:
54 days on 16,384 H100s, 419 unexpected interruptions, of which 148 faulty GPU,
72 HBM3 memory, 19 SRAM memory, 17 GPU system processor, 6 thermal interface or
sensor and 6 silent data corruption are the 268 GPU-category events;
`docs/notes/datacenter-realities.md`, section 7). Two classes. VISIBLE, 262 of
268 (97.8%): faults the hardware reports as they happen (ECC events, Xid errors,
a device or NVLink failure, a crash); the request stops and nothing wrong is
streamed. SILENT, 6 of 268 (2.2%): corruption of the compute path that reaches
the streamed tokens with no signal ([Dixit et al.
2021](https://arxiv.org/abs/2102.11245), [Hochschild et al.
2021](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s01-hochschild.pdf)),
the class M6 exists for. The mixture is a training fleet's; an inference
fleet's is assumed to be like it.

A detector is its coverage of each class, its cost as a fraction of the serving
computation, and what happens when it fires:

| Detector | Of VISIBLE | Of SILENT | Of all faults | Cost | Fires | Outcome |
|---|---|---|---|---|---|---|
| Hardware signals: ECC counters, Xid errors, watchdogs, crashes | 100% | 0 | 97.8% | 0 | as the fault happens | the request ends: truncation (S7) |
| Pre-streaming value check: NaN/Inf, range, degenerate logits on the activations and logits | 0 | 50% (assumption) | 1.1%; 98.9% together with the signals | `7.0e-6` of the compute on the 70B shape | before the token is streamed | truncation (S7) |
| Partial re-execution of a fraction `p` of the requests on idle capacity before round close, compared bit for bit | `p` | `p` | `p` | `p` | after the tokens are out, before `J` | declaration at `u(1)` |

The value check is an elementwise pass over the words a token produces: on the
70B shape of `docs/global-estimate.md` the logits and every layer's residual are
688,128 words per token, at two cost units per word against the request's
`ru_cost / tokens`, `7.0e-6` of the serving computation. Its coverage is an
assumption, taken as the top half of the word (`ValueCheck.sees(bit, width)`):
[Li et al. 2017](https://doi.org/10.1145/3126908.3126964) find that in
floating-point DNNs only flips of the high-order exponent bits cause silent
data corruption, mantissa and sign flips being benign, and that a range check
on the activations detects most of the SDC-causing ones; [Chen et al.
2021](https://arxiv.org/abs/2003.13874) (Ranger) cut SDC rates 3x to 50x by
range restriction at negligible overhead. The rows inject bit 12 of a 16-bit
word, which is in that class; the toy ISA has no NaN, so the coverage is
asserted by the bit position, not measured. Coverage composes as `1 - prod (1
- coverage)`: signals and check see 98.9% of all faults and 50% of the silent
ones; adding re-execution at `p` takes the silent coverage to `0.5 + 0.5 p`.

### 7.2 The two prices and the conservation law

`u(1) = W_V + log2 |S|` is the price of a declaration fixed before the
challenges (`docs/stress-tests.md`, M6). `u_post(1)` is the marginal charge of
one post-J declaration: the smaller of the two bounds `declared_bits` takes,
less the capacity without declarations, from the fold where it can be computed
(`fold_prices`) and from the closed-form rate as `rho log2 (1 / (1 - s))`
where it cannot (`analytic_prices`; `docs/stress-tests.md` quotes this form
for the headline). A fault left to M6 is declared, and charged, only when its
RU is opened, so its expected charge is `q u_post(1)`. Pinned values:

| | Toy shape, 1024 requests, `theta = (1/2, 1/8)`, fold | Same table, closed form | Headline, `q = 1.57e-8`, `s = 8.91e-3`, closed form |
|---|---|---|---|
| `u(1)` | 83.2 | 83.2 | 94.7 (`W_V = 16`, `log2 \|S\| = 78.7`) |
| `rho` | 830 | 904 | `4.74e11` |
| `u_post(1)` | 141.2 | 174.2 | `6.12e9` |
| `u_post(1) / u(1)` | 1.70 | 2.09 | `6.46e7` |
| `q u_post(1) / u(1)` | 0.848 | 1.047 | 1.015 |

Why the ratio is one. Where the scattered channel sets `rho`
(`src/veritor/analysis/rate.py`, (2) at `l = 1`; it binds at the headline and
on the 1024-request table), `rho = (W_V + log2 R + log2 m + 1) / log2 (1 / (1
- q s))`, and `W_V + log2 R + log2 m = u(1)`. So

`q u_post(1) = q rho log2 (1 / (1 - s)) = (u(1) + 1) q log2 (1 / (1 - s)) / log2 (1 / (1 - q s)) = (u(1) + 1) (1 + s / 2 + ...)`

which is `1.0106 x 1.0045 = 1.0151` at the headline, the tests pinning each
factor. In words: a post-J pardon hands the adaptive adversary one scattered
error's worth of acceptance probability, `s`, at the scattered channel's
exchange rate `rho`, which is by construction `u(1) + 1` bits per `log2 (1 /
(1 - q s))` of acceptance; the factors of `q` cancel. The excess over `u(1)`
is the 1 bit of the `log2 (l (l + 1))` term and `s / 2`. Where the fold is
computable it sits below the closed form by (i) of `rate.py` (about `log2 B`
bits per error where many errors are affordable): 0.81 of it on the
1024-request table, hence the ratio 0.848 there, where detection *loses* 12.6
bits per fault. At the headline the fold is out of reach and the closed form
is what M6 quotes; `rate.py` measures the closed form 17% to 27% above the fold
at small `s`, which would take the ratio to 0.80 to 0.87. Either way the
expected charge moves by a bit or two per fault, in one direction or the other.

The frontier. With a fraction `p` of the faults detected before `J` and
declared at `u(1)`, the expected charge per fault is `p u(1) + (1 - p) q
u_post(1) = 96.1 - 1.43 p` bits at the headline: 1.43 bits saved per fault at
`p = 1`, 0.72 at `p = 1/2`, against `p` of the serving computation for the
re-execution that finds them. No `p` halves the charge; that needs `q
u_post(1) >= 2 u(1)`, twice the conservation law.

Truncation. S7 as built (`TruncatedRequestsG`) charges `ceil(log2 max_new)`
bits for every request, truncated or not: 18 bits on the six-request run,
3,072 on the 1024-request table (against `u_post(1) = 141`), and `9 x 2.93e13
= 2.6e14` bits at the headline, 13.9 times `U_0 = 1.9e13`. That is a
constructor for a run in which every request may stop early, not a price for a
fault; at scale it is unusable. The information a truncation carries is which
requests stopped and where, `log2 C(R, k) + k log2 max_new`: 53.7 bits for one
of the year's `2.93e13` requests (`max_new = 512`), 47.9 per fault for 154 of
them, below `u(1) = 94.7` and about half of `q u_post(1)`. No constructor in
the repository charges that; it is the floor a sparse length advice would
reach, and the client loses the request's tail.

No action. An undeclared fault is a rejected round iff its RU is opened and its
VU sampled: `q s = 1.40e-10` per fault, `2.2e-8` per round at the headline's
154 silent faults. The charge is not bits but the round; section 5's P0 is the
comparison.

### 7.3 What detection buys: a certain charge, and headroom under `f_max`

Variance. Undetected, a fault's charge is 0 with probability `1 - q` and
`6.12e9` bits (0.032% of `U_0`) with probability `q`: a standard deviation of
`7.67e5` bits per fault against a mean of 96.1; `5.42e5` at `p = 1/2`, 0 at `p
= 1`. Per round at the Llama-3 rate the headline circuit (a year of `2.93e13`
requests, `5.45e8` device-hours or 62,000 GPU-years at `1e15` MAC/s, `2.83e-7`
SDC per device-hour) carries 154 silent faults, `q x 154 = 2.4e-6` of them
opened on average: the round pays `6.12e9` bits with probability `2.4e-6` and
nothing otherwise, or, at `p = 1`, `154 u(1) = 1.46e4` bits with certainty
(expected undetected: `1.48e4`).

Headroom. A round is rejected (`FAULTS_EXCEEDED`) when more than `f_max` faults
are declared, and only opened faults can be declared; with a fraction `p`
detected and handled before `J`, the opened undetected faults are Poisson with
mean `q mu (1 - p)` for `mu` silent faults per round. `P[Poisson(q mu (1 - p))
> f_max]` at the headline `q`:

| `mu` faults per round | `q mu` | `f_max` | `p = 0` | `p = 1/2` | `p = 0.9` | `p = 0.99` |
|---|---|---|---|---|---|---|
| 154 (the Llama-3 rate) | `2.4e-6` | 0 | `2.4e-6` | `1.2e-6` | `2.4e-7` | `2.4e-8` |
| 154 | `2.4e-6` | 4 | `6.9e-31` | `2.2e-32` | `6.9e-36` | `6.9e-41` |
| `1e6` | 0.016 | 0 | 0.016 | 0.0078 | 0.0016 | `1.6e-4` |
| `1e6` | 0.016 | 4 | `7.9e-12` | `2.5e-13` | `8.0e-17` | `8.0e-22` |
| `1e8` | 1.57 | 0 | 0.79 | 0.54 | 0.145 | 0.016 |
| `1e8` | 1.57 | 4 | 0.022 | 0.0013 | `7.0e-7` | `7.9e-12` |
| `1e9` | 15.7 | 4 | 0.999 | 0.89 | 0.022 | `7.0e-7` |

The cap sees only `q mu (1 - p)`, so a row at `1e9` and `p = 0.99` is the row
at `1e8` and `p = 0.9`. At the Llama-3 rate there is no headroom problem:
`f_max = 1` already has a tail of `2.9e-12` and `f_max = 4`, the day's budget
of M6, one of `6.9e-31`. Detection matters once `q mu` is of the order of
`f_max`, which at the headline is `1e8` faults per round, `6.5e5` times the
Llama-3 rate or 0.18 faults per device-hour: not random SDC but a broken fleet.
The detection that holds the tail below `1e-6` under `f_max = 4` is `q mu (1 -
p) <= 0.169`, `p >= 0.89` at `1e8` faults per round and `0.99` at `1e9`. A
mercurial core per thousand devices (Meta 2025,
`docs/notes/datacenter-realities.md`, section 7) corrupts every request it
serves: `2.9e10` faulty requests per round, 460 of them opened, and no cap
holds without `p > 0.9996`.
That is a device to find and remove (section 6), not a fault to declare. At
the simulation policy `q = 1/2` the same arithmetic bites at a handful of
faults per round: 3 faults under `f_max = 1` reject 44% of rounds undetected
and 1.0% at `p = 0.9`.

### 7.4 The rows

The fault is bit 12 of the output word of a step-1 projection dot of the first
of six requests (prompts of two to four tokens, `max_new = 8`), a flip in the
top half of the 16-bit word: 14 of the 16 step-1 embedding and projection dots
of that request change a streamed token when their bit 12 flips, and this one
changes 6 of the 7 downstream tokens, which are streamed. The server holds the
values, so replaying the RU finds exactly one VU that disagrees (`self_check`);
under a tokens-only recording policy it would have to pin up to 7 tokens
instead (section 9). Prices are read from the toy shape's 1024-request serving
table (`serving_table(..., "request", "row")`, fold uncapped) and from the
headline's closed form, because the six-request run's own fold is saturated:
its `U` is `|Out| = 768` bits with or without the declaration (`|S| = 3711`,
`u(1) = 75.9`, uncapped 801), so its own marginal is invisible.

- **H5a**, no action or a declaration after `J`. At `theta = (1, 1)` the
  undeclared fault is `RELATION_REJECTED` at VU 367. At `theta = (1/2, 1/8)`,
  `f_max = 1`, under eight fixed q-challenges, 2 open RU 1: the server
  self-checks it, declares the one VU, ACCEPTED, charged `u_post(1) = 141.2`
  bits on the 1024-request table; 6 do not open it: nothing is declared or
  charged, ACCEPTED. The charge is a lottery at `q`.
- **H5b**, caught before the token was streamed. A hardware signal or a value
  check on step 1's activations fires; the request ends after the one token
  the prefill produced, its 7 absent slots are blank check outputs
  (`TruncatedRequestsG`, lengths `(1, 8, 8, 8, 8, 8)`): 18 advice bits,
  ACCEPTED with no declaration at `theta = (1/2, 1/8)`, `U = 656 + 18` (the
  faulty step is not in the circuit; the truncated request becomes a kind of
  its own and the other five requests' tokens are unchanged).
- **H5c**, found before `J` by re-executing the request before the round
  closes. The first disagreeing gate is the flipped word, its VU the
  declaration; priced, not run, since the protocol takes declarations only
  after `J`: `u(1) = 83.2` bits on the 1024-request table (75.9 on this run:
  `Bound` at `theta = (1, 1)` goes 0 to 75.9), against `p = 1` of the serving
  computation.

### 7.5 What this section does not settle

The pre-J declaration is priced, not run: the protocol has no message for it
and adding one is a wire-format change. The value check's coverage of the
silent class is an assumption from the bit-flip literature, not a measurement
on the toy ISA, which has no NaN. S7's advice is dense; the sparse floor is a
number, not a constructor. The fault mixture is a training fleet's and counts
interruptions, not faults; the silent rate `2.8e-7` per device-hour is the
same source's 6 events.

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
`docs/data/stress-honest.json`: `H1*` fault classes, the per-class
declaration counts under the two recording policies (section 3: `H1a`-`H1h`
one fault of each of section 4's classes, `a` interior low bit, `b` interior
high bit, `c` token flip, `d` catastrophic word, `e` weight read fault, `f`
input read fault, `g` VU-output read fault, `h` KV word at rest, suffixed `r`
for RU = request and `s` for RU = step; `h` exists for RU = step only), `H2*`
the honest strategies P0-P3 on one run (section 5: `H2a`-`H2d` are P0-P3),
`H3*` epoch-scale runs with random flips through the epoch layer (section 5:
`H3a` P0, `H3b` P1 with tokens-only recording, `H3c` P1 with every-VU-output
recording), `H4*` systematic pod faults (section 6: `H4a` a corrupted weight
cell on one pod for an hour, `H4b` a stale weight version, `H4c` a wrong
kernel path, `H4d` a fleet-wide weight update mid-round), `H5*` detection
before the challenge (section 7: `H5a` the catastrophic fault left to M6, no
action or a post-J declaration; `H5b` the same fault caught before the token
was streamed and truncated; `H5c` the same fault found by re-execution before
the round closes and declared before `J`, priced), `H6*` round-close
logistics (section 8: `H6a` a crash mid-request and `H6at` the same with the
clients gone; `H6bw`, `H6bd`, `H6bt` a straggler waited for, deferred,
truncated; `H6c` a lost KV transfer; `H6dh`, `H6ds`, `H6dc` a request longer
than a round held, split, continued; `H6e` fleet churn).

Each row carries `declarations`, `charge_bits` and `recompute` besides the
catalogue's fields. The `H1` rows add `declarations_vu_outputs` (the count
under the other policy), `changed_tokens`, `fault_class`, `u1` and `u_post`;
the `H2` rows `pre_j`, `post_j` and `accepted`; the `H3` rows `rejections`,
`f_max` and `accepted`. An `H4` row's `declarations` is the P1 count under
`BOUNDARY` recording for the pod-hour; it also carries
`declarations_vu_outputs` (the same under `VU_OUTPUTS`), `faulty_boundary` and
`faulty_vu_outputs` (the pinned words and VUs of the pod-hour before the
`q`-cut), `toy_boundary` and `toy_vu_outputs` (the pins measured on the
fixture's pod) and `m6_bits_boundary`, `m6_bits_vu_outputs` (the M6 charge
under each policy at `u_post(1)`); `charge_bits` is the cheapest mechanism's.
The `H6` rows add `rounds`, `runs`, `outputs`, `check_outputs` and
`honest_cost`.

## Results

Generated by `python -m veritor.stress.report` from `docs/data/stress-honest.json`; declarations are what the honest prover had to declare for the run to be accepted, the charge is what those declarations add to `U` under the price the mechanism's stage admits (`docs/notes/late-advice.md`), recompute is the fraction of the production computation the prover re-executed to know what to declare, `U` is `Bound` at `eta = 2^-40` including the charge.

| ID | What happened | Mechanism | Declarations | Charge (bits) | Recompute | U (λ = 40) | Verdict |
|---|---|---|---|---|---|---|---|
| H1ar | interior low bit, RU = request (RequestsG): interior VU-output bit flip | M6 | 0 | 0 | 1 | 112 | BOUNDARY 0 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 0 declared |
| H1as | interior low bit, RU = step (ClusterG): interior VU-output bit flip | M6 | 0 | 0 | 1 | 142 | BOUNDARY 0 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 0 declared |
| H1br | interior high bit, RU = request (RequestsG): interior VU-output bit flip | M6 | 0 | 0 | 1 | 112 | BOUNDARY 0 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 0 declared |
| H1bs | interior high bit, RU = step (ClusterG): interior VU-output bit flip | M6 | 3 | 413 | 1 | 142 | BOUNDARY 3 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 3 declared |
| H1cr | token flip, RU = request (RequestsG): token flip | M6 | 2 | 265 | 1 | 112 | BOUNDARY 2 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 2 declared, FAULTS_EXCEEDED at f_max = 1 |
| H1cs | token flip, RU = step (ClusterG): token flip | M6 | 4 | 551 | 1 | 142 | BOUNDARY 4 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 4 declared, FAULTS_EXCEEDED at f_max = 3 |
| H1dr | catastrophic, RU = request (RequestsG): catastrophic | M6 | 3 | 397 | 1 | 112 | BOUNDARY 3 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 3 declared |
| H1ds | catastrophic, RU = step (ClusterG): catastrophic | M6 | 8 | 1,102 | 1 | 142 | BOUNDARY 8 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 8 declared |
| H1er | weight read, RU = request (RequestsG): weight-source read fault | M6 | 7 | 927 | 1 | 112 | BOUNDARY 7 / VU_OUTPUTS 18 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 7 declared |
| H1es | weight read, RU = step (ClusterG): weight-source read fault | M6 | 7 | 964 | 1 | 142 | BOUNDARY 7 / VU_OUTPUTS 18 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 7 declared |
| H1fr | input read, RU = request (RequestsG): input-source read fault | M6 | 2 | 265 | 1 | 112 | BOUNDARY 2 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 2 declared |
| H1fs | input read, RU = step (ClusterG): input-source read fault | M6 | 8 | 1,102 | 1 | 142 | BOUNDARY 8 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 8 declared |
| H1gr | VU-output read, RU = request (RequestsG): VU-output read fault | M6 | 0 | 0 | 1 | 112 | BOUNDARY 0 / VU_OUTPUTS 5 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 0 declared |
| H1gs | VU-output read, RU = step (ClusterG): VU-output read fault | M6 | 3 | 413 | 1 | 142 | BOUNDARY 3 / VU_OUTPUTS 5 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 3 declared |
| H1hs | boundary at rest, RU = step (ClusterG): boundary at rest | M6 | 1 | 138 | 1 | 142 | BOUNDARY 1 / VU_OUTPUTS 1 declarations; the tokens-only prover ACCEPTED at theta = (1, 1/8) with its 1 declared |
| H2a | P0, declares nothing; one run of four requests (RequestsG, RU = request) holding a token flip (RU 1), a catastrophic word (RU 2), a low-bit interior flip (RU 3) and a misread prompt token (RU 4); tokens-only recording; theta = (1/2, 1/8) | none | 0 | 0 | 0.486 | 192 | RELATION_REJECTED: gate at address 3201 violates add; opened request RUs [2, 3] of 1-4 (the same header as P1, f_max = 8, so the same J; nothing declared) |
| H2b | P1, replays the opened RUs pinned and declares their pins after J; one run of four requests (RequestsG, RU = request) holding a token flip (RU 1), a catastrophic word (RU 2), a low-bit interior flip (RU 3) and a misread prompt token (RU 4); tokens-only recording; theta = (1/2, 1/8) | M6 | 4 | 530 | 0.486 | 192 | ACCEPTED with 4 post-J declarations for the opened request RUs [2, 3] under a header budget f_max = 8 (every pin of the run, so that any J is covered); the 4 pins of the unopened RUs are never declared |
| H2c | P2, a value check before streaming flags RU 2 (its garbage word); its pins are declared before J at u(1), the other opened pins after J; one run of four requests (RequestsG, RU = request) holding a token flip (RU 1), a catastrophic word (RU 2), a low-bit interior flip (RU 3) and a misread prompt token (RU 4); tokens-only recording; theta = (1/2, 1/8) | M6 + pre-J pardons (counterfactual) | 4 | 297 | 0.486 | 192 | ACCEPTED under P1's declarations (the protocol has no pre-J message); counterfactual charge 4 pre-J at u(1) = 74.2 + 0 post-J at u_post(1) = 132.4 |
| H2d | P3, replays every RU before the boundary commitment and declares every pin at u(1); one run of four requests (RequestsG, RU = request) holding a token flip (RU 1), a catastrophic word (RU 2), a low-bit interior flip (RU 3) and a misread prompt token (RU 4); tokens-only recording; theta = (1/2, 1/8) | M6 + pre-J pardons (counterfactual) | 8 | 594 | 1 | 192 | ACCEPTED under P1's declarations (the protocol has no pre-J message); counterfactual charge 8 pre-J at u(1) = 74.2 + 0 post-J at u_post(1) = 132.4 |
| H3a | random SDC over a fleet of 9 runs of RequestsG (RU = request) in 3 rounds of 3 through run_epoch at theta = (1/2, 1/8): Poisson(1) bit flips per run at random dot words and bits (10 flips in 6 runs, 3 runs changed a streamed token); P0 declares nothing | none | 0 | 0 | 0.547 | 1,008 | epoch RELATION_REJECTED: 1 of 9 runs rejected; round 0: 0 declared, 0 of 3 rejected; round 1: 0 declared, 1 of 3 rejected (RELATION_REJECTED); round 2: 0 declared, 0 of 3 rejected |
| H3b | random SDC over a fleet of 9 runs of RequestsG (RU = request) in 3 rounds of 3 through run_epoch at theta = (1/2, 1/8): Poisson(1) bit flips per run at random dot words and bits (10 flips in 6 runs, 3 runs changed a streamed token); P1 with BOUNDARY recording declares the pins of the opened RUs after J under a round budget f_max = 6 | M6 | 4 | 2,443 | 0.436 | 1,008 | epoch ACCEPTED: 0 of 9 runs rejected, 4 of the fleet's 7 pinned VUs declared (those in opened RUs); round 0: 2 declared, 0 of 3 rejected; round 1: 2 declared, 0 of 3 rejected; round 2: 0 declared, 0 of 3 rejected |
| H3c | random SDC over a fleet of 9 runs of RequestsG (RU = request) in 3 rounds of 3 through run_epoch at theta = (1/2, 1/8): Poisson(1) bit flips per run at random dot words and bits (10 flips in 6 runs, 3 runs changed a streamed token); P1 with VU_OUTPUTS recording declares the pins of the opened RUs after J under a round budget f_max = 7 | M6 | 5 | 2,850 | 0.65 | 1,008 | epoch ACCEPTED: 0 of 9 runs rejected, 5 of the fleet's 10 pinned VUs declared (those in opened RUs); round 0: 1 declared, 0 of 3 rejected; round 1: 2 declared, 0 of 3 rejected; round 2: 2 declared, 0 of 3 rejected |
| H4a | corrupted weight cell on one pod, one hour | source-position pardon (pod scope) | 56,000 | 26 | 0.25 | 282 | one pardon, 26 bits, pre-J from the ECC log (post-J at the same price only under the forced-consistency argument, open); P1 needs 56,000 declarations under BOUNDARY recording or 72,000 under VU_OUTPUTS against f_max = 1: rejected |
| H4b | stale weight version on one pod, one hour | per-pod public weight root (M2/M8) | 632,000 | 0 | 0 | 256 | per-pod kappa_W in the constructor: 0 bits; else re-serve the pod-hour; P1 needs 632,000 (BOUNDARY) or 2,752,000 (VU_OUTPUTS) declarations: rejected |
| H4c | wrong kernel path on one pod, one hour | per-pod architecture (M8) | 632,000 | 0 | 0 | 256 | per-pod gate set in the constructor (ClusterG arches): 0 bits; a late lowering would cost 72,000 bits (1 per RU) and only under the note's conditions; P1 needs 632,000 (BOUNDARY) or 2,752,000 (VU_OUTPUTS) declarations: rejected |
| H4d | fleet-wide weight update mid-round | round close at the update (M2) | 614,445 | 0 | 0 | 256 | close the round and start a run under the new kappa_W: 0 bits; else re-serve the half-round; P1 needs 614,445 (BOUNDARY) or 2,752,000 (VU_OUTPUTS) declarations: rejected |
| H5a | catastrophic silent fault: bit 12 of a step-1 projection dot's output word (VU 367, RU 1, the first of 6 requests of max_new 8) flips, 6 of the request's 7 downstream tokens change and are streamed; nothing is detected before J | M6 post-J declaration (as built), or no action | 1 | 141 | 0.333 | 768 | no action, theta = (1, 1): RELATION_REJECTED at VU 367; post-J, theta = (1/2, 1/8), f_max = 1, 8 challenges: 2 open RU 1 (the first opens 2 of 7 RUs), self_check finds the one VU, declared, ACCEPTED; 6 do not open it, nothing is declared or charged, ACCEPTED |
| H5b | the same fault caught before the token was streamed (a hardware signal, or a value check on step 1's activations: bit 12 of 16 is in the top half); the request ends after its first token, the 7 absent slots are blank check outputs | pre-J truncation: S7 (TruncatedRequestsG), the generated length as advice | 0 | 18 | 0 | 674 | ACCEPTED with no declaration at theta = (1/2, 1/8); the faulty step is not in the circuit; U = 656 = the 41-token run's, plus 18 advice bits |
| H5c | the same fault found before J by re-executing the request on idle capacity before the round closes (the first disagreeing gate is the flipped word, VU 367), and declared before the q-challenge | pre-J declaration at u(1): a priced extension, not a protocol message | 1 | 83 | 1 | 768 | priced, not run: the protocol takes declarations only after J (H5a is the run); a declaration fixed before the challenges costs u(1) = 83.2 bits on the 1024-request table (75.9 on this run: Bound at theta = (1, 1) goes 0 -> 75.9) |
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

- **H1ar**: word 414 of RU 1 (add, correct 0xd868, stored as 0xd869); 0 of 7 streamed tokens changed; pinned VUs by RU: BOUNDARY none, VU_OUTPUTS {1: 1}.
- **H1as**: word 2018 of RU 2 (add, correct 0xc1ec, stored as 0xc1ed); 0 of 7 streamed tokens changed; pinned VUs by RU: BOUNDARY none, VU_OUTPUTS {2: 1}.
- **H1br**: word 226 of RU 1 (add, correct 0x2516, stored as 0xa516); 0 of 7 streamed tokens changed; pinned VUs by RU: BOUNDARY none, VU_OUTPUTS {1: 1}.
- **H1bs**: word 1797 of RU 2 (add, correct 0x49db, stored as 0xc9db); 0 of 7 streamed tokens changed; pinned VUs by RU: BOUNDARY {2: 3}, VU_OUTPUTS {2: 1}.
- **H1cr**: word 241 of RU 1 (add, correct 0x3031, stored as 0xb031); 3 of 7 streamed tokens changed (positions [0, 1, 2]); pinned VUs by RU: BOUNDARY {1: 2}, VU_OUTPUTS {1: 1}.
- **H1cs**: word 1812 of RU 2 (add, correct 0x3c4f, stored as 0xbc4f); 2 of 7 streamed tokens changed (positions [1, 2]); pinned VUs by RU: BOUNDARY {2: 4}, VU_OUTPUTS {2: 1}.
- **H1dr**: word 226 of RU 1 (add, correct 0x2516, stored as 0xdae9); 3 of 7 streamed tokens changed (positions [0, 1, 2]); pinned VUs by RU: BOUNDARY {1: 3}, VU_OUTPUTS {1: 1}.
- **H1ds**: word 1797 of RU 2 (add, correct 0x49db, stored as 0xb624); 2 of 7 streamed tokens changed (positions [1, 2]); pinned VUs by RU: BOUNDARY {2: 8}, VU_OUTPUTS {2: 1}.
- **H1er**: word 200 of RU 0 (weight, correct 0x4, read as 0x8004 by 36 gates); 5 of 7 streamed tokens changed (positions [0, 1, 2, 3, 5]); pinned VUs by RU: BOUNDARY {1: 3, 2: 4}, VU_OUTPUTS {1: 10, 2: 8}.
- **H1es**: word 200 of RU 0 (weight, correct 0x4, read as 0x8004 by 36 gates); 5 of 7 streamed tokens changed (positions [0, 1, 2, 3, 5]); pinned VUs by RU: BOUNDARY {1: 2, 2: 2, 3: 2, 4: 1}, VU_OUTPUTS {1: 8, 2: 4, 3: 4, 4: 2}.
- **H1fr**: word 201 of RU 1 (in, correct 0x1, read as 0x0 by 8 gates); 3 of 7 streamed tokens changed (positions [0, 1, 2]); pinned VUs by RU: BOUNDARY {1: 2}, VU_OUTPUTS {1: 1}.
- **H1fs**: word 201 of RU 1 (in, correct 0x1, read as 0x0 by 8 gates); 3 of 7 streamed tokens changed (positions [0, 1, 2]); pinned VUs by RU: BOUNDARY {1: 8}, VU_OUTPUTS {1: 1}.
- **H1gr**: word 226 of RU 1 (add, correct 0x2516, read as 0xa516 by 13 gates); 0 of 7 streamed tokens changed; pinned VUs by RU: BOUNDARY none, VU_OUTPUTS {1: 5}.
- **H1gs**: word 1797 of RU 2 (add, correct 0x49db, read as 0xc9db by 13 gates); 0 of 7 streamed tokens changed; pinned VUs by RU: BOUNDARY {2: 3}, VU_OUTPUTS {2: 5}.
- **H1hs**: word 498 of RU 1 (add, correct 0x62b9, read as 0x42b9 by 1 gates); 2 of 7 streamed tokens changed (positions [1, 2]); pinned VUs by RU: BOUNDARY {2: 1}, VU_OUTPUTS {2: 1}. Bits [14, 15] of this key word are inert for its reader (the polynomial softmax's square annihilates them): a flip there changes nothing any policy records; the class flips bit 13, the most significant live one.
- **H2a**: Pinned VUs by RU under tokens-only recording: {1: 2, 2: 4, 4: 2} (8 in all; the low-bit flip of RU 3 changed no token and pins nothing; every-VU-output recording would pin 4, one per fault). u(1) = 74.2 bits, u_post(1) = rho log2(1/(1-s)) = 132.4 bits (rho = 687); charge is the price of the declarations made, recompute the share of the run's replay cost the strategy re-executes (the weights RU costs 0). U is the fold with the strategy's header budget (the pins it leaves to post-J declarations) plus its pre-J pardons at u(1), capped at the run's \|Out\| = 192 bits, plus advice; the toy's fold is saturated, so every strategy certifies the cap and the charge column carries what U would lose.
- **H3a**: The rate is about a billion times the Llama-3 SDC rate (2.8e-07 per device-hour) for a run of this size, so that faults occur at all in a fleet we can afford to simulate; the mechanism, not the rate, is what these rows check. Tokens-only recording pins 7 VUs over the fleet (the flips that reached a token, with their cascades), every-VU-output recording 10 (one per flip, silent or not). A pinned VU the silent prover neither declares nor recomputes away is a rejection when sampled: 1 run(s) lost, and with them the epoch.
- **H3b**: Round budget f_max = fault_budget(q x pins / rounds = 1.17, tail 1e-3) = 6, carried by every header of the round and charged once per round: 6 x u_post(1) = 814 bits per round at u_post(1) = rho log2(1/(1-s)) = 135.7 bits (rho = 704 for the round's union of 3 tables at eta / 3), 2443 for the epoch; the same pins before J would cost u(1) = 75.1 bits each. U is the epoch's Bound at eta with the budget, summed over the rounds and capped at the outputs, plus the advice of every run. The two recording policies need different budgets, so their headers and with them their J differ; recompute is the opened share of the fleet's replay cost either way.
- **H3c**: Round budget f_max = fault_budget(q x pins / rounds = 1.67, tail 1e-3) = 7, carried by every header of the round and charged once per round: 7 x u_post(1) = 950 bits per round at u_post(1) = rho log2(1/(1-s)) = 135.7 bits (rho = 704 for the round's union of 3 tables at eta / 3), 2850 for the epoch; the same pins before J would cost u(1) = 75.1 bits each. U is the epoch's Bound at eta with the budget, summed over the rounds and capped at the outputs, plus the advice of every run. The two recording policies need different budgets, so their headers and with them their J differ; recompute is the opened share of the fleet's replay cost either way.
- **H4a**: a layer matrix cell is read by 1 VU per position; measured on pod 0 of the fixture (4 steps, 9 positions, 526 VUs, 79 recorded words, 7 tokens), pinned VUs BOUNDARY/VU_OUTPUTS per cell: low bit w_q 7/9, w_k 9/9, w_v 9/9, w_o 4/9, w_1 7/9, w_2 6/9; top bit w_q 0/3, w_k 3/3, w_v 3/3, w_o 2/3, w_1 0/6, w_2 2/3 (the toy's word is modular, so the top bit of a weight is silent for an even activation; W_k and W_v pin their readers under both policies because a step RU commits its KV entries; the others pin the tokens that flipped, between 0 and 1 of the pod's tokens on an 8-token vocabulary); a constant-table cell (16 readers on the pod) pins 17/2 low bit and 19/3 top bit (few readers move, early in the position, so the outputs' cut exceeds the readers' cut); the row scales W_q low bit: 144,000 readers and 112,000 flipped tokens in the pod-hour, 72,000 / 56,000 in opened RUs at q = 1/2; M6 would charge 10,266,597 / 7,985,131 bits at u_post(1) = 142.6; an RU-scoped source pardon per opened RU costs the same as declaring the RU's readers (10,266,597 bits); RU-scope pardons 24,273,772 bits pre-J and prohibited post-J; headline f_max for 16,384 devices is 2; diagnosis: the ECC scrub log names the cell, else replay the pod's opened RUs (q / pods of the run) and solve one opened VU for the cell
- **H4b**: 192 of 201 cells differ between the versions; measured on pod 0: every recorded word is pinned under BOUNDARY (79 of 79), every dot under VU_OUTPUTS (344 of 526 VUs); pod-hour 1,264,000 words and 5,504,000 dots; one run-wide source-position pardon per changed cell would cost 4,933 bits here and scales with \|W\| (GPT-2 Small 124,490,068 cells); the version is a public object known from the rollout log, so the constructor can carry a per-pod root and the verifier checks the pod's RUs against it
- **H4c**: every dot of the pod (344 of 526 VUs) stores a perturbed word; measured on pod 0: every recorded word pinned under BOUNDARY (79), every dot under VU_OUTPUTS (344); pod-hour 1,264,000 words and 5,504,000 dots; a source-position pardon cannot express a relation change; diagnosis without a kernel-path log: replay one opened RU under each candidate lowering
- **H4d**: measured with the update at step 3 of 6: RUs (7, 8, 9, 10) read the new version, 35 of their 36 recorded words are pinned under BOUNDARY (a token can agree with the honest run by chance on an 8-token vocabulary), every dot (160 of 240 VUs) under VU_OUTPUTS, and nothing before the update; the half-round after the update on 2 pods is 1,228,889 words and 5,504,000 dots; the header binds one kappa_W per run, so the update is a run boundary, not a pardon; RU-scope pardons for the half-round would cost 24,273,772 bits pre-J
- **H5a**: Charged only when the RU is opened (probability q). u_post(1) on the toy shape's 1024-request serving table at theta = (1/2, 1/8), uncapped fold: 141.2 bits (u(1) = 83.2, leverage 1.70, q u_post(1) / u(1) = 0.848); this run's own fold is saturated (\|S\| = 3711, u(1) = 75.9, U capped at \|Out\| = 768 bits with or without the declaration, uncapped 801), so its marginal is invisible. Headline (q = 1.57e-08, s = 8.91e-03): u_post(1) = 6.12e+09 bits per declaration, expected q u_post(1) = 96.1 = 1.015 u(1): the conservation law. 14 of the 16 step-1 embedding and projection dots change a token when bit 12 flips; a tokens-only recording policy would have to pin up to 7 downstream tokens instead of the one VU (section 9).
- **H5b**: The detector menu: hardware signals see 97.8% of the Llama-3 GPU fault mixture (262 of 268 events) at no cost and none of the silent 2.2%; a pre-streaming value check sees the half of the silent faults that blow a value up (an assumption, section 7) at 7.0e-06 of the serving compute on the 70B shape, 98.9% of all faults together. The truncation's price as built is a length for every request: 18 bits here, 3072 on the 1024-request table against u_post(1) = 141.2, 2.6e+14 at the headline (13.9 U_0); naming only the truncated request would cost 13.0 and 53.7 bits (log2 C(requests, 1) + log2 max_new), below u(1) = 94.7. The toy ISA has no NaN: the check's coverage is asserted by the bit position, not evaluated.
- **H5c**: Re-executing a fraction p of the requests finds a fraction p of the faults at a cost of p of the serving compute (here p = 100%); what it finds has been streamed, so the outcome is a declaration, not a truncation. Against leaving the fault to M6 it saves q u_post(1) - u(1) = -12.6 bits per fault at theta = (1/2, 1/8) on the 1024-request table and 1.4 at the headline: the conservation law q u_post(1) = 0.848 u(1) and 1.015 u(1). No p halves the expected charge; p = 1 takes its standard deviation from 7.7e+05 bits per fault to 0 at the headline and lowers the opened-fault count against f_max (section 7).
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
