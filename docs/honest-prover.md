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

*The matrix: landing site, magnitude, persistence, scope, detectability before
streaming, RU choice, recording policy; per cell the declarations needed,
whether a VU declaration can express the fault at all, the cheapest admissible
declaration kind, and the prover's diagnosis cost.*

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

*Corrupted weight cell, stale weight version, wrong kernel path, fleet-wide
update mid-round; what each costs under P1 and under the priced alternatives.*

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
`docs/data/stress-honest.json`: `H1*` honest replay by fault class (section
3: `H1a`-`H1h` one fault of each class, `a` interior low bit, `b` interior
high bit, `c` token flip, `d` catastrophic word, `e` weight read fault, `f`
input read fault, `g` VU-output read fault, `h` KV word at rest, suffixed `r`
for RU = request and `s` for RU = step; `h` exists for RU = step only), `H2*`
the strategies on one run (section 5: `H2a`-`H2d` are P0-P3), `H3*` random
SDC at fleet scale through the epoch layer (section 5: `H3a` P0, `H3b` P1
with tokens-only recording, `H3c` P1 with every-VU-output recording), `H4*`
systematic faults (section 6), `H5*` detection (section 7), `H6*` round-close
logistics (section 8). Each row carries `declarations`, `charge_bits` and
`recompute` besides the catalogue's fields; the `H1` rows add
`declarations_vu_outputs` (the count under the other policy), `changed_tokens`,
`fault_class`, `u1` and `u_post`, the `H2` rows `pre_j`, `post_j` and
`accepted`, the `H3` rows `rejections`, `f_max` and `accepted`.

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
