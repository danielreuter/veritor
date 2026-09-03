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
detection (section 7: `H5a` the catastrophic fault left to M6, no action or a
post-J declaration; `H5b` the same fault caught before the token was streamed
and truncated; `H5c` the same fault found by re-execution before the round
closes and declared before `J`, priced), `H6*` round-close logistics (section
8). Each row carries `declarations`, `charge_bits` and `recompute` besides the
catalogue's fields.

## Results

Generated by `python -m veritor.stress.report` from `docs/data/stress-honest.json`; declarations are what the honest prover had to declare for the run to be accepted, the charge is what those declarations add to `U` under the price the mechanism's stage admits (`docs/notes/late-advice.md`), recompute is the fraction of the production computation the prover re-executed to know what to declare, `U` is `Bound` at `eta = 2^-40` including the charge.

| ID | What happened | Mechanism | Declarations | Charge (bits) | Recompute | U (λ = 40) | Verdict |
|---|---|---|---|---|---|---|---|
| H5a | catastrophic silent fault: bit 12 of a step-1 projection dot's output word (VU 367, RU 1, the first of 6 requests of max_new 8) flips, 6 of the request's 7 downstream tokens change and are streamed; nothing is detected before J | M6 post-J declaration (as built), or no action | 1 | 141 | 0.333 | 768 | no action, theta = (1, 1): RELATION_REJECTED at VU 367; post-J, theta = (1/2, 1/8), f_max = 1, 8 challenges: 2 open RU 1 (the first opens 2 of 7 RUs), self_check finds the one VU, declared, ACCEPTED; 6 do not open it, nothing is declared or charged, ACCEPTED |
| H5b | the same fault caught before the token was streamed (a hardware signal, or a value check on step 1's activations: bit 12 of 16 is in the top half); the request ends after its first token, the 7 absent slots are blank check outputs | pre-J truncation: S7 (TruncatedRequestsG), the generated length as advice | 0 | 18 | 0 | 674 | ACCEPTED with no declaration at theta = (1/2, 1/8); the faulty step is not in the circuit; U = 656 = the 41-token run's, plus 18 advice bits |
| H5c | the same fault found before J by re-executing the request on idle capacity before the round closes (the first disagreeing gate is the flipped word, VU 367), and declared before the q-challenge | pre-J declaration at u(1): a priced extension, not a protocol message | 1 | 83 | 1 | 768 | priced, not run: the protocol takes declarations only after J (H5a is the run); a declaration fixed before the challenges costs u(1) = 83.2 bits on the 1024-request table (75.9 on this run: Bound at theta = (1, 1) goes 0 -> 75.9) |

Notes:

- **H5a**: Charged only when the RU is opened (probability q). u_post(1) on the toy shape's 1024-request serving table at theta = (1/2, 1/8), uncapped fold: 141.2 bits (u(1) = 83.2, leverage 1.70, q u_post(1) / u(1) = 0.848); this run's own fold is saturated (\|S\| = 3711, u(1) = 75.9, U capped at \|Out\| = 768 bits with or without the declaration, uncapped 801), so its marginal is invisible. Headline (q = 1.57e-08, s = 8.91e-03): u_post(1) = 6.12e+09 bits per declaration, expected q u_post(1) = 96.1 = 1.015 u(1): the conservation law. 14 of the 16 step-1 embedding and projection dots change a token when bit 12 flips; a tokens-only recording policy would have to pin up to 7 downstream tokens instead of the one VU (section 9).
- **H5b**: The detector menu: hardware signals see 97.8% of the Llama-3 GPU fault mixture (262 of 268 events) at no cost and none of the silent 2.2%; a pre-streaming value check sees the half of the silent faults that blow a value up (an assumption, section 7) at 7.0e-06 of the serving compute on the 70B shape, 98.9% of all faults together. The truncation's price as built is a length for every request: 18 bits here, 3072 on the 1024-request table against u_post(1) = 141.2, 2.6e+14 at the headline (13.9 U_0); naming only the truncated request would cost 13.0 and 53.7 bits (log2 C(requests, 1) + log2 max_new), below u(1) = 94.7. The toy ISA has no NaN: the check's coverage is asserted by the bit position, not evaluated.
- **H5c**: Re-executing a fraction p of the requests finds a fraction p of the faults at a cost of p of the serving compute (here p = 100%); what it finds has been streamed, so the outcome is a declaration, not a truncation. Against leaving the fault to M6 it saves q u_post(1) - u(1) = -12.6 bits per fault at theta = (1/2, 1/8) on the 1024-request table and 1.4 at the headline: the conservation law q u_post(1) = 0.848 u(1) and 1.015 u(1). No p halves the expected charge; p = 1 takes its standard deviation from 7.7e+05 bits per fault to 0 at the headline and lowers the opened-fault count against f_max (section 7).
