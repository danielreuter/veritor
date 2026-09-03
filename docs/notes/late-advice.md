# Late advice: which prover statements are admissible at which stage, and at what price

A prover statement is anything the verifier takes on the prover's word and
prices instead of checking: advice bits that fix data-dependent structure, a
declared-faulty VU, a pardoned RU, a "this weight cell read as `v'`". The
protocol as built admits exactly two kinds -- advice in the header and VU
declarations after the q-challenge (M6) -- and prices each with a proof. This
note asks what else could be admitted, at which stage, and at what price, so
that the honest prover's fault story (`docs/honest-prover.md`) never assumes a
price the protocol cannot back. The default is prohibition: a (stage, kind)
cell without a proof below is PROHIBITED.

The numbers in section 6 are computed and pinned by
`tests/veritor/analysis/test_late_advice.py` at the two operating points used
throughout the repository.

Notation. `n_RU` replay units, `|S|` verification units, `W_R` and `W_V` the
widest RU and VU output cuts, `q` and `s` the two sampling rates, `eta` the
acceptance threshold and `lambda = log2 (1 / eta)` (40 at `eta = 2^-40`).
`sigma_0(E) = E_J[(1 - s)^{N_J(E)}]` is the acceptance probability of a
transcript whose error set is `E` (`N_J(E)` errors inside the opened RUs),
`Y_0(eta) = {outputs of transcripts with sigma_0 > eta}`, `U_0(eta) = log2
|Y_0(eta)|` the bound with no declarations, and `rho = dU_0 / d lambda` its
slope in bits of capacity per bit of threshold; on the scattered channel
`rho = (u(1) + 1) / log2 (1 / (1 - q s))` (`analysis.rate`). `u(1) = W_V +
log2 |S|` names one VU and its contents (`analysis.faults.unit_fault_bits`).
`mu_f` is the largest Poisson mean with `P[X <= f] > eta` (`mu_0 = lambda
ln 2 = 27.73`, `mu_1 = 31.20`, `mu_2 = 34.15`). "Pre-J" means before the
opened set `J` is revealed, "post-J" after it and before the VU sample `T`,
"post-s" after `T`.

## 1. Adversary model

The protocol has five stages and the prover learns something at each of the
two challenges only.

1. Compile time. The prover fixes the constructor, the gate set, the advice
   `a`, and every header parameter: `eta`, `theta = (q, s)`, the fault budget
   `f_max`, the caps. Nothing about the challenges is known; the header digest
   binds all of it.
2. Admission. The verifier prices the header (`price_and_admit`) and admits.
   Still nothing about the challenges.
3. Boundary commitment. The prover commits the RU boundaries and the claimed
   outputs (`BoundaryMessage`). The q-challenge is then `J =
   bernoulli_subset(q_seed, boundary_phase(header, boundary))`: a PRF of the
   boundary-phase digest under a seed the verifier drew and kept
   (`protocol/challenge.py`, `protocol/phases.py`). The digest binds the
   selection to this transcript; the secret seed makes it unpredictable, so
   the prover cannot grind boundaries for a favourable `J`.
4. Post-J. The prover replays the opened RUs and commits their interiors
   together with its declarations (`InteriorMessage.manifest` carries
   `declarations`). The s-challenge is `T = bernoulli_subset(s_seed,
   interior_phase(replay_phase, interiors))`, again a PRF of the digest under
   a verifier seed, so the sample binds every declaration: changing the
   declaration set changes the sample, and the prover cannot predict either.
5. Post-s. Openings and proofs for `T`. No message may add a statement.

In the epoch layer (`protocol/epoch.py`) the two seeds of every run are
`derive_run_seed(round_seed, seal, round, run, label)`, an HMAC under a round
seed the verifier draws at `close_round` and never publishes before the seal;
the seal binds every admitted run's boundary. The stage structure is the same;
what the epoch adds is that all runs of a round reach stage 3 before any of
them reaches stage 4.

So the prover has exactly two pieces of information to exploit: `J` at stage
4 and `T` at stage 5. Every price below is the number of extra output
vectors that knowing `J` (or `T`) lets an adversary get accepted with
probability above `eta`, over and above `|Y_0(eta)|`.

## 2. The stage ladder

Two proof shapes give every entry.

**Fixed-in-advance.** If a statement of `b` bits is fixed before `J`
(stages 1-3), the accepted output is a function of the correct computation,
of the values at the unpardoned errors `E'` (which must survive the sampling
as before) and of the statement itself. Enumerating the statement,

    |Y| <= 2^b * |Y_0(eta)|,   U <= U_0(eta) + b.

The statement may name anything; its price is its length. This is how advice
is charged today (bit for bit) and how `fault_allowance_bits` charges a
declaration at `q = 1`.

**Adaptive.** If the statement is chosen after `J`, the adversary chooses it
to fit `J`. Two rigorous bounds, from `analysis.faults` (there for VU
declarations, here in general). Write `sigma_f` for the acceptance
probability with `f` statements chosen post-J.

- (i) *Survival*: if each statement can rescue at most one checked error,
  `sigma_f(E) <= (1 - s)^-f sigma_0(E)`, so `U_f <= U_0(eta (1 - s)^f) = U_0
  + f rho log2 (1 / (1 - s))`. This is a threshold shift; it is tight on the
  scattered channel and it is the charge `declared_bits` makes.
- (ii) *Union*: `max over statements <= sum over statements`, so some fixed
  statement has acceptance `> eta / (number of statements)`, and `U_f <=
  U_0(eta / K^f) + f b = U_0 + f (rho log2 K + b)` for `K` candidate
  statements of `b` bits each. Rigorous for every kind; expensive whenever
  `rho log2 K` is, and at the headline `rho = 4.7e11`, so every bit of `log2
  K` costs `2.5%` of `U_0`.
- (iii) *Forcing*: if the statement's scope is wide enough that any
  statement other than the one the transcript was computed under fails a
  checked relation with probability near one, the post-J choice is forced by
  the pre-J commitment and the price collapses to the message. Section 2.3
  makes this rigorous.

### 2.1 VU value pardons (M6)

Pre-J: `u(1)` each (fixed-in-advance). Post-J: `declared_bits(f) =
min(U_0(eta (1 - s)^f), U_0(eta / (1 + |S|)^f) + f u(1)) - U_0`, bound (i)
in every regime that matters, `rho log2 (1 / (1 - s))` per declaration.
Because `rho log2 (1 / (1 - s)) ~ (u(1) + 1) / q` on the scattered channel,
a post-J declaration costs `1 / q` pre-J ones: the adversary plants errors
everywhere and pardons only the caught fraction `q`. Post-s: PROHIBITED. The
attack pardons exactly the sampled bad VUs; with `f` pardons the adversary
survives `mu_f - mu_0` more sampled errors, each worth a `1 / (q s)` planting
leverage, so the first pardon alone is worth about `(mu_1 - mu_0) (u(1) + 1)
/ (q s)`: `2.4e12` bits at the headline, `12.5%` of `U_0` and `390` times the
post-J price.

### 2.2 RU-scope pardons

Pre-J: `W_R + log2 n_RU` (the pardoned RU's whole output is free). Post-J:
bound (ii) with the pardoned sets as statements. For a set `P` of pardoned
RUs, `1{P subset J}` times the survival of the rest, so `sigma_f(E) <=
sum_{|P| <= f} q^|P| sigma_0(E \ P)`; there are at most `(1 + q n_RU / (1 -
q))^f` weighted terms, giving `U_f <= U_0 + f [rho log2 (1 + q n_RU / (1 -
q)) + W_R + log2 n_RU]`, `0.47 U_0` per pardon at the headline. The matching
attack corrupts `(mu_f - mu_0) / q` whole RUs and pardons the opened ones:
`0.095 U_0` for the first pardon. Between a tenth and a half of the bound
per pardon: PROHIBITED post-J, and unnecessary, since an honest RU-wide fault
(a pod's request was garbage) is a truncation or a re-serve, both pre-J.

### 2.3 Source-cell pardons

A source-cell pardon says "in scope `X`, cell `w` of `kappa_W` (or of an
input or KV commitment) was read as `v'`"; the verifier substitutes `v'` into
the relation of every checked reader of `w` inside `X`. The message has
`log2 |M|` bits, `|M| = m n_W 2^W_cell` for `m` scopes: `51.9` bits at the
headline for one scope. The scope is the whole point.

Pre-J: `log2 |M|` (fixed-in-advance), any scope.

Post-J, the product bound. Fix the transcript. Let `Q` be the set of scopes
the prover computed under a substituted cell (an *effective* substitution:
at least one reader in the scope is not silent, i.e. its committed output
differs from the honest one; a substitution nobody reads is not a
corruption), `E` the remaining error set relative to the substituted circuit
`C_Q`. A scope `X in Q` is *hit* if one of its non-silent readers is opened
and sampled; a hit scope fails unless pardoned, since the relation is then
checked against the original cell. Acceptance therefore needs both `hits(Q)
<= f` and no sampled error in `E`, and the two events bound it separately:

    sigma_f <= min( P[hits(Q) <= f], sigma_0^{C_Q}(E) ).

Hits are independent across scopes (disjoint RUs) with `P[hit_X] = q_X = 1 -
prod_{RUs i in X} [(1 - q) + q (1 - s)^{r_i}]`, `r_i` the non-silent readers
in RU `i`; `q_X >= q s` always (one reader) and `q_X -> 1` once `q sum_i (1 -
(1 - s)^{r_i}) >> 1`. The substituted circuit has the same kind table, so
`|Y_0^{C_Q}(eta)| <= 2^{U_0(eta)}`, and

    |Y| <= N_Q(eta) 2^{U_0(eta)},   N_Q(eta) <= (1 + |M|)^{c_max},
    c_max = max { c : P[Binomial(c, q_min) <= f] > eta } ~ mu_f / q_min,

with `q_min` the smallest `q_X` over the scopes of the kind. The price is
`c_max log2 (1 + |M|)`, and it has three regimes:

- *Dense scopes* (`q_min ~ 1`): a fleet-year run of the 70B shape has `q N
  ~ 4.6e5` opened requests each reading every layer cell `1024` times, so
  `P[no hit] = e^{-4.6e5}`; `c_max = f` and the price is `f log2 (1 + |M|)`:
  the message, `52` bits per pardon. This is the forcing regime: any pardon
  other than the one the run was computed under fails, so knowing `J` buys
  nothing.
- *Sparse scopes* (a pod-hour run, `q_X ~ 5e-3` to `7e-3`): `c_max ~ mu_f /
  q_X ~ 5e3` for `f = 1`, price about `3e5` bits per round -- `1.5e-8 U_0`.
  The adversary can plant a different wrong cell in thousands of pod-hours
  and pardon the one that is hit, but each planted scope is worth only its
  message.
- *RU scope* (`q_X ~ q`): `c_max ~ mu_1 / q = 2.0e9`, price `2.0e9 x (log2
  |M| + log2 n_RU) = 1.9e11` bits, `1.0% U_0` per pardon, thirty times a VU
  pardon. Admissible but not cheap; the VU-level bound (i) never beats it
  here because a cell has `d = 1024` readers per request and bound (i) pays
  `d u_post(1) = 0.33 U_0`.

The product bound counts the `f = 0` configurations too (they are already
inside `U_0`), so the marginal price of the `f`-th pardon is smaller, about
`(mu_f - mu_{f-1}) / q_min` scopes' worth; the numbers above are the
rigorous, non-marginal ones. Conclusion: post-J source-cell pardons are
ADMISSIBLE at `c_max(q_min, f) log2 (1 + |M|)`, and the honest prover should
scope them to the run (the pod-hour), where the price is negligible, never
to the RU. Post-s: PROHIBITED, as for every kind (the post-s attack is scope
independent: pardon what was sampled).

### 2.4 Port pardons

"Port `p` of gate kind `k` read a wrong value" in scope `X`. If the port is
fed by one source cell for the whole scope, this is a source-cell pardon of
that cell and is priced as such. If it is per gate instance (each reader of
the port saw its own wrong value), it is not one statement but `|readers|`
VU-value statements and costs `|readers| u(1)` pre-J or `|readers|
u_post(1)` post-J; there is no cheaper argument because the readers' values
are independent degrees of freedom. So a port pardon is either a source-cell
pardon or a bundle of VU pardons; it is not a third kind. `|M| = n_ports
2^W_port` per scope where it applies.

### 2.5 Structural advice (routes, lengths, kernel paths)

Pre-J: bit for bit, as today. Post-J: section 3. Post-s: PROHIBITED.

### 2.6 The ladder

| kind \ stage | pre-J (fixed-in-advance) | post-J | post-s |
|---|---|---|---|
| VU value pardon | `u(1)` = 94.7 | `rho log2 (1/(1-s))` = 6.1e9 = 3.2e-4 `U_0` | PROHIBITED: first pardon 0.125 `U_0` |
| RU-scope pardon | `W_R + log2 n_RU` = 8237 | PROHIBITED: 0.095 to 0.47 `U_0` | PROHIBITED |
| source cell, run-wide, dense | `log2 \|M\|` = 51.9 | `f log2 (1 + \|M\|)` = 52 | PROHIBITED |
| source cell, run-wide, pod-hour | 51.9 + `log2 m` | `~ mu_f / q_X x log2 (1 + \|M\|)` = 3e5 | PROHIBITED |
| source cell, RU scope | 51.9 + `log2 n_RU` | `mu_f / q x (log2 \|M\| + log2 n_RU)` = 1.9e11 = 1.0% `U_0` | PROHIBITED |
| port pardon | a source-cell pardon, or `\|readers\| u(1)` | likewise | PROHIBITED |
| structural advice, per-RU family | `sum_R log2 \|V_R\|` | same (locality, section 3) | PROHIBITED |
| structural advice, shared per run | `log2 \|V\|` | `log2 \|V\| (1 + rho)` = 2.5% `U_0` per binary choice | PROHIBITED |
| structural advice, function of public config | 0 | 0 | 0 |

Headline numbers; section 6 has both operating points.

## 3. Late lowering

*Lowering* is the compiler's choice of a concrete variant for a kind: a
kernel path, a tile schedule, a rounding order. Fixed at compile time it is
free. The question is what it costs when the prover wants to choose the
variant after the boundary commitment or after `J`, because the pod that
served the request decided the kernel and the replay must follow it.

**Setting.** Every RU `R` has a finite variant family `V_R`, declared in the
header, each variant a different set of relations on the same positions. A
choice vector `v = (v_R)` fixes the circuit `C_v`; a committed transcript has
error set `E_v` relative to `C_v`, and inside RU `R` the count `n_R(v_R)`
depends on `v_R` alone.

**Locality lemma.** Let the prover choose `v` after seeing `J`. Its
acceptance probability is `E_J[max_v prod_{R in J} (1 - s)^{n_R(v_R)}]`. The
maximum of a product over independent coordinates is the product of the
maxima, so the maximiser is `v*_R = argmin n_R(v_R)`, which depends on `R`'s
committed values only and not on `J`. Hence

    E_J[max_v prod_{R in J} g_R(v_R)] = E_J[prod_{R in J} g_R(v*_R)]
                                       = sigma_{v*}(E_{v*}) = max_v sigma_v(E_v),

and the accepted set is `Y = union_v Y_v(eta)`, so

    |Y| <= sum_v |Y_v(eta)| <= 2^A max_v 2^{U_v(eta)},   A = sum_R log2 |V_R|.

Knowing `J` buys nothing: the price is the compile-time price `A`, with the
fold taken over the worst variant assignment. A proof by union over `v`
would instead divide the threshold by `2^A` and cost `rho A`; locality
removes that term, and it is the only reason late lowering is affordable at
all.

**Conditions.** The lemma needs, and the protocol must enforce:

1. `V_R` finite and in the header, so `A` is priced at admission.
2. Independence: `v_R` changes only RU `R`'s relations. No shared parameter,
   no cross-RU coupling.
3. The variant leaves the RU's positions, widths and boundary unchanged, so
   the index and the commitments are the same for every `v`.
4. The verifier folds `max_v U_v`: either the union-of-variants kind table
   (sound for mixed assignments, at most `log2 |V_R|` looser per erroneous RU
   on the scattered channel) or, where the per-kind series are pointwise
   dominated, the worst variant. The fold is monotone in each per-kind
   series (`multiply`, `power`, `cap`, knapsack, Laplace), so the union
   table is an upper bound.
5. No value rides along: a variant is a relation set, not a parameter. A
   parametrised family is admissible only if it is finite, listed in the
   header and counted in `|V_R|`.
6. The choice is fixed before `T`: it goes into the interior message so the
   sample binds it (post-s choice is the post-s pardon, PROHIBITED).
7. The boundary is committed before `J`, as always; the variant cannot
   change a committed value.

Condition (5) and (6) are exactly what forbids the case the architect feared:
advice that arrives after the RU is sampled and is injected into it. A
variant cannot carry a value; a value can only enter as a pardon, priced in
section 2.

**Caveat on the fold.** The error truncation of the default fold moves with
the grid, so on a saturated table `bound(max_faults = f).bits` can come out
below `bound().bits` (small table: `1895.8` at `eta`, `1887.1` at `eta (1 -
s)`; `1894.5` at both on a fine grid). Sound, but not monotone in `eta` at
`max_buckets = 2048`. Read declaration prices off the rate, or off a fine
grid.

**Granularity, and what it costs.** The price `A` counts every RU with a
choice. Three cases:

- *Per-RU independent choice* (one bit per request): `A = n_RU = 2.9e13`
  bits, `1.55 U_0` at the headline. Secure and unaffordable. Even a family
  restricted to a few kinds per request is priced per request. The
  rounding channel is why `A` must count every RU: two floating-point
  lowerings of the same request give two honest outputs, both accepted with
  probability one.
- *Shared choice per run* (one kernel path per pod per round): `v` is the
  same for every RU of the run, so `max_v prod_{R in J}` no longer factors;
  the post-J optimum depends on `J`, locality fails, and only bound (ii)
  applies: `log2 |V| + rho log2 |V|`, `4.7e11` bits or `2.5%` of `U_0` per
  binary choice per round. But the honest prover knows this choice at
  production time, so it belongs in the boundary message, pre-J, at `log2
  |V|` bits. Nothing is gained by lateness here and `2.5%` is lost.
- *A function of public configuration* (placement, rollout log, the pod's
  declared kernel version, all under the public-config mechanism M3): the
  verifier recomputes the variant; price zero at every stage.

So the compiler rule is: derive the variant from public configuration where
possible; otherwise declare shared choices pre-J; never leave per-request
freedom, secure though it is.

## 4. The epoch setting

**Per-round budget.** `EpochVerifier._capacity` folds `union(round tables)`
at `eta / rounds` with the round's `max_faults`, and `receive_interiors`
enforces the round total (`test_the_fault_budget_is_the_rounds_to_share`).
Both adaptive bounds of section 2 are pointwise in the transcript or unions
over statements; neither cares that the runs' sample challenges are released
one after another within the round, since each run's `T` is a PRF of its
own interior digest under a seed the prover never sees. No gap. A per-run
budget would be unsound: a fleet of small runs could plant a different wrong
cell in every RU and pardon, in every run, the opened one, at capacity `n_RU
log2 |M|` with nothing to pay for the selection. The budget is the round's.

**The round seed is load-bearing for M6.** If `s_i` were public before the
interior message, declarations would be chosen knowing `T`: the post-s
attack, `390` times the post-J price. The verifier-private `round_seed` is
what keeps every declaration at stage 4.

**A per-epoch budget is not cheaper under the current accounting.** The
epoch bound is a union over rounds: each round is bounded at `eta / rounds`
and the epoch's capacity is the sum. Under this accounting a budget of `f`
shared across `R` rounds is priced by a second union, over the allocations
`(f_1, ..., f_R)`: some allocation has acceptance `> eta / C(f + R - 1, f)`,
which costs at least `log2 (f + 1)` bits of threshold in *every* round,
`R rho log2 (f + 1)`: `4.7e14` bits at `R = 1000`, `f = 1`, twenty-five
`U_0`. Charging every round the full `f` instead costs `R f rho log2 (1 /
(1 - s))`, `6.1e12` at the same point, so the per-round budget is the right
object under the current epoch bound. `EpochVerifier` needs no change.

**What the per-round accounting hides.** The scattered channel's capacity is
`rho lambda` and `rho` does not depend on the size of the table (only on
`q`, `s`, `u(1)`), so a round of an hour and a round of a year have the same
capacity, `U_r(eta / R) ~ rho (lambda + log2 R)`. An epoch of `R` rounds
therefore certifies about `R U_0 (1 + log2 R / lambda)`: `1.15 U_0` per
hourly round and `1.0e4 U_0` for a year of hourly rounds, at which point the
bound is a third of every output bit the fleet emits. The honest prover's
M6 premium, `3.2e-4` of each round's bound, is a rounding error next to
this; the number of rounds is the first-order parameter. The union over
rounds is loose by about `R`: the epoch is accepted only if every round is,
acceptance probabilities multiply across rounds, so an adversary has
`lambda` bits of threshold to spend on the whole epoch, not per round.

**Conjecture (shared threshold).** The number of messages an adaptive
adversary can transmit over an `R`-round epoch with acceptance `> eta` is at
most `2^{U_0(eta) + rho + O(R log R)}`. Sketch: for the last two rounds, a
message accepted with probability `> eta` has `sigma_1 E[sigma_2] > eta`;
messages sharing `y_1` differ in `y_2` for every challenge history, and for
a fixed history at most `|Y_2(t)|` of them have `sigma_2 >= t`; averaging
over histories with a level-set argument at resolution `epsilon` bits gives
at most `2^{U_2(eta / sigma_1) + rho epsilon + log2 (1 / epsilon) + O(1)}`
messages per `y_1`, and summing over `y_1` by level of `sigma_1` gives
`2^{U_0(eta) + rho epsilon + O(log lambda)}`; iterate over rounds with
`epsilon = 1 / R`. If it holds, `R` rounds cost `rho` (2.5% of `U_0`) instead
of `R U_0`, the per-round M6 premium sums to `R f u_post(1)` (`0.32 U_0` at
`R = 1000`, `f = 1`) and the natural companion is a per-epoch budget at `f
u_post(1)` once, since the survival factor `(1 - s)^{-sum_r f_r}` is also a
product. This is the most valuable open item in the epoch layer and it is
not proven here.

## 5. Conservation law

A fault handled at stage `k` is charged its stage price with the
probability that the stage is reached with the fault still unexplained:
`p_k in {1, q, q s}` for pre-J, post-J, post-s. The expected charge is
`u(1)` at every stage, because the leverage of each stage is `1 / p_k`:

    q u_post(1) = q rho log2 (1 / (1 - s)) = (u(1) + 1) (1 + s / 2 + ...)
                = 1.015 u(1)  at the headline (measured 96.1 vs 94.7).

What differs between stages is not the expected charge but the option
premium the verifier must reserve before the challenge -- `f_max` times the
stage price, paid whether or not a fault occurs -- and the variance. Three
consequences, each measured in `docs/honest-prover.md`:

- Detection before `J` (hardware signals, value checks, partial
  re-execution) cannot lower the expected charge by more than the `1.5%`
  term: `1.4` bits per fault at the headline. It buys headroom under `f_max`
  and a certain charge instead of a `q`-probability one.
- Structure may be late for free (section 3, per-RU families) but values
  may not: every value statement after `J` carries its stage leverage in the
  premium.
- The cheapest honest response to a value fault is whichever stage's premium
  the prover is already paying. With `f_max = 0` the prover pays no premium
  and accepts a rejection with probability `q D` per round (`D` faults per
  round); with `f_max >= 1` it pays `3.2e-4` of the round's bound per unit
  of budget. At today's fleets `q D ~ 1e-7`, so the premium exceeds the
  expected loss by three orders of magnitude, but it is `3.2e-4`.

There is no algebraic shortcut in floating point (Freivalds-style checks are
out of scope; production matmuls are not exact), so the frontier of
detection is what section 7 of `docs/honest-prover.md` measures.

## 6. Numbers

Pinned by `tests/veritor/analysis/test_late_advice.py`.

**Headline** (`veritor.evaluation.global_estimate`: the 70B serving shape,
`n_RU = 2.93e13` requests, `q = 1.57e-8`, `s = 8.91e-3`, `eta = 2^-40`,
`|S| = 2^78.7`, `W_V = 16`, `W_R = 8192`, `u(1) = 94.7`, `rho = 4.74e11`,
`U_0 = 1.90e13`; the capacity is the closed-form rate, and every adaptive
charge is the same threshold shift `declared_bits` applies, read off the
rate).

| statement | bits | share of `U_0` |
|---|---|---|
| VU pardon, pre-J | 94.7 | 5.0e-12 |
| VU pardon, post-J, bound (i) | 6.12e9 | 3.2e-4 |
| VU pardon, post-J, bound (ii) | 3.73e13 | 1.96 (not the charge) |
| VU pardon, post-s, first | 2.37e12 | 0.125 |
| RU-scope pardon, pre-J | 8237 | 4.3e-10 |
| RU-scope pardon, post-J, bound (ii) | 8.9e12 | 0.47 |
| RU-scope pardon, post-J, attack | 1.8e12 | 0.095 |
| source cell, one dense scope, pre-J or post-J | 51.9 | 2.7e-12 |
| source cell, pod-hour scopes (`q_X = 6e-3`, `m = 1e4`), post-J, `f = 1` | 3.4e5 | 1.8e-8 |
| source cell, RU scope, post-J, `f = 1` | 1.92e11 | 1.0e-2 |
| readers' bound (i) for one RU's cell, `d = 1024` | 6.3e12 | 0.33 |
| late lowering, one bit per RU | 2.93e13 | 1.55 |
| shared binary choice, post-J, per round | 4.74e11 | 2.5e-2 |
| conservation: `q u_post(1) / u(1)` | 1.015 | |
| round of `R = 8760`: `U_r(eta / R) / U_0` | 1.14 | `1.0e4 U_0` a year |
| per-epoch budget by allocation union, `R = 1000`, `f = 1` | 4.74e14 | 25 |
| per-round budget, `R = 1000`, `f = 1` | 6.12e12 | 0.32 |

**Simulation** (`theta = (1/2, 1/8)` on the small datacenter run's kind table,
where the fold is exercised directly: `|S| = 3791`, `W_V = 64`, `u(1) =
75.9`, `n_RU = 20`, `W_R = 224`, `|Out| = 480`). The fold is saturated:
uncapped `1894.5` bits at every threshold on a fine grid, capped at `480`
with or without a budget, so the prices are the slopes the fold would add
before the cap: `rho = 755.9`; post-J VU `145.6` per pardon (`u(1) / q =
151.8`); bound (ii) `9062`; post-s `1214` (more than the whole 480-bit
output); RU-scope pre-J `228.2`, post-J `3267`. At `theta = (1, 1)` the
budget costs exactly `f u(1)`. Source-cell prices are not evaluated on the
toy: its weights are few and every reader is a dot VU, so the scope regimes
of section 2.3 do not separate.

## 7. Recording costs

What the honest prover must keep from production in order to make each
statement, per 70B request (`1.34e14` flops) and per GPT-2 Small request
(`5.77e10` flops), from `evaluation.serving`:

| policy | 70B | GPT-2 Small | needed for |
|---|---|---|---|
| all VU outputs (2 bytes per interior position) | 33.9 GB, `2.5e-4` B/flop; hashing at `1e5` MACs per 64-byte chunk is 79% of serving | 175 MB, `3.0e-3` B/flop; hashing 9.5x the compute | naming the faulty VU (pre-J at `u(1)`, or any-stage exact declarations) |
| KV boundary | 2.68 GB | 7.4 MB | RU = decode step; pinned replay of a step from its boundary |
| tokens only | 2 KB | 2 KB | pinned replay of the request (`docs/honest-prover.md` section 3): declares the flipped outputs, not the faulty kernel |
| kernel-path log | ~1 KB | ~1 KB | shared structural choices declared pre-J, or derived from public config (free) |
| ECC / machine-check log | bytes per event | | source-cell pardons pre-J; the diagnosis for the run-wide kind |

`q = 1` with recorded interiors buys `u(1)` pardons at every stage but costs
the full interior commitment (79% of serving at 70B): by the conservation
law that is the same trade as P3 in `docs/honest-prover.md`, paid in
hashing instead of premium.

## 8. Reconciliation with `docs/notes/declaration-kinds.md`

That note lists four open pricing questions against an earlier version of
this ladder.

1. *Run-wide source-cell pardon after `J`.* Resolved in section 2.3 by the
   product bound: `c_max(q_min, f) log2 (1 + |M|)`, the message in the dense
   regime (`52` bits), about `3e5` bits per round for pod-hour scopes,
   `1.0% U_0` per RU-scoped pardon. Its two gaps are answered: silent readers
   are excluded from `r_i` (a substitution with no non-silent reader is not
   a corruption and adds no output), and the sparse regime is priced rather
   than degenerated to the RU-scoped kind. Its plant-and-pardon estimate
   `(mu_f / q_scope) log2 |M|` is the same quantity; the rigorous version
   adds `log2 (1 + m)` for the choice of scopes and counts `mu_f` rather than
   `mu_f - mu_{f-1}`. The kind is admissible post-J at run scope.
2. *Where the budget lives.* The round's; section 4 says so and gives the
   per-run attack.
3. *`142.6` vs `145.6` for the toy `u_post(1)`.* Two fixtures (that note's is
   capped at 256 bits, this one's at 480); the argument is the same and the
   numbers are each fixture's slope. No action.
4. *Kind-level port pardon.* Agreed: a broadcast cell's source pardon, or a
   bundle of VU pardons; section 2.4.

Nothing in `docs/honest-prover.md` may quote a price this note does not
establish; the scenario pricing of its section 6 (`H4`) uses the pre-J
message prices, which stand, and may now also use the post-J run-wide price.
