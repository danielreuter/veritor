# 3. Security analysis

The four properties S1–S4 of §1.8 are proved (as sketches, with the
assumptions they need) in §3.2–§3.6 and composed in §3.7. §3.8 games the
protocol adversarially, §3.9 states what survives every defense, and §3.10
shows that dropping any one of the three ingredients breaks a property.

## 3.1. Assumptions

- **A1 (random oracle).** $\mathcal O$ is a random oracle. Used for seed
  derivation, the lottery, and the Fiat–Shamir steps inside the proofs.
- **A2 (tile-product unpredictability on $H$).** Fix $A\in\mathbb Z^{n\times k}$,
  $B\in\mathbb Z^{k\times m}$ before $\zeta$ is drawn; let $E,F$ be rank-$r$
  noise derived from $\zeta$ as in §2.2. Any algorithm running on hardware $H$
  that outputs correct values for a set $S$ of running-sum tiles
  $C'^{(\ell)}_{ij}$ of $(A+E)(B+F)$ uses at least $(1-\varepsilon_{\rm alg})\,r^3\,|S^\downarrow|$
  MAC-equivalents, where $S^\downarrow$ is the prefix closure of $S$ (a
  running sum at $\ell$ needs the products at all $\ell'\le\ell$). This is
  PoUW's Assumption 6.4 specialized to a hardware cost model and extended from
  "all intermediates" to "any subset", as Remark 2.1 of the paper already
  requires. Justification: each $r\times r$ block of $A+E$ and $B+F$ is
  marginally uniform (Lemma 6.5), and the known ways to exploit the rank-$r$
  correlation across blocks (compute $E_L[i](E_R B')$ etc.) still cost
  $r^3$ per output tile because there are $nkm/r^3$ distinct output tiles of
  $r^2$ entries each. $\varepsilon_{\rm alg}$ is the fraction a fast
  matrix-multiplication algorithm could save on a random $r\times r\times r$
  integer product on $H$; on tensor-core hardware, whose cost model is the
  naive algorithm, it is close to 0.
- **A3 (Verity's cryptography).** Position-binding vector commitments,
  zero-knowledge proofs of knowledge with negligible soundness error for the
  VU and ticket statements, hiding commitments. As in security-argument §1–§5.
- **A4 (inventory).** The auditor knows $H$ and its aggregate rate $\Phi$;
  the developer has no other compute. Physical audit, outside the protocol.
- **A5 (reference efficiency).** $\alpha$, the fraction of a saturated device's
  MAC-equivalents that the best implementation of §2.2 spends inside tile
  products, is known to the auditor to within $\varepsilon_\alpha$.
- **A6 (frozen baseline).** No weight commitment is created or changed during
  the epoch except as a declared output; ingress is warden-observed (PBA).
- **A7 (non-regeneration).** Within a $\Gamma$-bounded task, a value that is
  not baseline and not a function of the task's $\le X$ bits of imports cannot
  be produced; in particular an accumulated weight matrix cannot be
  recomputed from the baseline and a short seed. PBA Assumption A.7 / Remark
  A.5. This is where "training is not compressible into a task" lives.
- **A8 (hash cost).** Hashing a tile is not cheaper than computing it: the
  lottery hash is non-algebraic (bit-level mixing per entry), so its value on a
  tile cannot be obtained from a low-degree function of the operands. §4.1
  explains why a linear or quadratic hash would break A2 by a factor $r$.

## 3.2. S1: consumption

**Claim.** Under A1, A2, A3, if the epoch holds $K$ valid tickets then, except
with probability $\le\exp(-\tfrac{1}{3}\Delta^2 pM)$ over the oracle, the
developer performed at least
$$
M\,r^3(1-\varepsilon_{\rm alg}),\qquad M:=\frac{K}{p}(1-\Delta),
$$
MAC-equivalents computing running-sum tiles of declared $\mathrm{CMM}_r$ gates
on their committed operands under noise derived from the auditor's seeds.

**Proof sketch.** Each valid ticket $(j,g,i,\ell,j',h,\pi)$ has a proof of
knowledge (A3) of operands opened from request $j$'s commitments and a tile $T$
equal to the correct running sum on those operands with the noise derived from
$\zeta_g$, with $h=\mathcal O(\zeta_g,i,\ell,j',T)<p2^\lambda$. By A1, $h$ is
uniform unless the prover queried $\mathcal O$ on exactly this input, i.e.
unless it held the correct $T$. So every valid ticket corresponds to a
correctly computed tile that the prover held, and tickets with distinct
$(j,g,i,\ell,j')$ correspond to distinct tiles. Let $\mathcal S$ be the set of
correct running-sum tiles the prover ever held. The number of winning tiles in
$\mathcal S$ is $\mathrm{Bin}(|\mathcal S|,p)$ (independent uniform hashes), and
the prover can present at most that many valid tickets. By Chernoff,
$|\mathcal S|<K(1-\Delta)/p$ with probability at most $\exp(-\Delta^2pM/3)$.
For every tile in $\mathcal S$ whose operands are the honest values, those
operands were fixed by the header before $\zeta_g$ was known (§2.1, §3.3), so
A2 applies to that subset and gives the cost bound. Tiles whose operands are
*not* honest values may have been chosen after $\zeta_g$ and A2 does not
price them; §3.3 and §3.4 bound their number by $B$ row blocks across the
epoch, and the claim should be read with $M$ reduced by that amount. $\square$

Three remarks. Verity's sampling enters S1 only through that last
subtraction; the rest holds whatever the committed operands are. The bound is
about *tiles of declared gates* because the proof opens the operands at gate
$g$'s positions in request $j$'s circuit; a tile of an undeclared computation
has no such position. And the prover gains nothing by computing garbage for
tiles it hopes will not win: a garbage tile's hash is uniform, it wins with
probability $p$, and when it wins the ticket cannot be proved, so the prover
either forfeits it (and the compute is unaccounted) or is rejected.

## 3.3. Ordering lemma: no cheap operands

**Lemma.** If the header is bound before $\zeta$, then for every declared
$\mathrm{CMM}_r$ gate whose committed operands are the *honest* values, the
distribution of the noised operands, conditioned on the header, is that of
PoUW's honest encoder on a fixed instance. In particular no adversary can
arrange for a block of $A+E$ or $B+F$ to be low-rank, sparse, repeated, or
otherwise structured with probability better than an honest prover on random
noise.

**Proof.** The honest operand values of gate $g$ are a deterministic function
of $(\kappa_W,x,a)$, all fixed by the header, which is logged before $\zeta$ is
released. $E,F$ are therefore independent of them, and each $r\times r$ block of
$A+E$ is uniform (Lemma 6.5). $\square$

The attack this rules out is the one that would otherwise make PoUW useless
against a self-dealing developer: with $\zeta$ known first, a circuit whose
first matmul takes $x$ directly lets the developer set $x=-E+S$ with $S$
sparse, making every tile of that gate a sparse product. The Lemma also covers
rejection sampling on $\zeta$: since every $\zeta$ yields uniform blocks, there
is no favorable seed to wait for.

**The hypothesis "honest values" is necessary.** A *wrong* committed operand
is chosen by the prover, and the boundary is committed after execution, hence
after $\zeta$. So the prover can corrupt the replay unit that produces an
operand block, set that block to $-E_{i\ell}+S$ with $S$ sparse, and obtain
every tile of the consuming gate's row block for free while keeping the
downstream transcript self-consistent (the decode returns $A^{\rm wrong}B$, the
correct product of the wrong operand). What limits this is Verity, not the
ordering: a replay unit with a corrupted operand block is caught with
probability $q(1-(1-s)^{r^2})\approx q$ when its RU is sampled, and *any*
rejected request is a detection event. So the number of such units the prover
can afford across the whole epoch before being caught with probability
$\ge1-\eta$ is $B:=\lceil\ln(1/\eta)/q\rceil$ (10 at the Verity draft's
calibration $q=1/2,\eta=10^{-2}$; 46 at $q=0.1$),
and the free tiles they yield number at most $B$ row blocks' worth, about
$B\,r\,k\,m$ MACs of credit, or $\sim10^{11}$ against $\Phi_T\sim10^{26}$. This
is attack 23 in §3.8 and the $B$ term of §3.7.

Why not commit the operands' hashes into the seed instead (§2.7)? That also
fixes the operands before the noise and is the paper's original design. It
costs an operand hash per gate; commit-then-seed costs a round trip. Both give
the Lemma.

## 3.4. S2: attribution

**Claim.** Under A3 and Verity's soundness theorem (security-argument §5A/§5E),
if no request in the epoch is rejected then, except with probability $\le\eta$,
at most $B=\lceil\ln(1/\eta)/q\rceil$ replay units across the epoch contain a
committed value that differs from the honest value of the declared circuit on
$(\kappa_W,x)$. Consequently, of the $M$ tiles counted in S1, all but at most
$B\cdot(\text{tiles consuming one RU's output})$ are tiles of the honest
evaluation of the declared circuits.

**Proof sketch.** The ticket proof and Verity's VU proofs open the *same
leaves* of the same commitments (§2.4 step 3: every non-weight $\mathrm{CMM}$
operand is a boundary value; weights are $\kappa_W$). Verity's acceptance
probability for a transcript with error set $E$ is
$\sigma(E)=\prod_r\bigl(1-q+q(1-s)^{|E\cap R_r|}\bigr)\le(1-q)^{c}$ where $c$ is
the number of RUs with an error large enough that $(1-s)^{|E\cap R_r|}\approx0$
(any error touching a whole $r\times r$ operand block qualifies). Sampling is
independent across requests, so the probability that *no* request is rejected
while $c$ such RUs exist across the epoch is $\le(1-q)^c$, which is $\le\eta$
once $c\ge B$. An operand of gate $g$ that is not the honest value is produced
by such an RU, and it can make cheap only the tiles that read it. A tile
whose operands are all honest values is, by definition, a tile of the honest
evaluation. $\square$

Errors too small to affect a whole operand block are caught with the smaller
probability $q(1-(1-s)^{|E\cap R_r|})$; they cannot make a tile cheap (the
block stays uniform up to a few entries) and are exactly what Verity's
capacity bound $U$ accounts for (S4). The statement is epoch-level rather than
per-request because a single rejection is a detection event; the adversary is
not playing $R$ independent games with $\eta$ each.

Where this matters is the *dual-use* attack (§3.8, attack 4): the developer declares
"inference on $W_0$" with operand $A$ equal to an activation of a training run
on $W_0+\Delta$, computes the tiles of $(A+E)(W_0+F)$ (real compute, tickets
issued), and separately adds the cheap low-rank correction $A\Delta$ to obtain
the training forward pass. Without operand binding, PoUW alone certifies this
as declared inference. With it, $A$ must be the honest $W_0$-activation on the
declared input, which differs from the training activation at every layer
after the first, so the training run pays for its own forward pass and the
declared run pays for its own, and the training compute is unaccounted.

## 3.5. S3: shape

The structural check is a fold over the index and needs no assumption: the
partition either is $(\Gamma,X,Y,N)$-bounded or is not. What needs care is what
boundedness *implies*.

**Theorem (heavy MACs per task; PBA Theorem A.3 in the circuit model).** Let
$\mathsf T$ be a task with import allowance $\bar X$ and export allowance
$\bar Y$ in matrix entries. Let $g$ be a matmul gate with operands $A$ ($n\times k$),
$B$ ($k\times m$). Let $\mathcal A=\{(i,c):A[i,c]\text{ imported by }\mathsf T\}$,
$\mathcal B=\{(c,j):B[c,j]\text{ imported by }\mathsf T\}$, and $\mathcal Z$ the set
of output coordinates $(i,j)$ of $g$ for which $\mathsf T$ exports a partial
sum or the value. Then the number of MACs $(i,c,j)$ of $g$ performed in $\mathsf T$
with both operands imported and result contributing to an exported coordinate is
$$
\le\sqrt{|\mathcal A|\,|\mathcal B|\,|\mathcal Z|}\le\frac{\bar X}{2}\sqrt{\bar Y}.
$$

**Proof.** Each such MAC is a triple $(i,c,j)$ whose three projections lie in
$\mathcal A$, $\mathcal B$, $\mathcal Z$; Loomis–Whitney bounds the number of
points of a finite set in $\mathbb Z^3$ by the geometric mean of its three
coordinate-plane shadows. $|\mathcal A|+|\mathcal B|\le\bar X$ and AM–GM give
$|\mathcal A||\mathcal B|\le\bar X^2/4$; $|\mathcal Z|\le\bar Y$. Summing over
gates in $\mathsf T$ with $\sum_g(|\mathcal A_g|+|\mathcal B_g|)\le\bar X$ and
$\sum_g|\mathcal Z_g|\le\bar Y$, the sum of $\sqrt{a_gb_gz_g}$ is maximized by
concentrating on one gate. $\square$

**Corollary (training is not admissible).** A training step of a model with
$P$ parameters on a batch of $t$ tokens performs $\approx 6Pt$ MACs, essentially
all of them with both operands accumulated (weights $W_t$, and activations and
gradients that depend on $W_t$). Under A7 none of these operands is baseline
or in-task regenerable, so each is imported by whichever task performs the
MAC, and (see the caveat below) each result contributes to an exported
coordinate. With $\bar X=10^7$, $\bar Y=10^6$ (PBA's $X=10$ MB, $Y=1$ MB at 8
bits), a task performs $\le5\times10^9$ heavy MACs, so a step of a $10^{12}$-parameter
model on $10^6$ tokens needs $\ge10^9$ tasks, against $N\approx10^7$ per day.
Inference (one task per request, imports $=$ prompt, exports $=$ response)
performs zero heavy MACs.

**Where the theorem is weaker than it looks.** Three hypotheses carry weight.

1. *"Result contributes to an exported coordinate."* A task can import heavy
   operands, perform a heavy matmul, pass the result through a nonlinearity
   and a *baseline* projection to a few values, and export those. Those MACs
   are heavy in operands but not in results, and the theorem does not count
   them; a construction with a narrow baseline projection performs
   $\sim\bar X\bar Y$ such MACs, a factor $\sqrt{\bar Y}$ more. But their
   results are collapsed to $\le\bar Y$ values before leaving, and a training
   step needs the *full* product (the gradient, the update) to leave or be
   applied. The clean statement is information-theoretic — a task's exports
   are $\le Y$ bits, a function of its imports and baseline — and the
   counting theorem is the matmul-shaped instance of it. Turning "training
   needs $P$ values of progress per step, and progress must leave tasks" into
   a lower bound robust to compressed and restructured training is the open
   theoretical problem PBA flags (Remark A.5); the merged protocol inherits
   it unchanged.
2. *A7 (non-regeneration).* A task that can rebuild $W_t$ from $W_0$ and a
   seed within $\Gamma$ performs no heavy MACs. That is the definition of a
   small fine-tune, and PBA permits it deliberately.
3. *Small heavy state.* If the accumulated state fits in $\bar X$ and $\bar Y$
   (a LoRA adapter of $\le10^6$ entries), it is imported and exported freely,
   the base weights are baseline, and every MAC has at most one heavy operand.
   The theorem does not bound this. See §3.9(i).

The merged protocol adds nothing to the theorem and takes nothing from it.
What it adds is that the theorem is now applied to a circuit whose evaluation
is *proved to have consumed the compute it claims* (S1) *on its honest values*
(S2), so that a developer cannot satisfy the structural check with a circuit it
did not actually run while running something else.

## 3.6. S4: Verity's capacity bound, unchanged

`Bound` is a fold over unit kinds and their downstream cuts. The
$\mathrm{CMM}_r$ interior introduces partial-dot-product VUs with a
one-accumulator cut (same width as today's dot products), noise-generation
gates with an *empty* effective cut (their corruption cancels in decode), sum
and decode gates whose cuts are the output entries they feed, and `lottery`
marks that do not affect cuts. So $\mathrm{Bound}(C,I,\theta)$ for a
$\mathrm{CMM}$-compiled circuit equals that of the plain-matmul circuit up to
the additive decode units. Had the *tile* been the VU, the cut per unit would be
$r^2$ accumulators and $U$ would grow by $r^2$; this is why §2.2 marks partial
dot products.

Ticket proofs are zero-knowledge and open positions Verity already opens
under existing owners, plus the public `seed` owner. The transparent
verifier's trust boundary is unchanged (§2.6). S4 is Verity's theorem
verbatim.

## 3.7. Composition

**Theorem (informal).** Under A1–A8, if the auditor accepts epoch $e$ (every
request accepted, $K\ge K_{\min}$, baseline frozen, partition bounded), then
except with probability $\eta_e\le\eta+\exp(-\Delta^2pM/3)+\mathrm{negl}$:

1. The hardware spent at least
   $(1-\delta)(1-\Delta)(1-\varepsilon_{\rm alg})\,\alpha\,\Phi_T-B\,rkm$
   MAC-equivalents on tiles of declared gates with honest operands (S1 with
   $K\ge K_{\min}$, less the $B$ row blocks of §3.3).
2. At most $B$ replay units across the epoch had any dishonest value (S2), so
   the compute in (1) is the honest evaluation of the declared circuits.
3. The declared circuits contain $\le\lambda\Phi_T$ heavy MACs (S3).
4. Each declared output carries $\le U_{\max}+A$ bits beyond its function (S4).

Hence the free compute satisfies
$$
F\;\le\;\bigl(\delta+\Delta+\varepsilon_{\rm alg}+\varepsilon_\alpha\bigr)\Phi_T
\;+\;\lambda\Phi_T\;+\;B\,r\,k\,m\;+\;\text{(§3.9 residuals)},
$$
where $B\,rkm\le46\cdot128\cdot8192\cdot32768\approx1.6\times10^{11}$ is
negligible against $\Phi_T\sim10^{26}$.
No $(1-\alpha)$ term appears: the adversary's *undeclared* work runs at raw
efficiency, but its *declared* work must run the same instrumented algorithm
to earn tickets, so $\alpha$ sets the honest baseline rather than giving the
adversary a discount. If the adversary has an implementation with tile
fraction $\alpha'>\alpha$, it passes with undeclared fraction up to
$1-(1-\delta)\alpha/\alpha'\approx\delta+(\alpha'-\alpha)/\alpha$; the excess
over $\delta$ is $\varepsilon_\alpha$.

**Proof.** (1) is S1 with $K\ge K_{\min}$ solved for $M$, less the tiles S2
does not vouch for; since the hardware performs at most $\Phi_T$
MAC-equivalents (A4), the remainder is $\Phi_T-(1)$. (2) is S2, which is
already an epoch-level statement (one rejection is detection). (3) is S3 times
$N$. (4) is S4. Free compute is by definition (remainder) + (heavy MACs) +
(the $B$ row blocks), and the residuals of §3.9 are the terms this accounting
does not see. $\square$

## 3.8. Attacks

Each row is a strategy the developer might use to increase $F$ while being
accepted; the outcome is what the analysis above says about it.

| # | Attack | What it exploits | Outcome |
|---|---|---|---|
| 1 | Forge tickets without computing tiles | Hash predictability | Blocked by A1+A2: a ticket needs the correct tile's hash; the tile costs $r^3$ (§3.2) |
| 2 | Replay one computed tile as many tickets | Missing freshness | Blocked: $\zeta_g$ differs per request and gate; uniqueness on $(j,g,i,\ell,j')$ |
| 3 | Choose operands after seeing noise ($x=-E+S$) | Seed ordering | Blocked by header-before-$\zeta$ (§3.3) |
| 4 | Dual use: declare "inference on $W_0$" with training activations as operands, add $A\Delta$ separately | Operand binding | Blocked by S2: operands must be honest $W_0$-activations, which differ from layer 2 on |
| 5 | Declare training as inference against a *new* $\kappa_W$ each step | Baseline freeze | Blocked by A6 (policy): new weights only as declared outputs (then S3 applies) or warden-observed ingress |
| 6 | Declare a training circuit with a $(\Gamma,X,Y,N)$ partition | Structural check | Blocked by S3 corollary for dense frontier training; **not blocked** for state $\le\min(X,Y)$ (§3.9 i) |
| 7 | Undeclared training on part of the fleet; declared work on the rest | Accounting threshold | Visible as ticket deficit; bounded to $\delta\Phi_T$ (§3.7). The bound is relative; §3.9 ii |
| 8 | Undeclared training in *idle* time of an honest inference workload | Idle = undeclared | Same as attack 7; the honest developer must fill idle compute or eat the gap (§3.9 v) |
| 9 | Faster tile algorithm (Strassen-like) or faster PoUW implementation | $\varepsilon_{\rm alg}$, $\varepsilon_\alpha$ | Directly increases $F$ by that fraction; not cryptographic (§3.9 iii) |
| 10 | Skip PoUW on some declared gates (plain GEMM, no tickets), spend savings on training | Unticketed declared work | Counted as unaccounted; falls under the $\delta$ budget. No gain beyond attack 7 |
| 11 | Cache non-matmul work (softmax etc.) on repeated or degenerate inputs | Non-matmul freshness | **Not detectable**; §3.9 iv. Handled by crediting tile MACs only |
| 12 | Adversarial model whose gates ignore most of the input (constant activations), so useful work is cacheable | Input-independence | Tiles still cost $r^3$ each under fresh noise (Lemma §3.3). Only the non-matmul part (attack 11) is saved |
| 13 | Filler gates on structured operands (zero, low-rank) | Operand structure | No saving: "the all-zeroes product is as hard as any other" once the noise has rank $r$ |
| 14 | Noise of lower rank than the tile (if the developer could choose $r_n$) | Cross-term shortcut | Protocol fixes $r_n=r$; §2.2 |
| 15 | Corrupt the noise gate ($E=0$) to compute plain tiles | Verity error budget | Zero capacity, but every ticket of the row block fails; charged to $B$; no gain |
| 16 | Commit wrong operands to make tiles cheap | Verity error budget | Operands are fixed before $\zeta$ regardless of correctness (§3.3), so no tile is cheaper; charged to $B$ |
| 17 | Choose advice/batching after $\zeta$ to align noise with operands | Late structure choice | Stacking fixed rows against fresh uniform noise cannot create structure; no gain |
| 18 | Grind $\zeta$ (submit headers, abandon unfavorable seeds) | Seed choice | No favorable seed exists (§3.3); abandoned headers cost the developer, not the auditor |
| 19 | Present tickets from rejected Verity runs | Validity | Void by definition; compute unaccounted |
| 20 | Exceed the inventory (hidden hardware) | A4 | $K>K_{\max}$ rejected; otherwise undetectable by the protocol; physical audit |
| 21 | Distribute training thinly across every device, below per-device statistical resolution | Statistical precision | Same $\delta$ total; per-device accounting (§4.3) only changes the granularity, not the total budget |
| 22 | Launder an undeclared adapter through Verity's covert capacity into a self-client's prompt, then declare "inference with adapter in prompt" | S4 capacity + adapter escape | Undeclared adapter *training* is unaccounted (attack 7); the *use* of the adapter is attack 6's admissible case |
| 23 | Corrupt the RU producing an operand block, choosing the block as $-E+S$ *after* $\zeta$, so the consuming row block's tiles are free | Boundary is committed after $\zeta$; wrong operands are the prover's choice | Real, but each such RU is caught w.p. $\approx q$ and any rejection is detection; $\le B$ (10–46) row blocks per epoch before detection w.p. $1-\eta$ (§3.3). Yield $\sim10^{11}$ MACs against $\Phi_T\sim10^{26}$ |

## 3.9. What survives: the residual an adversary keeps

The theorem in §3.7 bounds free compute by a sum of terms. Each term is
either a parameter the auditor sets or a genuine hole. The holes:

**(i) The small-state escape.** $(\Gamma,X,Y,N)$-boundedness permits any
computation whose task-to-task state fits in $\min(X,Y)$ bits, and the
merged protocol certifies such a computation as compliant *and* pays for it
with tickets. Rank-1 LoRA on a 7B model is $\sim10^6$ parameters; at 8 bits
it fits $Y=1$ MB. A declared task "forward + loss + backward + adapter
gradient, adapter in the prompt, adapter update as the response" is a
legitimate bounded task, and chaining $N$ of them is adapter training at full
declared compute. PBA's position is that adapters this small elicit rather
than teach; whether that holds is an empirical question about models, not a
protocol question. The protocol makes this visible (the circuit structure
shows a prompt-derived matrix reused as a weight) but does not forbid it, and
forbidding it structurally collides with in-context learning, where attention
also reuses prompt-derived matrices. Open problem §4.5(e).

**(ii) Relative, not absolute.** The bound is $(\delta+\ldots)\Phi_T$. For a
$10^6$-accelerator datacenter and $\delta=1\%$ that is $10^4$ accelerators for
the whole epoch, which trains a large model in months. Compute accounting
turns "the entire datacenter could be training" into "one percent of it could
be", which is a large improvement and still not "none". The other PBA
mechanisms (wipes every $\tau$ hours, interconnect limits) act on this
residual: training in the $\delta$ budget must checkpoint through $R$ bits of
residual memory or the covert channel of S4, and its parallelism is bounded by
the wardens. Whether the composition is tight is §4.5(f).

**(iii) Non-cryptographic slack.** $\varepsilon_{\rm alg}$ and $\varepsilon_\alpha$
convert one-for-one into free compute. The first is a statement about fast
matrix multiplication on specific hardware for $r\approx128$ integer tiles,
plausibly near zero on tensor cores and 10–20 % on hardware where Strassen
helps. The second is an engineering race: the auditor must know how fast the
instrumented matmul *can* run, or the developer keeps the difference. Both
must be stated as assumptions and revisited per hardware generation.

**(iv) Unevidenced non-matmul work.** Roughly 10–30 % of an inference step's
operations are not tile products and produce no tickets; for an adversarial
circuit they need not be fresh (attack 11). Crediting them at the architecture's
ratio trusts the architecture; crediting only tiles makes the honest developer
buy filler to cover them. Either way the fraction is a parameter the auditor
sets and the adversary gets.

**(v) Idle is undeclared.** Decode is memory-bound; an honest inference fleet
at 40–60 % tensor-core utilization must fill the remainder with filler to
present as compliant. The bound's tightness is set by the developer's
willingness to burn power, and by whether filler can actually co-run with
memory-bound decode on the same devices.

**(vi) The heavy-MAC definition.** §3.5's theorem counts MACs whose results
leave the task through the same matmul. Restructured or compressed training
that keeps heavy products in-task and exports something else is bounded only
by the information-theoretic statement, not by the counting theorem. This is
PBA's open problem; the merge does not close it.

**(vii) Exact arithmetic.** The scheme requires exact cancellation of noise
and therefore integer or prime-field matmuls. INT8 inference with 7-bit
operands and 7-bit noise works; FP16 does not. A developer whose serving stack
is FP16 must change it, and the quality cost of losing a bit of quantization
is borne by the honest party.

Nothing in (i)–(vii) is a break of S1, S2, S3 or S4 as stated. They are the
terms the theorem carries along and the places where its statement is weaker
than the Stage 0 question.

## 3.10. Why all three ingredients are necessary

Removing any one of Verity, PoUW, or the structural check leaves a concrete
attack.

- **Verity + structure, no PoUW.** Verity certifies consistency, not
  freshness. The developer declares $10^6$ identical requests (or a model whose
  activations ignore the prompt), computes once, replays the transcript, and
  claims $10^6\times$ the compute. Every sampled gate is correct. The auditor
  could demand distinct inputs, but the adversary designs the model, so
  distinct inputs need not mean distinct work. PoUW's fresh, rank-$r$ noise is
  what makes every declared tile cost $r^3$ regardless of the model or the
  inputs (§3.3, attacks 12 and 13).
- **PoUW + structure, no Verity.** Tickets bind tiles to operands the prover
  commits, not to the circuit's honest values. The developer commits training
  activations as the operands of "inference on $W_0$" gates and earns tickets
  for the forward pass of a training run on $W_0+\Delta$ (attack 4). More simply, it
  commits *any* matmul it performs as some declared gate's operands; the
  declared circuit is fiction and the tickets are real. Verity's sampling is
  what ties committed operands to the declared DAG.
- **Verity + PoUW, no structure.** The declared circuit is proven to have been
  run, freshly, on honest values, and it is a training loop. The frozen
  baseline (A6) stops it from *using* the result as a model in later declared
  inference, but the accumulated weights can sit as a declared circuit output
  waiting for the monitoring regime to end. The $(\Gamma,X,Y,N)$ check is what
  says the declared circuit cannot have been that loop.

And the two pieces the merge does not include: PBA's wardens and wipes, which
act on the $\delta\Phi_T$ residual and on undeclared state; and physical
inventory (A4), without which $\Phi_T$ is unknown and S1 bounds nothing.
