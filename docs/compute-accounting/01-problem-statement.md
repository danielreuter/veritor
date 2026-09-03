# 1. Problem statement: compute-accounted bounded accumulation

This document works out, in stages, what it would mean to combine three
things: Verity (sparse zero-knowledge verification of a committed circuit,
currently framed around bounding covert exfiltration), the Komargodski–Weinstein
proof of useful work for matrix multiplication (arXiv 2504.09971, "PoUW"), and
the bounded-accumulation framing of the "Proofs of bounded accumulation" post
("PBA"). Each stage states a candidate problem, shows what is wrong or vague
about it, and refines. The final section fixes definitions that the protocol
(§2) and the security analysis (§3) use.

Notation follows the Verity draft where possible: a circuit $C$ with gates
$1,\ldots,n$, replay units (RUs) and verification units (VUs), sampling rates
$\theta=(q,s)$, threshold $\eta$, acceptance probability $\sigma(E)$ of an error
set $E$, weight commitment $\kappa_W$, boundary $\partial$, and capacity bound
$U=\mathrm{Bound}(C,I,\theta)$. Because $\sigma$ is taken, the PoUW seed is
written $\zeta$. PBA's per-task allowances $(C,X,Y,N)$ are written
$(\Gamma,X,Y,N)$ to avoid clashing with the circuit $C$.

## 1.1. Stage 0: the naive question

> Can a developer prove to an auditor that its datacenter is only doing
> inference, not training, without revealing its models, code, or data?

Two words are undefined: *only* and *inference*. "Only" is a statement about
everything the hardware did, including work the developer never mentions.
"Inference" is a statement about the shape of a computation, and every attempt
to pin it down (reads a fixed model, small prompt in, small response out) is a
statement about *declared* computations. The two halves of the question live
on different objects, and the gap between them is where every attack lives.

## 1.2. Stage 1: what Verity alone gives, and does not

Verity takes a declared computation $(F,x)$, compiles it to a circuit $C$,
and certifies that the claimed output $y^*$ can carry at most $U$ bits beyond
$F(x)$. This is a strong statement about the declared computation and a null
statement about undeclared ones. In particular:

- Verity does not know how many computations were run. It sees the ones the
  server chose to submit.
- Verity does not know that a computation cost anything. A transcript that is
  consistent with the circuit is accepted whether it was computed fresh,
  replayed from a cache, or (for an adversarially designed circuit whose gates
  ignore most of the input) computed once and reused a million times. The
  sampled gates are correct in every case.
- Verity's weights are pinned by $\kappa_W$. So an inference request against a
  *trained* model $W_t\ne W_0$ cannot be declared as inference on $W_0$: the
  matmul gates would be incorrect at every layer, far above the budget $B$ of
  tolerated incorrect units. This is the one structural fact Verity contributes
  to the accumulation problem, and it is important: whatever is declared runs
  against the frozen baseline, or is caught.

So Verity answers "is the declared work what it says it is?" and cannot
answer "how much undeclared work was there?".

## 1.3. Stage 2: the compute-accounting reframing

The only quantity an auditor can bound from the outside is compute. A
datacenter has an inventory $H$ of accelerators with a known aggregate rate
$\Phi$ (MAC-equivalents per second) and the monitoring period has length $T$;
write $\Phi_T:=\Phi T$ for the compute available in the period. If the
developer can prove that at least $(1-\delta)\Phi_T$ of it was spent on
declared, inference-shaped work, then at most $\delta\Phi_T$ was available for
anything else, training included.

> **Stage 2 problem.** Prove that at least $(1-\delta)\Phi_T$ MAC-equivalents
> were consumed by the honest evaluation of declared circuits that are
> inference-shaped.

Three sub-problems fall out, and they map onto the three source documents:

1. **Consumption.** Prove that compute was actually spent. Verity cannot (a
   consistent transcript proves nothing about freshness). PoUW can: a
   certificate stream whose rate is proportional to the number of tile products
   performed under fresh noise.
2. **Attribution.** Prove that the compute in (1) was spent on the *declared*
   circuit's *honest* evaluation, not on some other computation that happened
   to touch the same operands. PoUW cannot (its verifier is told $A,B$ by the
   prover). Verity can: its sampling binds the committed operands of every
   gate to the circuit's honest values, up to a small number $B$ of replay
   units per epoch (§1.8).
3. **Shape.** Prove that the declared circuit is inference-shaped. Neither
   Verity nor PoUW says anything here. PBA's $(\Gamma,X,Y,N)$-bounded
   accumulation is a structural predicate on a partition of the circuit, and
   Verity's replay-unit boundary is exactly the interface across which it is
   measured.

## 1.4. Stage 3: what "inference-shaped" means on a circuit

PBA defines accumulation informally as "the dependence of later computation
on the results of prior computation" and makes it quantitative through tasks:
a computation is $(\Gamma,X,Y,N)$-bounded if it can be carried out by at most
$N$ processes, each performing at most $\Gamma$ operations, reading at most
$X$ bits of *novel* information (anything not in the committed baseline) and
writing at most $Y$ bits of it. Training at frontier scale cannot be split
this way: every step reads and rewrites terabytes of state, and a 10 MB pipe
between 10 H100-hour tasks does not carry it. Inference can: one request reads
the baseline model and a prompt, computes, and emits a response.

On the epoch circuit this becomes a statement about a partition. Let $C_e$ be
the disjoint union of all declared circuits in the epoch, all reading the same
$\kappa_W$. A *task partition* $\mathcal T$ is a partition of the non-source
gates of $C_e$ into tasks. For a task $\mathsf T$:

- its *compute* is the number of MAC-equivalents of gates in $\mathsf T$;
- its *imports* are the values it reads that are neither `weight` gates (baseline,
  read from $\kappa_W$) nor produced inside $\mathsf T$: values produced by
  other tasks, and the request's `in` gates (the prompt is novel information
  from outside);
- its *exports* are the values produced in $\mathsf T$ and read by another task
  or by the circuit output.

$\mathcal T$ is $(\Gamma,X,Y,N)$-bounded if every task has compute
$\le\Gamma$, import width $\le X$, export width $\le Y$, and $|\mathcal T|\le N$.

Two observations that make this tractable:

- For independent inference requests, the natural partition is one task per
  request. There is then no task-to-task flow at all: imports are the prompt,
  exports are the response, and the KV cache stays inside the task. In
  Verity's implementation the request's replay units already declare the
  cross-unit values (`Out(R_r)`), so a task is a union of RUs and its import
  and export widths are sums over the index, computable without touching a
  gate, in the same way `Bound` and `Cost` are.
- The developer proposes the partition, as it proposes $\theta$ and the RU/VU
  marks. The auditor only checks it. A bad partition can only hurt the
  developer (fewer admissible circuits), never soundness, because the check is
  on the actual DAG.

What the predicate buys is PBA's Appendix A: within one task, the number of
scalar multiplications whose operands are *both* imported is at most
$\tfrac{\bar X}{2}\sqrt{\bar Y}$ (Loomis–Whitney on the three shadows of the
set of $(i,k,j)$ triples), where $\bar X,\bar Y$ are the allowances in matrix
entries. Training multiplies accumulated weights by accumulated activations;
neither is baseline, neither can be regenerated inside a $\Gamma$-bounded
task, so both must be imported, and the bound bites. Inference multiplies
baseline weights by in-task activations and needs zero such products. We
return to the precise hypotheses of this theorem in §3.5, because they are
where "inference-shaped" is weakest.

## 1.5. Stage 4: the binding problem

Consumption and attribution have to be tied together at the level of a
single tile product, or the adversary decouples them. Consider the four ways a
PoUW certificate could fail to mean what we want:

1. **Free certificates.** A certificate produced without computing the tile.
   PoUW's transcript-unpredictability assumption rules this out, provided the
   tile's operands are fixed before the noise seed is known.
2. **Replayed certificates.** The same tile, computed once, certified many
   times. Fresh noise per request rules this out, provided the noise depends
   on a per-request seed and the auditor de-duplicates on
   (request, gate, tile).
3. **Cheap operands.** The adversary arranges for the *noised* operands to be
   structured (e.g. sets an activation to $-E+S$ with $S$ sparse), making the
   tile products cheap. This needs the operands chosen after the noise. The
   fix is ordering: the request header (which fixes the circuit, the inputs
   and, through $\kappa_W$, every *honest* operand value) is committed before
   the auditor releases $\zeta$. A *dishonestly* committed operand can still
   be chosen after $\zeta$; that is a Verity error, and the number of such
   operands the developer can afford before a rejection becomes near-certain
   is what bounds the attack (§3.3, attack 23).
4. **Misattributed certificates.** A tile genuinely computed as part of some
   *other* computation (a training step on $W_0+\Delta$, say) presented as a
   tile of a declared inference gate. The certificate's proof must open the
   tile's operands from the *same commitment that Verity samples*, so that
   "the committed operand of gate $g$" and "the value Verity checks the
   producer of that operand against" are one object. Then a misattributed
   tile requires a wrong committed operand, which is a Verity error and is
   charged against $B$.

The fourth point is the crux of the merge. Without it, PoUW is a proof that
*some* matrix multiplications happened; with it, it is a proof that *these*
gates of *this* circuit were evaluated. It also fixes the seed's position in
the protocol: after the header, before execution, and independent of the
boundary commitment (which Verity requires to be fixed before its own seeds,
but which may be fixed after $\zeta$).

## 1.6. Stage 5: the accounting problem

Certificates only ever count *declared* compute. The auditor's threshold must
therefore be set against the hardware, $\Phi_T$, not against the declared
circuits' cost: a developer who declares little and trains a lot would match
any threshold set against its own declarations. Two consequences:

- **Idle compute is indistinguishable from undeclared compute.** An honest
  developer at 40 % utilization presents the same certificate count as a
  dishonest one spending 60 % on training. The honest developer must
  therefore *fill* idle compute with declared filler work (PoUW on arbitrary
  matrices is as expensive as on real ones, and is trivially
  inference-shaped), or accept that the gap is counted against it. This is
  the "burn to prove" cost of the approach, and it is real: decode is
  memory-bound, so an inference datacenter runs its tensor cores at well below
  peak, and filling that gap costs power.
- **Only tile products are evidenced.** Softmax, normalization, activation
  functions, the KV-cache reads and the rest of an inference step produce no
  certificates, and for an adversarial circuit their freshness cannot be
  established (a circuit whose softmax input ignores the prompt can cache
  it). So either the auditor credits them by trusting the architecture's
  matmul-to-other ratio, or it credits tile products only and the honest
  developer makes up the difference with filler. §4 quantifies this.

## 1.7. Stage 6: the game

The adversary is the developer. It controls the hardware and what runs on
it, the models (hence the circuits), the inputs (it can send itself
requests), the advice, $\theta$, the task partition, and the timing of
everything except the auditor's seeds. The auditor controls its seeds
$(\zeta,q\text{-seed},s\text{-seed})$ per request, knows the inventory $H$,
holds the frozen baseline commitments, and verifies.

**The accumulation game** $\mathcal G(\Phi_T,\delta,\Gamma,X,Y,N)$.

1. *Setup.* The developer commits the baseline: $\kappa_W$ for every model,
   the constructor set $\mathcal G$. The auditor fixes $(\delta,\Gamma,X,Y,N)$,
   the tile size $r$, the win probability $p$, Verity's $(\eta,U_{\max},A)$,
   and the reference efficiency $\alpha$ (§1.8).
2. *Epoch.* Repeatedly: the developer submits a header $(G,x,a,\theta)$; the
   auditor returns $\zeta$; the developer executes with PoUW-instrumented
   matmul gates, then runs the Verity protocol and submits the request's
   winning tickets; the auditor verifies the run and the tickets.
3. *Close.* The developer submits a task partition $\mathcal T$ of the
   epoch circuit. The auditor checks (i) every run was accepted, (ii) the
   ticket count $K$ is at least $K_{\min}(\Phi_T,\alpha,\delta,p,r)$, (iii)
   no baseline commitment changed, (iv) $\mathcal T$ is
   $(\Gamma,X,Y,N)$-bounded. It accepts the epoch or rejects.
4. *Payoff.* Let $F$ be the developer's *free compute*: the MAC-equivalents its
   hardware performed during the epoch that were not part of the honest
   evaluation of a declared circuit, plus the heavy MACs (§1.8) inside
   declared circuits. The developer wants to maximize $F$ subject to
   acceptance.

**Security goal.** For every developer strategy,
$$
\Pr\bigl[\text{accept}\ \wedge\ F>(\delta+\lambda)\,\Phi_T+\text{slack}\bigr]\le\eta_e,
$$
where $\lambda:=N\cdot\tfrac{\bar X}{2}\sqrt{\bar Y}\,/\,\Phi_T$ is the
heavy-MAC fraction the partition permits, $\eta_e$ is the epoch failure
probability, and the slack terms (Verity's $B$ replay units, statistical error in
$K$, algorithmic and implementation efficiency) are made explicit in §3.

This is deliberately *not* the exfiltration game of the Verity draft (§3
there). That game bounds what leaves the datacenter; this one bounds what the
datacenter could have been doing. They compose: Verity's per-request
capacity bound $U_{\max}+A$ is one of the four properties the merged protocol
certifies (§3.1, S4), so covert channels through declared outputs remain
bounded, and PBA's wardens and wipes bound the rest.

## 1.8. Definitions used by §2–§4

**Hardware.** An inventory $H$ of devices with aggregate rate $\Phi$
MAC-equivalents per second. A *MAC-equivalent* is the time of one
tensor-core multiply-accumulate at the device's declared precision; other
operations are converted at the device's throughput ratios. $\Phi_T=\Phi T$.

**Baseline.** $\kappa_W$: per-model Merkle commitments to the weight vector,
frozen for the epoch; the constructor set $\mathcal G$; the gate set. Anything
not derivable from these and a request's public inputs is *novel*.

**Canonical matmul gate** $\mathrm{CMM}_r$. A gate with operands
$A\in\mathbb Z^{n\times k}$, $B\in\mathbb Z^{k\times m}$, a public seed
$\zeta_g$, and output $A\cdot B$ computed exactly. Its interior is the
PoUW algorithm: rank-$r$ noise $E=E_LE_R$, $F=F_LF_R$ derived from $\zeta_g$,
the $(n/r)(k/r)(m/r)$ running-sum tiles $C'^{(\ell)}_{ij}$ of
$\mathrm{MatMul}_r(A+E,\,B+F)$, and the exact decode
$C=C'-(AF_L)F_R-E_L(E_R(B+F))$. Details in §2.2.

**Ticket.** For request $j$, gate $g$, tile index $(i,\ell,j')$: the value
$h=\mathcal O(\zeta_g,i,\ell,j',C'^{(\ell)}_{ij'})$. The tile *wins* if
$h<p\cdot2^\lambda$. A *ticket* is $(j,g,i,\ell,j',h,\pi)$ where $\pi$ is a
zero-knowledge proof that $h$ is the hash of the correct running-sum tile for
the operands committed in request $j$'s boundary and $\kappa_W$ under the
noise derived from $\zeta_g$. Tickets are *valid* if $\pi$ verifies, the
request was accepted by Verity, and $(j,g,i,\ell,j')$ is unique.

**Ticket count and threshold.** $K$ is the number of valid tickets in the
epoch. The auditor's threshold is
$$
K_{\min}:=p\cdot\frac{(1-\delta)\,\alpha\,\Phi_T}{r^3}\;-\;z\sqrt{p\cdot\frac{\alpha\,\Phi_T}{r^3}},
$$
where $\alpha$ is the fraction of a fully utilized device's MAC-equivalents
that the best known PoUW-instrumented implementation spends inside tile
products (the remainder is noise generation, hashing, decode), and $z$ sets
the honest false-rejection rate.

**Task partition; imports, exports, heavy MACs.** As in §1.4. A MAC in task
$\mathsf T$ is *heavy* if both of its operands are imported by $\mathsf T$
(neither is a `weight` gate nor produced in $\mathsf T$). $\lambda$ is the
heavy-MAC fraction the partition admits.

**Free compute** $F$. MAC-equivalents performed by $H$ during the epoch and
not attributable to the honest evaluation of an accepted declared circuit,
plus heavy MACs inside accepted declared circuits. "Attributable" is made
operational in §3.2 and §3.4: a MAC-equivalent is attributed if it is one of
the $r^3$ MACs of a tile whose ticket-eligible hash was computed, on operands
equal to the honest values of the declared circuit.

**Four certified properties.** The merged protocol is meant to certify, for
an accepted epoch:

- **S1 (consumption)** at least $(1-\delta)\alpha\Phi_T$ MAC-equivalents were
  spent computing running-sum tiles of declared $\mathrm{CMM}_r$ gates on
  their committed operands under noise derived from the auditor's seeds;
- **S2 (attribution)** for all but at most $B=\lceil\ln(1/\eta)/q\rceil$
  replay units across the epoch, those committed operands are the honest
  values of the declared circuit on $(\kappa_W,x)$ ($B\approx10$ at the Verity
  draft's calibration $q=1/2,\eta=10^{-2}$, 46 at $q=0.1$; the statement is
  epoch-level because a single rejected request is a detection event);
- **S3 (shape)** the epoch circuit admits the declared
  $(\Gamma,X,Y,N)$-bounded partition, so it contains at most
  $\lambda\Phi_T$ heavy MACs;
- **S4 (exfiltration)** every declared output carries at most $U_{\max}+A$
  bits beyond its declared function (Verity, unchanged).

S1–S3 together give the security goal of §1.7. §3 states the assumptions
under which each holds, gives proof sketches, and then lists what the
adversary can still do.
