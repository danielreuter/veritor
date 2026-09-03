# 2. The merged protocol

This section gives the protocol that §3 analyzes. It is Verity's two-stage
protocol with three additions: a third seed released before execution, a
canonical matmul gate whose interior is the PoUW algorithm, and a ticket
message whose proofs open the same commitments Verity samples. Epoch-level
accounting sits on top.

## 2.1. Three seeds and where they sit

Verity has two verifier seeds, revealed after the commitments they sample:
the $q$-seed after the boundary, the $s$-seed after the interiors. The merged
protocol adds $\zeta$, revealed *before execution* and *after the header*.
The order matters and each constraint has one reason:

| Seed | Revealed after | Revealed before | Why |
|---|---|---|---|
| $\zeta$ | header $(G,x,a,\theta,\kappa_W)$ | execution | PoUW needs the instance fixed before the noise (no cheap-operand attack); the prover needs the noise to execute |
| $q$-seed | boundary commitment | interior commitments | Verity: RU sample drawn from a fixed boundary |
| $s$-seed | interior commitments | VU proofs | Verity: VU sample drawn from fixed interiors |

$\zeta$ is independent of the boundary, so revealing it early does not weaken
Verity: Verity's soundness (security-argument §5A) only needs the boundary
fixed before the $q$-seed, and the header fixes every honest operand value as
a deterministic function of $(\kappa_W,x,a)$ whether or not $\zeta$ is known.
$\zeta$ can be a per-request response from the auditor or a public beacon value
$\zeta_t$ used by every header logged in $[t-1,t)$; the only requirement is that
the header is bound (logged or committed to the auditor) before $\zeta$ is
known to the developer.

Per-gate seeds are derived, not transmitted:
$\zeta_g:=\mathcal O(\zeta,\mathrm{header\ digest},g)$ for each matmul gate
address $g$. Two requests with identical inputs therefore have different noise,
and two gates in one request have different noise.

## 2.2. The canonical matmul gate $\mathrm{CMM}_r$

**Interface.** Operands $A\in\mathbb Z^{n\times k}$ (activations, an interior or
boundary value of the producing unit) and $B\in\mathbb Z^{k\times m}$ (either
`weight` gates read from $\kappa_W$, or another computed value, as in
attention). Output $C=A\cdot B$, exactly. As a *circuit gate* it is an ordinary
matmul; the PoUW machinery is entirely interior, and the gate's semantics do
not mention $\zeta$.

**Arithmetic.** Exact. The noise must cancel exactly in the decode, so the
gate is defined over integers (fixed point) or a prime field, not over floating
point. With INT8 tensor cores: quantize operands to 7 bits, draw noise entries
uniformly from 7 bits, so $A+E$ and $B+F$ are 8-bit, products are 16-bit,
$k$-term sums fit INT32 (for $k\le2^{15}$), and the whole computation is exact
in $\mathbb Z$. Block-scaled formats (MXFP8, NVFP4) are exact within a block
and can be handled with block-integer accumulation; FP16 with FP32 accumulation
cannot be used directly. This is the same restriction Attestable-style ZK
systems impose, for a different reason.

**Interior.** With $r\mid n,k,m$ and $\zeta_g$ public:

1. *Noise.* $E_L\in\mathbb Z^{n\times r},E_R\in\mathbb Z^{r\times k},F_L\in\mathbb Z^{k\times r},F_R\in\mathbb Z^{r\times m}\leftarrow\mathrm{PRG}(\zeta_g)$;
   $E:=E_LE_R$, $F:=F_LF_R$; $A':=A+E$, $B':=B+F$. Cost $O((nk+km)r)$.
2. *Tiles.* For $i\in[n/r]$, $j\in[m/r]$, $\ell\in[k/r]$:
   $C'^{(\ell)}_{ij}:=C'^{(\ell-1)}_{ij}+A'_{i\ell}B'_{\ell j}$, with $C'^{(0)}_{ij}=0$.
   Cost $nkm$ MACs, the useful work.
3. *Lottery.* After each tile: $h_{i\ell j}:=\mathcal O(\zeta_g,i,\ell,j,C'^{(\ell)}_{ij})$.
   Cost $r^2$ hash inputs per $r^3$ MACs (§4.1 on why this is the dominant overhead).
4. *Decode.* $C:=C'^{(k/r)}-(AF_L)F_R-E_L\bigl(E_R\,B'\bigr)$. Cost $O((nk+nm+km)r)$.

The noise rank equals the tile size $r$. This is not a convenience: if the
noise had rank $r_n<r$, every $r\times r$ block of $E$ would have rank
$\le r_n$, and the cross terms $E_{i\ell}B'_{\ell j}$ could be computed in
$r^2r_n$ rather than $r^3$, giving an adversary who does not need the useful
term $A_{i\ell}B_{\ell j}$ (because $A$ is cached, or zero) a factor $r/r_n$
discount on every ticket. With $r_n=r$ each block of $A'$ is marginally uniform
(PoUW Lemma 6.5) and the cross terms cost as much as the useful term.

**Verity marks inside the gate.** Three granularities, for three purposes:

- *Replay unit:* row block $i$ of the gate (rows $ir..(i+1)r$ of $A$ against
  all of $B$), $r\,k\,m$ MACs. Self-contained given $E_L[i],E_R,F_L,F_R$ and its
  operands, all of which are boundary values or $\kappa_W$. Its declared
  outputs are its $r\times m$ rows of $C$.
- *Verification unit:* one partial dot product inside a tile,
  $C'^{(\ell)}_{ij}[a,b]=C'^{(\ell-1)}_{ij}[a,b]+\sum_c A'_{i\ell}[a,c]B'_{\ell j}[c,b]$,
  $r$ MACs behind one accumulator value. Its downstream cut is that single
  accumulator, so `Bound` sees the same 32-bit cut per unit it sees for a
  dot product today. (Marking the whole $r\times r$ tile as a VU would give it
  an $r^2$-value cut and multiply the capacity bound by $r^2$; §3.6.)
- *Lottery unit:* the $r\times r$ running-sum tile $(i,\ell,j)$, a set of
  $r^2$ VUs. This is a new mark, `lottery`, on the tile kind; it tells the
  compiler which interior values are hashed and which VUs a ticket proof
  covers. It is not sampled by the verifier; it is sampled by $\mathcal O$.

The noise, sum and decode gates are ordinary interior gates. A corrupted noise
gate has *zero* capacity (the same wrong $E$ is used in encode and decode, so
$C=AB$ regardless) but invalidates every ticket of its row block, since the
ticket proof recomputes the noise from $\zeta_g$. It is charged against $B$
like any incorrect unit.

**Dead noise.** For a gate with no downstream consumer of $C$ (filler), the
decode step is dead and may be omitted; `Bound` already assigns dead gates the
empty cut.

## 2.3. Tickets

A tile *wins* if $h_{i\ell j}<p\cdot2^\lambda$. For a winning tile the prover
produces a ticket $(j,g,i,\ell,j',h,\pi)$ where $\pi$ is a zero-knowledge proof
of the following statement, with public inputs
$(\text{boundary root of request }j,\ \kappa_W,\ \zeta_g,\ i,\ell,j',\ h)$:

> There exist operand blocks $A_{i1},\ldots,A_{i\ell}$ opened from the
> boundary commitment at the positions the circuit assigns to gate $g$'s
> first operand, and $B_{1j'},\ldots,B_{\ell j'}$ opened from $\kappa_W$ (or
> from the boundary, when the second operand is itself computed, as in
> attention) at the positions of its second operand, such that with $E,F$
> derived from $\zeta_g$,
> $T=\sum_{\ell'\le\ell}(A_{i\ell'}+E_{i\ell'})(B_{\ell'j'}+F_{\ell'j'})$ and
> $h=\mathcal O(\zeta_g,i,\ell,j',T)$.

The proof recomputes $\ell$ tile products ($\ell r^3\le kr^2$ MACs) and opens
$2\ell$ blocks of $r^2$ values each. Since the statement is exact-arithmetic
matmul plus Merkle openings, it is the most SNARK-friendly workload there is;
§4.2 costs it. Tickets for one request are batched into one proof.

What the auditor learns from a ticket: that tile $(i,\ell,j')$ of gate $g$ won.
The winning positions are uniform over tiles ($\mathcal O$ is a random oracle
on an input the prover does not control), so they carry no information about
the values; $h$ is a random-oracle output on a hidden input; $\pi$ is
zero-knowledge. The auditor learns the *count* per gate, which is
$\mathrm{Binomial}(nkm/r^3,p)$ and reveals only the gate's shape, already
public in $(C,I)$.

**Validity.** A ticket is valid if $\pi$ verifies, the request's Verity run
was accepted, and no other valid ticket has the same $(j,g,i,\ell,j')$.
Tickets of rejected requests are void: their compute is unaccounted and counts
against the developer.

## 2.4. Per-request flow

The merged run, with the new steps marked.

~~~text
Prover                                          Verifier / Auditor
------                                          ------------------
1. header = (G, x, a, theta, eta, kappa_W)  -->  log header; check advice <= A,
                                                 Bound(C,I,theta) <= U_max,
                                                 work <= W_max
                                            <--  zeta                      [NEW]
2. execute C with CMM_r interiors,
   deriving zeta_g per matmul gate;
   hash every running-sum tile;
   record winning tiles                                                    [NEW]
3. commit boundary tau|_d (includes every
   CMM operand that is not a weight)        -->  open public I/O; check vs header
                                            <--  q-seed
4. replay sampled RUs; commit interiors     -->
                                            <--  s-seed
5. prove sampled VUs                        -->  verify VU proofs
6. prove winning tiles against the same
   boundary / kappa_W roots                 -->  verify ticket proofs;
                                                 record (j,g,i,l,j') set    [NEW]
                                                 accept run iff 3-6 pass
~~~

Two remarks on step 3. First, "every CMM operand that is not a weight" is
already a boundary value in Verity whenever the operand crosses a replay-unit
boundary, which it does for the natural per-layer or per-row-block RUs; the
requirement is that it *always* is, so the ticket proof and the Verity checks
open the same leaf. Second, this does not commit the tiles themselves: the
interior of a row block is committed only if the RU is sampled, exactly as
before. Ticket proofs recompute their tiles from the operands, so no tile is
ever stored.

Step 6 may run after step 5 or be merged into it. Ticket proofs are
unconditional (every winning tile must be proved), while VU proofs are sampled.

## 2.5. Epoch accounting

At the close of the epoch the auditor holds, for every accepted request, its
$(C_j,I_j)$ and its set of valid tickets, and it holds the frozen baseline
commitments. It checks:

1. **Count.** $K\ge K_{\min}$ with $K_{\min}$ as in §1.8, and also
   $K\le K_{\max}$, the count consistent with $H$ running PoUW at 100 % of
   peak (a larger count is inconsistent with the inventory and is rejected as
   evidence of undeclared hardware or a broken assumption).
2. **Uniqueness.** No two valid tickets share $(j,g,i,\ell,j')$; no ticket
   references a rejected request or a header not logged before its $\zeta$.
3. **Frozen baseline.** Every accepted header's $\kappa_W$ is one of the
   commitments registered at setup. New weights can enter only as declared
   circuit outputs (and are then subject to the partition check, §3.5) or
   through warden-observed ingress, which is PBA's concern.
4. **Partition.** The developer's proposed $\mathcal T$ is a partition of the
   epoch circuit into tasks; each task's compute, import width and export
   width are computed as folds over the requests' indices (a task is a union
   of replay units; its imports are the boundary positions it reads that
   belong to other tasks or to `in` gates; its exports the boundary positions
   other tasks read); $|\mathcal T|\le N$; every task within $(\Gamma,X,Y)$.
   For independent inference requests this is one task per request and the
   check is per-request, at admission.

If the developer's real workload leaves tensor cores idle (decode is
memory-bound), it fills the gap with **filler**: declared requests whose
circuit is a $\mathrm{CMM}_r$ gate on arbitrary (say zero) operands, with no
consumer. Filler has no decode, produces tickets at the full rate, and is
trivially one bounded task. Its cost is power.

## 2.6. What changes in `veritor`

Concretely, against the current `src/veritor`:

- `constructors`: a `PoUWMatmulG(r)` beside `MatmulG`, tracing the interior of
  §2.2 with `replay` marks on row blocks, `verification` marks on partial dot
  products, and a new `lottery` mark on tiles. The noise generator is a gate
  kind whose single argument is a `seed` source gate (a new zero-arity source
  like `in`/`weight`, whose value the verifier fixes from $\zeta$ and the gate
  address rather than from the header).
- `core/index.py`: `validate_marks` admits `lottery` as a mark that must lie
  inside one replay unit and be tiled by verification units; `Index` exposes
  per-kind tile counts so the expected ticket count of a compiled circuit is a
  fold.
- `protocol`: a `seed` phase between header acceptance and boundary commitment;
  a `tickets` message after the evidence; `_Layout.owner` gives `seed` gates
  their own owner (public, derived); `verify_transcript` re-derives
  $\zeta_g$ and checks ticket proofs against the recorded roots.
- `analysis/cost.py`: terms for noise generation, hashing, and decode
  ($O((nk+km+nm)r)$ and $nkm/r$ hash inputs per gate); `analysis/bound.py` is
  unchanged in form, since the new kinds have singleton cuts (partial dot
  products) or are dead (noise, decode of filler).
- a new `analysis/accumulation.py`: the $(\Gamma,X,Y,N)$ fold over a proposed
  task partition, reporting per-task compute, import and export widths and the
  heavy-MAC bound $\tfrac{\bar X}{2}\sqrt{\bar Y}$.
- an epoch-level `Account(requests, tickets, H, alpha, delta, p, r)` that
  returns $K$, $K_{\min}$, $K_{\max}$ and the verdict.

None of this touches the transparent verifier's trust boundary: the compiler
still produces $(C,I)$ from the description bytes, the verifier still derives
every domain and challenge itself, and the ticket proofs open positions under
existing owners plus the public `seed` owner.

## 2.7. Alternatives considered

- **Operand hash in the seed** ($\zeta_g=\mathcal O(\zeta,H(A_g))$, PoUW's
  original $\mathcal O(\sigma,A,B)$) instead of committing the header before
  $\zeta$. Also sound, and it removes the round trip, at the cost of hashing
  every matmul operand during execution (about $1/m$ of the gate's MACs) and
  of making the boundary commitment's operand subtrees the roots those hashes
  must equal. Commit-then-seed is simpler and reuses Verity's header. Either
  works; §3.3 needs only that the operands are fixed before the noise.
- **Tile products rather than running sums as lottery units.** Hashing
  $A'_{i\ell}B'_{\ell j}$ instead of $C'^{(\ell)}_{ij}$ would let a ticket proof
  recompute one tile product instead of $\ell$. But the running sum is what a
  GEMM holds in registers; the individual product is never materialized, and
  materializing it costs an extra $r^2$ store and add per tile. The paper's
  choice is the GEMM-friendly one; the proof cost ($\le kr^2$ MACs per ticket)
  is affordable (§4.2).
- **Self-canceling noise** (PoUW Appendix A: $A'=AR$, $B'=R^\top B$ with a
  fast pseudorandom rotation, no decode). Removes the $O((nk+km+nm)r)$ decode
  term, which for transformer shapes is the larger half of the overhead. It
  rests on a newer conjecture (A.1) and needs the operands to be high-rank,
  which an adversarial *model* need not be. Listed as an open problem (§4.5(a)).
- **Lottery as Verity's VU sample.** Winning tiles are a uniform random sample
  of tiles at rate $p$, proved unconditionally, so they are extra VU checks
  for free. But at the $p$ that makes accounting cheap ($10^{-9}$–$10^{-7}$)
  the extra factor $(1-p)^{|E\cap\text{tiles}|}$ in $\sigma(E)$ is negligible,
  and tickets check only tile products given committed operands, not the
  producers of those operands. Verity's own sampling is still what binds
  operands to the circuit. The two samplers stay separate.
