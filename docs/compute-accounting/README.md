# Compute-accounted bounded accumulation

Can Verity's sampled zero-knowledge verification of a committed circuit be
combined with proofs of useful work for matrix multiplication (Komargodski–
Weinstein, arXiv 2504.09971) so that a developer proves, to an auditor, that
its datacenter spent its compute on inference-shaped work and not on training?
These four documents work the question through: they refine the problem until
it has a definition, give a merged protocol, and then argue its security and
insecurity.

| File | Contents |
|---|---|
| [01-problem-statement.md](01-problem-statement.md) | Six stages of refinement from "prove you only do inference" to a game with a payoff; definitions of tickets, tasks, heavy MACs, free compute; the four properties S1–S4 |
| [02-protocol.md](02-protocol.md) | The third seed and where it sits; the canonical matmul gate $\mathrm{CMM}_r$ and its Verity marks; tickets and their proofs; the merged per-request flow; epoch accounting; changes to `veritor`; alternatives |
| [03-security-analysis.md](03-security-analysis.md) | Assumptions A1–A8; proof sketches of S1–S4; the ordering lemma; composition theorem; 23 attacks; the residual an adversary keeps; why all three ingredients are necessary |
| [04-costs-parameters-open-problems.md](04-costs-parameters-open-problems.md) | GPU overhead of the instrumentation (the bad news); proof and statistical costs; a worked parameter set; ten open problems; what is settled |

## Summary

**The merge is sound, and each part is load-bearing.** A PoUW certificate
("ticket") for a tile of a declared matmul gate, whose zero-knowledge proof
opens the tile's operands from the *same boundary commitment Verity samples*,
certifies three things at once: that $r^3$ MAC-equivalents were spent under
fresh noise (PoUW hardness), that they were spent on the honest values of the
declared circuit (Verity soundness, since a wrong operand lives in one of the
$\le B\approx10$–$46$ replay units per epoch that can be corrupted before a
rejection becomes near-certain), and — once the auditor checks a task
partition of the epoch circuit against $(\Gamma,X,Y,N)$ using the replay-unit
boundary as the task interface — that the declared circuit is one training
cannot be disguised as (Loomis–Whitney). Counting tickets against the hardware
inventory bounds undeclared compute; it does not need Verity's sampling to be
sound, and Verity's sampling does not need tickets to bound capacity. The
PoUW seed is released after the request header and before execution, which
is what keeps a self-dealing developer from choosing operands that cancel the
noise. Remove Verity and the developer launders training matmuls through
declared gates; remove PoUW and it replays one computation a million times;
remove the structural check and it declares the training loop honestly.

**The guarantee is relative, and the residual is large in absolute terms.**
Accepted epoch implies free compute
$\le(\delta+\Delta+\varepsilon_{\rm alg}+\varepsilon_\alpha)\Phi_T+\lambda\Phi_T$
plus small-state training that $(\Gamma,X,Y,N)$ admits by design. The
cryptographic and statistical terms can be made tiny. The two that cannot are
how much faster than the reference a tile can be computed ($\varepsilon_{\rm alg}$,
fast matrix multiplication on the given hardware) and how much faster than the
reference the instrumented matmul can run ($\varepsilon_\alpha$, an engineering
race). Both convert one-for-one into compute the developer keeps, and a few
percent of a $10^6$-device fleet trains large models. Compute accounting turns
"the whole datacenter might be training" into "a few percent of it might be";
PBA's wardens and wipes are what act on that residual.

**The instrumentation is expensive on GPUs, and attention is outside it.**
PoUW's $1+o(1)$ hides two constants: scalar work costs ~30 tensor MACs, and
the noise rank must equal the tile size. Hashing costs $\sim c/r$ of the
matmul and decode costs $\sim2r(1/n+1/k+1/m)$; the optimum is 15–40 % for
transformer MLP shapes depending on how cheap a provably unpredictable hash
can be made. For attention ($k=d_{\rm head}=128$) the decode alone is $\ge2\times$
and there is no configuration below ~50 %. Attention and all non-matmul work
are therefore *unevidenced*: an honest developer either buys filler to cover
them (and to cover memory-bound decode's idle tensor cores) or the auditor
trusts an architecture ratio, and either way an adversary gets that fraction.
A decode-free self-canceling transform (PoUW Appendix A) made exact and
robust to degenerate operands is the one route to low single digits, and it
is conjectural.

**Verdict.** Secure in the sense of S1–S4 under stated assumptions, with a
residual of order several percent of the fleet that no cryptographic
parameter reduces, and an honest cost that the source paper's overhead claim
does not predict. Whether that is worth having depends on whether the
$\delta\Phi_T$ residual, under wipes and interconnect limits, is enough to
train something that matters.
