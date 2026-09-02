# Section 6 ("Partial verification"): context dump for the writing agent

Compiled 2026-09-02 from the repository at `/Users/danielreuter/projects/veritor` (the working tree has one unrelated author modification, `docs/paper/section-5-covert-capacity.md`, which was read but not touched), from the Notion exports `/tmp/notion-outline.md` (whole paper; §6 at lines 385–468, the appendix argument at lines 523–551), `/tmp/notion-outline-sections-6-7.md`, `/tmp/notion-section6-scratch.md` (the §6.4 "EXAMPLE V2" worked example), `/tmp/notion-section7-outline.md`; from `docs/paper/section-5-covert-capacity.md` (the current §5 draft, 399 lines), `docs/paper/section-7-secure-circuit-compilation.md` (the §7 draft), `docs/security-argument.md`, `docs/overnight-report.md`, `docs/frontier-report.md`, `docs/data/frontier-70b.json`, `docs/sp1-benchmark-plan.md`, `README.md`; from the source under `src/veritor/` and the tests under `tests/veritor/`; and from the design transcript `eb746331-5a84-4468-9455-5c2a7c14f35c` ("the transcript" below; the author's turns are quoted verbatim with their date and time, and numbered `msg N` by message index in the JSONL as counted by the helper used to read it). `docs/calibration/` exists (created this morning) but is empty.

Every number marked *measured* was produced today with the three throwaway scripts described in §3.7, §6.5 and §7.5 of these notes (kept under `/tmp/s6/`, not in the repository). Every number marked *unsourced* or *estimate* is labelled as such.

Conventions in this file: code fences are `~~~`; inline math is `$...$`, display math `$$...$$`; code is cited as `path:line` or `path::symbol`. "Outline" means the author's Notion outline; "Scratch" means the worked-example page; "§5 draft" and "§7 draft" mean the two files in `docs/paper/`; "code" means the repository. Where these disagree there is a **Discrepancy** callout with each version and a recommendation. Addresses are 1-based in the paper and 0-based in the code; where a code output is reported the mapping is stated.

---

## 0. Read-me for the writing agent

### 0.1 What §6 is for in the paper's architecture

The paper's spine is three trusted functions on two objects, and §8 composes them:

- **Bound** $(C, I, \theta) \to U$ (§5): an upper bound on $\log_2|\mathcal Y(\mathcal T_{\mathcal V})|$, the number of bits one accepted response can carry, for a verifier that checks only sampled units.
- **Verify** (§6, this section): the sampling verifier itself. It takes $(C, I)$, the client's proposal $\theta = (q, s)$, the verifier's own $\eta$, $U_{\max}$, $W_{\max}$ and (with §7) $A$, and accepts or rejects one claimed evaluation $y^* = C(x)$. §6 must show that this verifier is the verifier §5 analysed: its acceptance function on a fixed transcript is $\sigma_\theta(E) = \prod_r(1-q+q(1-s)^{|E\cap\mathcal P_r|})$, and adaptivity (filling in an interior after learning the unit was chosen) does not raise acceptance above $\sigma_\theta$ of the completed transcript.
- **Compile** $(G, x, a) \to (C, I)$ (§7): where $(C, I)$ comes from without trusting the client, and why the verifier can answer every query §6 makes in $O(\text{depth})$.
- §8 composes: exfiltration security in the sense of §3 from $\Pr[\text{accept}]\le\sigma_\theta$, $|\mathcal Y|\le 2^{U}$, $U \le U_{\max}$ at admission, and the $2^{A}$ circuits of §7.

The author fixed this decomposition on Sep 1 (transcript msg 748, Tuesday, Sep 1, 2026, 3:14 PM): *"Maybe we want it to be like... 1. Bound (how to compute bound) 2. Verify (how to verify, conditional on having chosen some sampling strategy, also providing a cost model) 3. Strategize (choosing sampling strategy given bound) 4. Compile (building C and I) ... I think they each want constructions and so forth, so maybe they are top-level sections."* "Strategize" became §6.6 ("Parameter optimization") and the code's `Optimize`; the cost model is §6.5. The top-level API in `src/veritor/research.py` uses the paper's names: `Compile`, `Verify`, `Bound`, `Capacity`, `Cost`, `Optimize`.

The outline's one-line statement of the section (`/tmp/notion-outline-sections-6-7.md:2`): *"Section 5 assumed a verifier capable of checking that at most $L$ subcircuits of $C$ were evaluated incorrectly. This section designs such a verifier and gives a construction that costs about 1% of evaluating $C$."* Note that the current §5 draft no longer speaks of "at most $L$ subcircuits"; it speaks of the family $\mathfrak E_\eta$ of tolerated error sets and, for the two-stage verifier, of $\sigma_\theta$ (see 0.2). §6's opening sentence should be updated to match (Discrepancy D-0 in §9).

The section plan, from the outline (`/tmp/notion-outline-sections-6-7.md:3-9`):

1. describe how $C$ is partitioned for sampling (6.1)
2. give the protocol (6.2)
3. prove its security properties (6.3)
4. give a simple construction (6.4)
5. estimate its cost (6.5)
6. show how to optimize the sampling parameters (6.6)

### 0.2 What §6 may assume from §5 (names and numbers exactly as in the current draft)

From `docs/paper/section-5-covert-capacity.md`:

- **Circuit** $C$ of size $n$, gates $1,\ldots,n$, $v_i \leftarrow f_i(v_{j_1},\ldots,v_{j_{d_i}})$, $\operatorname{args}(i)$ an ordered tuple of gates *preceding* $i$; value set $V_i\subseteq\{0,1\}^*$ finite with $|V_i|\ge 2$; width $w_i := \log_2|V_i|$; a 16-bit gate has $w_i = 16$, a token gate $w_i = \log_2 50{,}257 \approx 15.6$; designated $\operatorname{in}(C)$ (input gates have $\operatorname{args}(j) = ()$, $f_j$ a constant, the constants form $x$) and $\operatorname{out}(C)$ (§5 draft lines 7–13). $C$ and $x$ fixed throughout §5.
- **Transcript** $\tau = (v_1,\ldots,v_n)$; values need not be well-typed ("any finite bitstring, of any length"); *well-typed* value/transcript; gate *correct* in $\tau$ iff each argument value is well-typed and $v_i = f_i(\ldots)$; $\operatorname{err}(\tau)\subseteq[n]$; exactly one transcript has $\operatorname{err}(\tau)=\varnothing$; $\tau|_S$; $W(S) := \sum_{i\in S} w_i$, $W_{\mathrm{out}} := W(\operatorname{out}(C))$; $2^{W(S)}$ well-typed assignments to $S$ (lines 15–27).
- **Verifiers** (line 29): $\mathcal V$ is given a partition $\mathcal P$ of the *non-input* gates into $N$ units and a distribution over subsets of $\mathcal P$; draws $\mathcal S\subseteq\mathcal P$; accepts iff the input gates are correct, the output gates are well-typed, and every gate of every unit in $\mathcal S$ is correct, *"(A gate is checked against the values its arguments hold in $\tau$, which may lie outside the unit.)"* $\operatorname{err}_{\mathcal P}(\tau)$; $G(E) := \bigcup_{U\in E}U$; $W(E)$, $\tau|_E$. $\Pr[\mathcal V(\tau)=1] = \Pr[\mathcal S\cap\operatorname{err}_{\mathcal P}(\tau)=\varnothing] =: \sigma(\operatorname{err}_{\mathcal P}(\tau))$ when inputs correct and output well-typed, else $0$; $\sigma$ monotone. The *sampling verifier* with rate $p$: $\sigma(E) = (1-p)^{|E|}$. Line 41: *"For the verifier of §6, which samples in two stages, $\sigma$ is given at the end of §5.4."*
- **Threshold and tolerated sets** (lines 43–55): $\eta\in(0,1)$; $\mathcal T_{\mathcal V} := \{\tau : \Pr[\mathcal V(\tau)=1] > \eta\}$; $\mathfrak E_\eta := \{E\subseteq\mathcal P : \sigma(E) > \eta\}$; $\mathcal T_{\mathcal V}$ is exactly the transcripts with correct inputs, well-typed output and $\operatorname{err}_{\mathcal P}(\tau)\in\mathfrak E_\eta$; $\mathfrak E_\eta$ is closed under subsets. For the sampling verifier $\mathfrak E_\eta = \{E : |E|\le B\}$ where *"$B$ is the largest integer with $(1-p)^B>\eta$"*; *"With $p=\eta=1\%$, $B=458$."* *"Transcripts outside $\mathcal T_{\mathcal V}$ are accepted with probability at most $\eta$, and §8 accounts for them separately."*
- **Reachable outputs** $\mathcal Y(\mathcal T) := \{\tau|_{\operatorname{out}(C)} : \tau\in\mathcal T\}$; capacity $= \log_2|\mathcal Y(\mathcal T_{\mathcal V})|$ (lines 57–63).
- **Lemma 5.1** (well-typed transcripts suffice), **Proposition 5.2** $|\mathcal Y(\mathcal T_{\mathcal V})|\le\sum_{E\in\mathfrak E_\eta}2^{W(E)}$, **Corollary 5.3** (value term $B\,W_{\max}$ and position term $B\log_2(eN/B)$; the §5.3 matmul example: $d = 1{,}024$, $k = 100$, $p=\eta=1\%$, $N = 102{,}400$, $W_{\max} = 32{,}752$, $B = 458$, value term $\approx 1.5\times10^7$, position term $\approx 4{,}200$, $W_{\mathrm{out}} = 16{,}384$).
- **Definitions 5.4–5.5** (path; $D$ cuts $A$ from $Z$; downstream cut for $A$ cuts $A$ from $\operatorname{out}(C)$), **Theorem 5.6** (downstream cuts determine outputs), **Corollary 5.7** ($|\mathcal Y(\mathcal T)|\le 2^{W(D)}$), $\kappa(A) := \min\{W(D) : D \text{ a downstream cut for } A\}$, $\kappa(A)\le W(A)$, $\kappa(A)\le W_{\mathrm{out}}$, $\kappa(E) := \kappa(G(E))$ (lines 131–157).
- **Lemma 5.8** ($|\mathcal Y(\mathcal T_{\mathcal A})|\le 2^{\kappa(\mathcal A)}$), **Theorem 5.9** $|\mathcal Y(\mathcal T_{\mathcal V})|\le\sum_{E\in\mathfrak E_\eta}2^{\kappa(E)}$, and the paragraph at line 177: *"The bound is a sum and not a maximum. The maximum $\max_E 2^{\kappa(E)}$ bounds an adversary that has fixed where its errors go; the choice of $E$ carries up to $\log_2|\mathfrak E_\eta|$ bits more ... When several sets in $\mathfrak E_\eta$ share a cut, Lemma 5.8 applied to their union is tighter than the sum over them; Proposition 5.15 and §6 both use this."* **Proposition 5.10** (closed form with $\kappa_{\max}$).
- **Definitions 5.11–5.13** (downstream of; $D(A)$ the downstream-most narrowest cut; regions $A_D$, $R$ their number), **Lemma 5.14** (units inside a region are interchangeable for the value term), **Proposition 5.15** (sum over $\mathfrak Q_\eta$ of $2^{W(D_Q)}$; closed form $\sum_{\ell\le B}\binom R\ell 2^{\ell\kappa_{\max}}$ when units lie inside regions), **Corollary 5.16** $\log_2|\mathcal Y|\le B(\kappa_{\max}+\log_2(eR/B))$.
- **The two-stage verifier, end of §5.4.2** (lines 245–251), verbatim: *"The verifier of §6 samples in two stages. Its replay units $R_1,\ldots,R_m$ partition the non-input gates, each a union of verification units; it includes each replay unit in its sample independently with probability $q$, and then each verification unit inside a chosen replay unit independently with probability $s$. Writing $\mathcal P_r := \{U\in\mathcal P : U\subseteq R_r\}$,*
  $$\sigma_\theta(E)=\prod_{r=1}^{m}\bigl(1-q+q(1-s)^{|E\cap\mathcal P_r|}\bigr),\qquad\theta=(q,s),$$
  *so Theorem 5.9 and the first bound of Proposition 5.15 apply to it, whatever its verification units. (Strictly speaking, $\sigma_\theta$ is the probability of accepting a transcript fixed in advance. The protocol of §6 lets the adversary fill in the interior of a replay unit after learning that it was chosen, and §6 shows that this does not raise the probability of acceptance above $\sigma_\theta$ of the completed transcript's set of incorrect units.) The sets in $\mathfrak E_\eta$ are then not small: as long as $1-q>\eta$, every set of verification units inside a single replay unit is in $\mathfrak E_\eta$, whatever its size, since it costs the adversary only the one factor $1-q$. Lemma 5.8 bounds what all such sets together can reach by $2^{\kappa(\mathcal P_r)}$."* This parenthetical is a promissory note §6 must discharge (the $\ell^\star$ lemma, §4.2 below).
- **§5.7, fourth point** (line 358), verbatim: *"Fourth, under the two-stage verifier, replay units enter the bound too. When $1-q>\eta$, every set of verification units inside one replay unit $R_r$ is in $\mathfrak E_\eta$, and all such sets together reach at most $2^{\kappa(\mathcal P_r)}$ outputs, by Lemma 5.8. The value term for an adversary that corrupts many units inside one replay unit is thus the cut width of the replay unit, not of the regions inside it, and the choice among replay units is a position term of its own. A replay unit should therefore be an operation whose outputs are few compared with its gates; §6 weighs this against the cost of replay."*
- **GPT-2 Small numbers** (§5.6, lines 309–344): 12 layers, $d = 768$, 12 heads of 64, MLP 3,072, vocabulary 50,257, 100-token prompt, 100 generated tokens; $n = 42{,}387{,}408{,}594$ non-input gates $= 42{,}361{,}101{,}422$ arithmetic $+ 26{,}307{,}172$ FP16 rounding gates; $W_{\mathrm{out}}\approx 1{,}562$ bits; $R = 31{,}987{,}771$ regions; $\kappa_{\max} = 32$ ($6{,}114{,}482$ regions with 32-bit cuts); average region $\approx 1{,}300$ gates, largest $2.5\times10^8$; Corollary 5.16 gives $458(32+17.5)\approx 22{,}700$ bits, Corollary 5.3 gives $458(32+27.9)\approx 27{,}400$, the output bound $1{,}562$. Line 344: *"The regions and cuts are the same for the two-stage verifier of §6; only the sum over them, in Proposition 5.15, changes."*

**What §5 does not give §6 and §6 must not assume it does.** The §5 draft has no notion of boundary, interior, commitment, replay, or the client choosing anything; it has no $U_{\max}$, no cost, no $\theta$ other than the symbol in $\sigma_\theta$; and it partitions the *non-input* gates (line 29 and line 245), whereas the implementation puts every gate, inputs included, inside units (§2.2 below; Discrepancy D-3).

### 0.3 What §6 may assume from §7, and must not re-explain

From `docs/paper/section-7-secure-circuit-compilation.md` (the draft; it postdates the outline but predates some code decisions):

- $\mathrm{Compile}(G, x, a) \to (C, I)$ is deterministic; a party holding $h_C$, $x$, $a$ computes the same $(C, I)$ as the verifier (**Lemma 7.1**); at most $2^{A}$ pairs per $(G, x)$.
- **Lemma 7.2 (Soundness of derived structure)**: (i) replay units partition the gates, verification units partition each replay unit *[the draft adds "and inputs lie in no unit"; the code disagrees, D-3]*; (ii) *"for every replay unit $R_r$, every address inside $R_r$ read by a gate outside $R_r$, and every circuit output inside $R_r$, lies in $\mathrm{Out}(R_r)$, so $\partial = \mathrm{in}(C)\cup\bigcup_r\mathrm{Out}(R_r)$ contains every value read across a replay-unit boundary"*; (iii) size, cost, $|\mathrm{In}|$, $|\mathrm{Out}|$ and the width of $\mathrm{Out}$ are identical for all copies of a kind. Its remark (line 92): *"The lemma is exactly what §6's $\ell_r^\star$ lemma needs—that the stitched partial transcript $\widehat\tau$ agrees with $\tau|_\partial$ on every value a selected unit reads from outside—since (ii) gives every such value an address in $\partial$."*
- **Proposition 7.3 (Efficiency)**: every query §6 makes ($C[i]$, the unit owning $i$, a unit's interface, the rank of $i$ in $\partial$) costs $O(d)$ descent steps; the verifier's state is $G$ and the per-kind table; nothing costs $\Omega(n)$.
- **Proposition 7.4 (Completeness)** and **Theorem 7.5 (No uncharged degrees of freedom)**: *"Fix $\Sigma$, $A$, $\eta$, $U_{\max}$ and $W_{\max}$ ... $(C, I)$ satisfies the assumptions of §6; and $U = \mathrm{Bound}(C, I, \theta)\le U_{\max}$. Consequently the set of outputs $y^*$ accepted with probability above $\eta$, over all admissible runs with the same $(G, x)$, has size at most $2^{U_{\max}+A}$."*
- The asymmetry principle (draft line 41): *"structure—$G$, its kinds, the per-kind table of §7.4—is small and can be analyzed globally; values—the transcript—are enormous and can only be checked locally. Every global fact the protocol needs is placed on the structure side and derived from $G$; the value side is touched only where a sample lands."*
- $\kappa_W$: the §7 draft (lines 55, 74, 139) describes $\kappa_W$ as *"the pre-committed subtree of the boundary tree"*. The code makes it a separate root with its own domain (D-6). §6 owns the protocol-level description of $\kappa_W$ and should state the implemented version; §7 should be brought in line.
- The §7 draft's own cross-check list (line 177) asks §6 to confirm: (a) the $\ell_r^\star$ hypothesis is phrased as "$\partial$ contains every value read across a replay-unit boundary"; (b) the expected-work formula $W = (|\mathrm{io}| + qsP)(1+\lceil\log_2 n\rceil) + qsS + qm + 1$ matches §6.5 and that $|\mathrm{io}|$ excludes weights; (c) numbering "§6.6" for choosing units and rates.

### 0.4 What §6 must deliver to §8

1. **The verifier's parameters and who sets them.** $\eta$ (the verifier's; it defines $\mathcal Y_\eta$ and travels in the header), $U_{\max}$ (the verifier's; admission), $W_{\max}$ (the verifier's; admission), $A$ (with §7; admission). The client proposes $\theta=(q,s)$ and $a$ and chooses $I$ (the marks). Code: `src/veritor/protocol/parameters.py::VerifierParameters` (`eta`, `max_capacity`, `max_advice_bits`, `max_work`), `src/veritor/core/policy.py::VerificationPolicy` ($q$, $s$ as `Fraction`s).
2. **Completeness**: an honest client is accepted with probability 1 (up to the sampler's $2^{-190}$ bias and resource limits).
3. **Capacity soundness** in the form §8 consumes: for every client strategy, if the run is accepted with probability $>\eta$ then the claimed output lies in $\mathcal Y_\eta := \bigcup_{E:\sigma_\theta(E)>\eta}\mathrm{Reach}(E)$, and $\log_2|\mathcal Y_\eta|\le U = \mathrm{Bound}(C,I,\theta)\le U_{\max}$. Two sub-claims: (a) adaptive acceptance $\le\sigma_\theta(E^\star)$ (the $\ell^\star$ lemma); (b) $\mathrm{Bound}$ certifies the union.
4. **The admission rule** and its consequence, *"every accepted request has capacity at most $U_{\max}+A$"* (`docs/overnight-report.md:120-121`; `parameters.py:47-50`).
5. **Zero-knowledge** (what is hidden by the ZK construction, and the outline's note that it is *"conditional on the physical assumptions about witnessing the wire"*, `/tmp/notion-outline-sections-6-7.md:93`).
6. **The transcript** as an object §8 can refer to: header plus five messages, re-verifiable offline (`verify_transcript`).
7. **The cost**, so that §8/§10 can say "at about 1% of evaluating $C$" with a stated cost model and calibration.

### 0.5 The author's stylistic constraints (verbatim where possible)

- **Transparent verifier as specification, then ZK by replacement.** Monday, Aug 31, 2026, 1:33 PM (msg 535): *"Let's just assume that the verifier does everything and the client just gives it its information. That'll just make the exposition easier and then we'll just say wherever we say the verifier like does something with plain text this can be replaced with the client sends a commitment to it along with a proof that it did so correctly. As in the client can just run the verifier (using public randomness) and prove it did so correctly."* Scratch page: *"The transparent verifier above is the specification. To hide the client's values, replace $\beta$ and each $\alpha_r$ with commitments made at the same points in the interaction. At each phase, the client proves the checks that the transparent verifier would have completed before releasing the next challenge. After the final challenge, the client proves that every sampled gate equation is satisfied by the committed values."*
- **The verifier only knows the address space.** Monday, Aug 31, 2026, 12:08 PM (msg 496): *"I think all the verifier knows it has to build everything out of its knowledge of the address space one through n. That's the only thing it knows. Everything else is too expensive for it to materialize."* And 12:52 PM (msg 522): *"I'd like to build everything out of as primitive of a function as possible ... for the circuit, we can get its gate and we can get its count ... for the index, I'm not entirely sure what else we can do. But like commitment, I think all we know how to do is like take hashes and so forth that's like the primitive. And for sampling, we only know how to like draw randomness."*
- **The asymmetry principle** (structure cheap, values expensive). Tuesday, Sep 1, 2026, 3:28 PM (msg 759): *"we can do a pass of every gate in the circuit just because the circuits are quite compressed ... The values cannot be compressed. We don't attempt to compress the values, and so they're enormous. And so we cannot check every value. And so we need everything to follow from just local checks of values, but because of that some of the values can be infinite. The point is that if you pass that into just a circuit that we know and that we have done this structural analysis of, that doesn't actually affect our bound. Our bound incorporates that."* And 3:38 PM (msg 768): *"We want to say that we assume we can run algorithms over every gate in the circuit, but only over a tiny percentage of the values."* In §6 terms: the boundary is a *compression scheme* for the transcript (coarse replay units make $|\partial|\ll n$), and everything global (∂, ranks, unit membership, $\mathrm{Bound}$, $W$) is derived from structure.
- **No notion of authorized computation.** Same turn (msg 759): *"at no point do we check anything. The whole point is we're just bounding the exfiltration rate. There's no notion of an authorized computation ... The only property it has is that it's fixed and that any additional choices are tracked."* Also: *"an important design problem here is there's no way to check whether some of the values are absolutely enormous because we can only locally check stuff."* (The implementation does check well-typedness of every *opened* value: `INVALID_VALUE`; unopened values are unconstrained, which is exactly why Lemma 5.1 is needed.)
- **Vocabulary.** *"I think we call it well-typed, not well-formed"* (Sep 1, 3:45 PM, msg 770). *"Generally I hate 'words.' I prefer talking about output widths of gates and/or value widths"* (Sep 1, 3:33 PM, msg 761). *"G is once per epoch"* (same). The task brief adds: no "kernel", "description object" or "instance" as named things; "design decisions", not "design challenges"; the paper's objects are $G, C, I, \Sigma$ (the gate set), $\mathrm{Compile}$, $\theta=(q,s)$, $\eta$, $U$, $U_{\max}$, $W_{\max}$, $\partial$, $\beta$, $\alpha_r$, $\mathrm{Int}(r)$, $R_r$, $V_{r,j}$, $J$, $T$. The §5 draft's vocabulary note (line 391) lists what it stripped: "acceptance function", "admissible error set", "budget", "certify/certificate", "canonical"; §6 should not reintroduce them without cause (the code uses "admissible" and "budget" as identifiers; the paper can say "sets in $\mathfrak E_\eta$" and "$\ln(1/\eta)$").
- **Figure conventions** (Aug 31, 3:59–5:09 PM, msgs 604–647): *"just call them $r_1$ and $r_2$, $v_{1,1}$ etc. -- the gates are implicit"*; *"have the gate boxes basically just contain the operator and then have arrows coming in from the prior positions saying which ones they read"* (msg 551); *"Don't need 'cross-unit read' as a thing"* (msg 554); *"we don't need explicit out"* (msg 616); *"we actually hardcode the values in the input gates"* (msg 622); *"blue in and red out ... the whole node colored like that ... I don't want 'out' as a text label"* (msg 627); *"values pills and gates are rectangles"* (msg 641); *"in the top right-hand side vertically stacked the legend ... The blue rectangles are Input gates and the red rectangles are output gates. Also probably x^2 is more idiomatic"* (msg 644). See §10.
- **Scale and ML circuits.** Sep 1, 3:50 PM (msg 772): *"the problem is it just doesn't really talk about scale. And also, we're immediately going to need to talk about ML circuits. Right? We're basically only going to be talking about ML circuits in the actual paper."* §6 should carry the 70B numbers of §6.5/6.6, not only the toy.
- **Every required property explicit, correct, compact** (task brief; see the checklist in §12).

---

## 1. The problem §6 solves (6.1 material)

### 1.1 The outline's text, verbatim

`/tmp/notion-outline-sections-6-7.md:10-25`:

> The verifier samples subcircuits of $C$ and checks them, so $\tau$ must be fixed before the draw. Two things make this hard:
> - **Commitment cost:** Committing to all of $\tau$ is expensive. For GPT-2, hashing every value costs about 16% of evaluating $C$, against a budget of 1%.
> - **Ephemeral values:** Evaluating $C$ discards intermediate values as it goes, so the client must recompute whatever the verifier checks. It cannot afford to recompute all of $C$; it must recompute only what is sampled.
>
> Our protocol uses two partitions of $C$, a coarse one into replay units and a fine one into verification units:
> - **Replay units:** When the verifier samples a replay unit, the client replays it, recomputing every value in it from its inputs, and commits to those values.
> - **Verification units:** When the verifier samples a verification unit, the client proves every gate in it correct against the committed values. Each verification unit lies inside a single replay unit. Verification units are the subcircuits that Section 5 counts.
>
> Replay units answer both problems. Before any draw, the client commits only to the *boundary*: the values read across replay units, together with the inputs and outputs of $C$. After the draw, it replays the sampled units and commits to their *interiors*, the values not on the boundary. Only sampled units are replayed, and only sampled units are committed in full.
>
> [Figure: the eight-gate example, with the boundary shaded.]
>
> Deferring the interior commitment gives the client a choice it did not have before: it picks a unit's interior after learning that the unit was sampled. The choice does not help it. The interior with the fewest incorrect verification units is determined by the unit's inputs and outputs, which were fixed before the draw, so the client could have committed to it up front. Section 6.3 makes this precise.
>
> Replay units should be coarse, so that the boundary is small. Verification units cannot be: proving is about $10^{10}\times$ native, so a whole replay unit is far too expensive to prove. Nor should they be single gates, since each proof carries a fixed overhead.
>
> A verification unit can be proved only against committed values, and interiors are committed one replay unit at a time. So each verification unit lies inside a single replay unit, and is sampled only after that unit is.
>
> The protocol therefore has two draws, one per commitment: commit the boundary; sample replay units; commit their interiors; sample verification units inside them; prove those. The verifier can sample only from what is already fixed.
>
> Compared with committing to all of $\tau$ and sampling verification units in one draw, the values committed fall from $n$ to the boundary plus the interiors of the sampled units, and the check on each sampled verification unit is unchanged. What is lost is that every verification unit inside an unsampled replay unit goes unchecked. Section 6.3 quantifies the loss.
>
> Nothing else constrains either partition. The bound of Section 5 holds for any partition into verification units, and the analysis of Section 6.3 for any partition into replay units, so both granularities are set by cost alone. Section 6.6 chooses them. [Argument in the appendix.]

The whole-paper outline (`/tmp/notion-outline.md:81`) phrases the second obstacle as: *"**Checking the sample:** zero-knowledge proving is $10^{10}\times$ native. Proving tiny units wastes the proof system's fixed overhead; proving large ones wastes work on gates that did not need to be proved."*

### 1.2 The "16%" figure is unsourced

The sentence *"For GPT-2, hashing every value costs about 16% of evaluating $C$, against a budget of 1%"* appears in the outline (Sep 1, 10:27 PM export and today's) and nowhere else. A search of every transcript in `agent-transcripts/*/*.jsonl` for `16%`, `16 percent`, `sixteen percent` finds it only inside the pasted outline text and in two unrelated contexts (a 16% share of a knapsack inflation, a 16% cut-bound row in a granularity table). No derivation, model of hash cost, or hardware assumption backs it. The assistant noticed the tension on Sep 2 (msg 1162, thinking aloud): *"Real hashing (SHA-256, Poseidon) costs far more than a single multiply-add — yet the paper claims hashing every value costs only 16% of evaluating the whole circuit for GPT-2, which would mean $h$ is actually much smaller than 1 relative to a gate's replay cost."* Under the implemented cost model ($h = 1$ gate-equivalent per committed value) hashing every value of the 70B serving circuit costs $n/\text{honest} = 0.667$, i.e. **67%**, not 16% (measured; §6.5). **Flag: unsourced; either derive it (values per FLOP for GPT-2 under a stated hash cost) or replace it with the cost model's own number.** One way to make it defensible: GPT-2 Small has $n\approx 4.24\times10^{10}$ gate values and $\approx 4.23\times10^{10}$ FLOPs (§5 draft line 396: "conventional contraction work is 42,257,061,888 FLOPs" appears in the transcript's cut-analysis report); a Merkle leaf over a 16-bit value costs one compression-function call per leaf plus one per internal node, so "16%" would need a hash to cost about $0.16$ FLOP-equivalents, which is not true of SHA-256 on a CPU but may be a reasonable figure for a GPU-batched arithmetic hash relative to a fused multiply-add. The paper must say which.

### 1.3 The two obstacles in the implementation's terms

- **Fixing $\tau$ before the draw.** The verifier owns both seeds and releases each only after the message it depends on: $J$ is derived from `(q_seed, header, BoundaryMessage)` and $T$ from `(s_seed, ..., InteriorMessage)` (`src/veritor/protocol/session.py:583-591`, `614-621`; `src/veritor/protocol/phases.py`). The prover cannot learn a selection before the values it depends on are fixed (`docs/security-argument.md` §2).
- **Ephemeral values.** The prover's default is to *replay* each selected unit from the boundary: `replay_unit(compiled, unit, boundary_values)` recomputes $\mathrm{Int}(r)$ in address order from the boundary values and the unit's own earlier interior values (`session.py:222-241`). `ProverSession.interiors` calls it per selected unit and commits the result (`session.py:329-348`). A dishonest prover is modelled by `assignment_replay(values)` (`session.py:244-250`), which commits a fixed (possibly wrong) assignment instead.
- **Only sampled units are committed in full.** `InteriorMessage` carries one `Commitment` per selected replay unit, in $J$ order (`messages.py:285-299`); `interior_domain(replay_phase_digest, compiled, r)` covers exactly $\mathrm{Int}(r)$ (`domains.py:76-84`).

### 1.4 Why each verification unit lies inside a single replay unit

Three reasons, all present in the code:

1. **Provability.** A verification unit is checked against committed values; a value is committed either in $\partial$ (before $J$) or in some $\mathrm{Int}(r)$ (after $J$, only if $r\in J$). A unit straddling $R_r$ and $R_{r'}$ could be checked only if both are selected, which breaks the product form of $\sigma_\theta$.
2. **Tiling is enforced structurally.** `validate_marks` (`src/veritor/core/index.py:642-705`) checks once per definition that replay marks tile the gates, verification marks tile every replay unit, and no mark nests inside a mark of the same role; a verification mark cannot straddle two replay units because *"it would have to contain a replay mark"* (`docs/security-argument.md` §8). `Index.verification_units(r)` enumerates the verification units inside $r$; global verification-unit indices are numbered block by block in replay-unit order (`index.py:296-303`).
3. **Ownership.** `_Layout.required(unit)` refuses a verification unit that reads an address owned by another replay unit that is not a boundary position (`session.py:203-220`, code `INVALID_COMPILED_RESULT`); the compiler makes this unconstructible.

### 1.5 The two draws and what is lost

With one draw over all $N$ verification units at rate $p$, $\sigma(E) = (1-p)^{|E|}$ and the client must commit all $n$ values. With two draws, $\sigma_\theta(E) = \prod_r(1-q+q(1-s)^{l_r})$ and the client commits $|\partial| + \sum_{r\in J}|\mathrm{Int}(r)|$ values. The loss: every verification unit inside an unsampled replay unit goes unchecked, and the factor $1-q$ per replay unit does not depend on how many units inside it are wrong. In the code's terms (`src/veritor/analysis/probability.py:1-15`): *"Writing $c(l) = -\ln f(l)$ and $\Lambda = \ln(1/\eta)$ this is $\sum_r c(l_r) < \Lambda$: a knapsack over replay units. $c$ is increasing and saturates at $-\ln(1-q)$, so many errors in one replay unit cost little more than a few -- concentration is cheap, and that falls out of the formula."*

### 1.6 "Nothing else constrains either partition": the appendix argument, in full

The outline's appendix (`/tmp/notion-outline.md:523-551`, headed "Why do we not put any constraints on the partitions?", with the note "(shoudl maybe put this in") is garbled by the export (LaTeX doubled). The author's assistant wrote it on Tuesday, Sep 1, 2026, 7:39 PM (msg 888); the clean text is:

> **Setting.** Circuit $C$ on addresses $[n]$. An index $I$ designates replay units $R_1,\dots,R_m$ partitioning $[n]$ and verification units refining them. $\theta=(q,s)$. Boundary $\partial := \mathrm{in}(C)\cup\mathrm{out}(C)\cup\bigcup_r\mathrm{Out}(R_r)$. The verifier fixes a threshold $\eta$ and a cap $U_{\max}$. Everything else, including $I$ and $\theta$, is chosen by the adversary and committed with $C$ before the run.
>
> **Claim.** For every $I$ satisfying partition and refinement, and every $\theta$, $U = \mathrm{Bound}(C,I,\theta)$ is a valid upper bound on the number of bits an accepted run can carry. Consequently the verifier needs no further condition on $I$: it computes $U$ and proceeds iff $U\le U_{\max}$.
>
> **Argument.**
>
> 1. *What is being bounded.* Let $Y_\eta = \{\tau|_{\mathrm{out}(C)} : \Pr[\text{accept }\tau] > \eta\}$, the outputs the client can produce while being accepted with probability above $\eta$. Capacity per run is $\log_2|Y_\eta|$. This is the quantity $U$ must dominate.
> 2. *Acceptance depends only on the error set, for any partition.* Every cross-unit read passes through a declared output of the producing unit, so $\partial$ contains every value one replay unit reads from another. Given the committed boundary, the interiors of different replay units are therefore independent, and the client's best adaptive strategy inside a selected unit is the completion minimizing its number of incorrect verification units. This is the $\ell^\star$ lemma, and it uses only that $\partial\supseteq\bigcup_r\mathrm{Out}(R_r)$, which holds by construction for every partition. So acceptance probability is at most
>    $$\sigma_\theta(E)=\prod_r\bigl(1-q+q(1-s)^{|E\cap R_r|}\bigr),$$
>    where $E$ is the set of verification units left incorrect. This formula is exact rather than a model: the verifier flips one independent $\mathrm{Bernoulli}(q)$ per replay unit and one independent $\mathrm{Bernoulli}(s)$ per verification unit inside a selected one. No size, width, cost, or count of units appears in it.
> 3. *Reachable outputs depend only on cuts, for any set.* If all incorrect gates lie in a subcircuit $S$, everything outside $S$ is computed correctly from the inputs and the values on $\mathrm{Out}(S)$, so at most $2^{|\mathrm{Out}(S)|\cdot B}$ outputs are reachable. Hence
>    $$|\mathrm{Reach}(E)|\le 2^{\kappa(E)},\qquad \kappa(E) := \min_{S\supseteq E}|\mathrm{Out}(S)|\cdot B,$$
>    the minimum over any family of available cuts (the nodes of $I$, or anything else). This is the downstream-cut theorem; it holds for every $E$ and every $S$, and the partition does not appear in it.
> 4. *Combine.* $Y_\eta\subseteq\bigcup_{E:\sigma_\theta(E)>\eta}\mathrm{Reach}(E)$, so
>    $$|Y_\eta|\le\sum_{E:\sigma_\theta(E)>\eta}2^{\kappa(E)} =: 2^{U}.$$
>    Both $\sigma_\theta$ and $\kappa$ are defined for every $E$ under every $I$, so $U$ is a valid bound for every $I$. Nothing in steps 2–4 was conditional on the shape of the units.
> 5. *What a bad partition does.* Coarse replay units let the adversary concentrate errors behind a single $(1-q)$ factor, making $\sigma$ large for many $E$; coarse verification units make $\kappa$ large per bad unit. Either way the sum grows and $U$ grows. The verifier sees $U$ before the run. So the guarantee is unconditional in the form that matters: *if the run is accepted, its capacity is at most $U_{\max}$*. The partition affects only whether acceptance happens, and it is priced in the open, so the adversary has no reason to prefer a bad partition and no way to profit from one.
> 6. *Why choosing $I$ is not itself a channel.* $I$ is committed together with $C$, before any run, and the set of admissible markings is finite (at most $3^{|\text{definitions}|}$). The bits spent choosing it are amortized over every run under that commitment, exactly as the bits spent choosing $C$ are. This is the one timing constraint: a client permitted to re-mark per run would be choosing a fresh circuit per run and would have to be charged $\log_2(\#I)$ bits accordingly.
> 7. *What the verifier does have to fix.* $\eta$, because it is part of the definition of $Y_\eta$ and hence of the guarantee itself; $U_{\max}$; and a cap $W_{\max}$ on its own work, since the client controls the number of units and the bit-length of $q,s$. None of these constrains the partition. Two further conditions exist but are about completeness, not soundness: a cap on per-unit proof cost so an honest client can produce proofs, and a cap on the denominators of $\theta$ so sampling terminates in bounded time.
>
> Two consequences for the implementation. (a) Step 3 must be evaluated on the client's actual units. Substituting gates for verification units, which is exact when $V$ is singletons, under-bounds for coarse $V$, and the audit exhibited cases where it does. (b) Step 4 is a sum, not a maximum. $\max_E\kappa(E)$ bounds a client who has fixed *where* its errors go; the game lets it choose per run, and the choice of $E$ carries up to $\log_2$ of the number of admissible error sets, capped only by the output cut. Bound must certify the union.

Three remarks for the writing agent. (i) In this argument $B$ is the word width (bits per output value), not §5's budget $B$; the §5 draft writes cut widths as $W(D)$ in bits and the position budget as $B$. Use $W(\mathrm{Out}(S))$ and avoid the clash. (ii) The $\partial$ here includes $\mathrm{out}(C)$ explicitly; in the code the circuit outputs are always inside $\bigcup_r\mathrm{Out}(R_r)$ (every output resolves through the declared interface of the replay unit that owns it: `index.py:329-331`), so `Index.boundary()` is $\mathrm{In}\cup\bigcup_r\mathrm{Out}(R_r)$ and $\mathrm{out}(C)\subseteq\partial$ is a consequence, checked at `_Layout.__init__` (`session.py:176-183`, raising if an output were not a boundary position). (iii) Step 6's finiteness of markings and the "committed with $C$ before the run" clause are §7's business (marks are bytes of $G$; $h_C$ binds them; `test_changing_a_role_mark_changes_the_compiled_digest`).

**How this is implemented as admission.** `VerifierSession._admit` (`session.py:454-498`) runs before any commitment is accepted and in this order: advice bits $8|a|\le$ `max_advice_bits` (else `POLICY_REJECTED`); `max_probability_denominator_bits` on $\max(\text{denominator bits of } q, s, \eta)$; `max_units` on the replay-unit count and the verification-unit count; `max_positions_per_unit` per kind; `expected_work(compiled, θ, |IO|) ≤ W_max` (else `WORK_BUDGET_EXCEEDED`); and, when `max_capacity` is set, `bound(compiled, θ, η).bits ≤ U_max` (else `POLICY_REJECTED`). Docstring: *"Price the run before any commitment; folds over the kinds, nothing per copy ... Together the two caps bound the request's capacity by `U_max + A`."* Tests: `tests/veritor/protocol/test_parameters.py::test_runs_whose_bound_exceeds_u_max_are_rejected_before_any_commitment`, `::test_runs_above_the_work_budget_are_rejected_before_any_commitment`, `::test_u_max_zero_admits_a_fully_checked_run`, `::test_u_max_is_checked_at_the_verifiers_eta`, `::test_a_transcript_recorded_under_a_permissive_cap_is_rejected_by_a_tight_one`.

### 1.7 The new finding to integrate: "replay units should be coarse" is in tension with the bound

The outline says replay units should be coarse so the boundary is small. The honest-server frontier (`docs/frontier-report.md`, Sep 2) shows the other side of the coin at scale. Because an unsampled replay unit's interior is never checked, with probability $1-q$ the adversary controls the unit's *entire declared interface*; if $1-q>\eta$ that costs one factor and is always admissible. So

$$\mathrm{Bound}(C,I,\theta)\ \ge\ \min\bigl(W_{\mathrm{out}},\ \max_r W(\mathrm{Out}(R_r))\bigr)\quad\text{whenever }1-q>\eta,$$

and more generally the adversary affords $\lfloor\Lambda/(-\ln(1-q))\rfloor$ whole replay units at once ($\Lambda=\ln(1/\eta)$): 6 at $q=1/2,\eta=10^{-2}$; 20 at $q=1/2,\eta=10^{-6}$ (`docs/overnight-report.md:348-351`). The consequence for a transformer (report, "Reading the frontier"):

> The four intermediate partitions are useless at this scale because their replay units are "super-escapes": a `prefill_step` exposes 320 Gbit of activations, a prefill `layer` 192 Mbit, a `square_block` 256 Mbit, all wider than the 16 Mbit output, so a single admitted unit error reaches every output and the bound is capped at `out_bits` for every grid policy (436 of the 588 points sit at the cap, all 336 points of these four partitions among them). The `request` unit is the only coarse unit whose interface (its 512 tokens) is narrower than the circuit's output, which is why `request/*` is informative and `step/*` is not, though `step` is finer.

Numbers, from the report's partition table (70B shape, output interface $2048\times512\times16 = 16{,}777{,}216$ bits):

| partition (replay/verification) | replay units | widest replay-unit kind | its `out_bits` | $U$ at $\eta=10^{-2}$, best point with overhead $\le1$, work $\le10$ |
|---|---|---|---|---|
| `request/row` | $2.05\times10^3$ | `request` | 8 kbit | 276 kbit (1.7%) at $q=1/8,s=1$ |
| `request/gate` | $2.05\times10^3$ | `request` | 8 kbit | 53.5 kbit (0.33%) at $q=1/2,s=1/8$ |
| `step/row` | $3.28\times10^4$ | `prefill_step(32)` | 320 Gbit | 16 Mbit (100%, vacuous) |
| `layer/row` | $8.7\times10^7$ | `layer(512 positions, prefill)` | 192 Mbit | 16 Mbit (100%) |
| `matvec/row` | $1.2\times10^{10}$ | `square_block(16777216)` | 256 Mbit | 16 Mbit (100%) |
| `row/gate` | $1.24\times10^{13}$ | `square_block(16777216)` | 256 Mbit | 16 Mbit (100%) |
| `cell/gate` | $3.47\times10^{13}$ | `cell_unit(eq)` (every unit) | 16 bit | 3.93 kbit (0.024%) at $q=1/2,s=1/8$ |

So the outline's sentence must be qualified: replay units should be coarse *in gates* but narrow *in interface*. A dot product (thousands of gates, one 16-bit output) is the ideal replay unit: the boundary is $n/k$ values for contraction length $k$, and a whole-unit corruption buys 16 bits. A decode step (millions of gates, a KV-cache-width interface) is the worst: its boundary is small, but a whole-unit corruption buys more than the entire output. The author's own reaction when shown this (Wednesday, Sep 2, 2026, 7:40 AM, msg 1081): *"I would have assumed for the KV state width we would break it up. I don't understand why that's any different from like the fact that a dot product, you know, like a matrix multiplication consists of many dot products, each of which has one small output."* The assistant's answer (msg 1082) is the paragraph the paper needs: *"In $Y = XW$, a row unit's outputs (`cols` words of $Y$) are circuit outputs. Corrupting the unit really does let the adversary set those `cols` words, so the cut charge `cols·B` is exact: cut = reach. In a transformer, a unit's outputs are internal state. A decode step writes $\sim 2Ld$ KV words per slot and emits one token; the network compresses $\sim10^5$ state words into one output word. So at any granularity coarser than a single dot product, cut $\gg$ reach for a transformer unit, and the cut bound over-charges by the compression ratio. Matmul never has this problem because its state is its output."* This is why two refinements are on the table: (a) the client marks at `cell` granularity (every unit one dot product), which the cost model prices as nearly free (§6.5); (b) $\mathrm{Bound}$ becomes reach-aware, $\min(\mathrm{out\_bits}, \mathrm{reach\_bits})$ per node (§7.6 below). The author's decision on which lever the paper leans on is in §7.3.

---

## 2. Objects and notation, exactly as implemented

### 2.1 The circuit $C$ and the gate set $\Sigma$

- **Addresses.** The code addresses gates $0,\ldots,n-1$ (`Circuit.__getitem__`, `Index.n`); the paper uses $1,\ldots,n$. Every reported code output below is translated where it matters.
- **Gate set.** `src/veritor/core/gates.py::Gate` has `name`, `arity`, `width` (output width in bits), `is_input`, `is_weight`, `replay_cost`, `proof_cost`, and the evaluation/check functions. `GateSet` is the public $\Sigma$; its digest is bound into the compiled digest. Two example gate sets: `make_word_gate_set(width)` (`gates.py:232-266`: `in`, `weight`, `add`, `mul`, ... over $\mathbb Z_{2^{\text{width}}}$) and `make_isa_gate_set(B)` (`gates.py:269-298`: `in`, `weight`, `add`, `sub`, `mul`, `lt`, `eq`, `shr` over $\mathbb Z_{2^B}$; replay costs `add/sub/lt/eq/shr` 1, `mul` 2, sources 0; proof costs the same for arithmetic, and 1 for sources: `docs/frontier-report.md` "Caveats"). All gates in a gate set used with $\kappa_W$ must have weight gates of one width (`domains.py::weight_width`).
- **Source gates.** `in` and `weight` are zero-arity gates of $\Sigma$ ("source gates"). The circuit's inputs are its `in` gates in address order (`Circuit.inputs`, `Circuit.input_rank`), its weights the `weight` gates (`Circuit.weights`, `Circuit.weight_rank`). Inputs are pinned to $x$ (`Header.public_inputs` by rank) and weights to $\kappa_W$ (by rank). Outputs are a designated tuple of addresses `Circuit.outputs` (declared positions, not copy gates).
- **Values.** Every gate has a fixed output width; values are encoded canonically as big-endian fixed-width bytes (`Circuit.encode/decode`; a non-canonical opening is `INVALID_VALUE`). Every arithmetic operation is modular word arithmetic (toy). The §5 draft's per-gate value sets $V_i$ and token widths $\log_2 50{,}257$ are more general than the code (one width per gate); this is a presentational, not a logical, gap (D-10).
- **Well-typed.** A value at address $i$ is well-typed iff it is a canonical encoding of an element of $\{0,\ldots,2^{w_i}-1\}$. The verifier checks this for every value it opens (`_check_unit`, `INVALID_VALUE`) and for every public I/O value (encode at header construction). It cannot check it for unopened values; §5's Lemma 5.1 is what makes that harmless.

### 2.2 The index $I$: kinds, copies, marks, and the two antichains

`src/veritor/core/index.py`. An index is a hierarchy of *frames* (`IndexNode`, `index.py:34-92`), one per copy of a *definition* (a kind). The root frame covers $[0,n)$; each child frame covers a contiguous interval `[base, base + size)`. Two roles mark definitions: `role="replay"` and `role="verification"` (`src/veritor/core/description.py::REPLAY/VERIFICATION`). The set of replay-marked frames and the set of verification-marked frames are two antichains in the tree:

- `Index.replay_units` (a `Units`, `index.py:133-190`): the replay-marked frames, numbered $0..m-1$ in address order; `Units.unit(r)` returns the frame, `Units.owner(address)` the unit containing an address, `Units.count`.
- `Index.verification_units(r)` (`index.py:296-302`): the verification-marked frames inside replay unit $r$, as a block of consecutive global indices; `Index.verification_unit(k)` (`index.py:310-322`) returns the $k$-th verification unit globally with its `replay_unit`; `Index.verification_unit_count`.
- **Tiling** (`validate_marks`, `index.py:642-705`; `docs/security-argument.md` §8): replay marks tile the gates of the root; verification marks tile the gates of every replay unit; a mark never nests inside another mark of the same role. Consequently **every gate, including every `in` and `weight` gate, belongs to exactly one replay unit and exactly one verification unit** (`docs/overnight-report.md` §2.1: *"The root definition has no ports. The circuit's inputs and weights are zero-arity source gates ... placed by descriptions like any other gate, so they sit inside replay and verification units and the tiling check covers them."*). Tests: `tests/veritor/security/test_tiling.py::test_every_gate_is_in_exactly_one_replay_unit_and_one_verification_unit`, `::test_verification_units_refine_replay_units`, `::test_marks_leaving_a_gate_uncovered_are_a_compile_error`, `::test_nested_or_straddling_marks_are_a_compile_error`.

**Decision 2.1 (Sep 1, msg 783, author chose "option one") and its consequences.** The alternative was the outline's picture: input positions outside every unit, `Out` copy gates, the root with ports. The author chose source gates inside units. Consequences (overnight report §2.1, verbatim list):

- *"A unit of source gates has `Out` width 0, so `Bound` gives it zero capacity without a new rule (a tightening is possible: a source-only kind can never hold an error, see §3)."*
- *"Outputs stay declared (an interface), not gates. The paper's `Out_1(6)` copy gate has no counterpart; the uniform statement is 'every gate belongs to one replay unit and one verification unit; the outputs are declared positions among them'."*
- *"`∂ = In ∪ ⋃_r Out(R_r)` with `Out` excluding pinned gates, and `Int(r) = R_r \ Out(R_r) \ pinned(R_r)`: source gates are never interior. κ_W covers the weight gates; the `exclude=range` carve-out of an input prefix is gone."*
- *"A sampled `in` gate is compared with `x[rank]`; a sampled `weight` gate is accepted on its κ_W opening alone; the boundary phase still opens every `in` gate (Θ(|x|), inherent)."*
- *"Every source gate is by default its own verification unit (`Tracer.inputs(n)` emits one-gate cells), so the priced work `q s Σ_v (proof(V_v) + c_0)` gains `(|x| + |W|)(1 + c_0)`. A client that wants a smaller `W_max` price defines wider input units; nothing forces the one-gate cells."*

### 2.3 Declared interface of a unit: `In`, `Out`, pinned, and "Out excludes pinned gates"

Per definition (`src/veritor/core/description.py`): `input_count` (declared ports read from the parent), `out_count`/`out_bits`/`out_runs` (declared output positions inside the definition, as strided runs), `source_inputs`/`source_weights` (counts of `in`/`weight` gates inside, "pinned"), `replay_cost`, `proof_cost`, `size`. `KindSummary` (`index.py:193-225`) carries these per kind with `copies`; `KindTable` (`index.py:228-256`) is the per-kind table Bound, Cost and `expected_work` fold over.

- $\mathrm{In}(V)$ for a unit is the set of outside addresses its gates read: `Circuit.In(node)` (used by `_Layout.required`).
- $\mathrm{Out}(R_r)$ is the unit's *declared* output positions: every address inside $R_r$ that a gate outside $R_r$ reads, plus every circuit output inside $R_r$, must be among them (§7 Lemma 7.2(ii); `_Layout.required` enforces the consequence at run time). **`Out` never contains a pinned gate** (`_Boundary` docstring: *"The two parts are disjoint: `Out` never contains a pinned gate."*): an `in` gate is already in the boundary by rank, and a weight gate lives under $\kappa_W$ and may not be a claimed output (`_Layout.__init__` raises "a weight gate cannot be a claimed output"). The compiler enforces the rule; the run-typed `Out` was fuzzed against the per-ordinal resolver after F7.
- `out_bits` of a node is the sum of the widths of its declared outputs, excluding sources; it is the cut width Bound charges for corrupting the node.

### 2.4 The boundary $\partial$ and the interiors $\mathrm{Int}(r)$: the exact definitions

`Index.boundary()` (`index.py:324-333`), docstring verbatim: *"`In ∪ ⋃_r Out(R_r)`: the addresses the boundary commitment covers. The input gates by rank, then the units' `Out` in unit order. The weight gates are committed under their own root and are not here. The circuit outputs are always inside this set: every output resolves through the declared interface of the replay unit that owns it."* Implemented by `_Boundary` (`index.py:474-546`): `count = |In| + out_total`; `rank(address)` is the input rank if the address is an `in` gate, else `|In| + frame.out_before + out_rank within the frame`; `unrank` inverts by descending the hierarchy with prefix sums (`out_before` per frame, `step_out`/`out_total` per definition). Both are $O(\text{depth})$ (Proposition 7.3). The identity digest of the domain is `identity_digest("veritor/indexed-domain/boundary/v4", {"index": index.digest})`.

`Index.interior(r)` (`index.py:334-348`), docstring: *"`R_r` minus the boundary and the weights: its interval minus `Out` and its pinned runs."* It is an `IntervalDifferenceDomain` = the unit's interval minus its `out_runs`, `input_runs`, `weight_runs`. So

$$\partial = \mathrm{In}\ \cup\ \bigcup_r \mathrm{Out}(R_r),\qquad \mathrm{Int}(r) = R_r\setminus\mathrm{Out}(R_r)\setminus\mathrm{pinned}(R_r),\qquad W = \text{weight gates},$$

and $[n] = \partial\ \sqcup\ W\ \sqcup\ \bigsqcup_r\mathrm{Int}(r)$ is a partition of the address space into commitment *owners*: `_Layout.owner(address)` returns `WEIGHT_OWNER = -2`, `BOUNDARY_OWNER = -1`, or the replay-unit index $r$ (`session.py:187-194`; `domains.py:33-34`). The paper's $\partial$ (Scratch, appendix) writes $\mathrm{in}(C)\cup\mathrm{out}(C)\cup\bigcup_r\mathrm{Out}(R_r)$; the code's omits $\mathrm{out}(C)$ because it is implied, and omits weights because they are under $\kappa_W$ (D-5, D-6).

**Discrepancy D-5 (∂ definition).** Outline/Scratch: $\partial = \mathrm{in}(C)\cup\mathrm{out}(C)\cup\bigcup_r\mathrm{Out}(R_r)$, "the values read across a replay-unit boundary, together with the inputs and outputs of $C$". Code: $\partial = \mathrm{In}\cup\bigcup_r\mathrm{Out}(R_r)$ with $\mathrm{out}(C)\subseteq\bigcup_r\mathrm{Out}(R_r)$ by construction (§7's compiler makes every circuit output a declared output of its owning replay unit) and weights excluded. §7 draft: $\partial = \mathrm{in}(C)\cup\bigcup_r\mathrm{Out}(R_r)$ (line 88), agreeing with the code. Recommendation: define $\partial := \mathrm{In}\cup\bigcup_r\mathrm{Out}(R_r)$, state as a one-line consequence that $\mathrm{out}(C)\subseteq\partial$ ("every output is a declared output of the replay unit that computes it"), and say once that weight gates are under $\kappa_W$ and therefore neither in $\partial$ nor in any interior.

### 2.5 The eight-gate example, recomputed with the code

Constructor used (kept at `/tmp/s6/eight.py`; a `Tracer` program over `make_word_gate_set(16)`), defining kinds `pair` (two `in` gates; verification), `addsq` (add then square; verification), `addmul` (add then multiply by a port; verification), `R1 = addsq(pair())`, `R2 = addmul(pair(), R1)` (replay), root `R2(R1())`:

~~~python
@t.definition(input_count=0, key="pair", role="verification")
def pair(_v):  return [inp(), inp()]            # V_{r,1}: two input gates
@t.definition(input_count=2, key="addsq", role="verification")
def addsq(v):  s = add(v[0], v[1]); return mul(s, s)   # V_{1,2}: gates 3,4
@t.definition(input_count=3, key="addmul", role="verification")
def addmul(v): s = add(v[0], v[1]); return mul(v[2], s) # V_{2,2}: gates 7,8
@t.definition(input_count=0, key="R1", role="replay")
def R1(_v):    p = pair(); return addsq(p[0], p[1])
@t.definition(input_count=1, key="R2", role="replay")
def R2(v):     p = pair(); return addmul(p[0], p[1], v[0])
@t.definition(input_count=0, key="root")
def root(_v):  return R2(R1())
~~~

Output of `Compile(G, x=(1,1,2,3), a=b"")` (measured; addresses shown 0-based with the paper's 1-based in parentheses):

~~~text
n = 8  description bytes = 2040
  address 0 (paper 1): op=in  args=()     width=16 is_input=True
  address 1 (paper 2): op=in  args=()     width=16 is_input=True
  address 2 (paper 3): op=add args=(0, 1) width=16
  address 3 (paper 4): op=mul args=(2, 2) width=16        # x^2
  address 4 (paper 5): op=in  args=()     width=16 is_input=True
  address 5 (paper 6): op=in  args=()     width=16 is_input=True
  address 6 (paper 7): op=add args=(4, 5) width=16
  address 7 (paper 8): op=mul args=(3, 6) width=16
outputs (0-based): (7,)   inputs: (0, 1, 4, 5)
replay units:       [(0, (0,1,2,3)), (1, (4,5,6,7))]
verification units: [(0, (0,1), in RU 0), (1, (2,3), in RU 0), (2, (4,5), in RU 1), (3, (6,7), in RU 1)]
boundary ∂ (0-based, rank order): [0, 1, 4, 5, 3, 7]   count 6
boundary ∂ (paper 1-based):        [1, 2, 5, 6, 4, 8]
Int(1) (paper) = [3]      Int(2) (paper) = [7]
honest tau (paper order): [1, 1, 2, 4, 2, 3, 5, 20]     y* = (20,)
~~~

So the code reproduces the Scratch page exactly: $\partial=\{1,2,4,5,6,8\}$ as a set, $\mathrm{Int}(1)=\{3\}$, $\mathrm{Int}(2)=\{7\}$, $\tau=(1,1,2,4,2,3,5,20)$. Differences to note for the writer:

- **Rank order of ∂** is not address order: inputs first by rank ($1,2,5,6$), then $\mathrm{Out}(R_1)=\{4\}$, then $\mathrm{Out}(R_2)=\{8\}$. The Scratch page lists $\partial$ as a set; if the paper shows the boundary Merkle tree's leaves it should use rank order.
- **Inputs count as boundary** (all four `in` gates are boundary positions by rank), **and the output gate 8 is a boundary position** because it is $\mathrm{Out}(R_2)$; there is no separate "outputs" part.
- **Both `in` gates of a replay unit form a verification unit** ($V_{1,1}=\{1,2\}$, $V_{2,1}=\{5,6\}$): source-only units, never incorrect (F4). The Scratch page has the same four units.
- **The x² gate** is `mul(3,3)` in the code (the word gate set has no unary square); the figure labels it $x^2$ (author, msg 644).
- **Per-kind table** the verifier actually holds (measured; kind digests truncated):

~~~text
root    role=None         copies=1 size=8 in=0 out=1 out_bits=16 src_in=4 replay_cost=6 proof_cost=10
R1      role=replay       copies=1 size=4 in=0 out=1 out_bits=16 src_in=2 replay_cost=3 proof_cost=5
pair    role=verification copies=2 size=2 in=0 out=0 out_bits=0  src_in=2 replay_cost=0 proof_cost=2
addsq   role=verification copies=1 size=2 in=2 out=1 out_bits=16 src_in=0 replay_cost=3 proof_cost=3
R2      role=replay       copies=1 size=4 in=1 out=1 out_bits=16 src_in=2 replay_cost=3 proof_cost=5
addmul  role=verification copies=1 size=2 in=3 out=1 out_bits=16 src_in=0 replay_cost=3 proof_cost=3
~~~

(`add` costs 1 to replay and 1 to prove, `mul` 2 and 2, `in` 0 and 1; so `addsq` replay 3, proof 3; `pair` replay 0, proof 2.) The honest cost of the circuit is the root's `replay_cost` = 6.

### 2.6 Value semantics and widths (what §6 should say)

Everything in the prototype is modular word arithmetic on 16-bit words (toy ISA) — `docs/overnight-report.md` discrepancy 10: *"Value semantics remain deferred. Everything is modular word arithmetic; the LLM constructors that need fixed-point or float semantics wait on that decision."* §5's draft is agnostic (any finite $V_i$). §6 needs only: each gate has a fixed output width $w_i$; `check_gate` is a deterministic relation on canonical encodings; source gates have no relation. The paper should not commit §6 to fixed-point vs float; it should say the verifier checks a gate by evaluating $f_i$ on the opened argument values and comparing (`Circuit.check_gate`, `_check_unit`), and that an ill-typed or non-canonical opened value is a rejection.

---

## 3. The protocol (6.2)

### 3.1 The transparent version (specification)

The Scratch page's walkthrough is the specification the author asked for ("the verifier does everything", msg 535). Reproduced here with the code's names in brackets, for the eight-gate circuit with $q=s=1/2$:

0. **Header.** Both parties fix $(C, I)$ [`compiled.digest` = $H(C,I)$], the constructor digest [`Header.constructor`], the advice $a$ [`Header.advice`], $x$ [`Header.public_inputs`, encoded, by input rank], $y^*$ [`Header.claimed_outputs`], the client's $\theta=(q,s)$ [`Header.policy`], the verifier's $\eta$ [`Header.eta`], $\kappa_W$ [`Header.weights`, `None` here since there are no weight gates], and a session id. The verifier prices the run (admission, §1.6) before anything else.
1. **The client sends the complete boundary assignment** $\beta=\{1\mapsto1,2\mapsto1,4\mapsto4,5\mapsto2,6\mapsto3,8\mapsto20\}$ before seeing any randomness. The verifier derives $\partial$ from $(C,I)$ [`Index.boundary()`], checks $\mathrm{dom}(\beta)=\partial$, that each $\beta_i\in\mathbb Z_{2^{16}}$, that $(\beta_1,\beta_2,\beta_5,\beta_6)=x$ and $\beta_8=y^*$; rejects on failure, else stores $\beta$. *"The value at gate 4 is now fixed at 4 even though the verifier has not checked whether gate 4 correctly squared gate 3."*
2. **The verifier samples replay units:** one fresh fair coin per replay unit; $R_1$: tails, $R_2$: heads, $J=\{R_2\}$. *"The client learns that $R_2$ must be replayed only after the boundary is fixed; it cannot change $\beta_4,\beta_5,\beta_6,\beta_8$ in response."*
3. **The client replays the selected unit and sends its interior.** $R_2=\{5,6,7,8\}$, $\mathrm{Int}(2)=\{7\}$; honest $\tau_7=\beta_5+\beta_6=5$, $\tau_8=\beta_4\cdot\tau_7=20$; $\alpha_2=\{7\mapsto5\}$. The verifier checks $\mathrm{dom}(\alpha_2)=\mathrm{Int}(2)$ and the type, stores it, and *"deliberately has not yet selected or checked a verification unit."*
4. **The verifier samples verification units inside $R_2$** only after every selected interior is stored: $V_{2,1}$: tails, $V_{2,2}$: heads, $T=\{V_{2,2}\}$.
5. **The verifier assembles** $\widehat\tau=\beta\cup\alpha_2=\{1\mapsto1,2\mapsto1,4\mapsto4,5\mapsto2,6\mapsto3,7\mapsto5,8\mapsto20\}$. Gate 3 is absent ($R_1$ not selected); gate 4 present (boundary). *"The client does not tell the verifier where to find argument values; the verifier resolves every address using $C$, $I$ and the stored assignments."*
6. **The verifier checks the selected unit:** $C[7]=\mathrm{Add}(5,6)$: $5=2+3$ passes; $C[8]=\mathrm{Mul}(4,7)$: $20=4\cdot5$ passes. *"Checking gate 8 used the fixed value at gate 4 but did not check gate 4's own square equation; that belongs to a verification unit in the unselected $R_1$."*
7. **Accept or reject.**

Then: *"The transparent verifier above is the specification. To hide the client's values, replace $\beta$ and each $\alpha_r$ with commitments made at the same points in the interaction."* (§5 below.)

### 3.2 The implemented staged version: messages and phases

Files: `src/veritor/protocol/session.py` (`ProverSession` 253–372, `VerifierSession` 374–710, `run_protocol` 715–761), `messages.py`, `phases.py`, `challenge.py`, `domains.py`, `merkle.py`, `verify.py`, `wire.py`. `PROTOCOL_VERSION` is in `messages.py`. A run is:

~~~text
Verifier                                        Prover
  |  [admission: A, denominators, unit counts, W_max, U_max]
  |  Header (both compute it; the digest binds everything below)
  |<------------- BoundaryMessage(commitment κ_∂, io_openings) ---------------|
  |  accept κ_∂ under boundary_domain(header, compiled); open & compare I/O
  |  J := bernoulli_subset(q_seed, "q/replay-unit", boundary_phase, m, q)
  |-------------- ReplayChallenge(q_seed, selected = J) --------------------->|
  |<------------- InteriorMessage(commitments κ_r for r in J, in J order) ----|
  |  accept each κ_r under interior_domain(replay_phase, compiled, r)
  |  T := bernoulli_subset(s_seed, "s/verification-unit", interior_phase, Σ_{r∈J}|V(r)|, s)
  |-------------- SampleChallenge(s_seed, selected = T) --------------------->|
  |<------------- EvidenceMessage(units = (openings for v) for v in T) -------|
  |  for each v in T: open exactly required(v) under their owners; check gates
  |  VerificationReport(ACCEPTED, J, T)   or   Reject(code, detail)
~~~

**Header** (`messages.py:185-248`), fields: `session_id`, `compiled_digest` ($H(C,I)$), `constructor` (digest of $G$), `advice` ($a$), `policy` ($\theta$), `eta`, `public_inputs` (encoded values of the `in` gates by rank), `claimed_outputs` (encoded $y^*$ by output order), `weights` (`Weights(count, root)` = $\kappa_W$, or `None`). Digest tag `veritor/protocol/header/v6`, over all of these plus `protocol_version`. Docstring: *"`compiled_digest` names `(C, I)`, `constructor` the digest of the `G` that produced it and `advice` the `a` it was run on, so a transcript is bound to one `Compile(G, x, a)`. `policy` is the client's `theta = (q, s)` and `eta` the verifier's acceptance threshold."* The verifier's seeds are **not** in the header (they are revealed in the challenges); the header is the same on both sides and the prover checks it matches its own expectation (`EXPECTATION_MISMATCH` otherwise).

**Expectation** (`session.py:63-101`, built by `make_expectation(compilation, proposal, claimed_outputs, *, parameters, weights=None, session_id=None, q_seed=None, s_seed=None)`): the verifier's side — `session_id`, `compiled_digest`, `constructor`, `advice`, `policy`, `parameters: VerifierParameters`, `public_inputs`, `claimed_outputs`, `q_seed`, `s_seed` (32-byte secrets; drawn with `secrets.token_bytes(32)` when not given), `weights: Weights | None`. Docstring of `make_expectation`: nothing is *"defaulted: the verifier states `eta`, `U_max`, `A` and `W_max`."*

**Phase chain** (`phases.py`): `boundary_phase = H("phase/boundary/v2", header.digest, boundary.manifest)`; `replay_phase = H("phase/replay/v2", boundary_phase, ReplayChallenge.manifest)`; `interior_phase = H("phase/interior/v2", replay_phase, InteriorMessage.manifest)`; `sample_phase = H("phase/sample/v2", interior_phase, SampleChallenge.manifest)`. *"Each phase digest covers the previous one and the message just received, so a commitment domain or challenge bound to a phase digest is bound to the entire prefix of the interaction."*

**Message 1, `BoundaryMessage`** (`messages.py:251-267`): `commitment: Commitment(root, count)` over $\partial$ and `io_openings: tuple[Opening, ...]` — one opening per public I/O address, in boundary rank order (`_Layout.io`: the distinct `in` addresses and output addresses sorted by boundary rank). Verifier (`receive_boundary`, `session.py:555-595`): accept the commitment under `boundary_domain(header, compiled)` (count must equal $|\partial|$; an empty domain must present `empty_root`), check the opened positions are exactly `io` (`COVERAGE_MISMATCH`), verify each opening (`INVALID_OPENING`), compare each `in` value with `header.public_inputs[rank]` and each output with `header.claimed_outputs` (`PUBLIC_IO_MISMATCH`). Then derive $J$.

**Discrepancy D-1 (outline §6.2 step "Check that input values equal x; output values are well-typed and equal y*" with the note "[these should use hiding commitments]").** Code: the I/O values are *opened in the clear* from the boundary tree and compared byte-for-byte with the header (which already contains them). In the transparent protocol this is by design (nothing is hidden). In the ZK construction the inputs and outputs are public anyway ($x$ and $y^*$ are in the header), so "hiding" would apply to the *other* boundary values, not to the I/O openings; the opening of I/O is needed to *bind* the tree to the header's $x$, $y^*$ (otherwise the tree could hold different values at those positions). Recommendation: §6.2 should say "the client opens the input and output positions of the boundary commitment; the verifier checks they equal $x$ and $y^*$" and note that in the ZK version this opening is a proof of equality to the public header values, while all other boundary values stay hidden.

**Challenge 1, `ReplayChallenge`** (`messages.py:270-285`): `seed` (the verifier's `q_seed`, now revealed) and `selected: tuple[int, ...]` = $J$, sorted. Derivation (`derive_replay_selection`, `challenge.py:252-263`): `bernoulli_subset(q_seed, b"q/replay-unit", boundary_phase, m, q)`. **Sublinear sampling** (`challenge.py` module docstring, verbatim): *"Every candidate unit is selected independently with an exact rational probability `p = a / b` -- the joint distribution of one `p`-coin per unit -- but the work is proportional to the number of units selected, not to the number of candidates. The count `K ~ Binomial(N, p)` is drawn first, by inverting the binomial CDF against a 256-bit uniform, and then a uniformly random `K`-subset of `range(N)` is drawn with Floyd's algorithm from exactly `K` further uniforms. All randomness is HMAC-SHA256 keyed by the verifier's seed over the phase digest and a purpose tag, expanded by rejection sampling, so no floating point is involved and both parties agree bit for bit. Because the phase digest covers every earlier message, the prover cannot learn a selection before the values it depends on are fixed. Exact rational evaluation of the CDF would need `N * log2(b)` bits, so it is evaluated in 512-bit fixed point instead; `_binomial_count` bounds the resulting bias below `2**-190` in total variation for every `N < 2**64`."* `_binomial_count` (`challenge.py:155-189`) walks the pmf up from $k=0$ with the exact recurrence in fixed point; `_floyd_subset` (`190-210`) is Bentley–Floyd: for $j$ from $N-K$ to $N-1$ draw $t$ uniform in $[0,j]$, take $t$ unless taken, else $j$. Cost $O(K\log N)$ HMACs plus $O(\log N)$. Domain tag `veritor/protocol/binomial-subset-hmac-sha256/v3`; stage tags `q/replay-unit`, `s/verification-unit`; `max_probability_denominator_bits` (default 64) caps $q$, $s$ denominators. Tests: `tests/veritor/protocol/test_challenge.py` (`test_selection_is_deterministic_and_bound_to_seed_stage_and_phase`, statistical tests of marginals and independence; the audit §3: *"`J`, `T` are independent Bernoulli coins within `2^-190`; acceptance rate is `sigma(E)` — proved + tested (statistically)"*).

**Message 2, `InteriorMessage`** (`messages.py:288-299`): `commitments: tuple[Commitment, ...]`, one per $r\in J$ in $J$ order. Honest prover: `replay_unit(compiled, r, boundary_values)` recomputes $\mathrm{Int}(r)$ in address order from the boundary values (raising if a needed argument is neither known interior nor boundary — which cannot happen for a compiled circuit) and commits under `interior_domain(replay_phase, compiled, r)` (`session.py:329-348`). Verifier (`receive_interiors`, `597-627`): the number of commitments must equal $|J|$ (`COVERAGE_MISMATCH`), each accepted under its domain; then derive $T$.

**Challenge 2, `SampleChallenge`** (`messages.py:302-316`): `seed` (= `s_seed`) and `selected` = $T$ as global verification-unit indices, sorted. Derivation (`derive_sample_selection`, `challenge.py:266-292`): the candidates are the verification units of the selected replay units, *"ranked block by block in `J` order; only the `O(|J|)` block sizes and the `O(|T|)` sampled ranks are ever touched"*; `bernoulli_subset(s_seed, b"s/verification-unit", interior_phase, Σ_{r∈J}|V(r)|, s)`, mapped back to global indices with a bisect over block ends.

**Message 3, `EvidenceMessage`** (`messages.py:319-335`): `units: tuple[tuple[Opening, ...], ...]`, one batch per $v\in T$ in $T$ order. Prover (`ProverSession.evidence`): for each $v$, `required(v)` (the `(owner, address)` pairs of every gate of $v$ and every outside address it reads, sorted by address), opened under the owner's tree at `position(owner, address)`. Verifier (`receive_evidence` + `_check_unit`, `629-703`): batch count $=|T|$; total openings $\le$ `max_openings`; for each unit, the opened positions must be *exactly* `required(v)` in order (`COVERAGE_MISMATCH`); each opening verified under its owner (`INVALID_OPENING`; "no commitment for owner" if the owner is an unselected replay unit — which `required` already forbids with `INVALID_COMPILED_RESULT`); each payload decoded canonically (`INVALID_VALUE`); then per gate of $v$: an `in` gate's payload must equal `header.public_inputs[input_rank]` (`PUBLIC_IO_MISMATCH`); a `weight` gate is accepted on its $\kappa_W$ opening alone; every other gate must satisfy `check_gate(args, value)` (`RELATION_REJECTED`; a raising gate is `TRUSTED_SERVICE_FAILURE`).

**Ownership rules** (`_Layout.owner/position`, `session.py:187-200`), verbatim: *"Who commits to `address`: `kappa_W` for a weight gate, the boundary for an input gate or a declared output, else the owning replay unit."* and *"Where `address` lives in its owner's domain: its rank under `kappa_W`, the address itself under the boundary or an interior."* So the boundary tree and every interior tree are indexed by **absolute address** (the leaf's `position` is the address; its `rank` in the tree is `domain.rank(address)`), while $\kappa_W$ is indexed by weight rank. **Cross-unit reads** are resolved by this rule alone: a gate in $V\subseteq R_r$ reading address $j\notin R_r$ finds $j$ either in $\partial$ (owner $-1$) or in $W$ (owner $-2$); `required` raises `INVALID_COMPILED_RESULT` if $j$ is owned by another replay unit $r'\ne r$ — impossible for a compiled $(C,I)$ (Lemma 7.2(ii)), tested on a forged circuit only (audit §8: *"cross-cut read on a forged circuit only"*). A gate reading an address $j\in R_r$ finds it in $\mathrm{Int}(r)$ (owner $r$, committed in this run because $r\in J$) or in $\partial$ if $j\in\mathrm{Out}(R_r)$ or is an `in` gate.

**Commitments and openings** (`merkle.py`, `messages.py:114-155`): `Commitment(root: 32 bytes, count: int)`; `Opening(position, value, path)`. `CommitmentDomain(binding, owner, positions)` has `domain_id = H("domain", binding, owner+2, positions.identity_digest, count)`; `leaf(rank, position, schema, value) = H("leaf", domain_id, rank, position, schema, value)` with `schema = "u<width>"`; `node(level, index, left, right) = H("node", domain_id, level, index, left, right)`; padding leaves `H("pad", domain_id, rank)`; `empty_root = H("empty", domain_id)`; `merkle_depth(count) = 0 if count <= 1 else (count-1).bit_length()`. `verify_opening` recomputes the leaf from `(domain, rank = domain.rank(position), position, schema, value)` and walks `path` (length must equal the depth). **Domain separation** (`domains.py`): boundary binding = `header.digest`, owner $-1$, positions = `Index.boundary()`; interior binding = `replay_phase` digest, owner $r$, positions = `Index.interior(r)`; weights binding = `H("veritor/protocol/weights/v3", gate_set.digest, owner=-2)`, owner $-2$, positions = `range(|W|)`. Audit §1 verdict: *"Single owner per address; leaf/node/padding binding; domain binding; no cross-session/phase/owner reuse; verifier-derived domains — proved + tested."*

**Acceptance codes** (`messages.py:43-62`, `VerificationCode`), with meaning:

| code | raised when |
|---|---|
| `ACCEPTED` | every check passed |
| `EXPECTATION_MISMATCH` | the prover's header (or a transcript's) differs from the verifier's expectation (different $x$, $y^*$, $\theta$, $\eta$, $\kappa_W$, digests, session) |
| `POLICY_REJECTED` | admission: advice too long, or $\mathrm{Bound}(C,I,\theta)>U_{\max}$ |
| `WORK_BUDGET_EXCEEDED` | admission: `expected_work` $> W_{\max}$ |
| `INVALID_PHASE` | a message out of order |
| `INVALID_COMMITMENT` | a root/count not valid for its domain (wrong count; nonempty root for an empty domain) |
| `INVALID_OPENING` | a Merkle opening fails (wrong path length, wrong root, wrong owner) |
| `INVALID_VALUE` | an opened value is not a canonical encoding for its gate's width |
| `PUBLIC_IO_MISMATCH` | an opened `in`/output value differs from the header |
| `CHALLENGE_MISMATCH` | (offline) a transcript's recorded $J$ or $T$ differs from the derived one |
| `COVERAGE_MISMATCH` | wrong set/order of I/O openings, interior commitments, evidence batches or opened positions |
| `RELATION_REJECTED` | a sampled gate's relation fails on the opened values |
| `INVALID_COMPILED_RESULT` | $(C,I)$ violates an invariant the protocol relies on (weight as output; cross-unit read outside ∂; $\kappa_W$ count mismatch) |
| `MALFORMED_TRANSCRIPT`, `NONCANONICAL_TRANSCRIPT` | (offline) bytes that do not decode, or decode but do not re-encode identically |
| `RESOURCE_LIMIT` | a `VerificationLimits` cap hit (`max_positions` 10^7, `max_units` 10^6, `max_positions_per_unit` 10^6, `max_openings` 10^7, `max_probability_denominator_bits` 64, ...; `core/limits.py:34-77`) |
| `TRUSTED_SERVICE_FAILURE` | a gate's `check_gate` raised (fail closed) |

**Transcript and re-verification.** `Transcript(header, boundary, replay_challenge, interiors, sample_challenge, evidence)` (`messages.py:338-345`); `encode_transcript` / `decode_transcript` (`wire.py:48, 160`) are strict canonical JSON (*"sorted keys, no whitespace, lowercase hex, reduced fractions, no floats, no unknown keys"*; duplicate keys and floats rejected). `verify_transcript(data, expectation, compiled, limits)` (`verify.py:16-63`) decodes, checks canonical re-encoding, builds a fresh `VerifierSession` from the expectation (so admission re-runs), checks the transcript's header equals the session's, replays the three messages through the session, and checks the recorded challenges equal the derived ones (`CHALLENGE_MISMATCH`). Audit §9: *"Offline verification recomputes every challenge; every post-hoc alteration caught — proved + tested."* The eight-gate transcript is 3,074 bytes (measured).

### 3.3 The prover's side

`ProverSession` (`session.py:253-372`): `__init__(compiled, header, boundary_values, weights_tree=None, replay=None)`; `boundary()` builds the boundary tree from the client's values over `Index.boundary()` and opens `io`; `interiors(challenge)` checks the challenge seed/selection against its own derivation (a verifier that lies about $J$ is caught: `CHALLENGE_MISMATCH` on the prover side), replays each selected unit (default `replay_unit`) and commits; `evidence(challenge)` re-derives $T$ and opens `required(v)` for each $v\in T$ from the right tree. `run_protocol(compiled, expectation, values, *, replay=None, limits=None, weight_tree=None)` (`718-761`) drives both sessions in-process and returns an object with `.report` and `.transcript` (the transcript is `None` when the run was rejected: `test_a_rejected_interaction_leaves_no_transcript`); `replay=assignment_replay(values)` makes the prover commit a fixed dishonest assignment. `research.py::build_executable_conformance_transcript` is the top-level fixture.

### 3.4 What the verifier holds and touches (sublinearity)

Verifier state: $G$ (or the compiled per-kind table), the header, three phase digests, the roots, $J$, $T$. Work: $O(|\mathrm{io}|)$ at construction (`_Layout` docstring: *"the input gates and the outputs are the only addresses either party touches wholesale"*), $O(|J|)$ interior roots, $O(\sum_{v\in T}|\mathrm{required}(v)|\cdot\log n)$ for evidence, $O(|J|\log m + |T|\log N)$ for sampling; every structural query ($C[i]$, owner, rank, unit membership) $O(\text{depth})$ (§7 Prop 7.3). The priced version of this is `expected_work` (§6.4 below). Nothing is $\Omega(n)$; the author's constraint (msg 496) is met.

### 3.5 Weights and $\kappa_W$ in the run

If the circuit has weight gates, `Expectation.weights` must be a `Weights(count, root)` with `count == |W|` (else `INVALID_COMPILED_RESULT`). The verifier holds it *before* the run (deployment rule; audit §10 "kappa_W provenance"). A sampled weight gate is opened at `weight_rank(address)` under $\kappa_W$ and accepted on the opening alone — there is no relation to check and no comparison with anything else. `commit_weights(gate_set, values)` (`domains.py:86-100`) builds the tree once per model. Test: `test_kappa_w_is_bound_to_the_gate_set_and_the_vector_not_the_description`; headline: one root accepted for two batch shapes of the same model (`docs/overnight-report.md` §2.5).

### 3.6 Admission, restated as the verifier's first move

`_admit` runs in `VerifierSession.__init__` before the header is even constructed; the order and codes are in §1.6. Note `VerifierParameters.policy(proposal)` (`parameters.py:80-91`) returns the client's proposal unchanged: *"The client chooses how much is sampled, never what the verifier's acceptance means: `eta` is this object's and is bound into the header alongside the proposal."* `test_u_max_is_checked_at_the_verifiers_eta`, `test_the_proposal_is_theta_alone_and_the_header_binds_the_verifiers_eta` and `test_a_transcript_recorded_under_another_eta_is_rejected` pin this.

### 3.7 The eight-gate run through `run_protocol` (measured)

Script `/tmp/s6/eight.py` (construction in §2.5). With `VerificationPolicy(1/2, 1/2)`, `VerifierParameters(eta=1/100, max_capacity=None)`, `session_id=b"s6-demo"`, and seeds searched so that the draw reproduces the Scratch page ($J=\{R_2\}$, $T=\{V_{2,2}\}$): `q_seed = (33).to_bytes(32,"big")`, `s_seed = (1033).to_bytes(32,"big")`. Output:

~~~text
seeds: q_seed = int 33   s_seed = int 1033
report: accepted   J = (1,)   T = (3,)                      # 0-based: R_2 and V_{2,2}
header digest: 87206f787d07d487 ...  eta 1/100  policy q=1/2 s=1/2
public_inputs (encoded): ['0001','0001','0002','0003']   claimed_outputs: ['0014']   # 0x14 = 20
BoundaryMessage: root c17198b7b7a99129...  count 6
  io_openings positions (0-based): [0, 1, 4, 5, 7]  values [1, 1, 2, 3, 20]  path lengths [3, 3, 3, 3, 3]
ReplayChallenge: selected (1,)
InteriorMessage: [(root 5a2d6e7ccbad0ace..., count 1)]        # one tree for Int(R_2) = {gate 7}
SampleChallenge: selected (3,)
Evidence for VU 3: openings (position, value, pathlen):
  [(3, 4, 3), (4, 2, 3), (5, 3, 3), (6, 5, 0), (7, 20, 3)]
  # paper addresses 4,5,6,7,8: 4 from ∂ (Out(R_1)), 5 and 6 from ∂ (inputs), 7 from Int(R_2) (a 1-leaf tree: path length 0), 8 from ∂ (Out(R_2))
required(3): ((-1, 3), (-1, 4), (-1, 5), (1, 6), (-1, 7))     # owners: -1 boundary, 1 = interior of R_2
expected_work: 37
cost: ExpectedCost(boundary=6, replay=4, proof=5/2, weights=0)     # total 12.5 vs honest 6
bound eta=1/2, 1/100, 1/10^6: bits=16.0  out_bits=16  knapsack=32.0  laplace=32.0  capped=True
survival (q=s=1/2): E={V12}: 3/4   E={V12,V22}: 9/16   E={V22}: 3/4
cheat (beta_4=6, y*=30, only V_{1,2} incorrect): accepted 312/400 = 0.780;  sigma(E) = 0.75
transcript bytes: 3074   re-verify: accepted
~~~

Reading: the boundary tree has 6 leaves (depth 3); the five I/O openings are the four inputs and the output; the interior tree of $R_2$ has one leaf (gate 7) and depth 0; the evidence for $V_{2,2}=\{7,8\}$ opens gates 7 and 8 and their arguments 4, 5, 6 — five openings, four under the boundary root and one under the interior root. $\mathrm{Bound}=16$ bits at every $\eta$: the output is one 16-bit word, and with $q=1/2>\eta$ the adversary can corrupt all of $R_2$ (or $R_1$) with probability $1/2$, so the bound is the cap (the uncapped fold gives 32 = both units' `out_bits`). The cheat shows the $\ell^\star$ lemma numerically: the client commits $\beta_4=6$ (so $y^*=30$), keeping $\tau_3=2$ honest; the only incorrect verification unit is $V_{1,2}$ ($6\ne2^2$), $\sigma_\theta(E)=1-q+q(1-s)=3/4$, and 400 fresh-seed runs accept 312 (0.78). Had the client instead put $\beta_4=4$ and $\tau_7=6$ ($y^*=24$), the incorrect unit would be $V_{2,2}$ ($24\ne 4\cdot 5$ or $6\ne 2+3$): still one unit, $\sigma=3/4$ — but it could not also make $V_{2,1}$ wrong to "hide" anything, since inputs are pinned.

### 3.8 What the tests specify (citable behaviour)

- `tests/veritor/protocol/test_protocol.py`: `test_honest_run_accepts_and_transcript_round_trips`; `test_sessions_produce_identical_transcripts_on_both_sides`; `test_forged_interior_is_rejected_when_every_unit_is_checked`; `test_forged_interior_survives_when_nothing_is_sampled`; `test_wrong_claimed_output_is_rejected_at_the_boundary`; `test_wrong_input_value_is_rejected_at_the_boundary`; `test_verifier_rejects_transcript_recorded_under_other_seeds`; `test_tampered_replay_selection_is_a_challenge_mismatch`; `test_tampered_opening_fails_authentication`.
- `tests/veritor/security/test_staging.py` (audit §2 list): `test_verifier_rejects_messages_out_of_order`, `test_prover_rejects_calls_out_of_order`, `test_replay_selection_is_derived_from_seed_header_and_boundary_only`, `test_prover_changing_its_boundary_after_seeing_j_is_invalid_opening`, `test_transcript_with_an_altered_selection_is_challenge_mismatch`, `test_make_expectation_draws_fresh_seeds_by_default`; **negative**: `test_reused_seeds_let_the_prover_predict_and_evade_both_selections`.
- `tests/veritor/security/test_local_checks.py`: `test_every_non_source_gate_of_a_sampled_unit_is_checked`, `test_wrong_input_value_is_caught_at_the_boundary_before_any_sampling`, `test_wrong_claimed_output_with_honest_values_is_caught_at_the_boundary`, `test_altered_weight_in_the_run_is_caught_only_when_a_reader_is_sampled`, `test_noncanonical_encoding_of_a_committed_value_is_invalid_value`, `test_value_outside_the_gate_width_is_invalid_value`, `test_evidence_must_open_exactly_the_required_addresses_in_order`, `test_boundary_must_open_exactly_the_public_io_in_order`, `test_gate_arguments_are_the_owners_committed_values_not_the_provers_claims`.
- `tests/veritor/security/test_binding.py`: leaf/root/padding binding; `test_equivocating_on_a_boundary_value_between_phases_is_invalid_opening`; `test_two_units_reading_one_address_cannot_be_shown_different_values`; `test_every_address_has_exactly_one_owner`; `test_the_wire_carries_no_prover_described_domain`; `test_kappa_w_is_bound_to_the_gate_set_and_the_vector_not_the_description`; `test_interior_domain_is_bound_to_the_replay_phase`.
- `tests/veritor/security/test_tiling.py`: `test_every_gate_is_in_exactly_one_replay_unit_and_one_verification_unit`, `test_verification_units_refine_replay_units`, `test_cross_unit_reads_go_only_through_declared_outputs`, `test_marks_leaving_a_gate_uncovered_are_a_compile_error`, `test_nested_or_straddling_marks_are_a_compile_error`, `test_layout_rejects_a_circuit_that_reads_across_the_cut`.
- `tests/veritor/security/test_sampling.py`: `test_acceptance_rate_matches_survival_of_the_error_set`, `test_selection_law_alone_matches_survival_over_many_seeds`, `test_survival_is_the_product_of_per_replay_unit_factors`.
- `tests/veritor/protocol/test_challenge.py`: `test_selection_is_deterministic_and_bound_to_seed_stage_and_phase`, `test_count_inversion_agrees_with_exact_rational_inversion`, `test_marginals_pairs_and_counts_match_independent_coins`, `test_cost_follows_the_selection_not_the_candidates`, `test_derivations_touch_only_selected_units_at_scale`, `test_sample_selection_ranks_the_selected_replay_units_blocks`.
- `tests/veritor/protocol/test_parameters.py`: the admission tests listed in §1.6, plus `test_the_proposal_is_theta_alone_and_the_header_binds_the_verifiers_eta`, `test_a_transcript_recorded_under_another_eta_is_rejected`, `test_huge_denominators_are_rejected_at_admission_and_in_derivation`, `test_expected_work_follows_the_documented_formula`.
- `tests/veritor/protocol/test_weights.py`: `test_one_weight_root_serves_every_batch_shape_of_the_model`, `test_sampled_evidence_opens_weights_under_kappa_w_at_their_ranks`, `test_a_prover_cannot_substitute_its_own_weight_root`, `test_the_weight_domain_is_the_rank_space_and_the_boundary_excludes_the_weight_gates`, `test_ownership_rule_weights_then_boundary_then_interior`, `test_verifier_time_drops_when_weights_leave_the_boundary`.
- `tests/veritor/protocol/test_scaling.py`, `test_verifier_cost.py`: `test_a_full_verifier_run_touches_only_sampled_addresses`, `test_verifier_phases_are_flat_in_the_number_of_replay_units`, `test_verifier_construction_time_is_flat_in_gate_count`, `test_changing_a_role_mark_changes_the_compiled_digest`.
- `tests/veritor/protocol/test_wire.py`, `tests/veritor/security/test_transcript.py`, `test_canonical.py`: canonical bytes; `test_altering_a_recorded_message_is_caught_with_the_expected_code`; `test_transcript_verdict_equals_the_interactive_verdict`; `test_the_recorded_transcript_verifies_only_under_its_own_expectation`.
- `tests/veritor/constructors/test_cluster_protocol.py`, `test_requests.py`: full runs on the toy decoder / continual-batching cluster / per-request units.

---

## 4. Security properties (6.3)

The outline's list (`/tmp/notion-outline-sections-6-7.md:56-93`): *Completeness* — "An honest client is accepted with probability 1." *Capacity soundness* — "For any client strategy, if the verifier accepts with probability greater than $\eta$, then the output lies in the set of outputs reachable with at most $L$ incorrect subcircuits ... the client's best strategy is to commit to the completion that minimizes the number of incorrect subcircuits, which is fixed by the boundary values." *Zero-knowledge* — "The verifier learns nothing beyond $(x, y^*, C, \theta)$ ... [ZK is conditional on the physical assumptions about witnessing the wire]." The current §5 draft has replaced "at most $L$ incorrect subcircuits" by "$\operatorname{err}_{\mathcal P}(\tau)\in\mathfrak E_\eta$", so capacity soundness should be stated with $\sigma_\theta$ and $\mathfrak E_\eta$ (D-0).

### 4.1 Completeness

An honest client (boundary values from the honest $\tau$, interiors by `replay_unit`) is accepted with probability 1: every opening verifies, every gate relation holds, I/O match the header. Bias: the sampler is within $2^{-190}$ of exact Bernoulli coins but this affects only *which* units are drawn, not acceptance. Caveats: admission must pass ($U\le U_{\max}$, $W\le W_{\max}$, $8|a|\le A$, denominators, unit counts) — these are conditions on $(C,I,\theta,a)$, not on honesty; and `VerificationLimits` (e.g. `max_openings`) can reject an honest run with too many sampled openings (`RESOURCE_LIMIT`, `test_a_limit_hit_during_the_run_is_a_reject_not_an_exception`). Tests: `test_honest_run_accepts_and_transcript_round_trips`, `test_honest_run_under_a_weight_root_accepts_and_round_trips`, the cluster/requests protocol tests.

### 4.2 Capacity soundness, part (a): acceptance is at most $\sigma_\theta(E^\star)$ — the $\ell^\star$ lemma

**Statement (audit Claim A, `docs/security-argument.md:336-343`, verbatim):** *"For every prover strategy and every boundary `B` it commits, let `l_r >= 1` for each replay unit `r` that no interior can make consistent with `B` (and `l_r = 0` otherwise). Then `P[accept] <= sigma(E*) + 2^-189` for any `E*` with `|E* ∩ R_r| = l_r`, and the claimed output is an output of a transcript with error set `E*`."* Here $l_r := \min_{E\in F_r(B)}|E|$ where $F_r(B)$ is the family of error sets some interior of $r$ can realize given the committed boundary — this minimum is the outline's $\ell_r^\star$.

**Proof (audit, verbatim, four steps):**

1. *"`B` is fixed before `J` (section 2). Given `B`, each replay unit `r` has a fixed family `F_r(B)` of error sets `E ⊆ R_r` that some interior of `r` can realize (verification units holding a gate whose relation fails on the committed values). `∅ ∈ F_r(B)` iff `Out(R_r)` in `B` is what `R_r` computes from its inputs in `B` and kappa_W. If the claimed output is not `C(x, W)`, then by induction along the address order some `r` has `∅ ∉ F_r(B)`: otherwise every boundary value would be the honest one."*
2. *"The interiors committed after `J` choose one member `E_r ∈ F_r(B)` for each `r ∈ J`; the choice may depend on `J` but only on `J`."*
3. *"Given `J` and the interiors, `T` is the PRF of the still-secret `s` seed over the interior phase; the prover cannot evaluate it, so (up to the sampler bias) `P[T ∩ E = ∅ | J, interiors] = prod_{r ∈ J} (1 - s)^{|E_r|} <= prod_{r ∈ J} (1 - s)^{l_r}` with `l_r = min_{E ∈ F_r(B)} |E|`."*
4. *"Averaging over `J`, which is `Bernoulli(q)` per unit independent of `B`: `P[accept] <= prod_r (1 - q + q (1 - s)^{l_r}) = sigma(E*)`. The claimed output is determined by `B`, and `B` is consistent with an interior whose error set is the minimizing `E*`, so the output is an output of a transcript with error set `E*`."*

**Exact preconditions the proof uses** (each is something §6 must state or §7 must supply):

- (P1) *Boundary before $J$; interiors before $T$; the seeds are secret until revealed and the derivations are PRFs of the phase digests* (audit §2; `phases.py`, `challenge.py`; assumption: HMAC-SHA256 is a PRF; seeds fresh — F1).
- (P2) *Every value a gate of $R_r$ reads from outside $R_r$ is in $\partial$ (or is a weight under $\kappa_W$).* This is what makes $F_r(B)$ depend on $B$ alone, so interiors of distinct replay units are independent given $B$ and the per-unit factors multiply. From §7 Lemma 7.2(ii); enforced at run time by `_Layout.required` (`INVALID_COMPILED_RESULT`); tested by `test_cross_unit_reads_go_only_through_declared_outputs`, `test_layout_rejects_a_circuit_that_reads_across_the_cut`. **This is the only structural hypothesis**, and it holds for every partition $I$ by construction — the reason "nothing else constrains either partition" (§1.6, step 2).
- (P3) *Binding commitments with a single owner per address* (audit §1): a unit reading address $j$ sees the value the owner committed, and two units reading $j$ see the same value (`test_two_units_reading_one_address_cannot_be_shown_different_values`, `test_gate_arguments_are_the_owners_committed_values_not_the_provers_claims`). Assumption: SHA-256 collision resistance.
- (P4) *Inputs pinned at the boundary and again at every sampled `in` gate; weights pinned to $\kappa_W$* — so source gates are never in $E$ and the induction in step 1 starts from honest sources.
- (P5) *Every non-source gate of a sampled unit is checked against the owners' values* (`_check_unit`; `test_every_non_source_gate_of_a_sampled_unit_is_checked`), so $E_r$ is exactly the set of verification units of $r$ holding a failing gate under the committed values.
- (P6) *$(C,I)$ is a deterministic function of the header's $(h_G, x, a)$* (§7 Lemma 7.1), so "the transcript" is over a circuit both parties agree on.

Numerically confirmed: `test_acceptance_rate_matches_survival_of_the_error_set` (statistical), and the eight-gate cheat in §3.7 (0.78 observed vs $\sigma = 0.75$ over 400 runs).

**How the paper should phrase the $\ell^\star$ lemma** (the outline's version, `/tmp/notion-outline-sections-6-7.md:63-65`): *"Deferring the interior commitment gives the client a choice ... The interior with the fewest incorrect verification units is determined by the unit's inputs and outputs, which were fixed before the draw, so the client could have committed to it up front."* The §7 draft's remark (line 92) says what §7 delivers: *"the stitched partial transcript $\widehat\tau$ agrees with $\tau|_\partial$ on every value a selected unit reads from outside."* The §5 draft's parenthetical (line 249) is the promise §6 keeps: *"§6 shows that this does not raise the probability of acceptance above $\sigma_\theta$ of the completed transcript's set of incorrect units."*

### 4.3 The two-stage acceptance function, exactly as coded

`src/veritor/analysis/probability.py` (module docstring verbatim): *"Each replay unit is selected with probability `q`; inside a selected replay unit each verification unit is selected with probability `s`. An error set `E` (verification units holding an incorrect gate) escapes detection with probability*

~~~text
sigma(E) = prod_r f(l_r),   f(l) = 1 - q + q (1 - s)^l,   l_r = |E ∩ R_r|,
~~~

*and is admissible iff `sigma(E) > eta` for the verifier's threshold `eta`. Writing `c(l) = -ln f(l)` and `Lambda = ln(1 / eta)` this is `sum_r c(l_r) < Lambda`: a knapsack over replay units. `c` is increasing and saturates at `-ln(1 - q)`, so many errors in one replay unit cost little more than a few -- concentration is cheap, and that falls out of the formula."* Functions: `survival_factor(policy, l)` $= f(l)$ as a `Fraction`; `survival(policy, (l_r))` $=\prod_r f(l_r)$; `admissible(policy, eta, (l_r))` $\iff \sigma>\eta$ (strict: `test_admissibility_is_strict_at_eta`); `unit_cost(policy, l)` $= -\ln f(l)$ rounded **down** (`cost - cost·2^-40 - 2^-48`); `saturation_cost(policy)` $= -\ln(1-q)$ ($\infty$ at $q=1$); `budget(eta)` $=\ln(1/\eta)$ rounded **up**. This matches the §5 draft's $\sigma_\theta$ exactly. Test: `test_survival_multiplies_over_replay_units_and_ignores_positions` — $\sigma$ depends only on the counts $l_r$, not on which units.

Useful closed forms for the text: with $\Lambda=\ln(1/\eta)$, the adversary affords $\lfloor\Lambda/c(1)\rfloor$ isolated single-unit errors where $c(1) = -\ln(1-qs)$, and $\lfloor\Lambda/(-\ln(1-q))\rfloor$ whole replay units. At $q=1/2$, $s=1/8$: $c(1)=-\ln(15/16)=0.0645$; $\Lambda/c(1) = 10.7,\ 71.4,\ 214$ at $\eta=1/2, 10^{-2}, 10^{-6}$ (measured, `frontier-report.md` "Reading the frontier"); whole units: $0.693$ per unit → $1, 6, 19$ (floor of $1.0, 6.64, 19.9$).

### 4.4 Capacity soundness, part (b): $\mathrm{Bound}(C,I,\theta)$ upper-bounds $\log_2|\mathcal Y_\eta|$

**Statement (audit Claim B):** *"`Bound(C, I, theta).bits` at threshold `eta` is at least `log2 |Y_eta|`, where `Y_eta` is the union, over all error sets `E` with `sigma(E) > eta`, of the outputs of transcripts with error set `E`."*

**Algorithm** (`src/veritor/analysis/bound.py` docstring, §1–3 of the audit). Inputs: the per-kind table (never the flat circuit), $\theta$, $\eta$, `BoundOptions(max_buckets=2048, resolution=16, max_errors=256, knapsack=True)`. Steps:

1. *Downstream cut on index nodes.* If all incorrect gates lie in index nodes $S_1..S_m$, at most $2^{\sum_j \mathrm{out\_bits}(S_j)}$ outputs are reachable (values outside are functions of the declared outputs; inputs and weights pinned). `reference.cover_bits`/`cut_bits` enumerate this for small circuits (`test_cover_by_index_nodes_is_never_below_the_exact_cut`).
2. *Per-kind covers.* `_Fold.series`: a verification kind covers its $l\ge1$ subsets by itself (`unit_series(out_bits)`); a replay kind convolves its children's series over copies (`power`, `multiply`) and `cap`s at its own `out_bits` — the cheaper of "cover the pieces" and "cover the whole node". So $V_K(l)$ = total $2^\kappa$ of the *distinct* covers of the $l$-subsets of one copy of kind $K$. Docstring: *"This is at most the per-set sum `sum_E 2**kappa(E)` and can be far below it when whole units may be corrupted."* (This is the paper's Proposition 5.15 / "Lemma 5.8 applied to their union is tighter than the sum" made algorithmic; and it is the **mega-unit relaxation**: covering all $l$-subsets of a replay unit by the unit's own interface costs one term $2^{\mathrm{out\_bits}(R_r)}$ regardless of $l$.)
3. *Admissibility as a knapsack.* Costs $c(l)$ rounded down onto a grid of `buckets` steps of `cost_step` nats (grid step inflated by $1+2^{-50}$ and the index stepped back so every cost lands on a lower bucket); each replay kind's cost polynomial raised to its copy count (`sparse_power`), kinds convolved, every bucket strictly below $\Lambda$ summed (`prefix_sums`). *"The grid result is exact for the relaxed survival `sigma~(E) = prod_r exp(-cost_step * floor(c(l_r) / cost_step))`, which exceeds `sigma(E)` by less than `exp(cost_step)` per touched replay unit."*
4. *Error-count truncation.* Subsets with more than `errors_limit` errors in a copy are lumped at cost $c(\text{limit}+1)$ — a lower cost, admitting more.
5. *Laplace (Chernoff) bound.* $\min_t\ t\Lambda + \sum_K n_K\ln\sum_l V_K(l)e^{-tc(l)}$ with the same rounded costs; a minimum over $t$ of upper bounds. Used alone (`knapsack=False`) at frontier scale because the knapsack grid would round tiny unit costs to zero.
6. *Cap and integer count.* `bits = min(knapsack, laplace, out_bits(C))`; then `_integer_count` replaces `bits` by $\log_2\lfloor 2^{\text{bits}}\rfloor$ (valid since $|\mathcal Y_\eta|$ is an integer; overnight decision 2.2: *"`Bound` reports the log of an integer count ... a fully checked run ... is now exactly `0.0`"*; F3 fixed the rounding).
7. *Series arithmetic* (`series.py`) rounds every entry up with explicit slack $(\text{terms}+4)\cdot2^{-50}$.

`BoundResult` fields: `bits`, `capped`, `out_bits`, `knapsack_bits`, `laplace_bits`, `cost_step`, `buckets`, `errors_limit`, `policy`, `eta`, `digest`. Rounding discipline (audit): *"Costs down; budget up; grid step up and grid index down; error counts beyond the limit lumped at a lower cost; series entries up; the final cap is an exact minimum; `_integer_count` ... never undercounts the integer ... There is no formal end-to-end error budget (Gap)."* Tests: `tests/veritor/security/test_bound_soundness.py` (`test_union_over_random_markings_is_below_the_fold`, `test_whole_unit_corruption_is_covered_by_the_unit_interface`, `test_fully_checked_run_has_exactly_zero_capacity`, `test_source_only_units_contribute_no_error_terms`, `test_source_only_rule_is_exact_against_the_enumerated_union`), `tests/veritor/analysis/test_bound.py` (`test_fold_sits_between_the_union_and_the_relaxed_per_set_sum`, `test_random_small_circuits_union_is_below_the_fold`, `test_paper_fanin_example_union_is_below_the_fold`, `test_all_outputs_are_reachable_when_nothing_is_checked`, `test_bound_on_the_matmul_counts_the_dots_and_not_the_source_units`, `test_fold_never_enumerates_copies`).

**Source-only rule (F4).** Docstring: *"a unit holding nothing but source gates has capacity `2**0`. Such a unit is never in the error set of a transcript the verifier can accept: every input gate is compared with the public input at the boundary, before any sampling, and a weight gate's only admissible value is its opening under `kappa_W`. The fold therefore gives a kind with no non-source gate the series of the empty subset alone (`l = 0`)."*

### 4.5 Per-strategy versus the union: what was wrong and how it was resolved

The audit found that an earlier reading, $\mathrm{Bound}=\max_E \kappa(E)$ over admissible $E$, certifies only an adversary that has *fixed where its errors go*; the paper's $\mathcal Y_\eta$ is the *union* over admissible $E$, and the choice of $E$ per run carries up to $\log_2|\mathfrak E_\eta|$ additional bits (the "position term" $B\log_2(eN/B)$ of Corollary 5.3, or $B\log_2(eR/B)$ of Corollary 5.16). Resolution: the fold certifies the union — it sums $2^\kappa$ over distinct covers of all admissible error sets (steps 2–3 above), which is at most $\sum_E 2^{\kappa(E)}$ (Theorem 5.9) and typically far less because covers coincide. `docs/overnight-report.md` discrepancy 3: *"`Bound` is the union over admissible error sets, not `max_E`. The audit found the earlier 'per-strategy' reading; the fold now certifies `|⋃_{E admissible} outputs(E)|`. The outline's headline numbers were computed under the per-strategy definition and should be recomputed; expect them to grow by at most the log of the number of admissible covers."* The §5 draft already says it (line 177). The frontier confirms the size of the position term: per admitted error at `cell/gate`, $16 + \log_2(3.47\times10^{13}) \approx 61$ bits, i.e. the position term ($\approx45$ bits) dominates the value term (16 bits) for fine units (measured 60/56/55 bits per error at the three $\eta$).

**Discrepancy D-8 (per-kind folds vs per-cut sums).** The paper proves per-cut bounds (Theorem 5.9, Proposition 5.15: sum over $\mathfrak Q_\eta$ of $2^{W(D_Q)}$ with $D_Q$ the narrowest downstream cut of the union). The code's cuts are *index nodes only* (a unit or an ancestor kind), not arbitrary downstream cuts; its covers are chosen per kind and convolved over copies. So the code computes a valid but possibly looser instance of Theorem 5.9 (it uses a restricted family of cuts). Recommendation: §6 should say "Bound instantiates Theorem 5.9 with the cuts available from $I$ — the declared interfaces of its nodes — so that it runs in time polynomial in the number of kinds"; the tightness numbers of §5.6 (regions, $\kappa_{\max}=32$) were computed on the flat GPT-2 circuit by the cut analysis, not by this fold. (Whether the paper's headline numbers use §5's cut analysis or §6's fold must be decided; D-8 in §9.)

### 4.6 Zero-knowledge

Outline: *"The verifier learns nothing beyond $(x,y^*,C,\theta)$ (and $\eta$). [ZK is conditional on the physical assumptions about witnessing the wire]."* What the transparent protocol reveals (audit §10 "No privacy"): *"Every opened value is in the clear; the transcript reveals inputs, outputs, sampled interiors and sampled weights."* Also revealed: the roots (binding, not hiding — plain SHA-256 Merkle leaves without blinding), $J$, $T$, and, through the openings, the values at $\mathrm{required}(v)$ for $v\in T$ (including boundary values of neighbouring units and weights). What the ZK construction hides: all committed values (hiding commitments — blinded leaves or a hiding vector commitment), replacing each opening + local check by a proof that the committed values satisfy the sampled relations and that the I/O positions equal the public $x$, $y^*$. What it cannot hide: $(C,I)$'s structure (public by design; $h_C$ is in the header), $|\partial|$, $|J|$, $|T|$, $\theta$, $\eta$, and the fact of acceptance. The "physical assumption about witnessing the wire" (outline) refers to §3's threat model: ZK is about the verifier's view of the *protocol*, and the exfiltration bound is about the *output*; the paper must say ZK protects the server's weights/activations from the verifier, not the user from the server. **The prototype implements no hiding and no proofs**; §6.4 must be written as a construction whose ZK instance is specified but not implemented (D-11).

### 4.7 Security audit findings that concern §6 (`docs/security-argument.md` "Findings"; status as of Sep 2)

- **F1 — Seed reuse defeats the protocol** (medium, operational; documented, not fixable in the protocol). *"A verifier that reuses a session's seeds lets the prover predict `J` and `T` and forge acceptance with certainty."* Proposed fix: derive seeds as `HMAC(master, session_id || tag)`. For §6: the paper must state that the verifier's randomness is fresh per session and secret until revealed; soundness is per session ($k$ attempts: $\le k\sigma(E)$).
- **F2 — Default `U_max = None` waived the capacity check** (medium; fixed). `max_capacity` is now a required keyword; `None` (waiver) must be written. For §6: $U_{\max}$ is part of the verifier's statement, not optional.
- **F3 — `_integer_count` could undercount by an ulp** (low; fixed). Bound now never undercounts and never exceeds its input.
- **F4 — Source-only kinds counted with $l\ge1$** (looseness; fixed). The $l=0$ rule.
- **F5 — $\kappa_W$ is not self-describing** (low, by design). Root binds gate set + vector, not the model name; the verifier chooses which root a request runs under. For §6: state the deployment rule ("the verifier holds $\kappa_W$ before the epoch").
- **F6 — Dead limits** (informational): `max_nesting_depth`, `max_artifact_bytes` unused. Not paper-relevant.
- **F7 — Output over-resolution in the run resolver** (medium; fixed ba43d29). A compiler bug that inflated `Out` (sound direction for Bound, wrong for Cost, and a liveness failure). For §6: the boundary is what the compiler *declares*; correctness of `Out` is §7's Lemma 7.2 and was fuzzed.

Verdict table rows relevant to §6: 1 (binding), 2 (staging; seed freshness gap), 3 (sampling law), 4 (local checks), 5A/5B (Claims A/B), 6 (admission), 9 (offline verification): all "proved + tested" except seed freshness (gap, convention) and the float error budget (sound by rounding discipline, no machine-checked budget).

### 4.8 Every assumption the soundness argument needs, by source

From **§7 / the compiler**: (i) $(C,I)$ deterministic from $(h_G, x, a)$ and identical on both sides (`compiled.digest` in the header); (ii) replay marks tile $[n]$, verification marks tile each $R_r$; (iii) $\partial\supseteq$ every cross-replay-unit read and every circuit output; a weight gate is never a claimed output; (iv) per-kind table exact (all copies of a kind have identical size/cost/interface); (v) $O(\text{depth})$ queries so the verifier is sublinear (efficiency, not soundness); (vi) at most $2^A$ pairs $(C,I)$ per $(G,x)$ — charged at admission.

From **cryptography**: (vii) SHA-256 collision resistance (Merkle binding, header digest, phase chain); (viii) HMAC-SHA256 is a PRF (selections unpredictable before the seed reveal); (ix) a CSPRNG for seeds and session ids (`secrets`); (x) for ZK: hiding commitments and a sound, zero-knowledge proof system for the sampled relations (not implemented).

From **the verifier's discipline**: (xi) $\eta$, $U_{\max}$, $W_{\max}$, $A$ fixed by the verifier and $\eta$ bound into the header; (xii) seeds fresh and secret until their challenge; (xiii) session ids unique; (xiv) $\kappa_W$ obtained from a trusted source before the epoch; (xv) attempts bounded (soundness per session); (xvi) $G$ sandboxed or compilation proven (§7).

From **§5**: (xvii) Theorem 5.6/Corollary 5.7 (downstream cuts determine outputs), Lemma 5.1 (well-typed transcripts suffice), Theorem 5.9 (sum over $\mathfrak E_\eta$) — the fold instantiates these with index-node cuts.

---

## 5. The simple construction (6.4)

### 5.1 Indexed Merkle commitments over the common address space

The outline (`/tmp/notion-outline-sections-6-7.md:96-103`): *"Simple construction: indexed Merkle commitments over the common address space $[n]$; one boundary tree; one interior tree per selected replay unit; batched ZK proofs of sampled verification units with an off-the-shelf zkVM."* The implementation is exactly this, with one addition ($\kappa_W$) and one precision (positions are addresses, but ranks are domain ranks):

- **One boundary tree** over `Index.boundary()` ($|\partial|$ leaves, rank order: inputs by rank, then each replay unit's declared outputs), bound to the header digest. Leaf for address $i$: $H(\text{"leaf"}, \text{domain\_id}, \mathrm{rank}_\partial(i), i, \text{"u}w_i\text{"}, \text{value})$.
- **One interior tree per selected replay unit** over `Index.interior(r)` ($|\mathrm{Int}(r)|$ leaves), bound to the replay-phase digest (hence to the header, the boundary root, and $J$). A prover cannot reuse an interior tree from another session or phase (`test_interior_domain_is_bound_to_the_replay_phase`, `test_boundary_message_replayed_from_another_session_is_invalid_opening`).
- **One weight tree $\kappa_W$** over ranks $0..|W|-1$, bound to a fixed tag and the gate set; built once per model (`commit_weights`), reused across every circuit compiled from the model (`test_one_weight_root_serves_every_batch_shape_of_the_model`).
- **Domain separation**: `domain_id` = $H(\text{"domain"}, \text{binding}, \text{owner}+2, \text{positions.identity\_digest}, \text{count})$; every leaf, node, padding and empty root includes `domain_id`, so a root cannot be reused across sessions, phases, owners or position sets (`test_root_is_bound_to_binding_owner_position_set_and_count`, `test_interior_committed_under_the_boundary_domain_is_invalid_opening`, `test_boundary_committed_under_an_interior_domain_is_invalid_opening`).
- **Openings** are `(position, value, path)`; the verifier computes the rank from the position via the domain (no prover-described layout: `test_the_wire_carries_no_prover_described_domain`) and checks the path length equals `merkle_depth(count)`.

This realizes the author's requirement (msg 496/522): the verifier builds everything from the address space and hashing; nothing is materialized.

### 5.2 From transparent to zero-knowledge, by replacement

Scratch: *"replace $\beta$ and each $\alpha_r$ with commitments made at the same points in the interaction. At each phase, the client proves the checks that the transparent verifier would have completed before releasing the next challenge. After the final challenge, the client proves that every sampled gate equation is satisfied by the committed values."* Concretely, phase by phase:

| transparent verifier does | staged prototype (implemented) | ZK construction (specified) |
|---|---|---|
| receives $\beta$, checks $\mathrm{dom}(\beta)=\partial$, types, $x$, $y^*$ | receives $\kappa_\partial$ (root, count $=|\partial|$) and openings of the I/O positions; checks them | receives a hiding $\kappa_\partial$ and a proof that the committed I/O positions hold $x, y^*$ and that every committed value is well-typed (or defers typing to the sampled checks; Lemma 5.1) |
| flips $m$ coins | derives $J$ from `q_seed`, reveals the seed | same |
| receives $\alpha_r$ for $r\in J$, checks domains and types | receives $\kappa_r$ for $r\in J$ | receives hiding $\kappa_r$ |
| flips coins inside $J$ | derives $T$ from `s_seed`, reveals it | same |
| checks each gate of each $v\in T$ on $\widehat\tau=\beta\cup\bigcup_r\alpha_r$ | receives openings of $\mathrm{required}(v)$ under the owners' roots; checks paths, types, relations | receives one batched proof that, for every $v\in T$, the committed values at $\mathrm{required}(v)$ satisfy the gate relations of $v$ (Merkle paths and gate evaluation inside the proof) |

The equivalence the paper needs to state: the ZK verifier accepts iff the transparent verifier would have accepted the same values (soundness of the proof system) and learns nothing else (hiding + ZK). Because the specification is the transparent verifier, every soundness statement of §6.3 is about the transparent verifier and transfers.

### 5.3 The proof system: SP1 plans and the numbers §6 can cite

`docs/sp1-benchmark-plan.md` (Aug 22) is a **plan**, not results: SP1 (RV64IM zkVM; precompiles for SHA-256/Keccak/Poseidon2, no float precompiles), a matrix of guest operators (`f32_dot`, `fixed_dot`, `softmax_row`, `merkle_sha256` at depth 20/30, `leaf_check` = read operands + auth path, verify path, recompute operator, commit), Phase A execution-only cycle counts, Phase B local proving calibration, Phase C derived seconds and bytes per sampled check. Priors listed there (to validate, not trust): soft-float f32 add/mul 30–150 RV64IM instructions; a 4096-dot ≈ 0.3–1.5M cycles; SHA-256 precompile ≈ $10^2$ cycle-equivalents per block vs pure-Rust 3–8K; local core proving $O(10^4$–$10^5)$ cycles/s. **No measured proving numbers exist in the repository.** The outline's "$10^{10}\times$ native" (`/tmp/notion-outline.md:62,81,405`) and "Attestable 50–100×, fixed-point matmul only" (`:62`) are both marked `[source]` in the outline and are unsourced here. For §6.4 the paper needs only: proofs are batched per run ($|T|$ units in one proof, amortizing the fixed overhead $c_0$), Merkle paths are verified inside the proof (depth $\lceil\log_2|\partial|\rceil$ for boundary leaves, $\lceil\log_2|\mathrm{Int}(r)|\rceil$ for interior leaves), and the relation checked is `check_gate` per gate.

### 5.4 The eight-gate example's concrete objects and function calls

The author asked (Aug 31, 12:08–1:33 PM, msgs 496–535) for the walkthrough in terms of *primitive* operations: circuit → `gate(i)`, `size`; index → unit membership, interface, rank; commitment → hash; sampling → randomness. The final agreed version is the Scratch page's transparent walkthrough (§3.1) plus "the verifier does everything". In the code's primitives, the same run (§3.7) is:

~~~text
# Setup (both parties)
compiled = Compile(G, x=(1,1,2,3), a=b"")             # (C, I), digest H(C,I)
∂  = compiled.index.boundary()                          # ranks: 1,2,5,6 | 4 | 8   (paper addresses)
Int(1) = compiled.index.interior(0) = {3};  Int(2) = compiled.index.interior(1) = {7}
header = Header(session, H(C,I), h_G, a, θ=(1/2,1/2), η=1/100, x=[1,1,2,3], y*=[20], κ_W=None)

# Message 1 (client): boundary tree over ∂ in rank order, values β = (1,1,2,3,4,20)
κ_∂ = MerkleTree(boundary_domain(header, compiled), {1:1, 2:1, 5:2, 6:3, 4:4, 8:20}).root
send BoundaryMessage(κ_∂, count=6, io_openings = open(1), open(2), open(5), open(6), open(8))
# Verifier: verify 5 paths of length 3; compare 1,1,2,3 with x and 20 with y*
J = bernoulli_subset(q_seed, "q/replay-unit", boundary_phase, count=2, p=1/2)  -> {R_2}
send ReplayChallenge(q_seed, J)

# Message 2 (client): replay R_2 from the boundary: τ_7 = β_5 + β_6 = 5
κ_2 = MerkleTree(interior_domain(replay_phase, compiled, r=2), {7: 5}).root        # 1 leaf, depth 0
send InteriorMessage([κ_2])
T = bernoulli_subset(s_seed, "s/verification-unit", interior_phase, count=|V(R_2)|=2, p=1/2) -> {V_{2,2}}
send SampleChallenge(s_seed, T)

# Message 3 (client): required(V_{2,2}) = {4:∂, 5:∂, 6:∂, 7:Int(2), 8:∂}
send EvidenceMessage([ open_∂(4)=4, open_∂(5)=2, open_∂(6)=3, open_2(7)=5, open_∂(8)=20 ])
# Verifier: 4 boundary paths (len 3) + 1 interior path (len 0); decode u16; then
#   gate 7 = add(5,6): check 5 == 2+3     gate 8 = mul(4,7): check 20 == 4*5
# -> ACCEPTED, J={R_2}, T={V_{2,2}}; transcript 3,074 bytes; verify_transcript re-derives J, T and accepts
~~~

The ZK version replaces the last message by one proof over the five committed leaves and the two relations, and the first message's five openings by a proof that leaves $1,2,5,6,8$ hold $x$ and $y^*$.

### 5.5 What the construction does *not* do (so the paper does not overclaim)

- No hiding, no proofs (transparent only).
- No aggregation of many runs; one transcript per request.
- The verifier holds $\kappa_W$ out of band; nothing binds a root to a model name (F5).
- The compile step is trusted in the prototype (the verifier runs $G$; audit §10) — §7's job.
- Per-session soundness; retries are the operator's.

---

## 6. Cost model (6.5)

### 6.1 The outline's terms and the coded formula

Outline (`/tmp/notion-outline-sections-6-7.md:105-118`): the prover's cost has three terms — *"boundary commitment (hashing $|\partial|$ values), replay ($q\cdot\sum_r$ (replay cost of $R_r$ + hashing its interior)), proof ($q s\cdot\sum_v$ (proof cost of $V_v$ + fixed overhead))"* — *"determined by: verification-unit granularity, replay-unit granularity, proving cost, replay cost net of commitments, replay share $q$, proving share $s$."*

`src/veritor/analysis/cost.py` (module docstring, verbatim): *"Committing the boundary `∂ = In ∪ ⋃_r Out(R_r)` costs `h` per position; a replay unit, selected with probability `q`, costs its replay and `h` per interior position it commits (its gates but `Out` and its pinned source gates); a verification unit, selected with probability `q s`, costs its proof and a fixed `c_0`:*

~~~text
Cost = h |∂| + q sum_r (Cost_replay(R_r) + h |Int(r)|) + q s sum_v (Cost_proof(V_v) + c_0)
~~~

*The weights are committed once per epoch under κ_W, `h |W|`, reported separately from the per-run total. Everything is a count per kind weighted by copies, so the fold is `O(#kinds)` and exact."*

$$\mathrm{Cost}(C,I,\theta) = h\,|\partial| \;+\; q\sum_{r}\bigl(\mathrm{replay}(R_r) + h\,|\mathrm{Int}(r)|\bigr) \;+\; qs\sum_{v}\bigl(\mathrm{proof}(V_v) + c_0\bigr),\qquad \text{plus } h|W| \text{ per epoch.}$$

Code (`cost.py:60-88`): `boundary = input_count + Σ_{replay kinds} copies·out_count`; `interior = size − out_count − source_inputs − source_weights` per replay kind; `replay = Σ copies·(replay_cost + h·interior)`; `proof = Σ_{verification kinds} copies·(proof_cost + c0)`; returns `ExpectedCost(h·boundary, q·replay, q·s·proof, h·weight_count)`; `.total` excludes `weights`. `CostParameters(hash_cost=h=1, proof_overhead=c0=0)` defaults. Tests: `test_cost_fold_matches_the_unit_by_unit_sum`, `test_cost_fold_matches_on_nested_indices`, `test_cost_terms_and_defaults`, `test_the_weight_commitment_is_priced_per_epoch_not_per_request`.

**Units.** Costs are in "gate-equivalents": the gate set's `replay_cost` (`add/sub/lt/eq/shr` 1, `mul` 2, sources 0) and `proof_cost` (same for arithmetic, sources 1). $h$ is the cost of one hash in the same unit. The honest cost base is `honest_cost(table) = replay_cost of the root kind` (`frontier.py:129-132`), i.e. the replay cost of the whole circuit — *"what the honest computation costs in the same units"*. Overhead is reported as `Cost.total / honest_cost`.

**What determines each term.**

- $h|\partial|$: replay-unit granularity (how many declared outputs) and $|x|$. Paid every run regardless of $\theta$. Negligible for the 70B shape at every partition ($|\partial|/\text{honest}$: $5\times10^{-12}$ for `request/*`, $6.7\times10^{-6}$ for `step/row`, $8.4\times10^{-5}$ for `cell/gate`, measured).
- $q\sum_r(\mathrm{replay}+h|\mathrm{Int}|)$: the replay share $q$ times (the honest cost plus hashing every interior value). For the 70B shape $\sum_r(\ldots)/\text{honest} = 1.6666$ at every partition (measured): $1$ for the recomputation plus $n_{\mathrm{int}}\cdot h/\text{honest}\approx 0.667$ for hashing (there are $n/\text{honest} = 0.667$ values per cost unit because `mul` costs 2).
- $qs\sum_v(\mathrm{proof}+c_0)$: $\sum_v\mathrm{proof}/\text{honest} = 1.0000$ (proof costs equal replay costs for arithmetic gates; sources add $|x|+|W|$, negligible). With $c_0 = 0$ the proof term is $qs\times$ honest.
- Hence, for the 70B shape, **prover overhead $\approx q\,(1.667 + s)$** at every partition (measured: `(overhead/q − s)` ∈ {1.6666, 1.6667, 1.6672, 1.6693, 1.6773, 1.7097, 1.8391, 2.3568} across all 588 points; the larger values are the $q=1/8192$ points where the constant $h|\partial|/q$ shows). At $q=1/2, s=1/8$: $0.896$; at $q=1/2,s=1$: $1.33$; at $q=1/2048, s=1$: $0.00139$.

### 6.2 The verifier's expected work and $W_{\max}$

`src/veritor/protocol/parameters.py::expected_work` (docstring verbatim): *"One operation is counted per opened leaf, per Merkle path hash, per gate check and per commitment received:*

~~~text
W = (|IO| + q s A) (1 + d) + q s S + 1 + q R
~~~

*with `R` replay units, `A = sum_k m_k (size_k + in_k)` and `S = sum_k m_k size_k` over the verification kinds (`m_k` copies of `size_k` gates with `in_k` declared inputs, which bound the outside addresses a copy reads; a copy is sampled with probability `q s`), and `d = merkle_depth(n)` bounding every path length. Evaluated in `O(#kinds)` from the per-kind table alone: nothing here enumerates an interface, so a client cannot make admission cost depend on the size of its inputs."* `positions_per_unit(kind) = size + input_count` for verification kinds, `size` for replay kinds. `|IO|` is the number of distinct public I/O addresses (inputs plus outputs; weights excluded — the §7 draft's cross-check (b) is satisfied). `DEFAULT_MAX_WORK = 1 << 32`. Admission: $W\le W_{\max}$ else `WORK_BUDGET_EXCEEDED`. Test: `test_expected_work_follows_the_documented_formula`.

For the 70B shape: $d = \mathrm{merkle\_depth}(2.7\times10^{17}) = 58$; **verifier work $= 118.7\,qs$ honest-costs** for `gate` verification units (measured to four digits across all `*/gate` points) and $79.13\,qs$ for `row` units. The $q R$ term (one operation per selected replay unit) and $|IO|(1+d)$ are negligible. At $q=1/2, s=1/8$: $7.42$; at $q=1/2048, s=1$: $0.058$; at $q=1/2, s=1$: $59.3$. Note this "work" counts one hash per Merkle path node and one gate check per gate in the same unit as the prover's replay; it is not the verifier's *proof verification* cost in the ZK construction (which would be roughly constant per batched proof) — the paper should say which verifier it prices (transparent: $W$; ZK: proof verification plus $|\mathrm{io}|$ openings).

### 6.3 Numbers for the toy circuits (measured with `/tmp/s6/eight.py`, `/tmp/s6/toys.py`)

Eight-gate circuit ($h=1$, $c_0=0$; honest cost 6): boundary $6$, replay $q\cdot(4+4)$, proof $qs\cdot 10$; at $q=s=1/2$: $6 + 4 + 2.5 = 12.5$ ($2.08\times$ honest); $W = 37$.

Default matmul fixture (`compile_matmul()`: $n=45$, 9 inputs, 6 weights, 5 replay units, 21 verification units, $|\partial|=15$, $W_{\mathrm{out}}=48$ bits, honest 48):

| $q$ | $s$ | boundary | replay | proof | total (× honest) | $W$ (× honest) | $U$ at $\eta=10^{-2}$ |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 15 | 72 | 63 | 150 (3.125) | 723 (15.1) | 0 |
| 1/2 | 1/2 | 15 | 36 | 15.75 | 66.75 (1.39) | 261.5 (5.45) | 48 (cap) |
| 1/4 | 1/2 | 15 | 18 | 7.875 | 40.9 (0.85) | 183.75 (3.83) | 48 (cap) |
| 1/8 | 1/4 | 15 | 9 | 1.97 | 26.0 (0.54) | 125.75 (2.62) | 48 (cap) |

Larger matmul ($4$ batches $\times 16$ rows $\times (32\times32)$, 8-bit words: $n = 132{,}096$, 2,048 inputs, 1,024 weights, 66 replay units, 5,120 verification units, $|\partial| = 4{,}096$, $W_{\mathrm{out}} = 16{,}384$ bits, honest 194,560):

| $q$ | $s$ | total (× honest) | $W$ (× honest) | $U$, $\eta=1/2$ | $U$, $\eta=10^{-2}$ |
|---|---|---|---|---|---|
| 1 | 1 | 523,264 (2.69) | 5,210,179 (26.8) | 0 | 0 |
| 1/2 | 1/2 | 214,272 (1.10) | 1,360,930 (7.0) | 262.0 | 1,593.0 |
| 1/4 | 1/2 | 109,184 (0.56) | 719,378 (3.7) | 523.0 | 4,144.8 |
| 1/8 | 1/4 | 50,464 (0.26) | 238,217 (1.22) | 1,302.9 | 8,816.6 |

`optimize` on this circuit: `max_cost = 2× honest, η = 1/100, PolicyGrid.uniform(8)` → $q=1, s=1/4$, $U=259.7$ bits, cost 375,040 (81 grid points evaluated); `max_bits = 200` → $q=1, s=3/8$, $U=152.5$, cost 399,744. Both pick $q=1$: at toy scale the replay term is cheap relative to the boundary and every replay unit's interface (a row of outputs) is a large share of the output, so the optimizer refuses to leave any replay unit unchecked.

### 6.4 Numbers for the 70B `FRONTIER_SHAPE` (from `docs/data/frontier-70b.json`; verified today)

Shape: `vocab=32768, d_model=8192, heads=64, layers=80, prompt=512, generated=512, requests=2048, batch=32, hidden_multiplier=4, width=16`; $n = 2.744\times10^{17}$; honest cost $4.1168\times10^{17}$ (identical for all seven partitions); inputs $1{,}048{,}576$, outputs $1{,}048{,}576$ ($16{,}777{,}216$ output bits), weights $6.496\times10^{10}$. Per-partition at $q=1/2, s=1/8$ (measured):

| partition | replay units | $|\partial|$ | boundary/honest | replay/honest | proof/honest | total/honest | $h|W|$/honest (epoch) |
|---|---|---|---|---|---|---|---|
| `request/row` | $2.05\times10^3$ | $2.10\times10^6$ | $5.1\times10^{-12}$ | 0.8333 | 0.0625 | 0.8958 | $1.58\times10^{-7}$ |
| `step/row` | $3.28\times10^4$ | $2.75\times10^{12}$ | $6.7\times10^{-6}$ | 0.8333 | 0.0625 | 0.8958 | same |
| `cell/gate` | $3.47\times10^{13}$ | $3.47\times10^{13}$ | $8.4\times10^{-5}$ | 0.8333 | 0.0625 | 0.8959 | same |

Headline `certify` results (measured; agree with `docs/frontier-report.md`):

| $\eta$ | overhead $\le 1$, work $\le 10$ | overhead $\le 1\%$, work $\le 10\%$ |
|---|---|---|
| $1/2$ | `cell/gate` $q=1/2,s=1/8$: $U = 640.0$ bits (0.0038%), overhead 0.896, work 7.42 | `cell/gate` $q=1/2048, s=1$: $U = 74{,}636$ (0.445%), overhead 0.00139, work 0.0579 |
| $1/100$ | same policy: $U = 4{,}021.3$ (0.0240%) | same policy: $U = 470{,}104$ (2.80%) |
| $10^{-6}$ | same policy: $U = 11{,}711.7$ (0.0698%) | same policy: $U = 1{,}365{,}477$ (8.14%) |

Points at the cap: 436 of 588 (all 336 points of `step/row`, `layer/row`, `matvec/row`, `row/gate`, plus 100 of the others at the cheapest policies). The report's own decomposition: $U\approx(\Lambda/c(1))\times(\text{bits per error})$ with $\Lambda/c(1) = 10.7, 71.4, 214$ and bits/error $= 59.6, 56.4, 54.7$ at `cell/gate`, $q=1/2$, $s=1/8$, versus $16+\log_2(3.47\times10^{13}) = 61.0$.

### 6.5 Is $h=1$ consistent with "16% for hashing everything"? What to say about calibrating $h$

Under $h=1$ (one hash costs one `add`), hashing every value of the 70B circuit costs $n\cdot h/\text{honest} = 0.667$ — **67% of the honest computation**, not 16%. The two numbers are not about the same thing (the outline's is "for GPT-2"; the report's is the toy ISA's cost units on a 70B shape), but they cannot both be right for the same $h$: 16% would need $h\approx0.24$ in these units (or a circuit with far fewer values per unit of cost, e.g. wide fused gates). Conversely the outline's own "1% budget" is not met by the frontier's headline policies ($0.896\times$ overhead at $q=1/2, s=1/8$); the 1%-overhead column of the calibration table exists ($q\le1/128$) and buys $U$ in the 100 kbit–3 Mbit range at $\eta=1/100$. Recommendations for the text:

1. Define $h$ explicitly as "the cost of committing one value, in units of one gate evaluation of $\Sigma$", state the default $h=1$ as conservative for an arithmetic hash on the same hardware, and present cost as a function of $h$ where it matters (the replay term $q(1 + h\,n_{\mathrm{int}}/\text{honest})$ and the boundary term $h|\partial|$).
2. Either derive the 16% (values per FLOP for GPT-2 under a stated hash cost) or drop it; if kept, it fixes $h$ and the frontier overheads should be restated at that $h$ (they scale as $q(1 + 0.667h + s)$).
3. Say plainly that proof cost is modelled as $c_{\mathrm{proof}}$ per gate with $c_{\mathrm{proof}} = c_{\mathrm{replay}}$ in the prototype's units, i.e. the $10^{10}\times$ zkVM factor is *not* in the cost model; the frontier's "overhead" is therefore *replay-and-hash* overhead, and the proving cost is a separate multiplier on the $qs$ term. The outline's §6.1 sentence ("proving is about $10^{10}\times$ native, so a whole replay unit is far too expensive to prove") is the qualitative reason $s\ll1$ matters; the numbers in §6.5 need $c_{\mathrm{proof}}$ calibrated from SP1 (§5.3, not yet measured).
4. Cite the honest-server numbers in the form "overhead $\approx q(1.67+s)$ honest-costs, verifier work $\approx 119\,qs$", with the caveat that both are unit counts, not measured time (`frontier-report.md` "Caveats").

---

## 7. Parameter optimization (6.6)

### 7.1 `optimize.py` as implemented

Module docstring: *"`Optimize`: the client's search for `theta` over a rational grid. Advisory and untrusted: the verifier fixes `eta` and checks `Bound(C, I, theta) <= U_max` itself, so nothing here is load-bearing. The search evaluates every grid point (each `Bound` takes milliseconds) and keeps the cheapest policy whose bound fits, or -- given a budget instead -- the tightest bound the budget affords."* `PolicyGrid(q, s)` (all pairs; `PolicyGrid.uniform(steps)` = $\{0, 1/\text{steps},\ldots,1\}^2$); `optimize(target, eta, grid, *, max_bits=None, max_cost=None, parameters=None, bound_options=None, accept=None)` — exactly one of `max_bits`/`max_cost`; `accept` may veto (e.g. $W_{\max}$ through `expected_work`); ties break toward the smaller secondary objective then the earlier grid point; returns `Optimization(policy, bound, cost, evaluated)` or `None`. Tests: `test_cheapest_policy_under_a_capacity_limit`, `test_full_checking_is_the_only_way_to_zero_capacity`, `test_extra_constraints_and_options_are_honoured`.

### 7.2 The outline's minimax framing and the Aug 29 appendix

Outline (`/tmp/notion-outline-sections-6-7.md:120-126`): *"maximize exfiltration security subject to a fixed total-cost budget; solve minimax ... [the two-stage minimax solution from Appendix A]"*. The author's Appendix A draft (Saturday, Aug 29, 2026, 4:49 PM, msg 210), verbatim:

> **The minimax problem.** Suppose replaying a request costs $\gamma$ times its original computation, proving a sampled verification unit costs $\alpha$ times its original computation, and the total overhead budget is $\beta$. In two-stage sampling, a request is replayed with probability $q$ and each verification unit within it is then sampled with probability $s$, giving marginal sampling probability $p=qs$. If a request has $\ell$ incorrect units, it passes with probability $1-q+q(1-s)^\ell$. Writing $A(\ell)$ for the maximum capacity of such an attack and $\Lambda=\ln 100$, the minimax problem is
> $$\min_{q,s}\;\Lambda\max_{1\le\ell\le N}\frac{A(\ell)}{-\ln\bigl(1-q+q(1-s)^\ell\bigr)}\quad\text{subject to}\quad\gamma q+\alpha qs\le\beta.$$
> For the approximation $A(\ell)=\min\{A_{\mathrm{req}},a\ell\}$, let $r=A_{\mathrm{req}}/a$ and let $c_\gamma>0$ solve $e^{c_\gamma}=1+c_\gamma+\gamma r/\alpha$.
> **Solution.** Under the small-probability approximation, $s^\star=c_\gamma/r$, $q^\star=\beta/(\gamma+\alpha s^\star)$, $p^\star=q^\star s^\star$, $C_\gamma/C_0=e^{c_\gamma}$, where $C_0=\Lambda a\alpha/\beta$ is the ideal one-stage capacity when replay is free.
> **Empirical calibration.** At $\alpha=10^6$, $\beta=0.01$, $A_{\mathrm{req}}=17{,}000$, $a=86$: [table: $\gamma = 0 \to q^\star=1, s^\star=10^{-8}$, 5.0 GB, 1.0×; $\gamma=1 \to 9.9\times10^{-5}, 10^{-4}$, 5.0 GB; $\gamma=10 \to$ 5.3 GB, 1.1×; $10^2\to$ 6.0 GB, 1.2×; $10^3\to$ 8.7 GB, 1.8×; $10^4\to$ 22 GB, 4.5×; $10^5\to$ 120 GB, 24×.]

Three things to flag about this appendix relative to the current state: (i) it models one replay unit per *request* with $A(\ell)=\min(A_{\mathrm{req}}, a\ell)$ (per-request cap 17,000 bits, 86 bits per error) — the same structure the fold computes, but with the cap now the unit's `out_bits` and the per-error term $\mathrm{out\_bits}(V)+\log_2(\#\text{units})$; (ii) its capacities ("5.0 GB") are for a whole deployment at $p\approx10^{-8}$ and a $10^6$ proving factor, and are not comparable with the frontier's per-run bits at $qs\ge 1/16384$; the paper should use one of the two consistently; (iii) the closed form is a small-probability approximation the code does not use — `optimize` is a grid search over exact `Bound`. Recommendation: keep the minimax *statement* (it is the right framing: $\min_\theta \mathrm{Bound}$ s.t. $\mathrm{Cost}\le\beta$, or $\min_\theta\mathrm{Cost}$ s.t. $\mathrm{Bound}\le U_{\max}$) and replace the closed-form solution by the grid search and the frontier tables; present the closed form, if at all, as intuition for why $s^\star$ is small when $\gamma/\alpha$ is small.

### 7.3 Who chooses what — the author's Sep 2 decision

The sequence on Wednesday, Sep 2, 2026 (msgs 1081–1087), verbatim:

- 7:40 AM (msg 1081), on the cut-vs-reach finding: *"I would have assumed for the KV state width we would break it up. I don't understand why that's any different from like the fact that a dot product, you know, like a matrix multiplication consists of many dot products, each of which has one small output."*
- 7:46 AM (msg 1083): *"Okay, wait so I was thinking that the replay unit would be like an entire inference request. We would have a bunch of these and then like the verification unit would be smaller. It's not obvious to me why it matters what we choose as the replay unit though. I thought everything cared about verification units."*
- 7:51 AM (msg 1085): *"Sorry, but if we don't get to choose what the replay units are, then this doesn't help us much. options here are we show what an honest inference server does so that we can calibrate the max bound to that and an honest inference server minimizes its stuff or alternatively we think about how we can place policies on the replay units. I dramatically prefer the first one I think."*

The assistant's reading (msg 1087), which drove the frontier work: *"since the adversary chooses the replay units rather than the verifier, the earlier observation about tighter bounds with request-level units doesn't necessarily help ... an adversary who chooses bad units just gets a larger computed bound and fails the $U\le U_{\max}$ check, so soundness isn't threatened. But the real concern is calibration—the verifier still has to pick $U_{\max}$ at a level honest servers can actually meet given their natural circuit/partition ... `Optimize` needs to search over partition granularity as well as $\theta$."*

So the paper's position for §6.6 is: **the server chooses $I$ (its marks) and $\theta$; the verifier fixes $\eta$, $U_{\max}$, $W_{\max}$, $A$ and enforces them at admission; $U_{\max}$ is calibrated to what an honest server achieves under a stated cost budget** (the honest-server frontier), not imposed as a rule about units. The answer to msg 1083 ("why does the replay unit matter") is §1.7: because with probability $1-q$ a replay unit is unchecked in its entirety, its whole interface enters $\mathfrak E_\eta$.

### 7.4 The honest-server frontier: `serving.py`, `frontier.py`

`src/veritor/evaluation/serving.py` writes the per-kind table the compiler would produce for the toy decoder of `constructors/lm.py` at any dimensions (docstring: *"the structure is the toy's, gate for gate: `dot_k` is `k` products and a sum tree, `attend_head_c` is `c` scores, their squares, the mix and the shift, a token's layer is four projections, the heads, two residual sums and the square MLP, a request is a prefill over its prompt then one decode step per generated token, and the run is the weights unit followed by the requests"*). Replay levels: `request` (one unit per request), `step` (one unit per synchronous decode step of a batch, `ClusterG`'s layout), `layer` (one token's layer), `matvec` (one matrix-vector product or one head), `row` (one dot product / head / one-hot / argmax / block of residual or square cells), `cell` (*"every unit has one output: a dot product or a single gate, so no unit's cut exceeds a word"*). Verification levels: `layer`, `row` (the toy's marks), `gate` (every gate its own unit). *"Every gate lies in exactly one unit of each level"*; source gates are always verification units; the weights are one replay unit. Validated against compiled `RequestsG` and `ClusterG` at toy scale (`test_request_row_is_the_profile_of_requests_g`, `test_step_row_is_the_profile_of_cluster_g_on_identical_requests`, `test_every_partition_tiles_the_same_gates`, `test_finer_replay_levels_have_more_units_and_the_cell_level_one_word_each`, `test_verification_levels_refine`, `test_a_frontier_sized_table_is_written_in_milliseconds`).

`src/veritor/evaluation/frontier.py`: `FRONTIER_SHAPE` (§6.4), `DEFAULT_ETAS = (1/2, 1/100, 1/10^6)`, `DEFAULT_GRID = q ∈ {1/2, 1/8, 1/32, 1/128, 1/512, 1/2048, 1/8192} × s ∈ {1, 1/8, 1/64, 1/512}`, `DEFAULT_PARTITIONS` = the seven of §1.7, `FRONTIER_OPTIONS = BoundOptions(knapsack=False, max_buckets=1<<22)` (Laplace only), `Point` (partition, $q$, $s$, $\eta$, `bits`, `out_bits`, `overhead` = Cost.total/honest, `work` = expected_work/honest, seconds), `honest_cost`, `price`, `sweep`, `certify(points, eta, max_overhead, max_work)` (smallest `bits` within budgets; ties to cheaper prover then verifier), `save`/`load`, `calibration_table`, `partition_table`. Tests: `tests/veritor/evaluation/test_frontier.py`.

### 7.5 Frontier results §6.6 should quote (from `docs/frontier-report.md`, verified against the JSON)

- **Calibration** (overhead $\le1\times$, work $\le10\times$): $U_{\max}\approx 640$ bits at $\eta=1/2$, $\approx4.0$ kbit at $\eta=1/100$, $\approx11.7$ kbit at $\eta=10^{-6}$, all at `cell/gate`, $q=1/2$, $s=1/8$. If the verifier must also admit `request/row` servers: $42.6$ / $283$ / $848$ kbit ($66$–$72\times$ looser); `request/gate`: $8.3$ / $54.8$ / $164$ kbit.
- **Tight budgets** (overhead $\le1\%$, work $\le10\%$): $74.6$ kbit / $470$ kbit / $1.37$ Mbit at `cell/gate`, $q=1/2048$, $s=1$.
- **The `cell/gate` grid at $\eta=1/100$** (U as % of output / overhead / work): $q=1/2$: $s=1$: 0.0024%/1.33/59.3; $s=1/8$: 0.024%/0.896/7.42; $s=1/64$: 0.19%/0.841/0.927; $s=1/512$: 1.4%/0.834/0.116. $q=1/128, s=1$: 0.19%/0.0209/0.927. $q=1/2048, s=1$: 2.8%/0.00139/0.0579.
- **The frontier is one-dimensional in $qs$** and bends on verifier work: $U$ depends on $\theta$ almost only through $1-qs$ ($q=1/2,s=1/64$ and $q=1/128,s=1$ both give work $0.927$ and $31.3$ vs $31.6$ kbit); within fixed $qs$, higher $q$ lowers $U$ by about 1% and raises prover overhead proportionally. Verifier work buys, at $\eta=1/100$: 3.33 Mbit at 1% ($qs=1/16384$), 453 kbit at 10% ($1/2048$), 30.5 kbit at 100% ($1/128$), 3.93 kbit at 1000% ($1/16$), 402 bit at 10000% ($1/2$).
- **Monotonicities**: $U$ nondecreasing as $\eta$, $q$, $s$ decrease (0 violations of 392/1764/882 pairs); $\eta$: $1/2\to1/100$ multiplies $U$ by $\approx6.3$, $\to10^{-6}$ by $\approx18$ (tracking $\ln(1/\eta)$).
- **Verification-unit width matters**: `request/row` vs `request/gate` share replay units, but a `row` unit covering the one-hot has 524 kbit of output; a single erroneous row is covered by the whole request (8,192 bits), so `request/row` is $276$ kbit vs `request/gate` $53.5$ kbit at $\eta=1/100$ under the headline budgets, and `request/gate` keeps its capacity as $s$ falls.
- **Prover overhead is the same for every partition at a given policy** (0.896 at $q=1/2,s=1/8$ for all seven; `cell/gate` adds under 0.01% for its boundary), so *"marking at `cell/gate` costs the prover nothing measurable over `request/row`"* — the crux of the calibration recommendation and the reason "replay units should be coarse" needs the §1.7 qualification.

### 7.6 The pending reach-aware refinement of Bound

Status: **not implemented**; theorem drafted in `docs/overnight-report.md` §4 item 4 and §5.1: *"with errors confined to $S_1\cup\cdots\cup S_m$, $|\text{reachable outputs}|\le\prod_j 2^{\min(\mathrm{out\_bits}(S_j),\ \mathrm{reach\_bits}(S_j))}$ where $\mathrm{reach\_bits}(S)$ is the width of the circuit outputs downstream of $S$. Proof: fix all cuts but $c_j$; varying $c_j$ changes outputs only inside $\mathrm{reach}(S_j)$ and through at most $2^{\mathrm{out\_bits}(S_j)}$ cut values, so the image grows by at most $2^{\min(\cdot,\cdot)}$ per node; induct on $j$."* This is Corollary 5.7 applied with the cut $\mathrm{reach}(S)\subseteq\mathrm{out}(C)$ itself (the outputs downstream of $S$ form a downstream cut for $S$), so it needs no new theorem in §5 — only the observation that $\kappa(A)\le W(\text{outputs reachable from }A)$ and its use per node in the fold. Implementation plan (report §5.1): a reverse-dependency pass over the hierarchy at $O(|\text{description}|)$, per-kind `reach_bits` as the max over copies (sound), fold with $\min(\mathrm{out\_bits},\mathrm{reach\_bits})$; validate against the BFS oracle on the cluster fixtures. Measured effect on the toy cluster (56,600 gates, 41 replay units, 2,032 output bits, $\eta=10^{-2}$): $q=1/2,s=1$: cap 2,032 → 869; $q=1/2,s=1/2$: 2,032 → 1,152; $q=1/4,s=1/2$: still capped.

**Why it matters for coarse units, and why it would not rescue `step`/`layer` at 70B scale (estimate, not measured).** Reach makes a whole-unit corruption cost $\min(\mathrm{out\_bits}, \mathrm{reach\_bits})$ instead of $\mathrm{out\_bits}$. For a `prefill_step(32)` unit the reach is its 32 occupants' remaining tokens, at most $32\times512\times16 = 262$ kbit, instead of 320 Gbit; for a prefill `layer` unit (one token's layer), the reach is that request's remaining tokens, $\le 8{,}192$ bits, instead of 192 Mbit; for a `square_block`/`matvec` unit, likewise $\le 8$ kbit. At $q=1/2$ the adversary affords $\lfloor\Lambda/0.693\rfloor$ whole units: 1, 6, 19 at the three $\eta$; at $q=1/8$ ($-\ln(7/8)=0.134$): 5, 34, 103. So with reach, `step/row` at $\eta=10^{-2}$, $q=1/2$ would be bounded by roughly $6\times262$ kbit $\approx1.6$ Mbit (10% of output) plus position terms; `layer/row` and `matvec/row` by roughly $6\times8$ kbit $\approx50$ kbit plus position terms $\approx 6\log_2(8.7\times10^7)\approx160$ bits — comparable to `request/gate` (53.5 kbit), still $\gtrsim10\times$ worse than `cell/gate` (4.0 kbit), because a whole `cell` unit is only 16 bits. The report's caveat is the right wording: the intermediate partitions' 100% rows *"should be read as 'not yet bounded' rather than 'unsafe'"*. For the paper: reach-awareness is what makes medium-coarse replay units (the ones a server wants for per-unit overhead) chargeable at what the adversary can actually influence; it does not change the conclusion that the narrowest-interface units give the smallest bound.

**Discrepancy D-9 (reach in the paper).** The §5 draft's $\kappa(A)$ already allows any downstream cut, including $\mathrm{out}(C)\cap\mathrm{reach}(A)$; the code uses only index-node interfaces. Recommendation: in §6.3 say "Bound uses, for each node, the smaller of its declared interface and the width of the outputs downstream of it (both are downstream cuts in the sense of §5.4)", and mark the second as the refinement implemented (or pending — decide before the numbers are final).

---

## 8. Numbers the section quotes or should quote, each with its source

| number | value | source / status |
|---|---|---|
| $p=\eta=1\%$ ⇒ $B$ | $B=458$: largest integer with $0.99^B > 0.01$ ($0.99^{458} = 0.01004$, $0.99^{459} = 0.00994$) | §5 draft line 51; verified arithmetically. The outline's "$L = 459$" (`/tmp/notion-outline.md`, older text: "at most $L$ subcircuits", and the transcript's Aug 30 "$L=459$") is the smallest $L$ with $0.99^L\le0.01$, i.e. $B+1$: an off-by-one between "tolerated" ($\le B$) and "first rejected" ($L$). **Current: $B=458$, tolerated sets have $|E|\le B$.** |
| GPT-2 Small $n$ | $42{,}387{,}408{,}594$ non-input gates $= 42{,}361{,}101{,}422$ arithmetic $+ 26{,}307{,}172$ FP16 rounding gates | §5 draft line 311 (cut analysis, transcript). The brief's "42,361,101,422" is the arithmetic count only. |
| GPT-2 $\kappa_{\max}$ | 32 bits | §5 draft line 315 |
| GPT-2 regions $R$ | $31{,}987{,}771$; $6{,}114{,}482$ with 32-bit cuts; average $\approx1{,}300$ gates, largest $2.5\times10^8$ | §5 draft lines 313–319 |
| GPT-2 $W_{\mathrm{out}}$ | $\approx1{,}562$ bits (100 tokens × 15.6) | §5 draft line 311 |
| GPT-2 bounds | Cor 5.16: $458(32+17.5)\approx22{,}700$ bits; Cor 5.3: $458(32+27.9)\approx27{,}400$; output bound 1,562 | §5 draft lines 323–331 (one-stage verifier; §6's two-stage numbers not yet computed on GPT-2) |
| §5 matmul example | $d=1{,}024$, $k=100$: $N=102{,}400$ units, $W_{\max}=32{,}752$, value term $\approx1.5\times10^7$, position term $\approx4{,}200$, $W_{\mathrm{out}}=16{,}384$; cut bound $\approx11{,}600$; attack $\approx8{,}300$ (7,300 values-only) | §5 draft §5.3/§5.4 |
| "16% for hashing GPT-2" | unsourced | outline only; see §1.2. Under $h=1$ the 70B figure is 67%. |
| "1% budget" | target | outline (`/tmp/notion-outline.md:66`: "at a cost of [1%] of the verified computation"); §5 draft line 61 ("e.g. 1% of the cost"). The frontier's 1% column: $q\le1/128$. |
| zkVM overhead | "$10^{10}\times$ native" | outline lines 62, 81, 405, marked `[source]`; unsourced here. |
| Attestable | "50–100× overhead; fixed-point matmul only" | outline line 62, marked `[source]`; unsourced here. |
| $\eta$ grid | $1/2$, $1/100$, $10^{-6}$ | `frontier.py::DEFAULT_ETAS` |
| $(q,s)$ grid | $q\in\{2^{-1},2^{-3},2^{-5},2^{-7},2^{-9},2^{-11},2^{-13}\}$, $s\in\{1,2^{-3},2^{-6},2^{-9}\}$ | `frontier.py::DEFAULT_GRID`; 28 policies × 7 partitions × 3 etas = 588 points |
| 70B shape | $d=8192$, 80 layers, 64 heads, hidden $4d$, vocab 32,768, 16-bit; 2,048 requests × (512+512) tokens, batch 32; $n=2.744\times10^{17}$; honest cost $4.1168\times10^{17}$; output $16{,}777{,}216$ bits; $|W| = 6.496\times10^{10}$ | `frontier.py::FRONTIER_SHAPE`; measured |
| 70B headline (overhead ≤ 1, work ≤ 10) | 640 / 4,021 / 11,712 bits at $\eta = 1/2, 10^{-2}, 10^{-6}$; `cell/gate`, $q=1/2$, $s=1/8$; overhead 0.896, work 7.42 | `docs/frontier-report.md`; verified |
| 70B tight budgets (≤ 1%, ≤ 10%) | 74.6 kbit / 470 kbit / 1.37 Mbit; `cell/gate`, $q=1/2048$, $s=1$; overhead 0.00139, work 0.058 | verified |
| 70B `request/row` | 42,581 / 282,809 / 848,261 bits at $q=1/8$, $s=1$ (report writes 41.6/276/828 kbit in binary units and 42.6/283/848 in decimal; use one) | report |
| 70B `request/gate` | 8,254 / 54,750 / 164,206 bits at $q=1/2$, $s=1/8$ | report |
| 70B intermediate partitions | 16,777,216 bits (cap) at every grid point | report; 436/588 points at cap |
| overhead law | $q(1.667+s)$ | measured on all 588 points |
| verifier work law | $118.7\,qs$ (`gate` units), $79.1\,qs$ (`row` units) | measured |
| whole units affordable | $\lfloor\ln(1/\eta)/(-\ln(1-q))\rfloor$: 1/6/19 at $q=1/2$ | arithmetic |
| single errors affordable | $\ln(1/\eta)/(-\ln(1-qs))$: 10.7/71.4/214 at $qs=1/16$ | report; verified |
| bits per admitted error (`cell/gate`) | 59.6 / 56.4 / 54.7 vs $16+\log_2(3.47\times10^{13}) = 61.0$ | verified |
| toy cluster reach numbers | cut widths 144–672 bits vs reach 16–80; 2,032 → 869 at $q=1/2,s=1,\eta=10^{-2}$ | `docs/overnight-report.md` §5.1 |
| eight-gate example | $n=8$, $|\partial|=6$, $\mathrm{Int}=\{3\},\{7\}$, $\tau=(1,1,2,4,2,3,5,20)$, cost 12.5 at $q=s=1/2$ (honest 6), $W=37$, $U=16$ (cap), transcript 3,074 B | measured |
| sampler bias | $<2^{-190}$ TV per selection; Claim A adds $2^{-189}$ | `challenge.py` docstring; audit Claim A |
| minimax appendix operating point | $\alpha=10^6$, $\beta=0.01$, $A_{\mathrm{req}}=17{,}000$, $a=86$, $\Lambda=\ln100$; "5.0 GB … 120 GB" | transcript msg 210 (Aug 29); superseded model, see §7.2 |
| outline's "$\theta = (q,s,\eta)$" | — | superseded: $\eta$ is the verifier's (D-7) |

---

## 9. Discrepancies and open decisions

Each item: what the outline says; what the code does; what the author decided (with the transcript turn where found); what remains open.

**D-0 — The opening sentence.** *Outline:* "Section 5 assumed a verifier capable of checking that at most $L$ subcircuits of $C$ were evaluated incorrectly." *§5 draft:* no $L$; the verifier is a distribution over unit subsets with acceptance $\sigma(E)$, tolerated sets $\mathfrak E_\eta$, and for two stages $\sigma_\theta$. *Recommendation:* "Section 5 analysed a verifier that samples units and accepts a transcript with probability $\sigma_\theta(E)$ depending only on its set $E$ of incorrect units; this section builds that verifier and shows it costs about [x]% of evaluating $C$." *Open:* the percentage (§6.5).

**D-1 — I/O check "should use hiding commitments".** See §3.2. *Code:* I/O positions opened in the clear against the header. *Recommendation:* opening = binding the tree to the public header; ZK version proves equality instead. *Open:* none.

**D-2 — Well-typedness of unopened values.** *Outline §6.2:* "output values are well-typed". *Code:* checks type only for opened values (I/O and sampled positions). *Author:* "Values can be infinite … Nothing is preventing committing to some values that are too big" (msg 768). *Resolution:* Lemma 5.1 (well-typed transcripts suffice), plus: the outputs *are* opened and typed at the boundary; interior values are typed when opened. *Open:* the paper must say this once (checklist item 6).

**D-3 — Inputs (and weights) inside units; the root has no ports.** *Outline/§5 draft/§7 draft:* units partition the *non-input* gates; inputs outside; `Out` copy gates; §7 draft Lemma 7.2(i) "inputs lie in no unit". *Code:* every gate, `in` and `weight` included, in exactly one replay and one verification unit (overnight §2.1). *Author:* chose option one, Sep 1 (msg 783). *Recommendation (overnight §4.1):* adopt the implemented statement in §5, §6, §7; it "removes a case from every proof ('positions not in any unit') and the `Out` copy gates from the figures". *Open:* §5 draft line 29/245 and §7 draft Lemma 7.2(i) need the edit; the eight-gate figure already shows it (source gates are in $V_{1,1}$, $V_{2,1}$).

**D-4 — Declared outputs without `Out` copy gates.** *Outline (V1):* explicit `Output` copy gates. *Code/Scratch V2:* outputs are declared positions ("Gate 8 is also the sole public output position. There is no separate output-copy gate."). *Decided* (msg 616: "we don't need explicit out"). *Open:* none.

**D-5 — Definition of $\partial$.** See §2.4. *Recommendation:* $\partial := \mathrm{In}\cup\bigcup_r\mathrm{Out}(R_r)$; $\mathrm{out}(C)\subseteq\partial$ as a consequence; weights excluded. *Open:* whether to write $\mathrm{in}(C)$ or $\mathrm{In}$ (the §5 draft uses $\operatorname{in}(C)$, the code `In`).

**D-6 — Weights as a separate commitment $\kappa_W$.** *Outline:* weights are part of $x$ (Θ($|W|$) per request at the boundary) or of $G$ (a $G$ per model epoch). *§7 draft:* $\kappa_W$ is "the pre-committed subtree of the boundary tree". *Code:* $\kappa_W$ is its own root over the weight *vector* by rank, bound to the gate set, once per model, never a boundary position (overnight §2.5; `test_the_weight_domain_is_the_rank_space_and_the_boundary_excludes_the_weight_gates`). *Pros:* one root per model across all batch shapes; boundary independent of $|W|$ (`test_verifier_time_drops_when_weights_leave_the_boundary`); a sampled weight is checked by opening alone. *Cons:* the root is not self-describing (F5); the verifier must hold it out of band; a third owner class in every statement. *Recommendation (overnight §4.2):* "model weights are a third class of source gate, opened against a per-epoch commitment; they are never boundary positions." *Open:* the §7 draft's "subtree" wording must change; and where in §6 to introduce $\kappa_W$ (header, §6.2).

**D-7 — $\eta$ is the verifier's; $\theta=(q,s)$ is the client's proposal; denominators capped.** *Outline:* $\theta=(q,s,\eta)$. *Code:* `VerifierParameters.eta` in the header; `VerificationPolicy(q,s)` proposed; `max_probability_denominator_bits=64` (overnight §4.6; `test_the_proposal_is_theta_alone_and_the_header_binds_the_verifiers_eta`). *Recommendation:* the split; "a client-chosen $\eta$ is a soundness hole, not a parameter". *Open:* none.

**D-8 — Per-kind folds vs per-cut bounds; which headline numbers.** See §4.5. *Open:* whether §6's numbers come from the fold (index-node cuts, union-certified, two-stage) or from §5's cut analysis (arbitrary cuts, one-stage, per-strategy in the outline's original numbers). Overnight §4.3: "The outline's headline numbers were computed under the per-strategy definition and should be recomputed."

**D-9 — Reach-aware Bound.** See §7.6. *Open:* implement before quoting intermediate-partition numbers, or quote them as "not yet bounded".

**D-10 — Value semantics.** Toy: 16-bit modular ISA gates; §5 draft: general $V_i$, tokens $\log_2 50{,}257$. *Open* (overnight §4.10): fixed-point vs float, what "correct" means for a float gate. §6 should stay agnostic (§2.6).

**D-11 — ZK not implemented; "conditional on physical assumptions".** *Outline:* zero-knowledge listed as a security property. *Code:* transparent only; "No privacy" (audit §10). *Recommendation:* state ZK as a property of the construction with hiding commitments and a proof system (§5.2), not of the prototype; keep the outline's caveat about the wire.

**D-12 — Advice in/out of the header.** *Outline:* charges $|a|$ in the capacity statement but does not say where. *Code:* `Header.advice` (bound); `8|a|\le A` at admission (`POLICY_REJECTED`); `Capacity = Bound + 8|a|` (overnight §2.6, §4.5). *Recommendation:* "every accepted run has capacity $\le U_{\max}+A$". *Open:* two knobs for $A$ (`Compile(max_advice_bits=)` and `VerifierParameters.max_advice_bits`) — an implementation nit.

**D-13 — $U_{\max}$ and $W_{\max}$ admission; "Bound reports $\log_2$ of an integer".** *Outline:* no admission step. *Code:* `_admit` (§1.6); $U_{\max}$ required (F2); $U_{\max}=0$ satisfiable (overnight §2.2). *Recommendation:* one paragraph in §6.2 ("Before any commitment the verifier prices the run…") and one sentence on $U=0$ for a fully checked run.

**D-14 — Position term.** *§5:* $B\log_2(eN/B)$ / $B\log_2(eR/B)$ in the closed forms. *Code:* implicit in the union-certifying fold (the sum over distinct covers); at `cell/gate` it is $\approx45$ of the $\approx61$ bits per error. *Recommendation:* say in §6.3 that the fold's sum is the position term of §5 made exact for the two-stage $\sigma_\theta$.

**D-15 — Audit: "substituting gates for verification units under-bounds for coarse $V$".** Appendix argument consequence (a). *Code:* the fold covers the client's actual verification kinds (`unit_series(out_bits)` per verification kind), not gates. *Recommendation:* one sentence: "Bound is evaluated on the units the client declared."

**D-16 — "Replay units should be coarse".** *Outline §6.1.* *Frontier:* coarse-in-interface units make Bound vacuous; the cheapest informative partition is `cell/gate` (one dot product per replay unit). *Author (msg 1085):* calibrate $U_{\max}$ to the honest server rather than constrain units. *Recommendation:* rewrite as "replay units should have few declared outputs per gate; a dot product is the natural unit: thousands of gates, one word out" and move the cost trade-off to §6.6. *Open:* whether the paper's honest server is `cell/gate` (nearly free in this cost model, but $3.5\times10^{13}$ replay units and one interior root per sampled unit — the report's cost model charges nothing per unit beyond `Cost`; the assistant's msg 1082 warned "with $10^{12}$ units, the per-unit overhead (an interior root and a message per sampled unit) is itself the size of the boundary") or `request/gate`.

**D-17 — $c_0$ and $h$ uncalibrated; "1%" not met by the headline policy.** See §6.5. *Open:* the paper's operating point.

**D-18 — The verifier runs $G$ in the prototype.** Overnight §4.9. §7's business; §6 should say "given $(C,I)$ from §7".

**D-19 — Source-only units hold no errors.** Overnight §4.8; F4. *Recommendation:* the three-line tightening in §6.3.

**D-20 — Figure's boundary/interior shading dropped.** *Outline:* "[Figure: the eight-gate example, with the boundary shaded.]" *Author (msg 618, 627, 641):* replaced shading by input/output colouring and a shape grammar; the final variant has no boundary/interior distinction. *Open:* if §6.1 needs the boundary shown, use `RoleVariant=0` (legend "Replay-boundary value / Replay-interior value") or a second panel.

---

## 10. Figures and examples for §6

### 10.1 The eight-gate figure: conventions settled on Aug 31 (msgs 604–647)

Three horizontal bands — **Index** (root → replay units $1,2$ → verification units $1,2$ per replay unit → addresses $1..8$), **Circuit** (gate boxes containing only the operator; arrows from the gates they read; blue-filled rectangles for input gates labelled $\mathsf{In}(1),\mathsf{In}(1),\mathsf{In}(2),\mathsf{In}(3)$ — the constant is hard-coded in the gate, msg 622; a red-filled rectangle for the output gate $\times$, no "out" text; $x^2$ for the square), **Transcript** (values as warm pills: $1,1,2,4,2,3,5,20$; dashed arrows gate → value). Legend top-right, vertically stacked: blue rectangle "Input gate", red rectangle "Output gate". Index nodes are circles in the final variant (author, msg 630: "maybe we want the indices to be like circles or something to suggest that they're kind of like different objects"), gates rectangles, values pills (msg 641 "flipped"). The final rendering is `veritor-index-object-shapes-flipped.png` (Aug 31 17:10) = `RoleVariant=4` of the file below; the author's last requests (msg 647: trim 50% below the values, 10% below the gates, compress the index 20%) may or may not be in the saved `.tex` (it was re-saved Sep 1 18:16 with the default `RoleVariant=0`). Build: `pdflatex "\def\RoleVariant{4}\input{veritor-index-layers-v2.tex}"`.

`/Users/danielreuter/projects/latex-diagrams/veritor-index-layers-v2.tex`, verbatim:

~~~latex
\documentclass[10pt]{article}

\usepackage[
  paperwidth=24.2cm,
  paperheight=15.2cm,
  margin=0.25cm,
  noheadfoot
]{geometry}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,positioning}
\pagestyle{empty}
\setlength{\parindent}{0pt}

\definecolor{ReplayBlue}{RGB}{50,103,168}
\definecolor{VerifyPurple}{RGB}{113,78,154}
\definecolor{BoundaryBlue}{RGB}{218,235,249}
\definecolor{InteriorGold}{RGB}{251,232,186}
\definecolor{IndexBand}{RGB}{244,246,250}
\definecolor{CircuitBand}{RGB}{250,250,252}
\definecolor{TranscriptBand}{RGB}{250,248,243}
\definecolor{InputInk}{RGB}{52,105,151}
\definecolor{InputFill}{RGB}{226,238,248}
\definecolor{OutputInk}{RGB}{164,62,72}
\definecolor{OutputFill}{RGB}{249,228,231}
\definecolor{ValueInk}{RGB}{132,123,108}
\definecolor{ValueFill}{RGB}{243,239,231}

% 0 = replay status, 1 = fills, 2 = badges, 3 = outlines, 4 = shape grammar.
\providecommand{\RoleVariant}{0}

\begin{document}
\noindent\makebox[\textwidth][c]{%
\begin{tikzpicture}[
  font=\sffamily,
  band/.style={
    rounded corners=3pt,
    draw=black!18,
    line width=0.7pt
  },
  layerlabel/.style={
    anchor=west,
    font=\bfseries\Large\sffamily,
    text=black!82
  },
  rowlabel/.style={
    anchor=west,
    font=\bfseries\small\sffamily,
    text=black!58
  },
  hierarchy/.style={
    draw=black!38,
    line width=0.7pt,
    -{Stealth[length=1.7mm]},
    shorten >=2pt,
    shorten <=2pt
  },
  addressing/.style={
    draw=black!30,
    line width=0.7pt,
    -{Stealth[length=1.7mm]},
    shorten >=2pt,
    shorten <=2pt
  },
  indexnode/.style={
    rounded corners=4pt,
    draw=VerifyPurple!78,
    line width=0.9pt,
    fill=VerifyPurple!7,
    minimum height=9mm,
    align=center,
    text=black!82,
    font=\bfseries\sffamily
  },
  indexcircle/.style={
    circle,
    draw=VerifyPurple!78,
    line width=0.9pt,
    fill=VerifyPurple!7,
    minimum size=9.5mm,
    inner sep=0pt,
    align=center,
    text=black!82,
    font=\small\sffamily
  },
  root/.style={
    indexnode,
    minimum width=27mm,
    font=\normalfont\normalsize\sffamily
  },
  replay/.style={
    indexnode,
    minimum width=24mm
  },
  verification/.style={
    indexnode,
    minimum width=16mm
  },
  address/.style={
    indexnode,
    minimum width=9mm,
    inner sep=0pt,
    font=\bfseries\small\sffamily
  },
  gate/.style={
    rounded corners=3pt,
    draw=black!75,
    line width=0.9pt,
    fill=white,
    minimum width=18mm,
    minimum height=11mm,
    align=center,
    inner sep=3pt,
    font=\large\sffamily
  },
  valuebox/.style={
    rounded corners=4pt,
    line width=0.9pt,
    minimum width=17mm,
    minimum height=9mm,
    align=center,
    font=\small\sffamily
  },
  legendvalue/.style={
    rounded corners=1.3pt,
    line width=0.6pt,
    minimum width=5.7mm,
    minimum height=3mm,
    inner sep=0pt
  },
  boundary/.style={
    valuebox,
    draw=ReplayBlue,
    fill=BoundaryBlue
  },
  interior/.style={
    valuebox,
    draw=orange!75!black,
    fill=InteriorGold
  },
  plainvalue/.style={
    valuebox,
    draw=black!42,
    fill=white
  },
  warmvalue/.style={
    valuebox,
    rounded corners=9pt,
    draw=ValueInk,
    fill=ValueFill
  },
  wire/.style={
    draw=black!72,
    line width=0.9pt,
    -{Stealth[length=2.0mm]},
    shorten >=2pt,
    shorten <=2pt,
    preaction={draw=white, line width=2.8pt}
  },
  produces/.style={
    draw=black!38,
    line width=0.8pt,
    dashed,
    -{Stealth[length=1.8mm]},
    shorten >=2pt,
    shorten <=2pt
  },
  legendtext/.style={
    anchor=west,
    font=\small\sffamily,
    text=black!68
  },
  rolebadge/.style={
    rounded corners=1.2pt,
    minimum width=5.5mm,
    minimum height=2.6mm,
    inner sep=0.4pt,
    text=white,
    font=\bfseries\tiny\sffamily
  },
  legendgate/.style={
    rectangle,
    line width=0.8pt,
    minimum width=6.5mm,
    minimum height=3.8mm,
    inner sep=0pt
  }
]

\ifcase\RoleVariant
  \tikzset{
    inputrole/.style={},
    outputrole/.style={},
    transcriptboundary/.style={boundary},
    transcriptinterior/.style={interior}
  }
  \def\gateinputone{$\mathsf{In}(1)$}
  \def\gateinputtwo{$\mathsf{In}(1)$}
  \def\gateinputthree{$\mathsf{In}(2)$}
  \def\gateinputfour{$\mathsf{In}(3)$}
  \def\gateoutput{$\times$}
\or
  \tikzset{
    inputrole/.style={
      draw=InputInk,
      fill=InputFill,
      line width=1.1pt
    },
    outputrole/.style={
      draw=OutputInk,
      fill=OutputFill,
      line width=1.2pt
    },
    transcriptboundary/.style={plainvalue},
    transcriptinterior/.style={plainvalue}
  }
  \def\gateinputone{$\mathsf{In}(1)$}
  \def\gateinputtwo{$\mathsf{In}(1)$}
  \def\gateinputthree{$\mathsf{In}(2)$}
  \def\gateinputfour{$\mathsf{In}(3)$}
  \def\gateoutput{$\times$}
\or
  \tikzset{
    inputrole/.style={},
    outputrole/.style={},
    transcriptboundary/.style={plainvalue},
    transcriptinterior/.style={plainvalue}
  }
  \def\gateinputone{$1$}
  \def\gateinputtwo{$1$}
  \def\gateinputthree{$2$}
  \def\gateinputfour{$3$}
  \def\gateoutput{$\times$}
\or
  \tikzset{
    inputrole/.style={
      draw=InputInk,
      fill=white,
      line width=1.3pt
    },
    outputrole/.style={
      draw=OutputInk,
      fill=white,
      line width=1.5pt
    },
    transcriptboundary/.style={plainvalue},
    transcriptinterior/.style={plainvalue}
  }
  \def\gateinputone{$\mathsf{In}(1)$}
  \def\gateinputtwo{$\mathsf{In}(1)$}
  \def\gateinputthree{$\mathsf{In}(2)$}
  \def\gateinputfour{$\mathsf{In}(3)$}
  \def\gateoutput{$\times$\\[-2pt]{\scriptsize out}}
\or
  \tikzset{
    root/.style={
      indexcircle,
      minimum size=13mm,
      font=\normalfont\normalsize\sffamily
    },
    replay/.style={indexcircle},
    verification/.style={indexcircle},
    address/.style={indexcircle},
    gate/.append style={
      rounded corners=0pt
    },
    warmvalue/.append style={
      rounded corners=4.5mm
    },
    inputrole/.style={
      draw=InputInk,
      fill=InputFill,
      line width=1.1pt
    },
    outputrole/.style={
      draw=OutputInk,
      fill=OutputFill,
      line width=1.2pt
    },
    transcriptboundary/.style={warmvalue},
    transcriptinterior/.style={warmvalue}
  }
  \def\gateinputone{$\mathsf{In}(1)$}
  \def\gateinputtwo{$\mathsf{In}(1)$}
  \def\gateinputthree{$\mathsf{In}(2)$}
  \def\gateinputfour{$\mathsf{In}(3)$}
  \def\gateoutput{$\times$}
\fi

% One grid controls every layer.
\def\xleft{-12.05}
\def\xlabels{-11.65}
\def\xright{10.55}

\def\xone{-7.85}
\def\xtwo{-5.43}
\def\xthree{-3.01}
\def\xfour{-0.59}
\def\xfive{1.83}
\def\xsix{4.25}
\def\xseven{6.67}
\def\xeight{9.09}

\def\yroot{12.17}
\def\yreplay{10.77}
\def\yverify{9.37}
\def\yaddress{6.38}
\def\ygate{4.61}
\def\yvalue{0.39}
\def\ylegend{-1.01}

% Full-width layer containers, including their label gutter.
\filldraw[band, fill=IndexBand]
  (\xleft,8.25) rectangle (\xright,13.69);
\filldraw[band, fill=CircuitBand]
  (\xleft,2.55) rectangle (\xright,8.05);
\filldraw[band, fill=TranscriptBand]
  (\xleft,-0.81) rectangle (\xright,2.30);

\node[layerlabel] at (\xlabels,13.04) {Index};
\node[layerlabel] at (\xlabels,7.48) {Circuit};
\node[layerlabel] at (\xlabels,1.56) {Transcript};

\ifnum\RoleVariant=4
  \node[legendgate, draw=InputInk, fill=InputFill]
    (legendinput) at (7.30,12.95) {};
  \node[legendtext, right=1.5mm of legendinput] {Input gate};
  \node[legendgate, draw=OutputInk, fill=OutputFill]
    (legendoutput) at (7.30,12.31) {};
  \node[legendtext, right=1.5mm of legendoutput] {Output gate};
\fi

% Index layer
\node[root] (root) at (0.62,\yroot) {root};

\node[replay] (r1) at (-4.22,\yreplay) {$1$};
\node[replay] (r2) at (5.46,\yreplay) {$2$};

\node[verification] (v11) at (-6.64,\yverify) {$1$};
\node[verification] (v12) at (-1.80,\yverify) {$2$};
\node[verification] (v21) at (3.04,\yverify) {$1$};
\node[verification] (v22) at (7.88,\yverify) {$2$};

% Gate addresses belong to the circuit. Verification units index them.
\node[address] (a1) at (\xone,\yaddress) {$1$};
\node[address] (a2) at (\xtwo,\yaddress) {$2$};
\node[address] (a3) at (\xthree,\yaddress) {$3$};
\node[address] (a4) at (\xfour,\yaddress) {$4$};
\node[address] (a5) at (\xfive,\yaddress) {$5$};
\node[address] (a6) at (\xsix,\yaddress) {$6$};
\node[address] (a7) at (\xseven,\yaddress) {$7$};
\node[address] (a8) at (\xeight,\yaddress) {$8$};

\draw[hierarchy] (root) -- (r1);
\draw[hierarchy] (root) -- (r2);
\draw[hierarchy] (r1) -- (v11);
\draw[hierarchy] (r1) -- (v12);
\draw[hierarchy] (r2) -- (v21);
\draw[hierarchy] (r2) -- (v22);
\draw[hierarchy] (v11) -- (a1);
\draw[hierarchy] (v11) -- (a2);
\draw[hierarchy] (v12) -- (a3);
\draw[hierarchy] (v12) -- (a4);
\draw[hierarchy] (v21) -- (a5);
\draw[hierarchy] (v21) -- (a6);
\draw[hierarchy] (v22) -- (a7);
\draw[hierarchy] (v22) -- (a8);

\node[rowlabel] at (\xlabels,\yroot) {Root};
\node[rowlabel] at (\xlabels,\yreplay) {Replay unit};
\node[rowlabel] at (\xlabels,\yverify) {Verification unit};

% Circuit layer
\node[gate,inputrole] (g1) at (\xone,\ygate)
  {\gateinputone};
\node[gate,inputrole] (g2) at (\xtwo,\ygate)
  {\gateinputtwo};
\node[gate] (g3) at (\xthree,\ygate)
  {$+$};
\node[gate] (g4) at (\xfour,\ygate)
  {$x^2$};
\node[gate,inputrole] (g5) at (\xfive,\ygate)
  {\gateinputthree};
\node[gate,inputrole] (g6) at (\xsix,\ygate)
  {\gateinputfour};
\node[gate] (g7) at (\xseven,\ygate)
  {$+$};
\node[gate,outputrole] (g8) at (\xeight,\ygate)
  {\gateoutput};

\ifnum\RoleVariant=2
  \node[rolebadge, fill=InputInk, anchor=south west]
    at ($(g1.north west)+(0.6mm,-1.5mm)$) {IN};
  \node[rolebadge, fill=InputInk, anchor=south west]
    at ($(g2.north west)+(0.6mm,-1.5mm)$) {IN};
  \node[rolebadge, fill=InputInk, anchor=south west]
    at ($(g5.north west)+(0.6mm,-1.5mm)$) {IN};
  \node[rolebadge, fill=InputInk, anchor=south west]
    at ($(g6.north west)+(0.6mm,-1.5mm)$) {IN};
  \node[rolebadge, fill=OutputInk, anchor=south east]
    at ($(g8.north east)+(-0.6mm,-1.5mm)$) {OUT};
\fi

\draw[addressing] (a1) -- (g1);
\draw[addressing] (a2) -- (g2);
\draw[addressing] (a3) -- (g3);
\draw[addressing] (a4) -- (g4);
\draw[addressing] (a5) -- (g5);
\draw[addressing] (a6) -- (g6);
\draw[addressing] (a7) -- (g7);
\draw[addressing] (a8) -- (g8);

\node[rowlabel] at (\xlabels,\yaddress) {Address};
\node[rowlabel] at (\xlabels,\ygate) {Gate};

% Transcript layer
\node[transcriptboundary] (t1) at (\xone,\yvalue) {$1$};
\node[transcriptboundary] (t2) at (\xtwo,\yvalue) {$1$};
\node[transcriptinterior] (t3) at (\xthree,\yvalue) {$2$};
\node[transcriptboundary] (t4) at (\xfour,\yvalue) {$4$};
\node[transcriptboundary] (t5) at (\xfive,\yvalue) {$2$};
\node[transcriptboundary] (t6) at (\xsix,\yvalue) {$3$};
\node[transcriptinterior] (t7) at (\xseven,\yvalue) {$5$};
\node[transcriptboundary] (t8) at (\xeight,\yvalue) {$20$};

\node[rowlabel] at (\xlabels,\yvalue) {Value};

% Dashed arrows associate each gate with the transcript value it produces.
\draw[produces] (g1.south) -- (t1.north);
\draw[produces] (g2.south) -- (t2.north);
\draw[produces] (g3.south) -- (t3.north);
\draw[produces] (g4.south) -- (t4.north);
\draw[produces] (g5.south) -- (t5.north);
\draw[produces] (g6.south) -- (t6.north);
\draw[produces] (g7.south) -- (t7.north);
\draw[produces] (g8.south) -- (t8.north);

% Black wires are circuit dependencies and remain entirely inside C.
\coordinate (g3in1) at (g3.west);
\coordinate (g3in2) at ($(g3.west)+(0,-0.24)$);
\coordinate (g7in1) at (g7.west);
\coordinate (g7in2) at ($(g7.west)+(0,-0.24)$);
\coordinate (g8in1) at (g8.west);
\coordinate (g8in2) at ($(g8.west)+(0,-0.24)$);

\draw[wire] (g2.east) -- (g3in1);
\draw[wire] (g3.east) -- (g4.west);
\draw[wire] (g6.east) -- (g7in1);
\draw[wire] (g7.east) -- (g8in1);

\draw[wire] (g1.east)
  .. controls ($(g1.east)+(0.65,-1.10)$)
  and ($(g3in2)+(-0.75,-0.90)$)
  .. (g3in2);
\draw[wire] (g5.east)
  .. controls ($(g5.east)+(0.60,-1.10)$)
  and ($(g7in2)+(-0.75,-0.90)$)
  .. (g7in2);
\draw[wire] (g4.east)
  .. controls ($(g4.east)+(0.55,-1.85)$)
  and ($(g8in2)+(-0.90,-1.75)$)
  .. (g8in2);

% The legend has its own footer row and cannot overlap the diagram.
\ifnum\RoleVariant=0
  \node[legendvalue, draw=ReplayBlue, fill=BoundaryBlue]
    (lb) at (-4.05,\ylegend) {};
  \node[legendtext, right=1.5mm of lb] {Replay-boundary value};
  \node[legendvalue, draw=orange!75!black, fill=InteriorGold]
    (li) at (2.05,\ylegend) {};
  \node[legendtext, right=1.5mm of li] {Replay-interior value};
\fi

\end{tikzpicture}
}
\end{document}
~~~

Note for the writer: the figure's $\mathsf{In}(1),\mathsf{In}(1),\mathsf{In}(2),\mathsf{In}(3)$ label the *constants* (gates 1 and 2 both hold 1), consistent with $x=(1,1,2,3)$ and with §5's "input gates are constants". A reader may mistake $\mathsf{In}(k)$ for "the $k$-th input"; the Scratch page says "Gates 1, 2, 5, and 6 contain the public input values 1, 1, 2, and 3", which is the intended reading.

### 10.2 The two-stage sampling schematic (Sep 1, 18:32)

`/Users/danielreuter/projects/latex-diagrams/veritor-circuit-partition.tex` (rendered as `veritor-circuit-partition.png` and, with `\Flow=1`, `veritor-circuit-partition-flow.png`): a 20-layer × 8-row grid circuit with 5×2 replay-unit tiles each containing 2×2 verification-unit tiles of 2×2 gates; sampled replay units drawn in `ReplayBlue` with light fill, sampled verification units inside them in `VerifyPurple`, their gates purple; callouts "gate", "verification unit", "replay unit"; legend "sampled replay unit / sampled verification unit"; $x$ in, $y$ out. The flow variant adds a dashed "Server (untrusted)" perimeter, a "Verifier (trusted)" box on its edge and a "Client" box outside, with arrows "$y^*$, commitments, proofs" (server → verifier), "$x$, challenges" (verifier → server), "$y^*$ on accept" (verifier → client), "$x$" (client → verifier). Build lines are in the file header. This is the natural §6.1 figure for "two draws"; the eight-gate figure is the §6.2/6.4 worked example. Its verbatim source (177 lines) is in the file; key styles: `sru` (sampled replay unit), `svu` (sampled verification unit), `\sampledru{col}{row}`, `\sampledvu{col}{row}{vcol}{vrow}` select which tiles are drawn as sampled.

### 10.3 The two-stage walkthrough with coins

Use the Scratch page's coin phrasing (§3.1): one fresh fair coin per replay unit, announced as flipped; then one per verification unit inside selected units, *not flipped until every selected interior is stored*. For the paper the coins are $\mathrm{Bernoulli}(q)$ and $\mathrm{Bernoulli}(s)$; the implemented derivation (seed + phase digest → binomial count → Floyd subset) can be a footnote or an appendix.

### 10.4 The matmul example

$Y = XW$ with $X$ a batch of rows: one replay unit per row $X_i W$ (its declared outputs are the row of $Y$), one verification unit per output inner product (`compile_matmul`, `constructors/matmul.py`; layout `[activations][weights][row × rows]`, overnight §2.1). Here cut $=$ reach (a row unit's outputs are circuit outputs), so Bound is exact in the sense of §1.7. Numbers in §6.3. This example also shows why `optimize` picks $q=1$ at toy scale (every replay unit's interface is a large fraction of the output).

### 10.5 A frontier figure

Suggested: $U$ (log scale, bits or % of output) against $qs$ (log scale) for each partition at $\eta=1/100$, with verifier work $=118.7\,qs$ as a secondary axis; `cell/gate`, `request/gate`, `request/row` as three curves, the intermediate four as a flat line at the cap (labelled "not reach-aware"). A second panel or table: the calibration table (work × overhead → $U_{\max}$) for $\eta=1/100$. Data: `docs/data/frontier-70b.json` via `veritor.evaluation.frontier.load`; `calibration_table`, `partition_table` render the tables.

---

## 11. Glossary: paper name ↔ implementation name (verified against the code)

| paper | implementation | where |
|---|---|---|
| circuit $C$, size $n$, gate $C[i]$ | `Circuit`, `Index.n`, `circuit[a]` (0-based) | `core/circuit.py`, `core/index.py` |
| gate set $\Sigma$ | `GateSet`, `make_word_gate_set`, `make_isa_gate_set` | `core/gates.py` |
| input gates $\operatorname{in}(C)$, $x$ | `in` gates; `Circuit.inputs`, `Circuit.input_rank`; `Index.inputs()`; `Header.public_inputs` | `core/circuit.py`, `core/index.py`, `protocol/messages.py` |
| weight gates $W$, $\kappa_W$ | `weight` gates; `Circuit.weights`, `weight_rank`; `Index.weights()`; `Weights(count, root)`; `weight_domain`, `commit_weights` | `protocol/domains.py`, `protocol/messages.py` |
| outputs $\operatorname{out}(C)$, $y^*$ | `Circuit.outputs`; `Header.claimed_outputs` | |
| index $I$; kinds; copies | `Index`; definitions (`Definition`); `IndexNode` frames; `KindSummary`, `KindTable` | `core/index.py`, `core/description.py` |
| replay units $R_r$ | `Index.replay_units` (`Units`), `Units.unit(r)`, `Units.owner(a)`; `role="replay"` | `core/index.py` |
| verification units $V_{r,j}$ | `Index.verification_units(r)` (block), `Index.verification_unit(k)` (global), `role="verification"` | |
| $\mathrm{In}(V)$, $\mathrm{Out}(R_r)$ | `Circuit.In(node)`; `Definition.out_runs/out_count/out_bits`; `KindSummary.input_count/out_count/out_bits` | |
| boundary $\partial$ | `Index.boundary()` (`_Boundary`); owner `BOUNDARY_OWNER = -1` | `core/index.py:324`, `protocol/domains.py` |
| interior $\mathrm{Int}(r)$ | `Index.interior(r)`; owner `r` | `core/index.py:334` |
| boundary assignment $\beta$ | prover's `boundary_values`; committed as `BoundaryMessage.commitment` | `protocol/session.py` |
| interior assignment $\alpha_r$ | `replay_unit(compiled, r, boundary_values)`; `InteriorMessage.commitments[j]` | |
| policy $\theta=(q,s)$ | `VerificationPolicy(q, s)` (`Fraction`s) | `core/policy.py` |
| threshold $\eta$ | `VerifierParameters.eta`; `Header.eta` | `protocol/parameters.py` |
| $U_{\max}$, $W_{\max}$, $A$ | `VerifierParameters.max_capacity`, `.max_work`, `.max_advice_bits` | |
| $U = \mathrm{Bound}(C,I,\theta)$ | `bound(target, policy, eta, options).bits`; `research.Bound` | `analysis/bound.py` |
| $\mathrm{Capacity} = U + 8|a|$ | `research.Capacity` | `research.py` |
| $\sigma_\theta(E)$, $f(l)$, $c(l)$, $\Lambda$ | `survival`, `survival_factor`, `unit_cost`, `budget`; `admissible` | `analysis/probability.py` |
| $\mathfrak E_\eta$ | admissible $(l_r)$: `sum_r c(l_r) < Lambda` | |
| $\mathrm{Cost}(C,I,\theta)$, $h$, $c_0$ | `cost(...)` → `ExpectedCost(boundary, replay, proof, weights)`; `CostParameters(hash_cost, proof_overhead)` | `analysis/cost.py` |
| verifier work $W$ | `expected_work(target, policy, io_count)` | `protocol/parameters.py` |
| Optimize | `optimize(target, eta, grid, max_bits=/max_cost=)`; `PolicyGrid`; `Optimization` | `analysis/optimize.py` |
| header / expectation | `Header` (both sides); `Expectation`, `make_expectation` (verifier's, with seeds) | `protocol/messages.py`, `protocol/session.py` |
| $J$ (selected replay units) | `ReplayChallenge.selected`; `derive_replay_selection` | `protocol/challenge.py` |
| $T$ (selected verification units) | `SampleChallenge.selected`; `derive_sample_selection` | |
| commitment / opening | `Commitment(root, count)`, `Opening(position, value, path)`; `CommitmentDomain`, `MerkleTree`, `verify_opening` | `protocol/merkle.py` |
| phases | `boundary_phase`, `replay_phase`, `interior_phase`, `sample_phase` | `protocol/phases.py` |
| transcript | `Transcript`; `encode_transcript`/`decode_transcript`; `verify_transcript` | `protocol/wire.py`, `protocol/verify.py` |
| verdict | `VerificationReport(code, sampled_replay_units, sampled_verification_units)`; `VerificationCode` | `protocol/messages.py` |
| Compile$(G,x,a)$, $h_C$, $h_G$ | `research.Compile`; `Compiled.digest`; `constructor_digest` | `research.py`, `compile/` |
| honest server / partitions | `serving_table(shape, replay_level, verification_level)`; `REPLAY_LEVELS`, `VERIFICATION_LEVELS` | `evaluation/serving.py` |
| frontier | `FRONTIER_SHAPE`, `DEFAULT_GRID`, `DEFAULT_ETAS`, `sweep`, `certify`, `calibration_table`, `partition_table`, `load` | `evaluation/frontier.py` |
| downstream cut / $\kappa$ / reach | `out_bits` per node (cut); `reach_bits` (planned); `reference.cut_bits`, `cover_bits` (small-circuit oracles) | `analysis/reference.py` |
| well-typed | `Circuit.encode/decode` canonical; `INVALID_VALUE` | |
| source gates | `Gate.is_input`, `Gate.is_weight`; `source_inputs`, `source_weights` | `core/gates.py`, `core/description.py` |

Terms the code uses that the paper should *not* adopt as names: "kind"/"copy" (say "definition" or "the same subcircuit repeated" if needed), "frame", "owner" (say "committed under"), "admissible"/"budget" (§5 stripped them), "expectation" (say "the verifier's parameters and seeds"), "compiled"/"artifact".

---

## 12. Checklist for the writing agent (in order; every item explicit, correct, compact)

1. State what §6 builds: a verifier for a committed $(C,I)$ whose acceptance on a fixed transcript is $\sigma_\theta(E)$ of §5.4, and that the adaptive protocol does not exceed it.
2. Name the two obstacles (commitment cost; ephemeral values) with a sourced number or none.
3. Define replay units, verification units, tiling and refinement; **every gate (inputs and weights included) in exactly one unit of each kind**; outputs are declared positions.
4. Define $\mathrm{Out}(R_r)$ (declared; contains every cross-unit read and every circuit output inside; no source gate), $\partial = \mathrm{In}\cup\bigcup_r\mathrm{Out}(R_r)$, $\mathrm{Int}(r)$, $W$ under $\kappa_W$; note $\mathrm{out}(C)\subseteq\partial$.
5. Give the protocol in five messages with the two draws, the ownership rule (weights → $\kappa_W$, inputs and declared outputs → $\partial$, else the owning replay unit's interior), and the rule that the verifier derives $\partial$, $J$, $T$ and every position itself.
6. Say what is checked and what is not: I/O against the header; every opened value well-typed; every non-source gate of a sampled unit against the owners' committed values; nothing about unopened values (Lemma 5.1).
7. Admission before any commitment: $8|a|\le A$, denominators, unit counts, $W\le W_{\max}$, $U=\mathrm{Bound}(C,I,\theta)\le U_{\max}$; $\eta$ is the verifier's and in the header.
8. Completeness (probability 1; sampler bias $<2^{-190}$ is on selection, not acceptance).
9. The $\ell^\star$ lemma with its exact hypothesis (every outside read of a replay unit is in $\partial$ or $W$) and conclusion ($\Pr[\text{accept}]\le\sigma_\theta(E^\star)$, output reachable from $E^\star$).
10. $\sigma_\theta(E)=\prod_r(1-q+q(1-s)^{|E\cap\mathcal P_r|})$; $\mathfrak E_\eta$ as the knapsack $\sum_r c(l_r)<\Lambda$; the saturation $c(\infty)=-\ln(1-q)$ and its consequence (whole replay units are cheap; their interface enters the bound).
11. Bound: instantiates Theorem 5.9 with index-node cuts; certifies the union (sum over distinct covers), not $\max_E$; source-only units contribute nothing; reports $\log_2$ of an integer, so a fully checked run has $U=0$; every approximation rounds toward admitting more.
12. "Nothing else constrains either partition" with the seven-step argument, and the qualification from the frontier: coarse-in-interface replay units are priced out by $U\le U_{\max}$, so the honest server marks narrow-interface units.
13. Zero-knowledge as a property of the construction (hiding commitments + proofs at the same points), what stays public, the wire caveat; the prototype is transparent.
14. The construction: indexed Merkle trees over $[n]$ with domain separation; one boundary tree, one interior tree per selected unit, $\kappa_W$ per model; openings by address; batched proofs per run.
15. Cost model with the exact formula, units, $h$ and $c_0$ named, honest base, and the verifier's $W$; the 70B laws overhead $\approx q(1.67+s)$ and work $\approx119\,qs$, with the caveats (unit counts; proving factor not included).
16. Parameter choice: server chooses $I$ and $\theta$; verifier enforces $\eta$, $U_{\max}$, $W_{\max}$, $A$; $U_{\max}$ calibrated to the honest-server frontier; quote the calibration numbers and the `cell/gate` result; minimax statement, grid search in practice.
17. Say what is pending: reach-aware Bound; SP1 calibration of $c_{\mathrm{proof}}$ and $h$; the 16% figure; value semantics.
18. Vocabulary: well-typed; output widths, not "words"; $G$ once per epoch; design decisions; no kernel/description object/instance; $R_r$, $V_{r,j}$, $J$, $T$, $\beta$, $\alpha_r$, $\partial$, $\mathrm{Int}(r)$, $\theta$, $\eta$, $U$, $U_{\max}$, $W_{\max}$.
19. Cross-check with §5: use $\mathcal P_r$, $\sigma_\theta$, $\mathfrak E_\eta$, $\kappa$, Theorem 5.9 / Proposition 5.15 by number; do not reintroduce "acceptance function", "admissible", "budget", "certify".
20. Cross-check with §7: cite Lemma 7.1 (determinism), 7.2(ii) (∂ contains every cross-read), Proposition 7.3 (O(depth) queries), Theorem 7.5 (capacity $\le U_{\max}+A$); flag that §7's "inputs outside units" and "$\kappa_W$ as a subtree" must be updated to match.


---

## Appendix A. Verbatim claims from `docs/security-argument.md` that §6.3 can lean on

The audit is organized as numbered claims with "Argument", "Attack tests", "Gaps", "Verdict". §6.3 should not cite the audit, but the writing agent can lift the *statements* below as the precise properties the construction has, and the "Gaps" as the caveats the section must not paper over. Quoted with the audit's own line ranges into the code; they refer to the Sep 1 state of `session.py` and have shifted. Today's positions: `_Layout.owner` 187, `_Layout.position` 197, `_Layout.required` 203–220, `_admit` 454–498, `receive_boundary` 556, `_check_unit` 651.

### A.1 Sampling distribution (audit §3)

*"**Claim.** The joint law of `J` is that of independent `Bernoulli(q)` coins, one per replay unit, within total variation `2^-190`; given `J`, the law of `T` is independent `Bernoulli(s)` coins over the verification units of the units in `J`. Consequently a transcript whose error set is `E` is accepted with probability `sigma(E) = prod_r (1 - q + q (1 - s)^{l_r})` up to that bias."*

*"**Argument.** `bernoulli_subset` (213-249) draws `K ~ Binomial(N, p)` by inverting the CDF against a 256-bit HMAC output (`_binomial_count`, evaluated in 512-bit fixed point; the docstring bounds the bias below `2^-190` for `N < 2^64`) and then a uniform `K`-subset with Floyd's algorithm from rejection-sampled uniforms (`_floyd_subset`, 190-210; `uniform_below`, 87-118). A `Binomial(N, p)` mixture of uniform `K`-subsets is exactly the law of `N` independent `p`-coins, so the only deviation is the count inversion's bias. The two stages use distinct tags and phase digests, so their PRF streams are independent. `derive_sample_selection` ranks the candidates block by block over the selected replay units (266-292), so `T` is `Bernoulli(s)` over exactly the units of `J`."*

Attack tests named: `test_acceptance_rate_matches_survival_of_the_error_set` (*"a prover corrupts a fixed `E` of two verification units, in distinct replay units (`sigma = 9/16`) and in the same one (`sigma = 5/8`), at `q = s = 1/2`; the acceptance rate over 2000 fresh seed pairs is within 4 standard deviations of the exact `survival`"*), `test_selection_law_alone_matches_survival_over_many_seeds`, `test_survival_is_the_product_of_per_replay_unit_factors` (all in `tests/veritor/protocol/test_sampling.py`).

Gaps to carry into the paper as caveats: *"The `2^-190` bias means `P[accept] <= sigma(E) + 2^-189`, not `sigma(E)`; `eta` is compared exactly."* — so a fully rigorous statement of capacity soundness is at threshold $\eta - 2^{-189}$, or the paper says "up to a $2^{-189}$ additive term". *"The statistical tests are 4-sigma checks, not proofs; the law itself rests on the PRF assumption."*

The two numbers to use in §6.2 when describing sublinear sampling: $9/16$ and $5/8$ are the exact survival values of two-unit error sets at $q=s=1/2$ ($\,(1-\tfrac12+\tfrac12\cdot\tfrac12)^2 = (3/4)^2 = 9/16$ across two replay units; $1-\tfrac12+\tfrac12\cdot\tfrac14 = 5/8$ inside one), and they illustrate the §5/§6 point that concentrating errors in one replay unit survives better ($5/8 > 9/16$).

### A.2 Local checks (audit §4)

*"**Claim.** For every sampled verification unit the verifier opens exactly the addresses the unit reads or writes, each under its owner, decodes each value canonically, compares every `in` gate with the header's public input, accepts a `weight` gate only as kappa_W's leaf, and checks every other gate's relation against the opened argument values. At the boundary, before any sampling, the public inputs and the claimed outputs are opened and compared with the header, exhaustively."*

*"`_Layout.required(unit)` (182-199) is the sorted set of the unit's gates plus `In(unit)`, each with its owner; `_check_unit` (607-658) demands the evidence open exactly those positions in that order (`COVERAGE_MISMATCH`), opens each under its owner through `_open` (492-510, `INVALID_OPENING`), decodes it with the circuit's codec (`INVALID_VALUE` for anything that does not round-trip, including a value outside the gate's width), then walks the unit's gates: an `in` gate's payload must equal the header's public input of its rank (`PUBLIC_IO_MISMATCH`); a `weight` gate has nothing to check beyond its opening under kappa_W; any other gate goes through `circuit.check_gate(args, out)` (`RELATION_REJECTED`). Arguments are the owners' committed values -- the prover never states an argument, only opens positions."*

*"`receive_boundary` (512-552) demands the boundary openings cover exactly the public I/O addresses in boundary order, opens each and compares inputs and claimed outputs with the header (`PUBLIC_IO_MISMATCH`). Because every `in` gate is a boundary position and every output is a boundary position (`_Layout.__init__`, 161-168), this check is exhaustive and precedes the reveal of the `q` seed."*

*"A gate whose semantics raise is `TRUSTED_SERVICE_FAILURE`: the verifier fails closed."*

Gaps: *"Only sampled units are checked; that is the design, and section 5 bounds what survives."* *"Canonicity of values is the gate set's codec (`encode`/`decode`). A gate set whose `decode` accepts two encodings of one value would let a prover commit the same value twice; the built-in codec is strict."* *"A weight gate is accepted as *whatever kappa_W says*; a wrong kappa_W is a provenance problem (section 10), not a protocol one."*

The last gap is the one the paper must state: the protocol binds the run to $\kappa_W$; it does not certify that $\kappa_W$ is the weights of any particular model. That is a statement about the header, and belongs with the sentence "the verifier holds $\kappa_W$" in §6.2.

### A.3 Admission (audit §6)

*"**Claim.** `eta` is the verifier's and bound into the header; the denominators of `theta` and `eta` are capped before any sampling; `U_max` and `W_max` are checked from the per-kind counts alone, before any commitment; a transcript recorded under another `eta` is `EXPECTATION_MISMATCH`."*

*"`_admit` (426-461) runs in `VerifierSession.__init__` before the phase is set to `boundary`: it enforces `max_probability_denominator_bits` on both `theta` and `eta`, `max_units` on both unit counts, `max_positions_per_unit` per kind, then `expected_work(compiled, theta, |IO|) > W_max` is `WORK_BUDGET_EXCEEDED` and, when `U_max` is set, `bound(...).bits > U_max` is `POLICY_REJECTED`. `expected_work` (90-121) is a closed form over `index.kinds()`. `verify_transcript` (`verify.py`, 38-49) compares the transcript's `eta` and then the whole header with the verifier's own."*

The demonstrated gap, now closed (F2): *"`VerifierParameters()` defaults to `U_max = None`, which waives the capacity check, and `policy()` accepts any `theta`, including `(0, 0)`. A verifier built with the defaults accepts any claimed output under `theta = (0, 0)`: nothing is sampled and the only checks compare the client's boundary with the client's header. `Bound` reports this honestly (`bits == out_bits`); with no default `U_max`, a verifier has to decide."* Test: `test_default_parameters_waive_u_max_and_admit_a_policy_that_checks_nothing`.

Remaining gap: *"`W_max` bounds *expected* work; the realized work of one run can exceed it (the selection is random). The hard limits (`max_openings`, `max_proof_bytes`, ...) bound the worst case."* §6.5 should say "expected" every time it says "work".

### A.4 Tiling and refinement (audit §8)

*"**Claim.** Every gate is in exactly one replay unit and exactly one verification unit; verification units refine replay units; a unit reads outside itself only through kappa_W, the boundary, or its own replay unit's interior, so `Out(R_r)` is the cut between replay units."*

*"`validate_marks` (598-660) checks once per definition: a replay-marked definition contains no replay mark and is tiled by verification marks; a verification-marked definition contains no mark of either role; the root is tiled by replay marks; a marked definition has gates. Tiling means every step above a mark is a call into a tiled definition, so no gate is left uncovered, no two marks overlap, and a verification mark cannot straddle two replay units (it would have to contain a replay mark). `_Layout.required` (182-199) independently refuses a unit that reads an address owned by another replay unit that is not a boundary position (`INVALID_COMPILED_RESULT`); the compiler makes this unconstructible, because a replay unit's declared outputs are exactly what may be read from outside and those are boundary positions."*

Tests: `tests/veritor/protocol/test_tiling.py::test_every_gate_is_in_exactly_one_replay_unit_and_one_verification_unit`, `::test_verification_units_refine_replay_units`, `::test_cross_unit_reads_go_only_through_declared_outputs`, `::test_marks_leaving_a_gate_uncovered_are_a_compile_error`, `::test_nested_or_straddling_marks_are_a_compile_error`, `::test_layout_rejects_a_circuit_that_reads_across_the_cut` (*"the compiler rule makes the case unconstructible through the API, so the test forges a `Compiled` whose circuit reads across the cut"*).

This is the §7 fact §6 consumes as "∂ contains every cross-replay-unit read" (§7 draft Lemma 7.2(ii)); the verifier's defensive check in `_Layout.required` means §6 can say the verifier *also* refuses a circuit that violates it, so the soundness argument does not depend on trusting the compiler at that point.

## Appendix B. The overnight decisions (2.1–2.7 of `docs/overnight-report.md`), verbatim where they bind §6

**2.1 Every gate is in a unit (commits b433dbd, 353b85c, 97a2c21).** *"The root definition has no ports. The circuit's inputs and weights are zero-arity source gates (`in`, `weight`) in the public gate set, placed by descriptions like any other gate, so they sit inside replay and verification units and the tiling check covers them. `x_k` is the value at the `k`-th `in` gate in address order (`Index.inputs()` is an `O(depth)` rank/unrank domain, as is `Index.weights()`). Ports survive only inside the hierarchy, between a definition and its children, which is what keeps kind sharing."* Consequences listed there and all relevant to §6:

- *"A unit of source gates has `Out` width 0, so `Bound` gives it zero capacity without a new rule (a tightening is possible: a source-only kind can never hold an error, see §3)."* (F4.)
- *"Outputs stay declared (an interface), not gates. The paper's `Out_1(6)` copy gate has no counterpart; the uniform statement is 'every gate belongs to one replay unit and one verification unit; the outputs are declared positions among them'."*
- *"`∂ = In ∪ ⋃_r Out(R_r)` with `Out` excluding pinned gates, and `Int(r) = R_r \ Out(R_r) \ pinned(R_r)`: source gates are never interior. κ_W covers the weight gates; the `exclude=range` carve-out of an input prefix is gone."* (This is exactly the code's `Index.boundary()`; the circuit outputs are not a separate term because every output resolves through the declared interface of the replay unit that owns it — `index.py:329-330` — and `_Layout.__init__` (`session.py:174-181`) *checks* rather than adds: it raises `ProtocolError("circuit output at address ... is not a boundary position (a weight gate cannot be a claimed output)")` if an output is not already in $\partial$. See §2.4 and D-5 of this dump.)
- *"A sampled `in` gate is compared with `x[rank]`; a sampled `weight` gate is accepted on its κ_W opening alone; the boundary phase still opens every `in` gate (Θ(|x|), inherent)."*
- *"Matmul layout: `[activations][weights][row × rows]`. The activation row is not inside its `row` unit ... 1024³: 4010-byte description, 9 definitions, 18 ms compile, 2.1·10⁹ gates, per-kind table 0.08 ms."*
- *"Every source gate is by default its own verification unit (`Tracer.inputs(n)` emits one-gate cells), so the priced work `q s Σ_v (proof(V_v) + c_0)` gains `(|x| + |W|)(1 + c_0)`. A client that wants a smaller `W_max` price defines wider input units; nothing forces the one-gate cells."* (This is why the eight-gate example's $V_{1,1}$ and $V_{2,1}$ are *pairs* of inputs: `EightG` groups them explicitly.)

**2.2 `Bound` reports the log of an integer count (a3d8f0f).** *"`|Y_η|` is an integer, so `|Y_η| ≤ 2^b` implies `|Y_η| ≤ ⌊2^b⌋`. The fold's upward rounding made a fully checked run bound to `~1e-14` bits and `U_max = 0` unsatisfiable; it is now exactly `0.0`. Still an upper bound (the power is rounded up before the floor)."*

**2.3 Interface resolution is bounded by what it produces (fd478dd).** *"Declared interfaces are runs `(start, count, stride, width)`. ... `CompilationLimits.max_output_runs` (256 per definition) and `max_output_runs_total` (16 384) cap the pieces a definition may resolve to."* Relevant to §6 only as the reason the boundary is a union of $O(1)$-described runs, so `_Layout` costs $O(\#\text{kinds}\cdot\#\text{runs})$, not $O(|\partial|)$.

**2.4 `In` of a kind is its declared port count (e3850a5).** *"Admission pricing (`W_max`) uses the declared input interface of a verification kind, not the set of inputs actually read; declared ≥ read, so the priced openings only grow, and it is `O(1)` per kind. `Definition.reads` is only ever evaluated for sampled units, after the work budget has admitted the run. A client can no longer make admission itself cost `Θ(|x|)`."* This is the `in_k` in `expected_work` (§6.2 of this dump).

**2.5 κ_W commits the weight vector, per model, not per description (676f379).** *"Before tonight the weight domain was `I.weights()`, whose identity digest depends on the compiled description, so κ_W would have changed with every request of a continual-batching cluster (each request compiles a different circuit from the same model). Now the domain is the rank space `0 .. |W|-1` bound to a fixed tag and the gate set digest: position `k` is the `k`-th `weight` gate in address order of whichever circuit is verified. `commit_weights(gate_set, values)` needs no circuit; a sampled weight gate at address `a` is opened at rank `weight_rank(a)`. Headline test: one root is accepted for two batch shapes of the same model."* *"Security note: κ_W binds the gate set and the vector, not a model name. Which root a request must run under is the verifier's choice in the `Expectation`; the verifier must have obtained κ_W before the epoch's first request (a deployment rule, not enforced in code)."*

**2.6 Advice is structural, and the verifier runs `G` (9641d18, 77a89a6).** *"Decision: structural only, as the paper's `Compile(G, x, a)` has it. ... (ii) charging `8·|a|` bits per request is the whole accounting, with no per-unit bookkeeping and no interaction with `Bound`."* *"The header binds `G.digest` and `a` (protocol v6); admission rejects advice over the verifier's `max_advice_bits`; `Capacity(compilation, θ, η) = Bound + 8·|a|`. With `U_max` enforced at admission and `max_advice_bits = A`, every accepted request has capacity ≤ `U_max + A`."* *"Two knobs exist for one bound (`Compile(max_advice_bits=)` pre-checks before running `G`; `VerifierParameters.max_advice_bits` is the authoritative admission check that also runs on the transcript path). Keep them equal; a mismatch only causes rejections, never acceptances."*

**2.7 `U_max` has no default (afa92d4).** *"`max_capacity` is now a required keyword (`None` still waives, but has to be written) and `make_expectation` / `make_verification_expectation` require the verifier's parameters. Test fixtures state `VerifierParameters(max_capacity=None)` explicitly; the README example states `max_capacity=0` for its fully checked run."*

## Appendix C. The protocol in code, as the README states it (usable as the paper's "reference usage")

`README.md:236-262` is the shortest complete honest run and shows which object is whose; the comments there are the author's own mapping to paper names and can be reused in §6.2/§6.4:

~~~python
from veritor import (
    VerificationPolicy,
    VerifierParameters,
    Verify,
    compile_matmul,
    make_verification_expectation,
    run_protocol,
)
from veritor.protocol import commit_weights, encode_transcript

compilation = compile_matmul(request)              # Compile(MatmulG, workload, b"")
compiled = compilation.compiled
values = dict(enumerate(compiled.circuit.evaluate(compilation.inputs, request.weight_values)))
outputs = tuple(values[a] for a in compiled.circuit.outputs)
weights, weight_tree = commit_weights(gate_set, request.weight_values)  # kappa_W, once per model

expectation = make_verification_expectation(       # the verifier's side of one run
    compilation,                                   # (C, I), G's digest, x by rank and a
    VerificationPolicy(q=1, s=1),                  # the client's proposal, theta
    outputs,
    parameters=VerifierParameters(eta=0, max_capacity=0),  # eta, U_max, A, W_max are the verifier's
    weights=weights,
)
run = run_protocol(compiled, expectation, values, weight_tree=weight_tree)
assert run.report.accepted
assert Verify(encode_transcript(run.transcript), expectation, compiled) == run.report
~~~

README prose right after (`README.md:264-269`): *"`ProverSession` and `VerifierSession` are the two state machines behind `run_protocol`; `Verify` (`verify_transcript`) re-derives both challenges from the verifier's seeds and checks a recorded transcript purely. The header binds `(C, I)`, `G`'s digest, the advice `a`, `θ` and the verifier's `η`, so a transcript recorded under another `G`, `a` or `η` is rejected; advice longer than `max_advice_bits` is rejected at admission, before any commitment."*

And the Bound paragraph (`README.md:273-281`), which is the tightest one-paragraph description of what §6.3 hands back to §5: *"`Bound(C, I, θ)` certifies `U`, a bound in bits on the outputs an adversary can reach with acceptance probability above `η`. It is a fold over the kinds of `I`: every error set is assigned a cover by index nodes, the reachable outputs of a cover are at most `2^{Σ out_bits}` (the downstream cut), and the distinct covers of admissible error sets are summed. Admissibility is a knapsack over replay units against the budget `ln(1/η)`, solved on a cost grid that only ever admits more; a grid-free Laplace bound is taken alongside and the smaller reported. No copy is ever enumerated: a `10^8`-gate transformer index bounds in milliseconds."*

The analysis snippet (`README.md:283-294`):

~~~python
from fractions import Fraction
from veritor import Bound, Capacity, Cost, CostParameters, Optimize, PolicyGrid, VerificationPolicy

theta = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
eta = Fraction(1, 100)                                  # the verifier's threshold
print(Bound(compiled, theta, eta).bits)                 # U in bits
print(Capacity(compilation, theta, eta))                # U + 8|a|: what the paper charges
print(Cost(compiled, theta, CostParameters(hash_cost=1, proof_overhead=0)).total)

best = Optimize(compiled, eta, PolicyGrid.uniform(8), max_bits=20)
~~~

`research.py` exposes `Compile`, `Verify`, `Bound`, `Capacity`, `Cost`, `Optimize`, `make_verification_expectation`, `build_executable_conformance_transcript` (`src/veritor/research.py:56-245`). `Capacity(compilation, θ, η) = Bound(compiled, θ, η).bits + 8·len(advice)` is the number §8 composes; §6 delivers `Bound` and the admission that makes `U ≤ U_max` hold for every accepted run.

## Appendix D. Index of author decisions found in the transcripts, by date (for the writing agent's citations)

All from `agent-transcripts/eb746331-.../eb746331-....jsonl` unless noted; "msg N" is the 0-based index of the user/assistant message in that file. Times are Pacific as printed in the transcript.

| Date / time | msg | Decision or statement | Where it lands in §6 |
|---|---|---|---|
| Aug 22 | — | `docs/sp1-benchmark-plan.md` written: SP1 as the zkVM candidate; operators `f32_dot`, `fixed_dot`, `softmax_row`, ...; no results yet | §6.4 proof system, §6.5 $c_{\mathrm{proof}}$ unsourced |
| Aug 29 | 210 | Minimax appendix with $\alpha=10^6,\ \beta=0.01,\ A_{\mathrm{req}}=17{,}000,\ a=86,\ \Lambda=\ln 100$; "5.0 GB … 120 GB" | §6.6 framing (superseded numerically) |
| Aug 29 | — | Outline appendix "Why do we not put any constraints on the partitions?" (seven-step argument) | §6.1, §6.3 |
| Aug 31 12:08 PM | 496 | *"all the verifier knows it has to build everything out of its knowledge of the address space one through n"* | §6.2, §6.4 |
| Aug 31 12:08–1:33 PM | 496–535 | Walkthrough in primitive operations; *"the verifier does everything and the client just gives it its information"* | §6.2 transparent protocol as specification |
| Aug 31 3:59–5:09 PM | 604–647 | Eight-gate figure conventions: $r_1, r_2, v_{1,1}$; operator-only gate boxes; blue inputs / red output; pills; legend top-right; $x^2$ label; boundary shading dropped for variant 4 | §6 figure |
| Sep 1 3:14 PM | 748 | Paper architecture: *"1. Bound 2. Verify 3. Compile"*, then composition | §0 |
| Sep 1 3:28 PM | 759 | Asymmetry principle; *"at no point do we check anything ... no notion of an authorized computation"* | §6.1, §6.3 |
| Sep 1 3:33 PM | 761 | *"I hate 'words.' I prefer talking about output widths of gates and/or value widths"* | vocabulary |
| Sep 1 3:45 PM | 770 | *"we call it well-typed, not well-formed"* | vocabulary |
| Sep 1 3:50 PM | 772 | *"it just doesn't really talk about scale ... we're immediately going to need to talk about ML circuits"* | §6.5/§6.6 numbers |
| Sep 1 afternoon | 783 | "option one": inputs and weights as source gates inside units; root has no ports; $\kappa_W$ separate; header binds it | §6.2 objects (D-3, D-5) |
| Sep 1 18:32 | — | Two-stage sampling schematic (`veritor-circuit-partition.tex`) | §6 figure |
| Sep 1 10:27 PM | — | Notion outline export with the 16% sentence | §6.1 (unsourced number) |
| Sep 1 night | — | Overnight decisions 2.1–2.8 landed (Appendix B) | throughout |
| Sep 2 7:40 AM | 1081 | *"I would have assumed for the KV state width we would break it up ..."* | §6.6 objection |
| Sep 2 7:46 AM | 1083 | *"I was thinking that the replay unit would be like an entire inference request ..."* | §6.1/§6.6 |
| Sep 2 7:51 AM | 1085 | *"if we don't get to choose what the replay units are, then this doesn't help us much ... show what an honest inference server does so that we can calibrate the max bits"* | §6.6 resolution: calibrate $U_{\max}$ |
| Sep 2 morning | — | Frontier sweep (50 min), `docs/frontier-report.md`, `docs/data/frontier-70b.json` | §6.5, §6.6 numbers |

What was searched for and *not* found in any transcript: the derivation of "16%"; a numerical value for $c_{\mathrm{proof}}$ or $c_0$; an explicit author statement choosing the string "Partial verification" as the §6 title (it is only in the outline); any author statement on whether $\eta$ should be $1/2$, $1/100$ or $10^{-6}$ in the headline (all three are computed; the outline's §5 numbers use $p=\eta=1\%$).

## Appendix E. Two small consistency checks the writing agent can run

Both take a few seconds with `.venv/bin/python`; they reproduce the two numbers most likely to be misquoted.

1. **$B$ at $p=\eta=1\%$.**

~~~python
from fractions import Fraction
p = eta = Fraction(1, 100)
B = 0
while (1 - p) ** (B + 1) > eta:
    B += 1
print(B)   # 458
~~~

2. **The two-stage survival of the eight-gate error sets** (matches the audit's $9/16$ and $5/8$ and the §4.3 formula):

~~~python
from fractions import Fraction
from veritor.analysis.probability import survival
from veritor.core.policy import VerificationPolicy
theta = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
print(survival(theta, [1, 1]))   # two replay units, one bad verification unit each: 9/16
print(survival(theta, [2]))      # one replay unit with two bad verification units: 5/8
~~~

(Signature verified today: `survival(policy: VerificationPolicy, errors_per_replay_unit: Iterable[int]) -> Fraction`, `probability.py:50`; `survival_factor(policy, errors) = 1 - q + q(1-s)^errors`, `probability.py:42-47`.)
