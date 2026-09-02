# 7. Secure circuit compilation

The verifier of this section neither receives a gate list from the client nor re-runs the client's compiler; it parses a small description in a grammar of three constructs and derives every structural fact about the circuit itself. §6 took the circuit and its units $R_1, \dots, R_m$ and $V_{r,k}$ as given, required nothing of them beyond tiling and refinement, and left to §8 who chooses them. This section discharges that assumption without trusting anything the client sends and in time sublinear in the circuit: it produces $(C, I)$ from inputs that are committed, pinned or charged, answers §6's structural queries in logarithmic time, and gives §8 the lemma that fixing $G$ and $x$ leaves at most $2^{A}$ reachable circuits. What it certifies is deliberately narrow. The verifier never asks whether $C$ is the circuit the client should have run; nothing in this paper depends on such a notion. It certifies that the circuit is *fixed* and every further choice the client makes is *charged*, so that $y^*$ carries at most a certified number of bits beyond what a fixed function of $(G, x)$ determines. No uncharged degrees of freedom: that is the whole security statement, and §7.3 makes it precise.

## 7.1. Basic idea

Suppose the client claims that the following program returned $y^*=20$ on the input $x=(1,1,2,3)$:

~~~python
def workload(a, b, c, d):
    total = a + b
    square = 0
    for i in range(total):
        if i % 2 == 0:
            square += total
        else:
            square += a
            square += b
    return square * (c + d)
~~~

Suppose further that the verifier can afford to rerun only one simple operation from this computation. Any procedure for doing so must satisfy the following constraints:

1. **The verifier's choice must be unpredictable**. If the client could predict which operation would be chosen, then it could always perform that correctly. This can be achieved via random sampling.
2. **The verifier must have access to the full list of operations**. Random sampling is only possible given a fixed population.
3. **The population must be fixed by things the client cannot vary per run without paying**. A circuit alone is not enough: a client free to pick a fresh circuit for each output fixes the population only after the fact. The circuit must therefore be a deterministic function $C = \mathrm{Compile}(G, x, a)$ of a program $G$ committed once per epoch, the public input $x$, and advice $a$ charged bit for bit, $|a| \le A$. Every degree of freedom is committed, pinned, or charged.
4. **The verifier must be able to answer questions about the population without reading it**. The circuits we care about have on the order of $10^{11}$ gates, of which the verifier will look at a few thousand. It needs the gate at address $i$, the interface of a unit, and the unit structure, each in time logarithmic in the circuit, from a description it can hold.

The operation cannot be fixed in advance. If the client knew which operation would be checked, it could perform that operation honestly and cheat everywhere else. In this partial-verification approach, the verifier therefore wants to choose an operation unpredictably after the client has fixed its claimed execution.

Before it can sample an operation, however, the verifier needs a fixed population of candidate operations. The source code does not directly provide one. In particular, it is unclear: *how many* operations will be performed, because the number of loop iterations depends on the input; *which* operations will be performed, because a different branch may execute in each iteration; and what should count as one operation, because a source-level statement may expand into several lower-level operations. Random sampling is well defined only after these choices have been resolved.

For the input $x=(1,1,2,3)$, the program sets $\mathsf{total}=2$. Its first loop iteration adds $2$ to $\mathsf{square}$; its second adds $1$ and then $1$. It therefore returns $4\cdot(2+3)=20$. Equivalently, the claimed arithmetic relation is $(1+1)^2(2+3)=20$.

A compiler can replace the program's loops, branches, and bookkeeping with a fixed directed graph of elementary operations. Such a graph is called a *circuit*. Every gate in a circuit has a stable address, so the verifier can now choose a random gate—or a random group of gates—from a well-defined population. In the figure below, the square gate summarizes the loop's arithmetic effect.

[Figure 7.1: the circuit for $(1+1)^2(2+3)$ with its index; already exists]

The circuit resolves constraints 1 and 2. Constraint 3 is where the exfiltration game parts from ordinary verifiable computation. The client is the developer: it wrote the program and the compiler and runs both, and if it may also choose the circuit at run time, the choice is a channel. With many compilers to pick from, it selects for each output it wants to send the one whose honest evaluation produces it, and the verifier of §6 finds nothing wrong because nothing is. The remedy is to price the choice, not forbid it: $\mathrm{Compile}$ is deterministic, $G$ is committed once per epoch, $x$ is public, and $a$—the only argument the client varies per run—is charged bit for bit. The client chose $G$ too, perhaps maliciously; but it chose once, before any input, and §8 fixes $G$ inside the game. Given $G$ and $x$, Lemma 7.1 leaves at most $2^{A}$ circuits, and $A$ enters the bound as an additive term.

Constraint 4 is quantitative. §6 asks the circuit for $C[i]$ at sampled addresses, for the units that own them and those units' interfaces, and for ranks in $\partial$ to locate values in the boundary tree; with $10^{11}$ gates a flat gate list runs to terabytes. This is the asymmetry that governs the design: structure—$G$, its kinds, the per-kind table of §7.4—is small and can be analyzed globally; values—the transcript—are enormous and can only be checked locally. Every global fact the protocol needs is placed on the structure side and derived from $G$; the value side is touched only where a sample lands.

Two observations complete the idea. First, the verifier does not check what the circuit computes. The loop of `workload` became a single square gate: the compiler replaced the program by a different program with the same output on this input, and the protocol is indifferent. What is certified is a property of the pair $(C, y^*)$, not a relation between $C$ and `workload`, and this puts constructors, tracers and JAX or PyTorch front ends outside the trusted base: a wrong lowering is just a different fixed circuit. Second, the circuit is the program's own structure. The tracer records the program's functions and loops as definitions and repetitions, and that call structure is precisely the compressed description the verifier needs; the client authors nothing extra. Data-dependent control flow that cannot be re-expressed as a fixed graph is padded to an envelope whose cost §7.5 quantifies.

§7.2 states the protocol, §7.3 its security properties, §7.4 the construction, §7.5 what it can express and what the envelope costs, and §7.6 its limitations.

## 7.2. Protocol

We give the protocol in its transparent form, with the discipline of §6.2: the transparent verifier is the specification, and the zero-knowledge construction of §8 replaces each value the verifier reads with a commitment and each check with a proof at the same point in the interaction. Here that is simple, because $\mathrm{Compile}$ is deterministic and $O(|G|)$: the client can prove it ran $\mathrm{Compile}$ correctly on a committed $G$.

The objects are as follows. The *gate set* $\Sigma$ is public; each gate has a name, an arity, an output width in bits, a replay cost, a proof cost and a relation, and a value is well-typed when it is a bitstring of the output width of the gate producing it. A *description* $G$ is a byte string in the grammar of §7.4: definitions built from gates, calls and repetitions of earlier definitions, some marked `replay` or `verification`. $\mathrm{Compile}(G, x, a)$ parses and validates $G$ against $\Sigma$, binds its integer parameters from the shape of $x$ and from $a$, and returns $(C, I)$ or rejects. The client proposes $\theta = (q, s)$; the threshold $\eta$, with $\Lambda = \ln(1/\eta)$, and the limits $A$, $U_{\max}$, $W_{\max}$ are the verifier's, fixed before any run.

**Once per model.**

1. The client sends the description $G$, with its marks, and the root $\kappa_W$ of the commitment to the model's weights.
2. The verifier parses $G$ and validates every definition—canonical encoding, gate names and arities against $\Sigma$, in-range relative references, dependency order, and the resource limits on bytes, definitions, steps, addresses, cost and nesting depth—and rejects otherwise.
3. The verifier checks the marks—the replay-marked definitions tile the gates (every gate lies in exactly one replay unit, so replay units do not nest), the verification-marked definitions tile every replay unit the same way, and every verification unit's proof cost is within the cap §6's completeness requires—and rejects otherwise.
4. The verifier summarizes every definition once (the per-kind table of §7.4) and publishes
$$h_C \;=\; H\bigl(\textsf{compiled},\, H(G),\, H(\Sigma)\bigr),$$
which binds the description, its marks and the gate set; it is just the hash of $G$, qualified by the identity of $\Sigma$, and by determinism it fixes $(C, I)$ for every $x$ and $a$.

**Once per run.**

1. The client sends the input $x$, the claimed output $y^*$, the advice $a$, and the proposal $\theta = (q, s)$ as exact rationals.
2. The verifier checks
$$|a| \;\stackrel{?}{\le}\; A, \qquad \mathrm{den}(q),\, \mathrm{den}(s) \;\stackrel{?}{\le}\; 2^{D},$$
where $D$ is a public bound on denominator size, so that the sampling of §6 is exact and its cost bounded, and rejects otherwise.
3. The verifier binds the parameters of $G$ from $\mathrm{shape}(x)$ and $a$, checks that the root's input count equals $|x|$, and forms $(C, I) = \mathrm{Compile}(G, x, a)$. Nothing is unrolled; every later query descends through the prefix sums of §7.4.
4. The verifier computes, from the per-kind table alone,
$$U \;=\; \mathrm{Bound}(C, I, \theta) \;\stackrel{?}{\le}\; U_{\max}, \qquad W(C, I, \theta) \;\stackrel{?}{\le}\; W_{\max},$$
and proceeds only if both hold. $W$ is its own expected work in §6, one operation per opened leaf, per hash on an authentication path, per gate relation checked and per commitment received:
$$W(C, I, \theta) \;=\; \bigl(|\mathrm{io}| + qs\,P\bigr)\bigl(1 + \lceil \log_2 n \rceil\bigr) + qs\,S + q\,m + 1,$$
where $P = \sum_{K} c_K\,(\mathrm{Size}_K + |\mathrm{In}_K|)$ and $S = \sum_{K} c_K\,\mathrm{Size}_K$ run over the verification kinds $K$ with $c_K$ copies, $|\mathrm{io}|$ counts the positions opened in §6's first step, and $n$ is the number of addresses.
5. The verifier hands $(C, I, \theta)$ to the protocol of §6, with $\kappa_W$ as the pre-committed subtree of the boundary tree.

The verifier never runs client code, never reads a gate list, and believes no quantity the client asserts about its own circuit; every number above is derived from $G$.

## 7.3. Security properties

The one idea is that determinism turns a choice into a count, whatever the client's compiler, constructor or intentions.

**Lemma 7.1 (Binding).** Fix $\Sigma$ and $A$, and let $a$ range over strings of exactly $A$ bits, shorter advice being padded. For every description $G$ and input $x$, the set $\{\mathrm{Compile}(G, x, a) : a \in \{0,1\}^{A}\}$ has at most $2^{A}$ elements, and a party holding $h_C$, $x$ and $a$ computes the same $(C, I)$ as the verifier.

*Proof.* $\mathrm{Compile}$ is a deterministic function of its three arguments, so with $G$ and $x$ fixed the map $a \mapsto (C, I)$ has a domain of size $2^{A}$. The second claim is the same determinism read by the other party, since $h_C$ determines $G$ up to collisions in $H$. $\square$

*Remarks.* The lemma does not say the $2^{A}$ circuits are similar or correct or that any computes the client's program; only that there are at most $2^{A}$, which is what §8 uses: the union over $a$ of §6's reachable output sets is at most $2^{A}$ times the largest.

**Lemma 7.2 (Soundness of derived structure).** For every $(C, I)$ returned by $\mathrm{Compile}$: (i) the replay units partition the gates of $C$, the verification units partition each replay unit, and inputs lie in no unit; (ii) for every replay unit $R_r$, every address inside $R_r$ read by a gate outside $R_r$, and every circuit output inside $R_r$, lies in $\mathrm{Out}(R_r)$, so $\partial = \mathrm{in}(C) \cup \bigcup_r \mathrm{Out}(R_r)$ contains every value read across a replay-unit boundary; (iii) $\mathrm{Size}$, $\mathrm{Cost}$, $|\mathrm{In}|$, $|\mathrm{Out}|$ and the width of $\mathrm{Out}$ in bits are identical for all copies of a kind and computed exactly from the definition.

*Proof.* (i) is the mark check of §7.2: above the replay cut every step is a call or repetition into a definition that is itself tiled, so no gate lies outside a replay unit; a replay-marked definition contains no other; and the same two conditions hold for verification marks inside each replay-marked definition. (ii) is a property of the grammar. A step's argument names a coordinate relative to the enclosing definition—one of its inputs or an output slot of an earlier step—so a parent can reach a gate inside a child copy only through the child's declared output ranges, and by induction on the resolution, every reference from outside a copy into it passes through the copy's declared outputs; for $R_r$ these are $\mathrm{Out}(R_r)$, the declared interface resolved to the gates the copy owns. The circuit outputs are the root's declared outputs and the root lies above the replay cut, so each is an input or resolves through the replay unit that owns it. (iii) holds because every copy of a definition is a translation of one layout, so each quantity is computed once per definition. $\square$

*Remarks.* $\mathrm{Out}(R_r)$ is a superset of what is actually read across the boundary; a declared output nobody reads still sits in $\partial$, where the client pays for its commitment, so the surplus costs the client, never soundness. The superset is also what makes $\partial$ copy-independent, and copy-independence is what makes rank and unrank in $\partial$ prefix sums of per-kind interface widths. The lemma is exactly what §6's $\ell_r^{\star}$ lemma needs—that the stitched partial transcript $\widehat\tau$ agrees with $\tau|_{\partial}$ on every value a selected unit reads from outside—since (ii) gives every such value an address in $\partial$. Since inputs lie in no unit, $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r)$.

**No condition on the marks.** The verifier imposes nothing on the marks beyond tiling and refinement. A client may mark a single dot product or a whole layer; every such choice is legal and every one is priced, by $\mathrm{Bound}$ through $\kappa$ and $\sigma_\theta$ and by $W$ through the sizes and interface counts of the verification kinds. §5.4 and §6.6 analyze how granularity moves $U$ and the cost. The point here is that the verifier need not judge whether the marks are sensible, only whether they tile: a client who chooses badly pays in cost or in a larger $U$, never in soundness.

**Proposition 7.3 (Efficiency).** Let $|G|$ be the byte size of the description, $d$ the nesting depth of its definitions and $n$ the number of addresses. Parsing, validation and the mark check cost $O(|G|)$; deriving the per-kind table costs $O(|G|)$ plus time linear in the declared interfaces it lists, of which the root's are $x$ and $y$, which the verifier reads in any case. Thereafter every query of §6 costs $O(d)$ descent steps, each a bisection or a division, times the arity or interface size when the answer is a set. The verifier's state is $G$ and the per-kind table; nothing costs $\Omega(n)$.

*Proof sketch.* Every check is per definition, every summary a sum over a definition's steps weighted by repetition counts, and every query a descent of §7.4: subtract a base, bisect a prefix sum, divide by a child size, recurse. $\square$

**Proposition 7.4 (Completeness).** Every description satisfying the grammar and limits of §7.4 compiles, and its marks pass step 3 of §7.2 whenever the replay-marked definitions tile the gates and the verification-marked definitions tile each of them. An honest client that evaluates $C$ on $x$ and follows §6 is accepted with the completeness probability of §6.

**Theorem 7.5 (No uncharged degrees of freedom).** Fix $\Sigma$, $A$, $\eta$, $U_{\max}$ and $W_{\max}$. Let $G$ be a description accepted once per model and $x$ an input. For every run the verifier of §7.2 admits, with advice $a$ and proposal $\theta$: $(C, I) = \mathrm{Compile}(G, x, a)$ is one of at most $2^{A}$ pairs determined by $(G, x)$; $(C, I)$ satisfies the assumptions of §6; and $U = \mathrm{Bound}(C, I, \theta) \le U_{\max}$. Consequently the set of outputs $y^*$ accepted with probability above $\eta$, over all admissible runs with the same $(G, x)$, has size at most $2^{U_{\max} + A}$: the output carries at most $U + A$ bits beyond what a fixed function of $(G, x)$ determines.

*Proof.* The three clauses are Lemma 7.1, Lemma 7.2 and step 4 of §7.2. §6 bounds the outputs reachable within one circuit by $2^{U}$; the union over at most $2^{A}$ circuits gives the count. $\square$

*Remarks.* The theorem is silent about which function of $(G, x)$ the output approximates, and the silence is the point: a client that ran a different program than it advertised, or lowered it wrongly, has produced some fixed circuit, and the theorem bounds what it can add to that circuit's honest output. A careful reader will also ask whether $\theta$, which the client proposes, is a degree of freedom. It is, and it is priced rather than charged: a proposal that lets too much through fails $U \le U_{\max}$, one that costs the verifier too much fails $W \le W_{\max}$, and between those the client is free, harmlessly, because $U$ is certified for whichever $\theta$ it picked.

## 7.4. Construction

**The grammar.** A description is a finite sequence of definitions in dependency order, each identified by the hash of its canonical body, so that identical definitions are one definition, stored once and named by its digest wherever it is used. Specifically, a *definition* declares an input count, an optional mark in $\{\mathtt{replay}, \mathtt{verification}\}$, a straight-line list of steps and a list of output ranges; a *step* is one of three constructs:

~~~text
gate    g   range*           one gate g of Σ applied to the values in the ranges
call    D   range*           one copy of an earlier definition D
repeat  c D jrange*          c copies of D; copy j reads each range shifted by j·jstride

range   = (space, start, count, stride)            space ∈ {input, local}
jrange  = (space, start, count, stride, jstride)
~~~

Here `input` coordinates name the enclosing definition's own inputs and `local` coordinates name output slots of earlier steps in the same definition: a gate step has one slot, a call as many as $D$ has outputs, a repetition $c$ times as many, copy-major. Coordinates are relative to the enclosing definition, so passing a whole vector, or the $j$-th column of a matrix to the $j$-th copy, is a single range of constant size whatever the vector's length, and a range with $\mathtt{jstride} = 0$ broadcasts an operand to every copy without duplicating it. Validation is per definition, hence $O(|G|)$: every range must lie within the definition's inputs or the slots defined so far, for every copy of a repetition, every called digest must be defined earlier, and the checks of §7.2 are the rest.

The matmul constructor exercises the whole grammar. For $Y = XW$ with $X$ a $64 \times 64$ activation block and $W$ a $64 \times 64$ matrix over $\mathbb{Z}_{2^{8}}$ it produces five definitions:

~~~text
mul                    inputs 2       gate mul(in[0], in[1])                           out loc[0]
add                    inputs 2       gate add(in[0], in[1])                           out loc[0]
dot   [verification]   inputs 128     repeat 64 mul(in[j], in[64+j])                   64 products
                                      repeat 32 add(loc[2j], loc[2j+1])                then 16, 8, 4, 2, 1
                                                                                       out loc[126]
row   [replay]         inputs 4160    repeat 64 dot(in[0..64), in[64+j :: 64])         x broadcast, column j of W
                                                                                       out loc[0..64)
batch                  inputs 8192    repeat 64 row(in[4096+64j ..+64), in[0..4096))   row j of X, W broadcast
                                                                                       out loc[0..4096)
~~~

One row $x_i W$ is a replay unit and each output dot product a verification unit. Rows, columns and the products inside a dot are repetitions and the dot's sum is a tree of repetitions, so the description is $O(\log k)$ in the contraction length $k$ and $O(1)$ in the numbers of rows and columns: $2{,}245$ bytes for $520{,}192$ gates at $k = 64$ and $2{,}834$ bytes for $2.1 \times 10^{9}$ gates at $k = 1024$, compiled in $0.23$ and $0.28\,\mathrm{ms}$ of single-threaded Python (Table 7.1).

**Weights are inputs.** In the display $W$ enters `batch` as an input and is broadcast to every row; the constructor could have written the weights as immediates, and must not. Kinds are definition digests, so two rows—or two layers of a transformer—are copies of one kind only when their bodies are identical and their weights flow in as arguments; weights as immediates would make every layer a distinct kind, with $|G| = \Theta(|W|)$, a description as large as the model. With weights as inputs, $|G|$ is proportional to the program: the weights are committed once per model under $\kappa_W$, a pre-committed subtree of §6's boundary tree, opened where a sample lands and never carried by a run.

**The compiler.** $\mathrm{Compile}$ parses $G$, validates and summarizes each definition once, checks the marks, and returns $(C, I)$; from then on everything is descent. Inputs occupy addresses $[0, |x|)$, the root's gates follow in step order, and every copy of a definition owns a contiguous interval of $\mathrm{Size}(D)$ addresses, so copy $j$ of the step at position $t$ in a copy with base $b$ begins at $b + \mathrm{pre}_t + j \cdot \mathrm{Size}(D_t)$, with $\mathrm{pre}_t$ the prefix sum of the earlier steps' sizes. To answer $C[i]$: subtract the base, bisect the prefix sums to find the step, divide by the child's size to find the copy, recurse; at a gate step, resolve each argument to an absolute address by walking back up the copies, a `local` reference to the slot of an earlier step and an `input` reference to the range the parent passed at this copy's index. Inputs are boundary positions outside the units; units contain gates.

The unit structure and the boundary are read off prefix sums exactly as addresses are. Per definition the compiler stores the number of replay units, verification units and boundary positions inside it, and per step the prefix sums of each; the $k$-th replay unit is found by bisecting the replay prefix sums and dividing by the child's count, the owner of an address by descending to the first replay-marked copy on its path, and the rank of a boundary address by descending to its replay unit and bisecting in its declared output offsets. The per-kind table lists, for every definition reachable from the root, its copy count, size, replay and proof cost, input and output counts, the width of $\mathrm{Out}$ in bits, the multiset of child kinds it calls, and the number and kinds of verification units inside it; copy counts flow from parents to children in one pass over the definition graph. This is profiling by kind: $\mathrm{Bound}$, $W$ and the cost model of §6 are folds over this table weighted by copy counts, and none enumerates a copy.

**Parameters.** The display is specialized to one shape. The design, not yet in the prototype, admits integer parameters in repetition counts, strides and definition selectors, bound before parsing from $\mathrm{shape}(x)$, which is public, and from $a$, which is charged—elaboration in the hardware sense, except that nothing is unrolled. Structural dependence on the *values* of $x$ is excluded by design; whatever needs it, such as mixture-of-experts routing, goes through advice and is paid for. Parameters leave §7.3 untouched: $\mathrm{Compile}$ remains a deterministic function of $(G, x, a)$.

**The tracer.** The client writes constructors in ordinary Python against an untrusted tracer, which runs the constructor once over symbolic wires, records each function as a definition and each call as a step, hash-conses definitions by the digest of their bodies, and serializes in dependency order. Python loops unroll into steps; an explicit `repeat` is what keeps $G$ small, and it is the construct a JAX or PyTorch front end would map `vmap` and `scan` to. Nothing about the tracer is trusted: the compiler re-validates every byte, and a tracer bug is a different fixed circuit.

**Prior art.** The description is a hierarchically defined graph in the sense of Lengauer and Wanke, whose bottom-up method solves problems on such graphs in time polynomial in the description rather than its expansion [TODO: cite Lengauer–Wanke]; the per-kind table and $\mathrm{Bound}$ are that method applied to interfaces, costs and covers. Hash-consing is Ershov's and Goto's [TODO: cite]. Relative coordinates are position-independent addressing: a copy is a translation of its definition. `repeat` with a per-copy stride is the `generate` loop of hardware description languages and the single affine loop of the polyhedral model. Tracing a program once over symbolic values to record its call structure is staging, as in JAX [TODO: cite]. Nothing off the shelf sits at the intersection of hierarchy-preserving, scalar-exact and small-decoder: tensor-level intermediate representations preserve hierarchy and are small, but each operator needs a trusted lowering to the scalar relations the verifier checks, while circuit formats for secure computation and proof systems are scalar-exact but flat, with size linear in the gates. Our trusted base is $\Sigma$ with a validator for three constructs, in the LCF architecture: a small trusted checker, arbitrary untrusted producers.

## 7.5. Expressiveness and the cost of the envelope

**What is expressible.** Over a complete gate set every function on fixed-width inputs has a circuit, hence a description—flat at worst, one gate step per gate. The prototype's $\{\mathrm{add}, \mathrm{mul}\}$ over $\mathbb{Z}_{2^{B}}$ is not complete—its circuits compute polynomial functions modulo $2^{B}$, and for $B \ge 2$ not every function is one—so a comparison or bit-extraction gate must join $\Sigma$ for general computation. Given one, any computation of bounded length is a `repeat` of a step circuit with memory as multiplexers, at the price zkVMs pay for the same generality. Inference needs none of this: after shape specialization a transformer forward pass is a static dataflow graph with affine indexing—products, elementwise maps, reductions and permutations, each a repetition over rows, columns or heads—and layers are copies of one kind because their weights are inputs.

**What the envelope costs on inference.** Four places in a language model are not natively static; each is handled by padding to a fixed envelope or by routing the data-dependent choice through advice. The embedding lookup selects row $t$ of a $V \times d$ table by the token id $t$, a value of $x$; since structure may not depend on values, the lookup is a $V$-way select of $O(Vd)$ gates per token, the order of the unembedding's $d \times V$ product, so it at most doubles that layer. Nonlinearities—GELU, softmax, normalization—are fixed-point polynomial approximations over the modular gates or gates of $\Sigma$ with a specified relation, a choice between more gates and a larger $\Sigma$ (§7.6). Variable sequence length is padding to a maximum $T_{\max}$, with masks as multiplication by zero and one, at the ratio of padded to actual length. Mixture-of-experts routing is the one real design trade. With $E$ experts and top-$k$ routing, the padded envelope evaluates every expert and costs $E/k$ on the feed-forward blocks; alternatively the routing decisions travel as advice, about $k \log_2 E$ bits per token per layer, charged to $A$. The first costs gates and the second costs the bound; both are sound, and §8's accounting is what makes the second admissible at all. [TODO: the advice bits and the $E/k$ overhead for one concrete model in §10.]

**Compile cost.** Table 7.1 gives circuit size, description size and the verifier's costs from this section for each model of §10; the matmul rows are measured.

| Circuit | Gates | $\lvert G \rvert$ (bytes) | Definitions | $\mathrm{Compile}$ | Per-kind table | $C[i]$ |
|---|---|---|---|---|---|---|
| matmul $64 \times 64 \times 64$, $B = 8$ | $520{,}192$ | $2{,}245$ | 5 | $0.23\,\mathrm{ms}$ | $10\,\mathrm{ms}$ | $4.6\,\mu\mathrm{s}$ |
| matmul $256 \times 256 \times 256$, $B = 8$ | $3.3 \times 10^{7}$ | $2{,}533$ | 5 | $0.26\,\mathrm{ms}$ | $0.17\,\mathrm{s}$ | $4.6\,\mu\mathrm{s}$ |
| matmul $1024 \times 1024 \times 1024$, $B = 8$ | $2.1 \times 10^{9}$ | $2{,}834$ | 5 | $0.28\,\mathrm{ms}$ | $2.8\,\mathrm{s}$ | $4.7\,\mu\mathrm{s}$ |
| GPT-2 small, 100+100 tokens | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [TODO: remaining models of §10] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

Table 7.1. Single-threaded Python on [TODO: machine]; $\mathrm{Compile}$ is the minimum of five runs, $C[i]$ the mean over two thousand random addresses, and the per-kind table is derived once per model.

## 7.6. Limitations

Parameters are integers bound before parsing, so a description cannot express structure that depends on the values of $x$; anything that does goes through charged advice. All gates of a set share one output width in the prototype, and per-gate widths, which mixed-precision inference wants, are a change to the value codec and to the width accounting of $\mathrm{Out}$ rather than to the grammar. A unit boundary must today be a definition boundary, so the constructor's factoring of the program is load-bearing for granularity, and canonical chunking of long step lists—so that a cut can fall inside a definition without the client's cooperation—is a later phase. Transcendental gates with specified rounding would shrink the nonlinearity envelope of §7.5 at the cost of a larger $\Sigma$ and a relation the verifier must check exactly.

---

**Notes for the author** (not part of the section)

- *Check against §5/§6 as written.* (a) Lemma 7.2's remarks say §6's $\ell_r^{\star}$ lemma needs only that $\partial$ contain every value read across a replay-unit boundary; confirm the lemma's hypothesis is phrased that way. (b) §7.2 step 4 gives $W$ as the prototype's formula, $W = (|\mathrm{io}| + qsP)(1 + \lceil \log_2 n \rceil) + qsS + qm + 1$ with sums over verification kinds; check it matches §6.5's cost expression, and that $|\mathrm{io}|$ (positions opened in §6's step 1, excluding weights under $\kappa_W$) is the quantity §6 uses. (c) I cite "§5.4" for granularity and "§6.6" for choosing units and rates; renumber if the outline differs. (d) Theorem 7.5 states the total as $U + A$ bits (union over $2^{A}$ circuits, each with $\le 2^{U}$ reachable outputs); your brief said "at most $U$ bits beyond a fixed function of $(G, x)$". If §8 defines the certified total to include $A$, adjust the theorem's last sentence to §8's symbol.
- *Measured numbers.* All matmul figures came from a throwaway script under `/tmp` importing the repo's `Compiler`, `MatmulG` and `Index` directly (the top-level `veritor` import is currently broken by the in-progress `VerificationPolicy` refactor; `research.py` still passes `eta`). The workload with 64 rows, $k = 64$, 64 columns has $520{,}192$ gates and $528{,}384$ addresses ($8{,}192$ inputs); your brief said "528,384 gates"—that is the address count. Compile times are the minimum of five runs with `public_inputs` hoisted out of the timed region (it is an $O(|x|)$ flatten that inflated my first measurement). The per-kind table's cold time is dominated by enumerating the root's `In` and `Out` (`Definition.reads`, `local_outputs`), so it is $O(|x| + |y|)$, not $O(|G|)$; Proposition 7.3 says so. If you want the table $O(|G|)$, the root's `in_count` need not be enumerated—only verification kinds' $|\mathrm{In}|$ enter $W$.
- *TODOs left:* `[TODO: cite Lengauer–Wanke]`; `[TODO: cite]` for hash-consing (Ershov 1958, Goto 1974; possibly Filliâtre–Conchon 2006); `[TODO: cite]` for JAX tracing; `[TODO: the advice bits and the $E/k$ overhead for one concrete model in §10]`; Table 7.1 rows for GPT-2 small (100+100 tokens) and the remaining §10 models; `[TODO: machine]` in the table caption.
- *Wording that commits the paper to a choice.* (1) $\partial = \mathrm{in}(C) \cup \bigcup_r \mathrm{Out}(R_r)$ with $\mathrm{Out}(R_r)$ the *declared* interface resolved to owned gates (pass-through outputs excluded), and $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r)$; this is what the code does and what makes rank/unrank prefix sums. (2) Inputs are boundary positions outside every unit; units contain gates only. The §4 toy figure shows input gates inside units and needs reconciling, as you noted. (3) $h_C = H(\textsf{compiled}, H(G), H(\Sigma))$ binds the marks (bytes of $G$) and the gate set's identity and nothing else—not $x$, $a$ or $\theta$. (4) Lemma 7.1 fixes advice at exactly $A$ bits (padding shorter advice); the prototype accepts up to $A$ bits, giving $2^{A+1} - 1$ circuits. Pick one and make §8 agree. (5) The parameter-binding step (§7.2 once-per-run step 3) is design, not prototype; in the prototype $G$ is already shape-specialized and that step reuses the once-per-model table. (6) The opening paragraph before §7.1 is mine; drop it if you prefer the section to open on "Suppose the client claims".
- *House-rule audit.* No "prover", "server", "well-formed", "word(s)", "kernel", "IR", "instance", "artifact", "plugin", "challenge(s)", "trust the compiler" or "authorized computation" in the section proper. "Well-typed" once (values); "design trade" once (MoE); "no uncharged degrees of freedom" as the theorem's name; the asymmetry principle once (§7.1, constraint 4). New symbols not in your list: $D$ (denominator bit bound), $d$ (nesting depth), $P$, $S$ (per-kind sums inside $W$), $c_K$ (copies of kind $K$), $\mathrm{pre}_t$ (address prefix sum), $|\mathrm{io}|$; none collides with $B$, $m$, $n$, $N$ or your notation, but rename freely.
