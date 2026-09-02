# Section 7 ("Secure circuit compilation"): context dump for the writing agent

Compiled 2026-09-02 from the repository at `/Users/danielreuter/projects/veritor` (HEAD as of this morning; the working tree has one unrelated modification, `docs/paper/section-5-covert-capacity.md`), from `docs/paper/section-7-secure-circuit-compilation.md` (the current §7 draft, 181 lines), `docs/paper/section-5-covert-capacity.md`, `docs/security-argument.md`, `docs/overnight-report.md`, `docs/frontier-report.md`, `docs/lean-collaborator-handoff.md`, `README.md`, the Notion exports in `/tmp/notion-outline.md`, `/tmp/notion-outline-sections-6-7.md`, `/tmp/notion-section7-outline.md`, `/tmp/notion-section6-scratch.md`, and the design transcript `eb746331-5a84-4468-9455-5c2a7c14f35c` (referred to below as "the transcript", with the author's turns quoted verbatim and numbered `L<n>` by JSONL line). Every number marked *measured* was produced this morning with the scripts described in §8 and §10 of these notes; every number marked *estimate* or *unsourced* is labelled as such.

Conventions in this file: code fences are `~~~`; inline math is `$...$`; code is cited as `path:line` or `path::symbol`. "Outline" means the author's Notion outline (§7 at `/tmp/notion-outline.md` lines 469–503 and the endorsed detailed outline `/tmp/notion-section7-outline.md`); "draft" means the existing §7 draft; "code" means the repository. Where the three disagree there is a **Discrepancy** callout with each version and a recommendation.

---

## 0. Read-me for the writing agent

### 0.1 What §7 is for, in the paper's architecture

The paper reduces the problem to three trusted functions on two objects. The outline (`/tmp/notion-outline.md` §4.2–4.3, and the transcript's §4.2 draft at L1214) has:

- **Compile** $(G, x, a) \to (C, I)$: the circuit $C$ and its index $I$. §7.
- **Bound** $(C, I, \theta) \to U$: a structural upper bound on the bits an accepted run can carry. §5.
- **Verify** $(\tau;\, C, I, \theta)$: the sampling protocol. §6.
- §8 composes the three.

The reading order §5 Bound, §6 Verify, §7 Compile is the reverse of data flow. The author's assistant put it this way at L1369 and the author kept it in the endorsed outline: *"The section's job is to discharge §6's assumption ('a committed $(C, I)$ with these properties') without trusting anything the client sends, and to do it sublinearly. Everything else is in service of that."* The draft's opening paragraph says the same in Thaler's register (`docs/paper/section-7-secure-circuit-compilation.md:3`).

The architecture diagram has four arrows. The author's words at L1213: *"I agree with the advice for production orders untenable. I agree with the fourth arrow."* The fourth arrow was proposed by the assistant at L1205 (item 4 of a criticism list: *"Draw Compile at the bottom producing $(C,I)$; Bound and Verify as siblings above it, both reading $(C,I)$; θ flowing from Bound to Verify; §8 on top."*), accepted by the author at L1213, and drawn in the §4.2 draft at L1214 with this caption: *"Compile at the bottom emits $(C,I)$. Bound and Verify sit above it, both reading $(C,I)$. A fourth arrow carries $\theta$ from Bound to Verify. §8 on top composes the three."* So the arrows are Compile → Bound, Compile → Verify, Bound → Verify (carrying $\theta$), and the three → §8. See §10.3 below for the diagram's later revision (θ is now the client's proposal priced by Bound rather than chosen by Bound; the arrow survives as "Bound admits θ").

### 0.2 What §7 may assume from §5 and §6, and what it must deliver

**May assume (names as in the current drafts).**

- From §5 (`docs/paper/section-5-covert-capacity.md`): a circuit $C$ is a finite list of gates with addresses $1..n$, each gate $i$ with an operation from a public gate set, argument addresses, and a value set $V_i$ (the §5 draft writes widths as $|V_i| = 2^{w_i}$; the code has one width $B$ per gate set for now). Input gates are constants (their value is pinned to $x$); output gates are a designated subset of addresses. A transcript $\tau$ assigns a value to every gate; a gate is *incorrect* if its value is not its operation applied to its arguments' values *or is not well-typed* (the author's word, L1359: *"also I think we call it well-typed, not well-formed -- the property you refer to"*). The downstream-cut bound $\kappa(E)$ and the union-over-admissible-$E$ definition of $U$.
- From §6: replay units $R_1, \dots, R_m$ (partition the gates), verification units $V_{r,k}$ (refine them), the boundary $\partial := \mathrm{In} \cup \bigcup_r \mathrm{Out}(R_r)$ (the §6 agent may write $\mathrm{in}(C) \cup \mathrm{out}(C) \cup \bigcup_r \mathrm{Out}(R_r)$; the code's `Index.boundary()` docstring notes the outputs are always inside the union already, `src/veritor/core/index.py:324-333`), interiors $\mathrm{Int}(r) := R_r \setminus \partial$, the policy $\theta = (q, s)$, the threshold $\eta$, the $\ell_r^\star$ lemma whose precondition is exactly "every value a selected unit reads from outside is in $\partial$", the verifier's expected work $W$, and the fact that §6 requires nothing of the units beyond tiling and refinement.

**Must deliver.**

1. To §6: a committed pair $(C, I)$ such that (a) replay marks tile the gates and verification marks tile each replay unit; (b) $\partial$ as derived from $I$ contains every cross-unit read; (c) $C[i]$, $\mathrm{In}/\mathrm{Out}/\mathrm{Size}/\mathrm{Cost}$ of a unit, `owner(i)`, and $\mathrm{rank}_\partial(i)$ are answerable in $O(\text{depth})$ without materializing $C$; (d) the per-kind table that Bound and $W$ fold over.
2. To §8: the lemma that, given committed $G$ and public $x$, at most $2^{A}$ circuits are reachable ($A$ = the advice bound), and the statement of where each client-supplied object is committed, pinned, charged or priced. The outline says: *"State this as the lemma §8 will cite"* (`/tmp/notion-section7-outline.md`, 7.3 first bullet).
3. To §8's accounting: the prototype's accounting is `Capacity = Bound + 8·|a|` and *"every accepted request has capacity ≤ U_max + A"* (`docs/overnight-report.md:120-121`; `src/veritor/protocol/parameters.py:36-60` docstring).

### 0.3 The author's stylistic constraints, verbatim where possible

Collected from the transcript. Each is a direct quote of the author unless marked "assistant, accepted".

1. **The paper's objects.** Assistant (L1369, in the outline the author then endorsed): *"don't reintroduce a kernel, a description object, or an instance as named things; the paper's objects are $G$, $C$, $I$, $\Sigma$, and Compile, and everything in this section is either one of those or an algorithm on them."* The author had driven this: at L1316 the author asked *"Why do we need 'kernel' or `load` -- why isn't Compile() fine? ... It seems like we can drop D and instead just do H(C, I) or something? I'm not sure what address set is. Are these just helper functions over our core paper-level methods (like In(), Out()). I'm not sure we need Instance but could be wrong."* The assistant agreed to drop all of them (L1317 and the restatement at L1348).
2. **7.1 is constraints and consequences, not a rationale essay.** Assistant (L1369, endorsed): *"Don't let 7.1 become the design-rationale essay; the rationale is 7.4's last paragraph, and 7.1 should read as constraints and consequences."*
3. **Compile outputs $(C, I)$ directly; no separate $\gamma$.** Author (L1176): *"Actually, no. I very much prefer this thing where compile spits out a circuit and an index over it rather than this like separate gamma thing. I also don't know what your shapes thing is or why that's a real thing."* (The "shapes thing" was the assistant's offline/online split conditioned on $\mathrm{shape}(x)$; see §9.11 below on what "shape(x)" now means.)
4. **"Well-typed", not "well-formed".** Author (L1234): *"(also I think we call it well-typed, not well-formed -- the property you refer to)."* Note the code and the README still say "well-formed" for the *description*; the author's rule was about *values*. Recommendation: in the paper, use "well-typed" for values (a value is a bitstring of its gate's output width) and avoid "well-formed" for descriptions; say "valid" or "in the grammar" (the draft already does this; see §12).
5. **"Design decisions", not "design challenges".** Author (L1215): *"Note that these core design challenges that you wrote for 4.2.3 aren't really challenges. No, we could maybe instead just say design decisions that we made instead of challenges. Maybe that would unify the sections better."*
6. **No "authorized computation"; the verifier checks nothing about what $C$ computes.** Author (L1215): *"Sorry, I don't know if trust is a real thing. We don't need it to be the program on X. Sorry, basically at no point do we check anything. The whole point is we're just bounding the exfiltration rate. There's no notion of an authorized computation. All we do is just build basically an arbitrary computation. It could be completely unrelated to the original one. The only property it has is that it's fixed and that any additional choices are tracked. This actually feels potentially important to mention."*
7. **The asymmetry principle.** Author (L1215): *"we can do a pass of every gate in the circuit just because the circuits are quite compressed. They have, they can all be described in the same way. They can be compressed. The values cannot be compressed. We don't attempt to compress the values, and so they're enormous. And so we cannot check every value. And so we need everything to follow from just local checks of values, but because of that some of the values can be infinite. The point is that if you pass that into just a circuit that we know and that we have done this structural analysis of, that doesn't actually affect our bound."* Author (L1232): *"We want to say that we assume we can run algorithms over every gate in the circuit, but only over a tiny percentage of the values."* The assistant's refinement (L1233, accepted): *"'a pass over every gate' should be 'a pass over the description.' $10^{15}$ gates can't be visited even structurally; the pass is over $10^3$–$10^6$ definitions, and everything is a fold over those."* The compressed form for the paper: *we can run algorithms over every gate but only over a tiny fraction of the values* (paraphrasing L1232). The author rejected the phrase "random-access asymmetry" (L1217: *"Not sure what random-access asymetry means"*) and the assistant restated it as *"structure is cheap to look up, values are expensive to look up"* (L1218).
8. **Values are not bounded by their width until locally checked.** Author (L1232): *"'Values can be infinite' becomes: a value is bounded by its gate's output width — Note this is not true. We would have to check this. But we can only check locally. Nothing is preventing committing to some values that are too big."* The assistant's fix, which the author accepted (L1233): a gate is *incorrect* if its value is not its operation applied to its arguments *or is not a well-typed value of its output width*; the code already does this (`decode_value` rejects anything `encode_value` would not produce). §7 should not claim well-typedness for unopened positions.
9. **No "words".** Author (L1217): *"'Fixed width words' what does this mean. Generally I hate 'words.' I prefer talking about output widths of gates and/or value widths."* Note the code's gate set is literally named `veritor.word-arithmetic` (`src/veritor/core/gates.py:263`); do not carry the name into the paper.
10. **$G$ is once per epoch.** Author (L1217): *"G is once per epoch."* And at L1294: *"epoch I think is like one run of the protocol btw (it's possible we want to forget about this for now)"*. And at L1292: *"I don't think it's important for G to be fixed before x is seen."* And at L1294: *"G and x are both public in some sense (like in the real confidentiality-preserving protocol they'd be committed prior)."*
11. **The developer is the adversary.** Author (L1340): *"No, the developer is the adversary. The only thing trusted in this system is the verifier and the channel. All code etc. is untrusted. even the selection of replay and everything that's all completely untrusted and i agree cost is a helper like cost is something we can yeah it's just it's just like a little property of a circuit or something like that. We can label the operators with cost or something like that as part of our public operator set or the public gate set."*
12. **Every gate is in a unit.** Author (L1453): *"I prefer every gate is in a unit. I want you to try to steer the code in that direction, we might change this later."* Landed as `docs/overnight-report.md` §2.1; it reverses the draft's "inputs lie in no unit". See §9.6.
13. **Advice for reduction/production orders is untenable.** Assistant (L1205), accepted by the author at L1213: *"Advice for reduction order is untenable. A permutation of a $10^4$-term reduction is $\sim 10^5$ bits of advice per reduction. Nondeterminism of that kind has to be removed by construction: exact arithmetic (modular integer, fixed-point) is associative, so the canonical order in $C$ and the client's actual order agree bit for bit. Advice is for sparse choices: sequence length, expert routing, sampling seeds."* The author: *"I agree with the advice for production orders untenable."* (The author wrote "production orders"; the assistant's text said "reduction order".)
14. **Client, not prover/server.** From the §6 agent's notation table the author forwarded (L1349): "Party: client; retire prover, server." The draft's house-rule audit (`section-7-secure-circuit-compilation.md:181`) lists the banned words: "prover", "server", "well-formed", "word(s)", "kernel", "IR", "instance", "artifact", "plugin", "challenge(s)", "trust the compiler", "authorized computation".
15. **Scale and ML.** Author (L1236, on §4; the figure remark is L1238): *"the problem is it just doesn't really talk about scale. And also, we're immediately going to need to talk about ML circuits. Right? We're basically only going to be talking about ML circuits in the actual paper."* And on the figure: *"I don't think we'd be able to fit 14 gates in the figure."* The 8-gate example is the compromise.
16. **The author's own 7.1 text.** At L1368 the author pasted a 7.1 (reproduced in §1.1 below) and asked: *"7.2 i think should just be protocol where you actually write out the protocol, then later at some point there should be a construction. The basic idea should also motivate our choices etc."*

### 0.4 What this dump is not

It is not a draft. It is the material. The writing agent should treat the existing draft (`docs/paper/section-7-secure-circuit-compilation.md`) as the baseline to revise, and use §12 of this file (paragraph-by-paragraph keep/revise/drop) as the revision plan.

---

## 1. §7.1 Basic idea

### 1.1 The `workload` example and the arithmetic relation

The author's 7.1 text (L1368), which the draft keeps verbatim (`section-7-secure-circuit-compilation.md:5-33`):

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

On $x = (1, 1, 2, 3)$: `total = 2`; iteration 0 adds 2, iteration 1 adds 1 then 1; `square = 4`; returns $4 \cdot 5 = 20$. The claimed arithmetic relation is $(1+1)^2 (2+3) = 20$. The author's text: *"A compiler can replace the program's loops, branches, and bookkeeping with a fixed directed graph of elementary operations. Such a graph is called a circuit. Every gate in a circuit has a stable address, so the verifier can now choose a random gate—or a random group of gates—from a well-defined population. In the figure below, the square gate summarizes the loop's arithmetic effect."*

The 8-gate circuit (`/tmp/notion-section6-scratch.md`, "EXAMPLE V2", 1-indexed as in the paper):

| gate | operation | reads | value (honest) | replay unit | verification unit |
|---|---|---|---|---|---|
| 1 | in ($x_1 = 1$) | — | 1 | $R_1$ | $V_{1,1}$ |
| 2 | in ($x_2 = 1$) | — | 1 | $R_1$ | $V_{1,1}$ |
| 3 | add | 1, 2 | 2 | $R_1$ | $V_{1,2}$ |
| 4 | square | 3 | 4 | $R_1$ | $V_{1,2}$ |
| 5 | in ($x_3 = 2$) | — | 2 | $R_2$ | $V_{2,1}$ |
| 6 | in ($x_4 = 3$) | — | 3 | $R_2$ | $V_{2,1}$ |
| 7 | add | 5, 6 | 5 | $R_2$ | $V_{2,2}$ |
| 8 | mul | 4, 7 | 20 | $R_2$ | $V_{2,2}$ |

$\tau = (1,1,2,4,2,3,5,20)$; $\partial = \{1,2,4,5,6,8\}$; $\mathrm{Int}(1) = \{3\}$, $\mathrm{Int}(2) = \{7\}$; output position 8, no output-copy gate.

**This circuit compiles today, exactly as drawn, with every gate in a unit.** Measured this morning (script in §10.2): tracing it with the untrusted `Tracer` gives a 2 040-byte canonical description of 6 definitions; `Compiler.compile` takes 0.38 ms; `C.evaluate((1,1,2,3))` returns `(1, 1, 2, 4, 2, 3, 5, 20)`; `Index.boundary()` enumerates `[1, 2, 5, 6, 4, 8]` (inputs by rank, then $\mathrm{Out}(R_1)$, then $\mathrm{Out}(R_2)$; as a set it is the paper's $\partial$); `Index.interior(0) = [3]`, `Index.interior(1) = [7]`; `C.Out(R_1) = [4]`, `C.In(R_2) = [4]`, `C.Out(V_{1,1}) = []` (source gates are excluded from $\mathrm{Out}$), `C.In(V_{2,2}) = [4, 5, 6]`. One detail for the figure: the public gate set has no `square` gate; gate 4 is `mul(3, 3)` (both arguments the same address). If the figure labels it $x^2$ the text should say the square is a multiplication whose two arguments coincide, or Σ must be given a `square` gate (the `evaluation/serving.py` table treats `square` as `mul(x, x)` too: `src/veritor/evaluation/serving.py:62`).

### 1.2 The four constraints

The author wrote constraints 1 and 2 (L1368):

1. *"The verifier's choice must be unpredictable. If the client could predict which operation would be chosen, then it could always perform that correctly. This can be achieved via random sampling."*
2. *"The verifier must have access to the full list of operations. Random sampling is only possible given a fixed population."*

The assistant added 3 and 4 (L1369), and the author endorsed the four in the outline (`/tmp/notion-section7-outline.md`):

3. *"The population is fixed by things the client can't vary per run without paying. A circuit per se isn't enough: if the client could pick a fresh circuit for each output, the population would be fixed only after the fact. So the circuit must be a deterministic function $\mathrm{Compile}(G, x, a)$ of a committed program $G$, the public input $x$, and advice $a$ that is charged bit-for-bit. Every degree of freedom is either committed, pinned, or charged. This is the security statement of the section."*
4. *"The verifier can answer questions about the population without reading it. $10^{11}$ gates; the verifier will look at a few thousand. So it needs $C[i]$, the interfaces of subcircuits, and the unit structure in time logarithmic in the circuit, from a description it can hold. This is the efficiency statement."*

What the code makes true of each:

- (1) is §6's business; the code's challenge derivation is `src/veritor/protocol/challenge.py` (binomial count from the seed, then Floyd's subset; commit `3e8ffab`, transcript L1348 item 10).
- (2) $C$ is `Circuit` (`src/veritor/core/circuit.py:89`), addresses $[0, n)$, `C[i]` a `GateRef(op, args, width, source)` (`circuit.py:45`).
- (3) `research.Compile(G, x, a, gate_set, *, limits, max_advice_bits)` (`src/veritor/research.py:56-112`): checks `8 * len(a) <= max_advice_bits`, validates `G.digest`, runs `G(x, a)` (any exception including `SystemExit` becomes a `CompileError`), compiles the returned bytes with `Compiler(gate_set, limits).compile(description, inputs)`, returns `Compilation(compiled, constructor_digest, inputs, advice)`. Determinism is tested at `tests/veritor/compile/test_compiler.py:20` (`compiled.digest == Compiler(GATES).compile(payload, inputs).digest`) and `tests/veritor/security/test_canonical.py:36`. The charge is `Capacity(compilation, θ, η) = Bound + 8·|a|` (`src/veritor/research.py::Capacity`; `docs/overnight-report.md:120`).
- (4) `DescriptionCircuit.__getitem__` (`circuit.py:418`) descends a `Frame` chain in $O(\text{depth})$; `Index.replay_units.unit(k)`, `Units.owner(i)`, `Index.boundary().rank(i)` are prefix-sum descents (`index.py:133-190`, `index.py:474-548`). Measured: `C[i]` 15–28 µs and `boundary.rank` 18–76 µs on matmul shapes from $8^3$ to $4096^3$ gates ($1.4 \times 10^{11}$ gates at $4096^3$), i.e. flat in $n$ (§8.2).

### 1.3 The two observations

Assistant (L1369, endorsed):

- *"The verifier does not check what the circuit computes. Your example already shows it: the compiler replaced a data-dependent loop with a square gate, a different program with the same output. The protocol is indifferent. There is no 'authorized computation'; there is a fixed circuit and charged choices. This is what lets the entire lowering pipeline, constructors, tracers, front ends from JAX or PyTorch, sit outside the trust boundary: a wrong lowering is just a different fixed circuit."*
- *"The circuit is the program's own structure. Programs come with functions and loops; the tracer records them, and that call structure is exactly the compressed description the verifier needs. The client authors nothing extra. Where data-dependent control flow can't be re-expressed (your loop), it is padded to an envelope; the envelope's cost is a constant factor for inference, and §7.5 says how much."*

Code backing: the tracer records each `@tracer.definition(...)` function as one hash-consed definition and each call as a `call` step (`src/veritor/constructors/tracer.py:275-327`); Python loops unroll into steps unless the constructor uses `tracer.repeat` (`tracer.py:329`). The `workload` loop is *not* re-expressed by any tool in the repo: the 8-gate circuit is hand-traced (the compiler in the prose is "a compiler", not the repository's). Say so if the writing agent wants to be exact: the repository has no loop-to-gate optimizer; the example illustrates what any lowering may do because the protocol does not care.

### 1.4 What the existing draft says about 7.1 and how it differs from the outline

The draft (`section-7-secure-circuit-compilation.md:5-45`) keeps the author's text verbatim, states the four constraints as a numbered list, then adds three paragraphs: "The circuit resolves constraints 1 and 2. Constraint 3 is where the exfiltration game parts from ordinary verifiable computation..." (with the *client is the developer* framing, correct per L1340); "Constraint 4 is quantitative..." (the asymmetry, phrased as "structure ... can be analyzed globally; values ... can only be checked locally", which matches the author's L1214/L1232 words); and the two observations. Differences from the outline:

- The outline's 7.1 has the two observations *after* the constraints and then *"Close 7.1 with the figure and the one-sentence roadmap"*. The draft places the figure between the author's text and the constraint discussion and ends with a roadmap sentence. Both are fine; the outline's order is figure last.
- The draft's constraint 3 says "$G$ committed once per epoch"; the author's L1217 says the same. Keep.
- The draft says $10^{11}$ gates in constraint 4; the §5 draft's GPT-2 Small number is $4.2 \times 10^{10}$ (§8.1 below); the serving table for a 70B-class shape is $1.9 \times 10^{17}$. Pick one scale and use it consistently; see §8.
- The draft's opening paragraph before 7.1 (line 3) is the assistant's, flagged as optional in its own notes (line 180 item 6).
- The draft's 7.1 says nothing about the *tracer being untrusted*; the outline's second observation makes the point. The draft's second observation does say "tracer records ... the client authors nothing extra." Adequate.

---

## 2. §7.2 Protocol, as implemented versus the outline

### 2.1 The outline's protocol (endorsed at L1369)

**Once per model.** Client sends $G$ (with replay and verification marks) and the weight commitment $\kappa_W$. Verifier: (1) parses $G$, rejecting anything outside the grammar or over resource limits; (2) checks well-formedness per definition: arity, in-range references, dependency order; (3) checks the marks per definition: replay marks tile the gates, verification marks tile each replay unit, no verification kind exceeds the proof-cost cap; (4) derives the per-kind table (copies, size, cost, interface widths); (5) publishes $H(G)$, which is $H(C, I)$.

**Once per run.** Client sends $x$, $y^*$, $a$, proposed $\theta$. Verifier: (6) checks $|a| \le A$ and $\theta$'s denominators; (7) binds the description's parameters from $\mathrm{shape}(x)$ and $a$, obtaining the closed circuit; (8) computes $U = \mathrm{Bound}(C, I, \theta)$ and its own work $W$; proceeds iff $U \le U_{\max}$ and $W \le W_{\max}$; (9) hands $(C, I, \theta)$ to §6.

*"Say explicitly what the verifier never does: run client code, read a gate list, or believe any quantity the client asserts about its own circuit. Every number in steps 3–8 is derived."*

### 2.2 What the code does, step by step, with pointers

There is no once-per-model / once-per-run split in the code. `Compile(G, x, a)` runs the constructor *per run* and the description it emits is *closed* (no parameters). What is fixed per model is the constructor's digest and $\kappa_W$; the compiled description's digest is computed per run. Concretely:

**Constructor identity (per model).** `constructor_digest(name, version, parameters)` (`src/veritor/compile/constructor.py:38`) is a domain-separated hash of the constructor class name, its version string and its parameters (e.g. `MatmulG(8).digest`, `ClusterG(shape, pods, slots, steps).digest`). It is bound in the header as `Header.constructor` (`src/veritor/protocol/messages.py:201`). It is *not* a hash of the constructor's Python code; it names public code both parties hold (`research.py:74-79` docstring). Test: `tests/veritor/constructors/test_matmul.py:169` (`MatmulG(8).digest != MatmulG(16).digest`), `test_requests.py:109-111`.

**Weights (per model).** `commit_weights(gate_set, values)` builds $\kappa_W$ over the weight *vector by rank*, bound to a fixed tag and the gate set digest, needing no circuit (`docs/overnight-report.md:89-101`, commit `676f379`). Position $k$ is the $k$-th `weight` gate in address order of whichever circuit is verified (`Index.weights()`, `index.py:291`). Bound in the header as `Header.weights` (`messages.py:207`). Test: one root accepted for two batch shapes of the same model (`overnight-report.md:97`).

**Parse and validate $G$'s output (per run; outline steps 1–2).** `parse_description(payload, gate_set, limits)` (`src/veritor/compile/description.py:381-435`):

- `len(payload) <= limits.max_description_bytes` (10 MB default) *before* decoding (`:391`);
- valid UTF-8 JSON; re-serialized canonical bytes must equal the payload byte for byte (`:404-408`, "description is not canonically serialized"); no duplicate keys, no non-finite numbers (`:102-112`);
- document keys exactly `{version, definitions, root}`, `version == 2` (`:410-412`);
- `len(definitions) <= limits.max_definitions` (100 000) (`:414`);
- each definition's digest must equal the digest of its canonical body (`:424`), appear once (`:422`), and be validated by `_definition` (`:250-325`): keys exactly `{input_count, role, steps, outputs}`; role in `{null, "replay", "verification"}`; `len(steps) <= max_steps_per_definition` (1 000 000); each step's kind in `{gate, call, repeat}`; gate names must exist in Σ and argument count must equal the gate's arity (`:213-219`); `call`/`repeat` digests must name a definition defined *earlier* in the list (`:234`, dependency order); every range must lie within the definition's ports or the slots defined so far, for every copy of a repeat (`_range`, `_ranges`, `:146-195`); at least one declared output (`:284`); per-definition totals (gates, slots, outputs) $\le$ `max_addresses` ($2^{63}-1$), replay and proof cost $\le$ `max_cost`, nesting depth $\le$ `max_depth` (256) (`:286-295`); declared outputs must resolve within `max_output_runs` (256) runs, and so must the definition's `in` and `weight` gate layouts (`:296-306`); the resolved position count must equal the declared `output_count` (`:307-312`, the F7 invariant); the owned output runs must be pairwise distinct (`:313-323`);
- the running total of resolved output runs over the description $\le$ `max_output_runs_total` (16 384) (`:429`);
- `root` must name a definition (`:433`) and the root's `input_count` must be 0 (`:435`, "the root has no ports; inputs are `in` gates").

**Marks (outline step 3).** `validate_marks(root, limits)` (`src/veritor/core/index.py:642-720`), run by `Index.__init__` (`index.py:264`), *once per definition, never per copy*:

1. Replay marks tile the gates: above the replay cut every step is a `call` or `repeat` into a covered definition; a bare `gate` step above the cut is rejected naming the step (`_offending_step`).
2. Verification marks tile every replay unit the same way.
3. No mark is nested inside a mark of the same role; a verification mark may not contain a replay mark; the root may not be marked verification.
4. Every verification-marked definition's `proof_cost <= limits.max_verification_unit_proof_cost` (`index.py:685-690`).
5. A marked definition must have at least one gate (`index.py:665`).

**The proof-cost cap exists in code**, as `CompilationLimits.max_verification_unit_proof_cost` (`src/veritor/core/limits.py:20`), default $2^{63}-1$ (i.e. unbounded unless a verifier sets it). It is a *completeness* condition (an honest client must be able to prove each unit), not a soundness one (transcript L1361 item 7; `docs/security-argument.md` §8). Test: `tests/veritor/compile/test_index.py:426` (`test_units_must_have_gates_and_verification_units_respect_the_proof_cap`).

**Per-kind table (outline step 4).** `Index.kinds()` (`index.py:350`) returns one `KindSummary` per reachable definition: `kind` (definition digest), `role`, `copies`, `size`, `replay_cost`, `proof_cost`, `input_count` (declared ports), `out_count`, `out_bits`, `source_inputs`, `source_weights`, `min_depth`, `max_depth`, `children` (child kind → copies called directly), `verification_units`, `verification_kinds` (`index.py:193-226`). `Compiled.kind_table()` wraps it with `root`, `n`, `input_count`, `weight_count`, `replay_unit_count` and the compiled digest (`src/veritor/core/compiled.py:34-38`, `index.py:228-256`). Bound, Cost and `expected_work` are functions of this table alone (`KindTable` docstring, `index.py:229-236`): a table *written from a model* of a circuit (`veritor.evaluation.serving`) is priced exactly like one profiled from a compiled description. Measured: `kinds()` 0.04–0.09 ms on every matmul shape (§8.2); tests `tests/veritor/compile/test_index.py:168,260`.

**Publish $H$ (outline step 5).** `Compiled.digest = identity_digest("veritor/compiled/v1", {"description": H(canonical description bytes), "gate_set": Σ.digest})` (`compiled.py:12,28-32`; `description_digest` at `compile/description.py:84`, docstring: *"The identity of a canonical description: H(C, I) up to the gate set"*). The description bytes *include the role marks* (they are the `role` field of each body), so a mark change changes the digest: tests `tests/veritor/security/test_canonical.py:76` and `tests/veritor/protocol/test_scaling.py:268`. Bound in the header as `Header.compiled_digest` (`messages.py:200`). This digest is per compiled description, i.e. per run, not per model.

**Once per run (outline steps 6–9).**

- (6) advice bound: `research.Compile` rejects `8 * len(a) > max_advice_bits` before running $G$ (`research.py:92`); `VerifierSession._admit` rejects `advice_bits > parameters.max_advice_bits` (`src/veritor/protocol/session.py:469-473`). Two knobs for one bound, kept equal (`overnight-report.md:127-130`). Denominators: `VerificationLimits.max_probability_denominator_bits = 64` enforced at `session.py:477`.
- (7) parameter binding: **does not exist.** The constructor emits a closed description; there are no integer parameters in the grammar (§4.7 below). The verifier "binds parameters" by running $G(x, a)$.
- (8) `_admit` (`session.py:454-500`): enforces `max_units` on replay and verification unit counts and `max_positions_per_unit` per kind (`:480-483`), computes `W = expected_work(compiled, policy, len(io))` and rejects `W > parameters.max_work` (`:484-490`), and if `parameters.max_capacity is not None` computes `bound(compiled, policy, parameters.eta).bits` and rejects `> max_capacity` (`:491-497`). All before any commitment is accepted (`docs/security-argument.md` §6; `docs/overnight-report.md:198`). `max_capacity` has no default (`parameters.py:36-60`; F2, `overnight-report.md:132-140`).
- (9) the session then runs §6 on `Compiled`.

The expected-work formula the code prices (`parameters.py:106-125` docstring): $W = (|IO| + qsA)(1 + d) + qsS + 1 + qR$, with $R$ replay units, $A = \sum_k m_k(\mathrm{size}_k + \mathrm{in}_k)$ and $S = \sum_k m_k \mathrm{size}_k$ over verification kinds, $d = $ Merkle depth. The draft's step 4 formula matches (§12).

**Header binding.** `Header` (`messages.py:187-207`) binds `compiled_digest`, `constructor` (digest), `advice` (bytes), `policy` ($\theta = (q, s)$), `eta` (the verifier's), `public_inputs` (encoded $x$), `claimed_outputs` ($y^*$), `weights` ($\kappa_W$ or `None`). Protocol version v6 after advice was added (`overnight-report.md:119`).

### 2.3 What the verifier never does, and what it does in the prototype

Outline: never runs client code, never reads a gate list, never believes a client-asserted quantity.

**In the prototype the verifier runs $G$.** The trust-model docstring (`src/veritor/research.py:74-82`), verbatim:

> Trust model.  In this prototype the verifier executes ``G`` as ordinary Python identified by a versioned digest: like the gate set, ``G`` is public code both parties hold, and the header binds ``G.digest`` and ``a`` so that everything about the run beyond the advice is a deterministic function of ``(G, x, a)``.  A deployment would run ``G`` sandboxed and metered, or have the client prove ``Compile(G, x, a) = (C, I)`` (paper §7).  The description-size and compilation limits bound the *output* of ``G``; nothing here bounds ``G``'s running time.

`docs/overnight-report.md:230-231`: *"no sandbox for G (only output limits; SystemExit is caught, os._exit is not)"*. `docs/overnight-report.md:289-291` (discrepancy 9): *"The verifier runs G in this prototype. §7 of the paper (secure circuit compilation) is the deployment answer; the outline should say the prototype's trust model plainly (public versioned G, output-bounded, not time-bounded)."*

**Discrepancy (verifier runs client code).** Outline/draft: the verifier never runs client code; $G$ is a *description* in a grammar; parameters are bound from $\mathrm{shape}(x)$ and $a$. Code: $G$ is a Python constructor; the verifier runs it per run; its output (the description) is what the grammar validates. Author (L1301): *"G and x are both public in some sense"*; overnight decision 2.6: *"Advice is structural, and the verifier runs G."* Recommendation: the paper should define $G$ as the description (data, not code) as the outline does, state that in the paper's protocol the client sends $G$ and the verifier only *reads* it, and then say in one sentence (in 7.2 after "what the verifier never does", or in 7.6) that the prototype instead lets the verifier execute a public constructor to *produce* the description per run, and that the two deployment options are a sandboxed, metered constructor or a proof that $\mathrm{Compile}(G, x, a) = (C, I)$. Do not claim the prototype has parametric descriptions.

### 2.4 The transparent-to-hiding discipline

The draft's 7.2 opening (line 49) says the transparent verifier is the specification and §8's construction replaces each read with a commitment and each check with a proof, adding: *"Here that is simple, because Compile is deterministic and $O(|G|)$: the client can prove it ran Compile correctly on a committed $G$."* Nothing in the code proves compilation; the sentence is design. The §6.4 scratch says the same discipline for §6 (`/tmp/notion-section6-scratch.md`, "Zero-knowledge implementation").

---

## 3. §7.3 Security properties: exact statements, where proven, where tested

The one idea (draft line 80, assistant): *"determinism turns a choice into a count, whatever the client's compiler, constructor or intentions."*

### 3.1 Determinism / binding (the lemma §8 cites)

**Statement (outline).** $(C, I)$ is a function of $(G, x, a)$; $H(G)$ binds it. Given committed $G$ and public $x$, at most $2^{|a|} \le 2^{A}$ circuits are reachable; this is the $A$ term in the end-to-end bound.

**Statement (draft Lemma 7.1, line 82).** *"Fix $\Sigma$ and $A$, and let $a$ range over strings of exactly $A$ bits, shorter advice being padded. For every description $G$ and input $x$, the set $\{\mathrm{Compile}(G, x, a) : a \in \{0,1\}^{A}\}$ has at most $2^{A}$ elements, and a party holding $h_C$, $x$ and $a$ computes the same $(C, I)$ as the verifier."*

**What the code makes true.**

- `Compile` is a pure function of bytes: *"Everything is a pure function of the bytes: no clock, no randomness, no environment"* (`docs/security-argument.md:560-562`). The parse rejects non-canonical bytes before any definition is examined (`compile/description.py:404-408`), so two parties with the same bytes have the same digest and layout; two parties with the same digest have (up to SHA-256 collisions) the same bytes, hence the same $(C, I)$.
- Advice: `research.Compile` accepts `8 * len(a) <= max_advice_bits` (`research.py:92`), i.e. *up to* $A$ bits in whole bytes, not exactly $A$. Counting all byte strings of length $\le A/8$ gives $\sum_{k=0}^{A/8} 256^k < 2^{A+1}$ circuits, not $2^{A}$. The draft's note (line 180, item 4) flags this: *"the prototype accepts up to $A$ bits, giving $2^{A+1}-1$ circuits. Pick one and make §8 agree."* The prototype's *charge* is `Capacity = Bound + 8·|a|`, i.e. the actual advice length, and the admission theorem is *"every accepted run has capacity ≤ U_max + A"* (`overnight-report.md:120, 283`). Either statement is sound; the paper needs one.
- The digest that binds: `Compiled.digest = H("veritor/compiled/v1", {description: H(bytes), gate_set: H(Σ)})` (`compiled.py:12,28-32`). It binds the marks (they are bytes of $G$) and Σ's identity, not $x$, $a$ or $\theta$; those are bound separately in the `Header` (§2.2).
- The $\mathrm{shape}$ dependence: in the prototype $G(x, a)$ may read *anything* about $x$, including values (it is Python); determinism holds regardless because $x$ is bound in the header as `public_inputs`. The paper's design excludes structural dependence on values of $x$ by making $G$ a description with integer parameters bound from $\mathrm{shape}(x)$. See §9.1–9.2.

**Tests.** `tests/veritor/compile/test_compiler.py:20` (`test_compile_returns_the_circuit_the_index_and_a_bound_digest`: two compiles of the same bytes give equal digests); `tests/veritor/security/test_canonical.py:36,53,76,118,152` (same bytes → same digest and layout; re-encoded bytes rejected before compile work; a mark change changes the digest and a transcript recorded under the other compilation is `EXPECTATION_MISMATCH`); `tests/veritor/constructors/test_matmul.py:169` and `test_requests.py:109` (constructor digests differ across parameters). Audit verdict: *"proved + tested"* (`security-argument.md:606`).

**Precise lemma text the writing agent can adapt.** Let $\Sigma$ be fixed. $\mathrm{Compile}$ is a deterministic function of $(G, x, a)$ that either rejects or returns $(C, I)$, and $H(C, I)$ is a deterministic function of $(G, \Sigma)$ alone (it does not depend on $x$ or $a$ in the paper's design where $G$ is the parametric description; in the prototype it depends on the closed description $G(x, a)$ emits). For fixed $(G, x)$ the number of pairs $(C, I)$ the verifier can admit is at most the number of admissible advice strings, at most $2^{A}$ if advice is padded to exactly $A$ bits, at most $2^{A+1}$ if any length up to $A$ is admitted. §8 uses: the union over admissible $a$ of §6's reachable output sets has size at most (number of admissible $a$) × $2^{U_{\max}}$, hence at most $U_{\max} + A$ (+1) bits.

### 3.2 Derived structure is sound (the precondition of §6's $\ell^\star$ lemma)

**Statement (outline).** The interfaces the verifier computes from the description are *supersets* of the flat circuit's actual cross-reads (a caller can only reach declared outputs), so $\partial$ as derived contains every cross-unit read *for any marks*. Size, cost and the antichains are exact.

**Statement (draft Lemma 7.2, line 88).** (i) replay units partition the gates, verification units partition each replay unit, *and inputs lie in no unit* [OUTDATED, see below]; (ii) every address inside $R_r$ read by a gate outside $R_r$, and every circuit output inside $R_r$, lies in $\mathrm{Out}(R_r)$, so $\partial = \mathrm{in}(C) \cup \bigcup_r \mathrm{Out}(R_r)$ contains every value read across a replay-unit boundary; (iii) Size, Cost, $|\mathrm{In}|$, $|\mathrm{Out}|$ and the width of Out in bits are identical for all copies of a kind and computed exactly from the definition.

**Why (ii) holds, from the grammar.** Every argument of every step is a `Range` in *relative* coordinates: `space = "input"` names the enclosing definition's own ports, `space = "local"` names output slots of earlier steps in the same definition (`core/description.py:12-19`). A call's slots are `child.output_count` many, i.e. exactly the child's *declared* outputs; a parent has no other coordinate for anything inside a child copy. By induction over nesting, every reference from outside a copy into it passes through a chain of declared-output slots, and the compiler resolves each declared output to the gate that owns it (`Definition.resolved_outputs`, `description.py:357-393`; `_call_pieces`, `:692`). $\mathrm{Out}(R_r)$ is exactly the set of gates in $R_r$ that any declared output of $R_r$'s definition resolves to, minus its source gates (pinned: they are boundary positions already under the input commitment or under $\kappa_W$) (`description.py:28-33` "Interfaces" paragraph). Hence any gate outside $R_r$ reading inside $R_r$ reads a member of $\mathrm{Out}(R_r)$ or a pinned source gate; both are in $\partial$ or under $\kappa_W$.

**Code.** `Index.boundary()` (`index.py:324-333`): *"``In ∪ ⋃_r Out(R_r)``: the addresses the boundary commitment covers. The input gates by rank, then the units' ``Out`` in unit order. The weight gates are committed under their own root and are not here. The circuit outputs are always inside this set: every output resolves through the declared interface of the replay unit that owns it."* `Index.interior(r)` (`index.py:335-348`) = the unit's interval minus its `out_runs`, `input_runs`, `weight_runs`. A second, independent check in the protocol: `_Layout.required` refuses any sampled unit that reads an address owned by another replay unit that is not a boundary position, verdict `INVALID_COMPILED_RESULT` (`security-argument.md:630-635`), *"the compiler makes this unconstructible"*.

**Tests.**

- `tests/veritor/security/test_tiling.py:60` `test_cross_unit_reads_go_only_through_declared_outputs`; `:152` `test_layout_rejects_a_circuit_that_reads_across_the_cut` (on a *forged* `Compiled`, since the compiler rule makes the case unconstructible; audit §8 verdict).
- `tests/veritor/compile/test_index.py:98` `test_boundary_and_interiors_match_the_reference_derivation` (boundary and interiors vs a flat reference derivation); `:42` units are the marked copies with lazy lookup; `:130` input and weight domains lazy and ranked in address order.
- `tests/veritor/compile/test_out_runs.py`: `:244` run interfaces match enumeration on a case table; `:249` `Out` runs are what the output ranges resolve to; `:286` **400 random strided declarations over permuted-output children against the per-ordinal resolver** (the fuzzer added for F7); `:302` a strided run inside one copy stops at the declared count (the F7 regression); `:339` `input_count` is the declared interface, not what is read; `:356-404` distinct-outputs rule (same slot twice rejected, overlapping strides rejected, duplicates through pass-through children rejected, disjoint interleaved strides accepted).
- `tests/veritor/core/test_circuit.py` (`DescriptionCircuit` vs `FlatCircuit`: `test_description_circuit_matches_flat`, `test_gate_at_is_o_depth`, per the summary of the prior session).

**F7, precisely** (`security-argument.md:819-845`). The within-copy branch of `_call_pieces` (added for the fan-out bound, commit `fd478dd`) *"computed the last slot of a copy at the declared stride and did not clamp it to the declared count, so a strided declaration inside one copy of a child resolved to every slot of that copy at that stride."* Consequences before the fix (`ba43d29`): when the extra gates collided with other declared gates the distinctness rule rejected a *valid* description (liveness: 480 of 2 325 small cluster shapes failed to compile); when they did not collide the definition compiled with an `Out` *larger* than declared, so $\partial$ held undeclared positions, `out_bits` and hence `Bound` were inflated (the sound direction), and `Cost` was off. *"Nothing let a prover open a wrong value: the per-ordinal resolver that places values was correct throughout; only the run-typed summary disagreed with it."* Fix: the clamp; and the compiler now rejects any definition whose resolved positions do not number `output_count` (`compile/description.py:307-312`, *"internal error in output resolution"*). Lesson recorded: *"every run-typed summary (Out, In, source runs) has a per-element reference path, and each should be fuzzed against it."*

**(iii) exactness.** `Definition.size`, `replay_cost`, `proof_cost`, `output_count`, `out_count`, `out_bits`, `input_total`, `weight_total`, `replay_units`, `verification_units` are per-definition sums over steps weighted by repeat counts (`description.py:276-310`, `450-490`). A copy is a translation of its definition (`description.py:22-27` "Layout"), so every quantity is copy-independent. Tests: `tests/veritor/compile/test_index.py:168` `test_kinds_table_summarizes_each_definition_once`; `:260` `test_kinds_table_matches_enumeration_of_every_copy`. **Caveat for the paper**: `Cost` labels are the gate set's `replay_cost`/`proof_cost` (`gates.py`), *outside the trusted base*: miscalibration costs efficiency, never soundness (author L1340; the §6 agent's brief).

**Discrepancy (draft's Lemma 7.2 (i) "inputs lie in no unit").** Draft: inputs occupy $[0, |x|)$ outside every unit; $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r)$. Code (after `overnight-report.md` §2.1, author L1453): inputs and weights are zero-arity source gates inside units; the root has no ports (`compile/description.py:435`); `In` = the set of `in` gates (`Index.inputs()`, `index.py:286`); $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r) \setminus \mathrm{pinned}(R_r)$ (`index.py:335`). Recommendation: rewrite (i) as *"every gate, source gates included, lies in exactly one replay unit and exactly one verification unit"* and (ii)'s $\partial$ as $\mathrm{In} \cup \bigcup_r \mathrm{Out}(R_r)$ with In the `in` gates; weights are not in $\partial$ (they are under $\kappa_W$).

### 3.3 No condition on the marks beyond tiling and refinement

**Statement (outline).** *"Partition and refinement are the only requirements; everything else is priced by Bound. (Point to the argument, or put it in an appendix.)"*

**The argument, as the author asked for it.** Author (L1360): *"Can you dump the argument for why we actually don't need to place any constraints on the partitions, except for the max U?"* The assistant's dump (L1361), condensed to its load-bearing steps and checked against the code:

1. The acceptance probability of *any* client strategy whose incorrect verification units are exactly $E$ is at most $\sigma_\theta(E) = \prod_r (1 - q + q(1-s)^{|E \cap R_r|})$. This is a statement about the sampling, proven for any tiling (audit §5 Claim A, *"$P[\text{accept}] \le \sigma(E^\star)$ for every strategy"*, `security-argument.md:336-366`); it uses only that every gate is in exactly one replay unit and one verification unit, that a sampled unit's every gate is checked against owners' values, and that the challenge is independent of the transcript (seed revealed after commitment).
2. The outputs reachable when all incorrect gates lie in units $S_1, \dots, S_m$ number at most $\prod_j 2^{\mathrm{out\_bits}(S_j)}$ (downstream cut; audit §5 Claim B step 1). This uses only that every value outside $S$ is a function of $S$'s declared outputs and the pinned inputs/weights, which is Lemma 7.2 (ii): true for any marks.
3. $\mathrm{Bound}$ is the $\log_2$ of the total weight of the distinct covers of all admissible $E$ (those with $\sigma_\theta(E) > \eta$), relaxed monotonically (knapsack over per-kind cost series, Laplace/Chernoff, cap at `out_bits`; every rounding admits more, `security-argument.md:376-427`). Neither ingredient assumes anything about unit shape, size, balance or count.
4. Therefore a "bad" partition (one giant unit and many tiny ones; a unit whose declared interface is wider than the circuit's output; a unit of a single gate) makes $U$ *larger*, never the bound *wrong*. The verifier admits iff $U \le U_{\max}$ and $W \le W_{\max}$; a partition that lets too much through fails the first test.

The author's worry (L1322): *"What if you just pick all your replay units to be up to generate size except for one? Like it just feels like there's edge cases that are."* The assistant's answer (L1323 and L1361): the edge case is priced; with one unit holding almost everything, $\kappa$ of that unit is almost the whole output and $\sigma_\theta(\{$that unit$\})= 1-q+q(1-s)^{1} > \eta$ for any $q < 1-\eta$, so $U \approx |\mathrm{out}(C)|$ and the run is rejected under any meaningful $U_{\max}$. This is now *measured*: `docs/frontier-report.md:35, 7`: *"the `step`, `layer`, `matvec` and `row` units expose internal activations that are wider than the whole output ... The four intermediate partitions ... certify nothing at any policy in the grid: their bound is the full 16 Mbit everywhere."* And on the cluster fixture, `overnight-report.md:256-270`: unit cut widths 144–672 bits against 160 output bits, so Bound is vacuous at every fractional policy until the fold is made reach-aware (pending; `frontier-report.md:220`).

**Where the code enforces exactly this and nothing more.** `validate_marks` (`index.py:642-720`; §2.2 above) checks tiling, refinement, non-nesting, non-empty marked definitions and the proof-cost completeness cap. There is no balance, size, count or interface condition. Tests: `tests/veritor/security/test_tiling.py:31,48,84,111`; `tests/veritor/compile/test_index.py:324-457` (a gate step above the replay cut is named in the error; stray source gates caught by the tiling check; a gate step inside a replay unit must be in a verification unit; marks may not nest; verification needs a replay unit; units must have gates; the root may itself be the only replay unit; the root has no ports). Audit §8 verdict: *"proved + tested (the cross-cut read is tested only on a forged circuit; the compiler rule is tested directly)"*.

**Positional bits.** The author raised (in the §5 review) the question of whether choosing *which* unit to corrupt is itself a channel. It is, and the union definition counts it: Bound certifies $|\bigcup_{E\ \mathrm{admissible}} \mathrm{outputs}(E)|$, not $\max_E$ (`overnight-report.md:251-255`; audit Claim B). §7 need only say that Bound is the union and that Lemma 7.2 gives it its inputs.

### 3.4 Efficiency

**Statement (outline).** Compile is $O(|G|)$; $C[i]$, In/Out/Size/Cost of a unit, `owner(i)`, and $\mathrm{rank}_\partial(i)$ are $O(\text{depth})$. *"This is what §6.4's construction assumed."*

**What the code does.**

- Parse and validate: one pass over the bytes (`json.loads` is $O(\text{bytes})$ under `max_description_bytes`), one pass over definitions; per definition, checks are linear in its steps plus the output-resolution work, which is bounded by `max_output_runs` runs per definition and the distinctness check is quadratic in the number of *runs*, never outputs (`compile/description.py:25-31`). `validate_marks` is one pass over reachable definitions (`index.py:642`). `Index.kinds()` is one pass, parents first, copy counts flowing down (`index.py:350-390`).
- `C[i]`: `DescriptionCircuit.__getitem__` (`circuit.py:418`) descends from the root `Frame`: bisect on step starts within a definition (`Frame.child`, `description.py:825`), divide by the child's size within a repeat, recurse; at a gate step, resolve each argument by walking back up the frame chain (an `input` reference to the range the parent passed at this copy's index, a `local` reference to a slot of an earlier step). This is the lazy `_Frame` chain from the prior session's fix (the summary's "Prover Session Scaling Problem": `_gate_at` used to rebuild argument tuples at every level, $O(N^3)$ in `occurrence_summary`; now $O(\text{depth} \times \text{arity})$).
- `Units.unit(k)`, `Units.owner(i)`: bisect the per-step prefix sums of unit counts, divide by the child's count in a repeat, descend to the first marked copy on the path (`index.py:133-190`).
- `boundary.rank(i)` / `unrank(r)`: the input gates by rank (`_Sources`, prefix-summed per step, `index.py:417-470`), then per replay unit the prefix sums of `out_count` (`_Boundary`, `index.py:474-548`); within a unit, `Definition.out_rank`/`out_offset` bisect the unit's `out_starts` (`description.py:450-480`).
- Interfaces as runs: `Definition.out_runs`, `input_runs`, `weight_runs` are tuples of `Run(start, count, stride)` computed once per definition (`description.py:395-448`); `In(unit)` / `Out(unit)` on `Circuit` return address sets (`IndexedDomain`: `contains`/`rank`/`unrank`/`size`, `core/indexed.py:18-40`; `IntervalDomain`, `IntervalDifferenceDomain`, `RangeIndexedDomain`, `:123-400`).

**Measured this morning** (single-threaded Python 3, this laptop; script in `/tmp/s7/mm.py`, gate set `make_word_gate_set(8)`, `MatmulG(8)`):

| shape (rows × k × cols) | gates $n$ | description bytes | definitions | compile | `kinds()` | `C[i]` mean | `boundary.rank` mean |
|---|---|---|---|---|---|---|---|
| $8^3$ | 1 152 | 2 886 | 9 | 0.5 ms | 0.06 ms | 15 µs | 18 µs |
| $64^3$ | 528 384 | 3 409 | 9 | 0.6 ms | 0.05 ms | 20 µs | 21 µs |
| $256^3$ | $3.36 \times 10^{7}$ | 3 710 | 9 | 0.6 ms | 0.05 ms | 24 µs | 27 µs |
| $1024^3$ | $2.15 \times 10^{9}$ | 4 010 | 9 | 14 ms (cold; 0.5–1 ms warm in other runs) | 0.06 ms | 26 µs | 28 µs |
| $4096^3$ | $1.37 \times 10^{11}$ | 4 293 | 9 | 1.0 ms | 0.09 ms | 28 µs | 76 µs |

The $4096^3$ row is $1.4 \times 10^{11}$ gates in 4.3 KB, compiled in a millisecond. The description grows by $\approx 300$ bytes per factor 4 in $k$ (one more sum-tree level: $O(\log k)$) and not at all in rows or columns. Tracing $G$ (the untrusted client side) took 6 ms at $8^3$ and 234 ms at $4096^3$ because the tracer materializes `Wires` slices; that is the client's cost, not the verifier's.

**Measured earlier (with sources).**

- `docs/overnight-report.md:43-48`: *"1024³ matmul: 4 010 B description, 2.1×10⁹ gates, compile 18 ms; `kinds()` 2.84 s → 0.36 ms"* after the per-kind table stopped enumerating the root's `In`/`Out` (the summary's "KindTable in_count/out_count O(|x|+|y|)" fix, landed as range arithmetic). The draft's Table 7.1 still shows *"per-kind table 2.8 s"* for $1024^3$ and $10$ ms for $64^3$; those are pre-fix numbers and should be replaced (today: 0.06 ms at every shape).
- `docs/overnight-report.md:48`: *"Bound on a 10⁸-gate synthetic index: 1–40 ms"* (Laplace and knapsack folds, per-kind).
- `tests/veritor/protocol/test_scaling.py:221` `test_verifier_phases_are_flat_in_the_number_of_replay_units` (1k → 32k units; the verifier's per-phase cost does not grow with the unit count), `:231` `test_a_full_verifier_run_touches_only_sampled_addresses`, `:255` end to end on the large tiling, `:190` `test_the_tiling_scales_as_described`.
- `tests/veritor/security/test_canonical.py:161` `test_a_description_of_a_trillion_gates_compiles_in_bounded_time`.
- `tests/veritor/compile/test_compiler.py:56` `test_compile_and_index_are_sublinear_in_the_gate_count`.
- `tests/veritor/compile/test_out_runs.py:548` `test_admission_does_not_scale_with_the_input_count` (monkeypatched to detect enumeration of $x$), `:451` a strided subset of a huge slot-linear interface is one run.
- Frontier sweep: 588 `Bound` evaluations on a $2.7 \times 10^{17}$-gate table took 50.3 min of bound computation (≈ 5 s each with `max_buckets = 2^{22}`, Laplace only) (`frontier-report.md:5`). This is Bound's cost, not Compile's, and is dominated by the knapsack grid size, not $n$.
- Toy LM / cluster (`tests/veritor/constructors/test_cluster.py:1-9` docstring; reproduced this morning): default fixture `LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)`, four requests on 2 pods × 2 slots × 6 steps: 6 564 gates, 8 replay units (weights + 7 occupied steps), 1 065 verification units, 42 kinds, description 34 507 B, traced 3.1 ms, compiled 4.3 ms; evaluating every gate ≈ 50 ms. `RequestsG` on the same requests: 33 731 B, 40 kinds, 5 replay units (weights + 4 requests), boundary 19 positions (vs 139 for the cluster: the step units expose KV-cache rows). A larger toy (`vocab 16, d_model 8, heads 2, layers 2, context 16`, 8 requests, 2 × 4 × 12): 94 956 gates, 46 kinds, 64 465 B, compiled 8.0 ms.

**Discrepancy (efficiency of the per-kind table).** Draft Proposition 7.3 says deriving the per-kind table costs $O(|G|)$ *"plus time linear in the declared interfaces it lists, of which the root's are $x$ and $y$"*, and the draft's note (line 178) says the table's cold time was $O(|x| + |y|)$. Code today: `KindSummary.input_count`, `out_count`, `out_bits` are computed from runs (`Definition.out_total`, `out_count`, `out_bits`, `description.py:450-490`), so the table is $O(|G|)$; `overnight-report.md:44` records the fix (2.84 s → 0.36 ms). Recommendation: state Proposition 7.3 as $O(|G|)$ outright and drop the interface caveat.

### 3.5 Completeness

**Statement (outline).** Every well-formed description compiles; an honest client's marks pass.

**Code.** Every constructor in the repo compiles and runs the protocol honestly: `tests/veritor/constructors/test_matmul.py`, `test_lm.py`, `test_cluster.py` (including `test_mixed_lifetimes_in_one_pod_compile_and_decode_like_the_reference`, the F7 liveness regression, and a sweep of small cluster shapes), `test_cluster_protocol.py`, `test_requests.py`. `Tracer` emits only canonical bytes (`tracer.py::serialize`). The proof-cost cap is the one completeness condition that a verifier can tune (`CompilationLimits.max_verification_unit_proof_cost`); `VerificationLimits.max_units` (1 000 000) and `max_positions_per_unit` (10 000 000) are admission-time completeness limits too (`limits.py:26-40`; `session.py:480-483`).

### 3.6 Resource-limit soundness (a malicious $G$ cannot make the verifier's compile step do unbounded work)

Not in the outline's 7.3 list, but the audit treats it as part of Compile's security (`security-argument.md:542-606`, §7 "Compile determinism and canonical encoding": *"description size, definition count, step count, nesting depth and output-run fan-out bound the compile work"*).

`CompilationLimits` (`src/veritor/core/limits.py:8-24`), defaults: `max_description_bytes = 10_000_000`; `max_definitions = 100_000`; `max_steps_per_definition = 1_000_000`; `max_addresses = 2^63 - 1`; `max_cost = 2^63 - 1`; `max_depth = 256`; `max_verification_unit_proof_cost = 2^63 - 1`; `max_output_runs = 256` (*"Pieces one definition's declared outputs may resolve to; bounds the work of resolving them"*); `max_output_runs_total = 16_384` (*"Resolved output runs over the whole description; bounds the distinctness check"*).

The attack the run limits close (commit `fd478dd`; the summary's "compile-time DoS via strided output ranges"): a small description with a `repeat` step whose declared output stride is coprime to the child's size forces one residue-class run per element; without a cap, the $O(\text{runs}^2)$ distinctness check is a DoS. Now `resolve_outputs(limit)` and `resolve_source_runs(source, limit)` return `None` as soon as the limit is exceeded and the compiler rejects (`compile/description.py:296-306`); `_call_pieces`, `_grid`, `_merge` are lazy generators (`core/description.py:602-750`); a shortcut recognizes slot-linear children and within-copy strides so common cases are one run (`test_out_runs.py:451`). Tests: `test_out_runs.py:461` `test_interfaces_resolving_to_too_many_runs_are_rejected_without_doing_the_work`; `:474` `test_the_total_number_of_runs_over_a_description_is_capped`; `test_canonical.py:184` nesting deeper than the limit is a compile error; `:53` oversize bytes rejected before decode.

Not bounded: the *running time of $G$ itself* in the prototype (`research.py:81-82`; audit §10). `SystemExit` raised by $G$ is converted to `CompileError`; `os._exit` is not catchable.

The audit's honest gaps (`security-argument.md:593-604`): determinism across machines rests on `json.dumps(sort_keys=True, ensure_ascii=True)` with no cross-implementation vectors; `json.loads` runs over the whole payload before the canonical check ($O(\text{bytes})$); *"Compilation work is bounded by limits, not by a proof of linearity in the description size."*

### 3.7 The theorem that ties it together (draft Theorem 7.5)

Draft line 102: *"Fix $\Sigma$, $A$, $\eta$, $U_{\max}$ and $W_{\max}$. Let $G$ be a description accepted once per model and $x$ an input. For every run the verifier of §7.2 admits, with advice $a$ and proposal $\theta$: $(C, I) = \mathrm{Compile}(G, x, a)$ is one of at most $2^{A}$ pairs determined by $(G, x)$; $(C, I)$ satisfies the assumptions of §6; and $U = \mathrm{Bound}(C, I, \theta) \le U_{\max}$. Consequently the set of outputs $y^*$ accepted with probability above $\eta$, over all admissible runs with the same $(G, x)$, has size at most $2^{U_{\max} + A}$."* The draft's remark on $\theta$ being *priced* rather than *charged* is right and matches the code (`_admit` prices $\theta$ through $U$ and $W$; `overnight-report.md:275-277` discrepancy 6: $\eta$ is the verifier's, $\theta$ the client's proposal).

**Caution.** The theorem's "$\le 2^{U_{\max}+A}$" counts outputs accepted with probability $> \eta$ *for one fixed $(G, x)$*. The prototype's accounting is per run, `Capacity = Bound + 8|a| ≤ U_max + A`. Both are fine; §8 must pick the form. Note also that a *different* $x$ is a different population and is not charged; $x$ is public and pinned (author L1301).

---

## 4. §7.4 Construction

### 4.1 The description language, exactly as implemented

Wire format (`src/veritor/compile/description.py:3-16`, verbatim):

~~~text
{"version": 2,
 "definitions": [{"digest": "<hex>", "body": {...}}, ...],   # dependency order
 "root": "<hex>"}

body  = {"input_count": n, "role": null | "replay" | "verification",
         "steps": [step, ...], "outputs": [range, ...]}
step  = {"kind": "gate", "gate": name, "args": [range, ...]}
      | {"kind": "call", "digest": "<hex>", "args": [range, ...]}
      | {"kind": "repeat", "count": n, "digest": "<hex>", "args": [jrange, ...]}
range = [space, start, count, stride]           space = "input" | "local"
jrange = [space, start, count, stride, jstride]
~~~

Canonical JSON: sorted keys, no whitespace, no floats, ASCII (`core/identity.py::canonical_json_bytes`); a definition's `digest` is SHA-256 of its canonical body under a domain tag (`definition_digest`, `compile/description.py:78`); the description's digest is SHA-256 of the canonical bytes (`description_digest`, `:84`). `FORMAT_VERSION = 2`.

Semantics (`core/description.py:1-38` module docstring, paraphrased tightly):

- A **definition** has `input_count` ports, a straight-line list of steps, declared `outputs` (ranges) and an optional `role` mark.
- A **gate step** is one scalar gate of Σ applied to the values in its argument ranges; the concatenated ranges must have exactly the gate's arity many elements (`_step`, `compile/description.py:213-219`). It has one output **slot**.
- A **call step** is one copy of an earlier definition $D$; its arguments, concatenated, must have exactly `D.input_count` elements; it has `D.output_count` slots.
- A **repeat step** is `count` copies of $D$; copy $j$ receives arguments whose start is shifted by `j * jstride`; it has `count * D.output_count` slots, copy-major. `jstride = 0` broadcasts an operand to every copy.
- A **range** `(space, start, count, stride)` names `count` coordinates `start, start+stride, ...` in `space = input` (the definition's own ports) or `space = local` (slots of earlier steps in the same definition). Every coordinate must be in range for every copy of a repeat (`_range`, `_ranges`, `compile/description.py:146-195`).
- **Source gates** `in` and `weight` are zero-arity gates of Σ (`Gate(..., source=INPUT_SOURCE | WEIGHT_SOURCE)`, `gates.py`); they take their values from the environment ($x$ by rank; $W$ by rank) and live inside units like every other gate. **The root has no ports** (`input_count` must be 0; `compile/description.py:435`).
- **Layout.** The circuit's addresses are the root's gates in step order, each call or repeat copy occupying one contiguous block. Copy $j$ of the step at position $t$ in a copy with base $b$ begins at $b + \mathrm{pre}_t + j \cdot \mathrm{Size}(D_t)$.
- **Interfaces.** `Out` of a definition is its declared outputs resolved to the gates a copy owns, *minus its source gates* (pinned positions). It is a union of arithmetic progressions kept as a tuple of `Run(start, count, stride)` (`description.py:116-145`), computed once per definition in time proportional to the description. **Distinct-outputs rule**: the runs must be pairwise disjoint (`_repeated_output`, `compile/description.py:367-380`, via `_common_offset` and an extended gcd). **`In` = declared ports**: `KindSummary.input_count` is the definition's `input_count`, not what is read (`test_out_runs.py:339`).
- **Declared outputs without Out copy gates.** A definition's output *is* a gate it owns (or a pass-through of a port, `PieceKind.PORT`, which is excluded from `Out` because the port already belongs to the parent's coordinates); there is no copy gate. The 8-gate example's output is gate 8 itself.

Resolution (`core/description.py:553-790`): `_declared_pieces` walks each declared range; `_output_pieces` maps a `local` coordinate to the step and slot; for a gate step the piece is that gate; for a call/repeat, `_call_pieces` (`:692`) maps the slot to the child's `resolved_outputs` translated by the copy's base; strided declarations that cross copies fall back to residue-class runs (`_grid`, `:602`) merged by `_merge` (`:626`); `_lift` (`:754`) translates a child's runs into the parent's address space. The F7 clamp lives in the within-copy branch of `_call_pieces`.

### 4.2 The eight-gate example as bytes

Traced this morning with the untrusted `Tracer` over `make_word_gate_set(16)` (the script is in §10.2; `pair_in` emits two `in` gates directly rather than through `tracer.inputs(2)` because the tracer's canonical source cells are themselves verification-marked and a verification mark may not contain another). Canonical bytes: **2 040 B, 6 definitions**. One definition per line, compacted (the file on the wire has no newlines or spaces):

~~~text
{"body":{"input_count":0,"outputs":[["local",0,2,1]],"role":"verification","steps":[{"args":[],"gate":"in","kind":"gate"},{"args":[],"gate":"in","kind":"gate"}]},"digest":"ebd99d60…2890"}
{"body":{"input_count":2,"outputs":[["local",1,1,0]],"role":"verification","steps":[{"args":[["input",0,2,1]],"gate":"add","kind":"gate"},{"args":[["local",0,2,0]],"gate":"mul","kind":"gate"}]},"digest":"47b8baf7…8821"}
{"body":{"input_count":0,"outputs":[["local",2,1,0]],"role":"replay","steps":[{"args":[],"digest":"ebd99d60…2890","kind":"call"},{"args":[["local",0,2,1]],"digest":"47b8baf7…8821","kind":"call"}]},"digest":"4591d6e3…a2fc"}
{"body":{"input_count":3,"outputs":[["local",1,1,0]],"role":"verification","steps":[{"args":[["input",0,2,1]],"gate":"add","kind":"gate"},{"args":[["input",2,1,0],["local",0,1,0]],"gate":"mul","kind":"gate"}]},"digest":"cdf24207…55e2"}
{"body":{"input_count":1,"outputs":[["local",2,1,0]],"role":"replay","steps":[{"args":[],"digest":"ebd99d60…2890","kind":"call"},{"args":[["local",0,2,1],["input",0,1,0]],"digest":"cdf24207…55e2","kind":"call"}]},"digest":"c440686c…451a"}
{"body":{"input_count":0,"outputs":[["local",1,1,0]],"role":null,"steps":[{"args":[],"digest":"4591d6e3…a2fc","kind":"call"},{"args":[["local",0,1,0]],"digest":"c440686c…451a","kind":"call"}]},"digest":"c7242b2a…b475"}
"root": "c7242b2a…b475", "version": 2
~~~

Full digests: `ebd99d6020fdd3eab97208f4b91e28fbd9f009e1ee0cce923e35a55a09272890` (pair of `in` gates, verification), `47b8baf77c26defbb9a32b658505c43a9f983d97875c64d767a36b12fed8b821` (add then square, verification), `4591d6e3e8417c8005ddb9a38904e1038042c1cb3a4c3e6df29d2cff1a89a2fc` ($R_1$), `cdf24207ea76c4c2a4b56c8fcf20582cf2846a88eb72c3af2df08168338f55e2` (add then mul, verification), `c440686c689cf27144afce202ecac531e8a6f2d252719eba538197ffedb9451a` ($R_2$), `c7242b2a6d582c7ed9446fd7a24fec45c60cba4a1895724237c4d567ad61b475` (root). Compiled digest `2d978a8f380c973687676ad9f67c48f19f38e8eb0efd94a821ed5bb3d16a1936` (against `make_word_gate_set(16)`, digest `f482b1fe…3314`).

Reading it as the paper's figure: the pair-of-inputs cell is *one kind with two copies* (`copies=2` in the kind table: $V_{1,1}$ and $V_{2,1}$ are the same definition), $R_1$ has no ports, $R_2$ has one port (the square, passed from the root's slot 0), and the root's declared output is `local 1` = $R_2$'s output = gate 8. The verifier's per-kind table for this circuit (`Index.kinds()`, measured):

~~~text
kind      role          copies size replay proof in out out_bits src_in src_w children                  vus
c7242b2a  (root)        1      8    6      10    0  1   16       4      0     4591d6e3×1, c440686c×1    4
4591d6e3  replay (R1)   1      4    3      5     0  1   16       2      0     ebd99d60×1, 47b8baf7×1    2
ebd99d60  verification  2      2    0      2     0  0   0        2      0     —                         1
47b8baf7  verification  1      2    3      3     2  1   16       0      0     —                         1
c440686c  replay (R2)   1      4    3      5     1  1   16       2      0     ebd99d60×1, cdf24207×1    2
cdf24207  verification  1      2    3      3     3  1   16       0      0     —                         1
~~~

(`replay`/`proof` are the gate set's cost labels summed: `add` 1/1, `mul` 2/2, `in` 0/1 in `make_word_gate_set`; `gates.py:232-262`.) Note `out_bits = 0` for the source-only cell: its declared outputs are pinned, so `Out` is empty and it contributes $\ell = 0$ to the knapsack (F4).

### 4.3 The matmul "in six lines"

**Discrepancy (six lines, three definitions).** Outline 7.4: *"Give the matmul in six lines (three definitions, one step each)."* Draft 7.4: five definitions (`mul`, `add`, `dot`, `row`, `batch`) with `batch` taking $X$ and $W$ as 8 192 ports. Code today (`src/veritor/constructors/matmul.py::MatmulG`): **nine definitions**, because (a) `mul` and `add` are wrapped as one-gate definitions so `repeat` can call them (a `repeat` needs a definition, not a bare gate), (b) the activations and the weights are two replay units of source gates (`activations_unit`, `weights_unit`, each a `repeat` of a one-gate canonical cell that the tracer marks `verification`), and (c) the root `batch` has no ports and calls those two units. The arithmetic core is still three definitions (`dot`, `row`, `batch`) and reads as six-ish lines. Recommendation: show the three arithmetic definitions as the "six lines", then say in one sentence that the sources are two more replay units of `in` and `weight` gates and that `mul`/`add` are one-gate definitions so that `repeat` can name them; give the true byte count (2.8–4.3 KB) and definition count (9).

Rendered from the real description for $X$ a $2 \times 3$ block, $W$ a $3 \times 2$ matrix, $B = 8$ (2 773 bytes, 32 gates; measured):

~~~text
in_cell      [verification] inputs 0: gate in()                                             -> out loc[0]
activations  [replay]       inputs 0: repeat 6 in_cell()                                    -> out loc[0:6:1]
weight_cell  [verification] inputs 0: gate weight()                                         -> out loc[0]
weights      [replay]       inputs 0: repeat 6 weight_cell()                                -> out loc[0:6:1]
mul          [-]            inputs 2: gate mul(in[0:2:1])                                   -> out loc[0]
add          [-]            inputs 2: gate add(in[0:2:1])                                   -> out loc[0]
dot3         [verification] inputs 6: repeat 3 mul(in[0:2:3] /j+1); repeat 1 add(loc[0:2:1] /j+2); gate add(loc[3], loc[2])   -> out loc[4]
row          [replay]       inputs 9: repeat 2 dot3(in[0:3:1] /j+0, in[3:3:2] /j+1)          -> out loc[0:2:1]
batch        [-]            inputs 0: call activations(); call weights(); repeat 2 row(loc[0:3:1] /j+3, loc[6:6:1] /j+0)   -> out loc[12:4:1]
~~~

Notation: `in[start:count:stride]`, `loc[start:count:stride]`, `/j+s` is the per-copy `jstride`. Read: `dot3` multiplies $x_i w_i$ for $i < 3$ (each copy reads ports $j$ and $3+j$ via stride 3), adds the first pair, then adds the carry; `row` computes two dot products, copy $j$ reading the whole $x$ (`jstride 0`, broadcast) and column $j$ of $W$ (`in[3:3:2]` shifted by $j$: elements $3+j, 5+j, 7+j$ — row-major $W$ with 2 columns); `batch` lays out the sources and then two rows, copy $j$ reading activation row $j$ (`loc[0:3:1]` shifted by $3j$) and all of $W$ (`loc[6:6:1]`, broadcast). Compiled: $n = 32$ (6 `in` + 6 `weight` + 2 rows × 2 dots × 5 gates), 4 replay units, 16 verification units, $|\partial| = 10$ (6 inputs + 4 dot outputs; the weights are under $\kappa_W$, not in $\partial$), outputs at addresses 16, 21, 26, 31.

The same nine definitions describe $4096^3$ (4 293 B): only the counts and strides change, plus $\log_2 k$ sum-tree levels in `dot`. Raw JSON for the small case is in `/tmp/s7` output and reproduced in §10.4.

### 4.4 The compiler

`Compiler(gate_set, limits).compile(description: bytes, inputs) -> Compiled` (`src/veritor/compile/compiler.py:19-60`). Docstring: *"``compile`` parses and validates the description (canonical encoding, arity, in-range relative references, dependency order, limits, a root without ports), summarizes every definition once, checks the role marks, and builds the lazy circuit and index. Only the input *count* is checked here, against the number of ``in`` gates (values are checked when they are encoded); weight values are not a compile input, they enter through the protocol's ``Weights``. Neither is the advice: shaping a description by the input or the advice is the constructor's job, done before the bytes reach the compiler (:func:`veritor.research.Compile` runs it)."*

Steps: `parse_description` → `Index(root, limits)` (runs `validate_marks`, builds the lazy unit structure) → `DescriptionCircuit(root, gate_set)` → check `len(inputs) == index.input_count` → `Compiled(circuit, index, Compiled.digest_of(parsed.digest, gate_set.digest))`.

Descent, in the outline's words and the code's: *"copies own contiguous intervals; `repeat` descends by division; step lists by bisection; ∂ rank is a prefix sum of per-kind interface widths, the same mechanism as addresses are prefix sums of size."* Code: `Frame.child(index, copy)` (`description.py:825`), `Frame.interval` (`:822`), `Frame.source_before(source)` (`:847`, prefix sums of source gates per step for input/weight rank); `Units` (`index.py:133`), `_Sources` (`:417`), `_Boundary` (`:474`).

What the per-kind table is *for*: *"profiling by kind: Bound, W and the cost model of §6 are folds over this table weighted by copy counts, and none enumerates a copy"* (draft line 142; code: `KindTable` docstring `index.py:229-236`, `analysis/bound.py` folds, `parameters.py::expected_work`). `veritor.evaluation.serving.serving_table` writes such a table *from a model of the circuit* so a 70B shape can be priced without ever being described (`serving.py:1-12`).

### 4.5 Marks and units, operationally

How a developer gets "one replay unit per request" (author L1318: *"suppose I'm a developer and I want to run some inference ... if I want to have my replay units be like ... inference requests or something. How can I make sure that that happens? And like what do I choose my verification unit?"*): mark the request function's definition `role="replay"` and the dot-product / attention-head / one-hot / argmax functions `role="verification"`; the tracer records the roles (`tracer.py::definition(..., role=)`), the compiler checks tiling, `Index.replay_units` enumerates the copies. `RequestsG` does exactly this (`src/veritor/constructors/requests.py`: one replay unit per request, no advice); `ClusterG` marks each occupied *step* of a pod as the replay unit (`cluster.py`), because under continual batching a request's tokens are spread over steps that also serve other requests. The tracer refuses source gates outside a replay unit or inside a verification unit (`tests/veritor/constructors/test_tracer.py:63`) and emits sources as one `repeat` of a canonical one-gate verification cell (`:17,124`).

### 4.6 The tracer (untrusted)

`src/veritor/constructors/tracer.py`: `Tracer(gate_set)`; `tracer.gate(name)` returns a callable emitting a gate step; `@tracer.definition(input_count=, key=, role=)` wraps a Python function that receives `Wires` (a symbolic strided range) and returns wires; the body runs *once* per distinct `key` and is hash-consed by the digest of its serialized body; `tracer.repeat(count, definition, *jranges)` emits a repeat step; `tracer.inputs(n)` / `tracer.weights(n)` emit `n` source gates as a repeat of a canonical cell (`tracer.py:sources`); `tracer.serialize(root)` emits canonical bytes in dependency order. Python `for` loops unroll into steps; only an explicit `repeat` keeps $G$ small (this is why the description size of `ClusterG` grows with the number of occupied slots per step: each step's slot list is a step list). The outline's sentence: *"JAX/PyTorch front ends live here."* Nothing about the tracer is trusted: *"the compiler re-validates every byte, and a tracer bug is a different fixed circuit"* (draft line 146). The tracer materializes `Wires` slices, which is why tracing a real GPT-2 or 70B model would not finish; `serving_table` exists for that.

### 4.7 Parameters (PENDING) and the current substitute

**Outline 7.4 "Parameters".** *"Integer parameters in `repeat` counts, strides, and definition selectors, bound from shape(x) (public) and a (charged). Elaboration, in the hardware sense. Structural dependence on x's values is excluded by design; anything that needs it (MoE routing) is advice."*

**Code.** No parameters. `count`, `start`, `stride`, `jstride` are literal integers in the bytes; digests are literal. The constructor $G$ emits a *closed* description per run, shaped by whatever it reads of $x$ and $a$: `MatmulG.__call__` uses `x.activation_shapes`, `x.weight_shape` (`matmul.py:263-273`); `ClusterG.__call__(requests, a)` decodes `a` as `Schedule.encode()` bytes — which request occupies which slot at which step — and lays the description out accordingly (`cluster.py`); `RequestsG.__call__(requests, b"")` uses the requests' prompt lengths and `max_new`. Advice is charged: `Capacity = Bound + 8|a|`; for the default cluster fixture $a$ is 100 B = 800 bits (measured; `Schedule.encode()` header `veritor/schedule/v1` plus fixed-width fields).

**Status.** `docs/overnight-report.md` §3 PENDING list: *"Phase 3: canonical chunking and parametric descriptions"*; the draft says *"The design, not yet in the prototype"* (line 144). See Discrepancy §9.1.

### 4.8 Prior art, with verification status

Collected from the transcript (L1272–L1286, the author's *"is there some design pattern we can refer to ... I'm looking for design patterns that we can appeal to it's not just like 'Claude did it'"*) and the draft (line 148). Status column: *verified* means the reference is standard and its relevance is correctly stated; *to-check* means the writing agent should confirm the citation.

| Reference | Used for | Status |
|---|---|---|
| Lengauer & Wanke, "Efficient solution of connectivity problems on hierarchically defined graphs" (SIAM J. Comput. 1988) and Lengauer's "Hierarchical planarity testing algorithms" (JACM 1989); the general "bottom-up method" for hierarchically defined graphs | The description is a hierarchical graph definition (cells calling cells); the per-kind table and Bound are the bottom-up method applied to interfaces, costs and covers; problems solved in time polynomial in the description rather than its expansion | verified as to existence and relevance; exact titles/years to-check |
| Hash-consing: Ershov 1958 ("On programming of arithmetic operations"), Goto 1974 (Lisp), Filliâtre & Conchon 2006 ("Type-safe modular hash-consing") | Identical definitions are one definition named by its digest | verified (standard); pick one or two |
| Content-addressed storage / Merkle DAGs (Merkle 1979; Git) | Same as above, for the "named by digest" framing | verified; optional |
| Position-independent code / relative addressing | Arguments in coordinates relative to the enclosing definition, so a copy is a translation of its definition | verified (folk); no citation needed |
| Verilog/VHDL `generate` loops; the polyhedral model's affine loops (Feautrier 1991) | `repeat` with a per-copy stride | verified as analogies; Feautrier to-check if cited |
| Staging / tracing: JAX (Bradbury et al. 2018; `jax.make_jaxpr`), tracing JITs | The tracer records the call structure by running the program once over symbolic values | verified; cite the JAX repo/whitepaper |
| LCF architecture (Milner 1972; Gordon–Milner–Wadsworth 1979) and the de Bruijn criterion | Small trusted checker (Σ plus a validator for three constructs), arbitrary untrusted producers (constructors, tracers, front ends) | verified; the transcript's assistant used both terms at L1283 |
| Tensor IRs (XLA HLO/StableHLO, MLIR, TVM Relay) and flat circuit formats (Bristol Fashion for MPC; R1CS/PLONKish for SNARKs) | Why nothing off the shelf: tensor IRs need a trusted per-op lowering to the scalar relation the verifier checks; circuit formats are scalar-exact but flat, size linear in gates | verified as characterization; no specific citation needed unless the author wants one |
| Hierarchical netlists (EDIF, structural Verilog) | The closest existing artifact: a hierarchical netlist with parametrized cells; lacks canonical bytes and small-decoder validation | verified as analogy |

The assistant's one-sentence version (L1281), which the author quoted back at L1282 while asking for a design pattern to appeal to: *"The verifier trusts a fixed set of scalar gates and a reader for a three-construct netlist format (definition, call, repeat). It never runs client code and never sees a gate list."* The author's reaction: *"I don't know if people know what a netlist is."* Recommendation: say "hierarchically defined graph" in the prior-art paragraph and avoid "netlist" and "IR" in the section proper (the draft's house-rule audit already bans "IR").

### 4.9 The rationale paragraph (7.4's last paragraph, per the outline)

Why roll our own and why this shape, in the author's and assistant's words:

- Author (L1274): *"it smells that we're defining our own intermediate representation. I mean, I guess we are trying to represent all computation or like we're trying to create a shared set of foundational semantics. Should we be using something off the shelf here? Or should we roll our own?"* Assistant (L1280): off-the-shelf tensor IRs make the verifier trust a lowering; flat circuit formats are the size of the circuit; the intersection (hierarchy-preserving, scalar-exact, small decoder) is empty, so a three-construct format with a validator is the minimal thing.
- Author (L1276): *"is there any way that the hierarchy can be separated? That we can just define like, like that the IR itself doesn't need to own the hierarchy. Like I was kind of imagining that the kernel would own the hierarchy and G would only ever work at the lowest level."* Assistant (L1283): the hierarchy *is* the compression; a flat $G$ is $\Theta(n)$ and nothing sublinear can be derived from it; the client must author the hierarchy, and the verifier validates it.
- Author (L1278): *"So, is your assessment that in order to do this we have to build a new hierarchical IR that Compile() knows how to read?"* Assistant: yes, and it is small: three constructs.
- Author (L1280): *"Is there like a short story we can tell about how the compiler works? Like is there any like universality statement or something like this? Like could we demonstrate that if we agree on expressive enough gates, this can express anything?"* → §5.1 below.

---

## 5. §7.5 Expressiveness and the cost of the envelope

### 5.1 What is expressible (the universality story)

The author asked for it (L1280): *"Is there like a short story we can tell about how the compiler works? Like is there any like universality statement or something like this? Like could we demonstrate that if we agree on expressive enough gates, this can express anything?"* The assistant's three tiers (L1348 item 11; the draft's 7.5 first paragraph):

1. **Any function on fixed-width inputs**, over a complete gate set, has a circuit, hence a description (flat at worst: one `gate` step per gate). $\{+, \times\}$ over $\mathbb{Z}_{2^B}$ alone is *not* complete: its circuits compute polynomial functions modulo $2^B$, and for $B \ge 2$ not every function $\mathbb{Z}_{2^B}^k \to \mathbb{Z}_{2^B}$ is a polynomial (standard; e.g. the indicator of $0$ is not a polynomial mod 4). The toy ISA adds `sub`, `lt`, `eq`, `shr` (`make_isa_gate_set`, `gates.py:268-300`, name `veritor.toy-isa`, version 1); `lt`/`eq` give comparison and, with `mul` as AND-with-a-bit and `sub`, selection; `shr` gives bit extraction. With those, any Boolean function is expressible bit by bit, hence any function on fixed-width inputs.
2. **Any bounded computation**: `repeat` $T$ of a step circuit whose memory is a bank of values selected by multiplexers, i.e. the zkVM reduction, at the zkVM price (each step touches every memory cell it *might* touch). The prototype does not do this; it is a statement, not a constructor.
3. **Inference, natively**: after shape specialization a transformer forward pass is a static dataflow graph with affine indexing: products, elementwise maps, reductions and permutations, each a `repeat` over rows, columns, heads or positions, and layers are copies of one kind because their weights flow in as source gates by rank, not as immediates. Evidence: `ToyLM` (`lm.py`) expresses embedding, attention with a KV cache, an MLP, an LM head and an argmax entirely in `gate`/`call`/`repeat` with `repeat` over `dot`, `square`, `shr`, `eq`; the serving table at GPT-2 and 70B shapes has 521–3 092 kinds (§8.3).

Gate sets in the repo: `make_word_gate_set(width)` (`add`, `mul`, `in`, `weight`; `gates.py:232-265`, named `veritor.word-arithmetic` in code — do not carry "word" into the paper) used by `MatmulG`/`DemoG`; `make_isa_gate_set(width)` (`add`, `sub`, `mul`, `lt`, `eq`, `shr`, `in`, `weight`) used by the LM constructors. All gates of a set share one output `width` (per-gate widths are a limitation, §6).

### 5.2 Dependent reads, the $V$-way multiplexer, and "carving the world at its hinges" (quotes)

The assistant had said the description could not express "dependent reads" (a gate whose *argument address* depends on a value) and proposed alternatives. The author (L1262):

> No, no, no. I don't think we can have dependent reads. I'm happy to... Wait, sorry. What do you mean V-Way multiplexer? I would have thought that we could just have no ops for the gates that aren't chosen, but then still just commit to them and sample them. Nope, I think we do want per gate output width. Whatever, fuck it. Yeah, let's not have that quite yet. That's fine. And then let's also just pad the length. Why not? Can you think carefully? Are we carving the world at its hinges here? It's a bit unclear to me. what our ontology is for representing stuff in compiler speak. It seems like we have a bunch of different concepts. We have definitions, and there's the kernel, and there's load. Can you just present to me your current ontology here?

The assistant (L1263):

> Agreed on all three: no dependent reads, pad lengths, per-gate widths later. And you're right about the multiplexer; I overstated it. A "$V$-way mux" is exactly what you described: for each candidate $v$, a gate computing $[id = v]$ and a masked copy of $E[v]$, summed. The unchosen ones are zeros, committed and sampled like anything else. Cost is $O(V \cdot d)$ per token, the same order as the unembedding matmul that's already there: roughly 20% on GPT-2 small, 5% at frontier shapes. Fine. The one place the envelope genuinely hurts is MoE, where computing all $E$ experts and masking is $E/k$ on the FFN; that's the case where routing goes through advice instead, which is your existing Pareto analysis. No new mechanism either way.

So the *author's* formulation ("no-ops for the gates that aren't chosen, committed and sampled like anything else") is the envelope, and the "$V$-way multiplexer" is the same thing under a hardware name. The "5–20%" figure is the assistant's back-of-envelope estimate ($Vd$ against the unembedding $dV$ plus the rest of a layer stack): **unsourced beyond that estimate**; see §8.

The "hinges" assessment, in the assistant's ontology critique (L1263, "Where it's not at the hinges"): the kernel conflated the gate set, the decoder and the query engine; one concept had four names (occurrence, unit, plan, partition); the index lived in two packages; the boundary was stored rather than derived; seven verbs for one act. The author's response was the sequence of ontology questions (L1264–L1316) that produced the shipped ontology (§7 below). Later (L1348 item 4) the assistant's verdict on expressiveness: *"Hierarchy is compression and can't be removed, but the protocol's hierarchy is separable from $G$'s factoring: units are nodes, chosen by marks, priced by Bound. $G$'s factoring determines only which cuts are cheap to name."* And item 1: *"No dependent reads. The padded envelope handles everything inference needs ... My earlier 'not viable' was wrong."*

### 5.3 The envelope costs for inference, item by item, with what the code does

- **Embedding lookup.** Code: `ToyLM.embed_row` (`lm.py:511-519`) = `onehot(token, constants)` (a `repeat vocab` of `eq(token, k)` against a constant table $0..V-1$ that lives under the *weight* commitment, since the grammar has no immediates; `lm.py:16-20`) followed by `matvec(vocab, d)` (the one-hot times $E$). This is exactly the $V$-way select: $V$ `eq` gates plus $Vd$ `mul` and $\approx Vd$ `add` per token. The unembedding is $dV$ `mul` plus adds, so the embedding at most doubles that layer (the draft's sentence, line 156). For GPT-2 small ($V = 50\,257$, $d = 768$): $\approx 3.9 \times 10^7$ multiply-adds per token for the embedding against $\approx 12 \times (4d^2 + 2 \cdot d \cdot 4d) \approx 8.5 \times 10^7$ for the layers plus $3.9 \times 10^7$ for the unembedding, so about $+24\%$ on a GPT-2-small-shaped forward pass at short context (this morning's arithmetic; consistent with "roughly 20% on GPT-2 small"). At $d = 8192$, $V = 128\,256$, 80 layers: $1.05 \times 10^9$ vs $80 \times (4 \cdot 8192^2 + 8 \cdot 8192^2) \approx 6.4 \times 10^{10}$, about $+1.6\%$ (the frontier figure of "5%" is generous). **Estimate, not measured on a compiled model.**
- **Nonlinearities.** The toy has none of the real ones: softmax is $w_j = s_j^2$ (no division), no layer norm, attention output shifted right by a constant (`lm.py:3-10`). For real models the choice is fixed-point polynomial approximations over `add`/`mul`, or new gates of Σ with a specified exact relation (GELU, exp, rsqrt with specified rounding); the draft (line 156) and outline say so. Pending on the value-semantics decision (§6).
- **Variable sequence length.** Padding to a maximum with masks as multiplication by 0/1 (author L1262: *"let's also just pad the length. Why not?"*). Code: `ClusterG` pads each pod to `slots` occupants and `steps` steps; an unoccupied slot is a `hole` with gates that compute on padding values; a request's tokens are read at the positions the schedule says (`cluster.py`). The toy's attention over a KV cache at context $c$ is a distinct kind `attend_head_c` for each $c$ (so kinds grow with the context length: 521 kinds for 100+100 tokens, 2 581 for 512+512 in the serving table), which is the cost of static shapes without parameters; `repeat` over positions with a padded context would collapse those.
- **Mixture of experts.** Two sound options, quantified only as formulas: evaluate all $E$ experts and mask (cost $E/k$ on the FFN blocks; e.g. $256/8 = 32\times$ on the FFN for a DeepSeek-style router, assistant L1263 reasoning), or route through advice at $\approx k \log_2 E$ bits per token per layer charged to $A$. The outline calls this *"the one real design trade"* and asks for *"the advice-versus-overhead frontier"*. **No MoE constructor exists**; nothing is measured. The "Pareto analysis" the assistant refers to is the author's earlier zero-advice-vs-advice analysis in the old §7 of the Notion draft (`/tmp/notion-outline.md` §4.2.1 region; not reproducible from the repo). Recommendation: state the two costs as formulas with the DeepSeek-V3 numbers ($E = 256$, $k = 8$; to-check) and defer a measured point to §10.
- **Continual batching (the case advice is needed) versus one unit per request (no advice).** `ClusterG` needs advice ($a$ = `Schedule.encode()`: which request sits in which slot of which pod at which step; 100 B for the default fixture) because the *structure* of a step depends on the schedule, which is the server's choice; `RequestsG` needs none because the structure depends only on prompt lengths and `max_new`, which are the request's (public $x$). The author's latest preference (from the frontier work; `frontier-report.md:5`): *"price what an honest inference server naturally achieves under the protocol, so that the verifier's maximum admissible capacity $U_{\max}$ can be calibrated to it rather than dictated."* The frontier finding that bears on §7.5's "which marks": replay units whose declared interface is wider than the circuit's output (a batched decode `step` exposing KV-cache rows; a `layer`; a `matvec`) make Bound vacuous under the current cut-width fold; `request` units (interface = the request's own tokens) and `cell` units (one value) do not (`frontier-report.md:35, 7`; `overnight-report.md:256-270`). Until the fold is reach-aware (`frontier-report.md:220`), §7.5 should say that the constructor's factoring decides which cuts are *cheap to name*, and Bound decides which are *worth naming*.

### 5.4 Compile-cost table (what can be measured today)

Measured this morning (single-threaded Python; laptop; minimum of one run unless stated; scripts `/tmp/s7/mm.py`, `/tmp/s7/lm.py`, `/tmp/s7/eight.py`):

| circuit | gates | $|G|$ bytes | definitions (kinds) | trace (client) | Compile (verifier) | `kinds()` | `C[i]` |
|---|---|---|---|---|---|---|---|
| 8-gate `workload`, $B = 16$ | 8 | 2 040 | 6 | — | 0.38 ms | — | — |
| matmul $2 \times 3 \cdot 3 \times 2$, $B = 8$ | 32 | 2 773 | 9 | — | 0.5 ms | — | — |
| matmul $8^3$ | 1 152 | 2 886 | 9 | 6 ms | 0.5 ms | 0.06 ms | 15 µs |
| matmul $64^3$ | 528 384 | 3 409 | 9 | 7 ms | 0.6 ms | 0.05 ms | 20 µs |
| matmul $256^3$ | $3.36 \times 10^7$ | 3 710 | 9 | 18 ms | 0.6 ms | 0.05 ms | 24 µs |
| matmul $1024^3$ | $2.15 \times 10^9$ | 4 010 | 9 | 61 ms | 0.5–14 ms | 0.06 ms | 26 µs |
| matmul $4096^3$ | $1.37 \times 10^{11}$ | 4 293 | 9 | 234 ms | 1.0 ms | 0.09 ms | 28 µs |
| toy LM, `RequestsG`, 4 requests | 6 564 | 33 731 | 40 | 3.1 ms | 3.9 ms | — | — |
| toy LM, `ClusterG` 2×2×6, 4 requests | 6 564 | 34 507 | 42 | 3.1 ms | 4.3 ms | — | — |
| toy LM, `ClusterG` 2×4×12, 8 requests, 2 layers | 94 956 | 64 465 | 46 | 5.8 ms | 8.0 ms | — | — |

Gate counts at scale come from `veritor.evaluation.serving.serving_table` (a model of the toy decoder's kinds at arbitrary dimensions, *not* a traced description; the tracer would not finish):

| shape (toy decoder structure) | gates $n$ | kinds (`request/row`) | kinds (`step/row`) | inputs | weights | table time |
|---|---|---|---|---|---|---|
| GPT-2-small dims, 1 request, 100 + 100 tokens | $4.65 \times 10^{10}$ | 521 | 620 | 100 | $1.34 \times 10^8$ | 5 ms |
| GPT-2-small dims, 8 requests, 512 + 512 | $2.03 \times 10^{12}$ | 2 581 | 3 092 | 4 096 | $1.34 \times 10^8$ | 26–32 ms |
| 70B-class ($d$ 8192, 80 L, 64 h, $V$ 128k), 2 048 req, 512 + 512, batch 256 | $1.89 \times 10^{17}$ | 2 581 | 3 092 | $1.05 \times 10^6$ | $4.5 \times 10^{10}$ | 30 ms |
| `FRONTIER_SHAPE` ($V$ 32 768, hidden $4d$, batch 32) (`frontier.py:63-74`) | $2.7 \times 10^{17}$ | — | — | — | — | (report) |

**Description size at scale: estimate.** The per-kind byte cost of the traced toy descriptions is 820–1 400 B/kind (34 507 B / 42 kinds; 64 465 B / 46 kinds; the larger figure because `ClusterG` step definitions list one call per occupied slot). At 521 kinds (GPT-2-small dims, 100+100 tokens) that is $\approx 0.4$–$0.7$ MB; at 2 581 kinds $\approx 2$–$3.6$ MB. A *single forward pass* at fixed context (one `attend_head_c`, no per-context kinds) is a few dozen kinds, i.e. tens of kilobytes. The outline's *"kilobytes for a GPT-2 forward pass against $10^{11}$ gates"* is right for one forward pass and an order of magnitude low for a 100-token decode under today's per-context kinds; with `repeat` over padded positions or with parametric descriptions the decode would also be kilobytes. Flag as estimate.

---

## 6. §7.6 Limitations (each with status)

1. **Parametric descriptions.** Not implemented (Phase 3 PENDING, `overnight-report.md` §3). Today: the constructor emits a closed description per run; the paper's design has integer parameters in `repeat` counts, strides, definition selectors, bound from $\mathrm{shape}(x)$ and $a$ before parsing; *beyond* integer parameters (e.g. structure depending on values of $x$) is excluded by design and goes through advice.
2. **Per-gate output widths.** One `width` per gate set (`GateSet(gates, name, version)`, every `Gate(name, arity, width, ...)` constructed with the set's width; `gates.py:232-300`). The author wants per-gate widths eventually (L1262: *"Nope, I think we do want per gate output width. Whatever, fuck it. Yeah, let's not have that quite yet."*). The §5 draft writes value sets $V_i$ with widths $w_i$ per gate; the code's `out_bits = out_count × width`. Mixed-precision inference (FP16 boundaries, FP32 accumulators in the §5 draft's `vllm-fp16-reference` profile) needs this. Changing it touches the value codec and `out_bits`, not the grammar (draft line 171).
3. **Canonical chunking of long step lists.** Not implemented (Phase 3). Today a unit boundary must be a definition boundary, so the constructor's factoring is load-bearing for granularity; the transcript's fix (L1317) was for Compile to build a canonical balanced chunking of long step lists so every scale exists in $I$ and a cut can fall inside a definition without the client's cooperation.
4. **Transcendental gates with specified rounding.** Not present; the toy ISA has `add sub mul lt eq shr` over $\mathbb{Z}_{2^{16}}$. Would shrink the nonlinearity envelope at the cost of a larger Σ and a relation the verifier checks exactly.
5. **Value semantics: undecided.** `overnight-report.md:293-295` discrepancy 10: *"Everything is modular word arithmetic; the LLM constructors that need fixed-point or float semantics wait on that decision."* The §5 draft's profile (FP16 boundary values, FP32 accumulators, named `vllm-fp16-reference`) is not what the toy computes. The PENDING list: *"llm-constructors: GPT-2 dense first; Kimi-K3 / DeepSeek-V4-Pro / Inkling after the value-semantics decision."*
6. **The verifier executes $G$.** §2.3. Deployment: sandboxed and metered $G$, or a proof of compilation. No proof-of-compilation prototype exists.
7. **Determinism across implementations.** Rests on Python's `json.dumps(sort_keys=True)`; no cross-implementation test vectors (`security-argument.md:594-596`).
8. **Distinctness check is quadratic in runs.** Bounded by `max_output_runs = 256`; a legitimate definition with more than 256 output runs (e.g. an oddly strided interface) is rejected, a completeness limit.
9. **Kinds per context length.** Without `repeat` over padded positions or parameters, attention at each KV length is a distinct kind, so $|G|$ grows linearly with the generated length (§5.4).
10. **Dead limits.** `VerificationLimits.max_nesting_depth` and `max_artifact_bytes` are declared but unenforced (F6, `security-argument.md:812-817`); not a §7 claim, but do not cite them as protections.
11. **Cost labels are conventions.** `replay_cost`/`proof_cost` per gate are the gate set's, not measured; Cost and $W$ are as good as the labels.
12. **Tracer ergonomics** (PENDING): 2-D ranges, a concatenation primitive; today `Wires` slicing materializes, which bounds what can be traced.

---

## 7. Ontology dump

### 7.1 The final restated ontology (assistant L1348, after the author's L1347 "state the whole fresh ontology")

Reproduced verbatim (LaTeX kept as the transcript had it):

> **Trusted.** The verifier and the channel. Nothing else: the developer, constructor, marks, proposed parameters, prover, and serving stack are all the adversary.
>
> **Public parameters.** Gate set $\Sigma$: scalar primitives, each with arity, output width, exact relation, and two cost labels (replay, proof). The only semantics in the trusted base. $U_{\max}$: the exfiltration bound the verifier will accept. $W_{\max}$: the verifier's own work budget.
>
> **From the client, before the run.** $G$: the description. A hash-consed sequence of definitions, each a straight-line list of `gate` / `call` / `repeat` steps with relative, range-typed arguments; some definitions marked *replay unit* or *verification unit*. $x$: inputs (weights included, under their own commitment).
>
> **From the client, at the start of the run.** $y^*$; advice $a$ with $|a| \le A$; proposed $\theta = (q, s)$.
>
> **Compile$(G, x, a) \to (C, I)$.** Trusted, deterministic, $O(|G|)$. Rejects malformed $G$ or invalid marks.
> - $C$: the circuit. `C[i]`; $\mathrm{In}, \mathrm{Out}, \mathrm{Size}, \mathrm{Cost}$ on subcircuits; semantics from $\Sigma$. No decomposition of its own.
> - $I$: the hierarchy of nested subcircuits over $C$, leaves = addresses, nodes tagged by kind (same kind ⇒ same structure, Size, Cost, Out). The marks designate two antichains: replay units partition the gates, verification units refine them. Checks, per definition: replay marks tile; verification marks tile each replay unit; verification kinds under the proof-cost cap.
> - Circuit commitment $H(C, I)$. Notation $\partial := x \cup y \cup \bigcup_r \mathrm{Out}(R_r)$, $\mathrm{Int}(r) := R_r \setminus \partial$.
>
> **How each client input is handled.** $G$: checked, everything derived from it. Marks and $\theta$: checked, then *priced*. $x, y^*$: pinned. $a$: charged. Transcript: sampled.
>
> **Verbs.**
> - **Bound**$(C, I, \theta) \to U$: fold over $I$ by kind, minimum over every node as a cut. Run before the run; proceed iff $U \le U_{\max}$ and verifier work $\le W_{\max}$.
> - **Verify**$(\tau; C, I, \theta)$: commit $\partial$; sample replay units at $q$; commit interiors; sample verification units at $s$; check `C[i]` relations on opened values. Reads $C$ and the two antichains only.
> - Cost is a helper (sum of labels). Optimize is not a method: it's the client's search over marks and $\theta$, untrusted, with Bound as its oracle.
>
> **Untrusted tooling.** Constructor code, the tracer that records call structure and marks into $G$, front ends from JAX or PyTorch. Unbelieved rather than trusted.

Two later corrections to this text: (i) $\eta$ is the verifier's and travels in the header; $\theta = (q, s)$ only is the client's proposal (`overnight-report.md:275-277`); (ii) $\partial$ is $\mathrm{In} \cup \bigcup_r \mathrm{Out}(R_r)$ with In the `in` gates, weights excluded (under $\kappa_W$), and $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r) \setminus \mathrm{pinned}(R_r)$ (`overnight-report.md:240-244`).

### 7.2 The author's ontology questions, in order, and the answers that stuck

| turn | author's question (quote) | resolution |
|---|---|---|
| L1264 | *"I think there should be: 1. Public gate set 2. Kernel 3. A Circuit object (this is what implements gate at etc. Pause: what is an occurrence summary? I would have thought that the idea was that these were essentially all just quick ways to compute properties of the circuit."* | Occurrence summaries, replay plans, boundary derivation were all implementations of In/Out/Size/Cost (L1348 item 5). `Circuit` protocol = `C[i]` plus `In`, `Out`, `Size`, `Cost` (`circuit.py:89`). |
| L1266 | *"Is that not an implementation of In() and Out() and Size() and Cost()?"* | Yes. |
| L1268 | *"it should also probably be C[i] (that's how we access the gate in the paper). Sorry, can you walk me through children a bit better? It feels like this is not a standard property of the circuit. It's like what saying there's like a canonical decomposition?"* | `children` is not a property of a circuit; it belongs to $I$ (L1348 item 5). Code: `KindSummary.children` on the index, `Circuit` has none. |
| L1270 | *"What the kernel loaded. Sorry, can you give me more of a sense for what description is concretely and how it's used so I can understand its place and whether it earns its keep?"* | The description is the compressed circuit; it is $G$ (bytes). Not a named paper object. |
| L1272 | *"This is basically like... an IR? This is like jaxpr (it looks like it)? It's like the compiler's internal IR?"* | Yes: hash-consed ANF with `call` and `repeat` over scalar cells; Lengauer–Wanke hierarchically defined graphs (L1348 item 2). Do not say "IR" in the paper. |
| L1274, L1276, L1278 | roll our own vs off the shelf; can the hierarchy be separated from the IR; must Compile read a new hierarchical IR? | Hierarchy is compression and cannot be removed; nothing off the shelf; three constructs (L1348 items 2, 4). |
| L1282, L1284 | design pattern / appeal to authority, *"not just like 'Claude did it'"* | LCF / de Bruijn small checker; Lengauer–Wanke; hash-consing; `generate`; staging (§4.8). |
| L1286 | *"state our ontology again ... carving the world at its hinges"* | L1287 restatement (kernel, D, address set, ρ, Instance still present). |
| L1294 | *"1. what's an instance? 2. epoch I think is like one run of the protocol ... 3. G and x are both public in some sense ... 4. So I think it's not enough for it to be a deterministic function. We need to know something about its structure so that we can efficiently traverse it"* | Instance dropped; *"Compile needs a traversal contract, not just determinism. DAG of definitions, relative addressing, contiguous copy intervals, bounded depth. Drop any one and something is $O(|C|)$"* (L1348 item 7). |
| L1296 | *"remind me what else needs D, except insofar as it holds C and I?"* | Nothing; $D$ dropped, $H(C, I)$ is a footnote about canonical serialization (L1317). |
| L1298 | *"Could we get the nested cuts thing via some sort of witness as opposed to like A circuit has no notion of same sub-circuit ... Or is this the cleanest way?"* | $I$ is one hierarchy with two special antichains; Bound needs nested cuts and sharing, both properties of $I$ (L1348 item 6). |
| L1304 | *"What is an anti-chain in general and in this context?"* | A set of nodes of the hierarchy none of which contains another; the replay units and the verification units are each an antichain that covers the leaves. |
| L1306 | *"Also, what is cost doing for us?"* | Cost is a label (two integers per gate in Σ) used by the cost model and $W$, unused by Bound (L1348 item 9). |
| L1310, L1312 | is the boundary a separate object; do we need that notation? | $\partial$ is a formula, not an object; Int is notation (L1350). |
| L1316 | *"Why do we need 'kernel' or `load` -- why isn't Compile() fine? Also let's say from the client before the run and from the client after the run. It seems like we can drop D and instead just do H(C, I) or something? I'm not sure what address set is. ... I'm not sure we need Instance but could be wrong. How is ρ chosen by optimize but then used in compile() -- temporally seems off, no?"* | All six accepted (L1317): kernel/load dropped (Compile is the function; `Compiler` is the code object); before/after adopted; $D$ dropped; address sets are return types of In/Out, an implementation requirement, not ontology; Instance dropped; ρ removed from Compile — Compile outputs the full $I$; selection happens after. |
| L1318, L1322 | operational: how does a developer get request-level replay units; *"Hold up, we're gonna let the client pick. ... What if you just pick all your replay units to be up to generate size except for one?"* | Marks in $G$; client picks, verifier prices; audit dispatched (L1323); result L1357: safely priced for every partition (§3.3). |
| L1336, L1340 | is Optimize trusted; *"the developer is the adversary"* | Optimize is not a protocol method (client's search, untrusted); ρ dropped entirely in favor of marks. |
| L1435, L1453 | *"why are inputs outside of every unit?"* / *"I prefer every gate is in a unit"* | Source gates inside units; root without ports (§9.6). |

### 7.3 Term-by-term map to the shipped code, and what the paper must not name

| paper | shipped code | status in paper |
|---|---|---|
| $\Sigma$, gate set | `GateSet` (`core/gates.py`), `Gate(name, arity, width, replay_cost, proof_cost, evaluate, source)`; `make_word_gate_set`, `make_isa_gate_set`; `GateSet.digest` | named object |
| $G$ | (a) in the paper: the description bytes; (b) in the code: the `Constructor` protocol (`compile/constructor.py:23`, `G(x, a) -> (description bytes, inputs)`, `.digest`) *plus* the bytes it emits | named object; say "description $G$"; do not name "constructor" as a paper object — it is "the client's tooling" |
| $C$ | `Circuit` protocol (`core/circuit.py:89`); `DescriptionCircuit` (lazy), `FlatCircuit` (reference) | named object |
| $I$ | `Index` (`core/index.py:258`) with `replay_units`, `verification_units`, `boundary()`, `interior(r)`, `inputs()`, `weights()`, `kinds()`, `kind_table()` | named object |
| Compile | `research.Compile(G, x, a, gate_set, *, limits, max_advice_bits)` → `Compilation(compiled, constructor_digest, inputs, advice)`; inside it `Compiler(gate_set, limits).compile(bytes, inputs)` → `Compiled(circuit, index, digest)` | named function |
| $H(C, I)$ | `Compiled.digest` | a footnote: hash of the canonical description bytes (marks included) qualified by $H(\Sigma)$ |
| marks | `Definition.role ∈ {None, "replay", "verification"}`; constants `REPLAY`, `VERIFICATION`, `ROLES` (`core/description.py:53-56`); tracer `role=` | "marks" or "marked definitions" |
| replay units $R_r$ | `Index.replay_units: Units` (`unit(k)`, `owner(i)`, `count`) | named |
| verification units $V_{r,k}$ | `Index.verification_units`, `Index.verification_unit(r, k)` | named |
| $\partial$ | `Index.boundary()` (an `IndexedDomain`) | notation |
| $\mathrm{Int}(r)$ | `Index.interior(r)` (`IntervalDifferenceDomain`) | notation |
| In, Out, Size, Cost | `Circuit.In(unit)`, `.Out(unit)`, `.Size(unit)`, `.Cost(unit)`; per definition `Definition.out_runs`, `input_count`, `size`, `replay_cost`, `proof_cost` | functions on subcircuits |
| kinds, per-kind table | `KindSummary`, `KindTable` (`core/index.py:193-256`) | "kinds"; "the per-kind table" |
| $a$, $A$ | advice `bytes`; `max_advice_bits` (`research.Compile` and `VerifierParameters`) | named |
| $\theta = (q, s)$ | `VerificationPolicy(q, s)` (`core/policy.py`); `Header.policy` | named |
| $\eta$ | `VerifierParameters.eta`; `Header.eta` | named, verifier's |
| $U_{\max}$, $W_{\max}$ | `VerifierParameters.max_capacity`, `max_work` | named |
| $U$ | `bound(compiled, policy, eta).bits` (`analysis/bound.py`) | Bound's output |
| $W$ | `expected_work(compiled, policy, io_count)` (`protocol/parameters.py`) | named |
| $\kappa_W$ | `commit_weights(gate_set, values)`; `Header.weights` | named |
| $x$ | `Header.public_inputs`; the values of the `in` gates by rank (`Index.inputs()`) | named |
| $y^*$ | `Header.claimed_outputs`; `Circuit.outputs` addresses | named |
| epoch | no code object; the author: one run of the protocol, or the lifetime of a $(G, \kappa_W)$ commitment; *"it's possible we want to forget about this for now"* (L1294) | avoid or define once |
| **kernel** | deleted (`call_dag.py` removed in `e6e2354`); `Compiler` is the code object | MUST NOT appear |
| **load** | deleted; `parse_description` is the first step of `compile` | MUST NOT appear |
| **description object / D** | `Definition`, `parse_description` exist in code; the paper speaks of $G$ (bytes) and of $H(C, I)$ | MUST NOT be named; say "$G$" or "the description" |
| **instance** | `Compilation` / `Header` in code | MUST NOT appear; say what is public |
| **address set** | `IndexedDomain` protocol (`core/indexed.py`) | implementation detail; the paper says "set of addresses" |
| **ρ / index rule / sampling policy (unit choice)** | gone; marks in $G$ | do not introduce |
| **Optimize** | `research.Optimize` exists as the client's advisory search | not a protocol method; mention once as the client's search with Bound as oracle |
| **occurrence, plan, partition, artifact, plugin, producer** | deleted or renamed (copy, unit, `Index`, `Compiled`, —, `Tracer`) | do not use |
| tracer | `constructors/tracer.py::Tracer` | may be named ("the tracer"), untrusted |
| copy | a definition at a path in the hierarchy (`Frame`) | may be used informally ("copies of a kind") |
| well-typed | `decode_value` strictness (`gates.py`) | for values only |

---

## 8. Numbers, each with its source

### 8.1 Circuit sizes

| number | value | source | status |
|---|---|---|---|
| GPT-2 Small, 100 prompt + 100 generated tokens, real primitive gates (LN, softmax exp/recip, GELU pieces) | $42\,361\,101\,422$ gates; prefill $17\,301\,481\,495$, decode $25\,059\,619\,827$, plus 100 argmax gates; $42\,257\,061\,888$ contraction FLOPs; transformer body with $n$ visible KV entries $= 170\,345\,378 + 37\,584 n$ gates; LM head $77\,144\,495$ | `src/circuit_cut_analysis` (the §5 analysis package), `tests/circuit_cut_analysis/test_gpt2.py:31-34`, `test_gpt2_circuit.py:263`; the §5 draft rounds it to $4.2 \times 10^{10}$ (`section-5-covert-capacity.md:305`) with 82 kinds and 119 wiring rules | measured (by that package; not by the §7 compiler) |
| Toy decoder at GPT-2-small dimensions, 1 request, 100 + 100 | $4.65 \times 10^{10}$ gates, 521 kinds (`request/row`) | `serving_table` this morning | measured (model of the toy's kinds) |
| Toy decoder at GPT-2-small dims, 8 requests, 512 + 512 | $2.03 \times 10^{12}$ | `serving_table` | measured (model) |
| 70B-class, 2 048 requests, 512 + 512, batch 32, $V = 32\,768$, hidden $4d$ (`FRONTIER_SHAPE`) | $2.7 \times 10^{17}$ gates; output interface $16\,777\,216$ bits; honest replay cost $4.117 \times 10^{17}$ | `docs/frontier-report.md:5`; `src/veritor/evaluation/frontier.py:63-74` | measured (model) |
| 70B-class, $V = 128\,256$, hidden $2d$, batch 256 (my variant) | $1.89 \times 10^{17}$ | `serving_table` this morning | measured (model) |
| matmul $4096^3$ | $1.37 \times 10^{11}$ gates in 4 293 B | this morning | measured (compiled) |
| the paper's "$10^{11}$ gates" | — | draft 7.1 constraint 4; outline 7.4 | round figure; matches GPT-2-small-scale decode ($4 \times 10^{10}$) to within a factor 2 |

### 8.2 Description sizes, compile and lookup times

All in §3.4 and §5.4 tables (measured this morning) plus:

| number | source | status |
|---|---|---|
| "kilobytes for a GPT-2 forward pass against $10^{11}$ gates" | outline 7.4 | estimate: tens of KB for one forward pass; 0.4–0.7 MB for a 100-token decode under per-context kinds (§5.4) |
| "$2\,245$ B for $520\,192$ gates at $k = 64$; $2\,834$ B for $2.1 \times 10^9$ at $k = 1024$; compiled in 0.23 and 0.28 ms" | draft Table 7.1 (pre-`in`/`weight`-gates layout; five definitions) | superseded: today 3 409 B / 4 010 B with nine definitions, 0.5–1 ms |
| "per-kind table 10 ms / 0.17 s / 2.8 s" | draft Table 7.1 | superseded: 0.05–0.09 ms after the range-arithmetic fix (`overnight-report.md:44`: 2.84 s → 0.36 ms) |
| "$C[i]$ 4.6–4.7 µs" | draft Table 7.1 | today 15–28 µs (a different, richer `GateRef` and the source-gate layout; still flat in $n$) |
| `kinds()` 1024³: 2.84 s → 0.36 ms | `overnight-report.md:44` | measured then |
| Bound on a $10^8$-gate synthetic index: 1–40 ms | `overnight-report.md:48` | measured then |
| Bound at the 70B frontier: ≈ 5 s per point (588 points, 50.3 min) with `max_buckets = 2^{22}` | `frontier-report.md:5` | measured |
| Sublinear sampling: ≈ 3 ms for $10^9$ candidates (binomial count + Floyd) | transcript L1348 item 10; commit `3e8ffab` | measured then |
| Cluster fixture: 6 564 gates, 8 RUs, 1 065 VUs, 42 kinds, 34.5 KB, 4 ms trace, 4 ms compile, 50 ms full evaluation | `tests/veritor/constructors/test_cluster.py:1-9`; reproduced | measured |
| Audit: 1 536 adversarial (partition, policy) cases, zero soundness violations, relaxation looseness ≤ 8 bits, mean < 0.1; boundary lemma on 2 550 random markings | transcript L1357 | measured then (the partition audit agent) |
| F7 liveness: 480 of 2 325 small cluster shapes failed to compile before the clamp | `security-argument.md:826-828` | measured then |

### 8.3 Overheads and envelope numbers

| number | source | status |
|---|---|---|
| Embedding as $V$-way select: "roughly 20% on GPT-2 small, 5% at frontier shapes"; outline: 5–20% | assistant L1263; outline 7.5 | **unsourced estimate**; my arithmetic gives ≈ 24% for GPT-2-small dims at short context and ≈ 1.6% at $d = 8192$, $V = 128$k (§5.3) |
| MoE envelope $E/k$; DeepSeek-style $256/8 = 32\times$ on the FFN; advice $k \log_2 E$ bits per token per layer | assistant L1263, draft 7.5 | formula; no constructor, no measurement |
| zkVM overhead $\sim 10^{10}\times$ native | outline §4.2 line 62 and line 81, marked `[source]` | unsourced in the outline; the assistant's criticism at L1205 item 4: *"Standard zkVM numbers are $10^4$–$10^6\times$ relative to CPU. $10^{10}$ is defensible only relative to tensor-core throughput. State what it's relative to and cite or measure it."* |
| Attestable 50–100× overhead, fixed-point matmul only | outline line 62, marked `[source]` | unsourced in the outline |
| "16% of the arithmetic spent on hashing at scalar granularity" (GPT-2, $4.2 \times 10^{10}$ values) | the §6 agent's brief (transcript L1349 forwarded text) | unsourced here; belongs to §6 |
| Frontier calibration: `cell/gate` at $q = 1/2, s = 1/8$ certifies 640 bits ($\eta = 1/2$), 4 021 bits ($1/100$), 11 712 bits ($10^{-6}$) at prover overhead 0.896× and verifier work 7.42×; `request/row` 42.6 k / 283 k / 848 k bits; `step`, `layer`, `matvec`, `row` partitions: full 16 Mbit (vacuous) | `frontier-report.md:7` | measured (Laplace-only fold; not reach-aware) |
| Cluster fixture cut widths 144–672 bits vs true output reach 16–80 bits, 160 output bits total | `overnight-report.md:259-262` | measured |

### 8.4 Limits and parameters (defaults in code)

`CompilationLimits`: 10 MB, 100 000 definitions, $10^6$ steps/definition, $2^{63}-1$ addresses/cost, depth 256, proof-cost cap $2^{63}-1$, 256 output runs/definition, 16 384 total (`limits.py:8-24`). `VerificationLimits`: `max_units = 1_000_000`, `max_positions_per_unit = 10_000_000`, `max_probability_denominator_bits = 64`, plus transcript-size limits (`limits.py:26-40`). `VerifierParameters`: `eta` (Fraction), `max_capacity` (required, may be `None` to waive explicitly), `max_advice_bits` (default 0), `max_work` (`parameters.py:36-60`). Advice charge: 8 bits per byte.

---

## 9. Discrepancies and open decisions

Format: **outline says / draft says / code does / author decided / open?** Items 1–14 are §7's own; 15–24 are the `overnight-report.md` discrepancies that touch §7.

**9.1 Parametric descriptions vs per-run constructor.** Outline: $G$ is a description with integer parameters bound from $\mathrm{shape}(x)$ and $a$ at step 7. Draft: same, flagged *"not yet in the prototype"* (line 144) and *"design, not prototype"* (line 180 item 5). Code: no parameters; `G(x, a)` is Python that emits a closed description per run; the verifier runs it. Author: *"G is once per epoch"* (L1217); *"I don't think it's important for G to be fixed before x is seen"* (L1292); *"G and x are both public in some sense"* (L1294). Open: whether the paper presents the parametric design as the protocol (recommended, with a one-sentence prototype caveat) or presents the prototype's per-run constructor. Recommendation: present the parametric design; in 7.6 or a footnote say the prototype has the verifier execute a public, versioned constructor and relies on output limits.

**9.2 Once-per-model / once-per-run split vs `Compile(G, x, a)` per run.** Outline and draft: parse, validate, marks, per-kind table, $H(G)$ once per model; bind parameters, Bound, $W$ per run. Code: everything except the constructor digest and $\kappa_W$ happens per run (`_admit`), because the description is per run; the compiled digest is per run and is bound in the header. Open: same as 9.1. Note that with parametric descriptions the per-kind table would *itself* be parametric (copy counts as functions of the parameters), which is what makes the split possible; today `KindSummary.copies` is an integer.

**9.3 Proof-cost cap per verification kind.** Outline: exists (step 3). Draft: exists (step 3; Prop 7.4). Code: exists, `CompilationLimits.max_verification_unit_proof_cost`, default unbounded; checked in `validate_marks` (`index.py:685-690`); test `test_index.py:426`. Also `VerificationLimits.max_positions_per_unit` at admission. Decided: it is a completeness condition, not soundness (L1350, L1361 step 7). Not open.

**9.4 Marks in the digest; $H(G)$ vs $H(C, I)$.** Outline: *"publishes $H(G)$, which is $H(C, I)$."* Draft: $h_C = H(\textsf{compiled}, H(G), H(\Sigma))$ (line 60). Code: `Compiled.digest = identity_digest("veritor/compiled/v1", {"description": description_digest(bytes), "gate_set": Σ.digest})`; marks are the `role` fields, hence in the bytes, hence in the digest (`test_canonical.py:76`; `test_scaling.py:268`). Precisely: $H(C, I)$ is the hash of the canonical description bytes *including marks*, qualified by the gate set's identity; nothing else ($x$, $a$, $\theta$ are in the header). The draft is right. Not open, but say it precisely: "$H(C, I)$ is, concretely, $H(G)$ qualified by $H(\Sigma)$; by determinism it fixes $(C, I)$."

**9.5 Weights: $\kappa_W$ by rank per model vs "sent once per model".** Outline 7.2: client sends $\kappa_W$ once per model. Draft: "the weights are committed once per model under $\kappa_W$, a pre-committed subtree of §6's boundary tree" (line 138). Code: `commit_weights(gate_set, values)` builds a root over the weight vector by rank with no reference to any circuit; the verifier checks a sampled `weight` gate's value against that root at the gate's rank (`Index.weights().rank(i)`); weights are *not* in $\partial$ and the weight tree is a *separate* root, not a subtree of the boundary tree (`index.py:329-330`; `overnight-report.md:89-101`: "two roots"). Provenance of $\kappa_W$ is by convention (F5; audit §10). Recommendation: the draft's "pre-committed subtree of the boundary tree" should become "a second commitment root, per model, over the weight vector by rank; weights are never boundary positions."

**9.6 Root without ports; source gates inside units; every gate in a unit.** Outline (§6.4 scratch and 6.1): every gate, inputs included, in exactly one replay and one verification unit; $\partial = \mathrm{in}(C) \cup \mathrm{out}(C) \cup \bigcup_r \mathrm{Out}(R_r)$. Draft: inputs occupy $[0, |x|)$ *outside* every unit; Lemma 7.2 (i) says so; note item (2) flags the conflict with the §4 figure. Code: `in`/`weight` are zero-arity gates inside units; the root has no ports (`compile/description.py:435`); `Index.inputs()`/`weights()` are lazy rank/unrank domains (`index.py:286-300`). Author: L1435 asked why inputs were outside; L1453 *"I prefer every gate is in a unit. I want you to try to steer the code in that direction, we might change this later."* Landed (`overnight-report.md` §2.1). Decided in favor of the outline; the draft must be updated (Lemma 7.2 (i), the "Inputs occupy addresses $[0,|x|)$" sentence in 7.4, the "Weights are inputs" paragraph, and the note's item (2)).

**9.7 "Out excludes pinned gates"; "In = declared ports"; declared outputs without Out copy gates.** Outline: `Out` copy gates appear in the §6.4 figures (per `overnight-report.md:240`: "the outline's circuit has In positions outside the units and Out copy gates"). Draft: $\mathrm{Out}(R_r)$ = declared interface resolved to owned gates, pass-through outputs excluded (note item (1)). Code: `Out` = resolved declared outputs minus source gates (pinned) minus `PORT` pieces; `In` = the definition's declared `input_count` ports, not what is read (`test_out_runs.py:339`); no copy gates; the root's declared outputs *are* the circuit's outputs (`Circuit.outputs`). Decided by the code and `overnight-report.md` discrepancy 1's recommendation. The writing agent should remove Out copy gates from any figure and state $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r) \setminus \mathrm{pinned}(R_r)$.

**9.8 The boundary formula.** Outline/§6: $\partial = \mathrm{in}(C) \cup \mathrm{out}(C) \cup \bigcup_r \mathrm{Out}(R_r)$. Draft: $\partial = \mathrm{in}(C) \cup \bigcup_r \mathrm{Out}(R_r)$ (outputs are inside the union). Code: `In ∪ ⋃_r Out(R_r)`, outputs always inside (`index.py:325-332`), weights excluded. Recommendation: use the draft's/code's form and remark that $\mathrm{out}(C) \subseteq \bigcup_r \mathrm{Out}(R_r)$ because the root lies above the replay cut and can only name a unit's declared outputs.

**9.9 Advice: exactly $A$ bits vs up to $A$; charged at $8|a|$.** Outline: $|a| \le A$; $2^{|a|} \le 2^{A}$ circuits. Draft: Lemma 7.1 pads to exactly $A$ bits (note item (4) flags it). Code: `8·len(a) ≤ max_advice_bits`; `Capacity = Bound + 8|a|`; theorem "every accepted run has capacity ≤ U_max + A" (`overnight-report.md:120-121, 283`). Open: pick one; the code's form (charge the actual length; bound the sum) is the simplest to state and costs at most one bit if one insists on counting circuits.

**9.10 $\theta$ and $\eta$.** Outline (older): $\theta = (q, s, \eta)$ client-proposed. Draft: $\theta = (q, s)$ proposed by client; $\eta$, $A$, $U_{\max}$, $W_{\max}$ verifier's (line 51). Code: same as the draft (`VerifierParameters.eta`; `Header.eta`; `overnight-report.md:275-277`). Decided; the draft is right.

**9.11 What $\mathrm{shape}(x)$ is when $x$ is a tuple of requests.** Outline: "shape(x)" undefined. Draft: same. Code: for `RequestsG`, the structure depends on each request's prompt length and `max_new` (both public, part of the request); for `ClusterG`, on the schedule, which is *not* a function of $x$ and is therefore advice; for `MatmulG`, on the matrix shapes. Author (L1176): *"I also don't know what your shapes thing is or why that's a real thing."* Open: the paper should define $\mathrm{shape}(x)$ as the public part of the input that the description's integer parameters may read (lengths and counts, never token values), or avoid the word and say "the integer parameters are bound from public lengths and from $a$".

**9.12 The verifier runs $G$ (prototype).** §2.3. Outline/draft: never. Code: does. Decided for the prototype (`overnight-report.md` 2.6, 9). Open for the paper's wording; recommendation in §2.3.

**9.13 Existing draft's terminology vs the shipped ontology.** The draft already avoids kernel/IR/instance/artifact/plugin (its own house-rule audit, line 181). Terms to fix: "Inputs occupy addresses $[0, |x|)$"; "inputs lie in no unit"; "Weights are inputs" (paragraph title, line 138: weights are *source gates under $\kappa_W$*, not inputs; the paragraph's point — weights must not be immediates or $|G| = \Theta(|W|)$ — is right and important, keep the argument, rename the paragraph, e.g. "Weights are sources, not immediates"); "the matmul constructor ... produces five definitions" (nine; §4.3); Table 7.1 numbers (§8.2); "$W$ enters `batch` as an input and is broadcast to every row" (today $W$ is a replay unit of `weight` gates whose slots `batch` passes to every row with `jstride 0`; the broadcast point survives).

**9.14 The transparent/hiding discipline and the proof of compilation.** Draft 7.2 opening: the client can prove it ran Compile on a committed $G$. Code: no such proof; the verifier runs Compile itself. Open: keep as design; label as such.

**9.15 (overnight 1) Inputs and weights are gates inside units; the root has no ports.** See 9.6.

**9.16 (overnight 2) Weights are neither inputs nor part of $G$.** See 9.5. Recommendation quoted: *"model weights are a third class of source gate, opened against a per-epoch commitment; they are never boundary positions."*

**9.17 (overnight 3) Bound is the union over admissible error sets, not $\max_E$.** §5's concern, but §7.3's "derived structure is sound" feeds it; the draft's Theorem 7.5 says "§6 bounds the outputs reachable within one circuit by $2^U$", which is consistent with either reading. Nothing for §7 to change; do not describe Bound as "max over strategies".

**9.18 (overnight 4) Cut width vs output reach.** Not a §7 statement, but it determines which marks are *useful*, which §7.5 discusses. Pending fix in the fold (`overnight-report.md` §5.1; `frontier-report.md:220`). Recommendation: §7.5 should not claim step-level replay units certify anything until the fold is reach-aware; say instead that request-level and cell-level units do (frontier report).

**9.19 (overnight 5) Advice charged in bits of its canonical encoding, before Bound, at admission.** See 9.9.

**9.20 (overnight 6) $\eta$ the verifier's, $\theta$ the client's.** See 9.10.

**9.21 (overnight 7) Bound reports $\log_2$ of an integer count**, so $U_{\max} = 0$ is satisfiable. §5/§6 detail; §7 may cite it when saying "$U \le U_{\max}$".

**9.22 (overnight 8) Source-only verification units hold no errors** (F4; `_Fold.series` sets $\ell = 0$). §7 relevance: this is why the tracer emits sources as their own verification cells (`tracer.py::sources`), and why `out_bits = 0` for the source cell in the 8-gate kind table. One sentence in 7.4 if the writing agent shows the kind table.

**9.23 (overnight 9) The verifier runs $G$.** See 9.12.

**9.24 (overnight 10) Value semantics deferred.** §6 item 5.

**9.25 The `square` gate.** The paper's figure has a `square` gate; Σ in the code has none; the 8-gate example uses `mul(3, 3)`. Open: add `square` to the paper's Σ for the example (the serving table already models `square` as `mul(x, x)`), or draw gate 4 as `mul` with both wires from gate 3.

**9.26 Circuit-model consistency with §5.** §5 draft: value sets $V_i$ per gate, input gates as constants, output gates a designated subset; a gate is incorrect if ill-typed or not the relation. Code: one width per gate set; `in` gates are zero-arity gates whose *value* is pinned to $x$ by rank (so "input gates as constants" holds by pinning, not by having the constant in the gate — which is exactly what keeps kinds shareable, L1441); outputs are the root's declared outputs; strict decoding enforces well-typedness on opened values only (§0.3 item 8). §7 should say once that §5's $V_i = \{0,1\}^{w}$ with $w$ the gate set's width for now.

**9.27 Attention kinds per context length.** Not in any doc; observed this morning (§5.4). Open: whether §7.5's "variable length by padding" should acknowledge that the toy does *not* pad the KV context and pays in kinds instead.

---

## 10. Figures and examples

### 10.1 The `workload` → 8-gate circuit figure

The §6 agent is collecting the TikZ. Conventions the author settled on (from the §6 work forwarded at L1349 and the scratch page): three layers, **Index** (unit brackets: $R_1, R_2$ above $V_{1,1}, V_{1,2}, V_{2,1}, V_{2,2}$), **Circuit** (gates 1–8 as nodes with wires), **Transcript** (values as pills under each gate); blue filled rectangles for input gates (1, 2, 5, 6), red filled rectangle for the output gate (8); legend top-right; gate 4 labelled $x^2$. With "every gate in a unit", the four blue input gates sit *inside* $V_{1,1}$ and $V_{2,1}$ (the draft's note item (2) wanted them outside; that is now wrong). $\partial = \{1, 2, 4, 5, 6, 8\}$ — both the paper's and the code's (the code's `boundary()` enumerates `1 2 5 6 4 8`: inputs by rank, then $\mathrm{Out}(R_1)$, then $\mathrm{Out}(R_2)$). Interiors $\{3\}$ and $\{7\}$. No Out copy gates.

Data for the caption or the walkthrough (measured): `C[1..8]` = `in, in, add(1,2), mul(3,3), in, in, add(5,6), mul(4,7)`; $\tau = (1,1,2,4,2,3,5,20)$; kinds: 6 (root, $R_1$, $R_2$, the pair-of-inputs cell ×2 copies, add-square, add-mul); `Cost_replay`: $R_1 = R_2 = 3$ (one `add` at 1 and one `mul` at 2; `in` costs 0); `Cost_proof`: 5 each (`in` costs 1 to prove).

### 10.2 The tracer script for the 8-gate circuit (client side; reproduces the bytes in §4.2)

~~~python
from veritor.core import make_word_gate_set, CompilationLimits
from veritor.compile import Compiler
from veritor.constructors.tracer import Tracer, Wires

gs = make_word_gate_set(16)
t = Tracer(gs)
add, mul, ing = t.gate("add"), t.gate("mul"), t.gate("in")

@t.definition(input_count=0, key="pair_in", role="verification")
def pair_in(_v: Wires):
    return [ing(), ing()]            # two `in` gates, one verification unit

@t.definition(input_count=2, key="add_sq", role="verification")
def add_sq(v: Wires):
    s = add(v[0], v[1])              # gate 3
    return mul(s, s)                 # gate 4: the square; the only declared output

@t.definition(input_count=3, key="add_mul", role="verification")
def add_mul(v: Wires):
    s = add(v[0], v[1])              # gate 7
    return mul(v[2], s)              # gate 8

@t.definition(input_count=0, key="R1", role="replay")
def R1(_v: Wires):
    ab = pair_in()
    return add_sq(ab[0], ab[1])

@t.definition(input_count=1, key="R2", role="replay")
def R2(v: Wires):
    cd = pair_in()
    return add_mul(cd[0], cd[1], v[0])

@t.definition(input_count=0, key="root")
def root(_v: Wires):
    return R2(R1())

desc = t.serialize(root)                              # 2 040 canonical bytes
compiled = Compiler(gs, CompilationLimits()).compile(desc, (1, 1, 2, 3))
compiled.circuit.evaluate((1, 1, 2, 3))               # (1, 1, 2, 4, 2, 3, 5, 20)
~~~

### 10.3 The architecture diagram

Final version (proposed by the assistant at L1205 item 4, accepted by the author at L1213 "I agree with the fourth arrow", drawn in the §4.2 draft at L1214, restated L1348 and in the endorsed §7 outline): Compile at the bottom emits $(C, I)$; Bound and Verify above it, both reading $(C, I)$; a fourth arrow from Bound to Verify carrying $\theta$ (now: Bound *admits* the client's proposed $\theta$ by certifying $U(\theta) \le U_{\max}$, and Verify runs with that $\theta$); §8 on top composes the three. Arrows: Compile → Bound; Compile → Verify; Bound → Verify; {Bound, Verify} → §8 (composition). The author's earlier ordering worry (L1202): *"1. Bound 2. Verify 3. Strategize 4. Compile ... now this is a bunch of top-level sections, man"*; "Strategize" became the client's untrusted Optimize and left the diagram.

### 10.4 The matmul description as bytes (small shape; 2 773 B)

One definition per line, full digests (the wire has no whitespace):

~~~text
{"body":{"input_count":0,"outputs":[["local",0,1,0]],"role":"verification","steps":[{"args":[],"gate":"in","kind":"gate"}]},"digest":"a321b054dd9ea555a4def0a2a46b8ef45dc75d705ff7ba566e5e8258ac2bef8c"}
{"body":{"input_count":0,"outputs":[["local",0,6,1]],"role":"replay","steps":[{"args":[],"count":6,"digest":"a321b054dd9ea555a4def0a2a46b8ef45dc75d705ff7ba566e5e8258ac2bef8c","kind":"repeat"}]},"digest":"c905bdbf5a2655272d2866a62c2970b1c67b846c4ee7721955e3d842413ae4c9"}
{"body":{"input_count":0,"outputs":[["local",0,1,0]],"role":"verification","steps":[{"args":[],"gate":"weight","kind":"gate"}]},"digest":"989cf0e0ef66fe114f18293f25949b8d211c2fa4e1b42374934eea93fe498109"}
{"body":{"input_count":0,"outputs":[["local",0,6,1]],"role":"replay","steps":[{"args":[],"count":6,"digest":"989cf0e0ef66fe114f18293f25949b8d211c2fa4e1b42374934eea93fe498109","kind":"repeat"}]},"digest":"32ce6e63c82cd337d14fdd60e2ba975ede13d503ae543ee6a5ebc3e2e8dbe9eb"}
{"body":{"input_count":2,"outputs":[["local",0,1,0]],"role":null,"steps":[{"args":[["input",0,2,1]],"gate":"mul","kind":"gate"}]},"digest":"f647b5302e2eb745db2f8f774454821433e31fcdf6334095e05f87806d6a6285"}
{"body":{"input_count":2,"outputs":[["local",0,1,0]],"role":null,"steps":[{"args":[["input",0,2,1]],"gate":"add","kind":"gate"}]},"digest":"3f39b641df98448be36b132cbb4669c77c7996d9daa56ec604e85ba54f0e98b1"}
{"body":{"input_count":6,"outputs":[["local",4,1,0]],"role":"verification","steps":[{"args":[["input",0,2,3,1]],"count":3,"digest":"f647b5302e2eb745db2f8f774454821433e31fcdf6334095e05f87806d6a6285","kind":"repeat"},{"args":[["local",0,2,1,2]],"count":1,"digest":"3f39b641df98448be36b132cbb4669c77c7996d9daa56ec604e85ba54f0e98b1","kind":"repeat"},{"args":[["local",3,1,0],["local",2,1,0]],"gate":"add","kind":"gate"}]},"digest":"985d382cc08d07d9b97a856817ab75a46744b8e83c749a841552fa7f6a1b453e"}
{"body":{"input_count":9,"outputs":[["local",0,2,1]],"role":"replay","steps":[{"args":[["input",0,3,1,0],["input",3,3,2,1]],"count":2,"digest":"985d382cc08d07d9b97a856817ab75a46744b8e83c749a841552fa7f6a1b453e","kind":"repeat"}]},"digest":"082128e8b6080fc8e51ab88eca587baa4d9ce793f82ee3ff4051677642ddca1d"}
{"body":{"input_count":0,"outputs":[["local",12,4,1]],"role":null,"steps":[{"args":[],"digest":"c905bdbf5a2655272d2866a62c2970b1c67b846c4ee7721955e3d842413ae4c9","kind":"call"},{"args":[],"digest":"32ce6e63c82cd337d14fdd60e2ba975ede13d503ae543ee6a5ebc3e2e8dbe9eb","kind":"call"},{"args":[["local",0,3,1,3],["local",6,6,1,0]],"count":2,"digest":"082128e8b6080fc8e51ab88eca587baa4d9ce793f82ee3ff4051677642ddca1d","kind":"repeat"}]},"digest":"e5bdff2c528fbe3cdb5c19f2a525b43d82f5d282b7b40c2ecf59e77d89094fa1"}
"root": "e5bdff2c528fbe3cdb5c19f2a525b43d82f5d282b7b40c2ecf59e77d89094fa1", "version": 2
~~~

For a display the writing agent can use the rendering in §4.3; for the paper's "six lines" the three arithmetic definitions suffice, with the sizes 2 886 B ($8^3$) to 4 293 B ($4096^3$).

### 10.5 The grammar display

The draft's `text` block (lines 113–119) is accurate to the code and can be kept verbatim; append "the root has no ports; `in` and `weight` are zero-arity gates of Σ".

---

## 11. Glossary (paper name ↔ code name), verified against the code

- $\Sigma$ ↔ `GateSet` (`core/gates.py`); a gate ↔ `Gate(name, arity, width, replay_cost, proof_cost, evaluate, source)`; $H(\Sigma)$ ↔ `GateSet.digest`.
- $G$ ↔ the description bytes emitted by a `Constructor` (`compile/constructor.py:23`); the paper's $G$ is the bytes; the code's constructor is client tooling with a `digest` (`constructor_digest`, `:38`) bound as `Header.constructor`.
- $C$ ↔ `Circuit` (`core/circuit.py:89`), realized by `DescriptionCircuit` (`:347`); `C[i]` ↔ `circuit[i] -> GateRef` (`:45`); $n$ ↔ `circuit.n`; $\mathrm{out}(C)$ ↔ `circuit.outputs`; `In/Out/Size/Cost` ↔ `circuit.In(unit)` etc.
- $I$ ↔ `Index` (`core/index.py:258`); replay units ↔ `index.replay_units` (`Units`, `:133`); verification units ↔ `index.verification_units`, `index.verification_unit(r, k)`; `owner(i)` ↔ `Units.owner`; kinds ↔ `index.kinds() -> tuple[KindSummary]` (`:193`); the per-kind table ↔ `KindTable` (`:228`, `Compiled.kind_table()`).
- $\partial$ ↔ `index.boundary()`; $\mathrm{Int}(r)$ ↔ `index.interior(r)`; In (the input gates) ↔ `index.inputs()`; the weight gates ↔ `index.weights()`.
- Compile ↔ `veritor.research.Compile` (`research.py:56`) wrapping `Compiler.compile` (`compile/compiler.py:19,44`); $(C, I)$ ↔ `Compiled(circuit, index, digest)` (`core/compiled.py:15`); the run's public record ↔ `Compilation(compiled, constructor_digest, inputs, advice)` (`compile/constructor.py:47`).
- $H(C, I)$ ↔ `Compiled.digest` = `Compiled.digest_of(description_digest, gate_set_digest)` (`compiled.py:28-32`); $H(G)$ ↔ `description_digest(bytes)` (`compile/description.py:84`).
- marks ↔ `Definition.role` ∈ {`None`, `REPLAY`, `VERIFICATION`} (`core/description.py:53-56`); the mark check ↔ `validate_marks` (`core/index.py:642`).
- definition / kind ↔ `Definition` (`core/description.py:203`) identified by its digest; copy ↔ `Frame` (`:793`); steps ↔ `GateStep` (`:147`), `CallStep` (`:170`, `count` ≥ 1 covers `repeat`); range ↔ `Range` (`:84`); run ↔ `Run` (`:116`).
- source gates ↔ `Gate.source ∈ {INPUT_SOURCE, WEIGHT_SOURCE}`; `in`, `weight` in both gate sets.
- $x$ ↔ `Header.public_inputs` / `Compilation.inputs` (values of the `in` gates by rank); $y^*$ ↔ `Header.claimed_outputs`; $a$ ↔ `Header.advice` / `Compilation.advice`; $A$ ↔ `max_advice_bits` (`research.Compile`, `VerifierParameters`); $\kappa_W$ ↔ `Header.weights` from `commit_weights`.
- $\theta = (q, s)$ ↔ `VerificationPolicy` (`core/policy.py:65`) / `Header.policy`; $\eta$ ↔ `VerifierParameters.eta`, `Header.eta`; $U_{\max}$ ↔ `VerifierParameters.max_capacity`; $W_{\max}$ ↔ `VerifierParameters.max_work`.
- Bound ↔ `veritor.research.Bound` / `analysis.bound.bound(compiled, policy, eta).bits`; Capacity ↔ `research.Capacity` (= Bound + 8|a|); $W$ ↔ `protocol.parameters.expected_work`; Cost ↔ `research.Cost`; Optimize ↔ `research.Optimize` (client-side advisory search).
- Compile's limits ↔ `CompilationLimits` (`core/limits.py:8`); admission limits ↔ `VerificationLimits` (`:26`).
- tracer ↔ `constructors.tracer.Tracer`; constructors ↔ `MatmulG`, `DemoG`, `RequestsG`, `ClusterG`, `ToyLM` (kinds), `Schedule`/`Request` (`constructors/schedule.py`).
- the serving model (a per-kind table without a description) ↔ `evaluation.serving.serving_table(ServingShape, replay, verification)`; the frontier sweep ↔ `evaluation.frontier`.

---

## 12. Annotated table of contents of `docs/paper/section-7-secure-circuit-compilation.md`

Line numbers are the draft's. Verdicts: **keep**, **revise**, **drop**.

| lines | paragraph | verdict | reason / pointer |
|---|---|---|---|
| 3 | Opening paragraph ("The verifier of this section neither receives a gate list...") | keep, optional | Accurate; the draft's own note (line 180 item 6) offers to drop it so the section opens on "Suppose the client claims". Author's outline closes 7.1 with the figure and roadmap; either is fine. |
| 5–20 | `workload` program | keep | Author's text (L1368). |
| 22–27 | Four constraints | revise lightly | Constraint 3: replace "$G$ committed once per epoch" with the author's phrasing only if the paper defines epoch (§7.3 table, "epoch"). Constraint 4: choose one scale ($4 \times 10^{10}$ GPT-2 Small per §5, or $10^{17}$ for 70B) and use it consistently. §1.2. |
| 29–33 | "The operation cannot be fixed in advance ... Random sampling is well defined ... $(1+1)^2(2+3)=20$ ... A compiler can replace..." | keep | Author's text. |
| 35 | Figure placeholder | revise | Reference the §6 figure; note inputs are inside units and gate 4 is `mul(3,3)` or a `square` gate added to Σ (§9.25). |
| 37 | "The circuit resolves constraints 1 and 2. Constraint 3 is where the exfiltration game parts from ordinary verifiable computation..." | keep | Matches L1340 (developer is the adversary) and L1213–1217. |
| 39 | "Constraint 4 is quantitative ... the asymmetry that governs the design" | keep | Matches L1214/L1232; §0.3 item 7. |
| 41 | Two observations | keep | Matches the outline's wording. |
| 43 | Roadmap sentence | keep | |
| 47 | 7.2 opening: transparent form, "the client can prove it ran Compile correctly" | revise | Label the proof of compilation as design, not prototype (§9.14). |
| 49 | Objects: Σ, description $G$, Compile binds parameters, client proposes $\theta$, verifier fixes $\eta$, $A$, $U_{\max}$, $W_{\max}$ | revise | "binds its integer parameters" is design (§9.1); the rest matches the code (§9.10). |
| 51–58 | Once per model 1–4 and $h_C$ | revise | Step 3 mark check is right and matches `validate_marks`; step 4's $h_C$ is right (§9.4). Add: the digest is per description; in the prototype that is per run (§9.2). |
| 60–74 | Once per run 1–5, $W$ formula | revise | Step 3 "binds the parameters of $G$ from shape(x)" is design; today Compile *runs* $G$ (§2.3, §9.1). $W$ formula matches `expected_work` (§2.2). Step 5's "$\kappa_W$ as the pre-committed subtree of the boundary tree" → a second root (§9.5). |
| 76 | "The verifier never runs client code, never reads a gate list..." | revise | Keep the sentence for the paper's protocol; add the one-sentence prototype caveat (§2.3). |
| 80 | "The one idea is that determinism turns a choice into a count" | keep | |
| 82–86 | Lemma 7.1 (Binding) + proof + remarks | revise | Exactly-$A$ vs up-to-$A$ (§9.9); otherwise right (§3.1). |
| 88–92 | Lemma 7.2 (Soundness of derived structure) + proof + remarks | revise | (i) "inputs lie in no unit" and "$\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r)$" are outdated (§9.6, §9.7); (ii) and (iii) right; the remarks' "since inputs lie in no unit" must go. Add the pinned-source exclusion from Out. §3.2. |
| 94 | "No condition on the marks" | keep, extend | Right; can now cite the measured consequence (frontier report) and the argument's structure (§3.3). |
| 96–98 | Proposition 7.3 (Efficiency) + sketch | revise | Drop "plus time linear in the declared interfaces": the table is $O(|G|)$ now (§3.4). |
| 100 | Proposition 7.4 (Completeness) | keep | Mention the proof-cost cap is the one completeness condition. |
| 102–106 | Theorem 7.5 (No uncharged degrees of freedom) + proof + remarks | revise lightly | $U + A$ vs the code's per-run `Capacity ≤ U_max + A` (§3.7, §9.9); the $\theta$ remark is right. |
| 110–121 | 7.4 The grammar (text block and explanation) | keep, append | Add "the root has no ports; `in`/`weight` are zero-arity gates" (§4.1, §10.5). |
| 123–135 | Matmul display: "five definitions", `batch` with 8 192 inputs | revise | Nine definitions; sources are units; `batch` has no ports; new byte counts (§4.3, §8.2). |
| 137–138 | "Weights are inputs" | revise | Rename and rewrite: weights are source gates under $\kappa_W$, never immediates (else $|G| = \Theta(|W|)$) and never boundary positions (§9.5, §9.13). The compression argument is right and important. |
| 140–142 | The compiler: address layout, descent, unit structure, per-kind table | revise | "Inputs occupy addresses $[0, |x|)$" and "Inputs are boundary positions outside the units" are outdated (§9.6). The descent description matches `Frame`, `Units`, `_Boundary`. The per-kind table description matches `KindSummary` (add `source_inputs`, `source_weights`). |
| 144 | Parameters | keep | Correctly flagged as design; cross-reference §9.1. |
| 146 | The tracer | keep | Matches `tracer.py`; add that `tracer.inputs/weights` emit canonical source cells. |
| 148 | Prior art | keep, fill TODOs | §4.8 has the references and status. Avoid "netlist"/"IR" if the house rule holds; the paragraph currently uses neither. |
| 152 | 7.5 What is expressible | keep | §5.1. |
| 154 | What the envelope costs on inference | revise | Embedding: cite `lm.py`'s one-hot-then-matvec; the "at most doubles that layer" claim is right; MoE numbers remain formulas (§5.3). Add the continual-batching vs per-request distinction and the frontier finding about which units certify (§5.3 last item). |
| 156–166 | Compile cost table | revise | Replace with this morning's numbers (§5.4); add the serving-table rows for GPT-2 and 70B shapes labelled as models; remove the pre-fix per-kind-table times. |
| 170 | 7.6 Limitations | revise, extend | Add value semantics, the verifier-runs-$G$ gap, per-context kinds (§6). |
| 174–181 | Notes for the author | drop from the section | Already served their purpose; the points are absorbed in §9 here. |

---

## 13. Checklist for the writing agent

Properties §7 must make explicit, in order, with the sentences the outline requires:

1. The section's job: discharge §6's assumption of a committed $(C, I)$ with tiling, refinement and a boundary containing every cross-unit read, without trusting anything the client sends, in time sublinear in the circuit; and give §8 the lemma bounding the reachable circuits by the advice.
2. 7.1: the author's `workload` text verbatim; the four constraints as constraints and consequences; the two observations; the figure; the one-sentence roadmap. Not a design-rationale essay.
3. 7.2: the transparent protocol with once-per-model and once-per-run steps as numbered lists; the exact sentence *"Say explicitly what the verifier never does: run client code, read a gate list, or believe any quantity the client asserts about its own circuit. Every number in steps 3–8 is derived."*; then one sentence on the prototype (public versioned constructor executed by the verifier; output-bounded; deployment sandboxes it or proves compilation).
4. 7.3 Determinism: *"State this as the lemma §8 will cite."* Decide exactly-$A$ or up-to-$A$ (§9.9) and make §8 agree.
5. 7.3 Derived structure: state that a caller can only reach declared outputs, hence $\partial \supseteq$ all cross-unit reads for any marks; that Out excludes pinned source gates; that every gate (sources included) lies in exactly one replay and one verification unit; that Size, Cost, $|\mathrm{In}|$, $|\mathrm{Out}|$, `out_bits` are exact per kind.
6. 7.3 No condition on the marks: partition and refinement only; everything else priced; point to the argument (§3.3, L1361) or an appendix; mention the proof-cost cap and $W_{\max}$ as completeness/work limits, not soundness conditions.
7. 7.3 Efficiency: Compile $O(|G|)$; $C[i]$, In/Out/Size/Cost, owner, $\mathrm{rank}_\partial$ in $O(\text{depth})$; "this is what §6.4's construction assumed"; the per-kind table is $O(|G|)$.
8. 7.3 Completeness: every valid description compiles; honest marks pass; the proof-cost cap.
9. 7.4 The description: definitions in dependency order, hash-consed; `gate`/`call`/`repeat`; strided relative ranges; canonical bytes; *"Give the matmul in six lines"* — with the honest count (three arithmetic definitions shown; nine in the file) and sizes ($O(1)$ in rows and columns, $O(\log k)$ in $k$; 2.9–4.3 KB measured); the "kilobytes for a GPT-2 forward pass" claim qualified as in §5.4.
10. 7.4 The compiler: parse; validate and summarize each definition once; check marks per definition; descent (intervals, division, bisection, prefix sums for $\partial$ rank).
11. 7.4 Parameters: design; excluded structural dependence on values of $x$; MoE routing through advice.
12. 7.4 The tracer: untrusted; one paragraph; "JAX/PyTorch front ends live here".
13. 7.4 Prior art in one paragraph (Lengauer–Wanke; hash-consing; position-independent addressing; `generate`/affine loops; staging; why nothing off the shelf; LCF). This is where the rationale lives.
14. 7.5: the three universality tiers (note $\{+, \times\}$ mod $2^B$ is not complete); the envelope items; *"the one real design trade"*: MoE advice-versus-overhead frontier, stated as formulas with a TODO for a measured point; the continual-batching (advice) vs per-request (no advice) contrast; which marks give non-vacuous bounds (request, cell) and which do not yet (step, layer, matvec, row) under the current fold.
15. 7.5 Compile-cost table: today's numbers; label serving-table rows as models, not traced descriptions.
16. 7.6: parametric descriptions; per-gate widths; canonical chunking; transcendental gates with specified rounding; value semantics; verifier-executes-$G$. One sentence each.
17. House rules throughout: no kernel, load, description object, instance, address set, IR, netlist, artifact, plugin, prover/server ("client"), "word(s)", "well-formed" for descriptions, "authorized computation", "design challenges" ("design decisions"). "Well-typed" for values only. Objects are $G, C, I, \Sigma$, Compile. Compile outputs $(C, I)$ directly.
18. Consistency with §5 and §6 as currently drafted: $\partial = \mathrm{In} \cup \bigcup_r \mathrm{Out}(R_r)$ (outputs inside); $\mathrm{Int}(r) = R_r \setminus \mathrm{Out}(R_r) \setminus \mathrm{pinned}(R_r)$; $\eta$ verifier's, $\theta$ client's; Bound is the union; weights under $\kappa_W$ and never in $\partial$; one width per gate set for now.

## 14. Appendices: tests as specifications, exact record layouts, descent algorithms, transcript index

### 14.1 Test inventory mapped to §7 claims

The author's instruction was that tests are precise specifications. Every §7 claim below is followed by the tests that pin it down (all paths under `tests/veritor/`). Test names are the sentences the writing agent can lean on; read the test body when a claim needs its exact preconditions.

**Determinism and binding (§3.1, Lemma 7.1 / Theorem 7.5).**

- `security/test_canonical.py::test_same_description_bytes_give_the_same_digest_and_layout` — same bytes, same `Compiled.digest`, same index layout.
- `security/test_canonical.py::test_reencoded_description_bytes_are_rejected_before_any_compile_work` — a semantically equal but non-canonical encoding is rejected at parse time, so there is exactly one byte string per description.
- `security/test_canonical.py::test_changing_a_mark_changes_the_digest_and_the_header` — marks are inside the digested bytes (the "marks in the digest" decision, §9.4).
- `security/test_canonical.py::test_canonical_bytes_are_unique_per_transcript`, `::test_transcript_with_a_noncanonical_or_malformed_encoding_is_rejected`.
- `core/test_identity.py::test_canonical_json_is_sorted_compact_and_order_independent`, `::test_canonical_json_rejects_noncanonical_values`, `::test_tagged_sha256_is_deterministic_and_domain_separated`, `::test_identity_digest_hashes_the_canonical_manifest`.
- `compile/test_description.py::test_canonical_encoding_is_enforced`, `::test_definition_digest_is_tagged_and_body_sensitive`.
- `core/test_gates.py::test_gate_set_identity_binds_declarations_not_callables` — $\Sigma$'s digest binds names, arities, widths, costs, not Python callables (the semantics are named by `(name, version)`).
- `compile/test_compiler.py::test_compile_returns_the_circuit_the_index_and_a_bound_digest`.
- `research_api/test_compile_and_exports.py::test_compile_runs_g_and_records_what_it_ran_on`, `::test_compile_checks_the_constructors_digest`, `::test_paper_functions_reject_anything_but_what_compile_produced` — `Bound`/`Cost`/`Expectation` accept only a `Compilation` produced by `Compile`.
- `research_api/test_conformance.py::test_the_header_binds_the_constructor_and_the_advice`.
- `constructors/test_cluster_protocol.py::test_the_compilation_binds_the_constructor_the_prompts_and_the_schedule`.
- `constructors/test_matmul.py::test_constructor_digest_names_the_class_version_and_width`, `::test_digest_depends_on_shape_and_width_not_public_values`; `constructors/test_demo_g.py::test_demo_g_digest_binds_the_batch_shape_not_its_values`; `constructors/test_cluster.py::test_digest_names_the_constructor_and_its_parameters`.

**Advice is charged (§3.1, the $A$ term).**

- `research_api/test_compile_and_exports.py::test_compile_charges_the_advice_g_consumes`, `::test_compile_rejects_advice_over_the_bound`, `::test_capacity_adds_exactly_the_advice_bits` — `Capacity = Bound + 8·|a|`.
- `research_api/test_conformance.py::test_advice_over_the_verifiers_bound_is_rejected_at_admission`; `constructors/test_cluster_protocol.py::test_the_advice_must_be_admitted_by_the_verifier`; `constructors/test_demo_g.py::test_demo_g_takes_advice_the_verifier_charges`.
- `constructors/test_matmul.py::test_constructor_rejects_foreign_inputs_and_advice` — a constructor that takes no advice rejects non-empty advice.

**Derived structure is sound (§3.2, Lemma 7.2).**

- `security/test_tiling.py::test_every_gate_is_in_exactly_one_replay_unit_and_one_verification_unit`, `::test_verification_units_refine_replay_units`, `::test_cross_unit_reads_go_only_through_declared_outputs` (the precondition of §6's $\ell^\star$ lemma), `::test_marks_leaving_a_gate_uncovered_are_a_compile_error`, `::test_nested_or_straddling_marks_are_a_compile_error`, `::test_layout_rejects_a_circuit_that_reads_across_the_cut`.
- `security/test_binding.py::test_every_address_has_exactly_one_owner`, `::test_two_units_reading_one_address_cannot_be_shown_different_values` (one committed copy per boundary value; the shared address space).
- `compile/test_index.py::test_boundary_and_interiors_match_the_reference_derivation` — `Index.boundary()`/`interior(r)` against a flat reference over the materialized circuit.
- `compile/test_index.py::test_kinds_table_matches_enumeration_of_every_copy`, `::test_kinds_table_summarizes_each_definition_once` — per-kind table equals enumeration.
- `compile/test_description_circuit.py::test_lazy_circuit_matches_flat_on_matmul`, `::test_interfaces_of_matmul_units_are_resolved_through_the_frame`, `::test_declared_interface_may_exceed_what_is_read` — `In` is the *declared* interface (a superset of actual reads), as §3.2 says.
- `compile/test_out_runs.py::test_run_interfaces_match_enumeration`, `::test_out_runs_are_what_the_output_ranges_resolve_to`, `::test_strided_outputs_over_children_fall_back_to_residue_runs`, `::test_random_strided_declarations_match_enumeration` (the randomized fuzzer against the per-ordinal resolver), `::test_a_strided_run_inside_one_copy_stops_at_the_declared_count` (the F7 clamp), `::test_fewer_elements_than_residue_classes_is_exact_enumeration`, `::test_input_count_is_the_declared_interface_not_what_is_read`.
- Distinct-outputs rule: `compile/test_out_runs.py::test_the_same_slot_declared_twice_is_rejected`, `::test_overlapping_strided_ranges_are_rejected`, `::test_duplicates_through_passthrough_children_are_rejected`, `::test_disjoint_interleaved_strides_are_accepted`.
- Source gates: `compile/test_description.py::test_source_gates_are_pinned_pieces_with_prefix_summed_ranks`, `::test_root_ports_and_repeated_source_gates_are_rejected`; `compile/test_index.py::test_input_and_weight_domains_are_lazy_and_ranked_in_address_order`, `::test_stray_source_gates_are_caught_by_the_tiling_check`, `::test_the_root_has_no_ports`; `core/test_gates.py::test_source_gates_are_exactly_the_zero_arity_gates`, `::test_word_gate_set_has_an_input_and_a_weight_source_gate`.
- Matmul specifics: `constructors/test_matmul.py::test_boundary_is_the_inputs_and_the_rows_outputs`, `::test_shared_weight_gates_fan_out_across_replay_units`, `::test_marks_make_source_units_rows_replay_units_and_dots_verification_units`.
- Cluster specifics: `constructors/test_cluster.py::test_marks_tile_every_gate_once_and_the_kv_cache_is_the_step_interface`, `::test_kinds_are_shared_across_pods_and_steps`.

**No condition on marks beyond tiling and refinement; the proof-cost cap (§3.3, §3.5).**

- `compile/test_index.py::test_gate_step_above_the_replay_cut_is_named`, `::test_gate_step_inside_a_replay_unit_must_be_in_a_verification_unit`, `::test_marks_may_not_nest_and_verification_needs_a_replay_unit`, `::test_units_must_have_gates_and_verification_units_respect_the_proof_cap` (uses `CompilationLimits(max_verification_unit_proof_cost=11)` vs `10`), `::test_root_may_itself_be_the_only_replay_unit` (one replay unit is legal; Bound then prices it).
- `compile/test_compiler.py::test_compile_checks_inputs_and_marks`.
- `constructors/test_tracer.py::test_source_gates_outside_a_replay_unit_or_inside_a_verification_unit_are_rejected`, `::test_the_canonical_cells_parse_as_pinned_verification_units`, `::test_source_gates_are_one_repeat_of_a_canonical_verification_cell`.
- That bad partitions are *priced*, not forbidden: `security/test_admission.py::test_policy_whose_bound_exceeds_u_max_is_policy_rejected`, `::test_waiving_u_max_admits_a_policy_that_checks_nothing` (why `max_capacity` is required, F2), `::test_run_whose_expected_work_exceeds_w_max_is_work_budget_exceeded`, `::test_admission_checks_unit_counts_against_the_limits`.
- Bound soundness over arbitrary markings: `security/test_bound_soundness.py::test_union_over_random_markings_is_below_the_fold`, `::test_whole_unit_corruption_is_covered_by_the_unit_interface`, `::test_fully_checked_run_has_exactly_zero_capacity`, `::test_source_only_units_contribute_no_error_terms`, `::test_source_only_rule_is_exact_against_the_enumerated_union`, `::test_integer_count_never_undercounts_and_never_exceeds_its_input` (F3).

**Efficiency (§3.4, Proposition 7.3).**

- `compile/test_compiler.py::test_compile_and_index_are_sublinear_in_the_gate_count`.
- `security/test_canonical.py::test_a_description_of_a_trillion_gates_compiles_in_bounded_time`.
- `compile/test_description.py::test_limits_bound_every_summary_without_unrolling`, `::test_repeat_shifts_arguments_per_copy_and_descends_by_division`, `::test_strided_ranges_describe_matrix_columns_in_constant_size`, `::test_nested_repeat_frames_carry_copy_indices`.
- `compile/test_out_runs.py::test_a_strided_subset_of_a_huge_slot_linear_interface_is_one_run`, `::test_admission_does_not_scale_with_the_input_count`, `::test_the_wide_layer_declares_and_reads_every_input`.
- `compile/test_index.py::test_units_are_the_marked_copies_with_lazy_lookup`.
- `core/test_indexed.py::test_range_domain_is_lazy_and_supports_steps`, `::test_range_domain_supports_empty_and_huge_counts_without_materialization`, `::test_lazy_domain_contract_requires_rank_unrank_without_materialization`.
- `constructors/test_matmul.py::test_description_size_does_not_grow_with_rows_or_columns`, `::test_description_size_is_logarithmic_in_the_contraction_length` (the "kilobytes against $10^{11}$ gates" scaling, §8.2).
- `constructors/test_lm.py::test_kinds_are_row_sized_and_shared_across_positions`.

**Resource-limit soundness (§3.6).**

- `compile/test_out_runs.py::test_interfaces_resolving_to_too_many_runs_are_rejected_without_doing_the_work`, `::test_the_total_number_of_runs_over_a_description_is_capped` (`max_output_runs`, `max_output_runs_total`).
- `compile/test_description.py::test_source_gate_runs_are_bounded_per_definition`, `::test_malformed_definitions_are_rejected`, `::test_call_arity_and_repeat_bounds`, `::test_limits_bound_every_summary_without_unrolling`.
- `security/test_canonical.py::test_nesting_deeper_than_the_limit_is_a_compile_error`.
- `security/test_admission.py::test_theta_with_an_enormous_denominator_is_resource_limit` (`max_probability_denominator_bits`).
- `research_api/test_compile_and_exports.py::test_compile_wraps_a_failing_constructor_into_a_rejection` (including `SystemExit`).
- `constructors/test_cluster.py::test_bad_requests_and_bad_advice_fail_to_trace`, `::test_the_compiler_checks_the_input_count_against_the_prompts`; `constructors/test_matmul.py::test_workload_rejects_malformed_shapes_and_values`; `constructors/test_demo_g.py::test_demo_g_rejects_malformed_batches`.

**Completeness (§3.5, Proposition 7.4): honest constructors compile and their runs are accepted.**

- `constructors/test_requests.py::test_an_honest_run_is_accepted_with_empty_advice`; `constructors/test_cluster_protocol.py::test_an_honest_cluster_run_is_accepted_and_its_transcript_round_trips`, `::test_a_hand_written_schedule_runs_under_the_same_weight_root`; `research_api/test_conformance.py::test_honest_conformance_transcript_verifies_purely`, `::test_interactive_run_and_pure_verification_agree`.
- Semantic transparency of the compiled circuit against a plain Python reference: `constructors/test_lm.py::test_single_request_circuit_decodes_like_the_reference`; `constructors/test_cluster.py::test_fcfs_run_generates_what_the_reference_generates`, `::test_a_hand_written_schedule_is_semantically_transparent`, `::test_mixed_lifetimes_in_one_pod_compile_and_decode_like_the_reference` (the F7 liveness case), `::test_two_layers_and_a_different_fcfs_shape`; `constructors/test_matmul.py::test_shared_weight_matmul_executes_in_canonical_order`, `::test_modular_overflow_is_applied_to_every_output`.

**Values are checked only where opened; well-typedness is local (§0.3 item 8).**

- `security/test_local_checks.py::test_value_outside_the_gate_width_is_invalid_value`, `::test_noncanonical_encoding_of_a_committed_value_is_invalid_value`, `::test_every_non_source_gate_of_a_sampled_unit_is_checked`, `::test_gate_arguments_are_the_owners_committed_values_not_the_provers_claims`, `::test_altered_weight_in_the_run_is_caught_only_when_a_reader_is_sampled` (a wrong weight is caught only if a unit reading it is sampled: values are never globally checked).
- `core/test_gates.py::test_gate_validates_arity_and_value_widths`, `::test_value_codec_is_fixed_width_big_endian`.

**Weights under $\kappa_W$ per model (§9.5).**

- `security/test_binding.py::test_kappa_w_is_bound_to_the_gate_set_and_the_vector_not_the_description`, `::test_kappa_w_with_another_count_is_rejected_before_any_commitment`, `::test_weight_opened_with_another_value_is_invalid_opening`.
- `constructors/test_cluster_protocol.py::test_a_hand_written_schedule_runs_under_the_same_weight_root` — two different descriptions (schedules) verified under one $\kappa_W$: the reason weights are ranked per model, not per description.

**Corruption is caught where §6 says it is (sanity for §7's "hand $(C, I)$ to §6").**

- `constructors/test_requests.py::test_a_corrupted_token_is_caught_at_the_boundary`, `::test_a_corrupted_interior_value_is_caught_when_everything_is_checked`; `constructors/test_cluster_protocol.py::test_an_altered_generated_token_is_rejected_at_the_boundary`, `::test_a_corrupted_interior_value_is_rejected_when_everything_is_sampled`; `security/test_local_checks.py::test_wrong_input_value_is_caught_at_the_boundary_before_any_sampling`, `::test_wrong_claimed_output_with_honest_values_is_caught_at_the_boundary`.
- `security/test_sampling.py::test_acceptance_rate_matches_survival_of_the_error_set`, `::test_survival_is_the_product_of_per_replay_unit_factors` — the survival function §6 exports is what the implementation realizes.

### 14.2 Exact record layouts (for precise statements of what is bound where)

`CompilationLimits` (`src/veritor/core/limits.py:11-24`), a frozen dataclass with defaults:

~~~
max_description_bytes: int = 10_000_000
max_definitions: int = 100_000
max_steps_per_definition: int = 1_000_000
max_addresses: int = (1 << 63) - 1
max_cost: int = (1 << 63) - 1
max_depth: int = 256
max_verification_unit_proof_cost: int = (1 << 63) - 1
max_output_runs: int = 256          # pieces one definition's declared outputs may resolve to
max_output_runs_total: int = 16_384 # resolved output runs over the whole description
~~~

`VerificationLimits` (`src/veritor/core/limits.py:37-46`):

~~~
max_manifest_bytes: int = 1 << 20
max_artifact_bytes: int = 64 << 20
max_positions: int = 10_000_000
max_units: int = 1_000_000
max_positions_per_unit: int = 1_000_000
max_openings: int = 10_000_000
max_proof_bytes: int = 64 << 20
max_transcript_bytes: int = 128 << 20
max_nesting_depth: int = 128
max_probability_denominator_bits: int = 64
~~~

`VerifierParameters` (`src/veritor/protocol/parameters.py:53-56`): `eta: Fraction`, `max_capacity: int | None` (required keyword since F2; `None` waives the check and is tested as such), `max_advice_bits: int`, `max_work: int`.

`Header` (`src/veritor/protocol/messages.py:199-208`), the per-run public parameters, digested into every later message:

~~~
session_id: bytes
compiled_digest: Digest        # H(C, I): digest of the canonical description bytes under Σ
constructor: Digest            # G's digest (class, version, shape parameters; not x, not a)
advice: bytes                  # a, charged at 8·len(a) bits
policy: VerificationPolicy     # θ = (q, s) as Fractions
eta: Fraction                  # the verifier's threshold, not the client's
public_inputs: tuple[bytes, ...]
claimed_outputs: tuple[bytes, ...]
weights: Weights | None        # κ_W root with its count, per model
digest: bytes                  # derived
~~~

`Weights` (`src/veritor/protocol/messages.py:159-167`): *"The root `kappa_W` over a model's weight vector of `count` values. Leaf `k` is the `k`-th weight, read by the `k`-th `weight` gate in address order of whichever circuit is verified, so the root is committed once per model and is the same for every description compiled from it. The verifier holds this and binds it into the header, and weight values are opened only where sampled, at their ranks."* This is the exact sentence to cite for §9.5.

`KindSummary` (`src/veritor/core/index.py:209-224`), one row per definition:

~~~
kind: str                        # definition digest
role: str | None                 # None | "replay" | "verification"
copies: int
size: int                        # gates per copy
replay_cost: int
proof_cost: int
input_count: int                 # declared ports (In), per copy
out_count: int                   # |Out| per copy, source gates excluded
out_bits: int                    # out_count × width
source_inputs: int               # `in` gates per copy
source_weights: int              # `weight` gates per copy
min_depth: int
max_depth: int
children: tuple[tuple[str, int], ...]             # (child kind, copies per copy)
verification_units: int                           # per copy
verification_kinds: tuple[tuple[str, int], ...]   # (verification kind, count per copy)
~~~

`Gate` (`src/veritor/core/gates.py:56-65`, docstring): *"One scalar gate: `name`, `arity`, output `width`, costs and `source`. `evaluate` and `check` are trusted executable semantics. They are excluded from identity; the enclosing gate set's name and version name the semantics. Arguments are validated against the gate's own width (every gate in a set has the same width for now). A gate with a `source` (`"input"` or `"weight"`) has arity `0` and no semantics: its value comes from the environment, so `evaluate` and `check` raise."*

### 14.3 The descent algorithms, as implemented (for §7.4's "then descent" paragraph)

All queries are walks down the `Frame` chain (`src/veritor/core/description.py:793-930`). A `Frame` is one copy of one definition: `(definition, base address, parent frame, step index in parent, copy index j)`. Per definition the validator precomputes `step_address` (prefix sums of step sizes), `step_slot` (prefix sums of step output counts), `step_source(source)` (prefix sums of `in`/`weight` counts), and `out_starts` (prefix sums of run lengths). Each loop iteration below is one level of the hierarchy, so every query is $O(\text{depth} \cdot \log(\text{steps}))$.

Gate at an address (`Frame.locate`, `:908-920`; `Frame.gate`, `:922-930`):

~~~
locate(address):
    frame, offset = root, address
    loop:
        d = frame.definition
        index = bisect_right(d.step_address, offset) - 1     # which step owns the offset
        step = d.steps[index]
        if step is a GateStep: return (frame, step)
        copy, offset = divmod(offset - d.step_address[index], step.child.size)   # repeat: divide
        frame = frame.child(index, copy)

gate(address):
    frame, step = locate(address)
    return step.gate, [frame.address(item.space, item.element(k)) for item in step.args for k in range(item.count)]
~~~

Resolving a relative argument to an absolute address (`Frame.address`, `slot_address`, `input_address`, `:879-902`): a LOCAL reference names a slot of the enclosing definition, which is either a gate step (address = `base + offset`) or an output of an earlier child (recurse into that child's `output_address`); an INPUT reference names a port of the enclosing definition, which is resolved by walking *up* to the parent, reading the parent's `call`/`repeat` argument item at that ordinal for copy $j$ (`item.element(element, frame.j)` applies the per-copy stride), and continuing. At the root an INPUT reference is an error: *"the root has no ports"* (`:893`).

Source gate of a given rank (`Frame.source_address`, `:850-867`) — the `unrank` of `Index.inputs()`/`weights()`:

~~~
source_address(source, rank):
    frame = root; rank -= frame.source_before(source)
    loop:
        d = frame.definition; sums = d.step_source(source)
        index = bisect_right(sums, rank) - 1
        step = d.steps[index]
        if step is a GateStep: return frame.base + d.step_address[index]
        copy, rank = divmod(rank - sums[index], step.child.source_total(source))
        frame = frame.child(index, copy)
~~~

and its inverse `source_rank(source, address)` (`:869-877`) is `locate` followed by a prefix-sum read. `Out` of a unit (`DescriptionCircuit.Out`, `circuit.py:433-438`) is `definition.out_offset(rank)` added to the frame's base, where `out_offset` bisects `out_starts` and steps along the run (`description.py:467-470`). `In`, `Size`, `Cost` are reads of the summarized definition (`circuit.py:427-450`). `Index.owner(address, role)` is `locate` stopping at the first frame whose definition carries the role. `Index.boundary().rank(address)` is `owner` followed by the per-kind prefix sum of interface widths accumulated along the path, plus the unit's local rank of the address inside its `Out` runs.

### 14.4 The README's statement of the objects (current, matches the code)

`README.md:22-44`, quoted for the writing agent because it is the shortest correct description of the shipped ontology:

> **Description**: the wire format a constructor `G` produces. A hash-consed sequence of definitions built from three steps, `gate`, `call`, and `repeat`, with relative range arguments, so a transformer-sized circuit is a few kilobytes. Definitions may carry a mark, `replay` or `verification`.
>
> **Circuit** `C`: `C[i]` gives the operator and the absolute argument addresses of gate `i`. The circuit's inputs are its `in` gates and its weights its `weight` gates: they sit inside units like any other gate, so the root definition has no ports (ports remain the relative addressing mechanism *inside* the hierarchy). `C.inputs` and `C.weights` list their addresses by rank. `DescriptionCircuit` answers these queries lazily in `O(depth)`.
>
> **Index** `I`: the hierarchy of copies over `C`. Every copy of a definition is a node; copies of the same definition are one *kind*. The marked nodes form two antichains: the replay units tile the gates — every gate, source gates included, lies in exactly one replay unit and one verification unit — and the verification units refine them. `I.inputs()` and `I.weights()` are the source gates by rank; `I.boundary()` is `In ∪ ⋃_r Out(R_r)`; `I.interior(r)` is `R_r` minus its interface minus its source gates. `In` and `Out` of a unit are its *declared* interfaces, held per kind as arithmetic runs of addresses, so the per-kind table, the boundary and the interiors cost by the number of runs, never by the addresses they span; `Out` excludes a unit's *pinned* (source) gates, which are boundary or `κ_W` positions already.

Note "Description" is a README/code name; the paper says "$G$" or "the description $G$ produces"; see §7.3 and §11.

### 14.5 Transcript line index used in this dump

All references `L<n>` are line numbers of `/Users/danielreuter/.cursor/projects/Users-danielreuter-projects-veritor/agent-transcripts/eb746331-5a84-4468-9455-5c2a7c14f35c/eb746331-5a84-4468-9455-5c2a7c14f35c.jsonl` (one JSON event per line; the `L` is the 1-indexed line).

| Line | Speaker | Content |
| --- | --- | --- |
| L1174 | author | Pastes "From programs to partial verification" (the outline's opening) |
| L1176 | author | "compile spits out a circuit and an index over it rather than this like separate gamma thing ... I also don't know what your shapes thing is" |
| L1183 | author | "do we implement bound as something that happens at compile time, like inside of compile? Or does it run after" |
| L1200 | author | "Sorry, how do we get theta?" |
| L1202 | author | Proposes the four top-level sections "1. Bound 2. Verify 3. Strategize 4. Compile"; "they each want constructions ... maybe they are top-level sections" |
| L1204 | author | Pastes the §4.2 Architecture draft for criticism |
| L1205 | assistant | Criticism list: no "data flows Compile → Verify → Bound"; "sampling caps" is wrong; $\log_2 10^{15} \approx 50$; zkVM $10^{10}\times$ needs a referent; "advice for reduction order is untenable"; the fourth arrow |
| L1213 | author | "I agree with the advice for production orders untenable. I agree with the fourth arrow. ... I think I disagree with everything. Let me just see what you got." |
| L1214 | assistant | §4.2 architecture draft with the diagram caption |
| L1215 | author | No trust, no authorized computation; values cannot be compressed; local checks only; "design decisions" not "challenges"; maybe reorder so compilation is not first |
| L1217 | author | "G is once per epoch"; no "words"; "not sure what random-access asymmetry means" |
| L1218 | assistant | Restatement: structure cheap to look up, values expensive |
| L1232 | author | Values not bounded by width until checked; "description" is fine for gates/subcircuits; "algorithms over every gate but only a tiny percentage of the values"; wants to discuss core objects (circuit, index, transcript) without defining them early |
| L1233 | assistant | Pass is over the description, not gates; incorrect includes ill-typed |
| L1234 | author | Show architecture through the simple example?; "we call it well-typed, not well-formed" |
| L1236 | author | "doesn't really talk about scale ... we're basically only going to be talking about ML circuits" |
| L1238 | author | "I don't think we'd be able to fit 14 gates in the figure" |
| L1240 | author | Overlap with the later definition of circuit; "what's the concept here?" |
| L1244 | author | "make sure that we're building out the repo still. Do you have any questions for me?" |
| L1248 | author | "Yes to one, inputs as gate inside units, weights as... What is a session header?" |
| L1250 | author | "Is this standard in protocol implementations? ... basically just public parameters" |
| L1254 | author | "I wanted to have an algorithm that would like walk the circuit and produce what it needs to produce" |
| L1257 | author | "Can you walk me through how the compiler now will work? ... G, x) -- forget a ... recursive memoized thing that literally builds up the leaves into descriptions" |
| L1259 | author | "How do we deal with the args with absolute versus relative positioning?" |
| L1262 | author | No dependent reads; "What do you mean V-Way multiplexer?"; per-gate width "not quite yet"; "pad the length"; "carving the world at its hinges"; asks for the ontology |
| L1264 | author | "1. Public gate set 2. Kernel 3. A Circuit object"; "what is an occurrence summary?" |
| L1266 | author | "Is that not an implementation of In() and Out() and Size() and Cost()?" |
| L1268 | author | "it should also probably be C[i]"; children as canonical decomposition? |
| L1270 | author | "What the kernel loaded ... what description is concretely and how it's used" |
| L1272 | author | "This is basically like... an IR? ... like jaxpr" |
| L1274 | author | "it smells that we're defining our own intermediate representation ... off the shelf or roll our own?" |
| L1276 | author | Can the hierarchy be separated from the IR? |
| L1278 | author | "do we have to build a new hierarchical IR that Compile() knows how to read?" |
| L1280 | author | Two papers or one; deploy in labs ASAP; "is there any like universality statement" |
| L1281 | assistant | "The verifier trusts a fixed set of scalar gates and a reader for a three-construct netlist format ..." |
| L1282 | author | Quotes L1281 back; "is there some design pattern we can refer to ... I don't know if people know what a netlist is" |
| L1284 | author | "not about the trusted computing base ... design patterns that we can appeal to it's not just like 'Claude did it'" |
| L1286 | author | "state our ontology again and architecture ... carving the world at its hinges" |
| L1288 | author | "on the index rule, can you walk me through what this decision is?" |
| L1290 | author | "is `load` anything special? Could it not just be... Compile()"; G insufficient without x and a; G program-like, x data-like |
| L1292 | author | "I don't think it's important for G to be fixed before x is seen"; x can contribute to structure |
| L1294 | author | "what's an instance?"; "epoch I think is like one run of the protocol"; G and x both public; need structure to traverse efficiently |
| L1296 | author | "remind me what else needs D, except insofar as it holds C and I?" |
| L1298 | author | Nested cuts via a witness instead of messing up the ontology? |
| L1300 | author | "Can you explain to me what I ends up being then?" (the figure) |
| L1302 | author | "there's just like special kinds of sub-circuits and these are like the protocol-defined ones?" |
| L1304 | author | "What is an anti-chain in general and in this context?" |
| L1306 | author | "what is cost doing for us?" |
| L1308 | author | Should $\partial$ also be a function? |
| L1310 | author | Is this an instance of an object we already have? |
| L1312 | author | Do we need the notation? Example of using boundary |
| L1314 | author | "write out our ontology one more time" |
| L1316 | author | "Why do we need 'kernel' or `load` -- why isn't Compile() fine? ... drop D and instead just do H(C, I) ... not sure what address set is ... not sure we need Instance ... How is ρ chosen by optimize but then used in compile()" |
| L1317 | assistant | Agrees to drop kernel/load/D/Instance; explains ρ |
| L1318 | author | "walk through that operationally in terms of selecting the replay units and the verification unit ... suppose I'm a developer" |
| L1320 | author | "Are there any requirements we have of the replay units, or can these just be arbitrarily chosen?"; I contains special units and we check properties |
| L1322 | author | "Hold up, we're gonna let the client pick ... What if you just pick all your replay units to be up to generate size except for one?"; dispatch sub-agents; keep building |
| L1323 | assistant | Answer: priced, not forbidden |
| L1334 | author | "Sorry, can you quickly lay out the ontology as we currently know it?" |
| L1336 | author | "is Optimize some deterministic thing or is it not? ... trusted code or untrusted code?" |
| L1338 | author | "So then Optimize isn't a protocol method ... think about it from [the developer's] perspective" |
| L1340 | author | "No, the developer is the adversary. The only thing trusted in this system is the verifier and the channel." |
| L1347 | author | "Can you walk me through what we've learned? First state, the whole fresh ontology." |
| L1348 | assistant | The final restated ontology (§7.1 of this dump) |
| L1349 | author | Forwards the §6 agent's plan and asks for advice on the §5–§6 interface; "Do we need interior as a separate concept?" |
| L1351 | author | "How are we doing on implementing?" |
| L1357 | assistant | Audit result: 1,536 adversarial (partition, policy) cases, zero soundness violations, looseness ≤ 8 bits; boundary lemma on 2,550 random markings; the counted-backend counterexample (34 cases) and the rule "capacity classes are kinds of units, never gates" |
| L1358 | author | "Did we figure out whether it matters, whether our result is good for arbitrary partitions or no?" |
| L1359 | assistant | "The reason is structural, not just empirical. Bound has two ingredients, and neither assumes anything about unit shape." |
| L1360 | author | "Can you dump the argument for why we actually don't need to place any constraints on the partitions, except for the max U?" |
| L1361 | assistant | The no-constraints-on-partitions argument (§3.3) |
| L1368 | author | Pastes the 7.1 draft ("workload") and asks for the §7 outline with the protocol in 7.2 and a construction later |
| L1369 | assistant | The §7 outline the author endorsed (7.1–7.6 and "two things I'd resist") |
| L1373 | author | "Can you take stock of how the implementation is doing? Are there any high level design decisions that might need to be thought through again ..." |
| L1406 | author | Dispatch a sub-agent to write §7 in the style of Thaler (Notion links) |
| L1435 | author | "Sorry, why are inputs outside of every unit?" |
| L1441 | assistant | Kinds shareable across pods and steps (cluster) |
| L1453 | author | "I prefer every gate is in a unit. I want you to try to steer the code in that direction, we might change this later." |
| L1464 | author | Pastes the outline's reduction sentence ("compile G(x) into a fixed circuit C; partially verify C; return a bound on the covert capacity"); "Section 5.5 of this shows matmuls" |
| L1466, L1467 | author | "you really need to convince yourself of the security here ... a lot of different components that do not have their security properties fully proved out" |

The writing agent should not cite transcript lines in the paper; they are here so that any quotation in this dump can be checked.
