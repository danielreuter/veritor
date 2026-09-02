# 5. Covert capacity of partially incorrect circuits

The verifier of §6 checks a random sample of a circuit's gates and nothing else. This section asks how much an adversary can communicate through the gates it does not check. We fix the circuit and its input, so that an honest evaluation has exactly one output, and bound the number of distinct outputs the adversary can produce while still being accepted with probability above a threshold. The logarithm of this number is the number of bits a single response can carry; we call it the capacity of the channel, and §8 turns a bound on it into exfiltration security in the sense of §3.

Two things make the bound harder than it first appears. The adversary communicates not only through the values it writes at incorrect gates but through its choice of which gates to corrupt, and in a circuit of $10^{10}$ gates the choice can be worth more bits than the value (§4.2.3). And an incorrect value does not reach the output intact: it is read by a few gates, combined with other values, and usually discarded, so that counting the bits of every incorrect value overstates what reaches the output by orders of magnitude. We handle the first problem by counting accepted transcripts rather than incorrect values (§5.2). The second is the subject of §5.3 and §5.4: an incorrect gate affects the output only through the gates that every path from it to the output passes through, and it is the narrowest such set, not the gate itself, that the bound should count. The same construction tells us which gates the verifier may treat as one unit without weakening the bound, which is what §6 needs from this section. §5.5 shows how to compute the cuts in circuits far too large to hold in memory, §5.6 does so for GPT-2 Small, and §5.7 draws the consequences for how §6 should choose its units.

## 5.1. Circuits, transcripts, and reachable outputs

A circuit $C$ of size $n$ is a sequence of *gates* numbered $1,\ldots,n$. Gate $i$ represents a single step in a static program, computing the value

$$
v_i\leftarrow f_i(v_{j_1},\ldots,v_{j_{d_i}}).
$$

Here $v_i\in\{0,1\}^{w_i}$, where $w_i\ge1$ is the *width* of gate $i$, and $\operatorname{args}(i)=(j_1,\ldots,j_{d_i})$ is an ordered tuple of gates preceding $i$ whose values are the arguments of $f_i$. A designated set $\operatorname{in}(C)\subseteq[n]$ consists of the *input gates*; each input gate $j$ has $\operatorname{args}(j)=()$, so $f_j$ is a constant, and these constants together form the input $x$. A designated set $\operatorname{out}(C)\subseteq[n]$ consists of the *output gates*. Throughout this section $C$, and with it $x$, is fixed.

We represent an evaluation of $C$ as a *transcript* $\tau=(v_1,\ldots,v_n)$ of $n$ finite bitstrings. These values need not have been computed correctly and can have arbitrary lengths. We can think of $C$ as a system of constraints on such transcripts. We say that

- a value $v_i$ is *well-typed* if $v_i\in\{0,1\}^{w_i}$, and a transcript is well-typed if all of its values are;
- gate $i$ is *correct* in $\tau$ if each argument value $v_{j_\ell}$ is well-typed and $v_i=f_i(v_{j_1},\ldots,v_{j_{d_i}})$;
- a gate that is not correct is *incorrect*, and $\operatorname{err}(\tau)\subseteq[n]$ is the set of incorrect gates.

Exactly one transcript has $\operatorname{err}(\tau)=\varnothing$, the honest evaluation of $C$ on $x$. For $S=\{s_1<\cdots<s_k\}\subseteq[n]$ we write

$$
\tau|_S:=(v_{s_1},\ldots,v_{s_k})
$$

for the restriction of $\tau$ to $S$, so that $\tau|_{\operatorname{in}(C)}$ and $\tau|_{\operatorname{out}(C)}$ are its input and its output. For a set of gates $S$ we write $W(S):=\sum_{i\in S}w_i$ for its width; in particular $W_{\mathrm{out}}:=W(\operatorname{out}(C))$ is the width of the output.

Nothing in this section uses more about the well-typed values of a gate than that there are $2^{w_i}$ of them. The width may therefore be any positive real number and $\{0,1\}^{w_i}$ any fixed set of that size. We use this freedom once, in §5.6, for gates whose value is one of $50{,}257$ tokens.

**Verifiers.** A verifier $\mathcal V$ is a randomized procedure that reads a transcript and accepts or rejects it. We are interested in verifiers cheap enough to run on every response, which rules out reading the whole transcript, and we fix the following class. A *sampling verifier* is specified by a partition $\mathcal P$ of the non-input gates into $N$ *units* and a rate $p\in(0,1]$. Given $\tau$, it checks that the input gates are correct and the output gates well-typed; it then includes each unit independently with probability $p$ and checks that every gate of each included unit is correct; and it accepts if every check passes. (A gate is checked against the values its arguments hold in $\tau$, which may lie outside the unit; how the verifier obtains those values without reading the whole transcript is the subject of §6.) Write $\operatorname{err}_{\mathcal P}(\tau)$ for the set of units that contain an incorrect gate. A unit passes its check exactly when it is not in $\operatorname{err}_{\mathcal P}(\tau)$, so

$$
\Pr[\mathcal V(\tau)=1]=(1-p)^{|\operatorname{err}_{\mathcal P}(\tau)|}
$$

when the inputs of $\tau$ are correct and its output is well-typed, and $\Pr[\mathcal V(\tau)=1]=0$ otherwise.

Fix a threshold $\eta\in(0,1)$ and let

$$
\mathcal T_{\mathcal V}:=\{\tau:\Pr[\mathcal V(\tau)=1]>\eta\}
$$

be the set of transcripts the adversary can submit with better than an $\eta$ chance of acceptance. Let $L$ be the least integer with $(1-p)^L\le\eta$. Then $\mathcal T_{\mathcal V}$ is exactly the set of transcripts with correct inputs, a well-typed output, and fewer than $L$ incorrect units. We call $L$ the *budget* of the verifier: the adversary may corrupt fewer than $L$ units and no more. With $p=\eta=1\%$, $L=459$. Transcripts outside $\mathcal T_{\mathcal V}$ are accepted with probability at most $\eta$, and §8 accounts for them separately.

The partition into units is a parameter of the verifier. How the bound depends on it is a theme of this section; how the cost of verification depends on it is the subject of §6.5; and §5.7 brings the two together.

**Reachable outputs.** For a set of transcripts $\mathcal T$, let

$$
\mathcal Y(\mathcal T):=\{\tau|_{\operatorname{out}(C)}:\tau\in\mathcal T\}
$$

be the set of outputs it produces. Our goal is an upper bound on $|\mathcal Y(\mathcal T_{\mathcal V})|$, the number of outputs the adversary can reach; $\log_2$ of this number is the capacity. The bound must be computable, and computable far more cheaply than evaluating $C$: the circuits we care about have $10^{10}$ gates or more, and the verifier will hold a description of the circuit (§7) but never the circuit itself. §5.5 addresses this.

## 5.2. Deriving a simple bound

The most basic bound is simply the width of the circuit's output:

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\le 2^{W_{\mathrm{out}}}.
$$

This is sound because $\mathcal V$ rejects any transcript whose output is ill-typed, so every reachable output is one of the $2^{W_{\mathrm{out}}}$ well-typed ones. We refer to it as the *output bound*. In many situations we would like to improve on it. For instance, an inference datacenter could exfiltrate its model weights in [TODO: Rinberg 2025 example].

In principle, we should be able to derive a tighter bound from $\mathcal T_{\mathcal V}$, since the verifier places substantial constraints on it and $|\mathcal Y(\mathcal T_{\mathcal V})|\le|\mathcal T_{\mathcal V}|$. An immediate concern, however, is that $|\mathcal T_{\mathcal V}|$ may not itself be bounded. This is because $\mathcal T_{\mathcal V}$ may contain ill-typed transcripts, that is, transcripts in which some gate holds a value of the wrong width, possibly an arbitrarily large one. For instance, gate $500$ could be assigned a value a billion bits long, which later gates could read and then write to the output. This may not be detected because the verifier can't afford to check every value in the transcript. However, below we prove that this is not actually a problem: there are no outputs reachable by an ill-typed transcript that are not also reachable by a well-typed transcript, and thus no adversary can exploit this seeming vulnerability to its advantage. It follows that we may restrict our count to the well-typed transcripts in $\mathcal T_{\mathcal V}$, of which there are finitely many.

**Lemma 5.1.** *Let $\mathcal T_{\mathcal V}^{\mathrm{wt}}\subseteq\mathcal T_{\mathcal V}$ be the set of well-typed transcripts in $\mathcal T_{\mathcal V}$. Then $\mathcal Y(\mathcal T_{\mathcal V})=\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})$.*

*Proof.* We know that $\mathcal T_{\mathcal V}^{\mathrm{wt}}\subseteq\mathcal T_{\mathcal V}$, and hence that $\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})\subseteq\mathcal Y(\mathcal T_{\mathcal V})$. Thus it suffices to show that $\mathcal Y(\mathcal T_{\mathcal V})\subseteq\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})$. To do so, we will show that for every $\tau\in\mathcal T_{\mathcal V}$ there exists a $\tau'\in\mathcal T_{\mathcal V}^{\mathrm{wt}}$ such that $\tau'|_{\operatorname{out}(C)}=\tau|_{\operatorname{out}(C)}$. We will do so by construction. Let $\tau=(v_i)_i\in\mathcal T_{\mathcal V}$ and define $\tau'=(v_i')_i$ by

$$
v_i':=\begin{cases}v_i & \text{if } v_i\in\{0,1\}^{w_i},\\[2pt] 0^{w_i} & \text{otherwise.}\end{cases}
$$

Then $\tau'$ is well-typed, since each $v_i'$ is either $v_i$ or a string of $w_i$ zeros, both of which are well-typed. It remains to show that $\tau'\in\mathcal T_{\mathcal V}$ and that $\tau'|_{\operatorname{out}(C)}=\tau|_{\operatorname{out}(C)}$. Consider any gate $i$ that is correct in $\tau$. Its value lies in $\{0,1\}^{w_i}$, since $f_i$ takes values there, and each of its arguments $v_j$ lies in $\{0,1\}^{w_j}$, by the definition of correctness; so $v_i'=v_i$ and $v_j'=v_j$ for every argument $j$, and gate $i$ is correct in $\tau'$. Hence $\operatorname{err}_{\mathcal P}(\tau')\subseteq\operatorname{err}_{\mathcal P}(\tau)$, and $\tau'$ has fewer than $L$ incorrect units. The input gates of $\tau$ are correct and its output gates are well-typed, so none of them is overwritten: $\tau'$ has correct inputs and $\tau'|_{\operatorname{out}(C)}=\tau|_{\operatorname{out}(C)}$. Thus $\tau'\in\mathcal T_{\mathcal V}$. $\square$

This proof demonstrates the necessity of our requirement from §5.1 that a correct gate have well-typed arguments. Without it, the adversary could write a million bits at gate $500$ at the cost of one incorrect gate and have a thousand correct gates each read a different $16$-bit slice of it, so that a single error would carry $16{,}000$ bits to the output.

We are now in a position to bound $|\mathcal Y(\mathcal T_{\mathcal V})|$ by counting the elements of $\mathcal T_{\mathcal V}^{\mathrm{wt}}$.

**Proposition 5.2.** *Let $\mathcal V$ be a sampling verifier with units $\mathcal P$, $N:=|\mathcal P|$, and budget $L$, and let $W_{\max}$ be the largest width of a unit in $\mathcal P$. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{\ell<L}\binom{N}{\ell}\,2^{\ell W_{\max}}.
$$

*Proof.* We know from Lemma 5.1 that $|\mathcal Y(\mathcal T_{\mathcal V})|=|\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})|\le|\mathcal T_{\mathcal V}^{\mathrm{wt}}|$. Thus it suffices to show that $|\mathcal T_{\mathcal V}^{\mathrm{wt}}|\le\sum_{\ell<L}\binom N\ell 2^{\ell W_{\max}}$. To do so, we will show that every $\tau\in\mathcal T_{\mathcal V}^{\mathrm{wt}}$ is determined by the pair $(S,\tau|_S)$, where $S:=\operatorname{err}_{\mathcal P}(\tau)$ and $\tau|_S$ is the restriction of $\tau$ to the gates of the units in $S$, and then count the possible pairs. Let $\tau\in\mathcal T_{\mathcal V}^{\mathrm{wt}}$ and consider its gates in index order. An input gate holds its input value, since the inputs of $\tau$ are correct. A gate in a unit of $S$ holds the value recorded in $\tau|_S$. Any other gate $i$ is correct, so $v_i$ is $f_i$ applied to the values of its arguments, all of which precede $i$. Thus each $v_i$ is determined by $(S,\tau|_S)$ and the earlier values, and by induction so is all of $\tau$. It remains to count the pairs. There are $\binom N\ell$ choices of $S$ with $|S|=\ell<L$, and since $\tau$ is well-typed, at most $\prod_{U\in S}2^{W(U)}\le2^{\ell W_{\max}}$ choices of $\tau|_S$ for each. $\square$

**Corollary 5.3.** *For $1\le L\le N$,*

$$
\log_2|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ L\Bigl(W_{\max}+\log_2\frac{eN}{L}\Bigr).
$$

*Proof.* Each term of the sum in Proposition 5.2 has $\ell<L$, so $2^{\ell W_{\max}}\le2^{LW_{\max}}$, and $\sum_{\ell<L}\binom N\ell\le\sum_{\ell\le L}\binom N\ell\le(eN/L)^L$ by the standard estimate. $\square$

The corollary is the form worth remembering: each incorrect unit costs the adversary at most $W_{\max}$ bits for its value and $\log_2(eN/L)$ bits for its position. We refer to the two parts as the *value term* and the *position term*. In the next section we investigate the tightness of the bound for the case of a matrix multiplication.

## 5.3. Testing the bound on a matrix multiplication

To see how loose Proposition 5.2 can be, consider the most common circuit in the world, that of a matrix multiplication. To give the circuit some depth, let $C$ compute a chain of $k$ matrix-vector products $x_t=A_t\,x_{t-1}$, $t=1,\ldots,k$, where the vector $x_0\in(\{0,1\}^{16})^d$ and the matrices $A_1,\ldots,A_k\in(\{0,1\}^{16})^{d\times d}$ are inputs. Each entry of $x_t$ is a dot product of length $d$, computed by $d$ multiplication gates and $d-1$ addition gates, all of width $16$. (Which $16$-bit arithmetic the gates perform is immaterial here; only the widths of their results matter.) The output gates are the $d$ gates holding $x_k$, so $W_{\mathrm{out}}=16d$. Take each dot product as a unit, which is the unit §6 uses for this circuit; then $N=kd$ and every unit has width $W(U)=16(2d-1)$.

Let $d=1{,}024$, $k=100$, and $p=\eta=1\%$, so that $N=102{,}400$, $W_{\max}=32{,}752$, and $L=459$. The value term in Corollary 5.3 is $459\cdot32{,}752\approx1.5\times10^{7}$ bits and the position term is $459\log_2\frac{e\cdot102{,}400}{459}\approx4{,}200$ bits, against $W_{\mathrm{out}}=16{,}384$. Proposition 5.2 is worse than the output bound by a factor of nearly a thousand, and the entire excess is in the value term.

The reason is that Proposition 5.2 counts transcripts, and here a great many transcripts have the same output. Nothing outside a dot product reads its products or its partial sums; the only value in the unit that any other gate reads is the final sum, which is $16$ bits wide. Two well-typed assignments to an incorrect unit that agree on the final sum therefore produce the same output, so the $2^{32{,}752}$ assignments Proposition 5.2 counts produce at most $2^{16}$ distinct outputs. If $W_{\max}$ could be replaced by $16$ in Corollary 5.3, the bound would be $459\cdot(16+9.2)\approx11{,}600$ bits. This is not far from what the adversary can in fact do: by corrupting $458$ dot products in the last layer it can set $458$ entries of $x_k$ to whatever it likes, which gives it $458\cdot16\approx7{,}300$ bits from the values alone and about $8{,}300$ once the choice of entries is counted.

The fact we used is that every path from a gate of a dot product to an output gate passes through the dot product's final addition. In the last layer this is obvious, since the final addition is itself an output gate. But it holds in every layer, and there it is not obvious at all: a corrupted dot product in layer $1$ changes every entry of $x_k$, yet the whole of that change is a function of the one $16$-bit value passed to layer $2$, so it too has at most $2^{16}$ distinct effects on the output. The next section shows that this is all that is needed: in Corollary 5.3, $W_{\max}$ may be replaced by the width of any set of gates that every path from a unit to the output must pass through.

## 5.4. Bounding capacity via the narrowest downstream cut

### 5.4.1. Downstream cuts determine outputs

In §5.3 we saw why Proposition 5.2 is loose: it counts every value inside an incorrect unit, but the rest of the circuit reads only the value the unit passes on. Here we make this observation precise and general. We define a *downstream cut* for a set of gates $A$ to be a set of gates that every path from $A$ to the output passes through, prove that the values on a downstream cut for the incorrect gates determine the output, and conclude that in Proposition 5.2 the width of a unit may be replaced by the width of its narrowest downstream cut.

The value at one gate can affect the value at a later gate only if the former is an argument of the latter, or an argument of one of its arguments, and so on. We call such a chain a path.

**Definition 5.4 (Path).** *A path from a gate $a$ to a gate $z$ is a sequence of gates $a=j_0,j_1,\ldots,j_t=z$ with $j_s\in\operatorname{args}(j_{s+1})$ for every $s<t$; we allow $t=0$. For $A,Z\subseteq[n]$, a path from $A$ to $Z$ is a path from some $a\in A$ to some $z\in Z$.*

Since arguments precede the gates that read them, the gates of a path are distinct and increase in index.

**Definition 5.5 (Downstream cut).** *Let $A,Z,D\subseteq[n]$. We say that $D$ cuts $A$ from $Z$ if every path from $A$ to $Z$ contains a gate of $D$. A downstream cut for $A$ is a set of gates that cuts $A$ from $\operatorname{out}(C)$.*

Both $A$ itself and $\operatorname{out}(C)$ are downstream cuts for $A$, since every path from $A$ to the output begins in the one and ends in the other. The useful cuts lie in between; for a dot product in §5.3, its final addition is a downstream cut of width $16$. Note that a path of length zero is a path, so a downstream cut for $A$ must contain every output gate in $A$.

An incorrect value can affect a gate only along a path, and a cut intercepts every path. So the values on a set that cuts the incorrect gates from $Z$ should determine the values at $Z$.

**Theorem 5.6 (Downstream cuts determine outputs).** *Let $\tau$ and $\tau'$ be transcripts of $C$, let $E:=\operatorname{err}(\tau)\cup\operatorname{err}(\tau')$ be the set of gates incorrect in either, let $Z\subseteq[n]$, and let $D$ cut $E$ from $Z$. If $\tau|_D=\tau'|_D$, then $\tau|_Z=\tau'|_Z$.*

*Proof.* Let $\Delta:=\{i\in[n]:v_i\neq v'_i\}$ be the set of gates on which the transcripts disagree. By hypothesis $\Delta$ contains no gate of $D$. Suppose for contradiction that $\Delta$ contains a gate of $Z$. We will construct a path from $E$ to $Z$ lying entirely in $\Delta$. Since $D$ cuts $E$ from $Z$, this path contains a gate of $D$, which contradicts $\Delta\cap D=\varnothing$.

We construct the path backward from $Z$. First observe that every gate $i\in\Delta\setminus E$ has an argument in $\Delta$: such a gate is correct in both transcripts, so if the transcripts agreed on all of its arguments, applying $f_i$ to the same values would give $v_i=v'_i$, contradicting $i\in\Delta$. (In particular an input gate, having no arguments, cannot lie in $\Delta\setminus E$.) Now let $j_0\in\Delta\cap Z$, and as long as $j_r\notin E$, choose $j_{r+1}\in\operatorname{args}(j_r)\cap\Delta$. Each step decreases the gate index, so the process terminates, and it can terminate only at a gate $j_m\in E$. Reversing the sequence gives a path $j_m,\ldots,j_0$ from $E$ to $Z$ consisting of gates in $\Delta$, as required. $\square$

We will use the theorem with $Z=\operatorname{out}(C)$ until §5.5, where a different target set is needed.

**Corollary 5.7.** *Let $\mathcal T$ be a set of well-typed transcripts, and let $D$ be a downstream cut for $\operatorname{err}(\tau)$ for every $\tau\in\mathcal T$. Then $|\mathcal Y(\mathcal T)|\le2^{W(D)}$.*

*Proof.* Let $\tau,\tau'\in\mathcal T$. Then $D$ is a downstream cut for $\operatorname{err}(\tau)\cup\operatorname{err}(\tau')$, since every path from the union begins in one of the two sets. By Theorem 5.6, if $\tau|_D=\tau'|_D$ then $\tau$ and $\tau'$ have the same output. So the output of a transcript in $\mathcal T$ is determined by its values on $D$, and since the transcripts are well-typed there are at most $2^{W(D)}$ possibilities for those. $\square$

We can now make the replacement promised in §5.3. For a set of gates $A$, let

$$
\kappa(A):=\min\{W(D):D\text{ is a downstream cut for }A\}
$$

be the width of the narrowest downstream cut for $A$. Since $A$ is a downstream cut for itself, $\kappa(A)\le W(A)$, and since $\operatorname{out}(C)$ is one, $\kappa(A)\le W_{\mathrm{out}}$. For a sampling verifier with units $\mathcal P$, let $\kappa_{\max}$ be the largest $\kappa(U)$ over $U\in\mathcal P$; then $\kappa_{\max}\le W_{\max}$.

**Proposition 5.8.** *Let $\mathcal V$, $\mathcal P$, $N$, and $L$ be as in Proposition 5.2. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{\ell<L}\binom{N}{\ell}\,2^{\ell\kappa_{\max}},
$$

*and consequently Corollary 5.3 holds with $\kappa_{\max}$ in place of $W_{\max}$.*

*Proof.* By Lemma 5.1 it suffices to bound $|\mathcal Y(\mathcal T^{\mathrm{wt}}_{\mathcal V})|$. For a set $S$ of units, let $\mathcal T_S$ be the set of $\tau\in\mathcal T^{\mathrm{wt}}_{\mathcal V}$ with $\operatorname{err}_{\mathcal P}(\tau)=S$. The sets $\mathcal T_S$ with $|S|<L$ cover $\mathcal T^{\mathrm{wt}}_{\mathcal V}$, so it suffices to show that $|\mathcal Y(\mathcal T_S)|\le2^{|S|\kappa_{\max}}$ for each such $S$. For each $U\in S$ choose a downstream cut $D_U$ for $U$ of width $\kappa(U)$, and let $D_S:=\bigcup_{U\in S}D_U$. Every incorrect gate of a transcript in $\mathcal T_S$ lies in some $U\in S$, since its inputs are correct, and every path from it to the output contains a gate of $D_U\subseteq D_S$. So $D_S$ is a downstream cut for $\operatorname{err}(\tau)$ for every $\tau\in\mathcal T_S$, and Corollary 5.7 gives $|\mathcal Y(\mathcal T_S)|\le2^{W(D_S)}\le2^{\sum_{U\in S}\kappa(U)}\le2^{|S|\kappa_{\max}}$. $\square$

The two bounds of §5.2 are the extreme cases of this argument: taking $D_U=U$ gives Proposition 5.2, and taking $D=\operatorname{out}(C)$ in Corollary 5.7 gives the output bound. For the circuit of §5.3, the final addition of a dot product is a downstream cut for it, so $\kappa(U)=16$ for every unit and Corollary 5.3 with $\kappa_{\max}$ in place of $W_{\max}$ gives about $11{,}600$ bits, against $1.5\times10^{7}$ from Proposition 5.2 and $16{,}384$ from the output. The attack of §5.3, which sets up to $458$ entries of $x_k$ at will, reaches $\sum_{\ell<L}\binom{1{,}024}{\ell}(2^{16}-1)^{\ell}$ outputs, about $8{,}300$ bits, so the bound is within a factor of $1.4$ of the truth. The value terms agree exactly. The gap is in the position term, which counts positions among all $N=kd$ dot products where the attack uses only the last $d$; whether corrupting earlier layers yields further distinct outputs depends on the matrices, since a corruption of $x_t$ reaches $x_k$ only through $A_k\cdots A_{t+1}$.

### 5.4.2. Grouping gates by their narrowest downstream cut

Proposition 5.8 depends on the partition into units through $\kappa_{\max}$ and $N$, and it is worth asking how. Merging two units into one never lowers $\kappa_{\max}$: a downstream cut for the merged unit is a downstream cut for each part, so its $\kappa$ is at least that of either part, and it can be far larger. A unit consisting of a whole layer of the circuit of §5.3 has $\kappa=16d$, the width of the layer's output, in place of $16$. But merging is sometimes free. The gates inside one dot product all have the same narrowest downstream cut, its final addition, and any set of them has $\kappa=16$, whether it is one gate or all $2d-1$. In this subsection we make this precise by assigning each gate a single cut and grouping the gates that share one. The groups, which we call regions, are the coarsest units the verifier can use without loss. They also improve the position term: every unit inside a region has the same cut, so the adversary gains nothing from its choice of unit within a region, and the position term should count regions rather than units.

A gate may have several narrowest downstream cuts. In the circuit of §5.3, the narrowest cuts of a multiplication gate in a dot product are the gate itself, each partial sum after it, and the final addition, all of width $16$. To assign each gate one cut we take the one furthest downstream, which groups the whole dot product behind its final addition; taking the nearest would make every gate its own group. The following order makes "furthest downstream" precise.

**Definition 5.9.** *Let $D$ and $D'$ be downstream cuts for $A$. We say that $D'$ is downstream of $D$ if $D'$ is a downstream cut for $D$, that is, if every path from $D$ to the output contains a gate of $D'$.*

Every downstream cut is downstream of itself, and if $D''$ is downstream of $D'$ and $D'$ downstream of $D$ then $D''$ is downstream of $D$, since a path from $D$ to the output contains a gate of $D'$ and its suffix from that gate contains a gate of $D''$.

**Lemma 5.10.** *Let $A$ be a set of gates. Among the narrowest downstream cuts for $A$ there is exactly one that is downstream of every other. We call it the downstream-most narrowest cut for $A$ and denote it $D(A)$.*

*Proof.* We show here that there is at most one; that there is at least one is Lemma 5.16 in §5.5, where the construction that computes $D(A)$ also shows it exists. Two cuts each downstream of every other are each downstream of the other, so it suffices to show that no two distinct narrowest downstream cuts for $A$ are each downstream of the other. Suppose $D_1\neq D_2$ were such a pair, and choose $d\in D_1\setminus D_2$, exchanging the names if necessary. We claim that $D_1\setminus\{d\}$ is still a downstream cut for $A$, which contradicts the minimality of $W(D_1)$ because $w_d>0$. Let $P$ be a path from $A$ to the output. If $d\notin P$ then $P$ contains a gate of $D_1$ other than $d$. If $d\in P$, the suffix of $P$ from $d$ is a path from $D_1$ to the output, so it contains a gate $d_2\in D_2$, which comes strictly after $d$ because $d\notin D_2$; and the suffix from $d_2$ is a path from $D_2$ to the output, so it contains a gate $d_1\in D_1$ at or after $d_2$, hence strictly after $d$. Since the gates of a path are distinct, $d_1\neq d$, and $P$ contains a gate of $D_1\setminus\{d\}$. $\square$

**Definition 5.11 (Regions).** *For a non-input gate $g$, the cut of $g$ is $D(g):=D(\{g\})$. For a set of gates $D$, the region of $D$ is $A_D:=\{g\notin\operatorname{in}(C):D(g)=D\}$. The nonempty regions partition the non-input gates; we write $R$ for their number.*

A gate with no path to the output has $\varnothing$ as its only narrowest downstream cut, so $D(g)=\varnothing$; such gates, if there are any, form one region, of width $0$. Every other gate has $\kappa(g)\ge w_{\min}$, the least width of a gate, since its cut is nonempty.

**Lemma 5.12 (Regions are lossless).** *Let $A_D$ be a region and let $U\subseteq A_D$ be nonempty. Then $D$ is a narrowest downstream cut for $U$; in particular $\kappa(U)=W(D)$.*

*Proof.* $D$ is a downstream cut for every $g\in U$, hence for $U$, so $\kappa(U)\le W(D)$. Conversely, a downstream cut for $U$ is a downstream cut for any $g\in U$, so its width is at least $\kappa(g)=W(D)$. $\square$

So a unit inside a region has the region's cut, however many gates it contains: within a region, refining the units gains nothing and merging them loses nothing. Across regions, merging does lose, as the layer example above shows. This is the granularity statement promised in §4.2.3; §5.7 draws its consequences. Its counterpart for the position term is the following.

**Proposition 5.13.** *Let $\mathcal V$ be a sampling verifier with budget $L$ each of whose units lies inside a region, and let $R$ be the number of regions. For a set $Q$ of regions write $D_Q$ for the union of their cuts. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{\ell<L}\ \sum_{|Q|=\ell}2^{W(D_Q)}\ \le\ \sum_{\ell<L}\binom{R}{\ell}\,2^{\ell\kappa_{\max}},
$$

*where the inner sum is over sets of $\ell$ regions.*

*Proof.* By Lemma 5.1 it suffices to bound $|\mathcal Y(\mathcal T^{\mathrm{wt}}_{\mathcal V})|$. For $\tau\in\mathcal T^{\mathrm{wt}}_{\mathcal V}$ let $Q(\tau)$ be the set of regions that contain an incorrect unit of $\tau$. Each unit lies in one region, so $|Q(\tau)|\le|\operatorname{err}_{\mathcal P}(\tau)|<L$, and the sets $\mathcal T_Q:=\{\tau\in\mathcal T^{\mathrm{wt}}_{\mathcal V}:Q(\tau)=Q\}$ with $|Q|<L$ cover $\mathcal T^{\mathrm{wt}}_{\mathcal V}$. Fix $Q$. Every incorrect gate of a transcript in $\mathcal T_Q$ lies in a region $A_D$ with $D\in Q$, and $D\subseteq D_Q$ is a downstream cut for it; so $D_Q$ is a downstream cut for $\operatorname{err}(\tau)$ for every $\tau\in\mathcal T_Q$, and Corollary 5.7 gives $|\mathcal Y(\mathcal T_Q)|\le2^{W(D_Q)}$. Summing over $Q$ gives the first inequality. For the second, $W(D_Q)\le\sum_{A_D\in Q}W(D)\le|Q|\,\kappa_{\max}$, because each region contains a unit and by Lemma 5.12 that unit has $\kappa(U)=W(D)$; and there are $\binom{R}{\ell}$ sets of $\ell$ regions. $\square$

**Corollary 5.14.** *Under the hypotheses of Proposition 5.13, for $1\le L\le R$,*

$$
\log_2|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ L\Bigl(\kappa_{\max}+\log_2\frac{eR}{L}\Bigr).
$$

*Proof.* As for Corollary 5.3. $\square$

Since the units refine the regions, $R\le N$, and the corollary improves on Corollary 5.3 in both terms. For the circuit of §5.3 the regions are exactly the dot products, so nothing changes there. For GPT-2 Small the regions are about $1{,}300$ times fewer than the gates, and §5.6 gives the numbers.

*Remark (other verifiers).* The proofs of Propositions 5.8 and 5.13 use the verifier only through the family of sets of incorrect units it admits, namely those of size less than $L$. Consider any verifier that checks the inputs and the output as ours does and otherwise accepts $\tau$ with a probability $\sigma(\operatorname{err}_{\mathcal P}(\tau))$ that depends only on which units are incorrect. The same argument gives

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{E:\ \sigma(E)>\eta}2^{\kappa(E)},
$$

where $E$ ranges over sets of units and $\kappa(E)$ is the width of the narrowest downstream cut for their gates, which is at most the sum of the cut widths of the regions they meet, since the union of those cuts is a downstream cut for them. The two-stage sampler of §6, which samples replay units at rate $q$ and verification units inside them at rate $s$, has $\sigma(E)=\prod_r\bigl(1-q+q(1-s)^{|E\cap R_r|}\bigr)$, and §6 uses the bound in this form.

## 5.5. Computing narrowest downstream cuts

Propositions 5.8 and 5.13 are only as useful as our ability to compute $\kappa$ and $D$. In §5.5.1 we compute the cut of one set of gates by a maximum-flow computation, which also proves the existence half of Lemma 5.10. In §5.5.2 we give three lemmas that determine the cuts of nearly all gates of an inference circuit by inspection of their wiring, with no flow computation at all. In §5.5.3 we use the repetition in such circuits to do this for all $4\times10^{10}$ gates of GPT-2 Small at once.

### 5.5.1. One set of gates at a time

Finding a narrowest downstream cut is a minimum cut problem in which the capacities sit on the gates rather than on the edges between them. The standard reduction splits each gate in two and moves its width onto the edge between the halves.

Let $M:=1+\sum_{i\in[n]}w_i$. The network $G_A$ has a source $s$, a sink $t$, and two vertices $i_{\mathrm{in}}$ and $i_{\mathrm{out}}$ for each gate $i$. Its edges are: $i_{\mathrm{in}}\to i_{\mathrm{out}}$ of capacity $w_i$ for each gate $i$, which we call the internal edge of $i$; $j_{\mathrm{out}}\to i_{\mathrm{in}}$ of capacity $M$ for each $i$ and each $j\in\operatorname{args}(i)$; $s\to a_{\mathrm{in}}$ of capacity $M$ for each $a\in A$; and $o_{\mathrm{out}}\to t$ of capacity $M$ for each $o\in\operatorname{out}(C)$. A path $a=j_0,\ldots,j_t$ from $A$ to the output lifts to the $s$–$t$ path $s,(j_0)_{\mathrm{in}},(j_0)_{\mathrm{out}},\ldots,(j_t)_{\mathrm{out}},t$, and since the edges of $G_A$ alternate between internal edges and edges from an out-vertex to an in-vertex, every $s$–$t$ path is the lift of a path from $A$ to the output. An $s$–$t$ cut is specified by its source side $S$, a set of vertices with $s\in S$ and $t\notin S$; its capacity is the total capacity of the edges leaving $S$. We write

$$
D(S):=\{i\in[n]:i_{\mathrm{in}}\in S\text{ and }i_{\mathrm{out}}\notin S\}
$$

for the set of gates whose internal edge leaves $S$.

**Proposition 5.15.** *The minimum capacity of an $s$–$t$ cut in $G_A$ is $\kappa(A)$, and if $S$ is the source side of a minimum cut then $D(S)$ is a narrowest downstream cut for $A$.*

*Proof.* Let $c$ be the minimum cut capacity. The source side consisting of $s$ and every in-vertex has as its leaving edges exactly the internal edges, so $c\le\sum_i w_i<M$, and no edge of capacity $M$ leaves the source side of a minimum cut.

Let $S$ be the source side of a minimum cut. The lift of a path from $A$ to the output starts at $s\in S$ and ends at $t\notin S$, so some edge of it leaves $S$; that edge has capacity less than $M$, so it is the internal edge of a gate of $D(S)$, and that gate lies on the path. Thus $D(S)$ is a downstream cut for $A$. The edges leaving $S$ are exactly the internal edges of the gates of $D(S)$, so $\kappa(A)\le W(D(S))=c$.

Conversely, let $D$ be any downstream cut for $A$, and let $S_D$ be the set of vertices reachable from $s$ without using an internal edge of a gate of $D$. Then $t\notin S_D$, since a path from $s$ to $t$ avoiding those edges would be the lift of a path from $A$ to the output avoiding $D$. Every edge leaving $S_D$ is an internal edge of a gate of $D$, for otherwise its head would be reachable too. So the cut with source side $S_D$ has capacity at most $W(D)$, and $c\le W(D)$. Minimizing over $D$ gives $c\le\kappa(A)$, hence $c=\kappa(A)$, and $W(D(S))=\kappa(A)$ for every minimum cut $S$. $\square$

The network has $2n+2$ vertices and $n+m+|A|+|\operatorname{out}(C)|$ edges, where $m$ is the number of argument occurrences in $C$, so one query is one maximum-flow computation on a graph of size $O(n+m)$.

The minimum cut is not unique in general, and different minimum cuts give different narrowest downstream cuts; on a chain of equal-width gates, every gate is one. The source sides of the minimum cuts of a network are closed under union and intersection [TODO: cite Ford–Fulkerson 1962, or Picard–Queyranne 1980], so there is a largest one, and it is the one we want. We give the direct argument, which also says how to compute it.

**Lemma 5.16.** *Fix a maximum flow in $G_A$ and let $S^{\max}$ be the set of vertices from which $t$ cannot be reached in the residual graph. Then $S^{\max}$ is the source side of a minimum cut and contains the source side of every minimum cut, and $D(S^{\max})$ is downstream of every narrowest downstream cut for $A$. In particular the cut $D(A)$ of Lemma 5.10 exists, and $D(A)=D(S^{\max})$.*

*Proof.* For any source side $S$, the value of the flow is the flow on the edges leaving $S$ minus the flow on the edges entering $S$. This is at most the capacity of the cut, with equality exactly when every edge leaving $S$ is saturated and every edge entering $S$ carries no flow, that is, when no residual edge leaves $S$. Since the flow is maximum, its value is the minimum cut capacity, and so $S$ is the source side of a minimum cut if and only if no residual edge leaves $S$.

The set $S^{\max}$ contains $s$, because a maximum flow admits no augmenting path, and does not contain $t$. No residual edge leaves $S^{\max}$: from the head of such an edge $t$ would be reachable, and then also from its tail. So $S^{\max}$ is the source side of a minimum cut. If $S$ is the source side of any minimum cut, then no residual edge leaves $S$, so no residual path from a vertex of $S$ can reach $t\notin S$, and $S\subseteq S^{\max}$.

Now let $D'$ be any narrowest downstream cut for $A$, and let $S_{D'}$ be the source side constructed from it in the proof of Proposition 5.15. Its cut has capacity at most $W(D')=\kappa(A)$, so it is a minimum cut, and $S_{D'}\subseteq S^{\max}$. We claim that $d_{\mathrm{in}}\in S_{D'}$ for every $d\in D'$. Since $D'$ is narrowest and $w_d>0$, the set $D'\setminus\{d\}$ is not a downstream cut for $A$, so some path from $A$ to the output meets $D'$ only at $d$; the lift of its prefix ending at $d$ reaches $d_{\mathrm{in}}$ from $s$ using no internal edge of a gate of $D'$. Finally, let $P$ be a path from $D'$ to the output, starting at some $d\in D'$. Its lift starts at $d_{\mathrm{in}}\in S^{\max}$ and ends at $t\notin S^{\max}$, so some edge of it leaves $S^{\max}$; as $S^{\max}$ is the source side of a minimum cut, that edge has capacity less than $M$, so it is the internal edge of a gate of $D(S^{\max})$ lying on $P$. Thus $D(S^{\max})$ is a downstream cut for $D'$, that is, downstream of $D'$. By Proposition 5.15 it is itself a narrowest downstream cut for $A$, so it is the cut of Lemma 5.10. $\square$

So $D(A)$ is computed by one maximum flow followed by one search backward from $t$ in the residual graph. (The set of vertices reachable from $s$ in the residual graph gives, symmetrically, the narrowest cut nearest to $A$.) Whenever we speak of *the* cut of a set of gates, we mean $D(A)$.

### 5.5.2. Locality and two rules that settle most gates

One maximum flow per gate is not an algorithm at $4\times10^{10}$ gates, and it is unnecessary. Three observations reduce the work, for inference circuits, to a check of the wiring of a few dozen kinds of gate.

The first is that hypotheses about paths to the output can be verified inside a small part of the circuit. For a set of gates $B$, let $\operatorname{Out}(B)$ be the set of gates of $B$ that are output gates or are arguments of gates outside $B$: the gates through which $B$ is read from outside.

**Lemma 5.17 (Locality).** *Let $A\subseteq B\subseteq[n]$. If $D$ cuts $A$ from $\operatorname{Out}(B)$, then $D$ is a downstream cut for $A$.*

*Proof.* Let $P$ be a path from $A$ to the output. If every gate of $P$ lies in $B$, its last gate is an output gate in $B$, hence in $\operatorname{Out}(B)$. Otherwise let $h$ be the first gate of $P$ outside $B$; the gate before $h$ on $P$ lies in $B$ and is an argument of $h$, hence in $\operatorname{Out}(B)$. Either way $P$ has a prefix that is a path from $A$ to $\operatorname{Out}(B)$, and $D$ contains a gate of it. $\square$

In particular, to show that every path from $A$ to the output passes through a gate $o\in B$, it suffices to check the paths from $A$ to $\operatorname{Out}(B)$; and the maximum-flow computation of §5.5.1, run on $B$ alone with $\operatorname{Out}(B)$ in place of $\operatorname{out}(C)$, returns a downstream cut for $A$, which is sound for Proposition 5.8 even when it is not narrowest. Theorem 5.6 with $Z=\operatorname{Out}(B)$ is the corresponding statement about values: two transcripts that agree on a set cutting their incorrect gates from $\operatorname{Out}(B)$ agree on everything $B$ passes on.

The second observation is that widths bound the size of a cut.

**Lemma 5.18.** *Let $w_{\min}$ be the least width of a gate of $C$. A narrowest downstream cut for a gate $g$ has at most $\lfloor w_g/w_{\min}\rfloor$ gates.*

*Proof.* Since $\{g\}$ is a downstream cut for $g$, a narrowest one $D$ has $|D|\,w_{\min}\le W(D)\le w_g$. $\square$

In the GPT-2 circuit of §5.6 every gate has width at least $\log_2 50{,}257\approx15.6$ and at most $32$, so the cut of a gate $g$ has at most two gates: it is empty, or $g$ itself, or another single gate that every path from $g$ to the output passes through, or a pair of gates such that every such path passes through one of them.

The third observation is that two local rules decide, for almost every gate, which of these it is. The first rule says that a set of gates whose only way to the output is through one gate, no wider than anything before it, shares that gate's cut.

**Lemma 5.19 (Exit rule).** *Let $S$ be a set of gates, each with a path to the output, and let $o$ be a gate such that every path from $S$ to the output contains $o$, and every gate that precedes $o$ on such a path has width at least $w_o$. Then $\kappa(g)=\kappa(o)$ and $D(g)=D(o)$ for every $g\in S$.*

*Proof.* Fix $g\in S$. Every downstream cut for $o$ is a downstream cut for $g$, since every path from $g$ to the output contains $o$ and therefore its suffix from $o$. Hence $\kappa(g)\le\kappa(o)\le w_o$.

Now let $D$ be any narrowest downstream cut for $g$. We claim that either $D$ is a narrowest downstream cut for $o$, or $D=\{d\}$ for a single gate $d$ that precedes $o$ and $\{o\}$ is a narrowest downstream cut for both $g$ and $o$. If $D$ is a downstream cut for $o$, then $\kappa(o)\le W(D)=\kappa(g)\le\kappa(o)$ and the first alternative holds. Otherwise some path $P$ from $o$ to the output avoids $D$. Let $Q$ be a path from $g$ to $o$; one exists because a path from $g$ to the output contains $o$. The concatenation of $Q$ and $P$ is a path from $g$ to the output, so it contains a gate $d\in D$, and $d$ lies on $Q$ before $o$ since $P$ avoids $D$. By hypothesis $w_d\ge w_o$, so $\kappa(g)=W(D)\ge w_o\ge\kappa(o)\ge\kappa(g)$. Thus all of these are equal, $D=\{d\}$, and $\{o\}$ is a narrowest downstream cut for $o$ and for $g$, which is the second alternative. In either case $\kappa(g)=\kappa(o)$.

In the second alternative, $\{o\}$ is downstream of $D=\{d\}$: a path from $d$ to the output, preceded by the part of $Q$ up to $d$, is a path from $g$ to the output and so contains $o$, which is not on $Q$ before $d$; hence $o$ lies on the path from $d$. Now $D(o)$ is a narrowest downstream cut for $o$, hence for $g$, so $D(g)$ is downstream of $D(o)$. If $D(g)$ were of the second kind, then $\{o\}$, a narrowest downstream cut for $g$, would be downstream of $D(g)$ while $D(g)$ is downstream of $\{o\}$, and the uniqueness in Lemma 5.10 would give $D(g)=\{o\}$, contradicting that $d$ precedes $o$. So $D(g)$ is a narrowest downstream cut for $o$, hence $D(o)$ is downstream of $D(g)$, and by the same uniqueness $D(g)=D(o)$. $\square$

The second rule says that a gate whose influence spreads is its own cut.

**Lemma 5.20 (Fan-out rule).** *Let $g$ be a gate with a path to the output such that no gate other than $g$ lies on every path from $g$ to the output, and such that every downstream cut for $g$ with two or more gates has width greater than $w_g$. Then $D(g)=\{g\}$ and $\kappa(g)=w_g$.*

*Proof.* $\{g\}$ is a downstream cut for $g$ of width $w_g$. Let $D\neq\{g\}$ be another. $D$ is nonempty because $g$ has a path to the output; if $|D|\ge2$ then $W(D)>w_g$ by hypothesis; and if $D=\{d\}$ with $d\neq g$ then $d$ lies on every path from $g$ to the output, contrary to hypothesis. So $\{g\}$ is the unique narrowest downstream cut for $g$. $\square$

The first hypothesis of Lemma 5.20 holds whenever two paths from $g$ to the output share no gate but $g$, for instance two paths to different output gates. The second holds whenever $w_g<2w_{\min}$, since a cut of two or more gates has width at least $2w_{\min}$; in the GPT-2 circuit this covers every $16$-bit gate. For a $32$-bit gate it fails only if some pair of gates of total width at most $32$ is a downstream cut for $g$, which requires two tokens ($31.2$ bits) or two $16$-bit values ($32$ bits, a tie that Lemma 5.10 resolves in favour of the pair, since the pair is downstream of $\{g\}$). §5.6 shows that this happens for $176{,}038$ of the $4\times10^{10}$ gates.

Between them the two rules settle the gates of an inference circuit as follows. Inside an inner product every product and partial sum has a single exit, the value written out at the end, and is at least as wide as it; by Lemma 5.19 the whole inner product has the cut of its write-out. The write-out is read by many gates, so by Lemma 5.20 it is its own cut, unless it feeds a single gate, in which case Lemma 5.19 passes the cut on. The same two steps place every reduction behind the statistic it computes and every logit of a language model behind the token chosen from it. What the rules leave undecided is a handful of gates whose two-gate cuts must be exhibited directly, and Lemma 5.18 makes the search short: only pairs need be considered.

### 5.5.3. Repetition

An inference circuit is a few dozen kinds of gate, each repeated across layers, positions, heads, and coordinates. The GPT-2 Small circuit of §5.6 has $4.2\times10^{10}$ gates but only $82$ families of them, and its wiring is given by $119$ rules of the form "gate $(\ell,t,i)$ of family $F$ reads gate $(\ell,t',j)$ of family $F'$" with affine relations among the indices. The hypotheses of Lemmas 5.19 and 5.20 are statements about which families read which, so they can be checked once per family and hold for every copy; and the number of copies is a product of index ranges, so regions are counted algebraically rather than enumerated. The same goes for the exceptional cases, which arise where an index range degenerates: the first position of a sequence, whose softmax has a single key, or the last, which no later position reads. Splitting each family by layer and by these boundary cases gives a partition of the gates into a few hundred thousand parametrized classes, each of which is assigned a cut by one of the two rules or by an explicit two-gate cut, in about a second.

Two checks guard against error in the rules. The wiring rules are compared against an explicit construction of the circuit on small configurations (one or two layers, a model dimension of one to three, a vocabulary of two to four tokens, two or three positions), and on those configurations the cuts obtained from the rules are compared gate for gate with the cuts computed by maximum flow as in §5.5.1, both the downstream-most and the nearest. The descriptions of §7 make this structure explicit for arbitrary circuits: a circuit is a hierarchy of definitions and repetitions, every copy of a definition is one kind, and cuts are computed once per kind, as is everything else the verifier needs to know about the shape of the circuit.

## 5.6. Case study: GPT-2 Small

We apply the construction of §5.4 and §5.5 to an inference circuit for GPT-2 Small ($12$ layers, model dimension $d=768$, $12$ heads of dimension $64$, MLP width $3{,}072$, vocabulary of $50{,}257$ tokens) processing a $100$-token prompt and then generating $100$ tokens greedily. The circuit has $n\approx4.24\times10^{10}$ gates: $42{,}361{,}101{,}422$ arithmetic gates and $26{,}307{,}172$ FP16 write-outs at the boundaries of operations. Widths follow the profile under which the model is served: values at the boundaries of operations are FP16 ($16$ bits), accumulators and statistics are FP32 ($32$ bits), and a token is one of $50{,}257$ values, so its width is $\log_2 50{,}257\approx15.6$. The output gates are the $100$ generated tokens, so $W_{\mathrm{out}}\approx1{,}562$ bits. We take $p=\eta=1\%$, so $L=459$, and units that lie inside regions, as Proposition 5.13 requires.

**Regions.** The $n$ gates fall into $R=31{,}987{,}772$ regions, listed in Table 5.1. Three facts about them matter for the bound. First, the widest cut is $32$ bits, so $\kappa_{\max}=32$: the same as the widest gate, and there are $6{,}114{,}482$ regions with $32$-bit cuts. Second, regions are large: the average is about $1{,}300$ gates, and the largest is $2.5\times10^{8}$. Third, cuts are simple: $2.80\%$ of gates lie in the dead region and have the empty cut, all but $176{,}038$ of the rest have a single-gate cut, and only $133$ regions have a two-gate cut.

| Model part | Gates sharing one cut | Gates per region | Regions | Cut (width in bits) | Share of gates |
|---|---|---:|---:|---|---:|
| Embedding | token and position addition | $1$–$2$ | $152{,}064$ | FP16 coordinate ($16$) | $<0.001\%$ |
| LayerNorm | mean reduction | $768$ | $4{,}531$ | FP32 mean ($32$) | $0.008\%$ |
|  | variance reduction and rsqrt | $1{,}538$ | $4{,}531$ | FP32 $1/\sigma$ ($32$) | $0.016\%$ |
|  | centered coordinate | $1$ | $3{,}479{,}808$ | FP32 intermediate ($32$) | $0.008\%$ |
|  | normalize, $\gamma$, $\beta$ | $3$ | $3{,}497{,}472$ | FP16 coordinate ($16$) | $0.025\%$ |
| $Q$, $K$, $V$ projections | one $768$-term inner product | $1{,}537$ | $5{,}313{,}792$ | FP16 coordinate ($16$) | $19.27\%$ |
| Attention scores | one $64$-term inner product and scaling | $129$ | $2{,}600{,}400$ | FP16 score ($16$) | $0.79\%$ |
| Softmax | running maximum | $n_t-1$ | $25{,}740$ | FP32 maximum ($32$) | $0.006\%$ |
|  | shift and exponential | $2$ | $2{,}574{,}000$ | FP32 exponential ($32$) | $0.012\%$ |
|  | denominator and reciprocal | $n_t$ | $25{,}740$ | FP32 $1/\Sigma$ ($32$) | $0.006\%$ |
|  | probability | $1$ | $2{,}600{,}400$ | FP16 probability ($16$) | $0.006\%$ |
| Attention output $PV$ | one $n_t$-term inner product | $2n_t$ | $1{,}672{,}704$ | FP16 coordinate ($16$) | $0.78\%$ |
| Output projection | inner product and residual addition | $1{,}538$ | $1{,}672{,}704$ | FP16 residual coordinate ($16$) | $6.07\%$ |
| MLP up-projection | inner product and GELU | $1{,}546$ | $6{,}690{,}816$ | FP16 coordinate ($16$) | $24.40\%$ |
| MLP down-projection | inner product and residual addition | $6{,}146$ | $1{,}672{,}704$ | FP16 residual coordinate ($16$) | $24.26\%$ |
| LM head and argmax | $50{,}257$ inner products, argmax, last-layer tail | $8.9\times10^{7}$ | $100$ | token ($15.6$) | $21.51\%$ |

*Table 5.1.* Regions of GPT-2 Small by model part, for a $100$-token prompt and $100$ generated tokens; $n_t$ is the number of keys visible at position $t$. Gates per region include the write-out; shares are of the arithmetic gates. The rows contain $31{,}987{,}506$ regions and $97.17\%$ of the gates. Not listed: the dead region ($2.80\%$ of gates); $132$ regions in which a one-key softmax places a head's whole query projection behind its single probability ($0.03\%$); and the $133$ regions with two-gate cuts ($0.0004\%$), namely $132$ two-key softmaxes whose $32$-bit statistics sit behind their two probabilities, and the $32$-bit statistics at the second-to-last position, which sit behind the last two tokens ($31.2$ bits).

**Where the regions come from.** Each row of the table is an instance of §5.5.2. Every residual coordinate that a later position reads fans out, to the next layer and, through $K$ and $V$, to that later position, and is $16$ bits wide, so by Lemma 5.20 it is its own cut. Funneling therefore happens only inside operations and at the token. One coordinate of a projection $xW+b$ is $768$ multiplications, $767$ additions, a bias addition, and an FP16 write-out, $1{,}537$ gates, all but the last of them $32$ bits wide; Lemma 5.19 places them behind the write-out, and Lemma 5.20 makes the write-out its own cut, so one incorrect multiplication is worth exactly $16$ bits. Where the write-out feeds a single gate, the cut moves to that gate: the residual addition after the output projection and the down-projection, the GELU after the up-projection. Reductions to an FP32 statistic behave the same way with a $32$-bit exit: the mean and the inverse standard deviation of a LayerNorm, the maximum and the denominator of a softmax. The $32$-bit values that branch are their own cuts, and there are many: every centered coordinate $x_i-\mu$ feeds both the variance and the normalization, and every softmax exponential feeds both a probability and the denominator. That is why $\kappa_{\max}$ is $32$ rather than $16$; an adversary with $458$ incorrect units is not short of $32$-bit regions.

At the output, each of the $50{,}257$ logits is an inner product read only by the argmax, whose result, the token, is $15.6$ bits wide, narrower than everything before it. Lemma 5.19 with $o$ the token places behind it the LM head ($77{,}144{,}495$ gates) and every other gate at that position whose only route to an output is that token: in the last layer, the $Q$ projection and everything after it, through the final LayerNorm, about $8.9\times10^{7}$ gates in all. At the last position, which no later position reads, the region is the entire forward pass, $2.5\times10^{8}$ gates. The dead region is the mirror image: at the $99$ prompt positions whose logits are never computed, the same gates, the last layer's $Q$ projection and everything after it, have no path to an output, and an adversary may write there freely to no effect. The two-gate cuts arise where two exits together are no wider than a $32$-bit gate: at the second-to-last position, the $32$-bit statistics that reach both of the last two tokens sit behind that pair, $31.2$ bits, and at a position with two keys, the $32$-bit softmax statistics reach the rest of the circuit only through two $16$-bit probabilities.

**The bound.** Corollary 5.14 gives

$$
\log_2|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ 459\Bigl(32+\log_2\frac{e\cdot31{,}987{,}772}{459}\Bigr)\ \approx\ 459\,(32+17.5)\ \approx\ 22{,}700\text{ bits}.
$$

For comparison, Corollary 5.3 with single gates as units gives $459\,(32+27.9)\approx27{,}500$ bits, and the output bound gives $1{,}562$. The regions save about $10$ bits per incorrect unit in the position term and cost nothing in the value term, since a unit inside a region certifies its region's cut whether it is one gate or $10^{8}$. But for this run the output bound is the tighter of the three, by a factor of $15$: the adversary cannot use $32$ bits per incorrect unit because the whole output has only $1{,}562$ bits to carry them. The two bounds cross when the output is long relative to $L\kappa_{\max}$. At $p=1\%$ this happens at about $1{,}500$ generated tokens, past which Corollary 5.14, which grows only logarithmically in the size of the circuit, is the binding one; at $100$ tokens it happens at a rate of about $p=15\%$, where $L=29$.

**What is certified.** For a region with cut $\{o\}$, three statements must be kept apart. First, $\{o\}$ is a narrowest downstream cut, so at most $2^{w_o}$ outputs are reachable by corrupting the region. Second, some corruption realizes all $2^{w_o}$ values at $o$; this holds for every singleton cut, since $o$ itself may be the incorrect gate. Third, those $2^{w_o}$ values yield $2^{w_o}$ distinct outputs. The third statement holds for the token regions, whose cut is an output gate, and is open for the others: $2^{32}$ values of a LayerNorm mean need not yield $2^{32}$ distinct $100$-token outputs, and in aggregate they cannot, by the output bound. What the analysis certifies for this circuit is therefore the granularity: which gates may be grouped into a unit at no loss in the bound, namely anything inside a region, up to an inner product with its write-out or an LM head with its token. Whether the $32$-bit regions realize their cut widths is a question about values, not wiring, and the output bound answers it in the negative for short generations.

## 5.7. Implications for verification granularity

The verifier of §6 must choose its units, and this section constrains the choice from the side of the bound; §6.5 constrains it from the side of cost. The following consequences of the results above carry over.

First, within a region, the choice does not matter. By Lemma 5.12 every unit inside a region certifies the region's cut, so refining a unit below its region buys nothing and merging units up to the region costs nothing. The regions of an inference circuit are its arithmetic operations, an inner product with its write-out, a reduction with its statistic, an LM head with its token, and these are the coarsest units the verifier may use for free. In GPT-2 Small an inner product of $1{,}537$ gates certifies $16$ bits whether it is checked as one unit or as $1{,}537$, and the LM head of $7.7\times10^{7}$ gates certifies $15.6$ bits as one unit.

Second, across regions, merging costs, and it can cost a great deal. A unit that spans several regions has $\kappa$ at most the sum of their cut widths, and when their cuts lie on different paths to the output, as those of adjacent inner products do, the sum is close to the truth until it reaches $W_{\mathrm{out}}$. A LayerNorm over $768$ coordinates checked as one unit in GPT-2 Small has $\kappa$ in the hundreds or thousands of bits in place of $32$, its narrowest cut being its $768$ outputs or the tokens it can reach, whichever is narrower; a unit spanning a whole layer at one position has $\kappa$ close to $W_{\mathrm{out}}$. Since Proposition 5.8 uses $\kappa_{\max}$ for every incorrect unit, one coarse unit raises the value term for all of them. Verification units, the units that are proved and that enter $\kappa$, should therefore not cross region boundaries where the cost model allows it. Replay units, which §6 samples and re-executes, do not enter $\kappa$ at all; they enter the acceptance probability $\sigma$ of the remark in §5.4.2, and from the point of view of this section they may be as coarse as cost dictates.

Third, the position term counts regions, not units (Proposition 5.13), so making units finer than regions does not inflate it. The verifier is free to choose small verification units for reasons of cost without paying for the freedom in the bound.

Fourth, the cut bound is not always the binding one. It beats the output bound when $L(\kappa_{\max}+\log_2(eR/L))<W_{\mathrm{out}}$, that is, for long outputs or dense sampling; for GPT-2 Small at $p=1\%$ this requires about $1{,}500$ generated tokens. Below that, the output bound governs the capacity and the analysis of this section governs the granularity. §6.6 chooses $p$ and the units in the light of both.

Finally, the verifier of §7 never sees a gate list and cannot run the flow computation of §5.5.1. It uses, for each unit, the downstream cut that its description exposes without computation: the unit's declared interface $\operatorname{Out}(U)$, which is a downstream cut for $U$ by Lemma 5.17. For units that are regions or lie inside them, and whose declared interface is the region's cut, this is exact, and the bound the verifier computes is the bound of this section.

---

**Notes for the author** (not part of the section)

- *Numbering and file.* The section is numbered §5 to match the outline and the cross-references in `section-7-secure-circuit-compilation.md` (which cites "§5.4" for granularity); the request said "section 4" with "4.1 and 4.2" as anchors, which I read as §5.1 and §5.2. Renumbering is a find-and-replace. Results are numbered §-prefixed (Lemma 5.1, ...) as in §7.
- *Anchors.* §5.1 is the pasted text, extended with the definitions §5.2 uses and did not have: $\operatorname{err}(\tau)$, units, $\operatorname{err}_{\mathcal P}$, the sampling verifier, $\eta$, the budget $L$, $W(S)$, and the alphabet remark. The threshold is now a parameter $\eta$ rather than the literal $1\%$, since §7 and the appendix use $\eta$; $1\%$ remains the running value. §5.2 is our final version with the numbering changed, one edit ("a value one billion times the size of $m^*$" became "a value a billion bits long", since $m^*$ is not defined in the section), and the Rinberg example still a TODO, as you asked.
- *Alphabets.* Kept $\{0,1\}^{w_i}$ as the presentation and licensed non-integer widths in one sentence at the end of §5.1, so that tokens can be $15.6$ bits in §5.6. Lemma 5.1's $0^{w_i}$ then means a fixed well-typed value. §7 currently says a value is well-typed when it is a bitstring of the gate's width; that is the integer case.
- *Theorem 5.6* no longer assumes equal inputs: an input gate on which two transcripts disagree is incorrect in one of them, so it lies in $E$, and the proof goes through. Corollary 5.7 likewise drops "with correct inputs."
- *Downstream-most.* Defined via "$D'$ is a downstream cut for $D$" (Definition 5.9). Uniqueness is proved in Lemma 5.10 directly; existence is Lemma 5.16 via the residual graph. This needs $w_i>0$, which §5.1 now states. The literature reference for the lattice of minimum cuts is a TODO (Ford–Fulkerson's book has the union/intersection lemma; Picard–Queyranne 1980 is the standard citation).
- *Beyond the sketch.* Two additions: the remark on other verifiers at the end of §5.4.2, which states the bound in the form $\sum_{E:\sigma(E)>\eta}2^{\kappa(E)}$ that §6's two-stage sampler needs; and Lemma 5.18 (a cut of a gate has at most $\lfloor w_g/w_{\min}\rfloor$ gates), which is what makes the two rules exhaustive for GPT-2 and is what the code's "bounded cut order" certificate is.
- *GPT-2 numbers* were regenerated from `circuit_cut_analysis` (`circuit-cut gpt2`, 1.3 s). The code's gate total ($42{,}361{,}101{,}422$) counts arithmetic primitives only and excludes casts; the FP16 write-outs are the cuts of most regions, so the text counts them as gates ($26{,}307{,}172$ of them, from the report's source-gate totals) and says so, while the table's shares keep the code's denominator. They agree with the earlier draft's totals ($R=31{,}987{,}772$; $6{,}114{,}482$ regions of $32$ bits; $176{,}038$ gates in two-gate-cut regions; $2.80\%$ dead) and correct five per-region gate counts, which now include the FP16 write-out: $Q$/$K$/$V$ $1{,}537$ (was $1{,}536$), scores $129$ (was $128$), output projection $1{,}538$ (was $1{,}537$), up-projection $1{,}546$ (was $1{,}545$), down-projection $6{,}146$ (was $6{,}145$). The break-even figures ($\approx1{,}500$ tokens at $p=1\%$; $p\approx15\%$ at $100$ tokens) assume $R$ scales linearly with the number of positions, which is roughly right.
- *§5.5.3* describes what the code does ($82$ families, $119$ wire rules, algebraic counting, validation against the explicit solver on small configurations) without naming the code; adjust if §9 wants the credit.
- *Cross-references to check.* §6.5 (cost model), §6.6 (parameter choice), §7 (declared interfaces as cuts; $\operatorname{Bound}$ as a fold over kinds), §8 (how $\eta$ and $|\mathcal Y(\mathcal T_{\mathcal V})|$ enter $\delta$), §4.2.3 (the granularity promise, now Lemma 5.12 and the exit rule). The last paragraph of §5.7 commits §7's $\operatorname{Bound}$ to using declared interfaces as cuts, which is what the README says it does.
