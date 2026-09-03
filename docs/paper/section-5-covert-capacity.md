# 5. Covert capacity of partially incorrect circuits

Apart from checking the fixed input and the type of the output, the verifier of §6 checks a computation only by checking randomly sampled units of it. This section bounds the number of distinct outputs an adversary can produce from a fixed circuit and input while being accepted with probability above a threshold. The logarithm of this number is the number of bits one response can carry, and §8 turns a bound on it into exfiltration security in the sense of §3. Our main result (Theorem 5.9) is that the number is at most $\sum_E2^{\kappa(E)}$, summed over the sets $E$ of units that the verifier will tolerate being incorrect, where $\kappa(E)$ is the width of the narrowest set of gates through which errors in $E$ can reach the output. For a verifier that samples each unit independently this is at most $B\bigl(\kappa_{\max}+\log_2(eR/B)\bigr)$ bits (Corollary 5.16), where $B$ is the number of units the adversary can afford to corrupt, $R$ is the number of distinct cuts in the circuit, and $\kappa_{\max}$ is the width of the widest; in GPT-2 Small, $\kappa_{\max}=32$ although the circuit has $4\times10^{10}$ gates and cuts shared by as many as $10^{8}$ of them.

## 5.1. Circuits, transcripts, and reachable outputs

A circuit $C$ of size $n$ is a sequence of *gates* numbered $1,\ldots,n$. Gate $i$ represents a single step in a static program, computing the value

$$
v_i\leftarrow f_i(v_{j_1},\ldots,v_{j_{d_i}}).
$$

Here $\operatorname{args}(i)=(j_1,\ldots,j_{d_i})$ is an ordered tuple of gates preceding $i$ whose values are the arguments of $f_i$, and $f_i:V_{j_1}\times\cdots\times V_{j_{d_i}}\to V_i$, where $V_i\subseteq\{0,1\}^*$ is a finite set with $|V_i|\ge2$, the *value set* of gate $i$; $w_i:=\log_2|V_i|$ is the *width* of gate $i$. A $16$-bit gate has $V_i=\{0,1\}^{16}$ and $w_i=16$; a gate whose value is one of $50{,}257$ tokens has $w_i=\log_2 50{,}257\approx15.6$. A designated set $\operatorname{in}(C)\subseteq[n]$ consists of the *input gates*; each input gate $j$ has $\operatorname{args}(j)=()$, so $f_j$ is a constant, and these constants together form the input $x$. A designated set $\operatorname{out}(C)\subseteq[n]$ consists of the *output gates*. Throughout this section $C$, and with it $x$, is fixed.

We represent an evaluation of $C$ as a *transcript* $\tau=(v_1,\ldots,v_n)$. These values need not have been computed correctly, and $v_i$ need not lie in $V_i$: it may be any finite bitstring, of any length. We can think of $C$ as a system of constraints on such transcripts. We say that

- a value $v_i$ is *well-typed* if $v_i\in V_i$, and a transcript is well-typed if all of its values are;
- gate $i$ is *correct* in $\tau$ if each argument value $v_{j_\ell}$ is well-typed and $v_i=f_i(v_{j_1},\ldots,v_{j_{d_i}})$;
- a gate that is not correct is *incorrect*, and $\operatorname{err}(\tau)\subseteq[n]$ is the set of incorrect gates.

Exactly one transcript has $\operatorname{err}(\tau)=\varnothing$, the honest evaluation of $C$ on $x$. For $S=\{s_1<\cdots<s_k\}\subseteq[n]$ we write

$$
\tau|_S:=(v_{s_1},\ldots,v_{s_k})
$$

for the restriction of $\tau$ to $S$, so that $\tau|_{\operatorname{in}(C)}$ and $\tau|_{\operatorname{out}(C)}$ are its input and its output. For a set of gates $S$ we write $W(S):=\sum_{i\in S}w_i$ for its width; in particular $W_{\mathrm{out}}:=W(\operatorname{out}(C))$ is the width of the output. A well-typed assignment to the gates of $S$ is one of $\prod_{i\in S}|V_i|=2^{W(S)}$ possibilities, which is how widths enter the bounds.

**Verifiers.** A verifier $\mathcal V$ is a randomized procedure that reads a transcript and accepts or rejects it. The verifiers of this paper check a random sample of it. Specifically, $\mathcal V$ is given a partition $\mathcal P$ of the non-input gates into $N$ *units* and a probability distribution over subsets of $\mathcal P$. Given $\tau$, it draws a set $\mathcal S\subseteq\mathcal P$ of units from this distribution and accepts if and only if the input gates are correct, the output gates are well-typed, and every gate of every unit in $\mathcal S$ is correct. (A gate is checked against the values its arguments hold in $\tau$, which may lie outside the unit.) Write $\operatorname{err}_{\mathcal P}(\tau)\subseteq\mathcal P$ for the set of units that contain an incorrect gate. We will often treat a set $E$ of units as the set $G(E):=\bigcup_{U\in E}U$ of gates in its units, abbreviating $W(E):=W(G(E))$ and $\tau|_E:=\tau|_{G(E)}$; thus $\operatorname{err}(\tau)\subseteq G(\operatorname{err}_{\mathcal P}(\tau))$ whenever the inputs of $\tau$ are correct. When the inputs of $\tau$ are correct and its output is well-typed,

$$
\Pr[\mathcal V(\tau)=1]=\Pr[\mathcal S\cap\operatorname{err}_{\mathcal P}(\tau)=\varnothing]=:\sigma(\operatorname{err}_{\mathcal P}(\tau)),
$$

and otherwise $\Pr[\mathcal V(\tau)=1]=0$. So the probability of acceptance depends on $\tau$ only through its set of incorrect units, and $\sigma$ is monotone, $\sigma(E')\ge\sigma(E)$ whenever $E'\subseteq E$, since a sample that avoids $E$ avoids $E'$. Our running example is the *sampling verifier* with rate $p\in(0,1]$, which includes each unit in $\mathcal S$ independently with probability $p$, so that

$$
\sigma(E)=(1-p)^{|E|}.
$$

For the verifier of §6, which samples in two stages, $\sigma$ is given at the end of §5.4.

Fix a threshold $\eta\in(0,1)$ and let

$$
\mathcal T_{\mathcal V}:=\{\tau:\Pr[\mathcal V(\tau)=1]>\eta\}
$$

be the set of transcripts the adversary can submit with better than an $\eta$ chance of acceptance, and let

$$
\mathfrak E_\eta:=\{E\subseteq\mathcal P:\sigma(E)>\eta\}
$$

be the family of sets of units whose corruption is accepted with probability above $\eta$. Then $\mathcal T_{\mathcal V}$ is exactly the set of transcripts with correct inputs, a well-typed output, and $\operatorname{err}_{\mathcal P}(\tau)\in\mathfrak E_\eta$; and since $\sigma$ is monotone, every subset of a set in $\mathfrak E_\eta$ is in $\mathfrak E_\eta$. For the sampling verifier, $\mathfrak E_\eta$ consists of the sets of at most $B$ units, where $B$ is the largest integer with $(1-p)^B>\eta$: the adversary may corrupt up to $B$ units and no more. With $p=\eta=1\%$, $B=458$. Transcripts outside $\mathcal T_{\mathcal V}$ are accepted with probability at most $\eta$, and §8 accounts for them separately.

**Reachable outputs.** For a set of transcripts $\mathcal T$, let

$$
\mathcal Y(\mathcal T):=\{\tau|_{\operatorname{out}(C)}:\tau\in\mathcal T\}
$$

be the set of outputs it produces. Our goal is an upper bound on $|\mathcal Y(\mathcal T_{\mathcal V})|$, the number of outputs the adversary can reach; $\log_2$ of this number is the capacity.

## 5.2. Deriving a simple bound

The most basic bound is simply the width of the circuit's output:

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\le 2^{W_{\mathrm{out}}}.
$$

This is sound because $\mathcal V$ rejects any transcript whose output is ill-typed, so every reachable output is one of the $2^{W_{\mathrm{out}}}$ well-typed ones. We refer to it as the *output bound*. In many situations we would like to improve on it. For instance, an inference datacenter could exfiltrate its model weights in [TODO: Rinberg 2025 example].

In principle, we should be able to derive a tighter bound from $\mathcal T_{\mathcal V}$, since the verifier places substantial constraints on it and $|\mathcal Y(\mathcal T_{\mathcal V})|\le|\mathcal T_{\mathcal V}|$. An immediate concern, however, is that $|\mathcal T_{\mathcal V}|$ may not itself be bounded. This is because $\mathcal T_{\mathcal V}$ may contain ill-typed transcripts, that is, transcripts in which some gate holds a value outside its value set, possibly an arbitrarily long one. For instance, gate $500$ could be assigned a value a billion bits long, which later gates could read and then write to the output. This may not be detected because the verifier can't afford to check every value in the transcript. However, below we prove that this is not actually a problem: there are no outputs reachable by an ill-typed transcript that are not also reachable by a well-typed transcript, and thus no adversary can exploit this seeming vulnerability to its advantage. It follows that we may restrict our count to the well-typed transcripts in $\mathcal T_{\mathcal V}$, of which there are finitely many.

**Lemma 5.1.** *Let $\mathcal T_{\mathcal V}^{\mathrm{wt}}\subseteq\mathcal T_{\mathcal V}$ be the set of well-typed transcripts in $\mathcal T_{\mathcal V}$. Then $\mathcal Y(\mathcal T_{\mathcal V})=\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})$.*

*Proof.* We know that $\mathcal T_{\mathcal V}^{\mathrm{wt}}\subseteq\mathcal T_{\mathcal V}$, and hence that $\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})\subseteq\mathcal Y(\mathcal T_{\mathcal V})$. Thus it suffices to show that $\mathcal Y(\mathcal T_{\mathcal V})\subseteq\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})$. To do so, we will show that for every $\tau\in\mathcal T_{\mathcal V}$ there exists a $\tau'\in\mathcal T_{\mathcal V}^{\mathrm{wt}}$ such that $\tau'|_{\operatorname{out}(C)}=\tau|_{\operatorname{out}(C)}$. We will do so by construction. For each gate $i$ fix a default value $z_i\in V_i$. Let $\tau=(v_i)_i\in\mathcal T_{\mathcal V}$ and define $\tau'=(v_i')_i$ by

$$
v_i':=\begin{cases}v_i & \text{if } v_i\in V_i,\\[2pt] z_i & \text{otherwise.}\end{cases}
$$

Then $\tau'$ is well-typed, since each $v_i'$ is either $v_i$ or $z_i$, both of which lie in $V_i$. It remains to show that $\tau'\in\mathcal T_{\mathcal V}$ and that $\tau'|_{\operatorname{out}(C)}=\tau|_{\operatorname{out}(C)}$. Consider any gate $i$ that is correct in $\tau$. Its value lies in $V_i$, since $f_i$ takes values there, and each of its arguments $v_j$ lies in $V_j$, by the definition of correctness; so $v_i'=v_i$ and $v_j'=v_j$ for every argument $j$, and gate $i$ is correct in $\tau'$. Hence $\operatorname{err}_{\mathcal P}(\tau')\subseteq\operatorname{err}_{\mathcal P}(\tau)$, so $\operatorname{err}_{\mathcal P}(\tau')\in\mathfrak E_\eta$, since every subset of a set in $\mathfrak E_\eta$ is in $\mathfrak E_\eta$. The input gates of $\tau$ are correct and its output gates are well-typed, so none of them is overwritten: $\tau'$ has correct inputs and $\tau'|_{\operatorname{out}(C)}=\tau|_{\operatorname{out}(C)}$. Thus $\tau'\in\mathcal T_{\mathcal V}$. $\square$

This proof demonstrates the necessity of our requirement that a correct gate have well-typed arguments. Without it, the adversary could write a million bits at gate $500$ at the cost of one incorrect gate and have a thousand correct gates each read a different $16$-bit slice of it, so that a single error would carry $16{,}000$ bits to the output.

We are now in a position to bound $|\mathcal Y(\mathcal T_{\mathcal V})|$ by counting the elements of $\mathcal T_{\mathcal V}^{\mathrm{wt}}$.

**Proposition 5.2.** *Let $\mathcal V$ be a verifier with units $\mathcal P$, and let $\mathfrak E_\eta$ be the family of sets $E$ of units with $\sigma(E)>\eta$. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{E\in\mathfrak E_\eta}2^{W(E)}.
$$

*For the sampling verifier, with $B$ as in §5.1, $N:=|\mathcal P|$, and $W_{\max}$ the largest width of a unit, this is at most*

$$
\sum_{\ell\le B}\binom{N}{\ell}\,2^{\ell W_{\max}}.
$$

*Proof.* We know from Lemma 5.1 that $|\mathcal Y(\mathcal T_{\mathcal V})|=|\mathcal Y(\mathcal T_{\mathcal V}^{\mathrm{wt}})|\le|\mathcal T_{\mathcal V}^{\mathrm{wt}}|$. Thus it suffices to show that $|\mathcal T_{\mathcal V}^{\mathrm{wt}}|\le\sum_{E\in\mathfrak E_\eta}2^{W(E)}$. To do so, we will show that every $\tau\in\mathcal T_{\mathcal V}^{\mathrm{wt}}$ is determined by the pair $(E,\tau|_E)$, where $E:=\operatorname{err}_{\mathcal P}(\tau)$, and then count the possible pairs. Let $\tau\in\mathcal T_{\mathcal V}^{\mathrm{wt}}$ and consider its gates in index order. An input gate holds its input value, since the inputs of $\tau$ are correct. A gate in a unit of $E$ holds the value recorded in $\tau|_E$. Any other gate $i$ is correct, so $v_i$ is $f_i$ applied to the values of its arguments, all of which precede $i$. Thus each $v_i$ is determined by $(E,\tau|_E)$ and the earlier values, and by induction so is all of $\tau$. It remains to count the pairs. The set $E$ lies in $\mathfrak E_\eta$, and since $\tau$ is well-typed there are at most $2^{W(E)}$ choices of $\tau|_E$ for each $E$. For the sampling verifier the sets in $\mathfrak E_\eta$ are those of at most $B$ units; there are $\binom N\ell$ of size $\ell$, and each has $W(E)\le\ell W_{\max}$. $\square$

**Corollary 5.3.** *For the sampling verifier with $1\le B\le N$,*

$$
\log_2|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ B\Bigl(W_{\max}+\log_2\frac{eN}{B}\Bigr).
$$

*Proof.* Each term of the sum in Proposition 5.2 has $\ell\le B$, so $2^{\ell W_{\max}}\le2^{BW_{\max}}$, and $\sum_{\ell\le B}\binom N\ell\le(eN/B)^B$ by the standard estimate. $\square$

The corollary is the form worth remembering: each incorrect unit costs the adversary at most $W_{\max}$ bits for its value and $\log_2(eN/B)$ bits for its position. We refer to the two parts as the *value term* and the *position term*. In the next section we investigate the tightness of the bound for the case of a matrix multiplication.

## 5.3. Testing the bound on a matrix multiplication

To see how loose Proposition 5.2 can be, consider the most common circuit in the world, that of a matrix multiplication. Let $C$ compute a chain of $k$ matrix-vector products $x_t=A_t\,x_{t-1}$, $t=1,\ldots,k$, where the vector $x_0$ of length $d$ and the $d\times d$ matrices $A_1,\ldots,A_k$ are inputs with $16$-bit entries. Each entry of $x_t$ is a dot product of length $d$, computed by $d$ multiplication gates and $d-1$ addition gates, all of width $16$. (Which $16$-bit arithmetic the gates perform is immaterial here; only the widths of their results matter.) The output gates are the $d$ gates holding $x_k$, so $W_{\mathrm{out}}=16d$. Take each dot product as a unit; then $N=kd$ and every unit has width $W(U)=16(2d-1)$.

Let $d=1{,}024$, $k=100$, and $p=\eta=1\%$, so that $N=102{,}400$, $W_{\max}=32{,}752$, and $B=458$. The value term in Corollary 5.3 is $458\cdot32{,}752\approx1.5\times10^{7}$ bits and the position term is $458\log_2\frac{e\cdot102{,}400}{458}\approx4{,}200$ bits, against $W_{\mathrm{out}}=16{,}384$. Proposition 5.2 is worse than the output bound by a factor of nearly a thousand, and the entire excess is in the value term.

The reason is that Proposition 5.2 counts transcripts, and here a great many transcripts have the same output. Nothing outside a dot product reads its products or its partial sums; the only value in the unit that any other gate reads is the final sum, which is $16$ bits wide. Two well-typed assignments to an incorrect unit that agree on the final sum therefore produce the same output, so the $2^{32{,}752}$ assignments Proposition 5.2 counts produce at most $2^{16}$ distinct outputs. If $W_{\max}$ could be replaced by $16$ in Corollary 5.3, the bound would be $458\cdot(16+9.2)\approx11{,}600$ bits. This is not far from what the adversary can in fact do: by corrupting $458$ dot products in the last layer it can set $458$ entries of $x_k$ to whatever it likes, which gives it $458\cdot16\approx7{,}300$ bits from the values alone and about $8{,}300$ once the choice of entries is counted.

The fact we used is that every path from a gate of a dot product to an output gate passes through the dot product's final addition. In the last layer this is obvious, since the final addition is itself an output gate. But it holds in every layer, and there it is not obvious at all: a corrupted dot product in layer $1$ changes every entry of $x_k$, yet the whole of that change is a function of the one $16$-bit value passed to layer $2$, so it too has at most $2^{16}$ distinct effects on the output. The next section shows that this is all that is needed: in Corollary 5.3, $W_{\max}$ may be replaced by the width of any set of gates that every path from a unit to the output must pass through.

## 5.4. Bounding capacity via the narrowest downstream cut

### 5.4.1. Downstream cuts determine outputs

Informally, the value at one gate can affect the value at a later gate only if the former is an argument of the latter, or an argument of one of its arguments, and so on. We call such a chain a path.

**Definition 5.4 (Path).** *A path from a gate $a$ to a gate $z$ is a sequence of gates $a=j_0,j_1,\ldots,j_t=z$ with $j_s\in\operatorname{args}(j_{s+1})$ for every $s<t$; we allow $t=0$. For $A,Z\subseteq[n]$, a path from $A$ to $Z$ is a path from some $a\in A$ to some $z\in Z$.*

Since arguments precede the gates that read them, the gates of a path are distinct and increase in index.

**Definition 5.5 (Downstream cut).** *Let $A,Z,D\subseteq[n]$. We say that $D$ cuts $A$ from $Z$ if every path from $A$ to $Z$ contains a gate of $D$. A downstream cut for $A$ is a set of gates that cuts $A$ from $\operatorname{out}(C)$.*

Both $A$ itself and $\operatorname{out}(C)$ are downstream cuts for $A$, since every path from $A$ to the output begins in the one and ends in the other. The useful cuts lie in between; for a dot product of the last section, its final addition is a downstream cut of width $16$. Note that a path of length zero is a path, so a downstream cut for $A$ must contain every output gate in $A$.

An incorrect value can affect a gate only along a path, and a cut intercepts every path. So the values on a set that cuts the incorrect gates from $Z$ should determine the values at $Z$.

**Theorem 5.6 (Downstream cuts determine outputs).** *Let $\tau$ and $\tau'$ be transcripts of $C$, let $E:=\operatorname{err}(\tau)\cup\operatorname{err}(\tau')$ be the set of gates incorrect in either, let $Z\subseteq[n]$, and let $D$ cut $E$ from $Z$. If $\tau|_D=\tau'|_D$, then $\tau|_Z=\tau'|_Z$.*

*Proof.* Let $\Delta:=\{i\in[n]:v_i\neq v'_i\}$ be the set of gates on which the transcripts disagree. By hypothesis $\Delta$ contains no gate of $D$. Suppose for contradiction that $\Delta$ contains a gate of $Z$. We will construct a path from $E$ to $Z$ lying entirely in $\Delta$. Since $D$ cuts $E$ from $Z$, this path contains a gate of $D$, which contradicts $\Delta\cap D=\varnothing$.

We construct the path backward from $Z$. First observe that every gate $i\in\Delta\setminus E$ has an argument in $\Delta$: such a gate is correct in both transcripts, so if the transcripts agreed on all of its arguments, applying $f_i$ to the same values would give $v_i=v'_i$, contradicting $i\in\Delta$. (In particular an input gate, having no arguments, cannot lie in $\Delta\setminus E$.) Now let $j_0\in\Delta\cap Z$, and as long as $j_r\notin E$, choose $j_{r+1}\in\operatorname{args}(j_r)\cap\Delta$. Each step decreases the gate index, so the process terminates, and it can terminate only at a gate $j_m\in E$. Reversing the sequence gives a path $j_m,\ldots,j_0$ from $E$ to $Z$ consisting of gates in $\Delta$, as required. $\square$

**Corollary 5.7.** *Let $\mathcal T$ be a set of well-typed transcripts, and let $D$ be a downstream cut for $\operatorname{err}(\tau)$ for every $\tau\in\mathcal T$. Then $|\mathcal Y(\mathcal T)|\le2^{W(D)}$.*

*Proof.* Let $\tau,\tau'\in\mathcal T$. Then $D$ is a downstream cut for $\operatorname{err}(\tau)\cup\operatorname{err}(\tau')$, since every path from the union begins in one of the two sets. By Theorem 5.6, if $\tau|_D=\tau'|_D$ then $\tau$ and $\tau'$ have the same output. So the output of a transcript in $\mathcal T$ is determined by its values on $D$, and since the transcripts are well-typed there are at most $2^{W(D)}$ possibilities for those. $\square$

We can now make the replacement promised in §5.3. For a set of gates $A$, let

$$
\kappa(A):=\min\{W(D):D\text{ is a downstream cut for }A\}
$$

be the width of the narrowest downstream cut for $A$. Since $A$ is a downstream cut for itself, $\kappa(A)\le W(A)$, and since $\operatorname{out}(C)$ is one, $\kappa(A)\le W_{\mathrm{out}}$. For a set $E$ of units, $\kappa(E):=\kappa(G(E))$.

Corollary 5.7 concerns transcripts with a common downstream cut. Applied to the transcripts whose incorrect units all lie in a given set, it gives the form of the bound that the rest of the section uses.

**Lemma 5.8.** *Let $\mathcal A\subseteq\mathcal P$ be a set of units and let $\mathcal T_{\mathcal A}:=\{\tau\in\mathcal T_{\mathcal V}:\operatorname{err}_{\mathcal P}(\tau)\subseteq\mathcal A\}$. Then $|\mathcal Y(\mathcal T_{\mathcal A})|\le2^{\kappa(\mathcal A)}$.*

*Proof.* The construction in the proof of Lemma 5.1 takes each $\tau\in\mathcal T_{\mathcal A}$ to a well-typed $\tau'\in\mathcal T_{\mathcal V}$ with the same output and $\operatorname{err}_{\mathcal P}(\tau')\subseteq\operatorname{err}_{\mathcal P}(\tau)\subseteq\mathcal A$, so $\mathcal Y(\mathcal T_{\mathcal A})$ is the set of outputs of the well-typed transcripts in $\mathcal T_{\mathcal A}$. Each of these has correct inputs, so its incorrect gates lie in $G(\operatorname{err}_{\mathcal P}(\tau))\subseteq G(\mathcal A)$, and a narrowest downstream cut $D$ for $G(\mathcal A)$ is a downstream cut for $\operatorname{err}(\tau)$. Corollary 5.7 gives $|\mathcal Y(\mathcal T_{\mathcal A})|\le2^{W(D)}=2^{\kappa(\mathcal A)}$. $\square$

The adversary chooses where its errors go per run, so the bound on all reachable outputs is a sum over the sets of units it can afford to corrupt.

**Theorem 5.9.** *Let $\mathcal V$ be a verifier with units $\mathcal P$, and let $\mathfrak E_\eta$ be the family of sets $E$ of units with $\sigma(E)>\eta$. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{E\in\mathfrak E_\eta}2^{\kappa(E)}.
$$

*Proof.* Every $\tau\in\mathcal T_{\mathcal V}$ has $\operatorname{err}_{\mathcal P}(\tau)\in\mathfrak E_\eta$, so $\mathcal T_{\mathcal V}=\bigcup_{E\in\mathfrak E_\eta}\mathcal T_E$ with $\mathcal T_E$ as in Lemma 5.8, and $|\mathcal Y(\mathcal T_{\mathcal V})|\le\sum_{E}|\mathcal Y(\mathcal T_E)|\le\sum_{E}2^{\kappa(E)}$. $\square$

The two bounds of the last section are the extreme cases of this argument: with the gates of $E$ themselves in place of a narrowest cut it gives Proposition 5.2, and with $\operatorname{out}(C)$ in place of one it gives the output bound.

The bound is a sum and not a maximum. The maximum $\max_E2^{\kappa(E)}$ bounds an adversary that has fixed where its errors go; the choice of $E$ carries up to $\log_2|\mathfrak E_\eta|$ bits more, and for the matrix multiplication that was the whole of the position term. When several sets in $\mathfrak E_\eta$ share a cut, Lemma 5.8 applied to their union is tighter than the sum over them; Proposition 5.15 and §6 both use this. For the sampling verifier the sum has the following closed form.

**Proposition 5.10.** *For the sampling verifier with $N$ units and $B$ as in §5.1, let $\kappa_{\max}:=\max_{U\in\mathcal P}\kappa(U)$. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{\ell\le B}\binom{N}{\ell}\,2^{\ell\kappa_{\max}},
$$

*and consequently Corollary 5.3 holds with $\kappa_{\max}$ in place of $W_{\max}$.*

*Proof.* The sets in $\mathfrak E_\eta$ are those of at most $B$ units, and there are $\binom N\ell$ of size $\ell$. For such a set $E$, choose for each $U\in E$ a downstream cut $D_U$ of width $\kappa(U)$; every path from a gate of $E$ to the output contains a gate of the $D_U$ of its unit, so $\bigcup_{U\in E}D_U$ is a downstream cut for $G(E)$, and $\kappa(E)\le\sum_{U\in E}\kappa(U)\le|E|\,\kappa_{\max}$. Substituting into Theorem 5.9 gives the sum, and the second claim follows as in Corollary 5.3, since $\kappa_{\max}\le W_{\max}$. $\square$

For the circuit of §5.3, the final addition of a dot product is a downstream cut for it, so $\kappa(U)=16$ for every unit and Corollary 5.3 with $\kappa_{\max}$ in place of $W_{\max}$ gives about $11{,}600$ bits, against $1.5\times10^{7}$ from Proposition 5.2 and $16{,}384$ from the output. The attack described there, which sets up to $458$ entries of $x_k$ to any values, reaches $\sum_{\ell\le B}\binom{1{,}024}{\ell}(2^{16}-1)^{\ell}$ outputs, about $8{,}300$ bits, so the bound is within a factor of $1.4$ of what the attack achieves. The value terms agree exactly; the gap is in the position term, which counts positions among all $N=kd$ dot products where the attack uses only the last $d$. Whether corrupting earlier layers yields further distinct outputs depends on the matrices.

### 5.4.2. Grouping gates by their narrowest downstream cut

Proposition 5.10 depends on the partition into units through $\kappa_{\max}$ and $N$. Merging two units never lowers $\kappa_{\max}$, since a downstream cut for the merged unit is a downstream cut for each part, and it can raise it a great deal: a unit consisting of a whole layer of the matrix-multiplication circuit has $\kappa=16d$ in place of $16$. But merging is sometimes free. Any set of gates inside one dot product has $\kappa=16$, whether it is one gate or all $2d-1$, because all of them have the final addition as a narrowest downstream cut. In this subsection we assign each gate a single cut and group the gates that share one. The groups, which we call regions, are exactly the sets of gates that can be merged into one unit without raising the cut width of any gate in them (Lemma 5.14), and in the position term their number replaces $N$ (Proposition 5.15).

A gate may have several narrowest downstream cuts. In that circuit, the narrowest cuts of a multiplication gate in a dot product are the gate itself, each partial sum after it, and the final addition, all of width $16$. To assign each gate one cut we take the one furthest downstream, which groups the whole dot product behind its final addition; taking the nearest would make every gate its own group. The following order makes "furthest downstream" precise.

**Definition 5.11.** *Let $D$ and $D'$ be downstream cuts for $A$. We say that $D'$ is downstream of $D$ if $D'$ is a downstream cut for $D$, that is, if every path from $D$ to the output contains a gate of $D'$.*

Every downstream cut is downstream of itself, and if $D''$ is downstream of $D'$ and $D'$ downstream of $D$ then $D''$ is downstream of $D$, since a path from $D$ to the output contains a gate of $D'$ and its suffix from that gate contains a gate of $D''$.

**Lemma 5.12.** *Let $A$ be a set of gates. Two narrowest downstream cuts for $A$ that are each downstream of the other are equal, and among the narrowest downstream cuts for $A$ there is exactly one that is downstream of every other. We call it the downstream-most narrowest cut for $A$ and denote it $D(A)$.*

*Proof.* For the first claim, let $D_1$ and $D_2$ be narrowest downstream cuts for $A$, each downstream of the other, and suppose some $d\in D_1$ is not in $D_2$. Then $D_1\setminus\{d\}$ is still a downstream cut for $A$, which contradicts the minimality of $W(D_1)$ because $w_d>0$. A path from $A$ to the output that avoids $d$ contains another gate of $D_1$; one that contains $d$ continues from $d$, so it contains a gate $d_2\in D_2$ after $d$, and continues from $d_2$, so it contains a gate of $D_1$ at or after $d_2$, which is not $d$ since the gates of a path are distinct. Hence $D_1\subseteq D_2$, and by symmetry $D_1=D_2$.

The first claim gives uniqueness in the second, since two cuts each downstream of every other are each downstream of the other. For existence, since there are finitely many narrowest downstream cuts for $A$ and "downstream of" is transitive, it suffices to show that any two, $D_1$ and $D_2$, have a narrowest downstream cut downstream of both. This is the vertex form of a standard fact about minimum cuts, that the source sides of any two are contained in the source side of a third [TODO: Ford–Fulkerson 1962; Picard–Queyranne 1980]: if $S_i$ is the set of gates reachable from $A$ by paths avoiding $D_i$, then the gates outside $S_1\cup S_2$ that lie in $A$ or read a gate of $S_1\cup S_2$ form a narrowest downstream cut for $A$ downstream of both $D_1$ and $D_2$. The verification is in Appendix 5.A. $\square$

**Definition 5.13 (Regions).** *For a non-input gate $g$ with a path to the output, the cut of $g$ is $D(g):=D(\{g\})$, which is nonempty. For a nonempty set of gates $D$, the region of $D$ is $A_D:=\{g:D(g)=D\}$. The nonempty regions partition the non-input gates that have a path to the output, and $R$ denotes their number.*

A non-input gate with no path to the output has $\varnothing$ as its only narrowest downstream cut; corrupting it cannot change the output, and such gates belong to no region and play no part in the bounds below.

**Lemma 5.14.** *Let $U$ be a nonempty set of non-input gates, each with a path to the output. Then $\kappa(U)=\kappa(g)$ for every $g\in U$ if and only if $U$ lies inside a region. If $U\subseteq A_D$, then $D$ is a narrowest downstream cut for $U$, so $\kappa(U)=W(D)$, and $D(U)=D$.*

*Proof.* Suppose $U\subseteq A_D$. Then $D$ is a downstream cut for every $g\in U$, hence for $U$, so $\kappa(U)\le W(D)$; conversely a downstream cut for $U$ is a downstream cut for any $g\in U$, so its width is at least $\kappa(g)=W(D)$. Hence $\kappa(U)=W(D)=\kappa(g)$ for every $g\in U$, and $D$ is a narrowest downstream cut for $U$. Moreover, any narrowest downstream cut $D'$ for $U$ is a downstream cut for each $g\in U$ of width $\kappa(g)$, hence a narrowest one, so $D=D(g)$ is downstream of $D'$. Thus $D$ is downstream of every narrowest downstream cut for $U$, and $D=D(U)$.

Conversely, suppose $\kappa(U)=\kappa(g)$ for every $g\in U$, and let $D'$ be a narrowest downstream cut for $U$. As before, $D'$ is a narrowest downstream cut for each $g\in U$, so each $D(g)$ is downstream of $D'$. Fix $g,h\in U$. Every path from $h$ to the output contains a gate of $D'$, and its suffix from that gate contains a gate of $D(g)$; so $D(g)$ is a downstream cut for $h$, of width $\kappa(g)=\kappa(h)$, hence a narrowest one, and $D(h)$ is downstream of $D(g)$. By symmetry $D(g)$ is downstream of $D(h)$, and by Lemma 5.12 $D(g)=D(h)$. So all gates of $U$ have the same cut, and $U$ lies inside a region. $\square$

So the units the verifier may form inside a region are interchangeable for the bound: each has the region's cut, whether it is one gate or all of them. (Strictly speaking, interchangeable for the value term; finer units still change the cost of verification and, since the same incorrect gates then touch more units, the probability of acceptance.) Merging across regions raises the cut width of some gate, by the lemma, and can raise it a great deal, as the layer example shows. Its counterpart for the position term is the following.

**Proposition 5.15.** *Let $\mathcal V$ be a verifier with units $\mathcal P$, and let $\mathfrak E_\eta$ be the family of sets $E$ of units with $\sigma(E)>\eta$. For a set $E$ of units let $Q(E)$ be the set of regions that meet $G(E)$, let $\mathfrak Q_\eta:=\{Q(E):E\in\mathfrak E_\eta\}$, and for a set $Q$ of regions let $D_Q$ be the union of their cuts. Then*

$$
|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ \sum_{Q\in\mathfrak Q_\eta}2^{W(D_Q)}.
$$

*If moreover every unit that contains a gate with a path to the output lies inside a region, then for the sampling verifier, with $B$ as in §5.1, $\mathfrak Q_\eta$ is the family of sets of at most $B$ regions, and the bound is at most*

$$
\sum_{\ell\le B}\binom{R}{\ell}\,2^{\ell\kappa_{\max}}.
$$

*Proof.* By Lemma 5.1 it suffices to bound $|\mathcal Y(\mathcal T^{\mathrm{wt}}_{\mathcal V})|$. For $\tau\in\mathcal T^{\mathrm{wt}}_{\mathcal V}$ let $Q(\tau):=Q(\operatorname{err}_{\mathcal P}(\tau))\in\mathfrak Q_\eta$, so that the sets $\mathcal T_Q:=\{\tau\in\mathcal T^{\mathrm{wt}}_{\mathcal V}:Q(\tau)=Q\}$, $Q\in\mathfrak Q_\eta$, cover $\mathcal T^{\mathrm{wt}}_{\mathcal V}$. Fix $Q$ and $\tau\in\mathcal T_Q$. An incorrect gate of $\tau$ lies in $G(\operatorname{err}_{\mathcal P}(\tau))$, so either it has no path to the output, or it lies in a region $A_D$ with $D\in Q$, and then $D\subseteq D_Q$ is a downstream cut for it. Hence $D_Q$ is a downstream cut for $\operatorname{err}(\tau)$, and Corollary 5.7 gives $|\mathcal Y(\mathcal T_Q)|\le2^{W(D_Q)}$. Summing over $Q$ gives the first claim.

For the second, a unit inside a region meets only that region, and a unit whose gates have no path to the output meets none, so $|Q(E)|\le|E|\le B$ for $E\in\mathfrak E_\eta$; conversely any set of at most $B$ regions is $Q(E)$ for the set $E$ of one unit from each, since under the hypothesis every region is a union of units. There are $\binom R\ell$ sets of $\ell$ regions, and $W(D_Q)\le\sum_{A_D\in Q}W(D)\le|Q|\,\kappa_{\max}$, since each $W(D)$ equals $\kappa(U)$ for a unit $U\subseteq A_D$ by Lemma 5.14. $\square$

**Corollary 5.16.** *For the sampling verifier under the hypothesis of Proposition 5.15, with $1\le B\le R$,*

$$
\log_2|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ B\Bigl(\kappa_{\max}+\log_2\frac{eR}{B}\Bigr).
$$

*Proof.* As for Corollary 5.3. $\square$

Under the hypothesis the units refine the regions, so $R\le N$ and the corollary improves on Corollary 5.3 in both terms. For the matrix-multiplication circuit the regions are exactly the dot products, so nothing changes there. For GPT-2 Small the regions are about $1{,}300$ times fewer than the gates, and §5.6 gives the numbers.

The verifier of §6 samples in two stages. Its *replay units* $R_1,\ldots,R_m$ partition the non-input gates, each a union of verification units; it includes each replay unit in its sample independently with probability $q$, and then each verification unit inside a chosen replay unit independently with probability $s$. Writing $\mathcal P_r:=\{U\in\mathcal P:U\subseteq R_r\}$,

$$
\sigma_\theta(E)=\prod_{r=1}^{m}\bigl(1-q+q(1-s)^{|E\cap\mathcal P_r|}\bigr),\qquad\theta=(q,s),
$$

so Theorem 5.9 and the first bound of Proposition 5.15 apply to it, whatever its verification units. (Strictly speaking, $\sigma_\theta$ is the probability of accepting a transcript fixed in advance. The protocol of §6 lets the adversary fill in the interior of a replay unit after learning that it was chosen, and §6 shows that this does not raise the probability of acceptance above $\sigma_\theta$ of the completed transcript's set of incorrect units.) The sets in $\mathfrak E_\eta$ are then not small: as long as $1-q>\eta$, every set of verification units inside a single replay unit is in $\mathfrak E_\eta$, whatever its size, since it costs the adversary only the one factor $1-q$. Lemma 5.8 bounds what all such sets together can reach by $2^{\kappa(\mathcal P_r)}$.

## 5.5. Computing narrowest downstream cuts

Theorem 5.9 and Proposition 5.15 are only as useful as our ability to compute $\kappa$ and $D$, and the circuits we care about have $10^{10}$ gates or more, of which the verifier holds a description (§7) but never a gate list. We first compute the cut of one set of gates by a maximum-flow computation, then give three lemmas that determine the cuts of nearly all gates of an inference circuit from their wiring alone, and finally use the repetition in such circuits to do this for all $4\times10^{10}$ gates of GPT-2 Small at once.

### 5.5.1. One set of gates at a time

Finding a narrowest downstream cut is a minimum cut problem in which the capacities sit on the gates rather than on the edges between them, and the standard reduction moves them onto edges. Let $M:=1+\sum_{i\in[n]}w_i$. The network $G_A$ has a source $s$, a sink $t$, and two vertices $i_{\mathrm{in}}$ and $i_{\mathrm{out}}$ for each gate $i$, joined by an edge $i_{\mathrm{in}}\to i_{\mathrm{out}}$ of capacity $w_i$; it has an edge $j_{\mathrm{out}}\to i_{\mathrm{in}}$ of capacity $M$ for each $j\in\operatorname{args}(i)$, an edge $s\to a_{\mathrm{in}}$ of capacity $M$ for each $a\in A$, and an edge $o_{\mathrm{out}}\to t$ of capacity $M$ for each output gate $o$. The $s$–$t$ paths of $G_A$ correspond to the paths from $A$ to the output, and an $s$–$t$ cut of capacity less than $M$ is determined by the set $D(S)$ of gates $i$ with $i_{\mathrm{in}}$ on its source side $S$ and $i_{\mathrm{out}}$ off it. Minimum cuts are not unique, but the source sides of any two are contained in the source side of a third [TODO: Ford–Fulkerson 1962; Picard–Queyranne 1980], so there is a largest one, and it can be read off the residual graph of any maximum flow.

**Proposition 5.17.** *The minimum capacity of an $s$–$t$ cut in $G_A$ is $\kappa(A)$, and if $S$ is the source side of a minimum cut then $D(S)$ is a narrowest downstream cut for $A$. If $S^{\max}$ is the set of vertices from which $t$ cannot be reached in the residual graph of a maximum flow, then $D(S^{\max})=D(A)$.*

The proof is in Appendix 5.A. So $D(A)$ costs one maximum-flow computation on a network with $2n+2$ vertices and $O(n+m)$ edges, where $m$ is the number of argument occurrences in $C$, followed by one search backward from $t$. Whenever we speak of *the* cut of a set of gates, we mean $D(A)$.

### 5.5.2. Three lemmas that settle most gates

One maximum flow per gate is not an algorithm at $4\times10^{10}$ gates, and it is unnecessary. Three observations reduce the work, for inference circuits, to a check of the wiring of a few dozen kinds of gate.

The first is that hypotheses about paths to the output can be verified inside a small part of the circuit. For a set of gates $B$, let $\operatorname{Out}(B)$ be the set of gates of $B$ that are output gates or are arguments of gates outside $B$: the gates through which $B$ is read from outside.

**Lemma 5.18.** *Let $A\subseteq B\subseteq[n]$. If $D$ cuts $A$ from $\operatorname{Out}(B)$, then $D$ is a downstream cut for $A$.*

*Proof.* Let $P$ be a path from $A$ to the output. If every gate of $P$ lies in $B$, its last gate is an output gate in $B$, hence in $\operatorname{Out}(B)$. Otherwise let $h$ be the first gate of $P$ outside $B$; the gate before $h$ on $P$ lies in $B$ and is an argument of $h$, hence in $\operatorname{Out}(B)$. Either way $P$ has a prefix that is a path from $A$ to $\operatorname{Out}(B)$, and $D$ contains a gate of it. $\square$

In particular, the maximum-flow computation run on $B$ alone, with $\operatorname{Out}(B)$ in place of $\operatorname{out}(C)$, returns a downstream cut for $A$, whose width may stand in for $\kappa(A)$ in Proposition 5.10 even when it is not narrowest, at the cost of a weaker bound.

The second observation is that widths bound the size of a cut.

**Lemma 5.19.** *Let $w_{\min}$ be the least width of a gate of $C$. A narrowest downstream cut for a gate $g$ has at most $\lfloor w_g/w_{\min}\rfloor$ gates.*

*Proof.* Since $\{g\}$ is a downstream cut for $g$, a narrowest one $D$ has $|D|\,w_{\min}\le W(D)\le w_g$. $\square$

In the GPT-2 circuit of the next section every gate has width at least $\log_2 50{,}257\approx15.6$ and at most $32$, so the cut of a gate $g$ has at most two gates: it is $g$ itself, or another single gate that every path from $g$ to the output passes through, or a pair of gates such that every such path passes through one of them.

The third observation is that two lemmas about the wiring decide, for almost every gate, which of these it is. The first says that a set of gates whose only way to the output is through one gate, no wider than anything before it, shares that gate's cut.

**Lemma 5.20.** *Let $S$ be a set of gates, each with a path to the output, and let $o$ be a gate such that every path from $S$ to the output contains $o$, and every gate that precedes $o$ on such a path has width at least $w_o$. Then $\kappa(g)=\kappa(o)$ and $D(g)=D(o)$ for every $g\in S$.*

*Proof.* Fix $g\in S$. Every downstream cut for $o$ is a downstream cut for $g$, since every path from $g$ to the output contains $o$ and hence its suffix from $o$; so $\kappa(g)\le\kappa(o)\le w_o$. Now let $D$ be a narrowest downstream cut for $g$. If $D$ is a downstream cut for $o$, then $\kappa(o)\le W(D)=\kappa(g)$, so $\kappa(g)=\kappa(o)$ and $D$ is a narrowest downstream cut for $o$. Otherwise some path $P$ from $o$ to the output avoids $D$. Let $Q$ be a path from $g$ to $o$. Then $QP$ is a path from $g$ to the output, so it meets $D$ at a gate $d$ that lies on $Q$ before $o$, and $w_d\ge w_o$ by hypothesis; hence $\kappa(g)=W(D)\ge w_d\ge w_o\ge\kappa(o)\ge\kappa(g)$, so all are equal, $D=\{d\}$, and $\{o\}$ is a narrowest downstream cut for $g$. Moreover $\{o\}$ is then downstream of $\{d\}$: a path from $d$ to the output, preceded by the part of $Q$ up to $d$, is a path from $g$ to the output and so contains $o$, which is not on $Q$ before $d$.

Apply this to $D=D(g)$. In the second case, $\{o\}$ is a narrowest downstream cut for $g$ downstream of $D(g)$, while $D(g)$ is downstream of every narrowest downstream cut for $g$; by Lemma 5.12, $D(g)=\{o\}$, contradicting $d\neq o$. So $D(g)$ is a narrowest downstream cut for $o$, whence $D(o)$ is downstream of $D(g)$; and $D(o)$, being a narrowest downstream cut for $o$ and hence for $g$, has $D(g)$ downstream of it. By Lemma 5.12, $D(g)=D(o)$. $\square$

The second says that a gate whose influence spreads is its own cut.

**Lemma 5.21.** *Let $g$ be a gate with a path to the output such that no gate other than $g$ lies on every path from $g$ to the output, and such that every downstream cut for $g$ with two or more gates has width greater than $w_g$. Then $D(g)=\{g\}$ and $\kappa(g)=w_g$.*

*Proof.* $\{g\}$ is a downstream cut for $g$ of width $w_g$. Let $D\neq\{g\}$ be another. $D$ is nonempty because $g$ has a path to the output; if $|D|\ge2$ then $W(D)>w_g$ by hypothesis; and if $D=\{d\}$ with $d\neq g$ then $d$ lies on every path from $g$ to the output, contrary to hypothesis. So $\{g\}$ is the unique narrowest downstream cut for $g$. $\square$

The first hypothesis of Lemma 5.21 holds whenever two paths from $g$ to the output share no gate but $g$, for instance two paths to different output gates. The second holds whenever $w_g<2w_{\min}$, since a cut of two or more gates has width at least $2w_{\min}$; in the GPT-2 circuit this covers every $16$-bit gate. For a $32$-bit gate it fails only if some pair of gates of total width at most $32$ is a downstream cut for $g$, which requires two tokens ($31.2$ bits) or two $16$-bit values ($32$ bits, a tie that Lemma 5.12 resolves in favour of the pair, since the pair is downstream of $\{g\}$). This happens for $176{,}038$ of the $4\times10^{10}$ gates of GPT-2 Small.

Between them the two lemmas settle the gates of an inference circuit. Lemma 5.20 places the gates of an inner product or a reduction behind the gate that holds its result, since every path out of the operation passes through that gate and nothing before it is narrower; Lemma 5.21 makes that gate its own cut when several gates read it; and when only one gate reads it, Lemma 5.20 passes the cut on. What they leave undecided is a handful of gates whose two-gate cuts must be exhibited directly, and by Lemma 5.19 only pairs need be considered.

### 5.5.3. Repetition

An inference circuit is a few dozen kinds of gate, each repeated across layers, positions, heads, and coordinates. The GPT-2 Small circuit has $4.2\times10^{10}$ gates but only $82$ kinds of them, and its wiring is given by $119$ rules of the form "gate $(\ell,t,i)$ of kind $F$ reads gate $(\ell,t',j)$ of kind $F'$" with affine relations among the indices. The hypotheses of Lemmas 5.20 and 5.21 are statements about which kinds read which, so they can be checked once per kind and hold for every copy, and the number of copies is a product of index ranges, so regions are counted rather than enumerated. The exceptional cases arise where an index range degenerates, at the first position of a sequence, whose softmax has a single key, and at the last, which no later position reads. Splitting each kind by layer and by these boundary cases gives a few hundred thousand parametrized classes of gates, each assigned a cut by one of the two lemmas or by an explicit two-gate cut, in about a second. The wiring rules, and the cuts they yield, are checked gate for gate against an explicit construction of the circuit and the maximum-flow computation on small configurations (one or two layers, a model dimension of one to three, a vocabulary of two to four tokens, two or three positions).

## 5.6. Case study: GPT-2 Small

We apply the results above to an inference circuit for GPT-2 Small ($12$ layers, model dimension $d=768$, $12$ heads of dimension $64$, MLP width $3{,}072$, vocabulary of $50{,}257$ tokens) processing a $100$-token prompt and then generating $100$ tokens greedily. Its input gates are the weights, the embeddings, and the prompt tokens; its $n=42{,}387{,}408{,}594$ non-input gates are $42{,}361{,}101{,}422$ arithmetic gates and $26{,}307{,}172$ gates that round a result to FP16 at the boundary of an operation. Widths follow the profile under which the model is served: values at the boundaries of operations are FP16 ($16$ bits), accumulators and statistics are FP32 ($32$ bits), and a token is one of $50{,}257$ values, so its width is $\log_2 50{,}257\approx15.6$. The output gates are the $100$ generated tokens, so $W_{\mathrm{out}}\approx1{,}562$ bits. We take the sampling verifier with $p=\eta=1\%$, so $B=458$, and units that lie inside regions, as the second part of Proposition 5.15 requires.

**Regions.** The gates with a path to the output fall into $R=31{,}987{,}771$ regions, listed in Table 5.1; the remaining $2.80\%$ of gates have no path to the output. Three facts about the regions matter for the bound. First, the widest cut is $32$ bits, so $\kappa_{\max}=32$: the same as the widest gate, and there are $6{,}114{,}482$ regions with $32$-bit cuts. Second, regions are large: the average is about $1{,}300$ gates, and the largest is $2.5\times10^{8}$. Third, cuts are simple: all but $176{,}038$ gates have a single-gate cut, and only $133$ regions have a two-gate cut.

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
| Attention output $PV$ | one $n_t$-term inner product | $2n_t$ | $1{,}672{,}704$ | FP16 coordinate ($16$) | $0.79\%$ |
| Output projection | inner product and residual addition | $1{,}538$ | $1{,}672{,}704$ | FP16 residual coordinate ($16$) | $6.07\%$ |
| MLP up-projection | inner product and GELU | $1{,}546$ | $6{,}690{,}816$ | FP16 coordinate ($16$) | $24.40\%$ |
| MLP down-projection | inner product and residual addition | $6{,}146$ | $1{,}672{,}704$ | FP16 residual coordinate ($16$) | $24.25\%$ |
| LM head and argmax | $50{,}257$ inner products, argmax, last-layer tail | $8.9\times10^{7}$ | $100$ | token ($15.6$) | $21.51\%$ |

*Table 5.1.* Regions of GPT-2 Small by model part, for a $100$-token prompt and $100$ generated tokens; $n_t$ is the number of keys visible at position $t$. Gates per region include the gate that rounds the result to FP16; shares are of the $n$ non-input gates. The rows contain $31{,}987{,}506$ regions and $97.17\%$ of the gates. Not listed: the $2.80\%$ of gates with no path to the output; $132$ regions in which a one-key softmax places a head's whole query projection behind its single probability ($0.03\%$); and the $133$ regions with two-gate cuts ($0.0004\%$), namely $132$ two-key softmaxes whose $32$-bit statistics sit behind their two probabilities, and the $32$-bit statistics at the second-to-last position, which sit behind the last two tokens ($31.2$ bits).

**Where the regions come from.** Each row of the table is an instance of Lemmas 5.20 and 5.21. Every residual coordinate is read by the next layer and, through $K$ and $V$, by later positions, so by Lemma 5.21 it is its own cut; paths narrow only inside operations and at the token. One coordinate of a projection $xW+b$ is $768$ multiplications, $767$ additions, a bias addition, and a rounding to FP16, $1{,}537$ gates of which all but the last are $32$ bits wide. Lemma 5.20 places them behind the rounding gate and Lemma 5.21 makes the rounding gate its own cut, so an error anywhere in the inner product reaches the output through at most $16$ bits, and corrupting the rounding gate itself realizes all $2^{16}$ values of the cut. Where a single gate reads the rounding gate, the cut moves to that gate (the residual addition, the GELU). Reductions to an FP32 statistic behave the same way, and the $32$-bit values that branch are their own cuts: each centered coordinate $x_i-\mu$ feeds both the variance and the normalization, each softmax exponential both a probability and the denominator. That is why $\kappa_{\max}$ is $32$ rather than $16$.

At the output, each of the $50{,}257$ logits is read only by the argmax, whose result, the token, is narrower than everything before it, so Lemma 5.20 places behind it the LM head and everything else at that position whose only route to an output is that token, the last layer's $Q$ projection onward: about $8.9\times10^{7}$ gates. At the last position, which no later position reads, the region is the entire forward pass, $2.5\times10^{8}$ gates. At the $99$ prompt positions whose logits are never computed, the same gates reach no output at all; these are the $2.80\%$ in no region. The two-gate cuts listed in the caption arise where two results together are no wider than a $32$-bit gate.

**The bound.** For the sampling verifier, Corollary 5.16 gives

$$
\log_2|\mathcal Y(\mathcal T_{\mathcal V})|\ \le\ 458\Bigl(32+\log_2\frac{e\cdot31{,}987{,}771}{458}\Bigr)\ \approx\ 458\,(32+17.5)\ \approx\ 22{,}700\text{ bits}.
$$

For comparison, Corollary 5.3 with single gates as units gives $458\,(32+27.9)\approx27{,}400$ bits, and the output bound gives $1{,}562$. The regions save about $10$ bits per incorrect unit in the position term and cost nothing in the value term. But for this run the output bound is the tighter of the three, by a factor of nearly $15$: the adversary cannot use $32$ bits per incorrect unit because the whole output has only $1{,}562$ bits to carry them. The two bounds cross when the output is long relative to $B\kappa_{\max}$, and within the $1{,}024$-token context of GPT-2 Small this does not happen at $p=1\%$: for the longest generation the context allows after a $100$-token prompt, $924$ tokens, $R=3.3\times10^{8}$ and Corollary 5.16 gives $24{,}200$ bits against $W_{\mathrm{out}}=14{,}430$. They cross at $p\approx1.7\%$ for that generation, and at $p\approx14.5\%$, where $B=29$, for $100$ tokens. The regions and cuts are the same for the two-stage verifier of §6; only the sum over them, in Proposition 5.15, changes.

**What the analysis shows.** It settles the granularity: which gates may be grouped into a unit without raising any gate's cut width, namely anything inside a region, up to an inner product with its rounding gate or an LM head with its token. It does not settle whether the cut widths are realized in the output. For a region with cut $\{o\}$, three statements must be kept apart. First, $\{o\}$ is a narrowest downstream cut, so at most $2^{w_o}$ outputs are reachable by corrupting the region. Second, some corruption realizes all $2^{w_o}$ values at $o$; this holds for every single-gate cut, since $o$ itself may be the incorrect gate. Third, those $2^{w_o}$ values yield $2^{w_o}$ distinct outputs. The third holds for the token regions, whose cut is an output gate, and is open for the others: whether an individual $32$-bit region yields $2^{32}$ distinct $100$-token outputs is a question about values, not wiring. The output bound shows only that the contributions of many such regions cannot be independent in a short generation.

## 5.7. Implications for verification granularity

The verifier of §6 must choose its units, and the results above constrain the choice from the side of the bound.

First, units inside a region are interchangeable for the bound (Lemma 5.14). In GPT-2 Small an inner product of $1{,}537$ gates has a $16$-bit cut whether it is checked as one unit or as $1{,}537$, and the LM head of $7.7\times10^{7}$ gates has a $15.6$-bit cut as one unit. The regions of an inference circuit are its arithmetic operations: an inner product with its rounding gate, a reduction with its statistic, an LM head with its token.

Second, merging across regions costs. A unit spanning several regions has $\kappa$ at least the cut width of any one of them and at most the sum, and since Proposition 5.10 uses $\kappa_{\max}$ for every incorrect unit, one coarse unit can raise the value term for all of them. For a LayerNorm over $768$ coordinates at one of the first positions, checked as one unit, the cuts one can exhibit are its $768$ outputs ($12{,}288$ bits) and the tokens generated after it ($1{,}562$ bits), both far wider than the $32$-bit cut of any one of its gates. (A merge that raises no cut width above $\kappa_{\max}$, such as absorbing a $16$-bit region into a unit whose cut is already $32$ bits wide, costs nothing in Proposition 5.10, but it forfeits the hypothesis of the second part of Proposition 5.15.) Verification units should therefore not cross region boundaries where the cost model allows it.

Third, the position term counts regions, not units (Proposition 5.15), so units finer than regions do not inflate it: the verifier may choose small units to save on verification without loss in the bound.

Fourth, under the two-stage verifier, replay units enter the bound too. When $1-q>\eta$, every set of verification units inside one replay unit $R_r$ is in $\mathfrak E_\eta$, and all such sets together reach at most $2^{\kappa(\mathcal P_r)}$ outputs, by Lemma 5.8. The value term for an adversary that corrupts many units inside one replay unit is thus the cut width of the replay unit, not of the regions inside it, and the choice among replay units is a position term of its own. A replay unit should therefore be an operation whose outputs are few compared with its gates; §6 weighs this against the cost of replay.

Finally, the verifier of §7 never sees a gate list and cannot run the maximum-flow computation. For a unit $U$ it can use the downstream cut that the description of the circuit exposes without computation, the interface $\operatorname{Out}(U)$, which is a downstream cut for $U$ by Lemma 5.18, and the resulting bound is sound. It is the bound of this section only when $\operatorname{Out}(U)$ is the cut of $U$'s region, and for a unit that is a proper part of a region it generally is not: the region's cut may lie downstream of $U$ and outside it. To obtain the bound of Proposition 5.15 for small units, the description must carry, for each unit, either its region or a downstream cut for it, which the verifier can check by Lemma 5.18.

## Appendix 5.A. Deferred proofs

**Existence in Lemma 5.12.** Let $D_1$ and $D_2$ be narrowest downstream cuts for $A$. We show that there is a narrowest downstream cut for $A$ downstream of both. For $i=1,2$ let $S_i$ be the set of gates reachable from $A$ by a path containing no gate of $D_i$, and for any set of gates $S$ let $F(S)$ be the set of gates outside $S$ that lie in $A$ or have an argument in $S$. We use three facts about $F$.

First, if $S$ contains no output gate, then $F(S)$ is a downstream cut for $A$: a path from $A$ to the output either starts outside $S$, at a gate of $A$ that lies in $F(S)$, or leaves $S$ at some first gate, which has its predecessor on the path as an argument in $S$ and so lies in $F(S)$. Neither $S_1$ nor $S_2$ contains an output gate, since a path from $A$ to an output gate avoiding $D_i$ would contradict that $D_i$ is a downstream cut, so $F(S_1\cup S_2)$ and $F(S_1\cap S_2)$ are downstream cuts for $A$.

Second, $F(S_i)=D_i$. Let $g\in F(S_i)$, so $g\notin S_i$. If $g\in A$, the path of length zero at $g$ must contain a gate of $D_i$, since otherwise $g\in S_i$; that gate is $g$. If instead $g$ has an argument $j\in S_i$, a path from $A$ to $j$ containing no gate of $D_i$ extends to $g$, and the extended path must contain a gate of $D_i$, since otherwise $g\in S_i$; that gate can only be $g$. Either way $g\in D_i$, so $F(S_i)\subseteq D_i$; and $F(S_i)$ is a downstream cut for $A$, so $W(F(S_i))\ge\kappa(A)=W(D_i)$, and since widths are positive $F(S_i)=D_i$.

Third, every gate of $F(S_1\cup S_2)$ lies in $F(S_1)$ or in $F(S_2)$, since it is outside both $S_1$ and $S_2$ and either lies in $A$ or has an argument in one of them; every gate of $F(S_1\cap S_2)$ does too, since it is outside $S_1$ or outside $S_2$ and either lies in $A$ or has an argument in both; and a gate in both $F(S_1\cup S_2)$ and $F(S_1\cap S_2)$ lies in both $F(S_1)$ and $F(S_2)$, since it is outside both $S_1$ and $S_2$ and either lies in $A$ or has an argument in both. Hence $W(F(S_1\cup S_2))+W(F(S_1\cap S_2))\le W(D_1)+W(D_2)=2\kappa(A)$. Both terms on the left are widths of downstream cuts for $A$, so both are at least $\kappa(A)$, and therefore both equal $\kappa(A)$. In particular $D_3:=F(S_1\cup S_2)$ is a narrowest downstream cut for $A$.

Finally, $D_3$ is downstream of $D_1$, and by symmetry of $D_2$. Let $P$ be a path from a gate $d\in D_1$ to the output. If $d\notin S_1\cup S_2$ then, as $d\in D_1=F(S_1)$, $d$ lies in $A$ or has an argument in $S_1\subseteq S_1\cup S_2$, so $d\in D_3$ and $P$ contains it. Otherwise $P$ starts inside $S_1\cup S_2$ and ends outside it, so it has a first gate outside $S_1\cup S_2$, and that gate lies in $D_3$. $\square$

**Proof of Proposition 5.17.** Call the edge $i_{\mathrm{in}}\to i_{\mathrm{out}}$ the *internal edge* of gate $i$. A path $a=j_0,\ldots,j_t$ from $A$ to the output *lifts* to the $s$–$t$ path $s,(j_0)_{\mathrm{in}},(j_0)_{\mathrm{out}},\ldots,(j_t)_{\mathrm{out}},t$, and since the edges of $G_A$ alternate between internal edges and edges into an in-vertex, every $s$–$t$ path is the lift of a path from $A$ to the output. A cut is specified by its source side $S$, with $s\in S$ and $t\notin S$; its capacity is the total capacity of the edges leaving $S$, and $D(S)$ is the set of gates whose internal edge leaves $S$.

Let $c$ be the minimum cut capacity. The source side consisting of $s$ and every in-vertex has as its leaving edges exactly the internal edges, so $c\le\sum_i w_i<M$, and no edge of capacity $M$ leaves the source side of a minimum cut.

Let $S$ be the source side of a minimum cut. The lift of a path from $A$ to the output starts at $s\in S$ and ends at $t\notin S$, so some edge of it leaves $S$; that edge has capacity less than $M$, so it is the internal edge of a gate of $D(S)$, and that gate lies on the path. Thus $D(S)$ is a downstream cut for $A$. The edges leaving $S$ are exactly the internal edges of the gates of $D(S)$, so $\kappa(A)\le W(D(S))=c$.

Conversely, let $D$ be any downstream cut for $A$, and let $S_D$ be the set of vertices reachable from $s$ without using an internal edge of a gate of $D$. Then $t\notin S_D$, since a path from $s$ to $t$ avoiding those edges would be the lift of a path from $A$ to the output avoiding $D$. Every edge leaving $S_D$ is an internal edge of a gate of $D$, for otherwise its head would be reachable too. So the cut with source side $S_D$ has capacity at most $W(D)$, and $c\le W(D)$. Minimizing over $D$ gives $c\le\kappa(A)$, hence $c=\kappa(A)$, and $W(D(S))=\kappa(A)$ for every minimum cut $S$. This proves the first two claims.

For the third, fix a maximum flow. For any source side $S$, the value of the flow is the flow on the edges leaving $S$ minus the flow on the edges entering $S$. This is at most the capacity of the cut, with equality exactly when every edge leaving $S$ is saturated and every edge entering $S$ carries no flow, that is, when no residual edge leaves $S$. Since the flow is maximum, its value is $c$, and so $S$ is the source side of a minimum cut if and only if no residual edge leaves $S$. The set $S^{\max}$ contains $s$, because a maximum flow admits no augmenting path, and does not contain $t$. No residual edge leaves $S^{\max}$: from the head of such an edge $t$ would be reachable, and then also from its tail. So $S^{\max}$ is the source side of a minimum cut, and $D(S^{\max})$ is a narrowest downstream cut for $A$. If $S$ is the source side of any minimum cut, then no residual edge leaves $S$, so no residual path from a vertex of $S$ reaches $t\notin S$, and $S\subseteq S^{\max}$.

Now let $D'$ be any narrowest downstream cut for $A$, and let $S_{D'}$ be the source side constructed from it above. Its cut has capacity at most $W(D')=\kappa(A)$, so it is a minimum cut, and $S_{D'}\subseteq S^{\max}$. We claim that $d_{\mathrm{in}}\in S_{D'}$ for every $d\in D'$. Since $D'$ is narrowest and $w_d>0$, the set $D'\setminus\{d\}$ is not a downstream cut for $A$, so some path from $A$ to the output meets $D'$ only at $d$; the lift of its prefix ending at $d$ reaches $d_{\mathrm{in}}$ from $s$ using no internal edge of a gate of $D'$. Finally, let $P$ be a path from $D'$ to the output, starting at some $d\in D'$. Its lift starts at $d_{\mathrm{in}}\in S^{\max}$ and ends at $t\notin S^{\max}$, so some edge of it leaves $S^{\max}$; as $S^{\max}$ is the source side of a minimum cut, that edge has capacity less than $M$, so it is the internal edge of a gate of $D(S^{\max})$ lying on $P$. Thus $D(S^{\max})$ is downstream of $D'$. Being itself a narrowest downstream cut for $A$, it is $D(A)$. $\square$

---

**Notes for the author** (not part of the section)

- *Numbering and file.* Numbered §5 to match the outline and the cross-references in `section-7-secure-circuit-compilation.md` (which cites "§5.4" for granularity); your "section 4" with "4.1 and 4.2" as anchors is read as §5.1 and §5.2. Results are numbered §-prefixed as in §7. Deferred proofs are in Appendix 5.A at the end of the file.
- *Vocabulary.* Defined terms used in the section, all of them from your §5.1/§5.2 or the outline except the last three: gate, value set, width, input/output gates, transcript, well-typed, correct/incorrect, unit, verifier, sampling verifier, reachable outputs, output bound, value term and position term (§5.2), path, downstream cut, narrowest downstream cut, downstream of (Definition 5.11), downstream-most (said once, Lemma 5.12), region (Definition 5.13), replay unit and verification unit (from §6), interface $\operatorname{Out}(U)$ (from §7). Symbols with no name attached: $\sigma$, $\mathfrak E_\eta$, $B$, $\kappa$, $D(A)$, $Q(E)$, $D_Q$. Stripped in this pass: "acceptance function", "admissible error set", "budget", "lossless", "live/dead region", "exit rule", "fan-out rule", "locality", "write-out" (now "the gate that rounds the result to FP16" / "rounding gate"), "funnel", "exit", "certify/certificate" (now "shows"/"has"), "canonical". If "region" is one too many, "class" is the plain alternative (the regions are the equivalence classes of gates under $D(g)=D(h)$); it is a single find-and-replace.
- *Anchors.* §5.1 is your text extended with what §5.2 needs: value sets $V_i$ with $w_i=\log_2|V_i|$, $\operatorname{err}(\tau)$, units, $\operatorname{err}_{\mathcal P}$, the sampling model of a verifier and $\sigma$, $\mathfrak E_\eta$, $B$, $W(S)$, $G(E)$. §5.2 is our final version with two edits: "one billion times the size of $m^*$" became "a billion bits long" ($m^*$ is not defined here), and ill-typed values are described as lying outside their value sets. The Rinberg example is still a TODO.
- *Generality.* The bounds are stated for any verifier that samples a set of units and checks them (Theorem 5.9, Proposition 5.15); the sampling verifier is the running case with $\mathfrak E_\eta=\{E:|E|\le B\}$, and §6's two-stage verifier is given at the end of §5.4.2 with $\mathcal P_r$ (units inside $R_r$) so the exponent is well-typed. The parenthetical there commits §6 to showing that adaptive completion of a chosen replay unit is accepted with probability at most $\sigma_\theta$ of the completed transcript's error set. The first bound of Proposition 5.15 needs no hypothesis on the units; only the $\binom R\ell$ closed form needs units inside regions.
- *Theorem 5.6* does not assume equal inputs: an input gate on which two transcripts disagree is incorrect in one of them, so it lies in $E$. Corollary 5.7 likewise has no "with correct inputs".
- *Downstream-most.* Uniqueness is proved in Lemma 5.12; existence is sketched there as the vertex form of the closure of minimum-cut source sides under union and proved in Appendix 5.A. Citation TODO: Ford–Fulkerson's book has the union/intersection lemma; Picard–Queyranne 1980 is the standard reference for the lattice of minimum cuts. Both need $w_i>0$, which $|V_i|\ge2$ guarantees.
- *GPT-2 numbers* are from `circuit-cut gpt2` (1.3 s). Denominator: $n=42{,}387{,}408{,}594$ non-input gates $=42{,}361{,}101{,}422$ arithmetic $+26{,}307{,}172$ FP16 rounding gates; every share in Table 5.1 and the text is over this $n$ (the earlier draft mixed denominators). $R=31{,}987{,}771$ excludes the gates with no path to the output (the code's $31{,}987{,}772$ includes them as one class). Per-region gate counts include the rounding gate. Break-even figures come from runs at $100/300/500/700/924$ generated tokens (GPT-2 Small's context is $1{,}024$; the earlier "$1{,}500$ tokens" was outside it): at $p=1\%$ the cut bound never beats the output bound within the context ($924$ tokens: $24{,}233$ vs $14{,}430$ bits); they cross at $p\in[1.70\%,1.71\%)$ for $924$ tokens and at $p\in[14.2\%,14.7\%)$, $B=29$, for $100$ tokens. Corollary 5.3's $27{,}400$ uses $\log_2(en/458)=27.9$ with the new $n$.
- *§5.5.3* describes what the code does ($82$ kinds, $119$ wire rules, algebraic counting, validation against the explicit solver on small configurations) without naming it; adjust if §9 wants the credit.
- *Open for §7.* The last paragraph of §5.7 claims only soundness for interfaces as cuts. A description node's local cut relative to its exits can be wider than the global one even without re-entry (two live one-bit exits that merge outside the node into one one-bit gate), so identifying per-kind cuts with regions would need a hypothesis on the exterior; and a unit that is a proper part of a region does not see its region's cut in its own interface at all. If §7 wants the Proposition 5.15 bound for small units, the description must carry a region label or a downstream cut per unit, checkable by Lemma 5.18.
- *Cross-references to check.* §6 (cost model and parameter choice; I dropped the "§6.5"/"§6.6" subsection numbers since I cannot see that file), §7 (interfaces as cuts; $\operatorname{Bound}$ as a fold over kinds), §8 (how $\eta$ and $|\mathcal Y(\mathcal T_{\mathcal V})|$ enter $\delta$), §4.2.3 (the granularity promise, discharged by Lemma 5.14 and §5.7).
