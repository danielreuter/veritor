# Staged commitments for sparse verification

The verifier computes the indexed circuit

\[
C=\mathsf{Compile}(G,x,a)
\]

itself. The client does not commit to a circuit description. The commitments
below bind only claimed values at positions determined by \(C\).

Let \(I=\operatorname{in}(C)\), let \(D_C=[n]\setminus I\) be the non-input
gates, and let \(O\) be the ordered output positions. The verifier
deterministically derives a public partition

\[
\mathcal P=(P_1,\ldots,P_m)
\]

of \(D_C\) into nonempty replay units. It also fixes sampling probabilities
\(q,s\in[0,1]\).

## Commitment domains

The boundary of \(\mathcal P\) is

\[
B
=
I
\cup
\operatorname{supp}(O)
\cup
\bigcup_{r=1}^m
\bigcup_{i\in P_r}
\left(\operatorname{args}(i)\setminus P_r\right).
\]

Thus \(B\) contains every input, every output, and every value read across
replay units. Define the interior of unit \(r\) by

\[
U_r:=P_r\setminus B.
\]

The sets \(B,U_1,\ldots,U_m\) are disjoint and partition \([n]\). For every
gate \(i\in P_r\), the value at \(i\) and every value in
\(\operatorname{args}(i)\) therefore lie in \(B\cup U_r\).

For a public indexed domain \(Q=\{d_1<\cdots<d_k\}\), the client commits to
values on \(Q\) using leaves

\[
H\!\left(
\mathsf{leaf},
\lambda_Q,
d_t,
V_{d_t},
\mathsf{enc}_{d_t}(v_{d_t})
\right).
\]

Here \(\lambda_Q\) domain-separates the protocol session, the
verifier-computed circuit, the partition, the exact ordered domain, and either
the boundary or unit \(r\). Tree shape, padding, and the empty-domain root are
canonical. Any circuit identifier appearing in \(\lambda_Q\) is computed by
the verifier; it is not a client circuit commitment.

An opening supplies a position, a typed value, and an authentication path. The
verifier rejects malformed values, positions outside \(Q\), invalid paths, and
missing openings. After publishing a root, the client cannot successfully
open one position to two values, even after earlier challenges, except with
negligible probability. A malformed root need not open every position;
failure to open only creates another reason to reject.

## Protocol

1. The client commits to claimed values on \(B\), producing \(R_B\).
2. The client opens every position in \(I\cup\operatorname{supp}(O)\). The
   verifier checks all openings and verifies

   \[
   v|_I=x,
   \qquad
   v|_O=y^*,
   \]

   where the second restriction preserves the order and multiplicity of
   \(O\).
3. Independently for every replay unit \(r\), the verifier includes \(r\) in
   \(J\) with probability \(q\), then reveals \(J\).
4. For every \(r\in J\), the client commits to all values on \(U_r\),
   producing \(R_r\). An honest client obtains these values by replaying the
   unit from its committed boundary values. The roots are fixed before the
   within-unit samples are drawn.
5. Independently for every \(i\in P_r\), \(r\in J\), the verifier includes
   \(i\) in \(T_r\) with probability \(s\), then reveals the sets \(T_r\).
6. For every \(i\in T_r\), the client opens \(v_i\) and every value in
   \(\operatorname{args}(i)\). Boundary positions are opened against \(R_B\);
   all other positions are opened against \(R_r\). The verifier authenticates
   and type-checks every opening and checks

   \[
   v_i=f_i\bigl(v_{\operatorname{args}(i)}\bigr).
   \]

The verifier accepts iff the input and output are pinned and every sampled
local check succeeds.

## Exact local-check soundness

Fix typed boundary values \(b\) satisfying \(b|_I=x\) and \(b|_O=y^*\). For
an assignment \(\tau_r\) to \(U_r\), let \(e_r(\tau_r;b)\) be the number of
incorrect gates in \(P_r\), and define

\[
\ell_r^\star(b)
:=
\min_{\tau_r}e_r(\tau_r;b).
\]

Because every cross-unit read lies in \(B\), these minimizations are
independent across replay units.

**Proposition.** In the ideal binding model, the optimal local-check
acceptance probability conditioned on \(b\) is

\[
\prod_{r=1}^m
\left(
1-q+q(1-s)^{\ell_r^\star(b)}
\right).
\]

For fixed \(C\) and \(\mathcal P\), the overall optimum is therefore

\[
\max_{\substack{b|_I=x\\b|_O=y^*}}
\prod_{r=1}^m
\left(
1-q+q(1-s)^{\ell_r^\star(b)}
\right).
\]

A cryptographic implementation adds at most its negligible binding error.

*Proof.* Condition on the selected set \(J\). For \(r\in J\), the client fixes
its interior before learning \(T_r\). An interior with \(\ell\) incorrect
gates passes exactly when all \(\ell\) gates escape the independent
\(s\)-sample, with probability \((1-s)^\ell\). Its best choice therefore
attains \(\ell_r^\star(b)\). Averaging over the independent \(q\)-selection of
replay units gives the product.

The same value is obtained under eager commitment to every interior before
either sampling stage. The minimizing interiors are compatible because they
are disjoint and share only \(b\); committing to all of them eagerly attains
the displayed product. Conversely, staging cannot improve any selected unit
beyond its minimum \(\ell_r^\star(b)\). Thus staged and eager commitment have
the same optimal local-check soundness value, although they have different
costs. \(\square\)

If every \(\ell_r^\star(b)=0\), the minimizing interiors combine with \(b\)
into a globally correct transcript. Since the inputs and outputs are pinned,
an incorrect claimed output therefore forces \(\ell_r^\star(b)>0\) for at
least one unit. This argument relies on placing every cross-unit read in
\(B\).

The proof does not require a malformed Merkle root to encode a complete
vector. Every successful opening fixes at most one typed value at its
position. Fill unopened positions arbitrarily. Gates that could pass before
this completion remain correct, while missing openings can only cause
rejection.

For one unit containing \(\ell\) incorrect gates, the survival factor is

\[
1-q+q(1-s)^\ell.
\]

Although each gate has marginal sampling probability \(qs\), gates in the
same replay unit share the first-stage event. Hence this factor is generally
larger than

\[
(1-qs)^\ell.
\]

Two-stage sampling is therefore not equivalent to independently sampling
every gate with probability \(qs\). Staging the commitments adds no further
loss relative to the chosen two-stage experiment.

Write

\[
f_\ell(q,s):=1-q+q(1-s)^\ell.
\]

Suppose independent first-stage units with error counts \(\ell_t\) contribute
at most \(A(\ell_t)\) bits of reachable-output capacity, and these
contributions add. Their joint survival probability is
\(\prod_t f_{\ell_t}(q,s)\). Hence acceptance probability greater than
\(e^{-\Lambda}\) implies

\[
\sum_t A(\ell_t)
<
\Lambda
\max_{\ell\geq1}
\frac{A(\ell)}{-\ln f_\ell(q,s)}.
\]

This is the minimax objective in Appendix A. The commitment theorem supplies
the survival factors; the choice of \(A(\ell)\) and the additivity assumption
come from the separate reachable-output analysis.

## Expected cost

Let \(c_B\) include the always-paid boundary commitment and mandatory
input/output openings. Let \(\gamma_r\) be the cost of reconstructing unit
\(r\) and committing to \(U_r\), and let \(\alpha_i\) be the additive cost of
opening and checking gate \(i\). Then

\[
\mathbb E[\mathsf{cost}]
=
c_B
+
q\sum_{r=1}^m\gamma_r
+
qs\sum_{i\in D_C}\alpha_i.
\]

This equality uses additive accounting. Shared multiproofs, batching, or
other nonlinear costs should instead be evaluated on the jointly sampled
sets.

Appendix A chooses \(q\) and \(s\) under a homogeneous approximation to this
cost. Its available budget must exclude the always-paid \(c_B\). Its
first-stage object must also match the replay unit used here; if one “request”
contains several replay units, its exact survival probability is the product
of their factors, not one factor with their error counts summed.

Appendix A's closed form is an interior small-probability approximation. The
bounds \(q,s\in[0,1]\), boundary optima, integer error counts, and the
\(\gamma=0\) case must be checked against the exact objective. Its numerical
rows are analytic calibrations rather than empirical measurements.
