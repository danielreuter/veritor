# Security argument

This document argues, component by component, that the protocol in
`src/veritor` has the soundness properties it claims, and points at the attack
test that demonstrates each failure mode is caught.  The tests live in
`tests/veritor/security/`; every protocol fixture they use (a small compiled
chain circuit, expectations, prover/verifier runs, tamper hooks) is in
`tests/veritor/security/conftest.py`.

Model.  A client (the adversary) runs a circuit `C` for a verifier.  The
verifier trusts only the compiler's `(C, I)`, its own parameters and seeds, and
SHA-256.  The protocol is transparent: values are opened in the clear.  The
claim is soundness -- a false claimed output is accepted with probability at
most `sigma(E)` for the error set `E` the prover must incur -- and the capacity
bound `Bound(C, I, theta)` on the outputs an adversary can reach at all.
Privacy is out of scope.

Cryptographic assumptions, used throughout: SHA-256 is collision resistant
(so a Merkle root fixes one value per position, and a canonical digest fixes
one message); HMAC-SHA256 keyed by a secret 32-byte seed is a PRF (so a
selection is unpredictable until its seed is revealed and independent of the
message it is derived over); Python's `secrets` is a CSPRNG.  Where a section
says "bound", it means bound through a tagged, length-framed SHA-256 hash under
these assumptions.

Throughout, "the header" is the object both parties fix before any message
(session id, compiled digest, `theta`, `eta`, public inputs, claimed outputs,
`kappa_W`), and "kappa_W" is the per-model commitment to the weight gates.
Field names are avoided deliberately; the header and kappa_W are changing on
`main`.

Verdict summary (proved+tested / tested only / gap) is in the final table.

---

## 1. Position binding and domain separation

Files: `protocol/merkle.py`, `protocol/domains.py`, `protocol/session.py`
(`_Layout`).

**Claim.**

1. Every address has exactly one owner: kappa_W for a `weight` gate, the
   boundary for an `in` gate or a declared replay-unit output, otherwise the
   replay unit `r` whose interior contains it.
2. A leaf hash binds `(domain_id, rank, position, schema, value)`; a node hash
   binds `(domain_id, level, index, left, right)`; padding leaves are
   `H(pad, domain_id, rank)`; the empty root is `H(empty, domain_id)`.
3. `domain_id` binds `(binding digest, owner, position-set identity digest,
   count)`, where the binding digest is the header digest for the boundary,
   the replay-phase digest for an interior, and a fixed tag for kappa_W.
4. Hence a root cannot be reused across sessions, phases, owners or position
   sets, and an opening authenticates only the value committed for that
   position of that domain.
5. The verifier derives every domain from trusted data; the wire carries no
   domain description.

**Argument.**

- Ownership is the three-way rule in `_Layout.owner` (`session.py`, lines
  172-180): weights first, then the boundary, then
  `index.replay_units.owner(address)`.  The three position sets are disjoint
  by construction of the index (`domains.py` module docstring; `weight` gates
  are never boundary positions, and interiors are `R_r` minus its declared
  outputs, `Index.interior`).  Both parties use the same rule, so a value has
  one committing root.
- `CommitmentDomain.__post_init__` (`merkle.py`, lines 63-78) computes
  `domain_id = H("domain", binding, owner + 2, positions.identity_digest,
  count)`.  `leaf` (84-92), `node` (94-95), `empty_root` (97-98) and the
  padding leaves in `MerkleTree.__init__` (131-134) all take `domain_id` as
  their first argument.  `_hash` (33-41) length-prefixes every part under a
  versioned frame, so no two argument lists collide by concatenation.
- The bindings: `boundary_domain` uses `header.digest` (`domains.py`, 46-49),
  which covers the session id, compiled digest, `theta`, `eta`, the public
  I/O and kappa_W; `interior_domain` uses the replay-phase digest (52-59),
  which chains back to the header and the boundary message (section 2);
  `weight_domain` uses a fixed tag plus the index's weight-position identity
  (30, 40-43), because kappa_W outlives sessions.
- `verify_opening` (`merkle.py`, 174-198) rejects a count that disagrees with
  the domain, a position that is not in the domain's position set (the
  `rank` lookup fails, so padding ranks are unreachable), a path of the wrong
  length, and then recomputes the root from the domain-bound leaf and node
  hashes.  Under collision resistance an opening that verifies is the leaf
  the committer hashed at that rank, with that value.
- The verifier builds the domains itself: `VerifierSession.__init__` (419)
  for kappa_W, `receive_boundary` (516-518) and `receive_interiors`
  (566-568).  `wire.py` encodes a commitment as `(root, count)` and an
  opening as `(position, value, path)` only.

**Attack tests** (`test_binding.py`).

- `test_root_is_bound_to_binding_owner_position_set_and_count`,
  `test_leaf_binds_rank_position_schema_and_value`,
  `test_padding_leaves_are_domain_bound_and_cannot_be_opened`: the primitive
  level, one field at a time.
- `test_interior_committed_under_the_boundary_domain_is_invalid_opening` and
  `test_boundary_committed_under_an_interior_domain_is_invalid_opening`: a
  prover commits under the wrong domain; every opening is `INVALID_OPENING`
  (an empty interior under the wrong domain is `INVALID_COMMITMENT`).
- `test_boundary_message_replayed_from_another_session_is_invalid_opening`:
  the boundary of session 1 in session 2 (same values, same seeds).
- `test_commitment_count_disagreeing_with_the_domain_is_invalid_commitment`.
- `test_equivocating_on_a_boundary_value_between_phases_is_invalid_opening`:
  `v` at the boundary, `v'` in the evidence for the same address.
- `test_two_units_reading_one_address_cannot_be_shown_different_values` and
  `test_every_address_has_exactly_one_owner`: single owner.
- `test_the_wire_carries_no_prover_described_domain`.
- `test_kappa_w_is_bound_to_the_index_of_its_model`,
  `test_kappa_w_with_another_count_is_rejected_before_any_commitment`,
  `test_weight_opened_with_another_value_is_invalid_opening`,
  `test_interior_domain_is_bound_to_the_replay_phase`.

**Gaps.**

- kappa_W's domain is bound to a fixed tag, the weight-position identity and
  the count, and its leaves to the value width -- not to the gate set's
  semantics or to the compiled digest.  A kappa_W computed for another model
  with the same weight positions and widths is a *valid root* for this
  index; the header (which carries the compiled digest) is what ties a
  kappa_W to a model, and obtaining the right kappa_W for a model is the
  operator's duty (section 10).  On `main` the domain becomes weight ranks,
  which binds even less; the header binding must carry the weight.
- The boundary root is bound to the header digest, so two sessions with
  identical headers share a boundary domain.  `make_expectation` draws a
  random 16-byte session id; a caller that supplies its own must keep it
  unique.
- All of the above assumes SHA-256 collision resistance; a collision on a
  leaf or node hash lets the prover equivocate.

Verdict: proved + tested.

---

## 2. Staged commitments and challenge derivation

Files: `protocol/phases.py`, `protocol/challenge.py`, `protocol/session.py`.

**Claim.**  `J` is a function of `(q_seed, header, boundary message)` and the
`q` seed is revealed only after the boundary commitment and its public-I/O
openings have been accepted; `T` is a function of `(s_seed, everything so
far)` and the `s` seed is revealed only after the interior commitments have
been accepted.  Neither selection can be ground: before the seed is revealed
the prover has no information about it, and after it is revealed the messages
it depends on are fixed.  Both state machines reject calls out of order.

**Argument.**

- `boundary_phase(header, boundary)` (`phases.py`, 20-24) hashes the header
  digest and the boundary message; `derive_replay_selection` (`challenge.py`,
  252-263) is `bernoulli_subset(q_seed, "q/replay-unit", boundary_phase, ...)`.
  The `ReplayChallenge` object carrying the seed is created inside
  `receive_boundary` (`session.py`, 547) after `_accept_commitment`, the
  coverage check and the public-I/O comparison have passed (516-538).
- `replay_phase` (27-31) covers the boundary phase and the replay challenge;
  `interior_phase` (34-38) covers the replay phase and the interior message;
  `derive_sample_selection` (266-292) is
  `bernoulli_subset(s_seed, "s/verification-unit", interior_phase, ...)`.  The
  `SampleChallenge` carrying the `s` seed is created inside
  `receive_interiors` (578) after every interior commitment was accepted
  under its domain.
- Seeds are 32-byte secrets: `Expectation.__post_init__` (85-89) refuses any
  other length, `make_expectation` (98-126) draws them with
  `secrets.token_bytes(32)`, and the PRF is HMAC-SHA256 keyed by the seed
  (`_prf`, 61-79).  Under the PRF assumption the prover's view before the
  reveal is independent of the selection; grinding the boundary would require
  evaluating the PRF without its key.
- After the reveal nothing can move: the boundary root is bound to the header
  and sits in the transcript; the interior domains are bound to the replay
  phase, which covers the boundary message and `J`
  (`interior_domain`, `domains.py` 52-59); the sample derivation covers the
  interiors.  A prover that changes its boundary after seeing `J` has no
  domain to open under.
- Ordering: `ProverSession._expect` (279-281) and `VerifierSession._expect`
  (463-468) fail closed; the verifier's code is `INVALID_PHASE`.

**Attack tests** (`test_staging.py`).

- `test_verifier_rejects_messages_out_of_order`,
  `test_prover_rejects_calls_out_of_order`.
- `test_replay_selection_is_derived_from_seed_header_and_boundary_only`.
- `test_prover_changing_its_boundary_after_seeing_j_is_invalid_opening`.
- `test_transcript_with_an_altered_selection_is_challenge_mismatch`.
- `test_make_expectation_draws_fresh_seeds_by_default`.
- **Negative result**:
  `test_reused_seeds_let_the_prover_predict_and_evade_both_selections`.  With
  the seeds of a completed session reused, the prover computes `J` from its
  own boundary and `T` from its own interiors before sending them, grinds a
  free interior value, and gets a false output accepted with certainty.

**Gaps.**

- Seed freshness is a convention of `make_expectation`; nothing in the
  protocol detects reuse (Finding F1).
- Soundness is per session.  A client may open many sessions; the chance that
  some one of `k` attempts accepts a false output is at most `k * sigma(E)`.
  Rate limiting and accounting for attempts is the operator's.
- A verifier that lets the client influence `session_id`, `q_seed` or
  `s_seed` loses the property.  The `Expectation` API takes them as plain
  arguments.

Verdict: proved + tested (seed freshness: gap, tested negatively).

---

## 3. Sampling distribution

File: `protocol/challenge.py`.

**Claim.**  The joint law of `J` is that of independent `Bernoulli(q)` coins,
one per replay unit, within total variation `2^-190`; given `J`, the law of
`T` is independent `Bernoulli(s)` coins over the verification units of the
units in `J`.  Consequently a transcript whose error set is `E` is accepted
with probability `sigma(E) = prod_r (1 - q + q (1 - s)^{l_r})` up to that
bias.

**Argument.**  `bernoulli_subset` (213-249) draws `K ~ Binomial(N, p)` by
inverting the CDF against a 256-bit HMAC output (`_binomial_count`, evaluated
in 512-bit fixed point; the docstring bounds the bias below `2^-190` for
`N < 2^64`) and then a uniform `K`-subset with Floyd's algorithm from
rejection-sampled uniforms (`_floyd_subset`, 190-210; `uniform_below`,
87-118).  A `Binomial(N, p)` mixture of uniform `K`-subsets is exactly the law
of `N` independent `p`-coins, so the only deviation is the count inversion's
bias.  The two stages use distinct tags and phase digests, so their PRF
streams are independent.  `derive_sample_selection` ranks the candidates block
by block over the selected replay units (266-292), so `T` is `Bernoulli(s)`
over exactly the units of `J`.  The existing tests cover the mechanism:
`test_count_inversion_agrees_with_exact_rational_inversion`,
`test_marginals_pairs_and_counts_match_independent_coins`,
`test_sample_selection_ranks_the_selected_replay_units_blocks` in
`tests/veritor/protocol/test_challenge.py`.

**Attack tests** (`test_sampling.py`).

- `test_acceptance_rate_matches_survival_of_the_error_set`: a prover corrupts
  a fixed `E` of two verification units, in distinct replay units
  (`sigma = 9/16`) and in the same one (`sigma = 5/8`), at `q = s = 1/2`; the
  acceptance rate over 2000 fresh seed pairs is within 4 standard deviations
  of the exact `survival` from `veritor.analysis.probability`.
- `test_selection_law_alone_matches_survival_over_many_seeds`: the same for
  the selections alone, without running the protocol.
- `test_survival_is_the_product_of_per_replay_unit_factors`.

**Gaps.**

- The `2^-190` bias means `P[accept] <= sigma(E) + 2^-189`, not `sigma(E)`;
  `eta` is compared exactly.
- `uniform_below` gives up after 4096 rejections (`ResourceLimit`); each
  attempt succeeds with probability at least `1/2`, so this is unreachable
  for an honest seed.
- The statistical tests are 4-sigma checks, not proofs; the law itself rests
  on the PRF assumption.

Verdict: proved + tested.

---

## 4. Local checks

Files: `protocol/session.py` (`_check_unit`, `receive_boundary`),
`core/circuit.py` (`check_gate`, `decode`).

**Claim.**  For every sampled verification unit the verifier opens exactly the
addresses the unit reads or writes, each under its owner, decodes each value
canonically, compares every `in` gate with the header's public input, accepts
a `weight` gate only as kappa_W's leaf, and checks every other gate's relation
against the opened argument values.  At the boundary, before any sampling,
the public inputs and the claimed outputs are opened and compared with the
header, exhaustively.

**Argument.**

- `_Layout.required(unit)` (182-199) is the sorted set of the unit's gates
  plus `In(unit)`, each with its owner; `_check_unit` (607-658) demands the
  evidence open exactly those positions in that order (`COVERAGE_MISMATCH`),
  opens each under its owner through `_open` (492-510, `INVALID_OPENING`),
  decodes it with the circuit's codec (`INVALID_VALUE` for anything that does
  not round-trip, including a value outside the gate's width), then walks the
  unit's gates: an `in` gate's payload must equal the header's public input
  of its rank (`PUBLIC_IO_MISMATCH`); a `weight` gate has nothing to check
  beyond its opening under kappa_W; any other gate goes through
  `circuit.check_gate(args, out)` (`RELATION_REJECTED`).  Arguments are the
  owners' committed values -- the prover never states an argument, only
  opens positions.
- `receive_boundary` (512-552) demands the boundary openings cover exactly
  the public I/O addresses in boundary order, opens each and compares inputs
  and claimed outputs with the header (`PUBLIC_IO_MISMATCH`).  Because every
  `in` gate is a boundary position and every output is a boundary position
  (`_Layout.__init__`, 161-168), this check is exhaustive and precedes the
  reveal of the `q` seed.
- A gate whose semantics raise is `TRUSTED_SERVICE_FAILURE`: the verifier
  fails closed.

**Attack tests** (`test_local_checks.py`).

- `test_every_non_source_gate_of_a_sampled_unit_is_checked`.
- `test_wrong_input_value_is_caught_at_the_boundary_before_any_sampling`: a
  prover that satisfies every gate relation from a wrong input.
- `test_wrong_claimed_output_with_honest_values_is_caught_at_the_boundary`.
- `test_altered_weight_in_the_run_is_caught_only_when_a_reader_is_sampled`:
  both outcomes -- `INVALID_OPENING` when a reader of the weight is sampled,
  `ACCEPTED` when none is.
- `test_noncanonical_encoding_of_a_committed_value_is_invalid_value`,
  `test_value_outside_the_gate_width_is_invalid_value`.
- `test_evidence_must_open_exactly_the_required_addresses_in_order`,
  `test_boundary_must_open_exactly_the_public_io_in_order`.
- `test_gate_arguments_are_the_owners_committed_values_not_the_provers_claims`.

**Gaps.**

- Only sampled units are checked; that is the design, and section 5 bounds
  what survives.
- Canonicity of values is the gate set's codec (`encode`/`decode`).  A gate
  set whose `decode` accepts two encodings of one value would let a prover
  commit the same value twice; the built-in codec is strict.
- A weight gate is accepted as *whatever kappa_W says*; a wrong kappa_W is a
  provenance problem (section 10), not a protocol one.

Verdict: proved + tested.

---

## 5. Soundness of the acceptance bound and of `Bound`

Files: `analysis/bound.py`, `analysis/probability.py`, `analysis/series.py`,
`analysis/reference.py`.

### Claim A (acceptance)

For every prover strategy and every boundary `B` it commits, let
`l_r >= 1` for each replay unit `r` that no interior can make consistent with
`B` (and `l_r = 0` otherwise).  Then `P[accept] <= sigma(E*) + 2^-189` for
any `E*` with `|E* ∩ R_r| = l_r`, and the claimed output is an output of a
transcript with error set `E*`.

**Argument (the reduction).**

1. `B` is fixed before `J` (section 2).  Given `B`, each replay unit `r` has a
   fixed family `F_r(B)` of error sets `E ⊆ R_r` that some interior of `r`
   can realize (verification units holding a gate whose relation fails on the
   committed values).  `∅ ∈ F_r(B)` iff `Out(R_r)` in `B` is what `R_r`
   computes from its inputs in `B` and kappa_W.  If the claimed output is not
   `C(x, W)`, then by induction along the address order some `r` has
   `∅ ∉ F_r(B)`: otherwise every boundary value would be the honest one.
2. The interiors committed after `J` choose one member `E_r ∈ F_r(B)` for each
   `r ∈ J`; the choice may depend on `J` but only on `J`.
3. Given `J` and the interiors, `T` is the PRF of the still-secret `s` seed
   over the interior phase; the prover cannot evaluate it, so (up to the
   sampler bias) `P[T ∩ E = ∅ | J, interiors] = prod_{r ∈ J} (1 - s)^{|E_r|}
   <= prod_{r ∈ J} (1 - s)^{l_r}` with `l_r = min_{E ∈ F_r(B)} |E|`.
4. Averaging over `J`, which is `Bernoulli(q)` per unit independent of `B`:
   `P[accept] <= prod_r (1 - q + q (1 - s)^{l_r}) = sigma(E*)`.  The claimed
   output is determined by `B`, and `B` is consistent with an interior whose
   error set is the minimizing `E*`, so the output is an output of a
   transcript with error set `E*`.

`survival` in `probability.py` (50-56) is exactly `prod_r f(l_r)` with
`f(l) = 1 - q + q (1 - s)^l` (42-47), as `Fraction`s.

### Claim B (capacity)

`Bound(C, I, theta).bits` at threshold `eta` is at least `log2 |Y_eta|`, where
`Y_eta` is the union, over all error sets `E` with `sigma(E) > eta`, of the
outputs of transcripts with error set `E`.

**Argument, docstring step by step against the code.**

1. *Downstream cut.*  If all incorrect gates lie in index nodes `S_1..S_m`,
   every value outside them is a function of the values on their declared
   outputs, and inputs and weights are pinned, so at most
   `2^(sum out_bits(S_j))` outputs are reachable.  `reference.cover_bits` and
   `reference.cut_bits` enumerate this for small circuits;
   `test_cover_by_index_nodes_is_never_below_the_exact_cut` checks cover >=
   cut.
2. *Per-kind covers.*  `_Fold.series` (244-264): a verification kind covers
   its `l >= 1` subsets by itself (`unit_series(out_bits)`); a replay kind
   convolves its children's series over copies (`power`, `multiply`) and then
   `cap`s at its own `out_bits`, i.e. takes the cheaper of "cover the pieces"
   and "cover the whole node".  Both are valid covers of every `l`-subset, so
   the per-`l` weight is at least the total weight of the distinct covers of
   those subsets.
3. *Admissibility as a knapsack.*  `unit_cost` (67-76) is `c(l) = -ln f(l)`
   rounded **down** (`cost - cost * 2^-40 - 2^-48`); `budget` (87-95) is
   `Lambda = ln(1/eta)` rounded **up**; `_Fold.bucket` (220-230) floors the
   cost onto a grid whose step is inflated by `1 + 2^-50` and then steps back
   while `index * step > cost`, so every cost is rounded **down** onto the
   grid.  Every rounding admits more error sets, never fewer.  `knapsack`
   (284-297) raises each replay kind's cost polynomial to its copy count
   (`sparse_power`), convolves the kinds, and sums every bucket strictly below
   the budget (`prefix_sums` reversed): the grid result is exact for the
   relaxed survival `sigma~ >= sigma` described in the module docstring.
4. *Error-count truncation.*  `_errors_limit` (232-240) stops where further
   errors no longer change the bucket; subsets with more errors are lumped
   into a tail at the cost of `limit + 1` errors -- a **lower** cost than
   theirs, admitting more.
5. *Laplace bound.*  `laplace` (301-337) is Chernoff's
   `min_t t Lambda + sum_K n_K ln sum_l V_K(l) e^{-t c(l)}` with the same
   rounded-down costs and rounded-up budget; a minimum over `t` of upper
   bounds is an upper bound.
6. *Cap and integer count.*  `bound` (131-167) takes
   `min(knapsack, laplace, out_bits)`; the circuit's own interface bounds
   everything.  `_integer_count` (170-195) replaces `bits` by
   `log2 floor(2^bits)`, valid because `|Y_eta|` is an integer; see the
   rounding note below.
7. *Series arithmetic.*  Every operation in `series.py` adds an explicit slack
   of `(terms + 4) * 2^-50` relative plus absolute (`_up`, 28-36), so every
   entry is an upper bound on the exact `log2` quantity.

**Rounding directions, in one place.**  Costs down; budget up; grid step up
and grid index down (costs land on a lower bucket); error counts beyond the
limit lumped at a lower cost; series entries up; the final cap is an exact
minimum; `_integer_count` floors a power that has been scaled **up** by
`1 + 2^-45` (never undercounts the integer), takes `log2` exactly for a power
of two and rounded **up** by one ulp otherwise, and never exceeds the input.
The remaining unguarded float operations (a few multiplications in `laplace`,
`copies * ...` in `value`) err by at most one ulp each, far inside the
`2^-40` relative margins on costs and budget and the `4 * 2^-50` slack of the
series; the `1 + 2^-45` scaling in the integer step absorbs roughly `2^7`
ulps more.  There is no formal end-to-end error budget (Gap).

### The rule implemented: source-only kinds contribute only `l = 0`

`_Fold.series`, lines 249-254, and the module docstring, lines 40-48:

~~~python
if row.role == VERIFICATION:
    if row.size == row.source_inputs + row.source_weights:
        # nothing but source gates: never incorrect, so only l = 0
        result = empty_series()
    else:
        result = unit_series(row.out_bits)
~~~

*Soundness.*  A verification unit made of source gates alone is never in the
error set of a transcript the verifier can accept: an `in` gate's committed
value must equal the header's public input (checked for every input at the
boundary, before sampling, and again when sampled), and a `weight` gate's only
admissible value is its leaf under kappa_W; neither has a relation to
violate.  So for every admissible `E` containing such a unit `V`,
`outputs(E) = outputs(E \ V)` and `sigma(E \ V) >= sigma(E) > eta`: the
union `Y_eta` is unchanged when those `E` are dropped, and the fold's sum only
loses terms of weight `2^0` per such subset.  The rule tightens the bound and
remains an upper bound.  Before the rule, a kind with `m` source-only copies
contributed a factor `sum_l C(m, l)` on error counts that cost survival
without buying any output.

**Attack tests** (`test_bound_soundness.py`).

- `test_union_over_random_markings_is_below_the_fold`: random small circuits
  with *random marks* (the client's choice of partition granularity, wide and
  narrow units, maximal and minimal interfaces) against the exhaustive union
  of `reference.accepted_outputs`; `log2 |Y_eta| <= bits`; `bits == 0` only
  when the union is one output, and a fully checked run (`q = s = 1`,
  `eta = 0`) has `bits == 0`; with `q = 0` the fold is capped at the
  interface and the union is every output.
- `test_whole_unit_corruption_is_covered_by_the_unit_interface`: every gate
  of a unit corrupted at once.
- `test_integer_count_never_undercounts_and_never_exceeds_its_input`: the
  tightening against exact `Decimal` logarithms.
- `test_fully_checked_run_has_exactly_zero_capacity`.
- `test_source_only_units_contribute_no_error_terms`,
  `test_source_only_rule_is_exact_against_the_enumerated_union`: the rule
  above, against the enumerated union and against the per-unit cover sum
  with and without source-only units.
- Existing: `test_random_small_circuits_union_is_below_the_fold`,
  `test_fold_sits_between_the_union_and_the_relaxed_per_set_sum`,
  `test_knife_edge_is_admitted_by_the_grid_only`,
  `test_a_unit_of_source_gates_has_no_capacity` in
  `tests/veritor/analysis/test_bound.py`.

**Gaps.**

- Claim A is argued, not mechanically checked; the empirical checks are
  section 3's.
- The union definition is over error sets; a transcript's outputs are
  bounded by the *cut* of its error set, and the fold covers by index nodes
  (`cover >= cut`).  This is loose when a unit's outputs are not all
  downstream of its wrong gates.
- Float arithmetic: sound by the rounding discipline above, without a formal
  error budget.  `bits` is compared with an integer `U_max`.

Verdict: Claim A proved (reduction) + tested statistically; Claim B proved
step by step + tested exhaustively on small circuits.

---

## 6. Admission

Files: `protocol/session.py` (`_admit`), `protocol/parameters.py`.

**Claim.**  `eta` is the verifier's and bound into the header; the
denominators of `theta` and `eta` are capped before any sampling; `U_max` and
`W_max` are checked from the per-kind counts alone, before any commitment; a
transcript recorded under another `eta` is `EXPECTATION_MISMATCH`.

**Argument.**  `VerifierParameters.policy` (`parameters.py`, 64-74) returns
the client's `theta` unchanged and the header takes `eta` from
`expectation.parameters` (`session.py`, 408-416).  `_admit` (426-461) runs in
`VerifierSession.__init__` before the phase is set to `boundary`: it enforces
`max_probability_denominator_bits` on both `theta` and `eta`, `max_units` on
both unit counts, `max_positions_per_unit` per kind, then
`expected_work(compiled, theta, |IO|) > W_max` is `WORK_BUDGET_EXCEEDED` and,
when `U_max` is set, `bound(...).bits > U_max` is `POLICY_REJECTED`.
`expected_work` (90-121) is a closed form over `index.kinds()`.
`verify_transcript` (`verify.py`, 38-49) compares the transcript's `eta` and
then the whole header with the verifier's own.

**Attack tests** (`test_admission.py`).

- `test_theta_with_an_enormous_denominator_is_resource_limit`.
- `test_policy_whose_bound_exceeds_u_max_is_policy_rejected`.
- `test_run_whose_expected_work_exceeds_w_max_is_work_budget_exceeded`.
- `test_eta_is_the_verifiers_and_bound_into_the_header`.
- `test_admission_checks_unit_counts_against_the_limits`.
- **Gap demonstrated**:
  `test_default_parameters_waive_u_max_and_admit_a_policy_that_checks_nothing`.

**Gaps.**

- `VerifierParameters()` defaults to `U_max = None`, which *waives* the
  capacity check, and `policy()` accepts any `theta`, including `(0, 0)`.  A
  verifier built with the defaults accepts any claimed output under
  `theta = (0, 0)`: nothing is sampled and the only checks compare the
  client's boundary with the client's header.  `Bound` reports this honestly
  (`bits == out_bits`), but nobody asks it (Finding F2).
- `W_max` bounds *expected* work; the realized work of one run can exceed it
  (the selection is random).  The hard limits (`max_openings`,
  `max_proof_bytes`, ...) bound the worst case.

Verdict: proved + tested; default `U_max` is a gap.

---

## 7. Compile determinism and canonical encoding

Files: `compile/description.py`, `core/identity.py`, `protocol/wire.py`.

**Claim.**  The same description bytes yield the same digest and the same
`(C, I)` on any machine; bytes that are not the canonical serialization of
their own JSON value are rejected before any definition is examined;
description size, definition count, step count, nesting depth and output-run
fan-out bound the compile work; marks are part of the digest; transcripts are
canonical bytes and anything else is `NONCANONICAL_TRANSCRIPT` or
`MALFORMED_TRANSCRIPT`.

**Argument.**  `parse_description` (`description.py`, 375-430): size limit,
strict JSON (duplicate keys and non-finite constants rejected), then
`canonical_description(value) != payload` is a `CompileError` before the
`version`, `definitions` or `root` fields are read.  Each definition's body
must match its digest, definitions are bounded by `max_definitions`, resolved
output runs by `max_output_runs` (per definition) and `max_output_runs_total`
(fd478dd).  The compiled digest is `Compiled.digest_of(description digest,
gate set digest)` (`compiler.py`, 71-73); the role marks are fields of the
definition bodies, hence of the digest.  Everything is a pure function of the
bytes: no clock, no randomness, no environment.  `wire.decode_transcript`
(158-250) enforces `max_transcript_bytes`, rejects floats and duplicate keys
while parsing, type-checks every field against a fixed schema, and finally
re-encodes the decoded transcript and compares it with the input bytes
(`NONCANONICAL_TRANSCRIPT` on any difference; `test_noncanonical_bytes_are_rejected`,
`test_malformed_bytes_are_rejected` in `tests/veritor/protocol/test_wire.py`).
Nesting depth is bounded by the schema itself; a `RecursionError` during
parsing is `MALFORMED_TRANSCRIPT`.

**Attack tests** (`test_canonical.py`).

- `test_same_description_bytes_give_the_same_digest_and_layout`.
- `test_reencoded_description_bytes_are_rejected_before_any_compile_work`:
  whitespace, key order, a duplicate key, invalid UTF-8, oversize bytes.
- `test_changing_a_mark_changes_the_digest_and_the_header`: the digest
  changes and the recorded transcript is `EXPECTATION_MISMATCH` under the
  other compilation.
- `test_transcript_with_a_noncanonical_or_malformed_encoding_is_rejected`,
  `test_canonical_bytes_are_unique_per_transcript`.
- `test_a_description_of_a_trillion_gates_compiles_in_bounded_time`,
  `test_nesting_deeper_than_the_limit_is_a_compile_error`.
- Existing, for the fan-out limits:
  `test_interfaces_resolving_to_too_many_runs_are_rejected_without_doing_the_work`,
  `test_the_total_number_of_runs_over_a_description_is_capped`,
  `test_admission_does_not_scale_with_the_input_count` in
  `tests/veritor/compile/test_out_runs.py`.

**Gaps.**

- Determinism across machines rests on `json.dumps` with sorted keys and
  ASCII output, and on the gate set's digest, both of which are pure Python;
  no cross-implementation vectors exist.
- `json.loads` runs over the whole payload before the canonical check; that
  is `O(bytes)` under `max_description_bytes`, not zero work.
- Compilation work is bounded by limits, not by a proof of linearity in the
  description size; the trillion-gate test shows the lazy circuit does not
  materialize gates.
- `VerificationLimits.max_nesting_depth` and `max_artifact_bytes` are
  declared but enforced nowhere (Finding F6); the transcript schema makes
  them unnecessary today.

Verdict: proved + tested.

---

## 8. Tiling and refinement

File: `core/index.py` (`validate_marks`, `Index`), `protocol/session.py`
(`_Layout.required`).

**Claim.**  Every gate is in exactly one replay unit and exactly one
verification unit; verification units refine replay units; a unit reads
outside itself only through kappa_W, the boundary, or its own replay unit's
interior, so `Out(R_r)` is the cut between replay units.

**Argument.**  `validate_marks` (598-660) checks once per definition: a
replay-marked definition contains no replay mark and is tiled by
verification marks; a verification-marked definition contains no mark of
either role; the root is tiled by replay marks; a marked definition has gates.
Tiling means every step above a mark is a call into a tiled definition, so no
gate is left uncovered, no two marks overlap, and a verification mark cannot
straddle two replay units (it would have to contain a replay mark).
`_Layout.required` (182-199) independently refuses a unit that reads an
address owned by another replay unit that is not a boundary position
(`INVALID_COMPILED_RESULT`); the compiler makes this unconstructible, because
a replay unit's declared outputs are exactly what may be read from outside
and those are boundary positions.

**Attack tests** (`test_tiling.py`).

- `test_every_gate_is_in_exactly_one_replay_unit_and_one_verification_unit`,
  `test_verification_units_refine_replay_units`,
  `test_cross_unit_reads_go_only_through_declared_outputs`.
- `test_marks_leaving_a_gate_uncovered_are_a_compile_error`,
  `test_nested_or_straddling_marks_are_a_compile_error`.
- `test_layout_rejects_a_circuit_that_reads_across_the_cut`: the compiler
  rule makes the case unconstructible through the API, so the test forges a
  `Compiled` whose circuit reads across the cut and checks that
  `_Layout.required` rejects it with `INVALID_COMPILED_RESULT`.

**Gaps.**

- The tiling is checked per definition digest, once, and relies on the
  description's call graph being what `Index` walks; an inconsistency between
  `DescriptionCircuit` and `Index` would not be caught by `validate_marks`
  (it is caught, defensively, by `_Layout.required`).

Verdict: proved + tested (the cross-cut read is tested only on a forged
circuit; the compiler rule is tested directly).

---

## 9. Transcript verification

File: `protocol/verify.py`.

**Claim.**  A third party holding the expectation (seeds, parameters, public
I/O, kappa_W) and `(C, I)` recomputes every challenge from the recorded
messages; any message the prover alters after the fact is caught with a
specific code, and the offline verdict equals the interactive one.

**Argument.**  `verify_transcript` (17-62) decodes canonically, builds a fresh
`VerifierSession` from the expectation (which re-runs admission and rebuilds
the header), compares `eta` and the header (`EXPECTATION_MISMATCH`), feeds the
recorded boundary and interiors through the same `receive_*` methods, and
compares the derived challenges with the recorded ones: a different seed is
`EXPECTATION_MISMATCH`, a different selection `CHALLENGE_MISMATCH`.  Because
the interior phase covers the interior roots, a changed root changes `T`;
because the boundary domain is bound to the header, a changed header or
boundary root fails every opening.

**Attack tests** (`test_transcript.py`).

- `test_altering_a_recorded_message_is_caught_with_the_expected_code`, one
  case per field: header (session id, compiled digest, `theta`, `eta`, public
  inputs, claimed outputs, kappa_W) -> `EXPECTATION_MISMATCH`; boundary root
  or any boundary opening value/path -> `INVALID_OPENING`; boundary count or
  interior count -> `INVALID_COMMITMENT`; boundary position or a dropped
  opening / batch / interior -> `COVERAGE_MISMATCH`; either seed ->
  `EXPECTATION_MISMATCH`; either selection -> `CHALLENGE_MISMATCH`; evidence
  values or paths under any owner -> `INVALID_OPENING`; evidence position ->
  `COVERAGE_MISMATCH`.
- `test_altered_interior_root_changes_the_sample`: with `s < 1` the altered
  root is `CHALLENGE_MISMATCH` (with `s = 1` the sample is fixed and the
  openings fail instead).
- `test_the_recorded_transcript_verifies_only_under_its_own_expectation`.
- `test_a_rejected_interaction_leaves_no_transcript`.
- `test_transcript_verdict_equals_the_interactive_verdict`, including a
  dishonest transcript whose error set escaped sampling: accepted offline
  too, with the same sampled units.

**Gaps.**

- A transcript proves what the *verifier with these seeds* would have said;
  a third party must trust the expectation it is handed (the seeds are in the
  transcript, but the parameters, public I/O and kappa_W are not
  independently authenticated).
- `VerifierSession.__init__` raises `ProtocolError` (an exception, not a
  report) when the expectation names another compiled digest or its values do
  not encode; `verify_transcript` does not catch it.  This is the verifier's
  own inconsistency, not prover-controlled.

Verdict: proved + tested.

---

## 10. What is NOT achieved, or achieved only by convention

- **Client code runs in the verifier's process.**  On `main` the verifier
  will execute the client's constructor `G` to obtain the description; there
  is no sandbox, only output-size limits.  At this commit the verifier
  compiles description *bytes* (data, not code) under `CompilationLimits`;
  whoever runs `G` runs untrusted code.
- **kappa_W provenance.**  The verifier must obtain the model's kappa_W from a
  trusted source before the epoch's first request.  Nothing in the code
  enforces when or from whom; the `Expectation` takes it as an argument.  A
  kappa_W over the wrong weights makes every weight check meaningless
  (section 4), and kappa_W is not self-describing (section 1).
- **Seed freshness and secrecy.**  `make_expectation` draws fresh 32-byte
  seeds with `secrets`; reuse or leakage before the reveal is fatal
  (section 2, negative test).  `session_id` uniqueness likewise.
- **Retries.**  Soundness is per session; the operator must bound attempts.
- **No privacy.**  Every opened value is in the clear; the transcript reveals
  inputs, outputs, sampled interiors and sampled weights.
- **Cryptographic assumptions.**  SHA-256 collision resistance (Merkle
  binding, digests, hash chain), HMAC-SHA256 as a PRF (selections), Python's
  `secrets` as the CSPRNG (seeds, session ids).  None are proven here.
- **Floats in verifier decisions.**  `Bound` is float arithmetic compared
  with an integer `U_max`; it is sound by the rounding discipline of
  section 5 (every approximation rounds toward admitting more, and the
  integer tightening rounds up), without a machine-checked error budget.  No
  other verifier decision uses floats: survival, `eta`, `theta` and the
  expected work are `Fraction`s; sampling is integer arithmetic.
- **Fail-closed on gate-set bugs.**  A gate whose `check_gate` raises is a
  rejection (`TRUSTED_SERVICE_FAILURE`), never an acceptance; a `decode` that
  is not strict is a gate-set bug the protocol cannot see.
- **Denial of service.**  Bounded by explicit limits (`CompilationLimits`,
  `VerificationLimits`), which are conservative defaults, not proofs of
  linear work.
- **Default parameters.**  `VerifierParameters()` waives `U_max` (Finding F2).
- **`ProtocolError` vs verdict.**  Misuse of the API by the verifier's own
  operator (wrong compiled, unencodable values, calls out of order on the
  prover side) raises exceptions rather than producing verdicts.

---

## Findings

Severity is about soundness impact when the component is used as documented.

**F1 -- Seed reuse defeats the protocol (medium, operational; documented,
not fixable in the protocol).**  `Expectation` accepts any 32 bytes as seeds;
a verifier that reuses a session's seeds lets the prover predict `J` and `T`
and forge acceptance with certainty
(`test_reused_seeds_let_the_prover_predict_and_evade_both_selections`).
`make_expectation` draws fresh seeds by default; the docstring says so.
Proposed fix: derive per-session seeds from a long-lived verifier secret and
the session id (`HMAC(master, session_id || "q")`), so that seed freshness
reduces to session-id uniqueness, and document that the seeds must never be
supplied by anything the client can influence.

**F2 -- Default `U_max = None` waives the capacity check (medium, design
footgun; documented, not fixed).**  `VerifierParameters()` has
`max_capacity=None` and `policy()` accepts every `theta`; under the defaults a
client proposing `theta = (0, 0)` gets any claimed output accepted with
nothing sampled (`test_default_parameters_waive_u_max_and_admit_a_policy_that_checks_nothing`).
The module docstring documents `None` as "waives the check", so this is a
documented default rather than a broken property, but it contradicts the
README's framing of `U_max` as the verifier's guarantee.  Proposed fix: make
`max_capacity` a required argument, or default it to `0` (only fully checked
runs admitted), or refuse `q = 0` / `s = 0` unless `U_max` is set.  Not
changed here because the `Expectation`/header API is being changed on `main`.

**F3 -- `_integer_count` could round below the true count (low; fixed).**
`floor(2.0 ** bits)` could return `n - 1` when `bits` was `math.log2(n)`
rounded down by an ulp, and `math.log2(count)` for a non-power-of-two is not
rounded up.  Both undercount by at most one part in `2^52`, which cannot flip
an integer `U_max` decision below 49 bits, but the docstring's "never below
the log2 of the union size" did not hold as stated.  Fixed in
`analysis/bound.py` by scaling the power by `1 + 2^-45` before the floor,
taking `log2` exactly for powers of two and `nextafter(..., inf)` otherwise,
and capping at the input (`test_integer_count_never_undercounts_and_never_exceeds_its_input`).

**F4 -- Source-only kinds were counted with `l >= 1` (informational,
looseness; fixed).**  The knapsack ranged over error counts of verification
kinds that hold nothing but source gates, which can never be incorrect.
Sound but loose.  Fixed by the `l = 0` rule of section 5, with the soundness
argument there and tests `test_source_only_units_contribute_no_error_terms`,
`test_source_only_rule_is_exact_against_the_enumerated_union`.

**F5 -- kappa_W is not self-describing (low, convention; documented).**  The
weight root's domain binds a fixed tag, the index's weight positions and the
count, not the compiled digest or gate-set semantics; the header binds kappa_W
to the model.  Operators must not reuse a kappa_W across models with the same
weight layout (`test_kappa_w_is_bound_to_the_index_of_its_model` shows what is
and is not bound).

**F6 -- Dead limits (informational).**  `VerificationLimits.max_nesting_depth`
and `max_artifact_bytes` are fields nothing reads.  Transcript nesting is
bounded by the fixed schema and Python's recursion limit
(`MALFORMED_TRANSCRIPT`), so there is no exposure; the fields should either be
enforced or removed so the limits object does not promise what it does not
check.

---

## Verdicts

| # | Property | Verdict |
|---|----------|---------|
| 1 | Single owner per address; leaf/node/padding binding; domain binding; no cross-session/phase/owner reuse; verifier-derived domains | proved + tested |
| 2 | `J` from `(q_seed, header, boundary)`, seed revealed after; `T` from everything so far; state machines | proved + tested |
| 2 | Seed freshness | gap (convention; negative test) |
| 3 | `J`, `T` are independent Bernoulli coins within `2^-190`; acceptance rate is `sigma(E)` | proved + tested (statistically) |
| 4 | Every gate of a sampled unit checked against owners' values; inputs exhaustive at the boundary; weights only via kappa_W; canonical values; exact coverage | proved + tested |
| 5A | `P[accept] <= sigma(E*)` for every strategy | proved (reduction) + tested statistically |
| 5B | `Bound` upper-bounds `log2 |Y_eta|` under the union definition; every approximation rounds toward admitting more | proved + tested exhaustively on small random circuits with random marks |
| 5 | Source-only rule `l = 0` | implemented + tested |
| 6 | `eta` the verifier's and in the header; denominators capped; `U_max`, `W_max` from counts before any commitment; other `eta` is `EXPECTATION_MISMATCH` | proved + tested |
| 6 | `U_max` enforced by default | gap (F2) |
| 7 | Deterministic digest and `(C, I)`; non-canonical bytes rejected first; bounded compile work; marks in the digest; canonical transcripts | proved + tested |
| 8 | Tiling, refinement, cut through declared outputs | proved + tested (cross-cut read on a forged circuit only) |
| 9 | Offline verification recomputes every challenge; every post-hoc alteration caught | proved + tested |
| 10 | Sandbox for `G`, kappa_W provenance, seed freshness, retries, privacy, crypto assumptions, float budget | not achieved / conventional, listed |
