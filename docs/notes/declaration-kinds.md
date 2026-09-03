# Declaration kinds beyond the VU: source-position, port and RU-scope pardons

Design note. Nothing here is implemented: `PROTOCOL_VERSION`, the wire
formats, the digests and the verifier's acceptance rules are unchanged, and
the three kinds below are specified so that they can be priced and argued
about before any of them is built. The prices are those of the ladder in
`docs/notes/late-advice.md`, which owns them; where this note's argument goes
beyond that ladder it says so, and section 8 lists every disagreement. The
counts that motivate the note are section 6 of `docs/honest-prover.md` (rows
`H4a`-`H4d`, `tests/veritor/stress/test_honest_systematic.py`).

Notation as in `late-advice.md`: `J` the q-challenge (opened RUs), `T` the
s-challenge (sampled VUs), `u(1) = W_V + log2 |S|` the price of one VU
declaration fixed before `J`, `u_post(1) = rho log2 (1 / (1 - s))` the price of
one made after `J`, `n_RU` the RUs of the run, `n_W` the cells of `kappa_W`,
`W_cell` the width of a committed word (16 bits), `W_R` the width of an RU's
outputs, `d` the readers of a cell in one RU, `mu_f` the largest Poisson mean
with `P[X <= f] > eta`. Headline operating point: `q = 1.57e-8`, `s = 8.91e-3`,
`rho = 4.74e11`, `U_0 = 1.90e13`, `n_RU = 2.93e13` requests of 1024 tokens,
`n_W = 7e10`, `W_R = 8192`, `d = 1024`, `u_post(1) = 6.12e9`. Simulation
point: `theta = (1/2, 1/8)` on the cluster fixture of the systematic test
(`rho = 740`, `u_post(1) = 142.6`, 413 committed positions, two pods, 144,000
step RUs in the hour-long run of two pods, `W_R = 320`).

## 1. What M6 says, and what it cannot say cheaply

The protocol has one declaration kind. `InteriorMessage.declarations` names up
to `Header.max_faults` VUs of the opened RUs; a declared VU keeps its
committed value, every reader is checked against that value, and its own
relation is never checked. The message is made after `J` (the prover only
learns its faults by replaying the opened RUs), so `Bound` charges
`declared_bits`, `u_post(1)` per declaration at the scattered channel:
`6.12e9` bits at the headline, `3.2e-4 U_0`.

A declaration names a symptom. A systematic fault has one cause and as many
symptoms as the wiring gives it: a weight cell read wrong on one pod for an
hour is one VU per position on that pod (144,000 at the toy shape, of order
`5e7` on an 8-GPU pod at production decode rates), or, under tokens-only
recording, the tokens whose argmax it moved; a stale version or a wrong kernel
path is every dot on the pod. Every count is far above any `f_max`, so the
honest prover as built re-serves the pod-hour or is rejected
(`docs/honest-prover.md`, section 6). The kinds below name the cause instead.

The common frame. A pardon of kind `k` is a message `m` from a finite set
`M_k`; the verifier's check of a sampled VU becomes "the relation holds under
`m`" instead of "the relation holds". The reachable outputs under a fixed `m`
are the honest outputs of a modified circuit `C_m` plus what the base channel
allows, so a pardon fixed before `J` costs `log2 |M_k|` bits by the
fixed-in-advance argument of `veritor.analysis.faults` (`|Y| <= sum_m
2^U_0(C_m) = |M_k| 2^U_0`, one factor per pardon in the budget). The same
message made after `J` is adaptive, and its price is whatever the adaptivity
argument allows: for a VU declaration `declared_bits`; for the kinds below,
what section 6 says. Nothing may be declared after `T`.

## 2. Source-position pardon

*Statement.* "In scope `S`, the committed word at position `w` of commitment
`K` was read as `v'`." `K` is `kappa_W` (a weight cell), the input commitment
(a prompt token) or the boundary commitment (an RU output: a KV entry or a
token another RU reads). The committed value `v` at `w` is right and stays
committed; the pardon is about the readers.

~~~
SourcePardon
  scope       RUN | POD(pod) | RU(unit)
  commitment  WEIGHTS | INPUTS | BOUNDARY
  position    the word's rank in that commitment
  value       v', the word the readers read (W_cell bits)
~~~

*What the verifier checks instead.* For every checked VU in `S`, each read of
position `w` of `K` is evaluated as `v'`; the VU's relation, its other
inputs and its outputs are checked as before. The opening of `K` at `w` is
unchanged and still opens to `v`. A checked VU outside `S` reads `v`. If `K`
is the boundary commitment, the RU that produced `w` is still checked against
the committed `v` (it computed `v`; the fault is in the consumers). The
prover's interior commitments and its evidence must be consistent with `v'`
at every reader in `S`: a reader in `S` that computed from `v` fails its
check. Two pardons of one `(K, w)` in a run are rejected; the budget is a
count per round, like `max_faults`.

*Scope, and why run-wide is the primary kind.* A misread is a property of the
memory that held the cell for the readers that read it: a pod's HBM, a pod's
KV store. Its natural extent is every reader on that pod for the fault's
duration -- the run's RUs on the pod (`POD`), or the run (`RUN`) when a run is
one pod's work, as under `RequestsG`. The `POD` scope is a refinement of `RUN`
by a public partition: `ClusterG`'s schedule places every step RU on a pod,
and the verifier can derive the partition from the advice, so a pod-scoped
pardon is a run-wide one over a public subset and costs `log2 n_pods` more.

The scope decides what the pardon can be used for after `J`. Under `RUN` (or
`POD`) the verifier substitutes `v'` into every opened reader of `w` in the
scope, including readers the prover computed correctly from `v`. A prover
whose scope mixes RUs computed from `v` and RUs computed from `v'` cannot
pardon either way: every opened reader on the other side fails. So a run-wide
pardon made after `J` cannot pick and choose among the opened RUs; the only
pardon that passes is the one the whole scope was computed under, and that was
fixed before `J` when the boundary was committed. That is the architect's
argument for pricing the run-wide pardon at its message even after `J`
(about `W_cell + log2 n_W = 52` bits at the headline); section 6 says what is
established and section 8 what is not.

Under `RU(unit)` the pardon has the selective-opening leverage of a VU
declaration: an adversary plants a different `(w, v')` in each of many RUs,
computes each RU honestly under its own modification, and pardons whichever
of them `J` opens. It must be priced like `declared_bits`, with the pardon's
message `W_cell + log2 n_W + log2 n_RU` in place of `u(1)`: the smaller of the
`d` readers it removes from the opened RU at `u_post(1)` each and the union
over the pardons it could have been (`rho` times the message). At both
operating points the first term is the smaller, so an RU-scoped source pardon
after `J` costs exactly what declaring the RU's readers under M6 costs (285
bits per opened step on the fixture, `d u_post(1) = 6.3e12 = 0.33 U_0` at the
headline). It exists in this note for completeness; nothing recommends it.

*Diagnosis.* The pardon needs `(K, w, v')`. An ECC scrub log names `w` and
often `v'`. Without a signal, under `VU_OUTPUTS` recording the first VU in
address order whose recorded output disagrees with its recompute is a reader,
`w` is among its `d_model` weight inputs, and one relation solves `v'`
(two readers confirm it); under `BOUNDARY` recording the first disagreement is
a KV entry or a token far downstream, and the prover has to re-execute one
opened RU on the faulty pod with VU-output logging to find the reader.

## 3. Port pardon

*Statement.* "RU `r` read its port `p` as `v'`." A port is one of the RU
kind's read positions: a boundary word (a KV entry or a token from another
RU), a weight cell, an input. The pardon is the consumer-side, one-RU form of
a read fault: a KV word that rotted in the consuming pod's cache after its
producer committed it right, a token one step misread, a weight cell one step
misread.

~~~
PortPardon
  unit    the RU
  port    index into the RU kind's read ports (its resolved interface)
  value   v', the word the RU read (W_port bits)
~~~

*What the verifier checks instead.* Every checked VU of RU `unit` whose gate
reads port `port` reads `value`; the port's committed value, in whichever
commitment owns it, is unchanged and still opens; every other RU reads the
committed value. The prover's interior for `unit` must be consistent with
`value` at every reader of the port.

*Scope.* One RU by construction. A port carries a different value in every
copy of a kind, so a pardon for "port `p` of every VU of kind `k`" is
meaningful only for a port that is the same word in every copy -- a broadcast
constant such as the toy's attention shift -- and is then a source-position
pardon of that constant's cell; the kind-level form is not a separate kind.
The RU-level form covers the KV/boundary-at-rest class of the fault matrix,
which the source-position pardon of the boundary commitment also covers, with
a smaller message when the RU kind has fewer ports than the run has committed
positions: a 70B step's KV ports number about `1.3e9` (30 bits) against every
KV word of the run.

*Price.* Before `J`: `W_port + log2 (n_RU n_ports)` (the ladder). After `J`
it has an RU's selective-opening leverage and the `declared_bits`-like bound
`d_p u_post(1)`, `d_p` the readers of the port in the RU (up to `heads x`
later positions for a KV word, which exceeds `U_0` at the headline); the
ladder prohibits it. Its use is before `J`, with the KV store's ECC as the
signal.

## 4. RU-scope pardon

*Statement.* "The interior of RU `r` is not claimed correct." Nothing in the
RU is checked; its outputs stay committed in the boundary and are read by
consumers as committed; the RU's claimed outputs are unverified, so the pardon
frees `W_R` bits of output and names an RU.

~~~
RUPardon
  unit    the RU whose interior is exempt
~~~

*What the verifier checks instead.* The RU is excluded from `J` (or opened and
skipped: no interior commitment is demanded, no VU of it is sampled). Its
boundary openings are checked as for any RU. Consumers in other RUs read the
committed outputs and are checked against them.

*Scope.* One RU; a set of RUs is a set of pardons. A pod-hour withdrawn this
way is 72,000 pardons on the toy shape and about `5e4` at the headline, and
the pod's outputs for the hour are then exactly as unverified as if they had
not been served under the protocol. It is the fallback when no cause can be
named.

*Price.* Before `J`: `W_R + log2 n_RU` (337 bits per step on the fixture,
`8,237` at the headline). After `J`: the ladder's bound is `rho log2 (1 + q
n_RU / (1 - q)) + W_R + log2 n_RU` per pardon, `0.47 U_0` at the headline
against a whole-RU attack worth `0.095 U_0`; prohibited.

## 5. Stage: what the honest prover can know when

Every price in section 6 is cheap before `J` and prohibited or open after it,
and the honest prover learns its faults by pinned replay of the opened RUs,
which happens after `J`. So the kinds are worth exactly what the prover can
detect before the round closes: an ECC scrub log for a weight cell at rest, a
rollout log for a stale version (for which a per-pod public weight root, M2/M8,
is the cheaper statement still: zero bits), a kernel-path log for a wrong
lowering (a per-pod gate set, M8, zero bits), the KV store's ECC for a rotted
boundary word, a pre-streaming range check for a catastrophic value. A pardon
message therefore belongs before the round closes -- after the boundary is
committed, before `J` is derived from it -- and the wire changes of section 7
put it there. A post-`J` slot would be added only if section 8's first
question is settled in favour of the message price.

## 6. Prices in `Bound`

| Kind | Before `J` (established: fixed-in-advance) | After `J` (established) | After `J` (argued, not established) | After `T` |
|---|---|---|---|---|
| VU declaration (M6, exists) | `u(1)`: 94.7 headline, 75.9 toy (the note's fixture) | `declared_bits`: `u_post(1) = 6.12e9` headline (`3.2e-4 U_0`), 142.6-145.6 toy | -- | prohibited (`0.125 U_0` for the first) |
| Source-position, `RUN` / `POD` | `log2 (1 + n_scopes n_cells 2^W_cell)`: 51.9 headline (`+ 11` for 2048 pods), 25.7 toy | `<= d u_post(1)`: `0.33 U_0` headline, 285 per opened step toy -- declaring the readers, no saving | message price `~ 52` under forced consistency (section 8, question 1) | prohibited |
| Source-position, `RU` | `W_cell + log2 (n_W n_RU)`: 96.6 headline, 42.8 toy | `min(d u_post(1), rho x message + message) = d u_post(1)`: `0.33 U_0`, 285 toy | -- | prohibited |
| Port | `W_port + log2 (n_RU n_ports)` | `<= d_p u_post(1)`; prohibited | -- | prohibited |
| RU-scope | `W_R + log2 n_RU`: 8,237 headline, 337 toy | `<= rho log2 (1 + q n_RU / (1 - q)) + W_R + log2 n_RU`: `0.47 U_0`; prohibited (attack `0.095 U_0`) | -- | prohibited |

`Bound(..., max_faults=f)` today adds `declared_bits`. With the new kinds the
fold adds, for each budget, the pre-`J` price times the budget --
`f_src log2 (1 + n_scopes n_cells 2^W_cell) + f_port (W_port + log2 (n_RU
n_ports)) + f_RU (W_R + log2 n_RU)` -- exactly as `fault_allowance_bits` would
for pre-`J` VU declarations; an RU-scope pardon also keeps the RU in the
table (the fold stays conservative) and frees its `W_R` output bits. The
interface cap applies after. In the epoch the budgets are the round's, carried
by every header and charged once in the union bound, as `max_faults` is.

The fleet-scale comparison the systematic rows make (`docs/honest-prover.md`,
section 6): the outputs of the pardoned readers are a function of `(w, v')`
given honest inputs, so what a source-position pardon carries is its message,
52 bits at the headline; M6 over the readers of one opened request carries
`1024 x 6.12e9 = 6.3e12` bits, `0.33 U_0`, and the run is rejected anyway
because 1024 exceeds `f_max = 2`. The comparison is between a pardon fixed
before `J` and declarations made after it.

## 7. Wire and verifier changes each kind needs

Common to all three (the pre-`J` stage):

1. `protocol/messages.py`: a `PardonMessage(sources, ports, units)` with the
   three record types above, its canonical manifest and digest (a new domain
   tag in `protocol/domains.py`); `Header` gains the budgets
   `max_source_pardons`, `max_port_pardons`, `max_ru_pardons` and the allowed
   scopes, bound by the header digest like `max_faults`; `Transcript` gains
   the message. `PROTOCOL_VERSION` moves from `v8` to `v9`.
2. `protocol/phases.py`: `boundary_phase(header, boundary, pardons)` so that
   the q-seed (and, in the epoch, the link and the seal) binds the pardons;
   `interior_phase` unchanged.
3. `protocol/session.py`: `VerifierSession.accept_boundary` (and the epoch's
   `receive_boundary`) accepts the pardon message: each budget, distinct
   positions, positions in range of their commitment, values of the
   commitment's width, well-formed scopes (a `POD` scope needs the placement
   of item 6), a new `VerificationCode.PARDONS_EXCEEDED`; the prover session
   gains `pardon(...)`.
4. `analysis/bound.py`, `analysis/faults.py`: the budgets enter `bound` as
   section 6 says; `veritor.evaluation` and the stress report thread them.
5. `protocol/epoch.py`: the round's budgets in every header, enforced across
   the round's runs as `receive_interiors` enforces `max_faults`; the link
   covers the pardon message digest.
6. Constructors: a public `placement(unit) -> pod` for `POD` scopes, derived
   from the description or the advice (`ClusterG`: the schedule; `RequestsG`:
   the run is the pod).

Source-position pardon:

7. `protocol/proofs/statement.py`: `PositionRef` gains `substitute: bytes |
   None`; a reader's input position of a pardoned `(K, w)` in scope carries
   `v'` and no longer demands an opening of `K` at `w` for that read (the
   opening of `K` at `w` is still demanded wherever `w` is an output position,
   and for readers outside the scope).
8. `protocol/proofs/derive.py`: `derive_obligations(..., pardons)` applies the
   substitution to every sampled VU in scope; `KindProgram` is unchanged (the
   relation is the same, one input value differs).
9. `protocol/proofs/transparent.py` and the zkVM backend: the transparent
   check reads `substitute` instead of folding a path for that position; the
   statement of a proof takes `v'` as a public input for that slot.
10. Prover side (`replay_unit`, the honest model's `replay_pinned`): the
    reconstruction of an opened RU in scope reads `v'` at `w`, so the
    committed interior is consistent with the pardon.

Port pardon:

11. Items 7-10 with the substitution keyed by `(unit, port)`: the RU kind's
    resolved interface (`Index`/`derive`) maps the port to its committed
    position; a port that resolves to a source or boundary position is
    substituted in every VU of the RU reading it.

RU-scope pardon:

12. `protocol/challenge.py` / `ReplayChallenge`: the pardoned RUs are removed
    from the population `J` is drawn from (or drawn and skipped: no interior
    commitment demanded, no VU sampled); `InteriorMessage.commitments` omits
    them; `derive_obligations` never sees their VUs.
13. `analysis/bound.py`: the RU stays in the table; its `W_R` is added as free
    output bits under the cap.

## 8. Open pricing questions

1. *The run-wide source-position pardon after `J`.* The architect's review
   prices it at its message (about 52 bits at the headline) because the
   post-`J` choice is forced by the pre-`J` commitment. The ladder in
   `late-advice.md` establishes only `d u_post(1)` (declaring the opened RU's
   readers; `0.33 U_0`) and prohibits the kind for want of anything cheaper.
   This note's argument for the message price: for any pardon `(w, v')`
   other than the one the scope was computed under, every opened non-silent
   reader of `w` in the scope fails, so the pardon survives with probability
   at most `(1 - s)^(readers of w among the opened RUs)`; if `q s x (readers
   of w in the scope) >> ln |M|` the union over all `|M|` wrong pardons is
   negligible against `eta` and the maximum over pardons collapses to the
   fixed one, giving `log2 |M| + o(1)`. Two gaps. (i) A *silent* reader --
   one whose output does not depend on the cell at its committed inputs (a
   zero activation; on the toy, an even activation under a top-bit flip,
   which made 6 of 9 readers silent) -- accepts several `v'`, so the condition
   must count non-silent opened readers, or the price must add the union over
   the values every opened reader accepts. (ii) The condition holds at the
   headline for a run of the fleet's scale (`q s x 3e16` positions a year
   gives `4e6` opened readers of every layer cell; a fleet-hour gives about
   470) and fails for a pod scope or a pod-hour run (`5e7` positions give
   `7e-3`): there the pardon degenerates to the RU-scoped kind, and the
   plant-and-pardon attack across scopes carries about `(mu_f / q_scope) x
   log2 |M|` bits per round -- about `2e6` bits at the headline for pod-hour
   scopes (`q_scope = 8e-4`, `f = 1`, `mu_1 = 31.2`), `1.0e11` for RU scopes
   -- between the message price and the established bound. Until the note proves a bound
   below `d u_post(1)` in the dense regime and states the regime, this note
   uses the message price before `J` only.
2. *Where the pardon budget lives.* If the budget were per run rather than per
   round, a fleet of small runs could plant a different `(w, v')` in every
   RU and pardon, in every run, whichever RU was opened: capacity `n_RU log2
   |M|` with no selection to pay for. The budget must be the round's, as
   `max_faults` is; the ladder does not say so explicitly.
3. *Where two arguments give two numbers on the toy.* The fixture's fold is
   saturated and capped at 256 bits, so every post-`J` charge on the toy
   (`u_post(1) = 142.6`, the RU-scoped 285 per step, the `H4` M6 charges) is
   what the fold would add before the cap; `late-advice.md`'s simulation point
   quotes 145.6 on its own fixture (`rho = 755.9`, `W_R = 224`). The numbers
   agree on the argument and differ on the fixture.
4. *The kind-level port pardon.* The brief's "a specific input port of a gate
   kind reads a wrong value" is priced by the ladder per RU (`W_port + log2
   (n_RU n_ports)`). Section 3 argues the kind-level form is a source-position
   pardon of a broadcast cell and only the RU-level form is a distinct kind;
   if the ladder intends the kind-level form, its message must name the kind
   and the port among the kind's ports and its scope must be stated.
