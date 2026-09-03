"""The narrative report: text only, every number from the :class:`Summary`."""

from __future__ import annotations

from .summary import Summary


def render(summary: Summary) -> str:
    w, m, c, p, b, k, h, a = (
        summary.workload,
        summary.model,
        summary.compile,
        summary.policy,
        summary.bound,
        summary.cost,
        summary.honest,
        summary.adversary,
    )
    lines: list[str] = []
    out = lines.append
    out(f"Veritor datacenter demo -- scale {summary.scale}, seed {summary.seed}")
    out("=" * 78)

    out("")
    out("1. Workload (timing)")
    out(
        f"  cluster: {w.pods} pods x {w.slots} slots, {w.steps} synchronous steps of "
        f"{w.step_seconds * 1000:.0f} ms ({w.steps * w.step_seconds:.2f} s of wall clock)"
    )
    out(
        f"  model: toy decoder, vocab {m.vocab} ({m.vocab_bits}-bit tokens), d_model {m.d_model}, "
        f"{m.heads} heads, {m.layers} layer(s), context {m.context}, {m.width}-bit words, "
        f"{m.weights} weights, sampling {'on' if m.sampling else 'off'}"
    )
    out(
        f"  arrivals: {w.arrivals} Poisson arrivals at {w.load:.1f}x the cluster's throughput; "
        f"{w.admitted} admitted, {w.unserved} still queued when the window closed"
    )
    out("     #   t(s)  prompt  max_new  request")
    for arrival in w.arrival_records:
        request = "-" if arrival.request is None else str(arrival.request)
        out(
            f"    {arrival.index:2d}  {arrival.time:5.2f}  {arrival.prompt_length:6d}  "
            f"{arrival.max_new:7d}  {request:>7}"
        )
    out(
        f"  schedule (the advice a): {w.joins} joins of {w.admitted} requests, "
        f"{w.restarts} restart(s) after {len(w.failures)} pod failure(s); "
        f"{w.token_steps} occupant-steps, utilization {w.utilization:.0%}"
    )
    out("    pod  step  slot  request  length  outcome    streamed positions")
    for t in w.attempts:
        positions = ",".join(map(str, t.streamed)) or "-"
        out(
            f"    {t.pod:3d}  {t.step:4d}  {t.slot:4d}  {t.request:7d}  {t.length:6d}  "
            f"{t.outcome:<9}  {positions}"
        )
    out("  occupancy (rows: pods; columns: steps; digit = occupants, x = down):")
    for pod, row in enumerate(w.occupancy):
        out(f"    pod {pod}: " + "".join("x" if n < 0 else str(n) if n else "." for n in row))
    out(
        f"  advice: Schedule.encode() = {w.advice_bytes} bytes = {w.advice_bits} bits "
        "(magic, 4 header words, 5 words per join); timing enters the circuit only here"
    )

    out("")
    out("2. Data-dependent control flow")
    out(
        f"  EOS token {w.eos_token}: {w.eos_stops} attempt(s) stopped early, {w.completed} reached "
        f"max_new, {w.cut_by_run_end} still running when the window closed"
    )
    out(
        f"  every response is a prefix of reference_generate: "
        f"{'yes' if w.matches_reference else 'NO'} ({w.tokens} tokens over {w.admitted} requests)"
    )

    out("")
    out("3. Hardware failures")
    if w.failures:
        for f in w.failures:
            aborted = ", ".join(map(str, f.aborted)) or "no occupants"
            out(f"  pod {f.pod} failed at step {f.step}: aborted request(s) {aborted}")
        out(
            f"  {w.restarts} restart(s): each aborted request rejoined from the prefill; its earlier "
            "tokens stand as outputs of the aborted join and are recomputed, not re-emitted"
        )
    else:
        out("  no failures in this run")

    out("")
    out("4. Nondeterminism")
    if m.sampling:
        out(
            f"  tokens are sampled by the `sample` VU: w_j = s_j^2, prefix sums, first j with "
            f"cdf_j > (r * total) >> {m.width}; r is a {m.random_bits}-bit public random word per "
            "generated position, an `in` gate of the circuit"
        )
    else:
        out("  sampling off: tokens are the argmax; the scheduler's choices are the nondeterminism")

    out("")
    out("5. Compile")
    out(
        f"  n = {c.n} gates, {c.kinds} kinds, {c.replay_units} RUs, {c.verification_units} VUs, "
        f"{c.weight_gates} weight gates, {c.input_gates} input gates, {c.outputs} outputs "
        f"({c.out_bits} bits)"
    )
    out(
        f"  description {c.description_bytes} bytes, advice {c.advice_bytes} bytes, "
        f"compile {c.compile_seconds * 1000:.0f} ms, honest evaluation {c.evaluate_seconds * 1000:.0f} ms"
    )
    out(
        f"  per occupant-step: {c.gates_per_token_step:.0f} gates, {c.vus_per_token_step:.1f} VUs; "
        f"head VU ({c.head_kind}): {c.head_vu_gates} gates, kappa = {c.head_vu_cut_bits} bits"
    )
    out(
        f"  interfaces: boundary {c.boundary_positions} positions, interior {c.interior_positions}; "
        f"W_R = {c.W_R:.0f} bits per RU, W_V = {c.W_V:.1f} bits per VU, "
        f"{c.positions_per_vu:.1f} positions opened per sampled VU"
    )

    out("")
    out("6. Honest protocol run")
    if p.optimize_q is not None:
        out(
            f"  Optimize (cheapest theta with Bound <= U_max = {c.out_bits} bits at eta = {b.eta}, "
            f"W <= W_max, {p.grid_evaluated} of {p.grid_points} grid points): "
            f"theta = ({p.optimize_q}, {p.optimize_s}), U = {p.optimize_bits:.0f} bits, "
            f"cost {p.optimize_cost:.0f}"
        )
    out(f"  policy theta = (q, s) = ({p.q}, {p.s}): {p.rule}")
    out(
        f"  verifier: eta = {b.eta}, U_max = {h.max_capacity} bits, A = {w.advice_bits} bits, "
        f"W_max = {p.work_budget}; expected work W = {p.expected_work:.0f}"
    )
    out(f"  kappa_W (weights, once per model) = {h.weight_root[:16]}...")
    out(f"  boundary root ({h.boundary_positions} positions) = {h.boundary_root[:16]}...")
    out(
        f"  q-challenge: seed {h.q_seed[:16]}... over the boundary phase -> {h.replay_units_opened} "
        f"of {c.replay_units} RUs replayed; interior roots "
        + ", ".join(root[:8] for root in h.interior_roots[:4])
        + (", ..." if len(h.interior_roots) > 4 else "")
    )
    out(
        f"  s-challenge: seed {h.s_seed[:16]}... over the interior phase -> "
        f"{h.verification_units_opened} VUs sampled, {h.openings} openings"
    )
    out(f"  verdict: {h.code}{' -- ' + h.detail if h.detail else ''}")
    out(
        f"  prover {h.prover_seconds * 1000:.0f} ms, verifier {h.verifier_seconds * 1000:.0f} ms; "
        f"transcript {h.transcript_bytes} bytes ("
        + ", ".join(f"{name} {size}" for name, size in h.message_bytes.items())
        + ")"
    )

    out("")
    out("7. Adversary and tightness")
    out(f"  channel: {a.channel}")
    out(
        f"  {a.bits_per_vu} bits realized per corrupted VU; Bound charges kappa = {a.kappa_per_vu} "
        "bits (the width of the word the VU decides)"
    )
    out(
        f"  survival sigma(E) = prod_r (1 - q + q (1 - s)^l_r); observed over {a.rows[0].trials} "
        "fresh challenge derivations per row; full protocol runs in the last column"
    )
    out("    bits  VUs  RUs  l_r        predicted  observed  |dev|/sigma  full protocol")
    for r in a.rows:
        l_r = ",".join(map(str, r.errors_per_replay_unit))
        if len(l_r) > 9:
            l_r = l_r[:7] + ".."
        out(
            f"    {r.bits:4d}  {r.vus_corrupted:3d}  {r.replay_units_touched:3d}  {l_r:<9}  "
            f"{r.predicted_survival:9.4f}  {r.observed_survival:8.4f}  {r.deviation_sigmas:11.1f}  "
            f"{r.protocol_accepted}/{r.protocol_trials} accepted"
        )
    last = a.rows[-1]
    out(
        f"  secret of the last row ({last.bits} bits): {last.secret[:48]}"
        f"{'...' if last.bits > 48 else ''}"
    )
    out(
        f"  every secret decodes from the streamed tokens: {'yes' if a.decoded else 'NO'}; "
        f"non-carrier tokens unchanged: {'yes' if all(r.honest_tokens_unchanged for r in a.rows) else 'NO'}"
    )
    out(
        f"  every row within {a.tolerance_sigmas:.0f} sigma of the prediction: "
        f"{'yes' if a.within_tolerance else 'NO'}; every rejection was RELATION_REJECTED at a "
        "corrupted VU"
    )

    out("")
    out("8. Bound and cost")
    capped = (
        f"capped by the output interface of {b.out_bits} bits"
        if b.capped
        else "below the interface"
    )
    out(
        f"  Bound(C, I, theta) at eta = {b.eta}: U = {b.bits:.1f} bits ({capped}; "
        f"knapsack {b.knapsack_bits:.1f}, Laplace {b.laplace_bits:.1f})"
    )
    out(
        f"  Lambda = ln(1/eta) = {b.budget_nats:.2f} nats; c(1) = -ln(1 - qs) = {b.unit_cost_nats:.4f} "
        f"nats per corrupted VU; -ln(1 - q) = {b.saturation_cost_nats:.3f} nats per whole RU"
    )
    out(
        f"  the attack above reaches survival eta after {b.vus_to_eta} corrupted VUs in distinct RUs: "
        f"{b.bits_realized_to_eta} bits realized, {b.bits_charged_to_eta} bits charged"
    )
    out(
        f"  Cost (h = 1 per position): boundary {k.boundary:.0f}, recompute {k.recompute:.0f}, "
        f"interior commit {k.commit_interior:.0f}, proofs {k.proof:.0f}; total {k.total:.0f} "
        f"(n = {c.n}); kappa_W {k.weights_per_epoch:.0f} once per epoch"
    )
    out(
        f"  verifier expected work W = {k.verifier_expected_work:.0f} operations (W_max {p.work_budget})"
    )

    out("")
    out("Notes")
    for note in summary.notes:
        out(f"  - {note}")
    out("")
    out(f"total {summary.total_seconds:.1f} s")
    return "\n".join(lines)


__all__ = ["render"]
