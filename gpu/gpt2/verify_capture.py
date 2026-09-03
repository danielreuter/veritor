"""The GPU capture through the framework: compile GPT-2 Small, address the capture, sample VUs, re-execute, run the protocol.

    python gpu/gpt2/verify_capture.py sample   --vus 4000      # (ii) sampled-VU re-execution with the framework's own challenge derivation
    python gpu/gpt2/verify_capture.py protocol --layers 12     # (iii) run_protocol on a slice: honest accept + one flipped bit rejected

Needs ``gpu/gpt2/results/{hf_gpt2_fp32.npz,capture_gpu.npz,hf_meta.json}``
(the pod run) and, for ``protocol``, ``libref_chain.so`` built from
``gpu/tensor-core-semantics/ref_chain.cpp`` for this host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from typing import cast

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_ops
from run_gpt2 import load_weights, meta
from weight_tree import WeightTree

from veritor.constructors import GPT2G, GPT2Shape, Request
from veritor.constructors.gpt2_reference import (
    GPT2Weights,
    NumpyOps,
    SparseValues,
    address_map,
    check_unit,
    evaluate_unit,
    forward,
    request_frames,
)
from veritor.core import VerificationLimits, VerificationPolicy
from veritor.protocol import (
    MerkleTree,
    VerificationCode,
    VerifierParameters,
    derive_sample_selection,
    encode_transcript,
    make_expectation,
    replay_unit,
    run_protocol,
)
from veritor.research import Compile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LIMITS = VerificationLimits(
    max_units=1 << 40,
    max_positions=1 << 40,
    max_positions_per_unit=1 << 40,
    max_openings=1 << 40,
    max_artifact_bytes=1 << 40,
    max_proof_bytes=1 << 40,
    max_transcript_bytes=1 << 40,
)


def load_capture(path: str) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    z = np.load(path)
    capture = {k: z[k] for k in z.files if not k.startswith("_")}
    return capture, [int(t) for t in z["_prompt"]], [int(t) for t in z["_tokens_out"]]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def owner_vu(compiled, address: int) -> int:
    index = compiled.index
    r = index.replay_units.owner(address)
    block = index.verification_units(r)
    return block.first + block.owner(address)


def kind_name(g: GPT2G, node) -> str:
    return g.model.kind_names().get(node.kind, node.kind[:12])


def cmd_sample(args) -> None:
    weights = load_weights(RESULTS)
    capture, prompt, tokens = load_capture(os.path.join(RESULTS, "capture_gpu.npz"))
    max_new = len(tokens)
    g = GPT2G(GPT2Shape.small())
    t0 = time.perf_counter()
    compilation = Compile(g, (Request(tuple(prompt), max_new),), b"", g.gate_set)
    compiled = compilation.compiled
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    amap = address_map(
        compiled, g.model, request_frames(compiled)[0], len(prompt), max_new
    )
    values = SparseValues(compiled, capture, amap, weights.flat(), compilation.inputs)
    map_s = time.perf_counter() - t0
    n_vus = compiled.index.verification_unit_count
    index = compiled.index
    request_unit = 1  # replay unit 0 is the weights: 124.5 M one-gate source VUs
    request_vus = index.verification_units(request_unit).count
    # the framework's derivation: the request RU selected, its VUs at rate s, from a verifier seed over a phase digest
    s = Fraction(args.vus, request_vus).limit_denominator(1 << 60)
    seed = hashlib.sha256(b"veritor/gpt2-silicon/sample-seed").digest()
    digest = hashlib.sha256(b"veritor/gpt2-silicon/interior-phase").digest()
    t0 = time.perf_counter()
    sampled = derive_sample_selection(
        seed, digest, compiled, (request_unit,), VerificationPolicy(1, s), LIMITS
    )
    sample_s = time.perf_counter() - t0

    def owner_vu(address: int) -> int:
        r = index.replay_units.owner(address)
        block = index.verification_units(r)
        return block.first + block.owner(address)

    # every VU of the rare kinds, located through the addresses of their recorded outputs
    rare = ["lnf.mean", "lnf.rstd", "argmax.best", "argmax.token"]
    for layer in range(12):
        rare += [
            f"L{layer}.ln1.mean",
            f"L{layer}.ln1.rstd",
            f"L{layer}.ln2.mean",
            f"L{layer}.ln2.rstd",
            f"L{layer}.m",
            f"L{layer}.S",
        ]
    targeted = set()
    for name in rare:
        addresses = amap[name]
        for a in addresses[addresses >= 0].reshape(-1).tolist():
            targeted.add(owner_vu(int(a)))
    targeted -= set(sampled)

    per_kind: dict[str, Counter] = defaultdict(Counter)
    t0 = time.perf_counter()
    bad = []
    for label, units in (("uniform", sampled), ("targeted", sorted(targeted))):
        for u in units:
            node = compiled.index.verification_unit(u)
            name = kind_name(g, node)
            checked, agreeing = check_unit(compiled, node, values)
            per_kind[name]["units"] += 1
            per_kind[name][label] += 1
            per_kind[name]["gates_checked"] += checked
            per_kind[name]["gates_agreeing"] += agreeing
            per_kind[name]["gates"] += node.size
            if checked != agreeing:
                bad.append((u, name, checked, agreeing))
    check_s = time.perf_counter() - t0
    # Circuit.evaluate_gate over a VU's definition against the GPU's words, one VU of every kind
    evaluated = {}
    seen = set()
    for u in list(sampled) + sorted(targeted):
        node = compiled.index.verification_unit(u)
        name = kind_name(g, node)
        if name in seen:
            continue
        seen.add(name)
        known = evaluate_unit(compiled, node, values)
        outputs = [a for a in known if a in values]
        evaluated[name] = {
            "gates": len(known),
            "recorded_outputs": len(outputs),
            "equal": all(known[a] == values[a] for a in outputs),
        }
    report = {
        "prompt_tokens": len(prompt),
        "max_new": max_new,
        "gates": compiled.circuit.n,
        "replay_units": compiled.index.replay_units.count,
        "verification_units": n_vus,
        "recorded_words": values.recorded_count,
        "sampling_rate_s": str(s),
        "request_vus": request_vus,
        "sampled_vus": len(sampled),
        "targeted_vus": len(targeted),
        "gates_checked": sum(c["gates_checked"] for c in per_kind.values()),
        "gates_agreeing": sum(c["gates_agreeing"] for c in per_kind.values()),
        "failures": bad[:20],
        "per_kind": {k: dict(v) for k, v in sorted(per_kind.items())},
        "evaluate_unit_per_kind": evaluated,
        "seconds": {
            "compile": compile_s,
            "address_map": map_s,
            "sample": sample_s,
            "check": check_s,
        },
    }
    with open(os.path.join(RESULTS, "sampled_vus.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("per_kind", "evaluate_unit_per_kind")
            },
            indent=2,
            default=str,
        )
    )
    for k, v in sorted(per_kind.items()):
        print(
            f"{k:28s} {v['units']:6d} VUs ({v.get('uniform', 0)} sampled, {v.get('targeted', 0)} targeted) {v['gates']:9d} gates {v['gates_checked']:8d} checked {v['gates_agreeing']:8d} agreeing"
        )
    print("evaluate_unit:", json.dumps(evaluated))


def cmd_slice(args) -> None:
    """A small committed slice of the GPU capture: a few VUs of every kind with their input and output words by address.

    ``tests/veritor/constructors/test_gpt2_capture.py`` re-executes them
    without the 28 MB capture or the 497 MB checkpoint.  Kinds are found
    with the same uniform sample as ``sample`` plus the rare kinds; the
    embedding dot (``dot(50272,...)``, 100 K input words) is left out to
    keep the file small -- ``sampled_vus.json`` covers it.
    """

    weights = load_weights(RESULTS)
    capture, prompt, tokens = load_capture(os.path.join(RESULTS, "capture_gpu.npz"))
    max_new = len(tokens)
    g = GPT2G(GPT2Shape.small())
    compilation = Compile(g, (Request(tuple(prompt), max_new),), b"", g.gate_set)
    compiled = compilation.compiled
    amap = address_map(
        compiled, g.model, request_frames(compiled)[0], len(prompt), max_new
    )
    values = SparseValues(compiled, capture, amap, weights.flat(), compilation.inputs)
    index = compiled.index
    request_vus = index.verification_units(1).count
    s = Fraction(args.vus, request_vus).limit_denominator(1 << 60)
    seed = hashlib.sha256(b"veritor/gpt2-silicon/sample-seed").digest()
    digest = hashlib.sha256(b"veritor/gpt2-silicon/interior-phase").digest()
    candidates = list(
        derive_sample_selection(
            seed, digest, compiled, (1,), VerificationPolicy(1, s), LIMITS
        )
    )
    rare = ["lnf.mean", "lnf.rstd", "argmax.best", "argmax.token"]
    for layer in range(12):
        rare += [
            f"L{layer}.ln1.mean",
            f"L{layer}.ln1.rstd",
            f"L{layer}.ln2.mean",
            f"L{layer}.ln2.rstd",
            f"L{layer}.m",
            f"L{layer}.S",
        ]
    for name in rare:
        addresses = amap[name]
        candidates += [
            owner_vu(compiled, int(a))
            for a in addresses[addresses >= 0].reshape(-1).tolist()
        ]
    per_kind: Counter = Counter()
    chosen: list[dict] = []
    seen: set[int] = set()
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        node = index.verification_unit(u)
        name = kind_name(g, node)
        if name.startswith("dot(50272"):
            continue
        cap = 1 if name.startswith("dot(3072") else args.per_kind
        if per_kind[name] >= cap:
            continue
        circuit = compiled.circuit
        inputs: dict[int, int] = {}
        outputs: dict[int, int] = {}
        for address in node.interval:
            ref = circuit[address]
            if ref.is_source:
                continue
            for a in ref.args:
                if a not in node.interval:
                    inputs[a] = values[a]
            if address in values:
                outputs[address] = values[address]
        if not outputs:
            continue
        per_kind[name] += 1
        chosen.append({"unit": u, "kind": name, "inputs": inputs, "outputs": outputs})
    out = {
        "source": "gpu/gpt2/results/capture_gpu.npz (RTX 4090, driver 550.127.05, CUDA 12.4); words by circuit address",
        "shape": GPT2Shape.small().manifest,
        "prompt": prompt,
        "max_new": max_new,
        "tokens": tokens,
        "gates": compiled.circuit.n,
        "verification_units": index.verification_unit_count,
        "description_digest": compiled.digest,
        "vus": chosen,
    }
    path = args.out
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(
        f"{len(chosen)} VUs of {len(per_kind)} kinds, {sum(len(v['inputs']) + len(v['outputs']) for v in chosen):,} words, {os.path.getsize(path):,} bytes -> {path}"
    )
    for name, count in sorted(per_kind.items()):
        print(f"  {name:28s} {count}")


def slice_weights(weights: GPT2Weights, layers: int) -> GPT2Weights:
    shape = GPT2Shape(
        layers=layers, d_model=768, heads=12, d_ff=3072, vocab=50257, context=1024
    )
    return GPT2Weights(
        shape,
        weights.wte,
        weights.wpe,
        weights.layers[:layers],
        weights.lnf_g,
        weights.lnf_b,
    )


class Timer:
    def __init__(self) -> None:
        self.total = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.total += time.perf_counter() - self._t


def cmd_protocol(args) -> None:
    """``run_protocol`` on GPT-2 Small with ``--layers`` layers (all 12 by default), prompt of ``--prompt`` tokens, ``--new`` generated."""

    from veritor.protocol.session import ProverSession, VerifierSession

    full = load_weights(RESULTS)
    weights = slice_weights(full, args.layers) if args.layers < 12 else full
    m = meta(RESULTS)
    prompt = tuple(m["prompt"][: args.prompt])
    # the client's run: the CPU reference forward (bit-identical to the GPU capture on the full prompt)
    ops = NumpyOps(
        cuda_ops.CppGemm(
            os.path.join(
                os.path.dirname(HERE), "tensor-core-semantics", "libref_chain.so"
            )
        )
    )
    t0 = time.perf_counter()
    run = forward(weights, prompt, args.new, ops)
    forward_s = time.perf_counter() - t0
    g = GPT2G(weights.shape)
    t0 = time.perf_counter()
    compilation = Compile(g, (Request(prompt, args.new),), b"", g.gate_set)
    compiled = compilation.compiled
    compile_s = time.perf_counter() - t0
    amap = address_map(
        compiled, g.model, request_frames(compiled)[0], len(prompt), args.new
    )
    values = SparseValues(
        compiled, run.capture, amap, weights.flat(), compilation.inputs
    )
    t0 = time.perf_counter()
    tree = WeightTree(
        g.gate_set, weights.flat()
    )  # root-identical to commit_weights, numpy-backed, hashed in parallel
    kappa = tree.weights
    commit_s = time.perf_counter() - t0
    policy = VerificationPolicy(1, Fraction(args.s_num, args.s_den))
    parameters = VerifierParameters(Fraction(1, 2**40), max_capacity=None)

    def expectation_for(label: bytes):
        tag = b"veritor/gpt2-silicon/" + label
        return make_expectation(
            compilation,
            policy,
            run.tokens,
            parameters=parameters,
            weights=kappa,
            session_id=hashlib.sha256(tag + b"/session").digest()[:16],
            q_seed=hashlib.sha256(tag + b"/q").digest(),
            s_seed=hashlib.sha256(tag + b"/s").digest(),
        )

    replayed: dict[int, dict] = {}

    def honest_replay(unit: int, boundary_values):
        if unit not in replayed:
            log(
                f"replaying RU {unit} ({compiled.index.replay_units.unit(unit).size:,} gates)"
            )
            t = time.perf_counter()
            replayed[unit] = replay_unit(compiled, unit, boundary_values)
            log(f"replayed RU {unit} in {time.perf_counter() - t:.1f} s")
        return replayed[unit]

    log(
        f"compiled: {compiled.circuit.n:,} gates, {compiled.index.verification_unit_count:,} VUs; weights committed in {commit_s:.1f} s"
    )
    expectation = expectation_for(b"honest")
    prover_t, verifier_t = Timer(), Timer()
    with verifier_t:
        verifier = VerifierSession(expectation, compiled, limits=LIMITS)
    prover = ProverSession(
        compiled,
        verifier.header,
        values,
        limits=LIMITS,
        weight_tree=cast(MerkleTree, tree),
        replay=honest_replay,
    )
    with prover_t:
        boundary = prover.boundary()
    with verifier_t:
        replay_challenge = verifier.receive_boundary(boundary)
    t0 = time.perf_counter()
    honest_replay(
        1, values
    )  # the request RU: time the honest replay itself (the prover then reads it from the cache)
    replay_s = time.perf_counter() - t0
    log("committing the interior")
    with prover_t:
        interiors = prover.interiors(replay_challenge)
    log(f"interior committed ({prover_t.total:.1f} s prover so far)")
    with verifier_t:
        sample_challenge = verifier.receive_interiors(interiors)
    with prover_t:
        evidence = prover.evidence(sample_challenge)
    log(f"evidence built ({sum(len(batch) for batch in evidence.units):,} openings)")
    with verifier_t:
        report = verifier.receive_evidence(evidence)
    transcript = verifier.transcript
    transcript_bytes = len(encode_transcript(transcript))
    log(
        f"honest verdict {report.code.value}; transcript {transcript_bytes:,} B; prover {prover_t.total:.1f} s, verifier {verifier_t.total:.1f} s"
    )
    honest = {
        "code": report.code.value,
        "detail": report.detail,
        "sampled_replay_units": list(report.sampled_replay_units),
        "sampled_vus": len(report.sampled_verification_units),
        "openings": sum(len(batch) for batch in evidence.units),
        "interior_positions": [c.count for c in interiors.commitments],
        "boundary_positions": boundary.commitment.count,
        "prover_seconds": prover_t.total,
        "verifier_seconds": verifier_t.total,
        "transcript_bytes": transcript_bytes,
    }
    # the rejection test: the prover's replay flips one bit of one dot output (a mix dot, read by every o-proj dot
    # of its position).  Only the flipped VU and its ``d_model`` readers are inconsistent, so a verifier sampling at
    # rate ``s`` catches it with probability ``1 - (1 - s)^(d_model + 1)``; each session is a fresh verifier seed.
    layer = args.layers - 1
    target = int(amap[f"L{layer}.mix"][len(prompt) - 1, 7])
    readers = (
        1 + weights.shape.d_model
    )  # the flipped VU itself and the o-proj dots that read the word
    p_detect = 1 - float((1 - policy.s) ** readers)

    def flipped(unit: int, boundary_values):
        known = dict(
            honest_replay(unit, boundary_values)
        )  # the honest replay, cached, with one bit flipped
        if target in known:
            known[target] ^= 1
        return known

    path = os.path.join(RESULTS, f"protocol_L{args.layers}_p{len(prompt)}_n{args.new}.json")
    sessions: list[dict] = []
    out = {
        "layers": args.layers,
        "prompt_tokens": len(prompt),
        "max_new": args.new,
        "tokens": list(run.tokens),
        "gates": compiled.circuit.n,
        "weights": compiled.index.weight_count,
        "replay_units": compiled.index.replay_units.count,
        "verification_units": compiled.index.verification_unit_count,
        "policy": {"q": "1", "s": str(policy.s)},
        "seconds": {
            "forward_cpu": forward_s,
            "compile": compile_s,
            "commit_weights": commit_s,
            "replay": replay_s,
        },
        "honest": honest,
        "rejection": {
            "flipped_address": target,
            "flipped_gate": compiled.circuit[target].op,
            "inconsistent_vus": readers,
            "detection_probability_per_session": p_detect,
            "sessions": sessions,
        },
    }

    def save() -> None:  # after the honest run and after every rejection attempt: nothing is lost to a long run
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    save()
    for attempt in range(args.reject_attempts):
        log(
            f"rejection attempt {attempt}: flipping bit 0 of address {target} ({compiled.circuit[target].op})"
        )
        t0 = time.perf_counter()
        bad = run_protocol(
            compiled,
            expectation_for(b"flipped-%d" % attempt),
            values,
            weight_tree=cast(MerkleTree, tree),
            replay=flipped,
            limits=LIMITS,
        )
        sessions.append(
            {
                "code": bad.report.code.value,
                "detail": bad.report.detail,
                "sampled_vus": len(bad.report.sampled_verification_units),
                "seconds": time.perf_counter() - t0,
            }
        )
        log(
            f"rejection attempt {attempt}: {bad.report.code.value} in {sessions[-1]['seconds']:.1f} s"
        )
        save()
        if bad.report.code is not VerificationCode.ACCEPTED:
            break
    print(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sample", "protocol", "slice"])
    parser.add_argument("--per-kind", type=int, default=3)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(HERE)),
            "tests",
            "veritor",
            "constructors",
            "golden",
            "gpt2_small_capture_slice.json",
        ),
    )
    parser.add_argument("--vus", type=int, default=4000)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--prompt", type=int, default=1)
    parser.add_argument("--new", type=int, default=1)
    parser.add_argument("--s-num", type=int, default=1)
    parser.add_argument("--s-den", type=int, default=1000)
    parser.add_argument("--reject-attempts", type=int, default=3)
    args = parser.parse_args()
    {"sample": cmd_sample, "protocol": cmd_protocol, "slice": cmd_slice}[args.command](
        args
    )


if __name__ == "__main__":
    main()
