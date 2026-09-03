"""Dump HF ``gpt2`` (fp32) to an npz, tokenize the prompt, record HF's own greedy tokens and cuBLAS times.

Runs in the pod's system Python (torch + transformers); everything else in
``gpu/gpt2/`` runs in a plain numpy interpreter.  Nothing measured here enters
the circuit: the HF tokens are a sanity comparison and the cuBLAS times the
denominator of the kernel-cost table.

    python dump_hf.py --prompt "..." --new 8 --out /workspace/gpt2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)


def cublas_ms(m: int, k: int, n: int, iters: int = 50) -> float:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and",
    )
    parser.add_argument("--new", type=int, default=8)
    parser.add_argument("--out", default="/workspace/gpt2")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    prompt = [int(t) for t in ids[0]]
    with torch.no_grad():
        generated = model.generate(
            ids,
            max_new_tokens=args.new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    hf_tokens = [int(t) for t in generated[0][len(prompt) :]]

    state = model.state_dict()
    arrays: dict[str, np.ndarray] = {
        "wte": state["transformer.wte.weight"].numpy(),
        "wpe": state["transformer.wpe.weight"].numpy(),
        "lnf_g": state["transformer.ln_f.weight"].numpy(),
        "lnf_b": state["transformer.ln_f.bias"].numpy(),
    }
    layers = model.config.n_layer
    for i in range(layers):
        p = f"transformer.h.{i}."
        w_attn = state[
            p + "attn.c_attn.weight"
        ].numpy()  # [768, 2304] = [in, (q | k | v)]
        b_attn = state[p + "attn.c_attn.bias"].numpy()
        d = w_attn.shape[0]
        for j, name in enumerate(("q", "k", "v")):
            arrays[f"layer{i}.w_{name}"] = w_attn[:, j * d : (j + 1) * d]
            arrays[f"layer{i}.b_{name}"] = b_attn[j * d : (j + 1) * d]
        arrays[f"layer{i}.w_o"] = state[p + "attn.c_proj.weight"].numpy()
        arrays[f"layer{i}.b_o"] = state[p + "attn.c_proj.bias"].numpy()
        arrays[f"layer{i}.ln1_g"] = state[p + "ln_1.weight"].numpy()
        arrays[f"layer{i}.ln1_b"] = state[p + "ln_1.bias"].numpy()
        arrays[f"layer{i}.ln2_g"] = state[p + "ln_2.weight"].numpy()
        arrays[f"layer{i}.ln2_b"] = state[p + "ln_2.bias"].numpy()
        arrays[f"layer{i}.w_fc"] = state[p + "mlp.c_fc.weight"].numpy()
        arrays[f"layer{i}.b_fc"] = state[p + "mlp.c_fc.bias"].numpy()
        arrays[f"layer{i}.w_proj"] = state[p + "mlp.c_proj.weight"].numpy()
        arrays[f"layer{i}.b_proj"] = state[p + "mlp.c_proj.bias"].numpy()
    np.savez(os.path.join(args.out, "hf_gpt2_fp32.npz"), **arrays)  # type: ignore[arg-type]

    # cuBLAS (torch.matmul, BF16 in, BF16 out) at the GEMM shapes of this run
    m_prefill = len(prompt)
    shapes = []
    for m in sorted({m_prefill, 1}):
        shapes += [(m, 768, 768), (m, 768, 3072), (m, 3072, 768), (m, 50272, 768)]
    shapes.append((1, 768, 50257))
    cublas = {f"{m}x{k}x{n}": cublas_ms(m, k, n) for m, k, n in shapes}
    with torch.no_grad():
        model_gpu = model.to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model_gpu.generate(
            ids.to("cuda"),
            max_new_tokens=args.new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        hf_fp32_generate_s = time.perf_counter() - t0
    meta = {
        "prompt_text": args.prompt,
        "prompt": prompt,
        "max_new": args.new,
        "hf_fp32_greedy_tokens": hf_tokens,
        "hf_fp32_greedy_text": tokenizer.decode(hf_tokens),
        "hf_fp32_generate_seconds_gpu": hf_fp32_generate_s,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cublas_bf16_ms": cublas,
    }
    with open(os.path.join(args.out, "hf_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
