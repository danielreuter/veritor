"""VUs of the RTX 4090 run of GPT-2 Small re-execute bit-exactly on the CPU through the circuit's own gates.

``golden/gpt2_small_capture_slice.json`` holds 262 VUs of 88 kinds (three
of each, one of the 3072-long MLP dots) cut from the GPU capture of
``gpu/gpt2/results/capture_gpu.npz``: the pod's greedy decode of a
27-token prompt for 8 tokens with the fixed-order tensor-core chains and
the pinned elementwise kernels (``gpu/gpt2/pinned_ops.cu``).  Each VU
comes with the words the GPU produced at its input addresses (weights,
activations) and at its recorded output addresses.  Here GPT-2 Small is
compiled for the same request, the description digest is checked, and every
VU is re-executed gate by gate with ``Circuit.evaluate_gate`` /
``check_gate`` -- the pinned semantics in numpy float32 -- to the GPU's
words.  The whole-forward comparison (11,664,980 words, 0 mismatches) and
the 19,822 + 17,403 sampled VUs are in ``gpu/gpt2/results/`` and
``docs/gpt2-silicon.md``; this is the part of it that fits in the
repository.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from veritor.constructors import GPT2G, GPT2Shape, Request
from veritor.constructors.gpt2_reference import check_unit, evaluate_unit
from veritor.core import Compiled
from veritor.research import Compile

SLICE = Path(__file__).parent / "golden" / "gpt2_small_capture_slice.json"


@pytest.fixture(scope="module")
def capture() -> dict:
    with SLICE.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def compiled(capture: dict) -> tuple[GPT2G, Compiled]:
    constructor = GPT2G(GPT2Shape.small())
    request = Request(tuple(capture["prompt"]), capture["max_new"])
    compilation = Compile(constructor, (request,), b"", constructor.gate_set)
    return constructor, compilation.compiled


def test_the_slice_is_the_documented_run_of_gpt2_small(
    capture: dict, compiled: tuple[GPT2G, Compiled]
) -> None:
    constructor, circuit = compiled
    assert capture["shape"] == GPT2Shape.small().manifest
    assert len(capture["prompt"]) == 27 and capture["max_new"] == 8
    assert capture["tokens"] == [
        318,
        262,
        38760,
        2615,
        287,
        262,
        995,
        13,
    ]  # " is the tallest building in the world."
    assert circuit.digest == capture["description_digest"]
    assert circuit.circuit.n == capture["gates"] == 423_850_313
    assert (
        circuit.index.verification_unit_count
        == capture["verification_units"]
        == 133_318_577
    )
    assert circuit.index.replay_units.count == 2
    kinds = Counter(vu["kind"] for vu in capture["vus"])
    assert len(capture["vus"]) == 262 and len(kinds) == 88
    names = set(constructor.model.kind_names().values())
    assert set(kinds) <= names
    families = {name.split("(")[0] for name in kinds}
    assert families == {
        "add_cell",
        "argmax_block",
        "argmax_top",
        "dot",
        "eq_cell",
        "exp_cell",
        "gelu_cell",
        "ln_mean",
        "ln_out",
        "ln_var",
        "prob_cell",
        "scale_cell",
        "softmax_max",
        "softmax_sum",
        "sub_cell",
        "widen_cell",
    }
    assert {name for name in kinds if name.startswith("dot(")} == {
        "dot(16,False,True)",
        "dot(32,False,True)",
        "dot(48,False,True)",
        "dot(64,False,False)",
        "dot(768,False,False)",
        "dot(768,True,False)",
        "dot(768,True,True)",
        "dot(3072,True,False)",
    }
    assert SLICE.stat().st_size < 1 << 20


def test_every_vu_of_the_slice_re_executes_to_the_gpu_words(
    capture: dict, compiled: tuple[GPT2G, Compiled]
) -> None:
    constructor, circuit = compiled
    index = circuit.index
    names = constructor.model.kind_names()
    checked = agreeing = gates = 0
    for vu in capture["vus"]:
        node = index.verification_unit(vu["unit"])
        assert names[node.kind] == vu["kind"]
        values = {int(a): w for a, w in vu["inputs"].items()}
        outputs = {int(a): w for a, w in vu["outputs"].items()}
        assert all(a in node.interval for a in outputs) and not any(
            a in node.interval for a in values
        )
        values.update(outputs)
        c, a = check_unit(circuit, node, values)
        assert c == a == len(outputs), vu["kind"]
        checked += c
        agreeing += a
        gates += node.size
        # the framework's evaluation of the whole VU from its inputs alone gives the GPU's outputs
        known = evaluate_unit(
            circuit, node, {int(a): w for a, w in vu["inputs"].items()}
        )
        assert len(known) == node.size and all(
            known[a] == w for a, w in outputs.items()
        ), vu["kind"]
    assert (
        checked == agreeing == 283 and gates == 16_189
    )  # recorded outputs; gates re-executed to reach them


def test_the_words_have_the_declared_widths(
    capture: dict, compiled: tuple[GPT2G, Compiled]
) -> None:
    _, circuit = compiled
    for vu in capture["vus"]:
        for a, w in {**vu["inputs"], **vu["outputs"]}.items():
            ref = circuit.circuit[int(a)]
            assert 0 <= w < 1 << ref.width, (vu["kind"], a)
