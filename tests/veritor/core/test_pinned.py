"""The pinned gate set: its CPU semantics against the RTX 4090 golden vectors, and its declarations.

``golden/ada_pinned_f32.json`` was written by ``gpu/gpt2/run_gpt2.py golden``
on an RTX 4090 (sm_89) with ``gpu/gpt2/pinned_ops.cu`` built with
``-fmad=false -prec-div=true -prec-sqrt=true -ftz=false``: 400 records per
elementwise gate over inputs that exercise every branch (specials, huge and
tiny magnitudes, ties), the fixed-tree reductions at every length GPT-2
uses, and one LayerNorm statistics row.  See ``docs/gpt2-silicon.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from veritor.constructors.gpt2_reference import NumpyOps, f32_add, tree_reduce
from veritor.core import Gate, GateSet, InvalidArtifact, make_pinned_gate_set
from veritor.core.silicon import f32_max, is_nan_word

GOLDEN = Path(__file__).parent / "golden" / "ada_pinned_f32.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gates() -> GateSet:
    return make_pinned_gate_set("sm_89", "bf16")


def words(hex_list: list[str]) -> list[int]:
    return [int(h, 16) for h in hex_list]


UNARY = ("f32_exp", "f32_tanh", "gelu_tanh", "ln_rstd", "f32_to_bf16")
BINARY = ("f32_add", "f32_sub", "f32_mul", "f32_div", "f32_max")


@pytest.mark.parametrize("name", UNARY + BINARY + ("argmax_select", "token_eq"))
def test_every_pinned_elementwise_gate_reproduces_the_gpu(
    golden: dict, gates: GateSet, name: str
) -> None:
    record = golden["gates"][name]
    gate = gates[name]
    assert (
        list(gate.arg_widths) == record["arg_widths"] and gate.width == record["width"]
    )
    outputs = words(record["y"])
    assert len(outputs) >= 300
    if name in UNARY:
        columns = [words(record["x"])]
    elif name in BINARY:
        columns = [words(record["a"]), words(record["b"])]
    elif name == "argmax_select":
        columns = [words(record[k]) for k in ("la", "lb", "ia", "ib")]
    else:
        columns = [words(record["t"]), words(record["j"])]
    checked = nan_rows = 0
    for args, expected in zip(zip(*columns, strict=True), outputs, strict=True):
        got = gate.evaluate(args)
        # a NaN output is one NaN pattern on the GPU and the canonical one here; the relation excludes it
        # (token outputs are not floating point: any 16-bit pattern is a token)
        if name not in ("argmax_select", "token_eq") and is_nan_word(
            gate.width, expected
        ):
            assert is_nan_word(gate.width, got)
            assert not gate.check(args, expected)
            nan_rows += 1
            continue
        assert got == expected, (
            name,
            [f"{a:#x}" for a in args],
            f"{got:#x}",
            f"{expected:#x}",
        )
        assert gate.check(args, expected)
        checked += 1
    assert checked >= 250, (name, checked, nan_rows)


def same_word(got: int, expected: int) -> bool:
    """Equal words, or both NaN (payloads are the one thing IEEE leaves open; the gate relations exclude NaN)."""

    return got == expected or (is_nan_word(32, got) and is_nan_word(32, expected))


def f32(hex_list: list[str]) -> np.ndarray:
    return np.array(words(hex_list), dtype=np.uint32).view(np.float32)


def test_the_tree_reductions_reproduce_the_gpu_at_every_length(golden: dict) -> None:
    lengths = []
    finite_sums = 0
    for row in golden["reductions"]:
        u = f32(row["u"])[None, :]
        total = int(tree_reduce(u, f32_add).view(np.uint32)[0])
        top = int(tree_reduce(u, f32_max).view(np.uint32)[0])
        assert same_word(total, words(row["tree_sum"])[0]), row["c"]
        assert top == words(row["tree_max"])[0], row["c"]
        finite_sums += not is_nan_word(32, total)
        lengths.append(row["c"])
    assert lengths == list(range(1, 25)) + [64, 768] and finite_sums >= 13


def test_the_gelu_of_the_run_reproduces_the_gpu(golden: dict, gates: GateSet) -> None:
    record = golden["gates"]["gelu_tanh_run"]
    gate = gates["gelu_tanh"]
    for x, y in zip(words(record["x"]), words(record["y"]), strict=True):
        assert gate.evaluate((x,)) == y and gate.check((x,), y)
    assert len(record["x"]) == 400


def test_the_layer_norm_rows_of_the_run_reproduce_the_gpu(golden: dict) -> None:
    """24 LayerNorm rows of the GPT-2 Small capture: the 768-long tree sums, mean, rstd (numpy vs the GPU kernels)."""

    ops = NumpyOps()
    n = np.float32(768.0)
    for row in golden["layer_norm_rows"]:
        x = f32(row["x"])[None, :]
        assert x.shape == (1, 768)
        mean, _center, rstd = ops.ln_stats(x, n)
        assert int(mean.view(np.uint32)[0]) == words(row["mean"])[0], row
        assert int(rstd.view(np.uint32)[0]) == words(row["rstd"])[0], row
    assert len(golden["layer_norm_rows"]) == 24


def test_the_softmax_rows_of_the_run_reproduce_the_gpu(golden: dict) -> None:
    """One causal softmax row per length 1..34 of the capture: tree max, shifted exponentials, tree sum, rounded probabilities."""

    ops = NumpyOps()
    for row in golden["softmax_rows"]:
        u = f32(row["u"])[None, :]
        m = ops.row_max(u)
        e = ops.exp_shift(u, m)
        total = ops.row_sum(e)
        p = ops.div_round(e, total)
        assert m.view(np.uint32).tolist() == words(row["m"]), row["c"]
        assert e.view(np.uint32).reshape(-1).tolist() == words(row["e"]), row["c"]
        assert total.view(np.uint32).tolist() == words(row["S"]), row["c"]
        assert p.reshape(-1).tolist() == words(row["p"]), row["c"]
    assert [row["c"] for row in golden["softmax_rows"]] == list(range(1, 35))


def test_the_pinned_gate_set_declares_mixed_widths_and_nan_free_relations(
    gates: GateSet,
) -> None:
    assert gates.id == "veritor.pinned.ada_bf16_m16n8k16@1"
    step = gates["tc_dot16"]
    assert (step.arity, step.width, step.arg_widths[:2], set(step.arg_widths[1:])) == (
        33,
        32,
        (32, 16),
        {16},
    )
    zero = gates["tc_dot16_0"]
    assert (zero.arity, zero.width, set(zero.arg_widths)) == (32, 32, {16})
    one = 0x3F80
    # 1 * 1 summed over one k-slot from zero and from an accumulator of 2.0
    assert zero.evaluate((one, *([0] * 15), one, *([0] * 15))) == 0x3F800000
    assert step.evaluate((0x40000000, one, *([0] * 15), one, *([0] * 15))) == 0x40400000
    assert gates["bf16_to_f32"].evaluate((one,)) == 0x3F800000
    assert gates["f32_to_bf16"].evaluate((0x3F800001,)) == one  # rounds down
    assert gates["f32_to_bf16"].evaluate((0x3F808000,)) == one  # tie to even
    assert gates["f32_to_bf16"].evaluate((0x3F818000,)) == 0x3F82  # tie to even, up
    assert (
        gates["argmax_select"].evaluate((0x3F800000, 0x3F800000, 5, 9)) == 5
    )  # ties keep the earlier
    assert (
        gates["token_eq"].evaluate((7, 7)) == one
        and gates["token_eq"].evaluate((7, 8)) == 0
    )
    nan = 0x7FC00000
    assert not gates["f32_add"].check(
        (nan, 0), nan
    )  # a NaN anywhere fails the relation
    assert gates["f32_add"].check((0x3F800000, 0x3F800000), 0x40000000)
    assert not gates["f32_add"].check((0x3F800000, 0x3F800000), 0x40000001)
    with pytest.raises(InvalidArtifact, match="16-bit"):
        step.evaluate((0, 0x10000, *([0] * 31)))  # an operand word wider than 16 bits
    assert {g.name for g in gates} >= {
        "in",
        "weight",
        "f32_exp",
        "f32_tanh",
        "gelu_tanh",
        "ln_rstd",
    }
    plain = Gate(
        "x",
        2,
        8,
        replay_cost=1,
        proof_cost=1,
        evaluate=lambda args: (args[0] + args[1]) & 255,
    )
    assert plain.arg_widths == (8, 8) and gates["f32_add"].arg_widths == (32, 32)
    assert "arg_widths" not in plain.manifest and gates["argmax_select"].manifest[
        "arg_widths"
    ] == [32, 32, 16, 16]
