from __future__ import annotations

from collections import Counter

import pytest

from circuit_cut_analysis.mincut import CutStatus, minimum_vertex_cut
from circuit_cut_analysis.motifs import (
    ExpandedMotif,
    Primitive,
    dot_product_counts,
    expand_dot_product,
    expand_elementwise_add,
    expand_gelu_new,
    expand_layer_norm,
    expand_softmax,
    gelu_new_counts,
    layer_norm_counts,
    softmax_counts,
)


@pytest.mark.parametrize("length", [1, 2, 5])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("scale", [False, True])
def test_dot_product_symbolic_counts_match_expansion(
    length: int,
    bias: bool,
    scale: bool,
) -> None:
    motif = expand_dot_product(length, bias=bias, scale=scale)

    assert motif.primitive_counts == dot_product_counts(
        length,
        bias=bias,
        scale=scale,
    )
    assert motif.unit_gate_count == sum(motif.primitive_counts.values())


@pytest.mark.parametrize("size", [1, 2, 4])
def test_layer_norm_symbolic_counts_match_expansion(size: int) -> None:
    motif = expand_layer_norm(size)

    assert motif.primitive_counts == layer_norm_counts(size)
    assert motif.unit_gate_count == 7 * size + 2


@pytest.mark.parametrize("size", [1, 2, 5])
def test_softmax_symbolic_counts_match_expansion(size: int) -> None:
    motif = expand_softmax(size)

    assert motif.primitive_counts == softmax_counts(size)
    assert motif.unit_gate_count == 5 * size - 1


def test_gelu_symbolic_counts_match_expansion() -> None:
    motif = expand_gelu_new()

    assert motif.primitive_counts == gelu_new_counts()
    assert motif.primitive_counts == Counter(
        {
            Primitive.MUL.value: 6,
            Primitive.ADD.value: 2,
            Primitive.TANH.value: 1,
        }
    )


@pytest.mark.parametrize(
    "motif",
    [
        expand_dot_product(4, bias=True),
        expand_dot_product(4, bias=False, scale=True),
        expand_gelu_new(),
        expand_layer_norm(4),
        expand_softmax(4),
        expand_elementwise_add(),
    ],
)
def test_declared_output_boundary_cuts_every_computed_gate(
    motif: ExpandedMotif,
) -> None:
    circuit = motif.circuit
    sources = motif.computed_gates
    output_cut = motif.bottlenecks.get(
        "output",
        motif.bottlenecks.get("outputs"),
    )

    assert output_cut is not None
    assert circuit.is_downstream_cut(sources, output_cut)


def test_narrow_output_closes_dot_product_and_gelu() -> None:
    dot = expand_dot_product(8, bias=True)
    gelu = expand_gelu_new()

    dot_result = minimum_vertex_cut(dot.circuit, dot.computed_gates)
    gelu_result = minimum_vertex_cut(gelu.circuit, gelu.computed_gates)

    assert dot_result.status is CutStatus.FINITE
    assert dot_result.width_bits == 16
    assert dot_result.downstream_most_cut == dot.bottlenecks["output"]
    assert gelu_result.width_bits == 16
    assert gelu_result.downstream_most_cut == gelu.bottlenecks["output"]


def test_shared_layer_norm_state_retains_fp32_self_cut() -> None:
    motif = expand_layer_norm(4)

    mean = minimum_vertex_cut(motif.circuit, {"mean"})
    variance_sum = minimum_vertex_cut(motif.circuit, {"variance_sum:3"})
    coordinate = minimum_vertex_cut(motif.circuit, {"normalized:0"})

    assert mean.width_bits == 32
    assert variance_sum.width_bits == 32
    assert coordinate.width_bits == 16
    assert coordinate.downstream_most_cut == motif.bottlenecks["output:0"]


def test_softmax_shared_values_do_not_collapse_to_one_probability() -> None:
    motif = expand_softmax(4)

    shifted = minimum_vertex_cut(motif.circuit, {"shifted:0"})
    reciprocal = minimum_vertex_cut(motif.circuit, {"reciprocal"})
    probability = minimum_vertex_cut(motif.circuit, {"probability:0"})

    assert shifted.width_bits == 32
    assert reciprocal.width_bits == 32
    assert probability.width_bits == 16
