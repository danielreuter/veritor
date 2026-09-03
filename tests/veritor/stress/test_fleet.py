"""Numerics-and-fleet scenarios N2 (tensor parallelism) and N3 (a heterogeneous fleet).

N2: the same requests through ``RequestsG`` at tensor-parallel degrees 1, 2
and 4.  A marked ``dot_k`` becomes ``t`` partial dots and a fixed-order
reduction: the VU kinds differ, the RU interfaces do not, so the tokens and
``U`` are unchanged (mechanism M2: public configuration).  N3: ``ClusterG``
over two pods on two namespaced copies of the toy ISA in one union gate set:
one circuit, two families of step kinds, a request prefilled on one
architecture and decoded on the other; a gate outside the union does not
compile.
"""

from __future__ import annotations

import pytest

from veritor.compile import CompileError, Compiler
from veritor.constructors import (
    ClusterG,
    Join,
    Request,
    RequestsG,
    Schedule,
    reference_generate,
)
from veritor.core.description import REPLAY, VERIFICATION
from veritor.stress.models import Model
from veritor.stress.rows import Recorder
from veritor.stress.serving import Served, serve

REQUESTS = (Request((1, 2, 3), 3), Request((5,), 4), Request((7, 0), 2), Request((2, 2, 2, 2), 3), Request((4, 6), 4))


def per_request(model: Model, requests: tuple[Request, ...], tensor_parallel: int = 1) -> Served:
    constructor = RequestsG(model.shape, tensor_parallel=tensor_parallel)
    layout = constructor.output_layout(requests)
    return serve(constructor, requests, b"", model.gate_set, model.weights, layout, len(requests))


# -- N2: tensor parallelism -------------------------------------------------------------


def test_n2_tensor_parallelism(scenario: Recorder, model: Model) -> None:
    reference = reference_generate(model.shape, model.parameters, REQUESTS)
    runs = {degree: per_request(model, REQUESTS, degree) for degree in (1, 2, 4)}

    assert all(run.tokens == reference for run in runs.values())
    assert len({run.digest for run in runs.values()}) == 3  # three circuits
    verification = {degree: set(run.kinds(VERIFICATION)) for degree, run in runs.items()}
    replay = {degree: set(run.kinds(REPLAY)) for degree, run in runs.items()}
    assert verification[1].isdisjoint(verification[2] - verification[1]) and verification[2] != verification[4]
    assert len(set.intersection(*replay.values())) == 1  # only the weights unit is shared: the request kinds differ
    # U is unchanged: the RU interfaces (prompts in, tokens out) do not depend on the reduction tree
    bounds = {degree: run.price.bound for degree, run in runs.items()}
    assert len({(result.bits, round(result.knapsack_bits, 6)) for result in bounds.values()}) == 1
    # a dot over k products needs k - 1 sums however they are grouped: the kinds differ in the tree, not in size
    dots = {degree: RequestsG(model.shape, tensor_parallel=degree).lm.dot(model.shape.hidden) for degree in (1, 2, 4)}
    sizes = {degree: runs[degree].measurement.compiled.kind_table() for degree in (1, 2, 4)}
    gates = {
        degree: next(row.size for row in sizes[degree].rows if row.kind == dots[degree].digest) for degree in (1, 2, 4)
    }
    assert gates[1] == gates[2] == gates[4] == 2 * model.shape.hidden - 1
    assert len({dot.digest for dot in dots.values()}) == 3

    for degree, letter in ((1, "a"), (2, "b"), (4, "c")):
        run = runs[degree]
        scenario.record(
            id=f"N2{letter}",
            what=f"tensor parallelism, RequestsG at TP = {degree}: marked dots are {degree} partial dots and a fixed-order reduction",
            mechanism="M2",
            advice_bits=0,
            capacity_bits=run.capacity_bits,
            overhead=run.overhead,
            description_bytes=run.description_bytes,
            verdict=(
                f"outputs identical across TP = 1, 2, 4; {len(verification[degree])} VU kinds "
                f"({len(verification[degree] - verification[1])} not in TP = 1); U identical (uncapped too); "
                f"dot_{model.shape.hidden} keeps {gates[degree]} gates, its sum tree regrouped"
            ),
            notes=run.notes("the TP degree is public configuration in the constructor's manifest, not advice"),
        )


# -- N3: heterogeneous fleet --------------------------------------------------------------


def test_n3_heterogeneous_fleet(scenario: Recorder, model: Model) -> None:
    """Two pods on ``sm80`` and ``sm90`` copies of the toy ISA: every request prefilled on ``sm80`` and decoded
    on ``sm90``, next to a homogeneous run of the same schedule."""

    requests = REQUESTS[:3]
    reference = reference_generate(model.shape, model.parameters, requests)
    joins = []
    for index, request in enumerate(requests):
        joins.append(Join(0, index, 0, index, 1))
        if request.max_new > 1:
            joins.append(Join(1, index + 1, index, index, request.max_new - 1, resume=True))
    schedule = Schedule(2, 3, 6, tuple(sorted(joins)))
    fleet = ClusterG(model.shape, 2, 3, 6, arches=("sm80", "sm90"))
    homogeneous = ClusterG(model.shape, 2, 3, 6)
    layout = fleet.output_layout(requests, schedule)
    mixed = serve(fleet, requests, schedule.encode(), fleet.gate_set, model.weights, layout, len(requests))
    same = serve(homogeneous, requests, schedule.encode(), model.gate_set, model.weights, layout, len(requests))

    assert mixed.tokens == reference == same.tokens
    assert fleet.gate_set.id == "veritor.toy-isa-fleet@1" and len(fleet.gate_set) == 2 * 6 + 2
    assert mixed.price.honest == same.price.honest  # the same gates under two names
    # step kinds carry their gate set: the sm80 prefills and the sm90 decodes are disjoint families
    steps_mixed = {kind for kind, _ in mixed.kinds(REPLAY).items()} - {fleet.lm.weights_unit().digest}
    steps_same = {kind for kind, _ in same.kinds(REPLAY).items()} - {homogeneous.lm.weights_unit().digest}
    assert steps_mixed.isdisjoint(steps_same) and len(steps_mixed) == len(steps_same)
    sm80, sm90 = fleet.models["sm80"], fleet.models["sm90"]
    verification = set(mixed.kinds(VERIFICATION))
    assert sm80.dot(model.shape.d_model).digest in verification and sm90.dot(model.shape.d_model).digest in verification
    assert sm80.argmax().digest in verification and sm90.argmax().digest in verification  # both heads decide
    # a description over the fleet's gates does not compile against the plain ISA, and vice versa
    with pytest.raises(CompileError, match="unknown gate"):
        Compiler(model.gate_set).compile(mixed.measurement.description, mixed.measurement.compilation.inputs)
    with pytest.raises(CompileError, match="unknown gate"):
        Compiler(fleet.gate_set).compile(same.measurement.description, same.measurement.compilation.inputs)

    scenario.record(
        id="N3",
        what="heterogeneous fleet, ClusterG: pod 0 on sm80, pod 1 on sm90 (two namespaced copies of the toy ISA in one Σ)",
        mechanism="M2",
        advice_bits=mixed.advice_bits,
        capacity_bits=mixed.capacity_bits,
        overhead=mixed.overhead,
        description_bytes=mixed.description_bytes,
        verdict=(
            f"outputs = reference; one circuit over union_gate_set ({len(fleet.gate_set)} gates: 6 operators x 2 + in, weight); "
            f"{len(steps_mixed)} step kinds carry their architecture; prefill on sm80 decoded on sm90; a foreign gate name fails to compile"
        ),
        notes=mixed.notes(
            "the pod-to-architecture map is public configuration in the manifest",
            f"same honest cost as the homogeneous run ({same.price.honest}); the homogeneous description differs in every gate name",
        ),
    )
