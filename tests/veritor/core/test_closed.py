"""``KindSummary.closed``: a kind fed nothing but source gates at every call site.

Hand-built descriptions through the tracer, indexed without compiling: the
flag is a property of the description alone.  A port is retained when every
caller feeds it a source gate or a retained port of its own; a computed
gate, another call's output or a transient port of the caller make it
transient, and one transient port anywhere makes the kind open.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from veritor.compile.description import parse_description
from veritor.constructors import TracedDefinition, Tracer
from veritor.core import Index, KindTable, make_word_gate_set

GATES = make_word_gate_set(8)


def closed_by_key(tracer: Tracer, root: TracedDefinition, **named: TracedDefinition) -> dict[str, bool]:
    """``closed`` of the named definitions (and ``root``) in the index of ``root``."""

    kinds = {row.kind: row for row in Index(parse_description(tracer.serialize(root), GATES).root).kinds()}
    named["root"] = root
    return {name: kinds[definition.digest].closed for name, definition in named.items()}


def cells(tracer: Tracer) -> tuple[TracedDefinition, TracedDefinition]:
    """Two one-gate verification units, ``add`` and ``mul``, with two ports each."""

    add, mul = tracer.gate("add"), tracer.gate("mul")
    plus = tracer.definition(input_count=2, key="plus", role="verification")(lambda v: add(v[0], v[1]))
    times = tracer.definition(input_count=2, key="times", role="verification")(lambda v: mul(v[0], v[1]))
    return plus, times


def test_a_kind_fed_weights_only_is_closed_and_so_are_the_root_and_the_source_cells() -> None:
    tracer = Tracer(GATES)
    plus, _ = cells(tracer)

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(2)
        x = tracer.inputs(2)
        return [plus(w[0], w[1]), plus(x[0], w[1]), plus(x[0], x[1])]

    flags = closed_by_key(tracer, root, plus=plus, w=tracer.source_cell("weight"), x=tracer.source_cell("input"))

    assert flags == {"plus": True, "w": True, "x": True, "root": True}


def test_a_kind_fed_a_computed_value_is_not_closed() -> None:
    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(3)
        product = times(w[0], w[1])  # retained ports
        return plus(product, w[2])  # port 0 reads a computed value

    assert closed_by_key(tracer, root, plus=plus, times=times) == {"plus": False, "times": True, "root": True}


def test_a_kind_fed_a_callers_port_is_closed_iff_that_port_is_retained() -> None:
    """Two-level nesting, port by port: ``outer`` hands ports to ``inner`` kinds."""

    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=3, key="outer")
    def outer(v):
        return [plus(v[1], v[2]), times(v[0], v[1])]  # ``plus`` reads ports 1, 2; ``times`` ports 0, 1

    @tracer.definition(input_count=0, key="retained", role="replay")
    def retained(_v):
        w = tracer.weights(3)
        return outer(w[0], w[1], w[2])

    assert closed_by_key(tracer, retained, outer=outer, plus=plus, times=times) == {
        "outer": True,
        "plus": True,
        "times": True,
        "root": True,
    }

    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=3, key="outer")
    def outer2(v):
        return [plus(v[1], v[2]), times(v[0], v[1])]

    @tracer.definition(input_count=0, key="transient", role="replay")
    def transient(_v):
        w = tracer.weights(4)
        first = plus(w[2], w[3])  # computed, into port 0 of ``outer`` only
        return outer2(first, w[1], w[2])

    # port 0 of ``outer`` is transient: ``times`` reads it, ``plus`` does not (though ``plus`` is
    # also called directly on weights in the root, and both sites are retained for it)
    assert closed_by_key(tracer, transient, outer=outer2, plus=plus, times=times) == {
        "outer": False,
        "plus": True,
        "times": False,
        "root": True,
    }


def test_a_kind_called_from_a_retained_and_a_transient_site_is_not_closed() -> None:
    tracer = Tracer(GATES)
    plus, _ = cells(tracer)

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(3)
        first = plus(w[0], w[1])  # weights only
        return plus(first, w[2])  # the same kind, fed a computed value

    assert closed_by_key(tracer, root, plus=plus) == {"plus": False, "root": True}


def test_a_repeat_reading_another_repeats_outputs_across_copies_is_not_closed() -> None:
    """Copy ``j`` of the second repeat reads output ``j`` of the first: a strided, transient argument."""

    tracer = Tracer(GATES)
    plus, times = cells(tracer)
    n = 5

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(n + 1)
        products = tracer.repeat(n, times, w[0].by(1), w[1].by(1))  # copy j: w[j] * w[j + 1], closed
        return tracer.repeat(n, plus, products[0].by(1), w[0].by(1))  # copy j: products[j] + w[j]

    assert closed_by_key(tracer, root, plus=plus, times=times) == {"plus": False, "times": True, "root": True}


def test_a_repeat_whose_strided_argument_crosses_from_weights_into_computed_values_is_not_closed() -> None:
    """One argument run of a repeat may cover source gates in early copies and computed values later."""

    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(4)
        tracer.repeat(2, times, w[0].by(1), w[1].by(1))  # two products in slots 4, 5 after the four weights
        # copy j reads slot 2 + j: copies 0 and 1 read w[2], w[3]; copies 2 and 3 read the products
        return tracer.repeat(4, plus, w[2].by(1), w[0].by(1))

    assert closed_by_key(tracer, root, plus=plus, times=times) == {"plus": False, "times": True, "root": True}


def test_a_repeat_over_a_callers_ports_is_closed_iff_the_ports_it_strides_over_are_retained() -> None:
    """Input-space arguments that shift per copy: copy ``j`` of ``inner`` reads ports ``2j, 2j + 1``."""

    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=4, key="outer")
    def outer(v):
        return tracer.repeat(2, plus, v[0].by(2), v[1].by(2))

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(4)
        product = times(w[0], w[1])
        return outer(w[0], w[1], product, w[3])  # port 2 of ``outer`` is transient: copy 1 of the repeat

    assert closed_by_key(tracer, root, outer=outer, plus=plus, times=times) == {
        "outer": False,
        "plus": False,
        "times": True,
        "root": True,
    }

    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=4, key="outer")
    def outer2(v):
        return tracer.repeat(2, plus, v[0].by(2), v[1].by(2))

    @tracer.definition(input_count=0, key="root", role="replay")
    def root2(_v):
        w = tracer.weights(4)
        return outer2(w[0], w[1], w[2], w[3])

    assert closed_by_key(tracer, root2, outer=outer2, plus=plus) == {"outer": True, "plus": True, "root": True}


def test_circuit_outputs_are_not_retained() -> None:
    """A value that is also an output of the root is still transient when fed to another kind."""

    tracer = Tracer(GATES)
    plus, times = cells(tracer)

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(3)
        product = times(w[0], w[1])
        return [product, plus(product, w[2])]

    assert closed_by_key(tracer, root, plus=plus, times=times) == {"plus": False, "times": True, "root": True}


def test_the_table_records_the_flag_and_requires_portless_kinds_to_be_closed() -> None:
    tracer = Tracer(GATES)
    plus, _ = cells(tracer)

    @tracer.definition(input_count=0, key="root", role="replay")
    def root(_v):
        w = tracer.weights(2)
        return plus(w[0], w[1])

    index = Index(parse_description(tracer.serialize(root), GATES).root)
    table = index.kind_table()
    assert all(row.closed for row in table.rows)
    rows = tuple(replace(row, closed=False) if row.input_count == 0 else row for row in table.rows)
    with pytest.raises(ValueError, match="no ports and must be closed"):
        KindTable(
            rows=rows,
            root=table.root,
            n=table.n,
            input_count=table.input_count,
            weight_count=table.weight_count,
            replay_unit_count=table.replay_unit_count,
            digest=table.digest,
        )
