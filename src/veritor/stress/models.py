"""The toy models the catalogue's scenarios run on.

The shape is the simulated datacenter's (:mod:`veritor.simulation`), with a
context long enough for chunked prefills and shared prefixes: small enough
that every scenario compiles, evaluates and prices in well under a second,
big enough that every kind of the toy decoder appears.
"""

from __future__ import annotations

from dataclasses import dataclass

from veritor.constructors.lm import LMShape, Parameters, random_parameters
from veritor.core.gates import GateSet, make_isa_gate_set

__all__ = ["SAMPLED", "SHAPE", "Model", "make_model"]

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=16, width=16)
"""The catalogue's toy LM: the simulated datacenter's shape, with the argmax head."""

SAMPLED = LMShape(
    vocab=8, d_model=4, heads=2, layers=1, context=16, width=16, sampling=True
)
"""The same model with the ``sample`` VU over public randomness."""


@dataclass(frozen=True)
class Model:
    """A toy LM shape with its gate set and one draw of parameters."""

    shape: LMShape
    gate_set: GateSet
    parameters: Parameters

    @property
    def weights(self) -> tuple[int, ...]:
        return self.parameters.flatten()


def make_model(shape: LMShape, seed: int = 7) -> Model:
    return Model(shape, make_isa_gate_set(shape.width), random_parameters(shape, seed))
