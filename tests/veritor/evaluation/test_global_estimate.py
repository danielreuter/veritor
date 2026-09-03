"""The headline estimate: explicit inputs, a budget that is met, and the closed-form reading."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from veritor.evaluation.global_estimate import Inputs, estimate, render, sensitivity


@pytest.fixture(scope="module")
def base():
    return estimate(Inputs(), grid=60)


def test_defaults_land_in_the_terabyte_range_and_meet_the_budget(base) -> None:
    assert 1e13 <= base.capacity_bits <= 5e13  # ~2.4 TB
    assert base.overhead <= base.inputs.budget * (1 + 1e-9)
    assert 0 < base.q <= 1 and 0 < base.s <= 1


def test_the_reading_formula_matches_when_the_scattered_channel_binds(base) -> None:
    i = base.inputs
    reading = i.lam * (base.rate.verification_bits + math.log2(base.verification_units) + 2) * i.alpha * math.log(2) / i.budget
    assert base.rate.binding == 1
    assert abs(base.capacity_bits / reading - 1) < 0.05


def test_capacity_scales_with_alpha_and_against_the_budget(base) -> None:
    dearer = estimate(replace(base.inputs, alpha=10 * base.inputs.alpha), grid=60)
    richer = estimate(replace(base.inputs, budget=10 * base.inputs.budget), grid=60)
    assert 8 < dearer.capacity_bits / base.capacity_bits < 12
    assert 8 < base.capacity_bits / richer.capacity_bits < 12


def test_gate_granularity_is_priced_but_barely_matters_at_a_tiny_q(base) -> None:
    gate = estimate(replace(base.inputs, interior="gate"), grid=60)
    assert gate.ru_positions > 1000 * base.ru_positions
    assert gate.commit_overhead > base.commit_overhead
    assert gate.capacity_bits < 1.1 * base.capacity_bits


def test_render_and_sensitivity_have_every_input() -> None:
    rows = sensitivity(replace(Inputs(), tokens_per_year=1e15))
    names = {name for name, _, _ in rows}
    assert {"alpha", "budget", "lam", "hash_macs", "values_per_leaf", "interior", "tokens_per_year"} <= names
    text = render(rows[0][2], rows[:2], [])
    assert "U(lambda = 40)" in text and "| alpha |" in text
