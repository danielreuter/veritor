from decimal import Decimal
from fractions import Fraction

import pytest

from veritor.core import VerificationPolicy, exact_fraction, rational_manifest


def test_policy_accepts_only_exact_inputs_and_reduces_them():
    policy = VerificationPolicy(
        1,
        Fraction(6, 10),
    )

    assert policy.q == Fraction(1, 1)
    assert policy.s == Fraction(3, 5)
    assert VerificationPolicy(1, Decimal("0.125")).s == Fraction(1, 8)
    assert rational_manifest(policy.s) == {
        "numerator": 3,
        "denominator": 5,
    }
    assert policy.denominator_bits == 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1/3", Fraction(1, 3)),
        (" 6/10 ", Fraction(3, 5)),
        ("0.125", Fraction(1, 8)),
        ("1e-400", Fraction(1, 10**400)),
        ("-0", Fraction(0)),
    ],
)
def test_exact_fraction_parses_decimal_and_fraction_strings(text, expected):
    assert exact_fraction(text) == expected


@pytest.mark.parametrize("value", [True, False, 0.0, 0.5, 1.0])
def test_policy_rejects_boolean_and_float_inputs(value):
    with pytest.raises(TypeError, match="int, Fraction, Decimal"):
        VerificationPolicy(value, 1)


@pytest.mark.parametrize(
    "value",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        "",
        "not-a-rational",
        "1/0",
    ],
)
def test_exact_fraction_rejects_invalid_exact_values(value):
    with pytest.raises(ValueError):
        exact_fraction(value)


@pytest.mark.parametrize(
    "values",
    [
        (-1, 0),
        ("1.0001", 0),
        (0, -1),
        (0, "4/3"),
    ],
)
def test_policy_enforces_closed_sampling_ranges(values):
    with pytest.raises(ValueError):
        VerificationPolicy(*values)


def test_policy_has_no_eta():
    """``eta`` is the verifier's (``VerifierParameters``), not part of ``theta``."""

    with pytest.raises(TypeError):
        VerificationPolicy(1, 1, 0)  # type: ignore[call-arg]
    assert not hasattr(VerificationPolicy(1, 1), "eta")


def test_equivalent_exact_inputs_have_identical_policy_identity():
    decimal = VerificationPolicy(Decimal("0.1"), "0.30")
    fraction = VerificationPolicy("1/10", Fraction(3, 10))

    assert decimal == fraction
    assert decimal.digest == fraction.digest
    assert decimal.manifest == {
        "q": {"numerator": 1, "denominator": 10},
        "s": {"numerator": 3, "denominator": 10},
    }
    assert decimal.replay_probability is decimal.q
    assert decimal.within_unit_probability is decimal.s
