import pytest

from veritor.core import (
    InvalidArtifact,
    canonical_json_bytes,
    identity_digest,
    tagged_sha256,
    validate_digest,
)


def digest(label: str):
    return identity_digest("test/digest", {"label": label})


def test_canonical_json_is_sorted_compact_and_order_independent():
    left = {"z": [3, 2], "a": {"β": "ok", "x": True}}
    right = {"a": {"x": True, "β": "ok"}, "z": (3, 2)}

    encoded = canonical_json_bytes(left)

    assert encoded == canonical_json_bytes(right)
    assert encoded == '{"a":{"x":true,"β":"ok"},"z":[3,2]}'.encode()


@pytest.mark.parametrize(
    "manifest",
    [
        {"probability": 0.1},
        {1: "non-string key"},
        {"bytes": b"not-json"},
        {"set": {1, 2}},
        {"callable": lambda: None},
    ],
)
def test_canonical_json_rejects_noncanonical_values(manifest):
    with pytest.raises(TypeError):
        canonical_json_bytes(manifest)


def test_tagged_sha256_is_deterministic_and_domain_separated():
    first = tagged_sha256("test/a", b"payload")

    assert first == tagged_sha256("test/a", memoryview(b"payload"))
    assert first != tagged_sha256("test/b", b"payload")
    assert first != tagged_sha256("test/a", b"payload!")
    assert len(first) == 64
    assert first == first.lower()


def test_identity_digest_hashes_the_canonical_manifest():
    assert digest("a") == digest("a")
    assert digest("a") != digest("b")
    assert digest("a") != identity_digest("test/other", {"label": "a"})


def test_digest_validation_rejects_noncanonical_sha256_text():
    valid = digest("valid")

    assert validate_digest(valid) == valid
    with pytest.raises(InvalidArtifact, match="lowercase"):
        validate_digest(valid.upper())
    with pytest.raises(InvalidArtifact, match="64-character"):
        validate_digest("abc")
    with pytest.raises(InvalidArtifact, match="hexadecimal"):
        validate_digest("z" * 64)
