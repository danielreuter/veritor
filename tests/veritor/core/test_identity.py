from dataclasses import FrozenInstanceError

import pytest

from veritor.core import (
    ArtifactKind,
    CompiledResultIdentity,
    InvalidArtifact,
    PartitionIdentity,
    PartitionKind,
    StructureIdentity,
    canonical_json_bytes,
    identity_digest,
    tagged_sha256,
    validate_digest,
)


def digest(label: str):
    return identity_digest("test/digest", {"label": label})


def structure(**changes):
    fields = {
        "schema_version": "1",
        "artifact_kind": ArtifactKind.EXECUTABLE_CIRCUIT,
        "compiler_id": "tests.compiler",
        "compiler_version": "1.2.3",
        "semantic_scope_id": "word-machine-v1",
        "representation_digest": digest("representation"),
        "value_registry_digest": digest("values"),
        "operator_registry_digest": digest("operators"),
    }
    fields.update(changes)
    return StructureIdentity(**fields)


def partition(
    structure_identity: StructureIdentity,
    kind: PartitionKind,
    suffix: str,
) -> PartitionIdentity:
    return PartitionIdentity(
        partition_kind=kind,
        structure_digest=structure_identity.digest,
        algorithm_id=f"test.{suffix}",
        algorithm_version="1",
        configuration_digest=digest(f"{suffix}-configuration"),
        representation_digest=digest(f"{suffix}-representation"),
    )


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


def test_identity_manifest_hashes_every_semantic_field():
    base = structure()

    assert base == structure()
    assert base.digest == structure().digest
    assert base.digest != structure(compiler_version="2").digest
    assert base.digest != structure(semantic_scope_id="other").digest
    assert base.digest != structure(operator_registry_digest=digest("other")).digest
    assert not hasattr(base, "__dict__")
    with pytest.raises(FrozenInstanceError):
        base.compiler_id = "mutated"


def test_structure_identity_can_hash_a_representation_manifest_directly():
    direct = StructureIdentity.from_manifest(
        {"gates": [{"operation": "copy", "reads": [0]}]},
        schema_version="1",
        artifact_kind=ArtifactKind.STRUCTURAL_CIRCUIT,
        compiler_id="tests.compiler",
        compiler_version="1",
        semantic_scope_id="structural-only",
    )

    assert direct.representation_digest == identity_digest(
        "veritor/structure-representation/v1",
        {"gates": [{"operation": "copy", "reads": [0]}]},
    )


def test_compiled_identity_binds_ordered_component_roles():
    structure_identity = structure()
    replay = partition(structure_identity, PartitionKind.REPLAY, "replay")
    verification = partition(
        structure_identity,
        PartitionKind.VERIFICATION,
        "verification",
    )

    compiled = CompiledResultIdentity.from_components(
        structure_identity,
        replay,
        verification,
    )

    assert compiled.structure_digest == structure_identity.digest
    assert compiled.replay_partition_digest == replay.digest
    assert compiled.verification_partition_digest == verification.digest
    with pytest.raises(InvalidArtifact, match="wrong partition kind"):
        CompiledResultIdentity.from_components(
            structure_identity,
            verification,
            replay,
        )


def test_digest_validation_rejects_noncanonical_sha256_text():
    valid = digest("valid")

    assert validate_digest(valid) == valid
    with pytest.raises(InvalidArtifact, match="lowercase"):
        validate_digest(valid.upper())
    with pytest.raises(InvalidArtifact, match="64-character"):
        validate_digest("abc")
    with pytest.raises(InvalidArtifact, match="hexadecimal"):
        validate_digest("z" * 64)
