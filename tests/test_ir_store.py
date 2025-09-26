"""
Test the IR store functionality with pytest.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.veritor.ir_store import IRStore, IRRole, IRFormat
from src.veritor.api import WorkloadDatabase


class TestIRBlobStorage:
    """R1: IR blob storage with content-addressable IDs"""

    def test_same_content_same_id(self):
        """Same payload from different call sites produces same blob_id"""
        store = IRStore()

        stablehlo = """
        module @test {
            func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
                return %arg0 : tensor<2x3xf32>
            }
        }
        """

        blob_id1 = store.put_blob(stablehlo, IRFormat.STABLEHLO)
        blob_id2 = store.put_blob(stablehlo, IRFormat.STABLEHLO)

        assert blob_id1 == blob_id2

    def test_whitespace_normalization(self):
        """Whitespace differences don't change blob_id"""
        store = IRStore()

        # With trailing spaces
        ir1 = "module @test { return }  \n"
        # Without trailing spaces
        ir2 = "module @test { return }\n"

        blob_id1 = store.put_blob(ir1, IRFormat.STABLEHLO)
        blob_id2 = store.put_blob(ir2, IRFormat.STABLEHLO)

        assert blob_id1 == blob_id2


class TestSidecarMapping:
    """R2: Sidecar mapping between graphs and IR roles"""

    def test_attach_and_retrieve_ir(self):
        """Can attach IR to graph and retrieve in O(1)"""
        store = IRStore()

        graph_id = "model_v1"
        logical_ir = "module @logical { return }"

        blob_id = store.attach_ir(graph_id, IRRole.LOGICAL, logical_ir, IRFormat.STABLEHLO)
        retrieved = store.get_ir(graph_id, IRRole.LOGICAL)

        assert retrieved is not None
        assert retrieved.blob_id == blob_id
        assert retrieved.content == logical_ir.encode('utf-8')

    def test_list_ir_roles(self):
        """Can list all IR roles for a graph"""
        store = IRStore()

        graph_id = "model_v1"
        store.attach_ir(graph_id, IRRole.LOGICAL, "logical", IRFormat.STABLEHLO)
        store.attach_ir(graph_id, IRRole.DISTRIBUTED, "distributed", IRFormat.HLO)

        roles = store.list_ir_roles(graph_id)

        assert len(roles) == 2
        role_types = {r for r, _, _ in roles}
        assert IRRole.LOGICAL in role_types
        assert IRRole.DISTRIBUTED in role_types


class TestProvenance:
    """R3: Provenance/transformation chain tracking"""

    def test_link_derivation(self):
        """Can record transformation between graphs"""
        store = IRStore()

        source = "logical_v1"
        target = "distributed_v1"

        store.attach_ir(source, IRRole.LOGICAL, "logical", IRFormat.STABLEHLO)
        store.attach_ir(target, IRRole.DISTRIBUTED, "distributed", IRFormat.HLO)
        store.link_derivation(source, target, "partition")

        lineage = store.get_lineage(target)

        assert len(lineage) == 1
        assert lineage[0].source_graph_id == source
        assert lineage[0].target_graph_id == target
        assert lineage[0].transformation_type == "partition"

    def test_find_related(self):
        """Can find related graphs"""
        store = IRStore()

        # Create chain: A -> B -> C
        #                 -> D
        store.link_derivation("A", "B", "transform1")
        store.link_derivation("B", "C", "transform2")
        store.link_derivation("A", "D", "transform3")

        children_of_a = store.find_related("A", "children")
        assert set(children_of_a) == {"B", "D"}

        parents_of_c = store.find_related("C", "parents")
        assert set(parents_of_c) == {"B"}

        siblings_of_b = store.find_related("B", "siblings")
        assert "D" in siblings_of_b


class TestCompatibilityMetadata:
    """R4: Compatibility metadata for precision tracking"""

    def test_precision_metadata(self):
        """Can set and retrieve precision metadata"""
        store = IRStore()

        graph_id = "mixed_precision_model"
        store.set_precision_metadata(
            graph_id,
            intent_precision="fp32",
            effective_precision="mixed"
        )

        intent, effective = store.get_precision_metadata(graph_id)

        assert intent == "fp32"
        assert effective == "mixed"


class TestPersistence:
    """R6: Persistence and integrity checks"""

    def test_save_and_load(self, tmp_path):
        """Can save and load IR store"""
        store = IRStore()

        # Add data
        ir_content = "module @test { return }"
        graph_id = "test_graph"
        blob_id = store.attach_ir(graph_id, IRRole.LOGICAL, ir_content, IRFormat.STABLEHLO)
        store.set_precision_metadata(graph_id, intent_precision="fp32")

        # Save
        save_path = tmp_path / "ir_store"
        store.save(str(save_path))

        # Load
        loaded = IRStore.load(str(save_path))

        # Verify
        retrieved = loaded.get_ir(graph_id, IRRole.LOGICAL)
        assert retrieved is not None
        assert retrieved.blob_id == blob_id

        intent, _ = loaded.get_precision_metadata(graph_id)
        assert intent == "fp32"

    def test_corruption_detection(self, tmp_path):
        """Detects corrupted IR blobs"""
        store = IRStore()

        ir_content = "module @test { return }"
        blob_id = store.put_blob(ir_content, IRFormat.STABLEHLO)

        # Save
        save_path = tmp_path / "ir_store"
        store.save(str(save_path))

        # Corrupt a blob file
        blob_file = save_path / "blobs" / f"{blob_id}.blob"
        with open(blob_file, 'ab') as f:
            f.write(b'CORRUPTED')

        # Should raise error on load
        with pytest.raises(ValueError, match="Integrity check failed"):
            IRStore.load(str(save_path))


class TestRoundTrip:
    """R7: Round-trip serialization preservation"""

    def test_round_trip_preservation(self):
        """Round-trip preserves bit-for-bit equality"""
        store = IRStore()

        original = "module @test { return }"
        original_id = store.put_blob(original, IRFormat.STABLEHLO)

        # Get and re-put
        blob = store.get_blob(original_id)
        new_id = store.put_blob(blob.content, blob.format)

        assert new_id == original_id

    def test_round_trip_method(self):
        """Built-in round_trip_test method works"""
        store = IRStore()

        blob_id = store.put_blob("test content", IRFormat.STABLEHLO)
        assert store.round_trip_test(blob_id)


class TestWorkloadDatabaseIntegration:
    """Test integration with WorkloadDatabase"""

    def test_store_graph_with_ir(self):
        """Can store graph with IR in WorkloadDatabase"""
        db = WorkloadDatabase()

        stablehlo = """
        module @model {
            func.func @forward() { return }
        }
        """

        graph_id = db.store_graph_with_ir(
            "model_v1",
            stablehlo,
            IRRole.LOGICAL,
            IRFormat.STABLEHLO,
            metadata={'version': 1}
        )

        # Retrieve IR
        ir_content = db.get_graph_ir(graph_id, IRRole.LOGICAL)
        assert ir_content == stablehlo.encode('utf-8')

    def test_database_persistence(self, tmp_path):
        """WorkloadDatabase saves and loads IR store"""
        db = WorkloadDatabase()

        # Add graph with IR
        graph_id = db.store_graph_with_ir(
            "model_v1",
            "module @test { return }",
            IRRole.LOGICAL,
            IRFormat.STABLEHLO
        )

        # Save
        save_path = tmp_path / "workload_db"
        db.save(str(save_path))

        # Load
        loaded_db = WorkloadDatabase.load(str(save_path))

        # Verify IR preserved
        ir_content = loaded_db.get_graph_ir(graph_id, IRRole.LOGICAL)
        assert ir_content == b"module @test { return }"