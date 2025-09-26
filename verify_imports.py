#!/usr/bin/env python3
"""
Verify that all imports work correctly after moving to src/veritor.
"""

import sys
import traceback

def test_imports():
    """Test all the imports."""

    print("Testing imports after src/veritor restructure...")
    print("=" * 60)

    errors = []

    # Test direct imports from src.veritor
    try:
        from src.veritor import IRStore, IRRole, IRFormat
        print("✓ Can import from src.veritor")
    except ImportError as e:
        errors.append(f"Failed to import from src.veritor: {e}")
        print(f"✗ {errors[-1]}")

    # Test data models
    try:
        from src.veritor.data_models import Graph, Trace, DataBundle
        print("✓ Can import data models")
    except ImportError as e:
        errors.append(f"Failed to import data models: {e}")
        print(f"✗ {errors[-1]}")

    # Test API
    try:
        from src.veritor.api import WorkloadDatabase
        print("✓ Can import WorkloadDatabase")
    except ImportError as e:
        errors.append(f"Failed to import WorkloadDatabase: {e}")
        print(f"✗ {errors[-1]}")

    # Test IR store
    try:
        from src.veritor.ir_store import IRStore, TransformationRecord
        print("✓ Can import IR store components")
    except ImportError as e:
        errors.append(f"Failed to import IR store: {e}")
        print(f"✗ {errors[-1]}")

    # Test creating instances
    try:
        from src.veritor.ir_store import IRStore
        from src.veritor.api import WorkloadDatabase

        store = IRStore()
        db = WorkloadDatabase()
        print("✓ Can create IRStore and WorkloadDatabase instances")
    except Exception as e:
        errors.append(f"Failed to create instances: {e}")
        print(f"✗ {errors[-1]}")

    # Test basic functionality
    try:
        from src.veritor.ir_store import IRStore, IRFormat
        store = IRStore()
        blob_id = store.put_blob("test", IRFormat.STABLEHLO)
        assert blob_id is not None
        print("✓ Basic IRStore functionality works")
    except Exception as e:
        errors.append(f"IRStore functionality failed: {e}")
        print(f"✗ {errors[-1]}")

    print("=" * 60)

    if errors:
        print(f"\n❌ {len(errors)} import errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All imports working correctly!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)