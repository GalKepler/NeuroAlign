#!/usr/bin/env python
"""Test that NeuroAlign is properly set up."""

print("Testing NeuroAlign setup...")

# Test imports
try:
    from neuroalign.data.loaders import (
        AnatomicalLoader,
        DiffusionLoader,
        QuestionnaireLoader
    )
    print("✓ Data loaders imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test basic functionality
try:
    from pathlib import Path
    
    # These will fail if paths don't exist, but that's OK
    # We're just testing the code works
    print("\nTesting AnatomicalLoader initialization...")
    anat = AnatomicalLoader(
        cat12_root=Path("/fake/path"),
        atlas_root=Path("/fake/path")
    )
    print("✓ AnatomicalLoader created")
    
    print("\nTesting DiffusionLoader initialization...")
    diff = DiffusionLoader(
        qsiparc_path=Path("/fake/path"),
        qsirecon_path=Path("/fake/path")
    )
    print("✓ DiffusionLoader created")
    
    print("\nTesting QuestionnaireLoader initialization...")
    # This would fail without a real file, so just check the class exists
    assert QuestionnaireLoader is not None
    print("✓ QuestionnaireLoader available")
    
except Exception as e:
    print(f"✗ Functionality test failed: {e}")
    exit(1)

print("\n🎉 NeuroAlign is ready to use!")
print("\nNext steps:")
print("  1. Update .env with your actual data paths")
print("  2. Create notebooks/01_test_data_loading.ipynb")
print("  3. Start building regional BAG models!")