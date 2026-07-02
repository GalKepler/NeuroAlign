"""Checks for the offline region_reference.json build pipeline (neuroalign.reference)."""

import json

import pytest

from neuroalign.reference import (
    APP_ASSETS_DIR,
    BAG_REGION_METRICS,
    bag_region_names,
    compute_centroids,
    merge_reference,
)

def _try_atlas_paths():
    from neuroalign.reference import _atlas_paths

    try:
        _atlas_paths()
        return True
    except (RuntimeError, FileNotFoundError):
        return False


_requires_atlas = pytest.mark.skipif(
    not _try_atlas_paths(),
    reason="Atlas dseg volume/TSV not available (CAT12_ATLAS_ROOT unset or missing).",
)


@_requires_atlas
def test_centroid_math_round_trips_for_label_one():
    """center_of_mass + affine round-trip: label 1's centroid must be inside the brain."""
    from neuroalign.reference import _atlas_paths

    nii_path, tsv_path = _atlas_paths()
    centroids = compute_centroids(nii_path, tsv_path)

    assert len(centroids) == 432
    for info in centroids.values():
        x, y, z = info["centroid_mni"]
        # Generous MNI152 brain-extent bounds (occipital/cerebellar regions reach y~-105).
        assert -100 < x < 100 and -120 < y < 100 and -100 < z < 100


def test_region_reference_contract_matches_bag_names():
    """The already-built region_reference.json must key exactly the BAG output's region names."""
    ref_path = APP_ASSETS_DIR / "region_reference.json"
    if not ref_path.exists() or not BAG_REGION_METRICS.exists():
        pytest.skip("region_reference.json or BAG region_metrics.parquet not built yet.")

    reference = json.loads(ref_path.read_text())
    assert set(reference.keys()) == bag_region_names()
    assert len(reference) == 432

    for name, info in reference.items():
        assert "centroid_mni" in info
        assert "plain_name" in info
        assert "terms" in info
        assert "studies" in info


def test_merge_reference_fails_loud_on_key_mismatch(monkeypatch, tmp_path):
    """The build-time contract must raise (not silently pass) on a key mismatch."""
    monkeypatch.setattr(
        "neuroalign.reference.bag_region_names", lambda: {"LH_Vis_1", "some_missing_region"}
    )
    monkeypatch.setattr("neuroalign.reference.APP_ASSETS_DIR", tmp_path)

    centroids = {"LH_Vis_1": {"index": 1, "centroid_mni": [0, 0, 0], "network": "Vis", "structure": "cortex"}}
    with pytest.raises(ValueError, match="don't match"):
        merge_reference(centroids, {})
