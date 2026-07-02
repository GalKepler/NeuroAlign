"""Static region reference pipeline: centroids + literature -> region_reference.json.

Builds the offline, versioned reference artifacts described in
`plans/app_kickoff.md` (steps 1-2): per-region MNI centroids from the atlas
volume, Neurosynth/NeuroQuery literature terms, and plain-language names,
merged into one `region_reference.json` keyed by region name (the same
keys used by the multivariate BAG output's `region_metrics.parquet`).

Usage:
    python -m neuroalign.reference centroids   # step 1: centroids + spot-check
    python -m neuroalign.reference literature  # step 2: enrich + merge + contract check
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.ndimage import center_of_mass

load_dotenv()

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"
APP_ASSETS_DIR = REPO_ROOT / "app" / "assets"
BAG_REGION_METRICS = (
    REPO_ROOT
    / "data"
    / "processed_full"
    / "bag"
    / "multivariate"
    / "anat_baseline"
    / "region_metrics.parquet"
)

# Schaefer 7-network + Tian structure abbreviations -> plain-language words.
# ponytail: small hand-written map; no plain-language lookup exists in-repo to reuse.
_NETWORK_NAMES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default Mode",
}
_SUBCORTEX_STRUCTURE_NAMES = {
    "HIP": "Hippocampus",
    "AMY": "Amygdala",
    "THA": "Thalamus",
    "NAc": "Nucleus Accumbens",
    "GP": "Globus Pallidus",
    "PUT": "Putamen",
    "CAU": "Caudate",
}
_HEMI_NAMES = {"lh": "Left", "rh": "Right", "LH": "Left", "RH": "Right"}


def _atlas_paths() -> tuple[Path, Path]:
    """Resolve the atlas dseg NIfTI + TSV paths from env (CAT12_ATLAS_ROOT, ATLAS_NAME)."""
    atlas_root = os.getenv("CAT12_ATLAS_ROOT")
    if not atlas_root:
        raise RuntimeError(
            "CAT12_ATLAS_ROOT not set. Point it at the dir containing "
            "atlas-<ATLAS_NAME>/ (see .env.example)."
        )
    atlas_name = os.getenv("ATLAS_NAME", "Schaefer2018N400n7Tian2020S2")
    atlas_dir = Path(atlas_root) / f"atlas-{atlas_name}"
    nii_path = atlas_dir / f"atlas-{atlas_name}_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz"
    tsv_path = atlas_dir / f"atlas-{atlas_name}_dseg.tsv"
    if not nii_path.exists():
        raise FileNotFoundError(nii_path)
    if not tsv_path.exists():
        raise FileNotFoundError(tsv_path)
    return nii_path, tsv_path


def compute_centroids(dseg_path: Path, tsv_path: Path) -> dict[str, dict[str, Any]]:
    """Compute per-region MNI-mm centroids from an atlas dseg volume.

    Returns a dict keyed by region name (matching the TSV `label` column and
    the BAG output's `region` column), each holding index/centroid/network/structure.
    """
    img = nib.load(dseg_path)
    data = img.get_fdata()
    tsv = pd.read_csv(tsv_path, sep="\t")

    labels = np.unique(data.astype(int))
    labels = labels[labels != 0]

    centroids: dict[str, dict[str, Any]] = {}
    for label_idx in labels:
        row = tsv.loc[tsv["index"] == label_idx]
        if row.empty:
            logger.warning("Atlas label %d has no TSV entry - skipping.", label_idx)
            continue
        row = row.iloc[0]
        com_voxel = center_of_mass(data == label_idx)
        mni = nib.affines.apply_affine(img.affine, com_voxel)
        structure = "subcortex" if row["atlas_name"].startswith("Tian") else "cortex"
        centroids[row["label"]] = {
            "index": int(label_idx),
            "centroid_mni": [round(float(c), 1) for c in mni],
            "network": None if pd.isna(row["network_label"]) else row["network_label"],
            "structure": structure,
        }
    return centroids


def _plain_language_name(region_name: str, network: str | None, structure: str) -> str:
    """Derive a readable name from the region label parts (no lookup table exists)."""
    if structure == "cortex":
        hemi_code, _, rest = region_name.partition("_")
        hemi = _HEMI_NAMES.get(hemi_code, hemi_code)
        net_name = _NETWORK_NAMES.get(network, network or rest)
        return f"{hemi} {net_name} ({region_name})"

    # Subcortical, e.g. "NAc-shell-lh" or "THA-DP-rh".
    parts = region_name.split("-")
    hemi = _HEMI_NAMES.get(parts[-1], parts[-1])
    struct = next((v for k, v in _SUBCORTEX_STRUCTURE_NAMES.items() if k in parts[0]), parts[0])
    return f"{hemi} {struct} ({region_name})"


def spot_check(centroids: dict[str, dict[str, Any]], n: int = 5) -> None:
    """Print a handful of centroids and save an all-region glass-brain PNG for eyeballing."""
    import matplotlib

    matplotlib.use("Agg")
    from nilearn import plotting

    names = list(centroids.keys())
    print(f"Spot-check ({len(names)} regions total):")
    for name in names[:n] + names[-n:]:
        print(f"  {name}: {centroids[name]}")

    coords = np.array([v["centroid_mni"] for v in centroids.values()])
    out_path = REFERENCE_DIR / "centroid_spotcheck.png"
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    display = plotting.plot_markers(
        node_values=np.zeros(len(coords)),
        node_coords=coords,
        node_size=15,
        display_mode="ortho",
        colorbar=False,
    )
    display.savefig(out_path)
    display.close()
    print(f"Wrote glass-brain spot-check PNG: {out_path}")


def build_centroids() -> dict[str, dict[str, Any]]:
    nii_path, tsv_path = _atlas_paths()
    centroids = compute_centroids(nii_path, tsv_path)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REFERENCE_DIR / "region_centroids.json"
    out_path.write_text(json.dumps(centroids, indent=2))
    logger.info("Wrote %d region centroids to %s", len(centroids), out_path)
    return centroids


def enrich_literature(
    centroids: dict[str, dict[str, Any]], radius_mm: float = 6.0
) -> dict[str, dict[str, Any]]:
    """Decode Neurosynth terms + contributing studies for each region centroid via NiMARE."""
    from nimare.extract import fetch_neurosynth
    from nimare.io import convert_neurosynth_to_dataset

    cache_dir = REFERENCE_DIR / "neurosynth"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = fetch_neurosynth(
        data_dir=str(cache_dir),
        version="7",
        overwrite=False,
        return_type="files",
        source="abstract",
        vocab="terms",
    )[0]
    dataset = convert_neurosynth_to_dataset(
        coordinates_file=files["coordinates"],
        metadata_file=files["metadata"],
        annotations_files=files["features"],
    )

    literature: dict[str, dict[str, Any]] = {}
    for region_name, info in centroids.items():
        x, y, z = info["centroid_mni"]
        ids = dataset.get_studies_by_coordinate([[x, y, z]], r=radius_mm)
        if not ids:
            logger.warning("No studies found near %s (%s) - leaving terms empty.", region_name, info["centroid_mni"])
            literature[region_name] = {"terms": [], "studies": []}
            continue

        sub_dataset = dataset.slice(ids)
        term_freqs = sub_dataset.annotations.filter(regex="^terms_abstract_tfidf__").mean(numeric_only=True)
        top_terms = term_freqs.sort_values(ascending=False).head(10)
        terms = [
            {"term": t.split("__", 1)[-1], "z": round(float(v), 3)} for t, v in top_terms.items() if v > 0
        ]
        studies = [
            {
                "pmid": str(sid).split("-")[0],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{str(sid).split('-')[0]}/",
            }
            for sid in ids[:10]
        ]
        literature[region_name] = {"terms": terms, "studies": studies}

    out_path = REFERENCE_DIR / "region_literature.json"
    out_path.write_text(json.dumps(literature, indent=2))
    logger.info("Wrote literature enrichment for %d regions to %s", len(literature), out_path)
    return literature


def bag_region_names() -> set[str]:
    """Region names present in the multivariate BAG output (the contract we must match)."""
    metrics = pd.read_parquet(BAG_REGION_METRICS)
    return set(metrics["region"])


@functools.lru_cache(maxsize=1)
def load_region_reference() -> dict[str, dict[str, Any]]:
    """Runtime (server-side) load of the built `region_reference.json`, keyed by region name."""
    return json.loads((APP_ASSETS_DIR / "region_reference.json").read_text())


def merge_reference(
    centroids: dict[str, dict[str, Any]], literature: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Merge centroids + literature + plain names -> region_reference.json.

    Fails loud (raises) if the resulting key set doesn't exactly match the BAG
    output's region names, per the app_kickoff.md build-time contract.
    """
    reference: dict[str, dict[str, Any]] = {}
    for name, c in centroids.items():
        reference[name] = {
            **c,
            "plain_name": _plain_language_name(name, c["network"], c["structure"]),
            **literature.get(name, {"terms": [], "studies": []}),
        }

    expected = bag_region_names()
    actual = set(reference.keys())
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"region_reference.json keys don't match BAG region names. "
            f"Missing from reference: {sorted(missing)}. Extra in reference: {sorted(extra)}."
        )

    APP_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = APP_ASSETS_DIR / "region_reference.json"
    out_path.write_text(json.dumps(reference, indent=2))
    logger.info("Wrote region_reference.json (%d regions, contract OK) to %s", len(reference), out_path)

    _copy_atlas_volume()
    return reference


def _copy_atlas_volume() -> None:
    """Copy the atlas dseg volume into app/assets/atlas/ so the shipped viewer has no
    external-path dependency (mirrors the external CAT12_ATLAS_ROOT/ATLAS_NAME location)."""
    import shutil

    nii_path, _ = _atlas_paths()
    atlas_dir = APP_ASSETS_DIR / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    dest = atlas_dir / nii_path.name
    shutil.copy2(nii_path, dest)
    logger.info("Copied atlas volume to %s", dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["centroids", "literature"])
    args = parser.parse_args()

    if args.step == "centroids":
        centroids = build_centroids()
        spot_check(centroids)
    else:
        centroids_path = REFERENCE_DIR / "region_centroids.json"
        if not centroids_path.exists():
            raise SystemExit("Run `centroids` step first.")
        centroids = json.loads(centroids_path.read_text())
        literature = enrich_literature(centroids)
        merge_reference(centroids, literature)


if __name__ == "__main__":
    main()
