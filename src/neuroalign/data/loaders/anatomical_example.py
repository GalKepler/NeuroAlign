"""
Command-line helper to aggregate CAT12 regional measures into pickles.

The notebook `age_cat12_new.ipynb` drives the same workflow interactively.
This script packages the key pieces into a repeatable CLI and adds
multi-processing for the heavy loading/parcellation step.

The CLI reads major paths from the project .env file (override with flags).

Example:
    python -m neuroaging.data.loaders.cat12 \\
        --sessions-csv ~/Downloads/linked_sessions.csv \\
        --atlas 4S456Parcels --jobs 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io
from parcellate.parcellation.volume import VolumetricParcellator
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from dotenv import load_dotenv

from neuroaging.data.loaders.utils import locate_anat_file, rsync_gunzip

CHAPTER_DIR = Path(__file__).parent
PROJ_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJ_ROOT / ".env")


@dataclass(frozen=True)
class Cat12Paths:
    sessions_csv: Path
    atlas_root: Path
    matlab_atlas_dir: Path
    dest_root: Path
    cat12_root: Path
    matlab_bin: Path
    spm_path: Path
    cat12_path: Path
    tiv_template: Path
    raw_data_dir: Path
    tmp_bids_dir: Path


def _env_path(name: str) -> Optional[Path]:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser()


def _resolve_required(
    arg_value: Optional[Path],
    env_value: Optional[Path],
    env_name: str,
    arg_name: str,
) -> Path:
    if arg_value is not None:
        return arg_value
    if env_value is not None:
        return env_value
    raise RuntimeError(f"Missing {env_name}; set it in .env or pass {arg_name}.")


@lru_cache
def _default_sessions_csv() -> Path:
    return Path("~/Downloads/linked_sessions.csv").expanduser()


def sanitize_subject_code(subject_code: Any) -> str:
    """Normalize subject codes used across the various CSV files."""
    return (
        str(subject_code)
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
    )


def clean_session_id(session_id: Any) -> str:
    """Normalize session id strings and strip separators."""
    return str(session_id).replace("_", "").replace("-", "").replace(" ", "")


def load_sessions(
    sessions_csv: Path,
    extras_root: Path,
) -> pd.DataFrame:
    """Load and merge all session sources used in the notebook."""
    dtype_map = {"subject_code": str, "session_id": str}
    sessions = pd.read_csv(sessions_csv, dtype=dtype_map)
    sessions = sessions.sort_values(["subject_code", "session_id"])
    sessions["session_id"] = sessions["session_id"].astype(str).str.zfill(12)
    sessions["subject_code"] = sessions["subject_code"].apply(sanitize_subject_code)
    sessions = sessions[sessions["subject_code"] != "0000"]

    def _load_if_exists(path: Path, loader) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return loader(path)

    def _load_legacy(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["source"] = "legacy"
        df["subject_code"] = df["subject_code"].apply(sanitize_subject_code)
        df["session_id"] = df["session_id"].astype(str)
        return df

    def _load_legacy_vip(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df["source"] = "legacy_vip"
        df["subject_id"] = df["ID "].astype(str).str.zfill(9)
        df["subject_code"] = df["subject_id"]
        df["session_id"] = df["session_id"].astype(str)
        return df

    def _load_legacy_repeated(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df["source"] = "legacy_repeated"
        df["session_id"] = df["session_id"].astype(str)
        df["subject_code"] = df["subject_code"].apply(sanitize_subject_code)
        return df

    def _load_cardiff(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["subject_code"] = df["full_name"].astype(str).apply(sanitize_subject_code)
        df["session_id"] = df["session_id"].astype(str)
        df["source"] = "cardiff"
        return df

    extras: List[pd.DataFrame] = [
        _load_if_exists(extras_root / "legacy_mprage.csv", _load_legacy),
        _load_if_exists(extras_root / "yaniv_vip_legacy_mprage.csv", _load_legacy_vip),
        _load_if_exists(extras_root / "repeated_mprage.csv", _load_legacy_repeated),
        _load_if_exists(extras_root / "cardiff.csv", _load_cardiff),
    ]
    for extra in extras:
        if not extra.empty:
            sessions = pd.concat([sessions, extra], ignore_index=True)

    sessions["session_id"] = sessions["session_id"].apply(clean_session_id)
    sessions["subject_code"] = sessions["subject_code"].apply(sanitize_subject_code)
    sessions = sessions.drop_duplicates(subset=["subject_code", "session_id"], keep="last")
    return sessions


def calculate_strip_score(
    input_file: Path,
    axis: int = 2,
    target_freq: float = 0.25,
    freq_tolerance: float = 0.05,
) -> float:
    """Calculate the striping score of a given NIfTI file along a given axis."""
    input_img = nib.load(str(input_file))
    data = input_img.get_fdata()  # type: ignore[attr-defined]

    mean_profile = np.mean(data, axis=tuple(i for i in range(data.ndim) if i != axis))

    n_voxels = len(mean_profile)
    yf = fft(mean_profile)
    xf = fftfreq(n_voxels, 1)[: n_voxels // 2]

    power_spectrum = 2.0 / n_voxels * np.abs(yf[: n_voxels // 2])

    freq_indices = np.where(
        (xf >= target_freq - freq_tolerance) & (xf <= target_freq + freq_tolerance)
    )[0]

    return float(np.sum(power_spectrum[freq_indices]))


def check_keys(dict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert mat_struct entries to dictionaries recursively."""
    from scipy.io.matlab.mio5_params import mat_struct

    for key in dict_data:
        if isinstance(dict_data[key], mat_struct):
            dict_data[key] = to_dict(dict_data[key])
    return dict_data


def to_dict(mat_obj: Any) -> Dict[str, Any]:
    """Recursive converter used by `check_keys`."""
    from scipy.io.matlab.mio5_params import mat_struct

    dict_data = {}
    for field_name in mat_obj._fieldnames:
        field_value = getattr(mat_obj, field_name)
        if isinstance(field_value, mat_struct):
            dict_data[field_name] = to_dict(field_value)
        elif isinstance(field_value, np.ndarray):
            dict_data[field_name] = parse_array(field_value)
        else:
            dict_data[field_name] = field_value
    return dict_data


def parse_array(array: np.ndarray) -> Any:
    """Helper to recursively convert arrays that contain mat_struct objects."""
    from scipy.io.matlab.mio5_params import mat_struct

    if array.size == 1:
        return to_dict(array.item()) if isinstance(array.item(), mat_struct) else array.item()
    return [to_dict(element) if isinstance(element, mat_struct) else element for element in array]


def load_quality_measures(file_path: Path) -> Dict[str, float]:
    """Extract CAT12 quality measures from the .mat file into a flat dict."""
    data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    data = check_keys(data)
    results: Dict[str, float] = {}
    s_block = data.get("S")
    if isinstance(s_block, dict):
        for main_key in ["qualitymeasures", "qualityratings"]:
            main_s = s_block.get(main_key)
            if not main_s:
                continue
            for key, value in main_s.items():
                if isinstance(value, (int, float, np.floating)):
                    results[f"{main_key}_{key}"] = float(value)
    return results


def get_t1w_file(
    subject: str, session: str, raw_data_dir: Path, tmp_bids_dir: Path
) -> Optional[Path]:
    """Fetch T1w file into the TMP_BIDS_DIRECTORY if it is missing."""
    bids_dir = raw_data_dir / f"sub-{subject}" / f"ses-{session}" / "anat"
    session_tmp_dir = tmp_bids_dir / f"sub-{subject}" / f"ses-{session}" / "anat"
    session_tmp_dir.mkdir(parents=True, exist_ok=True)
    t1w_file = (
        locate_anat_file(bids_dir)
        if "legacy" not in subject
        else locate_anat_file(session_tmp_dir, extension="nii")
    )
    if t1w_file is not None:
        target_t1w_file = session_tmp_dir / t1w_file.name.replace(".nii.gz", ".nii")
        if not target_t1w_file.exists():
            rsync_gunzip(t1w_file, session_tmp_dir)
        return target_t1w_file
    return None


def get_regional_volumes(
    cat12_directory: Path, parcellator: VolumetricParcellator
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load CAT12 probability maps and parcellate to regional measures."""
    gm_prob = list(cat12_directory.glob("mwp1*.nii"))
    wm_prob = list(cat12_directory.glob("mwp2*.nii"))
    ct_prob = list(cat12_directory.glob("wct*.nii"))
    gm_volumes = parcellator.transform(gm_prob[0]) if gm_prob else None
    wm_volumes = parcellator.transform(wm_prob[0]) if wm_prob else None
    ct = parcellator.transform(ct_prob[0]) if ct_prob else None
    return gm_volumes, wm_volumes, ct


def find_example_cat12_file(sessions: pd.DataFrame, cat12_root: Path) -> Path:
    """Locate one CAT12 mwp1 file to fit the VolumetricParcellator template."""
    for _, row in sessions.iterrows():
        subject = row["subject_code"]
        session = row["session_id"]
        candidate_dir = cat12_root / f"sub-{subject}" / f"ses-{session}" / "anat"
        gm_files = sorted(candidate_dir.glob("mwp1*.nii"))
        if gm_files:
            return gm_files[0]
    raise FileNotFoundError(
        f"No CAT12 outputs found under {cat12_root}. "
        "Run CAT12 first or provide a different --cat12-root."
    )


def build_parcellator(
    atlas_img: Path, parcels_path: Path, example_cat12: Path
) -> VolumetricParcellator:
    parcels = pd.read_csv(parcels_path, sep="\t")
    vp = VolumetricParcellator(atlas_img=atlas_img, lut=parcels, mask="gm")
    vp.fit(str(example_cat12))
    return vp


def build_template_df(
    sessions: pd.DataFrame, sample_volumes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create empty gm/wm/ct DataFrames with the expected columns."""
    template_columns = sessions.columns.tolist() + sample_volumes.columns.tolist() + ["strip_score"]
    gm = pd.DataFrame(columns=template_columns)
    gm["metric"] = "volume"
    gm["tissue"] = "gm"

    wm = pd.DataFrame(columns=template_columns)
    wm["metric"] = "volume"
    wm["tissue"] = "wm"

    ct = pd.DataFrame(columns=template_columns)
    ct["metric"] = "thickness"
    ct["tissue"] = "ct"
    return gm, wm, ct


_VP: Optional[VolumetricParcellator] = None


def _init_worker(atlas_img: Path, parcels_path: Path, example_cat12: Path) -> None:
    """Initializer for worker processes to build their own parcellator."""
    global _VP
    _VP = build_parcellator(atlas_img, parcels_path, example_cat12)


def _select_xml(cat12_dir: Path, subject: str, session: str) -> Optional[Path]:
    """Find the CAT12 XML needed for TIV calculation."""
    existing = sorted(cat12_dir.glob("cat_*sub-*.xml"))
    if existing:
        return existing[0]
    corrected = cat12_dir / f"cat_sub-{subject}_ses-{session}_ce-corrected_T1w.xml"
    if corrected.exists():
        return corrected
    uncorrected = cat12_dir / f"cat_sub-{subject}_ses-{session}_ce-uncorrected_T1w.xml"
    return uncorrected if uncorrected.exists() else None


def _process_session(
    row: Dict[str, Any],
    cat12_root: Path,
    raw_data_dir: Path,
    tmp_bids_dir: Path,
    include_strip_score: bool = True,
) -> Dict[str, Any]:
    """Worker entrypoint: load CAT12 outputs, QC, and strip score."""
    assert _VP is not None, "Parcellator not initialised in worker."

    subject = row["subject_code"]
    session = row["session_id"]
    cat12_dir = cat12_root / f"sub-{subject}" / f"ses-{session}" / "anat"
    if not cat12_dir.exists():
        t1w = get_t1w_file(subject, session, raw_data_dir, tmp_bids_dir)
        return {
            "status": "missing_cat12",
            "subject": subject,
            "session": session,
            "t1w": str(t1w) if t1w else None,
        }

    gm_volumes, wm_volumes, ct = get_regional_volumes(cat12_dir, _VP)
    if all(v is None for v in [gm_volumes, wm_volumes, ct]):
        return {
            "status": "missing_prob_maps",
            "subject": subject,
            "session": session,
        }

    qc = {}
    qc_files = list(cat12_dir.glob("cat_*.mat"))
    if qc_files:
        qc = load_quality_measures(qc_files[0])

    strip_score: Optional[float] = None
    if include_strip_score:
        t1w_file = get_t1w_file(subject, session, raw_data_dir, tmp_bids_dir)
        if t1w_file:
            strip_score = calculate_strip_score(t1w_file)

    payload: Dict[str, pd.DataFrame] = {}
    for label, df in (("gm", gm_volumes), ("wm", wm_volumes), ("ct", ct)):
        if df is None:
            continue
        out = df.copy()
        for key, value in row.items():
            out[key] = value
        for key, value in qc.items():
            out[key] = value
        if strip_score is not None:
            out["strip_score"] = strip_score
        out["tissue"] = label
        out["metric"] = "volume" if label in ("gm", "wm") else "thickness"
        payload[label] = out

    xml_file = _select_xml(cat12_dir, subject, session)

    return {
        "status": "success",
        "data": payload,
        "xml": str(xml_file) if xml_file else None,
        "subject": subject,
        "session": session,
    }


def filter_sessions(
    sessions: pd.DataFrame, existing_index: Iterable[Tuple[str, str]]
) -> pd.DataFrame:
    """Drop sessions that are already present unless overwrite is requested."""
    existing = set(existing_index)
    mask = ~sessions.apply(lambda r: (r["subject_code"], r["session_id"]) in existing, axis=1)
    return sessions.loc[mask]


def resolve_paths(args: argparse.Namespace) -> Cat12Paths:
    """Resolve loader paths from CLI args and .env."""
    sessions_csv = args.sessions_csv or _env_path("CAT12_SESSIONS_CSV") or _default_sessions_csv()
    atlas_root = _resolve_required(
        args.atlas_root, _env_path("CAT12_ATLAS_ROOT"), "CAT12_ATLAS_ROOT", "--atlas-root"
    )
    matlab_atlas_dir = _resolve_required(
        args.matlab_atlas_dir,
        _env_path("CAT12_MATLAB_ATLAS_DIR"),
        "CAT12_MATLAB_ATLAS_DIR",
        "--matlab-atlas-dir",
    )
    dest_root = _resolve_required(
        args.dest_root, _env_path("CAT12_DEST_ROOT"), "CAT12_DEST_ROOT", "--dest-root"
    )
    raw_data_dir = _resolve_required(
        args.raw_data_dir, _env_path("CAT12_RAW_DATA_DIR"), "CAT12_RAW_DATA_DIR", "--raw-data-dir"
    )
    tmp_bids_dir = _resolve_required(
        args.tmp_bids_dir, _env_path("CAT12_TMP_BIDS_DIR"), "CAT12_TMP_BIDS_DIR", "--tmp-bids-dir"
    )
    cat12_root = (
        args.cat12_root
        or _env_path("CAT12_ROOT")
        or (tmp_bids_dir / "derivatives" / "CAT12.9_2577.new")
    )
    tiv_template = (
        args.tiv_template or _env_path("CAT12_TIV_TEMPLATE") or (PROJ_ROOT / "cat12_tiv_template.m")
    )
    matlab_bin = _resolve_required(
        args.matlab_bin, _env_path("CAT12_MATLAB_BIN"), "CAT12_MATLAB_BIN", "--matlab-bin"
    )
    spm_path = _resolve_required(
        args.spm_path, _env_path("CAT12_SPM_PATH"), "CAT12_SPM_PATH", "--spm-path"
    )
    cat12_path = _resolve_required(
        args.cat12_path, _env_path("CAT12_PATH"), "CAT12_PATH", "--cat12-path"
    )
    return Cat12Paths(
        sessions_csv=sessions_csv,
        atlas_root=atlas_root,
        matlab_atlas_dir=matlab_atlas_dir,
        dest_root=dest_root,
        cat12_root=cat12_root,
        matlab_bin=matlab_bin,
        spm_path=spm_path,
        cat12_path=cat12_path,
        tiv_template=tiv_template,
        raw_data_dir=raw_data_dir,
        tmp_bids_dir=tmp_bids_dir,
    )


def main() -> None:
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-csv", type=Path, default=None)
    parser.add_argument("--atlas", type=str, default="4S456Parcels")
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=None,
        help="Base directory holding atlas-<name> folders (CAT12_ATLAS_ROOT).",
    )
    parser.add_argument(
        "--matlab-atlas-dir",
        type=Path,
        default=None,
        help="Directory containing atlas-<name>_dseg.tsv files (CAT12_MATLAB_ATLAS_DIR).",
    )
    parser.add_argument(
        "--cat12-root",
        type=Path,
        default=None,
        help="CAT12 derivatives root (CAT12_ROOT or derived from CAT12_TMP_BIDS_DIR).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="Destination root for gm/wm/ct pickles (CAT12_DEST_ROOT).",
    )
    parser.add_argument(
        "--tiv-template",
        type=Path,
        default=None,
        help="Path to cat12_tiv_template.m (CAT12_TIV_TEMPLATE).",
    )
    parser.add_argument(
        "--matlab-bin",
        type=Path,
        default=None,
        help="MATLAB executable for running CAT12 TIV (CAT12_MATLAB_BIN).",
    )
    parser.add_argument(
        "--spm-path",
        type=Path,
        default=None,
        help="SPM path to add inside MATLAB (CAT12_SPM_PATH).",
    )
    parser.add_argument(
        "--cat12-path",
        type=Path,
        default=None,
        help="CAT12 path to add inside MATLAB (CAT12_PATH).",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Raw BIDS root for T1w fetching (CAT12_RAW_DATA_DIR).",
    )
    parser.add_argument(
        "--tmp-bids-dir",
        type=Path,
        default=None,
        help="Temporary BIDS root for staging T1w files (CAT12_TMP_BIDS_DIR).",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Worker processes to use.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing pickles and rebuild from scratch.",
    )
    parser.add_argument(
        "--skip-strip-score",
        action="store_true",
        help="Do not compute strip score (saves time, skips FFT).",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=None,
        help="Optional path to write missing CAT12/T1w sessions.",
    )
    args = parser.parse_args()

    paths = resolve_paths(args)

    print("Execution paths:")
    print(paths)

    sessions = load_sessions(paths.sessions_csv, CHAPTER_DIR)
    print(f"Loaded {len(sessions)} session rows")

    atlas_img = (
        paths.atlas_root
        / f"atlas-{args.atlas}"
        / f"atlas-{args.atlas}_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz"
    )
    parcels_path = paths.matlab_atlas_dir / f"atlas-{args.atlas}_dseg.tsv"
    dest_dir = paths.dest_root / f"atlas-{args.atlas}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    example_cat12 = find_example_cat12_file(sessions, paths.cat12_root)
    parcellator = build_parcellator(atlas_img, parcels_path, example_cat12)
    sample_gm, _, _ = get_regional_volumes(example_cat12.parent, parcellator)
    if sample_gm is None:
        raise RuntimeError(f"Could not transform sample CAT12 file {example_cat12}")

    gm_vol_df_path = dest_dir / "gm_vol_df.pkl"
    wm_vol_df_path = dest_dir / "wm_vol_df.pkl"
    ct_df_path = dest_dir / "ct_df.pkl"

    if args.overwrite:
        gm_vol_df, wm_vol_df, ct_df = build_template_df(sessions, sample_gm)
    else:
        gm_vol_df = (
            pd.read_pickle(gm_vol_df_path)
            if gm_vol_df_path.exists()
            else build_template_df(sessions, sample_gm)[0]
        )
        wm_vol_df = (
            pd.read_pickle(wm_vol_df_path)
            if wm_vol_df_path.exists()
            else build_template_df(sessions, sample_gm)[1]
        )
        ct_df = (
            pd.read_pickle(ct_df_path)
            if ct_df_path.exists()
            else build_template_df(sessions, sample_gm)[2]
        )

    existing_idx = zip(gm_vol_df.get("subject_code", []), gm_vol_df.get("session_id", []))
    sessions_to_process = sessions if args.overwrite else filter_sessions(sessions, existing_idx)
    print(f"Processing {len(sessions_to_process)} sessions")

    results_gm: List[pd.DataFrame] = []
    results_wm: List[pd.DataFrame] = []
    results_ct: List[pd.DataFrame] = []
    missing: List[Tuple[str, str, Optional[str]]] = []
    errors: List[Tuple[str, str, str]] = []
    xmls: List[Tuple[str, str, str]] = []

    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(atlas_img, parcels_path, example_cat12),
    ) as pool:
        futures = [
            pool.submit(
                _process_session,
                row._asdict() if hasattr(row, "_asdict") else row.to_dict(),
                paths.cat12_root,
                paths.raw_data_dir,
                paths.tmp_bids_dir,
                not args.skip_strip_score,
            )
            for _, row in sessions_to_process.iterrows()
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CAT12"):
            res = fut.result()
            status = res.get("status")
            if status == "success":
                data = res["data"]
                if "gm" in data:
                    results_gm.append(data["gm"])
                if "wm" in data:
                    results_wm.append(data["wm"])
                if "ct" in data:
                    results_ct.append(data["ct"])
                xml_path = res.get("xml")
                if xml_path:
                    xmls.append((res.get("subject", ""), res.get("session", ""), xml_path))
            elif status == "missing_cat12":
                missing.append((res["subject"], res["session"], res.get("t1w")))
            elif status:
                errors.append((res.get("subject", ""), res.get("session", ""), status))

    if results_gm:
        gm_vol_df = pd.concat([gm_vol_df] + results_gm, ignore_index=True)
    if results_wm:
        wm_vol_df = pd.concat([wm_vol_df] + results_wm, ignore_index=True)
    if results_ct:
        ct_df = pd.concat([ct_df] + results_ct, ignore_index=True)

    gm_vol_df.to_pickle(gm_vol_df_path)
    wm_vol_df.to_pickle(wm_vol_df_path)
    ct_df.to_pickle(ct_df_path)

    print(f"Saved gm_vol_df to {gm_vol_df_path}")
    print(f"Saved wm_vol_df to {wm_vol_df_path}")
    print(f"Saved ct_df to {ct_df_path}")

    if args.missing_out and missing:
        args.missing_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.missing_out, "w") as f:
            for subj, sess, t1w in missing:
                f.write(f"{subj},{sess},{t1w or 'missing T1w'}\n")
        print(f"Wrote missing CAT12/T1w list to {args.missing_out}")

    if errors:
        print("The following sessions had issues:")
        for subj, sess, msg in errors:
            print(f"  {subj} {sess}: {msg}")

    # Calculate TIV via MATLAB if XMLs are available
    if xmls:
        tiv_out = dest_dir / "TIV.txt"
        filled_template = dest_dir / "cat12_tiv.m"

        xml_lines = "\n".join([f"'{xml_path}'" for _, _, xml_path in xmls])
        template_text = paths.tiv_template.read_text()
        filled_text = template_text.replace("$XMLS", xml_lines).replace("$OUT_FILE", str(tiv_out))
        filled_template.write_text(filled_text)

        cmd = " ".join(
            [
                str(paths.matlab_bin),
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                '"',
                f"addpath {paths.spm_path} {paths.cat12_path};",
                f"try, run('{filled_template}'); catch; end; exit;",
                '"',
            ]
        )
        print(f"Running MATLAB for TIV: {cmd}")
        subprocess.run(cmd, shell=True, check=False)

        if tiv_out.exists():
            tiv_data = pd.read_csv(tiv_out, header=None).values.flatten()
            for df in [gm_vol_df, wm_vol_df, ct_df]:
                for (subj, sess, _), tiv in zip(xmls, tiv_data):
                    mask = (df["subject_code"] == subj) & (df["session_id"] == sess)
                    df.loc[mask, "tiv"] = tiv
            gm_vol_df.to_pickle(gm_vol_df_path)
            wm_vol_df.to_pickle(wm_vol_df_path)
            ct_df.to_pickle(ct_df_path)
            print(f"Updated TIV in pickles using {tiv_out}")


if __name__ == "__main__":
    main()
