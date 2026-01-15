"""
Anatomical MRI Data Loader for NeuroAlign
==========================================

Loads and parcellates CAT12-processed anatomical data (GM volume, cortical thickness).
Uses multiprocessing for efficient batch loading.
Supports TIV (Total Intracranial Volume) calculation and volume normalization.

Example:
    >>> from neuroalign.data.loaders import AnatomicalLoader
    >>> loader = AnatomicalLoader(
    ...     cat12_root="/path/to/cat12/derivatives",
    ...     atlas_root="/path/to/atlases",
    ...     atlas_name="4S456Parcels"
    ... )
    >>> df = loader.load_sessions(
    ...     sessions_csv="linked_sessions.csv",
    ...     n_jobs=8,
    ...     normalize_by_tiv=True
    ... )
"""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io

try:
    from parcellate.parcellation.volume import VolumetricParcellator

    PARCELLATE_AVAILABLE = True
except ImportError:
    PARCELLATE_AVAILABLE = False
    VolumetricParcellator = None

logger = logging.getLogger(__name__)


@dataclass
class AnatomicalPaths:
    """Configuration for anatomical data paths."""

    cat12_root: Path
    atlas_root: Path
    atlas_name: str = "4S456Parcels"

    @property
    def atlas_img(self) -> Path:
        return (
            self.atlas_root
            / f"atlas-{self.atlas_name}"
            / f"atlas-{self.atlas_name}_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz"
        )

    @property
    def parcels_tsv(self) -> Path:
        return self.atlas_root / f"atlas-{self.atlas_name}" / f"atlas-{self.atlas_name}_dseg.tsv"


@dataclass
class TIVConfig:
    """Configuration for TIV calculation via MATLAB."""

    matlab_bin: Optional[Path] = None
    spm_path: Optional[Path] = None
    cat12_path: Optional[Path] = None
    tiv_template: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "TIVConfig":
        """Load TIV configuration from environment variables."""

        def _get_path(name: str) -> Optional[Path]:
            val = os.getenv(name)
            return Path(val).expanduser() if val else None

        return cls(
            matlab_bin=_get_path("MATLAB_BIN"),
            spm_path=_get_path("SPM_PATH"),
            cat12_path=_get_path("CAT12_PATH"),
            tiv_template=_get_path("TIV_TEMPLATE"),
        )

    def is_available(self) -> bool:
        """Check if all required paths are configured."""
        return all(
            [
                self.matlab_bin and self.matlab_bin.exists(),
                self.spm_path and self.spm_path.exists(),
                self.cat12_path and self.cat12_path.exists(),
                self.tiv_template and self.tiv_template.exists(),
            ]
        )

    @property
    def enabled(self) -> bool:
        """Alias for is_available() for clearer API."""
        return self.is_available()


# Global parcellator for multiprocessing workers
_WORKER_PARCELLATOR: Optional[VolumetricParcellator] = None


def _init_worker(atlas_img: str, parcels_path: str, example_file: str) -> None:
    """Initialize parcellator in worker process."""
    global _WORKER_PARCELLATOR
    parcels = pd.read_csv(parcels_path, sep="\t")
    _WORKER_PARCELLATOR = VolumetricParcellator(atlas_img=atlas_img, lut=parcels, mask="gm")
    _WORKER_PARCELLATOR.fit(example_file)


def _get_regional_volumes(
    cat12_directory: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load CAT12 probability maps and parcellate to regional measures."""
    assert _WORKER_PARCELLATOR is not None, "Parcellator not initialized in worker"

    gm_prob = list(cat12_directory.glob("mwp1*.nii"))
    wm_prob = list(cat12_directory.glob("mwp2*.nii"))
    ct_prob = list(cat12_directory.glob("wct*.nii"))

    gm_volumes = _WORKER_PARCELLATOR.transform(gm_prob[0]) if gm_prob else None
    wm_volumes = _WORKER_PARCELLATOR.transform(wm_prob[0]) if wm_prob else None
    ct = _WORKER_PARCELLATOR.transform(ct_prob[0]) if ct_prob else None

    return gm_volumes, wm_volumes, ct


def _check_keys(dict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert mat_struct entries to dictionaries recursively."""
    try:
        from scipy.io.matlab.mio5_params import mat_struct
    except ImportError:
        return dict_data

    for key in dict_data:
        if isinstance(dict_data[key], mat_struct):
            dict_data[key] = _to_dict(dict_data[key])
    return dict_data


def _to_dict(mat_obj: Any) -> Dict[str, Any]:
    """Recursive converter for mat_struct objects."""
    try:
        from scipy.io.matlab.mio5_params import mat_struct
    except ImportError:
        return {}

    dict_data = {}
    for field_name in mat_obj._fieldnames:
        field_value = getattr(mat_obj, field_name)
        if isinstance(field_value, mat_struct):
            dict_data[field_name] = _to_dict(field_value)
        elif isinstance(field_value, np.ndarray):
            dict_data[field_name] = _parse_array(field_value)
        else:
            dict_data[field_name] = field_value
    return dict_data


def _parse_array(array: np.ndarray) -> Any:
    """Helper to recursively convert arrays containing mat_struct objects."""
    try:
        from scipy.io.matlab.mio5_params import mat_struct
    except ImportError:
        return array

    if array.size == 1:
        item = array.item()
        return _to_dict(item) if isinstance(item, mat_struct) else item
    return [_to_dict(elem) if isinstance(elem, mat_struct) else elem for elem in array]


def _load_quality_measures(file_path: Path) -> Dict[str, float]:
    """Extract CAT12 quality measures from .mat file into a flat dict."""
    try:
        data = scipy.io.loadmat(str(file_path), struct_as_record=False, squeeze_me=True)
        data = _check_keys(data)
    except Exception:
        return {}

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


def _process_session(
    row: Dict[str, Any],
    cat12_root: Path,
    include_qc: bool = True,
) -> Dict[str, Any]:
    """Worker function: load CAT12 outputs for a single session."""
    subject = row["subject_code"]
    session = row["session_id"]
    cat12_dir = cat12_root / f"sub-{subject}" / f"ses-{session}" / "anat"

    if not cat12_dir.exists():
        return {
            "status": "missing_cat12",
            "subject": subject,
            "session": session,
        }

    gm_volumes, wm_volumes, ct = _get_regional_volumes(cat12_dir)
    if all(v is None for v in [gm_volumes, wm_volumes, ct]):
        return {
            "status": "missing_prob_maps",
            "subject": subject,
            "session": session,
        }

    # Load QC measures if requested
    qc: Dict[str, float] = {}
    if include_qc:
        qc_files = list(cat12_dir.glob("cat_*.mat"))
        if qc_files:
            qc = _load_quality_measures(qc_files[0])

    # Find XML file for TIV calculation
    xml_file = _select_xml(cat12_dir, subject, session)

    # Build output DataFrames
    payload: Dict[str, pd.DataFrame] = {}
    for label, df in (("gm", gm_volumes), ("wm", wm_volumes), ("ct", ct)):
        if df is None:
            continue
        out = df.copy()
        for key, value in row.items():
            out[key] = value
        for key, value in qc.items():
            out[key] = value
        out["modality"] = label
        out["metric"] = "volume" if label in ("gm", "wm") else "thickness"
        payload[label] = out

    return {
        "status": "success",
        "data": payload,
        "xml": str(xml_file) if xml_file else None,
        "subject": subject,
        "session": session,
    }


def _select_xml(cat12_dir: Path, subject: str, session: str) -> Optional[Path]:
    """Find the CAT12 XML needed for TIV calculation."""
    corrected = cat12_dir / f"cat_sub-{subject}_ses-{session}_ce-corrected_T1w.xml"
    if corrected.exists():
        return corrected
    uncorrected = cat12_dir / f"cat_sub-{subject}_ses-{session}_ce-uncorrected_T1w.xml"
    return uncorrected if uncorrected.exists() else None


class AnatomicalLoader:
    """
    Loader for CAT12-processed anatomical MRI data.

    Extracts regional volumes (GM, WM) and cortical thickness from
    preprocessed probability maps using volumetric parcellation.
    Uses multiprocessing for efficient batch loading.
    Supports TIV normalization for volumetric measures.

    Attributes:
        paths: AnatomicalPaths configuration
        tiv_config: TIVConfig for MATLAB-based TIV calculation
        parcellator: VolumetricParcellator instance (fitted on first use)
    """

    def __init__(
        self,
        cat12_root: Path,
        atlas_root: Path,
        atlas_name: str = "4S456Parcels",
        tiv_config: Optional[TIVConfig] = None,
    ):
        """
        Initialize anatomical data loader.

        Args:
            cat12_root: Path to CAT12 derivatives directory
            atlas_root: Path to atlas directory
            atlas_name: Name of parcellation atlas to use
            tiv_config: TIV calculation configuration (loads from env if None)
        """
        if not PARCELLATE_AVAILABLE:
            raise ImportError("parcellate package required. Install with: uv add parcellate")

        self.paths = AnatomicalPaths(
            cat12_root=Path(cat12_root),
            atlas_root=Path(atlas_root),
            atlas_name=atlas_name,
        )
        self.tiv_config = tiv_config or TIVConfig.from_env()
        self.parcellator: Optional[VolumetricParcellator] = None
        self._parcels: Optional[pd.DataFrame] = None
        self._example_file: Optional[Path] = None
        self._xml_files: List[Tuple[str, str, str]] = []

    def _init_parcellator(self, example_file: Path) -> None:
        """Initialize and fit the parcellator on an example file."""
        if self.parcellator is not None:
            return

        parcels = pd.read_csv(self.paths.parcels_tsv, sep="\t")
        self.parcellator = VolumetricParcellator(
            atlas_img=str(self.paths.atlas_img), lut=parcels, mask="gm"
        )
        self.parcellator.fit(str(example_file))
        self._parcels = parcels
        self._example_file = example_file

    def _find_example_file(self, sessions: pd.DataFrame) -> Path:
        """Find an example CAT12 file to fit the parcellator."""
        for _, row in sessions.iterrows():
            subject = row["subject_code"]
            session = row["session_id"]
            if isinstance(session, float) or isinstance(session, int):
                session = str(int(session))
            if isinstance(subject, float) or isinstance(subject, int):
                subject = (
                    str(int(subject)).replace("_", "").replace("-", "").replace("\t", "").zfill(4)
                )
            candidate_dir = self.paths.cat12_root / f"sub-{subject}" / f"ses-{session}" / "anat"
            gm_files = sorted(candidate_dir.glob("mwp1*.nii"))
            if gm_files:
                return gm_files[0]
        raise FileNotFoundError(
            f"No CAT12 outputs found under {self.paths.cat12_root}. "
            "Run CAT12 first or check the path."
        )

    def get_cat12_directory(self, subject: str, session: str) -> Optional[Path]:
        """Get CAT12 output directory for a subject/session."""
        cat12_dir = self.paths.cat12_root / f"sub-{subject}" / f"ses-{session}" / "anat"
        return cat12_dir if cat12_dir.exists() else None

    def get_regional_volumes(
        self, cat12_directory: Path
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Extract regional volumes from CAT12 probability maps."""
        if self.parcellator is None:
            gm_example = list(cat12_directory.glob("mwp1*.nii"))
            if not gm_example:
                raise RuntimeError(f"No GM probability maps found in {cat12_directory}.")
            self._init_parcellator(gm_example[0])

        gm_prob = list(cat12_directory.glob("mwp1*.nii"))
        wm_prob = list(cat12_directory.glob("mwp2*.nii"))
        ct_prob = list(cat12_directory.glob("wct*.nii"))

        gm_volumes = self.parcellator.transform(gm_prob[0]) if gm_prob else None
        wm_volumes = self.parcellator.transform(wm_prob[0]) if wm_prob else None
        ct = self.parcellator.transform(ct_prob[0]) if ct_prob else None

        return gm_volumes, wm_volumes, ct

    def load_session(
        self, subject: str, session: str, include_metadata: bool = True, include_tiv: bool = True,
        tiv_output_dir: Path = None
    ) -> Optional[pd.DataFrame]:
        """Load all anatomical data for a single session."""
        cat12_dir = self.get_cat12_directory(subject, session)
        if cat12_dir is None:
            return None

        gm_vol, wm_vol, ct = self.get_regional_volumes(cat12_dir)

        dfs = []
        for modality, df in [("gm", gm_vol), ("wm", wm_vol), ("ct", ct)]:
            if df is None:
                continue
            df = df.copy()
            df["modality"] = modality
            df["metric"] = "volume" if modality in ("gm", "wm") else "thickness"
            if include_metadata:
                df["subject_code"] = subject
                df["session_id"] = session
            dfs.append(df)

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        if include_tiv:
            xml_file = _select_xml(cat12_dir, subject, session)
            if xml_file:
                self._xml_files.append((subject, session, str(xml_file)))
                if not tiv_output_dir:
                    import tempfile

                    output_dir = Path(tempfile.mkdtemp(prefix="neuroalign_tiv_"))
                else:
                    output_dir = Path(tiv_output_dir)
                tiv_df = self._calculate_tiv(output_dir)
                if tiv_df is None or tiv_df.empty:
                    logger.warning("TIV calculation failed - normalization skipped")
                    return df

                # Merge TIV into data
                df = df.merge(
                    tiv_df[["subject_code", "session_id", "tiv"]],
                    on=["subject_code", "session_id"],
                    how="left",
                )
        return df

    def load_sessions(
        self,
        sessions_csv: Path,
        n_jobs: int = 1,
        progress: bool = True,
        include_qc: bool = True,
        normalize_by_tiv: bool = True,
        tiv_output_dir: Optional[Path] = None,
        calculate_tiv: bool = True,
    ) -> pd.DataFrame:
        """
        Load anatomical data for multiple sessions.

        Uses multiprocessing for efficient parallel loading when n_jobs > 1.
        Optionally calculates TIV and normalizes volume measures.

        Args:
            sessions_csv: Path to CSV with 'subject_code' and 'session_id' columns
            n_jobs: Number of parallel workers (default: 1 for serial processing)
            progress: Whether to show progress bar
            include_qc: Whether to include CAT12 quality measures
            normalize_by_tiv: Whether to normalize volumes by TIV (requires MATLAB)
            tiv_output_dir: Directory for TIV calculation files (uses temp if None)
            calculate_tiv: Whether to calculate TIV (adds 'tiv' column even without normalization)

        Returns:
            DataFrame with all sessions' regional features
        """
        sessions = pd.read_csv(sessions_csv, dtype={"subject_code": str, "session_id": str})

        # Find example file for parcellator initialization
        example_file = self._find_example_file(sessions)
        self._init_parcellator(example_file)

        # Load data
        if n_jobs == 1:
            df = self._load_sessions_serial(sessions, progress, include_qc)
        else:
            df = self._load_sessions_parallel(sessions, n_jobs, progress, include_qc)

        # Calculate TIV (always if calculate_tiv=True, normalize only if normalize_by_tiv=True)
        if calculate_tiv or normalize_by_tiv:
            df = self._apply_tiv(df, tiv_output_dir, normalize=normalize_by_tiv)

        return df

    def _load_sessions_serial(
        self,
        sessions: pd.DataFrame,
        progress: bool,
        include_qc: bool,
    ) -> pd.DataFrame:
        """Serial loading (original behavior)."""
        results = []
        self._xml_files = []
        iterator = sessions.iterrows()

        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(sessions), desc="Loading anatomical data")
            except ImportError:
                pass

        for _, row in iterator:
            subject = row["subject_code"]
            session = row["session_id"]
            # Don't calculate TIV per-session - batch calculation happens in _apply_tiv()
            session_data = self.load_session(subject=subject, session=session, include_tiv=False)
            if session_data is not None:
                for col in sessions.columns:
                    if col not in session_data.columns:
                        session_data[col] = row[col]
                results.append(session_data)

                # Track XML file for batch TIV calculation later
                cat12_dir = self.get_cat12_directory(subject, session)
                if cat12_dir:
                    xml_file = _select_xml(cat12_dir, subject, session)
                    if xml_file:
                        self._xml_files.append((subject, session, str(xml_file)))

        if not results:
            raise ValueError("No sessions successfully loaded")

        return pd.concat(results, ignore_index=True)

    def _load_sessions_parallel(
        self,
        sessions: pd.DataFrame,
        n_jobs: int,
        progress: bool,
        include_qc: bool,
    ) -> pd.DataFrame:
        """Parallel loading using ProcessPoolExecutor."""
        warnings.filterwarnings("ignore")

        results_gm: List[pd.DataFrame] = []
        results_wm: List[pd.DataFrame] = []
        results_ct: List[pd.DataFrame] = []
        self._xml_files = []

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(
                str(self.paths.atlas_img),
                str(self.paths.parcels_tsv),
                str(self._example_file),
            ),
        ) as pool:
            futures = [
                pool.submit(
                    _process_session,
                    row.to_dict(),
                    self.paths.cat12_root,
                    include_qc,
                )
                for _, row in sessions.iterrows()
            ]

            iterator = as_completed(futures)
            if progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        iterator,
                        total=len(futures),
                        desc="Loading anatomical data",
                    )
                except ImportError:
                    pass

            for fut in iterator:
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
                        self._xml_files.append((res["subject"], res["session"], xml_path))

        all_results = results_gm + results_wm + results_ct
        if not all_results:
            raise ValueError("No sessions successfully loaded")

        return pd.concat(all_results, ignore_index=True)

    def _apply_tiv(
        self,
        df: pd.DataFrame,
        output_dir: Optional[Path] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Calculate TIV and optionally normalize volume measures.

        Args:
            df: DataFrame with anatomical data
            output_dir: Directory for TIV calculation files (uses temp if None)
            normalize: Whether to normalize volumes by TIV (default: True)

        Returns:
            DataFrame with 'tiv' column added (and optionally normalized volumes)
        """
        import tempfile

        if not self._xml_files:
            logger.warning("No XML files collected - TIV calculation skipped")
            return df

        if not self.tiv_config.is_available():
            logger.warning(
                "MATLAB/CAT12 not configured - TIV calculation skipped. "
                "Set MATLAB_BIN, SPM_PATH, CAT12_PATH, TIV_TEMPLATE in .env"
            )
            return df

        logger.info(f"Calculating TIV for {len(self._xml_files)} sessions...")

        # Use context manager for temp directory to ensure proper cleanup
        if output_dir is None:
            with tempfile.TemporaryDirectory(prefix="neuroalign_tiv_") as tmp_dir:
                tiv_df = self._calculate_tiv(Path(tmp_dir))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tiv_df = self._calculate_tiv(output_dir)

        if tiv_df is None or tiv_df.empty:
            logger.warning("TIV calculation failed")
            return df

        # Merge TIV into data
        df = df.merge(
            tiv_df[["subject_code", "session_id", "tiv"]],
            on=["subject_code", "session_id"],
            how="left",
        )
        logger.info(f"Added TIV column for {tiv_df['tiv'].notna().sum()} sessions")

        # Optionally normalize volume columns by TIV
        if normalize:
            volume_mask = df["metric"] == "volume"
            if volume_mask.any() and "tiv" in df.columns:
                if "volume_mm3" in df.columns:
                    df.loc[volume_mask, "volume_mm3_normalized"] = (
                        df.loc[volume_mask, "volume_mm3"] / df.loc[volume_mask, "tiv"]
                    )
                    logger.info(f"Normalized {volume_mask.sum()} volume measurements by TIV")

        return df

    def _calculate_tiv(self, output_dir: Path) -> Optional[pd.DataFrame]:
        """Calculate TIV using MATLAB/CAT12."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tiv_out = output_dir / "TIV.txt"
        filled_template = output_dir / "cat12_tiv.m"

        # Build MATLAB script
        xml_lines = "\n".join([f"'{xml_path}'" for _, _, xml_path in self._xml_files])
        template_text = self.tiv_config.tiv_template.read_text()
        filled_text = template_text.replace("$XMLS", xml_lines).replace("$OUT_FILE", str(tiv_out))
        filled_template.write_text(filled_text)

        # Build MATLAB command as a single string for -r option
        spm = str(self.tiv_config.spm_path)
        cat = str(self.tiv_config.cat12_path)
        script = str(filled_template)
        matlab_cmd = (
            f"addpath('{spm}'); addpath('{cat}'); "
            f"try, run('{script}'); catch ME, disp(ME.message); end; exit;"
        )

        # Run MATLAB
        cmd = [
            str(self.tiv_config.matlab_bin),
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            matlab_cmd,
        ]
        logger.info(f"Running MATLAB for TIV calculation ({len(self._xml_files)} sessions)...")
        logger.debug(f"MATLAB command: {' '.join(cmd)}")
        logger.debug(f"TIV script: {filled_template}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"MATLAB TIV calculation failed (returncode={result.returncode})")
            logger.error(f"MATLAB stderr: {result.stderr}")

        # Log stdout for debugging (may contain warnings)
        if result.stdout:
            # Check for warnings in output
            if "Warning" in result.stdout or "Error" in result.stdout:
                logger.warning(f"MATLAB output contains warnings/errors:\n{result.stdout}")
            else:
                logger.debug(f"MATLAB stdout: {result.stdout[:500]}...")

        if not tiv_out.exists():
            logger.error(f"TIV output file not created: {tiv_out}")
            logger.error(f"Check MATLAB script at: {filled_template}")
            logger.error(f"XML files being processed: {len(self._xml_files)}")
            if self._xml_files:
                logger.error(f"First XML: {self._xml_files[0][2]}")
            return None

        # Parse results
        tiv_data = pd.read_csv(tiv_out, header=None).values.flatten()

        if len(tiv_data) != len(self._xml_files):
            logger.warning(
                f"TIV count mismatch: got {len(tiv_data)}, expected {len(self._xml_files)}"
            )

        result_df = pd.DataFrame(
            [
                {"subject_code": subj, "session_id": sess, "tiv": tiv}
                for (subj, sess, _), tiv in zip(self._xml_files, tiv_data, strict=False)
            ]
        )

        logger.info(f"TIV calculated for {len(result_df)} sessions")
        return result_df

    @property
    def region_names(self) -> Optional[List[str]]:
        """Get list of region names from the parcellation."""
        if self._parcels is None:
            return None
        return self._parcels["name"].tolist()

    @property
    def n_regions(self) -> int:
        """Get number of regions in the parcellation."""
        if self._parcels is None:
            return 0
        return len(self._parcels)
