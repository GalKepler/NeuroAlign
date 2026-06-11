"""
Feature Store for NeuroAlign regional brain features.

Organizes features in two formats:
1. Long format (raw) - Preserves all `TabularDerivativesLoader` output columns
2. Wide format - One file per metric/combination for easy modeling access

Structure:
    data/processed/
    ├── long/
    │   ├── anatomical.parquet              # Schaefer2018N400n7 + Tian2020S2, long format
    │   └── diffusion/
    │       ├── DSIStudio.parquet
    │       ├── AMICONODDI.parquet
    │       └── ...
    ├── wide/
    │   ├── anatomical/
    │   │   ├── anat_thickness_mean_mm.parquet
    │   │   ├── anat_volume_mm3.parquet
    │   │   └── ...
    │   └── diffusion/
    │       ├── DSIStudio_tensor_fa_mean.parquet
    │       └── ...
    ├── tiv.parquet
    ├── metadata.parquet
    └── manifest.json

Example:
    >>> store = FeatureStore("data/processed")
    >>> store.list_features()
    ['anat_thickness_mean_mm', 'DSIStudio_tensor_fa_mean', ...]
    >>> fa = store.load_feature("DSIStudio_tensor_fa_mean")
    >>> anat_long = store.load_long("anatomical")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Anatomical metric columns (Schaefer2018N400n7 cortex + Tian2020S2 subcortex,
# concatenated by TabularDerivativesLoader.load_anatomical). Cortex and
# subcortex regions only populate their own metrics - the rest are NaN.
ANATOMICAL_METRICS = [
    # cortex (surface-based, Schaefer2018N400n7)
    "num_vertices",
    "surface_area_mm2",
    "gray_matter_volume_mm3",
    "thickness_mean_mm",
    "thickness_std_mm",
    "mean_curvature",
    "gaussian_curvature",
    "folding_index",
    "curvature_index",
    "white_surf_area_mm2",
    "brain_seg_vol_mm3",
    "brain_seg_no_vent_mm3",
    "cortex_vol_mm3",
    "supratentorial_vol_mm3",
    # subcortex (volume-based, Tian2020S2)
    "num_voxels",
    "volume_mm3",
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "intensity_range",
    "intensity_snr",
    "subcort_gray_mm3",
]

# Diffusion metric columns, generated per (software, model, param, desc) combo
DIFFUSION_METRICS = [
    "mean",
    "std",
    "median",
    "sum",
    "cv",
    "robust_mean",
    "robust_std",
    "robust_cv",
    "mad_median",
    "z_filtered_mean",
    "z_filtered_std",
    "iqr_filtered_mean",
    "iqr_filtered_std",
    "skewness",
    "excess_kurtosis",
    "percentile_5",
    "percentile_25",
    "percentile_75",
    "percentile_95",
    "coverage",
    "volume_mm3",
    "voxel_count",
]

# Columns identifying a session - the join key across all stored tables
META_COLS = ["uid", "session_id"]


@dataclass
class FeatureInfo:
    """Information about a stored wide-format feature."""

    name: str
    modality: str  # "anatomical" or "diffusion"
    metric: str  # e.g., "thickness_mean_mm", "mean"
    n_regions: int
    n_sessions: int
    region_names: List[str]
    file_path: str
    created_at: str
    # Diffusion-only identifiers
    software: Optional[str] = None
    model: Optional[str] = None
    param: Optional[str] = None
    desc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "modality": self.modality,
            "metric": self.metric,
            "n_regions": self.n_regions,
            "n_sessions": self.n_sessions,
            "region_names": self.region_names,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "software": self.software,
            "model": self.model,
            "param": self.param,
            "desc": self.desc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureInfo":
        return cls(**data)


@dataclass
class LongFormatInfo:
    """Information about a long-format data file."""

    name: str
    modality: str
    n_rows: int
    n_sessions: int
    n_subjects: int
    columns: List[str]
    metrics_available: List[str]
    file_path: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "modality": self.modality,
            "n_rows": self.n_rows,
            "n_sessions": self.n_sessions,
            "n_subjects": self.n_subjects,
            "columns": self.columns,
            "metrics_available": self.metrics_available,
            "file_path": self.file_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongFormatInfo":
        return cls(**data)


@dataclass
class StoreManifest:
    """Manifest describing all data in the store."""

    version: str = "3.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    atlas_name: str = ""
    n_sessions: int = 0
    n_subjects: int = 0
    long_formats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    wide_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_tiv: bool = False

    def add_long_format(self, info: LongFormatInfo) -> None:
        self.long_formats[info.name] = info.to_dict()
        self.updated_at = datetime.now().isoformat()

    def add_feature(self, info: FeatureInfo) -> None:
        self.wide_features[info.name] = info.to_dict()
        self.updated_at = datetime.now().isoformat()

    def get_long_format(self, name: str) -> Optional[LongFormatInfo]:
        if name in self.long_formats:
            return LongFormatInfo.from_dict(self.long_formats[name])
        return None

    def get_feature(self, name: str) -> Optional[FeatureInfo]:
        if name in self.wide_features:
            return FeatureInfo.from_dict(self.wide_features[name])
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "atlas_name": self.atlas_name,
            "n_sessions": self.n_sessions,
            "n_subjects": self.n_subjects,
            "long_formats": self.long_formats,
            "wide_features": self.wide_features,
            "has_tiv": self.has_tiv,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoreManifest":
        return cls(
            version=data.get("version", "3.0"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            atlas_name=data.get("atlas_name", ""),
            n_sessions=data.get("n_sessions", 0),
            n_subjects=data.get("n_subjects", 0),
            long_formats=data.get("long_formats", {}),
            wide_features=data.get("wide_features", {}),
            has_tiv=data.get("has_tiv", False),
        )


class FeatureStore:
    """
    Storage and retrieval system for regional brain features.

    Supports two data formats:
    - Long format: Raw `TabularDerivativesLoader` output, all columns preserved
    - Wide format: One file per metric (anatomical) or per
      (software, model, param[, desc], metric) combination (diffusion)

    Also stores TIV (Total Intracranial Volume, `tiv_mm3`) separately for
    normalization - extracted directly from the anatomical long format.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        compression: Optional[str] = "snappy",
    ):
        """
        Initialize feature store.

        Args:
            root_dir: Root directory for the feature store
            compression: Parquet compression algorithm
        """
        self.root_dir = Path(root_dir)
        self.compression = compression
        self._manifest: Optional[StoreManifest] = None

    # -------------------------------------------------------------------------
    # Directory structure
    # -------------------------------------------------------------------------

    @property
    def long_dir(self) -> Path:
        return self.root_dir / "long"

    @property
    def wide_dir(self) -> Path:
        return self.root_dir / "wide"

    @property
    def anatomical_wide_dir(self) -> Path:
        return self.wide_dir / "anatomical"

    @property
    def diffusion_wide_dir(self) -> Path:
        return self.wide_dir / "diffusion"

    @property
    def diffusion_long_dir(self) -> Path:
        return self.long_dir / "diffusion"

    @property
    def tiv_path(self) -> Path:
        return self.root_dir / "tiv.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.root_dir / "metadata.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / "manifest.json"

    def _ensure_dirs(self) -> None:
        """Create directory structure."""
        self.long_dir.mkdir(parents=True, exist_ok=True)
        self.anatomical_wide_dir.mkdir(parents=True, exist_ok=True)
        self.diffusion_wide_dir.mkdir(parents=True, exist_ok=True)
        self.diffusion_long_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Manifest management
    # -------------------------------------------------------------------------

    def _load_manifest(self) -> StoreManifest:
        if self._manifest is not None:
            return self._manifest

        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self._manifest = StoreManifest.from_dict(json.load(f))
        else:
            self._manifest = StoreManifest()

        return self._manifest

    def _save_manifest(self) -> None:
        if self._manifest is None:
            return

        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest.to_dict(), f, indent=2)

    def exists(self) -> bool:
        """Check if the store exists and has data."""
        return self.manifest_path.exists()

    # -------------------------------------------------------------------------
    # Long format storage (raw TabularDerivativesLoader output)
    # -------------------------------------------------------------------------

    def save_anatomical_long(self, df: pd.DataFrame, atlas_name: str = "") -> str:
        """
        Save long-format anatomical data (`TabularDerivativesLoader.load_anatomical`).

        Args:
            df: Long-format DataFrame with `uid`, `session_id`, `label`,
                `structure`, `tiv_mm3`, and per-region metric columns.
            atlas_name: Name of the combined atlas (e.g. "Schaefer2018N400n7Tian2020S2").

        Returns:
            Name of the saved long format file ("anatomical").
        """
        self._ensure_dirs()
        manifest = self._load_manifest()
        if atlas_name:
            manifest.atlas_name = atlas_name

        name = "anatomical"
        file_path = self.long_dir / f"{name}.parquet"
        df.to_parquet(file_path, compression=self.compression, index=False)

        available_metrics = [c for c in df.columns if c in ANATOMICAL_METRICS]

        info = LongFormatInfo(
            name=name,
            modality="anatomical",
            n_rows=len(df),
            n_sessions=df[META_COLS].drop_duplicates().shape[0],
            n_subjects=df["uid"].nunique(),
            columns=df.columns.tolist(),
            metrics_available=available_metrics,
            file_path=str(file_path.relative_to(self.root_dir)),
            created_at=datetime.now().isoformat(),
        )
        manifest.add_long_format(info)
        manifest.n_sessions = max(manifest.n_sessions, info.n_sessions)
        manifest.n_subjects = max(manifest.n_subjects, info.n_subjects)

        self._save_manifest()
        logger.info(f"Saved {name}: {info.n_rows} rows, {info.n_sessions} sessions")

        return name

    def save_diffusion_long(self, df: pd.DataFrame, atlas_name: str = "") -> List[str]:
        """
        Save long-format diffusion data (`TabularDerivativesLoader.load_diffusion`),
        split into one file per `software`.

        Args:
            df: Long-format DataFrame with `uid`, `session_id`, `label`,
                `software`, `model`, `param`, `desc`, and per-region metric columns.
            atlas_name: Name of the combined atlas (e.g. "Schaefer2018N400n7Tian2020S2").

        Returns:
            Names of the saved long format files (one per `software`).
        """
        self._ensure_dirs()
        manifest = self._load_manifest()
        if atlas_name:
            manifest.atlas_name = atlas_name

        available_metrics = [c for c in df.columns if c in DIFFUSION_METRICS]
        names = []

        for software, sw_df in df.groupby("software"):
            name = f"diffusion_{software}"
            file_path = self.diffusion_long_dir / f"{software}.parquet"
            sw_df.to_parquet(file_path, compression=self.compression, index=False)

            info = LongFormatInfo(
                name=name,
                modality="diffusion",
                n_rows=len(sw_df),
                n_sessions=sw_df[META_COLS].drop_duplicates().shape[0],
                n_subjects=sw_df["uid"].nunique(),
                columns=sw_df.columns.tolist(),
                metrics_available=available_metrics,
                file_path=str(file_path.relative_to(self.root_dir)),
                created_at=datetime.now().isoformat(),
            )
            manifest.add_long_format(info)
            manifest.n_sessions = max(manifest.n_sessions, info.n_sessions)
            manifest.n_subjects = max(manifest.n_subjects, info.n_subjects)
            names.append(name)
            logger.info(f"Saved {name}: {info.n_rows} rows, {info.n_sessions} sessions")

        self._save_manifest()
        return names

    def load_long(self, name: str) -> pd.DataFrame:
        """
        Load long-format data.

        Args:
            name: Long format name (e.g., "anatomical", "diffusion_DSIStudio")

        Returns:
            DataFrame with all `TabularDerivativesLoader` columns
        """
        manifest = self._load_manifest()
        info = manifest.get_long_format(name)

        if info is None:
            raise ValueError(
                f"Long format '{name}' not found. Available: {self.list_long_formats()}"
            )

        file_path = self.root_dir / info.file_path
        return pd.read_parquet(file_path)

    def list_long_formats(self) -> List[str]:
        """List available long-format data files."""
        manifest = self._load_manifest()
        return sorted(manifest.long_formats.keys())

    # -------------------------------------------------------------------------
    # Wide format generation (from long format)
    # -------------------------------------------------------------------------

    def generate_wide_features(
        self,
        metrics: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate wide-format feature files from stored long-format data.

        Anatomical features are named `anat_<metric>` (one per metric in
        `ANATOMICAL_METRICS`, region columns from `label`). Diffusion features
        are named `<software>_<model>_<param>[_<desc>]_<metric>` (one per
        (software, model, param, desc) combination found in the data).

        Args:
            metrics: Specific metrics to generate (default: all available)
            modalities: Modalities to process, "anatomical" and/or "diffusion"
                (default: all)

        Returns:
            List of generated feature names
        """
        self._ensure_dirs()
        manifest = self._load_manifest()

        generated = []

        for long_name, long_info_dict in manifest.long_formats.items():
            long_info = LongFormatInfo.from_dict(long_info_dict)

            if modalities and long_info.modality not in modalities:
                continue

            df = self.load_long(long_name)

            if long_info.modality == "anatomical":
                generated.extend(self._generate_anatomical_wide(df, manifest, metrics))
            elif long_info.modality == "diffusion":
                generated.extend(self._generate_diffusion_wide(df, manifest, metrics))

        self._save_manifest()
        return generated

    def _generate_anatomical_wide(
        self,
        df: pd.DataFrame,
        manifest: StoreManifest,
        metrics: Optional[List[str]],
    ) -> List[str]:
        """Generate one `anat_<metric>.parquet` per anatomical metric."""
        generated = []
        metrics_to_gen = metrics or [m for m in ANATOMICAL_METRICS if m in df.columns]

        for metric in metrics_to_gen:
            if metric not in df.columns:
                continue

            feat_name = f"anat_{metric}"
            feat_path = self.anatomical_wide_dir / f"{feat_name}.parquet"

            wide_df = df.pivot_table(
                index=META_COLS,
                columns="label",
                values=metric,
                aggfunc="first",
            ).reset_index()
            wide_df.columns.name = None
            wide_df.to_parquet(feat_path, compression=self.compression, index=False)

            region_cols = [c for c in wide_df.columns if c not in META_COLS]

            info = FeatureInfo(
                name=feat_name,
                modality="anatomical",
                metric=metric,
                n_regions=len(region_cols),
                n_sessions=len(wide_df),
                region_names=region_cols,
                file_path=str(feat_path.relative_to(self.root_dir)),
                created_at=datetime.now().isoformat(),
            )
            manifest.add_feature(info)
            generated.append(feat_name)
            logger.info(
                f"Generated {feat_name}: {len(wide_df)} sessions, {len(region_cols)} regions"
            )

        return generated

    def _generate_diffusion_wide(
        self,
        df: pd.DataFrame,
        manifest: StoreManifest,
        metrics: Optional[List[str]],
    ) -> List[str]:
        """Generate one `<software>_<model>_<param>[_<desc>]_<metric>.parquet` per combo/metric."""
        generated = []
        metrics_to_gen = metrics or [m for m in DIFFUSION_METRICS if m in df.columns]

        for (software, model, param, desc), group in df.groupby(
            ["software", "model", "param", "desc"], dropna=False
        ):
            combo_label = "_".join(str(x) for x in (software, model, param))
            if pd.notna(desc):
                combo_label = f"{combo_label}_{desc}"

            for metric in metrics_to_gen:
                if metric not in group.columns:
                    continue

                feat_name = f"{combo_label}_{metric}"
                feat_path = self.diffusion_wide_dir / f"{feat_name}.parquet"

                wide_df = group.pivot_table(
                    index=META_COLS,
                    columns="label",
                    values=metric,
                    aggfunc="first",
                ).reset_index()
                wide_df.columns.name = None
                wide_df.to_parquet(feat_path, compression=self.compression, index=False)

                region_cols = [c for c in wide_df.columns if c not in META_COLS]

                info = FeatureInfo(
                    name=feat_name,
                    modality="diffusion",
                    metric=metric,
                    n_regions=len(region_cols),
                    n_sessions=len(wide_df),
                    region_names=region_cols,
                    file_path=str(feat_path.relative_to(self.root_dir)),
                    created_at=datetime.now().isoformat(),
                    software=software,
                    model=model,
                    param=param,
                    desc=desc if pd.notna(desc) else None,
                )
                manifest.add_feature(info)
                generated.append(feat_name)
                logger.info(
                    f"Generated {feat_name}: {len(wide_df)} sessions, {len(region_cols)} regions"
                )

        return generated

    # -------------------------------------------------------------------------
    # TIV storage
    # -------------------------------------------------------------------------

    def save_tiv(self, anatomical_df: pd.DataFrame) -> str:
        """
        Extract and save TIV (`tiv_mm3`) from long-format anatomical data.

        Saves TIV both as a standalone file (for `load_tiv()`) and as a
        wide-format anatomical feature (for `load_feature("tiv")`).

        Args:
            anatomical_df: Long-format anatomical DataFrame with `uid`,
                `session_id`, and `tiv_mm3` columns (one value per session,
                already present in the tabular derivatives).

        Returns:
            Feature name ("tiv")
        """
        self._ensure_dirs()
        manifest = self._load_manifest()

        required = META_COLS + ["tiv_mm3"]
        missing = [c for c in required if c not in anatomical_df.columns]
        if missing:
            raise ValueError(f"Anatomical DataFrame missing columns: {missing}")

        tiv_df = anatomical_df[required].drop_duplicates()

        # Save as standalone TIV file
        tiv_df.to_parquet(self.tiv_path, compression=self.compression, index=False)

        # Also save as wide-format anatomical feature
        feat_name = "tiv"
        feat_path = self.anatomical_wide_dir / f"{feat_name}.parquet"
        tiv_df.to_parquet(feat_path, compression=self.compression, index=False)

        info = FeatureInfo(
            name=feat_name,
            modality="anatomical",
            metric="tiv_mm3",
            n_regions=1,  # TIV is a single global measure
            n_sessions=len(tiv_df),
            region_names=["tiv_mm3"],
            file_path=str(feat_path.relative_to(self.root_dir)),
            created_at=datetime.now().isoformat(),
        )
        manifest.add_feature(info)
        manifest.has_tiv = True
        self._save_manifest()

        logger.info(f"Saved TIV for {len(tiv_df)} sessions")
        return feat_name

    def load_tiv(self) -> pd.DataFrame:
        """Load TIV data."""
        if not self.tiv_path.exists():
            raise FileNotFoundError("TIV file not found. Run `save_tiv()` first.")
        return pd.read_parquet(self.tiv_path)

    def has_tiv(self) -> bool:
        """Check if TIV data is available."""
        return self.tiv_path.exists()

    # -------------------------------------------------------------------------
    # Metadata storage
    # -------------------------------------------------------------------------

    def save_metadata(self, df: pd.DataFrame) -> None:
        """
        Save session metadata (AGE, sex, lab, ... from `BehavioralLoader`).

        Args:
            df: DataFrame with `uid`, `session_id`, and demographic/
                questionnaire columns (e.g. from `BehavioralLoader.get_sessions()`).
        """
        meta_df = df.drop_duplicates(subset=META_COLS)
        meta_df.to_parquet(self.metadata_path, compression=self.compression, index=False)
        logger.info(f"Saved metadata: {len(meta_df)} sessions")

    def load_metadata(self) -> pd.DataFrame:
        """Load session metadata."""
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=META_COLS)
        return pd.read_parquet(self.metadata_path)

    # -------------------------------------------------------------------------
    # Wide feature loading (for modeling)
    # -------------------------------------------------------------------------

    def list_features(self, modality: Optional[str] = None) -> List[str]:
        """List available wide-format features."""
        manifest = self._load_manifest()

        features = []
        for name, info in manifest.wide_features.items():
            if modality is None or info.get("modality") == modality:
                features.append(name)

        return sorted(features)

    def get_feature_info(self, name: str) -> Optional[FeatureInfo]:
        """Get information about a specific feature."""
        manifest = self._load_manifest()
        return manifest.get_feature(name)

    def load_feature(
        self,
        name: str,
        include_metadata: bool = True,
        include_tiv: bool = False,
    ) -> pd.DataFrame:
        """
        Load a wide-format feature.

        Args:
            name: Feature name (e.g., "anat_thickness_mean_mm", "DSIStudio_tensor_fa_mean")
            include_metadata: Whether to merge with metadata (AGE, sex, ...)
            include_tiv: Whether to include the `tiv_mm3` column

        Returns:
            DataFrame with `uid`, `session_id`, and region columns
        """
        manifest = self._load_manifest()
        info = manifest.get_feature(name)

        if info is None:
            raise ValueError(f"Feature '{name}' not found. Available: {self.list_features()}")

        file_path = self.root_dir / info.file_path
        df = pd.read_parquet(file_path)

        if include_metadata and self.metadata_path.exists():
            meta_df = pd.read_parquet(self.metadata_path)
            df = df.merge(meta_df, on=META_COLS, how="left")

        if include_tiv and self.tiv_path.exists():
            tiv_df = pd.read_parquet(self.tiv_path)
            df = df.merge(tiv_df, on=META_COLS, how="left")

        return df

    def load_features(
        self,
        names: List[str],
        include_metadata: bool = True,
        include_tiv: bool = False,
    ) -> pd.DataFrame:
        """Load and merge multiple wide-format features."""
        if not names:
            raise ValueError("No feature names provided")

        result = self.load_feature(names[0], include_metadata=False, include_tiv=False)

        for name in names[1:]:
            feature_df = self.load_feature(name, include_metadata=False, include_tiv=False)
            result = result.merge(feature_df, on=META_COLS, how="outer")

        if include_metadata and self.metadata_path.exists():
            meta_df = pd.read_parquet(self.metadata_path)
            result = result.merge(meta_df, on=META_COLS, how="left")

        if include_tiv and self.tiv_path.exists():
            tiv_df = pd.read_parquet(self.tiv_path)
            result = result.merge(tiv_df, on=META_COLS, how="left")

        return result

    def get_regions(self, name: str) -> List[str]:
        """Get region names for a feature."""
        info = self.get_feature_info(name)
        if info is None:
            raise ValueError(f"Feature '{name}' not found")
        return info.region_names

    # -------------------------------------------------------------------------
    # Existing sessions (for incremental loading)
    # -------------------------------------------------------------------------

    def get_existing_sessions(self) -> pd.DataFrame:
        """Get all (uid, session_id) pairs in the store."""
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=META_COLS)

        meta = pd.read_parquet(self.metadata_path)
        return meta[META_COLS].drop_duplicates()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics about the feature store."""
        manifest = self._load_manifest()

        anatomical_features = [
            n for n, f in manifest.wide_features.items() if f.get("modality") == "anatomical"
        ]
        diffusion_features = [
            n for n, f in manifest.wide_features.items() if f.get("modality") == "diffusion"
        ]

        return {
            "root_dir": str(self.root_dir),
            "atlas_name": manifest.atlas_name,
            "n_sessions": manifest.n_sessions,
            "n_subjects": manifest.n_subjects,
            "long_formats": list(manifest.long_formats.keys()),
            "n_wide_features": len(manifest.wide_features),
            "anatomical_features": anatomical_features,
            "diffusion_features": diffusion_features,
            "has_tiv": manifest.has_tiv,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
        }
