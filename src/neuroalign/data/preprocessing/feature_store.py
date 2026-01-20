"""
Feature Store for NeuroAlign regional brain features.

Organizes features in two formats:
1. Long format (raw) - Preserves all parcellator output columns
2. Wide format - One file per metric for easy modeling access

Structure:
    data/processed/
    ├── long/
    │   ├── anatomical_gm.parquet
    │   ├── anatomical_wm.parquet
    │   ├── anatomical_ct.parquet
    │   └── diffusion/
    │       ├── AMICONODDI.parquet
    │       └── ...
    ├── wide/
    │   ├── anatomical/
    │   │   ├── gm_volume_mm3.parquet
    │   │   ├── gm_mean.parquet
    │   │   └── ...
    │   └── diffusion/
    │       └── ...
    ├── tiv.parquet
    ├── metadata.parquet
    └── manifest.json

Example:
    >>> store = FeatureStore("data/processed")
    >>> store.list_features()
    ['gm_volume_mm3', 'gm_mean', 'ct_mean', ...]
    >>> gm_vol = store.load_feature("gm_volume_mm3")
    >>> gm_long = store.load_long("anatomical_gm")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Regional summary columns from VolumetricParcellator
ANATOMICAL_METRICS = ["volume_mm3", "mean", "std", "median", "sum", "robust_std", "mad_median"]
# Diffusion summary columns from VolumetricParcellator (same parcellator, similar columns)
DIFFUSION_METRICS = [
    "mean",
    "std",
    "median",
    "robust_mean",
    "robust_std",
    "mad_median",
    "z_filtered_mean",
    "z_filtered_std",
    "iqr_filtered_mean",
    "iqr_filtered_std",
    "volume_mm3",
    "voxel_count",
]
# Columns that identify the region (not metrics)
REGION_ID_COLS = [
    "index",
    "label",
    "network_label",
    "label_7network",
    "index_17network",
    "label_17network",
    "network_label_17network",
    "atlas_name",
    "network_id",
]
# Metadata columns
META_COLS = ["subject_code", "session_id"]


@dataclass
class FeatureInfo:
    """Information about a stored feature."""

    name: str
    modality: str  # "anatomical" or "diffusion"
    metric: str  # e.g., "volume_mm3", "mean", "ICVF"
    n_regions: int
    n_sessions: int
    region_names: List[str]
    file_path: str
    created_at: str
    # Additional metadata
    source_modality: Optional[str] = None  # gm, wm, ct
    workflow: Optional[str] = None  # For diffusion
    model: Optional[str] = None
    param: Optional[str] = None

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
            "source_modality": self.source_modality,
            "workflow": self.workflow,
            "model": self.model,
            "param": self.param,
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

    version: str = "2.0"
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
            version=data.get("version", "2.0"),
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
    - Long format: Raw parcellator output with all columns preserved
    - Wide format: One file per metric for efficient modeling

    Also stores TIV (Total Intracranial Volume) separately for normalization.
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
    # Long format storage (raw parcellator output)
    # -------------------------------------------------------------------------

    def save_anatomical_long(
        self,
        df: pd.DataFrame,
        modality: Literal["gm", "wm", "ct"],
        atlas_name: str = "",
    ) -> str:
        """
        Save raw anatomical parcellator output in long format.

        Preserves ALL columns from the parcellator (volume, mean, std, etc.).

        Args:
            df: Long-format DataFrame from AnatomicalLoader/parcellator
            modality: Source modality (gm, wm, ct)
            atlas_name: Name of atlas used

        Returns:
            Name of saved long format file
        """
        self._ensure_dirs()
        manifest = self._load_manifest()
        if atlas_name:
            manifest.atlas_name = atlas_name

        name = f"anatomical_{modality}"
        file_path = self.long_dir / f"{name}.parquet"

        df.to_parquet(file_path, compression=self.compression, index=False)

        # Identify available metrics
        available_metrics = [c for c in df.columns if c in ANATOMICAL_METRICS]

        info = LongFormatInfo(
            name=name,
            modality="anatomical",
            n_rows=len(df),
            n_sessions=df[["subject_code", "session_id"]].drop_duplicates().shape[0],
            columns=df.columns.tolist(),
            metrics_available=available_metrics,
            file_path=str(file_path.relative_to(self.root_dir)),
            created_at=datetime.now().isoformat(),
        )
        manifest.add_long_format(info)

        # Update session counts
        n_sessions = df[["subject_code", "session_id"]].drop_duplicates().shape[0]
        n_subjects = df["subject_code"].nunique()
        manifest.n_sessions = max(manifest.n_sessions, n_sessions)
        manifest.n_subjects = max(manifest.n_subjects, n_subjects)

        self._save_manifest()
        logger.info(f"Saved {name}: {info.n_rows} rows, {info.n_sessions} sessions")

        return name

    def save_diffusion_long(
        self,
        df: pd.DataFrame,
        workflow: str,
    ) -> str:
        """
        Save raw diffusion parcellator output in long format.

        Args:
            df: Long-format DataFrame from DiffusionLoader
            workflow: Workflow name (e.g., "AMICONODDI", "DSIStudio")

        Returns:
            Name of saved long format file
        """
        self._ensure_dirs()
        manifest = self._load_manifest()

        name = f"diffusion_{workflow}"
        file_path = self.diffusion_long_dir / f"{workflow}.parquet"

        df.to_parquet(file_path, compression=self.compression, index=False)

        # Identify available summary statistics (mean, std, median, etc.)
        available_metrics = [c for c in df.columns if c in DIFFUSION_METRICS]

        info = LongFormatInfo(
            name=name,
            modality="diffusion",
            n_rows=len(df),
            n_sessions=df[["subject_code", "session_id"]].drop_duplicates().shape[0],
            columns=df.columns.tolist(),
            metrics_available=available_metrics,
            file_path=str(file_path.relative_to(self.root_dir)),
            created_at=datetime.now().isoformat(),
        )
        manifest.add_long_format(info)

        n_sessions = df[["subject_code", "session_id"]].drop_duplicates().shape[0]
        n_subjects = df["subject_code"].nunique()
        manifest.n_sessions = max(manifest.n_sessions, n_sessions)
        manifest.n_subjects = max(manifest.n_subjects, n_subjects)

        self._save_manifest()
        logger.info(f"Saved {name}: {info.n_rows} rows, {info.n_sessions} sessions")

        return name

    def load_long(self, name: str) -> pd.DataFrame:
        """
        Load long-format data.

        Args:
            name: Long format name (e.g., "anatomical_gm", "diffusion_AMICONODDI")

        Returns:
            DataFrame with all parcellator columns
        """
        manifest = self._load_manifest()
        info = manifest.get_long_format(name)

        if info is None:
            raise ValueError(f"Long format '{name}' not found. Available: {self.list_long_formats()}")

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

        Args:
            metrics: Specific metrics to generate (default: all available)
            modalities: Modalities to process (default: all)

        Returns:
            List of generated feature names
        """
        self._ensure_dirs()
        manifest = self._load_manifest()

        generated = []

        # Process anatomical long formats
        for long_name, long_info_dict in manifest.long_formats.items():
            long_info = LongFormatInfo.from_dict(long_info_dict)

            if long_info.modality == "anatomical":
                if modalities and not any(m in long_name for m in modalities):
                    continue

                # Load long format
                df = self.load_long(long_name)
                source_mod = long_name.replace("anatomical_", "")  # gm, wm, ct

                # Generate wide for each metric
                metrics_to_gen = metrics or long_info.metrics_available
                for metric in metrics_to_gen:
                    if metric not in df.columns:
                        continue

                    feat_name = f"{source_mod}_{metric}"
                    feat_path = self.anatomical_wide_dir / f"{feat_name}.parquet"

                    # Pivot to wide
                    wide_df = df.pivot_table(
                        index=["subject_code", "session_id"],
                        columns="label",
                        values=metric,
                        aggfunc="first",
                    ).reset_index()
                    wide_df.columns.name = None

                    wide_df.to_parquet(feat_path, compression=self.compression, index=False)

                    # Get region names
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
                        source_modality=source_mod,
                    )
                    manifest.add_feature(info)
                    generated.append(feat_name)

                    logger.info(f"Generated {feat_name}: {len(wide_df)} sessions, {len(region_cols)} regions")

            elif long_info.modality == "diffusion":
                # Load long format
                df = self.load_long(long_name)
                workflow = long_name.replace("diffusion_", "")

                # Determine region column
                region_col = "name" if "name" in df.columns else "label"
                if region_col not in df.columns:
                    for alt in ["label", "name", "region"]:
                        if alt in df.columns:
                            region_col = alt
                            break

                # Find available summary statistics in this diffusion data
                available_metrics = [m for m in DIFFUSION_METRICS if m in df.columns]
                if not available_metrics:
                    logger.warning(f"No summary statistics found in {long_name}")
                    continue

                # Get unique model/param combinations
                if "model" in df.columns and "param" in df.columns:
                    for (model, param), group in df.groupby(["model", "param"]):
                        # Generate wide format for each available summary statistic
                        for metric in available_metrics:
                            if metric not in group.columns:
                                continue

                            # Feature name includes the summary statistic
                            feat_name = f"{workflow}_{model}_{param}_{metric}"
                            feat_path = self.diffusion_wide_dir / f"{feat_name}.parquet"

                            # Pivot to wide
                            wide_df = group.pivot_table(
                                index=["subject_code", "session_id"],
                                columns=region_col,
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
                                workflow=workflow,
                                model=model,
                                param=param,
                            )
                            manifest.add_feature(info)
                            generated.append(feat_name)

                            logger.info(f"Generated {feat_name}: {len(wide_df)} sessions")

        self._save_manifest()
        return generated

    # -------------------------------------------------------------------------
    # TIV storage
    # -------------------------------------------------------------------------

    def save_tiv(self, df: pd.DataFrame) -> str:
        """
        Save TIV (Total Intracranial Volume) data.

        Saves TIV both as a standalone file (for load_tiv()) and as a
        wide-format anatomical feature (for load_feature("tiv")).

        Args:
            df: DataFrame with subject_code, session_id, tiv columns

        Returns:
            Feature name ("tiv")
        """
        self._ensure_dirs()
        manifest = self._load_manifest()

        # Ensure required columns
        required = ["subject_code", "session_id", "tiv"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"TIV DataFrame missing columns: {missing}")

        tiv_df = df[required].drop_duplicates()

        # Save as standalone TIV file (backwards compatibility)
        tiv_df.to_parquet(self.tiv_path, compression=self.compression, index=False)

        # Also save as wide-format anatomical feature
        feat_name = "tiv"
        feat_path = self.anatomical_wide_dir / f"{feat_name}.parquet"
        tiv_df.to_parquet(feat_path, compression=self.compression, index=False)

        # Register as a feature in manifest
        info = FeatureInfo(
            name=feat_name,
            modality="anatomical",
            metric="tiv",
            n_regions=1,  # TIV is a single global measure
            n_sessions=len(tiv_df),
            region_names=["tiv"],
            file_path=str(feat_path.relative_to(self.root_dir)),
            created_at=datetime.now().isoformat(),
            source_modality="global",
        )
        manifest.add_feature(info)
        manifest.has_tiv = True
        self._save_manifest()

        logger.info(f"Saved TIV for {len(tiv_df)} sessions")
        return feat_name

    def load_tiv(self) -> pd.DataFrame:
        """Load TIV data."""
        if not self.tiv_path.exists():
            raise FileNotFoundError("TIV file not found. Run pipeline with TIV calculation enabled.")
        return pd.read_parquet(self.tiv_path)

    def has_tiv(self) -> bool:
        """Check if TIV data is available."""
        return self.tiv_path.exists()

    # -------------------------------------------------------------------------
    # Metadata storage
    # -------------------------------------------------------------------------

    def save_metadata(self, df: pd.DataFrame, age_col: str = "AGE") -> None:
        """
        Save session metadata (AGE, etc.).

        Args:
            df: DataFrame with subject_code, session_id, and metadata columns
            age_col: Name of age column in input
        """
        cols = ["subject_code", "session_id"]

        # Handle age column
        for col in [age_col, "AGE", "Age@Scan", "age", "Age"]:
            if col in df.columns:
                cols.append(col)
                break

        # Include other metadata columns
        feature_prefixes = ("gm_", "wm_", "ct_", "DSIStudio", "MRtrix", "DIPY", "AMICO")
        for col in df.columns:
            if col not in cols and not any(col.startswith(p) for p in feature_prefixes):
                if col not in ANATOMICAL_METRICS and col not in REGION_ID_COLS:
                    cols.append(col)

        # Only keep columns that exist
        cols = [c for c in cols if c in df.columns]

        meta_df = df[cols].drop_duplicates()

        # Standardize age column name
        if age_col in meta_df.columns and age_col != "AGE":
            meta_df = meta_df.rename(columns={age_col: "AGE"})

        meta_df.to_parquet(self.metadata_path, compression=self.compression, index=False)
        logger.info(f"Saved metadata: {len(meta_df)} sessions")

    def load_metadata(self) -> pd.DataFrame:
        """Load session metadata."""
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=["subject_code", "session_id"])
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
            name: Feature name (e.g., "gm_volume_mm3", "ct_mean")
            include_metadata: Whether to merge with metadata (AGE)
            include_tiv: Whether to include TIV column

        Returns:
            DataFrame with subject_code, session_id, and region columns
        """
        manifest = self._load_manifest()
        info = manifest.get_feature(name)

        if info is None:
            raise ValueError(f"Feature '{name}' not found. Available: {self.list_features()}")

        file_path = self.root_dir / info.file_path
        df = pd.read_parquet(file_path)

        if include_metadata and self.metadata_path.exists():
            meta_df = pd.read_parquet(self.metadata_path)
            df = df.merge(meta_df, on=["subject_code", "session_id"], how="left")

        if include_tiv and self.tiv_path.exists():
            tiv_df = pd.read_parquet(self.tiv_path)
            df = df.merge(tiv_df, on=["subject_code", "session_id"], how="left")

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
            result = result.merge(feature_df, on=["subject_code", "session_id"], how="outer")

        if include_metadata and self.metadata_path.exists():
            meta_df = pd.read_parquet(self.metadata_path)
            result = result.merge(meta_df, on=["subject_code", "session_id"], how="left")

        if include_tiv and self.tiv_path.exists():
            tiv_df = pd.read_parquet(self.tiv_path)
            result = result.merge(tiv_df, on=["subject_code", "session_id"], how="left")

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
        """Get all subject-session pairs in the store."""
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=["subject_code", "session_id"])

        meta = pd.read_parquet(self.metadata_path)
        return meta[["subject_code", "session_id"]].drop_duplicates()

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
