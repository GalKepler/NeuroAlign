"""
Data preparation pipeline for NeuroAlign.

Loads neuroimaging data from multiple modalities and outputs to a FeatureStore
with long format (raw parcellator output) and wide format (per metric) files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from neuroalign.data.loaders import AnatomicalLoader, DiffusionLoader
from .config import PipelineConfig
from .feature_store import FeatureStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from data preparation pipeline."""

    store: FeatureStore
    metadata: Dict[str, Any]
    output_path: Path
    long_formats_saved: List[str]
    wide_features_generated: List[str]
    n_new_sessions: int = 0
    n_skipped_sessions: int = 0


class DataPreparationPipeline:
    """
    Main pipeline for preparing NeuroAlign feature matrices.

    Saves data in two formats:
    - Long format: Raw parcellator output with all columns preserved
    - Wide format: One file per metric for efficient modeling
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._sessions_df: Optional[pd.DataFrame] = None

    def _load_sessions_csv(self) -> pd.DataFrame:
        """Load and cache sessions CSV."""
        if self._sessions_df is None:
            self._sessions_df = pd.read_csv(
                self.config.paths.sessions_csv,
                dtype={"subject_code": str, "session_id": str},
            )
            logger.info(f"Loaded {len(self._sessions_df)} sessions from CSV")
        return self._sessions_df

    def _validate_sessions_csv(self) -> None:
        """Validate that sessions CSV has required columns."""
        df = self._load_sessions_csv()
        required = ["subject_code", "session_id"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Sessions CSV missing required columns: {missing}")

    def _init_anatomical_loader(self) -> Optional[AnatomicalLoader]:
        """Initialize anatomical loader if paths are configured."""
        if not self.config.modalities.anatomical:
            logger.info("Anatomical modality disabled")
            return None

        if self.config.paths.cat12_root is None:
            logger.warning("Anatomical modality enabled but cat12_root not set")
            return None

        if self.config.paths.atlas_root is None:
            logger.warning("Anatomical modality enabled but atlas_root not set")
            return None

        return AnatomicalLoader(
            cat12_root=self.config.paths.cat12_root,
            atlas_root=self.config.paths.atlas_root,
            atlas_name=self.config.atlas_name,
        )

    def _init_diffusion_loader(self) -> Optional[DiffusionLoader]:
        """Initialize diffusion loader if paths are configured."""
        if not self.config.modalities.diffusion:
            logger.info("Diffusion modality disabled")
            return None

        if self.config.paths.qsiparc_path is None:
            logger.warning("Diffusion modality enabled but qsiparc_path not set")
            return None

        if self.config.paths.qsirecon_path is None:
            logger.warning("Diffusion modality enabled but qsirecon_path not set")
            return None

        return DiffusionLoader(
            qsiparc_path=self.config.paths.qsiparc_path,
            qsirecon_path=self.config.paths.qsirecon_path,
            workflows=self.config.modalities.diffusion_workflows,
            atlas_name=self.config.atlas_name,
        )

    def _get_anatomical_modalities(self) -> List[str]:
        """Get list of enabled anatomical modalities."""
        modalities = []
        if self.config.modalities.gray_matter:
            modalities.append("gm")
        if self.config.modalities.white_matter:
            modalities.append("wm")
        if self.config.modalities.cortical_thickness:
            modalities.append("ct")
        return modalities

    def _get_sessions_to_load(self, store: FeatureStore) -> tuple[pd.DataFrame, int]:
        """
        Get sessions that need to be loaded.

        If force=True or store doesn't exist, returns all sessions.
        Otherwise, returns only sessions not already in the store.
        """
        all_sessions = self._load_sessions_csv()

        if self.config.force or not store.exists():
            logger.info("Loading all sessions (force=True or new store)")
            return all_sessions, 0

        existing = store.get_existing_sessions()

        if existing.empty:
            return all_sessions, 0

        merged = all_sessions.merge(
            existing,
            on=["subject_code", "session_id"],
            how="left",
            indicator=True,
        )
        new_sessions = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        n_skipped = len(all_sessions) - len(new_sessions)

        if n_skipped > 0:
            logger.info(
                f"Incremental mode: {n_skipped} sessions already in store, "
                f"{len(new_sessions)} new sessions to load"
            )

        return new_sessions, n_skipped

    def _write_temp_sessions_csv(self, sessions_df: pd.DataFrame) -> Path:
        """Write a temporary sessions CSV for the loaders."""
        temp_dir = self.config.paths.output_dir / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_csv = temp_dir / "sessions_to_load.csv"
        sessions_df.to_csv(temp_csv, index=False)
        return temp_csv

    def load_anatomical_data(
        self,
        sessions_csv: Optional[Path] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load anatomical data (all modalities combined).

        Returns:
            Long-format DataFrame with all parcellator columns
        """
        loader = self._init_anatomical_loader()
        if loader is None:
            return None

        csv_path = sessions_csv or self.config.paths.sessions_csv

        logger.info(f"Loading anatomical data (n_jobs={self.config.n_jobs})...")
        try:
            # Use output directory for TIV files (easier to debug than temp)
            tiv_work_dir = self.config.paths.output_dir / ".tiv_work"

            # Load with TIV calculation but WITHOUT normalization
            # This adds 'tiv' column to the data without modifying volume values
            long_df = loader.load_sessions(
                sessions_csv=csv_path,
                n_jobs=self.config.n_jobs,
                progress=self.config.progress,
                normalize_by_tiv=False,  # Don't normalize - store raw values
                calculate_tiv=True,  # But do calculate TIV and add to dataframe
                tiv_output_dir=tiv_work_dir,  # Use known directory for debugging
            )
        except Exception as e:
            logger.error(f"Failed to load anatomical data: {e}")
            return None

        # Filter to requested modalities
        modalities = self._get_anatomical_modalities()
        if modalities:
            long_df = long_df[long_df["modality"].isin(modalities)]

        logger.info(f"Loaded {len(long_df)} anatomical records")
        return long_df

    def load_diffusion_data(
        self,
        sessions_csv: Optional[Path] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load diffusion data.

        Returns:
            Long-format DataFrame from DiffusionLoader
        """
        loader = self._init_diffusion_loader()
        if loader is None:
            return None

        csv_path = sessions_csv or self.config.paths.sessions_csv

        logger.info("Loading diffusion data...")
        try:
            long_df = loader.load_sessions(
                sessions_csv=csv_path,
                progress=self.config.progress,
            )
        except Exception as e:
            logger.error(f"Failed to load diffusion data: {e}")
            return None

        logger.info(f"Loaded {len(long_df)} diffusion records")
        return long_df

    def _extract_tiv(self, anatomical_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract TIV from anatomical data if present.

        The anatomical loader adds 'tiv' column when MATLAB/CAT12 is configured.
        """
        if anatomical_df is None or "tiv" not in anatomical_df.columns:
            return None

        # Get unique TIV per session
        tiv_df = (
            anatomical_df[["subject_code", "session_id", "tiv"]]
            .drop_duplicates()
            .dropna(subset=["tiv"])
        )

        if tiv_df.empty:
            return None

        logger.info(f"Extracted TIV for {len(tiv_df)} sessions")
        return tiv_df

    def _prepare_metadata_df(
        self,
        anatomical_df: Optional[pd.DataFrame],
        diffusion_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Prepare metadata DataFrame with AGE from sessions CSV."""
        session_pairs = []

        if anatomical_df is not None and len(anatomical_df) > 0:
            session_pairs.append(
                anatomical_df[["subject_code", "session_id"]].drop_duplicates()
            )

        if diffusion_df is not None and len(diffusion_df) > 0:
            session_pairs.append(
                diffusion_df[["subject_code", "session_id"]].drop_duplicates()
            )

        if not session_pairs:
            raise ValueError("No data loaded from any modality")

        all_sessions = pd.concat(session_pairs, ignore_index=True).drop_duplicates()

        # Add age from sessions CSV
        sessions = self._load_sessions_csv()
        age_col = self.config.age_column

        age_source_col = None
        for col_name in [age_col, "AGE", "Age@Scan", "age", "Age"]:
            if col_name in sessions.columns:
                age_source_col = col_name
                break

        if age_source_col:
            metadata = all_sessions.merge(
                sessions[["subject_code", "session_id", age_source_col]].drop_duplicates(),
                on=["subject_code", "session_id"],
                how="left",
            )
            if age_source_col != "AGE":
                metadata = metadata.rename(columns={age_source_col: "AGE"})
        else:
            metadata = all_sessions.copy()
            logger.warning("No age column found in sessions CSV")

        return metadata

    def _compute_metadata(self, store: FeatureStore) -> Dict[str, Any]:
        """Compute metadata about the output."""
        summary = store.summary()

        age_stats = {"min": None, "max": None, "mean": None, "missing": 0}
        if store.metadata_path.exists():
            meta = pd.read_parquet(store.metadata_path)
            if "AGE" in meta.columns:
                age_stats = {
                    "min": float(meta["AGE"].min()) if not meta["AGE"].isna().all() else None,
                    "max": float(meta["AGE"].max()) if not meta["AGE"].isna().all() else None,
                    "mean": float(meta["AGE"].mean()) if not meta["AGE"].isna().all() else None,
                    "missing": int(meta["AGE"].isna().sum()),
                }

        return {
            "n_subjects": summary.get("n_subjects", 0),
            "n_sessions": summary.get("n_sessions", 0),
            "long_formats": summary.get("long_formats", []),
            "n_wide_features": summary.get("n_wide_features", 0),
            "anatomical_features": summary.get("anatomical_features", []),
            "diffusion_features": summary.get("diffusion_features", []),
            "has_tiv": summary.get("has_tiv", False),
            "age_stats": age_stats,
            "atlas_name": self.config.atlas_name,
        }

    def run(self) -> PipelineResult:
        """
        Execute the full data preparation pipeline.

        Steps:
        1. Load anatomical data and save as long format (per modality)
        2. Load diffusion data and save as long format (per workflow)
        3. Extract and save TIV separately
        4. Generate wide-format features from long format
        5. Save metadata

        Returns:
            PipelineResult with store, metadata, and saved formats
        """
        logger.info("Starting data preparation pipeline...")

        self._validate_sessions_csv()

        # Initialize feature store
        store = FeatureStore(
            root_dir=self.config.paths.output_dir,
            compression=self.config.output.compression,
        )

        # Determine which sessions to load
        sessions_to_load, n_skipped = self._get_sessions_to_load(store)

        if sessions_to_load.empty:
            logger.info("All sessions already in store - nothing to do")
            metadata = self._compute_metadata(store)
            return PipelineResult(
                store=store,
                metadata=metadata,
                output_path=self.config.paths.output_dir,
                long_formats_saved=store.list_long_formats(),
                wide_features_generated=store.list_features(),
                n_new_sessions=0,
                n_skipped_sessions=n_skipped,
            )

        # Prepare temp CSV if incremental
        temp_csv = None
        is_incremental = not self.config.force and store.exists()

        if is_incremental:
            temp_csv = self._write_temp_sessions_csv(sessions_to_load)
            sessions_csv_path = temp_csv
        else:
            sessions_csv_path = None

        n_new_sessions = len(sessions_to_load[["subject_code", "session_id"]].drop_duplicates())

        long_formats_saved = []
        tiv_df = None

        # =====================================================================
        # Load and save anatomical data (long format per modality)
        # =====================================================================
        anatomical_df = self.load_anatomical_data(sessions_csv=sessions_csv_path)

        if anatomical_df is not None and len(anatomical_df) > 0:
            logger.info("Saving anatomical long format data...")

            # Extract TIV before splitting by modality
            tiv_df = self._extract_tiv(anatomical_df)

            # Save each modality separately
            for modality in anatomical_df["modality"].unique():
                mod_df = anatomical_df[anatomical_df["modality"] == modality].copy()
                name = store.save_anatomical_long(
                    mod_df,
                    modality=modality,
                    atlas_name=self.config.atlas_name,
                )
                long_formats_saved.append(name)

        # =====================================================================
        # Load and save diffusion data (long format per workflow)
        # =====================================================================
        diffusion_df = self.load_diffusion_data(sessions_csv=sessions_csv_path)

        if diffusion_df is not None and len(diffusion_df) > 0:
            logger.info("Saving diffusion long format data...")

            # Save each workflow separately
            for workflow in diffusion_df["workflow"].unique():
                wf_df = diffusion_df[diffusion_df["workflow"] == workflow].copy()
                name = store.save_diffusion_long(wf_df, workflow=workflow)
                long_formats_saved.append(name)

        # Clean up temp file
        if temp_csv and temp_csv.exists():
            temp_csv.unlink()
            if temp_csv.parent.exists():
                try:
                    temp_csv.parent.rmdir()
                except OSError:
                    pass

        if not long_formats_saved:
            raise ValueError("No data was saved - check your data paths")

        # =====================================================================
        # Save TIV (as both standalone file and anatomical feature)
        # =====================================================================
        tiv_feature_name = None
        if tiv_df is not None and len(tiv_df) > 0:
            logger.info("Saving TIV data...")
            tiv_feature_name = store.save_tiv(tiv_df)

        # =====================================================================
        # Save metadata
        # =====================================================================
        logger.info("Saving metadata...")
        metadata_df = self._prepare_metadata_df(anatomical_df, diffusion_df)
        store.save_metadata(metadata_df, age_col=self.config.age_column)

        # =====================================================================
        # Generate wide-format features
        # =====================================================================
        logger.info("Generating wide-format features...")
        wide_features = store.generate_wide_features()

        # Include TIV in wide features list if it was saved
        if tiv_feature_name and tiv_feature_name not in wide_features:
            wide_features.append(tiv_feature_name)

        # Compute final metadata
        metadata = self._compute_metadata(store)

        logger.info("Pipeline complete!")

        return PipelineResult(
            store=store,
            metadata=metadata,
            output_path=self.config.paths.output_dir,
            long_formats_saved=long_formats_saved,
            wide_features_generated=wide_features,
            n_new_sessions=n_new_sessions,
            n_skipped_sessions=n_skipped,
        )
