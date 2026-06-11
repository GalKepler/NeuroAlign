"""
Data preparation pipeline for NeuroAlign.

Loads behavioral data (brainlink) and pre-parcellated tabular derivatives
(anatomical + diffusion), and writes them to a `FeatureStore` with long
format (raw `TabularDerivativesLoader` output) and wide format (per
metric/combination) files.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from neuroalign.data.loaders import BehavioralLoader, TabularDerivativesLoader

from .config import PipelineConfig
from .feature_store import META_COLS, FeatureStore

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

    1. `BehavioralLoader.get_sessions()` provides the canonical session list
       (uid, session_id, AGE, sex, ...).
    2. `TabularDerivativesLoader.load_anatomical/load_diffusion()` provide
       long-format regional tables for those sessions.
    3. `FeatureStore` saves long format, TIV, metadata, and generates wide
       format features.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._behavioral = BehavioralLoader(config.paths.brainlink_db)
        self._derivatives = TabularDerivativesLoader(
            config.paths.tabular_derivatives_root,
            atlas_name=config.atlas_name,
            anat_atlases=config.anat_atlases,
            session_variant=config.session_variant,
        )
        self._sessions_df: Optional[pd.DataFrame] = None

    def _load_sessions(self) -> pd.DataFrame:
        """Load and cache the canonical session list from brainlink."""
        if self._sessions_df is None:
            self._sessions_df = self._behavioral.get_sessions(
                labs=self.config.labs,
                require_complete_mapping=self.config.require_complete_mapping,
            )
            logger.info(f"Loaded {len(self._sessions_df)} sessions from brainlink")
        return self._sessions_df

    def _get_sessions_to_load(self, store: FeatureStore) -> Tuple[pd.DataFrame, int]:
        """
        Get sessions that need to be loaded.

        If force=True or store doesn't exist, returns all sessions.
        Otherwise, returns only sessions not already in the store.
        """
        all_sessions = self._load_sessions()

        if self.config.force or not store.exists():
            logger.info("Loading all sessions (force=True or new store)")
            return all_sessions, 0

        existing = store.get_existing_sessions()

        if existing.empty:
            return all_sessions, 0

        merged = all_sessions.merge(existing, on=META_COLS, how="left", indicator=True)
        new_sessions = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        n_skipped = len(all_sessions) - len(new_sessions)

        if n_skipped > 0:
            logger.info(
                f"Incremental mode: {n_skipped} sessions already in store, "
                f"{len(new_sessions)} new sessions to load"
            )

        return new_sessions, n_skipped

    def _load_derivatives(
        self, sessions: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load anatomical and diffusion long-format data concurrently."""
        anatomical_df = None
        diffusion_df = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if self.config.modalities.anatomical:
                futures["anatomical"] = executor.submit(self._derivatives.load_anatomical, sessions)

            if self.config.modalities.diffusion:
                futures["diffusion"] = executor.submit(self._derivatives.load_diffusion, sessions)

            for name, future in futures.items():
                try:
                    result = future.result()
                    if name == "anatomical":
                        anatomical_df = result
                    else:
                        diffusion_df = result
                except Exception as e:
                    logger.error(f"Failed to load {name} data: {e}")

        return anatomical_df, diffusion_df

    def _compute_metadata(self, store: FeatureStore) -> Dict[str, Any]:
        """Compute metadata about the output."""
        summary = store.summary()

        age_stats = {"min": None, "max": None, "mean": None, "missing": 0}
        if store.metadata_path.exists():
            meta = store.load_metadata()
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
        1. Load the canonical session list from brainlink.
        2. Load anatomical and diffusion long-format data for new sessions.
        3. Save long format, TIV, and metadata.
        4. Generate wide-format features from long format.

        Returns:
            PipelineResult with store, metadata, and saved formats
        """
        logger.info("Starting data preparation pipeline...")

        store = FeatureStore(
            root_dir=self.config.paths.output_dir,
            compression=self.config.output.compression,
        )

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

        n_new_sessions = len(sessions_to_load[META_COLS].drop_duplicates())

        anatomical_df, diffusion_df = self._load_derivatives(sessions_to_load)

        long_formats_saved: List[str] = []

        if anatomical_df is not None and len(anatomical_df) > 0:
            logger.info("Saving anatomical long format data...")
            name = store.save_anatomical_long(anatomical_df, atlas_name=self.config.atlas_name)
            long_formats_saved.append(name)

        if diffusion_df is not None and len(diffusion_df) > 0:
            logger.info("Saving diffusion long format data...")
            names = store.save_diffusion_long(diffusion_df, atlas_name=self.config.atlas_name)
            long_formats_saved.extend(names)

        if not long_formats_saved:
            raise ValueError("No data was saved - check your data paths")

        # =====================================================================
        # Save TIV (extracted directly from anatomical long format)
        # =====================================================================
        tiv_feature_name = None
        if (
            anatomical_df is not None
            and "tiv_mm3" in anatomical_df.columns
            and len(anatomical_df) > 0
        ):
            logger.info("Saving TIV data...")
            tiv_feature_name = store.save_tiv(anatomical_df)

        # =====================================================================
        # Save metadata (AGE, sex, ... from brainlink)
        # =====================================================================
        logger.info("Saving metadata...")
        store.save_metadata(sessions_to_load)

        # =====================================================================
        # Generate wide-format features
        # =====================================================================
        logger.info("Generating wide-format features...")
        wide_features = store.generate_wide_features()

        if tiv_feature_name and tiv_feature_name not in wide_features:
            wide_features.append(tiv_feature_name)

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
