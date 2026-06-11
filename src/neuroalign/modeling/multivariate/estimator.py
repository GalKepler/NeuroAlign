"""Multivariate regional BAG estimation via `regional-stacker`.

Combines multiple regional feature tables (e.g. cortical thickness, diffusion
FA) into a single per-region feature block, then fits a
`RegionalStackingRegressor` (per-region base learners + a meta-learner) under
group-aware cross-validation to predict chronological age.
``BAG = predicted_age - actual_age``, with optional post-hoc bias correction.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from regional_stacker import RegionalStackingRegressor, wide_to_stacker_input
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from neuroalign.data.preprocessing import FeatureStore
from neuroalign.data.preprocessing.feature_store import META_COLS
from neuroalign.modeling._shared import apply_bias_correction, compute_ipw_weights
from neuroalign.modeling.config import BAGConfig
from neuroalign.modeling.result import BAGResult

logger = logging.getLogger(__name__)


def _run_outer_cv(
    x: np.ndarray,
    ages: np.ndarray,
    uids: np.ndarray,
    region_mapping: dict,
    cfg: BAGConfig,
) -> np.ndarray:
    """Cross-validated out-of-fold age predictions, one per session."""
    n_subjects = len(np.unique(uids))

    if cfg.splits == "group_kfold":
        if cfg.n_splits > n_subjects:
            raise ValueError(
                f"n_splits={cfg.n_splits} cannot be greater than "
                f"the number of unique subjects={n_subjects}."
            )
        outer_cv = GroupKFold(n_splits=cfg.n_splits)
        n_outer_splits = cfg.n_splits
    else:
        outer_cv = LeaveOneGroupOut()
        n_outer_splits = n_subjects

    stage1_cv = max(2, min(5, n_subjects))

    splits = enumerate(outer_cv.split(x, ages, groups=uids))
    if cfg.progress:
        try:
            from tqdm.auto import tqdm

            splits = tqdm(splits, total=n_outer_splits, desc="Outer folds")
        except ImportError:
            pass

    logger.info("Starting %d-fold cross-validation.", n_outer_splits)
    pred = np.full(len(ages), np.nan)
    for _, (train_idx, test_idx) in splits:
        train_ages = ages[train_idx]
        sample_weight = compute_ipw_weights(train_ages, cfg.ipw_bandwidth) if cfg.ipw else None

        stacker = RegionalStackingRegressor(
            region_mapping=region_mapping,
            outer_cv=max(2, min(stage1_cv, len(train_idx))),
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        stacker.fit(x[train_idx], train_ages, sample_weight=sample_weight)
        pred[test_idx] = stacker.predict(x[test_idx])
    logger.info("Cross-validation finished.")
    return pred


def _compute_region_metrics(
    x: np.ndarray,
    ages: np.ndarray,
    region_mapping: dict,
    cfg: BAGConfig,
) -> pd.DataFrame:
    """Per-region stage-1 (base learner) OOF diagnostics from a full-data fit."""
    n_subjects_proxy = max(2, min(5, len(ages)))
    logger.info("Fitting full-data stacker for region diagnostics.")
    full_weight = compute_ipw_weights(ages, cfg.ipw_bandwidth) if cfg.ipw else None
    full_stacker = RegionalStackingRegressor(
        region_mapping=region_mapping,
        outer_cv=n_subjects_proxy,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    full_stacker.fit(x, ages, sample_weight=full_weight)

    metrics_rows = []
    for region_idx, region in enumerate(full_stacker.region_names_):
        oof = full_stacker.oof_predictions_[:, region_idx]
        metrics_rows.append(
            {
                "region": region,
                "n_features": len(region_mapping[region]),
                "r2": r2_score(ages, oof),
                "mae": mean_absolute_error(ages, oof),
                "correlation": pearsonr(ages, oof)[0],
            }
        )
    return pd.DataFrame(metrics_rows)


class MultivariateRegionalBAGEstimator:
    """Multivariate (multimodal) regional Brain Age Gap estimator.

    Parameters
    ----------
    config : BAGConfig
        Estimation configuration. ``model_type``/``polynomial_degree`` are
        not used here (the base/meta learners default to `Ridge`); see
        `RegionalStackingRegressor` to customize them.
    """

    def __init__(self, config: BAGConfig | None = None) -> None:
        self.config = config or BAGConfig()

    def fit_predict(
        self,
        store: FeatureStore,
        feature_names: Sequence[str],
    ) -> BAGResult:
        """Run cross-validated multivariate regional BAG estimation.

        Parameters
        ----------
        store : FeatureStore
            Feature store containing the wide-format features named in
            *feature_names* and session metadata (`AGE`/`age_col`).
        feature_names : Sequence[str]
            Wide-format feature names to combine, e.g.
            ``["anat_thickness_mean_mm", "DSIStudio_tensor_fa_mean"]``. Each
            table is reindexed to the (uid, session_id) intersection across
            all tables and concatenated per region via
            `regional_stacker.wide_to_stacker_input`.

        Returns
        -------
        BAGResult
            ``bag``/``bag_uncorrected``/``predicted_age`` are
            ``[uid, session_id, <value>]`` DataFrames (one row per session,
            single value - the multivariate model produces one age
            prediction per session, not per region). ``region_metrics``
            reports per-region stage-1 (base learner) out-of-fold diagnostics
            from a fit on the full dataset.
        """
        cfg = self.config

        tables = {
            name: store.load_feature(name, include_metadata=False).set_index(META_COLS)
            for name in feature_names
        }

        x, region_mapping, sessions = wide_to_stacker_input(tables)
        logger.info(
            "Stacker input: %d sessions x %d features across %d regions.",
            x.shape[0],
            x.shape[1],
            len(region_mapping),
        )

        meta = store.load_metadata().set_index(META_COLS).reindex(sessions)
        missing_age = meta[cfg.age_col].isna()
        if missing_age.any():
            n_missing = int(missing_age.sum())
            logger.warning("Dropping %d session(s) missing '%s'.", n_missing, cfg.age_col)
            keep = ~missing_age.to_numpy()
            x = x[keep]
            meta = meta[keep]
            sessions = [s for s, k in zip(sessions, keep, strict=True) if k]

        if np.isnan(x).any():
            raise ValueError(
                "Feature matrix contains NaNs after combining tables; "
                "RegionalStackingRegressor requires complete data."
            )

        ages = meta[cfg.age_col].to_numpy(dtype=float)
        uids = meta.index.get_level_values("uid").to_numpy()

        pred = _run_outer_cv(x, ages, uids, region_mapping, cfg)

        ids = pd.DataFrame(sessions, columns=META_COLS)

        predicted_age = ids.copy()
        predicted_age["predicted_age"] = pred

        bag_uncorrected = ids.copy()
        bag_uncorrected["bag"] = pred - ages

        if cfg.bias_correction:
            logger.info("Applying bias correction.")
            corrected = apply_bias_correction(bag_uncorrected[["bag"]], ages)
            bag = ids.copy()
            bag["bag"] = corrected["bag"].values
        else:
            bag = bag_uncorrected.copy()

        # --- Region-level diagnostics: stage-1 OOF from a full-data fit ---
        region_metrics = _compute_region_metrics(x, ages, region_mapping, cfg)

        logger.info("Multivariate BAG estimation complete.")
        return BAGResult(
            bag=bag,
            bag_uncorrected=bag_uncorrected,
            predicted_age=predicted_age,
            region_metrics=region_metrics,
            config=cfg,
        )
