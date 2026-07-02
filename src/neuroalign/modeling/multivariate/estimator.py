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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuroalign.data.preprocessing import FeatureStore
from neuroalign.data.preprocessing.feature_store import META_COLS
from neuroalign.modeling._shared import apply_bias_correction, compute_ipw_weights
from neuroalign.modeling.config import BAGConfig
from neuroalign.modeling.result import BAGResult

logger = logging.getLogger(__name__)


def _build_pipeline() -> Pipeline:
    """Median-impute + scale + RidgeCV, so per-region NaNs (missing coverage) don't crash the fit.

    RidgeCV (not a fixed-alpha Ridge) tunes regularization per region/fold via
    internal CV — an untuned alpha=1 leaves regions with collinear/high-dim
    feature blocks under-regularized, which is what produced physically
    impossible age predictions (e.g. large negative ages) downstream.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-5, 5, 20))),
        ]
    )


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
            base_estimator=_build_pipeline(),
            meta_estimator=_build_pipeline(),
            outer_cv=max(2, min(stage1_cv, len(train_idx))),
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            allow_nan=True,
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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Per-region stage-1 (base learner) OOF diagnostics from a full-data fit.

    Also returns the raw per-session x per-region stage-1 OOF predicted-age
    matrix and the fitted linear meta-learner's per-region contribution to the
    overall (scalar) predicted age - both are discarded by the caller unless
    asked for, so a single full-data fit serves both diagnostics and the
    regional BAG / contribution outputs.
    """
    n_subjects_proxy = max(2, min(5, len(ages)))
    logger.info("Fitting full-data stacker for region diagnostics.")
    full_weight = compute_ipw_weights(ages, cfg.ipw_bandwidth) if cfg.ipw else None
    full_stacker = RegionalStackingRegressor(
        region_mapping=region_mapping,
        base_estimator=_build_pipeline(),
        meta_estimator=_build_pipeline(),
        outer_cv=n_subjects_proxy,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        allow_nan=True,
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

    # Per-region contribution to the overall predicted age: the meta-learner is
    # a linear pipeline (impute + scale + RidgeCV), so
    # contribution[:, r] = coef_[r] * scaled_oof[:, r], and
    # contribution.sum(axis=1) + intercept_ == meta_pipe.predict(oof_predictions_).
    meta_pipe = full_stacker.meta_estimator_
    scaled_oof = meta_pipe.named_steps["scaler"].transform(
        meta_pipe.named_steps["imputer"].transform(full_stacker.oof_predictions_)
    )
    ridge = meta_pipe.named_steps["model"]
    contribution = scaled_oof * ridge.coef_[np.newaxis, :]

    return (
        pd.DataFrame(metrics_rows),
        full_stacker.oof_predictions_,
        contribution,
        full_stacker.region_names_,
    )


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
        tiv_normalize: Sequence[str] | None = None,
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
        _tiv_normalize = set(tiv_normalize or [])
        tiv = None
        if _tiv_normalize:
            tiv_df = store.load_tiv().dropna(subset=["tiv_mm3"])
            dupes = tiv_df.duplicated(subset=META_COLS).sum()
            if dupes:
                logger.warning(
                    "Dropping %d duplicate (uid, session_id) row(s) from TIV table.", dupes
                )
                tiv_df = tiv_df.drop_duplicates(subset=META_COLS, keep="first")
            tiv = tiv_df.set_index(META_COLS)["tiv_mm3"]

        tables = {}
        for name in feature_names:
            df = store.load_feature(name, include_metadata=False).set_index(META_COLS)
            if name in _tiv_normalize:
                df = df.div(tiv, axis=0)
            tables[name] = df

        # Keep sessions present in >=50% of tables (not just the full intersection);
        # missing tables become NaN rows, imputed downstream by the per-region pipeline.
        n_tables = len(tables)
        dfs = list(tables.values())
        union_index = dfs[0].index
        for df in dfs[1:]:
            union_index = union_index.union(df.index)
        coverage = pd.Series(0, index=union_index)
        for df in dfs:
            coverage.loc[df.index] += 1
        keep_sessions = coverage[coverage >= n_tables / 2].index.sort_values()
        dropped = len(coverage) - len(keep_sessions)
        if dropped:
            logger.warning(
                "Dropping %d session(s) present in fewer than 50%% of feature tables.", dropped
            )
        tables = {name: df.reindex(keep_sessions) for name, df in tables.items()}

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

        n_nan = int(np.isnan(x).sum())
        if n_nan:
            logger.warning(
                "Feature matrix contains %d NaN value(s) after combining tables; "
                "median-imputed inside each fold.",
                n_nan,
            )

        ages = meta[cfg.age_col].to_numpy(dtype=float)
        uids = meta.index.get_level_values("uid").to_numpy()

        pred = _run_outer_cv(x, ages, uids, region_mapping, cfg)

        # Snapshot before the implausible-prediction filter below reassigns `sessions` -
        # the regional matrices are keyed to this full (x, ages) session set.
        full_sessions = list(sessions)

        implausible = (pred < 0) | (pred > 120)
        n_implausible = int(implausible.sum())
        if n_implausible:
            logger.warning(
                "Dropping %d session(s) with physically implausible predicted age "
                "(outside [0, 120]) - likely a data-quality issue in the underlying "
                "feature values.",
                n_implausible,
            )
            keep = ~implausible
            pred = pred[keep]
            kept_ages = ages[keep]
            sessions = [s for s, k in zip(sessions, keep, strict=True) if k]
        else:
            kept_ages = ages

        ids = pd.DataFrame(sessions, columns=META_COLS)

        predicted_age = ids.copy()
        predicted_age["predicted_age"] = pred

        bag_uncorrected = ids.copy()
        bag_uncorrected["bag"] = pred - kept_ages

        if cfg.bias_correction:
            logger.info("Applying bias correction.")
            corrected = apply_bias_correction(bag_uncorrected[["bag"]], kept_ages)
            bag = ids.copy()
            bag["bag"] = corrected["bag"].values
        else:
            bag = bag_uncorrected.copy()

        # --- Region-level diagnostics + regional BAG / contribution: full-data fit ---
        region_metrics, oof_predictions, contribution, region_names = _compute_region_metrics(
            x, ages, region_mapping, cfg
        )

        region_ids = pd.DataFrame(full_sessions, columns=META_COLS)

        def _wide(matrix: np.ndarray) -> pd.DataFrame:
            return pd.concat([region_ids, pd.DataFrame(matrix, columns=region_names)], axis=1)

        regional_bag_uncorrected = _wide(oof_predictions - ages[:, np.newaxis])

        if cfg.bias_correction:
            corrected = apply_bias_correction(regional_bag_uncorrected[region_names], ages)
            regional_bag = _wide(corrected.values)
        else:
            regional_bag = regional_bag_uncorrected.copy()

        regional_contribution = _wide(contribution)

        logger.info("Multivariate BAG estimation complete.")
        return BAGResult(
            bag=bag,
            bag_uncorrected=bag_uncorrected,
            predicted_age=predicted_age,
            region_metrics=region_metrics,
            config=cfg,
            regional_bag=regional_bag,
            regional_bag_uncorrected=regional_bag_uncorrected,
            regional_contribution=regional_contribution,
        )
