"""Univariate regional BAG estimation.

For each brain region independently, predicts chronological age from that
region's metric value (+ covariates), then computes
BAG = predicted_age - actual_age with optional post-hoc bias correction.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import PolynomialFeatures

from neuroalign.modeling._shared import apply_bias_correction, compute_ipw_weights
from neuroalign.modeling.config import BAGConfig
from neuroalign.modeling.result import BAGResult

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_model(config: BAGConfig) -> RegressorMixin:
    """Model factory: returns an unfitted scikit-learn-compatible estimator."""
    logger.debug("Building model: %s", config.model_type)
    if config.model_type == "ridge":
        return Ridge(alpha=1.0, random_state=config.random_state)

    if config.model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=config.random_state,
            verbosity=0,
        )

    if config.model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=config.random_state,
            verbose=-1,
        )

    raise ValueError(f"Unknown model_type: {config.model_type!r}")


def _fit_region(
    region: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sample_weight: np.ndarray | None,
    config: BAGConfig,
) -> np.ndarray:
    """Fit a single region model and return test predictions."""
    model = _build_model(config)
    fit_kwargs: dict = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, **fit_kwargs)
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RegionalBAGEstimator:
    """Univariate regional Brain Age Gap estimator.

    For each region, trains an independent model to predict chronological age
    from that region's metric value and covariates (sex, TIV).  BAG is computed
    as ``predicted_age - actual_age`` with optional bias correction.

    Parameters
    ----------
    config : BAGConfig
        Estimation configuration.
    """

    def __init__(self, config: BAGConfig | None = None) -> None:
        self.config = config or BAGConfig()

    # ---- main entry point ------------------------------------------------

    def fit_predict(
        self,
        features: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> BAGResult:
        """Run cross-validated BAG estimation for all regions.

        Parameters
        ----------
        features : pd.DataFrame
            Wide-format feature matrix with columns
            ``[subject_col, session_col, region_1, ..., region_N]``.
        metadata : pd.DataFrame
            Must contain ``[subject_col, session_col, age_col, sex_col, tiv_col]``.

        Returns
        -------
        BAGResult
        """
        logger.info("Starting regional BAG estimation.")
        cfg = self.config
        id_cols = [cfg.subject_col, cfg.session_col]

        # --- 1. Merge & validate ---
        logger.info("Input features: %d rows, %d columns.", features.shape[0], features.shape[1])
        logger.info("Input metadata: %d rows, %d columns.", metadata.shape[0], metadata.shape[1])
        df = features.merge(metadata, on=id_cols, how="inner")
        required = [cfg.age_col, cfg.sex_col, cfg.tiv_col]
        df = df.dropna(subset=required)
        logger.info("Merged data: %d rows, %d columns.", df.shape[0], df.shape[1])

        # Encode sex as 0/1
        sex_values = df[cfg.sex_col]
        if sex_values.dtype == object or sex_values.dtype.name == "category":
            unique = sorted(sex_values.unique())
            sex_map = {v: i for i, v in enumerate(unique)}
            logger.info("Encoding sex: %s", sex_map)
            df["_sex_encoded"] = sex_values.map(sex_map).astype(float)
        else:
            df["_sex_encoded"] = sex_values.astype(float)

        # --- 2. Identify region columns ---
        regions = [c for c in features.columns if c not in id_cols]
        if not regions:
            raise ValueError("No region columns found in features DataFrame.")
        logger.info("Fitting %d regions across %d sessions.", len(regions), len(df))

        # --- 3. Prepare arrays ---
        logger.debug("Preparing data arrays for cross-validation.")
        ages = df[cfg.age_col].values
        groups = df[cfg.subject_col].values
        sex_enc = df["_sex_encoded"].values
        tiv = df[cfg.tiv_col].values

        # Pre-allocate prediction storage (sessions x regions)
        n = len(df)
        pred_matrix = np.full((n, len(regions)), np.nan)

        # --- 4. GroupKFold cross-validation ---
        # gkf = GroupKFold(n_splits=cfg.n_splits)
        if cfg.splits == "group_kfold":
            gkf = GroupKFold(n_splits=cfg.n_splits)
        elif cfg.splits == "loo":
            gkf = LeaveOneGroupOut()
        use_poly = cfg.model_type == "ridge"

        if cfg.progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                tqdm = None  # type: ignore[assignment]
        else:
            tqdm = None  # type: ignore[assignment]

        if cfg.n_splits > len(np.unique(groups)):
            raise ValueError(
                f"n_splits={cfg.n_splits} cannot be greater than "
                f"the number of unique subjects={len(np.unique(groups))}."
            )
        if cfg.splits == "loo":
            logger.info("Using Leave-One-Group-Out cross-validation.")
            n_splits = len(np.unique(groups))
        else:
            n_splits = cfg.n_splits
        logger.info("Starting %d-fold cross-validation.", n_splits)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
            logger.debug("Fold %d: train=%d, test=%d", fold_idx, len(train_idx), len(test_idx))

            train_ages = ages[train_idx]

            # IPW weights on training set
            if cfg.ipw:
                weights = compute_ipw_weights(train_ages, cfg.ipw_bandwidth)
            else:
                weights = None

            iterator = enumerate(regions)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=len(regions),
                    desc=f"Fold {fold_idx + 1}/{n_splits}",
                    leave=False,
                )

            for region_idx, region in iterator:
                metric_train = df[region].values[train_idx]
                metric_test = df[region].values[test_idx]

                valid_train = ~np.isnan(metric_train)
                valid_test = ~np.isnan(metric_test)

                if valid_train.sum() < 2:
                    continue

                m_train = metric_train[valid_train]
                m_test = metric_test[valid_test]
                t_ages = train_ages[valid_train]
                t_weights = weights[valid_train] if weights is not None else None
                s_train = sex_enc[train_idx][valid_train]
                s_test = sex_enc[test_idx][valid_test]
                v_train = tiv[train_idx][valid_train]
                v_test = tiv[test_idx][valid_test]

                # Build feature matrices: [metric, sex, tiv]
                # Polynomial expansion on metric only (for ridge)
                if use_poly and cfg.polynomial_degree > 1:
                    poly = PolynomialFeatures(degree=cfg.polynomial_degree, include_bias=False)
                    metric_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
                    metric_test_poly = poly.transform(m_test.reshape(-1, 1))
                else:
                    metric_train_poly = m_train.reshape(-1, 1)
                    metric_test_poly = m_test.reshape(-1, 1)

                X_train = np.column_stack([metric_train_poly, s_train, v_train])
                X_test = np.column_stack([metric_test_poly, s_test, v_test])

                preds = _fit_region(region, X_train, t_ages, X_test, t_weights, cfg)
                pred_matrix[test_idx[valid_test], region_idx] = preds
        logger.info("Cross-validation finished.")

        # --- 5. Assemble results ---
        logger.info("Assembling results.")
        predicted_age = pd.DataFrame(pred_matrix, columns=regions)
        predicted_age.insert(0, cfg.session_col, df[cfg.session_col].values)
        predicted_age.insert(0, cfg.subject_col, df[cfg.subject_col].values)

        bag_uncorrected_vals = pred_matrix - ages[:, np.newaxis]
        bag_uncorrected = pd.DataFrame(bag_uncorrected_vals, columns=regions)
        bag_uncorrected.insert(0, cfg.session_col, df[cfg.session_col].values)
        bag_uncorrected.insert(0, cfg.subject_col, df[cfg.subject_col].values)

        # --- 6. Bias correction ---
        if cfg.bias_correction:
            logger.info("Applying bias correction.")
            bag_corrected_vals = apply_bias_correction(
                pd.DataFrame(bag_uncorrected_vals, columns=regions), ages
            )
            bag = bag_corrected_vals.copy()
        else:
            logger.info("Skipping bias correction.")
            bag = pd.DataFrame(bag_uncorrected_vals, columns=regions)

        bag.insert(0, cfg.session_col, df[cfg.session_col].values)
        bag.insert(0, cfg.subject_col, df[cfg.subject_col].values)

        # --- 7. Region metrics ---
        logger.info("Computing region-wise metrics.")
        metrics_rows = []
        for region_idx, region in enumerate(regions):
            pred = pred_matrix[:, region_idx]
            valid = ~np.isnan(pred)
            if valid.sum() < 2:
                metrics_rows.append({"region": region, "r2": np.nan, "mae": np.nan, "correlation": np.nan})
                continue
            r2 = r2_score(ages[valid], pred[valid])
            mae = mean_absolute_error(ages[valid], pred[valid])
            corr, _ = pearsonr(ages[valid], pred[valid])
            metrics_rows.append({"region": region, "r2": r2, "mae": mae, "correlation": corr})
        region_metrics = pd.DataFrame(metrics_rows)

        logger.info("BAG estimation complete.")
        return BAGResult(
            bag=bag,
            bag_uncorrected=bag_uncorrected,
            predicted_age=predicted_age,
            region_metrics=region_metrics,
            config=cfg,
        )

    # ---- utilities -------------------------------------------------------

    @staticmethod
    def long_to_wide(
        long_df: pd.DataFrame,
        metric_col: str,
        region_col: str = "label",
        id_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Pivot a long-format feature DataFrame to wide (sessions x regions).

        Parameters
        ----------
        long_df : pd.DataFrame
            Long-format data with one row per (session, region).
        metric_col : str
            Column containing the metric value to pivot.
        region_col : str
            Column identifying the brain region.
        id_cols : list[str] | None
            Columns identifying a session. Defaults to
            ``["subject_code", "session_id"]``.

        Returns
        -------
        pd.DataFrame
            Wide DataFrame with ``id_cols`` + one column per region.
        """
        if id_cols is None:
            id_cols = ["subject_code", "session_id"]

        wide = long_df.pivot_table(
            index=id_cols,
            columns=region_col,
            values=metric_col,
            aggfunc="first",
        ).reset_index()

        wide.columns.name = None
        return wide
