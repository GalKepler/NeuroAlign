"""Shared helpers for BAG estimation (univariate and multivariate)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


def compute_ipw_weights(ages: np.ndarray, bandwidth: float) -> np.ndarray:
    """KDE-based inverse probability weights normalised to mean 1."""
    logger.debug("Computing IPW weights with bandwidth=%.2f", bandwidth)
    kde = KernelDensity(bandwidth=bandwidth).fit(ages.reshape(-1, 1))
    log_density = kde.score_samples(ages.reshape(-1, 1))
    density = np.exp(log_density)
    weights = 1.0 / density
    weights /= weights.mean()
    return weights


def apply_bias_correction(bag_df: pd.DataFrame, ages: np.ndarray) -> pd.DataFrame:
    """De Lange & Cole post-hoc linear bias correction per column.

    For each column, fits ``BAG = alpha * age + beta`` and subtracts the
    fitted values so that the corrected BAG is uncorrelated with age.
    """
    logger.debug("Applying linear bias correction to %d column(s).", len(bag_df.columns))
    corrected = bag_df.copy()
    for col in bag_df.columns:
        bag_vals = bag_df[col].values
        a_matrix = np.column_stack([ages, np.ones_like(ages)])
        coeffs, *_ = np.linalg.lstsq(a_matrix, bag_vals, rcond=None)
        corrected[col] = bag_vals - (coeffs[0] * ages + coeffs[1])
    return corrected
