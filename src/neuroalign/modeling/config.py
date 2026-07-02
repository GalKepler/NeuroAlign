"""Shared configuration for BAG estimation."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BAGConfig(BaseModel):
    """Configuration for Brain Age Gap estimation.

    Used by both univariate (per-region) and multivariate (joint) estimators.
    """

    splits: Literal["group_kfold", "loo"] = "group_kfold"
    n_splits: int = Field(default=5, ge=2, description="Number of GroupKFold splits")
    model_type: Literal["ridge", "xgboost", "lightgbm"] = "ridge"
    polynomial_degree: int = Field(
        default=2,
        ge=1,
        description="Polynomial features degree for metric values (ridge only)",
    )
    bias_correction: bool = True
    ipw: bool = True
    ipw_bandwidth: float = Field(
        default=2.0, gt=0, description="KDE bandwidth for inverse probability weighting"
    )
    random_state: int = 42
    n_jobs: int = Field(default=1, description="Parallel region fitting (-1 for all cores)")
    progress: bool = True

    # Column name mapping
    age_col: str = "age"
    sex_col: str = "sex"
    tiv_col: str = "tiv"
    subject_col: str = "subject_code"
    session_col: str = "session_id"
