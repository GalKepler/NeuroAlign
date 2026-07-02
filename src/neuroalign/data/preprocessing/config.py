"""
Configuration models for the data preparation pipeline.
"""

from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """Paths configuration for data sources."""

    brainlink_db: Path = Field(..., description="Path to the brainlink SQLite DB")
    tabular_derivatives_root: Path = Field(
        ..., description="Root of the pre-parcellated tabular derivatives tree"
    )
    output_dir: Path = Field(default=Path("data/processed"), description="Output directory")


class ModalityConfig(BaseModel):
    """Configuration for which modalities to include."""

    anatomical: bool = True
    diffusion: bool = True


class OutputConfig(BaseModel):
    """Output configuration."""

    prefix: str = "neuroalign"
    compression: Optional[str] = "snappy"


class BAGEstimationConfig(BaseModel):
    """Configuration for automatic regional Brain Age Gap (BAG) estimation.

    Column mapping (age/sex/tiv/subject/session) is fixed to the
    `FeatureStore` schema (``AGE``, ``sex``, ``tiv_mm3``, ``uid``,
    ``session_id``) and is not configurable here; the remaining fields
    mirror `neuroalign.modeling.config.BAGConfig`.
    """

    enabled: bool = False
    univariate_features: List[str] = Field(
        default_factory=lambda: ["anat_thickness_mean_mm"],
        description="Wide-format feature names for per-region univariate BAG estimation",
    )
    multivariate_feature_sets: List[List[str]] = Field(
        default_factory=list,
        description="Combinations of wide-format feature names for multivariate BAG estimation",
    )
    multivariate_tiv_normalize: List[str] = Field(
        default_factory=list,
        description="Feature names to divide by TIV before multivariate BAG estimation",
    )

    splits: Literal["group_kfold", "loo"] = "group_kfold"
    n_splits: int = Field(default=5, ge=2)
    model_type: Literal["ridge", "xgboost", "lightgbm"] = "ridge"
    polynomial_degree: int = Field(default=2, ge=1)
    bias_correction: bool = True
    ipw: bool = True
    ipw_bandwidth: float = Field(default=2.0, gt=0)
    random_state: int = 42
    n_jobs: int = 1
    progress: bool = True
    min_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of metadata sessions a feature must cover "
            "(after merge) to run BAG estimation; sparser features are skipped."
        ),
    )


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    paths: DataPaths
    modalities: ModalityConfig = Field(default_factory=ModalityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    bag_estimation: BAGEstimationConfig = Field(default_factory=BAGEstimationConfig)
    atlas_name: str = "Schaefer2018N400n7Tian2020S2"
    anat_atlases: Tuple[str, str] = ("Schaefer2018N400n7", "Tian2020S2")
    session_variant: Literal["cross", "plain", "subject"] = "cross"
    labs: Optional[List[str]] = Field(None, description="Restrict to these brainlink labs")
    require_complete_mapping: bool = True
    force: bool = False  # If False, skip sessions already in the store
