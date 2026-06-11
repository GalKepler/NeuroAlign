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


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    paths: DataPaths
    modalities: ModalityConfig = Field(default_factory=ModalityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    atlas_name: str = "Schaefer2018N400n7Tian2020S2"
    anat_atlases: Tuple[str, str] = ("Schaefer2018N400n7", "Tian2020S2")
    session_variant: Literal["cross", "plain", "subject"] = "cross"
    labs: Optional[List[str]] = Field(None, description="Restrict to these brainlink labs")
    require_complete_mapping: bool = True
    force: bool = False  # If False, skip sessions already in the store
