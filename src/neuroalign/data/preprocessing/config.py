"""
Configuration models for the data preparation pipeline.
"""

from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """Paths configuration for data sources."""

    sessions_csv: Path = Field(..., description="Path to sessions CSV with AGE column")
    cat12_root: Optional[Path] = Field(None, description="CAT12 derivatives root")
    atlas_root: Optional[Path] = Field(None, description="Atlas directory root")
    qsiparc_path: Optional[Path] = Field(None, description="QSIParc derivatives path")
    qsirecon_path: Optional[Path] = Field(None, description="QSIRecon derivatives path")
    output_dir: Path = Field(default=Path("data/processed"), description="Output directory")


class ModalityConfig(BaseModel):
    """Configuration for which modalities to include."""

    anatomical: bool = True
    diffusion: bool = True

    # Anatomical sub-modalities
    gray_matter: bool = True
    white_matter: bool = True
    cortical_thickness: bool = True

    # Diffusion workflows (None = all available)
    diffusion_workflows: Optional[List[str]] = None


class OutputConfig(BaseModel):
    """Output configuration."""

    prefix: str = "neuroalign"
    compression: Optional[str] = "snappy"


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    paths: DataPaths
    modalities: ModalityConfig = Field(default_factory=ModalityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    atlas_name: str = "4S456Parcels"
    age_column: str = "AGE"  # Column name for age in sessions CSV
    n_jobs: int = 1  # Number of parallel workers (1 = serial)
    progress: bool = True
    force: bool = False  # If False, skip sessions already in the store
