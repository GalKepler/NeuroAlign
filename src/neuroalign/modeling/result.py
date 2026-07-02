"""Shared result container for BAG estimation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import BAGConfig

logger = logging.getLogger(__name__)


@dataclass
class BAGResult:
    """Container for BAG estimation results.

    Attributes
    ----------
    bag : pd.DataFrame
        Bias-corrected BAG values (sessions x regions, wide format).
    bag_uncorrected : pd.DataFrame
        Raw BAG values before bias correction.
    predicted_age : pd.DataFrame
        Predicted brain age per region (sessions x regions).
    region_metrics : pd.DataFrame
        Per-region quality metrics (r2, mae, correlation).
    config : BAGConfig
        Configuration used to produce these results.
    """

    bag: pd.DataFrame
    bag_uncorrected: pd.DataFrame
    predicted_age: pd.DataFrame
    region_metrics: pd.DataFrame
    config: BAGConfig

    def save(self, output_dir: Path) -> None:
        """Save all result components to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving BAG result to %s", output_dir)

        self.bag.to_parquet(output_dir / "bag.parquet", index=False)
        self.bag_uncorrected.to_parquet(output_dir / "bag_uncorrected.parquet", index=False)
        self.predicted_age.to_parquet(output_dir / "predicted_age.parquet", index=False)
        self.region_metrics.to_parquet(output_dir / "region_metrics.parquet", index=False)

        config_path = output_dir / "config.json"
        config_path.write_text(self.config.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> BAGResult:
        """Load a previously saved BAGResult from *path*."""
        path = Path(path)
        logger.info("Loading BAG result from %s", path)

        config = BAGConfig(**json.loads((path / "config.json").read_text()))

        return cls(
            bag=pd.read_parquet(path / "bag.parquet"),
            bag_uncorrected=pd.read_parquet(path / "bag_uncorrected.parquet"),
            predicted_age=pd.read_parquet(path / "predicted_age.parquet"),
            region_metrics=pd.read_parquet(path / "region_metrics.parquet"),
            config=config,
        )
