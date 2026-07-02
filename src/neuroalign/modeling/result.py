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
    regional_bag : pd.DataFrame, optional
        Per-session, per-region BAG (wide: `[uid, session_id, <432 region cols>]`).
        Only produced by `MultivariateRegionalBAGEstimator` (stage-1 OOF predicted
        age minus chronological age, bias-corrected).
    regional_bag_uncorrected : pd.DataFrame, optional
        Same as `regional_bag` before bias correction.
    regional_contribution : pd.DataFrame, optional
        Per-session, per-region signed contribution to the overall predicted age
        from the (linear) meta-learner - `coef_[r] * standardized(oof[:, r])`,
        summing to `prediction - intercept_`.
    """

    bag: pd.DataFrame
    bag_uncorrected: pd.DataFrame
    predicted_age: pd.DataFrame
    region_metrics: pd.DataFrame
    config: BAGConfig
    regional_bag: pd.DataFrame | None = None
    regional_bag_uncorrected: pd.DataFrame | None = None
    regional_contribution: pd.DataFrame | None = None

    def save(self, output_dir: Path) -> None:
        """Save all result components to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving BAG result to %s", output_dir)

        self.bag.to_parquet(output_dir / "bag.parquet", index=False)
        self.bag_uncorrected.to_parquet(output_dir / "bag_uncorrected.parquet", index=False)
        self.predicted_age.to_parquet(output_dir / "predicted_age.parquet", index=False)
        self.region_metrics.to_parquet(output_dir / "region_metrics.parquet", index=False)

        if self.regional_bag is not None:
            self.regional_bag.to_parquet(output_dir / "regional_bag.parquet", index=False)
        if self.regional_bag_uncorrected is not None:
            self.regional_bag_uncorrected.to_parquet(
                output_dir / "regional_bag_uncorrected.parquet", index=False
            )
        if self.regional_contribution is not None:
            self.regional_contribution.to_parquet(
                output_dir / "regional_contribution.parquet", index=False
            )

        config_path = output_dir / "config.json"
        config_path.write_text(self.config.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> BAGResult:
        """Load a previously saved BAGResult from *path*."""
        path = Path(path)
        logger.info("Loading BAG result from %s", path)

        config = BAGConfig(**json.loads((path / "config.json").read_text()))

        def _load_optional(name: str) -> pd.DataFrame | None:
            p = path / name
            return pd.read_parquet(p) if p.exists() else None

        return cls(
            bag=pd.read_parquet(path / "bag.parquet"),
            bag_uncorrected=pd.read_parquet(path / "bag_uncorrected.parquet"),
            predicted_age=pd.read_parquet(path / "predicted_age.parquet"),
            region_metrics=pd.read_parquet(path / "region_metrics.parquet"),
            config=config,
            regional_bag=_load_optional("regional_bag.parquet"),
            regional_bag_uncorrected=_load_optional("regional_bag_uncorrected.parquet"),
            regional_contribution=_load_optional("regional_contribution.parquet"),
        )
