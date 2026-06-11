"""Result containers for brain twins and regional BAG interpretation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class InterpretationResult:
    """Top-level LLM interpretation of brain twins."""

    query_subject: str
    query_session: str
    narrative: str
    key_shared_traits: list[str]
    cluster_profile: str
    raw_response: str
    n_twins_used: int
    # Per-scale structured insights (new field, defaults to empty list)
    scale_highlights: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"InterpretationResult(subject={self.query_subject!r}, "
            f"n_twins={self.n_twins_used})"
        )


@dataclass
class NetworkStat:
    """BAG statistics for a single brain network or structure."""

    name: str
    display_name: str
    mean_z: float
    n_parcels: int
    is_subcortical: bool

    @property
    def direction(self) -> str:
        return "older" if self.mean_z > 0 else "younger"

    @property
    def abs_z(self) -> float:
        return abs(self.mean_z)


@dataclass
class RegionalInterpretationResult:
    """LLM interpretation of a participant's regional BAG profile."""

    query_subject: str
    query_session: str
    narrative: str
    network_highlights: list[dict]
    overall_summary: str
    network_stats: list[NetworkStat]
    raw_response: str

    def to_stats_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "network": s.name,
                "display_name": s.display_name,
                "mean_z": s.mean_z,
                "abs_z": s.abs_z,
                "direction": s.direction,
                "n_parcels": s.n_parcels,
                "is_subcortical": s.is_subcortical,
            }
            for s in self.network_stats
        ]).sort_values("abs_z", ascending=False)

    def __repr__(self) -> str:
        return (
            f"RegionalInterpretationResult(subject={self.query_subject!r}, "
            f"n_networks={len(self.network_stats)})"
        )
