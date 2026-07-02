"""Session ↔ BAG store: load, merge, and serve pre-computed regional BAG results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .result import BAGResult

logger = logging.getLogger(__name__)

_ID_COLS = ["uid", "session_id"]


class BAGStore:
    """Access layer for pre-computed regional BAG results.

    Wraps ``<root_dir>/bag/`` written by the pipeline's BAG estimation step.
    Each result lives at ``bag/univariate/<feature>/`` or
    ``bag/multivariate/<combo>/`` and was saved by :meth:`BAGResult.save`.

    Parameters
    ----------
    root_dir:
        The ``data/processed/`` directory (parent of ``bag/``).
    metadata_path:
        Optional path to ``metadata.parquet``. Defaults to
        ``<root_dir>/metadata.parquet``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self._bag_dir = self.root_dir / "bag"
        self._meta_path = Path(metadata_path) if metadata_path else self.root_dir / "metadata.parquet"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list(self, kind: Optional[str] = None) -> list[str]:
        """List available BAG result names.

        Parameters
        ----------
        kind:
            ``"univariate"``, ``"multivariate"``, or ``None`` for both.

        Returns
        -------
        list[str]
            Names like ``"univariate/anat_thickness_mean_mm"`` or
            ``"multivariate/anat_thickness_mean_mm_DSIStudio_tensor_fa_mean"``.
        """
        if not self._bag_dir.exists():
            return []

        results = []
        for kind_dir in sorted(self._bag_dir.iterdir()):
            if not kind_dir.is_dir():
                continue
            if kind and kind_dir.name != kind:
                continue
            for result_dir in sorted(kind_dir.iterdir()):
                if (result_dir / "bag.parquet").exists():
                    results.append(f"{kind_dir.name}/{result_dir.name}")
        return results

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _meta(self) -> pd.DataFrame:
        if self._meta_path.exists():
            return pd.read_parquet(self._meta_path)
        return pd.DataFrame(columns=_ID_COLS)

    def _result_dir(self, name: str) -> Path:
        return self._bag_dir / name

    def load_result(self, name: str) -> BAGResult:
        """Load a raw :class:`BAGResult` by name."""
        d = self._result_dir(name)
        if not d.exists():
            raise FileNotFoundError(f"BAG result not found: {d}")
        return BAGResult.load(d)

    def load(
        self,
        name: str,
        include_metadata: bool = True,
        corrected: bool = True,
    ) -> pd.DataFrame:
        """Load BAG values as a wide DataFrame, optionally merged with metadata.

        Parameters
        ----------
        name:
            BAG result name (from :meth:`list`).
        include_metadata:
            Merge session metadata (age, sex, …) from ``metadata.parquet``.
        corrected:
            Use bias-corrected BAG (``bag.parquet``). Set ``False`` for raw.

        Returns
        -------
        pd.DataFrame
            Columns: ``[uid, session_id, <metadata cols>, <region cols>]``.
        """
        result = self.load_result(name)
        df = result.bag if corrected else result.bag_uncorrected

        if include_metadata:
            meta = self._meta()
            if not meta.empty:
                df = meta.merge(df, on=_ID_COLS, how="right")

        return df.reset_index(drop=True)

    def get_session(
        self,
        name: str,
        uid: str,
        session_id: str,
        corrected: bool = True,
    ) -> pd.Series:
        """Return a single session's regional BAG vector.

        Parameters
        ----------
        name:
            BAG result name.
        uid:
            Subject identifier.
        session_id:
            Session identifier.
        corrected:
            Use bias-corrected values.

        Returns
        -------
        pd.Series
            Index = region names, values = BAG floats.

        Raises
        ------
        KeyError
            If the session is not found.
        """
        result = self.load_result(name)
        src = result.bag if corrected else result.bag_uncorrected
        mask = (src["uid"] == uid) & (src["session_id"] == session_id)
        rows = src[mask]
        if rows.empty:
            raise KeyError(f"Session not found: uid={uid!r}, session_id={session_id!r}")
        region_cols = [c for c in src.columns if c not in _ID_COLS]
        return rows.iloc[0][region_cols]

    def bag_matrix(
        self,
        name: str,
        corrected: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return metadata DataFrame and BAG matrix for FAISS / retrieval.

        Parameters
        ----------
        name:
            BAG result name.
        corrected:
            Use bias-corrected values.

        Returns
        -------
        meta : pd.DataFrame
            Session metadata (uid, session_id, age, sex, …), one row per session,
            aligned to *matrix* row order.
        matrix : np.ndarray, shape (n_sessions, n_regions)
            Regional BAG values.
        """
        df = self.load(name, include_metadata=True, corrected=corrected)
        region_cols = [c for c in df.columns if c not in self._meta().columns or c in _ID_COLS]
        # region cols = everything except metadata non-id cols
        meta_cols_extra = [c for c in self._meta().columns if c not in _ID_COLS]
        non_region = set(_ID_COLS + meta_cols_extra)
        region_cols = [c for c in df.columns if c not in non_region]

        meta = df[[c for c in df.columns if c not in region_cols]].copy()
        matrix = df[region_cols].to_numpy(dtype=np.float32)
        return meta, matrix

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def region_metrics(self, name: str) -> pd.DataFrame:
        """Per-region quality metrics (r², MAE, correlation) for a BAG result."""
        return self.load_result(name).region_metrics

    def summary(self, name: str) -> dict:
        """Quick summary of a BAG result."""
        result = self.load_result(name)
        df = result.bag
        region_cols = [c for c in df.columns if c not in _ID_COLS]
        return {
            "name": name,
            "n_sessions": len(df),
            "n_regions": len(region_cols),
            "bag_mean": float(df[region_cols].values.mean()),
            "bag_std": float(df[region_cols].values.std()),
            "config": result.config.model_dump(),
        }
