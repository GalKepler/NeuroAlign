"""
Long-to-wide format transformers for neuroimaging data.
"""

from typing import List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AnatomicalWideTransformer:
    """
    Transform anatomical long-format data to wide-format.

    Input columns: label, modality, metric, volume_mm3, mean, subject_code, session_id
    Output: One row per subject-session with columns like gm_volume_LH_Vis_1, ...
    """

    def __init__(
        self,
        modalities: Optional[List[str]] = None,
    ):
        """
        Initialize transformer.

        Args:
            modalities: List of modalities to include (default: ["gm", "wm", "ct"])
        """
        self.modalities = modalities or ["gm", "wm", "ct"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform long-format anatomical data to wide format.

        Args:
            df: Long-format DataFrame from AnatomicalLoader

        Returns:
            Wide-format DataFrame with one row per subject-session
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # Filter to requested modalities
        df = df[df["modality"].isin(self.modalities)].copy()

        if len(df) == 0:
            logger.warning("No data remaining after modality filter")
            return pd.DataFrame()

        # Create feature name: {modality}_{metric}_{label}
        df["feature_name"] = df["modality"] + "_" + df["metric"] + "_" + df["label"]

        # Select value column based on metric type
        # For volume: use volume_mm3, for thickness: use mean
        df["value"] = df.apply(
            lambda row: row["volume_mm3"] if row["metric"] == "volume" else row["mean"],
            axis=1,
        )

        # Pivot to wide format
        wide_df = df.pivot_table(
            index=["subject_code", "session_id"],
            columns="feature_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Flatten column names if MultiIndex
        if isinstance(wide_df.columns, pd.MultiIndex):
            wide_df.columns = [
                "_".join(str(c) for c in col).strip("_") if isinstance(col, tuple) else col
                for col in wide_df.columns
            ]

        logger.info(
            f"Transformed anatomical data: {len(wide_df)} sessions, "
            f"{len(wide_df.columns) - 2} features"
        )

        return wide_df


class DiffusionWideTransformer:
    """
    Transform diffusion long-format data to wide-format.

    Input columns: name (region), mean (value), workflow, model, param, subject_code, session_id
    Output: One row per subject-session with columns like AMICONODDI_NODDI_ICVF_LH_Vis_1, ...
    """

    def __init__(
        self,
        workflows: Optional[List[str]] = None,
        region_col: str = "name",
        value_col: str = "mean",
    ):
        """
        Initialize transformer.

        Args:
            workflows: List of workflows to include (default: all)
            region_col: Column name for region identifier
            value_col: Column name for the value to use
        """
        self.workflows = workflows
        self.region_col = region_col
        self.value_col = value_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform long-format diffusion data to wide format.

        Args:
            df: Long-format DataFrame from DiffusionLoader

        Returns:
            Wide-format DataFrame with one row per subject-session
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()

        df = df.copy()

        # Filter to requested workflows if specified
        if self.workflows is not None:
            df = df[df["workflow"].isin(self.workflows)]

        if len(df) == 0:
            logger.warning("No data remaining after workflow filter")
            return pd.DataFrame()

        # Determine region column (may be 'name', 'label', or 'index')
        region_col = self.region_col
        if region_col not in df.columns:
            # Try alternatives
            for alt in ["label", "name", "region"]:
                if alt in df.columns:
                    region_col = alt
                    break
            else:
                # Use index if nothing else
                if "index" in df.columns:
                    region_col = "index"
                else:
                    raise ValueError(
                        f"Could not find region column. Available: {df.columns.tolist()}"
                    )

        # Determine value column
        value_col = self.value_col
        if value_col not in df.columns:
            for alt in ["mean", "Mean", "value"]:
                if alt in df.columns:
                    value_col = alt
                    break
            else:
                raise ValueError(
                    f"Could not find value column '{self.value_col}'. "
                    f"Available: {df.columns.tolist()}"
                )

        # Create feature name: {workflow}_{model}_{param}_{region}
        df["feature_name"] = (
            df["workflow"]
            + "_"
            + df["model"]
            + "_"
            + df["param"]
            + "_"
            + df[region_col].astype(str)
        )

        # Pivot to wide format
        wide_df = df.pivot_table(
            index=["subject_code", "session_id"],
            columns="feature_name",
            values=value_col,
            aggfunc="first",
        ).reset_index()

        # Flatten column names if MultiIndex
        if isinstance(wide_df.columns, pd.MultiIndex):
            wide_df.columns = [
                "_".join(str(c) for c in col).strip("_") if isinstance(col, tuple) else col
                for col in wide_df.columns
            ]

        logger.info(
            f"Transformed diffusion data: {len(wide_df)} sessions, "
            f"{len(wide_df.columns) - 2} features"
        )

        return wide_df
