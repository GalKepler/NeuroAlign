"""
Tabular Derivatives Loader for NeuroAlign
==========================================

Loads pre-parcellated anatomical (CAT12) and diffusion (QSIPrep/QSIRecon)
regional tables from `/mnt/62/Processed_Data/derivatives/tabular`. Replaces
`AnatomicalLoader` and `DiffusionLoader` (which parcellated NIfTI volumes
on the fly) - parcellation has already been done upstream.

Layout:
    sub-<UID>/
    ├── anat/atlas-<Atlas>/sub-<UID>_atlas-<Atlas>_structure-<...>.csv     # subject-level (template)
    ├── ses-<SESSIONID>/anat/atlas-<Atlas>/..._structure-<...>.csv
    ├── ses-<SESSIONID>.cross/anat/atlas-<Atlas>/..._structure-<...>.csv  # cross-sectional variant
    └── ses-<SESSIONID>{,.cross}/dwi/atlas-<Atlas>/..._diffmap.tsv

Example:
    >>> from neuroalign.data.loaders import TabularDerivativesLoader
    >>> loader = TabularDerivativesLoader("/mnt/62/Processed_Data/derivatives/tabular")
    >>> anat = loader.load_anatomical(sessions)  # sessions has uid + session_id columns
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd

from neuroalign.data.loaders.diffusion import parse_bids_entities

logger = logging.getLogger(__name__)

_STRUCTURE_RE = re.compile(r"structure-([A-Za-z0-9]+)")

# Rows for the medial wall / unparcellated background - not real regions, and
# absent from the diffusion atlases, so they're dropped on load.
_BACKGROUND_LABELS = {"Background+FreeSurfer_Defined_Medial_Wall"}


class TabularDerivativesLoader:
    """Loads long-format regional tables from pre-parcellated tabular derivatives."""

    def __init__(
        self,
        root: Path | str,
        atlas_name: str = "Schaefer2018N400n7Tian2020S2",
        anat_atlases: Tuple[str, ...] = ("Schaefer2018N400n7", "Tian2020S2"),
        session_variant: Literal["cross", "plain", "subject"] = "cross",
    ):
        """
        Args:
            root: Root of the tabular derivatives tree (`sub-<UID>/...`).
            atlas_name: Combined atlas label written to the `atlas` column
                of the output (matches the diffusion folder naming).
            anat_atlases: Anatomical atlas folder names to load and
                concatenate (cortex + subcortex) into `atlas_name`.
            session_variant: Which session directory to prefer:
                - ``"cross"`` (default): prefer `ses-<id>.cross/`, falling
                  back to `ses-<id>/`.
                - ``"plain"``: prefer `ses-<id>/`, falling back to
                  `ses-<id>.cross/`.
                - ``"subject"``: use the subject-level `sub-<uid>/anat/`
                  directory (no per-session breakdown; anatomical only).
        """
        self.root = Path(root)
        self.atlas_name = atlas_name
        self.anat_atlases = tuple(anat_atlases)
        if session_variant not in ("cross", "plain", "subject"):
            raise ValueError(f"invalid session_variant: {session_variant!r}")
        self.session_variant = session_variant

    def _session_dir(self, uid: str, session_id: str) -> Optional[Path]:
        """Resolve the directory holding `anat/`/`dwi/` for a session, per `session_variant`."""
        sub_dir = self.root / f"sub-{uid}"

        if self.session_variant == "subject":
            return sub_dir if sub_dir.exists() else None

        cross_dir = sub_dir / f"ses-{session_id}.cross"
        plain_dir = sub_dir / f"ses-{session_id}"
        preferred, fallback = (
            (cross_dir, plain_dir) if self.session_variant == "cross" else (plain_dir, cross_dir)
        )
        if preferred.exists():
            return preferred
        if fallback.exists():
            return fallback
        return None

    @staticmethod
    def _load_anat_atlas_csv(session_dir: Path, atlas: str) -> Optional[pd.DataFrame]:
        atlas_dir = session_dir / "anat" / f"atlas-{atlas}"
        csv_files = sorted(atlas_dir.glob("*.csv"))
        if not csv_files:
            return None

        df = pd.read_csv(csv_files[0])
        match = _STRUCTURE_RE.search(csv_files[0].name)
        df["structure"] = match.group(1) if match else None
        return df

    def load_anatomical(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Load long-format anatomical data for the given sessions.

        Args:
            sessions: DataFrame with `uid` and `session_id` columns
                (e.g. from `BehavioralLoader.get_sessions()`).

        Returns:
            Long-format DataFrame: `uid`, `session_id`, `atlas`, `structure`,
            `index`, `label`, `hemisphere`, plus the per-row metric columns
            (`gray_matter_volume_mm3`, `thickness_mean_mm`, `volume_mm3`,
            ..., `tiv_mm3`). Cortex and subcortex rows have different metric
            columns; missing values are NaN after concatenation.
        """
        frames = []
        for uid, session_id in (
            sessions[["uid", "session_id"]].drop_duplicates().itertuples(index=False)
        ):
            session_dir = self._session_dir(uid, session_id)
            if session_dir is None:
                continue

            for atlas in self.anat_atlases:
                df = self._load_anat_atlas_csv(session_dir, atlas)
                if df is None:
                    logger.debug("missing anat atlas %s for sub-%s ses-%s", atlas, uid, session_id)
                    continue

                df = df[~df["label"].isin(_BACKGROUND_LABELS)].copy()
                df.drop(columns=["subject_id"], errors="ignore", inplace=True)
                df.insert(0, "session_id", session_id)
                df.insert(0, "uid", uid)
                df["atlas"] = self.atlas_name
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _dwi_atlas_dir(self, uid: str, session_id: str) -> Optional[Path]:
        """Resolve the `dwi/atlas-<atlas_name>` directory for a session.

        Diffusion derivatives only exist under `ses-<id>/` (not `.cross/`),
        so this checks both regardless of `session_variant` (except that
        ``"plain"`` is tried first when requested).
        """
        sub_dir = self.root / f"sub-{uid}"
        cross_dwi = sub_dir / f"ses-{session_id}.cross" / "dwi" / f"atlas-{self.atlas_name}"
        plain_dwi = sub_dir / f"ses-{session_id}" / "dwi" / f"atlas-{self.atlas_name}"
        preferred, fallback = (
            (plain_dwi, cross_dwi) if self.session_variant == "plain" else (cross_dwi, plain_dwi)
        )
        if preferred.exists():
            return preferred
        if fallback.exists():
            return fallback
        return None

    def load_diffusion(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Load long-format diffusion data for the given sessions.

        Args:
            sessions: DataFrame with `uid` and `session_id` columns
                (e.g. from `BehavioralLoader.get_sessions()`).

        Returns:
            Long-format DataFrame: `uid`, `session_id`, `atlas`, `software`,
            `model`, `param`, `desc`, `index`, `label`, `hemisphere`, plus the
            per-region metric columns (`mean`, `std`, `median`, ..., `scalar`).
            One row per (session, software/model/param/desc, region).
        """
        frames = []
        for uid, session_id in (
            sessions[["uid", "session_id"]].drop_duplicates().itertuples(index=False)
        ):
            dwi_dir = self._dwi_atlas_dir(uid, session_id)
            if dwi_dir is None:
                logger.debug(
                    "missing dwi atlas %s for sub-%s ses-%s", self.atlas_name, uid, session_id
                )
                continue

            for tsv_file in sorted(dwi_dir.glob("*_diffmap.tsv")):
                entities = parse_bids_entities(tsv_file.name)
                df = pd.read_csv(tsv_file, sep="\t")
                df.insert(0, "session_id", session_id)
                df.insert(0, "uid", uid)
                df["atlas"] = self.atlas_name
                df["software"] = entities.get("software")
                df["model"] = entities.get("model")
                df["param"] = entities.get("param")
                df["desc"] = entities.get("desc")
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
