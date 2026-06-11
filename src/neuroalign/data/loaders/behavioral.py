"""
Behavioral Data Loader for NeuroAlign
======================================

Thin wrapper around `brainlink.BrainLinkDB`, providing the canonical session
list (uid / subject_code / session_id + demographics) and questionnaire data
used as the entry point for the data preparation pipeline.

Example:
    >>> from neuroalign.data.loaders import BehavioralLoader
    >>> loader = BehavioralLoader("/media/storage/brainlink/brainlink.db")
    >>> sessions = loader.get_sessions()
    >>> sessions.columns
    Index(['session_id', 'uid', 'subject_code', ..., 'AGE', 'sex', ...])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from brainlink import BrainLinkDB


class BehavioralLoader:
    """Wraps `BrainLinkDB` to provide pipeline-ready session/questionnaire tables."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._db = BrainLinkDB(str(self.db_path))

    def get_sessions(
        self,
        labs: Optional[Sequence[str]] = None,
        require_imaging: Optional[Sequence[str]] = None,
        require_complete_mapping: bool = True,
    ) -> pd.DataFrame:
        """Return the canonical session list with demographics.

        Args:
            labs: Restrict to these labs (e.g. ``["YA", "SNBB"]``).
            require_imaging: Only include sessions with at least one file
                for each listed imaging pipeline (e.g. ``["bids"]``).
            require_complete_mapping: Only include sessions with a complete
                uid / subject_code / session_id mapping.

        Returns:
            One row per session, including ``uid``, ``subject_code``,
            ``session_id``, ``AGE`` (renamed from ``age_at_scan``), ``sex``,
            and other demographics columns.
        """
        sessions = self._db.query(
            labs=list(labs) if labs is not None else None,
            include=["demographics"],
            require_complete_mapping=require_complete_mapping,
            require_imaging=list(require_imaging) if require_imaging is not None else None,
            styled=False,
        )
        return sessions.rename(columns={"age_at_scan": "AGE"})

    def get_questionnaires(
        self,
        instruments: Optional[Sequence[str]] = None,
        labs: Optional[Sequence[str]] = None,
        require_complete_mapping: bool = True,
    ) -> pd.DataFrame:
        """Return the session list joined with pivoted questionnaire columns.

        Args:
            instruments: Restrict to these questionnaire instruments
                (e.g. ``["BDI", "PHQ9"]``); ``None`` returns all available.
            labs: Restrict to these labs.
            require_complete_mapping: Only include sessions with a complete
                uid / subject_code / session_id mapping.

        Returns:
            One row per session, including demographics and one column per
            questionnaire item.
        """
        sessions = self._db.query(
            labs=list(labs) if labs is not None else None,
            include=["demographics", "questionnaires"],
            questionnaire_instruments=list(instruments) if instruments is not None else None,
            require_complete_mapping=require_complete_mapping,
            styled=False,
        )
        return sessions.rename(columns={"age_at_scan": "AGE"})
