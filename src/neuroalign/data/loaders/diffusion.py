"""
Diffusion MRI Data Loader for NeuroAlign
=========================================

Save this file as: src/neuroalign/data/loaders/diffusion.py

Loads QSIPrep/QSIRecon-processed diffusion data (DTI, NODDI derivatives).

Example:
    >>> from neuroalign.data.loaders import DiffusionLoader
    >>> loader = DiffusionLoader(
    ...     qsiparc_path="/path/to/qsiparc",
    ...     workflows=["AMICONODDI"]
    ... )
    >>> df = loader.load_sessions(sessions_csv="linked_sessions.csv")
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DiffusionPaths:
    """Configuration for diffusion data paths."""

    qsiparc_path: Path
    qsirecon_path: Path
    atlas_name: str = "4S456Parcels"

    @property
    def atlas_tsv(self) -> Path:
        return (
            self.qsirecon_path
            / "atlases"
            / f"atlas-{self.atlas_name}"
            / f"atlas-{self.atlas_name}_dseg.tsv"
        )


def parse_bids_entities(filename: str) -> Dict[str, str]:
    """
    Parse BIDS filename entities.

    Args:
        filename: BIDS-formatted filename

    Returns:
        Dictionary of entity key-value pairs

    Example:
        >>> parse_bids_entities("sub-001_ses-01_model-DTI_param-MD_dseg.tsv")
        {'sub': '001', 'ses': '01', 'model': 'DTI', 'param': 'MD'}
    """
    entities = {}
    parts = Path(filename).name.split("_")
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
    return entities


class DiffusionLoader:
    """
    Loader for QSIPrep/QSIRecon-processed diffusion MRI data.

    Extracts regional microstructure parameters (MD, FA, RD, NODDI derivatives)
    from parcellated reconstruction outputs.

    Attributes:
        paths: DiffusionPaths configuration
        workflows: List of reconstruction workflows to load
    """

    def __init__(
        self,
        qsiparc_path: Path,
        qsirecon_path: Path,
        workflows: Optional[List[str]] = None,
        atlas_name: str = "4S456Parcels",
        n_jobs: int = 1,
    ):
        """
        Initialize diffusion data loader.

        Args:
            qsiparc_path: Path to qsiparc derivatives
            qsirecon_path: Path to qsirecon derivatives
            workflows: List of workflow names to load (e.g., ["AMICONODDI"])
                      If None, will auto-detect all qsirecon-* workflows
            atlas_name: Name of parcellation atlas
            n_jobs: Number of parallel workers for loading (default: 1 = serial)
        """
        self.paths = DiffusionPaths(
            qsiparc_path=Path(qsiparc_path),
            qsirecon_path=Path(qsirecon_path),
            atlas_name=atlas_name,
        )
        self.n_jobs = n_jobs

        if workflows is None:
            # Auto-detect workflows
            self.workflows = self._discover_workflows()
        else:
            self.workflows = workflows

    def _discover_workflows(self) -> List[str]:
        """
        Discover available QSIRecon workflows.

        Returns:
            List of workflow names (without 'qsirecon-' prefix)
        """
        workflows = []
        if not self.paths.qsiparc_path.exists():
            return workflows

        for workflow_dir in self.paths.qsiparc_path.iterdir():
            if workflow_dir.is_dir() and workflow_dir.name.startswith("qsirecon-"):
                workflow_name = workflow_dir.name.replace("qsirecon-", "")
                workflows.append(workflow_name)

        return workflows

    def get_session_directory(self, subject: str, session: str, workflow: str) -> Optional[Path]:
        """
        Get QSIParc directory for a subject/session/workflow.

        Args:
            subject: Subject code (without 'sub-' prefix)
            session: Session ID (without 'ses-' prefix)
            workflow: Workflow name (without 'qsirecon-' prefix)

        Returns:
            Path to session directory or None if not found
        """
        session_dir = (
            self.paths.qsiparc_path
            / f"qsirecon-{workflow}"
            / f"sub-{subject}"
            / f"ses-{session}"
            / "dwi"
            / f"atlas-{self.paths.atlas_name}"
        )
        return session_dir if session_dir.exists() else None

    def load_session(
        self,
        subject: str,
        session: str,
        workflow: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load diffusion data for a single session.

        Args:
            subject: Subject code
            session: Session ID
            workflow: Specific workflow to load, or None for all workflows
            include_metadata: Whether to include subject/session in output

        Returns:
            DataFrame with regional features or None if session not found
        """
        workflows_to_load = [workflow] if workflow else self.workflows

        dfs = []
        for wf in workflows_to_load:
            session_dir = self.get_session_directory(subject, session, wf)
            if session_dir is None:
                continue

            # Load all TSV files in the directory
            for tsv_file in session_dir.glob("*_parc.tsv"):
                entities = parse_bids_entities(tsv_file.name)

                df = pd.read_csv(tsv_file, sep="\t")
                df["workflow"] = wf
                df["model"] = entities.get("model", "unknown")
                df["param"] = entities.get("param", "unknown")
                df["desc"] = entities.get("desc", "unknown")

                if include_metadata:
                    df["subject_code"] = subject
                    df["session_id"] = session

                dfs.append(df)

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True, copy=False)

    def _load_session_worker(
        self,
        row: Dict[str, Any],
        workflow: Optional[str],
    ) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """
        Worker function for parallel session loading.

        Args:
            row: Dictionary with session metadata (subject_code, session_id, etc.)
            workflow: Specific workflow to load, or None for all workflows

        Returns:
            Tuple of (subject_code, session_id, DataFrame or None)
        """
        subject = row["subject_code"]
        session = row["session_id"]
        session_data = self.load_session(
            subject=subject,
            session=session,
            workflow=workflow,
        )
        if session_data is not None:
            # Add metadata columns from sessions CSV
            for col, val in row.items():
                if col not in session_data.columns:
                    session_data[col] = val
        return subject, session, session_data

    def load_sessions(
        self,
        sessions_csv: Path,
        workflow: Optional[str] = None,
        progress: bool = True,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load diffusion data for multiple sessions.

        Args:
            sessions_csv: Path to CSV with 'subject_code' and 'session_id' columns
            workflow: Specific workflow to load, or None for all workflows
            progress: Whether to show progress bar
            n_jobs: Number of parallel workers (overrides instance setting)

        Returns:
            DataFrame with all sessions' regional features
        """
        sessions = pd.read_csv(sessions_csv, dtype={"subject_code": str, "session_id": str})
        effective_jobs = n_jobs if n_jobs is not None else self.n_jobs

        if effective_jobs == 1:
            return self._load_sessions_serial(sessions, workflow, progress)
        else:
            return self._load_sessions_parallel(sessions, workflow, progress, effective_jobs)

    def _load_sessions_serial(
        self,
        sessions: pd.DataFrame,
        workflow: Optional[str],
        progress: bool,
    ) -> pd.DataFrame:
        """
        Serial loading (original behavior).

        Args:
            sessions: DataFrame with subject_code and session_id columns
            workflow: Specific workflow to load
            progress: Whether to show progress bar

        Returns:
            DataFrame with all sessions' regional features
        """
        results = []
        iterator = sessions.iterrows()

        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(sessions), desc="Loading diffusion data")
            except ImportError:
                pass

        for _, row in iterator:
            session_data = self.load_session(
                subject=row["subject_code"], session=row["session_id"], workflow=workflow
            )
            if session_data is not None:
                for col in sessions.columns:
                    if col not in session_data.columns:
                        session_data[col] = row[col]
                results.append(session_data)

        if not results:
            raise ValueError("No sessions successfully loaded")

        return pd.concat(results, ignore_index=True, copy=False)

    def _load_sessions_parallel(
        self,
        sessions: pd.DataFrame,
        workflow: Optional[str],
        progress: bool,
        n_jobs: int,
    ) -> pd.DataFrame:
        """
        Parallel loading using ThreadPoolExecutor.

        Uses threads (not processes) since diffusion loading is I/O-bound
        and doesn't require heavy CPU computation.

        Args:
            sessions: DataFrame with subject_code and session_id columns
            workflow: Specific workflow to load
            progress: Whether to show progress bar
            n_jobs: Number of parallel workers

        Returns:
            DataFrame with all sessions' regional features
        """
        results: List[pd.DataFrame] = []
        success_count = 0
        skip_count = 0
        error_count = 0
        failed_sessions: List[tuple] = []

        logger.info(f"Loading diffusion data with {n_jobs} parallel workers")
        logger.debug(f"Total sessions to process: {len(sessions)}")

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(
                    self._load_session_worker,
                    row.to_dict(),
                    workflow,
                )
                for _, row in sessions.iterrows()
            ]

            iterator = as_completed(futures)
            if progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=len(futures), desc="Loading diffusion data")
                except ImportError:
                    pass

            for fut in iterator:
                try:
                    subject, session, session_data = fut.result()
                    if session_data is not None:
                        results.append(session_data)
                        success_count += 1
                        logger.debug(f"  SUCCESS: sub-{subject}_ses-{session}")
                    else:
                        skip_count += 1
                        failed_sessions.append((subject, session, "no_data"))
                        logger.debug(f"  SKIP: sub-{subject}_ses-{session} - no data found")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Worker exception: {e}", exc_info=True)

        # Log summary
        logger.info(
            f"Diffusion loading complete: "
            f"{success_count} success, {skip_count} skipped, {error_count} errors"
        )

        if failed_sessions:
            logger.debug(f"Failed/skipped sessions ({len(failed_sessions)} total):")
            for subj, sess, reason in failed_sessions[:50]:
                logger.debug(f"  sub-{subj}_ses-{sess}: {reason}")
            if len(failed_sessions) > 50:
                logger.debug(f"  ... and {len(failed_sessions) - 50} more")

        if not results:
            logger.error(
                f"No sessions successfully loaded. "
                f"Attempted {len(sessions)} sessions. "
                f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}"
            )
            raise ValueError("No sessions successfully loaded")

        logger.info(f"Successfully loaded {len(results)} sessions")
        return pd.concat(results, ignore_index=True, copy=False)

    def get_available_parameters(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get available parameters grouped by model.

        Args:
            df: DataFrame loaded from load_sessions

        Returns:
            Dictionary mapping model names to lists of parameters
        """
        params_by_model = {}
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            params_by_model[model] = sorted(model_df["param"].unique().tolist())

        return params_by_model
