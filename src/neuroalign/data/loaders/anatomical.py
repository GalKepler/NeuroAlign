"""
Anatomical MRI Data Loader for NeuroAlign
==========================================

Save this file as: src/neuroalign/data/loaders/anatomical.py

Loads and parcellates CAT12-processed anatomical data (GM volume, cortical thickness).

Example:
    >>> from neuroalign.data.loaders import AnatomicalLoader
    >>> loader = AnatomicalLoader(
    ...     cat12_root="/path/to/cat12/derivatives",
    ...     atlas_root="/path/to/atlases",
    ...     atlas_name="4S456Parcels"
    ... )
    >>> df = loader.load_sessions(sessions_csv="linked_sessions.csv")
"""

from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import nibabel as nib
import numpy as np
from dataclasses import dataclass

try:
    from parcellate.parcellation.volume import VolumetricParcellator
    PARCELLATE_AVAILABLE = True
except ImportError:
    PARCELLATE_AVAILABLE = False
    print("Warning: parcellate package not available. Install with: uv add parcellate")


@dataclass
class AnatomicalPaths:
    """Configuration for anatomical data paths."""
    cat12_root: Path
    atlas_root: Path
    atlas_name: str = "4S456Parcels"
    
    @property
    def atlas_img(self) -> Path:
        return (
            self.atlas_root
            / f"atlas-{self.atlas_name}"
            / f"atlas-{self.atlas_name}_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz"
        )
    
    @property
    def parcels_tsv(self) -> Path:
        return self.atlas_root / f"atlas-{self.atlas_name}" / f"atlas-{self.atlas_name}_dseg.tsv"


class AnatomicalLoader:
    """
    Loader for CAT12-processed anatomical MRI data.
    
    Extracts regional volumes (GM, WM) and cortical thickness from
    preprocessed probability maps using volumetric parcellation.
    
    Attributes:
        paths: AnatomicalPaths configuration
        parcellator: VolumetricParcellator instance (fitted on first use)
    """
    
    def __init__(
        self,
        cat12_root: Path,
        atlas_root: Path,
        atlas_name: str = "4S456Parcels"
    ):
        """
        Initialize anatomical data loader.
        
        Args:
            cat12_root: Path to CAT12 derivatives directory
            atlas_root: Path to atlas directory
            atlas_name: Name of parcellation atlas to use
        """
        if not PARCELLATE_AVAILABLE:
            raise ImportError(
                "parcellate package required. Install with: uv add parcellate"
            )
        
        self.paths = AnatomicalPaths(
            cat12_root=Path(cat12_root),
            atlas_root=Path(atlas_root),
            atlas_name=atlas_name
        )
        self.parcellator: Optional[VolumetricParcellator] = None
        self._parcels: Optional[pd.DataFrame] = None
    
    def _init_parcellator(self, example_file: Path) -> None:
        """Initialize and fit the parcellator on an example file."""
        if self.parcellator is not None:
            return
        
        parcels = pd.read_csv(self.paths.parcels_tsv, sep="\t")
        self.parcellator = VolumetricParcellator(
            atlas_img=str(self.paths.atlas_img),
            lut=parcels,
            mask="gm"
        )
        self.parcellator.fit(str(example_file))
        self._parcels = parcels
    
    def get_cat12_directory(
        self, 
        subject: str, 
        session: str
    ) -> Optional[Path]:
        """
        Get CAT12 output directory for a subject/session.
        
        Args:
            subject: Subject code (without 'sub-' prefix)
            session: Session ID (without 'ses-' prefix)
            
        Returns:
            Path to CAT12 directory or None if not found
        """
        cat12_dir = (
            self.paths.cat12_root 
            / f"sub-{subject}" 
            / f"ses-{session}" 
            / "anat"
        )
        return cat12_dir if cat12_dir.exists() else None
    
    def get_regional_volumes(
        self,
        cat12_directory: Path
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Extract regional volumes from CAT12 probability maps.
        
        Args:
            cat12_directory: Path to CAT12 anat directory
            
        Returns:
            Tuple of (gm_volumes, wm_volumes, cortical_thickness) DataFrames
            Returns None for missing modalities
        """
        if self.parcellator is None:
            # Find an example file to fit the parcellator
            gm_example = list(cat12_directory.glob("mwp1*.nii"))
            if not gm_example:
                raise RuntimeError(
                    f"No GM probability maps found in {cat12_directory}. "
                    "Cannot initialize parcellator."
                )
            self._init_parcellator(gm_example[0])
        
        # Find probability maps
        gm_prob = list(cat12_directory.glob("mwp1*.nii"))
        wm_prob = list(cat12_directory.glob("mwp2*.nii"))
        ct_prob = list(cat12_directory.glob("wct*.nii"))
        
        # Parcellate each modality
        gm_volumes = self.parcellator.transform(gm_prob[0]) if gm_prob else None
        wm_volumes = self.parcellator.transform(wm_prob[0]) if wm_prob else None
        ct = self.parcellator.transform(ct_prob[0]) if ct_prob else None
        
        return gm_volumes, wm_volumes, ct
    
    def load_session(
        self,
        subject: str,
        session: str,
        include_metadata: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load all anatomical data for a single session.
        
        Args:
            subject: Subject code
            session: Session ID
            include_metadata: Whether to include subject/session in output
            
        Returns:
            DataFrame with regional features or None if session not found
        """
        cat12_dir = self.get_cat12_directory(subject, session)
        if cat12_dir is None:
            return None
        
        gm_vol, wm_vol, ct = self.get_regional_volumes(cat12_dir)
        
        # Combine modalities
        dfs = []
        for modality, df in [("gm", gm_vol), ("wm", wm_vol), ("ct", ct)]:
            if df is None:
                continue
            df = df.copy()
            df["modality"] = modality
            df["metric"] = "volume" if modality in ("gm", "wm") else "thickness"
            if include_metadata:
                df["subject_code"] = subject
                df["session_id"] = session
            dfs.append(df)
        
        if not dfs:
            return None
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_sessions(
        self,
        sessions_csv: Path,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Load anatomical data for multiple sessions.
        
        Args:
            sessions_csv: Path to CSV with 'subject_code' and 'session_id' columns
            progress: Whether to show progress bar
            
        Returns:
            DataFrame with all sessions' regional features
        """
        sessions = pd.read_csv(
            sessions_csv,
            dtype={"subject_code": str, "session_id": str}
        )
        
        results = []
        iterator = sessions.iterrows()
        
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(sessions), desc="Loading anatomical data")
            except ImportError:
                pass
        
        for _, row in iterator:
            session_data = self.load_session(
                subject=row["subject_code"],
                session=row["session_id"]
            )
            if session_data is not None:
                # Add any additional metadata from sessions CSV
                for col in sessions.columns:
                    if col not in session_data.columns:
                        session_data[col] = row[col]
                results.append(session_data)
        
        if not results:
            raise ValueError("No sessions successfully loaded")
        
        return pd.concat(results, ignore_index=True)
    
    @property
    def region_names(self) -> Optional[List[str]]:
        """Get list of region names from the parcellation."""
        if self._parcels is None:
            return None
        return self._parcels["name"].tolist()
    
    @property
    def n_regions(self) -> int:
        """Get number of regions in the parcellation."""
        if self._parcels is None:
            return 0
        return len(self._parcels)
