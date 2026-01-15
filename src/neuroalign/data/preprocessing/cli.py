"""
CLI entry point for the data preparation pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from neuroalign.data.preprocessing.config import (
    PipelineConfig,
    DataPaths,
    ModalityConfig,
    OutputConfig,
)
from neuroalign.data.preprocessing.pipeline import DataPreparationPipeline

# Load environment variables from .env file
load_dotenv()


def _get_env_path(var_name: str) -> Optional[Path]:
    """Get a path from environment variable, expanding ~ if present."""
    value = os.getenv(var_name)
    if value:
        return Path(os.path.expanduser(value))
    return None


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare NeuroAlign feature matrices from neuroimaging data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all defaults from .env file
  neuroalign-prepare

  # Full pipeline with explicit paths
  neuroalign-prepare --sessions /path/to/sessions.csv \\
      --cat12-root /path/to/cat12 \\
      --atlas-root /path/to/atlases \\
      --qsiparc /path/to/qsiparc \\
      --qsirecon /path/to/qsirecon

  # Anatomical only
  neuroalign-prepare --no-diffusion

  # Diffusion only with specific workflows
  neuroalign-prepare --no-anatomical --workflows AMICONODDI DSIStudio

Environment variables (loaded from .env):
  SESSIONS_CSV      - Path to sessions CSV
  CAT12_ROOT        - Path to CAT12 derivatives
  CAT12_ATLAS_ROOT  - Path to atlas directory
  QSIPARC_PATH      - Path to QSIParc derivatives
  QSIRECON_PATH     - Path to QSIRecon derivatives
  ATLAS_NAME        - Atlas name (default: 4S456Parcels)
        """,
    )

    # Path arguments (with env var defaults)
    paths_group = parser.add_argument_group("Data paths (override .env with CLI args)")
    paths_group.add_argument(
        "--sessions",
        "-s",
        type=Path,
        default=_get_env_path("SESSIONS_CSV"),
        help="Path to sessions CSV (env: SESSIONS_CSV)",
    )
    paths_group.add_argument(
        "--cat12-root",
        type=Path,
        default=_get_env_path("CAT12_ROOT"),
        help="Path to CAT12 derivatives directory (env: CAT12_ROOT)",
    )
    paths_group.add_argument(
        "--atlas-root",
        type=Path,
        default=_get_env_path("CAT12_ATLAS_ROOT"),
        help="Path to atlas directory (env: CAT12_ATLAS_ROOT)",
    )
    paths_group.add_argument(
        "--qsiparc",
        type=Path,
        default=_get_env_path("QSIPARC_PATH"),
        help="Path to QSIParc derivatives directory (env: QSIPARC_PATH)",
    )
    paths_group.add_argument(
        "--qsirecon",
        type=Path,
        default=_get_env_path("QSIRECON_PATH"),
        help="Path to QSIRecon derivatives directory (env: QSIRECON_PATH)",
    )
    paths_group.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)",
    )

    # Modality selection
    modality_group = parser.add_argument_group("Modality selection")
    modality_group.add_argument(
        "--no-anatomical",
        action="store_true",
        help="Disable anatomical data loading",
    )
    modality_group.add_argument(
        "--no-diffusion",
        action="store_true",
        help="Disable diffusion data loading",
    )
    modality_group.add_argument(
        "--no-gm",
        action="store_true",
        help="Disable gray matter volume",
    )
    modality_group.add_argument(
        "--no-wm",
        action="store_true",
        help="Disable white matter volume",
    )
    modality_group.add_argument(
        "--no-ct",
        action="store_true",
        help="Disable cortical thickness",
    )
    modality_group.add_argument(
        "--workflows",
        nargs="+",
        help="Specific diffusion workflows to include (default: all)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--prefix",
        default="neuroalign",
        help="Output file prefix (default: neuroalign)",
    )
    output_group.add_argument(
        "--compression",
        choices=["snappy", "gzip", "brotli", "none"],
        default="snappy",
        help="Parquet compression (default: snappy)",
    )

    # General options
    parser.add_argument(
        "--atlas-name",
        default=os.getenv("ATLAS_NAME", "4S456Parcels"),
        help="Atlas name (env: ATLAS_NAME, default: 4S456Parcels)",
    )
    parser.add_argument(
        "--age-column",
        default="AGE",
        help="Column name for age in sessions CSV (default: AGE, also tries Age@Scan)",
    )
    parser.add_argument(
        "--n-jobs",
        "-j",
        type=int,
        default=int(os.getenv("N_JOBS", "1")),
        help="Number of parallel workers (env: N_JOBS, default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reload all sessions (ignore existing data in store)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build pipeline configuration from CLI arguments."""
    paths = DataPaths(
        sessions_csv=args.sessions,
        cat12_root=args.cat12_root,
        atlas_root=args.atlas_root,
        qsiparc_path=args.qsiparc,
        qsirecon_path=args.qsirecon,
        output_dir=args.output,
    )

    modalities = ModalityConfig(
        anatomical=not args.no_anatomical,
        diffusion=not args.no_diffusion,
        gray_matter=not args.no_gm,
        white_matter=not args.no_wm,
        cortical_thickness=not args.no_ct,
        diffusion_workflows=args.workflows,
    )

    output = OutputConfig(
        prefix=args.prefix,
        compression=args.compression if args.compression != "none" else None,
    )

    return PipelineConfig(
        paths=paths,
        modalities=modalities,
        output=output,
        atlas_name=args.atlas_name,
        age_column=args.age_column,
        n_jobs=args.n_jobs,
        progress=not args.no_progress,
        force=args.force,
    )


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Validate required arguments
    if args.sessions is None:
        logger.error(
            "Sessions CSV is required. Provide via --sessions or set SESSIONS_CSV in .env"
        )
        return 1

    try:
        config = build_config(args)
        pipeline = DataPreparationPipeline(config)
        result = pipeline.run()

        # Print summary
        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {result.output_path}")
        print(f"Total sessions in store: {result.metadata['n_sessions']}")
        print(f"Unique subjects: {result.metadata['n_subjects']}")

        # Show incremental loading stats
        if result.n_skipped_sessions > 0 or result.n_new_sessions > 0:
            print()
            print(f"This run: {result.n_new_sessions} new sessions loaded")
            if result.n_skipped_sessions > 0:
                print(f"          {result.n_skipped_sessions} sessions already in store (skipped)")

        print()
        print(f"Long formats saved: {len(result.long_formats_saved)}")
        for fmt in result.long_formats_saved:
            print(f"  - {fmt}")

        print()
        print(f"Wide feature types: {result.metadata['n_wide_features']}")
        anat_feats = result.metadata.get("anatomical_features", [])
        diff_feats = result.metadata.get("diffusion_features", [])
        print(f"  Anatomical: {len(anat_feats)}")
        for feat in anat_feats:
            print(f"    - {feat}")
        print(f"  Diffusion: {len(diff_feats)}")
        for feat in diff_feats:
            print(f"    - {feat}")

        if result.metadata["age_stats"]["min"] is not None:
            print()
            print(
                f"Age range: {result.metadata['age_stats']['min']:.1f} - "
                f"{result.metadata['age_stats']['max']:.1f} "
                f"(mean: {result.metadata['age_stats']['mean']:.1f})"
            )
            if result.metadata["age_stats"]["missing"] > 0:
                print(f"  Missing age: {result.metadata['age_stats']['missing']} sessions")

        print()
        print("Usage example:")
        print("  from neuroalign.data.preprocessing import FeatureStore")
        print(f"  store = FeatureStore('{result.output_path}')")
        print("  gm = store.load_feature('gm_volume')")
        print("  multi = store.load_features(['gm_volume', 'ct_thickness'])")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
