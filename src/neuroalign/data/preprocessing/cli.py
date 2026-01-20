"""
CLI entry point for the data preparation pipeline.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
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


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> Optional[Path]:
    """Configure logging with optional file output.

    Args:
        verbose: If True, console shows DEBUG level; otherwise INFO.
        log_file: If provided, write detailed DEBUG logs to this file.
                  If set to a directory, auto-generate timestamped filename.

    Returns:
        Path to the log file if file logging is enabled, None otherwise.
    """
    # Determine console level
    console_level = logging.DEBUG if verbose else logging.INFO

    # Format for all handlers
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all; handlers filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    actual_log_path = None
    if log_file is not None:
        # If log_file is a directory, generate timestamped filename
        if log_file.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            actual_log_path = log_file / f"neuroalign_prepare_{timestamp}.log"
        else:
            actual_log_path = log_file
            # Ensure parent directory exists
            actual_log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(actual_log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG in file

        # More detailed format for file (includes line numbers)
        file_format = (
            "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=date_format))
        root_logger.addHandler(file_handler)

        # Log startup info to file
        logger = logging.getLogger(__name__)
        logger.debug("=" * 80)
        logger.debug("NEUROALIGN DATA PREPARATION LOG")
        logger.debug(f"Started at: {datetime.now().isoformat()}")
        logger.debug(f"Log file: {actual_log_path}")
        logger.debug("=" * 80)

    return actual_log_path


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
        "--log-file",
        "-l",
        type=Path,
        default=None,
        help=(
            "Write detailed DEBUG logs to this file. "
            "If a directory is provided, auto-generates timestamped filename. "
            "Log file always captures DEBUG level regardless of --verbose."
        ),
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
    log_path = setup_logging(args.verbose, args.log_file)

    logger = logging.getLogger(__name__)

    # Log configuration at startup (will appear in file if enabled)
    if log_path:
        print(f"Logging to: {log_path}")
        logger.debug("CLI Arguments:")
        for arg, value in vars(args).items():
            logger.debug(f"  {arg}: {value}")

    # Validate required arguments
    if args.sessions is None:
        logger.error(
            "Sessions CSV is required. Provide via --sessions or set SESSIONS_CSV in .env"
        )
        return 1

    try:
        config = build_config(args)

        # Log resolved configuration
        logger.debug("Resolved Configuration:")
        logger.debug(f"  Paths: {config.paths}")
        logger.debug(f"  Modalities: {config.modalities}")
        logger.debug(f"  Output: {config.output}")
        logger.debug(f"  Atlas: {config.atlas_name}")
        logger.debug(f"  N jobs: {config.n_jobs}")

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

        if log_path:
            print()
            print(f"Detailed log saved to: {log_path}")

        print()
        print("Usage example:")
        print("  from neuroalign.data.preprocessing import FeatureStore")
        print(f"  store = FeatureStore('{result.output_path}')")
        print("  gm = store.load_feature('gm_volume')")
        print("  multi = store.load_features(['gm_volume', 'ct_thickness'])")
        print("=" * 60)

        logger.debug("Pipeline completed successfully")
        return 0

    except Exception as e:
        # Always log full traceback to file, console depends on verbose
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        if log_path:
            print(f"\nPipeline failed. See detailed log at: {log_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
