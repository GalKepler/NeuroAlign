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
    BAGEstimationConfig,
    DataPaths,
    ModalityConfig,
    OutputConfig,
    PipelineConfig,
)
from neuroalign.data.preprocessing.pipeline import DataPreparationPipeline, PipelineResult

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
  neuroalign-prepare --brainlink-db /path/to/brainlink.db \\
      --tabular-derivatives-root /path/to/derivatives/tabular

  # Anatomical only, restricted to a single lab
  neuroalign-prepare --no-diffusion --labs TS

  # Force a full reload of all sessions
  neuroalign-prepare --force

Environment variables (loaded from .env):
  BRAINLINK_DB_PATH         - Path to the brainlink SQLite DB
  TABULAR_DERIVATIVES_ROOT  - Root of the pre-parcellated tabular derivatives tree
  ATLAS_NAME                - Atlas name (default: Schaefer2018N400n7Tian2020S2)
  ANAT_ATLASES              - Comma-separated anatomical atlas folders
                               (default: Schaefer2018N400n7,Tian2020S2)
  SESSION_VARIANT           - Diffusion session variant: cross|plain|subject
                               (default: cross)
        """,
    )

    # Path arguments (with env var defaults)
    paths_group = parser.add_argument_group("Data paths (override .env with CLI args)")
    paths_group.add_argument(
        "--brainlink-db",
        type=Path,
        default=_get_env_path("BRAINLINK_DB_PATH"),
        help="Path to the brainlink SQLite DB (env: BRAINLINK_DB_PATH)",
    )
    paths_group.add_argument(
        "--tabular-derivatives-root",
        type=Path,
        default=_get_env_path("TABULAR_DERIVATIVES_ROOT"),
        help="Root of the pre-parcellated tabular derivatives tree (env: TABULAR_DERIVATIVES_ROOT)",
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
        default=os.getenv("ATLAS_NAME", "Schaefer2018N400n7Tian2020S2"),
        help="Atlas name (env: ATLAS_NAME, default: Schaefer2018N400n7Tian2020S2)",
    )
    parser.add_argument(
        "--anat-atlases",
        default=os.getenv("ANAT_ATLASES", "Schaefer2018N400n7,Tian2020S2"),
        help=(
            "Comma-separated anatomical atlas folders, cortex first "
            "(env: ANAT_ATLASES, default: Schaefer2018N400n7,Tian2020S2)"
        ),
    )
    parser.add_argument(
        "--session-variant",
        choices=["cross", "plain", "subject"],
        default=os.getenv("SESSION_VARIANT", "cross"),
        help="Diffusion session variant (env: SESSION_VARIANT, default: cross)",
    )
    parser.add_argument(
        "--labs",
        nargs="+",
        default=None,
        help="Restrict to these brainlink labs (default: all)",
    )
    parser.add_argument(
        "--allow-incomplete-mapping",
        action="store_true",
        help="Include sessions without a complete subject_code/uid mapping in brainlink",
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
        "--force",
        "-f",
        action="store_true",
        help="Force reload all sessions (ignore existing data in store)",
    )

    # BAG estimation
    bag_group = parser.add_argument_group("Regional Brain Age Gap (BAG) estimation")
    bag_group.add_argument(
        "--run-bag-estimation",
        action="store_true",
        help="Run regional BAG estimation after building the feature store",
    )
    bag_group.add_argument(
        "--bag-univariate-features",
        nargs="+",
        default=None,
        metavar="FEATURE",
        help=(
            "Wide-format feature names for per-region univariate BAG estimation "
            "(default: anat_thickness_mean_mm)"
        ),
    )
    bag_group.add_argument(
        "--bag-multivariate-features",
        nargs="+",
        default=None,
        metavar="FEATURE1,FEATURE2,...",
        help=(
            "Comma-separated wide-format feature combinations for multivariate BAG "
            "estimation. Repeat to run multiple combinations, "
            "e.g. --bag-multivariate-features anat_thickness_mean_mm,DSIStudio_tensor_fa_mean"
        ),
    )
    bag_group.add_argument(
        "--bag-model-type",
        choices=["ridge", "xgboost", "lightgbm"],
        default="ridge",
        help="Base model type for BAG estimation (default: ridge)",
    )
    bag_group.add_argument(
        "--bag-n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits for BAG estimation (default: 5)",
    )
    bag_group.add_argument(
        "--bag-n-jobs",
        type=int,
        default=1,
        help="Parallel region fitting for BAG estimation, -1 for all cores (default: 1)",
    )
    bag_group.add_argument(
        "--bag-no-bias-correction",
        action="store_true",
        help="Disable post-hoc BAG bias correction",
    )
    bag_group.add_argument(
        "--bag-no-ipw",
        action="store_true",
        help="Disable inverse probability weighting in BAG estimation",
    )
    bag_group.add_argument(
        "--bag-all-univariate",
        action="store_true",
        help=(
            "Run univariate BAG estimation for every wide-format feature in the store. "
            "Overrides --bag-univariate-features."
        ),
    )
    bag_group.add_argument(
        "--bag-all-multivariate",
        action="store_true",
        help=(
            "Run one multivariate BAG estimation combining all wide-format features in the store. "
            "Diffusion filtered to --bag-multivariate-diffusion-metric (default: robust_mean). "
            "Anatomical restricted to anat_volume_mm3 (TIV-normalised) and anat_thickness_mean_mm. "
            "Overrides --bag-multivariate-features."
        ),
    )
    bag_group.add_argument(
        "--bag-multivariate-diffusion-metric",
        default="robust_mean",
        metavar="METRIC",
        help="Diffusion aggregation to use with --bag-all-multivariate (default: robust_mean)",
    )
    bag_group.add_argument(
        "--bag-multivariate-tiv-normalize",
        nargs="+",
        default=None,
        metavar="FEATURE",
        help=(
            "Feature names to divide by TIV before multivariate BAG estimation "
            "(default when --bag-all-multivariate: anat_volume_mm3)"
        ),
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build pipeline configuration from CLI arguments."""
    paths = DataPaths(
        brainlink_db=args.brainlink_db,
        tabular_derivatives_root=args.tabular_derivatives_root,
        output_dir=args.output,
    )

    modalities = ModalityConfig(
        anatomical=not args.no_anatomical,
        diffusion=not args.no_diffusion,
    )

    output = OutputConfig(
        prefix=args.prefix,
        compression=args.compression if args.compression != "none" else None,
    )

    anat_atlases = tuple(a.strip() for a in args.anat_atlases.split(","))

    bag_kwargs = {
        "enabled": args.run_bag_estimation,
        "model_type": args.bag_model_type,
        "n_splits": args.bag_n_splits,
        "n_jobs": args.bag_n_jobs,
        "bias_correction": not args.bag_no_bias_correction,
        "ipw": not args.bag_no_ipw,
    }
    if getattr(args, "bag_all_univariate", False) or getattr(args, "bag_all_multivariate", False):
        from neuroalign.data.preprocessing.feature_store import FeatureStore
        _store = FeatureStore(args.output)
        all_features = _store.list_features()
        if not all_features:
            flag = "--bag-all-univariate" if getattr(args, "bag_all_univariate", False) else "--bag-all-multivariate"
            raise ValueError(
                f"{flag}: no features found in store at "
                f"{args.output}. Run the pipeline first to build the feature store."
            )
        if getattr(args, "bag_all_univariate", False):
            bag_kwargs["univariate_features"] = all_features
        if getattr(args, "bag_all_multivariate", False):
            diff_metric = getattr(args, "bag_multivariate_diffusion_metric", "robust_mean")
            diff_features = [
                name for name in all_features
                if (info := _store.get_feature_info(name))
                and info.modality == "diffusion"
                and info.metric == diff_metric
            ]
            _anat_wanted = ["anat_volume_mm3", "anat_thickness_mean_mm"]
            anat_features = [f for f in _anat_wanted if f in all_features]
            bag_kwargs["multivariate_feature_sets"] = [anat_features + diff_features]
            tiv_norm = getattr(args, "bag_multivariate_tiv_normalize", None)
            if tiv_norm is None:
                tiv_norm = [f for f in ["anat_volume_mm3"] if f in all_features]
            bag_kwargs["multivariate_tiv_normalize"] = tiv_norm
    else:
        if args.bag_univariate_features is not None:
            bag_kwargs["univariate_features"] = args.bag_univariate_features
        if args.bag_multivariate_features is not None:
            bag_kwargs["multivariate_feature_sets"] = [
                combo.split(",") for combo in args.bag_multivariate_features
            ]
        if getattr(args, "bag_multivariate_tiv_normalize", None) is not None:
            bag_kwargs["multivariate_tiv_normalize"] = args.bag_multivariate_tiv_normalize
    bag_estimation = BAGEstimationConfig(**bag_kwargs)

    return PipelineConfig(
        paths=paths,
        modalities=modalities,
        output=output,
        bag_estimation=bag_estimation,
        atlas_name=args.atlas_name,
        anat_atlases=anat_atlases,
        session_variant=args.session_variant,
        labs=args.labs,
        require_complete_mapping=not args.allow_incomplete_mapping,
        force=args.force,
    )


def _print_wide_features(result: PipelineResult) -> None:
    """Print the generated wide-format feature breakdown."""
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


def _print_summary(result: PipelineResult, log_path: Optional[Path]) -> None:
    """Print a human-readable summary of a `PipelineResult`."""
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

    _print_wide_features(result)

    age_stats = result.metadata["age_stats"]
    if age_stats["min"] is not None:
        print()
        print(
            f"Age range: {age_stats['min']:.1f} - {age_stats['max']:.1f} "
            f"(mean: {age_stats['mean']:.1f})"
        )
        if age_stats["missing"] > 0:
            print(f"  Missing age: {age_stats['missing']} sessions")

    if result.bag_results_saved:
        print()
        print("BAG estimation results saved:")
        for bag_path in result.bag_results_saved:
            print(f"  - {bag_path}")

    if log_path:
        print()
        print(f"Detailed log saved to: {log_path}")

    print()
    print("Usage example:")
    print("  from neuroalign.data.preprocessing import FeatureStore")
    print(f"  store = FeatureStore('{result.output_path}')")
    print("  meta = store.load_metadata()")
    print("  thickness = store.load_feature('anat_thickness_mean_mm')")
    print("=" * 60)


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
    if args.brainlink_db is None:
        logger.error(
            "Brainlink DB path is required. "
            "Provide via --brainlink-db or set BRAINLINK_DB_PATH in .env"
        )
        return 1
    if args.tabular_derivatives_root is None:
        logger.error(
            "Tabular derivatives root is required. "
            "Provide via --tabular-derivatives-root or set TABULAR_DERIVATIVES_ROOT in .env"
        )
        return 1

    try:
        config = build_config(args)

        # Log resolved configuration
        logger.debug("Resolved Configuration:")
        logger.debug(f"  Paths: {config.paths}")
        logger.debug(f"  Modalities: {config.modalities}")
        logger.debug(f"  Output: {config.output}")
        logger.debug(f"  Atlas: {config.atlas_name} ({config.anat_atlases})")
        logger.debug(f"  Session variant: {config.session_variant}")
        logger.debug(f"  Labs: {config.labs}")
        logger.debug(f"  BAG estimation: {config.bag_estimation}")

        pipeline = DataPreparationPipeline(config)
        result = pipeline.run()

        _print_summary(result, log_path)

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
