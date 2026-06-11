"""Logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Literal

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    *,
    force: bool = True,
) -> None:
    """Configure the root logger for a script or notebook.

    Parameters
    ----------
    level : str, optional
        The minimum logging level to display, by default "INFO".
    force : bool, optional
        If True, remove any existing handlers and re-configure.
        Helpful in interactive (notebook) environments.
    """
    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=force,
    )
    logging.info("Logging configured to level %s", level)
