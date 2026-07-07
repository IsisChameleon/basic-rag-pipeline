from __future__ import annotations

import os
import sys

from loguru import logger

# Default DEBUG so the per-page ingest skip logs are actually emitted; override
# with LOG_LEVEL=INFO (etc.) in production if they get too noisy.
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()


def configure_logging() -> None:
    """Replace loguru's default handler with one at our chosen level, so DEBUG
    logs are emitted regardless of loguru's implicit defaults. Idempotent."""
    logger.remove()
    logger.add(sys.stderr, level=_LOG_LEVEL)
