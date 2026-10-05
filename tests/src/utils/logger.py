"""
src/utils/logger.py
-------------------
Loguru-based logger with rotating file sinks.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import sys
from pathlib import Path
from loguru import logger as _logger

_configured = False


def _configure_logger(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default sink
    _logger.remove()

    # Console sink — coloured, human-readable
    _logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    # General app log
    _logger.add(
        log_path / "app.log",
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        filter=lambda record: "retrieval" not in record["extra"]
        and "evaluation" not in record["extra"],
    )

    # Retrieval-specific log
    _logger.add(
        log_path / "retrieval.log",
        level="DEBUG",
        rotation="5 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        filter=lambda record: record["extra"].get("retrieval", False),
    )

    # Evaluation log
    _logger.add(
        log_path / "evaluation.log",
        level="DEBUG",
        rotation="5 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        filter=lambda record: record["extra"].get("evaluation", False),
    )

    _configured = True


def get_logger(name: str = "physics_rag"):
    """Return a bound logger for the given module name."""
    try:
        from src.utils.config import settings
        _configure_logger(settings.log_dir, settings.log_level)
    except Exception:
        _configure_logger()
    return _logger.bind(module=name)


# Pre-configure on import with defaults
_configure_logger()
