"""Structured logging for the integration pipeline.

Provides JSON-formatted structured logging with contextual fields for
pipeline phase tracking, source system identification, and performance metrics.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional


class StructuredFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def __init__(self, include_timestamp: bool = True) -> None:
        super().__init__()
        self._include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {}
        if self._include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_entry["level"] = record.levelname
        log_entry["logger"] = record.name
        log_entry["message"] = record.getMessage()

        # Attach any extra structured fields
        for key in ("phase", "source_system", "entity", "duration_s",
                     "row_count", "error_type", "detail"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format for development."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(
    name: str,
    level: str = "INFO",
    fmt: str = "json",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger instance.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__``).
    level:
        Logging level string.
    fmt:
        ``"json"`` for structured output, ``"text"`` for human-readable.
    log_file:
        Optional path; when set a rotating-style file handler is added.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = StructuredFormatter() if fmt == "json" else TextFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


@contextmanager
def log_phase(
    logger: logging.Logger,
    phase: str,
    entity: Optional[str] = None,
    source_system: Optional[str] = None,
) -> Generator[None, None, None]:
    """Context manager that logs the start/end of a pipeline phase with timing.

    Usage::

        with log_phase(logger, "discovery", entity="customer"):
            run_discovery()
    """
    extra: dict[str, Any] = {"phase": phase}
    if entity:
        extra["entity"] = entity
    if source_system:
        extra["source_system"] = source_system

    logger.info("Phase started: %s", phase, extra=extra)
    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        extra["duration_s"] = round(elapsed, 3)
        logger.info("Phase completed: %s (%.3fs)", phase, elapsed, extra=extra)
    except Exception:
        elapsed = time.perf_counter() - start
        extra["duration_s"] = round(elapsed, 3)
        logger.exception("Phase failed: %s (%.3fs)", phase, elapsed, extra=extra)
        raise
