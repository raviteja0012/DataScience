"""
Structured logging with correlation ID support.

Every log entry carries a correlation_id so that a single payment transaction
can be traced end-to-end across onboarding, processing, settlement, and
exception handling. Output is JSON-structured for compatibility with log
aggregation platforms (Splunk, ELK, Datadog).
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

# Context variable for request-scoped correlation IDs.
_correlation_id: ContextVar[str] = ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set (or generate) a correlation ID for the current context."""
    cid = cid or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    cid = _correlation_id.get()
    if not cid:
        cid = set_correlation_id()
    return cid


class StructuredFormatter(logging.Formatter):
    """Formats log records as single-line JSON with standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Merge any extra fields passed via the `extra` kwarg
        if hasattr(record, "event_data"):
            log_entry["data"] = record.event_data  # type: ignore[attr-defined]

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger configured with structured JSON output.

    Usage:
        logger = get_logger(__name__)
        logger.info("Payment processed", extra={"event_data": {"txn_id": "abc"}})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
