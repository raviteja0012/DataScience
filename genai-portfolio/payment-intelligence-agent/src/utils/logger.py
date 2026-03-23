"""Structured logging configuration for the Payment Intelligence Agent.

Provides a consistent, structured logging interface across all modules
with support for context binding, correlation IDs, and log level filtering.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def _configure_structlog() -> None:
    """Configure structlog with processors for structured output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ],
            ),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_configured = False


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance bound to the given module name.

    Args:
        name: Logger name, typically __name__ of the calling module.
        **initial_context: Key-value pairs to bind to every log entry.

    Returns:
        A bound structlog logger instance.
    """
    global _configured
    if not _configured:
        _configure_structlog()
        _configured = True

    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
