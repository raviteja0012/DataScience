"""Security utilities for input sanitization and PII masking.

Provides defense-in-depth measures for handling user input and ensuring
sensitive payment data is never exposed in logs, responses, or error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import bleach

from .logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PIIPattern:
    """Defines a PII detection pattern with its masking replacement."""

    name: str
    pattern: re.Pattern[str]
    replacement: str


# Standard PII patterns for payment data
_PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="credit_card",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        replacement="[CARD ****]",
    ),
    PIIPattern(
        name="ssn",
        pattern=re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
        replacement="[SSN ***-**-****]",
    ),
    PIIPattern(
        name="email",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        replacement="[EMAIL REDACTED]",
    ),
    PIIPattern(
        name="phone",
        pattern=re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
        replacement="[PHONE REDACTED]",
    ),
    PIIPattern(
        name="cvv",
        pattern=re.compile(r"\bCVV[:\s]*\d{3,4}\b", re.IGNORECASE),
        replacement="[CVV ***]",
    ),
    PIIPattern(
        name="account_number",
        pattern=re.compile(r"\b(?:account|acct)[#:\s]*\d{8,17}\b", re.IGNORECASE),
        replacement="[ACCOUNT ****]",
    ),
    PIIPattern(
        name="routing_number",
        pattern=re.compile(r"\b(?:routing|aba)[#:\s]*\d{9}\b", re.IGNORECASE),
        replacement="[ROUTING ****]",
    ),
]

# SQL injection indicators beyond what sqlparse catches
_SQL_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r";\s*(?:DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC)", re.IGNORECASE),
    re.compile(r"UNION\s+(?:ALL\s+)?SELECT", re.IGNORECASE),
    re.compile(r"(?:--|#|/\*)\s*$", re.MULTILINE),
    re.compile(r"'\s*(?:OR|AND)\s+['\d]", re.IGNORECASE),
    re.compile(r"(?:xp_|sp_)\w+", re.IGNORECASE),
    re.compile(r"WAITFOR\s+DELAY", re.IGNORECASE),
    re.compile(r"BENCHMARK\s*\(", re.IGNORECASE),
    re.compile(r"LOAD_FILE\s*\(", re.IGNORECASE),
    re.compile(r"INTO\s+(?:OUT|DUMP)FILE", re.IGNORECASE),
]


class PIIMasker:
    """Masks personally identifiable information in text and data structures.

    Scans text for known PII patterns (credit card numbers, SSNs, emails, etc.)
    and replaces them with safe placeholder tokens. Operates recursively on
    nested data structures.
    """

    def __init__(self, additional_patterns: list[PIIPattern] | None = None) -> None:
        self._patterns = list(_PII_PATTERNS)
        if additional_patterns:
            self._patterns.extend(additional_patterns)

    def mask_text(self, text: str) -> str:
        """Replace all detected PII patterns in the given text.

        Args:
            text: Raw text that may contain PII.

        Returns:
            Text with all detected PII replaced by safe placeholders.
        """
        masked = text
        for pii in self._patterns:
            masked = pii.pattern.sub(pii.replacement, masked)
        return masked

    def mask_dict(self, data: dict[str, Any], sensitive_keys: set[str] | None = None) -> dict[str, Any]:
        """Recursively mask PII in dictionary values.

        Args:
            data: Dictionary potentially containing PII in values.
            sensitive_keys: Field names whose values should be fully redacted.

        Returns:
            New dictionary with PII masked.
        """
        if sensitive_keys is None:
            sensitive_keys = {
                "card_number", "card_pan", "cvv", "ssn",
                "account_number", "routing_number", "password",
                "secret", "token", "api_key",
            }

        masked: dict[str, Any] = {}
        for key, value in data.items():
            if key.lower() in sensitive_keys:
                masked[key] = "[REDACTED]"
            elif isinstance(value, str):
                masked[key] = self.mask_text(value)
            elif isinstance(value, dict):
                masked[key] = self.mask_dict(value, sensitive_keys)
            elif isinstance(value, list):
                masked[key] = [
                    self.mask_dict(item, sensitive_keys) if isinstance(item, dict)
                    else self.mask_text(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked[key] = value
        return masked


@dataclass
class InputSanitizer:
    """Sanitizes and validates user input before processing.

    Applies HTML stripping, length limits, and SQL injection detection
    to all user-provided text before it reaches any processing pipeline.
    """

    max_input_length: int = 2000
    allowed_tags: list[str] = field(default_factory=list)

    def sanitize(self, user_input: str) -> str:
        """Clean and validate user input.

        Args:
            user_input: Raw user input string.

        Returns:
            Sanitized input string.

        Raises:
            ValueError: If input exceeds length limits or contains injection attempts.
        """
        if not user_input or not user_input.strip():
            raise ValueError("Input cannot be empty")

        # Strip HTML/script tags
        cleaned = bleach.clean(user_input, tags=self.allowed_tags, strip=True)

        # Enforce length limit
        if len(cleaned) > self.max_input_length:
            raise ValueError(
                f"Input exceeds maximum length of {self.max_input_length} characters"
            )

        # Check for SQL injection patterns
        for pattern in _SQL_INJECTION_PATTERNS:
            if pattern.search(cleaned):
                logger.warning(
                    "sql_injection_attempt_detected",
                    pattern=pattern.pattern,
                    input_preview=cleaned[:50],
                )
                raise ValueError("Input contains potentially unsafe SQL patterns")

        return cleaned.strip()

    def sanitize_identifier(self, identifier: str) -> str:
        """Sanitize a SQL identifier (table name, column name).

        Args:
            identifier: Raw identifier string.

        Returns:
            Safe identifier containing only alphanumeric characters and underscores.

        Raises:
            ValueError: If identifier is empty or contains only invalid characters.
        """
        safe = re.sub(r"[^a-zA-Z0-9_]", "", identifier)
        if not safe:
            raise ValueError(f"Invalid identifier after sanitization: '{identifier}'")
        return safe
