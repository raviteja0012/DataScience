"""
Data validation utilities for payment processing.

Provides reusable validation logic used across the processing pipeline —
account number format checks, amount range enforcement, duplicate detection,
and biller-specific rule evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional

from src.models.biller import Biller, PaymentTypeConfig
from src.models.payment import Payment, PaymentStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when a validation rule fails."""

    def __init__(self, code: str, message: str, field_name: Optional[str] = None):
        self.code = code
        self.field_name = field_name
        super().__init__(message)


@dataclass
class ValidationResult:
    """Aggregate result of running a validation pipeline."""
    is_valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)

    def add_error(self, code: str, message: str, field_name: Optional[str] = None) -> None:
        self.is_valid = False
        self.errors.append(ValidationError(code, message, field_name))

    @property
    def error_messages(self) -> list[str]:
        return [str(e) for e in self.errors]


def validate_account_number(account_number: str, biller: Biller) -> ValidationResult:
    """
    Validate an account number against the biller's format rules.

    Different utilities use different account number schemes — some are purely
    numeric, others have alpha prefixes for division codes.
    """
    result = ValidationResult()

    if not account_number:
        result.add_error("ACCT_EMPTY", "Account number is required", "account_number")
        return result

    rules = biller.validation_rules

    if len(account_number) != rules.account_number_length:
        result.add_error(
            "ACCT_LENGTH",
            f"Account number must be {rules.account_number_length} characters, "
            f"got {len(account_number)}",
            "account_number",
        )

    if not re.match(rules.account_number_format, account_number):
        result.add_error(
            "ACCT_FORMAT",
            f"Account number does not match expected format: {rules.account_number_format}",
            "account_number",
        )

    return result


def validate_payment_amount(
    amount: Decimal,
    biller: Biller,
    payment_type: str,
    bill_amount: Optional[Decimal] = None,
) -> ValidationResult:
    """
    Validate a payment amount against biller rules.

    Checks min/max bounds, partial payment allowance, and overpayment rules.
    """
    result = ValidationResult()

    if amount <= Decimal("0.00"):
        result.add_error("AMT_NEGATIVE", "Payment amount must be positive", "amount")
        return result

    pt_config = biller.get_payment_type_config(payment_type)
    if pt_config is None:
        result.add_error(
            "PMT_TYPE_UNSUPPORTED",
            f"Biller does not support payment type: {payment_type}",
            "payment_type",
        )
        return result

    if amount < pt_config.min_amount:
        result.add_error(
            "AMT_BELOW_MIN",
            f"Amount {amount} is below minimum {pt_config.min_amount}",
            "amount",
        )

    if amount > pt_config.max_amount:
        result.add_error(
            "AMT_ABOVE_MAX",
            f"Amount {amount} exceeds maximum {pt_config.max_amount}",
            "amount",
        )

    if bill_amount is not None:
        if amount < bill_amount and not biller.validation_rules.allow_partial_payments:
            result.add_error(
                "PARTIAL_NOT_ALLOWED",
                "This biller does not accept partial payments",
                "amount",
            )

        if amount > bill_amount and not biller.validation_rules.allow_overpayments:
            result.add_error(
                "OVERPAY_NOT_ALLOWED",
                "This biller does not accept overpayments",
                "amount",
            )

    return result


def validate_payment_channel(
    biller: Biller,
    payment_type: str,
    channel: str,
) -> ValidationResult:
    """Verify that the biller accepts the given channel for this payment type."""
    result = ValidationResult()

    if not biller.accepts_channel(payment_type, channel):
        result.add_error(
            "CHANNEL_NOT_SUPPORTED",
            f"Channel '{channel}' is not supported for payment type '{payment_type}' "
            f"on biller '{biller.biller_id}'",
            "channel",
        )

    return result


class DuplicateDetector:
    """
    Detects duplicate payment submissions within a configurable time window.

    Uses an idempotency key derived from biller + account + amount + date.
    In a production system this would be backed by Redis or a database;
    here we use an in-memory dict for simulation.
    """

    def __init__(self, window_hours: int = 24):
        self._window = timedelta(hours=window_hours)
        self._seen: dict[str, datetime] = {}

    def is_duplicate(self, payment: Payment) -> bool:
        """Check whether a payment with the same idempotency key was recently seen."""
        self._evict_stale()

        key = payment.idempotency_key or ""
        if key in self._seen:
            logger.warning(
                "Duplicate payment detected",
                extra={"event_data": {
                    "idempotency_key": key,
                    "transaction_id": payment.transaction_id,
                    "original_timestamp": self._seen[key].isoformat(),
                }},
            )
            return True

        self._seen[key] = datetime.utcnow()
        return False

    def _evict_stale(self) -> None:
        cutoff = datetime.utcnow() - self._window
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

    def clear(self) -> None:
        self._seen.clear()
