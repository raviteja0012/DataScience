"""
Payment transaction models.

Covers the full lifecycle of a payment from initiation through authorization,
settlement, and potential reversal. The Payment dataclass is the central record
that flows through the processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid


class PaymentStatus(str, Enum):
    INITIATED = "initiated"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    SETTLED = "settled"
    FAILED = "failed"
    REVERSED = "reversed"
    REFUNDED = "refunded"
    PENDING_RETRY = "pending_retry"
    TIMEOUT = "timeout"


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    ACH = "ach"
    CHECK = "check"


class PaymentChannel(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    IVR = "ivr"
    AGENT = "agent"
    KIOSK = "kiosk"


class PaymentType(str, Enum):
    ONE_TIME = "one_time"
    AUTOPAY = "autopay"
    BUDGET_BILLING = "budget_billing"
    PREPAY = "prepay"


@dataclass
class Payment:
    """
    Canonical payment transaction record.

    Created when a customer initiates a payment, then updated at each stage
    of processing (validation, authorization, capture, settlement). Carries
    all the context needed for reconciliation and exception handling downstream.
    """
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    biller_id: str = ""
    customer_account_id: str = ""
    bill_id: Optional[str] = None

    amount: Decimal = Decimal("0.00")
    convenience_fee: Decimal = Decimal("0.00")
    total_amount: Decimal = Decimal("0.00")

    payment_method: PaymentMethod = PaymentMethod.ACH
    payment_type: PaymentType = PaymentType.ONE_TIME
    channel: PaymentChannel = PaymentChannel.WEB

    status: PaymentStatus = PaymentStatus.INITIATED
    status_reason: Optional[str] = None

    # Timestamps for the processing pipeline
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    authorized_at: Optional[datetime] = None
    captured_at: Optional[datetime] = None
    settled_at: Optional[datetime] = None

    # Gateway-side identifiers
    authorization_code: Optional[str] = None
    gateway_reference: Optional[str] = None
    processor_response_code: Optional[str] = None

    # Retry tracking
    attempt_number: int = 1
    max_attempts: int = 3

    # Correlation
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.total_amount == Decimal("0.00") and self.amount > Decimal("0.00"):
            self.total_amount = self.amount + self.convenience_fee
        if self.idempotency_key is None:
            self.idempotency_key = (
                f"{self.biller_id}:{self.customer_account_id}:"
                f"{self.amount}:{self.initiated_at.strftime('%Y%m%d')}"
            )

    @property
    def is_terminal(self) -> bool:
        """Whether the payment has reached a final state."""
        return self.status in (
            PaymentStatus.SETTLED,
            PaymentStatus.FAILED,
            PaymentStatus.REVERSED,
            PaymentStatus.REFUNDED,
        )

    @property
    def is_retryable(self) -> bool:
        return (
            self.status in (PaymentStatus.FAILED, PaymentStatus.TIMEOUT, PaymentStatus.PENDING_RETRY)
            and self.attempt_number < self.max_attempts
        )

    def advance_status(self, new_status: PaymentStatus, reason: Optional[str] = None) -> None:
        """Transition the payment to a new status, recording the timestamp."""
        now = datetime.utcnow()
        self.status = new_status
        self.status_reason = reason

        status_timestamp_map = {
            PaymentStatus.VALIDATED: "validated_at",
            PaymentStatus.AUTHORIZED: "authorized_at",
            PaymentStatus.CAPTURED: "captured_at",
            PaymentStatus.SETTLED: "settled_at",
        }
        ts_field = status_timestamp_map.get(new_status)
        if ts_field:
            setattr(self, ts_field, now)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output or logging."""
        return {
            "transaction_id": self.transaction_id,
            "biller_id": self.biller_id,
            "customer_account_id": self.customer_account_id,
            "bill_id": self.bill_id,
            "amount": str(self.amount),
            "convenience_fee": str(self.convenience_fee),
            "total_amount": str(self.total_amount),
            "payment_method": self.payment_method.value,
            "payment_type": self.payment_type.value,
            "channel": self.channel.value,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "initiated_at": self.initiated_at.isoformat(),
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "attempt_number": self.attempt_number,
            "correlation_id": self.correlation_id,
        }


@dataclass
class PaymentBatch:
    """A batch of payments submitted together (e.g., autopay nightly run)."""
    batch_id: str = field(default_factory=lambda: f"BATCH-{uuid.uuid4().hex[:12].upper()}")
    biller_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    payments: list[Payment] = field(default_factory=list)
    status: str = "pending"

    @property
    def total_amount(self) -> Decimal:
        return sum((p.total_amount for p in self.payments), Decimal("0.00"))

    @property
    def payment_count(self) -> int:
        return len(self.payments)

    @property
    def success_count(self) -> int:
        return sum(1 for p in self.payments if p.status == PaymentStatus.SETTLED)

    @property
    def failure_count(self) -> int:
        return sum(1 for p in self.payments if p.status == PaymentStatus.FAILED)
