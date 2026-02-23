"""
Payment gateway connector.

Simulates the interface between the payment platform and the downstream
payment processor / acquiring bank. In production this would be an HTTPS
integration with a processor like FIS, Fiserv, or Paymentech.

The gateway handles:
    - Authorization requests
    - Capture / settlement requests
    - Void / reversal requests
    - Transaction status inquiries
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from src.models.payment import Payment, PaymentMethod, PaymentStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GatewayResponseCode(str, Enum):
    """ISO 8583 response codes (simplified)."""
    APPROVED = "00"
    REFER_TO_ISSUER = "01"
    INVALID_MERCHANT = "03"
    DO_NOT_HONOR = "05"
    INVALID_TRANSACTION = "12"
    INVALID_AMOUNT = "13"
    INVALID_CARD_NUMBER = "14"
    NO_SUCH_ISSUER = "15"
    INSUFFICIENT_FUNDS = "51"
    EXPIRED_CARD = "54"
    INCORRECT_PIN = "55"
    TRANSACTION_NOT_PERMITTED = "57"
    SUSPECTED_FRAUD = "59"
    EXCEEDS_LIMIT = "61"
    RESTRICTED_CARD = "62"
    SYSTEM_MALFUNCTION = "96"
    TIMEOUT = "99"


@dataclass
class GatewayRequest:
    """Authorization or capture request to the payment gateway."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str = ""
    amount: Decimal = Decimal("0.00")
    payment_method: PaymentMethod = PaymentMethod.ACH
    card_last_four: str = "0000"
    merchant_id: str = "BILLPAY-001"
    request_type: str = "authorization"  # authorization, capture, void
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GatewayResponse:
    """Response from the payment gateway."""
    request_id: str = ""
    response_code: GatewayResponseCode = GatewayResponseCode.APPROVED
    authorization_code: Optional[str] = None
    reference_number: Optional[str] = None
    response_message: str = ""
    processor_timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0

    @property
    def is_approved(self) -> bool:
        return self.response_code == GatewayResponseCode.APPROVED


class PaymentGateway:
    """
    Simulated payment gateway with configurable behavior.

    Supports realistic response patterns: most transactions approve, a small
    percentage decline for various reasons, and rare timeouts occur. Latency
    is simulated to mimic real-world gateway response times.
    """

    def __init__(
        self,
        decline_rate: float = 0.03,
        timeout_rate: float = 0.005,
        avg_latency_ms: float = 250.0,
    ):
        self._decline_rate = decline_rate
        self._timeout_rate = timeout_rate
        self._avg_latency_ms = avg_latency_ms
        self._transaction_log: list[tuple[GatewayRequest, GatewayResponse]] = []

    def authorize(self, payment: Payment) -> GatewayResponse:
        """Submit an authorization request for a payment."""
        request = GatewayRequest(
            transaction_id=payment.transaction_id,
            amount=payment.total_amount,
            payment_method=payment.payment_method,
            request_type="authorization",
        )

        # Simulate processing latency
        latency = max(50, random.gauss(self._avg_latency_ms, self._avg_latency_ms * 0.3))

        logger.info(
            f"Gateway authorization request: {request.request_id}",
            extra={"event_data": {
                "request_id": request.request_id,
                "transaction_id": payment.transaction_id,
                "amount": str(payment.total_amount),
                "method": payment.payment_method.value,
            }},
        )

        # Simulate timeout
        if random.random() < self._timeout_rate:
            response = GatewayResponse(
                request_id=request.request_id,
                response_code=GatewayResponseCode.TIMEOUT,
                response_message="Gateway timeout — no response from processor",
                latency_ms=30000.0,
            )
            self._transaction_log.append((request, response))
            return response

        # Simulate decline
        if random.random() < self._decline_rate:
            decline_code = random.choice([
                GatewayResponseCode.INSUFFICIENT_FUNDS,
                GatewayResponseCode.DO_NOT_HONOR,
                GatewayResponseCode.EXPIRED_CARD,
                GatewayResponseCode.INVALID_CARD_NUMBER,
                GatewayResponseCode.EXCEEDS_LIMIT,
            ])
            response = GatewayResponse(
                request_id=request.request_id,
                response_code=decline_code,
                response_message=f"Declined: {decline_code.value} — {decline_code.name.replace('_', ' ').title()}",
                latency_ms=latency,
            )
            self._transaction_log.append((request, response))
            return response

        # Approved
        response = GatewayResponse(
            request_id=request.request_id,
            response_code=GatewayResponseCode.APPROVED,
            authorization_code=f"A{random.randint(100000, 999999)}",
            reference_number=f"REF-{uuid.uuid4().hex[:12].upper()}",
            response_message="Approved",
            latency_ms=latency,
        )

        self._transaction_log.append((request, response))
        return response

    def capture(self, payment: Payment, auth_code: str) -> GatewayResponse:
        """Capture a previously authorized payment."""
        request = GatewayRequest(
            transaction_id=payment.transaction_id,
            amount=payment.total_amount,
            payment_method=payment.payment_method,
            request_type="capture",
        )

        latency = max(30, random.gauss(self._avg_latency_ms * 0.5, 50))

        response = GatewayResponse(
            request_id=request.request_id,
            response_code=GatewayResponseCode.APPROVED,
            authorization_code=auth_code,
            reference_number=f"CAP-{uuid.uuid4().hex[:12].upper()}",
            response_message="Captured",
            latency_ms=latency,
        )

        self._transaction_log.append((request, response))
        return response

    def void(self, payment: Payment) -> GatewayResponse:
        """Void a previously authorized but uncaptured payment."""
        request = GatewayRequest(
            transaction_id=payment.transaction_id,
            amount=payment.total_amount,
            payment_method=payment.payment_method,
            request_type="void",
        )

        response = GatewayResponse(
            request_id=request.request_id,
            response_code=GatewayResponseCode.APPROVED,
            reference_number=f"VOID-{uuid.uuid4().hex[:12].upper()}",
            response_message="Voided",
        )

        self._transaction_log.append((request, response))
        return response

    def get_transaction_log(self) -> list[tuple[GatewayRequest, GatewayResponse]]:
        return list(self._transaction_log)

    def get_stats(self) -> dict:
        total = len(self._transaction_log)
        approved = sum(1 for _, r in self._transaction_log if r.is_approved)
        declined = total - approved

        avg_latency = (
            sum(r.latency_ms for _, r in self._transaction_log) / total
            if total > 0
            else 0
        )

        return {
            "total_requests": total,
            "approved": approved,
            "declined": declined,
            "approval_rate": f"{approved / total:.1%}" if total else "N/A",
            "avg_latency_ms": round(avg_latency, 1),
        }
