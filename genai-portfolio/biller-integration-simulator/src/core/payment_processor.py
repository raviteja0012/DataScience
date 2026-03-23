"""
Payment processing simulation engine.

Orchestrates the end-to-end payment lifecycle:
    1. Validate  — account number format, amount bounds, channel eligibility
    2. Duplicate check — idempotency key match within time window
    3. Authorize — simulate gateway authorization (with configurable failure rates)
    4. Capture  — mark funds as captured
    5. Post to CIS — notify the biller's CIS to apply the payment
    6. Settle   — move payment to settled state for reconciliation

Designed for high-volume batch processing as well as single-payment flows.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

from src.models.biller import Biller
from src.models.payment import (
    Payment,
    PaymentBatch,
    PaymentChannel,
    PaymentMethod,
    PaymentStatus,
    PaymentType,
)
from src.core.exception_handler import ExceptionHandler, ExceptionType
from src.utils.logger import get_logger, set_correlation_id
from src.utils.validators import (
    DuplicateDetector,
    ValidationResult,
    validate_account_number,
    validate_payment_amount,
    validate_payment_channel,
)

logger = get_logger(__name__)


class PaymentProcessingError(Exception):
    """Non-retryable payment processing failure."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class ProcessingResult:
    """Outcome of processing a single payment."""
    payment: Payment
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.payment.transaction_id,
            "status": self.payment.status.value,
            "success": self.success,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class PaymentProcessor:
    """
    Simulates payment processing with realistic failure modes.

    The processor validates each payment against the biller's rules, runs
    it through a simulated authorization gateway, and tracks it through
    capture and settlement. Configurable parameters control simulated
    failure rates, processing latency, and timeout behavior.
    """

    def __init__(
        self,
        exception_handler: Optional[ExceptionHandler] = None,
        auth_failure_rate: float = 0.03,
        timeout_rate: float = 0.01,
        processing_delay_ms: float = 0.0,
    ):
        self._exception_handler = exception_handler or ExceptionHandler()
        self._duplicate_detector = DuplicateDetector(window_hours=24)
        self._auth_failure_rate = auth_failure_rate
        self._timeout_rate = timeout_rate
        self._processing_delay_ms = processing_delay_ms

        # Counters for reporting
        self._processed_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_amount = Decimal("0.00")

    def process_payment(
        self,
        payment: Payment,
        biller: Biller,
    ) -> ProcessingResult:
        """
        Process a single payment through the full lifecycle.

        Returns a ProcessingResult indicating success or failure with details.
        """
        start_time = time.monotonic()
        cid = set_correlation_id(payment.correlation_id)

        logger.info(
            f"Processing payment {payment.transaction_id}",
            extra={"event_data": {
                "transaction_id": payment.transaction_id,
                "biller_id": biller.biller_id,
                "amount": str(payment.amount),
                "method": payment.payment_method.value,
            }},
        )

        try:
            # Step 1: Biller status check
            if not biller.is_active():
                raise PaymentProcessingError(
                    "BILLER_INACTIVE",
                    f"Biller {biller.biller_id} is not active (status: {biller.status.value})",
                )

            # Step 2: Validate
            self._validate(payment, biller)

            # Step 3: Duplicate check
            if self._duplicate_detector.is_duplicate(payment):
                raise PaymentProcessingError(
                    "DUPLICATE_PAYMENT",
                    f"Duplicate payment detected (idempotency_key: {payment.idempotency_key})",
                )

            payment.advance_status(PaymentStatus.VALIDATED)

            # Step 4: Authorize
            self._authorize(payment)

            # Step 5: Capture
            self._capture(payment)

            # Step 6: Settle (immediate for simulation)
            payment.advance_status(PaymentStatus.SETTLED)

            # Simulated processing delay
            if self._processing_delay_ms > 0:
                time.sleep(self._processing_delay_ms / 1000)

            elapsed = (time.monotonic() - start_time) * 1000
            self._processed_count += 1
            self._success_count += 1
            self._total_amount += payment.amount

            logger.info(
                f"Payment {payment.transaction_id} settled successfully",
                extra={"event_data": {
                    "transaction_id": payment.transaction_id,
                    "status": "settled",
                    "elapsed_ms": round(elapsed, 2),
                }},
            )

            return ProcessingResult(
                payment=payment,
                success=True,
                processing_time_ms=elapsed,
            )

        except PaymentProcessingError as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            payment.advance_status(PaymentStatus.FAILED, reason=str(exc))
            self._processed_count += 1
            self._failure_count += 1

            # Create exception case for tracking
            exc_type = ExceptionType.AUTHORIZATION_FAILURE
            if exc.code == "DUPLICATE_PAYMENT":
                exc_type = ExceptionType.DUPLICATE_PAYMENT
            elif exc.code in ("ACCT_FORMAT", "ACCT_LENGTH", "ACCT_EMPTY"):
                exc_type = ExceptionType.ACCOUNT_NOT_FOUND
            elif exc.code == "TIMEOUT":
                exc_type = ExceptionType.TIMEOUT

            self._exception_handler.create_case(
                exception_type=exc_type,
                description=str(exc),
                transaction_id=payment.transaction_id,
                biller_id=biller.biller_id,
                amount=payment.amount,
            )

            logger.warning(
                f"Payment {payment.transaction_id} failed: {exc.code}",
                extra={"event_data": {
                    "transaction_id": payment.transaction_id,
                    "error_code": exc.code,
                    "error": str(exc),
                }},
            )

            return ProcessingResult(
                payment=payment,
                success=False,
                error_code=exc.code,
                error_message=str(exc),
                processing_time_ms=elapsed,
            )

    def process_batch(
        self,
        payments: list[Payment],
        biller: Biller,
    ) -> list[ProcessingResult]:
        """Process a batch of payments sequentially."""
        results: list[ProcessingResult] = []

        logger.info(
            f"Starting batch processing: {len(payments)} payments for {biller.biller_id}",
            extra={"event_data": {
                "biller_id": biller.biller_id,
                "batch_size": len(payments),
            }},
        )

        for payment in payments:
            result = self.process_payment(payment, biller)
            results.append(result)

        success = sum(1 for r in results if r.success)
        failed = len(results) - success

        logger.info(
            f"Batch complete: {success}/{len(results)} succeeded, {failed} failed",
            extra={"event_data": {
                "total": len(results),
                "success": success,
                "failed": failed,
                "biller_id": biller.biller_id,
            }},
        )

        return results

    def get_stats(self) -> dict:
        return {
            "processed": self._processed_count,
            "success": self._success_count,
            "failed": self._failure_count,
            "total_amount": str(self._total_amount),
            "success_rate": (
                f"{self._success_count / self._processed_count:.1%}"
                if self._processed_count > 0
                else "N/A"
            ),
        }

    def reset_stats(self) -> None:
        self._processed_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_amount = Decimal("0.00")
        self._duplicate_detector.clear()

    # ---------------------------------------------------------------
    # Internal processing steps
    # ---------------------------------------------------------------

    def _validate(self, payment: Payment, biller: Biller) -> None:
        """Run all validation checks. Raises PaymentProcessingError on failure."""

        # Account number validation
        acct_result = validate_account_number(
            payment.customer_account_id, biller
        )
        if not acct_result.is_valid:
            err = acct_result.errors[0]
            raise PaymentProcessingError(err.code, str(err))

        # Amount validation
        amt_result = validate_payment_amount(
            payment.amount,
            biller,
            payment.payment_type.value,
        )
        if not amt_result.is_valid:
            err = amt_result.errors[0]
            raise PaymentProcessingError(err.code, str(err))

        # Channel validation
        chan_result = validate_payment_channel(
            biller,
            payment.payment_type.value,
            payment.channel.value,
        )
        if not chan_result.is_valid:
            err = chan_result.errors[0]
            raise PaymentProcessingError(err.code, str(err))

    def _authorize(self, payment: Payment) -> None:
        """Simulate payment gateway authorization."""

        # Simulate timeout
        if random.random() < self._timeout_rate:
            payment.advance_status(PaymentStatus.TIMEOUT)
            raise PaymentProcessingError(
                "TIMEOUT",
                "Authorization request timed out after 30 seconds",
            )

        # Simulate decline
        if random.random() < self._auth_failure_rate:
            decline_reasons = [
                ("INSUFFICIENT_FUNDS", "Insufficient funds"),
                ("CARD_DECLINED", "Card declined by issuer"),
                ("INVALID_CARD", "Invalid card number"),
                ("EXPIRED_CARD", "Card expired"),
                ("DO_NOT_HONOR", "Do not honor — contact issuing bank"),
            ]
            code, msg = random.choice(decline_reasons)
            raise PaymentProcessingError(code, msg)

        # Successful authorization
        payment.authorization_code = f"AUTH-{uuid.uuid4().hex[:8].upper()}"
        payment.gateway_reference = f"GW-{uuid.uuid4().hex[:12].upper()}"
        payment.processor_response_code = "00"  # Approved
        payment.advance_status(PaymentStatus.AUTHORIZED)

    def _capture(self, payment: Payment) -> None:
        """Capture the authorized funds."""
        payment.advance_status(PaymentStatus.CAPTURED)
