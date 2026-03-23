"""Tests for the payment processing engine."""

import pytest
from decimal import Decimal

from src.core.onboarding import OnboardingEngine
from src.core.payment_processor import PaymentProcessor, PaymentProcessingError
from src.core.exception_handler import ExceptionHandler
from src.models.biller import Biller, BillerStatus
from src.models.payment import (
    Payment,
    PaymentChannel,
    PaymentMethod,
    PaymentStatus,
    PaymentType,
)


@pytest.fixture
def active_biller() -> Biller:
    """Onboard and activate the first biller from config."""
    engine = OnboardingEngine()
    configs = engine.get_raw_biller_configs()
    biller, _ = engine.onboard_biller(configs[0])
    engine.activate_biller(biller.biller_id)
    return biller


@pytest.fixture
def exception_handler() -> ExceptionHandler:
    return ExceptionHandler()


@pytest.fixture
def processor(exception_handler: ExceptionHandler) -> PaymentProcessor:
    return PaymentProcessor(
        exception_handler=exception_handler,
        auth_failure_rate=0.0,   # No random failures in tests
        timeout_rate=0.0,
    )


def _make_payment(
    biller_id: str = "UTIL-GA-POWER-001",
    account_id: str = "1234567890",
    amount: float = 150.00,
    method: PaymentMethod = PaymentMethod.ACH,
    channel: PaymentChannel = PaymentChannel.WEB,
    payment_type: PaymentType = PaymentType.ONE_TIME,
) -> Payment:
    """Helper to create a test payment."""
    return Payment(
        biller_id=biller_id,
        customer_account_id=account_id,
        amount=Decimal(str(amount)),
        payment_method=method,
        channel=channel,
        payment_type=payment_type,
    )


class TestPaymentProcessor:
    """Tests for payment processing pipeline."""

    def test_successful_payment(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """A valid payment should go through to settled."""
        payment = _make_payment()
        result = processor.process_payment(payment, active_biller)

        assert result.success
        assert payment.status == PaymentStatus.SETTLED
        assert payment.authorization_code is not None

    def test_invalid_account_format(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Account number not matching format regex should fail validation."""
        payment = _make_payment(account_id="ABC")  # Not 10 digits
        result = processor.process_payment(payment, active_biller)

        assert not result.success
        assert payment.status == PaymentStatus.FAILED
        assert result.error_code in ("ACCT_FORMAT", "ACCT_LENGTH")

    def test_amount_below_minimum(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Payment below the biller's minimum should fail."""
        payment = _make_payment(amount=0.50)  # Below $1.00 minimum
        result = processor.process_payment(payment, active_biller)

        assert not result.success
        assert result.error_code == "AMT_BELOW_MIN"

    def test_amount_above_maximum(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Payment above the biller's maximum should fail."""
        payment = _make_payment(amount=50000.00)  # Above $25,000 max
        result = processor.process_payment(payment, active_biller)

        assert not result.success
        assert result.error_code == "AMT_ABOVE_MAX"

    def test_unsupported_channel(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Payment via an unsupported channel should fail."""
        payment = _make_payment(channel=PaymentChannel.KIOSK)
        result = processor.process_payment(payment, active_biller)

        assert not result.success
        assert result.error_code == "CHANNEL_NOT_SUPPORTED"

    def test_inactive_biller_rejected(
        self, processor: PaymentProcessor,
    ) -> None:
        """Payments to an inactive biller should be rejected."""
        engine = OnboardingEngine()
        configs = engine.get_raw_biller_configs()
        # Use the pending biller (Pacific Coast, index 2)
        biller, _ = engine.onboard_biller(configs[2])
        # Don't activate it

        payment = _make_payment(biller_id=biller.biller_id)
        result = processor.process_payment(payment, biller)

        assert not result.success
        assert result.error_code == "BILLER_INACTIVE"

    def test_duplicate_detection(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Submitting the same payment twice should be caught as duplicate."""
        payment1 = _make_payment()
        payment2 = _make_payment()
        # Force same idempotency key
        payment2.idempotency_key = payment1.idempotency_key

        result1 = processor.process_payment(payment1, active_biller)
        result2 = processor.process_payment(payment2, active_biller)

        assert result1.success
        assert not result2.success
        assert result2.error_code == "DUPLICATE_PAYMENT"

    def test_batch_processing(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Batch processing should handle multiple payments."""
        payments = [
            _make_payment(account_id=f"{i:010d}") for i in range(1000000000, 1000000005)
        ]

        results = processor.process_batch(payments, active_biller)

        assert len(results) == 5
        assert all(r.success for r in results)

    def test_processing_stats(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Stats should track processed, success, and failure counts."""
        processor.reset_stats()

        # One success
        good = _make_payment(account_id="1111111111")
        processor.process_payment(good, active_biller)

        # One failure (bad account)
        bad = _make_payment(account_id="BAD")
        processor.process_payment(bad, active_biller)

        stats = processor.get_stats()
        assert stats["processed"] == 2
        assert stats["success"] == 1
        assert stats["failed"] == 1

    def test_payment_timestamps(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Settled payments should have all lifecycle timestamps populated."""
        payment = _make_payment()
        processor.process_payment(payment, active_biller)

        assert payment.validated_at is not None
        assert payment.authorized_at is not None
        assert payment.captured_at is not None
        assert payment.settled_at is not None

    def test_exception_cases_created_on_failure(
        self,
        processor: PaymentProcessor,
        active_biller: Biller,
        exception_handler: ExceptionHandler,
    ) -> None:
        """Failed payments should create exception cases."""
        payment = _make_payment(account_id="INVALID")
        processor.process_payment(payment, active_biller)

        cases = exception_handler.get_open_cases()
        assert len(cases) >= 1

    def test_zero_amount_rejected(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Zero-amount payment should be rejected."""
        payment = _make_payment(amount=0.00)
        result = processor.process_payment(payment, active_biller)

        assert not result.success
        assert result.error_code == "AMT_NEGATIVE"

    def test_negative_amount_rejected(
        self, processor: PaymentProcessor, active_biller: Biller,
    ) -> None:
        """Negative amount should be rejected."""
        payment = _make_payment(amount=-50.00)
        result = processor.process_payment(payment, active_biller)

        assert not result.success

    def test_payment_to_dict(self) -> None:
        """Payment serialization should include all key fields."""
        payment = _make_payment()
        d = payment.to_dict()

        assert "transaction_id" in d
        assert "amount" in d
        assert "status" in d
        assert d["payment_method"] == "ach"
