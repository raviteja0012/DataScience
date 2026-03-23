"""Tests for the three-way settlement reconciliation engine."""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from src.core.settlement_engine import SettlementEngine
from src.core.exception_handler import ExceptionHandler
from src.models.settlement import (
    ExceptionSeverity,
    MatchStatus,
    SettlementRecord,
)


@pytest.fixture
def exception_handler() -> ExceptionHandler:
    return ExceptionHandler()


@pytest.fixture
def engine(exception_handler: ExceptionHandler) -> SettlementEngine:
    return SettlementEngine(exception_handler=exception_handler)


def _make_record(
    txn_id: str,
    source: str,
    amount: float = 100.00,
    settlement_date: date = None,
) -> SettlementRecord:
    """Helper to create a settlement record."""
    return SettlementRecord(
        transaction_id=txn_id,
        source=source,
        customer_account_id="1234567890",
        amount=Decimal(str(amount)),
        payment_date=date.today(),
        settlement_date=settlement_date or date.today(),
        biller_id="UTIL-GA-POWER-001",
    )


class TestSettlementEngine:
    """Tests for three-way reconciliation logic."""

    def test_perfect_match(self, engine: SettlementEngine) -> None:
        """All three records agree — should produce a MATCHED result."""
        txn = "TXN-001"
        biller = [_make_record(txn, "biller", 100.00)]
        platform = [_make_record(txn, "platform", 100.00)]
        bank = [_make_record(txn, "bank", 100.00)]

        batch = engine.reconcile(biller, platform, bank)

        assert len(batch.results) == 1
        assert batch.results[0].match_status == MatchStatus.MATCHED
        assert batch.matched_count == 1
        assert batch.exception_count == 0

    def test_amount_mismatch(self, engine: SettlementEngine) -> None:
        """Amounts differ beyond tolerance — should flag as AMOUNT_MISMATCH."""
        txn = "TXN-002"
        biller = [_make_record(txn, "biller", 100.00)]
        platform = [_make_record(txn, "platform", 100.00)]
        bank = [_make_record(txn, "bank", 95.00)]  # $5 off

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.AMOUNT_MISMATCH
        assert batch.results[0].severity == ExceptionSeverity.HIGH

    def test_penny_tolerance(self, engine: SettlementEngine) -> None:
        """A $0.01 difference should still match (within tolerance)."""
        txn = "TXN-003"
        biller = [_make_record(txn, "biller", 100.00)]
        platform = [_make_record(txn, "platform", 100.00)]
        bank = [_make_record(txn, "bank", 100.01)]

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.MATCHED

    def test_missing_bank_record(self, engine: SettlementEngine) -> None:
        """Transaction in biller and platform but not bank — CRITICAL."""
        txn = "TXN-004"
        biller = [_make_record(txn, "biller")]
        platform = [_make_record(txn, "platform")]
        bank: list[SettlementRecord] = []

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.MISSING_BANK
        assert batch.results[0].severity == ExceptionSeverity.CRITICAL

    def test_missing_biller_record(self, engine: SettlementEngine) -> None:
        """Transaction in platform and bank but not biller — MEDIUM severity."""
        txn = "TXN-005"
        biller: list[SettlementRecord] = []
        platform = [_make_record(txn, "platform")]
        bank = [_make_record(txn, "bank")]

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.MISSING_BILLER
        assert batch.results[0].severity == ExceptionSeverity.MEDIUM

    def test_missing_platform_record(self, engine: SettlementEngine) -> None:
        """Transaction in biller and bank but not platform — HIGH severity."""
        txn = "TXN-006"
        biller = [_make_record(txn, "biller")]
        platform: list[SettlementRecord] = []
        bank = [_make_record(txn, "bank")]

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.MISSING_PLATFORM
        assert batch.results[0].severity == ExceptionSeverity.HIGH

    def test_orphan_single_source(self, engine: SettlementEngine) -> None:
        """Record in only one source is an ORPHAN."""
        txn = "TXN-007"
        biller = [_make_record(txn, "biller")]
        platform: list[SettlementRecord] = []
        bank: list[SettlementRecord] = []

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.ORPHAN
        assert batch.results[0].severity == ExceptionSeverity.LOW

    def test_date_mismatch(self, engine: SettlementEngine) -> None:
        """Settlement dates differ beyond tolerance — DATE_MISMATCH."""
        txn = "TXN-008"
        today = date.today()
        biller = [_make_record(txn, "biller", 100.00, settlement_date=today)]
        platform = [_make_record(txn, "platform", 100.00, settlement_date=today)]
        bank = [_make_record(txn, "bank", 100.00,
                             settlement_date=today + timedelta(days=5))]

        batch = engine.reconcile(biller, platform, bank)

        assert batch.results[0].match_status == MatchStatus.DATE_MISMATCH

    def test_multiple_transactions(self, engine: SettlementEngine) -> None:
        """Reconcile multiple transactions in one batch."""
        biller = [
            _make_record("T1", "biller", 50.00),
            _make_record("T2", "biller", 75.00),
            _make_record("T3", "biller", 120.00),
        ]
        platform = [
            _make_record("T1", "platform", 50.00),
            _make_record("T2", "platform", 75.00),
            _make_record("T3", "platform", 120.00),
        ]
        bank = [
            _make_record("T1", "bank", 50.00),
            _make_record("T2", "bank", 75.00),
            _make_record("T3", "bank", 120.00),
        ]

        batch = engine.reconcile(biller, platform, bank)

        assert len(batch.results) == 3
        assert batch.matched_count == 3
        assert batch.exception_count == 0
        assert batch.match_rate == 1.0

    def test_mixed_results(self, engine: SettlementEngine) -> None:
        """Batch with some matches and some exceptions."""
        biller = [
            _make_record("T1", "biller", 100.00),
            _make_record("T2", "biller", 200.00),
        ]
        platform = [
            _make_record("T1", "platform", 100.00),
            _make_record("T2", "platform", 200.00),
        ]
        bank = [
            _make_record("T1", "bank", 100.00),
            _make_record("T2", "bank", 190.00),  # Mismatch
        ]

        batch = engine.reconcile(biller, platform, bank)

        assert batch.matched_count == 1
        assert batch.exception_count == 1

    def test_batch_summary(self, engine: SettlementEngine) -> None:
        """Summary report should contain all expected fields."""
        biller = [_make_record("T1", "biller", 100.00)]
        platform = [_make_record("T1", "platform", 100.00)]
        bank = [_make_record("T1", "bank", 100.00)]

        batch = engine.reconcile(biller, platform, bank, biller_id="TEST-001")
        summary = batch.summary()

        assert "batch_id" in summary
        assert "record_counts" in summary
        assert "amounts" in summary
        assert "reconciliation" in summary
        assert summary["biller_id"] == "TEST-001"

    def test_generate_report(self, engine: SettlementEngine) -> None:
        """JSON report generation should not raise."""
        biller = [_make_record("T1", "biller", 100.00)]
        platform = [_make_record("T1", "platform", 100.00)]
        bank = [_make_record("T1", "bank", 90.00)]

        batch = engine.reconcile(biller, platform, bank)
        report = engine.generate_report(batch)

        assert isinstance(report, str)
        assert "amount_mismatch" in report

    def test_exception_cases_created(
        self,
        engine: SettlementEngine,
        exception_handler: ExceptionHandler,
    ) -> None:
        """Reconciliation exceptions should create exception cases."""
        txn = "TXN-EXC"
        biller = [_make_record(txn, "biller", 100.00)]
        platform = [_make_record(txn, "platform", 100.00)]
        bank: list[SettlementRecord] = []  # Missing

        engine.reconcile(biller, platform, bank)

        cases = exception_handler.get_open_cases()
        # Should have at least one case (some might be auto-resolved)
        total = len(cases) + len(exception_handler.get_resolved_cases())
        assert total >= 1
