"""
Settlement and reconciliation models.

Defines the data structures used by the three-way settlement reconciliation
engine. Each settlement cycle produces a batch of SettlementRecords that are
then matched across three sources (biller, platform, bank) to produce
ReconciliationResults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid


class MatchStatus(str, Enum):
    """Outcome of the three-way reconciliation for a single transaction."""
    MATCHED = "matched"
    PARTIAL_MATCH = "partial_match"
    AMOUNT_MISMATCH = "amount_mismatch"
    DATE_MISMATCH = "date_mismatch"
    MISSING_BILLER = "missing_biller"
    MISSING_PLATFORM = "missing_platform"
    MISSING_BANK = "missing_bank"
    DUPLICATE = "duplicate"
    ORPHAN = "orphan"


class ExceptionSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResolutionAction(str, Enum):
    AUTO_MATCH = "auto_match"
    DEFER = "defer"
    WRITE_OFF = "write_off"
    MANUAL_REVIEW = "manual_review"
    ESCALATE = "escalate"


@dataclass
class SettlementRecord:
    """
    A single transaction's settlement data from one of the three sources.

    During reconciliation, we compare SettlementRecords from biller, platform,
    and bank to look for agreement or discrepancies.
    """
    transaction_id: str
    source: str  # "biller", "platform", or "bank"
    customer_account_id: str = ""
    amount: Decimal = Decimal("0.00")
    fee_amount: Decimal = Decimal("0.00")
    net_amount: Decimal = Decimal("0.00")
    payment_date: Optional[date] = None
    settlement_date: Optional[date] = None
    biller_id: str = ""
    reference_number: Optional[str] = None
    record_timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.net_amount == Decimal("0.00") and self.amount > Decimal("0.00"):
            self.net_amount = self.amount - self.fee_amount


@dataclass
class ReconciliationResult:
    """
    The outcome of reconciling a single transaction across all three sources.

    Captures the match status, any discrepancies found, and the resolution
    action (if one was applied automatically).
    """
    reconciliation_id: str = field(
        default_factory=lambda: f"RECON-{uuid.uuid4().hex[:12].upper()}"
    )
    transaction_id: str = ""
    match_status: MatchStatus = MatchStatus.MATCHED
    severity: ExceptionSeverity = ExceptionSeverity.LOW

    # The records from each source (None if missing from that source)
    biller_record: Optional[SettlementRecord] = None
    platform_record: Optional[SettlementRecord] = None
    bank_record: Optional[SettlementRecord] = None

    # Discrepancy details
    amount_difference: Decimal = Decimal("0.00")
    date_difference_days: int = 0
    discrepancy_details: str = ""

    # Resolution
    resolution_action: Optional[ResolutionAction] = None
    resolution_reason: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_exception(self) -> bool:
        return self.match_status != MatchStatus.MATCHED

    @property
    def is_resolved(self) -> bool:
        return self.resolution_action is not None

    def to_dict(self) -> dict:
        return {
            "reconciliation_id": self.reconciliation_id,
            "transaction_id": self.transaction_id,
            "match_status": self.match_status.value,
            "severity": self.severity.value,
            "amount_difference": str(self.amount_difference),
            "date_difference_days": self.date_difference_days,
            "discrepancy_details": self.discrepancy_details,
            "resolution_action": self.resolution_action.value if self.resolution_action else None,
            "resolution_reason": self.resolution_reason,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SettlementBatch:
    """
    A batch of settlement records processed in a single reconciliation cycle.

    Aggregates results and provides summary statistics for reporting.
    """
    batch_id: str = field(
        default_factory=lambda: f"SETTLE-{uuid.uuid4().hex[:12].upper()}"
    )
    biller_id: str = ""
    settlement_date: date = field(default_factory=date.today)
    created_at: datetime = field(default_factory=datetime.utcnow)

    biller_records: list[SettlementRecord] = field(default_factory=list)
    platform_records: list[SettlementRecord] = field(default_factory=list)
    bank_records: list[SettlementRecord] = field(default_factory=list)

    results: list[ReconciliationResult] = field(default_factory=list)

    @property
    def total_biller_amount(self) -> Decimal:
        return sum((r.amount for r in self.biller_records), Decimal("0.00"))

    @property
    def total_platform_amount(self) -> Decimal:
        return sum((r.amount for r in self.platform_records), Decimal("0.00"))

    @property
    def total_bank_amount(self) -> Decimal:
        return sum((r.amount for r in self.bank_records), Decimal("0.00"))

    @property
    def matched_count(self) -> int:
        return sum(1 for r in self.results if r.match_status == MatchStatus.MATCHED)

    @property
    def exception_count(self) -> int:
        return sum(1 for r in self.results if r.is_exception)

    @property
    def match_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.matched_count / len(self.results)

    def summary(self) -> dict:
        """Generate a summary report of the settlement batch."""
        status_counts: dict[str, int] = {}
        for r in self.results:
            key = r.match_status.value
            status_counts[key] = status_counts.get(key, 0) + 1

        severity_counts: dict[str, int] = {}
        for r in self.results:
            if r.is_exception:
                key = r.severity.value
                severity_counts[key] = severity_counts.get(key, 0) + 1

        return {
            "batch_id": self.batch_id,
            "biller_id": self.biller_id,
            "settlement_date": self.settlement_date.isoformat(),
            "record_counts": {
                "biller": len(self.biller_records),
                "platform": len(self.platform_records),
                "bank": len(self.bank_records),
            },
            "amounts": {
                "biller_total": str(self.total_biller_amount),
                "platform_total": str(self.total_platform_amount),
                "bank_total": str(self.total_bank_amount),
            },
            "reconciliation": {
                "total_results": len(self.results),
                "matched": self.matched_count,
                "exceptions": self.exception_count,
                "match_rate": f"{self.match_rate:.1%}",
                "by_status": status_counts,
                "exceptions_by_severity": severity_counts,
            },
        }
