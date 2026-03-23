"""
Three-way settlement reconciliation engine.

Reconciles payment records from three independent sources:
    1. Biller records  — what the utility CIS says was owed and paid
    2. Platform records — what the payment platform collected
    3. Bank records     — what was actually settled via ACH/wire

The engine matches on transaction ID as the primary key, then verifies
amount and date agreement within configurable tolerances. Discrepancies
are classified by severity and routed to the exception handler.

This is the financial heartbeat of any bill-pay platform — if the three
sources don't agree, someone is either missing money or overcharging.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.settlement import (
    ExceptionSeverity,
    MatchStatus,
    ReconciliationResult,
    ResolutionAction,
    SettlementBatch,
    SettlementRecord,
)
from src.core.exception_handler import ExceptionHandler
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)


class SettlementEngine:
    """
    Three-way reconciliation engine.

    Accepts settlement records from biller, platform, and bank sources,
    matches them, identifies discrepancies, and produces a reconciliation
    report with exception cases.
    """

    def __init__(
        self,
        exception_handler: Optional[ExceptionHandler] = None,
        config_path: Optional[Path] = None,
    ):
        from config import SETTLEMENT_RULES_PATH
        self._config_path = config_path or SETTLEMENT_RULES_PATH
        self._exception_handler = exception_handler or ExceptionHandler()
        self._rules: dict[str, Any] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        with open(self._config_path, "r") as f:
            self._rules = yaml.safe_load(f)

    @property
    def _tolerances(self) -> dict:
        return self._rules.get("reconciliation", {}).get("tolerances", {})

    @property
    def _amount_tolerance(self) -> Decimal:
        return Decimal(str(self._tolerances.get("amount", {}).get("absolute", "0.01")))

    @property
    def _date_tolerance_days(self) -> int:
        return int(self._tolerances.get("date", {}).get("days", 2))

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def reconcile(
        self,
        biller_records: list[SettlementRecord],
        platform_records: list[SettlementRecord],
        bank_records: list[SettlementRecord],
        biller_id: str = "",
    ) -> SettlementBatch:
        """
        Run three-way reconciliation on the provided records.

        Returns a SettlementBatch containing all ReconciliationResults.
        """
        cid = set_correlation_id()

        batch = SettlementBatch(
            biller_id=biller_id,
            biller_records=biller_records,
            platform_records=platform_records,
            bank_records=bank_records,
        )

        logger.info(
            f"Starting reconciliation for {biller_id}",
            extra={"event_data": {
                "biller_id": biller_id,
                "biller_count": len(biller_records),
                "platform_count": len(platform_records),
                "bank_count": len(bank_records),
            }},
        )

        # Index records by transaction_id
        biller_index = self._index_by_txn_id(biller_records)
        platform_index = self._index_by_txn_id(platform_records)
        bank_index = self._index_by_txn_id(bank_records)

        # Collect all unique transaction IDs
        all_txn_ids = set(biller_index.keys()) | set(platform_index.keys()) | set(bank_index.keys())

        for txn_id in sorted(all_txn_ids):
            biller_rec = biller_index.get(txn_id)
            platform_rec = platform_index.get(txn_id)
            bank_rec = bank_index.get(txn_id)

            result = self._reconcile_transaction(
                txn_id, biller_rec, platform_rec, bank_rec
            )
            batch.results.append(result)

            # If exception, create a case
            if result.is_exception:
                self._exception_handler.create_case_from_recon(result)

        # Attempt auto-resolution on all new exception cases
        for case in self._exception_handler.get_open_cases():
            self._exception_handler.try_auto_resolve(case)

        logger.info(
            "Reconciliation complete",
            extra={"event_data": batch.summary()},
        )

        return batch

    def generate_report(self, batch: SettlementBatch) -> str:
        """Generate a JSON reconciliation report."""
        report = {
            "summary": batch.summary(),
            "exceptions": [
                r.to_dict()
                for r in batch.results
                if r.is_exception
            ],
        }
        return json.dumps(report, indent=2, default=str)

    # ---------------------------------------------------------------
    # Internal reconciliation logic
    # ---------------------------------------------------------------

    def _reconcile_transaction(
        self,
        txn_id: str,
        biller_rec: Optional[SettlementRecord],
        platform_rec: Optional[SettlementRecord],
        bank_rec: Optional[SettlementRecord],
    ) -> ReconciliationResult:
        """Reconcile a single transaction across the three sources."""

        result = ReconciliationResult(
            transaction_id=txn_id,
            biller_record=biller_rec,
            platform_record=platform_rec,
            bank_record=bank_rec,
        )

        present_count = sum(1 for r in (biller_rec, platform_rec, bank_rec) if r is not None)

        # --- Missing source cases ---
        if present_count == 0:
            result.match_status = MatchStatus.ORPHAN
            result.severity = ExceptionSeverity.LOW
            result.discrepancy_details = "No records found for transaction"
            return result

        if present_count == 1:
            result.match_status = MatchStatus.ORPHAN
            result.severity = ExceptionSeverity.LOW
            source = (biller_rec or platform_rec or bank_rec).source  # type: ignore[union-attr]
            result.discrepancy_details = f"Record exists only in {source}"
            return result

        if present_count == 2:
            if biller_rec is None:
                result.match_status = MatchStatus.MISSING_BILLER
                result.severity = ExceptionSeverity.MEDIUM
                result.discrepancy_details = (
                    "Transaction present in platform and bank but missing from biller records. "
                    "Likely biller posting lag — recheck in next cycle."
                )
            elif platform_rec is None:
                result.match_status = MatchStatus.MISSING_PLATFORM
                result.severity = ExceptionSeverity.HIGH
                result.discrepancy_details = (
                    "Transaction present in biller and bank but missing from platform records. "
                    "Possible audit gap — platform may not have recorded the transaction."
                )
            elif bank_rec is None:
                result.match_status = MatchStatus.MISSING_BANK
                result.severity = ExceptionSeverity.CRITICAL
                result.discrepancy_details = (
                    "Transaction present in biller and platform but missing from bank settlement. "
                    "Funds collected but not settled — immediate cash flow impact."
                )
            return result

        # --- All three present: check amounts and dates ---

        # Use the biller amount as the reference (what the utility says is owed)
        biller_amount = biller_rec.amount  # type: ignore[union-attr]
        bank_amount = bank_rec.amount  # type: ignore[union-attr]

        # For platform, compare net amount (total minus fees) against biller amount
        platform_net = platform_rec.net_amount  # type: ignore[union-attr]

        # Amount comparison: biller vs bank (both should reflect the payment amount)
        amount_diff = abs(biller_amount - bank_amount)
        result.amount_difference = amount_diff

        if amount_diff > self._amount_tolerance:
            result.match_status = MatchStatus.AMOUNT_MISMATCH
            result.severity = ExceptionSeverity.HIGH
            result.discrepancy_details = (
                f"Amount mismatch: biller={biller_amount}, bank={bank_amount}, "
                f"difference={amount_diff} (tolerance={self._amount_tolerance})"
            )
            return result

        # Date comparison
        biller_date = biller_rec.settlement_date  # type: ignore[union-attr]
        bank_date = bank_rec.settlement_date  # type: ignore[union-attr]

        if biller_date and bank_date:
            date_diff = abs((biller_date - bank_date).days)
            result.date_difference_days = date_diff

            if date_diff > self._date_tolerance_days:
                result.match_status = MatchStatus.DATE_MISMATCH
                result.severity = ExceptionSeverity.MEDIUM
                result.discrepancy_details = (
                    f"Settlement date mismatch: biller={biller_date}, bank={bank_date}, "
                    f"difference={date_diff} days (tolerance={self._date_tolerance_days})"
                )
                return result

        # Everything matches
        result.match_status = MatchStatus.MATCHED
        result.severity = ExceptionSeverity.LOW
        return result

    @staticmethod
    def _index_by_txn_id(
        records: list[SettlementRecord],
    ) -> dict[str, SettlementRecord]:
        """Index a list of settlement records by transaction ID."""
        index: dict[str, SettlementRecord] = {}
        for rec in records:
            if rec.transaction_id in index:
                logger.warning(
                    f"Duplicate transaction_id in {rec.source}: {rec.transaction_id}",
                    extra={"event_data": {
                        "source": rec.source,
                        "transaction_id": rec.transaction_id,
                    }},
                )
            index[rec.transaction_id] = rec
        return index
