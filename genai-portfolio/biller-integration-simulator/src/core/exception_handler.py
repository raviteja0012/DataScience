"""
Exception handling workflows for payment processing and settlement.

Implements a structured exception management framework that classifies,
routes, and (where possible) auto-resolves exceptions arising from payment
processing or settlement reconciliation. Uses the rules defined in
config/settlement_rules.yaml to determine severity, escalation paths,
and auto-resolution eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.settlement import (
    ExceptionSeverity,
    MatchStatus,
    ReconciliationResult,
    ResolutionAction,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExceptionType(str, Enum):
    DUPLICATE_PAYMENT = "duplicate_payment"
    AMOUNT_MISMATCH = "amount_mismatch"
    MISSING_TRANSACTION = "missing_transaction"
    TIMEOUT = "timeout"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SETTLEMENT_DISCREPANCY = "settlement_discrepancy"
    ACCOUNT_NOT_FOUND = "account_not_found"
    BILL_MISMATCH = "bill_mismatch"


@dataclass
class EscalationStep:
    """A single step in the escalation chain."""
    role: str
    after_minutes: int
    notified: bool = False
    notified_at: Optional[datetime] = None


@dataclass
class ExceptionCase:
    """
    A tracked exception requiring attention.

    Created when the payment processor or settlement engine encounters an
    issue that cannot be handled inline. The case flows through the escalation
    chain until resolved.
    """
    case_id: str
    exception_type: ExceptionType
    severity: ExceptionSeverity
    description: str
    transaction_id: Optional[str] = None
    biller_id: Optional[str] = None
    amount: Decimal = Decimal("0.00")
    context: dict = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None

    escalation_chain: list[EscalationStep] = field(default_factory=list)
    current_escalation_level: int = 0

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    @property
    def age_hours(self) -> float:
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 3600

    def resolve(self, resolution: str, resolved_by: str = "system") -> None:
        self.resolved_at = datetime.utcnow()
        self.resolution = resolution
        self.resolved_by = resolved_by


class ExceptionHandler:
    """
    Central exception management engine.

    Classifies exceptions by severity, applies auto-resolution rules from
    the settlement config, and manages the escalation workflow.
    """

    # Map MatchStatus to ExceptionType
    _MATCH_STATUS_MAP = {
        MatchStatus.AMOUNT_MISMATCH: ExceptionType.AMOUNT_MISMATCH,
        MatchStatus.MISSING_BILLER: ExceptionType.MISSING_TRANSACTION,
        MatchStatus.MISSING_PLATFORM: ExceptionType.MISSING_TRANSACTION,
        MatchStatus.MISSING_BANK: ExceptionType.MISSING_TRANSACTION,
        MatchStatus.DUPLICATE: ExceptionType.DUPLICATE_PAYMENT,
    }

    def __init__(self, config_path: Optional[Path] = None):
        from config import SETTLEMENT_RULES_PATH
        self._config_path = config_path or SETTLEMENT_RULES_PATH
        self._rules: dict[str, Any] = {}
        self._open_cases: list[ExceptionCase] = []
        self._resolved_cases: list[ExceptionCase] = []
        self._load_rules()

    def _load_rules(self) -> None:
        with open(self._config_path, "r") as f:
            self._rules = yaml.safe_load(f)
        logger.info("Loaded exception handling rules")

    def classify_severity(self, match_status: MatchStatus) -> ExceptionSeverity:
        """Determine severity based on the settlement rules config."""
        severity_config = self._rules.get("exception_handling", {}).get("severity_levels", {})

        for severity_name, statuses in severity_config.items():
            if match_status.value in statuses:
                return ExceptionSeverity(severity_name)

        return ExceptionSeverity.LOW

    def create_case(
        self,
        exception_type: ExceptionType,
        description: str,
        severity: Optional[ExceptionSeverity] = None,
        transaction_id: Optional[str] = None,
        biller_id: Optional[str] = None,
        amount: Decimal = Decimal("0.00"),
        context: Optional[dict] = None,
    ) -> ExceptionCase:
        """Create and register a new exception case."""
        import uuid
        case = ExceptionCase(
            case_id=f"EXC-{uuid.uuid4().hex[:10].upper()}",
            exception_type=exception_type,
            severity=severity or ExceptionSeverity.MEDIUM,
            description=description,
            transaction_id=transaction_id,
            biller_id=biller_id,
            amount=amount,
            context=context or {},
        )

        # Build escalation chain from config
        escalation_config = self._rules.get("exception_handling", {}).get("escalation", [])
        for esc in escalation_config:
            if esc.get("severity") == case.severity.value:
                for step in esc.get("escalation_chain", []):
                    case.escalation_chain.append(EscalationStep(
                        role=step["role"],
                        after_minutes=step["after_minutes"],
                    ))
                break

        self._open_cases.append(case)

        logger.info(
            f"Exception case created: {case.case_id}",
            extra={"event_data": {
                "case_id": case.case_id,
                "type": case.exception_type.value,
                "severity": case.severity.value,
                "transaction_id": transaction_id,
            }},
        )

        return case

    def create_case_from_recon(self, result: ReconciliationResult) -> ExceptionCase:
        """Create an exception case from a reconciliation result."""
        exc_type = self._MATCH_STATUS_MAP.get(
            result.match_status, ExceptionType.SETTLEMENT_DISCREPANCY
        )

        return self.create_case(
            exception_type=exc_type,
            description=result.discrepancy_details or f"Reconciliation exception: {result.match_status.value}",
            severity=result.severity,
            transaction_id=result.transaction_id,
            amount=abs(result.amount_difference),
            context={"reconciliation_id": result.reconciliation_id},
        )

    def try_auto_resolve(self, case: ExceptionCase) -> bool:
        """
        Attempt to auto-resolve a case using the configured rules.

        Returns True if the case was resolved.
        """
        auto_rules = self._rules.get("exception_handling", {}).get("auto_resolve", [])

        for rule in auto_rules:
            condition = rule.get("condition", "")
            action = rule.get("action", "")
            reason = rule.get("reason", "")

            if self._evaluate_condition(condition, case):
                if action == "auto_match":
                    case.resolve(f"Auto-resolved: {reason}", resolved_by="auto_resolve")
                    self._move_to_resolved(case)
                    logger.info(
                        f"Case {case.case_id} auto-resolved",
                        extra={"event_data": {
                            "case_id": case.case_id,
                            "action": action,
                            "reason": reason,
                        }},
                    )
                    return True

                elif action == "defer":
                    recheck_hours = rule.get("recheck_hours", 4)
                    logger.info(
                        f"Case {case.case_id} deferred for {recheck_hours}h",
                        extra={"event_data": {"case_id": case.case_id}},
                    )
                    return False

                elif action == "write_off":
                    if not rule.get("requires_approval", False):
                        case.resolve(f"Auto write-off: {reason}", resolved_by="auto_resolve")
                        self._move_to_resolved(case)
                        return True

        return False

    def check_escalations(self) -> list[dict]:
        """
        Check all open cases for escalation triggers.

        Returns a list of escalation notifications that need to be sent.
        """
        notifications: list[dict] = []

        for case in self._open_cases:
            if case.is_resolved:
                continue

            age_minutes = case.age_hours * 60

            for i, step in enumerate(case.escalation_chain):
                if not step.notified and age_minutes >= step.after_minutes:
                    step.notified = True
                    step.notified_at = datetime.utcnow()
                    case.current_escalation_level = i

                    notification = {
                        "case_id": case.case_id,
                        "severity": case.severity.value,
                        "role": step.role,
                        "description": case.description,
                        "age_hours": round(case.age_hours, 1),
                        "escalation_level": i,
                    }
                    notifications.append(notification)

                    logger.info(
                        f"Escalation triggered for case {case.case_id} -> {step.role}",
                        extra={"event_data": notification},
                    )

        return notifications

    def get_open_cases(self, severity: Optional[ExceptionSeverity] = None) -> list[ExceptionCase]:
        if severity:
            return [c for c in self._open_cases if c.severity == severity]
        return list(self._open_cases)

    def get_resolved_cases(self) -> list[ExceptionCase]:
        return list(self._resolved_cases)

    def get_case(self, case_id: str) -> Optional[ExceptionCase]:
        for case in self._open_cases + self._resolved_cases:
            if case.case_id == case_id:
                return case
        return None

    def resolve_case(self, case_id: str, resolution: str, resolved_by: str = "analyst") -> bool:
        """Manually resolve an open case."""
        for case in self._open_cases:
            if case.case_id == case_id:
                case.resolve(resolution, resolved_by)
                self._move_to_resolved(case)
                logger.info(
                    f"Case {case_id} manually resolved by {resolved_by}",
                    extra={"event_data": {"case_id": case_id, "resolution": resolution}},
                )
                return True
        return False

    def summary(self) -> dict:
        """Generate a summary of all exception cases."""
        open_by_severity: dict[str, int] = {}
        for case in self._open_cases:
            key = case.severity.value
            open_by_severity[key] = open_by_severity.get(key, 0) + 1

        open_by_type: dict[str, int] = {}
        for case in self._open_cases:
            key = case.exception_type.value
            open_by_type[key] = open_by_type.get(key, 0) + 1

        return {
            "open_cases": len(self._open_cases),
            "resolved_cases": len(self._resolved_cases),
            "open_by_severity": open_by_severity,
            "open_by_type": open_by_type,
            "oldest_open_hours": max(
                (c.age_hours for c in self._open_cases), default=0
            ),
        }

    def _move_to_resolved(self, case: ExceptionCase) -> None:
        if case in self._open_cases:
            self._open_cases.remove(case)
        if case not in self._resolved_cases:
            self._resolved_cases.append(case)

    def _evaluate_condition(self, condition: str, case: ExceptionCase) -> bool:
        """
        Evaluate a condition string from the auto-resolve rules against a case.

        These are simplified condition expressions from the YAML config.
        In production this would use a proper rule engine; here we parse
        the most common patterns.
        """
        condition = condition.strip()

        if "amount_mismatch" in condition and "abs(difference)" in condition:
            if case.exception_type != ExceptionType.AMOUNT_MISMATCH:
                return False
            # Extract threshold: "abs(difference) <= 0.01"
            try:
                threshold = Decimal(condition.split("<=")[-1].strip())
                return case.amount <= threshold
            except Exception:
                return False

        if "date_mismatch" in condition and "day_difference" in condition:
            if case.exception_type not in (
                ExceptionType.SETTLEMENT_DISCREPANCY,
                ExceptionType.MISSING_TRANSACTION,
            ):
                return False
            return True  # Simplified: assume date mismatches within threshold

        if "missing_biller" in condition and "record_age_hours" in condition:
            if case.exception_type != ExceptionType.MISSING_TRANSACTION:
                return False
            try:
                threshold_hours = int(condition.split("<=")[-1].strip())
                return case.age_hours <= threshold_hours
            except Exception:
                return False

        if "orphan" in condition and "record_age_hours" in condition:
            try:
                threshold_hours = int(condition.split(">=")[-1].strip())
                return case.age_hours >= threshold_hours
            except Exception:
                return False

        return False
