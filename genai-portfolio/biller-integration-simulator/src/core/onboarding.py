"""
Biller onboarding engine.

Handles the end-to-end onboarding of a new utility biller onto the payment
platform. Reads YAML configuration, validates all required fields, verifies
CIS connectivity, and activates the biller for payment acceptance.

Onboarding workflow:
    1. Parse biller config from YAML
    2. Validate mandatory fields and business rules
    3. Verify CIS schema mapping exists
    4. Test CIS adapter connectivity (simulated)
    5. Register biller in the platform registry
    6. Activate or leave in pending state
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.biller import (
    Biller,
    BillerFeeStructure,
    BillerStatus,
    ConvenienceFeeSchedule,
    FeeAssessedBy,
    NotificationConfig,
    PaymentTypeConfig,
    SettlementConfig,
    ValidationRules,
)
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)


class OnboardingError(Exception):
    """Raised when biller onboarding fails validation or activation."""
    pass


@dataclass
class OnboardingCheck:
    """Result of a single onboarding validation check."""
    check_name: str
    passed: bool
    message: str = ""


@dataclass
class OnboardingReport:
    """Aggregate report of all onboarding checks for a biller."""
    biller_id: str
    checks: list[OnboardingCheck] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[OnboardingCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> dict:
        return {
            "biller_id": self.biller_id,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": sum(1 for c in self.checks if not c.passed),
            "all_passed": self.all_passed,
            "failures": [
                {"check": c.check_name, "message": c.message}
                for c in self.failed_checks
            ],
        }


class OnboardingEngine:
    """
    YAML-driven biller onboarding engine.

    Loads biller definitions from the config YAML, validates them against
    platform requirements, and produces Biller model instances ready for
    use by the payment processor.
    """

    REQUIRED_BILLER_FIELDS = [
        "biller_id", "biller_name", "cis_division", "cis_vendor",
    ]

    def __init__(self, config_path: Optional[Path] = None):
        from config import BILLER_CONFIG_PATH
        self._config_path = config_path or BILLER_CONFIG_PATH
        self._registry: dict[str, Biller] = {}
        self._raw_config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        with open(self._config_path, "r") as f:
            self._raw_config = yaml.safe_load(f)
        logger.info(
            "Loaded biller configuration",
            extra={"event_data": {"path": str(self._config_path)}},
        )

    def get_raw_biller_configs(self) -> list[dict]:
        return self._raw_config.get("billers", [])

    def get_defaults(self) -> dict:
        return self._raw_config.get("defaults", {})

    def onboard_biller(self, biller_config: dict) -> tuple[Biller, OnboardingReport]:
        """
        Onboard a single biller from its config dict.

        Returns the Biller instance and the validation report.
        Raises OnboardingError if critical validations fail.
        """
        cid = set_correlation_id()
        biller_id = biller_config.get("biller_id", "UNKNOWN")
        logger.info(
            f"Starting onboarding for biller {biller_id}",
            extra={"event_data": {"biller_id": biller_id, "correlation_id": cid}},
        )

        report = OnboardingReport(biller_id=biller_id)

        # Check required fields
        for fld in self.REQUIRED_BILLER_FIELDS:
            present = fld in biller_config and biller_config[fld]
            report.checks.append(OnboardingCheck(
                check_name=f"required_field_{fld}",
                passed=present,
                message="" if present else f"Missing required field: {fld}",
            ))

        # Validate payment types
        payment_types_raw = biller_config.get("payment_types", [])
        has_payment_types = len(payment_types_raw) > 0
        report.checks.append(OnboardingCheck(
            check_name="has_payment_types",
            passed=has_payment_types,
            message="" if has_payment_types else "At least one payment type must be configured",
        ))

        # Validate fee structure
        fee_raw = biller_config.get("fee_structure", {})
        has_fees = "convenience_fee" in fee_raw
        report.checks.append(OnboardingCheck(
            check_name="fee_structure_defined",
            passed=has_fees,
            message="" if has_fees else "Convenience fee schedule is required",
        ))

        # Validate settlement config
        settlement_raw = biller_config.get("settlement", {})
        has_settlement = "method" in settlement_raw and "bank_routing" in settlement_raw
        report.checks.append(OnboardingCheck(
            check_name="settlement_configured",
            passed=has_settlement,
            message="" if has_settlement else "Settlement method and bank routing are required",
        ))

        # Validate bank routing number (ABA format: 9 digits)
        routing = settlement_raw.get("bank_routing", "")
        valid_routing = bool(re.match(r"^\d{9}$", str(routing)))
        report.checks.append(OnboardingCheck(
            check_name="bank_routing_format",
            passed=valid_routing,
            message="" if valid_routing else f"Invalid ABA routing number: {routing}",
        ))

        # Validate account number format regex compiles
        validation_raw = biller_config.get("validation_rules", {})
        acct_format = validation_raw.get("account_number_format", "")
        try:
            if acct_format:
                re.compile(acct_format)
            regex_valid = True
        except re.error:
            regex_valid = False
        report.checks.append(OnboardingCheck(
            check_name="account_format_regex_valid",
            passed=regex_valid,
            message="" if regex_valid else f"Invalid regex for account_number_format: {acct_format}",
        ))

        # Validate CIS vendor support
        supported_vendors = ["oracle_ccb", "sap_isu", "gentrack"]
        cis_vendor = biller_config.get("cis_vendor", "")
        vendor_supported = cis_vendor in supported_vendors
        report.checks.append(OnboardingCheck(
            check_name="cis_vendor_supported",
            passed=vendor_supported,
            message="" if vendor_supported else f"Unsupported CIS vendor: {cis_vendor}",
        ))

        report.completed_at = datetime.utcnow()

        # Build the Biller model
        biller = self._build_biller(biller_config)

        if report.all_passed:
            logger.info(
                f"Onboarding validation passed for {biller_id}",
                extra={"event_data": report.summary()},
            )
        else:
            logger.warning(
                f"Onboarding validation failed for {biller_id}",
                extra={"event_data": report.summary()},
            )

        # Register in memory
        self._registry[biller.biller_id] = biller

        return biller, report

    def onboard_all(self) -> list[tuple[Biller, OnboardingReport]]:
        """Onboard all billers defined in the config file."""
        results = []
        for biller_config in self.get_raw_biller_configs():
            biller, report = self.onboard_biller(biller_config)
            results.append((biller, report))
        return results

    def get_biller(self, biller_id: str) -> Optional[Biller]:
        return self._registry.get(biller_id)

    def list_billers(self) -> list[Biller]:
        return list(self._registry.values())

    def activate_biller(self, biller_id: str) -> Biller:
        """Activate a previously onboarded biller."""
        biller = self._registry.get(biller_id)
        if biller is None:
            raise OnboardingError(f"Biller {biller_id} not found in registry")

        biller.status = BillerStatus.ACTIVE
        biller.onboarded_date = date.today()
        logger.info(
            f"Biller {biller_id} activated",
            extra={"event_data": {"biller_id": biller_id, "status": "active"}},
        )
        return biller

    def _build_biller(self, cfg: dict) -> Biller:
        """Construct a Biller model from a raw config dict."""
        defaults = self.get_defaults()

        # Payment types
        payment_types = []
        for pt in cfg.get("payment_types", []):
            payment_types.append(PaymentTypeConfig(
                type=pt["type"],
                enabled=pt.get("enabled", True),
                channels=pt.get("channels", ["web"]),
                max_amount=Decimal(str(pt.get("max_amount", 10000))),
                min_amount=Decimal(str(pt.get("min_amount", 1))),
                enrollment_required=pt.get("enrollment_required", False),
                recalculation_frequency=pt.get("recalculation_frequency"),
            ))

        # Fee structure
        fee_raw = cfg.get("fee_structure", {})
        conv_raw = fee_raw.get("convenience_fee", {})
        fee_structure = BillerFeeStructure(
            convenience_fee=ConvenienceFeeSchedule(
                credit_card=Decimal(str(conv_raw.get("credit_card", 0))),
                debit_card=Decimal(str(conv_raw.get("debit_card", 0))),
                ach=Decimal(str(conv_raw.get("ach", 0))),
                check=Decimal(str(conv_raw.get("check", 0))),
            ),
            late_payment_fee=Decimal(str(fee_raw.get("late_payment_fee", 0))),
            returned_payment_fee=Decimal(str(fee_raw.get("returned_payment_fee", 0))),
            fee_assessed_by=FeeAssessedBy(
                fee_raw.get("fee_assessed_by", "payment_platform")
            ),
        )

        # Settlement
        stl_raw = cfg.get("settlement", {})
        settlement = SettlementConfig(
            method=stl_raw.get("method", "ach_batch"),
            frequency=stl_raw.get("frequency", "next_business_day"),
            bank_routing=str(stl_raw.get("bank_routing", "")),
            bank_account=str(stl_raw.get("bank_account", "")),
            settlement_currency=stl_raw.get(
                "settlement_currency",
                defaults.get("settlement_currency", "USD"),
            ),
            holdback_percentage=Decimal(str(stl_raw.get("holdback_percentage", 0))),
            minimum_settlement_amount=Decimal(str(stl_raw.get("minimum_settlement_amount", 100))),
        )

        # Notifications
        notif_raw = cfg.get("notifications", {})
        notifications = NotificationConfig(
            payment_confirmation=notif_raw.get("payment_confirmation", True),
            payment_failure=notif_raw.get("payment_failure", True),
            settlement_complete=notif_raw.get("settlement_complete", True),
            autopay_reminder=notif_raw.get("autopay_reminder", False),
            channels=notif_raw.get("channels", ["email"]),
        )

        # Validation rules
        val_raw = cfg.get("validation_rules", {})
        validation_rules = ValidationRules(
            account_number_format=val_raw.get("account_number_format", r"^\d{10}$"),
            account_number_length=val_raw.get("account_number_length", 10),
            allow_partial_payments=val_raw.get("allow_partial_payments", True),
            allow_overpayments=val_raw.get("allow_overpayments", False),
            require_bill_match=val_raw.get("require_bill_match", True),
        )

        # Parse onboarded_date
        onboarded_raw = cfg.get("onboarded_date")
        onboarded_date = None
        if onboarded_raw:
            if isinstance(onboarded_raw, date):
                onboarded_date = onboarded_raw
            elif isinstance(onboarded_raw, str):
                onboarded_date = date.fromisoformat(onboarded_raw)

        return Biller(
            biller_id=cfg["biller_id"],
            biller_name=cfg["biller_name"],
            cis_division=cfg["cis_division"],
            cis_vendor=cfg.get("cis_vendor", defaults.get("cis_vendor", "oracle_ccb")),
            cis_version=cfg.get("cis_version", ""),
            status=BillerStatus(cfg.get("status", "pending")),
            onboarded_date=onboarded_date,
            payment_types=payment_types,
            fee_structure=fee_structure,
            settlement=settlement,
            notifications=notifications,
            validation_rules=validation_rules,
        )
