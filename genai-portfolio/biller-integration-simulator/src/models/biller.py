"""
Biller entity models.

A "biller" in the payment platform context is a utility company (or a specific
operating division within one) that has been onboarded to accept payments through
the platform. These models capture the biller's configuration — payment types,
fee schedules, settlement preferences, and validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class BillerStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"


class FeeAssessedBy(str, Enum):
    PAYMENT_PLATFORM = "payment_platform"
    BILLER = "biller"


@dataclass
class ConvenienceFeeSchedule:
    """Fee schedule per payment tender type."""
    credit_card: Decimal = Decimal("0.00")
    debit_card: Decimal = Decimal("0.00")
    ach: Decimal = Decimal("0.00")
    check: Decimal = Decimal("0.00")


@dataclass
class BillerFeeStructure:
    """Complete fee configuration for a biller."""
    convenience_fee: ConvenienceFeeSchedule = field(
        default_factory=ConvenienceFeeSchedule
    )
    late_payment_fee: Decimal = Decimal("0.00")
    returned_payment_fee: Decimal = Decimal("0.00")
    fee_assessed_by: FeeAssessedBy = FeeAssessedBy.PAYMENT_PLATFORM


@dataclass
class PaymentTypeConfig:
    """Configuration for a single payment type offered by a biller."""
    type: str  # one_time, autopay, budget_billing, prepay
    enabled: bool = True
    channels: list[str] = field(default_factory=lambda: ["web"])
    max_amount: Decimal = Decimal("10000.00")
    min_amount: Decimal = Decimal("1.00")
    enrollment_required: bool = False
    recalculation_frequency: Optional[str] = None  # for budget billing


@dataclass
class SettlementConfig:
    """How and when funds are settled to the biller."""
    method: str = "ach_batch"
    frequency: str = "next_business_day"
    bank_routing: str = ""
    bank_account: str = ""  # masked in config
    settlement_currency: str = "USD"
    holdback_percentage: Decimal = Decimal("0.0")
    minimum_settlement_amount: Decimal = Decimal("100.00")


@dataclass
class ValidationRules:
    """Account-level validation rules applied when a customer initiates payment."""
    account_number_format: str = r"^\d{10}$"
    account_number_length: int = 10
    allow_partial_payments: bool = True
    allow_overpayments: bool = False
    require_bill_match: bool = True


@dataclass
class NotificationConfig:
    """Notification preferences for biller-related events."""
    payment_confirmation: bool = True
    payment_failure: bool = True
    settlement_complete: bool = True
    autopay_reminder: bool = False
    channels: list[str] = field(default_factory=lambda: ["email"])


@dataclass
class Biller:
    """
    Top-level biller entity.

    Represents a single utility biller onboarded onto the payment platform.
    All runtime behavior — payment acceptance, fee calculation, settlement
    processing, notifications — is governed by this configuration.
    """
    biller_id: str
    biller_name: str
    cis_division: str
    cis_vendor: str = "oracle_ccb"
    cis_version: str = "2.9.0.1"
    status: BillerStatus = BillerStatus.PENDING
    onboarded_date: Optional[date] = None

    payment_types: list[PaymentTypeConfig] = field(default_factory=list)
    fee_structure: BillerFeeStructure = field(default_factory=BillerFeeStructure)
    settlement: SettlementConfig = field(default_factory=SettlementConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    validation_rules: ValidationRules = field(default_factory=ValidationRules)

    def is_active(self) -> bool:
        return self.status == BillerStatus.ACTIVE

    def accepts_payment_type(self, payment_type: str) -> bool:
        return any(
            pt.type == payment_type and pt.enabled
            for pt in self.payment_types
        )

    def accepts_channel(self, payment_type: str, channel: str) -> bool:
        for pt in self.payment_types:
            if pt.type == payment_type and pt.enabled and channel in pt.channels:
                return True
        return False

    def get_convenience_fee(self, tender_type: str) -> Decimal:
        """Look up the convenience fee for a given tender (credit_card, debit_card, etc.)."""
        return getattr(
            self.fee_structure.convenience_fee, tender_type, Decimal("0.00")
        )

    def get_payment_type_config(self, payment_type: str) -> Optional[PaymentTypeConfig]:
        for pt in self.payment_types:
            if pt.type == payment_type:
                return pt
        return None
