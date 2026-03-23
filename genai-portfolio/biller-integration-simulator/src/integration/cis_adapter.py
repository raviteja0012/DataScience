"""
CIS (Customer Information System) adapter.

Provides a unified interface for communicating with the biller's Oracle CC&B
system. In production, this would make SOAP/REST calls to the CC&B web services
layer or read from flat-file extracts. Here we simulate CIS interactions using
synthetic data.

Supported operations:
    - Account lookup (CI_ACCT + CI_PER)
    - Bill inquiry (CI_BILL + CI_BILL_SEG)
    - Payment posting confirmation
    - Service agreement validation (CI_SA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from src.models.biller import Biller
from src.models.oracle_ccb import (
    AcctStatusFlg,
    BillStatusFlg,
    CiAcct,
    CiBill,
    CiPer,
    CiSa,
)
from src.models.payment import Payment
from src.utils.generators import (
    generate_account,
    generate_bill,
    generate_person,
    generate_service_agreement,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CISConnectionError(Exception):
    """Raised when the CIS system is unreachable."""
    pass


class AccountNotFoundError(Exception):
    """Raised when an account lookup returns no results."""
    pass


@dataclass
class AccountInquiryResponse:
    """Response from an account lookup in the CIS."""
    account: CiAcct
    person: CiPer
    service_agreements: list[CiSa] = field(default_factory=list)
    open_bills: list[CiBill] = field(default_factory=list)
    found: bool = True


@dataclass
class PaymentPostingResponse:
    """Response from posting a payment to the CIS."""
    success: bool
    cis_confirmation_number: Optional[str] = None
    posted_amount: Decimal = Decimal("0.00")
    posted_date: Optional[date] = None
    error_message: Optional[str] = None


class CISAdapter:
    """
    Adapter for Oracle CC&B CIS interactions.

    Simulates the real-time and batch interfaces that a payment platform uses
    to communicate with the utility's customer information system. Each method
    mirrors a typical CIS web service operation.
    """

    def __init__(self, biller: Biller):
        self._biller = biller
        # In-memory "database" for simulation
        self._accounts: dict[str, CiAcct] = {}
        self._persons: dict[str, CiPer] = {}
        self._bills: dict[str, CiBill] = {}
        self._service_agreements: dict[str, CiSa] = {}
        self._posted_payments: list[dict] = []

    def seed_data(self, num_accounts: int = 20) -> None:
        """Pre-populate the simulated CIS with synthetic data."""
        logger.info(
            f"Seeding CIS adapter with {num_accounts} accounts",
            extra={"event_data": {
                "biller_id": self._biller.biller_id,
                "num_accounts": num_accounts,
            }},
        )

        for _ in range(num_accounts):
            person = generate_person()
            self._persons[person.per_id] = person

            acct = generate_account(
                cis_division=self._biller.cis_division,
                per_id=person.per_id,
            )
            self._accounts[acct.acct_id] = acct

            # Service agreements
            sa = generate_service_agreement(acct.acct_id)
            self._service_agreements[sa.sa_id] = sa

            # Bills for active accounts
            if acct.is_active():
                bill = generate_bill(
                    acct_id=acct.acct_id,
                    sa_ids=[sa.sa_id],
                )
                self._bills[bill.bill_id] = bill

    def lookup_account(self, account_id: str) -> AccountInquiryResponse:
        """
        Look up an account in the CIS by account ID.

        Returns account details, the linked person, active service agreements,
        and any open bills.
        """
        logger.info(
            f"CIS account lookup: {account_id}",
            extra={"event_data": {
                "account_id": account_id,
                "biller_id": self._biller.biller_id,
            }},
        )

        acct = self._accounts.get(account_id)
        if acct is None:
            logger.warning(f"Account not found in CIS: {account_id}")
            raise AccountNotFoundError(
                f"Account {account_id} not found in CIS division {self._biller.cis_division}"
            )

        # Find linked person
        person = self._persons.get(acct.per_id or "")
        if person is None:
            person = generate_person(per_id=acct.per_id)

        # Find active service agreements
        sas = [
            sa for sa in self._service_agreements.values()
            if sa.acct_id == account_id and sa.is_active()
        ]

        # Find open bills
        open_bills = [
            bill for bill in self._bills.values()
            if bill.acct_id == account_id and bill.is_payable
        ]

        return AccountInquiryResponse(
            account=acct,
            person=person,
            service_agreements=sas,
            open_bills=open_bills,
        )

    def get_bill(self, bill_id: str) -> Optional[CiBill]:
        """Retrieve a specific bill by ID."""
        return self._bills.get(bill_id)

    def get_account_balance(self, account_id: str) -> Decimal:
        """Return the total outstanding balance for an account."""
        total = Decimal("0.00")
        for bill in self._bills.values():
            if bill.acct_id == account_id and bill.is_payable:
                total += bill.tot_amt_due
        return total

    def post_payment(self, payment: Payment) -> PaymentPostingResponse:
        """
        Post a payment confirmation back to the CIS.

        In a real integration, this would call the CC&B payment upload service
        or write to the payment staging table for batch posting.
        """
        import uuid

        logger.info(
            f"Posting payment to CIS: txn={payment.transaction_id}, "
            f"acct={payment.customer_account_id}, amount={payment.amount}",
            extra={"event_data": {
                "transaction_id": payment.transaction_id,
                "account_id": payment.customer_account_id,
                "amount": str(payment.amount),
            }},
        )

        # Verify account exists
        acct = self._accounts.get(payment.customer_account_id)
        if acct is None:
            return PaymentPostingResponse(
                success=False,
                error_message=f"Account {payment.customer_account_id} not found",
            )

        # Verify account is eligible
        if not acct.is_eligible_for_payment():
            return PaymentPostingResponse(
                success=False,
                error_message=(
                    f"Account {payment.customer_account_id} is not eligible for payment "
                    f"(status: {acct.acct_status_flg.value})"
                ),
            )

        confirmation = f"CIS-{uuid.uuid4().hex[:10].upper()}"

        self._posted_payments.append({
            "confirmation": confirmation,
            "transaction_id": payment.transaction_id,
            "account_id": payment.customer_account_id,
            "amount": payment.amount,
            "posted_date": date.today(),
        })

        # Reduce the bill balance if bill_id is provided
        if payment.bill_id and payment.bill_id in self._bills:
            bill = self._bills[payment.bill_id]
            bill.tot_amt_due = max(
                Decimal("0.00"),
                bill.tot_amt_due - payment.amount,
            )

        return PaymentPostingResponse(
            success=True,
            cis_confirmation_number=confirmation,
            posted_amount=payment.amount,
            posted_date=date.today(),
        )

    def validate_account_for_payment(self, account_id: str) -> tuple[bool, str]:
        """
        Quick check whether an account is valid and eligible for payment.

        Returns (is_valid, reason).
        """
        acct = self._accounts.get(account_id)
        if acct is None:
            return False, "Account not found"

        if not acct.is_eligible_for_payment():
            return False, f"Account status is {acct.acct_status_flg.value}"

        return True, "Account is eligible"

    def get_accounts(self) -> list[CiAcct]:
        """Return all accounts in the simulated CIS."""
        return list(self._accounts.values())

    def get_active_account_ids(self) -> list[str]:
        """Return IDs of all active accounts."""
        return [a.acct_id for a in self._accounts.values() if a.is_active()]
