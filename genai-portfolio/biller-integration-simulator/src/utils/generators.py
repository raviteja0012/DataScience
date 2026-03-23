"""
Synthetic data generators for testing and simulation.

Generates realistic-looking Oracle CC&B records, payment transactions, and
settlement files. The data patterns are modeled after real utility billing
cycles — bill amounts follow seasonal distributions, payment timing clusters
around due dates, and a configurable percentage of records include deliberate
anomalies for testing exception handling.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from src.models.oracle_ccb import (
    AcctStatusFlg,
    BillStatusFlg,
    CiAcct,
    CiBill,
    CiBillSeg,
    CiPer,
    CiSa,
    PerOrBusFlg,
    SaStatusFlg,
)
from src.models.payment import (
    Payment,
    PaymentChannel,
    PaymentMethod,
    PaymentStatus,
    PaymentType,
)
from src.models.settlement import SettlementRecord


# Realistic name pools
_FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
]
_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

_SA_TYPES = ["E-RES", "E-COM", "G-RES", "G-COM", "W-RES", "W-COM"]
_BILL_CYCLE_CODES = ["01", "02", "03", "04", "05", "10", "15", "20"]


def _random_decimal(low: float, high: float, places: int = 2) -> Decimal:
    value = random.uniform(low, high)
    return Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)


def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))


def generate_account_id(length: int = 10) -> str:
    return "".join(random.choices(string.digits, k=length))


def generate_person(per_id: Optional[str] = None) -> CiPer:
    """Generate a realistic person record."""
    per_id = per_id or str(random.randint(1000000000, 9999999999))
    is_business = random.random() < 0.15  # ~15% business accounts

    first = random.choice(_FIRST_NAMES)
    last = random.choice(_LAST_NAMES)

    return CiPer(
        per_id=per_id,
        per_or_bus_flg=PerOrBusFlg.BUSINESS if is_business else PerOrBusFlg.PERSON,
        emailid=f"{first.lower()}.{last.lower()}@example.com",
        lang_cd=random.choice(["ENG", "ENG", "ENG", "SPA"]),
        life_support_flg="Y" if random.random() < 0.02 else "N",
        name1=f"{last},{first}" if not is_business else f"{last} Enterprises LLC",
    )


def generate_account(
    acct_id: Optional[str] = None,
    cis_division: str = "GP01",
    per_id: Optional[str] = None,
) -> CiAcct:
    """Generate a realistic account record."""
    acct_id = acct_id or generate_account_id()
    setup_date = _random_date(date(2015, 1, 1), date(2024, 6, 1))

    # Most accounts are active, but include some closed/suspended for realism
    status_weights = [0.85, 0.05, 0.08, 0.02]
    statuses = [
        AcctStatusFlg.ACTIVE,
        AcctStatusFlg.SUSPENDED,
        AcctStatusFlg.CLOSED,
        AcctStatusFlg.WRITTEN_OFF,
    ]
    status = random.choices(statuses, weights=status_weights, k=1)[0]

    return CiAcct(
        acct_id=acct_id,
        cis_division=cis_division,
        setup_dt=setup_date,
        acct_status_flg=status,
        currency_cd="USD",
        bill_cycle_cd=random.choice(_BILL_CYCLE_CODES),
        per_id=per_id,
    )


def generate_service_agreement(
    acct_id: str,
    sa_id: Optional[str] = None,
    sa_type_cd: Optional[str] = None,
) -> CiSa:
    """Generate a service agreement linked to an account."""
    sa_id = sa_id or str(random.randint(1000000000, 9999999999))
    sa_type_cd = sa_type_cd or random.choice(_SA_TYPES)

    start = _random_date(date(2015, 1, 1), date(2024, 1, 1))
    # ~10% of SAs are stopped
    is_stopped = random.random() < 0.10

    return CiSa(
        sa_id=sa_id,
        acct_id=acct_id,
        sa_type_cd=sa_type_cd,
        sa_status_flg=SaStatusFlg.STOPPED if is_stopped else SaStatusFlg.ACTIVE,
        start_dt=start,
        end_dt=start + timedelta(days=random.randint(365, 3650)) if is_stopped else None,
        char_prem_id=str(random.randint(100000, 999999)),
    )


def generate_bill(
    acct_id: str,
    bill_date: Optional[date] = None,
    sa_ids: Optional[list[str]] = None,
    seasonal_factor: float = 1.0,
) -> CiBill:
    """
    Generate a bill with segments.

    The seasonal_factor allows callers to simulate higher summer bills (e.g., 1.4)
    or lower spring bills (e.g., 0.8).
    """
    bill_id = str(random.randint(100000000000, 999999999999))
    bill_date = bill_date or date.today() - timedelta(days=random.randint(1, 60))
    due_date = bill_date + timedelta(days=21)  # Standard 21-day payment window

    # Base residential bill amount, adjusted by season
    base_amount = _random_decimal(45.0, 350.0) * Decimal(str(seasonal_factor))
    prev_unpaid = _random_decimal(0, 50) if random.random() < 0.20 else Decimal("0.00")

    bill = CiBill(
        bill_id=bill_id,
        acct_id=acct_id,
        bill_dt=bill_date,
        due_dt=due_date,
        bill_amt=base_amount,
        prev_unpaid_amt=prev_unpaid,
        bill_status_flg=BillStatusFlg.COMPLETE,
        bill_cyc_cd=random.choice(_BILL_CYCLE_CODES),
    )

    # Generate segments — one per SA
    sa_ids = sa_ids or [str(random.randint(1000000000, 9999999999))]
    segment_count = len(sa_ids)
    remaining = base_amount

    for i, sa_id in enumerate(sa_ids):
        if i == segment_count - 1:
            seg_amount = remaining
        else:
            seg_amount = _random_decimal(
                float(remaining) * 0.3,
                float(remaining) * 0.7,
            )
            remaining -= seg_amount

        seg_start = bill_date - timedelta(days=30)
        seg_end = bill_date - timedelta(days=1)

        segment = CiBillSeg(
            billseg_id=str(random.randint(100000000000, 999999999999)),
            bill_id=bill_id,
            sa_id=sa_id,
            start_dt=seg_start,
            end_dt=seg_end,
            calc_amt=seg_amount,
        )
        bill.segments.append(segment)

    return bill


def generate_payment(
    biller_id: str,
    account_id: str,
    amount: Optional[Decimal] = None,
    bill_id: Optional[str] = None,
    status: PaymentStatus = PaymentStatus.INITIATED,
) -> Payment:
    """Generate a payment transaction."""
    amount = amount or _random_decimal(25.0, 400.0)

    method = random.choices(
        [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD,
         PaymentMethod.ACH, PaymentMethod.CHECK],
        weights=[0.30, 0.25, 0.35, 0.10],
        k=1,
    )[0]

    channel = random.choices(
        [PaymentChannel.WEB, PaymentChannel.MOBILE,
         PaymentChannel.IVR, PaymentChannel.AGENT],
        weights=[0.45, 0.30, 0.15, 0.10],
        k=1,
    )[0]

    # Convenience fee depends on tender type
    fee_map = {
        PaymentMethod.CREDIT_CARD: _random_decimal(1.99, 2.95),
        PaymentMethod.DEBIT_CARD: _random_decimal(0.99, 1.75),
        PaymentMethod.ACH: Decimal("0.00"),
        PaymentMethod.CHECK: Decimal("0.00"),
    }
    fee = fee_map[method]

    return Payment(
        biller_id=biller_id,
        customer_account_id=account_id,
        bill_id=bill_id,
        amount=amount,
        convenience_fee=fee,
        payment_method=method,
        payment_type=PaymentType.ONE_TIME,
        channel=channel,
        status=status,
        initiated_at=datetime.utcnow() - timedelta(
            hours=random.randint(0, 72),
            minutes=random.randint(0, 59),
        ),
    )


def generate_settlement_records(
    payments: list[Payment],
    anomaly_rate: float = 0.05,
) -> tuple[list[SettlementRecord], list[SettlementRecord], list[SettlementRecord]]:
    """
    Generate three-way settlement records from a list of payments.

    Returns (biller_records, platform_records, bank_records).

    A configurable anomaly_rate introduces deliberate mismatches:
    - amount discrepancies (rounding, fee differences)
    - missing records from one source
    - date offsets
    """
    biller_records: list[SettlementRecord] = []
    platform_records: list[SettlementRecord] = []
    bank_records: list[SettlementRecord] = []

    for payment in payments:
        if payment.status not in (PaymentStatus.SETTLED, PaymentStatus.CAPTURED):
            continue

        base_date = (payment.settled_at or payment.captured_at or payment.initiated_at).date()
        is_anomaly = random.random() < anomaly_rate

        # Biller record
        biller_rec = SettlementRecord(
            transaction_id=payment.transaction_id,
            source="biller",
            customer_account_id=payment.customer_account_id,
            amount=payment.amount,
            fee_amount=Decimal("0.00"),
            payment_date=base_date,
            settlement_date=base_date + timedelta(days=1),
            biller_id=payment.biller_id,
        )

        # Platform record
        platform_rec = SettlementRecord(
            transaction_id=payment.transaction_id,
            source="platform",
            customer_account_id=payment.customer_account_id,
            amount=payment.total_amount,
            fee_amount=payment.convenience_fee,
            net_amount=payment.amount,
            payment_date=base_date,
            settlement_date=base_date + timedelta(days=1),
            biller_id=payment.biller_id,
        )

        # Bank record
        bank_rec = SettlementRecord(
            transaction_id=payment.transaction_id,
            source="bank",
            customer_account_id=payment.customer_account_id,
            amount=payment.amount,
            fee_amount=Decimal("0.00"),
            payment_date=base_date,
            settlement_date=base_date + timedelta(days=1),
            biller_id=payment.biller_id,
        )

        if is_anomaly:
            anomaly_type = random.choice([
                "amount_mismatch",
                "missing_biller",
                "missing_bank",
                "date_offset",
            ])

            if anomaly_type == "amount_mismatch":
                # Introduce a small discrepancy in the bank record
                offset = _random_decimal(-5.0, 5.0)
                bank_rec.amount += offset
                bank_rec.net_amount = bank_rec.amount - bank_rec.fee_amount
            elif anomaly_type == "missing_biller":
                biller_records.append(biller_rec)  # skip — don't add
                platform_records.append(platform_rec)
                bank_records.append(bank_rec)
                continue
            elif anomaly_type == "missing_bank":
                biller_records.append(biller_rec)
                platform_records.append(platform_rec)
                # skip bank record
                continue
            elif anomaly_type == "date_offset":
                bank_rec.settlement_date = base_date + timedelta(
                    days=random.randint(2, 5)
                )

        biller_records.append(biller_rec)
        platform_records.append(platform_rec)
        bank_records.append(bank_rec)

    return biller_records, platform_records, bank_records


@dataclass
class DataSetGenerator:
    """
    Generates a complete synthetic dataset for a biller.

    Produces correlated persons, accounts, service agreements, bills,
    payments, and settlement records — suitable for end-to-end testing
    of the full processing pipeline.
    """
    biller_id: str
    cis_division: str
    num_accounts: int = 50

    def generate(self) -> dict:
        """Generate and return the full dataset as a dict of lists."""
        persons: list[CiPer] = []
        accounts: list[CiAcct] = []
        service_agreements: list[CiSa] = []
        bills: list[CiBill] = []
        payments: list[Payment] = []

        for _ in range(self.num_accounts):
            person = generate_person()
            persons.append(person)

            acct = generate_account(
                cis_division=self.cis_division,
                per_id=person.per_id,
            )
            accounts.append(acct)

            # 1-3 SAs per account
            sa_count = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15], k=1)[0]
            acct_sas: list[CiSa] = []
            for _ in range(sa_count):
                sa = generate_service_agreement(acct.acct_id)
                service_agreements.append(sa)
                acct_sas.append(sa)

            # Generate 1-3 bills per active account
            if acct.is_active():
                bill_count = random.randint(1, 3)
                for b in range(bill_count):
                    bill_date = date.today() - timedelta(days=30 * (b + 1))
                    bill = generate_bill(
                        acct_id=acct.acct_id,
                        bill_date=bill_date,
                        sa_ids=[sa.sa_id for sa in acct_sas if sa.is_active()],
                    )
                    bills.append(bill)

                    # ~70% of bills get a payment
                    if random.random() < 0.70:
                        pmt = generate_payment(
                            biller_id=self.biller_id,
                            account_id=acct.acct_id,
                            amount=bill.tot_amt_due,
                            bill_id=bill.bill_id,
                            status=PaymentStatus.SETTLED,
                        )
                        payments.append(pmt)

        biller_recs, platform_recs, bank_recs = generate_settlement_records(
            payments, anomaly_rate=0.08,
        )

        return {
            "persons": persons,
            "accounts": accounts,
            "service_agreements": service_agreements,
            "bills": bills,
            "payments": payments,
            "settlement": {
                "biller_records": biller_recs,
                "platform_records": platform_recs,
                "bank_records": bank_recs,
            },
        }
