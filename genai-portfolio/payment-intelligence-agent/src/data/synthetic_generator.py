"""Synthetic payment data generator for testing and demo purposes.

Generates realistic synthetic transaction, merchant, customer, settlement,
and chargeback data with configurable volume, time ranges, and anomaly
injection. Designed to produce data that exercises all analytics and
anomaly detection capabilities.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Merchant template data
_MERCHANT_TEMPLATES = [
    ("TechStore Global", "5732", "Electronics Stores", "North America", "US", "LOW"),
    ("FoodMart Express", "5411", "Grocery Stores", "North America", "US", "LOW"),
    ("CloudServices Inc", "7372", "Computer Services", "North America", "US", "LOW"),
    ("RetailHub Premium", "5311", "Department Stores", "North America", "US", "LOW"),
    ("GameZone Digital", "5816", "Digital Goods", "North America", "US", "MEDIUM"),
    ("TravelNow Agency", "4722", "Travel Agencies", "Europe", "GB", "MEDIUM"),
    ("HealthPlus Pharmacy", "5912", "Drug Stores", "North America", "US", "LOW"),
    ("AutoParts Direct", "5533", "Auto Parts", "North America", "US", "LOW"),
    ("BookWorld Online", "5942", "Book Stores", "Europe", "DE", "LOW"),
    ("FashionForward Co", "5651", "Clothing Stores", "Europe", "FR", "LOW"),
    ("GadgetPro Shop", "5734", "Computer Software", "Asia Pacific", "JP", "MEDIUM"),
    ("GreenGrocers Ltd", "5499", "Misc Food Stores", "Europe", "GB", "LOW"),
    ("PetCare Central", "5995", "Pet Shops", "North America", "US", "LOW"),
    ("HomeDecor Plus", "5200", "Home Supply", "North America", "CA", "LOW"),
    ("SportsFit Outlet", "5941", "Sporting Goods", "North America", "US", "LOW"),
    ("ElectroMart", "5065", "Electronic Parts", "Asia Pacific", "KR", "MEDIUM"),
    ("CoffeeBean Roasters", "5814", "Fast Food", "North America", "US", "LOW"),
    ("MusicStream Pro", "5815", "Digital Goods", "North America", "US", "MEDIUM"),
    ("ArtSupplies Hub", "5970", "Art Supply", "Europe", "NL", "LOW"),
    ("ToyLand Express", "5945", "Hobby Stores", "North America", "US", "LOW"),
    ("QuickMart Fuel", "5541", "Service Stations", "North America", "US", "LOW"),
    ("LuxuryWatch Co", "5944", "Jewelry Stores", "Europe", "CH", "HIGH"),
    ("CryptoExchange Ltd", "6051", "Money Orders", "Europe", "MT", "HIGH"),
    ("OnlineBets Pro", "7995", "Gambling", "Latin America", "CR", "HIGH"),
    ("HighRisk Trading", "6211", "Security Brokers", "Asia Pacific", "HK", "HIGH"),
]


class SyntheticDataGenerator:
    """Generates realistic synthetic payment data for testing.

    Creates interconnected datasets across all five core tables
    (MERCHANTS, CUSTOMERS, TRANSACTIONS, SETTLEMENTS, CHARGEBACKS)
    with configurable parameters and optional anomaly injection.

    Usage:
        generator = SyntheticDataGenerator(n_transactions=10000)
        data = generator.generate_all()
        transactions_df = data["transactions"]
    """

    def __init__(
        self,
        n_transactions: int = 10_000,
        n_customers: int = 500,
        n_merchants: int = 25,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        anomaly_rate: float = 0.03,
        seed: int = 42,
    ) -> None:
        self.n_transactions = n_transactions
        self.n_customers = n_customers
        self.n_merchants = min(n_merchants, len(_MERCHANT_TEMPLATES))
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.anomaly_rate = anomaly_rate
        self.rng = np.random.default_rng(seed)

        self._merchants_df: pd.DataFrame | None = None
        self._customers_df: pd.DataFrame | None = None
        self._transactions_df: pd.DataFrame | None = None

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all synthetic datasets.

        Returns:
            Dictionary with keys: merchants, customers, transactions,
            settlements, chargebacks.
        """
        merchants = self.generate_merchants()
        customers = self.generate_customers()
        transactions = self.generate_transactions(merchants, customers)
        settlements = self.generate_settlements(transactions, merchants)
        chargebacks = self.generate_chargebacks(transactions, merchants)

        logger.info(
            "synthetic_data_generated",
            merchants=len(merchants),
            customers=len(customers),
            transactions=len(transactions),
            settlements=len(settlements),
            chargebacks=len(chargebacks),
        )

        return {
            "merchants": merchants,
            "customers": customers,
            "transactions": transactions,
            "settlements": settlements,
            "chargebacks": chargebacks,
        }

    def generate_merchants(self) -> pd.DataFrame:
        """Generate synthetic merchant data."""
        templates = _MERCHANT_TEMPLATES[:self.n_merchants]
        records: list[dict[str, Any]] = []

        for name, mcc, mcc_desc, region, country, risk in templates:
            records.append({
                "MERCHANT_ID": str(uuid.uuid4()),
                "MERCHANT_NAME": name,
                "MCC_CODE": mcc,
                "MCC_DESCRIPTION": mcc_desc,
                "REGION": region,
                "COUNTRY_CODE": country,
                "RISK_TIER": risk,
                "ONBOARDING_DATE": (
                    self.start_date - timedelta(days=int(self.rng.integers(30, 730)))
                ).date(),
                "STATUS": "ACTIVE",
                "MONTHLY_VOLUME_LIMIT": float(
                    self.rng.choice([100_000, 250_000, 500_000, 1_000_000, 5_000_000])
                ),
                "CREATED_AT": datetime.utcnow(),
            })

        self._merchants_df = pd.DataFrame(records)
        return self._merchants_df

    def generate_customers(self) -> pd.DataFrame:
        """Generate synthetic customer data."""
        import hashlib

        segments = ["PREMIUM", "STANDARD", "NEW"]
        segment_weights = [0.15, 0.60, 0.25]
        countries = ["US", "GB", "CA", "DE", "FR", "JP", "AU", "BR"]
        country_weights = [0.40, 0.12, 0.10, 0.08, 0.07, 0.08, 0.08, 0.07]
        regions = {
            "US": "North America", "CA": "North America",
            "GB": "Europe", "DE": "Europe", "FR": "Europe",
            "JP": "Asia Pacific", "AU": "Asia Pacific",
            "BR": "Latin America",
        }

        records: list[dict[str, Any]] = []
        for _ in range(self.n_customers):
            cust_id = str(uuid.uuid4())
            country = self.rng.choice(countries, p=country_weights)
            segment = self.rng.choice(segments, p=segment_weights)

            records.append({
                "CUSTOMER_ID": cust_id,
                "CUSTOMER_HASH": hashlib.sha256(cust_id.encode()).hexdigest(),
                "CUSTOMER_SEGMENT": segment,
                "COUNTRY_CODE": country,
                "REGION": regions.get(country, "Other"),
                "ACCOUNT_CREATED_DATE": (
                    self.start_date - timedelta(days=int(self.rng.integers(1, 1095)))
                ).date(),
                "RISK_SCORE": round(float(np.clip(self.rng.normal(30, 15), 0, 100)), 2),
                "LIFETIME_TRANSACTION_COUNT": int(self.rng.integers(1, 500)),
                "LIFETIME_TRANSACTION_VALUE": round(float(self.rng.lognormal(8, 1.5)), 2),
                "CREATED_AT": datetime.utcnow(),
            })

        self._customers_df = pd.DataFrame(records)
        return self._customers_df

    def generate_transactions(
        self,
        merchants: pd.DataFrame,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate synthetic transaction data with injected anomalies."""
        merchant_ids = merchants["MERCHANT_ID"].values
        customer_ids = customers["CUSTOMER_ID"].values

        # Weight merchants by risk tier (high-risk get fewer transactions)
        merchant_weights = np.array([
            0.8 if r == "LOW" else 0.5 if r == "MEDIUM" else 0.2
            for r in merchants["RISK_TIER"]
        ])
        merchant_weights = merchant_weights / merchant_weights.sum()

        n_normal = int(self.n_transactions * (1 - self.anomaly_rate))
        n_anomalous = self.n_transactions - n_normal

        # Generate normal transactions
        normal_records = self._generate_normal_transactions(
            n_normal, merchant_ids, customer_ids, merchant_weights, merchants,
        )

        # Generate anomalous transactions
        anomaly_records = self._generate_anomalous_transactions(
            n_anomalous, merchant_ids, customer_ids, merchants,
        )

        all_records = normal_records + anomaly_records
        self._transactions_df = pd.DataFrame(all_records)
        self._transactions_df = self._transactions_df.sort_values(
            "TRANSACTION_DATE"
        ).reset_index(drop=True)

        return self._transactions_df

    def _generate_normal_transactions(
        self,
        n: int,
        merchant_ids: np.ndarray,
        customer_ids: np.ndarray,
        merchant_weights: np.ndarray,
        merchants: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Generate normal transaction records."""
        records: list[dict[str, Any]] = []
        date_range_days = (self.end_date - self.start_date).days

        methods = ["CREDIT", "DEBIT", "ACH", "WIRE"]
        method_weights = [0.50, 0.30, 0.15, 0.05]
        brands = {"CREDIT": ["VISA", "MASTERCARD", "AMEX", "DISCOVER"],
                   "DEBIT": ["VISA", "MASTERCARD"]}
        statuses = ["APPROVED", "DECLINED", "PENDING", "REFUNDED"]
        status_weights = [0.90, 0.06, 0.02, 0.02]
        channels = ["ONLINE", "IN_STORE", "MOBILE", "PHONE"]
        channel_weights = [0.45, 0.30, 0.20, 0.05]

        decline_reasons = [
            "Insufficient Funds", "Card Expired", "Invalid CVV",
            "Suspected Fraud", "Velocity Limit", "Do Not Honor",
        ]

        for _ in range(n):
            merchant_idx = self.rng.choice(len(merchant_ids), p=merchant_weights)
            merchant_id = merchant_ids[merchant_idx]
            merchant_row = merchants.iloc[merchant_idx]

            method = self.rng.choice(methods, p=method_weights)
            status = self.rng.choice(statuses, p=status_weights)
            txn_date = self.start_date + timedelta(
                seconds=int(self.rng.integers(0, date_range_days * 86400))
            )

            # Amount follows log-normal distribution, varies by payment method
            amount_mean = {"CREDIT": 4.5, "DEBIT": 3.8, "ACH": 6.0, "WIRE": 7.5}
            amount = round(float(self.rng.lognormal(amount_mean.get(method, 4.5), 0.8)), 2)
            amount = min(amount, 25000)  # Cap at reasonable maximum

            card_brand = None
            card_last_four = None
            if method in ("CREDIT", "DEBIT"):
                card_brand = self.rng.choice(brands.get(method, ["VISA"]))
                card_last_four = f"{self.rng.integers(1000, 9999)}"

            records.append({
                "TRANSACTION_ID": str(uuid.uuid4()),
                "MERCHANT_ID": merchant_id,
                "CUSTOMER_ID": self.rng.choice(customer_ids),
                "TRANSACTION_DATE": txn_date,
                "AMOUNT": amount,
                "CURRENCY": "USD",
                "PAYMENT_METHOD": method,
                "CARD_BRAND": card_brand,
                "CARD_LAST_FOUR": card_last_four,
                "STATUS": status,
                "DECLINE_REASON": self.rng.choice(decline_reasons) if status == "DECLINED" else None,
                "AUTH_CODE": f"{self.rng.integers(100000, 999999)}" if status == "APPROVED" else None,
                "RISK_SCORE": round(float(np.clip(self.rng.normal(25, 12), 0, 100)), 2),
                "CHANNEL": self.rng.choice(channels, p=channel_weights),
                "REGION": merchant_row["REGION"],
                "COUNTRY_CODE": merchant_row["COUNTRY_CODE"],
                "FEE_AMOUNT": round(amount * self.rng.uniform(0.015, 0.035), 2),
                "CREATED_AT": datetime.utcnow(),
            })

        return records

    def _generate_anomalous_transactions(
        self,
        n: int,
        merchant_ids: np.ndarray,
        customer_ids: np.ndarray,
        merchants: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Generate anomalous transaction records with various fraud patterns."""
        records: list[dict[str, Any]] = []
        date_range_days = (self.end_date - self.start_date).days

        # High-risk merchants get more anomalies
        high_risk_merchants = merchants[merchants["RISK_TIER"] == "HIGH"]["MERCHANT_ID"].values
        if len(high_risk_merchants) == 0:
            high_risk_merchants = merchant_ids[:3]

        # Small pool of "compromised" customers
        compromised_customers = customer_ids[:max(5, int(len(customer_ids) * 0.01))]

        anomaly_types = ["large_amount", "card_testing", "velocity_burst", "geographic", "off_hours"]

        for i in range(n):
            anomaly_type = self.rng.choice(anomaly_types)
            txn_date = self.start_date + timedelta(
                seconds=int(self.rng.integers(0, date_range_days * 86400))
            )

            if anomaly_type == "large_amount":
                amount = float(self.rng.uniform(5000, 50000))
                risk_score = float(self.rng.uniform(70, 98))
                status = self.rng.choice(["APPROVED", "DECLINED"], p=[0.3, 0.7])
            elif anomaly_type == "card_testing":
                amount = float(self.rng.uniform(0.01, 1.50))
                risk_score = float(self.rng.uniform(50, 85))
                status = self.rng.choice(["APPROVED", "DECLINED"], p=[0.2, 0.8])
            elif anomaly_type == "velocity_burst":
                amount = float(self.rng.lognormal(4.0, 0.5))
                risk_score = float(self.rng.uniform(60, 90))
                status = "APPROVED"
                # Cluster timestamps
                txn_date = self.start_date + timedelta(days=int(self.rng.integers(0, date_range_days)))
                txn_date = txn_date.replace(hour=int(self.rng.integers(0, 4)))
            elif anomaly_type == "geographic":
                amount = float(self.rng.lognormal(5.5, 1.0))
                risk_score = float(self.rng.uniform(65, 95))
                status = self.rng.choice(["APPROVED", "DECLINED"], p=[0.4, 0.6])
            else:  # off_hours
                amount = float(self.rng.lognormal(5.0, 1.0))
                risk_score = float(self.rng.uniform(55, 85))
                status = "APPROVED"
                txn_date = txn_date.replace(hour=int(self.rng.integers(1, 5)))

            records.append({
                "TRANSACTION_ID": str(uuid.uuid4()),
                "MERCHANT_ID": self.rng.choice(high_risk_merchants),
                "CUSTOMER_ID": self.rng.choice(compromised_customers),
                "TRANSACTION_DATE": txn_date,
                "AMOUNT": round(amount, 2),
                "CURRENCY": "USD",
                "PAYMENT_METHOD": self.rng.choice(["CREDIT", "DEBIT"]),
                "CARD_BRAND": self.rng.choice(["VISA", "MASTERCARD"]),
                "CARD_LAST_FOUR": f"{self.rng.integers(1000, 9999)}",
                "STATUS": status,
                "DECLINE_REASON": "Suspected Fraud" if status == "DECLINED" else None,
                "AUTH_CODE": f"{self.rng.integers(100000, 999999)}" if status == "APPROVED" else None,
                "RISK_SCORE": round(risk_score, 2),
                "CHANNEL": "ONLINE",
                "REGION": self.rng.choice(["Eastern Europe", "West Africa", "Southeast Asia"]),
                "COUNTRY_CODE": self.rng.choice(["RO", "NG", "VN", "UA"]),
                "FEE_AMOUNT": round(amount * 0.035, 2),
                "CREATED_AT": datetime.utcnow(),
            })

        return records

    def generate_settlements(
        self,
        transactions: pd.DataFrame,
        merchants: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate settlement data from approved transactions."""
        approved = transactions[transactions["STATUS"] == "APPROVED"].copy()
        if approved.empty:
            return pd.DataFrame()

        approved["TRANSACTION_DATE"] = pd.to_datetime(approved["TRANSACTION_DATE"])
        approved["SETTLEMENT_WEEK"] = approved["TRANSACTION_DATE"].dt.to_period("W")

        records: list[dict[str, Any]] = []
        for (merchant_id, week), group in approved.groupby(["MERCHANT_ID", "SETTLEMENT_WEEK"]):
            gross = group["AMOUNT"].sum()
            fees = group["FEE_AMOUNT"].sum()
            settlement_days = int(self.rng.integers(1, 5))

            records.append({
                "SETTLEMENT_ID": str(uuid.uuid4()),
                "MERCHANT_ID": merchant_id,
                "SETTLEMENT_DATE": (week.end_time + timedelta(days=settlement_days)).date(),
                "TRANSACTION_COUNT": len(group),
                "GROSS_AMOUNT": round(gross, 2),
                "FEE_AMOUNT": round(fees, 2),
                "NET_AMOUNT": round(gross - fees, 2),
                "CURRENCY": "USD",
                "PAYMENT_METHOD": group["PAYMENT_METHOD"].mode().iloc[0] if not group["PAYMENT_METHOD"].mode().empty else "CREDIT",
                "STATUS": "COMPLETED",
                "SETTLEMENT_DAYS": settlement_days,
                "CREATED_AT": datetime.utcnow(),
            })

        return pd.DataFrame(records).sort_values("SETTLEMENT_DATE").reset_index(drop=True)

    def generate_chargebacks(
        self,
        transactions: pd.DataFrame,
        merchants: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate chargeback data from approved transactions."""
        approved = transactions[transactions["STATUS"] == "APPROVED"]
        if approved.empty:
            return pd.DataFrame()

        # Higher chargeback rate for high-risk merchants and high-risk scores
        chargeback_prob = np.where(
            approved["RISK_SCORE"] > 60, 0.08,
            np.where(approved["RISK_SCORE"] > 40, 0.02, 0.005),
        )
        chargeback_mask = self.rng.random(len(approved)) < chargeback_prob
        chargeback_txns = approved[chargeback_mask]

        reason_codes = [
            ("4837", "No Cardholder Authorization"),
            ("4853", "Cardholder Dispute - Not as Described"),
            ("4863", "Cardholder Does Not Recognize"),
            ("4855", "Non-Receipt of Merchandise"),
            ("4834", "Transaction Amount Differs"),
            ("4841", "Cancelled Recurring Transaction"),
        ]

        records: list[dict[str, Any]] = []
        for _, txn in chargeback_txns.iterrows():
            reason_code, reason_desc = reason_codes[int(self.rng.integers(0, len(reason_codes)))]
            txn_date = pd.to_datetime(txn["TRANSACTION_DATE"])
            chargeback_days = int(self.rng.integers(5, 90))
            resolution_days = int(self.rng.integers(15, 120))
            status = self.rng.choice(["OPEN", "WON", "LOST", "EXPIRED"], p=[0.15, 0.35, 0.40, 0.10])

            records.append({
                "CHARGEBACK_ID": str(uuid.uuid4()),
                "TRANSACTION_ID": txn["TRANSACTION_ID"],
                "MERCHANT_ID": txn["MERCHANT_ID"],
                "CHARGEBACK_DATE": (txn_date + timedelta(days=chargeback_days)).date(),
                "AMOUNT": txn["AMOUNT"],
                "CURRENCY": "USD",
                "REASON_CODE": reason_code,
                "REASON_DESCRIPTION": reason_desc,
                "STATUS": status,
                "RESOLUTION_DATE": (
                    (txn_date + timedelta(days=chargeback_days + resolution_days)).date()
                    if status != "OPEN" else None
                ),
                "RESOLUTION_DAYS": resolution_days if status != "OPEN" else None,
                "REPRESENTMENT_SUBMITTED": status in ("WON", "LOST"),
                "CREATED_AT": datetime.utcnow(),
            })

        return pd.DataFrame(records).sort_values("CHARGEBACK_DATE").reset_index(drop=True)
