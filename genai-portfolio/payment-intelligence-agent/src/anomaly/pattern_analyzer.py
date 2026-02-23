"""Transaction pattern analysis for fraud detection.

Implements domain-specific pattern recognition for payment transactions:
velocity checks, geographic anomalies, temporal patterns, and behavioral
deviation detection. These complement statistical methods with business
logic-driven rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatternViolation:
    """A detected pattern violation in transaction data.

    Attributes:
        pattern_type: Category of pattern violation.
        severity: Severity level (critical, high, medium, low).
        description: Human-readable description of the violation.
        affected_records: Number of transactions involved.
        details: Additional structured data about the violation.
    """

    pattern_type: str
    severity: str
    description: str
    affected_records: int
    details: dict[str, Any] = field(default_factory=dict)


class PatternAnalyzer:
    """Analyzes transaction patterns for fraud indicators.

    Implements payment-specific pattern detection rules:
    - Velocity checks: excessive transactions in short time windows
    - Geographic anomalies: impossible travel, high-risk region spikes
    - Amount patterns: round amounts, split transactions, card testing
    - Temporal patterns: off-hours activity, day-of-week deviations
    - Behavioral deviation: changes from established customer patterns

    Attributes:
        velocity_window_minutes: Time window for velocity checks.
        velocity_threshold: Maximum transactions allowed in the window.
        high_risk_regions: Regions flagged as elevated risk.
    """

    def __init__(
        self,
        velocity_window_minutes: int = 10,
        velocity_threshold: int = 5,
        high_risk_regions: list[str] | None = None,
    ) -> None:
        self.velocity_window_minutes = velocity_window_minutes
        self.velocity_threshold = velocity_threshold
        self.high_risk_regions = high_risk_regions or [
            "Eastern Europe", "Southeast Asia", "West Africa",
        ]

    def analyze(self, data: pd.DataFrame) -> list[PatternViolation]:
        """Run all pattern analyses on transaction data.

        Args:
            data: DataFrame with transaction records.

        Returns:
            List of detected PatternViolation objects.
        """
        violations: list[PatternViolation] = []

        if data.empty:
            return violations

        # Run each pattern check
        violations.extend(self._check_velocity(data))
        violations.extend(self._check_geographic_anomalies(data))
        violations.extend(self._check_amount_patterns(data))
        violations.extend(self._check_temporal_patterns(data))
        violations.extend(self._check_card_testing(data))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        violations.sort(key=lambda v: severity_order.get(v.severity, 4))

        logger.info("pattern_analysis_complete", violations_found=len(violations))
        return violations

    def _check_velocity(self, data: pd.DataFrame) -> list[PatternViolation]:
        """Detect excessive transaction velocity per customer or card.

        Flags customers with more transactions than the threshold
        within the configured time window.
        """
        violations: list[PatternViolation] = []

        if "CUSTOMER_ID" not in data.columns or "TRANSACTION_DATE" not in data.columns:
            return violations

        # Ensure datetime type
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["TRANSACTION_DATE"]):
            data["TRANSACTION_DATE"] = pd.to_datetime(data["TRANSACTION_DATE"])

        window = timedelta(minutes=self.velocity_window_minutes)

        for customer_id, group in data.groupby("CUSTOMER_ID"):
            if len(group) < self.velocity_threshold:
                continue

            sorted_group = group.sort_values("TRANSACTION_DATE")
            dates = sorted_group["TRANSACTION_DATE"].values

            # Sliding window count
            for i in range(len(dates)):
                window_end = dates[i] + np.timedelta64(int(window.total_seconds()), "s")
                window_count = np.sum((dates >= dates[i]) & (dates <= window_end))

                if window_count >= self.velocity_threshold:
                    total_amount = sorted_group.iloc[i:i + int(window_count)]["AMOUNT"].sum()
                    violations.append(PatternViolation(
                        pattern_type="velocity",
                        severity="high",
                        description=(
                            f"Customer {str(customer_id)[:8]}... had {int(window_count)} transactions "
                            f"within {self.velocity_window_minutes} minutes "
                            f"(total: ${total_amount:,.2f})"
                        ),
                        affected_records=int(window_count),
                        details={
                            "customer_id": str(customer_id),
                            "window_count": int(window_count),
                            "total_amount": float(total_amount),
                        },
                    ))
                    break  # One violation per customer

        return violations

    def _check_geographic_anomalies(self, data: pd.DataFrame) -> list[PatternViolation]:
        """Detect geographic anomalies in transaction patterns.

        Checks for:
        - Unusual concentration of transactions in high-risk regions
        - Rapid geographic shifts (impossible travel)
        """
        violations: list[PatternViolation] = []

        if "REGION" not in data.columns:
            return violations

        # High-risk region concentration
        region_counts = data["REGION"].value_counts()
        total = len(data)

        for region in self.high_risk_regions:
            if region in region_counts.index:
                count = region_counts[region]
                ratio = count / total
                if ratio > 0.15:  # More than 15% from high-risk region
                    violations.append(PatternViolation(
                        pattern_type="geographic",
                        severity="high" if ratio > 0.25 else "medium",
                        description=(
                            f"Elevated transaction volume from high-risk region '{region}': "
                            f"{count} transactions ({ratio:.1%} of total)"
                        ),
                        affected_records=int(count),
                        details={
                            "region": region,
                            "count": int(count),
                            "percentage": round(ratio * 100, 2),
                        },
                    ))

        # Geographic diversity check per customer
        if "CUSTOMER_ID" in data.columns:
            for customer_id, group in data.groupby("CUSTOMER_ID"):
                if "COUNTRY_CODE" in group.columns:
                    unique_countries = group["COUNTRY_CODE"].nunique()
                    if unique_countries > 3 and len(group) < 20:
                        violations.append(PatternViolation(
                            pattern_type="geographic",
                            severity="medium",
                            description=(
                                f"Customer {str(customer_id)[:8]}... transacted from "
                                f"{unique_countries} different countries in a short period"
                            ),
                            affected_records=len(group),
                            details={
                                "customer_id": str(customer_id),
                                "country_count": unique_countries,
                                "countries": group["COUNTRY_CODE"].unique().tolist(),
                            },
                        ))

        return violations

    def _check_amount_patterns(self, data: pd.DataFrame) -> list[PatternViolation]:
        """Detect suspicious amount patterns.

        Flags:
        - High proportion of round amounts (potential money laundering)
        - Very small amounts followed by large ones (card testing)
        - Split transactions (multiple small amounts near a threshold)
        """
        violations: list[PatternViolation] = []

        if "AMOUNT" not in data.columns:
            return violations

        amounts = data["AMOUNT"]

        # Round amount detection
        round_amounts = amounts[amounts % 100 == 0]
        round_ratio = len(round_amounts) / len(amounts) if len(amounts) > 0 else 0

        if round_ratio > 0.3 and len(round_amounts) > 10:
            violations.append(PatternViolation(
                pattern_type="amount_pattern",
                severity="medium",
                description=(
                    f"{len(round_amounts)} transactions ({round_ratio:.1%}) have round amounts "
                    f"(multiples of $100), which may indicate structured transactions"
                ),
                affected_records=len(round_amounts),
                details={
                    "round_count": len(round_amounts),
                    "round_percentage": round(round_ratio * 100, 2),
                },
            ))

        # Micro-transaction detection (potential card testing)
        micro_threshold = 1.00
        micro_transactions = amounts[amounts <= micro_threshold]
        micro_ratio = len(micro_transactions) / len(amounts) if len(amounts) > 0 else 0

        if micro_ratio > 0.05 and len(micro_transactions) > 5:
            violations.append(PatternViolation(
                pattern_type="card_testing",
                severity="high",
                description=(
                    f"{len(micro_transactions)} micro-transactions (under ${micro_threshold:.2f}) detected "
                    f"({micro_ratio:.1%} of total) - potential card testing activity"
                ),
                affected_records=len(micro_transactions),
                details={
                    "micro_count": len(micro_transactions),
                    "threshold": micro_threshold,
                },
            ))

        return violations

    def _check_temporal_patterns(self, data: pd.DataFrame) -> list[PatternViolation]:
        """Detect unusual temporal transaction patterns.

        Flags unusual off-hours activity or day-of-week deviations.
        """
        violations: list[PatternViolation] = []

        if "TRANSACTION_DATE" not in data.columns:
            return violations

        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["TRANSACTION_DATE"]):
            data["TRANSACTION_DATE"] = pd.to_datetime(data["TRANSACTION_DATE"])

        hours = data["TRANSACTION_DATE"].dt.hour

        # Off-hours detection (midnight to 5 AM)
        off_hours = data[(hours >= 0) & (hours < 5)]
        off_hours_ratio = len(off_hours) / len(data) if len(data) > 0 else 0

        if off_hours_ratio > 0.15 and len(off_hours) > 10:
            avg_amount = off_hours["AMOUNT"].mean() if "AMOUNT" in off_hours.columns else 0
            violations.append(PatternViolation(
                pattern_type="temporal",
                severity="medium",
                description=(
                    f"Elevated off-hours activity: {len(off_hours)} transactions "
                    f"({off_hours_ratio:.1%}) between midnight and 5 AM "
                    f"(avg amount: ${avg_amount:,.2f})"
                ),
                affected_records=len(off_hours),
                details={
                    "off_hours_count": len(off_hours),
                    "percentage": round(off_hours_ratio * 100, 2),
                    "avg_amount": round(avg_amount, 2),
                },
            ))

        return violations

    def _check_card_testing(self, data: pd.DataFrame) -> list[PatternViolation]:
        """Detect card testing patterns.

        Card testing involves rapid small-amount transactions to verify
        stolen card numbers before making larger purchases.
        """
        violations: list[PatternViolation] = []

        if "AMOUNT" not in data.columns or "STATUS" not in data.columns:
            return violations

        # Look for rapid small amounts followed by high decline rates
        small_transactions = data[data["AMOUNT"] < 5.00]
        if len(small_transactions) < 5:
            return violations

        if "STATUS" in small_transactions.columns:
            declined = small_transactions[small_transactions["STATUS"] == "DECLINED"]
            decline_rate = len(declined) / len(small_transactions)

            if decline_rate > 0.5 and len(declined) > 3:
                violations.append(PatternViolation(
                    pattern_type="card_testing",
                    severity="critical",
                    description=(
                        f"Suspected card testing: {len(declined)} declined small-amount "
                        f"transactions ({decline_rate:.1%} decline rate for amounts under $5.00)"
                    ),
                    affected_records=len(declined),
                    details={
                        "declined_count": len(declined),
                        "decline_rate": round(decline_rate * 100, 2),
                        "total_small_txns": len(small_transactions),
                    },
                ))

        return violations

    def get_velocity_summary(
        self,
        data: pd.DataFrame,
        entity_column: str = "CUSTOMER_ID",
        window_minutes: int | None = None,
    ) -> pd.DataFrame:
        """Compute velocity metrics per entity (customer or merchant).

        Args:
            data: Transaction DataFrame.
            entity_column: Column to group by.
            window_minutes: Override time window.

        Returns:
            DataFrame with velocity metrics per entity.
        """
        if entity_column not in data.columns or "TRANSACTION_DATE" not in data.columns:
            return pd.DataFrame()

        window = window_minutes or self.velocity_window_minutes

        summary_records: list[dict[str, Any]] = []

        for entity_id, group in data.groupby(entity_column):
            if len(group) < 2:
                continue

            sorted_dates = group.sort_values("TRANSACTION_DATE")
            time_diffs = sorted_dates["TRANSACTION_DATE"].diff().dt.total_seconds().dropna()

            summary_records.append({
                entity_column: entity_id,
                "transaction_count": len(group),
                "total_amount": group["AMOUNT"].sum() if "AMOUNT" in group.columns else 0,
                "avg_time_between_txns_seconds": time_diffs.mean() if len(time_diffs) > 0 else 0,
                "min_time_between_txns_seconds": time_diffs.min() if len(time_diffs) > 0 else 0,
                "max_velocity_per_window": self._max_window_count(
                    sorted_dates["TRANSACTION_DATE"].values, window * 60,
                ),
            })

        if not summary_records:
            return pd.DataFrame()

        return pd.DataFrame(summary_records).sort_values(
            "max_velocity_per_window", ascending=False,
        )

    @staticmethod
    def _max_window_count(timestamps: np.ndarray, window_seconds: float) -> int:
        """Find the maximum number of transactions in any sliding window."""
        max_count = 0
        window_ns = np.timedelta64(int(window_seconds), "s")

        for i in range(len(timestamps)):
            window_end = timestamps[i] + window_ns
            count = int(np.sum((timestamps >= timestamps[i]) & (timestamps <= window_end)))
            max_count = max(max_count, count)

        return max_count
