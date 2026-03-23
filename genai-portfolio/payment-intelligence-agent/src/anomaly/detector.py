"""Anomaly detection engine that orchestrates statistical and pattern analysis.

Provides a unified interface for running anomaly detection across transaction
data, combining statistical outlier detection with domain-specific pattern
analysis and generating structured alerts.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .statistical import StatisticalDetector, DetectionMethod
from .pattern_analyzer import PatternAnalyzer
from .alert_manager import AlertManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Unified anomaly detection engine for payment transactions.

    Orchestrates statistical detection, pattern analysis, and alert
    generation. Provides both detailed analysis results and concise
    summaries suitable for the conversational interface.

    Usage:
        detector = AnomalyDetector()
        results = detector.analyze("Are there anomalies in recent transactions?", demo_mode=True)
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        contamination: float = 0.05,
    ) -> None:
        self._statistical = StatisticalDetector(
            zscore_threshold=zscore_threshold,
            iqr_multiplier=iqr_multiplier,
            contamination=contamination,
        )
        self._pattern = PatternAnalyzer()
        self._alert_manager = AlertManager()

    def analyze(
        self,
        query: str,
        data: pd.DataFrame | None = None,
        demo_mode: bool = True,
    ) -> dict[str, Any]:
        """Run anomaly detection based on the user query.

        In demo mode, generates synthetic transaction data and runs
        the full detection pipeline against it. In production mode,
        operates on the provided DataFrame.

        Args:
            query: User's natural language query about anomalies.
            data: Transaction DataFrame (optional, uses synthetic in demo).
            demo_mode: Whether to use synthetic data.

        Returns:
            Dictionary containing:
                - anomalies: DataFrame of flagged anomalous transactions.
                - summary: Natural language summary.
                - alerts: List of alert dictionaries.
                - statistics: Detection statistics.
        """
        query_lower = query.lower()

        # Generate or use provided data
        if data is None and demo_mode:
            data = self._generate_demo_data(query_lower)

        if data is None or data.empty:
            return {
                "anomalies": pd.DataFrame(),
                "summary": "No transaction data available for analysis.",
                "alerts": [],
                "statistics": {},
            }

        # Determine analysis focus from query
        focus = self._determine_focus(query_lower)

        # Run statistical detection
        numeric_cols = self._get_analysis_columns(data, focus)
        statistical_scores = self._statistical.detect(
            data,
            columns=numeric_cols,
            methods=[DetectionMethod.ZSCORE, DetectionMethod.IQR, DetectionMethod.MODIFIED_ZSCORE],
        )

        # Run pattern analysis
        violations = self._pattern.analyze(data)

        # Generate alerts
        self._alert_manager.clear_alerts()

        # Alerts from statistical scores
        for score in statistical_scores:
            if score.is_anomaly:
                self._alert_manager.generate_alert(
                    score=score.score,
                    category="statistical",
                    message=f"Statistical anomaly detected (score: {score.score:.2f}) - "
                            f"Features: {', '.join(f'{k}={v:.2f}' for k, v in list(score.features.items())[:3])}",
                    details={"score": score.score, "features": score.features},
                )

        # Alerts from pattern violations
        self._alert_manager.generate_alerts_from_violations(violations)

        # Build anomaly DataFrame
        anomaly_indices = [s.index for s in statistical_scores if s.is_anomaly]
        if anomaly_indices:
            anomalies_df = data.iloc[anomaly_indices].copy()
            anomalies_df["anomaly_score"] = [
                s.score for s in statistical_scores if s.is_anomaly
            ]
        else:
            anomalies_df = pd.DataFrame()

        # Generate summary
        alerts = self._alert_manager.get_alerts()
        alert_dicts = [a.to_dict() for a in alerts]
        summary = self._generate_summary(data, anomalies_df, violations, statistical_scores)

        return {
            "anomalies": anomalies_df,
            "summary": summary,
            "alerts": alert_dicts,
            "statistics": {
                "total_transactions": len(data),
                "anomalies_detected": len(anomalies_df),
                "pattern_violations": len(violations),
                "alert_count": len(alerts),
                "anomaly_rate": len(anomalies_df) / len(data) if len(data) > 0 else 0,
            },
        }

    def _determine_focus(self, query: str) -> str:
        """Determine the analysis focus from the user query."""
        if any(word in query for word in ["velocity", "rapid", "fast", "quick"]):
            return "velocity"
        if any(word in query for word in ["geographic", "region", "location", "country"]):
            return "geographic"
        if any(word in query for word in ["amount", "value", "large", "small"]):
            return "amount"
        if any(word in query for word in ["pattern", "behavior", "trend"]):
            return "pattern"
        if any(word in query for word in ["merchant", "chargeback"]):
            return "merchant"
        return "general"

    @staticmethod
    def _get_analysis_columns(data: pd.DataFrame, focus: str) -> list[str]:
        """Select numeric columns for analysis based on focus area."""
        available_numeric = data.select_dtypes(include=[np.number]).columns.tolist()

        if focus == "amount":
            priority = ["AMOUNT", "FEE_AMOUNT", "RISK_SCORE"]
        elif focus == "velocity":
            priority = ["AMOUNT", "RISK_SCORE"]
        elif focus == "merchant":
            priority = ["AMOUNT", "RISK_SCORE", "FEE_AMOUNT"]
        else:
            priority = ["AMOUNT", "RISK_SCORE", "FEE_AMOUNT"]

        columns = [c for c in priority if c in available_numeric]
        return columns if columns else available_numeric[:3]

    def _generate_summary(
        self,
        data: pd.DataFrame,
        anomalies: pd.DataFrame,
        violations: list[Any],
        scores: list[Any],
    ) -> str:
        """Generate a natural language summary of detection results."""
        parts: list[str] = []

        total = len(data)
        anomaly_count = len(anomalies)
        anomaly_rate = (anomaly_count / total * 100) if total > 0 else 0

        parts.append(
            f"Analyzed **{total:,}** transactions. "
            f"Found **{anomaly_count}** statistical anomalies ({anomaly_rate:.1f}% anomaly rate)."
        )

        if not anomalies.empty and "AMOUNT" in anomalies.columns:
            avg_anomaly_amount = anomalies["AMOUNT"].mean()
            avg_normal_amount = data["AMOUNT"].mean()
            if avg_normal_amount > 0:
                ratio = avg_anomaly_amount / avg_normal_amount
                parts.append(
                    f"Anomalous transactions average **${avg_anomaly_amount:,.2f}** "
                    f"({ratio:.1f}x the normal average of ${avg_normal_amount:,.2f})."
                )

        if violations:
            critical = [v for v in violations if v.severity == "critical"]
            high = [v for v in violations if v.severity == "high"]

            if critical:
                parts.append(
                    f"\n**{len(critical)} CRITICAL pattern violation(s)** requiring immediate attention."
                )
            if high:
                parts.append(
                    f"**{len(high)} high-severity** pattern violation(s) detected."
                )

        alert_summary = self._alert_manager.get_alert_summary()
        if alert_summary["total_active"] > 0:
            parts.append(
                f"\n**{alert_summary['total_active']} active alert(s)** generated across "
                f"{len(alert_summary['by_category'])} categories."
            )

        return "\n".join(parts)

    def _generate_demo_data(self, query: str) -> pd.DataFrame:
        """Generate synthetic transaction data for demo analysis.

        Creates a realistic dataset with injected anomalies for
        demonstrating detection capabilities.
        """
        rng = np.random.default_rng(42)
        n_normal = 950
        n_anomalous = 50

        # Normal transactions
        dates = pd.date_range(end="2024-12-31", periods=n_normal, freq="h")
        normal_data = {
            "TRANSACTION_ID": [f"TXN-{i:06d}" for i in range(n_normal)],
            "TRANSACTION_DATE": dates,
            "AMOUNT": rng.lognormal(mean=4.5, sigma=0.8, size=n_normal).round(2),
            "RISK_SCORE": rng.normal(25, 10, n_normal).clip(0, 100).round(2),
            "CUSTOMER_ID": [f"CUST-{rng.integers(1, 200):04d}" for _ in range(n_normal)],
            "MERCHANT_ID": [f"MERCH-{rng.integers(1, 50):03d}" for _ in range(n_normal)],
            "PAYMENT_METHOD": rng.choice(["CREDIT", "DEBIT", "ACH", "WIRE"], n_normal, p=[0.5, 0.3, 0.15, 0.05]),
            "STATUS": rng.choice(["APPROVED", "DECLINED", "PENDING"], n_normal, p=[0.92, 0.06, 0.02]),
            "REGION": rng.choice(
                ["North America", "Europe", "Asia Pacific", "Latin America", "Eastern Europe"],
                n_normal,
                p=[0.45, 0.25, 0.15, 0.10, 0.05],
            ),
            "COUNTRY_CODE": rng.choice(["US", "GB", "DE", "JP", "BR", "RO"], n_normal, p=[0.45, 0.15, 0.10, 0.10, 0.10, 0.10]),
            "CHANNEL": rng.choice(["ONLINE", "IN_STORE", "MOBILE"], n_normal, p=[0.5, 0.3, 0.2]),
            "FEE_AMOUNT": rng.uniform(0.5, 15, n_normal).round(2),
        }

        # Anomalous transactions
        anomaly_dates = pd.date_range(end="2024-12-31", periods=n_anomalous, freq="3h")
        anomaly_data = {
            "TRANSACTION_ID": [f"TXN-A{i:05d}" for i in range(n_anomalous)],
            "TRANSACTION_DATE": anomaly_dates,
            "AMOUNT": np.concatenate([
                rng.uniform(8000, 50000, 20),  # Unusually large amounts
                rng.uniform(0.01, 0.99, 15),   # Card testing amounts
                rng.uniform(100, 500, 15) * 100,  # Round amounts
            ]).round(2),
            "RISK_SCORE": rng.normal(75, 15, n_anomalous).clip(0, 100).round(2),
            "CUSTOMER_ID": [f"CUST-{rng.integers(1, 10):04d}" for _ in range(n_anomalous)],
            "MERCHANT_ID": [f"MERCH-{rng.integers(1, 5):03d}" for _ in range(n_anomalous)],
            "PAYMENT_METHOD": rng.choice(["CREDIT", "DEBIT"], n_anomalous),
            "STATUS": rng.choice(["APPROVED", "DECLINED"], n_anomalous, p=[0.4, 0.6]),
            "REGION": rng.choice(
                ["Eastern Europe", "Southeast Asia", "West Africa", "North America"],
                n_anomalous,
                p=[0.35, 0.25, 0.25, 0.15],
            ),
            "COUNTRY_CODE": rng.choice(["RO", "NG", "VN", "US"], n_anomalous, p=[0.3, 0.3, 0.2, 0.2]),
            "CHANNEL": rng.choice(["ONLINE", "MOBILE"], n_anomalous),
            "FEE_AMOUNT": rng.uniform(10, 100, n_anomalous).round(2),
        }

        normal_df = pd.DataFrame(normal_data)
        anomaly_df = pd.DataFrame(anomaly_data)

        combined = pd.concat([normal_df, anomaly_df], ignore_index=True)
        combined = combined.sort_values("TRANSACTION_DATE").reset_index(drop=True)

        logger.info("demo_data_generated", total_rows=len(combined), anomalies_injected=n_anomalous)
        return combined

    def run_time_series_detection(
        self,
        data: pd.DataFrame,
        date_column: str = "TRANSACTION_DATE",
        value_column: str = "AMOUNT",
        frequency: str = "D",
    ) -> dict[str, Any]:
        """Run time-series anomaly detection on aggregated data.

        Aggregates transactions by the specified frequency and detects
        anomalous periods using statistical methods.

        Args:
            data: Transaction DataFrame.
            date_column: Column containing timestamps.
            value_column: Column to aggregate and analyze.
            frequency: Aggregation frequency (D=daily, H=hourly, etc.).

        Returns:
            Dictionary with time-series anomaly results.
        """
        if date_column not in data.columns or value_column not in data.columns:
            return {"anomalies": pd.DataFrame(), "summary": "Required columns not found."}

        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])

        # Aggregate by frequency
        agg = data.set_index(date_column).resample(frequency)[value_column].agg(
            ["sum", "count", "mean"]
        ).reset_index()
        agg.columns = [date_column, "total_amount", "transaction_count", "avg_amount"]

        # Run detection on aggregated metrics
        scores = self._statistical.detect(
            agg,
            columns=["total_amount", "transaction_count"],
        )

        anomaly_periods = [s for s in scores if s.is_anomaly]

        if anomaly_periods:
            anomaly_indices = [s.index for s in anomaly_periods]
            anomaly_df = agg.iloc[anomaly_indices].copy()
            anomaly_df["anomaly_score"] = [s.score for s in anomaly_periods]
        else:
            anomaly_df = pd.DataFrame()

        return {
            "time_series": agg,
            "anomalies": anomaly_df,
            "summary": (
                f"Analyzed {len(agg)} time periods. "
                f"Found {len(anomaly_periods)} anomalous period(s)."
            ),
        }
