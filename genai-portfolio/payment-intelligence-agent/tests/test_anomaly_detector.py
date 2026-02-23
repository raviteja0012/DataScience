"""Tests for the anomaly detection engine.

Validates statistical detection methods, pattern analysis, alert generation,
and the end-to-end detection pipeline using synthetic payment data.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.anomaly.statistical import StatisticalDetector, DetectionMethod
from src.anomaly.pattern_analyzer import PatternAnalyzer, PatternViolation
from src.anomaly.alert_manager import AlertManager, Alert, Severity
from src.anomaly.detector import AnomalyDetector


@pytest.fixture
def normal_data() -> pd.DataFrame:
    """Generate a DataFrame of normal transactions."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "AMOUNT": rng.lognormal(4.5, 0.5, n).round(2),
        "RISK_SCORE": rng.normal(25, 8, n).clip(0, 100).round(2),
        "FEE_AMOUNT": rng.uniform(1, 10, n).round(2),
    })


@pytest.fixture
def data_with_outliers() -> pd.DataFrame:
    """Generate a DataFrame with injected outliers."""
    rng = np.random.default_rng(42)
    n_normal = 190
    n_outliers = 10

    normal_amounts = rng.lognormal(4.5, 0.5, n_normal)
    outlier_amounts = rng.uniform(10000, 50000, n_outliers)

    amounts = np.concatenate([normal_amounts, outlier_amounts])
    risk_scores = np.concatenate([
        rng.normal(25, 8, n_normal).clip(0, 100),
        rng.uniform(80, 100, n_outliers),
    ])

    return pd.DataFrame({
        "AMOUNT": amounts.round(2),
        "RISK_SCORE": risk_scores.round(2),
    })


@pytest.fixture
def transaction_data() -> pd.DataFrame:
    """Generate transaction data with columns needed for pattern analysis."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    return pd.DataFrame({
        "TRANSACTION_ID": [f"TXN-{i:04d}" for i in range(n)],
        "TRANSACTION_DATE": dates,
        "AMOUNT": rng.lognormal(4.5, 0.8, n).round(2),
        "CUSTOMER_ID": [f"CUST-{rng.integers(1, 10):03d}" for _ in range(n)],
        "REGION": rng.choice(
            ["North America", "Europe", "Eastern Europe", "Asia Pacific"],
            n, p=[0.5, 0.25, 0.15, 0.1],
        ),
        "COUNTRY_CODE": rng.choice(["US", "GB", "RO", "JP"], n),
        "STATUS": rng.choice(["APPROVED", "DECLINED"], n, p=[0.9, 0.1]),
        "RISK_SCORE": rng.normal(30, 15, n).clip(0, 100).round(2),
    })


class TestStatisticalDetector:
    """Test statistical anomaly detection methods."""

    def test_zscore_detects_outliers(self, data_with_outliers: pd.DataFrame) -> None:
        detector = StatisticalDetector(zscore_threshold=3.0)
        scores = detector.detect(
            data_with_outliers,
            columns=["AMOUNT"],
            methods=[DetectionMethod.ZSCORE],
        )
        anomalies = [s for s in scores if s.is_anomaly]
        assert len(anomalies) > 0
        # The injected outliers should be detected
        assert len(anomalies) <= 20  # Reasonable upper bound

    def test_iqr_detects_outliers(self, data_with_outliers: pd.DataFrame) -> None:
        detector = StatisticalDetector(iqr_multiplier=1.5)
        scores = detector.detect(
            data_with_outliers,
            columns=["AMOUNT"],
            methods=[DetectionMethod.IQR],
        )
        anomalies = [s for s in scores if s.is_anomaly]
        assert len(anomalies) > 0

    def test_modified_zscore(self, data_with_outliers: pd.DataFrame) -> None:
        detector = StatisticalDetector()
        scores = detector.detect(
            data_with_outliers,
            columns=["AMOUNT"],
            methods=[DetectionMethod.MODIFIED_ZSCORE],
        )
        anomalies = [s for s in scores if s.is_anomaly]
        assert len(anomalies) > 0

    def test_normal_data_few_anomalies(self, normal_data: pd.DataFrame) -> None:
        detector = StatisticalDetector(zscore_threshold=3.0)
        scores = detector.detect(normal_data, columns=["AMOUNT"])
        anomalies = [s for s in scores if s.is_anomaly]
        # Normal data should have very few anomalies
        assert len(anomalies) < len(normal_data) * 0.1

    def test_empty_dataframe(self) -> None:
        detector = StatisticalDetector()
        scores = detector.detect(pd.DataFrame())
        assert len(scores) == 0

    def test_single_column_detection(self, data_with_outliers: pd.DataFrame) -> None:
        detector = StatisticalDetector()
        scores = detector.detect(data_with_outliers, columns=["RISK_SCORE"])
        assert len(scores) > 0

    def test_scores_are_normalized(self, data_with_outliers: pd.DataFrame) -> None:
        detector = StatisticalDetector()
        scores = detector.detect(data_with_outliers, columns=["AMOUNT"])
        for score in scores:
            assert 0.0 <= score.score <= 1.0

    def test_amount_outlier_detection(self, data_with_outliers: pd.DataFrame) -> None:
        detector = StatisticalDetector()
        outliers = detector.detect_amount_outliers(data_with_outliers, "AMOUNT")
        assert isinstance(outliers, pd.DataFrame)


class TestPatternAnalyzer:
    """Test payment pattern analysis."""

    def test_geographic_anomaly_detection(self, transaction_data: pd.DataFrame) -> None:
        analyzer = PatternAnalyzer()
        violations = analyzer._check_geographic_anomalies(transaction_data)
        # Eastern Europe is flagged as high-risk, should detect elevated volume
        assert isinstance(violations, list)

    def test_temporal_patterns(self, transaction_data: pd.DataFrame) -> None:
        analyzer = PatternAnalyzer()
        violations = analyzer._check_temporal_patterns(transaction_data)
        assert isinstance(violations, list)

    def test_amount_patterns(self, transaction_data: pd.DataFrame) -> None:
        analyzer = PatternAnalyzer()
        violations = analyzer._check_amount_patterns(transaction_data)
        assert isinstance(violations, list)

    def test_full_analysis(self, transaction_data: pd.DataFrame) -> None:
        analyzer = PatternAnalyzer()
        violations = analyzer.analyze(transaction_data)
        assert isinstance(violations, list)
        for v in violations:
            assert isinstance(v, PatternViolation)
            assert v.severity in ("critical", "high", "medium", "low")
            assert v.pattern_type
            assert v.description

    def test_empty_data(self) -> None:
        analyzer = PatternAnalyzer()
        violations = analyzer.analyze(pd.DataFrame())
        assert violations == []

    def test_velocity_summary(self, transaction_data: pd.DataFrame) -> None:
        analyzer = PatternAnalyzer()
        summary = analyzer.get_velocity_summary(transaction_data)
        assert isinstance(summary, pd.DataFrame)


class TestAlertManager:
    """Test alert generation and management."""

    def test_generate_critical_alert(self) -> None:
        manager = AlertManager()
        alert = manager.generate_alert(
            score=0.95,
            category="statistical",
            message="Critical anomaly detected",
        )
        assert alert is not None
        assert alert.severity == Severity.CRITICAL

    def test_generate_high_alert(self) -> None:
        manager = AlertManager()
        alert = manager.generate_alert(
            score=0.75,
            category="velocity",
            message="High velocity detected",
        )
        assert alert is not None
        assert alert.severity == Severity.HIGH

    def test_below_threshold_no_alert(self) -> None:
        manager = AlertManager()
        alert = manager.generate_alert(
            score=0.3,
            category="statistical",
            message="Low score",
        )
        assert alert is None

    def test_deduplication(self) -> None:
        manager = AlertManager()
        alert1 = manager.generate_alert(0.8, "test", "Same message")
        alert2 = manager.generate_alert(0.8, "test", "Same message")
        assert alert1 is not None
        assert alert2 is None  # Deduplicated

    def test_acknowledge_alert(self) -> None:
        manager = AlertManager()
        alert = manager.generate_alert(0.8, "test", "Test alert")
        assert alert is not None
        assert not alert.acknowledged
        result = manager.acknowledge_alert(alert.alert_id)
        assert result is True
        assert alert.acknowledged

    def test_get_alert_summary(self) -> None:
        manager = AlertManager()
        manager.generate_alert(0.95, "cat1", "Alert 1")
        manager.generate_alert(0.75, "cat2", "Alert 2")
        summary = manager.get_alert_summary()
        assert summary["total_active"] == 2
        assert "critical" in summary["by_severity"] or "high" in summary["by_severity"]

    def test_clear_alerts(self) -> None:
        manager = AlertManager()
        manager.generate_alert(0.8, "test", "Alert 1")
        manager.clear_alerts()
        assert manager.active_alert_count == 0


class TestAnomalyDetector:
    """Test the end-to-end anomaly detection engine."""

    def test_demo_mode_analysis(self) -> None:
        detector = AnomalyDetector()
        result = detector.analyze("Are there anomalies in recent transactions?", demo_mode=True)

        assert "anomalies" in result
        assert "summary" in result
        assert "alerts" in result
        assert "statistics" in result
        assert isinstance(result["anomalies"], pd.DataFrame)
        assert result["statistics"]["total_transactions"] > 0

    def test_analysis_finds_injected_anomalies(self) -> None:
        detector = AnomalyDetector()
        result = detector.analyze("Check for anomalies", demo_mode=True)

        # The demo data has injected anomalies, so some should be detected
        assert result["statistics"]["anomalies_detected"] > 0

    def test_time_series_detection(self) -> None:
        detector = AnomalyDetector()
        data = detector._generate_demo_data("time series analysis")
        ts_result = detector.run_time_series_detection(data)

        assert "time_series" in ts_result
        assert "anomalies" in ts_result
        assert isinstance(ts_result["time_series"], pd.DataFrame)

    def test_summary_generation(self) -> None:
        detector = AnomalyDetector()
        result = detector.analyze("Show anomalies", demo_mode=True)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "transactions" in result["summary"].lower()
