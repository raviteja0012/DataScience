"""Standalone anomaly detection demo.

Demonstrates statistical anomaly detection, transaction pattern analysis,
alert generation, and time-series anomaly detection using synthetic
payment data.

Run: python -m demo.demo_anomaly
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anomaly.detector import AnomalyDetector
from src.anomaly.statistical import StatisticalDetector, DetectionMethod
from src.anomaly.pattern_analyzer import PatternAnalyzer
from src.anomaly.alert_manager import AlertManager
from src.data.synthetic_generator import SyntheticDataGenerator


def main() -> None:
    print("=" * 80)
    print("  PAYMENT INTELLIGENCE AGENT - Anomaly Detection Demo")
    print("=" * 80)

    # Generate synthetic data
    print("\n[1/5] Generating synthetic transaction data...")
    generator = SyntheticDataGenerator(n_transactions=5000, anomaly_rate=0.05)
    datasets = generator.generate_all()
    transactions = datasets["transactions"]
    print(f"  Generated {len(transactions):,} transactions")
    print(f"  Date range: {transactions['TRANSACTION_DATE'].min()} to {transactions['TRANSACTION_DATE'].max()}")
    print(f"  Status distribution:")
    for status, count in transactions["STATUS"].value_counts().items():
        print(f"    {status}: {count:,}")

    # Statistical detection
    print(f"\n{'─' * 80}")
    print("[2/5] Running statistical anomaly detection...")
    stat_detector = StatisticalDetector(zscore_threshold=3.0, iqr_multiplier=1.5)
    scores = stat_detector.detect(
        transactions,
        columns=["AMOUNT", "RISK_SCORE"],
        methods=[DetectionMethod.ZSCORE, DetectionMethod.IQR, DetectionMethod.MODIFIED_ZSCORE],
    )
    anomalies = [s for s in scores if s.is_anomaly]
    print(f"  Total scores computed: {len(scores)}")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Anomaly rate: {len(anomalies) / len(transactions) * 100:.2f}%")

    if anomalies:
        print(f"\n  Top 5 anomalies by score:")
        for score in anomalies[:5]:
            features_str = ", ".join(f"{k}={v:.2f}" for k, v in list(score.features.items())[:3])
            print(f"    Score={score.score:.3f} | {features_str}")

    # Amount outlier detection
    print(f"\n{'─' * 80}")
    print("[3/5] Detecting amount outliers...")
    amount_outliers = stat_detector.detect_amount_outliers(transactions, "AMOUNT")
    print(f"  Amount outliers found: {len(amount_outliers)}")
    if not amount_outliers.empty:
        print(f"  Outlier amount range: ${amount_outliers['AMOUNT'].min():,.2f} - ${amount_outliers['AMOUNT'].max():,.2f}")
        print(f"  Normal amount range: ${transactions['AMOUNT'].quantile(0.05):,.2f} - ${transactions['AMOUNT'].quantile(0.95):,.2f}")

    # Pattern analysis
    print(f"\n{'─' * 80}")
    print("[4/5] Running pattern analysis...")
    pattern_analyzer = PatternAnalyzer()
    violations = pattern_analyzer.analyze(transactions)
    print(f"  Pattern violations found: {len(violations)}")
    for v in violations:
        print(f"    [{v.severity.upper():8s}] {v.pattern_type}: {v.description[:70]}")

    # Alert generation
    print(f"\n{'─' * 80}")
    print("[5/5] Generating alerts...")
    alert_manager = AlertManager()

    for score in anomalies[:20]:
        alert_manager.generate_alert(
            score=score.score,
            category="statistical",
            message=f"Amount anomaly detected (score: {score.score:.2f})",
            details=score.features,
        )

    alert_manager.generate_alerts_from_violations(violations)

    summary = alert_manager.get_alert_summary()
    print(f"  Total active alerts: {summary['total_active']}")
    print(f"  By severity:")
    for sev, count in summary["by_severity"].items():
        print(f"    {sev.upper()}: {count}")
    print(f"  By category:")
    for cat, count in summary["by_category"].items():
        print(f"    {cat}: {count}")

    # End-to-end detection
    print(f"\n{'─' * 80}")
    print("End-to-end detection via AnomalyDetector...")
    detector = AnomalyDetector()
    result = detector.analyze("Check for anomalies", data=transactions, demo_mode=False)
    print(f"\n  Summary:\n  {result['summary']}")

    # Time-series detection
    print(f"\n{'─' * 80}")
    print("Time-series anomaly detection...")
    ts_result = detector.run_time_series_detection(transactions, frequency="D")
    print(f"  {ts_result['summary']}")
    if not ts_result["anomalies"].empty:
        print(f"  Anomalous periods:")
        for _, row in ts_result["anomalies"].head(5).iterrows():
            print(f"    {row['TRANSACTION_DATE'].date()}: {row['transaction_count']} txns, ${row['total_amount']:,.2f}")

    print(f"\n{'=' * 80}")
    print("  Demo complete. All anomaly detection methods executed successfully.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
