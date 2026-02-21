"""
Model Performance Monitoring System
====================================
Monitors a deployed ML model for:
  - Data drift (Kolmogorov-Smirnov test, Population Stability Index)
  - Concept drift (accuracy / F1 degradation over time windows)
  - Feature distribution shifts per column

Includes:
  - Synthetic production data generator that introduces controlled drift
  - Configurable alert system with severity levels
  - Automatic retraining trigger logic
  - Visualisation of drift metrics over time

Usage:
    python model_monitor.py
"""

import json
import time
import datetime
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "monitoring_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """Represents a single monitoring alert."""
    timestamp: str
    alert_type: str          # data_drift | concept_drift | performance
    severity: str            # info | warning | critical
    feature: Optional[str]
    metric_name: str
    metric_value: float
    threshold: float
    message: str


@dataclass
class MonitoringReport:
    """Aggregated report for one monitoring window."""
    window_id: int
    timestamp: str
    n_samples: int
    data_drift: Dict[str, Dict]
    concept_drift: Dict[str, float]
    performance_metrics: Dict[str, float]
    alerts: List[Alert]
    should_retrain: bool


# ---------------------------------------------------------------------------
# Statistical tests for drift
# ---------------------------------------------------------------------------

def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (statistic, p_value).
    """
    stat, p_val = stats.ks_2samp(reference, current)
    return float(stat), float(p_val)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI interpretation:
      < 0.10  no significant shift
      0.10-0.25  moderate shift
      > 0.25  significant shift
    """
    # Create bins from the reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# Synthetic data generator with controllable drift
# ---------------------------------------------------------------------------

class DriftingDataGenerator:
    """Generates synthetic production data with optional drift.

    Drift modes:
      - "none"       : data identical to training distribution
      - "gradual"    : slow feature mean shift over time
      - "sudden"     : abrupt shift after a specified window
      - "concept"    : feature distributions unchanged but label relationship changes
    """

    def __init__(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        feature_names: List[str],
        random_state: int = 42,
    ):
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.feature_names = feature_names
        self.rng = np.random.RandomState(random_state)

        # Reference statistics
        self.ref_means = X_ref.mean(axis=0)
        self.ref_stds = X_ref.std(axis=0)

    def generate(
        self,
        n_samples: int = 100,
        window_id: int = 0,
        drift_mode: str = "none",
        drift_magnitude: float = 0.5,
        sudden_window: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y_true) for one monitoring window."""
        # Sample with replacement from the reference
        indices = self.rng.choice(len(self.X_ref), size=n_samples, replace=True)
        X = self.X_ref[indices].copy()
        y = self.y_ref[indices].copy()

        # Add baseline noise
        noise = self.rng.normal(0, 0.05 * self.ref_stds, size=X.shape)
        X = X + noise

        if drift_mode == "gradual":
            # Shift grows linearly with the window id
            shift = drift_magnitude * (window_id / 10.0) * self.ref_stds
            X = X + shift

        elif drift_mode == "sudden":
            if window_id >= sudden_window:
                shift = drift_magnitude * self.ref_stds * 2.0
                X = X + shift

        elif drift_mode == "concept":
            # Same features but randomly flip a fraction of labels
            flip_rate = min(0.05 * window_id, 0.4)
            flip_mask = self.rng.random(n_samples) < flip_rate
            n_classes = len(np.unique(self.y_ref))
            y[flip_mask] = self.rng.randint(0, n_classes, size=flip_mask.sum())

        return X, y


# ---------------------------------------------------------------------------
# Model Monitor
# ---------------------------------------------------------------------------

class ModelMonitor:
    """Core monitoring engine."""

    def __init__(
        self,
        model,
        X_reference: np.ndarray,
        y_reference: np.ndarray,
        feature_names: List[str],
        ks_threshold: float = 0.1,
        psi_threshold: float = 0.15,
        accuracy_threshold: float = 0.85,
        f1_threshold: float = 0.80,
        drift_window_trigger: int = 3,
    ):
        self.model = model
        self.X_ref = X_reference
        self.y_ref = y_reference
        self.feature_names = feature_names

        # Thresholds
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.accuracy_threshold = accuracy_threshold
        self.f1_threshold = f1_threshold
        self.drift_window_trigger = drift_window_trigger

        # History
        self.reports: List[MonitoringReport] = []
        self.consecutive_drift_windows = 0

    # ----- public API -----

    def evaluate_window(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        window_id: int = 0,
    ) -> MonitoringReport:
        """Evaluate a single window of production data."""
        timestamp = datetime.datetime.utcnow().isoformat()
        alerts: List[Alert] = []

        # 1. Data drift per feature
        data_drift = {}
        for i, fname in enumerate(self.feature_names):
            ks_stat, ks_p = ks_test(self.X_ref[:, i], X[:, i])
            psi = population_stability_index(self.X_ref[:, i], X[:, i])

            drifted = ks_stat > self.ks_threshold or psi > self.psi_threshold

            data_drift[fname] = {
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_p, 4),
                "psi": round(psi, 4),
                "drifted": drifted,
            }

            if drifted:
                severity = "critical" if psi > 0.25 else "warning"
                alerts.append(Alert(
                    timestamp=timestamp,
                    alert_type="data_drift",
                    severity=severity,
                    feature=fname,
                    metric_name="psi",
                    metric_value=round(psi, 4),
                    threshold=self.psi_threshold,
                    message=f"Data drift detected on {fname}: PSI={psi:.4f}, KS={ks_stat:.4f}",
                ))

        # 2. Performance metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        perf = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
        }

        if accuracy < self.accuracy_threshold:
            alerts.append(Alert(
                timestamp=timestamp,
                alert_type="performance",
                severity="critical" if accuracy < self.accuracy_threshold - 0.1 else "warning",
                feature=None,
                metric_name="accuracy",
                metric_value=round(accuracy, 4),
                threshold=self.accuracy_threshold,
                message=f"Accuracy degraded to {accuracy:.4f} (threshold {self.accuracy_threshold})",
            ))

        if f1 < self.f1_threshold:
            alerts.append(Alert(
                timestamp=timestamp,
                alert_type="performance",
                severity="warning",
                feature=None,
                metric_name="f1_macro",
                metric_value=round(f1, 4),
                threshold=self.f1_threshold,
                message=f"F1 degraded to {f1:.4f} (threshold {self.f1_threshold})",
            ))

        # 3. Concept drift – compare rolling performance against baseline
        concept_drift = self._compute_concept_drift(accuracy, f1)

        # 4. Retraining trigger
        any_drift = any(d["drifted"] for d in data_drift.values())
        perf_degraded = accuracy < self.accuracy_threshold or f1 < self.f1_threshold

        if any_drift or perf_degraded:
            self.consecutive_drift_windows += 1
        else:
            self.consecutive_drift_windows = 0

        should_retrain = self.consecutive_drift_windows >= self.drift_window_trigger

        if should_retrain:
            alerts.append(Alert(
                timestamp=timestamp,
                alert_type="concept_drift",
                severity="critical",
                feature=None,
                metric_name="consecutive_drift_windows",
                metric_value=self.consecutive_drift_windows,
                threshold=self.drift_window_trigger,
                message=(
                    f"Retraining recommended: {self.consecutive_drift_windows} consecutive "
                    f"windows with drift or degradation"
                ),
            ))

        report = MonitoringReport(
            window_id=window_id,
            timestamp=timestamp,
            n_samples=len(X),
            data_drift=data_drift,
            concept_drift=concept_drift,
            performance_metrics=perf,
            alerts=alerts,
            should_retrain=should_retrain,
        )
        self.reports.append(report)
        return report

    def _compute_concept_drift(self, accuracy: float, f1: float) -> Dict[str, float]:
        """Compute concept drift by comparing latest metrics to historical mean."""
        if len(self.reports) < 2:
            return {"accuracy_delta": 0.0, "f1_delta": 0.0}

        hist_acc = np.mean([r.performance_metrics["accuracy"] for r in self.reports[-5:]])
        hist_f1 = np.mean([r.performance_metrics["f1_macro"] for r in self.reports[-5:]])

        return {
            "accuracy_delta": round(accuracy - hist_acc, 4),
            "f1_delta": round(f1 - hist_f1, 4),
        }

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return monitoring history as a DataFrame."""
        rows = []
        for r in self.reports:
            row = {
                "window_id": r.window_id,
                "timestamp": r.timestamp,
                "n_samples": r.n_samples,
                "accuracy": r.performance_metrics["accuracy"],
                "f1_macro": r.performance_metrics["f1_macro"],
                "n_alerts": len(r.alerts),
                "should_retrain": r.should_retrain,
            }
            # Add per-feature PSI
            for fname, drift in r.data_drift.items():
                row[f"psi_{fname}"] = drift["psi"]
                row[f"ks_{fname}"] = drift["ks_statistic"]
            rows.append(row)
        return pd.DataFrame(rows)

    def save_report(self, path: Path = None):
        """Persist all reports to JSON."""
        path = path or OUTPUT_DIR / "monitoring_reports.json"
        data = []
        for r in self.reports:
            entry = {
                "window_id": r.window_id,
                "timestamp": r.timestamp,
                "n_samples": r.n_samples,
                "performance_metrics": r.performance_metrics,
                "concept_drift": r.concept_drift,
                "n_drifted_features": sum(1 for d in r.data_drift.values() if d["drifted"]),
                "should_retrain": r.should_retrain,
                "alerts": [
                    {
                        "type": a.alert_type,
                        "severity": a.severity,
                        "feature": a.feature,
                        "metric": a.metric_name,
                        "value": a.metric_value,
                        "threshold": a.threshold,
                        "message": a.message,
                    }
                    for a in r.alerts
                ],
            }
            data.append(entry)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Reports saved to {path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_monitoring_dashboard(monitor: ModelMonitor, output_path: Path = None):
    """Generate a multi-panel monitoring dashboard."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available, skipping plots")
        return

    df = monitor.get_history_dataframe()
    if df.empty:
        return

    output_path = output_path or OUTPUT_DIR / "monitoring_dashboard.png"
    feature_names = monitor.feature_names

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Monitoring Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Accuracy and F1 over time
    ax = axes[0, 0]
    ax.plot(df["window_id"], df["accuracy"], "o-", label="Accuracy", linewidth=2)
    ax.plot(df["window_id"], df["f1_macro"], "s-", label="F1 Macro", linewidth=2)
    ax.axhline(y=monitor.accuracy_threshold, color="red", linestyle="--", alpha=0.7, label="Accuracy Threshold")
    ax.axhline(y=monitor.f1_threshold, color="orange", linestyle="--", alpha=0.7, label="F1 Threshold")
    ax.set_xlabel("Window")
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 2: PSI per feature
    ax = axes[0, 1]
    psi_cols = [c for c in df.columns if c.startswith("psi_")]
    for col in psi_cols:
        ax.plot(df["window_id"], df[col], "o-", label=col.replace("psi_", ""), alpha=0.7)
    ax.axhline(y=monitor.psi_threshold, color="red", linestyle="--", alpha=0.7, label="PSI Threshold")
    ax.set_xlabel("Window")
    ax.set_ylabel("PSI")
    ax.set_title("Population Stability Index per Feature")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: KS statistic per feature
    ax = axes[1, 0]
    ks_cols = [c for c in df.columns if c.startswith("ks_")]
    for col in ks_cols:
        ax.plot(df["window_id"], df[col], "o-", label=col.replace("ks_", ""), alpha=0.7)
    ax.axhline(y=monitor.ks_threshold, color="red", linestyle="--", alpha=0.7, label="KS Threshold")
    ax.set_xlabel("Window")
    ax.set_ylabel("KS Statistic")
    ax.set_title("Kolmogorov-Smirnov Statistic per Feature")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: Alerts per window
    ax = axes[1, 1]
    colors = ["green" if not r else "red" for r in df["should_retrain"]]
    ax.bar(df["window_id"], df["n_alerts"], color=colors, alpha=0.7)
    ax.set_xlabel("Window")
    ax.set_ylabel("Number of Alerts")
    ax.set_title("Alerts per Window (red = retrain triggered)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved to {output_path}")


# ---------------------------------------------------------------------------
# Demo simulation
# ---------------------------------------------------------------------------

def run_monitoring_simulation():
    """Full end-to-end demonstration with synthetic drift."""
    print("\n" + "=" * 60)
    print("  Model Monitoring Simulation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Train a reference model
    # ------------------------------------------------------------------
    print("\n[Phase 1] Training reference model ...")
    data = load_iris()
    feature_names = [f"feature_{i}" for i in range(data.data.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    model.fit(X_train, y_train)

    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    # ------------------------------------------------------------------
    # 2. Set up monitor
    # ------------------------------------------------------------------
    monitor = ModelMonitor(
        model=model,
        X_reference=X_test,
        y_reference=y_test,
        feature_names=feature_names,
        ks_threshold=0.15,
        psi_threshold=0.15,
        accuracy_threshold=0.85,
        f1_threshold=0.80,
        drift_window_trigger=3,
    )

    generator = DriftingDataGenerator(
        X_ref=X_test,
        y_ref=y_test,
        feature_names=feature_names,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 3. Simulate monitoring windows
    # ------------------------------------------------------------------
    scenarios = [
        # (window_id, drift_mode, drift_magnitude, description)
        (0, "none", 0.0, "Stable production data"),
        (1, "none", 0.0, "Stable production data"),
        (2, "none", 0.0, "Stable production data"),
        (3, "gradual", 0.3, "Gradual drift begins"),
        (4, "gradual", 0.3, "Gradual drift continues"),
        (5, "gradual", 0.3, "Gradual drift continues"),
        (6, "gradual", 0.5, "Drift intensifies"),
        (7, "gradual", 0.5, "Drift intensifies"),
        (8, "sudden", 1.0, "Sudden shift"),
        (9, "sudden", 1.0, "Sudden shift persists"),
        (10, "concept", 0.0, "Concept drift"),
        (11, "concept", 0.0, "Concept drift worsens"),
        (12, "concept", 0.0, "Concept drift worsens"),
        (13, "concept", 0.0, "Concept drift worsens further"),
        (14, "none", 0.0, "Recovery after retraining"),
    ]

    print(f"\n[Phase 2] Running {len(scenarios)} monitoring windows ...\n")

    for wid, drift_mode, magnitude, desc in scenarios:
        X_prod, y_prod = generator.generate(
            n_samples=200,
            window_id=wid,
            drift_mode=drift_mode,
            drift_magnitude=magnitude,
            sudden_window=8,
        )

        report = monitor.evaluate_window(X_prod, y_prod, window_id=wid)

        status = "RETRAIN" if report.should_retrain else "OK"
        n_drifted = sum(1 for d in report.data_drift.values() if d["drifted"])
        print(
            f"  Window {wid:2d} | {desc:35s} | "
            f"acc={report.performance_metrics['accuracy']:.3f} "
            f"f1={report.performance_metrics['f1_macro']:.3f} | "
            f"drifted_features={n_drifted} | "
            f"alerts={len(report.alerts):2d} | {status}"
        )

    # ------------------------------------------------------------------
    # 4. Save results and visualise
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Saving results ...")
    monitor.save_report()

    df = monitor.get_history_dataframe()
    df.to_csv(OUTPUT_DIR / "monitoring_history.csv", index=False)
    print(f"  History CSV saved to {OUTPUT_DIR / 'monitoring_history.csv'}")

    plot_monitoring_dashboard(monitor)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    total_alerts = sum(len(r.alerts) for r in monitor.reports)
    retrain_windows = sum(1 for r in monitor.reports if r.should_retrain)

    print(f"\n{'=' * 60}")
    print(f"  Simulation Summary")
    print(f"  Windows evaluated:    {len(monitor.reports)}")
    print(f"  Total alerts raised:  {total_alerts}")
    print(f"  Retrain triggered in: {retrain_windows} windows")
    print(f"  Final accuracy:       {monitor.reports[-1].performance_metrics['accuracy']:.4f}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_monitoring_simulation()
