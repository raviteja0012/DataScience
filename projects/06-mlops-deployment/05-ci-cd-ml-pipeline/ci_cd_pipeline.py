"""
CI/CD Pipeline Simulation for Machine Learning
================================================
Simulates a production CI/CD pipeline for ML models including:
  - Automated test framework (data validation, model performance, regression)
  - A/B testing framework with statistical significance
  - Canary deployment simulation with progressive traffic shifting
  - Rollback logic with health-check gating
  - Pipeline stage orchestration with pass/fail reporting

This is a self-contained simulation -- no external services required.

Usage:
    python ci_cd_pipeline.py
"""

import json
import time
import hashlib
import datetime
import warnings
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "ci_cd_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DeploymentStrategy(str, Enum):
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"


@dataclass
class StageResult:
    name: str
    status: StageStatus
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    pipeline_name: str
    run_id: str
    status: StageStatus
    stages: List[StageResult]
    started_at: str
    finished_at: str
    total_seconds: float


# ---------------------------------------------------------------------------
# Test Framework for ML Models
# ---------------------------------------------------------------------------

class MLTestSuite:
    """Automated testing framework for ML models and data."""

    def __init__(self, model, X_train, y_train, X_test, y_test, feature_names, target_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.target_names = target_names
        self.results: List[Dict] = []

    def _record(self, test_name: str, passed: bool, details: dict = None):
        self.results.append({
            "test": test_name,
            "passed": passed,
            "details": details or {},
        })

    # ---- Data Validation Tests ----

    def test_no_missing_values(self, X: pd.DataFrame) -> bool:
        """Verify that the dataset contains no missing values."""
        n_missing = X.isnull().sum().sum()
        passed = n_missing == 0
        self._record("data_no_missing_values", passed, {"n_missing": int(n_missing)})
        return passed

    def test_feature_ranges(self, X: pd.DataFrame, min_val: float = -100, max_val: float = 100) -> bool:
        """Verify all features are within expected bounds."""
        out_of_range = ((X < min_val) | (X > max_val)).sum().sum()
        passed = out_of_range == 0
        self._record("data_feature_ranges", passed, {
            "min_val": min_val, "max_val": max_val, "out_of_range": int(out_of_range),
        })
        return passed

    def test_no_duplicate_rows(self, X: pd.DataFrame) -> bool:
        """Verify there are no exact duplicate rows."""
        n_dups = X.duplicated().sum()
        passed = n_dups == 0
        self._record("data_no_duplicates", passed, {"n_duplicates": int(n_dups)})
        return passed

    def test_class_balance(self, y: pd.Series, max_imbalance_ratio: float = 10.0) -> bool:
        """Verify class distribution is not extremely imbalanced."""
        counts = y.value_counts()
        ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
        passed = ratio <= max_imbalance_ratio
        self._record("data_class_balance", passed, {
            "imbalance_ratio": round(ratio, 2),
            "max_allowed": max_imbalance_ratio,
            "class_counts": counts.to_dict(),
        })
        return passed

    def test_feature_count(self, X: pd.DataFrame, expected: int) -> bool:
        """Verify the feature count matches expectation."""
        actual = X.shape[1]
        passed = actual == expected
        self._record("data_feature_count", passed, {"expected": expected, "actual": actual})
        return passed

    def test_schema_consistency(self, X: pd.DataFrame) -> bool:
        """Verify column names match expected feature names."""
        actual = list(X.columns)
        passed = actual == self.feature_names
        self._record("data_schema_consistency", passed, {
            "expected": self.feature_names, "actual": actual,
        })
        return passed

    # ---- Model Performance Tests ----

    def test_accuracy_threshold(self, threshold: float = 0.85) -> bool:
        """Model accuracy must meet minimum threshold."""
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        passed = acc >= threshold
        self._record("model_accuracy_threshold", passed, {
            "accuracy": round(acc, 4), "threshold": threshold,
        })
        return passed

    def test_f1_threshold(self, threshold: float = 0.80) -> bool:
        """Model F1 score must meet minimum threshold."""
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        passed = f1 >= threshold
        self._record("model_f1_threshold", passed, {
            "f1_macro": round(f1, 4), "threshold": threshold,
        })
        return passed

    def test_no_class_has_zero_recall(self) -> bool:
        """Every class must have non-zero recall (no complete failure on any class)."""
        y_pred = self.model.predict(self.X_test)
        recalls = recall_score(self.y_test, y_pred, average=None, zero_division=0)
        min_recall = float(recalls.min())
        passed = min_recall > 0.0
        self._record("model_no_zero_recall", passed, {
            "per_class_recall": [round(r, 4) for r in recalls],
            "min_recall": round(min_recall, 4),
        })
        return passed

    def test_cross_validation_stability(self, min_cv_mean: float = 0.80, max_cv_std: float = 0.10) -> bool:
        """Cross-validation results must be stable and above threshold."""
        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.concatenate([self.y_train, self.y_test])
        scores = cross_val_score(self.model, X_all, y_all, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        std_score = scores.std()
        passed = mean_score >= min_cv_mean and std_score <= max_cv_std
        self._record("model_cv_stability", passed, {
            "cv_mean": round(float(mean_score), 4),
            "cv_std": round(float(std_score), 4),
            "min_cv_mean": min_cv_mean,
            "max_cv_std": max_cv_std,
        })
        return passed

    def test_prediction_latency(self, max_ms: float = 100.0, n_samples: int = 100) -> bool:
        """Single-sample prediction must complete within latency budget."""
        X_single = self.X_test[:1]
        times = []
        for _ in range(n_samples):
            t0 = time.time()
            self.model.predict(X_single)
            times.append((time.time() - t0) * 1000)
        p95 = np.percentile(times, 95)
        passed = p95 <= max_ms
        self._record("model_prediction_latency", passed, {
            "p95_ms": round(p95, 3),
            "max_ms": max_ms,
            "mean_ms": round(np.mean(times), 3),
        })
        return passed

    # ---- Regression Tests ----

    def test_performance_regression(self, baseline_accuracy: float, tolerance: float = 0.02) -> bool:
        """New model must not regress more than tolerance below baseline."""
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        passed = acc >= baseline_accuracy - tolerance
        self._record("model_performance_regression", passed, {
            "current_accuracy": round(acc, 4),
            "baseline_accuracy": round(baseline_accuracy, 4),
            "tolerance": tolerance,
            "delta": round(acc - baseline_accuracy, 4),
        })
        return passed

    def test_prediction_determinism(self, n_runs: int = 5) -> bool:
        """Model predictions must be deterministic across multiple runs."""
        preds = []
        for _ in range(n_runs):
            preds.append(tuple(self.model.predict(self.X_test)))
        unique = len(set(preds))
        passed = unique == 1
        self._record("model_determinism", passed, {"unique_prediction_sets": unique})
        return passed

    # ---- Run all tests ----

    def run_all(self, baseline_accuracy: float = 0.90) -> Tuple[bool, List[Dict]]:
        """Run the complete test suite and return (all_passed, results)."""
        self.results = []

        X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        y_train_s = pd.Series(self.y_train)

        # Data validation
        self.test_no_missing_values(X_test_df)
        self.test_feature_ranges(X_test_df)
        self.test_no_duplicate_rows(X_train_df)
        self.test_class_balance(y_train_s)
        self.test_feature_count(X_test_df, expected=len(self.feature_names))
        self.test_schema_consistency(X_test_df)

        # Model performance
        self.test_accuracy_threshold(threshold=0.85)
        self.test_f1_threshold(threshold=0.80)
        self.test_no_class_has_zero_recall()
        self.test_cross_validation_stability()
        self.test_prediction_latency()

        # Regression
        self.test_performance_regression(baseline_accuracy)
        self.test_prediction_determinism()

        all_passed = all(r["passed"] for r in self.results)
        return all_passed, self.results


# ---------------------------------------------------------------------------
# A/B Testing Framework
# ---------------------------------------------------------------------------

class ABTestFramework:
    """Statistical A/B testing between two models."""

    def __init__(
        self,
        model_a,
        model_b,
        X_test: np.ndarray,
        y_test: np.ndarray,
        confidence_level: float = 0.95,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.X_test = X_test
        self.y_test = y_test
        self.confidence_level = confidence_level

    def run(self, n_bootstrap: int = 1000, random_state: int = 42) -> Dict[str, Any]:
        """Run A/B test and return detailed results."""
        rng = np.random.RandomState(random_state)

        y_pred_a = self.model_a.predict(self.X_test)
        y_pred_b = self.model_b.predict(self.X_test)

        acc_a = accuracy_score(self.y_test, y_pred_a)
        acc_b = accuracy_score(self.y_test, y_pred_b)
        f1_a = f1_score(self.y_test, y_pred_a, average="macro", zero_division=0)
        f1_b = f1_score(self.y_test, y_pred_b, average="macro", zero_division=0)

        # Bootstrap confidence interval for the accuracy difference
        n = len(self.y_test)
        diffs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            ba = accuracy_score(self.y_test[idx], y_pred_a[idx])
            bb = accuracy_score(self.y_test[idx], y_pred_b[idx])
            diffs.append(bb - ba)

        diffs = np.array(diffs)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(diffs, 100 * alpha / 2)
        ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
        p_b_better = (diffs > 0).mean()

        # McNemar test for paired comparison
        correct_a = y_pred_a == self.y_test
        correct_b = y_pred_b == self.y_test
        # Cells: both correct, only A correct, only B correct, neither
        b_only = ((~correct_a) & correct_b).sum()
        a_only = (correct_a & (~correct_b)).sum()

        if b_only + a_only > 0:
            mcnemar_stat = (abs(b_only - a_only) - 1) ** 2 / (b_only + a_only)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0.0
            mcnemar_p = 1.0

        significant = mcnemar_p < (1 - self.confidence_level)

        winner = "B" if acc_b > acc_a else "A" if acc_a > acc_b else "tie"
        if not significant:
            winner = "no_significant_difference"

        result = {
            "model_a": {
                "accuracy": round(acc_a, 4),
                "f1_macro": round(f1_a, 4),
            },
            "model_b": {
                "accuracy": round(acc_b, 4),
                "f1_macro": round(f1_b, 4),
            },
            "accuracy_diff_b_minus_a": round(float(acc_b - acc_a), 4),
            "bootstrap_ci": {
                "lower": round(float(ci_lower), 4),
                "upper": round(float(ci_upper), 4),
                "confidence_level": self.confidence_level,
            },
            "p_b_better": round(float(p_b_better), 4),
            "mcnemar_test": {
                "statistic": round(float(mcnemar_stat), 4),
                "p_value": round(float(mcnemar_p), 4),
            },
            "statistically_significant": significant,
            "winner": winner,
        }
        return result


# ---------------------------------------------------------------------------
# Canary Deployment Simulation
# ---------------------------------------------------------------------------

class CanaryDeployment:
    """Simulates a canary deployment with progressive traffic shifting."""

    def __init__(
        self,
        current_model,
        candidate_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        traffic_steps: List[float] = None,
        error_threshold: float = 0.10,
        min_accuracy: float = 0.85,
    ):
        self.current_model = current_model
        self.candidate_model = candidate_model
        self.X_test = X_test
        self.y_test = y_test
        self.traffic_steps = traffic_steps or [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        self.error_threshold = error_threshold
        self.min_accuracy = min_accuracy
        self.log: List[Dict] = []

    def run(self, random_state: int = 42) -> Dict[str, Any]:
        """Execute the canary deployment simulation."""
        rng = np.random.RandomState(random_state)
        n = len(self.X_test)
        rollback = False
        final_traffic = 0.0

        for step, canary_pct in enumerate(self.traffic_steps):
            # Split traffic
            canary_size = max(1, int(n * canary_pct))
            indices = rng.permutation(n)
            canary_idx = indices[:canary_size]
            current_idx = indices[canary_size:]

            # Get predictions from both models
            canary_preds = self.candidate_model.predict(self.X_test[canary_idx])
            canary_acc = accuracy_score(self.y_test[canary_idx], canary_preds)
            canary_errors = 1.0 - canary_acc

            current_preds = self.current_model.predict(self.X_test[current_idx]) if len(current_idx) > 0 else np.array([])
            current_acc = accuracy_score(self.y_test[current_idx], current_preds) if len(current_idx) > 0 else 1.0

            entry = {
                "step": step,
                "canary_traffic_pct": canary_pct,
                "canary_accuracy": round(canary_acc, 4),
                "canary_error_rate": round(canary_errors, 4),
                "current_accuracy": round(current_acc, 4),
                "canary_samples": canary_size,
                "passed": True,
            }

            # Check health gates
            if canary_errors > self.error_threshold:
                entry["passed"] = False
                entry["failure_reason"] = f"Error rate {canary_errors:.4f} exceeds threshold {self.error_threshold}"
                self.log.append(entry)
                rollback = True
                break

            if canary_acc < self.min_accuracy:
                entry["passed"] = False
                entry["failure_reason"] = f"Accuracy {canary_acc:.4f} below minimum {self.min_accuracy}"
                self.log.append(entry)
                rollback = True
                break

            self.log.append(entry)
            final_traffic = canary_pct

        return {
            "completed": not rollback,
            "final_canary_traffic": final_traffic,
            "rollback": rollback,
            "steps": self.log,
        }


# ---------------------------------------------------------------------------
# Rollback Manager
# ---------------------------------------------------------------------------

class RollbackManager:
    """Manages model versions and handles rollback on failure."""

    def __init__(self):
        self.versions: List[Dict] = []
        self.active_idx: int = -1

    def register(self, model, version_tag: str, metrics: dict):
        self.versions.append({
            "version": version_tag,
            "model": model,
            "metrics": metrics,
            "deployed_at": datetime.datetime.utcnow().isoformat(),
        })
        self.active_idx = len(self.versions) - 1

    def rollback(self) -> Dict[str, Any]:
        """Roll back to the previous version."""
        if self.active_idx <= 0:
            return {"success": False, "reason": "No previous version to roll back to"}

        old_version = self.versions[self.active_idx]["version"]
        self.active_idx -= 1
        new_version = self.versions[self.active_idx]["version"]
        return {
            "success": True,
            "rolled_back_from": old_version,
            "rolled_back_to": new_version,
            "active_metrics": self.versions[self.active_idx]["metrics"],
        }

    @property
    def active_model(self):
        if self.active_idx >= 0:
            return self.versions[self.active_idx]["model"]
        return None

    @property
    def active_version(self):
        if self.active_idx >= 0:
            return self.versions[self.active_idx]["version"]
        return None


# ---------------------------------------------------------------------------
# CI/CD Pipeline Orchestrator
# ---------------------------------------------------------------------------

class MLCICDPipeline:
    """Orchestrates the full CI/CD pipeline for an ML model."""

    def __init__(self, pipeline_name: str = "ml-cicd"):
        self.pipeline_name = pipeline_name
        self.run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.stages: List[StageResult] = []

    def _run_stage(self, name: str, fn: Callable, **kwargs) -> StageResult:
        """Execute a single pipeline stage and record the result."""
        print(f"\n  --- Stage: {name} ---")
        t0 = time.time()
        try:
            result = fn(**kwargs)
            duration = time.time() - t0
            passed = result.get("passed", True) if isinstance(result, dict) else bool(result)
            status = StageStatus.PASSED if passed else StageStatus.FAILED
            details = result if isinstance(result, dict) else {}
            stage = StageResult(name=name, status=status, duration_seconds=round(duration, 3), details=details)
        except Exception as exc:
            duration = time.time() - t0
            stage = StageResult(
                name=name,
                status=StageStatus.FAILED,
                duration_seconds=round(duration, 3),
                error=str(exc),
            )
        self.stages.append(stage)
        print(f"  Result: {stage.status.value} ({stage.duration_seconds}s)")
        return stage

    def run(self) -> PipelineResult:
        """Execute the full CI/CD pipeline."""
        started_at = datetime.datetime.utcnow().isoformat()
        t_total = time.time()

        print(f"\n{'='*60}")
        print(f"  CI/CD Pipeline: {self.pipeline_name}")
        print(f"  Run ID:         {self.run_id}")
        print(f"{'='*60}")

        # ---- 1. Data Preparation ----
        data = load_iris()
        feature_names = [f"feature_{i}" for i in range(data.data.shape[1])]
        target_names = list(data.target_names)

        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=42, stratify=data.target,
        )

        # ---- 2. Train baseline model ----
        baseline = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, random_state=42))])
        baseline.fit(X_train, y_train)
        baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

        rollback_mgr = RollbackManager()
        rollback_mgr.register(baseline, "v1.0-baseline", {"accuracy": round(baseline_acc, 4)})
        print(f"\n  Baseline model accuracy: {baseline_acc:.4f}")

        # ---- 3. Train candidate model ----
        candidate = Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])
        candidate.fit(X_train, y_train)
        candidate_acc = accuracy_score(y_test, candidate.predict(X_test))
        print(f"  Candidate model accuracy: {candidate_acc:.4f}")

        # ---- Stage: Data Validation ----
        def run_data_validation():
            suite = MLTestSuite(candidate, X_train, y_train, X_test, y_test, feature_names, target_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            y_train_s = pd.Series(y_train)

            results = []
            results.append({"test": "no_missing", "passed": suite.test_no_missing_values(X_test_df)})
            results.append({"test": "feature_ranges", "passed": suite.test_feature_ranges(X_test_df)})
            results.append({"test": "class_balance", "passed": suite.test_class_balance(y_train_s)})
            results.append({"test": "feature_count", "passed": suite.test_feature_count(X_test_df, len(feature_names))})
            results.append({"test": "schema", "passed": suite.test_schema_consistency(X_test_df)})

            all_pass = all(r["passed"] for r in results)
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"    [{status}] {r['test']}")
            return {"passed": all_pass, "tests": results}

        stage = self._run_stage("data_validation", run_data_validation)
        if stage.status == StageStatus.FAILED:
            return self._finalise(started_at, t_total)

        # ---- Stage: Model Tests ----
        def run_model_tests():
            suite = MLTestSuite(candidate, X_train, y_train, X_test, y_test, feature_names, target_names)
            all_passed, results = suite.run_all(baseline_accuracy=baseline_acc)
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"    [{status}] {r['test']}")
            return {"passed": all_passed, "total": len(results), "failed": sum(1 for r in results if not r["passed"]), "results": results}

        stage = self._run_stage("model_tests", run_model_tests)
        if stage.status == StageStatus.FAILED:
            return self._finalise(started_at, t_total)

        # ---- Stage: A/B Test ----
        def run_ab_test():
            ab = ABTestFramework(baseline, candidate, X_test, y_test, confidence_level=0.95)
            result = ab.run()
            print(f"    Model A accuracy: {result['model_a']['accuracy']}")
            print(f"    Model B accuracy: {result['model_b']['accuracy']}")
            print(f"    Winner: {result['winner']}")
            print(f"    Significant: {result['statistically_significant']}")
            # Pass if B is not significantly worse than A
            is_not_worse = result["accuracy_diff_b_minus_a"] >= -0.02
            return {"passed": is_not_worse, **result}

        self._run_stage("ab_testing", run_ab_test)

        # ---- Stage: Canary Deployment ----
        def run_canary():
            canary = CanaryDeployment(
                current_model=baseline,
                candidate_model=candidate,
                X_test=X_test,
                y_test=y_test,
                traffic_steps=[0.05, 0.10, 0.25, 0.50, 0.75, 1.00],
                error_threshold=0.15,
                min_accuracy=0.80,
            )
            result = canary.run()
            for step in result["steps"]:
                status = "PASS" if step["passed"] else "FAIL"
                print(f"    [{status}] traffic={step['canary_traffic_pct']:.0%} acc={step['canary_accuracy']:.4f}")
            return {"passed": result["completed"], **result}

        stage = self._run_stage("canary_deployment", run_canary)

        # ---- Stage: Deploy or Rollback ----
        if stage.status == StageStatus.PASSED:
            def deploy():
                rollback_mgr.register(candidate, "v2.0-candidate", {"accuracy": round(candidate_acc, 4)})
                print(f"    Deployed {rollback_mgr.active_version}")
                return {"passed": True, "deployed_version": rollback_mgr.active_version}
            self._run_stage("deploy", deploy)
        else:
            def rollback():
                result = rollback_mgr.rollback()
                if result["success"]:
                    print(f"    Rolled back to {result['rolled_back_to']}")
                else:
                    print(f"    Rollback failed: {result['reason']}")
                return {"passed": result["success"], **result}
            self._run_stage("rollback", rollback)

        # ---- Stage: Post-deploy Health Check ----
        def post_deploy_check():
            active = rollback_mgr.active_model
            preds = active.predict(X_test)
            acc = accuracy_score(y_test, preds)
            healthy = acc >= 0.80
            print(f"    Active model accuracy: {acc:.4f} (healthy={healthy})")
            return {"passed": healthy, "accuracy": round(acc, 4)}

        self._run_stage("post_deploy_health_check", post_deploy_check)

        return self._finalise(started_at, t_total)

    def _finalise(self, started_at: str, t_total_start: float) -> PipelineResult:
        total_seconds = round(time.time() - t_total_start, 2)
        all_passed = all(s.status in (StageStatus.PASSED, StageStatus.SKIPPED) for s in self.stages)
        overall_status = StageStatus.PASSED if all_passed else StageStatus.FAILED

        result = PipelineResult(
            pipeline_name=self.pipeline_name,
            run_id=self.run_id,
            status=overall_status,
            stages=self.stages,
            started_at=started_at,
            finished_at=datetime.datetime.utcnow().isoformat(),
            total_seconds=total_seconds,
        )

        # Save report
        report_path = OUTPUT_DIR / f"pipeline_report_{self.run_id}.json"
        report_data = {
            "pipeline_name": result.pipeline_name,
            "run_id": result.run_id,
            "status": result.status.value,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "total_seconds": result.total_seconds,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration_seconds": s.duration_seconds,
                    "error": s.error,
                    "details": _make_serialisable(s.details),
                }
                for s in result.stages
            ],
        }
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*60}")
        print(f"  Pipeline Result: {overall_status.value.upper()}")
        print(f"  Duration: {total_seconds}s")
        print(f"  Report: {report_path}")
        print(f"{'='*60}")
        for s in self.stages:
            icon = "PASS" if s.status == StageStatus.PASSED else "FAIL"
            print(f"  [{icon}] {s.name} ({s.duration_seconds}s)")
        print()

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pipeline = MLCICDPipeline(pipeline_name="iris-classifier-cicd")
    result = pipeline.run()

    # Also demonstrate the A/B testing framework standalone
    print("\n" + "=" * 60)
    print("  Standalone A/B Test Demonstration")
    print("=" * 60)

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target,
    )

    model_a = Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=50, random_state=1))])
    model_b = Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=200, random_state=2))])
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    ab = ABTestFramework(model_a, model_b, X_test, y_test)
    ab_result = ab.run()
    print(f"\n  A/B Test Result:\n{json.dumps(ab_result, indent=4)}\n")


if __name__ == "__main__":
    main()
