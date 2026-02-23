"""Statistical anomaly detection methods for payment transactions.

Implements multiple complementary statistical approaches for identifying
outliers in transaction data: Z-score, modified Z-score (MAD-based),
IQR fencing, and isolation forest scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DetectionMethod(str, Enum):
    """Available statistical detection methods."""

    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


@dataclass(frozen=True)
class AnomalyScore:
    """Score for a single observation across detection methods.

    Attributes:
        index: Row index in the original DataFrame.
        score: Composite anomaly score (0.0 = normal, 1.0 = highly anomalous).
        method_scores: Individual scores from each detection method.
        is_anomaly: Whether this observation is flagged as anomalous.
        features: Feature values that contributed to the anomaly score.
    """

    index: int
    score: float
    method_scores: dict[str, float]
    is_anomaly: bool
    features: dict[str, float]


class StatisticalDetector:
    """Multi-method statistical anomaly detector.

    Runs multiple detection algorithms and produces a composite score
    using weighted averaging. This ensemble approach reduces false positives
    compared to any single method.

    Attributes:
        zscore_threshold: Z-score threshold for outlier detection.
        iqr_multiplier: IQR multiplier for fence calculation.
        contamination: Expected proportion of outliers (for isolation forest).
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        contamination: float = 0.05,
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination

    def detect(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        methods: list[DetectionMethod] | None = None,
    ) -> list[AnomalyScore]:
        """Run anomaly detection on the given data.

        Args:
            data: DataFrame containing numeric features.
            columns: Specific columns to analyze (defaults to all numeric).
            methods: Detection methods to use (defaults to all).

        Returns:
            List of AnomalyScore objects for flagged anomalies.
        """
        if data.empty:
            return []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            logger.warning("no_numeric_columns_for_detection")
            return []

        if methods is None:
            methods = [DetectionMethod.ZSCORE, DetectionMethod.IQR, DetectionMethod.MODIFIED_ZSCORE]

        # Run each method and collect scores
        all_scores: dict[int, dict[str, float]] = {}

        for method in methods:
            method_scores = self._run_method(data, columns, method)
            for idx, score in method_scores.items():
                if idx not in all_scores:
                    all_scores[idx] = {}
                all_scores[idx][method.value] = score

        # Compute composite scores
        results: list[AnomalyScore] = []
        for idx, method_scores in all_scores.items():
            composite = np.mean(list(method_scores.values()))

            # Extract feature values for this row
            features = {}
            for col in columns:
                if idx < len(data):
                    features[col] = float(data.iloc[idx][col])

            results.append(AnomalyScore(
                index=idx,
                score=composite,
                method_scores=method_scores,
                is_anomaly=composite > 0.5,
                features=features,
            ))

        # Sort by composite score descending
        results.sort(key=lambda x: x.score, reverse=True)

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "statistical_detection_complete",
            total_rows=len(data),
            anomalies_found=anomaly_count,
            methods_used=[m.value for m in methods],
        )

        return results

    def _run_method(
        self,
        data: pd.DataFrame,
        columns: list[str],
        method: DetectionMethod,
    ) -> dict[int, float]:
        """Run a single detection method and return per-row scores.

        Scores are normalized to [0, 1] where higher means more anomalous.
        """
        if method == DetectionMethod.ZSCORE:
            return self._zscore_detect(data, columns)
        elif method == DetectionMethod.MODIFIED_ZSCORE:
            return self._modified_zscore_detect(data, columns)
        elif method == DetectionMethod.IQR:
            return self._iqr_detect(data, columns)
        elif method == DetectionMethod.ISOLATION_FOREST:
            return self._isolation_forest_detect(data, columns)
        else:
            logger.warning("unknown_method", method=method.value)
            return {}

    def _zscore_detect(self, data: pd.DataFrame, columns: list[str]) -> dict[int, float]:
        """Z-score based anomaly detection.

        Flags observations where any feature has an absolute z-score
        exceeding the threshold.
        """
        scores: dict[int, float] = {}

        for col in columns:
            values = data[col].dropna()
            if len(values) < 3:
                continue

            mean = values.mean()
            std = values.std()
            if std == 0:
                continue

            z_scores = np.abs((values - mean) / std)

            for idx, z in zip(values.index, z_scores):
                # Normalize to [0, 1] using sigmoid-like scaling
                normalized = min(z / (self.zscore_threshold * 2), 1.0)
                if idx in scores:
                    scores[idx] = max(scores[idx], normalized)
                else:
                    scores[idx] = normalized

        return scores

    def _modified_zscore_detect(self, data: pd.DataFrame, columns: list[str]) -> dict[int, float]:
        """Modified Z-score using Median Absolute Deviation (MAD).

        More robust to outliers than standard Z-score since it uses
        median instead of mean.
        """
        scores: dict[int, float] = {}

        for col in columns:
            values = data[col].dropna()
            if len(values) < 3:
                continue

            median = np.median(values)
            mad = np.median(np.abs(values - median))

            if mad == 0:
                continue

            # 0.6745 is the 0.75th quantile of the standard normal distribution
            modified_z = 0.6745 * np.abs(values - median) / mad

            for idx, mz in zip(values.index, modified_z):
                normalized = min(mz / (self.zscore_threshold * 2), 1.0)
                if idx in scores:
                    scores[idx] = max(scores[idx], normalized)
                else:
                    scores[idx] = normalized

        return scores

    def _iqr_detect(self, data: pd.DataFrame, columns: list[str]) -> dict[int, float]:
        """Interquartile range (IQR) based anomaly detection.

        Flags observations outside the IQR fences:
        lower fence = Q1 - multiplier * IQR
        upper fence = Q3 + multiplier * IQR
        """
        scores: dict[int, float] = {}

        for col in columns:
            values = data[col].dropna()
            if len(values) < 4:
                continue

            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_fence = q1 - self.iqr_multiplier * iqr
            upper_fence = q3 + self.iqr_multiplier * iqr

            for idx, val in zip(values.index, values):
                if val < lower_fence:
                    distance = (lower_fence - val) / iqr
                    normalized = min(distance / 3.0, 1.0)
                elif val > upper_fence:
                    distance = (val - upper_fence) / iqr
                    normalized = min(distance / 3.0, 1.0)
                else:
                    normalized = 0.0

                if idx in scores:
                    scores[idx] = max(scores[idx], normalized)
                else:
                    scores[idx] = normalized

        return scores

    def _isolation_forest_detect(self, data: pd.DataFrame, columns: list[str]) -> dict[int, float]:
        """Isolation forest based anomaly detection.

        Uses scikit-learn's IsolationForest for unsupervised detection.
        Falls back to IQR if sklearn is not available.
        """
        try:
            from sklearn.ensemble import IsolationForest

            numeric_data = data[columns].dropna()
            if len(numeric_data) < 10:
                return self._iqr_detect(data, columns)

            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            model.fit(numeric_data)

            # score_samples returns negative scores; lower = more anomalous
            raw_scores = model.score_samples(numeric_data)

            # Normalize to [0, 1]
            min_score = raw_scores.min()
            max_score = raw_scores.max()
            score_range = max_score - min_score

            scores: dict[int, float] = {}
            if score_range > 0:
                for idx, raw in zip(numeric_data.index, raw_scores):
                    # Invert so higher = more anomalous
                    normalized = 1.0 - (raw - min_score) / score_range
                    scores[idx] = float(normalized)
            else:
                for idx in numeric_data.index:
                    scores[idx] = 0.0

            return scores

        except ImportError:
            logger.warning("sklearn_not_available", fallback="iqr")
            return self._iqr_detect(data, columns)

    def detect_amount_outliers(
        self,
        data: pd.DataFrame,
        amount_column: str = "AMOUNT",
        group_column: str | None = None,
    ) -> pd.DataFrame:
        """Detect amount-based outliers with optional group-level analysis.

        When a group column is provided (e.g., MERCHANT_ID), outlier detection
        is performed per-group, enabling merchant-specific baseline comparisons.

        Args:
            data: Transaction DataFrame.
            amount_column: Column containing transaction amounts.
            group_column: Optional grouping column for per-group detection.

        Returns:
            DataFrame of flagged outlier transactions with scores.
        """
        if amount_column not in data.columns:
            return pd.DataFrame()

        if group_column and group_column in data.columns:
            outlier_frames: list[pd.DataFrame] = []
            for group_val, group_df in data.groupby(group_column):
                if len(group_df) < 5:
                    continue
                scores = self.detect(group_df, columns=[amount_column])
                flagged = [s for s in scores if s.is_anomaly]
                if flagged:
                    flagged_indices = [s.index for s in flagged]
                    outlier_df = group_df.loc[group_df.index.isin(flagged_indices)].copy()
                    outlier_df["anomaly_score"] = [s.score for s in flagged]
                    outlier_frames.append(outlier_df)

            if outlier_frames:
                return pd.concat(outlier_frames).sort_values("anomaly_score", ascending=False)
            return pd.DataFrame()
        else:
            scores = self.detect(data, columns=[amount_column])
            flagged = [s for s in scores if s.is_anomaly]
            if not flagged:
                return pd.DataFrame()
            flagged_indices = [s.index for s in flagged]
            result = data.loc[data.index.isin(flagged_indices)].copy()
            result["anomaly_score"] = [s.score for s in flagged]
            return result.sort_values("anomaly_score", ascending=False)
