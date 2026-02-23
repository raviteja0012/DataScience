"""Alert generation and routing for detected anomalies.

Manages the lifecycle of anomaly alerts: creation, deduplication,
severity classification, and formatting for display. Supports
configurable severity thresholds and alert aggregation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """Alert severity levels aligned with PCI DSS incident response."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Alert:
    """A generated anomaly alert.

    Attributes:
        alert_id: Unique alert identifier.
        severity: Alert severity level.
        category: Alert category (statistical, velocity, geographic, etc.).
        message: Human-readable alert message.
        details: Structured alert data.
        created_at: When the alert was generated.
        acknowledged: Whether the alert has been reviewed.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    severity: Severity = Severity.MEDIUM
    category: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to a dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """Manages anomaly alert generation, deduplication, and routing.

    Provides threshold-based alert generation with built-in deduplication
    to prevent alert fatigue. Alerts are classified by severity based on
    configurable thresholds.

    Attributes:
        critical_threshold: Anomaly score threshold for critical alerts.
        high_threshold: Score threshold for high-severity alerts.
        medium_threshold: Score threshold for medium-severity alerts.
    """

    def __init__(
        self,
        critical_threshold: float = 0.9,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.5,
        max_alerts: int = 50,
    ) -> None:
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.max_alerts = max_alerts
        self._active_alerts: list[Alert] = []
        self._alert_fingerprints: set[str] = set()

    @property
    def active_alert_count(self) -> int:
        """Number of unacknowledged active alerts."""
        return sum(1 for a in self._active_alerts if not a.acknowledged)

    def generate_alert(
        self,
        score: float,
        category: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Generate an alert if the score exceeds the threshold.

        Applies deduplication to prevent repeated alerts for the same
        underlying condition. Returns None if the alert is suppressed.

        Args:
            score: Anomaly score (0.0 to 1.0).
            category: Alert category name.
            message: Alert message.
            details: Additional alert details.

        Returns:
            Generated Alert or None if suppressed.
        """
        severity = self._classify_severity(score)
        if severity is None:
            return None

        # Deduplication using fingerprint
        fingerprint = f"{category}:{message[:50]}"
        if fingerprint in self._alert_fingerprints:
            logger.debug("alert_deduplicated", fingerprint=fingerprint)
            return None

        alert = Alert(
            severity=severity,
            category=category,
            message=message,
            details=details or {},
        )

        self._active_alerts.append(alert)
        self._alert_fingerprints.add(fingerprint)

        # Enforce max alerts limit
        if len(self._active_alerts) > self.max_alerts:
            self._active_alerts = self._active_alerts[-self.max_alerts:]

        logger.info(
            "alert_generated",
            alert_id=alert.alert_id,
            severity=severity.value,
            category=category,
        )

        return alert

    def generate_alerts_from_scores(
        self,
        scores: list[dict[str, Any]],
        category: str = "statistical",
    ) -> list[Alert]:
        """Generate alerts from a batch of anomaly scores.

        Args:
            scores: List of score dictionaries with 'score', 'message', and 'details'.
            category: Alert category for all generated alerts.

        Returns:
            List of generated Alert objects.
        """
        alerts: list[Alert] = []
        for score_entry in scores:
            alert = self.generate_alert(
                score=score_entry.get("score", 0),
                category=category,
                message=score_entry.get("message", "Anomaly detected"),
                details=score_entry.get("details", {}),
            )
            if alert is not None:
                alerts.append(alert)

        return alerts

    def generate_alerts_from_violations(
        self,
        violations: list[Any],
    ) -> list[Alert]:
        """Generate alerts from pattern violation objects.

        Args:
            violations: List of PatternViolation objects.

        Returns:
            List of generated Alert objects.
        """
        alerts: list[Alert] = []
        severity_score_map = {
            "critical": 0.95,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
        }

        for violation in violations:
            score = severity_score_map.get(violation.severity, 0.5)
            alert = self.generate_alert(
                score=score,
                category=violation.pattern_type,
                message=violation.description,
                details=violation.details,
            )
            if alert is not None:
                alerts.append(alert)

        return alerts

    def get_alerts(
        self,
        severity_filter: Severity | None = None,
        category_filter: str | None = None,
        unacknowledged_only: bool = False,
    ) -> list[Alert]:
        """Retrieve alerts with optional filtering.

        Args:
            severity_filter: Only return alerts of this severity.
            category_filter: Only return alerts in this category.
            unacknowledged_only: Only return unacknowledged alerts.

        Returns:
            Filtered list of Alert objects.
        """
        alerts = self._active_alerts

        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]

        if category_filter:
            alerts = [a for a in alerts if a.category == category_filter]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: ID of the alert to acknowledge.

        Returns:
            True if alert was found and acknowledged, False otherwise.
        """
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("alert_acknowledged", alert_id=alert_id)
                return True
        return False

    def get_alert_summary(self) -> dict[str, Any]:
        """Get a summary of current alert status.

        Returns:
            Dictionary with alert counts by severity and category.
        """
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}

        for alert in self._active_alerts:
            if not alert.acknowledged:
                by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
                by_category[alert.category] = by_category.get(alert.category, 0) + 1

        return {
            "total_active": self.active_alert_count,
            "by_severity": by_severity,
            "by_category": by_category,
        }

    def clear_alerts(self) -> None:
        """Clear all alerts and reset deduplication state."""
        self._active_alerts.clear()
        self._alert_fingerprints.clear()
        logger.info("alerts_cleared")

    def _classify_severity(self, score: float) -> Severity | None:
        """Classify anomaly score into a severity level.

        Returns None if score is below all thresholds.
        """
        if score >= self.critical_threshold:
            return Severity.CRITICAL
        elif score >= self.high_threshold:
            return Severity.HIGH
        elif score >= self.medium_threshold:
            return Severity.MEDIUM
        return None
