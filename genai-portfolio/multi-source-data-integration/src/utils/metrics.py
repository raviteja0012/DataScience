"""Pipeline metrics collection and export.

Tracks row counts, durations, error rates, and other KPIs across every
pipeline phase so that operators can monitor migration health at a glance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    TIMER = "timer"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A single recorded metric data-point."""

    name: str
    value: float
    metric_type: MetricType
    phase: str = ""
    entity: str = ""
    source_system: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MetricsCollector:
    """Accumulates ``Metric`` instances throughout a pipeline run and exports
    them as JSON for downstream consumption by dashboards or alerting systems.
    """

    def __init__(self) -> None:
        self._metrics: List[Metric] = []
        self._timers: Dict[str, float] = {}

    # -- Recording helpers ----------------------------------------------------

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        phase: str = "",
        entity: str = "",
        source_system: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self._metrics.append(
            Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                phase=phase,
                entity=entity,
                source_system=source_system,
                tags=tags or {},
            )
        )

    def increment(
        self, name: str, delta: float = 1.0, **kwargs: Any
    ) -> None:
        self.record(name, delta, MetricType.COUNTER, **kwargs)

    def start_timer(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str, **kwargs: Any) -> float:
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        elapsed = round(time.perf_counter() - start, 4)
        self.record(name, elapsed, MetricType.TIMER, **kwargs)
        return elapsed

    # -- Query helpers --------------------------------------------------------

    def get_metrics(
        self,
        phase: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> List[Metric]:
        result = self._metrics
        if phase:
            result = [m for m in result if m.phase == phase]
        if entity:
            result = [m for m in result if m.entity == entity]
        return result

    def summary(self) -> Dict[str, Any]:
        """Return a top-level summary dict suitable for the final report."""
        total_duration = sum(
            m.value for m in self._metrics if m.metric_type == MetricType.TIMER
        )
        error_count = sum(
            m.value for m in self._metrics if "error" in m.name.lower()
        )
        row_metrics = [
            m for m in self._metrics
            if m.metric_type == MetricType.GAUGE and "row" in m.name.lower()
        ]
        total_rows = sum(m.value for m in row_metrics)

        return {
            "total_metrics_recorded": len(self._metrics),
            "total_pipeline_duration_s": round(total_duration, 2),
            "total_rows_processed": int(total_rows),
            "total_errors": int(error_count),
            "phases": sorted({m.phase for m in self._metrics if m.phase}),
        }

    # -- Export ---------------------------------------------------------------

    def to_dict(self) -> List[Dict[str, Any]]:
        results = []
        for m in self._metrics:
            results.append({
                "name": m.name,
                "value": m.value,
                "type": m.metric_type.value,
                "phase": m.phase,
                "entity": m.entity,
                "source_system": m.source_system,
                "tags": m.tags,
                "timestamp": m.timestamp,
            })
        return results

    def export_json(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "metrics": self.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

    def reset(self) -> None:
        self._metrics.clear()
        self._timers.clear()
