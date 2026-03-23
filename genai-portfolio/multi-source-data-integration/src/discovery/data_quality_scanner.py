"""Data quality assessment engine.

Scores each table and column across four quality dimensions:
completeness, uniqueness, validity, and consistency.  The composite
quality score drives go/no-go decisions for migration phases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.discovery.source_profiler import ColumnProfile, TableProfile
from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class QualityDimension:
    """Score for a single quality dimension (0.0 – 1.0)."""

    name: str
    score: float
    details: str = ""


@dataclass
class ColumnQuality:
    name: str
    dimensions: List[QualityDimension] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        if not self.dimensions:
            return 0.0
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions), 4)


@dataclass
class TableQuality:
    table_name: str
    source_system: str
    row_count: int = 0
    columns: List[ColumnQuality] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        if not self.columns:
            return 0.0
        return round(sum(c.composite_score for c in self.columns) / len(self.columns), 4)


@dataclass
class QualityReport:
    source_system: str
    tables: List[TableQuality] = field(default_factory=list)
    overall_score: float = 0.0
    threshold: float = 0.70
    passed: bool = True

    def compute_overall(self) -> None:
        if self.tables:
            self.overall_score = round(
                sum(t.composite_score for t in self.tables) / len(self.tables), 4
            )
        self.passed = self.overall_score >= self.threshold


# ---------------------------------------------------------------------------
# Validation patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_PHONE_RE = re.compile(r"^[\d\s\-\(\)\+\.]{7,20}$")
_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")


class DataQualityScanner:
    """Score data quality from profiled table metadata or raw records."""

    def __init__(self, threshold: float = 0.70) -> None:
        self._threshold = threshold

    # -- Dimension scorers ---------------------------------------------------

    def _completeness(self, col: ColumnProfile) -> QualityDimension:
        """Ratio of non-null values."""
        non_null_pct = 1.0 - (col.null_count / max(col.total_count, 1))
        return QualityDimension(
            name="completeness",
            score=round(non_null_pct, 4),
            details=f"{col.null_pct}% null",
        )

    def _uniqueness(self, col: ColumnProfile) -> QualityDimension:
        """Ratio of distinct values to total non-null values."""
        non_null = col.total_count - col.null_count
        if non_null == 0:
            return QualityDimension(name="uniqueness", score=0.0, details="all null")
        ratio = col.distinct_count / non_null
        return QualityDimension(
            name="uniqueness",
            score=round(min(ratio, 1.0), 4),
            details=f"{col.distinct_count} distinct / {non_null} non-null",
        )

    def _validity(self, col: ColumnProfile, values: Sequence[Any] = ()) -> QualityDimension:
        """Pattern-based validity check for known semantic types."""
        if not values:
            # Fall back to a neutral score when raw values are unavailable
            return QualityDimension(name="validity", score=0.90, details="no raw values")

        name_lower = col.name.lower()
        pattern = None
        if "email" in name_lower:
            pattern = _EMAIL_RE
        elif "phone" in name_lower:
            pattern = _PHONE_RE
        elif "zip" in name_lower or "postal" in name_lower:
            pattern = _ZIP_RE

        if pattern is None:
            return QualityDimension(name="validity", score=1.0, details="no pattern check")

        non_null = [v for v in values if v is not None]
        if not non_null:
            return QualityDimension(name="validity", score=0.0, details="all null")

        valid = sum(1 for v in non_null if pattern.match(str(v)))
        score = valid / len(non_null)
        return QualityDimension(
            name="validity",
            score=round(score, 4),
            details=f"{valid}/{len(non_null)} match pattern",
        )

    def _consistency(self, col: ColumnProfile, values: Sequence[Any] = ()) -> QualityDimension:
        """Check formatting consistency (e.g. consistent casing, length patterns)."""
        non_null_strs = [str(v) for v in values if v is not None] if values else []
        if not non_null_strs:
            return QualityDimension(name="consistency", score=0.90, details="no raw values")

        # Length consistency: coefficient of variation of string lengths
        lengths = [len(s) for s in non_null_strs]
        if len(set(lengths)) <= 1:
            return QualityDimension(name="consistency", score=1.0, details="uniform length")

        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0:
            return QualityDimension(name="consistency", score=1.0)

        std_len = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
        cv = std_len / mean_len
        # A CV < 0.3 is considered consistent; above 1.0 is poor
        score = max(0.0, 1.0 - cv)
        return QualityDimension(
            name="consistency",
            score=round(score, 4),
            details=f"length CV={cv:.2f}",
        )

    # -- Table / source scanning ---------------------------------------------

    def scan_table_profile(
        self,
        table_profile: TableProfile,
        raw_data: Optional[List[Dict[str, Any]]] = None,
    ) -> TableQuality:
        """Score quality from a ``TableProfile`` and optional raw data."""
        col_map: Dict[str, List[Any]] = {}
        if raw_data:
            for row in raw_data:
                for col, val in row.items():
                    col_map.setdefault(col, []).append(val)

        col_qualities: List[ColumnQuality] = []
        for cp in table_profile.columns:
            values = col_map.get(cp.name, [])
            dims = [
                self._completeness(cp),
                self._uniqueness(cp),
                self._validity(cp, values),
                self._consistency(cp, values),
            ]
            col_qualities.append(ColumnQuality(name=cp.name, dimensions=dims))

        tq = TableQuality(
            table_name=table_profile.table_name,
            source_system=table_profile.source_system,
            row_count=table_profile.row_count,
            columns=col_qualities,
        )
        logger.info(
            "Quality scan %s.%s — composite score %.2f",
            table_profile.source_system, table_profile.table_name,
            tq.composite_score,
        )
        return tq

    def scan_source(
        self,
        table_profiles: List[TableProfile],
        raw_tables: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> QualityReport:
        """Scan all tables for a source system and produce a ``QualityReport``."""
        raw_tables = raw_tables or {}
        system_id = table_profiles[0].source_system if table_profiles else "unknown"

        table_qualities: List[TableQuality] = []
        for tp in table_profiles:
            raw = raw_tables.get(tp.table_name)
            tq = self.scan_table_profile(tp, raw_data=raw)
            table_qualities.append(tq)

        report = QualityReport(
            source_system=system_id,
            tables=table_qualities,
            threshold=self._threshold,
        )
        report.compute_overall()
        logger.info(
            "Quality report for %s: overall=%.4f, passed=%s",
            system_id, report.overall_score, report.passed,
        )
        return report
