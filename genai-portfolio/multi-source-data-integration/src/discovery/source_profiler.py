"""Source system profiling engine.

Generates a statistical profile for every table in a source system, including
row counts, column types, null percentages, cardinality, and value
distributions.  When run against synthetic data (via dicts) it operates
entirely in-memory.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    data_type: str
    total_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    sample_values: List[Any] = field(default_factory=list)

    @property
    def null_pct(self) -> float:
        return round(self.null_count / max(self.total_count, 1) * 100, 2)

    @property
    def uniqueness(self) -> float:
        non_null = self.total_count - self.null_count
        return round(self.distinct_count / max(non_null, 1) * 100, 2)


@dataclass
class TableProfile:
    """Aggregate profile for one table."""

    table_name: str
    schema_name: str
    source_system: str
    row_count: int = 0
    column_count: int = 0
    columns: List[ColumnProfile] = field(default_factory=list)
    primary_key: Optional[str] = None
    profiled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class SourceProfile:
    """Complete profile for a source system."""

    system_id: str
    system_name: str
    platform: str
    tables: List[TableProfile] = field(default_factory=list)
    total_row_count: int = 0
    total_table_count: int = 0
    profiled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SourceProfiler:
    """Profile a source system from either live database metadata or in-memory
    record dictionaries.

    Parameters
    ----------
    system_config:
        Parsed YAML dict for the source system.
    """

    def __init__(self, system_config: Dict[str, Any]) -> None:
        self._config = system_config
        self._system = system_config.get("system", {})

    def _infer_type(self, value: Any) -> str:
        if value is None:
            return "UNKNOWN"
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "NUMERIC"
        return "VARCHAR"

    def _profile_column(self, col_name: str, values: List[Any]) -> ColumnProfile:
        total = len(values)
        nulls = sum(1 for v in values if v is None)
        non_null = [v for v in values if v is not None]
        distinct = len(set(non_null))
        data_type = self._infer_type(non_null[0]) if non_null else "UNKNOWN"

        profile = ColumnProfile(
            name=col_name,
            data_type=data_type,
            total_count=total,
            null_count=nulls,
            distinct_count=distinct,
            sample_values=non_null[:5],
        )

        # Numeric stats
        numeric = [v for v in non_null if isinstance(v, (int, float))]
        if numeric:
            profile.min_value = min(numeric)
            profile.max_value = max(numeric)
            profile.mean_value = round(statistics.mean(numeric), 4)
            profile.median_value = round(statistics.median(numeric), 4)

        # String-length stats
        strings = [v for v in non_null if isinstance(v, str)]
        if strings:
            lengths = [len(s) for s in strings]
            profile.min_length = min(lengths)
            profile.max_length = max(lengths)
            profile.avg_length = round(statistics.mean(lengths), 2)
            if not numeric:
                profile.min_value = min(strings)
                profile.max_value = max(strings)

        return profile

    def profile_records(
        self,
        table_name: str,
        records: Sequence[Dict[str, Any]],
        schema_name: str = "",
    ) -> TableProfile:
        """Profile a table represented as a sequence of row dictionaries."""
        if not records:
            return TableProfile(
                table_name=table_name,
                schema_name=schema_name,
                source_system=self._system.get("id", "unknown"),
            )

        columns: Dict[str, List[Any]] = {}
        for row in records:
            for col, val in row.items():
                columns.setdefault(col, []).append(val)

        col_profiles = [
            self._profile_column(col_name, col_values)
            for col_name, col_values in columns.items()
        ]

        tp = TableProfile(
            table_name=table_name,
            schema_name=schema_name,
            source_system=self._system.get("id", "unknown"),
            row_count=len(records),
            column_count=len(col_profiles),
            columns=col_profiles,
        )
        logger.info(
            "Profiled %s.%s — %d rows, %d columns",
            schema_name, table_name, tp.row_count, tp.column_count,
        )
        return tp

    def profile_from_config(self) -> SourceProfile:
        """Build a profile purely from the YAML metadata (no DB connection).

        Useful for planning phases where connectivity is not yet available.
        """
        tables: List[TableProfile] = []
        total_rows = 0
        for schema in self._config.get("schemas", []):
            schema_name = schema.get("name", "")
            for tbl in schema.get("tables", []):
                cols = [
                    ColumnProfile(
                        name=c["name"],
                        data_type=c["type"],
                        total_count=tbl.get("estimated_rows", 0),
                        null_count=0 if not c.get("nullable", True)
                        else int(tbl.get("estimated_rows", 0) * 0.05),
                    )
                    for c in tbl.get("columns", [])
                ]
                tp = TableProfile(
                    table_name=tbl["name"],
                    schema_name=schema_name,
                    source_system=self._system.get("id", ""),
                    row_count=tbl.get("estimated_rows", 0),
                    column_count=len(cols),
                    columns=cols,
                    primary_key=tbl.get("primary_key"),
                )
                tables.append(tp)
                total_rows += tp.row_count

        sp = SourceProfile(
            system_id=self._system.get("id", ""),
            system_name=self._system.get("name", ""),
            platform=self._system.get("platform", ""),
            tables=tables,
            total_row_count=total_rows,
            total_table_count=len(tables),
        )
        logger.info(
            "Profiled source %s from config: %d tables, %d estimated rows",
            sp.system_id, sp.total_table_count, sp.total_row_count,
        )
        return sp

    def profile_from_records(
        self, tables: Dict[str, List[Dict[str, Any]]]
    ) -> SourceProfile:
        """Profile multiple tables from in-memory record dicts."""
        profiles: List[TableProfile] = []
        total = 0
        for table_name, records in tables.items():
            tp = self.profile_records(table_name, records)
            profiles.append(tp)
            total += tp.row_count

        return SourceProfile(
            system_id=self._system.get("id", ""),
            system_name=self._system.get("name", ""),
            platform=self._system.get("platform", ""),
            tables=profiles,
            total_row_count=total,
            total_table_count=len(profiles),
        )
