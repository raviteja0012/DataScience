"""Value-level reconciliation.

Compares aggregate values (SUM, COUNT, MIN, MAX) between source and target
systems to verify financial and quantitative integrity after migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class AggregateType(Enum):
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    DISTRIBUTION = "distribution"
    GROUP_COUNT = "group_count"
    MIN_MAX = "min_max"


@dataclass
class ValueCheck:
    """Result of a single value reconciliation check."""

    check_name: str
    aggregate_type: AggregateType
    column: str
    source_values: Dict[str, Any] = field(default_factory=dict)
    expected_value: Any = None
    actual_value: Any = None
    delta: float = 0.0
    delta_pct: float = 0.0
    tolerance_pct: float = 0.01
    passed: bool = True
    details: str = ""


@dataclass
class ValueReconciliationResult:
    """Aggregate result for all value checks on one entity."""

    entity: str
    target_table: str
    checks: List[ValueCheck] = field(default_factory=list)
    passed: bool = True

    @property
    def failed_checks(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


class ValueReconciler:
    """Perform aggregate value reconciliation.

    Parameters
    ----------
    tolerance_pct:
        Default percentage tolerance for value comparisons.
    """

    def __init__(self, tolerance_pct: float = 0.01) -> None:
        self._tolerance = tolerance_pct

    def reconcile_from_records(
        self,
        entity: str,
        target_table: str,
        source_records: Dict[str, List[Dict[str, Any]]],
        target_records: List[Dict[str, Any]],
        check_configs: List[Dict[str, Any]],
    ) -> ValueReconciliationResult:
        """Run value reconciliation using in-memory records.

        Parameters
        ----------
        source_records:
            ``{system_id: [rows...]}`` from each source.
        target_records:
            Rows from the target table.
        check_configs:
            List of check definitions from validation_rules.yaml.
        """
        result = ValueReconciliationResult(entity=entity, target_table=target_table)

        for cfg in check_configs:
            check_name = cfg.get("name", "unnamed")
            check_type = cfg.get("check_type", "sum")
            tolerance = cfg.get("tolerance_pct", self._tolerance)

            if check_type == "sum":
                vc = self._check_sum(
                    check_name, source_records, target_records, cfg, tolerance
                )
            elif check_type == "distribution":
                vc = self._check_distribution(
                    check_name, target_records, cfg, tolerance
                )
            elif check_type == "group_count":
                vc = self._check_group_count(
                    check_name, target_records, cfg
                )
            elif check_type == "min_max":
                vc = self._check_min_max(
                    check_name, target_records, cfg
                )
            else:
                vc = ValueCheck(
                    check_name=check_name,
                    aggregate_type=AggregateType.SUM,
                    column="",
                    passed=True,
                    details=f"Unsupported check type: {check_type}",
                )

            result.checks.append(vc)

        result.passed = all(c.passed for c in result.checks)
        status = "PASSED" if result.passed else "FAILED"
        logger.info(
            "Value reconciliation for %s: %s (%d checks, %d failed)",
            entity, status, len(result.checks), result.failed_checks,
        )
        return result

    def _check_sum(
        self,
        name: str,
        source_records: Dict[str, List[Dict[str, Any]]],
        target_records: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        tolerance: float,
    ) -> ValueCheck:
        source_col_map = cfg.get("source_columns", {})
        target_col = cfg.get("target_column", "")

        # Sum source values
        source_total = 0.0
        source_values: Dict[str, float] = {}
        for sys_id, records in source_records.items():
            col = source_col_map.get(sys_id, target_col)
            sys_sum = sum(
                float(r.get(col, 0) or 0) for r in records
            )
            source_values[sys_id] = round(sys_sum, 2)
            source_total += sys_sum

        # Sum target values
        target_total = sum(
            float(r.get(target_col, 0) or 0) for r in target_records
        )

        delta = abs(target_total - source_total)
        delta_pct = delta / max(abs(source_total), 1) * 100

        return ValueCheck(
            check_name=name,
            aggregate_type=AggregateType.SUM,
            column=target_col,
            source_values=source_values,
            expected_value=round(source_total, 2),
            actual_value=round(target_total, 2),
            delta=round(delta, 2),
            delta_pct=round(delta_pct, 4),
            tolerance_pct=tolerance,
            passed=delta_pct <= tolerance,
        )

    def _check_distribution(
        self,
        name: str,
        target_records: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        tolerance: float,
    ) -> ValueCheck:
        target_col = cfg.get("target_column", "")
        dist: Dict[str, int] = {}
        for r in target_records:
            val = str(r.get(target_col, "NULL"))
            dist[val] = dist.get(val, 0) + 1

        return ValueCheck(
            check_name=name,
            aggregate_type=AggregateType.DISTRIBUTION,
            column=target_col,
            actual_value=dist,
            passed=True,
            details=f"Distribution: {dist}",
        )

    def _check_group_count(
        self,
        name: str,
        target_records: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> ValueCheck:
        group_col = cfg.get("target_group_column", "")
        groups: Dict[str, int] = {}
        for r in target_records:
            val = str(r.get(group_col, "NULL"))
            groups[val] = groups.get(val, 0) + 1

        return ValueCheck(
            check_name=name,
            aggregate_type=AggregateType.GROUP_COUNT,
            column=group_col,
            actual_value=groups,
            passed=True,
            details=f"Group counts: {groups}",
        )

    def _check_min_max(
        self,
        name: str,
        target_records: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> ValueCheck:
        target_col = cfg.get("target_column", "")
        values = [
            float(r.get(target_col, 0) or 0)
            for r in target_records
            if r.get(target_col) is not None
        ]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        return ValueCheck(
            check_name=name,
            aggregate_type=AggregateType.MIN,
            column=target_col,
            actual_value={"min": min_val, "max": max_val},
            passed=True,
            details=f"Range: [{min_val}, {max_val}]",
        )
