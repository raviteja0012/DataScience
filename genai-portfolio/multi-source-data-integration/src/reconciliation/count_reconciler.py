"""Row count reconciliation.

Compares row counts between source systems (combined) and the target table,
accounting for expected deltas from deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class CountCheck:
    """Result of a single count-reconciliation check."""

    check_name: str
    source_counts: Dict[str, int] = field(default_factory=dict)
    expected_count: int = 0
    actual_count: int = 0
    delta: int = 0
    delta_pct: float = 0.0
    tolerance_pct: float = 0.0
    passed: bool = True
    dedup_delta: int = 0
    notes: str = ""


@dataclass
class CountReconciliationResult:
    """Aggregate result for all count checks on one entity."""

    entity: str
    target_table: str
    checks: List[CountCheck] = field(default_factory=list)
    passed: bool = True

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def failed_checks(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


class CountReconciler:
    """Perform row-count reconciliation between sources and target.

    Parameters
    ----------
    tolerance_pct:
        Default tolerance for count mismatches (0.0 = exact match).
    """

    def __init__(self, tolerance_pct: float = 0.0) -> None:
        self._tolerance = tolerance_pct

    def reconcile(
        self,
        entity: str,
        target_table: str,
        source_counts: Dict[str, int],
        target_count: int,
        dedup_count: int = 0,
        filter_checks: Optional[List[Dict[str, Any]]] = None,
        filtered_source_counts: Optional[Dict[str, Dict[str, int]]] = None,
        filtered_target_counts: Optional[Dict[str, int]] = None,
    ) -> CountReconciliationResult:
        """Run all count reconciliation checks.

        Parameters
        ----------
        source_counts:
            ``{system_id: row_count}`` for each source.
        target_count:
            Total rows in the target table.
        dedup_count:
            Number of records removed by deduplication.
        filter_checks:
            Additional checks with WHERE filters.
        filtered_source_counts:
            ``{check_name: {system_id: count}}`` for filtered checks.
        filtered_target_counts:
            ``{check_name: count}`` for filtered checks.
        """
        result = CountReconciliationResult(
            entity=entity, target_table=target_table
        )

        # Total row count check
        total_source = sum(source_counts.values())
        expected = total_source - dedup_count
        delta = target_count - expected
        delta_pct = abs(delta) / max(expected, 1) * 100

        total_check = CountCheck(
            check_name="total_row_count",
            source_counts=dict(source_counts),
            expected_count=expected,
            actual_count=target_count,
            delta=delta,
            delta_pct=round(delta_pct, 4),
            tolerance_pct=self._tolerance,
            passed=delta_pct <= self._tolerance,
            dedup_delta=dedup_count,
            notes=f"Sources: {total_source}, Deduped: {dedup_count}, Expected: {expected}",
        )
        result.checks.append(total_check)

        # Per-source count checks
        for sys_id, src_count in source_counts.items():
            # These are informational — we just record them
            result.checks.append(CountCheck(
                check_name=f"source_count_{sys_id}",
                source_counts={sys_id: src_count},
                expected_count=src_count,
                actual_count=src_count,
                passed=True,
                notes=f"Source {sys_id} contributed {src_count} rows",
            ))

        # Filtered checks (e.g. active customer count)
        if filter_checks and filtered_source_counts and filtered_target_counts:
            for fc in filter_checks:
                check_name = fc.get("name", "unnamed")
                tolerance = fc.get("tolerance_pct", self._tolerance)
                fsrc = filtered_source_counts.get(check_name, {})
                ftgt = filtered_target_counts.get(check_name, 0)
                fexpected = sum(fsrc.values())
                fdelta = ftgt - fexpected
                fdelta_pct = abs(fdelta) / max(fexpected, 1) * 100

                result.checks.append(CountCheck(
                    check_name=check_name,
                    source_counts=dict(fsrc),
                    expected_count=fexpected,
                    actual_count=ftgt,
                    delta=fdelta,
                    delta_pct=round(fdelta_pct, 4),
                    tolerance_pct=tolerance,
                    passed=fdelta_pct <= tolerance,
                ))

        result.passed = all(c.passed for c in result.checks)
        status = "PASSED" if result.passed else "FAILED"
        logger.info(
            "Count reconciliation for %s: %s (%d/%d checks passed)",
            entity, status,
            result.total_checks - result.failed_checks,
            result.total_checks,
        )
        return result
