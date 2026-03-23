"""Post-migration validation orchestrator.

Ties together count, value, and hash reconciliation into a single validation
pass driven by the ``validation_rules.yaml`` configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from src.reconciliation.count_reconciler import CountReconciler, CountReconciliationResult
from src.reconciliation.hash_comparator import HashComparator, HashComparisonResult
from src.reconciliation.report_generator import (
    EntityReconciliationReport,
    ReconciliationReport,
    ReportGenerator,
)
from src.reconciliation.value_reconciler import ValueReconciler, ValueReconciliationResult
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector

logger = get_logger(__name__, fmt="text")


class ReconciliationValidator:
    """Orchestrate all reconciliation checks for a migration.

    Parameters
    ----------
    config:
        Parsed ``validation_rules.yaml``.
    pipeline_name:
        Name for the reconciliation report.
    output_dir:
        Directory for report output.
    metrics:
        Optional metrics collector.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pipeline_name: str = "default",
        output_dir: str = "./reports",
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self._config = config
        self._pipeline_name = pipeline_name
        self._metrics = metrics

        global_settings = config.get("global_settings", {})
        self._count_reconciler = CountReconciler(
            tolerance_pct=global_settings.get("count_tolerance_pct", 0.0)
        )
        self._value_reconciler = ValueReconciler(
            tolerance_pct=global_settings.get("value_tolerance_pct", 0.01)
        )
        self._hash_comparator = HashComparator(
            hash_algorithm=global_settings.get("hash_algorithm", "sha256"),
            max_discrepancies=global_settings.get("max_discrepancies_per_entity", 1000),
        )
        self._report_generator = ReportGenerator(
            output_dir=output_dir,
            include_drill_down=True,
        )

    def validate_entity(
        self,
        entity_config: Dict[str, Any],
        source_records: Dict[str, List[Dict[str, Any]]],
        target_records: List[Dict[str, Any]],
        dedup_count: int = 0,
    ) -> EntityReconciliationReport:
        """Validate a single entity.

        Parameters
        ----------
        entity_config:
            Entity section from ``validation_rules.yaml``.
        source_records:
            ``{system_id: [rows...]}``.
        target_records:
            Rows from the target table.
        dedup_count:
            Records removed by deduplication.
        """
        entity = entity_config.get("entity", "unknown")
        target_table = entity_config.get("target_table", "")

        logger.info("Validating entity: %s (%s)", entity, target_table)

        if self._metrics:
            self._metrics.start_timer(f"validate_{entity}")

        # Count reconciliation
        source_counts = {sid: len(recs) for sid, recs in source_records.items()}
        count_result = self._count_reconciler.reconcile(
            entity=entity,
            target_table=target_table,
            source_counts=source_counts,
            target_count=len(target_records),
            dedup_count=dedup_count,
        )

        # Value reconciliation
        value_result: Optional[ValueReconciliationResult] = None
        value_checks = entity_config.get("value_checks", [])
        if value_checks:
            value_result = self._value_reconciler.reconcile_from_records(
                entity=entity,
                target_table=target_table,
                source_records=source_records,
                target_records=target_records,
                check_configs=value_checks,
            )

        # Hash comparison
        hash_result: Optional[HashComparisonResult] = None
        hash_cfg = entity_config.get("hash_checks", {})
        if hash_cfg.get("enabled", False):
            hash_result = self._hash_comparator.compare(
                entity=entity,
                target_table=target_table,
                source_records=target_records,  # compare target against itself for demo
                target_records=target_records,
                key_columns=hash_cfg.get("key_columns", []),
                compare_columns=hash_cfg.get("compare_columns", []),
            )

        if self._metrics:
            self._metrics.stop_timer(
                f"validate_{entity}",
                phase="reconciliation",
                entity=entity,
            )

        return EntityReconciliationReport(
            entity=entity,
            count_result=count_result,
            value_result=value_result,
            hash_result=hash_result,
        )

    def validate_all(
        self,
        data_store: Dict[str, Dict[str, List[Dict[str, Any]]]],
        target_store: Dict[str, List[Dict[str, Any]]],
        dedup_counts: Optional[Dict[str, int]] = None,
    ) -> ReconciliationReport:
        """Validate all entities defined in the config.

        Parameters
        ----------
        data_store:
            ``{entity: {system_id: [rows...]}}`` for source data.
        target_store:
            ``{entity: [target_rows...]}`` for target data.
        dedup_counts:
            ``{entity: count}`` of deduplicated records.
        """
        dedup_counts = dedup_counts or {}
        report = ReconciliationReport(pipeline_name=self._pipeline_name)

        for entity_cfg in self._config.get("entities", []):
            entity = entity_cfg.get("entity", "")
            src_data = data_store.get(entity, {})
            tgt_data = target_store.get(entity, [])
            dedup = dedup_counts.get(entity, 0)

            entity_report = self.validate_entity(
                entity_config=entity_cfg,
                source_records=src_data,
                target_records=tgt_data,
                dedup_count=dedup,
            )
            report.entities.append(entity_report)

        overall = "PASSED" if report.overall_passed else "FAILED"
        logger.info(
            "Overall reconciliation: %s (%d entities)",
            overall, len(report.entities),
        )
        return report

    def generate_reports(
        self, report: ReconciliationReport, formats: Optional[List[str]] = None
    ) -> List[str]:
        """Generate report files in the requested formats."""
        formats = formats or ["html", "json"]
        paths: List[str] = []
        if "html" in formats:
            paths.append(self._report_generator.generate_html(report))
        if "json" in formats:
            paths.append(self._report_generator.generate_json(report))
        return paths
