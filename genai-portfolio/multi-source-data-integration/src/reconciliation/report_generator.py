"""Reconciliation report generation.

Produces HTML, JSON, or CSV reconciliation reports with drill-down into
discrepancies.  The HTML report includes summary dashboards, per-entity
breakdowns, and detailed discrepancy tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.reconciliation.count_reconciler import CountReconciliationResult
from src.reconciliation.hash_comparator import HashComparisonResult
from src.reconciliation.value_reconciler import ValueReconciliationResult
from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class EntityReconciliationReport:
    """All reconciliation results for one entity."""

    entity: str
    count_result: Optional[CountReconciliationResult] = None
    value_result: Optional[ValueReconciliationResult] = None
    hash_result: Optional[HashComparisonResult] = None

    @property
    def passed(self) -> bool:
        results = [
            self.count_result.passed if self.count_result else True,
            self.value_result.passed if self.value_result else True,
            self.hash_result.passed if self.hash_result else True,
        ]
        return all(results)


@dataclass
class ReconciliationReport:
    """Complete reconciliation report across all entities."""

    pipeline_name: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    entities: List[EntityReconciliationReport] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        return all(e.passed for e in self.entities)


class ReportGenerator:
    """Generate reconciliation reports in multiple formats.

    Parameters
    ----------
    output_dir:
        Directory for report files.
    include_drill_down:
        Whether to include row-level discrepancy details.
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        include_drill_down: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._drill_down = include_drill_down

    def generate_html(self, report: ReconciliationReport) -> str:
        """Generate an HTML reconciliation report."""
        html_parts: List[str] = []
        overall = "PASSED" if report.overall_passed else "FAILED"
        overall_color = "#28a745" if report.overall_passed else "#dc3545"

        html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reconciliation Report — {report.pipeline_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  h3 {{ color: #0f3460; }}
  .summary-card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; color: white; font-size: 0.9em; }}
  .status-pass {{ background: #28a745; }}
  .status-fail {{ background: #dc3545; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; background: white; border-radius: 4px; overflow: hidden; }}
  th {{ background: #16213e; color: white; padding: 10px 12px; text-align: left; font-size: 0.85em; text-transform: uppercase; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.9em; }}
  tr:hover {{ background: #f8f9fa; }}
  .metric {{ font-size: 2em; font-weight: bold; color: #16213e; }}
  .metric-label {{ font-size: 0.85em; color: #666; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .drill-down {{ margin: 10px 0; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 0 4px 4px 0; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">
<h1>Reconciliation Report</h1>
<div class="summary-card">
  <p><strong>Pipeline:</strong> {report.pipeline_name}</p>
  <p><strong>Generated:</strong> {report.generated_at}</p>
  <p><strong>Overall Status:</strong> <span class="status" style="background:{overall_color}">{overall}</span></p>
  <p><strong>Entities Validated:</strong> {len(report.entities)}</p>
</div>
""")

        # Summary metrics
        total_checks = 0
        passed_checks = 0
        for entity_report in report.entities:
            if entity_report.count_result:
                total_checks += entity_report.count_result.total_checks
                passed_checks += (entity_report.count_result.total_checks -
                                  entity_report.count_result.failed_checks)
            if entity_report.value_result:
                total_checks += len(entity_report.value_result.checks)
                passed_checks += len(entity_report.value_result.checks) - entity_report.value_result.failed_checks

        html_parts.append(f"""
<div class="metrics-grid">
  <div class="metric-card"><div class="metric">{len(report.entities)}</div><div class="metric-label">Entities</div></div>
  <div class="metric-card"><div class="metric">{total_checks}</div><div class="metric-label">Total Checks</div></div>
  <div class="metric-card"><div class="metric">{passed_checks}</div><div class="metric-label">Checks Passed</div></div>
  <div class="metric-card"><div class="metric">{total_checks - passed_checks}</div><div class="metric-label">Checks Failed</div></div>
</div>
""")

        # Per-entity details
        for entity_report in report.entities:
            entity_status = "PASSED" if entity_report.passed else "FAILED"
            entity_color = "#28a745" if entity_report.passed else "#dc3545"
            html_parts.append(f"""
<h2>{entity_report.entity} <span class="status" style="background:{entity_color}">{entity_status}</span></h2>
""")

            # Count reconciliation
            if entity_report.count_result:
                html_parts.append(self._render_count_html(entity_report.count_result))

            # Value reconciliation
            if entity_report.value_result:
                html_parts.append(self._render_value_html(entity_report.value_result))

            # Hash comparison
            if entity_report.hash_result:
                html_parts.append(self._render_hash_html(entity_report.hash_result))

        html_parts.append("""
<div class="footer">
  <p>Multi-Source Data Integration Framework — Reconciliation Engine</p>
</div>
</div>
</body>
</html>""")

        html_content = "\n".join(html_parts)
        output_path = self._output_dir / f"reconciliation_{report.pipeline_name}.html"
        output_path.write_text(html_content)
        logger.info("HTML report generated: %s", output_path)
        return str(output_path)

    def _render_count_html(self, cr: CountReconciliationResult) -> str:
        rows = ""
        for c in cr.checks:
            status_cls = "status-pass" if c.passed else "status-fail"
            status_txt = "PASS" if c.passed else "FAIL"
            rows += f"""<tr>
  <td>{c.check_name}</td>
  <td>{c.expected_count:,}</td>
  <td>{c.actual_count:,}</td>
  <td>{c.delta:+,}</td>
  <td>{c.delta_pct:.2f}%</td>
  <td><span class="status {status_cls}">{status_txt}</span></td>
  <td>{c.notes}</td>
</tr>"""
        return f"""<h3>Row Count Reconciliation</h3>
<table>
<tr><th>Check</th><th>Expected</th><th>Actual</th><th>Delta</th><th>Delta %</th><th>Status</th><th>Notes</th></tr>
{rows}
</table>"""

    def _render_value_html(self, vr: ValueReconciliationResult) -> str:
        rows = ""
        for c in vr.checks:
            status_cls = "status-pass" if c.passed else "status-fail"
            status_txt = "PASS" if c.passed else "FAIL"
            expected = c.expected_value
            actual = c.actual_value
            if isinstance(expected, float):
                expected = f"{expected:,.2f}"
            if isinstance(actual, float):
                actual = f"{actual:,.2f}"
            if isinstance(actual, dict):
                actual = json.dumps(actual, default=str)[:100]
            rows += f"""<tr>
  <td>{c.check_name}</td>
  <td>{c.aggregate_type.value}</td>
  <td>{c.column}</td>
  <td>{expected}</td>
  <td>{actual}</td>
  <td><span class="status {status_cls}">{status_txt}</span></td>
</tr>"""
        return f"""<h3>Value Reconciliation</h3>
<table>
<tr><th>Check</th><th>Type</th><th>Column</th><th>Expected</th><th>Actual</th><th>Status</th></tr>
{rows}
</table>"""

    def _render_hash_html(self, hr: HashComparisonResult) -> str:
        status_cls = "status-pass" if hr.passed else "status-fail"
        status_txt = "PASS" if hr.passed else "FAIL"
        html = f"""<h3>Hash-Based Row Comparison <span class="status {status_cls}">{status_txt}</span></h3>
<div class="metrics-grid">
  <div class="metric-card"><div class="metric">{hr.total_source_rows:,}</div><div class="metric-label">Source Rows</div></div>
  <div class="metric-card"><div class="metric">{hr.total_target_rows:,}</div><div class="metric-label">Target Rows</div></div>
  <div class="metric-card"><div class="metric">{hr.matching_rows:,}</div><div class="metric-label">Matching</div></div>
  <div class="metric-card"><div class="metric">{hr.match_pct}%</div><div class="metric-label">Match Rate</div></div>
</div>"""

        if self._drill_down and hr.discrepancies:
            disc_rows = ""
            for d in hr.discrepancies[:50]:  # Limit drill-down
                disc_rows += f"""<tr>
  <td>{d.key}</td>
  <td>{d.discrepancy_type}</td>
  <td>{', '.join(d.differing_columns) if d.differing_columns else '-'}</td>
  <td>{json.dumps(d.source_values, default=str)[:80] if d.source_values else '-'}</td>
  <td>{json.dumps(d.target_values, default=str)[:80] if d.target_values else '-'}</td>
</tr>"""
            html += f"""
<h4>Discrepancy Detail (top {min(50, len(hr.discrepancies))} of {len(hr.discrepancies)})</h4>
<table>
<tr><th>Key</th><th>Type</th><th>Differing Columns</th><th>Source Values</th><th>Target Values</th></tr>
{disc_rows}
</table>"""

        return html

    def generate_json(self, report: ReconciliationReport) -> str:
        """Generate a JSON reconciliation report."""
        data: Dict[str, Any] = {
            "pipeline_name": report.pipeline_name,
            "generated_at": report.generated_at,
            "overall_passed": report.overall_passed,
            "entities": [],
        }

        for er in report.entities:
            entity_data: Dict[str, Any] = {
                "entity": er.entity,
                "passed": er.passed,
            }
            if er.count_result:
                entity_data["count_reconciliation"] = {
                    "passed": er.count_result.passed,
                    "checks": [
                        {
                            "name": c.check_name,
                            "expected": c.expected_count,
                            "actual": c.actual_count,
                            "delta": c.delta,
                            "passed": c.passed,
                        }
                        for c in er.count_result.checks
                    ],
                }
            if er.value_result:
                entity_data["value_reconciliation"] = {
                    "passed": er.value_result.passed,
                    "checks": [
                        {
                            "name": c.check_name,
                            "type": c.aggregate_type.value,
                            "column": c.column,
                            "passed": c.passed,
                        }
                        for c in er.value_result.checks
                    ],
                }
            if er.hash_result:
                entity_data["hash_comparison"] = {
                    "passed": er.hash_result.passed,
                    "matching_rows": er.hash_result.matching_rows,
                    "missing_in_target": er.hash_result.missing_in_target,
                    "missing_in_source": er.hash_result.missing_in_source,
                    "value_mismatches": er.hash_result.value_mismatches,
                    "match_pct": er.hash_result.match_pct,
                }
            data["entities"].append(entity_data)

        output_path = self._output_dir / f"reconciliation_{report.pipeline_name}.json"
        output_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("JSON report generated: %s", output_path)
        return str(output_path)
