"""Pipeline orchestrator and CLI entry-point.

Coordinates the full integration pipeline: discovery, mapping, identity
resolution, cutover, and reconciliation.  Can be run end-to-end or as
individual phases via command-line arguments.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure the project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.cutover.checkpoint import CheckpointManager
from src.cutover.parallel_runner import ParallelRunner
from src.cutover.rollback_manager import RollbackManager
from src.cutover.strategy import CutoverStrategyPlanner
from src.discovery.data_quality_scanner import DataQualityScanner
from src.discovery.dependency_mapper import DependencyMapper
from src.discovery.schema_analyzer import SchemaAnalyzer
from src.discovery.source_profiler import SourceProfiler
from src.identity.resolver import IdentityResolver
from src.mapping.schema_mapper import SchemaMapper
from src.reconciliation.validator import ReconciliationValidator
from src.utils.generators import SyntheticDataGenerator
from src.utils.logger import get_logger, log_phase
from src.utils.metrics import MetricsCollector

logger = get_logger(__name__, fmt="text")


class PipelineOrchestrator:
    """Coordinates all phases of the multi-source integration pipeline.

    Parameters
    ----------
    config_dir:
        Path to the ``config/`` directory.
    output_dir:
        Path for reports, metrics, and logs.
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        output_dir: str = "./output",
    ) -> None:
        self._config_dir = Path(config_dir or str(_PROJECT_ROOT / "config"))
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = MetricsCollector()

        # Load main pipeline config
        self._pipeline_cfg = self._load_yaml("pipeline_config.yaml")
        self._pipeline_name = self._pipeline_cfg.get("pipeline", {}).get(
            "name", "unnamed"
        )

        # Load source system configs
        self._source_configs: Dict[str, Dict[str, Any]] = {}
        for src in self._pipeline_cfg.get("source_systems", []):
            cfg_file = src["config_file"]
            self._source_configs[src["id"]] = self._load_yaml(cfg_file)

        # Load target config
        target_ref = self._pipeline_cfg.get("target_system", {})
        self._target_config = self._load_yaml(target_ref.get("config_file", ""))

        # Synthetic data (populated on demand)
        self._synthetic_data: Optional[Dict[str, Any]] = None

    def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
        full_path = self._config_dir / relative_path
        if not full_path.exists():
            logger.warning("Config file not found: %s", full_path)
            return {}
        with open(full_path) as f:
            return yaml.safe_load(f) or {}

    def _ensure_synthetic_data(self) -> Dict[str, Any]:
        if self._synthetic_data is None:
            gen = SyntheticDataGenerator(
                num_customers=500,
                num_products=100,
                num_orders=1000,
                seed=42,
            )
            self._synthetic_data = gen.generate_all()
        return self._synthetic_data

    # -- Phase: Discovery ----------------------------------------------------

    def run_discovery(self) -> Dict[str, Any]:
        """Profile all source systems and assess data quality."""
        with log_phase(logger, "discovery"):
            self._metrics.start_timer("discovery")
            data = self._ensure_synthetic_data()
            results: Dict[str, Any] = {"profiles": {}, "quality": {}, "dependencies": {}}

            for sys_id, sys_cfg in self._source_configs.items():
                # Source profiling
                profiler = SourceProfiler(sys_cfg)
                tables: Dict[str, List[Dict[str, Any]]] = {}

                if sys_id == "legacy_oracle":
                    tables["CUST_MASTER"] = [c.to_dict() for c in data["oracle_customers"]]
                    tables["PRODUCT_CATALOG"] = [p.to_dict() for p in data["oracle_products"]]
                    tables["ORDER_HEADER"] = [o.to_dict() for o in data["oracle_orders"]]
                    tables["ORDER_DETAIL"] = [d.to_dict() for d in data["oracle_order_details"]]
                elif sys_id == "legacy_sqlserver":
                    tables["Customers"] = [c.to_dict() for c in data["sqlserver_customers"]]
                    tables["Products"] = [p.to_dict() for p in data["sqlserver_products"]]
                    tables["SalesOrders"] = [o.to_dict() for o in data["sqlserver_orders"]]
                    tables["SalesOrderDetails"] = [d.to_dict() for d in data["sqlserver_order_details"]]

                profile = profiler.profile_from_records(tables)
                results["profiles"][sys_id] = profile

                # Quality scanning
                scanner = DataQualityScanner(threshold=0.70)
                quality = scanner.scan_source(profile.tables, raw_tables=tables)
                results["quality"][sys_id] = quality

                # Dependency mapping
                dep_mapper = DependencyMapper()
                graph = dep_mapper.build_graph(sys_cfg)
                results["dependencies"][sys_id] = graph

                self._metrics.record(
                    f"discovery_{sys_id}_tables", profile.total_table_count,
                    phase="discovery", source_system=sys_id,
                )
                self._metrics.record(
                    f"discovery_{sys_id}_rows", profile.total_row_count,
                    phase="discovery", source_system=sys_id,
                )

            # Schema analysis
            analyzer = SchemaAnalyzer()
            comparisons = []
            for sys_id, sys_cfg in self._source_configs.items():
                comps = analyzer.analyze_source_config(sys_cfg, self._target_config)
                comparisons.extend(comps)
            results["schema_comparisons"] = comparisons

            self._metrics.stop_timer("discovery", phase="discovery")
            return results

    # -- Phase: Mapping ------------------------------------------------------

    def run_mapping(self) -> Dict[str, Any]:
        """Apply schema mappings to transform source data."""
        with log_phase(logger, "mapping"):
            self._metrics.start_timer("mapping")
            data = self._ensure_synthetic_data()
            mapper = SchemaMapper()

            # Load mapping files
            mapping_dir = self._config_dir / "mapping_rules"
            for entity_file in self._pipeline_cfg.get("mapping", {}).get("entities", []):
                mapper.load_mapping_file(str(mapping_dir / entity_file))

            results: Dict[str, List[Dict[str, Any]]] = {}

            # Map customers
            ora_custs = [c.to_dict() for c in data["oracle_customers"]]
            sql_custs = [c.to_dict() for c in data["sqlserver_customers"]]
            results["oracle_customers_mapped"] = mapper.apply("customer", "legacy_oracle", ora_custs)
            results["sqlserver_customers_mapped"] = mapper.apply("customer", "legacy_sqlserver", sql_custs)

            total_mapped = sum(len(v) for v in results.values())
            self._metrics.record("mapping_total_rows", total_mapped, phase="mapping")
            self._metrics.stop_timer("mapping", phase="mapping")
            return results

    # -- Phase: Identity Resolution ------------------------------------------

    def run_identity_resolution(self) -> Dict[str, Any]:
        """Resolve identities across source systems."""
        with log_phase(logger, "identity_resolution"):
            self._metrics.start_timer("identity_resolution")
            data = self._ensure_synthetic_data()
            ir_config = self._pipeline_cfg.get("identity_resolution", {})

            # Build source priority map
            priorities = {
                src["id"]: src.get("priority", 99)
                for src in self._pipeline_cfg.get("source_systems", [])
            }

            resolver = IdentityResolver(
                entity_type="customer",
                config=ir_config,
                source_priorities=priorities,
                metrics=self._metrics,
            )

            ora_custs = [c.to_dict() for c in data["oracle_customers"]]
            sql_custs = [c.to_dict() for c in data["sqlserver_customers"]]

            result = resolver.resolve({
                "legacy_oracle": ora_custs,
                "legacy_sqlserver": sql_custs,
            })

            self._metrics.stop_timer("identity_resolution", phase="identity_resolution")
            return {
                "resolution_result": result,
                "total_input": result.total_input_records,
                "total_resolved": result.total_resolved_entities,
                "duplicates_found": result.total_duplicates_found,
            }

    # -- Phase: Cutover ------------------------------------------------------

    def run_cutover(self) -> Dict[str, Any]:
        """Plan and execute the data cutover."""
        with log_phase(logger, "cutover"):
            self._metrics.start_timer("cutover")
            cutover_cfg = self._pipeline_cfg.get("cutover", {})

            planner = CutoverStrategyPlanner(cutover_cfg)
            plan = planner.build_plan()

            checkpoint_mgr = CheckpointManager(
                checkpoint_dir=cutover_cfg.get("checkpoint_dir", "/tmp/msdi_checkpoints"),
                pipeline_name=self._pipeline_name,
                enabled=cutover_cfg.get("checkpoint_enabled", True),
            )
            rollback_mgr = RollbackManager(
                snapshot_dir=cutover_cfg.get("rollback_snapshot_dir", "/tmp/msdi_rollback"),
                enabled=cutover_cfg.get("rollback_enabled", True),
            )

            runner = ParallelRunner(
                max_workers=cutover_cfg.get("max_parallel_loads", 4),
                checkpoint_mgr=checkpoint_mgr,
                rollback_mgr=rollback_mgr,
                metrics=self._metrics,
            )

            cutover_result = runner.execute(plan)
            self._metrics.stop_timer("cutover", phase="cutover")
            return {
                "cutover_result": cutover_result,
                "plan_summary": plan.summary(),
            }

    # -- Phase: Reconciliation -----------------------------------------------

    def run_reconciliation(self) -> Dict[str, Any]:
        """Run post-migration reconciliation validation."""
        with log_phase(logger, "reconciliation"):
            self._metrics.start_timer("reconciliation")
            data = self._ensure_synthetic_data()

            # Load validation rules
            val_rules = self._load_yaml("validation_rules.yaml")

            # Build source/target data stores for reconciliation
            source_store: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                "customer": {
                    "legacy_oracle": [c.to_dict() for c in data["oracle_customers"]],
                    "legacy_sqlserver": [c.to_dict() for c in data["sqlserver_customers"]],
                },
                "product": {
                    "legacy_oracle": [p.to_dict() for p in data["oracle_products"]],
                    "legacy_sqlserver": [p.to_dict() for p in data["sqlserver_products"]],
                },
            }

            # For demo: target = combined sources (post-mapping)
            target_store: Dict[str, List[Dict[str, Any]]] = {
                "customer": (
                    [c.to_dict() for c in data["oracle_customers"]]
                    + [c.to_dict() for c in data["sqlserver_customers"]]
                ),
                "product": (
                    [p.to_dict() for p in data["oracle_products"]]
                    + [p.to_dict() for p in data["sqlserver_products"]]
                ),
            }

            report_dir = str(self._output_dir / "reports")
            validator = ReconciliationValidator(
                config=val_rules,
                pipeline_name=self._pipeline_name,
                output_dir=report_dir,
                metrics=self._metrics,
            )

            report = validator.validate_all(
                data_store=source_store,
                target_store=target_store,
            )
            paths = validator.generate_reports(report, formats=["html", "json"])

            self._metrics.stop_timer("reconciliation", phase="reconciliation")
            return {
                "report": report,
                "report_files": paths,
                "overall_passed": report.overall_passed,
            }

    # -- Full pipeline -------------------------------------------------------

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute all pipeline phases in sequence."""
        run_id = uuid.uuid4().hex[:8]
        logger.info("=" * 70)
        logger.info("STARTING FULL PIPELINE RUN: %s (run_id=%s)", self._pipeline_name, run_id)
        logger.info("=" * 70)

        results: Dict[str, Any] = {"run_id": run_id}

        results["discovery"] = self.run_discovery()
        results["mapping"] = self.run_mapping()
        results["identity_resolution"] = self.run_identity_resolution()
        results["cutover"] = self.run_cutover()
        results["reconciliation"] = self.run_reconciliation()

        # Export metrics
        metrics_path = self._metrics.export_json(
            str(self._output_dir / "metrics" / "pipeline_metrics.json")
        )
        results["metrics_file"] = metrics_path
        results["metrics_summary"] = self._metrics.summary()

        logger.info("=" * 70)
        logger.info("PIPELINE RUN COMPLETE: %s", self._pipeline_name)
        logger.info("Metrics: %s", self._metrics.summary())
        logger.info("=" * 70)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Source Data Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --phase discovery
  python -m src.main --phase identity_resolution
  python -m src.main --phase all
  python -m src.main --phase reconciliation --output-dir ./my_output
        """,
    )
    parser.add_argument(
        "--phase",
        choices=["discovery", "mapping", "identity_resolution",
                 "cutover", "reconciliation", "all"],
        default="all",
        help="Pipeline phase to run (default: all)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to config directory (default: project config/)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Path for output files (default: ./output)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )

    phase_map = {
        "discovery": orchestrator.run_discovery,
        "mapping": orchestrator.run_mapping,
        "identity_resolution": orchestrator.run_identity_resolution,
        "cutover": orchestrator.run_cutover,
        "reconciliation": orchestrator.run_reconciliation,
        "all": orchestrator.run_full_pipeline,
    }

    run_fn = phase_map[args.phase]
    result = run_fn()

    if args.phase == "all":
        print(f"\nPipeline complete. Metrics: {result.get('metrics_summary', {})}")
    else:
        print(f"\nPhase '{args.phase}' complete.")


if __name__ == "__main__":
    main()
