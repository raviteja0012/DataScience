#!/usr/bin/env python3
"""End-to-end pipeline demonstration.

Runs the complete integration pipeline on synthetic data:
  1. Discovery & Profiling
  2. Schema Mapping
  3. Identity Resolution
  4. Cutover Execution
  5. Reconciliation Validation

Produces metrics, logs, and HTML reconciliation reports.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main import PipelineOrchestrator


def main() -> None:
    print("=" * 70)
    print("MULTI-SOURCE DATA INTEGRATION — FULL PIPELINE")
    print("=" * 70)
    print()

    output_dir = str(PROJECT_ROOT / "output")
    orchestrator = PipelineOrchestrator(output_dir=output_dir)

    # -------------------------------------------------------------------------
    # Phase 1: Discovery
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 1: SOURCE SYSTEM DISCOVERY")
    print("=" * 50)

    discovery = orchestrator.run_discovery()
    for sys_id, profile in discovery["profiles"].items():
        print(f"\n  [{sys_id}] {profile.system_name}")
        print(f"    Tables: {profile.total_table_count}")
        print(f"    Total rows: {profile.total_row_count:,}")
        for tp in profile.tables:
            print(f"      {tp.table_name}: {tp.row_count:,} rows, {tp.column_count} cols")

    for sys_id, quality in discovery["quality"].items():
        print(f"\n  [{sys_id}] Quality Score: {quality.overall_score:.4f} "
              f"({'PASS' if quality.passed else 'FAIL'})")
        for tq in quality.tables:
            print(f"    {tq.table_name}: {tq.composite_score:.4f}")

    print("\n  Schema comparisons:")
    for comp in discovery["schema_comparisons"]:
        print(f"    {comp.source_table} -> {comp.target_table}: "
              f"{comp.mapping_coverage_pct}% coverage")

    for sys_id, graph in discovery["dependencies"].items():
        waves = graph.load_waves()
        print(f"\n  [{sys_id}] Load waves:")
        for i, wave in enumerate(waves):
            print(f"    Wave {i + 1}: {wave}")

    # -------------------------------------------------------------------------
    # Phase 2: Schema Mapping
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 2: SCHEMA MAPPING")
    print("=" * 50)

    mapping = orchestrator.run_mapping()
    for key, records in mapping.items():
        print(f"  {key}: {len(records)} records mapped")
        if records:
            sample = records[0]
            print(f"    Sample target columns: {list(sample.keys())[:6]}...")

    # -------------------------------------------------------------------------
    # Phase 3: Identity Resolution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 3: IDENTITY RESOLUTION")
    print("=" * 50)

    ir_result = orchestrator.run_identity_resolution()
    print(f"\n  Total input:       {ir_result['total_input']}")
    print(f"  Resolved entities: {ir_result['total_resolved']}")
    print(f"  Duplicates found:  {ir_result['duplicates_found']}")

    resolution = ir_result["resolution_result"]
    cross = [e for e in resolution.resolved_entities if len(set(e.source_systems)) > 1]
    print(f"  Cross-source matches: {len(cross)}")
    if cross:
        print(f"\n  Sample cross-source entity:")
        sample_entity = cross[0]
        print(f"    Business key: {sample_entity.business_key}")
        print(f"    Confidence: {sample_entity.match_confidence:.4f}")
        print(f"    Sources: {sample_entity.source_systems}")

    # -------------------------------------------------------------------------
    # Phase 4: Cutover
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 4: CUTOVER EXECUTION")
    print("=" * 50)

    cutover = orchestrator.run_cutover()
    cutover_result = cutover["cutover_result"]
    print(f"\n  Strategy: {cutover['plan_summary']['strategy']}")
    print(f"  Success: {cutover_result.success}")
    print(f"  Total phases: {len(cutover_result.phases)}")
    print(f"  Total rows loaded: {cutover_result.total_rows_loaded:,}")
    print(f"  Duration: {cutover_result.total_duration_s:.2f}s")
    for pr in cutover_result.phases:
        print(f"    Phase '{pr.phase_name}': {len(pr.tasks)} tasks, "
              f"{pr.total_rows:,} rows, {'OK' if pr.success else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Phase 5: Reconciliation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 5: RECONCILIATION VALIDATION")
    print("=" * 50)

    recon = orchestrator.run_reconciliation()
    report = recon["report"]
    print(f"\n  Overall: {'PASSED' if report.overall_passed else 'FAILED'}")
    for er in report.entities:
        print(f"  [{er.entity}]: {'PASS' if er.passed else 'FAIL'}")
        if er.count_result:
            for c in er.count_result.checks[:2]:
                print(f"    Count: {c.check_name} = {c.actual_count:,} "
                      f"(expected {c.expected_count:,}, delta={c.delta})")

    print(f"\n  Reports generated:")
    for path in recon["report_files"]:
        print(f"    {path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Pipeline: {orchestrator._pipeline_name}")
    summary = orchestrator._metrics.summary()
    print(f"  Total metrics recorded: {summary['total_metrics_recorded']}")
    print(f"  Total pipeline duration: {summary['total_pipeline_duration_s']:.2f}s")
    print(f"  Phases executed: {summary['phases']}")
    print(f"  Total rows processed: {summary['total_rows_processed']:,}")

    metrics_path = orchestrator._metrics.export_json(
        str(Path(output_dir) / "metrics" / "pipeline_metrics.json")
    )
    print(f"  Metrics exported to: {metrics_path}")
    print()


if __name__ == "__main__":
    main()
