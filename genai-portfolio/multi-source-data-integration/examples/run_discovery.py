#!/usr/bin/env python3
"""Source system discovery demonstration.

Profiles synthetic data from both legacy Oracle and SQL Server systems,
runs data quality scanning, analyses schema structure, and maps
table dependencies.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.discovery.data_quality_scanner import DataQualityScanner
from src.discovery.dependency_mapper import DependencyMapper
from src.discovery.schema_analyzer import SchemaAnalyzer
from src.discovery.source_profiler import SourceProfiler
from src.utils.generators import SyntheticDataGenerator


def main() -> None:
    print("=" * 70)
    print("SOURCE SYSTEM DISCOVERY & PROFILING")
    print("=" * 70)

    # Generate synthetic data
    gen = SyntheticDataGenerator(num_customers=200, num_products=50, num_orders=300, seed=42)
    data = gen.generate_all()

    # Load configs
    config_dir = PROJECT_ROOT / "config"
    with open(config_dir / "source_systems" / "legacy_oracle.yaml") as f:
        oracle_cfg = yaml.safe_load(f)
    with open(config_dir / "source_systems" / "legacy_sqlserver.yaml") as f:
        sqlserver_cfg = yaml.safe_load(f)
    with open(config_dir / "source_systems" / "target_snowflake.yaml") as f:
        target_cfg = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # 1. Source Profiling
    # -------------------------------------------------------------------------
    print("\n--- 1. SOURCE PROFILING ---\n")

    oracle_tables = {
        "CUST_MASTER": [c.to_dict() for c in data["oracle_customers"]],
        "PRODUCT_CATALOG": [p.to_dict() for p in data["oracle_products"]],
        "ORDER_HEADER": [o.to_dict() for o in data["oracle_orders"]],
        "ORDER_DETAIL": [d.to_dict() for d in data["oracle_order_details"]],
    }
    sql_tables = {
        "Customers": [c.to_dict() for c in data["sqlserver_customers"]],
        "Products": [p.to_dict() for p in data["sqlserver_products"]],
        "SalesOrders": [o.to_dict() for o in data["sqlserver_orders"]],
        "SalesOrderDetails": [d.to_dict() for d in data["sqlserver_order_details"]],
    }

    oracle_profiler = SourceProfiler(oracle_cfg)
    oracle_profile = oracle_profiler.profile_from_records(oracle_tables)
    print(f"Oracle ERP: {oracle_profile.total_table_count} tables, "
          f"{oracle_profile.total_row_count:,} total rows")
    for tp in oracle_profile.tables:
        print(f"  {tp.table_name}: {tp.row_count:,} rows, {tp.column_count} columns")
        for cp in tp.columns[:3]:
            print(f"    {cp.name}: type={cp.data_type}, null_pct={cp.null_pct}%, "
                  f"distinct={cp.distinct_count}")
        if tp.column_count > 3:
            print(f"    ... and {tp.column_count - 3} more columns")

    print()
    sql_profiler = SourceProfiler(sqlserver_cfg)
    sql_profile = sql_profiler.profile_from_records(sql_tables)
    print(f"SQL Server CRM: {sql_profile.total_table_count} tables, "
          f"{sql_profile.total_row_count:,} total rows")
    for tp in sql_profile.tables:
        print(f"  {tp.table_name}: {tp.row_count:,} rows, {tp.column_count} columns")

    # -------------------------------------------------------------------------
    # 2. Data Quality Scanning
    # -------------------------------------------------------------------------
    print("\n--- 2. DATA QUALITY ASSESSMENT ---\n")

    scanner = DataQualityScanner(threshold=0.70)

    oracle_quality = scanner.scan_source(oracle_profile.tables, raw_tables=oracle_tables)
    print(f"Oracle Quality Score: {oracle_quality.overall_score:.4f} "
          f"({'PASSED' if oracle_quality.passed else 'FAILED'})")
    for tq in oracle_quality.tables:
        print(f"  {tq.table_name}: {tq.composite_score:.4f}")
        # Show lowest-quality columns
        sorted_cols = sorted(tq.columns, key=lambda c: c.composite_score)
        for cq in sorted_cols[:2]:
            dims = ", ".join(f"{d.name}={d.score:.2f}" for d in cq.dimensions)
            print(f"    {cq.name}: {cq.composite_score:.4f} ({dims})")

    print()
    sql_quality = scanner.scan_source(sql_profile.tables, raw_tables=sql_tables)
    print(f"SQL Server Quality Score: {sql_quality.overall_score:.4f} "
          f"({'PASSED' if sql_quality.passed else 'FAILED'})")
    for tq in sql_quality.tables:
        print(f"  {tq.table_name}: {tq.composite_score:.4f}")

    # -------------------------------------------------------------------------
    # 3. Schema Analysis
    # -------------------------------------------------------------------------
    print("\n--- 3. SCHEMA STRUCTURE ANALYSIS ---\n")

    analyzer = SchemaAnalyzer()
    comparisons = analyzer.analyze_source_config(oracle_cfg, target_cfg)
    for comp in comparisons:
        print(f"  {comp.source_table} -> {comp.target_table}")
        print(f"    Matched: {len(comp.matched_columns)}, "
              f"Unmapped Source: {len(comp.unmapped_source_columns)}, "
              f"Unmapped Target: {len(comp.unmapped_target_columns)}")
        print(f"    Coverage: {comp.mapping_coverage_pct}%")
        for mc in comp.matched_columns[:3]:
            print(f"      {mc.source_column} -> {mc.target_column} "
                  f"(confidence={mc.confidence.value})")

    # -------------------------------------------------------------------------
    # 4. Dependency Mapping
    # -------------------------------------------------------------------------
    print("\n--- 4. DEPENDENCY MAPPING ---\n")

    dep_mapper = DependencyMapper()

    oracle_graph = dep_mapper.build_graph(oracle_cfg)
    print(f"Oracle dependency graph: {oracle_graph.table_count} tables, "
          f"{len(oracle_graph.edges)} FK edges")
    print(f"  Root tables (no dependencies): {oracle_graph.root_tables}")
    print(f"  Leaf tables: {oracle_graph.leaf_tables}")
    print(f"  Topological load order: {oracle_graph.topological_order()}")
    print(f"  Parallel load waves:")
    for i, wave in enumerate(oracle_graph.load_waves()):
        print(f"    Wave {i + 1}: {wave}")

    print()
    sql_graph = dep_mapper.build_graph(sqlserver_cfg)
    print(f"SQL Server dependency graph: {sql_graph.table_count} tables, "
          f"{len(sql_graph.edges)} FK edges")
    print(f"  Topological load order: {sql_graph.topological_order()}")
    print(f"  Parallel load waves:")
    for i, wave in enumerate(sql_graph.load_waves()):
        print(f"    Wave {i + 1}: {wave}")

    # Merged graph
    merged = dep_mapper.merge_graphs([oracle_graph, sql_graph])
    print(f"\nMerged graph: {merged.table_count} tables, {len(merged.edges)} edges")

    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
