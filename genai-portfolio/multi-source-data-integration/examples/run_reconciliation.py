#!/usr/bin/env python3
"""Reconciliation validation demonstration.

Runs count, value, and hash-based reconciliation on synthetic data and
generates an HTML report.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.reconciliation.count_reconciler import CountReconciler
from src.reconciliation.value_reconciler import ValueReconciler
from src.reconciliation.hash_comparator import HashComparator
from src.reconciliation.report_generator import (
    EntityReconciliationReport,
    ReconciliationReport,
    ReportGenerator,
)
from src.reconciliation.validator import ReconciliationValidator
from src.utils.generators import SyntheticDataGenerator


def main() -> None:
    print("=" * 70)
    print("RECONCILIATION VALIDATION DEMONSTRATION")
    print("=" * 70)

    gen = SyntheticDataGenerator(num_customers=300, num_products=80, num_orders=500, seed=42)
    data = gen.generate_all()

    ora_custs = [c.to_dict() for c in data["oracle_customers"]]
    sql_custs = [c.to_dict() for c in data["sqlserver_customers"]]
    ora_prods = [p.to_dict() for p in data["oracle_products"]]
    sql_prods = [p.to_dict() for p in data["sqlserver_products"]]

    # For demonstration: target = combined sources (simulating post-migration)
    target_custs = ora_custs + sql_custs
    target_prods = ora_prods + sql_prods

    # -------------------------------------------------------------------------
    # 1. Count Reconciliation
    # -------------------------------------------------------------------------
    print("\n--- 1. COUNT RECONCILIATION ---\n")

    count_reconciler = CountReconciler(tolerance_pct=0.0)

    cust_count = count_reconciler.reconcile(
        entity="customer",
        target_table="INTEGRATED.DIM_CUSTOMER",
        source_counts={
            "legacy_oracle": len(ora_custs),
            "legacy_sqlserver": len(sql_custs),
        },
        target_count=len(target_custs),
        dedup_count=0,
    )
    print(f"Customer count reconciliation: {'PASSED' if cust_count.passed else 'FAILED'}")
    for check in cust_count.checks:
        print(f"  {check.check_name}: expected={check.expected_count}, "
              f"actual={check.actual_count}, delta={check.delta} "
              f"({'PASS' if check.passed else 'FAIL'})")

    prod_count = count_reconciler.reconcile(
        entity="product",
        target_table="INTEGRATED.DIM_PRODUCT",
        source_counts={
            "legacy_oracle": len(ora_prods),
            "legacy_sqlserver": len(sql_prods),
        },
        target_count=len(target_prods),
        dedup_count=0,
    )
    print(f"\nProduct count reconciliation: {'PASSED' if prod_count.passed else 'FAILED'}")
    for check in prod_count.checks:
        print(f"  {check.check_name}: expected={check.expected_count}, "
              f"actual={check.actual_count}")

    # -------------------------------------------------------------------------
    # 2. Value Reconciliation
    # -------------------------------------------------------------------------
    print("\n--- 2. VALUE RECONCILIATION ---\n")

    value_reconciler = ValueReconciler(tolerance_pct=0.01)

    # Check product prices
    prod_value = value_reconciler.reconcile_from_records(
        entity="product",
        target_table="INTEGRATED.DIM_PRODUCT",
        source_records={
            "legacy_oracle": ora_prods,
            "legacy_sqlserver": sql_prods,
        },
        target_records=target_prods,
        check_configs=[
            {
                "name": "total_unit_price",
                "check_type": "sum",
                "source_columns": {"legacy_oracle": "UNIT_PRICE", "legacy_sqlserver": "ListPrice"},
                "target_column": "UNIT_PRICE",
                "tolerance_pct": 1.0,  # relaxed for demo since SKU overlap affects totals
            },
            {
                "name": "price_range",
                "check_type": "min_max",
                "target_column": "UNIT_PRICE",
            },
            {
                "name": "category_distribution",
                "check_type": "distribution",
                "target_column": "PROD_CAT_CD",
            },
        ],
    )
    print(f"Product value reconciliation: {'PASSED' if prod_value.passed else 'FAILED'}")
    for check in prod_value.checks:
        print(f"  {check.check_name} ({check.aggregate_type.value}): "
              f"{'PASS' if check.passed else 'FAIL'}")
        if check.details:
            print(f"    {check.details[:100]}")

    # -------------------------------------------------------------------------
    # 3. Hash-Based Comparison
    # -------------------------------------------------------------------------
    print("\n--- 3. HASH-BASED ROW COMPARISON ---\n")

    hash_comp = HashComparator(hash_algorithm="sha256", max_discrepancies=20)

    # Compare target against itself (should be 100% match)
    hash_result = hash_comp.compare(
        entity="customer",
        target_table="INTEGRATED.DIM_CUSTOMER",
        source_records=target_custs,
        target_records=target_custs,
        key_columns=["CUST_ID"] if "CUST_ID" in target_custs[0] else ["CustomerID"],
        compare_columns=list(target_custs[0].keys())[:5],
    )
    print(f"Self-comparison (sanity check): {'PASSED' if hash_result.passed else 'FAILED'}")
    print(f"  Match rate: {hash_result.match_pct}%")
    print(f"  Matching rows: {hash_result.matching_rows:,}")
    print(f"  Missing in target: {hash_result.missing_in_target}")
    print(f"  Value mismatches: {hash_result.value_mismatches}")

    # Introduce some discrepancies for demo
    modified_target = [dict(r) for r in target_custs]
    if modified_target:
        # Modify a few records
        modified_target[0]["CUST_NM" if "CUST_NM" in modified_target[0] else "FirstName"] = "MODIFIED"
        if len(modified_target) > 1:
            modified_target.pop()  # Remove one record

    hash_result_diff = hash_comp.compare(
        entity="customer_with_diffs",
        target_table="INTEGRATED.DIM_CUSTOMER",
        source_records=target_custs,
        target_records=modified_target,
        key_columns=["CUST_ID"] if "CUST_ID" in target_custs[0] else ["CustomerID"],
        compare_columns=list(target_custs[0].keys())[:5],
    )
    print(f"\nComparison with modifications: {'PASSED' if hash_result_diff.passed else 'FAILED'}")
    print(f"  Match rate: {hash_result_diff.match_pct}%")
    print(f"  Missing in target: {hash_result_diff.missing_in_target}")
    print(f"  Value mismatches: {hash_result_diff.value_mismatches}")
    if hash_result_diff.discrepancies:
        print(f"  Sample discrepancies:")
        for disc in hash_result_diff.discrepancies[:3]:
            print(f"    Key={disc.key}, Type={disc.discrepancy_type}")
            if disc.differing_columns:
                print(f"    Changed columns: {disc.differing_columns}")

    # -------------------------------------------------------------------------
    # 4. HTML Report Generation
    # -------------------------------------------------------------------------
    print("\n--- 4. REPORT GENERATION ---\n")

    report = ReconciliationReport(pipeline_name="demo-reconciliation")
    report.entities.append(EntityReconciliationReport(
        entity="customer",
        count_result=cust_count,
        hash_result=hash_result_diff,
    ))
    report.entities.append(EntityReconciliationReport(
        entity="product",
        count_result=prod_count,
        value_result=prod_value,
    ))

    output_dir = str(PROJECT_ROOT / "output" / "reports")
    report_gen = ReportGenerator(output_dir=output_dir)
    html_path = report_gen.generate_html(report)
    json_path = report_gen.generate_json(report)

    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")
    print(f"Overall result: {'PASSED' if report.overall_passed else 'FAILED'}")

    print("\n" + "=" * 70)
    print("RECONCILIATION DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
