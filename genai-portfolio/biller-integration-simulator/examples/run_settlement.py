#!/usr/bin/env python3
"""
Example: Run three-way settlement reconciliation.

Demonstrates the settlement engine's reconciliation workflow:
    1. Onboard a biller and generate synthetic payment data
    2. Create settlement records from three sources (biller, platform, bank)
    3. Run three-way reconciliation
    4. Display match results and exception details
    5. Show auto-resolution of eligible exceptions

Run from the project root:
    python examples/run_settlement.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.onboarding import OnboardingEngine
from src.core.settlement_engine import SettlementEngine
from src.core.exception_handler import ExceptionHandler
from src.integration.notification_service import NotificationService
from src.models.settlement import MatchStatus
from src.utils.generators import DataSetGenerator


def main() -> None:
    print("\n" + "=" * 70)
    print("  Three-Way Settlement Reconciliation Example")
    print("=" * 70)

    # --- Step 1: Set up biller ---
    print("\n  [1] Onboarding biller...\n")
    onboarding = OnboardingEngine()
    configs = onboarding.get_raw_biller_configs()
    biller, report = onboarding.onboard_biller(configs[0])
    onboarding.activate_biller(biller.biller_id)
    print(f"       Biller: {biller.biller_id} ({biller.biller_name})")

    # --- Step 2: Generate dataset ---
    print("\n  [2] Generating synthetic dataset...\n")
    generator = DataSetGenerator(
        biller_id=biller.biller_id,
        cis_division=biller.cis_division,
        num_accounts=40,
    )
    dataset = generator.generate()

    print(f"       Accounts:   {len(dataset['accounts'])}")
    print(f"       Bills:      {len(dataset['bills'])}")
    print(f"       Payments:   {len(dataset['payments'])}")

    settlement_data = dataset["settlement"]
    print(f"       Settlement records:")
    print(f"         Biller:   {len(settlement_data['biller_records'])}")
    print(f"         Platform: {len(settlement_data['platform_records'])}")
    print(f"         Bank:     {len(settlement_data['bank_records'])}")

    # --- Step 3: Run reconciliation ---
    print("\n  [3] Running three-way reconciliation...\n")
    exception_handler = ExceptionHandler()
    settlement_engine = SettlementEngine(exception_handler=exception_handler)

    batch = settlement_engine.reconcile(
        biller_records=settlement_data["biller_records"],
        platform_records=settlement_data["platform_records"],
        bank_records=settlement_data["bank_records"],
        biller_id=biller.biller_id,
    )

    # --- Step 4: Display results ---
    summary = batch.summary()
    recon = summary["reconciliation"]

    print(f"  {'='*60}")
    print(f"  RECONCILIATION RESULTS")
    print(f"  {'='*60}\n")
    print(f"    Batch ID:       {summary['batch_id']}")
    print(f"    Settlement Date:{summary['settlement_date']}")
    print()
    print(f"    Record Counts:")
    for source, count in summary["record_counts"].items():
        print(f"      {source:>10}: {count}")
    print()
    print(f"    Totals:")
    for label, amt in summary["amounts"].items():
        print(f"      {label:>16}: ${amt}")
    print()
    print(f"    Match Results:")
    print(f"      Total Results:  {recon['total_results']}")
    print(f"      Matched:        {recon['matched']}")
    print(f"      Exceptions:     {recon['exceptions']}")
    print(f"      Match Rate:     {recon['match_rate']}")

    if recon.get("by_status"):
        print(f"\n    Status Breakdown:")
        for status, count in sorted(recon["by_status"].items()):
            print(f"      {status:>22}: {count}")

    if recon.get("exceptions_by_severity"):
        print(f"\n    Exception Severity:")
        for sev, count in recon["exceptions_by_severity"].items():
            print(f"      {sev:>10}: {count}")

    # --- Step 5: Show exception details ---
    exceptions = [r for r in batch.results if r.is_exception]
    if exceptions:
        print(f"\n    Exception Details (showing first 5):")
        print(f"    {'-'*55}")
        for exc in exceptions[:5]:
            print(f"      TXN: {exc.transaction_id[:20]}...")
            print(f"        Status:     {exc.match_status.value}")
            print(f"        Severity:   {exc.severity.value}")
            if exc.amount_difference > 0:
                print(f"        Amt Diff:   ${exc.amount_difference}")
            if exc.date_difference_days > 0:
                print(f"        Date Diff:  {exc.date_difference_days} days")
            print(f"        Details:    {exc.discrepancy_details[:60]}")
            print()

    # --- Step 6: Exception handling ---
    print(f"  [4] Exception handling summary...\n")
    exc_summary = exception_handler.summary()
    print(f"    Open Cases:     {exc_summary['open_cases']}")
    print(f"    Resolved:       {exc_summary['resolved_cases']}")

    if exc_summary["open_by_severity"]:
        print(f"    Open by Severity:")
        for sev, count in exc_summary["open_by_severity"].items():
            print(f"      {sev:>10}: {count}")

    # Check escalations
    escalations = exception_handler.check_escalations()
    if escalations:
        print(f"\n    Escalation Notifications:")
        for esc in escalations[:3]:
            print(f"      Case {esc['case_id']} -> {esc['role']} "
                  f"(severity: {esc['severity']})")

    # --- Step 7: Settlement notifications ---
    print(f"\n  [5] Sending settlement notifications...\n")
    notif_service = NotificationService()
    notif_service.notify_settlement_complete(
        biller=biller,
        batch_id=batch.batch_id,
        total_amount=str(batch.total_platform_amount),
        record_count=len(batch.platform_records),
    )
    print(f"       Settlement completion notification sent to biller.")

    # Generate JSON report
    print(f"\n  [6] JSON reconciliation report (excerpt)...\n")
    report_json = settlement_engine.generate_report(batch)
    report_data = json.loads(report_json)
    # Show just the summary
    print(json.dumps(report_data["summary"], indent=4, default=str))

    print("\n" + "=" * 70)
    print("  Settlement reconciliation example complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
