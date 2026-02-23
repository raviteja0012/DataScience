"""
Biller Integration Simulator — CLI entry point.

Provides a command-line interface for running the major simulation workflows:
    - onboard    : Onboard billers from YAML configuration
    - process    : Process a batch of simulated payments
    - settle     : Run three-way settlement reconciliation
    - report     : Generate exception and settlement reports
    - demo       : Run a full end-to-end demonstration

Usage:
    python -m src.main onboard
    python -m src.main process --biller UTIL-GA-POWER-001 --count 100
    python -m src.main settle --biller UTIL-GA-POWER-001
    python -m src.main demo
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from typing import Optional

from src.core.onboarding import OnboardingEngine
from src.core.payment_processor import PaymentProcessor
from src.core.settlement_engine import SettlementEngine
from src.core.exception_handler import ExceptionHandler
from src.integration.cis_adapter import CISAdapter
from src.integration.notification_service import NotificationService
from src.utils.generators import (
    DataSetGenerator,
    generate_payment,
    generate_settlement_records,
)
from src.utils.logger import get_logger, set_correlation_id
from src.models.payment import PaymentStatus

logger = get_logger(__name__)


def cmd_onboard(args: argparse.Namespace) -> None:
    """Onboard all billers from the configuration file."""
    engine = OnboardingEngine()
    results = engine.onboard_all()

    print(f"\n{'='*60}")
    print("  BILLER ONBOARDING REPORT")
    print(f"{'='*60}\n")

    for biller, report in results:
        status_icon = "PASS" if report.all_passed else "FAIL"
        print(f"  [{status_icon}] {biller.biller_id} — {biller.biller_name}")
        print(f"        Status: {biller.status.value}")
        print(f"        CIS: {biller.cis_vendor} {biller.cis_version}")
        print(f"        Checks: {len(report.checks)} total, "
              f"{len(report.failed_checks)} failed")

        if report.failed_checks:
            for check in report.failed_checks:
                print(f"        [FAIL] {check.check_name}: {check.message}")
        print()

    # Activate billers that passed
    for biller, report in results:
        if report.all_passed and biller.status.value == "pending":
            engine.activate_biller(biller.biller_id)
            print(f"  Activated: {biller.biller_id}")

    print(f"\n{'='*60}\n")


def cmd_process(args: argparse.Namespace) -> None:
    """Process a batch of simulated payments."""
    biller_id = args.biller
    count = args.count

    # Onboard the biller first
    engine = OnboardingEngine()
    results = engine.onboard_all()

    biller = None
    for b, r in results:
        if b.biller_id == biller_id and r.all_passed:
            engine.activate_biller(b.biller_id)
            biller = b
            break

    if biller is None:
        print(f"Error: Biller '{biller_id}' not found or failed onboarding.")
        sys.exit(1)

    # Set up CIS adapter and generate data
    cis = CISAdapter(biller)
    cis.seed_data(num_accounts=max(count // 2, 10))

    # Generate payments
    active_accounts = cis.get_active_account_ids()
    payments = []
    import random
    for _ in range(count):
        acct_id = random.choice(active_accounts)
        pmt = generate_payment(
            biller_id=biller.biller_id,
            account_id=acct_id,
        )
        payments.append(pmt)

    # Process
    exception_handler = ExceptionHandler()
    processor = PaymentProcessor(
        exception_handler=exception_handler,
        auth_failure_rate=0.05,
    )

    print(f"\n{'='*60}")
    print(f"  PAYMENT PROCESSING — {biller.biller_name}")
    print(f"{'='*60}\n")
    print(f"  Processing {count} payments...\n")

    results = processor.process_batch(payments, biller)

    stats = processor.get_stats()
    print(f"  Results:")
    print(f"    Processed:    {stats['processed']}")
    print(f"    Successful:   {stats['success']}")
    print(f"    Failed:       {stats['failed']}")
    print(f"    Success Rate: {stats['success_rate']}")
    print(f"    Total Amount: ${stats['total_amount']}")

    exc_summary = exception_handler.summary()
    if exc_summary["open_cases"] > 0:
        print(f"\n  Exceptions:")
        print(f"    Open Cases:  {exc_summary['open_cases']}")
        for sev, cnt in exc_summary.get("open_by_severity", {}).items():
            print(f"      {sev}: {cnt}")

    print(f"\n{'='*60}\n")


def cmd_settle(args: argparse.Namespace) -> None:
    """Run three-way settlement reconciliation."""
    biller_id = args.biller

    # Onboard
    engine = OnboardingEngine()
    results = engine.onboard_all()

    biller = None
    for b, r in results:
        if b.biller_id == biller_id and r.all_passed:
            engine.activate_biller(b.biller_id)
            biller = b
            break

    if biller is None:
        print(f"Error: Biller '{biller_id}' not found or failed onboarding.")
        sys.exit(1)

    # Generate synthetic settlement data
    gen = DataSetGenerator(
        biller_id=biller.biller_id,
        cis_division=biller.cis_division,
        num_accounts=30,
    )
    dataset = gen.generate()
    settlement_data = dataset["settlement"]

    # Run reconciliation
    exception_handler = ExceptionHandler()
    settlement_engine = SettlementEngine(exception_handler=exception_handler)

    batch = settlement_engine.reconcile(
        biller_records=settlement_data["biller_records"],
        platform_records=settlement_data["platform_records"],
        bank_records=settlement_data["bank_records"],
        biller_id=biller.biller_id,
    )

    print(f"\n{'='*60}")
    print(f"  SETTLEMENT RECONCILIATION — {biller.biller_name}")
    print(f"{'='*60}\n")

    summary = batch.summary()
    print(f"  Batch ID: {summary['batch_id']}")
    print(f"  Date:     {summary['settlement_date']}")
    print()
    print(f"  Record Counts:")
    for source, count in summary["record_counts"].items():
        print(f"    {source:>10}: {count}")
    print()
    print(f"  Amounts:")
    for source, amt in summary["amounts"].items():
        print(f"    {source:>16}: ${amt}")
    print()
    print(f"  Reconciliation:")
    recon = summary["reconciliation"]
    print(f"    Total:      {recon['total_results']}")
    print(f"    Matched:    {recon['matched']}")
    print(f"    Exceptions: {recon['exceptions']}")
    print(f"    Match Rate: {recon['match_rate']}")
    print()

    if recon.get("by_status"):
        print(f"  Status Breakdown:")
        for status, cnt in recon["by_status"].items():
            print(f"    {status:>20}: {cnt}")

    if recon.get("exceptions_by_severity"):
        print(f"\n  Exception Severity:")
        for sev, cnt in recon["exceptions_by_severity"].items():
            print(f"    {sev:>10}: {cnt}")

    print(f"\n{'='*60}\n")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a full end-to-end demonstration."""
    print(f"\n{'='*60}")
    print("  BILLER INTEGRATION SIMULATOR — END-TO-END DEMO")
    print(f"{'='*60}\n")

    # Phase 1: Onboard
    print("  Phase 1: Biller Onboarding")
    print("  " + "-" * 40)

    onboarding = OnboardingEngine()
    onboard_results = onboarding.onboard_all()

    active_billers = []
    for biller, report in onboard_results:
        if report.all_passed:
            onboarding.activate_biller(biller.biller_id)
            active_billers.append(biller)
            print(f"    [OK] {biller.biller_id} — {biller.biller_name}")
        else:
            print(f"    [--] {biller.biller_id} — skipped (validation failed)")

    if not active_billers:
        print("\n  No billers available. Exiting.")
        return

    demo_biller = active_billers[0]
    print(f"\n  Using {demo_biller.biller_id} for demo.\n")

    # Phase 2: Generate & Process Payments
    print("  Phase 2: Payment Processing")
    print("  " + "-" * 40)

    cis = CISAdapter(demo_biller)
    cis.seed_data(num_accounts=25)

    import random
    active_accts = cis.get_active_account_ids()
    payments = [
        generate_payment(
            biller_id=demo_biller.biller_id,
            account_id=random.choice(active_accts),
        )
        for _ in range(50)
    ]

    exception_handler = ExceptionHandler()
    processor = PaymentProcessor(exception_handler=exception_handler)
    proc_results = processor.process_batch(payments, demo_biller)

    stats = processor.get_stats()
    print(f"    Processed: {stats['processed']}, "
          f"Success: {stats['success']}, "
          f"Failed: {stats['failed']}")
    print(f"    Total: ${stats['total_amount']}\n")

    # Phase 3: Settlement
    print("  Phase 3: Settlement Reconciliation")
    print("  " + "-" * 40)

    settled = [p for p in payments if p.status == PaymentStatus.SETTLED]
    biller_recs, platform_recs, bank_recs = generate_settlement_records(
        settled, anomaly_rate=0.10,
    )

    settlement_engine = SettlementEngine(exception_handler=exception_handler)
    batch = settlement_engine.reconcile(
        biller_records=biller_recs,
        platform_records=platform_recs,
        bank_records=bank_recs,
        biller_id=demo_biller.biller_id,
    )

    summary = batch.summary()
    recon = summary["reconciliation"]
    print(f"    Records:    {recon['total_results']}")
    print(f"    Matched:    {recon['matched']}")
    print(f"    Exceptions: {recon['exceptions']}")
    print(f"    Match Rate: {recon['match_rate']}\n")

    # Phase 4: Exception Summary
    print("  Phase 4: Exception Summary")
    print("  " + "-" * 40)

    exc = exception_handler.summary()
    print(f"    Open:     {exc['open_cases']}")
    print(f"    Resolved: {exc['resolved_cases']}")

    if exc["open_by_type"]:
        print(f"    By Type:")
        for t, c in exc["open_by_type"].items():
            print(f"      {t}: {c}")

    print(f"\n{'='*60}")
    print("  Demo complete.")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="biller-integration-simulator",
        description="Utility CIS-to-Payment Platform Integration Simulator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # onboard
    sub_onboard = subparsers.add_parser("onboard", help="Onboard billers from config")
    sub_onboard.set_defaults(func=cmd_onboard)

    # process
    sub_process = subparsers.add_parser("process", help="Process simulated payments")
    sub_process.add_argument(
        "--biller", type=str, default="UTIL-GA-POWER-001",
        help="Biller ID to process payments for",
    )
    sub_process.add_argument(
        "--count", type=int, default=50,
        help="Number of payments to simulate",
    )
    sub_process.set_defaults(func=cmd_process)

    # settle
    sub_settle = subparsers.add_parser("settle", help="Run settlement reconciliation")
    sub_settle.add_argument(
        "--biller", type=str, default="UTIL-GA-POWER-001",
        help="Biller ID to reconcile",
    )
    sub_settle.set_defaults(func=cmd_settle)

    # demo
    sub_demo = subparsers.add_parser("demo", help="Run full end-to-end demo")
    sub_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
