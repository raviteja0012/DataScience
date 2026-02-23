#!/usr/bin/env python3
"""
Example: Process a batch of simulated payments.

Demonstrates the payment processing pipeline end-to-end:
    1. Onboard and activate a biller
    2. Set up the CIS adapter with synthetic accounts
    3. Generate a batch of payment transactions
    4. Process them through validation, authorization, and settlement
    5. Display results and exception summary

Run from the project root:
    python examples/process_batch_payments.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decimal import Decimal

from src.core.onboarding import OnboardingEngine
from src.core.payment_processor import PaymentProcessor
from src.core.exception_handler import ExceptionHandler
from src.integration.cis_adapter import CISAdapter
from src.integration.notification_service import NotificationService
from src.models.payment import (
    Payment,
    PaymentChannel,
    PaymentMethod,
    PaymentStatus,
    PaymentType,
)
from src.utils.generators import generate_payment


def main() -> None:
    print("\n" + "=" * 70)
    print("  Batch Payment Processing Example")
    print("=" * 70)

    # --- Step 1: Onboard biller ---
    print("\n  [1] Onboarding biller...\n")
    onboarding = OnboardingEngine()
    configs = onboarding.get_raw_biller_configs()
    biller, report = onboarding.onboard_biller(configs[0])
    onboarding.activate_biller(biller.biller_id)
    print(f"       Biller:  {biller.biller_id} ({biller.biller_name})")
    print(f"       Status:  {biller.status.value}")

    # --- Step 2: Set up CIS ---
    print("\n  [2] Setting up CIS adapter with synthetic accounts...\n")
    cis = CISAdapter(biller)
    cis.seed_data(num_accounts=30)
    active_accounts = cis.get_active_account_ids()
    print(f"       Active accounts: {len(active_accounts)}")

    # --- Step 3: Generate payments ---
    batch_size = 40
    print(f"\n  [3] Generating {batch_size} payment transactions...\n")

    payments: list[Payment] = []
    for _ in range(batch_size):
        acct_id = random.choice(active_accounts)

        # Look up the account's balance
        balance = cis.get_account_balance(acct_id)
        if balance <= Decimal("0.00"):
            balance = Decimal(str(random.uniform(50, 300)))

        pmt = generate_payment(
            biller_id=biller.biller_id,
            account_id=acct_id,
            amount=balance,
        )
        payments.append(pmt)

    # Show a sample
    sample = payments[0]
    print(f"       Sample payment:")
    print(f"         Transaction: {sample.transaction_id}")
    print(f"         Account:     {sample.customer_account_id}")
    print(f"         Amount:      ${sample.amount}")
    print(f"         Method:      {sample.payment_method.value}")
    print(f"         Channel:     {sample.channel.value}")

    # --- Step 4: Process ---
    print(f"\n  [4] Processing {batch_size} payments...\n")
    exception_handler = ExceptionHandler()
    notification_service = NotificationService()
    processor = PaymentProcessor(
        exception_handler=exception_handler,
        auth_failure_rate=0.05,  # 5% simulated decline rate
        timeout_rate=0.02,       # 2% simulated timeout rate
    )

    results = processor.process_batch(payments, biller)

    # --- Step 5: Send notifications ---
    print("  [5] Sending notifications...\n")
    for result in results:
        if result.success:
            notification_service.notify_payment_confirmation(
                result.payment, biller
            )
        else:
            notification_service.notify_payment_failure(
                result.payment, biller,
                reason=result.error_message or "Unknown error",
            )

    # --- Results ---
    stats = processor.get_stats()
    print(f"  {'='*60}")
    print(f"  PROCESSING RESULTS")
    print(f"  {'='*60}\n")
    print(f"    Total Processed:  {stats['processed']}")
    print(f"    Successful:       {stats['success']}")
    print(f"    Failed:           {stats['failed']}")
    print(f"    Success Rate:     {stats['success_rate']}")
    print(f"    Total Amount:     ${stats['total_amount']}")

    # Show failures
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\n    Failed Payments:")
        for f in failures[:5]:  # Show first 5
            print(f"      {f.payment.transaction_id[:12]}... "
                  f"${f.payment.amount} - {f.error_code}: {f.error_message}")
        if len(failures) > 5:
            print(f"      ... and {len(failures) - 5} more")

    # Exception summary
    exc = exception_handler.summary()
    print(f"\n    Exception Cases:")
    print(f"      Open:     {exc['open_cases']}")
    print(f"      Resolved: {exc['resolved_cases']}")
    if exc["open_by_type"]:
        for t, c in exc["open_by_type"].items():
            print(f"        {t}: {c}")

    # Notification summary
    notif_stats = notification_service.get_stats()
    print(f"\n    Notifications Sent: {notif_stats['total_dispatched']}")
    for event, count in notif_stats.get("by_event", {}).items():
        print(f"      {event}: {count}")

    # --- CIS posting ---
    print(f"\n  [6] Posting successful payments to CIS...\n")
    posted = 0
    for result in results:
        if result.success:
            posting = cis.post_payment(result.payment)
            if posting.success:
                posted += 1
    print(f"       Posted {posted} payments to CIS")

    print("\n" + "=" * 70)
    print("  Batch processing example complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
