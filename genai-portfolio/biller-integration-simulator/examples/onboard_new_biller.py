#!/usr/bin/env python3
"""
Example: Onboard a new utility biller onto the payment platform.

Demonstrates the YAML-driven onboarding workflow:
    1. Load biller configurations from YAML
    2. Run validation checks on each biller
    3. Activate billers that pass all checks
    4. Display the onboarding report

Run from the project root:
    python examples/onboard_new_biller.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.onboarding import OnboardingEngine
from src.core.schema_parser import SchemaParser


def main() -> None:
    print("\n" + "=" * 70)
    print("  Biller Onboarding Example")
    print("=" * 70)

    # --- Phase 1: Onboard from YAML ---
    print("\n  [1] Loading biller configurations...\n")
    engine = OnboardingEngine()
    results = engine.onboard_all()

    for biller, report in results:
        summary = report.summary()
        status = "PASS" if report.all_passed else "FAIL"

        print(f"  [{status}] {biller.biller_id}")
        print(f"       Name:        {biller.biller_name}")
        print(f"       CIS:         {biller.cis_vendor} v{biller.cis_version}")
        print(f"       Division:    {biller.cis_division}")
        print(f"       Status:      {biller.status.value}")
        print(f"       Checks:      {summary['passed']}/{summary['total_checks']} passed")

        if summary["failures"]:
            for f in summary["failures"]:
                print(f"       [FAIL] {f['check']}: {f['message']}")

        # Show payment types
        pt_names = [pt.type for pt in biller.payment_types]
        print(f"       Pay Types:   {', '.join(pt_names)}")

        # Show fee info
        cc_fee = biller.fee_structure.convenience_fee.credit_card
        print(f"       CC Fee:      ${cc_fee}")
        print(f"       Settlement:  {biller.settlement.method} / {biller.settlement.frequency}")
        print()

    # --- Phase 2: Activate passing billers ---
    print("  [2] Activating billers that passed validation...\n")
    activated = []
    for biller, report in results:
        if report.all_passed and biller.status.value != "active":
            engine.activate_biller(biller.biller_id)
            activated.append(biller)
            print(f"       Activated: {biller.biller_id}")

    if not activated:
        print("       No new billers to activate (already active or failed checks).")

    # --- Phase 3: Verify schema mapping ---
    print("\n  [3] Verifying CC&B schema mappings...\n")
    parser = SchemaParser()

    mappings_to_check = [
        "account_mapping",
        "person_mapping",
        "bill_mapping",
        "bill_segment_mapping",
        "service_agreement_mapping",
    ]

    for mapping_name in mappings_to_check:
        table = parser.get_table_name(mapping_name)
        fields = parser.get_field_names(mapping_name)
        required = parser.get_required_fields(mapping_name)
        print(f"       {table:15} -> {len(fields)} fields mapped ({len(required)} required)")

    # --- Phase 4: Test a sample field mapping ---
    print("\n  [4] Sample field mapping (CI_ACCT -> canonical)...\n")
    sample_record = {
        "ACCT_ID": "0007654321",
        "CIS_DIVISION": "GP01",
        "SETUP_DT": "2023-08-10",
        "ACCT_STATUS_FLG": "20",
        "CURRENCY_CD": "USD",
        "COLL_CL_CD": "NORM",
    }

    mapped = parser.map_record("account_mapping", sample_record)
    print("       Source (CC&B)             -> Target (Platform)")
    print("       " + "-" * 50)
    for target_field, value in mapped.items():
        if value is not None:
            print(f"       {target_field:28} = {value}")

    print("\n" + "=" * 70)
    print("  Onboarding example complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
