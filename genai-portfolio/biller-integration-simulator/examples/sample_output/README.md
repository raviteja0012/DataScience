# Sample Output - Biller Integration Simulator

This directory contains representative example output from the three demo
scripts in `examples/`. Reviewers can read these files to understand what
the simulator produces without running any code.

## Files

| File | Source script | What it shows |
| --- | --- | --- |
| `onboarding_result.txt` | `examples/onboard_new_biller.py` | Three billers loaded from YAML, validated against a 10-check ruleset, activated, and verified against the CC&B schema mapping. Includes a sample CC&B -> canonical field mapping for `CI_ACCT`. |
| `batch_payments_result.txt` | `examples/process_batch_payments.py` | A 40-payment batch run end-to-end: synthetic accounts seeded, payments processed, idempotency duplicate detection, exception case classification, multi-channel notification dispatch, and CIS posting. |
| `settlement_report.txt` | `examples/run_settlement.py` | Three-way reconciliation across biller / platform / bank record sets, surfacing one amount mismatch and one auto-resolved missing record. |
| `settlement_report.json` | `examples/run_settlement.py` | Structured JSON variant of the settlement report - the format consumed by downstream finance systems. Includes match-rate breakdown, exception list with severity, and the settlement-complete notification record. |
| `exception_log.json` | aggregated from batch + settlement runs | Four representative exception cases (`duplicate_payment`, `amount_mismatch`, `missing_transaction`, `authorization_failure`) showing the full schema: classification rule, severity, escalation, resolution, and correlation IDs. |

## Reading Order

1. Start with `onboarding_result.txt` to see how billers are configured.
2. Read `batch_payments_result.txt` to follow a single batch through the pipeline.
3. Inspect `settlement_report.txt` and the matching `settlement_report.json`
   to see how three-way reconciliation produces both a human report and a
   machine-readable artifact.
4. Open `exception_log.json` for the canonical exception schema used by the
   `ExceptionHandler`.

## Reproducing

To regenerate these outputs from scratch on your own machine:

```bash
./quickstart.sh
```

The quickstart script installs dependencies, runs the three example scripts,
and writes the live output into `output/` at the project root. Compare the
live output against these reference samples for a quick smoke test.

## Notes

- All transaction IDs, account numbers, amounts, and biller IDs are
  synthetic and produced deterministically from seeded random generators.
- Timestamps are illustrative - real runs will have current timestamps.
- The simulator is offline by design: no external API calls are made.
