# Configuration Reference

This document is the authoritative schema reference for the three YAML configuration files that drive the Biller Integration Simulator. The system is designed so that adding a new utility biller, mapping a new CC&B installation, or tuning settlement behavior requires **zero code changes** -- everything is configuration-driven.

| File | Purpose | Loaded by |
|------|---------|-----------|
| `config/biller_config.yaml` | Per-biller onboarding parameters (channels, fees, settlement, validation) | `src/core/onboarding.py` |
| `config/schema_mapping.yaml` | CC&B-to-canonical column mappings + transform definitions | `src/core/schema_parser.py` |
| `config/settlement_rules.yaml` | Reconciliation tolerances, exception severity, escalation, batch windows | `src/core/settlement_engine.py`, `src/core/exception_handler.py` |

---

## 1. `biller_config.yaml`

Top-level structure:

```yaml
billers:           # list[Biller]      -- required
  - <biller entry>
defaults:          # GlobalDefaults    -- required
  ...
```

### 1.1 Biller Entry

Each entry under `billers:` defines one utility biller.

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `biller_id` | string | yes | Globally unique biller identifier. Convention: `UTIL-<STATE>-<TYPE>-<SEQ>` | `UTIL-GA-POWER-001` |
| `biller_name` | string | yes | Human-readable utility name | `Georgia Power & Light` |
| `cis_division` | string | yes | CC&B division code (4-char common) | `GP01` |
| `cis_vendor` | string | yes | CIS vendor identifier; currently only `oracle_ccb` | `oracle_ccb` |
| `cis_version` | string | yes | CC&B version string (semver-style) | `2.9.0.1` |
| `status` | enum string | yes | One of: `active`, `pending`, `suspended`, `disabled` | `active` |
| `onboarded_date` | date \| null | no | ISO date when biller went live; `null` if not yet onboarded | `2024-03-15` |
| `payment_types` | list | yes | See [Payment Types](#payment-types) | -- |
| `fee_structure` | object | yes | See [Fee Structure](#fee-structure) | -- |
| `settlement` | object | yes | See [Settlement](#settlement) | -- |
| `notifications` | object | yes | See [Notifications](#notifications) | -- |
| `validation_rules` | object | yes | See [Validation Rules](#validation-rules) | -- |

**Validation rules (enforced by `onboarding.py`):**
- `biller_id` must match `^UTIL-[A-Z]{2,4}-[A-Z]+-\d{3}$` (advisory) and be unique across the file.
- `status: active` requires a non-null `onboarded_date`.
- At least one entry in `payment_types` must have `enabled: true`.

### Payment Types

`payment_types` is a list of objects -- one per supported payment type per biller.

| Field | Type | Required | Description | Allowed Values |
|-------|------|----------|-------------|----------------|
| `type` | enum | yes | Payment type | `one_time`, `autopay`, `budget_billing`, `prepay` |
| `enabled` | bool | yes | Whether the type is active for this biller | `true`/`false` |
| `channels` | list[enum] | yes | Allowed channels | `web`, `mobile`, `ivr`, `agent`, `kiosk` |
| `max_amount` | decimal | yes | Maximum single transaction amount (USD) | `25000.00` |
| `min_amount` | decimal | no (default: `1.00`) | Minimum transaction amount | `5.00` |
| `enrollment_required` | bool | no | Required for `autopay`; signals enrollment workflow | `true` |
| `recalculation_frequency` | enum | no | For `budget_billing` only | `monthly`, `quarterly`, `annual` |

**Validation rules:**
- `min_amount <= max_amount`
- `channels` must be a non-empty subset of the global channel set.
- If `type == "autopay"`, `enrollment_required` must be `true`.
- If `type == "budget_billing"`, `recalculation_frequency` is required.

### Fee Structure

```yaml
fee_structure:
  convenience_fee:
    credit_card: 2.65
    debit_card: 1.50
    ach: 0.00
    check: 0.00
  late_payment_fee: 10.00
  returned_payment_fee: 25.00
  fee_assessed_by: "payment_platform"     # or "biller"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `convenience_fee.credit_card` | decimal | yes | Flat or percentage fee (depends on `fee_assessed_by`) |
| `convenience_fee.debit_card` | decimal | yes | -- |
| `convenience_fee.ach` | decimal | yes | Typically 0.00 |
| `convenience_fee.check` | decimal | yes | Typically 0.00 |
| `late_payment_fee` | decimal | yes | Fee assessed when payment is past due |
| `returned_payment_fee` | decimal | yes | Fee for NSF / returned ACH |
| `fee_assessed_by` | enum | yes | `payment_platform` (added at gateway) or `biller` (added by CIS) |

### Settlement

```yaml
settlement:
  method: "ach_batch"                # ach_batch | wire | rtp
  frequency: "next_business_day"     # next_business_day | t_plus_2 | daily | weekly
  bank_routing: "061000052"          # 9-digit ABA routing
  bank_account: "****7834"           # Masked; full value loaded from secrets manager
  settlement_currency: "USD"
  holdback_percentage: 0.0           # 0.0 - 100.0
  minimum_settlement_amount: 100.00  # Don't settle batches below this
```

**Validation rules:**
- `bank_routing` must be 9 digits and pass ABA check-digit validation.
- `holdback_percentage` in range `[0.0, 100.0]`.
- `minimum_settlement_amount >= 0.00`.

### Notifications

```yaml
notifications:
  payment_confirmation: true
  payment_failure: true
  settlement_complete: true
  autopay_reminder: true
  channels: ["email", "sms"]    # subset of: email, sms, push, postal
```

### Validation Rules

```yaml
validation_rules:
  account_number_format: "^\\d{10}$"     # Python regex (escaped)
  account_number_length: 10
  allow_partial_payments: true
  allow_overpayments: false
  require_bill_match: true
```

| Field | Type | Description |
|-------|------|-------------|
| `account_number_format` | regex string | Python regex applied to inbound account IDs |
| `account_number_length` | int | Expected length (used as fast-fail check before regex) |
| `allow_partial_payments` | bool | If `false`, customer must pay full bill amount |
| `allow_overpayments` | bool | If `true`, payment amount may exceed total amount due |
| `require_bill_match` | bool | If `true`, payment must reference an existing `bill_id` |

### 1.2 Global Defaults

Applied when a biller-level value is missing.

```yaml
defaults:
  cis_vendor: "oracle_ccb"
  settlement_currency: "USD"
  max_retry_attempts: 3
  retry_backoff_seconds: [30, 120, 600]    # one entry per attempt
  payment_timeout_seconds: 30
  batch_size: 500
  idempotency_window_hours: 24
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cis_vendor` | string | `oracle_ccb` | Fallback CIS vendor for billers that omit it |
| `settlement_currency` | ISO 4217 string | `USD` | Default settlement currency |
| `max_retry_attempts` | int | `3` | Used by `Payment.max_attempts` |
| `retry_backoff_seconds` | list[int] | `[30, 120, 600]` | Backoff per attempt; length must equal `max_retry_attempts` |
| `payment_timeout_seconds` | int | `30` | Gateway-side timeout |
| `batch_size` | int | `500` | Settlement batch chunk size |
| `idempotency_window_hours` | int | `24` | Window in which duplicate-detection checks `idempotency_key` |

---

## 2. `schema_mapping.yaml`

Maps Oracle CC&B table.column names to the platform's canonical fields. Used by `schema_parser.py` during onboarding (one-time) and at runtime (for live extracts).

### 2.1 Top-Level Structure

```yaml
account_mapping:               # MappingGroup -- maps CI_ACCT
  table: "CI_ACCT"
  fields: [<FieldMap>, ...]
person_mapping:                # MappingGroup -- maps CI_PER
bill_mapping:                  # MappingGroup -- maps CI_BILL
bill_segment_mapping:          # MappingGroup -- maps CI_BILL_SEG
service_agreement_mapping:     # MappingGroup -- maps CI_SA
transforms:                    # dict[str, TransformDef]
  <transform_name>: <TransformDef>
```

### 2.2 MappingGroup

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `table` | string | yes | CC&B table name (e.g., `CI_ACCT`) |
| `fields` | list[FieldMap] | yes | One entry per source column to map |

### 2.3 FieldMap

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | yes | CC&B column name |
| `target` | string | yes | Canonical field name on the platform side |
| `transform` | string \| null | no | Name of a transform defined in the `transforms:` section |
| `required` | bool | no (default: `true`) | If `true`, mapping fails when source value is missing |
| `default` | any | no | Value to substitute when source is null and `required: false` |
| `lookup` | dict[str, str] | no | Required when transform is `status_flag_decode`, `flag_decode`, or `sa_type_decode` |
| `description` | string | no | Free-form documentation |

### 2.4 Transform Types

The `transforms:` section is a dictionary of named transformations. Each entry has:

```yaml
<name>:
  description: "Free-form description"
  function: "<lambda or named function>"
```

The currently supported transforms are:

| Transform | Description | Input | Output | Example |
|-----------|-------------|-------|--------|---------|
| `strip_leading_zeros` | Remove leading zeros from numeric ID strings | `"0001234567"` | `"1234567"` | Used on ACCT_ID, BILL_SEG_ID |
| `oracle_date_to_iso8601` | Convert Oracle DATE format to ISO 8601 | `"2024-03-15 14:32:11"` | `"2024-03-15T14:32:11Z"` | All `*_DT` fields |
| `status_flag_decode` | Decode CC&B numeric status flag using `lookup` | `"20"` | `"active"` | `ACCT_STATUS_FLG`, `SA_STATUS_FLG` |
| `flag_decode` | Decode single-character CC&B flag | `"P"` | `"person"` | `PER_OR_BUS_FLG` |
| `boolean_flag` | Convert Y/N to boolean | `"Y"` | `True` | `LIFE_SUPPORT_FLG`, `CR_NOTE_FLG` |
| `lowercase` | Lowercase string (null-safe) | `"USER@EX.COM"` | `"user@ex.com"` | `EMAILID` |
| `decimal_2` | Round to 2 decimal places | `100.555` | `100.56` | All monetary amounts |
| `parse_alert_flags` | Split pipe-delimited string into list | `"DNF|MED|VIP"` | `["DNF", "MED", "VIP"]` | `ALERT_INFO` |
| `sa_type_decode` | Decode service-agreement-type code | `"E-RES"` | `"electric_residential"` | `SA_TYPE_CD` |

### 2.5 Lookup-Backed Transforms

Three transforms (`status_flag_decode`, `flag_decode`, `sa_type_decode`) require a `lookup` mapping on the field:

```yaml
- source: "ACCT_STATUS_FLG"
  target: "account_status"
  transform: "status_flag_decode"
  required: true
  lookup:
    "20": "active"
    "40": "suspended"
    "60": "closed"
    "80": "written_off"
```

If a value arrives that is not in the lookup, the parser raises a `SchemaParserError` (or substitutes `default` if defined).

### 2.6 Adding a New Transform

To add a new transform without changing application code, add an entry under `transforms:`:

```yaml
transforms:
  my_custom_transform:
    description: "Trim and uppercase a string"
    function: "lambda x: x.strip().upper() if x else x"
```

Then reference it from any `FieldMap.transform`. Lambda strings are evaluated by `schema_parser.py` in a sandboxed namespace; named functions resolve to module-level callables.

---

## 3. `settlement_rules.yaml`

Governs three-way reconciliation, exception classification, escalation, batch scheduling, and reporting.

### 3.1 Top-Level Structure

```yaml
reconciliation:        # MatchingConfig
  match_keys: ...
  tolerances: ...
  statuses: ...
exception_handling:    # ExceptionConfig
  severity_levels: ...
  auto_resolve: ...
  escalation: ...
settlement_processing: # ProcessingConfig
  batch_windows: ...
  file_formats: ...
  reports: ...
```

### 3.2 `reconciliation`

#### `match_keys`

| Field | Type | Description |
|-------|------|-------------|
| `primary` | string | Field used for primary matching (typically `transaction_id`) |
| `secondary` | list[string] | Additional fields required to confirm a match |
| `fallback` | list[string] | Alternate keys when primary is missing |

#### `tolerances`

| Field | Type | Description |
|-------|------|-------------|
| `amount.absolute` | decimal | Maximum absolute amount difference treated as a match (e.g., `0.01` for penny rounding) |
| `amount.percentage` | decimal | Maximum % amount difference (0.0 disables) |
| `date.days` | int | Allowed days offset between source dates |
| `timestamp.seconds` | int | Allowed seconds offset between source timestamps |

#### `statuses`

A documentation-only mapping of `MatchStatus` enum values to human-readable descriptions. Must include keys for every value in the `MatchStatus` enum (`matched`, `partial_match`, `amount_mismatch`, `date_mismatch`, `missing_biller`, `missing_platform`, `missing_bank`, `duplicate`, `orphan`).

### 3.3 `exception_handling`

#### `severity_levels`

Maps severity tier (`critical`, `high`, `medium`, `low`) to a list of `MatchStatus` values that map to that severity.

```yaml
severity_levels:
  critical:
    - "missing_bank"     # Money collected, not settled
    - "duplicate"        # Risk of double-charge
  high:
    - "amount_mismatch"
    - "missing_platform"
  medium:
    - "missing_biller"
    - "date_mismatch"
  low:
    - "orphan"
    - "partial_match"
```

Every status returned by the matcher must be classified into exactly one severity tier.

#### `auto_resolve`

A list of rules. Each rule is evaluated in order and the first match applies.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `condition` | expression string | yes | A simple boolean expression with named variables (`difference`, `day_difference`, `record_age_hours`) |
| `action` | enum | yes | One of `auto_match`, `defer`, `write_off`, `manual_review`, `escalate` |
| `reason` | string | yes | Audit-trail text written to `ReconciliationResult.resolution_reason` |
| `recheck_hours` | int | no | Required when `action: defer`; how long until next attempt |
| `requires_approval` | bool | no | If `true`, action stages but does not commit until human approval |

Example:

```yaml
auto_resolve:
  - condition: "amount_mismatch AND abs(difference) <= 0.01"
    action: "auto_match"
    reason: "Rounding difference within penny tolerance"
  - condition: "missing_biller AND record_age_hours <= 24"
    action: "defer"
    reason: "Biller posting lag - recheck in next cycle"
    recheck_hours: 4
```

#### `escalation`

A list of escalation chains, one per severity.

| Field | Type | Description |
|-------|------|-------------|
| `severity` | enum | `critical`, `high`, `medium`, `low` |
| `initial_response_minutes` | int | SLA target for first acknowledgement |
| `escalation_chain` | list | Sequence of `(role, after_minutes)` steps |
| `escalation_chain[].role` | string | Org role to notify |
| `escalation_chain[].after_minutes` | int | Minutes since exception creation when this role is paged |
| `notification_channels` | list[enum] | Subset of `email`, `sms`, `pagerduty`, `push` |

### 3.4 `settlement_processing`

#### `batch_windows`

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Logical batch name (used in logs and metrics) |
| `cron` | cron string | 5-field POSIX cron spec |
| `description` | string | Free-form |

#### `file_formats`

Three named record-layout descriptors:

- `biller_extract` -- fixed-width with field offsets
- `platform_extract` -- CSV with header row
- `bank_settlement` -- NACHA/ACH binary format (no schema required; parser is built-in)

For `fixed_width`, each field entry requires:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Canonical field name |
| `start` | int | 1-indexed start column |
| `length` | int | Field width in characters |
| `decimal_places` | int | (Optional) For numeric fields, implied decimal places |
| `format` | string | (Optional) For dates, e.g., `YYYYMMDD` |

#### `reports`

Each report definition includes:

| Field | Type | Description |
|-------|------|-------------|
| `output_format` | enum | `csv`, `json`, `parquet` |
| `include_matched` | bool | Whether to include `MATCHED` results (false for exception-only reports) |
| `include_exceptions` | bool | Whether to include exception results |
| `group_by` | list[string] | Fields to group/aggregate by (for `monthly_summary`) |
| `retention_days` | int | Retention period; `2555` for 7-year financial audit compliance |

---

## Configuration Validation

All three files are validated at startup by `src/core/onboarding.py` and `src/core/schema_parser.py`. Validation errors raise typed exceptions:

| Exception | Raised When |
|-----------|-------------|
| `BillerConfigError` | Schema violation in `biller_config.yaml` |
| `SchemaMappingError` | Missing transforms, invalid lookups, unknown table refs |
| `SettlementRulesError` | Severity gaps, malformed escalation chain, invalid cron |

Run validation standalone with:

```bash
python -m src.main validate-config
```

This loads all three files, runs structural and cross-reference validation, and exits with code `0` on success or `1` with a list of errors on failure.
