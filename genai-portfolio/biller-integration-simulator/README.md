# Biller Integration Simulator

A production-grade simulation of utility CIS (Customer Information System) to payment platform integration. This project models the real-world complexities of onboarding utility billers, processing payments, and reconciling settlements across multiple financial systems.

Built from hands-on experience with Oracle Utilities CC&B integrations, payment gateway implementations, and settlement reconciliation workflows in the utility billing space.

## What This Project Does

Utility companies run Customer Information Systems (typically Oracle CC&B) that manage accounts, billing, and service agreements. When these utilities want to accept payments through third-party platforms (web portals, mobile apps, IVR systems), a complex integration layer is required to:

1. **Map CIS schemas** to the payment platform's data model
2. **Validate and process payments** against biller-specific rules
3. **Reconcile settlements** across three independent financial systems (biller, platform, bank)
4. **Handle exceptions** — duplicates, mismatches, missing records — that inevitably arise in high-volume payment processing

This simulator models that entire pipeline with realistic data structures, configurable business rules, and synthetic data generation.

## Key Features

### Oracle CC&B Schema Models
Faithful representations of core CC&B tables (`CI_ACCT`, `CI_PER`, `CI_BILL`, `CI_BILL_SEG`, `CI_SA`) with proper field names, status flags, and entity relationships. These models drive the CIS adapter layer.

### YAML-Driven Biller Onboarding
Billers are configured entirely through YAML — payment types, fee schedules, settlement preferences, account validation rules, and notification settings. The onboarding engine validates each configuration against platform requirements before activation.

### Three-Way Settlement Reconciliation
The settlement engine compares records from three sources:
- **Biller records** — what the utility says was paid
- **Platform records** — what the payment platform collected
- **Bank records** — what the acquiring bank actually settled

Transactions are matched on ID, amount (with configurable tolerance), and date. Discrepancies are classified by severity and routed through the exception handling workflow.

### Exception Management
Structured exception handling with:
- Severity classification (critical/high/medium/low)
- Auto-resolution rules for common scenarios (penny rounding, posting lag)
- Configurable escalation chains with role-based routing
- Full audit trail

## Project Structure

```
biller-integration-simulator/
├── config/                    # YAML-driven configuration
│   ├── biller_config.yaml     # Biller onboarding parameters
│   ├── schema_mapping.yaml    # CC&B-to-platform field mappings
│   └── settlement_rules.yaml  # Reconciliation rules & tolerances
├── src/
│   ├── models/                # Data models
│   │   ├── oracle_ccb.py      # CC&B schema (CI_ACCT, CI_PER, CI_BILL, etc.)
│   │   ├── biller.py          # Biller entity & configuration
│   │   ├── payment.py         # Payment transaction lifecycle
│   │   └── settlement.py      # Settlement & reconciliation records
│   ├── core/                  # Processing engines
│   │   ├── onboarding.py      # Biller onboarding validation
│   │   ├── schema_parser.py   # CC&B schema mapping engine
│   │   ├── payment_processor.py   # Payment processing pipeline
│   │   ├── settlement_engine.py   # Three-way reconciliation
│   │   └── exception_handler.py   # Exception workflows
│   ├── integration/           # External system adapters
│   │   ├── cis_adapter.py     # Oracle CC&B CIS interface
│   │   ├── payment_gateway.py # Payment processor connector
│   │   └── notification_service.py  # Event notifications
│   └── utils/                 # Utilities
│       ├── validators.py      # Validation & duplicate detection
│       ├── generators.py      # Synthetic data generation
│       └── logger.py          # Structured JSON logging
├── tests/                     # Test suite
├── examples/                  # Runnable example scripts
└── docs/                      # Architecture documentation
```

## Quick Start

### Installation

```bash
# Clone and install
cd biller-integration-simulator
pip install -r requirements.txt
```

### Run the Examples

Each example generates synthetic data and runs end-to-end:

```bash
# Onboard billers from YAML configuration
python examples/onboard_new_biller.py

# Process a batch of simulated payments
python examples/process_batch_payments.py

# Run three-way settlement reconciliation
python examples/run_settlement.py
```

### CLI

```bash
# Full end-to-end demo
python -m src.main demo

# Onboard all configured billers
python -m src.main onboard

# Process payments for a specific biller
python -m src.main process --biller UTIL-GA-POWER-001 --count 100

# Run settlement for a biller
python -m src.main settle --biller UTIL-GA-POWER-001
```

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

### Biller Configuration (`config/biller_config.yaml`)

Defines each utility biller's integration parameters:

- **Payment types**: one-time, autopay, budget billing, prepay
- **Fee structures**: convenience fees by tender type, late/returned payment fees
- **Settlement**: ACH batch vs. wire, frequency, holdback percentages
- **Validation rules**: account number format, partial/overpayment policies

### Schema Mapping (`config/schema_mapping.yaml`)

Maps Oracle CC&B table columns to the payment platform's canonical data model. Includes transform functions for:

- Leading zero stripping (account IDs)
- Oracle date format conversion
- Status flag decoding via lookup tables
- Decimal precision enforcement

### Settlement Rules (`config/settlement_rules.yaml`)

Governs the reconciliation process:

- Match tolerances (amount, date)
- Exception severity classification
- Auto-resolution rules
- Escalation chains and notification preferences

## Design Decisions

**Dataclasses over ORMs**: The CC&B models use Python dataclasses rather than SQLAlchemy or similar. This keeps the simulation portable and focuses attention on the domain logic rather than persistence mechanics.

**YAML configuration**: Biller parameters change frequently in real deployments (new fee structures, added payment channels, updated validation rules). YAML configs make these changes deployable without code changes.

**Three-way reconciliation**: The settlement engine deliberately models all three sources independently rather than assuming any single source is authoritative. In production, discrepancies between biller, platform, and bank records are a daily reality.

**Structured logging**: Every log entry carries a correlation ID for end-to-end traceability. This is non-negotiable in payment processing — when a customer disputes a charge, you need to reconstruct the complete transaction history.

## Technology

- Python 3.10+
- PyYAML for configuration
- Standard library only (no database, no web framework)
- pytest for testing

## Author

**Ravi Potluru** — Utility billing systems integration and payment platform architecture.
