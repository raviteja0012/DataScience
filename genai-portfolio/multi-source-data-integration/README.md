# Multi-Source Data Integration Framework

A production-quality, configurable Python pipeline for integrating data from acquired company systems into a consolidated enterprise data warehouse. Built to handle the real-world complexity of M&A data migrations: heterogeneous source schemas, inconsistent data quality, cross-system identity resolution, and strict reconciliation requirements.

## Overview

When a company acquires another, integrating their data systems is one of the most complex and high-stakes technical challenges. This framework addresses the full lifecycle:

1. **Discovery** -- Profile source systems, assess data quality, and map dependencies
2. **Mapping** -- Transform heterogeneous schemas into a unified target model
3. **Identity Resolution** -- Match and merge duplicate entities across systems
4. **Cutover** -- Execute the migration with parallel loading, checkpoints, and rollback
5. **Reconciliation** -- Validate data integrity with count, value, and row-level checks

## Architecture

```
Source Systems                    Target
+------------------+
| Oracle 12c (ERP) |---+
+------------------+   |    +----------+    +-----------+    +----------+    +-----------+
                       +--->| Discovery|--->| Mapping   |--->| Identity |--->| Cutover   |
+------------------+   |    | & Profile|    | Engine    |    | Resolver |    | Manager   |
| SQL Server (CRM) |---+    +----------+    +-----------+    +----------+    +-----------+
+------------------+              |                                               |
                                  |         +----------------+                    |
                                  +-------->| Reconciliation |<-------------------+
                                            | Validator      |
                                            +----------------+
                                                   |
                                            +------+------+
                                            | HTML Report  |
                                            +-------------+
```

## Key Features

### Source System Discovery
- Statistical profiling: row counts, column types, null percentages, cardinality
- Data quality scoring across four dimensions: completeness, uniqueness, validity, consistency
- Foreign key dependency mapping with DAG-based load ordering
- Schema structure analysis with automated column-matching heuristics

### Schema Mapping Engine
- YAML-driven declarative mappings between source and target schemas
- 18+ transform types: direct, concatenation, splitting, lookup, type conversion, normalization
- Cross-platform type conversion (Oracle to Snowflake, SQL Server to Snowflake)
- Mapping completeness validation with coverage reporting

### Identity Resolution
- **Multiple matching strategies**: Exact, Levenshtein, Jaro-Winkler, Soundex, Metaphone
- **Configurable weighted scoring** with per-strategy thresholds
- **Blocking** to reduce O(n^2) comparison space
- **Survivorship rules**: newest wins, most complete wins, source priority, per-field overrides
- **Golden record creation** with field-level lineage tracking
- **Cross-source and within-source deduplication**

### Cutover Management
- Big-bang and phased cutover strategies
- Parallel load execution with configurable concurrency
- Checkpoint/restart for failed loads
- Before-state snapshots with rollback capabilities
- Retry logic with exponential backoff

### Reconciliation Validation
- Row count reconciliation with dedup-aware tolerances
- Aggregate value checks (SUM, COUNT, MIN/MAX, distribution)
- SHA-256 hash-based row-level comparison
- Detailed HTML reports with drill-down into discrepancies

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline demo
python examples/run_full_pipeline.py

# Run individual demos
python examples/run_discovery.py
python examples/run_identity_resolution.py
python examples/run_reconciliation.py

# Run tests
pytest tests/ -v

# Use the CLI
python -m src.main --phase all
python -m src.main --phase discovery
python -m src.main --phase identity_resolution
```

## Configuration

All pipeline behavior is driven by YAML configuration:

| File | Purpose |
|------|---------|
| `config/pipeline_config.yaml` | Main pipeline settings, phase config, thresholds |
| `config/source_systems/*.yaml` | Source/target system connection and schema metadata |
| `config/mapping_rules/*.yaml` | Entity-level source-to-target column mappings |
| `config/validation_rules.yaml` | Post-migration reconciliation check definitions |

### Example: Customer Mapping

```yaml
# Oracle CUST_MASTER -> Snowflake DIM_CUSTOMER
- source_column: "CUST_NM"
  target_column: "CUSTOMER_NAME"
  transform: "TRIM_UPPER_FIRST"

- source_columns: ["CUST_ADDR_1", "CUST_ADDR_2"]
  target_column: "FULL_ADDRESS"
  transform: "CONCATENATE"
  separator: ", "
  skip_null: true

- source_column: "CUST_TYPE"
  target_column: "CUSTOMER_TYPE"
  transform: "LOOKUP"
  lookup_map:
    "B": "Business"
    "I": "Individual"
```

## Project Structure

```
multi-source-data-integration/
├── config/                     # YAML-driven configuration
│   ├── pipeline_config.yaml
│   ├── source_systems/         # Oracle, SQL Server, Snowflake definitions
│   ├── mapping_rules/          # Customer, product, transaction mappings
│   └── validation_rules.yaml
├── src/
│   ├── main.py                 # Pipeline orchestrator & CLI
│   ├── discovery/              # Profiling, quality scanning, dependency mapping
│   ├── mapping/                # Schema mapping, transforms, type conversion
│   ├── identity/               # Matching strategies, merge rules, dedup engine
│   ├── cutover/                # Strategy planning, parallel execution, rollback
│   ├── reconciliation/         # Count/value/hash validation, HTML reports
│   └── utils/                  # Logging, metrics, database connectors, generators
├── tests/                      # Pytest test suite
├── examples/                   # Runnable demos with synthetic data
└── docs/                       # Architecture documentation
```

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test modules
pytest tests/test_identity_resolver.py -v
pytest tests/test_reconciliation.py -v
```

## Technology Stack

- **Python 3.9+** -- Core implementation
- **PyYAML** -- Configuration management
- **SQLAlchemy** -- Database abstraction (optional, for live connections)
- **pytest** -- Testing framework

## Design Decisions

1. **YAML-driven configuration** -- All mapping rules, thresholds, and system metadata are externalized to YAML, enabling non-developer operators to modify behavior without code changes.

2. **Synthetic data generators** -- Every example runs entirely self-contained using realistic synthetic data, including intentional cross-source overlaps and data quality issues.

3. **Pluggable matching strategies** -- The identity resolution engine uses a strategy pattern, making it straightforward to add new matching algorithms.

4. **Phase independence** -- Each pipeline phase (discovery, mapping, identity, cutover, reconciliation) can run independently, supporting iterative development and debugging.

5. **Checkpoint/restart** -- Long-running migrations can be interrupted and resumed from the last successful checkpoint.
