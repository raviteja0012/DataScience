# GenAI Portfolio

[![CI](https://github.com/raviteja0012/DataScience/actions/workflows/ci.yml/badge.svg)](https://github.com/raviteja0012/DataScience/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/raviteja0012/DataScience/branch/main/graph/badge.svg)](https://codecov.io/gh/raviteja0012/DataScience)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

A collection of three production-oriented GenAI and data engineering projects demonstrating end-to-end system design across utility billing integration, conversational AI analytics, and large-scale data migration. Each project emphasizes testability, clean architecture, and real-world domain complexity -- from Oracle CC&B schema modeling and PCI-DSS compliance to M&A identity resolution and checkpoint/restart pipelines.

---

## Projects at a Glance

| Project | Description | Key Tech | Tests |
|---------|-------------|----------|------:|
| [Biller Integration Simulator](biller-integration-simulator/) | Utility CIS-to-payment platform integration simulation | Python 3.11, dataclasses, pyyaml, structlog, pytest | 53 |
| [Payment Intelligence Agent](payment-intelligence-agent/) | Snowflake Cortex-powered conversational analytics agent | Streamlit, Plotly, FAISS, sentence-transformers, NumPy, pandas | 101 |
| [Multi-Source Data Integration Framework](multi-source-data-integration/) | M&A data migration pipeline with identity resolution | pandas, NumPy, Jinja2, pyyaml | 86 |

**Total test coverage: 240 tests**

---

## Biller Integration Simulator

**`biller-integration-simulator/`** -- Utility CIS-to-payment platform integration simulation

### Architecture Highlights

- **Oracle CC&B Schema Models** -- Faithfully represents core Customer Information System entities (`CI_ACCT`, `CI_PER`, `CI_BILL`) using Python dataclasses, enabling realistic billing simulation without a live database.
- **YAML-Driven Biller Onboarding** -- Biller configurations are declared in YAML, making it straightforward to add new billers, modify fee structures, or adjust settlement rules without code changes.
- **Three-Way Settlement Reconciliation** -- Matches payment records across biller, platform, and bank channels to surface discrepancies and confirm settlement integrity.
- **Exception Handling** -- Structured logging via `structlog` provides traceable diagnostics for failed transactions, mismatched amounts, and onboarding validation errors.

---

## Payment Intelligence Agent

**`payment-intelligence-agent/`** -- Snowflake Cortex-powered conversational agent for payment analytics

### Architecture Highlights

- **NL-to-SQL Analytics** -- Translates natural language questions into Snowflake-compatible SQL with built-in SQL injection defense, allowing non-technical users to query payment data conversationally.
- **PCI-DSS v4.0 RAG** -- Retrieval-augmented generation backed by a FAISS vector index and sentence-transformer embeddings, providing accurate compliance guidance grounded in PCI-DSS v4.0 documentation.
- **Statistical Anomaly Detection** -- Combines Z-score, IQR, and isolation forest methods to flag unusual payment patterns, giving analysts multiple lenses on potential fraud or system errors.
- **Streamlit Chat UI** -- Interactive chat interface with Plotly visualizations for inline charting of query results and anomaly distributions.

---

## Multi-Source Data Integration Framework

**`multi-source-data-integration/`** -- M&A data migration pipeline

### Architecture Highlights

- **Source Discovery and Profiling** -- Automatically catalogs source systems and generates column-level statistics to inform mapping decisions before any data moves.
- **Schema Mapping (Oracle/SQL Server to Snowflake)** -- Declarative mapping layer that translates heterogeneous source schemas into a unified Snowflake target model, handling data type conversions and naming conventions.
- **Identity Resolution** -- Five matching algorithms (exact, Levenshtein, Jaro-Winkler, Soundex, Metaphone) work in concert to deduplicate and link records across acquired entities, each tunable by confidence threshold.
- **Checkpoint/Restart Cutover** -- Long-running migration jobs persist progress at configurable intervals, allowing restart from the last successful checkpoint after failures rather than reprocessing from scratch.
- **Reconciliation with HTML Reporting** -- Count-level, value-level, and row-level reconciliation checks run automatically post-migration, producing Jinja2-templated HTML reports for stakeholder review.

---

## Quick Start

Each project is self-contained with its own dependencies. To get started with any project:

```bash
# Clone the repository
git clone <repository-url>
cd genai-portfolio

# Navigate to a project
cd biller-integration-simulator   # or payment-intelligence-agent, multi-source-data-integration

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the project (each project includes a main entry point)
python -m src.main
```

> Refer to the README inside each project directory for project-specific setup instructions and configuration details.

---

## Testing

All three projects use **pytest**. To run the full suite of 240 tests:

```bash
# Run all tests across every project from the portfolio root
cd genai-portfolio

# Biller Integration Simulator (53 tests)
cd biller-integration-simulator && python -m pytest tests/ -v && cd ..

# Payment Intelligence Agent (101 tests)
cd payment-intelligence-agent && python -m pytest tests/ -v && cd ..

# Multi-Source Data Integration Framework (86 tests)
cd multi-source-data-integration && python -m pytest tests/ -v && cd ..
```

Or use the Makefile for convenience:

```bash
make test              # Run all 240 tests
make coverage          # Run all tests with coverage report
make coverage-biller   # Coverage for a single project (with HTML report)
```

### Coverage

Each project targets **80%+ line coverage**. Coverage is reported in CI via [Codecov](https://codecov.io/gh/raviteja0012/DataScience) and can be generated locally:

```bash
cd biller-integration-simulator
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov
open htmlcov/index.html
```
