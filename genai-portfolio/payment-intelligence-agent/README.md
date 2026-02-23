# Payment Intelligence Agent

A Snowflake Cortex-powered conversational agent for payment analytics, PCI DSS compliance Q&A, and transaction anomaly detection. Built as a production-grade Streamlit application with full demo mode support.

## Overview

The Payment Intelligence Agent provides three core capabilities through a natural language chat interface:

- **Payment Analytics (NL-to-SQL)** - Ask questions about transactions, merchants, settlements, and chargebacks in plain English. The agent translates queries to optimized Snowflake SQL, executes them, and presents results with auto-generated charts and natural language summaries.

- **PCI Compliance RAG** - Ask questions about PCI DSS v4.0 requirements. The agent retrieves relevant sections from compliance documentation using vector similarity search and synthesizes accurate, sourced answers.

- **Anomaly Detection** - Request analysis of transaction patterns for fraud indicators. The agent runs statistical detection (Z-score, IQR, isolation forest), pattern analysis (velocity, geographic, temporal), and generates severity-classified alerts.

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone and navigate to the project
cd payment-intelligence-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application (Demo Mode)

```bash
# From the project root
streamlit run src/app.py
```

The app runs in demo mode by default, using synthetic data and local embedding models. No Snowflake connection is required.

### Run Demo Scripts

Each subsystem has a standalone demo that runs without the Streamlit UI:

```bash
# Analytics demo - NL-to-SQL translation
python -m demo.demo_analytics

# Anomaly detection demo
python -m demo.demo_anomaly

# RAG compliance Q&A demo
python -m demo.demo_rag
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
payment-intelligence-agent/
├── config/
│   ├── agent_config.yaml          # Agent orchestration config
│   ├── snowflake_config.yaml      # Snowflake connection (template)
│   └── pci_rules.yaml             # PCI-DSS compliance rules
├── src/
│   ├── app.py                     # Streamlit application
│   ├── agent/                     # Orchestration layer
│   │   ├── orchestrator.py        # Central query routing
│   │   ├── intent_classifier.py   # NL intent classification
│   │   ├── context_manager.py     # Conversation memory
│   │   └── response_formatter.py  # Response & visualization
│   ├── analytics/                 # NL-to-SQL pipeline
│   │   ├── nl_to_sql.py           # Query translation
│   │   ├── query_validator.py     # SQL safety validation
│   │   ├── schema_introspector.py # Schema discovery
│   │   └── result_analyzer.py     # Result summarization
│   ├── rag/                       # Compliance RAG pipeline
│   │   ├── document_loader.py     # Document ingestion
│   │   ├── chunker.py             # Section-aware chunking
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── vector_store.py        # FAISS vector search
│   │   └── retriever.py           # End-to-end retrieval
│   ├── anomaly/                   # Fraud detection
│   │   ├── detector.py            # Detection orchestrator
│   │   ├── statistical.py         # Z-score, IQR, isolation
│   │   ├── pattern_analyzer.py    # Transaction patterns
│   │   └── alert_manager.py       # Alert lifecycle
│   ├── data/
│   │   ├── schema.sql             # Snowflake DDL
│   │   ├── sample_pci_docs/       # PCI DSS documentation
│   │   └── synthetic_generator.py # Synthetic data
│   └── utils/
│       ├── snowflake_client.py    # Connection wrapper
│       ├── logger.py              # Structured logging
│       └── security.py            # Input sanitization, PII masking
├── tests/
├── demo/
└── docs/
    └── architecture.md
```

## Example Queries

### Payment Analytics
- "Show me top 10 merchants by transaction volume last month"
- "What is the average settlement time by payment method?"
- "How many chargebacks occurred last month?"
- "Compare revenue across regions"
- "What is the decline rate by payment method?"
- "Show me the daily transaction volume trend"

### PCI Compliance
- "What are the requirements for storing cardholder data?"
- "How should encryption keys be managed?"
- "What does PCI DSS say about access control?"
- "What are the penetration testing requirements?"

### Anomaly Detection
- "Are there any anomalies in recent transactions?"
- "Show me suspicious transaction patterns"
- "Check for velocity-based anomalies"

## Architecture

The system uses a tiered architecture:

1. **Streamlit Frontend** - Chat-style UI with embedded charts and tables
2. **Agent Orchestrator** - Routes queries via intent classification
3. **Subsystem Pipelines** - Analytics, RAG, and anomaly detection
4. **Snowflake Platform** - Tables, Cortex LLM, Cortex Search (production)

Intent classification uses a three-tier approach (keyword > similarity > LLM) that handles 95%+ of queries without an LLM call.

See [docs/architecture.md](docs/architecture.md) for detailed technical documentation.

## Configuration

### Snowflake Connection (Production)

Copy the config template and fill in credentials:

```bash
cp config/snowflake_config.yaml config/snowflake_config.local.yaml
```

Set environment variables:
```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_user"
export SNOWFLAKE_PASSWORD="your_password"
```

### Agent Behavior

Edit `config/agent_config.yaml` to adjust:
- Intent classification thresholds and keywords
- Response formatting options
- SQL safety guardrails
- Conversation context settings

## Security

- **Input sanitization** with HTML stripping and SQL injection detection
- **PII masking** for credit cards, SSNs, emails, phones, and account numbers
- **Query validation** with statement type enforcement and operation blocklists
- **Schema boundary** checks prevent queries outside the payment schema
- Credentials are never stored in code; use environment variables

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Snowflake Cortex (mistral-large2) |
| Embeddings | sentence-transformers / Cortex EMBED |
| Vector Search | FAISS / Cortex Search |
| Visualization | Plotly |
| Data Processing | pandas, numpy, scipy, scikit-learn |
| SQL Parsing | sqlparse |
| Logging | structlog |

## License

MIT
