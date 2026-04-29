# API Reference

The Payment Intelligence Agent exposes its capabilities through three interfaces:

1. **Programmatic API** - The `Orchestrator.query()` method (Python)
2. **Streamlit Web UI** - Chat-style interface at `http://localhost:8501`
3. **Subsystem APIs** - Direct access to NL-to-SQL, RAG, and Anomaly Detection

This document covers all three.

---

## 1. Orchestrator API

The orchestrator is the top-level entry point that routes user queries to the correct subsystem.

### `Orchestrator.query()`

```python
from src.agent.orchestrator import Orchestrator

orch = Orchestrator(config_path="config/agent_config.yaml")
response = orch.query(
    user_input="Show me top 10 merchants by transaction volume last month",
    session_id="user-abc-123",
)
```

#### Input Schema

| Field         | Type   | Required | Description                                         |
|---------------|--------|----------|-----------------------------------------------------|
| `user_input`  | `str`  | Yes      | Natural language query from the user                |
| `session_id`  | `str`  | No       | Conversation context key. Defaults to a fresh session |
| `mode`        | `str`  | No       | Force routing: `"analytics"`, `"compliance"`, `"anomaly"`, or `"auto"` (default) |

#### Output Schema

```python
{
    "intent": "analytics",                    # Resolved intent
    "confidence": 0.94,                       # Intent classifier confidence
    "subsystem": "nl_to_sql",                 # Subsystem that handled the query
    "response": {
        "type": "tabular",                    # "tabular" | "text" | "alert_list" | "rag_answer"
        "summary": "Top 10 merchants...",     # Natural-language summary
        "data": [...],                        # List of dicts (rows) for tabular
        "sql": "SELECT ...",                  # Generated SQL (analytics only)
        "chart_spec": {...}                   # Plotly chart spec (optional)
    },
    "session_id": "user-abc-123",
    "elapsed_ms": 412,
    "warnings": []                            # Any non-fatal issues
}
```

---

## 2. Intent Classification

Intent is determined via a 3-tier classifier in `src/agent/intent_classifier.py`:

1. **Keyword matching** - High-precision rules
2. **Jaccard similarity** - Against curated example queries
3. **Context-aware follow-up detection** - Inherits intent from prior turn

### Supported Intent Keywords

| Intent       | Keywords / Phrases                                                       |
|--------------|--------------------------------------------------------------------------|
| `analytics`  | `top`, `count`, `revenue`, `volume`, `trend`, `merchants`, `chargeback rate`, `settlement time`, `decline rate`, `breakdown`, `by region`, `last month`, `last 30 days` |
| `compliance` | `PCI`, `requirement`, `cardholder data`, `encryption key`, `access control`, `vulnerability`, `incident response`, `audit`, `compliance` |
| `anomaly`    | `anomaly`, `anomalies`, `fraud`, `unusual`, `suspicious`, `outlier`, `velocity check`, `detect`, `alert` |

If keyword matching returns no signal, the classifier falls back to Jaccard similarity over a curated set of example queries per intent.

---

## 3. NL-to-SQL Subsystem

Translates natural-language analytics questions into Snowflake-compatible SQL.

### Programmatic API

```python
from src.analytics.nl_to_sql import NLToSQLTranslator

translator = NLToSQLTranslator()
result = translator.translate("Show top 10 merchants by transaction volume last month")
```

#### Output Schema

```python
{
    "template_key": "top_merchants_by_volume",     # Matched template (or None for generic)
    "sql": "SELECT m.merchant_id, m.name, ...",   # Generated SQL
    "params": {                                    # Parameterized values (no string interpolation)
        "limit": 10,
        "start_date": "2024-09-01",
        "end_date": "2024-09-30"
    },
    "warnings": []                                 # Validation warnings
}
```

### Pre-Built Query Templates

Located in `src/analytics/nl_to_sql.py::_QUERY_TEMPLATES`:

| Template Key                     | Triggers (Example Phrasing)                                |
|----------------------------------|------------------------------------------------------------|
| `top_merchants_by_volume`        | "top N merchants by transaction volume"                    |
| `settlement_time_by_method`      | "average settlement time by payment method"                |
| `chargeback_summary`             | "chargeback summary", "chargeback rate"                    |
| `revenue_by_region`              | "revenue by region/country/state"                          |
| `decline_rate_by_method`         | "decline rate", "failure rate by payment method"           |
| `daily_volume_trend`             | "daily volume trend", "transactions over time"             |
| `merchant_chargeback_rate`       | "merchants with highest chargeback rate"                   |
| `payment_method_breakdown`       | "breakdown by payment method"                              |

### SQL Safety Validation

Every generated SQL statement is validated by `query_validator.py` before execution. The validator enforces:

1. **Statement type allowlist** - only `SELECT` is permitted
2. **Blocked keywords** - `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, `CREATE`, `GRANT`, `REVOKE`, `EXEC`, `CALL`, `MERGE`, `COPY` are rejected
3. **Injection pattern detectors** - 8 patterns including stacked queries, comment-based escapes, and union-based attacks
4. **Function blocklist** - blocks `LOAD_FILE`, `INTO OUTFILE`, `pg_sleep`, etc.
5. **Subquery depth limit** - max 5 nested levels
6. **Join count limit** - max 8 joins per query

If validation fails, the orchestrator returns a `ValidationError` to the user without executing the query.

---

## 4. RAG (PCI Compliance) Subsystem

Retrieves and synthesizes answers from PCI DSS v4.0 documentation.

### Programmatic API

```python
from src.rag.retriever import RAGRetriever

retriever = RAGRetriever(docs_dir="src/data/sample_pci_docs/")
retriever.initialize()  # Loads, chunks, and embeds documents

answer = retriever.query("How should encryption keys be managed?")
```

#### Output Schema

```python
{
    "answer": "Encryption keys must be stored separately from...",
    "sources": [
        {
            "doc": "pci_dss_requirements.md",
            "section": "Requirement 3: Protect stored cardholder data",
            "subsection": "3.5 Document and implement procedures...",
            "score": 0.87,                    # Cosine similarity
            "snippet": "..."
        },
        ...
    ],
    "pci_requirement_ids": ["3.5", "3.6", "3.7"],   # Auto-extracted requirement IDs
    "confidence": "high"                              # "high" | "medium" | "low"
}
```

### Document Format

PCI documents must be markdown with the following structure to be properly parsed:

```markdown
# Document Title

## Requirement N: <Title>

### N.M <Sub-requirement title>

Body text. Reference IDs in the form `N.M.K` are auto-extracted.

### N.M+1 <Next sub-requirement>
...
```

The `document_loader.py` extracts headings as section/subsection metadata. The `chunker.py` splits long sections at paragraph boundaries with configurable overlap (default 100 chars).

### Embedding Generation

Embeddings are produced by:

1. **Snowflake Cortex** (`EMBED_TEXT_768`) when `SNOWFLAKE_ACCOUNT` env is set
2. **sentence-transformers** (`all-MiniLM-L6-v2`) as a local fallback
3. **Hash-based pseudo-embeddings** as a deterministic fallback for tests

All paths are wrapped behind `embeddings.py::EmbeddingGenerator.embed()`.

---

## 5. Anomaly Detection Subsystem

Multi-method ensemble detector with severity classification.

### Programmatic API

```python
from src.anomaly.detector import AnomalyDetector
from src.anomaly.alert_manager import AlertManager

detector = AnomalyDetector()
alerts = detector.detect(transactions_df)

am = AlertManager()
am.ingest(alerts)
critical = am.get_alerts(severity_filter=Severity.CRITICAL)
```

### Alert JSON Schema

```json
{
  "alert_id": "alrt_8a4f2c1d",
  "severity": "critical",
  "score": 0.94,
  "category": "velocity_violation",
  "transaction_ids": ["txn_abc123", "txn_abc124", "txn_abc125"],
  "merchant_id": "merch_42",
  "detected_at": "2024-09-15T14:32:11Z",
  "method": "pattern_analyzer",
  "summary": "8 transactions in 60 seconds from merchant merch_42",
  "details": {
    "transaction_count": 8,
    "time_window_seconds": 60,
    "total_amount": 4250.00,
    "velocity_threshold": 5,
    "geography": "US-CA"
  },
  "fingerprint": "vel:merch_42:14:32",
  "acknowledged": false,
  "acknowledged_by": null,
  "acknowledged_at": null
}
```

### Severity Levels

| Severity   | Score Threshold | Response SLA | Example Triggers                                           |
|------------|-----------------|--------------|------------------------------------------------------------|
| `critical` | >= 0.90         | 15 minutes   | Velocity > 10 txn/min, multi-region same card in <5 min, declined-then-approved patterns |
| `high`     | >= 0.75         | 1 hour       | Z-score > 4 on amount, isolation forest score > 0.7        |
| `medium`   | >= 0.50         | 4 hours      | IQR outliers, unusual time-of-day patterns                 |
| `low`      | >= 0.30         | 24 hours     | Marginal Z-score deviations, single-instance amount outliers |

Below `low_threshold` (default 0.30), no alert is emitted.

### Detection Methods

| Method               | File                              | Use Case                                                   |
|----------------------|-----------------------------------|------------------------------------------------------------|
| Z-score              | `statistical.py::ZScoreDetector`  | Univariate amount outliers, fast                           |
| Modified Z-score (MAD)| `statistical.py::ModifiedZScoreDetector` | Robust to extreme values; better for heavy-tailed amount distributions |
| IQR fence            | `statistical.py::IQRDetector`     | Distribution-free; good baseline                           |
| Isolation Forest     | `statistical.py::IsolationForestDetector` | Multi-feature anomalies (amount + time + location)         |
| Velocity check       | `pattern_analyzer.py::VelocityChecker` | Rapid-fire transactions from same card/merchant            |
| Geographic anomaly   | `pattern_analyzer.py::GeoChecker` | Same card used in distant locations within short timeframe |
| Temporal anomaly     | `pattern_analyzer.py::TemporalChecker` | Transactions outside merchant's normal hours               |

### Alert Deduplication

Alerts with the same `fingerprint` within a configurable window (default 5 minutes) are coalesced into a single alert with an updated `transaction_ids` list. This prevents alert fatigue during sustained attacks.

---

## 6. Streamlit Chat Interface

The Streamlit app (`src/app.py`) provides a chat UI at `http://localhost:8501`.

### Session State Keys

| Key                  | Type                  | Purpose                                          |
|----------------------|-----------------------|--------------------------------------------------|
| `messages`           | `list[dict]`          | Conversation history (role/content/metadata)    |
| `session_id`         | `str`                 | Unique session identifier                        |
| `mode`               | `str`                 | Current routing mode (set by sidebar selector)  |
| `orchestrator`       | `Orchestrator`        | Initialized orchestrator instance (cached)      |
| `last_response`      | `dict`                | Most recent orchestrator response                |
| `chart_history`      | `list[Figure]`        | Plotly figures rendered in chat                  |

### Routes

The app is single-page; navigation is via the sidebar:

- **Chat** (default) - Conversational interface
- **Analytics Mode** - Forces all queries to NL-to-SQL
- **Compliance Mode** - Forces all queries to RAG
- **Anomaly Mode** - Forces all queries to anomaly detection
- **About** - Project info and configuration

### Health Endpoint

Streamlit exposes a built-in health check at `http://localhost:8501/_stcore/health`. Used by the Docker `HEALTHCHECK` directive.

---

## Configuration

All subsystems read from `config/agent_config.yaml`. See that file for:

- Embedding model selection
- Vector index parameters
- Anomaly thresholds
- NL-to-SQL row limits
- Streamlit theming

Environment variables override YAML values where supported. See `.env.example` for the complete list.

---

## Error Codes

| Code | HTTP Equivalent | Meaning                                         |
|------|-----------------|-------------------------------------------------|
| `E_INTENT_AMBIGUOUS` | 400 | Could not classify intent above confidence threshold |
| `E_SQL_VALIDATION`   | 400 | Generated SQL failed safety validation         |
| `E_NO_RESULTS`       | 200 | Query executed but returned 0 rows             |
| `E_RAG_NOT_INIT`     | 503 | RAG retriever not yet initialized              |
| `E_SNOWFLAKE_AUTH`   | 401 | Snowflake credentials invalid or missing       |
| `E_RATE_LIMIT`       | 429 | User exceeded query rate limit                 |
| `E_INTERNAL`         | 500 | Unhandled exception (see logs for details)     |

All error responses include a human-readable `message` field and a `request_id` for log correlation.
