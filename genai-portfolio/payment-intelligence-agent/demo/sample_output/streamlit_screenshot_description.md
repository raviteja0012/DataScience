# Streamlit UI - Visual Reference

This document describes what a reviewer would see when launching the
Payment Intelligence Agent Streamlit app via:

```bash
streamlit run src/app.py
```

Because the repository ships only source code (no PNG screenshots), this
file serves as the visual reference until you can launch the app yourself.

---

## Top Navigation

Tabs across the top bar:

```
[ Home ]   [ Analytics ]   [ Anomaly Detection ]   [ Compliance Q&A ]   [ Settings ]
```

The active tab is highlighted blue. All tabs share a common header strip
that shows:

- App title: "Payment Intelligence Agent"
- Connection status badge: "Connected to ANALYTICS_WH" (green) when a
  Snowflake session is active, "Demo mode (synthetic data)" (amber)
  otherwise.
- A "Last refreshed" timestamp.

---

## Tab 1: Home (default landing)

A two-column layout:

- Left (60%): four KPI cards stacked horizontally
    - "Total Volume (30d)" - large number, e.g. `$42.1M`, with a small
      sparkline showing the 30-day trend.
    - "Approval Rate" - percentage with a delta vs. prior period.
    - "Active Alerts" - count, color-coded by max severity.
    - "Open Compliance Issues" - count from RAG-backed audits.
- Right (40%): a "Recent Activity" feed listing the last 8 events:
  alerts created, queries run, settlement batches closed.

Below the cards: a Plotly line chart "Daily transaction volume - last 30
days" with weekend dips clearly visible.

---

## Tab 2: Analytics (NL-to-SQL)

Layout:

- A prominent text box at the top labelled
  "Ask a question about your payments data".
- Below it, a "Run query" button and a "Recent questions" expander.

After the user types a question (e.g. "Top 10 merchants by volume last
month") and clicks Run:

1. A status bar shows progressive stages: `Parsing -> SQL gen -> Validating -> Executing -> Summarizing`.
2. A collapsible "Generated SQL" panel appears, syntax-highlighted, with
   a "Copy" button and validation badge ("Safe" green or "Blocked" red).
3. A results table rendered via `st.dataframe` with sortable columns.
4. A natural-language summary block (bordered, light-blue background)
   restating the result in business terms.
5. An "Explain this further" follow-up prompt below the results.

See `analytics_session.txt` and `analytics_session.json` in this directory
for the actual content of these panels.

---

## Tab 3: Anomaly Detection

Layout:

- A sidebar (left) with filters: severity, category, date range, status.
- Main area splits into:
    - Top: a stacked bar chart "Alerts by severity over time".
    - Middle: a paginated alert table sortable by score, severity, age.
    - Bottom: a detail pane that opens when an alert is clicked, showing
      the full alert JSON, recommended actions, and an "Acknowledge" /
      "Dismiss" / "Escalate" button row.

See `anomaly_alerts.json` for the schema of each alert row.

---

## Tab 4: Compliance Q&A (RAG)

Layout:

- A search-style input box: "Ask a PCI DSS or compliance question".
- Below it, after submission:
    - A "Answer" card with the synthesized answer (markdown rendered).
    - A "Relevant Requirements" chip row (e.g. `Req 3.5`, `Req 8.4`).
    - A "Sources" panel with each cited chunk: document name, section,
      similarity score, and a short snippet.
    - A "Verify in original document" link that scrolls the document
      viewer (right pane) to the cited section.

See `rag_query_session.txt` for an example of the answer + sources
content for six representative compliance queries.

---

## Tab 5: Settings

A simple form covering:

- Snowflake connection (account, warehouse, database, schema, role)
- Detection thresholds (z-score, IQR multiplier, velocity rules)
- Retrieval parameters (top-k, chunk size, chunk overlap)
- Logging level

A "Save" button persists changes via `config/app_config.yaml`.

---

## Color Palette and Style

- Background: white with subtle gray panels.
- Primary accent: blue (#1f77b4) - matches Plotly default.
- Severity colors:
    - critical: `#d62728` (red)
    - high: `#ff7f0e` (orange)
    - medium: `#bcbd22` (yellow-olive)
    - low: `#2ca02c` (green)
- Monospace blocks (SQL, JSON, log): `#f7f7f9` background.

---

## Responsive Behavior

- Below 800px width, the tab bar collapses into a hamburger menu.
- KPI cards stack vertically.
- Charts use `use_container_width=True` so they always fill horizontally.
