# Sample Output - Payment Intelligence Agent

This directory contains representative example output from the three demo
scripts in `demo/`, plus a textual reference for the Streamlit UI.
Reviewers can read these files to understand what the agent produces
without running any code.

## Files

| File | Source | What it shows |
| --- | --- | --- |
| `analytics_session.txt` | `demo/demo_analytics.py` | Six natural-language queries translated to Snowflake SQL. Each entry shows the matched template, generated SQL, validation pass, sample result rows, and an LLM-style natural-language summary. |
| `analytics_session.json` | `demo/demo_analytics.py` | Structured JSON variant of the same session. The schema is what the analytics pipeline returns programmatically: `query`, `sql`, `validation`, `results`, `summary`, plus per-query execution metadata. |
| `anomaly_alerts.json` | `demo/demo_anomaly.py` | Nine representative anomaly alerts spanning critical / high / medium / low severities and the four detection categories (statistical, velocity, fraud_pattern, time_series). Includes the full alert schema with recommended actions and pattern-violation cross-references. |
| `rag_query_session.txt` | `demo/demo_rag.py` | Six PCI DSS compliance questions answered via the RAG pipeline. Shows the synthesized answer, relevant requirement IDs, and the top-3 source chunks with similarity scores. |
| `streamlit_screenshot_description.md` | `streamlit run src/app.py` | Visual reference for the five Streamlit tabs (Home, Analytics, Anomaly Detection, Compliance Q&A, Settings) covering layout, components, color palette, and responsive behavior. |

## Reading Order

1. `analytics_session.txt` - intuition for the NL-to-SQL pipeline.
2. `analytics_session.json` - the structured shape returned by the API.
3. `anomaly_alerts.json` - the alert schema and a realistic distribution
   of anomaly types.
4. `rag_query_session.txt` - the compliance Q&A surface area.
5. `streamlit_screenshot_description.md` - how all of the above are
   surfaced in the interactive UI.

## Reproducing

```bash
./quickstart.sh
```

The quickstart installs dependencies, runs the three demo scripts, and
captures their output into `output/quickstart/`. After it finishes, run

```bash
streamlit run src/app.py
```

to open the interactive UI described in `streamlit_screenshot_description.md`.

## Notes

- All merchant names, transaction IDs, and amounts in these samples are
  synthetic. They were designed to look realistic and to exercise every
  branch of the pipeline (template-matched and custom queries; all four
  alert severities; all detection categories; multiple compliance
  requirement domains).
- Real demo runs against the synthetic data generator will produce
  slightly different numbers, but the structure (templates, validation
  results, alert schema, citation format) will match these samples.
- The Streamlit app runs in "demo mode" with synthetic data when
  Snowflake credentials are not configured.
