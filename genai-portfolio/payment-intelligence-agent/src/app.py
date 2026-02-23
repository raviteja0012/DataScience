"""Payment Intelligence Agent - Streamlit Application.

Conversational interface for payment analytics, PCI compliance Q&A,
and anomaly detection powered by Snowflake Cortex Agent orchestration.

Run with: streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the Python path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import pandas as pd

from src.agent.orchestrator import PaymentAgentOrchestrator, OrchestratorConfig
from src.agent.response_formatter import ResponseType


# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Payment Intelligence Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .sql-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    .alert-critical {
        background-color: #fde8e8;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #fd7e14;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .alert-medium {
        background-color: #fffde7;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    div[data-testid="stChatMessage"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialize all session state variables."""
    if "orchestrator" not in st.session_state:
        config = OrchestratorConfig(
            demo_mode=True,
            show_sql=True,
            max_display_rows=100,
        )
        orchestrator = PaymentAgentOrchestrator(config=config)
        orchestrator.initialize_rag()
        st.session_state.orchestrator = orchestrator

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "auto"


_init_session_state()


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Payment Intelligence Agent")
    st.caption("Snowflake Cortex-Powered Analytics")

    st.divider()

    # Mode Selection
    st.subheader("Analysis Mode")
    mode = st.radio(
        "Route queries to:",
        options=["auto", "analytics", "compliance", "anomaly"],
        format_func=lambda x: {
            "auto": "Auto-detect (recommended)",
            "analytics": "Payment Analytics",
            "compliance": "PCI Compliance",
            "anomaly": "Anomaly Detection",
        }[x],
        index=0,
        key="mode_radio",
    )
    st.session_state.active_mode = mode

    st.divider()

    # Configuration
    st.subheader("Settings")
    show_sql = st.toggle("Show generated SQL", value=True, key="show_sql")
    show_charts = st.toggle("Show visualizations", value=True, key="show_charts")
    max_rows = st.slider("Max display rows", 10, 500, 100, key="max_rows")

    st.divider()

    # Quick Actions
    st.subheader("Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.orchestrator.reset_session()
            st.rerun()
    with col2:
        if st.button("New Session", use_container_width=True):
            st.session_state.messages = []
            config = OrchestratorConfig(
                demo_mode=True,
                show_sql=show_sql,
                max_display_rows=max_rows,
            )
            st.session_state.orchestrator = PaymentAgentOrchestrator(config=config)
            st.session_state.orchestrator.initialize_rag()
            st.rerun()

    st.divider()

    # Example Queries
    st.subheader("Example Queries")

    example_queries = {
        "Analytics": [
            "Show me top 10 merchants by transaction volume last month",
            "What is the average settlement time by payment method?",
            "Show the daily transaction volume trend",
            "What is the decline rate by payment method?",
        ],
        "Compliance": [
            "What are the requirements for storing cardholder data?",
            "How should encryption keys be managed?",
            "What does PCI DSS say about access control?",
            "What are the penetration testing requirements?",
        ],
        "Anomaly Detection": [
            "Are there any anomalies in recent transactions?",
            "Show me suspicious transaction patterns",
            "Check for velocity-based anomalies",
            "What merchants have unusual activity?",
        ],
    }

    for category, queries in example_queries.items():
        with st.expander(category, expanded=False):
            for q in queries:
                if st.button(q, key=f"example_{q[:30]}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

    st.divider()

    # Session Info
    with st.expander("Session Info", expanded=False):
        session_summary = st.session_state.orchestrator.get_session_summary()
        st.json(session_summary)

    st.caption("Running in demo mode with synthetic data")


# ──────────────────────────────────────────────────────────────
# Main Chat Interface
# ──────────────────────────────────────────────────────────────

# Header
st.markdown("## Payment Intelligence Agent")
st.markdown(
    "Ask questions about payment analytics, PCI compliance, "
    "or transaction anomalies."
)

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render rich content for assistant messages
        if message["role"] == "assistant" and "response_data" in message:
            response_data = message["response_data"]
            _render_rich_content(response_data) if callable(globals().get("_render_rich_content")) else None


def _render_rich_content(response_data: dict) -> None:
    """Render charts, tables, and alerts from response data."""
    show_charts_setting = st.session_state.get("show_charts", True)
    show_sql_setting = st.session_state.get("show_sql", True)

    # SQL display
    if show_sql_setting and response_data.get("sql"):
        with st.expander("Generated SQL", expanded=False):
            st.code(response_data["sql"], language="sql")

    # Chart display
    if show_charts_setting and response_data.get("chart") is not None:
        st.plotly_chart(response_data["chart"], use_container_width=True)

    # Table display
    if response_data.get("dataframe") is not None:
        df = response_data["dataframe"]
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)

    # Alert display
    if response_data.get("alerts"):
        for alert in response_data["alerts"]:
            severity = alert.get("severity", "medium")
            css_class = f"alert-{severity}"
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>[{severity.upper()}]</strong> {alert.get("message", "")}'
                f'</div>',
                unsafe_allow_html=True,
            )


# Process new messages (handles both typed input and example button clicks)
needs_response = (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and (
        len(st.session_state.messages) < 2
        or st.session_state.messages[-2]["role"] != "assistant"
        or st.session_state.messages[-1].get("_processed") is not True
    )
)

if needs_response:
    last_user_message = st.session_state.messages[-1]
    if not last_user_message.get("_processed"):
        last_user_message["_processed"] = True

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                orchestrator: PaymentAgentOrchestrator = st.session_state.orchestrator

                # Update config from sidebar
                orchestrator.config.show_sql = st.session_state.get("show_sql", True)
                orchestrator.config.max_display_rows = st.session_state.get("max_rows", 100)
                orchestrator._formatter.show_sql = orchestrator.config.show_sql
                orchestrator._formatter.max_display_rows = orchestrator.config.max_display_rows

                response = orchestrator.process_query(last_user_message["content"])

            # Display text response
            st.markdown(response.text)

            # Build response data for rendering
            response_data = {
                "sql": response.sql,
                "chart": response.chart,
                "dataframe": response.dataframe,
                "alerts": response.alerts,
                "response_type": response.response_type.value,
            }

            _render_rich_content(response_data)

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.text,
                "response_data": response_data,
            })

# Chat input
user_input = st.chat_input("Ask about payments, compliance, or anomalies...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Payment Intelligence Agent v1.0.0")
with col2:
    st.caption("Powered by Snowflake Cortex")
with col3:
    st.caption("Demo Mode - Synthetic Data")
