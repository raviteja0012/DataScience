"""Response formatting and visualization for agent outputs.

Transforms raw query results, RAG responses, and anomaly detections
into rich, presentation-ready formats including Plotly charts, formatted
tables, and natural language summaries for the Streamlit interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResponseType(str, Enum):
    """Types of formatted responses the agent can produce."""

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    ALERT = "alert"
    MIXED = "mixed"


@dataclass
class FormattedResponse:
    """A structured agent response with optional visualizations.

    Attributes:
        text: Natural language response text.
        response_type: Primary response format.
        dataframe: Tabular data for display (if applicable).
        chart: Plotly figure object (if applicable).
        sql: SQL query that produced the results (if applicable).
        metadata: Additional response metadata.
        alerts: List of alert messages (for anomaly detection).
    """

    text: str
    response_type: ResponseType = ResponseType.TEXT
    dataframe: pd.DataFrame | None = None
    chart: go.Figure | None = None
    sql: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)


class ResponseFormatter:
    """Formats agent outputs into rich, interactive responses.

    Automatically selects appropriate visualization types based on
    the data shape and query intent, with support for Plotly charts,
    formatted tables, and composite responses.
    """

    def __init__(self, max_display_rows: int = 100, show_sql: bool = True) -> None:
        self.max_display_rows = max_display_rows
        self.show_sql = show_sql

    def format_analytics_response(
        self,
        data: pd.DataFrame,
        query: str,
        sql: str | None = None,
        summary: str | None = None,
    ) -> FormattedResponse:
        """Format an analytics query result with auto-detected visualization.

        Args:
            data: Query result DataFrame.
            query: Original user query for context.
            sql: SQL query that produced the results.
            summary: Natural language summary of results.

        Returns:
            FormattedResponse with appropriate chart and table.
        """
        if data.empty:
            return FormattedResponse(
                text="The query returned no results. Try broadening your search criteria.",
                response_type=ResponseType.TEXT,
                sql=sql if self.show_sql else None,
            )

        # Truncate for display
        display_df = data.head(self.max_display_rows)
        truncated = len(data) > self.max_display_rows

        # Build summary text
        text_parts = []
        if summary:
            text_parts.append(summary)
        else:
            text_parts.append(f"Query returned **{len(data):,}** rows.")

        if truncated:
            text_parts.append(f"Showing first {self.max_display_rows} of {len(data):,} rows.")

        # Auto-detect best chart type
        chart = self._auto_chart(display_df, query)

        return FormattedResponse(
            text="\n\n".join(text_parts),
            response_type=ResponseType.MIXED if chart else ResponseType.TABLE,
            dataframe=display_df,
            chart=chart,
            sql=sql if self.show_sql else None,
            metadata={"total_rows": len(data), "truncated": truncated},
        )

    def format_compliance_response(
        self,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
        requirement_ids: list[str] | None = None,
    ) -> FormattedResponse:
        """Format a PCI compliance RAG response.

        Args:
            answer: Generated answer text.
            sources: Source document references.
            requirement_ids: Relevant PCI DSS requirement IDs.

        Returns:
            FormattedResponse with sourced compliance answer.
        """
        text_parts = [answer]

        if requirement_ids:
            req_str = ", ".join(f"Req. {rid}" for rid in requirement_ids)
            text_parts.append(f"\n**Relevant PCI DSS Requirements:** {req_str}")

        if sources:
            text_parts.append("\n**Sources:**")
            for i, source in enumerate(sources, 1):
                doc_name = source.get("document", "Unknown")
                section = source.get("section", "")
                score = source.get("score", 0.0)
                text_parts.append(f"{i}. *{doc_name}* - {section} (relevance: {score:.2f})")

        return FormattedResponse(
            text="\n".join(text_parts),
            response_type=ResponseType.TEXT,
            metadata={
                "sources": sources or [],
                "requirement_ids": requirement_ids or [],
            },
        )

    def format_anomaly_response(
        self,
        anomalies: pd.DataFrame,
        summary: str,
        alerts: list[dict[str, Any]] | None = None,
    ) -> FormattedResponse:
        """Format anomaly detection results with alerts and visualizations.

        Args:
            anomalies: DataFrame of detected anomalies.
            summary: Natural language summary of findings.
            alerts: Structured alert objects with severity levels.

        Returns:
            FormattedResponse with anomaly chart and alert list.
        """
        alert_list = alerts or []
        chart = None

        if not anomalies.empty:
            chart = self._build_anomaly_chart(anomalies)

        # Format alert text
        text_parts = [summary]
        if alert_list:
            text_parts.append(f"\n**{len(alert_list)} Alert(s) Generated:**")
            for alert in alert_list:
                severity = alert.get("severity", "medium").upper()
                icon = {"CRITICAL": "!!!", "HIGH": "!!", "MEDIUM": "!", "LOW": ""}.get(severity, "")
                text_parts.append(
                    f"- [{severity}] {icon} {alert.get('message', 'Anomaly detected')}"
                )

        return FormattedResponse(
            text="\n".join(text_parts),
            response_type=ResponseType.MIXED if chart else ResponseType.ALERT,
            dataframe=anomalies if not anomalies.empty else None,
            chart=chart,
            alerts=alert_list,
            metadata={"anomaly_count": len(anomalies)},
        )

    def format_error_response(self, error_message: str, suggestion: str | None = None) -> FormattedResponse:
        """Format an error response with optional recovery suggestion.

        Args:
            error_message: Description of what went wrong.
            suggestion: Optional suggestion for the user.

        Returns:
            FormattedResponse with error information.
        """
        text = f"I encountered an issue: {error_message}"
        if suggestion:
            text += f"\n\n**Suggestion:** {suggestion}"
        return FormattedResponse(text=text, response_type=ResponseType.TEXT)

    def _auto_chart(self, df: pd.DataFrame, query: str) -> go.Figure | None:
        """Automatically select and build a chart based on data shape and query.

        Heuristics:
        - Single numeric column with a categorical column -> bar chart
        - Time column with numeric -> line chart
        - Multiple numeric columns -> grouped bar or scatter
        - Percentage/rate data -> pie chart if few categories

        Args:
            df: DataFrame to visualize.
            query: Original query for intent hints.

        Returns:
            Plotly figure or None if no suitable chart is detected.
        """
        if len(df) < 2 or len(df.columns) < 2:
            return None

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        query_lower = query.lower()

        # Time-series detection
        if datetime_cols and numeric_cols:
            fig = px.line(
                df,
                x=datetime_cols[0],
                y=numeric_cols[0],
                title=self._derive_title(query),
            )
            fig.update_layout(template="plotly_white")
            return fig

        # Trend keywords suggest line chart
        if any(word in query_lower for word in ["trend", "over time", "daily", "monthly", "weekly"]):
            if len(numeric_cols) >= 1 and len(df.columns) >= 2:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
                fig = px.line(
                    df,
                    x=x_col,
                    y=numeric_cols[0],
                    title=self._derive_title(query),
                )
                fig.update_layout(template="plotly_white")
                return fig

        # Top/bottom queries -> horizontal bar
        if any(word in query_lower for word in ["top", "bottom", "highest", "lowest", "ranking"]):
            if categorical_cols and numeric_cols:
                fig = px.bar(
                    df,
                    y=categorical_cols[0],
                    x=numeric_cols[0],
                    orientation="h",
                    title=self._derive_title(query),
                )
                fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
                return fig

        # Distribution / breakdown queries -> pie chart (small cardinality)
        if any(word in query_lower for word in ["breakdown", "distribution", "share", "percentage"]):
            if categorical_cols and numeric_cols and len(df) <= 10:
                fig = px.pie(
                    df,
                    names=categorical_cols[0],
                    values=numeric_cols[0],
                    title=self._derive_title(query),
                )
                fig.update_layout(template="plotly_white")
                return fig

        # Default: bar chart for categorical + numeric
        if categorical_cols and numeric_cols:
            fig = px.bar(
                df,
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=self._derive_title(query),
            )
            fig.update_layout(template="plotly_white")
            return fig

        return None

    def _build_anomaly_chart(self, anomalies: pd.DataFrame) -> go.Figure:
        """Build a scatter plot highlighting anomalous transactions.

        Args:
            anomalies: DataFrame with anomaly scores and transaction data.

        Returns:
            Plotly figure with anomaly visualization.
        """
        fig = go.Figure()

        # Determine axes based on available columns
        x_col = None
        y_col = None
        color_col = None

        for col in anomalies.columns:
            col_lower = col.lower()
            if "date" in col_lower or "time" in col_lower:
                x_col = col
            elif "amount" in col_lower or "value" in col_lower:
                y_col = col
            elif "score" in col_lower or "severity" in col_lower:
                color_col = col

        if x_col and y_col:
            fig.add_trace(go.Scatter(
                x=anomalies[x_col],
                y=anomalies[y_col],
                mode="markers",
                marker=dict(
                    size=10,
                    color=anomalies[color_col] if color_col else "red",
                    colorscale="RdYlGn_r" if color_col else None,
                    showscale=bool(color_col),
                ),
                text=anomalies.apply(
                    lambda row: "<br>".join(f"{k}: {v}" for k, v in row.items()),
                    axis=1,
                ),
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.update_layout(
                title="Detected Anomalies",
                xaxis_title=x_col,
                yaxis_title=y_col,
                template="plotly_white",
            )
        else:
            # Fallback: simple index-based scatter
            numeric_cols = anomalies.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                fig.add_trace(go.Scatter(
                    y=anomalies[numeric_cols[0]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name="Anomaly",
                ))
                fig.update_layout(
                    title="Detected Anomalies",
                    yaxis_title=numeric_cols[0],
                    template="plotly_white",
                )

        return fig

    @staticmethod
    def _derive_title(query: str) -> str:
        """Derive a chart title from the user query.

        Cleans up the query to produce a concise, title-case chart heading.

        Args:
            query: Original user query.

        Returns:
            Title-cased chart title.
        """
        # Remove common question prefixes
        import re
        title = re.sub(
            r"^(?:show me|what (?:is|are|was|were)|give me|display|list|find|get)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        # Truncate long titles
        if len(title) > 60:
            title = title[:57] + "..."
        return title.strip().title()
