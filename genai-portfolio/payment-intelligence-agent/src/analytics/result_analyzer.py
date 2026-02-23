"""Query result interpretation and natural language insight generation.

Analyzes query results to produce human-readable summaries, highlight
notable patterns, and identify actionable insights that complement
the raw tabular data presented to the user.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResultAnalyzer:
    """Generates natural language summaries and insights from query results.

    Examines the shape, distribution, and notable features of result data
    to produce contextual summaries that help users understand the data
    without needing to interpret raw numbers.
    """

    def summarize(self, df: pd.DataFrame, query: str) -> str:
        """Generate a natural language summary of query results.

        Args:
            df: Query result DataFrame.
            query: Original user query for context.

        Returns:
            Human-readable summary string.
        """
        if df.empty:
            return "No data found matching your criteria."

        parts: list[str] = []

        # Basic shape description
        row_count = len(df)
        col_count = len(df.columns)

        # Identify key numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Context-specific summaries
        query_lower = query.lower()

        if "top" in query_lower and categorical_cols and numeric_cols:
            parts.append(self._summarize_ranking(df, categorical_cols[0], numeric_cols[0], query_lower))
        elif "trend" in query_lower or "daily" in query_lower:
            parts.append(self._summarize_trend(df, numeric_cols))
        elif "rate" in query_lower or "percentage" in query_lower:
            parts.append(self._summarize_rates(df, numeric_cols, categorical_cols))
        elif "chargeback" in query_lower:
            parts.append(self._summarize_chargebacks(df))
        elif "settlement" in query_lower:
            parts.append(self._summarize_settlements(df))
        else:
            parts.append(self._summarize_general(df, numeric_cols, categorical_cols))

        # Add notable observations
        observations = self._find_notable_patterns(df, numeric_cols)
        if observations:
            parts.append("\n**Notable observations:**")
            for obs in observations[:3]:
                parts.append(f"- {obs}")

        return "\n".join(parts)

    def _summarize_ranking(
        self,
        df: pd.DataFrame,
        name_col: str,
        value_col: str,
        query: str,
    ) -> str:
        """Summarize a ranked result set (e.g., top N merchants)."""
        top_name = df.iloc[0][name_col]
        top_value = df.iloc[0][value_col]
        total = df[value_col].sum()
        top_share = (top_value / total * 100) if total > 0 else 0

        parts = [
            f"Found **{len(df)}** results ranked by {value_col.lower().replace('_', ' ')}.",
            f"**{top_name}** leads with {self._format_number(top_value)} "
            f"({top_share:.1f}% of total).",
        ]

        if len(df) >= 3:
            top3_total = df.head(3)[value_col].sum()
            top3_share = (top3_total / total * 100) if total > 0 else 0
            parts.append(
                f"The top 3 account for {top3_share:.1f}% of the total "
                f"({self._format_number(top3_total)})."
            )

        return " ".join(parts)

    def _summarize_trend(self, df: pd.DataFrame, numeric_cols: list[str]) -> str:
        """Summarize a time-series trend."""
        if not numeric_cols:
            return f"Trend data with {len(df)} data points."

        primary_col = numeric_cols[0]
        values = df[primary_col].dropna()

        if len(values) < 2:
            return f"Trend data with {len(df)} data points."

        first_half = values.iloc[:len(values)//2].mean()
        second_half = values.iloc[len(values)//2:].mean()
        change_pct = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0

        direction = "upward" if change_pct > 2 else "downward" if change_pct < -2 else "stable"
        trend_icon = "increasing" if change_pct > 2 else "decreasing" if change_pct < -2 else "stable"

        parts = [
            f"Showing **{len(df)}** data points for {primary_col.lower().replace('_', ' ')}.",
            f"The trend is **{trend_icon}** with a {abs(change_pct):.1f}% "
            f"{'increase' if change_pct > 0 else 'decrease'} comparing the first and second halves of the period.",
            f"Range: {self._format_number(values.min())} to {self._format_number(values.max())} "
            f"(avg: {self._format_number(values.mean())}).",
        ]

        return " ".join(parts)

    def _summarize_rates(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> str:
        """Summarize rate/percentage data."""
        rate_cols = [c for c in numeric_cols if "rate" in c.lower() or "pct" in c.lower()]
        if not rate_cols:
            return self._summarize_general(df, numeric_cols, categorical_cols)

        rate_col = rate_cols[0]
        name_col = categorical_cols[0] if categorical_cols else df.columns[0]

        highest = df.loc[df[rate_col].idxmax()]
        lowest = df.loc[df[rate_col].idxmin()]

        return (
            f"**{highest[name_col]}** has the highest rate at {highest[rate_col]:.2f}%, "
            f"while **{lowest[name_col]}** has the lowest at {lowest[rate_col]:.2f}%. "
            f"Average across all categories: {df[rate_col].mean():.2f}%."
        )

    def _summarize_chargebacks(self, df: pd.DataFrame) -> str:
        """Summarize chargeback data."""
        total_count = df["CHARGEBACK_COUNT"].sum() if "CHARGEBACK_COUNT" in df.columns else len(df)
        total_amount = df["TOTAL_AMOUNT"].sum() if "TOTAL_AMOUNT" in df.columns else 0

        parts = [
            f"Found **{int(total_count):,}** chargebacks totaling **${total_amount:,.2f}**.",
        ]

        if "REASON_DESCRIPTION" in df.columns and "CHARGEBACK_COUNT" in df.columns:
            top_reason = df.loc[df["CHARGEBACK_COUNT"].idxmax()]
            parts.append(
                f"Most common reason: **{top_reason['REASON_DESCRIPTION']}** "
                f"({int(top_reason['CHARGEBACK_COUNT']):,} cases)."
            )

        if "WIN_RATE_PCT" in df.columns:
            avg_win_rate = df["WIN_RATE_PCT"].mean()
            parts.append(f"Overall dispute win rate: **{avg_win_rate:.1f}%**.")

        return " ".join(parts)

    def _summarize_settlements(self, df: pd.DataFrame) -> str:
        """Summarize settlement data."""
        if "AVG_SETTLEMENT_DAYS" in df.columns:
            avg_days = df["AVG_SETTLEMENT_DAYS"].mean()
            fastest = df.loc[df["AVG_SETTLEMENT_DAYS"].idxmin()]
            name_col = "PAYMENT_METHOD" if "PAYMENT_METHOD" in df.columns else df.columns[0]

            return (
                f"Average settlement time across all methods: **{avg_days:.1f} days**. "
                f"**{fastest[name_col]}** is fastest at {fastest['AVG_SETTLEMENT_DAYS']:.1f} days."
            )

        return self._summarize_general(
            df,
            df.select_dtypes(include=["number"]).columns.tolist(),
            df.select_dtypes(include=["object"]).columns.tolist(),
        )

    def _summarize_general(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> str:
        """General-purpose summary for any result set."""
        parts = [f"Query returned **{len(df):,}** rows across **{len(df.columns)}** columns."]

        if numeric_cols:
            primary = numeric_cols[0]
            total = df[primary].sum()
            avg = df[primary].mean()
            parts.append(
                f"Total {primary.lower().replace('_', ' ')}: {self._format_number(total)}, "
                f"average: {self._format_number(avg)}."
            )

        return " ".join(parts)

    def _find_notable_patterns(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> list[str]:
        """Identify notable statistical patterns in the results."""
        observations: list[str] = []

        for col in numeric_cols[:3]:
            values = df[col].dropna()
            if len(values) < 3:
                continue

            # Check for high concentration
            if len(values) >= 5:
                top_val = values.max()
                total = values.sum()
                if total > 0 and (top_val / total) > 0.3:
                    observations.append(
                        f"High concentration: the top value accounts for "
                        f"{top_val / total * 100:.1f}% of total {col.lower().replace('_', ' ')}"
                    )

            # Check for large variance
            cv = values.std() / values.mean() if values.mean() != 0 else 0
            if cv > 1.0:
                observations.append(
                    f"High variability in {col.lower().replace('_', ' ')} "
                    f"(coefficient of variation: {cv:.2f})"
                )

            # Check for outliers using IQR
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_count = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
                if outlier_count > 0:
                    observations.append(
                        f"{outlier_count} outlier(s) detected in {col.lower().replace('_', ' ')}"
                    )

        return observations

    @staticmethod
    def _format_number(value: float) -> str:
        """Format a number for human-readable display."""
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:,.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.2f}M"
        if abs(value) >= 1_000:
            return f"${value:,.0f}"
        return f"{value:,.2f}"
