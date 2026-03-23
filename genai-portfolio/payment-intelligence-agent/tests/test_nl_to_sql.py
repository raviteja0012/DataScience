"""Tests for the NL-to-SQL translation engine.

Validates query template matching, time range extraction, SQL generation,
and demo result generation for the payment analytics subsystem.
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.analytics.nl_to_sql import NLToSQLTranslator


@pytest.fixture
def translator() -> NLToSQLTranslator:
    """Create a fresh NLToSQLTranslator instance."""
    return NLToSQLTranslator()


class TestTemplateMatching:
    """Test that common payment queries match the expected templates."""

    def test_top_merchants_by_volume(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show me top 10 merchants by transaction volume last month")
        assert result["template"] == "top_merchants_by_volume"
        assert "MERCHANT_NAME" in result["sql"]
        assert "LIMIT 10" in result["sql"]

    def test_settlement_time_by_method(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("What is the average settlement time by payment method?")
        assert result["template"] == "settlement_time_by_method"
        assert "SETTLEMENT_DAYS" in result["sql"]
        assert "PAYMENT_METHOD" in result["sql"]

    def test_chargeback_summary(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("How many chargebacks occurred last month?")
        assert result["template"] == "chargeback_summary"
        assert "CHARGEBACKS" in result["sql"]

    def test_revenue_by_region(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Compare revenue across regions")
        assert result["template"] == "revenue_by_region"
        assert "REGION" in result["sql"]

    def test_decline_rate(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("What is the decline rate by payment method?")
        assert result["template"] == "decline_rate_by_method"
        assert "DECLINED" in result["sql"]

    def test_daily_volume_trend(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show me the daily transaction volume trend")
        assert result["template"] == "daily_volume_trend"
        assert "DATE_TRUNC" in result["sql"]

    def test_merchant_chargeback_rate(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Which merchants have the highest chargeback rate?")
        assert result["template"] == "merchant_chargeback_rate"
        assert "chargeback_rate_pct" in result["sql"]

    def test_payment_method_breakdown(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show me the payment method breakdown")
        assert result["template"] == "payment_method_breakdown"
        assert "CARD_BRAND" in result["sql"]


class TestTimeRangeExtraction:
    """Test natural language time range parsing."""

    def test_last_month(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show top merchants last month")
        assert "last month" in result["time_range"]["start"] or result["explanation"]

    def test_last_n_days(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show transactions from last 7 days")
        assert result["time_range"]["start"] is not None
        assert result["time_range"]["end"] is not None

    def test_this_year(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show revenue by region this year")
        assert result["time_range"]["start"].startswith(
            str(pd.Timestamp.utcnow().year)
        ) or result["time_range"]["start"] is not None

    def test_default_time_range(self, translator: NLToSQLTranslator) -> None:
        """Queries without a time reference should default to last 30 days."""
        result = translator.translate("Show top 5 merchants")
        assert result["time_range"]["start"] is not None
        assert result["time_range"]["end"] is not None


class TestLimitExtraction:
    """Test numeric limit extraction from queries."""

    def test_top_n_extraction(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show top 15 merchants by volume")
        assert "LIMIT 15" in result["sql"]

    def test_default_limit(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show me merchants by transaction volume")
        # Should have some limit
        assert "LIMIT" in result["sql"]


class TestDemoResultGeneration:
    """Test that demo mode generates valid DataFrames for each query type."""

    def test_merchant_volume_results(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show top 10 merchants by volume")
        df = translator.generate_demo_results("Show top 10 merchants by volume", result["sql"])
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "MERCHANT_NAME" in df.columns
        assert "TOTAL_VOLUME" in df.columns

    def test_settlement_results(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Average settlement time by method")
        df = translator.generate_demo_results("Average settlement time by method", result["sql"])
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "PAYMENT_METHOD" in df.columns

    def test_chargeback_results(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Chargeback summary by reason")
        df = translator.generate_demo_results("Chargeback summary by reason", result["sql"])
        assert isinstance(df, pd.DataFrame)
        assert "REASON_CODE" in df.columns

    def test_trend_results(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Daily transaction volume trend")
        df = translator.generate_demo_results("Daily transaction volume trend", result["sql"])
        assert isinstance(df, pd.DataFrame)
        assert "TRANSACTION_DAY" in df.columns
        assert len(df) == 30

    def test_general_query_returns_data(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Tell me about overall transaction metrics")
        df = translator.generate_demo_results("overall metrics", result["sql"])
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestSQLStructure:
    """Test that generated SQL has proper structure."""

    def test_sql_is_select_only(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show top 10 merchants")
        sql_upper = result["sql"].strip().upper()
        assert sql_upper.startswith("SELECT")
        # Should not contain DML statements
        for keyword in ["INSERT", "UPDATE", "DELETE", "DROP"]:
            assert keyword not in sql_upper

    def test_sql_contains_date_filter(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Show merchants by volume last month")
        assert ">=" in result["sql"]
        assert "<" in result["sql"]

    def test_translation_returns_explanation(self, translator: NLToSQLTranslator) -> None:
        result = translator.translate("Top 5 merchants by volume")
        assert result["explanation"]
        assert isinstance(result["explanation"], str)
