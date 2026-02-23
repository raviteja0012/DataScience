"""Natural language to SQL translation engine for payment analytics.

Translates user questions into safe, optimized Snowflake SQL queries using
a combination of template matching, schema-aware construction, and Cortex
LLM generation. The translator understands payment domain terminology and
maps it to the underlying schema automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .schema_introspector import SchemaIntrospector
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TimeRange:
    """Parsed time range from natural language."""

    start: str
    end: str
    label: str


# Common payment analytics query templates keyed by pattern
_QUERY_TEMPLATES: dict[str, str] = {
    "top_merchants_by_volume": """
        SELECT m.MERCHANT_NAME, COUNT(*) AS transaction_count,
               SUM(t.AMOUNT) AS total_volume,
               AVG(t.AMOUNT) AS avg_transaction
        FROM TRANSACTIONS t
        JOIN MERCHANTS m ON t.MERCHANT_ID = m.MERCHANT_ID
        WHERE t.TRANSACTION_DATE >= '{start_date}'
          AND t.TRANSACTION_DATE < '{end_date}'
          AND t.STATUS = 'APPROVED'
        GROUP BY m.MERCHANT_NAME
        ORDER BY total_volume DESC
        LIMIT {limit}
    """,
    "settlement_time_by_method": """
        SELECT s.PAYMENT_METHOD,
               AVG(s.SETTLEMENT_DAYS) AS avg_settlement_days,
               MIN(s.SETTLEMENT_DAYS) AS min_settlement_days,
               MAX(s.SETTLEMENT_DAYS) AS max_settlement_days,
               COUNT(*) AS settlement_count
        FROM SETTLEMENTS s
        WHERE s.SETTLEMENT_DATE >= '{start_date}'
          AND s.SETTLEMENT_DATE < '{end_date}'
        GROUP BY s.PAYMENT_METHOD
        ORDER BY avg_settlement_days ASC
    """,
    "chargeback_summary": """
        SELECT cb.REASON_CODE, cb.REASON_DESCRIPTION,
               COUNT(*) AS chargeback_count,
               SUM(cb.AMOUNT) AS total_amount,
               AVG(cb.RESOLUTION_DAYS) AS avg_resolution_days,
               SUM(CASE WHEN cb.STATUS = 'WON' THEN 1 ELSE 0 END) AS won_count,
               ROUND(SUM(CASE WHEN cb.STATUS = 'WON' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS win_rate_pct
        FROM CHARGEBACKS cb
        WHERE cb.CHARGEBACK_DATE >= '{start_date}'
          AND cb.CHARGEBACK_DATE < '{end_date}'
        GROUP BY cb.REASON_CODE, cb.REASON_DESCRIPTION
        ORDER BY chargeback_count DESC
    """,
    "revenue_by_region": """
        SELECT t.REGION,
               COUNT(*) AS transaction_count,
               SUM(t.AMOUNT) AS total_revenue,
               AVG(t.AMOUNT) AS avg_transaction,
               SUM(t.FEE_AMOUNT) AS total_fees
        FROM TRANSACTIONS t
        WHERE t.TRANSACTION_DATE >= '{start_date}'
          AND t.TRANSACTION_DATE < '{end_date}'
          AND t.STATUS = 'APPROVED'
        GROUP BY t.REGION
        ORDER BY total_revenue DESC
    """,
    "decline_rate_by_method": """
        SELECT t.PAYMENT_METHOD,
               COUNT(*) AS total_transactions,
               SUM(CASE WHEN t.STATUS = 'DECLINED' THEN 1 ELSE 0 END) AS declined_count,
               ROUND(SUM(CASE WHEN t.STATUS = 'DECLINED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS decline_rate_pct,
               SUM(CASE WHEN t.STATUS = 'APPROVED' THEN 1 ELSE 0 END) AS approved_count
        FROM TRANSACTIONS t
        WHERE t.TRANSACTION_DATE >= '{start_date}'
          AND t.TRANSACTION_DATE < '{end_date}'
        GROUP BY t.PAYMENT_METHOD
        ORDER BY decline_rate_pct DESC
    """,
    "daily_volume_trend": """
        SELECT DATE_TRUNC('DAY', t.TRANSACTION_DATE) AS transaction_day,
               COUNT(*) AS transaction_count,
               SUM(t.AMOUNT) AS total_amount,
               AVG(t.AMOUNT) AS avg_amount
        FROM TRANSACTIONS t
        WHERE t.TRANSACTION_DATE >= '{start_date}'
          AND t.TRANSACTION_DATE < '{end_date}'
          AND t.STATUS = 'APPROVED'
        GROUP BY DATE_TRUNC('DAY', t.TRANSACTION_DATE)
        ORDER BY transaction_day ASC
    """,
    "merchant_chargeback_rate": """
        SELECT m.MERCHANT_NAME,
               t.total_transactions,
               COALESCE(cb.chargeback_count, 0) AS chargeback_count,
               ROUND(COALESCE(cb.chargeback_count, 0) * 100.0 / NULLIF(t.total_transactions, 0), 4) AS chargeback_rate_pct
        FROM MERCHANTS m
        JOIN (
            SELECT MERCHANT_ID, COUNT(*) AS total_transactions
            FROM TRANSACTIONS
            WHERE STATUS = 'APPROVED'
            GROUP BY MERCHANT_ID
        ) t ON m.MERCHANT_ID = t.MERCHANT_ID
        LEFT JOIN (
            SELECT MERCHANT_ID, COUNT(*) AS chargeback_count
            FROM CHARGEBACKS
            GROUP BY MERCHANT_ID
        ) cb ON m.MERCHANT_ID = cb.MERCHANT_ID
        ORDER BY chargeback_rate_pct DESC
        LIMIT {limit}
    """,
    "payment_method_breakdown": """
        SELECT t.PAYMENT_METHOD,
               t.CARD_BRAND,
               COUNT(*) AS transaction_count,
               SUM(t.AMOUNT) AS total_volume,
               ROUND(SUM(t.AMOUNT) * 100.0 / SUM(SUM(t.AMOUNT)) OVER(), 2) AS volume_share_pct
        FROM TRANSACTIONS t
        WHERE t.TRANSACTION_DATE >= '{start_date}'
          AND t.TRANSACTION_DATE < '{end_date}'
          AND t.STATUS = 'APPROVED'
        GROUP BY t.PAYMENT_METHOD, t.CARD_BRAND
        ORDER BY total_volume DESC
    """,
}

# Patterns that map user query intents to template keys
_INTENT_PATTERNS: list[tuple[str, str, str]] = [
    (r"top\s+(\d+)?\s*merchants?\s+(?:by\s+)?(?:transaction\s+)?volume", "top_merchants_by_volume", "Top merchants by transaction volume"),
    (r"(?:average|avg)\s+settlement\s+time", "settlement_time_by_method", "Average settlement time by payment method"),
    (r"chargebacks?\s+(?:summary|breakdown|by\s+reason)", "chargeback_summary", "Chargeback summary by reason code"),
    (r"revenue\s+(?:by|across|per)\s+region", "revenue_by_region", "Revenue breakdown by region"),
    (r"decline\s+rate", "decline_rate_by_method", "Decline rate by payment method"),
    (r"(?:daily|transaction)\s+(?:volume\s+)?trend", "daily_volume_trend", "Daily transaction volume trend"),
    (r"merchant.*chargeback\s+rate", "merchant_chargeback_rate", "Merchant chargeback rates"),
    (r"payment\s+method\s+(?:breakdown|distribution|share|mix)", "payment_method_breakdown", "Payment method volume breakdown"),
    (r"how\s+many\s+chargebacks", "chargeback_summary", "Chargeback counts and details"),
    (r"compare\s+(?:revenue|volume|transactions?).*(?:region|area)", "revenue_by_region", "Regional comparison"),
]


class NLToSQLTranslator:
    """Translates natural language payment queries into SQL.

    Uses a tiered approach:
    1. Pattern matching against known query templates
    2. Schema-aware SQL construction for novel queries
    3. Cortex LLM generation as a fallback (when available)

    The translator extracts time ranges, limits, and entity references
    from the natural language query and injects them into parameterized
    templates for safe, predictable SQL generation.
    """

    def __init__(self) -> None:
        self._schema = SchemaIntrospector()

    def translate(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate a natural language query into SQL.

        Args:
            query: Natural language question about payment data.
            context: Conversation context with entity references.

        Returns:
            Dictionary containing:
                - sql: Generated SQL query string.
                - template: Template key used (if applicable).
                - explanation: Human-readable explanation of the query.
                - time_range: Extracted time range.
        """
        query_lower = query.lower().strip()
        time_range = self._extract_time_range(query_lower, context)
        limit = self._extract_limit(query_lower)

        # Try template matching first
        for pattern, template_key, explanation in _INTENT_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                # Extract limit from regex capture group if present
                if match.groups():
                    captured_limit = match.group(1)
                    if captured_limit and captured_limit.isdigit():
                        limit = int(captured_limit)

                template = _QUERY_TEMPLATES[template_key]
                sql = template.format(
                    start_date=time_range.start,
                    end_date=time_range.end,
                    limit=limit,
                ).strip()

                logger.info(
                    "nl_to_sql_translated",
                    method="template",
                    template=template_key,
                    time_range=time_range.label,
                )

                return {
                    "sql": sql,
                    "template": template_key,
                    "explanation": f"{explanation} for {time_range.label}",
                    "time_range": {"start": time_range.start, "end": time_range.end},
                }

        # Fallback: construct a general-purpose query
        sql = self._construct_general_query(query_lower, time_range, limit)

        return {
            "sql": sql,
            "template": None,
            "explanation": f"Custom query generated from: '{query[:80]}'",
            "time_range": {"start": time_range.start, "end": time_range.end},
        }

    def _extract_time_range(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> TimeRange:
        """Extract time range from natural language query.

        Supports patterns like "last month", "past 7 days", "this quarter",
        "in January 2024", and falls back to context or last 30 days.
        """
        now = datetime.utcnow()

        # Pattern: "last N days/weeks/months"
        match = re.search(r"(?:last|past|previous)\s+(\d+)\s+(day|week|month|quarter|year)s?", query)
        if match:
            count = int(match.group(1))
            unit = match.group(2)
            delta_map = {
                "day": timedelta(days=count),
                "week": timedelta(weeks=count),
                "month": timedelta(days=count * 30),
                "quarter": timedelta(days=count * 91),
                "year": timedelta(days=count * 365),
            }
            delta = delta_map.get(unit, timedelta(days=30))
            start = (now - delta).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
            return TimeRange(start=start, end=end, label=f"last {count} {unit}(s)")

        # Pattern: "last month" / "this month"
        if "last month" in query:
            first_of_current = now.replace(day=1)
            last_month_end = first_of_current - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return TimeRange(
                start=last_month_start.strftime("%Y-%m-%d"),
                end=first_of_current.strftime("%Y-%m-%d"),
                label="last month",
            )

        if "this month" in query:
            first_of_current = now.replace(day=1)
            return TimeRange(
                start=first_of_current.strftime("%Y-%m-%d"),
                end=now.strftime("%Y-%m-%d"),
                label="this month",
            )

        if "this quarter" in query:
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            quarter_start = now.replace(month=quarter_start_month, day=1)
            return TimeRange(
                start=quarter_start.strftime("%Y-%m-%d"),
                end=now.strftime("%Y-%m-%d"),
                label="this quarter",
            )

        if "this year" in query or "ytd" in query:
            year_start = now.replace(month=1, day=1)
            return TimeRange(
                start=year_start.strftime("%Y-%m-%d"),
                end=now.strftime("%Y-%m-%d"),
                label="year-to-date",
            )

        if "today" in query:
            today = now.strftime("%Y-%m-%d")
            tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            return TimeRange(start=today, end=tomorrow, label="today")

        if "yesterday" in query:
            yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            today = now.strftime("%Y-%m-%d")
            return TimeRange(start=yesterday, end=today, label="yesterday")

        # Check context for previous time range
        if context and "active_time_range" in context:
            prev = context["active_time_range"]
            if "reference" in prev:
                return TimeRange(
                    start=(now - timedelta(days=30)).strftime("%Y-%m-%d"),
                    end=now.strftime("%Y-%m-%d"),
                    label=prev["reference"],
                )

        # Default: last 30 days
        start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        return TimeRange(start=start, end=end, label="last 30 days")

    @staticmethod
    def _extract_limit(query: str) -> int:
        """Extract a numeric limit from the query (e.g., 'top 10')."""
        match = re.search(r"(?:top|first|limit)\s+(\d+)", query)
        if match:
            return min(int(match.group(1)), 1000)

        match = re.search(r"(\d+)\s+(?:merchants?|customers?|transactions?)", query)
        if match:
            return min(int(match.group(1)), 1000)

        return 20  # Default limit

    def _construct_general_query(
        self,
        query: str,
        time_range: TimeRange,
        limit: int,
    ) -> str:
        """Construct a general-purpose SQL query from the natural language.

        When no template matches, this method builds a query by identifying
        likely tables, aggregation functions, and grouping columns from
        the query text.
        """
        # Detect primary table
        table = "TRANSACTIONS"
        if "merchant" in query:
            table = "TRANSACTIONS t JOIN MERCHANTS m ON t.MERCHANT_ID = m.MERCHANT_ID"
        elif "settlement" in query:
            table = "SETTLEMENTS"
        elif "chargeback" in query:
            table = "CHARGEBACKS"
        elif "customer" in query:
            table = "TRANSACTIONS t JOIN CUSTOMERS c ON t.CUSTOMER_ID = c.CUSTOMER_ID"

        # Detect metrics
        metrics = []
        if any(w in query for w in ["count", "how many", "number of"]):
            metrics.append("COUNT(*) AS total_count")
        if any(w in query for w in ["total", "sum", "revenue", "volume"]):
            metrics.append("SUM(AMOUNT) AS total_amount")
        if any(w in query for w in ["average", "avg", "mean"]):
            metrics.append("AVG(AMOUNT) AS avg_amount")

        if not metrics:
            metrics = ["COUNT(*) AS total_count", "SUM(AMOUNT) AS total_amount"]

        # Detect grouping
        group_by = ""
        select_prefix = ""
        if "by region" in query or "per region" in query or "across region" in query:
            select_prefix = "REGION, "
            group_by = "GROUP BY REGION"
        elif "by payment method" in query or "by method" in query:
            select_prefix = "PAYMENT_METHOD, "
            group_by = "GROUP BY PAYMENT_METHOD"
        elif "by status" in query:
            select_prefix = "STATUS, "
            group_by = "GROUP BY STATUS"
        elif "by channel" in query:
            select_prefix = "CHANNEL, "
            group_by = "GROUP BY CHANNEL"

        select_cols = select_prefix + ", ".join(metrics)
        date_col = "TRANSACTION_DATE" if "TRANSACTIONS" in table.upper() else "CREATED_AT"

        sql = f"""
            SELECT {select_cols}
            FROM {table}
            WHERE {date_col} >= '{time_range.start}'
              AND {date_col} < '{time_range.end}'
            {group_by}
            ORDER BY {metrics[0].split(' AS ')[1]} DESC
            LIMIT {limit}
        """.strip()

        return sql

    def generate_demo_results(self, query: str, sql: str) -> pd.DataFrame:
        """Generate synthetic query results for demo mode.

        Creates realistic-looking data that matches what the SQL query
        would return, enabling full pipeline testing without a database.

        Args:
            query: Original natural language query.
            sql: Generated SQL query (used to infer result shape).

        Returns:
            Synthetic result DataFrame.
        """
        rng = np.random.default_rng(42)
        query_lower = query.lower()
        sql_lower = sql.lower()

        # Determine result shape from SQL
        if "merchant_name" in sql_lower and ("top" in query_lower or "volume" in query_lower):
            merchants = [
                "TechStore Global", "FoodMart Express", "CloudServices Inc",
                "RetailHub Premium", "GameZone Digital", "TravelNow Agency",
                "HealthPlus Pharmacy", "AutoParts Direct", "BookWorld Online",
                "FashionForward Co", "GadgetPro Shop", "GreenGrocers Ltd",
                "PetCare Central", "HomeDecor Plus", "SportsFit Outlet",
                "ElectroMart", "CoffeeBean Roasters", "MusicStream Pro",
                "ArtSupplies Hub", "ToyLand Express",
            ]
            n = min(len(merchants), self._extract_limit(query_lower))
            volumes = sorted(rng.uniform(50_000, 5_000_000, n), reverse=True)
            return pd.DataFrame({
                "MERCHANT_NAME": merchants[:n],
                "TRANSACTION_COUNT": rng.integers(500, 50_000, n),
                "TOTAL_VOLUME": np.round(volumes, 2),
                "AVG_TRANSACTION": np.round(rng.uniform(25, 250, n), 2),
            })

        if "settlement_days" in sql_lower or "settlement_time" in query_lower:
            methods = ["CREDIT", "DEBIT", "ACH", "WIRE"]
            return pd.DataFrame({
                "PAYMENT_METHOD": methods,
                "AVG_SETTLEMENT_DAYS": [2.3, 1.8, 3.5, 1.2],
                "MIN_SETTLEMENT_DAYS": [1, 1, 2, 1],
                "MAX_SETTLEMENT_DAYS": [5, 4, 7, 3],
                "SETTLEMENT_COUNT": rng.integers(1000, 50000, len(methods)),
            })

        if "reason_code" in sql_lower or "chargeback" in query_lower:
            return pd.DataFrame({
                "REASON_CODE": ["4837", "4853", "4863", "4855", "4834"],
                "REASON_DESCRIPTION": [
                    "No Cardholder Authorization",
                    "Cardholder Dispute - Not as Described",
                    "Cardholder Does Not Recognize",
                    "Non-Receipt of Merchandise",
                    "Transaction Amount Differs",
                ],
                "CHARGEBACK_COUNT": [342, 256, 198, 167, 89],
                "TOTAL_AMOUNT": [125340.50, 89420.75, 72150.00, 56890.25, 31200.00],
                "AVG_RESOLUTION_DAYS": [32.5, 28.1, 35.7, 22.3, 18.9],
                "WON_COUNT": [187, 128, 72, 101, 56],
                "WIN_RATE_PCT": [54.68, 50.00, 36.36, 60.48, 62.92],
            })

        if "region" in sql_lower:
            regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"]
            return pd.DataFrame({
                "REGION": regions,
                "TRANSACTION_COUNT": [125000, 89000, 67000, 34000, 12000],
                "TOTAL_REVENUE": [45_200_000, 32_100_000, 24_800_000, 11_500_000, 4_200_000],
                "AVG_TRANSACTION": [361.60, 360.67, 370.15, 338.24, 350.00],
                "TOTAL_FEES": [1_356_000, 963_000, 744_000, 345_000, 126_000],
            })

        if "decline_rate" in sql_lower or "decline" in query_lower:
            return pd.DataFrame({
                "PAYMENT_METHOD": ["CREDIT", "DEBIT", "ACH", "WIRE"],
                "TOTAL_TRANSACTIONS": [250000, 180000, 45000, 12000],
                "DECLINED_COUNT": [12500, 5400, 2250, 240],
                "DECLINE_RATE_PCT": [5.00, 3.00, 5.00, 2.00],
                "APPROVED_COUNT": [237500, 174600, 42750, 11760],
            })

        if "date_trunc" in sql_lower or "trend" in query_lower or "daily" in query_lower:
            dates = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
            base_volume = rng.integers(8000, 12000, 30)
            # Add weekend dip pattern
            for i, d in enumerate(dates):
                if d.weekday() >= 5:
                    base_volume[i] = int(base_volume[i] * 0.65)
            return pd.DataFrame({
                "TRANSACTION_DAY": dates,
                "TRANSACTION_COUNT": base_volume,
                "TOTAL_AMOUNT": np.round(base_volume * rng.uniform(85, 120, 30), 2),
                "AVG_AMOUNT": np.round(rng.uniform(85, 120, 30), 2),
            })

        if "payment_method" in sql_lower and "card_brand" in sql_lower:
            return pd.DataFrame({
                "PAYMENT_METHOD": ["CREDIT", "CREDIT", "CREDIT", "CREDIT", "DEBIT", "ACH", "WIRE"],
                "CARD_BRAND": ["VISA", "MASTERCARD", "AMEX", "DISCOVER", "VISA", None, None],
                "TRANSACTION_COUNT": [95000, 72000, 38000, 15000, 85000, 32000, 8000],
                "TOTAL_VOLUME": [38_500_000, 28_200_000, 19_800_000, 5_100_000, 22_100_000, 8_900_000, 4_200_000],
                "VOLUME_SHARE_PCT": [30.36, 22.24, 15.62, 4.02, 17.43, 7.02, 3.31],
            })

        # Generic fallback
        return pd.DataFrame({
            "METRIC": ["Total Transactions", "Total Volume", "Avg Transaction", "Approval Rate"],
            "VALUE": [327_000, 117_800_000, 360.25, 95.2],
        })
