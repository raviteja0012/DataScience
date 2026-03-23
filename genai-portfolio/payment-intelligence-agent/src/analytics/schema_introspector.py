"""Snowflake schema discovery and metadata management.

Provides schema-aware context to the NL-to-SQL translator by introspecting
the payment database schema. In demo mode, returns a predefined schema
that mirrors the production DDL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ColumnInfo:
    """Metadata for a single table column.

    Attributes:
        name: Column name.
        data_type: SQL data type.
        nullable: Whether the column allows NULL values.
        description: Human-readable description for NL-to-SQL context.
        is_pii: Whether this column contains personally identifiable information.
    """

    name: str
    data_type: str
    nullable: bool = True
    description: str = ""
    is_pii: bool = False


@dataclass
class TableInfo:
    """Metadata for a single database table.

    Attributes:
        name: Fully qualified table name.
        description: Human-readable table description.
        columns: List of column metadata objects.
        primary_key: Primary key column name(s).
        foreign_keys: Foreign key relationships to other tables.
        row_count_estimate: Approximate row count for query optimization hints.
    """

    name: str
    description: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: dict[str, str] = field(default_factory=dict)
    row_count_estimate: int | None = None

    def get_column_names(self) -> list[str]:
        """Return all column names in this table."""
        return [col.name for col in self.columns]

    def get_column(self, name: str) -> ColumnInfo | None:
        """Look up a column by name (case-insensitive)."""
        name_upper = name.upper()
        for col in self.columns:
            if col.name.upper() == name_upper:
                return col
        return None

    def to_ddl_context(self) -> str:
        """Generate a DDL-like context string for LLM prompts."""
        lines = [f"-- {self.description}", f"TABLE {self.name} ("]
        for col in self.columns:
            null_str = "" if col.nullable else " NOT NULL"
            desc = f"  -- {col.description}" if col.description else ""
            lines.append(f"  {col.name} {col.data_type}{null_str},{desc}")
        if self.primary_key:
            lines.append(f"  PRIMARY KEY ({', '.join(self.primary_key)})")
        lines.append(")")
        if self.foreign_keys:
            for col, ref in self.foreign_keys.items():
                lines.append(f"-- FK: {col} -> {ref}")
        return "\n".join(lines)


class SchemaIntrospector:
    """Discovers and caches database schema metadata.

    In production, queries Snowflake INFORMATION_SCHEMA to build the schema
    map. In demo mode, returns a predefined payment schema that matches
    the sample DDL.
    """

    def __init__(self) -> None:
        self._schema_cache: dict[str, TableInfo] | None = None

    def get_schema(self, force_refresh: bool = False) -> dict[str, TableInfo]:
        """Get the complete schema map for payment tables.

        Args:
            force_refresh: Force re-introspection even if cached.

        Returns:
            Dictionary mapping table names to TableInfo objects.
        """
        if self._schema_cache is not None and not force_refresh:
            return self._schema_cache

        self._schema_cache = self._build_demo_schema()
        logger.info("schema_loaded", table_count=len(self._schema_cache))
        return self._schema_cache

    def get_schema_context(self) -> str:
        """Generate a compact schema description for LLM prompt injection.

        Returns:
            Multi-line string describing all tables, columns, and relationships.
        """
        schema = self.get_schema()
        parts = []
        for table_info in schema.values():
            parts.append(table_info.to_ddl_context())
        return "\n\n".join(parts)

    def get_table_names(self) -> list[str]:
        """Return a list of all available table names."""
        return list(self.get_schema().keys())

    def find_tables_for_query(self, query: str) -> list[TableInfo]:
        """Identify which tables are likely relevant to a natural language query.

        Args:
            query: User's natural language query.

        Returns:
            List of TableInfo objects for relevant tables.
        """
        query_upper = query.upper()
        schema = self.get_schema()
        relevant = []

        for table_name, table_info in schema.items():
            # Check if table name or description matches
            if table_name in query_upper:
                relevant.append(table_info)
                continue

            # Check column names and descriptions
            for col in table_info.columns:
                if col.name in query_upper or (col.description and any(
                    word in query_upper for word in col.description.upper().split()
                    if len(word) > 3
                )):
                    relevant.append(table_info)
                    break

        # If nothing matched, include core tables
        if not relevant:
            core_tables = ["TRANSACTIONS", "MERCHANTS", "CUSTOMERS"]
            relevant = [schema[t] for t in core_tables if t in schema]

        return relevant

    @staticmethod
    def _build_demo_schema() -> dict[str, TableInfo]:
        """Build the predefined payment schema for demo mode."""
        schema: dict[str, TableInfo] = {}

        # TRANSACTIONS table
        schema["TRANSACTIONS"] = TableInfo(
            name="TRANSACTIONS",
            description="Core payment transaction records with amounts, status, and metadata",
            columns=[
                ColumnInfo("TRANSACTION_ID", "VARCHAR(36)", False, "Unique transaction identifier (UUID)"),
                ColumnInfo("MERCHANT_ID", "VARCHAR(36)", False, "Reference to MERCHANTS table"),
                ColumnInfo("CUSTOMER_ID", "VARCHAR(36)", False, "Reference to CUSTOMERS table"),
                ColumnInfo("TRANSACTION_DATE", "TIMESTAMP_NTZ", False, "When the transaction occurred"),
                ColumnInfo("AMOUNT", "DECIMAL(12,2)", False, "Transaction amount in USD"),
                ColumnInfo("CURRENCY", "VARCHAR(3)", False, "ISO 4217 currency code"),
                ColumnInfo("PAYMENT_METHOD", "VARCHAR(20)", False, "Payment method: CREDIT, DEBIT, ACH, WIRE"),
                ColumnInfo("CARD_BRAND", "VARCHAR(20)", True, "Card network: VISA, MASTERCARD, AMEX, DISCOVER"),
                ColumnInfo("CARD_LAST_FOUR", "VARCHAR(4)", True, "Last four digits of card number"),
                ColumnInfo("STATUS", "VARCHAR(20)", False, "Transaction status: APPROVED, DECLINED, PENDING, REFUNDED"),
                ColumnInfo("DECLINE_REASON", "VARCHAR(100)", True, "Reason code if declined"),
                ColumnInfo("AUTH_CODE", "VARCHAR(10)", True, "Authorization code from processor"),
                ColumnInfo("RISK_SCORE", "DECIMAL(5,2)", True, "Transaction risk score 0-100"),
                ColumnInfo("CHANNEL", "VARCHAR(20)", True, "Transaction channel: ONLINE, IN_STORE, MOBILE, PHONE"),
                ColumnInfo("REGION", "VARCHAR(50)", True, "Geographic region of the transaction"),
                ColumnInfo("COUNTRY_CODE", "VARCHAR(2)", True, "ISO 3166-1 alpha-2 country code"),
                ColumnInfo("FEE_AMOUNT", "DECIMAL(8,2)", True, "Processing fee amount"),
                ColumnInfo("CREATED_AT", "TIMESTAMP_NTZ", False, "Record creation timestamp"),
            ],
            primary_key=["TRANSACTION_ID"],
            foreign_keys={
                "MERCHANT_ID": "MERCHANTS.MERCHANT_ID",
                "CUSTOMER_ID": "CUSTOMERS.CUSTOMER_ID",
            },
            row_count_estimate=5_000_000,
        )

        # MERCHANTS table
        schema["MERCHANTS"] = TableInfo(
            name="MERCHANTS",
            description="Merchant profiles with business details and risk categorization",
            columns=[
                ColumnInfo("MERCHANT_ID", "VARCHAR(36)", False, "Unique merchant identifier (UUID)"),
                ColumnInfo("MERCHANT_NAME", "VARCHAR(200)", False, "Business name"),
                ColumnInfo("MCC_CODE", "VARCHAR(4)", False, "Merchant Category Code"),
                ColumnInfo("MCC_DESCRIPTION", "VARCHAR(100)", True, "MCC category description"),
                ColumnInfo("REGION", "VARCHAR(50)", True, "Operating region"),
                ColumnInfo("COUNTRY_CODE", "VARCHAR(2)", True, "Country of registration"),
                ColumnInfo("RISK_TIER", "VARCHAR(10)", True, "Risk classification: LOW, MEDIUM, HIGH"),
                ColumnInfo("ONBOARDING_DATE", "DATE", True, "Date merchant was onboarded"),
                ColumnInfo("STATUS", "VARCHAR(20)", False, "Account status: ACTIVE, SUSPENDED, CLOSED"),
                ColumnInfo("MONTHLY_VOLUME_LIMIT", "DECIMAL(14,2)", True, "Monthly processing volume limit"),
                ColumnInfo("CREATED_AT", "TIMESTAMP_NTZ", False, "Record creation timestamp"),
            ],
            primary_key=["MERCHANT_ID"],
            row_count_estimate=50_000,
        )

        # CUSTOMERS table
        schema["CUSTOMERS"] = TableInfo(
            name="CUSTOMERS",
            description="Customer profiles with account details (PII masked)",
            columns=[
                ColumnInfo("CUSTOMER_ID", "VARCHAR(36)", False, "Unique customer identifier (UUID)"),
                ColumnInfo("CUSTOMER_HASH", "VARCHAR(64)", False, "Hashed customer identifier for privacy"),
                ColumnInfo("CUSTOMER_SEGMENT", "VARCHAR(20)", True, "Segment: PREMIUM, STANDARD, NEW"),
                ColumnInfo("COUNTRY_CODE", "VARCHAR(2)", True, "Customer country"),
                ColumnInfo("REGION", "VARCHAR(50)", True, "Customer region"),
                ColumnInfo("ACCOUNT_CREATED_DATE", "DATE", True, "Account creation date"),
                ColumnInfo("RISK_SCORE", "DECIMAL(5,2)", True, "Customer risk score 0-100"),
                ColumnInfo("LIFETIME_TRANSACTION_COUNT", "INTEGER", True, "Total transactions processed"),
                ColumnInfo("LIFETIME_TRANSACTION_VALUE", "DECIMAL(14,2)", True, "Total transaction value USD"),
                ColumnInfo("CREATED_AT", "TIMESTAMP_NTZ", False, "Record creation timestamp"),
            ],
            primary_key=["CUSTOMER_ID"],
            row_count_estimate=500_000,
        )

        # SETTLEMENTS table
        schema["SETTLEMENTS"] = TableInfo(
            name="SETTLEMENTS",
            description="Settlement batches linking transactions to merchant payouts",
            columns=[
                ColumnInfo("SETTLEMENT_ID", "VARCHAR(36)", False, "Unique settlement identifier"),
                ColumnInfo("MERCHANT_ID", "VARCHAR(36)", False, "Reference to MERCHANTS table"),
                ColumnInfo("SETTLEMENT_DATE", "DATE", False, "Date funds were settled"),
                ColumnInfo("TRANSACTION_COUNT", "INTEGER", False, "Number of transactions in settlement"),
                ColumnInfo("GROSS_AMOUNT", "DECIMAL(14,2)", False, "Total gross amount"),
                ColumnInfo("FEE_AMOUNT", "DECIMAL(10,2)", False, "Total fees deducted"),
                ColumnInfo("NET_AMOUNT", "DECIMAL(14,2)", False, "Net amount paid to merchant"),
                ColumnInfo("CURRENCY", "VARCHAR(3)", False, "Settlement currency"),
                ColumnInfo("PAYMENT_METHOD", "VARCHAR(20)", True, "Primary payment method in batch"),
                ColumnInfo("STATUS", "VARCHAR(20)", False, "Settlement status: PENDING, COMPLETED, FAILED"),
                ColumnInfo("SETTLEMENT_DAYS", "INTEGER", True, "Days from transaction to settlement"),
                ColumnInfo("CREATED_AT", "TIMESTAMP_NTZ", False, "Record creation timestamp"),
            ],
            primary_key=["SETTLEMENT_ID"],
            foreign_keys={"MERCHANT_ID": "MERCHANTS.MERCHANT_ID"},
            row_count_estimate=200_000,
        )

        # CHARGEBACKS table
        schema["CHARGEBACKS"] = TableInfo(
            name="CHARGEBACKS",
            description="Chargeback disputes with reason codes and resolution status",
            columns=[
                ColumnInfo("CHARGEBACK_ID", "VARCHAR(36)", False, "Unique chargeback identifier"),
                ColumnInfo("TRANSACTION_ID", "VARCHAR(36)", False, "Original transaction reference"),
                ColumnInfo("MERCHANT_ID", "VARCHAR(36)", False, "Merchant involved in dispute"),
                ColumnInfo("CHARGEBACK_DATE", "DATE", False, "Date chargeback was filed"),
                ColumnInfo("AMOUNT", "DECIMAL(12,2)", False, "Disputed amount"),
                ColumnInfo("CURRENCY", "VARCHAR(3)", False, "Currency of dispute"),
                ColumnInfo("REASON_CODE", "VARCHAR(10)", False, "Chargeback reason code"),
                ColumnInfo("REASON_DESCRIPTION", "VARCHAR(200)", True, "Human-readable reason"),
                ColumnInfo("STATUS", "VARCHAR(20)", False, "Status: OPEN, WON, LOST, EXPIRED"),
                ColumnInfo("RESOLUTION_DATE", "DATE", True, "Date dispute was resolved"),
                ColumnInfo("RESOLUTION_DAYS", "INTEGER", True, "Days from filing to resolution"),
                ColumnInfo("REPRESENTMENT_SUBMITTED", "BOOLEAN", True, "Whether representment was filed"),
                ColumnInfo("CREATED_AT", "TIMESTAMP_NTZ", False, "Record creation timestamp"),
            ],
            primary_key=["CHARGEBACK_ID"],
            foreign_keys={
                "TRANSACTION_ID": "TRANSACTIONS.TRANSACTION_ID",
                "MERCHANT_ID": "MERCHANTS.MERCHANT_ID",
            },
            row_count_estimate=25_000,
        )

        return schema
