"""SQL query safety validation and guardrails.

Implements defense-in-depth validation for SQL queries generated from
natural language. Checks for injection patterns, blocked operations,
excessive complexity, and schema boundary violations before any query
is executed against the database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML

import yaml

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of SQL query validation.

    Attributes:
        is_safe: Whether the query passed all safety checks.
        reason: Explanation if the query was rejected.
        warnings: Non-blocking issues found during validation.
        normalized_sql: The parsed and normalized form of the query.
    """

    is_safe: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    normalized_sql: str = ""


class QueryValidator:
    """Validates SQL queries for safety, compliance, and schema boundaries.

    Validation layers:
    1. Statement type check (only SELECT allowed)
    2. Blocked keyword detection (DROP, DELETE, etc.)
    3. SQL injection pattern detection
    4. Subquery depth limiting
    5. Schema boundary enforcement
    6. Resource guard checks (row limits, join counts)
    """

    # Operations that are never allowed in analytics queries
    DEFAULT_BLOCKED_OPS = frozenset({
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
        "INSERT", "UPDATE", "MERGE", "GRANT", "REVOKE",
        "EXEC", "EXECUTE", "CALL",
    })

    # Dangerous functions that could be used for exploitation
    BLOCKED_FUNCTIONS = frozenset({
        "SYSTEM$", "SYSTEM_", "XP_", "SP_",
        "LOAD_FILE", "INTO OUTFILE", "INTO DUMPFILE",
        "BENCHMARK", "SLEEP", "WAITFOR",
    })

    def __init__(
        self,
        blocked_operations: set[str] | None = None,
        max_subquery_depth: int = 3,
        max_joins: int = 5,
        allowed_tables: set[str] | None = None,
    ) -> None:
        self.blocked_operations = blocked_operations or set(self.DEFAULT_BLOCKED_OPS)
        self.max_subquery_depth = max_subquery_depth
        self.max_joins = max_joins
        self.allowed_tables = allowed_tables or {
            "TRANSACTIONS", "MERCHANTS", "CUSTOMERS",
            "SETTLEMENTS", "CHARGEBACKS",
        }

        # Load additional guardrails from config if available
        self._load_config_overrides()

    def _load_config_overrides(self) -> None:
        """Load guardrail overrides from agent_config.yaml."""
        config_path = Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                guardrails = config.get("guardrails", {})
                extra_blocked = guardrails.get("blocked_operations", [])
                if extra_blocked:
                    self.blocked_operations.update(op.upper() for op in extra_blocked)
            except Exception as exc:
                logger.warning("config_load_warning", error=str(exc))

    def validate(self, sql: str) -> ValidationResult:
        """Run the full validation pipeline on a SQL query.

        Args:
            sql: The SQL query to validate.

        Returns:
            ValidationResult indicating whether the query is safe.
        """
        if not sql or not sql.strip():
            return ValidationResult(is_safe=False, reason="Empty query")

        warnings: list[str] = []

        # Normalize and parse
        try:
            normalized = self._normalize_sql(sql)
        except Exception as exc:
            return ValidationResult(
                is_safe=False,
                reason=f"SQL parsing failed: {exc}",
            )

        # Check 1: Statement type
        type_result = self._check_statement_type(normalized)
        if not type_result.is_safe:
            return type_result

        # Check 2: Blocked operations
        blocked_result = self._check_blocked_operations(normalized)
        if not blocked_result.is_safe:
            return blocked_result

        # Check 3: Injection patterns
        injection_result = self._check_injection_patterns(sql)
        if not injection_result.is_safe:
            return injection_result

        # Check 4: Blocked functions
        func_result = self._check_blocked_functions(normalized)
        if not func_result.is_safe:
            return func_result

        # Check 5: Subquery depth
        depth = self._measure_subquery_depth(normalized)
        if depth > self.max_subquery_depth:
            return ValidationResult(
                is_safe=False,
                reason=f"Query subquery depth ({depth}) exceeds maximum ({self.max_subquery_depth})",
            )
        if depth > 1:
            warnings.append(f"Query uses {depth} levels of subqueries")

        # Check 6: Join count
        join_count = len(re.findall(r"\bJOIN\b", normalized, re.IGNORECASE))
        if join_count > self.max_joins:
            return ValidationResult(
                is_safe=False,
                reason=f"Query join count ({join_count}) exceeds maximum ({self.max_joins})",
            )
        if join_count > 3:
            warnings.append(f"Query uses {join_count} JOINs, which may impact performance")

        # Check 7: Table references are within allowed schema
        table_result = self._check_table_references(normalized)
        if not table_result.is_safe:
            warnings.append(table_result.reason)

        # Check 8: No LIMIT on large-table queries (add warning)
        if not re.search(r"\bLIMIT\b", normalized, re.IGNORECASE):
            if not re.search(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", normalized, re.IGNORECASE):
                warnings.append("Query has no LIMIT clause; results may be large")

        logger.info(
            "query_validated",
            is_safe=True,
            warnings=warnings,
            sql_preview=normalized[:80],
        )

        return ValidationResult(
            is_safe=True,
            warnings=warnings,
            normalized_sql=normalized,
        )

    def _normalize_sql(self, sql: str) -> str:
        """Parse and normalize SQL for consistent analysis."""
        formatted = sqlparse.format(
            sql.strip().rstrip(";"),
            keyword_case="upper",
            strip_comments=True,
            strip_whitespace=True,
        )
        return formatted

    def _check_statement_type(self, sql: str) -> ValidationResult:
        """Ensure only SELECT statements are allowed."""
        parsed = sqlparse.parse(sql)
        if not parsed:
            return ValidationResult(is_safe=False, reason="Unable to parse SQL")

        statement = parsed[0]
        stmt_type = statement.get_type()

        if stmt_type and stmt_type.upper() != "SELECT":
            logger.warning("blocked_statement_type", type=stmt_type)
            return ValidationResult(
                is_safe=False,
                reason=f"Only SELECT queries are allowed. Got: {stmt_type}",
            )

        # Also check the first meaningful token
        first_keyword = None
        for token in statement.tokens:
            if token.ttype in (DML, Keyword) and str(token).strip():
                first_keyword = str(token).strip().upper()
                break

        if first_keyword and first_keyword not in ("SELECT", "WITH"):
            return ValidationResult(
                is_safe=False,
                reason=f"Query must start with SELECT or WITH. Found: {first_keyword}",
            )

        return ValidationResult(is_safe=True)

    def _check_blocked_operations(self, sql: str) -> ValidationResult:
        """Check for blocked SQL operations."""
        sql_upper = sql.upper()
        for operation in self.blocked_operations:
            # Match as whole word to avoid false positives
            pattern = rf"\b{re.escape(operation)}\b"
            if re.search(pattern, sql_upper):
                logger.warning("blocked_operation_detected", operation=operation)
                return ValidationResult(
                    is_safe=False,
                    reason=f"Blocked SQL operation detected: {operation}",
                )
        return ValidationResult(is_safe=True)

    def _check_injection_patterns(self, sql: str) -> ValidationResult:
        """Detect SQL injection patterns."""
        injection_patterns = [
            (r";\s*\w", "Multiple statements detected (semicolon followed by keyword)"),
            (r"(?:--|#)\s*$", "Comment-based injection pattern"),
            (r"'\s*(?:OR|AND)\s+['\d]", "Boolean-based injection pattern"),
            (r"UNION\s+(?:ALL\s+)?SELECT", "UNION-based injection pattern"),
            (r"(?:CHAR|CHR|CONCAT)\s*\(.*\d{2,}", "Character encoding injection"),
            (r"0x[0-9a-fA-F]+", "Hex-encoded injection"),
        ]

        for pattern, description in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE | re.MULTILINE):
                logger.warning("injection_pattern_detected", pattern=description)
                return ValidationResult(
                    is_safe=False,
                    reason=f"Potential SQL injection: {description}",
                )

        return ValidationResult(is_safe=True)

    def _check_blocked_functions(self, sql: str) -> ValidationResult:
        """Check for dangerous SQL functions."""
        sql_upper = sql.upper()
        for func in self.BLOCKED_FUNCTIONS:
            if func in sql_upper:
                return ValidationResult(
                    is_safe=False,
                    reason=f"Blocked function detected: {func}",
                )
        return ValidationResult(is_safe=True)

    def _check_table_references(self, sql: str) -> ValidationResult:
        """Verify that referenced tables are within the allowed schema."""
        # Extract table references using a simplified pattern
        # Matches FROM <table> and JOIN <table> patterns
        table_pattern = r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)"
        referenced_tables = re.findall(table_pattern, sql, re.IGNORECASE)

        unknown_tables = []
        for table_ref in referenced_tables:
            # Handle schema-qualified names (e.g., SCHEMA.TABLE)
            table_name = table_ref.split(".")[-1].upper()
            if table_name not in self.allowed_tables:
                unknown_tables.append(table_name)

        if unknown_tables:
            return ValidationResult(
                is_safe=True,  # Warning, not a hard block
                reason=f"References to unknown tables: {', '.join(unknown_tables)}",
            )

        return ValidationResult(is_safe=True)

    @staticmethod
    def _measure_subquery_depth(sql: str) -> int:
        """Measure the maximum nesting depth of subqueries."""
        max_depth = 0
        current_depth = 0
        in_string = False
        string_char = None

        for char in sql:
            if in_string:
                if char == string_char:
                    in_string = False
                continue

            if char in ("'", '"'):
                in_string = True
                string_char = char
            elif char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth = max(0, current_depth - 1)

        return max_depth
