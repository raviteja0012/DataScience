"""Tests for SQL query safety validation.

Validates that the query validator correctly identifies and blocks
dangerous SQL patterns while allowing legitimate analytics queries.
"""

from __future__ import annotations

import pytest

from src.analytics.query_validator import QueryValidator, ValidationResult


@pytest.fixture
def validator() -> QueryValidator:
    """Create a QueryValidator with default settings."""
    return QueryValidator()


class TestSafeQueries:
    """Test that legitimate analytics queries pass validation."""

    def test_simple_select(self, validator: QueryValidator) -> None:
        result = validator.validate("SELECT * FROM TRANSACTIONS LIMIT 10")
        assert result.is_safe

    def test_aggregation_query(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT MERCHANT_ID, COUNT(*), SUM(AMOUNT) "
            "FROM TRANSACTIONS GROUP BY MERCHANT_ID"
        )
        assert result.is_safe

    def test_join_query(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT m.MERCHANT_NAME, COUNT(*) "
            "FROM TRANSACTIONS t "
            "JOIN MERCHANTS m ON t.MERCHANT_ID = m.MERCHANT_ID "
            "GROUP BY m.MERCHANT_NAME"
        )
        assert result.is_safe

    def test_subquery(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS "
            "WHERE AMOUNT > (SELECT AVG(AMOUNT) FROM TRANSACTIONS)"
        )
        assert result.is_safe

    def test_cte_query(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "WITH daily AS (SELECT DATE_TRUNC('DAY', TRANSACTION_DATE) AS d, "
            "COUNT(*) AS cnt FROM TRANSACTIONS GROUP BY d) "
            "SELECT * FROM daily ORDER BY d"
        )
        assert result.is_safe

    def test_window_function(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT MERCHANT_ID, AMOUNT, "
            "ROW_NUMBER() OVER (PARTITION BY MERCHANT_ID ORDER BY AMOUNT DESC) AS rn "
            "FROM TRANSACTIONS"
        )
        assert result.is_safe


class TestBlockedOperations:
    """Test that dangerous SQL operations are correctly blocked."""

    def test_drop_table(self, validator: QueryValidator) -> None:
        result = validator.validate("DROP TABLE TRANSACTIONS")
        assert not result.is_safe
        assert "DROP" in result.reason

    def test_delete(self, validator: QueryValidator) -> None:
        result = validator.validate("DELETE FROM TRANSACTIONS WHERE AMOUNT < 0")
        assert not result.is_safe

    def test_insert(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "INSERT INTO TRANSACTIONS (TRANSACTION_ID) VALUES ('test')"
        )
        assert not result.is_safe

    def test_update(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "UPDATE TRANSACTIONS SET AMOUNT = 0 WHERE AMOUNT < 0"
        )
        assert not result.is_safe

    def test_truncate(self, validator: QueryValidator) -> None:
        result = validator.validate("TRUNCATE TABLE TRANSACTIONS")
        assert not result.is_safe

    def test_grant(self, validator: QueryValidator) -> None:
        result = validator.validate("GRANT SELECT ON TRANSACTIONS TO PUBLIC")
        assert not result.is_safe

    def test_alter(self, validator: QueryValidator) -> None:
        result = validator.validate("ALTER TABLE TRANSACTIONS ADD COLUMN test VARCHAR")
        assert not result.is_safe


class TestInjectionDetection:
    """Test that SQL injection patterns are detected and blocked."""

    def test_semicolon_injection(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS; DROP TABLE TRANSACTIONS"
        )
        assert not result.is_safe

    def test_union_injection(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS UNION SELECT * FROM CUSTOMERS"
        )
        assert not result.is_safe

    def test_boolean_injection(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS WHERE MERCHANT_ID = '' OR '1'='1'"
        )
        assert not result.is_safe

    def test_comment_injection(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS WHERE MERCHANT_ID = 'test' --"
        )
        assert not result.is_safe

    def test_hex_injection(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS WHERE MERCHANT_ID = 0x74657374"
        )
        assert not result.is_safe


class TestBlockedFunctions:
    """Test that dangerous SQL functions are blocked."""

    def test_system_function(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT SYSTEM$TYPEOF(AMOUNT) FROM TRANSACTIONS"
        )
        assert not result.is_safe

    def test_load_file(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT LOAD_FILE('/etc/passwd')"
        )
        assert not result.is_safe

    def test_sleep_function(self, validator: QueryValidator) -> None:
        result = validator.validate(
            "SELECT SLEEP(10)"
        )
        assert not result.is_safe


class TestResourceGuards:
    """Test query complexity limits."""

    def test_excessive_joins(self) -> None:
        validator = QueryValidator(max_joins=2)
        result = validator.validate(
            "SELECT * FROM TRANSACTIONS t "
            "JOIN MERCHANTS m ON t.MERCHANT_ID = m.MERCHANT_ID "
            "JOIN CUSTOMERS c ON t.CUSTOMER_ID = c.CUSTOMER_ID "
            "JOIN SETTLEMENTS s ON t.MERCHANT_ID = s.MERCHANT_ID "
        )
        assert not result.is_safe

    def test_deep_subqueries(self) -> None:
        validator = QueryValidator(max_subquery_depth=2)
        result = validator.validate(
            "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT 1)))"
        )
        assert not result.is_safe

    def test_no_limit_warning(self, validator: QueryValidator) -> None:
        result = validator.validate("SELECT * FROM TRANSACTIONS")
        assert result.is_safe
        assert any("LIMIT" in w for w in result.warnings)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query(self, validator: QueryValidator) -> None:
        result = validator.validate("")
        assert not result.is_safe

    def test_whitespace_only(self, validator: QueryValidator) -> None:
        result = validator.validate("   ")
        assert not result.is_safe

    def test_case_insensitive_blocking(self, validator: QueryValidator) -> None:
        result = validator.validate("drop table transactions")
        assert not result.is_safe

    def test_normalized_output(self, validator: QueryValidator) -> None:
        result = validator.validate("select count(*) from transactions")
        assert result.is_safe
        # Normalized SQL should have uppercase keywords
        assert "SELECT" in result.normalized_sql
        assert "FROM" in result.normalized_sql
