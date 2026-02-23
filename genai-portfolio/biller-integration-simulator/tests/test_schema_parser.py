"""Tests for the Oracle CC&B schema parser and field mapping engine."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from src.core.schema_parser import SchemaParser, SchemaMappingError


@pytest.fixture
def parser() -> SchemaParser:
    return SchemaParser()


class TestSchemaParser:
    """Tests for CC&B-to-platform schema mapping."""

    def test_account_mapping(self, parser: SchemaParser) -> None:
        """CI_ACCT fields should map to canonical account fields."""
        source = {
            "ACCT_ID": "0001234567",
            "CIS_DIVISION": "GP01",
            "SETUP_DT": "2022-03-15",
            "ACCT_STATUS_FLG": "20",
            "CURRENCY_CD": "USD",
        }

        result = parser.map_record("account_mapping", source)

        assert result["customer_account_id"] == "1234567"  # stripped leading zeros
        assert result["biller_division_code"] == "GP01"
        assert result["account_status"] == "active"  # decoded from "20"
        assert result["currency_code"] == "USD"

    def test_person_mapping(self, parser: SchemaParser) -> None:
        """CI_PER fields should map to canonical person fields."""
        source = {
            "PER_ID": "9876543210",
            "PER_OR_BUS_FLG": "P",
            "EMAILID": "John.Smith@Example.COM",
            "LANG_CD": "ENG",
        }

        result = parser.map_record("person_mapping", source)

        assert result["person_id"] == "9876543210"
        assert result["entity_type"] == "person"
        assert result["email_address"] == "john.smith@example.com"

    def test_bill_mapping(self, parser: SchemaParser) -> None:
        """CI_BILL fields should map with proper decimal and date transforms."""
        source = {
            "BILL_ID": "100000000001",
            "ACCT_ID": "0001234567",
            "BILL_DT": "2024-01-15",
            "DUE_DT": "2024-02-05",
            "BILL_AMT": "145.67",
            "PREV_UNPAID_AMT": "23.10",
            "TOT_AMT_DUE": "168.77",
            "BILL_STATUS_FLG": "C",
        }

        result = parser.map_record("bill_mapping", source)

        assert result["bill_id"] == "100000000001"
        assert result["current_charges"] == Decimal("145.67")
        assert result["previous_balance"] == Decimal("23.10")
        assert result["total_amount_due"] == Decimal("168.77")
        assert result["bill_status"] == "complete"

    def test_bill_segment_mapping(self, parser: SchemaParser) -> None:
        """CI_BILL_SEG should map with date and amount transforms."""
        source = {
            "BILLSEG_ID": "200000000001",
            "BILL_ID": "100000000001",
            "SA_ID": "3000000001",
            "START_DT": "2023-12-15",
            "END_DT": "2024-01-14",
            "CALC_AMT": "145.67",
        }

        result = parser.map_record("bill_segment_mapping", source)

        assert result["bill_segment_id"] == "200000000001"
        assert result["service_agreement_id"] == "3000000001"
        assert result["calculated_amount"] == Decimal("145.67")

    def test_service_agreement_mapping(self, parser: SchemaParser) -> None:
        """CI_SA should map with SA type decode."""
        source = {
            "SA_ID": "3000000001",
            "ACCT_ID": "0001234567",
            "SA_TYPE_CD": "E-RES",
            "SA_STATUS_FLG": "20",
            "START_DT": "2020-06-01",
        }

        result = parser.map_record("service_agreement_mapping", source)

        assert result["service_type"] == "electric_residential"
        assert result["agreement_status"] == "active"

    def test_missing_required_field_raises(self, parser: SchemaParser) -> None:
        """Missing a required field with no default should raise an error."""
        source = {
            "CIS_DIVISION": "GP01",
            # ACCT_ID is missing
        }

        with pytest.raises(SchemaMappingError, match="Required field.*ACCT_ID"):
            parser.map_record("account_mapping", source)

    def test_default_value_applied(self, parser: SchemaParser) -> None:
        """Optional fields with defaults should use the default when missing."""
        source = {
            "PER_ID": "1111111111",
            "PER_OR_BUS_FLG": "B",
            # LANG_CD is missing — should default to "ENG"
        }

        result = parser.map_record("person_mapping", source)
        assert result["preferred_language"] == "ENG"

    def test_status_flag_unknown_value(self, parser: SchemaParser) -> None:
        """An unrecognized status flag should pass through as-is."""
        source = {
            "ACCT_ID": "0001234567",
            "CIS_DIVISION": "GP01",
            "ACCT_STATUS_FLG": "99",  # Not in lookup
            "CURRENCY_CD": "USD",
        }

        result = parser.map_record("account_mapping", source)
        # Unknown code falls through as-is
        assert result["account_status"] == "99"

    def test_batch_mapping(self, parser: SchemaParser) -> None:
        """map_records should handle multiple records."""
        records = [
            {
                "PER_ID": "1111111111",
                "PER_OR_BUS_FLG": "P",
                "EMAILID": "a@b.com",
            },
            {
                "PER_ID": "2222222222",
                "PER_OR_BUS_FLG": "B",
                "EMAILID": "C@D.com",
            },
        ]

        results = parser.map_records("person_mapping", records)
        assert len(results) == 2
        assert results[1]["email_address"] == "c@d.com"

    def test_get_table_name(self, parser: SchemaParser) -> None:
        assert parser.get_table_name("account_mapping") == "CI_ACCT"
        assert parser.get_table_name("bill_mapping") == "CI_BILL"

    def test_get_required_fields(self, parser: SchemaParser) -> None:
        required = parser.get_required_fields("bill_mapping")
        assert "BILL_ID" in required
        assert "ACCT_ID" in required
        assert "BILL_DT" in required

    def test_validate_source_record(self, parser: SchemaParser) -> None:
        """Validation should report missing required fields."""
        incomplete = {"BILL_ID": "123"}  # Missing many required fields
        errors = parser.validate_source_record("bill_mapping", incomplete)
        assert len(errors) > 0
        assert any("ACCT_ID" in e for e in errors)

    def test_boolean_flag_transform(self, parser: SchemaParser) -> None:
        """Y/N flags should convert to boolean."""
        source = {
            "PER_ID": "1111111111",
            "PER_OR_BUS_FLG": "P",
            "LIFE_SUPPORT_FLG": "Y",
        }

        result = parser.map_record("person_mapping", source)
        assert result["life_support_flag"] is True

    def test_oracle_date_formats(self, parser: SchemaParser) -> None:
        """Parser should handle multiple Oracle date formats."""
        source_with_datetime = {
            "ACCT_ID": "0001234567",
            "CIS_DIVISION": "GP01",
            "SETUP_DT": "2022-03-15 14:30:00",
            "ACCT_STATUS_FLG": "20",
            "CURRENCY_CD": "USD",
        }

        result = parser.map_record("account_mapping", source_with_datetime)
        assert result["account_created_date"] == "2022-03-15"
