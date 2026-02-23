"""Tests for the schema mapping engine and transform functions."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mapping.transform_engine import TransformEngine
from src.mapping.schema_mapper import SchemaMapper, ColumnMapping
from src.mapping.type_converter import TypeConverter, Platform
from src.mapping.mapping_validator import MappingValidator, Severity


class TestTransformEngine:
    """Unit tests for individual transform functions."""

    def setup_method(self) -> None:
        self.engine = TransformEngine()

    def test_direct_passthrough(self) -> None:
        row = {"name": "Alice", "age": 30}
        result = self.engine.execute("DIRECT", row, source_column="name")
        assert result == "Alice"

    def test_direct_missing_column(self) -> None:
        row = {"name": "Alice"}
        result = self.engine.execute("DIRECT", row, source_column="missing")
        assert result is None

    def test_constant(self) -> None:
        row = {"x": 1}
        result = self.engine.execute("CONSTANT", row, params={"value": "legacy_oracle"})
        assert result == "legacy_oracle"

    def test_cast_to_varchar(self) -> None:
        row = {"id": 12345}
        result = self.engine.execute("CAST_TO_VARCHAR", row, source_column="id")
        assert result == "12345"
        assert isinstance(result, str)

    def test_concatenate(self) -> None:
        row = {"first": "John", "last": "Doe"}
        result = self.engine.execute(
            "CONCATENATE", row,
            source_columns=["first", "last"],
            params={"separator": " ", "skip_null": True},
        )
        assert result == "John Doe"

    def test_concatenate_with_null(self) -> None:
        row = {"addr1": "123 Main St", "addr2": None}
        result = self.engine.execute(
            "CONCATENATE", row,
            source_columns=["addr1", "addr2"],
            params={"separator": ", ", "skip_null": True},
        )
        assert result == "123 Main St"

    def test_lowercase(self) -> None:
        row = {"email": "John.Doe@EXAMPLE.COM"}
        result = self.engine.execute("LOWERCASE", row, source_column="email")
        assert result == "john.doe@example.com"

    def test_lookup_transform(self) -> None:
        row = {"status": "A"}
        result = self.engine.execute(
            "LOOKUP", row,
            source_column="status",
            params={
                "lookup_map": {"A": "Active", "I": "Inactive"},
                "default": "Unknown",
            },
        )
        assert result == "Active"

    def test_lookup_default(self) -> None:
        row = {"status": "X"}
        result = self.engine.execute(
            "LOOKUP", row,
            source_column="status",
            params={
                "lookup_map": {"A": "Active"},
                "default": "Unknown",
            },
        )
        assert result == "Unknown"

    def test_normalize_phone_10_digits(self) -> None:
        row = {"phone": "(555) 123-4567"}
        result = self.engine.execute("NORMALIZE_PHONE", row, source_column="phone")
        assert result == "+15551234567"

    def test_normalize_phone_11_digits(self) -> None:
        row = {"phone": "+12125551234"}
        result = self.engine.execute("NORMALIZE_PHONE", row, source_column="phone")
        assert result == "+12125551234"

    def test_extract_first_name(self) -> None:
        row = {"full_name": "Jane Alice Smith"}
        result = self.engine.execute(
            "EXTRACT_FIRST_NAME", row, source_columns=["full_name"]
        )
        assert result == "Jane"

    def test_extract_last_name(self) -> None:
        row = {"full_name": "Jane Alice Smith"}
        result = self.engine.execute(
            "EXTRACT_LAST_NAME", row, source_columns=["full_name"]
        )
        assert result == "Alice Smith"

    def test_pad_iso_country(self) -> None:
        row = {"country": "US"}
        result = self.engine.execute("PAD_ISO_COUNTRY", row, source_column="country")
        assert result == "USA"

    def test_prefix(self) -> None:
        row = {"id": 12345}
        result = self.engine.execute(
            "PREFIX", row, source_column="id", params={"prefix": "ORA-"}
        )
        assert result == "ORA-12345"

    def test_bit_to_boolean(self) -> None:
        assert self.engine.execute("BIT_TO_BOOLEAN", {"x": 1}, source_column="x") is True
        assert self.engine.execute("BIT_TO_BOOLEAN", {"x": 0}, source_column="x") is False

    def test_trim_upper_first(self) -> None:
        row = {"name": "  jOHN dOE  "}
        result = self.engine.execute("TRIM_UPPER_FIRST", row, source_column="name")
        assert result == "John Doe"

    def test_unknown_transform_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transform"):
            self.engine.execute("NONEXISTENT", {})


class TestSchemaMapper:
    """Integration tests for the schema mapper with YAML config."""

    def setup_method(self) -> None:
        self.mapper = SchemaMapper()
        config_dir = PROJECT_ROOT / "config" / "mapping_rules"
        self.mapper.load_mapping_file(str(config_dir / "customer_mapping.yaml"))

    def test_entities_loaded(self) -> None:
        assert "customer" in self.mapper.list_entities()

    def test_oracle_customer_mapping(self) -> None:
        row = {
            "CUST_ID": 1001,
            "CUST_NM": "John Smith",
            "CUST_ADDR_1": "123 Oak St",
            "CUST_ADDR_2": "Suite 5",
            "CUST_CITY": "Denver",
            "CUST_ST": "CO",
            "CUST_ZIP": "80201",
            "CUST_CNTRY": "USA",
            "CUST_EMAIL": "John.Smith@CORP.COM",
            "CUST_PHONE": "(303) 555-1234",
            "TAX_ID": "12-3456789",
            "CUST_TYPE": "I",
            "CUST_STATUS": "A",
            "CREATED_DT": "2020-01-15 10:30:00",
            "MODIFIED_DT": "2024-06-01 14:20:00",
        }
        results = self.mapper.apply("customer", "legacy_oracle", [row])
        assert len(results) == 1
        target = results[0]
        assert target["SOURCE_ID"] == "1001"
        assert target["CUSTOMER_NAME"] == "John Smith"
        assert target["FULL_ADDRESS"] == "123 Oak St, Suite 5"
        assert target["EMAIL"] == "john.smith@corp.com"
        assert target["CUSTOMER_TYPE"] == "Individual"
        assert target["STATUS"] == "Active"
        assert target["SOURCE_SYSTEM"] == "legacy_oracle"

    def test_sqlserver_customer_mapping(self) -> None:
        row = {
            "CustomerID": 5001,
            "FirstName": "Jane",
            "LastName": "Doe",
            "CompanyName": None,
            "EmailAddress": "jane.doe@gmail.com",
            "PhoneNumber": "212-555-9876",
            "AddressLine1": "456 Elm Ave",
            "AddressLine2": None,
            "City": "New York",
            "StateProvince": "NY",
            "PostalCode": "10001",
            "CountryCode": "US",
            "TaxIdentifier": "98-7654321",
            "CustomerType": "Individual",
            "Status": "Active",
            "CreatedDate": "2019-08-20T09:15:00",
            "ModifiedDate": "2024-03-10T11:45:00",
        }
        results = self.mapper.apply("customer", "legacy_sqlserver", [row])
        assert len(results) == 1
        target = results[0]
        assert target["SOURCE_ID"] == "5001"
        assert target["CUSTOMER_NAME"] == "Jane Doe"
        assert target["COUNTRY_CODE"] == "USA"
        assert target["SOURCE_SYSTEM"] == "legacy_sqlserver"


class TestTypeConverter:
    """Tests for cross-platform type conversion."""

    def setup_method(self) -> None:
        self.converter = TypeConverter()

    def test_oracle_varchar2_to_snowflake(self) -> None:
        result = self.converter.convert("VARCHAR2(200)", Platform.ORACLE)
        assert "VARCHAR" in result.target_type
        assert result.source_platform == Platform.ORACLE

    def test_oracle_number_to_snowflake(self) -> None:
        result = self.converter.convert("NUMBER(12,2)", Platform.ORACLE)
        assert "NUMBER" in result.target_type

    def test_oracle_date_to_snowflake(self) -> None:
        result = self.converter.convert("DATE", Platform.ORACLE)
        assert "TIMESTAMP" in result.target_type

    def test_sqlserver_bit_to_snowflake(self) -> None:
        result = self.converter.convert("BIT", Platform.SQLSERVER)
        assert result.target_type == "BOOLEAN"

    def test_sqlserver_nvarchar_to_snowflake(self) -> None:
        result = self.converter.convert("NVARCHAR(255)", Platform.SQLSERVER)
        assert "VARCHAR" in result.target_type

    def test_sqlserver_datetime2_to_snowflake(self) -> None:
        result = self.converter.convert("DATETIME2", Platform.SQLSERVER)
        assert "TIMESTAMP" in result.target_type

    def test_unknown_type_defaults_to_variant(self) -> None:
        result = self.converter.convert("CUSTOM_TYPE", Platform.ORACLE)
        assert result.target_type == "VARIANT"


class TestMappingValidator:
    """Tests for mapping completeness validation."""

    def test_valid_mapping(self) -> None:
        mapping_config = {
            "entity": "test",
            "target_table": "INTEGRATED.DIM_TEST",
            "sources": [
                {
                    "system_id": "src1",
                    "source_table": "schema.TABLE1",
                    "mappings": [
                        {"source_column": "COL_A", "target_column": "COLUMN_A", "transform": "DIRECT"},
                        {"source_column": "COL_B", "target_column": "COLUMN_B", "transform": "DIRECT"},
                    ],
                }
            ],
            "generated_columns": [
                {"column": "COLUMN_C", "strategy": "SEQUENCE"},
            ],
        }
        source_configs = {
            "src1": {
                "schemas": [{"name": "schema", "tables": [
                    {"name": "TABLE1", "columns": [
                        {"name": "COL_A", "type": "VARCHAR"},
                        {"name": "COL_B", "type": "VARCHAR"},
                    ]}
                ]}]
            }
        }
        target_config = {
            "schemas": [{"name": "INTEGRATED", "tables": [
                {"name": "DIM_TEST", "columns": [
                    {"name": "COLUMN_A", "type": "VARCHAR", "nullable": False},
                    {"name": "COLUMN_B", "type": "VARCHAR", "nullable": True},
                    {"name": "COLUMN_C", "type": "NUMBER", "nullable": False},
                ]}
            ]}]
        }

        validator = MappingValidator(strict_mode=True)
        result = validator.validate(mapping_config, source_configs, target_config)
        assert result.is_valid
        assert result.error_count == 0
