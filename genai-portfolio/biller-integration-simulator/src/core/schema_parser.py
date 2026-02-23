"""
Oracle CC&B schema parsing and field mapping engine.

Translates data between the Oracle CC&B schema (CI_ACCT, CI_PER, CI_BILL, etc.)
and the payment platform's canonical data model. Mappings are defined in
config/schema_mapping.yaml and applied at runtime when ingesting CIS extracts
or pushing payment confirmations back to the CIS.

This is the "Rosetta Stone" layer that lets the platform speak to any CIS
regardless of its internal schema — provided a mapping config exists.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaMappingError(Exception):
    """Raised when a field mapping fails."""
    pass


class SchemaParser:
    """
    Parses Oracle CC&B schema mapping configuration and applies transforms.

    Usage:
        parser = SchemaParser()
        canonical = parser.map_record("account_mapping", ccb_row)
    """

    def __init__(self, config_path: Optional[Path] = None):
        from config import SCHEMA_MAPPING_PATH
        self._config_path = config_path or SCHEMA_MAPPING_PATH
        self._mappings: dict[str, Any] = {}
        self._transforms: dict[str, Callable] = {}
        self._load_config()
        self._register_transforms()

    def _load_config(self) -> None:
        with open(self._config_path, "r") as f:
            self._mappings = yaml.safe_load(f)
        logger.info(
            "Loaded schema mapping configuration",
            extra={"event_data": {"path": str(self._config_path)}},
        )

    def _register_transforms(self) -> None:
        """Register the built-in transform functions referenced in the YAML."""
        self._transforms = {
            "strip_leading_zeros": self._strip_leading_zeros,
            "oracle_date_to_iso8601": self._oracle_date_to_iso8601,
            "status_flag_decode": None,  # requires lookup — handled specially
            "flag_decode": None,
            "sa_type_decode": None,
            "boolean_flag": self._boolean_flag,
            "lowercase": self._lowercase,
            "decimal_2": self._decimal_2,
            "parse_alert_flags": self._parse_alert_flags,
        }

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def map_record(
        self,
        mapping_name: str,
        source_record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Map a single source record (CC&B row) to the canonical platform model.

        Args:
            mapping_name: One of "account_mapping", "person_mapping",
                          "bill_mapping", "bill_segment_mapping",
                          "service_agreement_mapping".
            source_record: Dict with CC&B column names as keys.

        Returns:
            Dict with canonical platform field names and transformed values.
        """
        mapping_section = self._mappings.get(mapping_name)
        if mapping_section is None:
            raise SchemaMappingError(f"Unknown mapping: {mapping_name}")

        field_defs = mapping_section.get("fields", [])
        result: dict[str, Any] = {}

        for field_def in field_defs:
            source_col = field_def["source"]
            target_col = field_def["target"]
            transform_name = field_def.get("transform")
            required = field_def.get("required", False)
            default = field_def.get("default")
            lookup = field_def.get("lookup")

            raw_value = source_record.get(source_col)

            # Handle missing values
            if raw_value is None:
                if required and default is None:
                    raise SchemaMappingError(
                        f"Required field '{source_col}' missing from source record "
                        f"(mapping: {mapping_name})"
                    )
                raw_value = default

            # Apply transform
            if raw_value is not None and transform_name:
                try:
                    raw_value = self._apply_transform(
                        transform_name, raw_value, lookup
                    )
                except Exception as exc:
                    logger.warning(
                        f"Transform '{transform_name}' failed for {source_col}={raw_value}",
                        extra={"event_data": {
                            "mapping": mapping_name,
                            "field": source_col,
                            "error": str(exc),
                        }},
                    )
                    if required:
                        raise SchemaMappingError(
                            f"Transform '{transform_name}' failed on required field "
                            f"'{source_col}': {exc}"
                        ) from exc

            result[target_col] = raw_value

        return result

    def map_records(
        self,
        mapping_name: str,
        source_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map a batch of source records."""
        results = []
        for i, rec in enumerate(source_records):
            try:
                results.append(self.map_record(mapping_name, rec))
            except SchemaMappingError as exc:
                logger.error(
                    f"Mapping failed for record {i}",
                    extra={"event_data": {"index": i, "error": str(exc)}},
                )
                raise
        return results

    def get_table_name(self, mapping_name: str) -> str:
        """Return the CC&B table name for a given mapping."""
        section = self._mappings.get(mapping_name, {})
        return section.get("table", "UNKNOWN")

    def get_field_names(self, mapping_name: str) -> list[str]:
        """Return the list of source (CC&B) field names for a mapping."""
        section = self._mappings.get(mapping_name, {})
        return [f["source"] for f in section.get("fields", [])]

    def get_required_fields(self, mapping_name: str) -> list[str]:
        """Return the list of required source fields for a mapping."""
        section = self._mappings.get(mapping_name, {})
        return [
            f["source"]
            for f in section.get("fields", [])
            if f.get("required", False)
        ]

    def validate_source_record(
        self,
        mapping_name: str,
        source_record: dict[str, Any],
    ) -> list[str]:
        """
        Check a source record for missing required fields.

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []
        section = self._mappings.get(mapping_name, {})

        for field_def in section.get("fields", []):
            if field_def.get("required", False):
                source_col = field_def["source"]
                if source_col not in source_record or source_record[source_col] is None:
                    if field_def.get("default") is None:
                        errors.append(f"Missing required field: {source_col}")

        return errors

    # ---------------------------------------------------------------
    # Transform implementations
    # ---------------------------------------------------------------

    def _apply_transform(
        self,
        transform_name: str,
        value: Any,
        lookup: Optional[dict] = None,
    ) -> Any:
        """Route to the appropriate transform function."""
        # Lookup-based transforms
        if transform_name in ("status_flag_decode", "flag_decode", "sa_type_decode"):
            return self._lookup_decode(value, lookup)

        func = self._transforms.get(transform_name)
        if func is None:
            logger.warning(f"Unknown transform: {transform_name}, returning raw value")
            return value

        return func(value)

    @staticmethod
    def _strip_leading_zeros(value: Any) -> str:
        s = str(value)
        return s.lstrip("0") or "0"

    @staticmethod
    def _oracle_date_to_iso8601(value: Any) -> str:
        """Convert Oracle DATE/TIMESTAMP string to ISO 8601."""
        if isinstance(value, (date, datetime)):
            return value.isoformat()

        s = str(value).strip()
        # Try common Oracle date formats
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d-%b-%Y",
            "%d-%b-%y",
            "%Y/%m/%d",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.date().isoformat()
            except ValueError:
                continue

        # If nothing matches, return as-is
        return s

    @staticmethod
    def _boolean_flag(value: Any) -> bool:
        return str(value).upper() in ("Y", "YES", "1", "TRUE")

    @staticmethod
    def _lowercase(value: Any) -> str:
        return str(value).lower() if value else ""

    @staticmethod
    def _decimal_2(value: Any) -> Decimal:
        try:
            return Decimal(str(value)).quantize(Decimal("0.01"))
        except (InvalidOperation, ValueError):
            return Decimal("0.00")

    @staticmethod
    def _parse_alert_flags(value: Any) -> list[str]:
        s = str(value) if value else ""
        return [x.strip() for x in s.split("|") if x.strip()]

    @staticmethod
    def _lookup_decode(value: Any, lookup: Optional[dict] = None) -> Any:
        if lookup is None:
            return value
        key = str(value)
        decoded = lookup.get(key)
        if decoded is None:
            logger.warning(
                f"Lookup miss: key='{key}' not found in lookup table",
                extra={"event_data": {"key": key, "available_keys": list(lookup.keys())}},
            )
            return key
        return decoded
