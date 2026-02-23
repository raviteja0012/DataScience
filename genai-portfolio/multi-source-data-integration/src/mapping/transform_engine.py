"""Data transformation rule engine.

Implements every transform type referenced in the mapping YAML files.
Each transform is a pure function: ``(row, params) -> value``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")

# ISO 3166-1 alpha-2 to alpha-3 (subset for demo)
_COUNTRY_2_TO_3 = {
    "US": "USA", "CA": "CAN", "MX": "MEX", "GB": "GBR", "DE": "DEU",
    "FR": "FRA", "JP": "JPN", "CN": "CHN", "IN": "IND", "BR": "BRA",
    "AU": "AUS", "KR": "KOR", "IT": "ITA", "ES": "ESP", "NL": "NLD",
}


class TransformEngine:
    """Registry of named transforms, each producing a single target value
    from a source row and parameters.
    """

    def __init__(self) -> None:
        self._transforms: Dict[str, Callable[..., Any]] = {
            "DIRECT": self._direct,
            "CONSTANT": self._constant,
            "CAST_TO_VARCHAR": self._cast_to_varchar,
            "CAST_TIMESTAMP": self._cast_timestamp,
            "CAST_DATE": self._cast_date,
            "CONCATENATE": self._concatenate,
            "TRIM": self._trim,
            "TRIM_UPPER_FIRST": self._trim_upper_first,
            "LOWERCASE": self._lowercase,
            "LOOKUP": self._lookup,
            "NORMALIZE_PHONE": self._normalize_phone,
            "EXTRACT_FIRST_NAME": self._extract_first_name,
            "EXTRACT_LAST_NAME": self._extract_last_name,
            "PAD_ISO_COUNTRY": self._pad_iso_country,
            "PREFIX": self._prefix,
            "BIT_TO_BOOLEAN": self._bit_to_boolean,
            "CONDITIONAL": self._conditional,
            "SURROGATE_KEY_LOOKUP": self._surrogate_key_lookup,
        }

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._transforms[name] = fn

    def execute(
        self,
        transform_name: str,
        row: Dict[str, Any],
        source_column: Optional[str] = None,
        source_columns: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        fn = self._transforms.get(transform_name)
        if fn is None:
            raise ValueError(f"Unknown transform: {transform_name}")
        return fn(
            row=row,
            source_column=source_column,
            source_columns=source_columns,
            params=params or {},
        )

    # -- Transform implementations -------------------------------------------

    @staticmethod
    def _direct(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Any:
        if source_column is None:
            return None
        return row.get(source_column)

    @staticmethod
    def _constant(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Any:
        return params.get("value")

    @staticmethod
    def _cast_to_varchar(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        return str(val) if val is not None else None

    @staticmethod
    def _cast_timestamp(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.isoformat()
        # Already a string; normalise
        return str(val).replace("T", " ")

    @staticmethod
    def _cast_date(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        return str(val)[:10]  # YYYY-MM-DD

    @staticmethod
    def _concatenate(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        cols = source_columns or []
        separator = params.get("separator", " ")
        skip_null = params.get("skip_null", False)
        parts: List[str] = []
        for col in cols:
            val = row.get(col)
            if val is None:
                if not skip_null:
                    parts.append("")
            else:
                parts.append(str(val).strip())
        result = separator.join(p for p in parts if p) if skip_null else separator.join(parts)
        return result if result.strip() else None

    @staticmethod
    def _trim(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        return val.strip() if isinstance(val, str) else val

    @staticmethod
    def _trim_upper_first(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        if not isinstance(val, str):
            return val
        return " ".join(w.capitalize() for w in val.strip().split())

    @staticmethod
    def _lowercase(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        return val.lower() if isinstance(val, str) else val

    @staticmethod
    def _lookup(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Any:
        val = row.get(source_column) if source_column else None
        lookup_map = params.get("lookup_map", {})
        default = params.get("default")
        if val is None:
            return default
        return lookup_map.get(str(val), default)

    @staticmethod
    def _normalize_phone(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        digits = re.sub(r"\D", "", str(val))
        if len(digits) == 10:
            return f"+1{digits}"
        if len(digits) == 11 and digits[0] == "1":
            return f"+{digits}"
        return f"+{digits}" if digits else None

    @staticmethod
    def _extract_first_name(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        cols = source_columns or ([source_column] if source_column else [])
        val = row.get(cols[0]) if cols else None
        if not isinstance(val, str):
            return None
        parts = val.strip().split()
        return parts[0] if parts else None

    @staticmethod
    def _extract_last_name(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        cols = source_columns or ([source_column] if source_column else [])
        val = row.get(cols[0]) if cols else None
        if not isinstance(val, str):
            return None
        parts = val.strip().split()
        return " ".join(parts[1:]) if len(parts) > 1 else None

    @staticmethod
    def _pad_iso_country(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        code2 = str(val).strip().upper()
        return _COUNTRY_2_TO_3.get(code2, code2)

    @staticmethod
    def _prefix(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[str]:
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        return f"{params.get('prefix', '')}{val}"

    @staticmethod
    def _bit_to_boolean(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[bool]:
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        return bool(int(val))

    @staticmethod
    def _conditional(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Any:
        condition = params.get("condition", "")
        # Simple evaluator for "COLUMN = 'VALUE'" style conditions
        if "=" in condition:
            col, expected = [s.strip().strip("'\"") for s in condition.split("=", 1)]
            actual = str(row.get(col, ""))
            if actual == expected:
                val_col = params.get("value_if_true")
                return row.get(val_col) if val_col else None
        return params.get("value_if_false")

    @staticmethod
    def _surrogate_key_lookup(
        row: Dict[str, Any],
        source_column: Optional[str],
        source_columns: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Any:
        """Placeholder for surrogate-key resolution during load.

        In a real pipeline this would query the target dimension table.
        During mapping-only runs we return a synthetic placeholder.
        """
        val = row.get(source_column) if source_column else None
        if val is None:
            return None
        return int(val)  # pass-through for demo
