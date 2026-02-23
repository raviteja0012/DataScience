"""Cross-platform type conversion utilities.

Provides mapping tables and conversion logic for translating data types
between Oracle, SQL Server, and Snowflake.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class Platform(Enum):
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"


@dataclass
class TypeMapping:
    source_platform: Platform
    source_type: str
    target_platform: Platform
    target_type: str
    notes: str = ""
    precision_loss: bool = False


# ---------------------------------------------------------------------------
# Canonical type-mapping tables
# ---------------------------------------------------------------------------

_ORACLE_TO_SNOWFLAKE: Dict[str, str] = {
    "NUMBER": "NUMBER",
    "VARCHAR2": "VARCHAR",
    "CHAR": "VARCHAR",
    "NVARCHAR2": "VARCHAR",
    "NCHAR": "VARCHAR",
    "CLOB": "VARCHAR(16777216)",
    "NCLOB": "VARCHAR(16777216)",
    "BLOB": "BINARY",
    "DATE": "TIMESTAMP_NTZ",
    "TIMESTAMP": "TIMESTAMP_NTZ",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMP_TZ",
    "TIMESTAMP WITH LOCAL TIME ZONE": "TIMESTAMP_LTZ",
    "RAW": "BINARY",
    "LONG": "VARCHAR(16777216)",
    "LONG RAW": "BINARY",
    "FLOAT": "FLOAT",
    "BINARY_FLOAT": "FLOAT",
    "BINARY_DOUBLE": "FLOAT",
    "XMLTYPE": "VARIANT",
}

_SQLSERVER_TO_SNOWFLAKE: Dict[str, str] = {
    "INT": "NUMBER(10,0)",
    "BIGINT": "NUMBER(19,0)",
    "SMALLINT": "NUMBER(5,0)",
    "TINYINT": "NUMBER(3,0)",
    "BIT": "BOOLEAN",
    "DECIMAL": "NUMBER",
    "NUMERIC": "NUMBER",
    "MONEY": "NUMBER(19,4)",
    "SMALLMONEY": "NUMBER(10,4)",
    "FLOAT": "FLOAT",
    "REAL": "FLOAT",
    "VARCHAR": "VARCHAR",
    "NVARCHAR": "VARCHAR",
    "CHAR": "VARCHAR",
    "NCHAR": "VARCHAR",
    "TEXT": "VARCHAR(16777216)",
    "NTEXT": "VARCHAR(16777216)",
    "DATE": "DATE",
    "DATETIME": "TIMESTAMP_NTZ",
    "DATETIME2": "TIMESTAMP_NTZ",
    "DATETIMEOFFSET": "TIMESTAMP_TZ",
    "SMALLDATETIME": "TIMESTAMP_NTZ",
    "TIME": "TIME",
    "BINARY": "BINARY",
    "VARBINARY": "BINARY",
    "IMAGE": "BINARY",
    "UNIQUEIDENTIFIER": "VARCHAR(36)",
    "XML": "VARIANT",
    "SQL_VARIANT": "VARIANT",
}


class TypeConverter:
    """Convert data-type strings between source and target platforms."""

    def __init__(self) -> None:
        self._maps: Dict[Tuple[Platform, Platform], Dict[str, str]] = {
            (Platform.ORACLE, Platform.SNOWFLAKE): _ORACLE_TO_SNOWFLAKE,
            (Platform.SQLSERVER, Platform.SNOWFLAKE): _SQLSERVER_TO_SNOWFLAKE,
        }

    def _base_type(self, full_type: str) -> Tuple[str, str]:
        """Split ``VARCHAR2(200)`` into ``('VARCHAR2', '(200)')``."""
        paren = full_type.find("(")
        if paren >= 0:
            return full_type[:paren].strip().upper(), full_type[paren:]
        return full_type.strip().upper(), ""

    def convert(
        self,
        source_type: str,
        source_platform: Platform,
        target_platform: Platform = Platform.SNOWFLAKE,
    ) -> TypeMapping:
        """Return the recommended target type for a given source type."""
        base, precision = self._base_type(source_type)
        lookup = self._maps.get((source_platform, target_platform), {})
        target_base = lookup.get(base)

        if target_base is None:
            logger.warning(
                "No mapping for %s(%s) -> %s; defaulting to VARIANT",
                base, source_platform.value, target_platform.value,
            )
            return TypeMapping(
                source_platform=source_platform,
                source_type=source_type,
                target_platform=target_platform,
                target_type="VARIANT",
                notes="unmapped type — manual review required",
            )

        # Transfer precision where applicable
        target_type = target_base
        precision_loss = False
        if precision and target_base in ("VARCHAR", "NUMBER", "BINARY"):
            target_type = f"{target_base}{precision}"
        elif precision and "(" not in target_base:
            # Precision may be lost (e.g. CHAR(2) -> BOOLEAN)
            precision_loss = True

        return TypeMapping(
            source_platform=source_platform,
            source_type=source_type,
            target_platform=target_platform,
            target_type=target_type,
            precision_loss=precision_loss,
        )

    def convert_all(
        self,
        columns: List[Dict[str, str]],
        source_platform: Platform,
        target_platform: Platform = Platform.SNOWFLAKE,
    ) -> List[TypeMapping]:
        """Batch-convert a list of column defs ``[{name, type}, ...]``."""
        results: List[TypeMapping] = []
        for col in columns:
            tm = self.convert(col["type"], source_platform, target_platform)
            results.append(tm)
        return results

    def build_ddl_column(self, col_name: str, mapping: TypeMapping, nullable: bool = True) -> str:
        """Generate a Snowflake DDL column fragment."""
        null_clause = "" if nullable else " NOT NULL"
        return f"  {col_name} {mapping.target_type}{null_clause}"
