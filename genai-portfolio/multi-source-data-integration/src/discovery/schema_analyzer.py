"""Schema structure analysis.

Compares source and target schemas to identify structural differences, type
mismatches, missing columns, and naming convention divergences.  Produces an
analysis report that feeds into the schema-mapping phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class MatchConfidence(Enum):
    EXACT = "exact"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    description: str = ""


@dataclass
class TableInfo:
    name: str
    schema_name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_key: Optional[str] = None
    row_count: int = 0


@dataclass
class ColumnMatch:
    """Represents a proposed mapping between a source and target column."""

    source_column: str
    target_column: str
    confidence: MatchConfidence
    reason: str = ""
    type_compatible: bool = True
    type_conversion_needed: Optional[str] = None


@dataclass
class SchemaComparison:
    """Result of comparing a source table to a target table."""

    source_table: str
    target_table: str
    source_system: str
    matched_columns: List[ColumnMatch] = field(default_factory=list)
    unmapped_source_columns: List[str] = field(default_factory=list)
    unmapped_target_columns: List[str] = field(default_factory=list)

    @property
    def mapping_coverage_pct(self) -> float:
        total_source = len(self.matched_columns) + len(self.unmapped_source_columns)
        if total_source == 0:
            return 0.0
        return round(len(self.matched_columns) / total_source * 100, 1)


class SchemaAnalyzer:
    """Analyse and compare source/target schemas for mapping compatibility.

    Uses heuristic name matching (normalised lower-case, common abbreviation
    expansion) and type-compatibility checks to propose column mappings
    before the human-curated mapping YAML is written.
    """

    # Common abbreviations seen in legacy schemas
    _ABBREVIATIONS: Dict[str, str] = {
        "nm": "name", "nme": "name",
        "addr": "address",
        "dt": "date",
        "amt": "amount",
        "qty": "quantity",
        "pct": "percent",
        "desc": "description",
        "cd": "code",
        "cntry": "country",
        "st": "state",
        "cust": "customer",
        "prod": "product",
        "cat": "category",
        "id": "id",
        "sk": "surrogate_key",
        "bk": "business_key",
        "fk": "foreign_key",
    }

    def __init__(self) -> None:
        self._type_compat = _TypeCompatibilityChecker()

    def _normalize_name(self, name: str) -> str:
        """Convert to lowercase, strip underscores, expand abbreviations."""
        tokens = name.lower().replace("-", "_").split("_")
        expanded = [self._ABBREVIATIONS.get(t, t) for t in tokens if t]
        return "_".join(expanded)

    def _match_confidence(
        self, source_norm: str, target_norm: str
    ) -> MatchConfidence:
        if source_norm == target_norm:
            return MatchConfidence.EXACT
        # One is a substring of the other
        if source_norm in target_norm or target_norm in source_norm:
            return MatchConfidence.HIGH
        # Share significant overlap (Jaccard on tokens)
        src_tokens = set(source_norm.split("_"))
        tgt_tokens = set(target_norm.split("_"))
        if src_tokens and tgt_tokens:
            jaccard = len(src_tokens & tgt_tokens) / len(src_tokens | tgt_tokens)
            if jaccard >= 0.5:
                return MatchConfidence.MEDIUM
            if jaccard > 0:
                return MatchConfidence.LOW
        return MatchConfidence.NONE

    def compare_tables(
        self,
        source: TableInfo,
        target: TableInfo,
        source_system: str = "",
    ) -> SchemaComparison:
        """Compare source and target table schemas, proposing column matches."""
        matched: List[ColumnMatch] = []
        used_targets: set[str] = set()

        for src_col in source.columns:
            src_norm = self._normalize_name(src_col.name)
            best_match: Optional[ColumnMatch] = None
            best_confidence = MatchConfidence.NONE

            for tgt_col in target.columns:
                if tgt_col.name in used_targets:
                    continue
                tgt_norm = self._normalize_name(tgt_col.name)
                confidence = self._match_confidence(src_norm, tgt_norm)

                if confidence.value < best_confidence.value or best_match is None:
                    type_ok, conv = self._type_compat.check(
                        src_col.data_type, tgt_col.data_type
                    )
                    candidate = ColumnMatch(
                        source_column=src_col.name,
                        target_column=tgt_col.name,
                        confidence=confidence,
                        reason=f"name_similarity({src_norm} -> {tgt_norm})",
                        type_compatible=type_ok,
                        type_conversion_needed=conv,
                    )
                    if (confidence != MatchConfidence.NONE and
                            (best_match is None or
                             list(MatchConfidence).index(confidence) <
                             list(MatchConfidence).index(best_confidence))):
                        best_match = candidate
                        best_confidence = confidence

            if best_match and best_confidence != MatchConfidence.NONE:
                matched.append(best_match)
                used_targets.add(best_match.target_column)

        unmapped_src = [
            c.name for c in source.columns
            if c.name not in {m.source_column for m in matched}
        ]
        unmapped_tgt = [
            c.name for c in target.columns
            if c.name not in used_targets
        ]

        comp = SchemaComparison(
            source_table=f"{source.schema_name}.{source.name}",
            target_table=f"{target.schema_name}.{target.name}",
            source_system=source_system,
            matched_columns=matched,
            unmapped_source_columns=unmapped_src,
            unmapped_target_columns=unmapped_tgt,
        )
        logger.info(
            "Schema comparison %s -> %s: %d matched, %d unmapped src, %d unmapped tgt (%.1f%% coverage)",
            comp.source_table, comp.target_table,
            len(matched), len(unmapped_src), len(unmapped_tgt),
            comp.mapping_coverage_pct,
        )
        return comp

    def analyze_source_config(
        self, source_cfg: Dict[str, Any], target_cfg: Dict[str, Any]
    ) -> List[SchemaComparison]:
        """Compare all tables in a source config against the target config."""
        target_tables = self._extract_tables(target_cfg)
        source_tables = self._extract_tables(source_cfg)
        system_id = source_cfg.get("system", {}).get("id", "")
        results: List[SchemaComparison] = []

        for src_tbl in source_tables:
            # Find best-matching target table by name similarity
            best_tgt: Optional[TableInfo] = None
            best_score = -1
            src_norm = self._normalize_name(src_tbl.name)
            for tgt_tbl in target_tables:
                tgt_norm = self._normalize_name(tgt_tbl.name)
                score = len(set(src_norm.split("_")) & set(tgt_norm.split("_")))
                if score > best_score:
                    best_score = score
                    best_tgt = tgt_tbl

            if best_tgt and best_score > 0:
                results.append(self.compare_tables(src_tbl, best_tgt, system_id))

        return results

    def _extract_tables(self, cfg: Dict[str, Any]) -> List[TableInfo]:
        tables: List[TableInfo] = []
        for schema in cfg.get("schemas", []):
            schema_name = schema.get("name", "")
            for tbl in schema.get("tables", []):
                cols = [
                    ColumnInfo(
                        name=c["name"],
                        data_type=c.get("type", "VARCHAR"),
                        nullable=c.get("nullable", True),
                    )
                    for c in tbl.get("columns", [])
                ]
                tables.append(TableInfo(
                    name=tbl["name"],
                    schema_name=schema_name,
                    columns=cols,
                    primary_key=tbl.get("primary_key"),
                    row_count=tbl.get("estimated_rows", 0),
                ))
        return tables


class _TypeCompatibilityChecker:
    """Check whether a source data type is directly compatible with a target
    type and, if not, suggest the required conversion."""

    _COMPATIBLE_GROUPS = {
        "integer": {"NUMBER", "INT", "BIGINT", "INTEGER", "SMALLINT", "NUMBER(38,0)"},
        "decimal": {"NUMBER", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"},
        "string": {"VARCHAR", "VARCHAR2", "NVARCHAR", "CHAR", "NCHAR", "TEXT", "STRING"},
        "datetime": {"DATE", "DATETIME", "DATETIME2", "TIMESTAMP", "TIMESTAMP_NTZ",
                      "TIMESTAMP_LTZ", "TIMESTAMP_TZ"},
        "boolean": {"BOOLEAN", "BIT", "CHAR(1)"},
    }

    def _base_type(self, full_type: str) -> str:
        """Strip size/precision to get the base type name."""
        paren = full_type.find("(")
        return full_type[:paren].strip().upper() if paren > 0 else full_type.strip().upper()

    def _group(self, base: str) -> Optional[str]:
        for group, members in self._COMPATIBLE_GROUPS.items():
            if base in members:
                return group
        return None

    def check(self, source_type: str, target_type: str) -> Tuple[bool, Optional[str]]:
        src_base = self._base_type(source_type)
        tgt_base = self._base_type(target_type)

        if src_base == tgt_base:
            return True, None

        src_group = self._group(src_base)
        tgt_group = self._group(tgt_base)

        if src_group and src_group == tgt_group:
            return True, f"CAST({source_type} AS {target_type})"

        if src_group and tgt_group:
            return False, f"CONVERT({source_type} -> {target_type})"

        return False, f"MANUAL_CONVERSION({source_type} -> {target_type})"
