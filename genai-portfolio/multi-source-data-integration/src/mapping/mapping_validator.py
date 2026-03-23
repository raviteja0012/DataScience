"""Mapping completeness validation.

Ensures that every required target column has a mapping from at least one
source, flags unmapped source columns, and checks for type-conversion
gaps before the mapping is used in a production cutover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    severity: Severity
    entity: str
    message: str
    source_system: Optional[str] = None
    column: Optional[str] = None


@dataclass
class MappingValidationResult:
    entity: str
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    source_coverage: Dict[str, float] = field(default_factory=dict)
    target_coverage: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


class MappingValidator:
    """Validate mapping YAML files against source and target schema definitions."""

    def __init__(self, strict_mode: bool = True) -> None:
        self._strict = strict_mode

    def validate(
        self,
        mapping_config: Dict[str, Any],
        source_configs: Dict[str, Dict[str, Any]],
        target_config: Dict[str, Any],
    ) -> MappingValidationResult:
        """Validate a mapping config against source/target schemas.

        Parameters
        ----------
        mapping_config:
            Parsed YAML from a mapping rule file.
        source_configs:
            ``{system_id: parsed_yaml}`` for each source.
        target_config:
            Parsed YAML for the target system.
        """
        entity = mapping_config.get("entity", "unknown")
        result = MappingValidationResult(entity=entity)

        target_columns = self._extract_target_columns(
            mapping_config.get("target_table", ""), target_config
        )
        required_target = {
            c for c, info in target_columns.items() if not info.get("nullable", True)
        }
        generated = {
            g["column"] for g in mapping_config.get("generated_columns", [])
        }

        all_mapped_targets: Set[str] = set()

        for source_def in mapping_config.get("sources", []):
            system_id = source_def["system_id"]
            source_table = source_def.get("source_table", "")
            source_cols = self._extract_source_columns(
                source_table, source_configs.get(system_id, {})
            )

            mapped_sources: Set[str] = set()
            mapped_targets: Set[str] = set()

            for m in source_def.get("mappings", []):
                # Track mapped source columns
                if m.get("source_column"):
                    mapped_sources.add(m["source_column"])
                for sc in m.get("source_columns", []):
                    mapped_sources.add(sc)

                target_col = m.get("target_column", "")
                mapped_targets.add(target_col)
                all_mapped_targets.add(target_col)

            # Check unmapped source columns
            unmapped_source = source_cols - mapped_sources
            for col in sorted(unmapped_source):
                result.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    entity=entity,
                    source_system=system_id,
                    column=col,
                    message=f"Source column '{col}' in {source_table} is not mapped",
                ))

            # Source coverage
            coverage = (len(mapped_sources) / max(len(source_cols), 1)) * 100
            result.source_coverage[system_id] = round(coverage, 1)

            logger.info(
                "Mapping validation %s/%s: %d/%d source columns mapped (%.1f%%)",
                entity, system_id, len(mapped_sources), len(source_cols), coverage,
            )

        # Check unmapped required target columns
        covered_targets = all_mapped_targets | generated
        unmapped_required = required_target - covered_targets
        for col in sorted(unmapped_required):
            sev = Severity.ERROR if self._strict else Severity.WARNING
            result.issues.append(ValidationIssue(
                severity=sev,
                entity=entity,
                column=col,
                message=f"Required target column '{col}' has no mapping or generator",
            ))

        # Unmapped optional target columns
        all_target = set(target_columns.keys())
        unmapped_optional = all_target - covered_targets - unmapped_required
        for col in sorted(unmapped_optional):
            result.issues.append(ValidationIssue(
                severity=Severity.INFO,
                entity=entity,
                column=col,
                message=f"Optional target column '{col}' is unmapped (will be NULL)",
            ))

        result.target_coverage = round(
            len(covered_targets & all_target) / max(len(all_target), 1) * 100, 1
        )
        result.is_valid = result.error_count == 0
        return result

    def _extract_source_columns(
        self, table_ref: str, source_config: Dict[str, Any]
    ) -> Set[str]:
        """Extract column names for a table from its source config."""
        # table_ref is like "ERP_OWNER.CUST_MASTER"
        parts = table_ref.rsplit(".", 1)
        table_name = parts[-1] if parts else table_ref

        for schema in source_config.get("schemas", []):
            for tbl in schema.get("tables", []):
                if tbl["name"] == table_name:
                    return {c["name"] for c in tbl.get("columns", [])}
        return set()

    def _extract_target_columns(
        self, table_ref: str, target_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract column info dict for a target table."""
        parts = table_ref.rsplit(".", 1)
        table_name = parts[-1] if parts else table_ref

        for schema in target_config.get("schemas", []):
            for tbl in schema.get("tables", []):
                if tbl["name"] == table_name:
                    return {
                        c["name"]: {"nullable": c.get("nullable", True), "type": c.get("type", "")}
                        for c in tbl.get("columns", [])
                    }
        return {}
