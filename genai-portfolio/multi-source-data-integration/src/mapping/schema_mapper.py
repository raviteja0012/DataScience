"""Schema-to-schema mapping engine.

Reads YAML mapping rule files and applies configured transformations to
translate source records into target schema records.  Each mapping rule is a
declarative instruction (e.g. ``DIRECT``, ``CONCATENATE``, ``LOOKUP``) that
the engine dispatches to the appropriate transform function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import yaml

from src.mapping.transform_engine import TransformEngine
from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class ColumnMapping:
    """A single source-to-target column mapping rule."""

    source_column: Optional[str]
    source_columns: Optional[List[str]]
    target_column: str
    transform: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnMapping":
        return cls(
            source_column=d.get("source_column"),
            source_columns=d.get("source_columns"),
            target_column=d["target_column"],
            transform=d.get("transform", "DIRECT"),
            params={k: v for k, v in d.items()
                    if k not in ("source_column", "source_columns",
                                 "target_column", "transform", "description")},
        )


@dataclass
class SourceMapping:
    """Mapping rules for one source table -> one target table."""

    system_id: str
    source_table: str
    target_table: str
    column_mappings: List[ColumnMapping] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], target_table: str) -> "SourceMapping":
        return cls(
            system_id=d["system_id"],
            source_table=d["source_table"],
            target_table=target_table,
            column_mappings=[ColumnMapping.from_dict(m) for m in d.get("mappings", [])],
        )


@dataclass
class EntityMapping:
    """All source mappings for one logical entity (e.g. customer)."""

    entity: str
    target_table: str
    sources: List[SourceMapping] = field(default_factory=list)
    generated_columns: List[Dict[str, Any]] = field(default_factory=list)


class SchemaMapper:
    """Load mapping rules from YAML and transform source records.

    Usage::

        mapper = SchemaMapper()
        mapper.load_mapping_file("config/mapping_rules/customer_mapping.yaml")
        target_records = mapper.apply("legacy_oracle", source_records)
    """

    def __init__(self) -> None:
        self._entity_mappings: Dict[str, EntityMapping] = {}
        self._transform_engine = TransformEngine()

    def load_mapping_file(self, path: str) -> EntityMapping:
        """Parse a mapping YAML file and register it."""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        entity = cfg.get("entity", "unknown")
        target_table = cfg.get("target_table", "")

        # Handle the simple format (single target_table at top level)
        if "sources" in cfg:
            sources = [
                SourceMapping.from_dict(s, target_table)
                for s in cfg["sources"]
            ]
            em = EntityMapping(
                entity=entity,
                target_table=target_table,
                sources=sources,
                generated_columns=cfg.get("generated_columns", []),
            )
            self._entity_mappings[entity] = em
            logger.info(
                "Loaded mapping for entity=%s, target=%s, %d source(s)",
                entity, target_table, len(sources),
            )
            return em

        # Handle the multi-table format (transaction_mapping.yaml)
        if "tables" in cfg:
            for tbl_cfg in cfg["tables"]:
                tgt = tbl_cfg["target_table"]
                sources = [
                    SourceMapping.from_dict(s, tgt)
                    for s in tbl_cfg.get("sources", [])
                ]
                sub_entity = f"{entity}_{tgt.split('.')[-1].lower()}"
                em = EntityMapping(
                    entity=sub_entity,
                    target_table=tgt,
                    sources=sources,
                    generated_columns=tbl_cfg.get("generated_columns", []),
                )
                self._entity_mappings[sub_entity] = em
            logger.info(
                "Loaded multi-table mapping for entity=%s (%d sub-entities)",
                entity, len(cfg["tables"]),
            )
            # Return the first for convenience
            first_key = next(
                k for k in self._entity_mappings if k.startswith(entity)
            )
            return self._entity_mappings[first_key]

        raise ValueError(f"Unsupported mapping format in {path}")

    def get_entity_mapping(self, entity: str) -> Optional[EntityMapping]:
        return self._entity_mappings.get(entity)

    def list_entities(self) -> List[str]:
        return list(self._entity_mappings.keys())

    def apply(
        self,
        entity: str,
        system_id: str,
        source_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply mapping rules to transform source records into target schema.

        Parameters
        ----------
        entity:
            Entity key (e.g. ``"customer"``).
        system_id:
            Source system identifier (e.g. ``"legacy_oracle"``).
        source_records:
            List of source row dicts.

        Returns
        -------
        list[dict]
            Transformed rows in the target schema.
        """
        em = self._entity_mappings.get(entity)
        if not em:
            raise KeyError(f"No mapping loaded for entity '{entity}'")

        source_mapping: Optional[SourceMapping] = None
        for sm in em.sources:
            if sm.system_id == system_id:
                source_mapping = sm
                break

        if not source_mapping:
            raise KeyError(
                f"No mapping for system '{system_id}' in entity '{entity}'"
            )

        results: List[Dict[str, Any]] = []
        errors = 0
        for row in source_records:
            try:
                target_row = self._apply_mappings(row, source_mapping.column_mappings)
                results.append(target_row)
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    logger.warning("Transform error on row: %s", exc)

        logger.info(
            "Mapped %d records for %s/%s -> %s (%d errors)",
            len(results), system_id, entity, em.target_table, errors,
        )
        return results

    def _apply_mappings(
        self,
        row: Dict[str, Any],
        mappings: List[ColumnMapping],
    ) -> Dict[str, Any]:
        target_row: Dict[str, Any] = {}
        for cm in mappings:
            value = self._transform_engine.execute(
                transform_name=cm.transform,
                row=row,
                source_column=cm.source_column,
                source_columns=cm.source_columns,
                params=cm.params,
            )
            target_row[cm.target_column] = value
        return target_row
