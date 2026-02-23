"""Survivorship and golden record merge rules.

When identity resolution identifies two or more records as representing the
same real-world entity, the merge engine must decide which field values to
keep.  This module implements several strategies:

- **newest**: Keep the value from the most recently modified record.
- **most_complete**: Keep the value from the record with the fewest NULLs.
- **source_priority**: Keep the value from the highest-priority source system.
- **longest**: Keep the longest non-null string value.
- **custom**: Evaluate a per-field callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class SurvivorshipRule(Enum):
    NEWEST = "newest"
    MOST_COMPLETE = "most_complete"
    SOURCE_PRIORITY = "source_priority"
    LONGEST = "longest"


@dataclass
class RecordCandidate:
    """A single record participating in a merge, with metadata for
    survivorship evaluation."""

    data: Dict[str, Any]
    source_system: str
    source_priority: int = 99
    modified_at: Optional[str] = None
    completeness: float = 0.0  # 0..1

    def __post_init__(self) -> None:
        if self.completeness == 0.0:
            non_null = sum(1 for v in self.data.values() if v is not None)
            total = max(len(self.data), 1)
            self.completeness = round(non_null / total, 4)


@dataclass
class GoldenRecord:
    """The merged output record with lineage information."""

    data: Dict[str, Any]
    sources: List[str] = field(default_factory=list)
    field_lineage: Dict[str, str] = field(default_factory=dict)
    merge_strategy_used: str = ""


class MergeEngine:
    """Apply survivorship rules to produce a golden record from candidate records."""

    def __init__(
        self,
        default_rule: SurvivorshipRule = SurvivorshipRule.MOST_COMPLETE,
        field_overrides: Optional[Dict[str, SurvivorshipRule]] = None,
    ) -> None:
        self._default_rule = default_rule
        self._field_overrides = field_overrides or {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MergeEngine":
        """Build from the ``survivorship`` section of the pipeline config."""
        default = SurvivorshipRule(config.get("default_rule", "most_complete"))
        overrides = {
            field_name: SurvivorshipRule(rule)
            for field_name, rule in config.get("field_overrides", {}).items()
        }
        return cls(default_rule=default, field_overrides=overrides)

    def merge(self, candidates: List[RecordCandidate]) -> GoldenRecord:
        """Merge a cluster of candidate records into one golden record.

        Parameters
        ----------
        candidates:
            Two or more records identified as the same entity.

        Returns
        -------
        GoldenRecord
            The merged record with field-level lineage.
        """
        if not candidates:
            return GoldenRecord(data={})
        if len(candidates) == 1:
            return GoldenRecord(
                data=dict(candidates[0].data),
                sources=[candidates[0].source_system],
                merge_strategy_used="single_source",
            )

        # Collect all field names across candidates
        all_fields: set[str] = set()
        for c in candidates:
            all_fields.update(c.data.keys())

        merged: Dict[str, Any] = {}
        lineage: Dict[str, str] = {}

        for fld in sorted(all_fields):
            rule = self._field_overrides.get(fld, self._default_rule)
            winner = self._pick_winner(fld, candidates, rule)
            merged[fld] = winner.data.get(fld)
            lineage[fld] = f"{winner.source_system} ({rule.value})"

        return GoldenRecord(
            data=merged,
            sources=[c.source_system for c in candidates],
            field_lineage=lineage,
            merge_strategy_used=self._default_rule.value,
        )

    def _pick_winner(
        self,
        field_name: str,
        candidates: List[RecordCandidate],
        rule: SurvivorshipRule,
    ) -> RecordCandidate:
        """Select the winning candidate for a given field."""
        # Filter to candidates that have a non-null value for this field
        with_value = [c for c in candidates if c.data.get(field_name) is not None]
        if not with_value:
            return candidates[0]

        if rule == SurvivorshipRule.NEWEST:
            return self._pick_newest(with_value)
        if rule == SurvivorshipRule.MOST_COMPLETE:
            return self._pick_most_complete(with_value)
        if rule == SurvivorshipRule.SOURCE_PRIORITY:
            return self._pick_source_priority(with_value)
        if rule == SurvivorshipRule.LONGEST:
            return self._pick_longest(field_name, with_value)

        return with_value[0]

    @staticmethod
    def _pick_newest(candidates: List[RecordCandidate]) -> RecordCandidate:
        def _sort_key(c: RecordCandidate) -> str:
            return c.modified_at or "0000-00-00"
        return max(candidates, key=_sort_key)

    @staticmethod
    def _pick_most_complete(candidates: List[RecordCandidate]) -> RecordCandidate:
        return max(candidates, key=lambda c: c.completeness)

    @staticmethod
    def _pick_source_priority(candidates: List[RecordCandidate]) -> RecordCandidate:
        return min(candidates, key=lambda c: c.source_priority)

    @staticmethod
    def _pick_longest(
        field_name: str, candidates: List[RecordCandidate]
    ) -> RecordCandidate:
        def _len(c: RecordCandidate) -> int:
            val = c.data.get(field_name)
            return len(str(val)) if val is not None else 0
        return max(candidates, key=_len)
