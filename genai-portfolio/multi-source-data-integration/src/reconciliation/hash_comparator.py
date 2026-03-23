"""Hash-based row-level data comparison.

Computes a hash over selected columns for each row in both source and target,
then compares hashes to identify rows that were modified, lost, or introduced
during migration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class RowDiscrepancy:
    """A single row that differs between source and target."""

    key: str
    discrepancy_type: str  # "missing_in_target" | "missing_in_source" | "value_mismatch"
    source_hash: Optional[str] = None
    target_hash: Optional[str] = None
    source_values: Optional[Dict[str, Any]] = None
    target_values: Optional[Dict[str, Any]] = None
    differing_columns: List[str] = field(default_factory=list)


@dataclass
class HashComparisonResult:
    """Result of a hash-based comparison for one entity."""

    entity: str
    target_table: str
    hash_algorithm: str
    total_source_rows: int = 0
    total_target_rows: int = 0
    matching_rows: int = 0
    missing_in_target: int = 0
    missing_in_source: int = 0
    value_mismatches: int = 0
    passed: bool = True
    discrepancies: List[RowDiscrepancy] = field(default_factory=list)
    max_discrepancies: int = 1000

    @property
    def match_pct(self) -> float:
        total = max(self.total_source_rows, 1)
        return round(self.matching_rows / total * 100, 2)


class HashComparator:
    """Compare source and target data at the row level using hash fingerprints.

    Parameters
    ----------
    hash_algorithm:
        Hash function name (``"sha256"``, ``"md5"``, etc.).
    max_discrepancies:
        Maximum discrepancies to report before truncating.
    """

    def __init__(
        self,
        hash_algorithm: str = "sha256",
        max_discrepancies: int = 1000,
    ) -> None:
        self._algorithm = hash_algorithm
        self._max_disc = max_discrepancies

    def _hash_row(
        self, row: Dict[str, Any], columns: List[str]
    ) -> str:
        """Compute a deterministic hash of selected columns."""
        parts = []
        for col in sorted(columns):
            val = row.get(col)
            parts.append(f"{col}={val}" if val is not None else f"{col}=__NULL__")
        payload = "|".join(parts).encode("utf-8")
        return hashlib.new(self._algorithm, payload).hexdigest()

    def _build_key(
        self, row: Dict[str, Any], key_columns: List[str]
    ) -> str:
        """Build a composite key string from key columns."""
        return "::".join(str(row.get(k, "")) for k in key_columns)

    def compare(
        self,
        entity: str,
        target_table: str,
        source_records: List[Dict[str, Any]],
        target_records: List[Dict[str, Any]],
        key_columns: List[str],
        compare_columns: List[str],
    ) -> HashComparisonResult:
        """Compare source and target records using hash fingerprints.

        Parameters
        ----------
        key_columns:
            Columns that uniquely identify a row (e.g. ``["SOURCE_SYSTEM", "SOURCE_ID"]``).
        compare_columns:
            Columns to include in the hash comparison.
        """
        result = HashComparisonResult(
            entity=entity,
            target_table=target_table,
            hash_algorithm=self._algorithm,
            total_source_rows=len(source_records),
            total_target_rows=len(target_records),
            max_discrepancies=self._max_disc,
        )

        # Build source hash index
        source_index: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for row in source_records:
            key = self._build_key(row, key_columns)
            h = self._hash_row(row, compare_columns)
            source_index[key] = (h, row)

        # Build target hash index
        target_index: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for row in target_records:
            key = self._build_key(row, key_columns)
            h = self._hash_row(row, compare_columns)
            target_index[key] = (h, row)

        disc_count = 0

        # Check source against target
        for key, (src_hash, src_row) in source_index.items():
            if key not in target_index:
                result.missing_in_target += 1
                if disc_count < self._max_disc:
                    result.discrepancies.append(RowDiscrepancy(
                        key=key,
                        discrepancy_type="missing_in_target",
                        source_hash=src_hash,
                        source_values={c: src_row.get(c) for c in compare_columns},
                    ))
                    disc_count += 1
            else:
                tgt_hash, tgt_row = target_index[key]
                if src_hash == tgt_hash:
                    result.matching_rows += 1
                else:
                    result.value_mismatches += 1
                    if disc_count < self._max_disc:
                        differing = [
                            col for col in compare_columns
                            if src_row.get(col) != tgt_row.get(col)
                        ]
                        result.discrepancies.append(RowDiscrepancy(
                            key=key,
                            discrepancy_type="value_mismatch",
                            source_hash=src_hash,
                            target_hash=tgt_hash,
                            source_values={c: src_row.get(c) for c in differing},
                            target_values={c: tgt_row.get(c) for c in differing},
                            differing_columns=differing,
                        ))
                        disc_count += 1

        # Check target keys not in source
        for key in target_index:
            if key not in source_index:
                result.missing_in_source += 1
                if disc_count < self._max_disc:
                    tgt_hash, tgt_row = target_index[key]
                    result.discrepancies.append(RowDiscrepancy(
                        key=key,
                        discrepancy_type="missing_in_source",
                        target_hash=tgt_hash,
                        target_values={c: tgt_row.get(c) for c in compare_columns},
                    ))
                    disc_count += 1

        result.passed = (
            result.missing_in_target == 0
            and result.missing_in_source == 0
            and result.value_mismatches == 0
        )

        status = "PASSED" if result.passed else "FAILED"
        logger.info(
            "Hash comparison for %s: %s — %d matching, %d missing_tgt, "
            "%d missing_src, %d mismatches (%.1f%% match rate)",
            entity, status, result.matching_rows, result.missing_in_target,
            result.missing_in_source, result.value_mismatches, result.match_pct,
        )
        return result
