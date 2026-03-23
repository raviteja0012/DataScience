"""Deduplication engine.

Identifies and clusters duplicate records both within a single source system
and across multiple sources.  Uses configurable blocking keys to reduce the
comparison space, then applies matching strategies to score candidate pairs.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.identity.matching_strategies import MatchingStrategy, get_strategy
from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class DuplicateCluster:
    """A group of records that refer to the same real-world entity."""

    cluster_id: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    source_systems: List[str] = field(default_factory=list)
    match_scores: List[float] = field(default_factory=list)
    avg_score: float = 0.0

    def __post_init__(self) -> None:
        if self.match_scores:
            self.avg_score = round(
                sum(self.match_scores) / len(self.match_scores), 4
            )


@dataclass
class DedupResult:
    """Summary of a deduplication run."""

    total_input_records: int = 0
    total_clusters: int = 0
    total_duplicates: int = 0
    unique_records: int = 0
    within_source_dupes: int = 0
    cross_source_dupes: int = 0
    clusters: List[DuplicateCluster] = field(default_factory=list)


def _blocking_key(record: Dict[str, Any], fields: List[str]) -> str:
    """Create a blocking key from selected fields.

    Blocking reduces the O(n^2) comparison space by only comparing records
    that share the same blocking key.
    """
    parts: List[str] = []
    for f in fields:
        val = record.get(f)
        if val is None:
            parts.append("")
        else:
            # Normalise: first 3 chars lower, strip whitespace
            parts.append(str(val).strip().lower()[:3])
    return "|".join(parts)


class DedupEngine:
    """Configurable deduplication across and within source systems.

    Parameters
    ----------
    strategies:
        List of (strategy_name, fields, weight, min_similarity) tuples.
    match_threshold:
        Minimum weighted score to consider a pair a duplicate.
    blocking_fields:
        Fields used to build blocking keys (reduces comparisons).
    batch_size:
        Number of records per comparison batch.
    """

    def __init__(
        self,
        strategies: Optional[List[Dict[str, Any]]] = None,
        match_threshold: float = 0.82,
        blocking_fields: Optional[List[str]] = None,
        batch_size: int = 10000,
    ) -> None:
        self._match_threshold = match_threshold
        self._batch_size = batch_size
        self._blocking_fields = blocking_fields or ["customer_name", "email"]

        # Parse strategy configs
        self._strategies: List[Tuple[MatchingStrategy, List[str], float, float]] = []
        for scfg in (strategies or []):
            strat = get_strategy(scfg.get("algorithm", scfg.get("name", "exact")))
            fields = scfg.get("fields", [])
            weight = scfg.get("weight", 1.0)
            min_sim = scfg.get("min_similarity", 0.0)
            self._strategies.append((strat, fields, weight, min_sim))

        if not self._strategies:
            # Sensible defaults
            self._strategies = [
                (get_strategy("exact"), ["email"], 1.0, 1.0),
                (get_strategy("jaro_winkler"), ["customer_name"], 0.85, 0.88),
            ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DedupEngine":
        """Build from the ``identity_resolution`` section of the pipeline config."""
        return cls(
            strategies=config.get("strategies", []),
            match_threshold=config.get("match_threshold", 0.82),
            batch_size=config.get("dedup", {}).get("batch_size", 10000),
        )

    def _score_pair(
        self,
        rec_a: Dict[str, Any],
        rec_b: Dict[str, Any],
    ) -> float:
        """Compute the weighted match score for a pair of records."""
        total_weight = 0.0
        weighted_sum = 0.0

        for strategy, fields, weight, min_sim in self._strategies:
            field_scores: List[float] = []
            for fld in fields:
                val_a = rec_a.get(fld)
                val_b = rec_b.get(fld)
                if val_a is None or val_b is None:
                    continue
                s = strategy.score(val_a, val_b)
                if s < min_sim:
                    s = 0.0
                field_scores.append(s)

            if field_scores:
                avg = sum(field_scores) / len(field_scores)
                weighted_sum += avg * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    def deduplicate(
        self,
        records: List[Dict[str, Any]],
        source_system: str = "",
    ) -> DedupResult:
        """Find duplicates within a single list of records."""
        return self._find_duplicates(
            [(r, source_system) for r in records],
            within_source_only=True,
        )

    def cross_source_deduplicate(
        self,
        records_by_source: Dict[str, List[Dict[str, Any]]],
    ) -> DedupResult:
        """Find duplicates across multiple source systems."""
        all_records: List[Tuple[Dict[str, Any], str]] = []
        for sys_id, recs in records_by_source.items():
            for r in recs:
                all_records.append((r, sys_id))
        return self._find_duplicates(all_records, within_source_only=False)

    def _find_duplicates(
        self,
        records_with_source: List[Tuple[Dict[str, Any], str]],
        within_source_only: bool = False,
    ) -> DedupResult:
        """Core deduplication logic with blocking."""
        # Build blocking index
        blocks: Dict[str, List[int]] = defaultdict(list)
        for idx, (rec, _) in enumerate(records_with_source):
            key = _blocking_key(rec, self._blocking_fields)
            blocks[key].append(idx)

        # Union-Find for clustering
        parent: Dict[int, int] = {i: i for i in range(len(records_with_source))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        match_scores: Dict[Tuple[int, int], float] = {}

        # Compare within blocks
        comparisons = 0
        for block_ids in blocks.values():
            for i in range(len(block_ids)):
                for j in range(i + 1, len(block_ids)):
                    idx_a, idx_b = block_ids[i], block_ids[j]
                    rec_a, src_a = records_with_source[idx_a]
                    rec_b, src_b = records_with_source[idx_b]

                    if within_source_only and src_a != src_b:
                        continue

                    score = self._score_pair(rec_a, rec_b)
                    comparisons += 1

                    if score >= self._match_threshold:
                        union(idx_a, idx_b)
                        pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                        match_scores[pair_key] = score

        # Build clusters
        cluster_map: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(records_with_source)):
            root = find(idx)
            cluster_map[root].append(idx)

        clusters: List[DuplicateCluster] = []
        within_source_dupes = 0
        cross_source_dupes = 0

        for root, members in cluster_map.items():
            if len(members) < 2:
                continue
            recs = [records_with_source[m][0] for m in members]
            sources = [records_with_source[m][1] for m in members]
            scores = [
                match_scores.get((min(members[i], members[j]),
                                  max(members[i], members[j])), 0.0)
                for i in range(len(members))
                for j in range(i + 1, len(members))
                if (min(members[i], members[j]),
                    max(members[i], members[j])) in match_scores
            ]

            cluster_id = hashlib.md5(
                str(sorted(members)).encode()
            ).hexdigest()[:12]

            cluster = DuplicateCluster(
                cluster_id=cluster_id,
                records=recs,
                source_systems=sources,
                match_scores=scores,
            )
            clusters.append(cluster)

            unique_sources = set(sources)
            if len(unique_sources) > 1:
                cross_source_dupes += len(members) - 1
            else:
                within_source_dupes += len(members) - 1

        total_dupes = within_source_dupes + cross_source_dupes
        logger.info(
            "Dedup complete: %d records, %d clusters, %d dupes "
            "(within=%d, cross=%d), %d comparisons",
            len(records_with_source), len(clusters), total_dupes,
            within_source_dupes, cross_source_dupes, comparisons,
        )

        return DedupResult(
            total_input_records=len(records_with_source),
            total_clusters=len(clusters),
            total_duplicates=total_dupes,
            unique_records=len(records_with_source) - total_dupes,
            within_source_dupes=within_source_dupes,
            cross_source_dupes=cross_source_dupes,
            clusters=clusters,
        )
