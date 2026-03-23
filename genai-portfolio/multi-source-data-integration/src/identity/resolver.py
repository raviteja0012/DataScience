"""Master identity resolution engine.

Orchestrates the full identity-resolution workflow:

1. Normalise records from all source systems into a common comparison schema.
2. Run the deduplication engine (within-source and cross-source).
3. Apply merge rules to produce golden records for each cluster.
4. Assign business keys and confidence scores.

The resolver is the primary entry-point for identity resolution; downstream
modules (``matching_strategies``, ``merge_rules``, ``dedup_engine``) are
composed together here.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.identity.dedup_engine import DedupEngine, DedupResult, DuplicateCluster
from src.identity.merge_rules import (
    GoldenRecord,
    MergeEngine,
    RecordCandidate,
    SurvivorshipRule,
)
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector

logger = get_logger(__name__, fmt="text")


@dataclass
class ResolvedEntity:
    """A single resolved entity with its golden record and lineage."""

    business_key: str
    golden_record: Dict[str, Any]
    source_records: List[Dict[str, Any]] = field(default_factory=list)
    source_systems: List[str] = field(default_factory=list)
    match_confidence: float = 1.0
    field_lineage: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResolutionResult:
    """Complete result of an identity resolution run."""

    entity_type: str
    total_input_records: int = 0
    total_resolved_entities: int = 0
    total_duplicates_found: int = 0
    cross_source_matches: int = 0
    within_source_matches: int = 0
    resolved_entities: List[ResolvedEntity] = field(default_factory=list)
    dedup_result: Optional[DedupResult] = None


class IdentityResolver:
    """End-to-end identity resolution for a single entity type.

    Parameters
    ----------
    entity_type:
        e.g. ``"customer"``, ``"product"``.
    config:
        The ``identity_resolution`` section of the pipeline config.
    source_priorities:
        ``{system_id: priority_int}`` — lower is higher priority.
    metrics:
        Optional metrics collector.
    """

    def __init__(
        self,
        entity_type: str,
        config: Dict[str, Any],
        source_priorities: Optional[Dict[str, int]] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self._entity_type = entity_type
        self._config = config
        self._source_priorities = source_priorities or {}
        self._metrics = metrics

        self._dedup = DedupEngine.from_config(config)
        self._merge = MergeEngine.from_config(config.get("survivorship", {}))

    # -- Public API ----------------------------------------------------------

    def resolve(
        self,
        records_by_source: Dict[str, List[Dict[str, Any]]],
    ) -> ResolutionResult:
        """Run full identity resolution across all sources.

        Parameters
        ----------
        records_by_source:
            ``{system_id: [row_dict, ...]}`` for each source system.

        Returns
        -------
        ResolutionResult
        """
        total_input = sum(len(recs) for recs in records_by_source.values())
        logger.info(
            "Starting identity resolution for '%s': %d total records from %d sources",
            self._entity_type, total_input, len(records_by_source),
        )

        if self._metrics:
            self._metrics.start_timer(f"identity_resolution_{self._entity_type}")

        # 1. Normalise all records
        normalised = self._normalise_all(records_by_source)

        # 2. Cross-source deduplication
        dedup_result = self._dedup.cross_source_deduplicate(normalised)

        # 3. Build resolved entities
        resolved: List[ResolvedEntity] = []

        # Process clustered duplicates
        for cluster in dedup_result.clusters:
            golden = self._merge_cluster(cluster)
            resolved.append(golden)

        # Process singletons (non-duplicate records)
        clustered_indices = set()
        all_records: List[Tuple[Dict[str, Any], str]] = []
        for sys_id, recs in normalised.items():
            for r in recs:
                all_records.append((r, sys_id))

        for cluster in dedup_result.clusters:
            for rec in cluster.records:
                for idx, (r, _) in enumerate(all_records):
                    if r is rec:
                        clustered_indices.add(idx)
                        break

        for idx, (rec, sys_id) in enumerate(all_records):
            if idx not in clustered_indices:
                bk = self._generate_business_key(rec)
                resolved.append(ResolvedEntity(
                    business_key=bk,
                    golden_record=dict(rec),
                    source_records=[rec],
                    source_systems=[sys_id],
                    match_confidence=1.0,
                ))

        result = ResolutionResult(
            entity_type=self._entity_type,
            total_input_records=total_input,
            total_resolved_entities=len(resolved),
            total_duplicates_found=dedup_result.total_duplicates,
            cross_source_matches=dedup_result.cross_source_dupes,
            within_source_matches=dedup_result.within_source_dupes,
            resolved_entities=resolved,
            dedup_result=dedup_result,
        )

        if self._metrics:
            elapsed = self._metrics.stop_timer(
                f"identity_resolution_{self._entity_type}",
                phase="identity_resolution",
                entity=self._entity_type,
            )
            self._metrics.record(
                f"identity_{self._entity_type}_input_rows",
                total_input,
                phase="identity_resolution",
                entity=self._entity_type,
            )
            self._metrics.record(
                f"identity_{self._entity_type}_resolved_entities",
                len(resolved),
                phase="identity_resolution",
                entity=self._entity_type,
            )
            self._metrics.record(
                f"identity_{self._entity_type}_duplicates",
                dedup_result.total_duplicates,
                phase="identity_resolution",
                entity=self._entity_type,
            )

        logger.info(
            "Identity resolution for '%s' complete: %d input -> %d resolved "
            "(%d duplicates: %d cross-source, %d within-source)",
            self._entity_type,
            total_input,
            len(resolved),
            dedup_result.total_duplicates,
            dedup_result.cross_source_dupes,
            dedup_result.within_source_dupes,
        )
        return result

    # -- Internal helpers ----------------------------------------------------

    def _normalise_all(
        self, records_by_source: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalise field names to a common comparison schema."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for sys_id, recs in records_by_source.items():
            normalised = [self._normalise_record(r, sys_id) for r in recs]
            result[sys_id] = normalised
        return result

    def _normalise_record(
        self, record: Dict[str, Any], system_id: str
    ) -> Dict[str, Any]:
        """Map system-specific field names to canonical names for comparison.

        This creates a comparison-friendly record while preserving the
        originals under ``_source_*`` keys.
        """
        normalised: Dict[str, Any] = {"_source_system": system_id}

        # Copy all original fields with prefix
        for k, v in record.items():
            normalised[f"_src_{k}"] = v

        # Canonical field extraction by convention
        name_fields = ["CUST_NM", "customer_name", "CustomerName"]
        for f in name_fields:
            if f in record:
                normalised["customer_name"] = str(record[f]).strip().lower() if record[f] else None
                break

        # Attempt to construct customer_name from first + last
        if "customer_name" not in normalised:
            first = record.get("FirstName", record.get("first_name", ""))
            last = record.get("LastName", record.get("last_name", ""))
            if first or last:
                name = f"{first or ''} {last or ''}".strip().lower()
                normalised["customer_name"] = name if name else None

        # Email
        for f in ["CUST_EMAIL", "EmailAddress", "email"]:
            if f in record and record[f]:
                normalised["email"] = str(record[f]).strip().lower()
                break

        # Phone
        for f in ["CUST_PHONE", "PhoneNumber", "phone"]:
            if f in record and record[f]:
                import re
                normalised["phone"] = re.sub(r"\D", "", str(record[f]))
                break

        # Tax ID
        for f in ["TAX_ID", "TaxIdentifier", "tax_id"]:
            if f in record and record[f]:
                normalised["tax_id"] = str(record[f]).strip()
                break

        # Company name
        for f in ["CompanyName", "company_name"]:
            if f in record and record[f]:
                normalised["company_name"] = str(record[f]).strip().lower()
                break

        # Address / City / State / Postal
        for f in ["CUST_CITY", "City", "city"]:
            if f in record and record[f]:
                normalised["city"] = str(record[f]).strip().lower()
                break

        for f in ["CUST_ST", "StateProvince", "state"]:
            if f in record and record[f]:
                normalised["state"] = str(record[f]).strip().lower()
                break

        for f in ["CUST_ZIP", "PostalCode", "postal_code"]:
            if f in record and record[f]:
                normalised["postal_code"] = str(record[f]).strip()
                break

        # Modified date for survivorship
        for f in ["MODIFIED_DT", "ModifiedDate", "modified_at"]:
            if f in record and record[f]:
                normalised["_modified_at"] = str(record[f])
                break

        return normalised

    def _merge_cluster(self, cluster: DuplicateCluster) -> ResolvedEntity:
        """Merge a duplicate cluster into a single resolved entity."""
        candidates: List[RecordCandidate] = []
        for rec, sys_id in zip(cluster.records, cluster.source_systems):
            candidates.append(RecordCandidate(
                data=rec,
                source_system=sys_id,
                source_priority=self._source_priorities.get(sys_id, 99),
                modified_at=rec.get("_modified_at"),
            ))

        golden: GoldenRecord = self._merge.merge(candidates)
        bk = self._generate_business_key(golden.data)
        avg_score = cluster.avg_score if cluster.avg_score > 0 else (
            sum(cluster.match_scores) / max(len(cluster.match_scores), 1)
        )

        return ResolvedEntity(
            business_key=bk,
            golden_record=golden.data,
            source_records=cluster.records,
            source_systems=cluster.source_systems,
            match_confidence=round(avg_score, 4),
            field_lineage=golden.field_lineage,
        )

    @staticmethod
    def _generate_business_key(record: Dict[str, Any]) -> str:
        """Deterministic business key from core identifying fields."""
        components = [
            str(record.get("email", "")),
            str(record.get("tax_id", "")),
            str(record.get("customer_name", "")),
            str(record.get("postal_code", "")),
        ]
        key_input = "|".join(c.lower().strip() for c in components if c)
        if not key_input:
            return f"BK-{uuid.uuid4().hex[:12]}"
        return "BK-" + hashlib.sha256(key_input.encode()).hexdigest()[:12]
