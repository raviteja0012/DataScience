#!/usr/bin/env python3
"""Identity resolution demonstration.

Shows the full identity resolution workflow: normalisation, blocking,
multi-strategy matching (exact, fuzzy, phonetic), survivorship merge,
and golden record creation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.identity.matching_strategies import (
    ExactMatchStrategy,
    JaroWinklerStrategy,
    LevenshteinStrategy,
    MetaphoneStrategy,
    SoundexStrategy,
)
from src.identity.merge_rules import MergeEngine, RecordCandidate, SurvivorshipRule
from src.identity.dedup_engine import DedupEngine
from src.identity.resolver import IdentityResolver
from src.utils.generators import SyntheticDataGenerator
from src.utils.metrics import MetricsCollector


def demo_matching_strategies() -> None:
    """Demonstrate individual matching strategies on sample pairs."""
    print("--- MATCHING STRATEGY DEMONSTRATIONS ---\n")

    pairs = [
        ("John Smith", "John Smith"),
        ("John Smith", "JOHN SMITH"),
        ("John Smith", "Jon Smith"),
        ("John Smith", "John Smyth"),
        ("John Smith", "Jonathan Smith"),
        ("Robert Johnson", "Bob Johnson"),
        ("John Smith", "Jane Doe"),
    ]

    strategies = [
        ("Exact", ExactMatchStrategy()),
        ("Levenshtein", LevenshteinStrategy()),
        ("Jaro-Winkler", JaroWinklerStrategy()),
        ("Soundex", SoundexStrategy()),
        ("Metaphone", MetaphoneStrategy()),
    ]

    # Header
    header = f"{'Pair':<40s}"
    for name, _ in strategies:
        header += f" {name:<13s}"
    print(header)
    print("-" * len(header))

    for a, b in pairs:
        label = f"{a} vs {b}"
        row = f"{label:<40s}"
        for _, strat in strategies:
            score = strat.score(a, b)
            row += f" {score:<13.4f}"
        print(row)


def demo_merge_rules() -> None:
    """Demonstrate survivorship merge rules."""
    print("\n\n--- SURVIVORSHIP MERGE RULES ---\n")

    # Two candidate records for the same person
    oracle_rec = RecordCandidate(
        data={
            "customer_name": "john smith",
            "email": "JSMITH@ACME.COM",
            "phone": "3035551234",
            "address": "123 Oak Street, Suite 400",
            "city": "Denver",
            "state": "CO",
            "tax_id": "12-3456789",
        },
        source_system="legacy_oracle",
        source_priority=1,
        modified_at="2023-01-15",
    )

    sqlserver_rec = RecordCandidate(
        data={
            "customer_name": "john d. smith",
            "email": "john.smith@newmail.com",
            "phone": None,
            "address": "123 Oak St",
            "city": "Denver",
            "state": "CO",
            "tax_id": None,
        },
        source_system="legacy_sqlserver",
        source_priority=2,
        modified_at="2024-06-01",
    )

    rules_to_demo = [
        ("most_complete", SurvivorshipRule.MOST_COMPLETE),
        ("newest", SurvivorshipRule.NEWEST),
        ("source_priority", SurvivorshipRule.SOURCE_PRIORITY),
    ]

    for rule_name, rule in rules_to_demo:
        engine = MergeEngine(default_rule=rule)
        golden = engine.merge([oracle_rec, sqlserver_rec])
        print(f"Rule: {rule_name}")
        for field, value in golden.data.items():
            lineage = golden.field_lineage.get(field, "")
            print(f"  {field:<20s} = {str(value):<35s}  <-- {lineage}")
        print()

    # With field-level overrides
    print("Rule: mixed (default=most_complete, email=newest, tax_id=source_priority)")
    engine = MergeEngine(
        default_rule=SurvivorshipRule.MOST_COMPLETE,
        field_overrides={
            "email": SurvivorshipRule.NEWEST,
            "tax_id": SurvivorshipRule.SOURCE_PRIORITY,
        },
    )
    golden = engine.merge([oracle_rec, sqlserver_rec])
    for field, value in golden.data.items():
        lineage = golden.field_lineage.get(field, "")
        print(f"  {field:<20s} = {str(value):<35s}  <-- {lineage}")


def demo_full_resolution() -> None:
    """Run full identity resolution on synthetic data."""
    print("\n\n--- FULL IDENTITY RESOLUTION ---\n")

    gen = SyntheticDataGenerator(
        num_customers=300,
        num_products=50,
        num_orders=200,
        cross_source_overlap_pct=0.15,
        seed=42,
    )
    data = gen.generate_all()

    oracle_custs = [c.to_dict() for c in data["oracle_customers"]]
    sql_custs = [c.to_dict() for c in data["sqlserver_customers"]]

    print(f"Oracle customers: {len(oracle_custs)}")
    print(f"SQL Server customers: {len(sql_custs)}")
    print(f"Known cross-source overlaps: {len(data['shared_identities'])}")

    # Load pipeline config for identity resolution settings
    config_dir = PROJECT_ROOT / "config"
    with open(config_dir / "pipeline_config.yaml") as f:
        pipeline_cfg = yaml.safe_load(f)

    ir_config = pipeline_cfg.get("identity_resolution", {})
    metrics = MetricsCollector()

    resolver = IdentityResolver(
        entity_type="customer",
        config=ir_config,
        source_priorities={"legacy_oracle": 1, "legacy_sqlserver": 2},
        metrics=metrics,
    )

    result = resolver.resolve({
        "legacy_oracle": oracle_custs,
        "legacy_sqlserver": sql_custs,
    })

    print(f"\nResolution Results:")
    print(f"  Total input records: {result.total_input_records}")
    print(f"  Resolved entities:   {result.total_resolved_entities}")
    print(f"  Duplicates found:    {result.total_duplicates_found}")
    print(f"  Cross-source matches: {result.cross_source_matches}")
    print(f"  Within-source matches: {result.within_source_matches}")

    # Show some example resolved entities with cross-source matches
    cross_source = [
        e for e in result.resolved_entities
        if len(set(e.source_systems)) > 1
    ]
    print(f"\n  Sample cross-source resolved entities ({len(cross_source)} total):")
    for entity in cross_source[:5]:
        print(f"\n  Business Key: {entity.business_key}")
        print(f"  Confidence: {entity.match_confidence:.4f}")
        print(f"  Sources: {entity.source_systems}")
        golden = entity.golden_record
        display_fields = ["customer_name", "email", "phone", "city", "state"]
        for field in display_fields:
            val = golden.get(field, "N/A")
            lineage = entity.field_lineage.get(field, "")
            if val:
                print(f"    {field:<20s} = {str(val):<35s}  [{lineage}]")

    # Metrics summary
    print(f"\nMetrics summary: {metrics.summary()}")


def main() -> None:
    print("=" * 70)
    print("IDENTITY RESOLUTION DEMONSTRATION")
    print("=" * 70)

    demo_matching_strategies()
    demo_merge_rules()
    demo_full_resolution()

    print("\n" + "=" * 70)
    print("IDENTITY RESOLUTION DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
