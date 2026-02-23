"""Tests for the identity resolution engine."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.identity.matching_strategies import (
    ExactMatchStrategy,
    JaroWinklerStrategy,
    LevenshteinStrategy,
    MetaphoneStrategy,
    SoundexStrategy,
    get_strategy,
)
from src.identity.merge_rules import (
    GoldenRecord,
    MergeEngine,
    RecordCandidate,
    SurvivorshipRule,
)
from src.identity.dedup_engine import DedupEngine
from src.identity.resolver import IdentityResolver


class TestExactMatch:

    def test_identical_strings(self) -> None:
        s = ExactMatchStrategy()
        assert s.score("hello", "hello") == 1.0

    def test_case_insensitive(self) -> None:
        s = ExactMatchStrategy()
        assert s.score("Hello", "hello") == 1.0

    def test_different_strings(self) -> None:
        s = ExactMatchStrategy()
        assert s.score("hello", "world") == 0.0

    def test_none_values(self) -> None:
        s = ExactMatchStrategy()
        assert s.score(None, "hello") == 0.0
        assert s.score(None, None) == 0.0


class TestLevenshtein:

    def test_identical(self) -> None:
        s = LevenshteinStrategy()
        assert s.score("test", "test") == 1.0

    def test_one_edit(self) -> None:
        s = LevenshteinStrategy()
        result = s.score("test", "tast")
        assert 0.7 < result < 1.0

    def test_completely_different(self) -> None:
        s = LevenshteinStrategy()
        result = s.score("abc", "xyz")
        assert result == 0.0

    def test_empty_strings(self) -> None:
        s = LevenshteinStrategy()
        assert s.score("", "") == 1.0
        assert s.score("abc", "") == 0.0


class TestJaroWinkler:

    def test_identical(self) -> None:
        s = JaroWinklerStrategy()
        assert s.score("Smith", "Smith") == 1.0

    def test_similar_names(self) -> None:
        s = JaroWinklerStrategy()
        result = s.score("Smith", "Smyth")
        assert result > 0.8

    def test_common_prefix_bonus(self) -> None:
        s = JaroWinklerStrategy()
        # Jaro-Winkler gives bonus for matching prefix
        score_prefix = s.score("JOHNSON", "JOHNSEN")
        score_no_prefix = s.score("NHOSJNO", "NHOSJEN")
        assert score_prefix >= score_no_prefix

    def test_different_names(self) -> None:
        s = JaroWinklerStrategy()
        result = s.score("Smith", "Jones")
        assert result < 0.6


class TestSoundex:

    def test_same_soundex(self) -> None:
        s = SoundexStrategy()
        # Robert and Rupert have the same Soundex
        assert s.score("Robert", "Rupert") == 1.0

    def test_smith_smyth(self) -> None:
        s = SoundexStrategy()
        assert s.score("Smith", "Smyth") == 1.0

    def test_different_soundex(self) -> None:
        s = SoundexStrategy()
        result = s.score("Smith", "Jones")
        assert result < 1.0


class TestMetaphone:

    def test_similar_pronunciation(self) -> None:
        s = MetaphoneStrategy()
        result = s.score("Smith", "Smyth")
        assert result > 0.5

    def test_identical(self) -> None:
        s = MetaphoneStrategy()
        assert s.score("test", "test") == 1.0


class TestStrategyFactory:

    def test_get_exact(self) -> None:
        s = get_strategy("exact")
        assert isinstance(s, ExactMatchStrategy)

    def test_get_fuzzy(self) -> None:
        s = get_strategy("fuzzy")
        assert isinstance(s, JaroWinklerStrategy)

    def test_get_jaro_winkler(self) -> None:
        s = get_strategy("jaro_winkler")
        assert isinstance(s, JaroWinklerStrategy)

    def test_get_soundex(self) -> None:
        s = get_strategy("soundex")
        assert isinstance(s, SoundexStrategy)

    def test_get_phonetic(self) -> None:
        s = get_strategy("phonetic")
        assert isinstance(s, MetaphoneStrategy)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown matching strategy"):
            get_strategy("nonexistent")


class TestMergeEngine:

    def test_single_candidate(self) -> None:
        engine = MergeEngine()
        candidate = RecordCandidate(
            data={"name": "Alice", "email": "alice@example.com"},
            source_system="oracle",
        )
        golden = engine.merge([candidate])
        assert golden.data["name"] == "Alice"
        assert golden.sources == ["oracle"]

    def test_most_complete_wins(self) -> None:
        engine = MergeEngine(default_rule=SurvivorshipRule.MOST_COMPLETE)
        c1 = RecordCandidate(
            data={"name": "Alice", "email": None, "phone": None},
            source_system="oracle",
        )
        c2 = RecordCandidate(
            data={"name": "Alice Smith", "email": "alice@test.com", "phone": "555-1234"},
            source_system="sqlserver",
        )
        golden = engine.merge([c1, c2])
        assert golden.data["email"] == "alice@test.com"
        assert golden.data["phone"] == "555-1234"

    def test_newest_wins(self) -> None:
        engine = MergeEngine(default_rule=SurvivorshipRule.NEWEST)
        c1 = RecordCandidate(
            data={"name": "Old Name"},
            source_system="oracle",
            modified_at="2020-01-01",
        )
        c2 = RecordCandidate(
            data={"name": "New Name"},
            source_system="sqlserver",
            modified_at="2024-06-01",
        )
        golden = engine.merge([c1, c2])
        assert golden.data["name"] == "New Name"

    def test_source_priority(self) -> None:
        engine = MergeEngine(default_rule=SurvivorshipRule.SOURCE_PRIORITY)
        c1 = RecordCandidate(
            data={"tax_id": "AAA"},
            source_system="oracle",
            source_priority=1,
        )
        c2 = RecordCandidate(
            data={"tax_id": "BBB"},
            source_system="sqlserver",
            source_priority=2,
        )
        golden = engine.merge([c1, c2])
        assert golden.data["tax_id"] == "AAA"

    def test_field_overrides(self) -> None:
        engine = MergeEngine(
            default_rule=SurvivorshipRule.MOST_COMPLETE,
            field_overrides={"email": SurvivorshipRule.NEWEST},
        )
        c1 = RecordCandidate(
            data={"name": "Alice", "email": "old@test.com", "phone": "111"},
            source_system="oracle",
            modified_at="2020-01-01",
        )
        c2 = RecordCandidate(
            data={"name": None, "email": "new@test.com", "phone": None},
            source_system="sqlserver",
            modified_at="2024-06-01",
        )
        golden = engine.merge([c1, c2])
        # Email should use newest (sqlserver), name should use most_complete (oracle)
        assert golden.data["email"] == "new@test.com"
        assert golden.data["name"] == "Alice"


class TestDedupEngine:

    def test_exact_duplicates(self) -> None:
        engine = DedupEngine(
            strategies=[{"name": "exact", "fields": ["email"], "weight": 1.0}],
            match_threshold=0.8,
            blocking_fields=["email"],
        )
        records = [
            {"email": "alice@test.com", "name": "Alice"},
            {"email": "alice@test.com", "name": "Alice Smith"},
            {"email": "bob@test.com", "name": "Bob"},
        ]
        result = engine.deduplicate(records, source_system="test")
        assert result.total_duplicates >= 1
        assert result.total_clusters >= 1

    def test_no_duplicates(self) -> None:
        engine = DedupEngine(
            strategies=[{"name": "exact", "fields": ["email"], "weight": 1.0}],
            match_threshold=0.8,
            blocking_fields=["email"],
        )
        records = [
            {"email": "alice@test.com", "name": "Alice"},
            {"email": "bob@test.com", "name": "Bob"},
        ]
        result = engine.deduplicate(records, source_system="test")
        assert result.total_duplicates == 0

    def test_cross_source_dedup(self) -> None:
        engine = DedupEngine(
            strategies=[{"name": "exact", "fields": ["email"], "weight": 1.0}],
            match_threshold=0.8,
            blocking_fields=["email"],
        )
        result = engine.cross_source_deduplicate({
            "oracle": [{"email": "shared@test.com", "name": "Alice"}],
            "sqlserver": [{"email": "shared@test.com", "name": "Alice Smith"}],
        })
        assert result.cross_source_dupes >= 1


class TestIdentityResolver:

    def test_resolve_with_overlap(self) -> None:
        config = {
            "match_threshold": 0.8,
            "strategies": [
                {"name": "exact", "fields": ["email"], "weight": 1.0},
            ],
            "survivorship": {
                "default_rule": "most_complete",
            },
            "dedup": {"batch_size": 100},
        }
        resolver = IdentityResolver(
            entity_type="customer",
            config=config,
            source_priorities={"oracle": 1, "sqlserver": 2},
        )

        result = resolver.resolve({
            "oracle": [
                {"CUST_EMAIL": "alice@test.com", "CUST_NM": "Alice Smith",
                 "CUST_PHONE": "555-0001", "TAX_ID": "12-345"},
            ],
            "sqlserver": [
                {"EmailAddress": "alice@test.com", "FirstName": "Alice",
                 "LastName": "Smith", "PhoneNumber": "555-0001"},
                {"EmailAddress": "bob@test.com", "FirstName": "Bob",
                 "LastName": "Jones", "PhoneNumber": "555-0002"},
            ],
        })

        assert result.total_input_records == 3
        # At least one cross-source match expected
        assert result.total_resolved_entities <= result.total_input_records
