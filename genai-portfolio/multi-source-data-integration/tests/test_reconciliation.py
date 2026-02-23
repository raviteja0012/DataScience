"""Tests for the reconciliation and validation modules."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconciliation.count_reconciler import CountReconciler
from src.reconciliation.value_reconciler import ValueReconciler
from src.reconciliation.hash_comparator import HashComparator


class TestCountReconciler:

    def setup_method(self) -> None:
        self.reconciler = CountReconciler(tolerance_pct=0.0)

    def test_exact_match(self) -> None:
        result = self.reconciler.reconcile(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_counts={"oracle": 500, "sqlserver": 300},
            target_count=800,
            dedup_count=0,
        )
        assert result.passed
        total_check = result.checks[0]
        assert total_check.delta == 0

    def test_with_dedup(self) -> None:
        result = self.reconciler.reconcile(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_counts={"oracle": 500, "sqlserver": 300},
            target_count=750,
            dedup_count=50,
        )
        assert result.passed
        total_check = result.checks[0]
        assert total_check.expected_count == 750
        assert total_check.delta == 0

    def test_count_mismatch_fails(self) -> None:
        result = self.reconciler.reconcile(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_counts={"oracle": 500, "sqlserver": 300},
            target_count=700,
            dedup_count=0,
        )
        assert not result.passed

    def test_tolerance_allows_small_delta(self) -> None:
        reconciler = CountReconciler(tolerance_pct=1.0)
        result = reconciler.reconcile(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_counts={"oracle": 1000},
            target_count=995,
            dedup_count=0,
        )
        assert result.passed


class TestValueReconciler:

    def setup_method(self) -> None:
        self.reconciler = ValueReconciler(tolerance_pct=0.01)

    def test_sum_check_passes(self) -> None:
        source_records = {
            "oracle": [{"AMOUNT": 100.0}, {"AMOUNT": 200.0}],
            "sqlserver": [{"AMOUNT": 150.0}],
        }
        target_records = [
            {"TOTAL": 100.0}, {"TOTAL": 200.0}, {"TOTAL": 150.0},
        ]
        check_configs = [{
            "name": "total_amount",
            "check_type": "sum",
            "source_columns": {"oracle": "AMOUNT", "sqlserver": "AMOUNT"},
            "target_column": "TOTAL",
            "tolerance_pct": 0.01,
        }]
        result = self.reconciler.reconcile_from_records(
            entity="order",
            target_table="FACT_ORDER",
            source_records=source_records,
            target_records=target_records,
            check_configs=check_configs,
        )
        assert result.passed

    def test_sum_check_fails(self) -> None:
        source_records = {"oracle": [{"AMT": 1000.0}]}
        target_records = [{"TOTAL": 500.0}]
        check_configs = [{
            "name": "total_amount",
            "check_type": "sum",
            "source_columns": {"oracle": "AMT"},
            "target_column": "TOTAL",
            "tolerance_pct": 0.01,
        }]
        result = self.reconciler.reconcile_from_records(
            entity="order",
            target_table="FACT_ORDER",
            source_records=source_records,
            target_records=target_records,
            check_configs=check_configs,
        )
        assert not result.passed

    def test_distribution_check(self) -> None:
        target_records = [
            {"TYPE": "A"}, {"TYPE": "A"}, {"TYPE": "B"}, {"TYPE": "B"}, {"TYPE": "C"},
        ]
        check_configs = [{
            "name": "type_distribution",
            "check_type": "distribution",
            "target_column": "TYPE",
        }]
        result = self.reconciler.reconcile_from_records(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_records={},
            target_records=target_records,
            check_configs=check_configs,
        )
        assert result.passed
        dist_check = result.checks[0]
        assert dist_check.actual_value == {"A": 2, "B": 2, "C": 1}

    def test_min_max_check(self) -> None:
        target_records = [{"PRICE": 10.0}, {"PRICE": 500.0}, {"PRICE": 250.0}]
        check_configs = [{
            "name": "price_range",
            "check_type": "min_max",
            "target_column": "PRICE",
        }]
        result = self.reconciler.reconcile_from_records(
            entity="product",
            target_table="DIM_PRODUCT",
            source_records={},
            target_records=target_records,
            check_configs=check_configs,
        )
        assert result.passed
        assert result.checks[0].actual_value["min"] == 10.0
        assert result.checks[0].actual_value["max"] == 500.0


class TestHashComparator:

    def setup_method(self) -> None:
        self.comparator = HashComparator(hash_algorithm="sha256")

    def test_identical_data(self) -> None:
        records = [
            {"ID": "1", "NAME": "Alice", "EMAIL": "alice@test.com"},
            {"ID": "2", "NAME": "Bob", "EMAIL": "bob@test.com"},
        ]
        result = self.comparator.compare(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_records=records,
            target_records=records,
            key_columns=["ID"],
            compare_columns=["NAME", "EMAIL"],
        )
        assert result.passed
        assert result.matching_rows == 2
        assert result.missing_in_target == 0
        assert result.value_mismatches == 0

    def test_missing_in_target(self) -> None:
        source = [
            {"ID": "1", "NAME": "Alice"},
            {"ID": "2", "NAME": "Bob"},
        ]
        target = [
            {"ID": "1", "NAME": "Alice"},
        ]
        result = self.comparator.compare(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_records=source,
            target_records=target,
            key_columns=["ID"],
            compare_columns=["NAME"],
        )
        assert not result.passed
        assert result.missing_in_target == 1
        assert result.matching_rows == 1

    def test_value_mismatch(self) -> None:
        source = [{"ID": "1", "NAME": "Alice", "EMAIL": "alice@test.com"}]
        target = [{"ID": "1", "NAME": "Alice", "EMAIL": "alice@changed.com"}]
        result = self.comparator.compare(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_records=source,
            target_records=target,
            key_columns=["ID"],
            compare_columns=["NAME", "EMAIL"],
        )
        assert not result.passed
        assert result.value_mismatches == 1
        assert result.discrepancies[0].differing_columns == ["EMAIL"]

    def test_missing_in_source(self) -> None:
        source = [{"ID": "1", "NAME": "Alice"}]
        target = [
            {"ID": "1", "NAME": "Alice"},
            {"ID": "2", "NAME": "Extra"},
        ]
        result = self.comparator.compare(
            entity="customer",
            target_table="DIM_CUSTOMER",
            source_records=source,
            target_records=target,
            key_columns=["ID"],
            compare_columns=["NAME"],
        )
        assert not result.passed
        assert result.missing_in_source == 1
