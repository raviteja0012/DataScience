"""Tests for the cutover strategy, checkpoint, and rollback modules."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cutover.strategy import CutoverStrategyPlanner, CutoverType, LoadTask
from src.cutover.checkpoint import CheckpointManager
from src.cutover.rollback_manager import RollbackManager
from src.cutover.parallel_runner import ParallelRunner


class TestCutoverStrategy:

    def test_phased_plan(self) -> None:
        config = {
            "strategy": "phased",
            "max_parallel_loads": 4,
            "phases": [
                {"name": "reference", "entities": ["products", "categories"], "order": 1},
                {"name": "master", "entities": ["customers"], "order": 2, "depends_on": ["reference"]},
                {"name": "transactional", "entities": ["orders"], "order": 3, "depends_on": ["master"]},
            ],
        }
        planner = CutoverStrategyPlanner(config)
        plan = planner.build_plan()

        assert plan.strategy == CutoverType.PHASED
        assert len(plan.phases) == 3
        assert plan.total_tasks == 4
        assert plan.phases[0].name == "reference"
        assert plan.phases[1].depends_on == ["reference"]

    def test_big_bang_plan(self) -> None:
        config = {
            "strategy": "big_bang",
            "phases": [
                {"name": "all", "entities": ["customers", "products", "orders"]},
            ],
        }
        planner = CutoverStrategyPlanner(config)
        plan = planner.build_plan()

        assert plan.strategy == CutoverType.BIG_BANG
        assert len(plan.phases) == 1
        assert plan.phases[0].name == "big_bang"

    def test_plan_summary(self) -> None:
        config = {
            "strategy": "phased",
            "phases": [
                {"name": "p1", "entities": ["a", "b"], "order": 1},
            ],
        }
        planner = CutoverStrategyPlanner(config)
        plan = planner.build_plan()
        summary = plan.summary()
        assert summary["strategy"] == "phased"
        assert summary["total_phases"] == 1
        assert summary["total_tasks"] == 2


class TestCheckpointManager:

    def test_create_and_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, pipeline_name="test")
            mgr.create("run001")
            mgr.record_task("task_1", "customer", "phase_1", "completed", rows_loaded=100)
            mgr.record_task("task_2", "product", "phase_1", "failed", error_message="timeout")

            restored = mgr.restore("run001")
            assert restored is not None
            assert len(restored.completed_tasks()) == 1
            assert len(restored.failed_tasks()) == 1

    def test_resumable_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, pipeline_name="test")
            mgr.create("run002")
            mgr.record_task("t1", "a", "p1", "completed", rows_loaded=50)
            mgr.record_task("t2", "b", "p1", "failed", error_message="err")
            mgr.record_task("t3", "c", "p2", "pending")

            resumable = mgr.get_resumable_tasks("run002")
            assert "t2" in resumable
            assert "t3" in resumable
            assert "t1" not in resumable

    def test_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, pipeline_name="test")
            mgr.create("run003")
            mgr.record_task("t1", "a", "p1", "completed")
            mgr.cleanup("run003")

            restored = mgr.restore("run003")
            assert restored is None

    def test_disabled_mode(self) -> None:
        mgr = CheckpointManager(enabled=False)
        mgr.create("run_disabled")
        mgr.record_task("t1", "a", "p1", "completed")
        # Should not raise


class TestRollbackManager:

    def test_snapshot_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RollbackManager(snapshot_dir=tmpdir)
            original_data = [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
            mgr.take_snapshot("DIM_CUSTOMER", current_data=original_data)

            # Simulate a failed load — target is now different
            target_store = {"DIM_CUSTOMER": [{"id": 99, "name": "Bad Data"}]}

            action = mgr.rollback("DIM_CUSTOMER", target_store)
            assert action.success
            assert action.rows_affected == 2
            assert len(target_store["DIM_CUSTOMER"]) == 2
            assert target_store["DIM_CUSTOMER"][0]["name"] == "Alice"

    def test_rollback_no_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RollbackManager(snapshot_dir=tmpdir)
            action = mgr.rollback("NONEXISTENT")
            assert not action.success

    def test_rollback_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RollbackManager(snapshot_dir=tmpdir)
            mgr.take_snapshot("TABLE_A", current_data=[{"x": 1}])
            mgr.take_snapshot("TABLE_B", current_data=[{"y": 2}])

            store = {"TABLE_A": [], "TABLE_B": []}
            actions = mgr.rollback_all(store)
            assert len(actions) == 2
            assert all(a.success for a in actions)


class TestParallelRunner:

    def test_execute_simple_plan(self) -> None:
        config = {
            "strategy": "phased",
            "phases": [
                {"name": "p1", "entities": ["a", "b"], "order": 1},
            ],
        }
        planner = CutoverStrategyPlanner(config)
        plan = planner.build_plan()

        runner = ParallelRunner(max_workers=2, max_retries=1)
        result = runner.execute(plan)

        assert result.success
        assert len(result.phases) == 1
        assert result.phases[0].success

    def test_execute_with_dependencies(self) -> None:
        config = {
            "strategy": "phased",
            "phases": [
                {"name": "ref", "entities": ["products"], "order": 1},
                {"name": "master", "entities": ["customers"], "order": 2, "depends_on": ["ref"]},
            ],
        }
        planner = CutoverStrategyPlanner(config)
        plan = planner.build_plan()

        runner = ParallelRunner(max_workers=2, max_retries=1)
        result = runner.execute(plan)

        assert result.success
        assert len(result.phases) == 2

    def test_execute_with_failure(self) -> None:
        def failing_load(task, **kwargs):
            if task.entity == "bad_entity":
                raise RuntimeError("Simulated failure")
            return 100

        config = {
            "strategy": "phased",
            "phases": [
                {"name": "p1", "entities": ["good_entity", "bad_entity"], "order": 1},
            ],
        }
        planner = CutoverStrategyPlanner(config)
        plan = planner.build_plan()

        runner = ParallelRunner(
            max_workers=2, max_retries=1,
            load_fn=failing_load,
        )
        result = runner.execute(plan)
        assert not result.success
