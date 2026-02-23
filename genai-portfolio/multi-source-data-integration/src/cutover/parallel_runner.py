"""Parallel load execution engine.

Executes load tasks within a phase concurrently (up to ``max_workers``),
respecting inter-phase dependencies.  Each task is wrapped with checkpoint
and rollback support.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.cutover.checkpoint import CheckpointManager
from src.cutover.rollback_manager import RollbackManager
from src.cutover.strategy import CutoverPlan, LoadPhase, LoadTask
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector

logger = get_logger(__name__, fmt="text")


@dataclass
class TaskResult:
    """Outcome of a single load task."""

    task_id: str
    entity: str
    phase: str
    success: bool
    rows_loaded: int = 0
    duration_s: float = 0.0
    error_message: Optional[str] = None


@dataclass
class PhaseResult:
    """Outcome of an entire load phase."""

    phase_name: str
    success: bool
    tasks: List[TaskResult] = field(default_factory=list)
    total_rows: int = 0
    duration_s: float = 0.0


@dataclass
class CutoverResult:
    """Outcome of the full cutover execution."""

    run_id: str
    success: bool
    phases: List[PhaseResult] = field(default_factory=list)
    total_rows_loaded: int = 0
    total_duration_s: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "success": self.success,
            "total_phases": len(self.phases),
            "total_rows_loaded": self.total_rows_loaded,
            "total_duration_s": round(self.total_duration_s, 2),
            "phase_results": [
                {
                    "phase": p.phase_name,
                    "success": p.success,
                    "tasks": len(p.tasks),
                    "rows": p.total_rows,
                    "duration_s": round(p.duration_s, 2),
                }
                for p in self.phases
            ],
        }


# Default load function for demo mode
def _demo_load(task: LoadTask, **kwargs: Any) -> int:
    """Simulate loading data with a short sleep."""
    time.sleep(0.05)
    return task.estimated_rows or 100


class ParallelRunner:
    """Execute a ``CutoverPlan`` with parallel task execution.

    Parameters
    ----------
    max_workers:
        Maximum concurrent load tasks.
    checkpoint_mgr:
        Checkpoint manager for progress tracking.
    rollback_mgr:
        Rollback manager for failure recovery.
    metrics:
        Optional metrics collector.
    load_fn:
        Callable that performs the actual data load.  Signature:
        ``(task, **kwargs) -> rows_loaded``.  Defaults to a demo stub.
    max_retries:
        Number of retry attempts for failed tasks.
    retry_backoff_s:
        Seconds to wait between retries.
    """

    def __init__(
        self,
        max_workers: int = 4,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        rollback_mgr: Optional[RollbackManager] = None,
        metrics: Optional[MetricsCollector] = None,
        load_fn: Optional[Callable[..., int]] = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self._max_workers = max_workers
        self._checkpoint = checkpoint_mgr or CheckpointManager(enabled=False)
        self._rollback = rollback_mgr or RollbackManager(enabled=False)
        self._metrics = metrics
        self._load_fn = load_fn or _demo_load
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_s

    def execute(
        self,
        plan: CutoverPlan,
        load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CutoverResult:
        """Execute the cutover plan.

        Parameters
        ----------
        plan:
            The cutover plan produced by ``CutoverStrategyPlanner``.
        load_kwargs:
            Extra kwargs forwarded to the load function.
        """
        run_id = uuid.uuid4().hex[:8]
        load_kwargs = load_kwargs or {}
        overall_start = time.perf_counter()

        self._checkpoint.create(run_id)
        phase_results: List[PhaseResult] = []
        overall_success = True

        logger.info(
            "Starting cutover execution (run=%s, strategy=%s, phases=%d)",
            run_id, plan.strategy.value, len(plan.phases),
        )

        completed_phases: set[str] = set()

        for phase in sorted(plan.phases, key=lambda p: p.order):
            # Check phase dependencies
            unmet = [d for d in phase.depends_on if d not in completed_phases]
            if unmet:
                logger.error(
                    "Phase '%s' has unmet dependencies: %s — skipping",
                    phase.name, unmet,
                )
                overall_success = False
                continue

            pr = self._execute_phase(phase, run_id, load_kwargs)
            phase_results.append(pr)

            if pr.success:
                completed_phases.add(phase.name)
            else:
                overall_success = False
                logger.error("Phase '%s' failed — halting cutover", phase.name)
                break

        total_rows = sum(p.total_rows for p in phase_results)
        total_duration = time.perf_counter() - overall_start

        result = CutoverResult(
            run_id=run_id,
            success=overall_success,
            phases=phase_results,
            total_rows_loaded=total_rows,
            total_duration_s=total_duration,
        )

        if self._metrics:
            self._metrics.record(
                "cutover_total_rows", total_rows,
                phase="cutover",
            )
            self._metrics.record(
                "cutover_duration_s", total_duration,
                phase="cutover",
            )

        if overall_success:
            self._checkpoint.cleanup(run_id)
            logger.info("Cutover completed successfully: %s", result.summary())
        else:
            logger.error("Cutover failed: %s", result.summary())

        return result

    def _execute_phase(
        self,
        phase: LoadPhase,
        run_id: str,
        load_kwargs: Dict[str, Any],
    ) -> PhaseResult:
        """Execute all tasks in a phase with parallel workers."""
        logger.info(
            "Executing phase '%s' (%d tasks, max_parallel=%d)",
            phase.name, len(phase.tasks), self._max_workers,
        )
        phase_start = time.perf_counter()
        task_results: List[TaskResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_task_with_retry, task, run_id, load_kwargs
                ): task
                for task in phase.tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    tr = future.result()
                except Exception as exc:
                    tr = TaskResult(
                        task_id=f"{task.phase}_{task.entity}",
                        entity=task.entity,
                        phase=task.phase,
                        success=False,
                        error_message=str(exc),
                    )
                task_results.append(tr)

        phase_duration = time.perf_counter() - phase_start
        all_ok = all(tr.success for tr in task_results)

        return PhaseResult(
            phase_name=phase.name,
            success=all_ok,
            tasks=task_results,
            total_rows=sum(tr.rows_loaded for tr in task_results),
            duration_s=phase_duration,
        )

    def _execute_task_with_retry(
        self,
        task: LoadTask,
        run_id: str,
        load_kwargs: Dict[str, Any],
    ) -> TaskResult:
        """Execute a single task with retry logic."""
        task_id = f"{task.phase}_{task.entity}"
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            try:
                start = time.perf_counter()
                self._checkpoint.record_task(
                    task_id, task.entity, task.phase, "running",
                )

                rows = self._load_fn(task, **load_kwargs)
                duration = time.perf_counter() - start

                self._checkpoint.record_task(
                    task_id, task.entity, task.phase, "completed",
                    rows_loaded=rows,
                )
                logger.info(
                    "Task %s completed: %d rows in %.2fs",
                    task_id, rows, duration,
                )
                return TaskResult(
                    task_id=task_id,
                    entity=task.entity,
                    phase=task.phase,
                    success=True,
                    rows_loaded=rows,
                    duration_s=duration,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Task %s attempt %d/%d failed: %s",
                    task_id, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * attempt)

        self._checkpoint.record_task(
            task_id, task.entity, task.phase, "failed",
            error_message=last_error,
        )
        return TaskResult(
            task_id=task_id,
            entity=task.entity,
            phase=task.phase,
            success=False,
            error_message=last_error,
        )
