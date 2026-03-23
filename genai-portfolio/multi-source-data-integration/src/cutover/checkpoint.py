"""Checkpoint and restart support.

Persists pipeline progress to disk so that a failed run can be resumed
from the last successful checkpoint rather than starting from scratch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class CheckpointEntry:
    """State of a single load task at checkpoint time."""

    task_id: str
    entity: str
    phase: str
    status: str  # pending | completed | failed
    rows_loaded: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PipelineCheckpoint:
    """Full pipeline state at a point in time."""

    pipeline_name: str
    run_id: str
    checkpoint_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    entries: List[CheckpointEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def completed_tasks(self) -> List[str]:
        return [e.task_id for e in self.entries if e.status == "completed"]

    def failed_tasks(self) -> List[str]:
        return [e.task_id for e in self.entries if e.status == "failed"]

    def pending_tasks(self) -> List[str]:
        return [e.task_id for e in self.entries if e.status == "pending"]


class CheckpointManager:
    """Persist and restore pipeline checkpoints to/from disk.

    Parameters
    ----------
    checkpoint_dir:
        Directory for checkpoint files.
    pipeline_name:
        Pipeline identifier for namespacing.
    enabled:
        When ``False``, all operations are no-ops.
    """

    def __init__(
        self,
        checkpoint_dir: str = "/tmp/msdi_checkpoints",
        pipeline_name: str = "default",
        enabled: bool = True,
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._pipeline_name = pipeline_name
        self._enabled = enabled
        self._current: Optional[PipelineCheckpoint] = None

        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, run_id: str) -> Path:
        return self._dir / f"{self._pipeline_name}_{run_id}.json"

    def create(self, run_id: str) -> PipelineCheckpoint:
        """Create a new checkpoint for a pipeline run."""
        cp = PipelineCheckpoint(
            pipeline_name=self._pipeline_name,
            run_id=run_id,
        )
        self._current = cp
        logger.info("Checkpoint created for run %s", run_id)
        return cp

    def record_task(
        self,
        task_id: str,
        entity: str,
        phase: str,
        status: str,
        rows_loaded: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Record the state of a task in the current checkpoint."""
        if not self._enabled or self._current is None:
            return

        # Update or create entry
        existing = next(
            (e for e in self._current.entries if e.task_id == task_id), None
        )
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            existing.status = status
            existing.rows_loaded = rows_loaded
            existing.error_message = error_message
            if status == "completed":
                existing.completed_at = now
        else:
            entry = CheckpointEntry(
                task_id=task_id,
                entity=entity,
                phase=phase,
                status=status,
                rows_loaded=rows_loaded,
                started_at=now,
                completed_at=now if status == "completed" else None,
                error_message=error_message,
            )
            self._current.entries.append(entry)

        self._persist()

    def _persist(self) -> None:
        """Write the current checkpoint to disk."""
        if not self._enabled or self._current is None:
            return
        path = self._checkpoint_path(self._current.run_id)
        data = {
            "pipeline_name": self._current.pipeline_name,
            "run_id": self._current.run_id,
            "checkpoint_time": datetime.now(timezone.utc).isoformat(),
            "entries": [
                {
                    "task_id": e.task_id,
                    "entity": e.entity,
                    "phase": e.phase,
                    "status": e.status,
                    "rows_loaded": e.rows_loaded,
                    "started_at": e.started_at,
                    "completed_at": e.completed_at,
                    "error_message": e.error_message,
                }
                for e in self._current.entries
            ],
            "metadata": self._current.metadata,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug("Checkpoint persisted to %s", path)

    def restore(self, run_id: str) -> Optional[PipelineCheckpoint]:
        """Restore a checkpoint from disk."""
        path = self._checkpoint_path(run_id)
        if not path.exists():
            logger.warning("No checkpoint found for run %s", run_id)
            return None

        data = json.loads(path.read_text())
        entries = [
            CheckpointEntry(**e) for e in data.get("entries", [])
        ]
        cp = PipelineCheckpoint(
            pipeline_name=data["pipeline_name"],
            run_id=data["run_id"],
            checkpoint_time=data.get("checkpoint_time", ""),
            entries=entries,
            metadata=data.get("metadata", {}),
        )
        self._current = cp
        logger.info(
            "Checkpoint restored for run %s: %d completed, %d failed, %d pending",
            run_id,
            len(cp.completed_tasks()),
            len(cp.failed_tasks()),
            len(cp.pending_tasks()),
        )
        return cp

    def get_resumable_tasks(self, run_id: str) -> List[str]:
        """Return task IDs that need to be re-run after a failure."""
        cp = self.restore(run_id)
        if cp is None:
            return []
        return cp.failed_tasks() + cp.pending_tasks()

    def cleanup(self, run_id: str) -> None:
        """Remove a checkpoint file after successful completion."""
        path = self._checkpoint_path(run_id)
        if path.exists():
            path.unlink()
            logger.info("Checkpoint cleaned up for run %s", run_id)
