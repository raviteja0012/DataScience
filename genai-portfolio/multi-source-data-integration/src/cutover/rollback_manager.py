"""Rollback capabilities for failed cutover loads.

Before each load task, captures a "before-state" snapshot of the target
table (row counts, hashes, or full data) so that the load can be reversed
if validation fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class Snapshot:
    """Before-state snapshot for a single target table."""

    table_name: str
    snapshot_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    row_count: int = 0
    data: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None


@dataclass
class RollbackAction:
    """A recorded rollback action."""

    table_name: str
    action: str  # "restore" | "truncate" | "delete_delta"
    rows_affected: int = 0
    executed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    success: bool = True


class RollbackManager:
    """Manage before-state snapshots and rollback execution.

    Parameters
    ----------
    snapshot_dir:
        Directory for snapshot files.
    enabled:
        When ``False`` all operations are no-ops.
    """

    def __init__(
        self,
        snapshot_dir: str = "/tmp/msdi_rollback",
        enabled: bool = True,
    ) -> None:
        self._dir = Path(snapshot_dir)
        self._enabled = enabled
        self._snapshots: Dict[str, Snapshot] = {}
        self._rollback_log: List[RollbackAction] = []

        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def take_snapshot(
        self,
        table_name: str,
        current_data: Optional[List[Dict[str, Any]]] = None,
        row_count: int = 0,
    ) -> Snapshot:
        """Capture the current state of a target table before loading.

        For small tables the full data is retained in memory; for large tables
        only the row count and metadata are stored.
        """
        if not self._enabled:
            return Snapshot(table_name=table_name)

        snap = Snapshot(
            table_name=table_name,
            row_count=row_count or (len(current_data) if current_data else 0),
        )

        if current_data is not None:
            # Persist to disk for large datasets
            snap_path = self._dir / f"{table_name.replace('.', '_')}_snapshot.json"
            snap_path.write_text(json.dumps(
                {"table": table_name, "rows": current_data, "count": snap.row_count},
                indent=2, default=str,
            ))
            snap.file_path = str(snap_path)
            snap.data = current_data if len(current_data) <= 10000 else None

        self._snapshots[table_name] = snap
        logger.info(
            "Snapshot taken for %s: %d rows%s",
            table_name, snap.row_count,
            f" (persisted to {snap.file_path})" if snap.file_path else "",
        )
        return snap

    def rollback(
        self,
        table_name: str,
        target_data_store: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> RollbackAction:
        """Restore a table to its pre-load state.

        In demo mode this operates on in-memory data stores.  In production
        this would issue DELETE + re-INSERT SQL against the target database.
        """
        snap = self._snapshots.get(table_name)
        if snap is None:
            logger.warning("No snapshot found for %s — cannot rollback", table_name)
            action = RollbackAction(
                table_name=table_name, action="no_snapshot",
                rows_affected=0, success=False,
            )
            self._rollback_log.append(action)
            return action

        rows_restored = 0
        if target_data_store is not None:
            # In-memory rollback (demo mode)
            if snap.data is not None:
                target_data_store[table_name] = list(snap.data)
                rows_restored = len(snap.data)
            elif snap.file_path and Path(snap.file_path).exists():
                restored = json.loads(Path(snap.file_path).read_text())
                target_data_store[table_name] = restored.get("rows", [])
                rows_restored = len(target_data_store[table_name])
            else:
                target_data_store[table_name] = []

        action = RollbackAction(
            table_name=table_name,
            action="restore",
            rows_affected=rows_restored,
            success=True,
        )
        self._rollback_log.append(action)
        logger.info("Rolled back %s: restored %d rows", table_name, rows_restored)
        return action

    def rollback_all(
        self,
        target_data_store: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[RollbackAction]:
        """Rollback all tables with snapshots."""
        actions: List[RollbackAction] = []
        for table_name in list(self._snapshots.keys()):
            action = self.rollback(table_name, target_data_store)
            actions.append(action)
        return actions

    def get_rollback_log(self) -> List[RollbackAction]:
        return list(self._rollback_log)

    def cleanup_snapshots(self) -> None:
        """Remove snapshot files after successful migration."""
        for snap in self._snapshots.values():
            if snap.file_path and Path(snap.file_path).exists():
                Path(snap.file_path).unlink()
        self._snapshots.clear()
        logger.info("All snapshots cleaned up")
