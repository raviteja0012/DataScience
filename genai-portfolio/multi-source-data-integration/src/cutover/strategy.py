"""Cutover strategy planner.

Supports two primary strategies:

- **Big-bang**: All entities are loaded in a single coordinated window.
- **Phased**: Entities are grouped into ordered phases with explicit
  dependencies (e.g. reference data before master data before transactional).

The planner consults the dependency DAG to produce an executable load plan
respecting both inter-entity dependencies and the chosen strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class CutoverType(Enum):
    BIG_BANG = "big_bang"
    PHASED = "phased"


@dataclass
class LoadTask:
    """A single entity-load unit of work."""

    entity: str
    source_system: str
    target_table: str
    estimated_rows: int = 0
    depends_on: List[str] = field(default_factory=list)
    phase: str = ""
    order: int = 0
    status: str = "pending"  # pending | running | completed | failed | rolled_back


@dataclass
class LoadPhase:
    """A group of tasks that can execute after their dependencies are met."""

    name: str
    order: int
    tasks: List[LoadTask] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"

    @property
    def estimated_rows(self) -> int:
        return sum(t.estimated_rows for t in self.tasks)


@dataclass
class CutoverPlan:
    """Complete cutover execution plan."""

    strategy: CutoverType
    phases: List[LoadPhase] = field(default_factory=list)
    total_tasks: int = 0
    total_estimated_rows: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "total_phases": len(self.phases),
            "total_tasks": self.total_tasks,
            "total_estimated_rows": self.total_estimated_rows,
            "phases": [
                {
                    "name": p.name,
                    "order": p.order,
                    "tasks": len(p.tasks),
                    "estimated_rows": p.estimated_rows,
                    "depends_on": p.depends_on,
                }
                for p in self.phases
            ],
        }


class CutoverStrategyPlanner:
    """Build a ``CutoverPlan`` from the pipeline configuration.

    Parameters
    ----------
    config:
        The ``cutover`` section of the pipeline config YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._strategy = CutoverType(config.get("strategy", "phased"))

    def build_plan(
        self,
        entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> CutoverPlan:
        """Generate the cutover plan.

        Parameters
        ----------
        entity_metadata:
            Optional ``{entity_name: {target_table, estimated_rows, ...}}``
            for enriching load tasks.
        """
        entity_metadata = entity_metadata or {}

        if self._strategy == CutoverType.BIG_BANG:
            return self._build_big_bang(entity_metadata)
        return self._build_phased(entity_metadata)

    def _build_phased(
        self, entity_metadata: Dict[str, Dict[str, Any]]
    ) -> CutoverPlan:
        phases_cfg = self._config.get("phases", [])
        phases: List[LoadPhase] = []

        for pcfg in sorted(phases_cfg, key=lambda p: p.get("order", 0)):
            tasks: List[LoadTask] = []
            for entity_name in pcfg.get("entities", []):
                meta = entity_metadata.get(entity_name, {})
                tasks.append(LoadTask(
                    entity=entity_name,
                    source_system="all",
                    target_table=meta.get("target_table", f"INTEGRATED.{entity_name.upper()}"),
                    estimated_rows=meta.get("estimated_rows", 0),
                    depends_on=pcfg.get("depends_on", []),
                    phase=pcfg["name"],
                    order=pcfg.get("order", 0),
                ))

            lp = LoadPhase(
                name=pcfg["name"],
                order=pcfg.get("order", 0),
                tasks=tasks,
                depends_on=pcfg.get("depends_on", []),
            )
            phases.append(lp)

        plan = CutoverPlan(
            strategy=CutoverType.PHASED,
            phases=phases,
            total_tasks=sum(len(p.tasks) for p in phases),
            total_estimated_rows=sum(p.estimated_rows for p in phases),
        )
        logger.info("Built phased cutover plan: %s", plan.summary())
        return plan

    def _build_big_bang(
        self, entity_metadata: Dict[str, Dict[str, Any]]
    ) -> CutoverPlan:
        all_entities = set()
        for pcfg in self._config.get("phases", []):
            all_entities.update(pcfg.get("entities", []))

        tasks = [
            LoadTask(
                entity=e,
                source_system="all",
                target_table=entity_metadata.get(e, {}).get(
                    "target_table", f"INTEGRATED.{e.upper()}"
                ),
                estimated_rows=entity_metadata.get(e, {}).get("estimated_rows", 0),
                phase="big_bang",
            )
            for e in sorted(all_entities)
        ]

        plan = CutoverPlan(
            strategy=CutoverType.BIG_BANG,
            phases=[LoadPhase(name="big_bang", order=1, tasks=tasks)],
            total_tasks=len(tasks),
            total_estimated_rows=sum(t.estimated_rows for t in tasks),
        )
        logger.info("Built big-bang cutover plan: %s", plan.summary())
        return plan
