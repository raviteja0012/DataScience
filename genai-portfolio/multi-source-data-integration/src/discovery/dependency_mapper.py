"""Table and foreign-key dependency mapping.

Builds a directed acyclic graph (DAG) of table dependencies from foreign-key
metadata.  The DAG determines safe load ordering during the cutover phase:
parent (referenced) tables must be loaded before child (referencing) tables.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


@dataclass
class ForeignKey:
    name: str
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass
class TableNode:
    """A node in the dependency DAG."""

    name: str
    schema_name: str = ""
    primary_key: Optional[str] = None
    row_count: int = 0
    depends_on: List[str] = field(default_factory=list)
    depended_by: List[str] = field(default_factory=list)
    depth: int = 0  # topological depth (0 = root)


@dataclass
class DependencyGraph:
    """Wrapper around the table dependency DAG with helper methods."""

    source_system: str
    nodes: Dict[str, TableNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)

    @property
    def table_count(self) -> int:
        return len(self.nodes)

    @property
    def root_tables(self) -> List[str]:
        """Tables with no inbound dependencies (load first)."""
        return [n for n, node in self.nodes.items() if not node.depends_on]

    @property
    def leaf_tables(self) -> List[str]:
        """Tables that nothing else depends on (load last)."""
        return [n for n, node in self.nodes.items() if not node.depended_by]

    def topological_order(self) -> List[str]:
        """Return tables in a valid load order (Kahn's algorithm).

        Returns
        -------
        list[str]
            Table names in dependency-safe load order.

        Raises
        ------
        ValueError
            If a cycle is detected (should not happen in a well-formed schema).
        """
        in_degree: Dict[str, int] = {n: 0 for n in self.nodes}
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for child, parent in self.edges:
            adjacency[parent].append(child)
            in_degree[child] = in_degree.get(child, 0) + 1

        queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
        order: List[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbour in adjacency.get(current, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(self.nodes):
            loaded = set(order)
            cycle_candidates = [n for n in self.nodes if n not in loaded]
            raise ValueError(
                f"Cycle detected in dependency graph involving: {cycle_candidates}"
            )

        return order

    def load_waves(self) -> List[List[str]]:
        """Group tables into parallel-safe load waves.

        Tables in the same wave have no dependencies on each other and can be
        loaded concurrently.
        """
        remaining = dict(self.nodes)
        loaded: Set[str] = set()
        waves: List[List[str]] = []

        while remaining:
            wave = [
                name for name, node in remaining.items()
                if all(dep in loaded for dep in node.depends_on)
            ]
            if not wave:
                raise ValueError("Circular dependency prevents wave computation.")
            waves.append(sorted(wave))
            for name in wave:
                loaded.add(name)
                del remaining[name]

        return waves


class DependencyMapper:
    """Build a ``DependencyGraph`` from source-system YAML configuration."""

    def __init__(self, max_depth: int = 5) -> None:
        self._max_depth = max_depth

    def build_graph(self, source_config: Dict[str, Any]) -> DependencyGraph:
        """Parse FK relationships from a source-system YAML and return the DAG."""
        system_id = source_config.get("system", {}).get("id", "unknown")
        graph = DependencyGraph(source_system=system_id)

        for schema in source_config.get("schemas", []):
            schema_name = schema.get("name", "")

            # Register all tables as nodes
            for tbl in schema.get("tables", []):
                table_name = tbl["name"]
                graph.nodes[table_name] = TableNode(
                    name=table_name,
                    schema_name=schema_name,
                    primary_key=tbl.get("primary_key"),
                    row_count=tbl.get("estimated_rows", 0),
                )

            # Add edges from foreign keys
            for fk_def in schema.get("foreign_keys", []):
                fk = ForeignKey(**fk_def)
                child = fk.source_table
                parent = fk.target_table

                if child in graph.nodes and parent in graph.nodes:
                    graph.edges.append((child, parent))
                    graph.nodes[child].depends_on.append(parent)
                    graph.nodes[parent].depended_by.append(child)

        # Compute depth for each node
        self._compute_depths(graph)

        logger.info(
            "Dependency graph for %s: %d tables, %d FK edges, roots=%s",
            system_id, graph.table_count, len(graph.edges),
            graph.root_tables,
        )
        return graph

    def _compute_depths(self, graph: DependencyGraph) -> None:
        """BFS from root nodes to assign depth levels."""
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()

        for root in graph.root_tables:
            queue.append((root, 0))
            visited.add(root)

        while queue:
            name, depth = queue.popleft()
            if depth > self._max_depth:
                continue
            graph.nodes[name].depth = depth
            for child_name in graph.nodes[name].depended_by:
                if child_name not in visited:
                    visited.add(child_name)
                    queue.append((child_name, depth + 1))

    def merge_graphs(self, graphs: List[DependencyGraph]) -> DependencyGraph:
        """Merge multiple source-system graphs into a unified view.

        Useful when planning a cross-system cutover that must respect
        dependencies from both Oracle and SQL Server simultaneously.
        """
        merged = DependencyGraph(source_system="merged")
        for g in graphs:
            prefix = g.source_system + "."
            for name, node in g.nodes.items():
                qualified = prefix + name
                merged.nodes[qualified] = TableNode(
                    name=qualified,
                    schema_name=node.schema_name,
                    primary_key=node.primary_key,
                    row_count=node.row_count,
                    depends_on=[prefix + d for d in node.depends_on],
                    depended_by=[prefix + d for d in node.depended_by],
                    depth=node.depth,
                )
            for child, parent in g.edges:
                merged.edges.append((prefix + child, prefix + parent))

        logger.info(
            "Merged %d source graphs -> %d total nodes, %d edges",
            len(graphs), merged.table_count, len(merged.edges),
        )
        return merged
