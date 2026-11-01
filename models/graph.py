"""
Road network domain model for Opti-Route (Phase 1).

Contains:
- NodeType : what kind of place a node is (junction, depot, hub).
- Node     : a location on the map.
- Edge     : a road segment between two nodes.
- RoadGraph: the whole network, with editing, change notifications and an
             adjacency matrix. This class has no GUI code, so it can be tested
             on its own.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - lightweight fallback when numpy is unavailable
    class _MatrixFlags:
        def __init__(self) -> None:
            self.writeable: bool = True

    class _FallbackRow:
        def __init__(self, data: List[float], flags: _MatrixFlags) -> None:
            self._data = data
            self._flags = flags

        def __getitem__(self, idx: int) -> float:
            return self._data[idx]

        def __setitem__(self, idx: int, value: float) -> None:
            if not self._flags.writeable:
                raise ValueError("assignment destination is read-only")
            self._data[idx] = float(value)

        def __len__(self) -> int:
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def __eq__(self, other: object) -> bool:
            if isinstance(other, list):
                return self._data == other
            if isinstance(other, _FallbackRow):
                return self._data == other._data
            return False

    class _FallbackNDArray:
        def __init__(self, shape: Tuple[int, int], fill_value: float = 0.0) -> None:
            rows, cols = shape
            self.shape: Tuple[int, int] = (rows, cols)
            self.flags = _MatrixFlags()
            self._grid: List[List[float]] = [
                [float(fill_value) for _ in range(cols)] for _ in range(rows)
            ]

        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, tuple):
                r, c = key
                return self._grid[r][c]
            return _FallbackRow(self._grid[key], self.flags)

        def __setitem__(self, key: Any, value: float) -> None:
            if not self.flags.writeable:
                raise ValueError("assignment destination is read-only")
            if isinstance(key, tuple):
                r, c = key
                self._grid[r][c] = float(value)
            else:
                self._grid[key] = [float(v) for v in value]

        def __len__(self) -> int:
            return self.shape[0]

        def tolist(self) -> List[List[float]]:
            return [list(row) for row in self._grid]

    class _FallbackNumpy:
        ndarray = _FallbackNDArray
        inf = math.inf

        @staticmethod
        def full(shape: Tuple[int, int], fill_value: float) -> _FallbackNDArray:
            return _FallbackNDArray(shape, fill_value)

        @staticmethod
        def fill_diagonal(matrix: _FallbackNDArray, val: float) -> None:
            n = min(matrix.shape[0], matrix.shape[1])
            for idx in range(n):
                matrix[idx, idx] = float(val)

    np = _FallbackNumpy()  # type: ignore


class NodeType(str, Enum):
    """The role a node plays in the road network."""

    JUNCTION = "junction"
    DEPOT = "depot"
    HUB = "hub"


@dataclass
class Node:
    """Represents a node (intersection/depot/location) in the road network."""

    id: str
    x: float
    y: float
    name: str = ""
    node_type: NodeType = NodeType.JUNCTION

    @property
    def is_depot(self) -> bool:
        """True if this node is a depot."""
        return self.node_type == NodeType.DEPOT

    def distance_to(self, other: "Node") -> float:
        """Calculate Euclidean distance between this node and another node."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Edge:
    """Represents a road segment connecting two nodes with a cost/distance weight."""

    source: str
    target: str
    weight: float
    directed: bool = False

    def connects(self, first: str, second: str) -> bool:
        """
        True if this edge lets you travel from `first` to `second`.
        An undirected edge works in both directions.
        """
        if self.source == first and self.target == second:
            return True
        return (not self.directed) and self.source == second and self.target == first

    def touches(self, node_id: str) -> bool:
        """True if this edge starts or ends at node_id."""
        return node_id in (self.source, self.target)


# A listener receives the event name and keyword details about what changed.
GraphListener = Callable[..., None]


class RoadGraph:
    """
    Road network graph with weighted directed and undirected edges.

    The list of Edge objects is the single source of truth. The adjacency
    list used by the solvers is always built from it.
    """

    def __init__(self, distance_unit: str = "km"):
        if not isinstance(distance_unit, str) or not distance_unit.strip():
            raise ValueError("distance_unit must be a non-empty string.")

        self.nodes: Dict[str, Node] = {}
        self.distance_unit: str = distance_unit
        self._edges: List[Edge] = []
        self._adj: Dict[str, List[Tuple[str, float]]] = {}
        self._listeners: List[GraphListener] = []

        # Cached adjacency matrix; rebuilt only after a structural change.
        self._matrix_cache: Optional[np.ndarray] = None
        self._index_cache: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Change notifications (so the GUI can redraw when the graph changes)
    # ------------------------------------------------------------------

    def register_listener(self, callback: GraphListener) -> None:
        """Register callback(event_name, **details) to run after every change."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unregister_listener(self, callback: GraphListener) -> None:
        """Stop notifying a previously registered callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, event: str, **details: Any) -> None:
        for listener in list(self._listeners):
            listener(event, **details)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a Node. Raises ValueError for a bad node or a duplicate id."""
        if not isinstance(node.id, str) or not node.id.strip():
            raise ValueError("Node id must be a non-empty string.")
        if node.id in self.nodes:
            raise ValueError(f"Node '{node.id}' already exists in graph.")
        self._check_coordinates(node.x, node.y)
        if not isinstance(node.node_type, NodeType):
            raise ValueError("node_type must be a NodeType value.")

        self.nodes[node.id] = node
        self._adj[node.id] = []
        self._invalidate_matrix()
        self._notify("node_added", node_id=node.id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and every edge connected to it."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist in graph.")

        self._edges = [e for e in self._edges if not e.touches(node_id)]
        del self.nodes[node_id]
        self._rebuild_adjacency()
        self._invalidate_matrix()
        self._notify("node_removed", node_id=node_id)

    def move_node(self, node_id: str, x: float, y: float) -> None:
        """
        Change a node's canvas position. Edge weights stay as they are,
        because a weight is the length of the road, not the straight line.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist in graph.")
        self._check_coordinates(x, y)

        node = self.nodes[node_id]
        node.x = float(x)
        node.y = float(y)
        self._notify("node_moved", node_id=node_id, x=node.x, y=node.y)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve node object by ID."""
        return self.nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self.nodes

    def node_count(self) -> int:
        """Return total number of nodes in graph."""
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str, weight: float, directed: bool = False) -> None:
        """
        Add a weighted road between two existing nodes.

        Raises KeyError if a node does not exist, and ValueError if the
        weight is not a positive finite number or source equals target.
        Adding the same pair twice creates a parallel road.
        """
        if source not in self.nodes:
            raise KeyError(f"Source node '{source}' does not exist in graph.")
        if target not in self.nodes:
            raise KeyError(f"Target node '{target}' does not exist in graph.")
        if source == target:
            raise ValueError("An edge cannot connect a node to itself.")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("Edge weight must be a positive finite number.")

        edge = Edge(source, target, float(weight), directed)
        self._edges.append(edge)
        self._adj[source].append((target, edge.weight))
        if not directed:
            self._adj[target].append((source, edge.weight))
        self._invalidate_matrix()
        self._notify("edge_added", source=source, target=target, weight=edge.weight)

    def remove_edge(self, source: str, target: str) -> int:
        """
        Remove every road that lets you travel from source to target
        (an undirected road is removed whichever way round you name it).
        Returns how many edges were removed; raises KeyError if none.
        """
        kept = [e for e in self._edges if not e.connects(source, target)]
        removed = len(self._edges) - len(kept)
        if removed == 0:
            raise KeyError(f"No edge found between '{source}' and '{target}'.")

        self._edges = kept
        self._rebuild_adjacency()
        self._invalidate_matrix()
        self._notify("edge_removed", source=source, target=target)
        return removed

    def has_edge(self, source: str, target: str) -> bool:
        """True if you can travel directly from source to target."""
        return any(e.connects(source, target) for e in self._edges)

    @property
    def edges(self) -> Tuple[Edge, ...]:
        """Read-only snapshot of all edges."""
        return tuple(self._edges)

    def edge_count(self) -> int:
        """Return the number of roads (an undirected road counts once)."""
        return len(self._edges)

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Return list of (neighbor_node_id, edge_weight) tuples for given node_id."""
        return self._adj.get(node_id, [])

    # ------------------------------------------------------------------
    # Adjacency matrix
    # ------------------------------------------------------------------

    def node_index(self) -> Dict[str, int]:
        """Map each node id to its row/column number in the adjacency matrix."""
        self._ensure_matrix()
        assert self._index_cache is not None
        return dict(self._index_cache)

    def adjacency_matrix(self) -> np.ndarray:
        """
        Return the adjacency matrix. Entry [i][j] is the cheapest direct road
        from node i to node j, 0 on the diagonal, and infinity if there is
        no direct road. Rows follow node_index(). The array is read-only.
        """
        self._ensure_matrix()
        assert self._matrix_cache is not None
        return self._matrix_cache

    def _ensure_matrix(self) -> None:
        if self._matrix_cache is not None:
            return

        index = {node_id: i for i, node_id in enumerate(self.nodes)}
        size = len(index)
        matrix = np.full((size, size), np.inf)
        np.fill_diagonal(matrix, 0.0)

        for edge in self._edges:
            i, j = index[edge.source], index[edge.target]
            matrix[i, j] = min(matrix[i, j], edge.weight)
            if not edge.directed:
                matrix[j, i] = min(matrix[j, i], edge.weight)

        matrix.flags.writeable = False
        self._matrix_cache = matrix
        self._index_cache = index

    def _invalidate_matrix(self) -> None:
        self._matrix_cache = None
        self._index_cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_adjacency(self) -> None:
        """Recreate the adjacency lists from the edge list."""
        self._adj = {node_id: [] for node_id in self.nodes}
        for edge in self._edges:
            self._adj[edge.source].append((edge.target, edge.weight))
            if not edge.directed:
                self._adj[edge.target].append((edge.source, edge.weight))

    @staticmethod
    def _check_coordinates(x: float, y: float) -> None:
        for value in (x, y):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Node coordinates must be finite numbers.")