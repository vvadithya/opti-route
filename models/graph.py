import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Node:
    """Represents a node (intersection/depot/location) in the road network."""

    id: str
    x: float
    y: float
    name: str = ""

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


class RoadGraph:
    """
    Adjacency-list based Road Network Graph data structure supporting
    both directed and undirected weighted edges.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._adj: Dict[str, List[Tuple[str, float]]] = {}

    def add_node(self, node: Node) -> None:
        """Add a Node to the graph."""
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self._adj[node.id] = []

    def add_edge(self, source: str, target: str, weight: float, directed: bool = False) -> None:
        """
        Add a weighted edge between source and target nodes.
        Nodes are automatically registered if not already present.
        """
        if source not in self.nodes:
            raise KeyError(f"Source node '{source}' does not exist in graph.")
        if target not in self.nodes:
            raise KeyError(f"Target node '{target}' does not exist in graph.")

        self._adj[source].append((target, weight))
        if not directed:
            self._adj[target].append((source, weight))

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve node object by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Return list of (neighbor_node_id, edge_weight) tuples for given node_id."""
        return self._adj.get(node_id, [])

    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self.nodes

    def node_count(self) -> int:
        """Return total number of nodes in graph."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return total number of edges in graph."""
        total = sum(len(neighbors) for neighbors in self._adj.values())
        return total  # count of directed connections
