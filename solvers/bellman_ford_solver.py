import time
from typing import Any, Dict, List, Optional, Tuple, Union
from ..models.graph import RoadGraph
from .base_solver import BaseSolver, OptimizationResult, RouteResult, SolverState


class BellmanFordSolver(BaseSolver):
    """
    Exact single-source shortest path solver using the Bellman-Ford Algorithm.
    Supports dynamic edge relaxation across |V| - 1 passes with early stopping
    and negative-weight cycle detection.
    """

    def __init__(self, steps_per_edge: int = 10):
        super().__init__(name="BellmanFord")
        self.steps_per_edge = steps_per_edge

    def solve(
        self,
        graph: RoadGraph,
        fleet: Optional[Union[str, Any]] = None,
        job_queue: Optional[Union[str, Any]] = None,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Computes exact shortest path between start_node and end_node using Bellman-Ford relaxation.

        :param graph: RoadGraph instance.
        :param fleet: Optional vehicle fleet model or start_node ID string.
        :param job_queue: Optional job queue model or end_node ID string.
        :param start_node: Optional start location ID string.
        :param end_node: Optional target location ID string.
        :param kwargs: Optional steps_per_edge or solver parameters.
        :return: RouteResult containing path, total cost, trajectory, and relaxation telemetry.
        """
        self.set_state(SolverState.RUNNING)
        self.notify_progress(0.0, "Starting Bellman-Ford search")
        start_time = time.perf_counter()

        # Resolve start_node
        if start_node is None:
            if isinstance(fleet, str):
                start_node = fleet
            elif fleet is not None:
                start_node = getattr(fleet, "start_depot", getattr(fleet, "start_node", None))
            if start_node is None:
                start_node = kwargs.get("start_node")

        # Resolve end_node
        if end_node is None:
            if isinstance(job_queue, str):
                end_node = job_queue
            elif job_queue is not None:
                end_node = getattr(job_queue, "end_node", None)
                if end_node is None and callable(getattr(job_queue, "peek", None)):
                    first_job = job_queue.peek()
                    if first_job is not None:
                        end_node = getattr(first_job, "dropoff", None)
            if end_node is None:
                end_node = kwargs.get("end_node")

        if not start_node or not graph.has_node(start_node):
            self.set_state(SolverState.FAILED)
            raise ValueError(f"Start node '{start_node}' not found in RoadGraph.")
        if not end_node or not graph.has_node(end_node):
            self.set_state(SolverState.FAILED)
            raise ValueError(f"End node '{end_node}' not found in RoadGraph.")

        steps = kwargs.get("steps_per_edge", self.steps_per_edge)

        if start_node == end_node:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            trajectory = self.generate_trajectory(graph, [start_node], steps)
            self.notify_progress(100.0, "Completed Bellman-Ford search")
            self.set_state(SolverState.COMPLETED)
            return RouteResult(
                path=[start_node],
                routes={"Vehicle-1": [start_node]},
                total_cost=0.0,
                total_distance=0.0,
                trajectory=trajectory,
                nodes_visited=1,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=True,
                telemetry={"relaxation_passes": 0, "edges_relaxed": 0},
            )

        node_ids = list(graph.nodes.keys())
        num_vertices = len(node_ids)

        # Collect all directed transitions (u, v, weight) from the adjacency list
        directed_edges: List[Tuple[str, str, float]] = []
        for u in node_ids:
            for v, weight in graph.get_neighbors(u):
                directed_edges.append((u, v, float(weight)))

        distances: Dict[str, float] = {nid: float("inf") for nid in node_ids}
        predecessor: Dict[str, Optional[str]] = {nid: None for nid in node_ids}
        distances[start_node] = 0.0

        relaxation_passes = 0
        edges_relaxed = 0

        # Perform up to |V| - 1 relaxation passes with early termination
        for _ in range(max(0, num_vertices - 1)):
            if self.state == SolverState.STOPPED:
                break
            relaxation_passes += 1
            updated_in_pass = False
            for u, v, weight in directed_edges:
                dist_u = distances[u]
                if dist_u != float("inf") and dist_u + weight < distances[v]:
                    distances[v] = dist_u + weight
                    predecessor[v] = u
                    edges_relaxed += 1
                    updated_in_pass = True
            if not updated_in_pass:
                break

        # Check for negative-weight cycles reachable from start_node
        for u, v, weight in directed_edges:
            dist_u = distances[u]
            if dist_u != float("inf") and dist_u + weight < distances[v] - 1e-9:
                self.set_state(SolverState.FAILED)
                raise ValueError("Negative-weight cycle detected in RoadGraph.")

        reachable_count = sum(1 for d in distances.values() if d != float("inf"))

        if distances.get(end_node, float("inf")) == float("inf"):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.notify_progress(100.0, "Path not found")
            self.set_state(SolverState.COMPLETED)
            return RouteResult(
                path=[],
                total_cost=float("inf"),
                total_distance=float("inf"),
                trajectory=[],
                nodes_visited=reachable_count,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=False,
                telemetry={"relaxation_passes": relaxation_passes, "edges_relaxed": edges_relaxed},
            )

        # Reconstruct shortest path
        path: List[str] = []
        curr: Optional[str] = end_node
        visited_guard = set()
        while curr is not None and curr not in visited_guard:
            visited_guard.add(curr)
            path.append(curr)
            curr = predecessor.get(curr)
        path.reverse()

        total_cost = distances[end_node]
        trajectory = self.generate_trajectory(graph, path, steps)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        routes_map: Dict[str, List[str]] = {}
        fleet_vehicles = getattr(fleet, "vehicles", None) if fleet is not None else None
        if fleet_vehicles and len(fleet_vehicles) > 0:
            routes_map[fleet_vehicles[0].id] = list(path)
        else:
            routes_map["Vehicle-1"] = list(path)

        self.notify_progress(100.0, "Completed Bellman-Ford search")
        self.set_state(SolverState.COMPLETED)
        return RouteResult(
            path=path,
            routes=routes_map,
            total_cost=total_cost,
            total_distance=total_cost,
            trajectory=trajectory,
            nodes_visited=reachable_count,
            execution_time_ms=elapsed_ms,
            algorithm_name=self.name,
            found=True,
            telemetry={"relaxation_passes": relaxation_passes, "edges_relaxed": edges_relaxed},
        )
