import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from ..models.graph import RoadGraph
from .base_solver import BaseSolver, OptimizationResult, RouteResult, SolverState


class DijkstraSolver(BaseSolver):
    """
    Exact point-to-point shortest path solver using Dijkstra's Algorithm
    with binary min-heap priority queue.
    """

    def __init__(self, steps_per_edge: int = 10):
        super().__init__(name="Dijkstra")
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
        Computes exact shortest path and cost between start_node and end_node in graph.

        :param graph: RoadGraph instance.
        :param fleet: Optional vehicle fleet model or start_node ID string.
        :param job_queue: Optional job queue model or end_node ID string.
        :param start_node: Optional start location ID string.
        :param end_node: Optional target location ID string.
        :param kwargs: Optional steps_per_edge or hyper-parameters.
        :return: OptimizationResult containing path, total cost, trajectory, and performance stats.
        """
        self.set_state(SolverState.RUNNING)
        self.notify_progress(0.0, "Starting Dijkstra search")
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
        raw_waypoints = kwargs.get("waypoints") or []
        valid_waypoints = [w for w in raw_waypoints if graph.has_node(w) and w not in (start_node, end_node)]

        if valid_waypoints:
            seq = [start_node] + valid_waypoints + [end_node]
            full_path: List[str] = []
            total_cost = 0.0
            total_visited = 0
            adj = {nid: graph.get_neighbors(nid) for nid in graph.nodes}

            for i in range(len(seq) - 1):
                u_start, u_end = seq[i], seq[i + 1]
                if u_start == u_end:
                    if not full_path:
                        full_path.append(u_start)
                    continue
                pq_seg: List[Tuple[float, str]] = [(0.0, u_start)]
                dists_seg: Dict[str, float] = {u_start: 0.0}
                prev_seg: Dict[str, Optional[str]] = {u_start: None}
                vis_seg: Set[str] = set()
                while pq_seg:
                    c_u, curr_u = heapq.heappop(pq_seg)
                    if curr_u in vis_seg:
                        continue
                    vis_seg.add(curr_u)
                    total_visited += 1
                    if curr_u == u_end:
                        break
                    for nb, w in adj.get(curr_u, ()):
                        if nb not in vis_seg:
                            nc = c_u + w
                            if nc < dists_seg.get(nb, float("inf")):
                                dists_seg[nb] = nc
                                prev_seg[nb] = curr_u
                                heapq.heappush(pq_seg, (nc, nb))
                if u_end not in dists_seg:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.set_state(SolverState.COMPLETED)
                    return RouteResult(
                        path=[],
                        total_cost=float("inf"),
                        total_distance=float("inf"),
                        trajectory=[],
                        nodes_visited=total_visited,
                        execution_time_ms=elapsed_ms,
                        algorithm_name=self.name,
                        found=False,
                    )
                total_cost += dists_seg[u_end]
                seg_p: List[str] = []
                cur_n: Optional[str] = u_end
                while cur_n is not None:
                    seg_p.append(cur_n)
                    cur_n = prev_seg.get(cur_n)
                seg_p.reverse()
                if not full_path:
                    full_path.extend(seg_p)
                else:
                    full_path.extend(seg_p[1:])

            trajectory = self.generate_trajectory(graph, full_path, steps)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            routes_map: Dict[str, List[str]] = {}
            fleet_vehicles = getattr(fleet, "vehicles", None) if fleet is not None else None
            if fleet_vehicles and len(fleet_vehicles) > 0:
                routes_map[fleet_vehicles[0].id] = list(full_path)
            else:
                routes_map["Vehicle-1"] = list(full_path)

            self.notify_progress(100.0, "Completed Sequential Dijkstra search")
            self.set_state(SolverState.COMPLETED)
            return RouteResult(
                path=full_path,
                routes=routes_map,
                total_cost=total_cost,
                total_distance=total_cost,
                trajectory=trajectory,
                nodes_visited=total_visited,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=True,
            )

        # Special case: Start is destination
        if start_node == end_node:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            trajectory = self.generate_trajectory(graph, [start_node], steps)
            self.notify_progress(100.0, "Completed Dijkstra search")
            self.set_state(SolverState.COMPLETED)
            return RouteResult(
                path=[start_node],
                total_cost=0.0,
                total_distance=0.0,
                trajectory=trajectory,
                nodes_visited=1,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=True,
            )

        # Dijkstra priority queue initialization
        pq: List[Tuple[float, str]] = [(0.0, start_node)]
        distances: Dict[str, float] = {start_node: 0.0}
        previous: Dict[str, Optional[str]] = {start_node: None}
        visited: Set[str] = set()

        nodes_visited_count = 0

        while pq:
            curr_cost, curr_node = heapq.heappop(pq)

            if curr_node in visited:
                continue

            visited.add(curr_node)
            nodes_visited_count += 1

            # Early termination if destination node reached
            if curr_node == end_node:
                break

            for neighbor, weight in graph.get_neighbors(curr_node):
                if neighbor in visited:
                    continue

                new_cost = curr_cost + weight
                if new_cost < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_cost
                    previous[neighbor] = curr_node
                    heapq.heappush(pq, (new_cost, neighbor))

        # Check path existence
        if end_node not in distances:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.notify_progress(100.0, "Path not found")
            self.set_state(SolverState.COMPLETED)
            return RouteResult(
                path=[],
                total_cost=float("inf"),
                total_distance=float("inf"),
                trajectory=[],
                nodes_visited=nodes_visited_count,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=False,
            )

        # Reconstruct path
        path: List[str] = []
        curr: Optional[str] = end_node
        while curr is not None:
            path.append(curr)
            curr = previous.get(curr)

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

        self.notify_progress(100.0, "Completed Dijkstra search")
        self.set_state(SolverState.COMPLETED)
        return RouteResult(
            path=path,
            routes=routes_map,
            total_cost=total_cost,
            total_distance=total_cost,
            trajectory=trajectory,
            nodes_visited=nodes_visited_count,
            execution_time_ms=elapsed_ms,
            algorithm_name=self.name,
            found=True,
        )
