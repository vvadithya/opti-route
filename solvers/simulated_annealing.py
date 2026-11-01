import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from ..models.graph import RoadGraph
from .base_solver import BaseSolver, OptimizationResult, RouteResult, SolverState
from .dijkstra_solver import DijkstraSolver
from .mutation_operators import apply_mutation, MutationOperatorType


@dataclass
class SAResult(RouteResult):
    """Extended RouteResult with Simulated Annealing / Meta-heuristic telemetry data."""

    convergence_history: List[Tuple[int, float, float]] = field(default_factory=list)  # (iteration, temp, cost)
    temperature_steps: int = 0
    peak_memory_mb: float = 0.0


class _LazyPathMatrix(dict):
    """Reconstructs shortest-path node sequences on demand from cached integer predecessor arrays."""

    def __init__(self, idx_to_node: List[str], node_to_idx: Dict[str, int], prev_arrays: Dict[str, List[int]]):
        super().__init__()
        self._idx_to_node = idx_to_node
        self._node_to_idx = node_to_idx
        self._prev_arrays = prev_arrays

    def get(self, key: Tuple[str, str], default: Any = None) -> Any:
        if key in self:
            return super().get(key, default)
        src, dst = key
        prev = self._prev_arrays.get(src)
        dst_idx = self._node_to_idx.get(dst, -1)
        if prev is None or dst_idx < 0 or (src != dst and prev[dst_idx] == -1):
            self[key] = []
            return []
        p: List[str] = []
        curr = dst_idx
        idx_to_node = self._idx_to_node
        while curr != -1:
            p.append(idx_to_node[curr])
            curr = prev[curr]
        p.reverse()
        self[key] = p
        return p


def precompute_shortest_paths(
    graph: RoadGraph, source_nodes: Iterable[str]
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], List[str]]]:
    """
    Ultra-fast integer-indexed single-source Dijkstra precomputation for a set of waypoint nodes,
    minimizing Python dictionary allocations and tracemalloc overhead on large graphs.
    """
    dist_matrix: Dict[Tuple[str, str], float] = {}
    unique_sources = [s for s in dict.fromkeys(source_nodes) if graph.has_node(s)]
    if not unique_sources:
        return dist_matrix, {}

    idx_to_node = list(graph.nodes.keys())
    node_to_idx = {nid: i for i, nid in enumerate(idx_to_node)}
    num_v = len(idx_to_node)
    inf = float("inf")

    # Build integer-indexed adjacency table once
    adj_int: List[List[Tuple[int, float]]] = [
        [(node_to_idx[v], float(w)) for v, w in graph.get_neighbors(nid) if v in node_to_idx]
        for nid in idx_to_node
    ]

    src_indices = [(s, node_to_idx[s]) for s in unique_sources]
    target_flags = [False] * num_v
    for _, s_idx in src_indices:
        target_flags[s_idx] = True
    total_targets = len(src_indices)

    prev_arrays: Dict[str, List[int]] = {}
    heappop = heapq.heappop
    heappush = heapq.heappush

    for src_str, src_idx in src_indices:
        dists = [inf] * num_v
        prev = [-1] * num_v
        visited = [False] * num_v
        dists[src_idx] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src_idx)]
        rem = total_targets

        while pq:
            cost_u, u = heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if target_flags[u]:
                rem -= 1
                if rem == 0:
                    break
            for v, w in adj_int[u]:
                if not visited[v]:
                    nd = cost_u + w
                    if nd < dists[v]:
                        dists[v] = nd
                        prev[v] = u
                        heappush(pq, (nd, v))

        prev_arrays[src_str] = prev
        for dst_str, dst_idx in src_indices:
            dist_matrix[(src_str, dst_str)] = dists[dst_idx]

    return dist_matrix, _LazyPathMatrix(idx_to_node, node_to_idx, prev_arrays)


def nearest_neighbor_sequence(
    start_node: str,
    waypoints: List[str],
    end_node: str,
    dist_matrix: Dict[Tuple[str, str], float],
) -> List[str]:
    """Construct a fast Greedy Nearest-Neighbor initial tour over waypoints polished with a 2-opt pass."""
    remaining = list(waypoints)
    tour = [start_node]
    curr = start_node
    while remaining:
        best_next = min(
            remaining,
            key=lambda w: dist_matrix.get((curr, w), float("inf")),
        )
        tour.append(best_next)
        remaining.remove(best_next)
        curr = best_next
    tour.append(end_node)

    # Fast O(n^2) deterministic 2-opt local search polish on intermediate waypoints
    n = len(tour)
    if n > 3:
        best_c = sum(dist_matrix.get((tour[k], tour[k + 1]), float("inf")) for k in range(n - 1))
        for _pass in range(2):
            improved = False
            for i in range(1, n - 2):
                u_prev, u_curr = tour[i - 1], tour[i]
                d_before_i = dist_matrix.get((u_prev, u_curr), float("inf"))
                for j in range(i + 1, n - 1):
                    v_curr, v_next = tour[j], tour[j + 1]
                    # Quick lower-bound filter for symmetric/near-symmetric road networks
                    d_old_edges = d_before_i + dist_matrix.get((v_curr, v_next), float("inf"))
                    d_new_edges = (
                        dist_matrix.get((u_prev, v_curr), float("inf"))
                        + dist_matrix.get((u_curr, v_next), float("inf"))
                    )
                    if d_new_edges < d_old_edges - 1e-9:
                        candidate = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                        new_c = sum(dist_matrix.get((candidate[k], candidate[k + 1]), float("inf")) for k in range(n - 1))
                        if new_c < best_c - 1e-9:
                            tour = candidate
                            best_c = new_c
                            u_curr = tour[i]
                            d_before_i = dist_matrix.get((u_prev, u_curr), float("inf"))
                            improved = True
            if not improved:
                break
    return tour


def expand_full_route_and_fleet(
    solver: BaseSolver,
    graph: RoadGraph,
    best_solution: List[str],
    start_node: str,
    end_node: str,
    fleet: Optional[Any],
    jobs_list: List[Any],
    dist_matrix: Dict[Tuple[str, str], float],
    path_matrix: Dict[Tuple[str, str], List[str]],
    steps_per_edge: int = 10,
) -> Tuple[List[str], List[Tuple[float, float]], Dict[str, List[str]], Dict[str, Any]]:
    """Expand a sequence of waypoints into a full road-level path, trajectory, and fleet assignment."""
    full_path: List[str] = []
    for i in range(len(best_solution) - 1):
        u, v = best_solution[i], best_solution[i + 1]
        seg_path = path_matrix.get((u, v))
        if seg_path:
            if not full_path:
                full_path.extend(seg_path)
            else:
                full_path.extend(seg_path[1:])

    full_trajectory = solver.generate_trajectory(graph, full_path, steps_per_edge) if full_path else []

    routes_map: Dict[str, List[str]] = {}
    telemetry_data: Dict[str, Any] = {}
    fleet_vehicles = getattr(fleet, "vehicles", None) if fleet is not None else None

    if fleet_vehicles and len(fleet_vehicles) > 0:
        intermediate = best_solution[1:-1]
        num_vehicles = min(len(fleet_vehicles), max(1, len(intermediate)))
        vehicle_loads: Dict[str, float] = {}
        vehicle_fuel: Dict[str, float] = {}
        vehicle_hours: Dict[str, float] = {}

        for v_idx in range(num_vehicles):
            veh = fleet_vehicles[v_idx]
            chunk = intermediate[v_idx::num_vehicles]
            depot_node = veh.home_depot if graph.has_node(veh.home_depot) else start_node
            v_seq = [depot_node] + chunk + [end_node]
            v_full_path: List[str] = []
            v_dist = 0.0
            for s_i in range(len(v_seq) - 1):
                su, sv = v_seq[s_i], v_seq[s_i + 1]
                s_path = path_matrix.get((su, sv))
                if not s_path:
                    sub_d, sub_p = precompute_shortest_paths(graph, [su, sv])
                    s_path = sub_p.get((su, sv), [])
                    dist_matrix[(su, sv)] = sub_d.get((su, sv), float("inf"))
                if s_path:
                    if not v_full_path:
                        v_full_path.extend(s_path)
                    else:
                        v_full_path.extend(s_path[1:])
                v_dist += dist_matrix.get((su, sv), float("inf"))
            routes_map[veh.id] = v_full_path if v_full_path else v_seq
            if math.isfinite(v_dist):
                vehicle_fuel[veh.id] = round(veh.fuel_used(v_dist), 3)
                vehicle_hours[veh.id] = round(veh.travel_time_hours(v_dist), 3)

        if jobs_list:
            for j_idx, job in enumerate(jobs_list):
                veh = fleet_vehicles[j_idx % num_vehicles]
                vehicle_loads[veh.id] = vehicle_loads.get(veh.id, 0.0) + getattr(job, "weight", 0.0)

        telemetry_data = {
            "vehicle_loads": vehicle_loads,
            "vehicle_fuel": vehicle_fuel,
            "vehicle_travel_hours": vehicle_hours,
            "jobs_routed": len(jobs_list),
        }
    else:
        routes_map["Vehicle-1"] = full_path if full_path else best_solution

    return full_path, full_trajectory, routes_map, telemetry_data


class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated Annealing (SA) Meta-Heuristic Solver for Vehicle Routing Problems (VRP) / TSP.
    Supports dynamic mutation operators (2-Opt Edge Reversal, Customer Swap, Customer Insertion),
    geometric cooling schedules, precomputed all-pairs shortest path lookups, and Metropolis acceptance.
    """

    def __init__(
        self,
        initial_temp: float = 80.0,
        cooling_rate: float = 0.75,
        min_temp: float = 0.5,
        iterations_per_temp: int = 6,
        mutation_operator: Union[str, MutationOperatorType] = MutationOperatorType.RANDOM,
        seed: Optional[int] = None,
    ):
        super().__init__(name="SimulatedAnnealing")
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        self.mutation_operator = mutation_operator
        self.dijkstra = DijkstraSolver()

        if seed is not None:
            random.seed(seed)

    def calculate_route_cost(
        self, graph: RoadGraph, node_sequence: List[str], dist_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """
        Calculates total objective cost of traversing node_sequence in order.

        :param graph: RoadGraph instance.
        :param node_sequence: List of node IDs representing route order.
        :param dist_matrix: Cache of shortest path costs between pairs of nodes.
        :return: Total path cost.
        """
        if not node_sequence or len(node_sequence) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(len(node_sequence) - 1):
            u, v = node_sequence[i], node_sequence[i + 1]
            cost = dist_matrix.get((u, v))
            if cost is None:
                res = self.dijkstra.solve(graph, start_node=u, end_node=v)
                cost = res.total_cost if res.found else float("inf")
                dist_matrix[(u, v)] = cost

            if math.isinf(cost):
                return float("inf")
            total_cost += cost

        return total_cost

    def metropolis_criterion(self, delta_cost: float, temperature: float) -> bool:
        """
        Evaluates Metropolis Acceptance Criterion:
        P = 1.0 if delta_cost < 0 else exp(-delta_cost / T)
        """
        if delta_cost < 0:
            return True
        if temperature <= 0:
            return False

        probability = math.exp(-delta_cost / temperature)
        return random.random() < probability

    def _generate_basic_neighbor(self, solution: List[str]) -> List[str]:
        """Basic neighbor generator swapping two intermediate nodes in the solution route."""
        if len(solution) <= 3:
            return list(solution)

        neighbor = list(solution)
        i, j = random.sample(range(1, len(solution) - 1), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def solve(
        self,
        graph: RoadGraph,
        fleet: Optional[Union[str, Any]] = None,
        job_queue: Optional[Union[str, Any]] = None,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **kwargs,
    ) -> SAResult:
        """
        Execute Simulated Annealing optimization to visit target nodes or solve path route.
        """
        self.set_state(SolverState.RUNNING)
        self.notify_progress(0.0, "Starting Simulated Annealing search")
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
            if end_node is None:
                end_node = kwargs.get("end_node")
            if end_node is None and start_node is not None:
                end_node = start_node

        if not start_node or not graph.has_node(start_node):
            self.set_state(SolverState.FAILED)
            raise ValueError(f"Start node '{start_node}' not found in graph.")
        if not end_node or not graph.has_node(end_node):
            self.set_state(SolverState.FAILED)
            raise ValueError(f"End node '{end_node}' not found in graph.")

        waypoints: List[str] = list(kwargs.get("waypoints", []))

        # Extract waypoints from DynamicJobQueue if none were explicitly provided
        jobs_list: List[Any] = []
        if job_queue is not None and not isinstance(job_queue, str):
            all_jobs_fn = getattr(job_queue, "all_jobs", None)
            if callable(all_jobs_fn):
                jobs_list = list(all_jobs_fn())

        if not waypoints and jobs_list:
            seen_wp = set()
            for job in jobs_list:
                for loc in (getattr(job, "pickup", None), getattr(job, "dropoff", None)):
                    if loc and graph.has_node(loc) and loc not in (start_node, end_node) and loc not in seen_wp:
                        waypoints.append(loc)
                        seen_wp.add(loc)

        valid_waypoints = [w for w in waypoints if graph.has_node(w) and w not in (start_node, end_node)]
        initial_sequence = [start_node] + valid_waypoints + [end_node]

        if len(initial_sequence) <= 2:
            dijkstra_res = self.dijkstra.solve(graph, start_node=start_node, end_node=end_node, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.notify_progress(100.0, "Completed SA via Dijkstra fallback")
            self.set_state(SolverState.COMPLETED)
            return SAResult(
                path=dijkstra_res.path,
                routes=dijkstra_res.routes,
                total_cost=dijkstra_res.total_cost,
                total_distance=dijkstra_res.total_cost,
                trajectory=dijkstra_res.trajectory,
                nodes_visited=dijkstra_res.nodes_visited,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=dijkstra_res.found,
                convergence_history=[(0, self.initial_temp, dijkstra_res.total_cost)],
                temperature_steps=1,
            )

        # Precompute all-pairs shortest paths once across the active node set
        relevant_nodes = list(dict.fromkeys([start_node] + valid_waypoints + [end_node]))
        fleet_vehicles = getattr(fleet, "vehicles", None) if fleet is not None else None
        if fleet_vehicles:
            for veh in fleet_vehicles:
                if getattr(veh, "home_depot", None) and graph.has_node(veh.home_depot):
                    if veh.home_depot not in relevant_nodes:
                        relevant_nodes.append(veh.home_depot)

        dist_matrix, path_matrix = precompute_shortest_paths(graph, relevant_nodes)

        # Evaluate initial sequence
        current_solution = list(initial_sequence)
        current_cost = self.calculate_route_cost(graph, current_solution, dist_matrix)

        best_solution = list(current_solution)
        best_cost = current_cost

        # Hyper-parameters
        temp = float(kwargs.get("initial_temp", self.initial_temp))
        alpha = float(kwargs.get("cooling_rate", self.cooling_rate))
        min_temp = float(kwargs.get("min_temp", self.min_temp))
        k_iters = int(kwargs.get("iterations_per_temp", self.iterations_per_temp))
        mutation_op = kwargs.get("mutation_operator", self.mutation_operator)

        convergence_history: List[Tuple[int, float, float]] = [(0, temp, best_cost)]

        # Also evaluate Nearest-Neighbor tour on step 1 for rapid convergence
        nn_solution = nearest_neighbor_sequence(start_node, valid_waypoints, end_node, dist_matrix)
        nn_cost = self.calculate_route_cost(graph, nn_solution, dist_matrix)

        iteration_count = 0
        temp_step = 0
        stagnant_temp_steps = 0

        # Core SA Execution Loop with adaptive stagnation early-stopping
        while temp > min_temp:
            if self.state == SolverState.STOPPED:
                break
            temp_step += 1
            improved_in_step = False
            for step_i in range(k_iters):
                if self.state == SolverState.STOPPED:
                    break
                iteration_count += 1

                if iteration_count == 1 and nn_cost < current_cost:
                    neighbor_solution = nn_solution
                    neighbor_cost = nn_cost
                else:
                    neighbor_solution = apply_mutation(current_solution, mutation_op)
                    neighbor_cost = self.calculate_route_cost(graph, neighbor_solution, dist_matrix)

                delta = neighbor_cost - current_cost

                if self.metropolis_criterion(delta, temp):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

                    if current_cost < best_cost - 1e-9:
                        best_solution = list(current_solution)
                        best_cost = current_cost
                        improved_in_step = True

                convergence_history.append((iteration_count, temp, best_cost))

            if improved_in_step:
                stagnant_temp_steps = 0
            else:
                stagnant_temp_steps += 1
                if temp_step >= 5 and stagnant_temp_steps >= 5:
                    break

            temp *= alpha

        full_path, full_trajectory, routes_map, telemetry_data = expand_full_route_and_fleet(
            self,
            graph,
            best_solution,
            start_node,
            end_node,
            fleet,
            jobs_list,
            dist_matrix,
            path_matrix,
            steps_per_edge=kwargs.get("steps_per_edge", 10),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if self.state != SolverState.STOPPED:
            self.notify_progress(100.0, "Completed Simulated Annealing search")
            self.set_state(SolverState.COMPLETED)

        return SAResult(
            path=full_path if full_path else best_solution,
            routes=routes_map,
            total_cost=best_cost,
            total_distance=best_cost,
            trajectory=full_trajectory,
            nodes_visited=iteration_count,
            execution_time_ms=elapsed_ms,
            algorithm_name=self.name,
            found=not math.isinf(best_cost),
            convergence_history=convergence_history,
            temperature_steps=temp_step,
            telemetry=telemetry_data,
        )
