import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from ..models.graph import RoadGraph
from .base_solver import BaseSolver, OptimizationResult, RouteResult, SolverState
from .dijkstra_solver import DijkstraSolver
from .mutation_operators import apply_mutation, MutationOperatorType


@dataclass
class SAResult(RouteResult):
    """Extended RouteResult with Simulated Annealing telemetry data."""

    convergence_history: List[Tuple[int, float, float]] = field(default_factory=list)  # (iteration, temp, cost)
    temperature_steps: int = 0
    peak_memory_mb: float = 0.0


class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated Annealing (SA) Meta-Heuristic Solver for Vehicle Routing Problems (VRP) / TSP.
    Supports dynamic mutation operators (2-Opt Edge Reversal, Customer Swap, Customer Insertion),
    geometric cooling schedules, and Metropolis acceptance criteria.
    """

    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.01,
        iterations_per_temp: int = 50,
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
            if (u, v) not in dist_matrix:
                res = self.dijkstra.solve(graph, start_node=u, end_node=v)
                dist_matrix[(u, v)] = res.total_cost if res.found else float("inf")

            cost = dist_matrix[(u, v)]
            if math.isinf(cost):
                return float("inf")
            total_cost += cost

        return total_cost

    def metropolis_criterion(self, delta_cost: float, temperature: float) -> bool:
        """
        Evaluates Metropolis Acceptance Criterion:
        P = 1.0 if delta_cost < 0 else exp(-delta_cost / T)

        :param delta_cost: Cost difference (new_cost - current_cost).
        :param temperature: Current temperature T.
        :return: True if new solution is accepted, False otherwise.
        """
        if delta_cost < 0:
            return True
        if temperature <= 0:
            return False

        probability = math.exp(-delta_cost / temperature)
        return random.random() < probability

    def _generate_basic_neighbor(self, solution: List[str]) -> List[str]:
        """
        Basic neighbor generator swapping two intermediate nodes in the solution route.
        """
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

        :param graph: RoadGraph instance.
        :param fleet: Optional vehicle fleet model or start_node ID string.
        :param job_queue: Optional job queue model or end_node ID string.
        :param start_node: Start node ID override.
        :param end_node: End node ID override.
        :param kwargs: Optional 'waypoints' (List[str]), hyper-parameters overrides.
        :return: SAResult object with optimal solution, trajectory, and convergence analytics.
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

        # Construct initial node sequence: start_node -> waypoints -> end_node
        valid_waypoints = [w for w in waypoints if graph.has_node(w) and w not in (start_node, end_node)]
        initial_sequence = [start_node] + valid_waypoints + [end_node]

        if len(initial_sequence) <= 2:
            # Fallback to exact Dijkstra for simple point-to-point without waypoints
            dijkstra_res = self.dijkstra.solve(graph, start_node=start_node, end_node=end_node, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.notify_progress(100.0, "Completed SA via Dijkstra fallback")
            self.set_state(SolverState.COMPLETED)
            return SAResult(
                path=dijkstra_res.path,
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

        # Distance matrix cache to speed up repeated objective function calls
        dist_matrix: Dict[Tuple[str, str], float] = {}

        # Initial solution setup
        current_solution = list(initial_sequence)
        current_cost = self.calculate_route_cost(graph, current_solution, dist_matrix)

        best_solution = list(current_solution)
        best_cost = current_cost

        # Hyper-parameters
        temp = kwargs.get("initial_temp", self.initial_temp)
        alpha = kwargs.get("cooling_rate", self.cooling_rate)
        min_temp = kwargs.get("min_temp", self.min_temp)
        k_iters = kwargs.get("iterations_per_temp", self.iterations_per_temp)
        mutation_op = kwargs.get("mutation_operator", self.mutation_operator)

        convergence_history: List[Tuple[int, float, float]] = []
        iteration_count = 0
        temp_step = 0

        # Core SA Execution Loop
        while temp > min_temp:
            if self.state == SolverState.STOPPED:
                break
            temp_step += 1
            for _ in range(k_iters):
                if self.state == SolverState.STOPPED:
                    break
                iteration_count += 1

                # Generate neighbor solution using dynamic mutation operator strategy
                neighbor_solution = apply_mutation(current_solution, mutation_op)
                neighbor_cost = self.calculate_route_cost(graph, neighbor_solution, dist_matrix)

                delta = neighbor_cost - current_cost

                # Evaluate Metropolis Acceptance Criterion
                if self.metropolis_criterion(delta, temp):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best_solution = list(current_solution)
                        best_cost = current_cost

                convergence_history.append((iteration_count, temp, best_cost))

            # Geometric Temperature Decay
            temp *= alpha

        # Build detailed trajectory for final best route using Dijkstra segment trajectories
        full_path: List[str] = []
        full_trajectory: List[Tuple[float, float]] = []

        for i in range(len(best_solution) - 1):
            seg_start, seg_end = best_solution[i], best_solution[i + 1]
            seg_res = self.dijkstra.solve(graph, start_node=seg_start, end_node=seg_end)
            if seg_res.found:
                if not full_path:
                    full_path.extend(seg_res.path)
                else:
                    full_path.extend(seg_res.path[1:])  # avoid duplicate node join
                full_trajectory.extend(seg_res.trajectory)

        # Build per-vehicle routes and capacity/job telemetry if fleet/jobs exist
        routes_map: Dict[str, List[str]] = {}
        telemetry_data: Dict[str, Any] = {}
        fleet_vehicles = getattr(fleet, "vehicles", None) if fleet is not None else None

        if fleet_vehicles and len(fleet_vehicles) > 0:
            intermediate = best_solution[1:-1]
            num_vehicles = min(len(fleet_vehicles), max(1, len(intermediate)))
            vehicle_Loads: Dict[str, float] = {}
            vehicle_fuel: Dict[str, float] = {}
            vehicle_hours: Dict[str, float] = {}

            # Partition intermediate stops across fleet vehicles
            for v_idx in range(num_vehicles):
                veh = fleet_vehicles[v_idx]
                chunk = intermediate[v_idx::num_vehicles]
                v_seq = [veh.home_depot if graph.has_node(veh.home_depot) else start_node] + chunk + [end_node]
                v_full_path: List[str] = []
                for s_i in range(len(v_seq) - 1):
                    s_res = self.dijkstra.solve(graph, start_node=v_seq[s_i], end_node=v_seq[s_i + 1])
                    if s_res.found:
                        if not v_full_path:
                            v_full_path.extend(s_res.path)
                        else:
                            v_full_path.extend(s_res.path[1:])
                routes_map[veh.id] = v_full_path if v_full_path else v_seq
                v_dist = self.calculate_route_cost(graph, v_seq, dist_matrix)
                if math.isfinite(v_dist):
                    vehicle_fuel[veh.id] = round(veh.fuel_used(v_dist), 3)
                    vehicle_hours[veh.id] = round(veh.travel_time_hours(v_dist), 3)

            if jobs_list:
                for j_idx, job in enumerate(jobs_list):
                    veh = fleet_vehicles[j_idx % num_vehicles]
                    vehicle_Loads[veh.id] = vehicle_Loads.get(veh.id, 0.0) + getattr(job, "weight", 0.0)

            telemetry_data = {
                "vehicle_loads": vehicle_Loads,
                "vehicle_fuel": vehicle_fuel,
                "vehicle_travel_hours": vehicle_hours,
                "jobs_routed": len(jobs_list),
            }
        else:
            routes_map["Vehicle-1"] = full_path if full_path else best_solution

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
