import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from ..models.graph import RoadGraph
from .base_solver import BaseSolver, SolverState
from .dijkstra_solver import DijkstraSolver
from .mutation_operators import apply_mutation, MutationOperatorType
from .simulated_annealing import (
    SAResult,
    expand_full_route_and_fleet,
    nearest_neighbor_sequence,
    precompute_shortest_paths,
)


@dataclass
class TLBOResult(SAResult):
    """Extended RouteResult with Teaching-Learning-Based Optimization (TLBO) telemetry."""
    pass


def order_crossover_waypoints(parent_a: List[str], parent_b: List[str]) -> List[str]:
    """
    Order Crossover (OX1) on full route sequences [start_node, ...waypoints..., end_node],
    preserving fixed start and end nodes and permuting intermediate waypoints without duplicates.
    """
    if len(parent_a) <= 3 or len(parent_a) != len(parent_b):
        return list(parent_a)

    inner_a = parent_a[1:-1]
    inner_b = parent_b[1:-1]
    n = len(inner_a)
    i, j = sorted(random.sample(range(n), 2))

    slice_a = inner_a[i : j + 1]
    slice_set = set(slice_a)
    remaining = [gene for gene in inner_b if gene not in slice_set]

    child_inner = remaining[:i] + slice_a + remaining[i:]
    return [parent_a[0]] + child_inner + [parent_a[-1]]


class TLBOSolver(BaseSolver):
    """
    Teaching-Learning-Based Optimization (TLBO) Meta-Heuristic Solver for VRP / TSP.
    Operates in two parameter-free phases per generation (Rao et al.):
    1. Teacher Phase: Learners move toward the best individual (Teacher) relative to the population mean.
    2. Learner Phase: Learners interact pairwise and learn from superior peers.
    """

    def __init__(
        self,
        population_size: int = 8,
        num_generations: int = 8,
        mutation_operator: Union[str, MutationOperatorType] = MutationOperatorType.TWO_OPT,
        seed: Optional[int] = None,
    ):
        super().__init__(name="TLBO")
        self.population_size = max(4, int(population_size))
        self.num_generations = max(1, int(num_generations))
        self.mutation_operator = mutation_operator
        self.dijkstra = DijkstraSolver()

        if seed is not None:
            random.seed(seed)

    def _route_cost(self, seq: List[str], dist_matrix: Dict[Tuple[str, str], float]) -> float:
        total = 0.0
        for i in range(len(seq) - 1):
            c = dist_matrix.get((seq[i], seq[i + 1]), float("inf"))
            if math.isinf(c):
                return float("inf")
            total += c
        return total

    def _compute_mean_sequence(
        self, population: List[List[str]], start_node: str, end_node: str
    ) -> List[str]:
        """Compute the consensus (mean rank) waypoint sequence across all learners in the classroom."""
        if not population or len(population[0]) <= 2:
            return [start_node, end_node]

        waypoints = population[0][1:-1]
        pos_sums: Dict[str, float] = {w: 0.0 for w in waypoints}
        for ind in population:
            for idx, node_id in enumerate(ind[1:-1]):
                pos_sums[node_id] = pos_sums.get(node_id, 0.0) + idx

        sorted_wps = sorted(waypoints, key=lambda w: pos_sums.get(w, 0.0))
        return [start_node] + sorted_wps + [end_node]

    def solve(
        self,
        graph: RoadGraph,
        fleet: Optional[Union[str, Any]] = None,
        job_queue: Optional[Union[str, Any]] = None,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **kwargs,
    ) -> TLBOResult:
        """Execute TLBO optimization over delivery waypoints."""
        self.set_state(SolverState.RUNNING)
        self.notify_progress(0.0, "Starting TLBO optimization")
        start_time = time.perf_counter()

        if start_node is None:
            if isinstance(fleet, str):
                start_node = fleet
            elif fleet is not None:
                start_node = getattr(fleet, "start_depot", getattr(fleet, "start_node", None))
            if start_node is None:
                start_node = kwargs.get("start_node")

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
            self.notify_progress(100.0, "Completed TLBO via Dijkstra fallback")
            self.set_state(SolverState.COMPLETED)
            return TLBOResult(
                path=dijkstra_res.path,
                routes=dijkstra_res.routes,
                total_cost=dijkstra_res.total_cost,
                total_distance=dijkstra_res.total_cost,
                trajectory=dijkstra_res.trajectory,
                nodes_visited=dijkstra_res.nodes_visited,
                execution_time_ms=elapsed_ms,
                algorithm_name=self.name,
                found=dijkstra_res.found,
                convergence_history=[(0, dijkstra_res.total_cost, dijkstra_res.total_cost)],
                temperature_steps=1,
            )

        relevant_nodes = list(dict.fromkeys([start_node] + valid_waypoints + [end_node]))
        fleet_vehicles = getattr(fleet, "vehicles", None) if fleet is not None else None
        if fleet_vehicles:
            for veh in fleet_vehicles:
                if getattr(veh, "home_depot", None) and graph.has_node(veh.home_depot):
                    if veh.home_depot not in relevant_nodes:
                        relevant_nodes.append(veh.home_depot)

        dist_matrix, path_matrix = precompute_shortest_paths(graph, relevant_nodes)

        pop_size = max(4, int(kwargs.get("population_size", self.population_size)))
        generations = max(1, int(kwargs.get("num_generations", kwargs.get("iterations_per_temp", self.num_generations))))
        mutation_op = kwargs.get("mutation_operator", self.mutation_operator)

        # Initialize classroom population
        population: List[List[str]] = [list(initial_sequence)]
        nn_seq = nearest_neighbor_sequence(start_node, valid_waypoints, end_node, dist_matrix)
        for _ in range(pop_size - 1):
            shuffled = list(valid_waypoints)
            random.shuffle(shuffled)
            population.append([start_node] + shuffled + [end_node])

        costs: List[float] = [self._route_cost(ind, dist_matrix) for ind in population]
        best_idx = min(range(len(population)), key=lambda idx: costs[idx])
        best_solution = list(population[best_idx])
        best_cost = costs[best_idx]

        convergence_history: List[Tuple[int, float, float]] = [
            (0, sum(c for c in costs if math.isfinite(c)) / max(1, len(costs)), best_cost)
        ]
        evaluations = len(population)
        stagnant_gens = 0

        for gen in range(1, generations + 1):
            if self.state == SolverState.STOPPED:
                break

            improved_in_gen = False
            # Seed teacher with Nearest-Neighbor on generation 1 if better
            if gen == 1:
                nn_cost = self._route_cost(nn_seq, dist_matrix)
                evaluations += 1
                if nn_cost < best_cost - 1e-9:
                    best_cost = nn_cost
                    best_solution = list(nn_seq)
                    worst_i = max(range(pop_size), key=lambda idx: costs[idx])
                    population[worst_i] = list(nn_seq)
                    costs[worst_i] = nn_cost
                    improved_in_gen = True

            teacher_idx = min(range(pop_size), key=lambda idx: costs[idx])
            teacher = population[teacher_idx]
            mean_seq = self._compute_mean_sequence(population, start_node, end_node)

            # 1. TEACHER PHASE
            for i in range(pop_size):
                tf = random.choice((1, 2))
                if tf == 1:
                    candidate = order_crossover_waypoints(population[i], teacher)
                else:
                    inter = order_crossover_waypoints(population[i], mean_seq)
                    candidate = order_crossover_waypoints(inter, teacher)
                candidate = apply_mutation(candidate, mutation_op)
                cand_cost = self._route_cost(candidate, dist_matrix)
                evaluations += 1
                if cand_cost < costs[i]:
                    population[i] = candidate
                    costs[i] = cand_cost
                    if cand_cost < best_cost - 1e-9:
                        best_cost = cand_cost
                        best_solution = list(candidate)
                        improved_in_gen = True

            # 2. LEARNER PHASE
            for i in range(pop_size):
                peers = [idx for idx in range(pop_size) if idx != i]
                j = random.choice(peers)
                if costs[j] < costs[i]:
                    candidate = order_crossover_waypoints(population[i], population[j])
                else:
                    candidate = order_crossover_waypoints(population[j], population[i])
                candidate = apply_mutation(candidate, mutation_op)
                cand_cost = self._route_cost(candidate, dist_matrix)
                evaluations += 1
                if cand_cost < costs[i]:
                    population[i] = candidate
                    costs[i] = cand_cost
                    if cand_cost < best_cost - 1e-9:
                        best_cost = cand_cost
                        best_solution = list(candidate)
                        improved_in_gen = True

            finite_costs = [c for c in costs if math.isfinite(c)]
            mean_cost = sum(finite_costs) / len(finite_costs) if finite_costs else best_cost
            convergence_history.append((gen, mean_cost, best_cost))

            if improved_in_gen:
                stagnant_gens = 0
            else:
                stagnant_gens += 1
                if gen >= 4 and stagnant_gens >= 4:
                    break

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
        telemetry_data["population_size"] = pop_size
        telemetry_data["generations"] = generations

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if self.state != SolverState.STOPPED:
            self.notify_progress(100.0, "Completed TLBO optimization")
            self.set_state(SolverState.COMPLETED)

        return TLBOResult(
            path=full_path if full_path else best_solution,
            routes=routes_map,
            total_cost=best_cost,
            total_distance=best_cost,
            trajectory=full_trajectory,
            nodes_visited=evaluations,
            execution_time_ms=elapsed_ms,
            algorithm_name=self.name,
            found=not math.isinf(best_cost),
            convergence_history=convergence_history,
            temperature_steps=generations,
            telemetry=telemetry_data,
        )
