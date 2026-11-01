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
from .tlbo_solver import order_crossover_waypoints


@dataclass
class GAResult(SAResult):
    """Extended RouteResult with Genetic Algorithm (GA) evolutionary telemetry."""
    pass


class GeneticAlgorithmSolver(BaseSolver):
    """
    Genetic Algorithm (GA) Evolutionary Meta-Heuristic Solver for VRP / TSP.
    Features k-way Tournament Selection, Order Crossover (OX1), Elitism,
    and configurable Neighborhood Mutation (2-Opt, Swap, Insertion, Random).
    """

    def __init__(
        self,
        population_size: int = 10,
        generations: int = 10,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        elite_size: int = 2,
        tournament_size: int = 3,
        mutation_operator: Union[str, MutationOperatorType] = MutationOperatorType.RANDOM,
        seed: Optional[int] = None,
    ):
        super().__init__(name="GeneticAlgorithm")
        self.population_size = max(4, int(population_size))
        self.generations = max(1, int(generations))
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elite_size = max(1, min(self.population_size - 1, int(elite_size)))
        self.tournament_size = max(2, int(tournament_size))
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

    def _tournament_select(self, population: List[List[str]], costs: List[float]) -> List[str]:
        """Select the fittest parent from a random tournament subset."""
        k = min(len(population), self.tournament_size)
        contenders = random.sample(range(len(population)), k)
        winner = min(contenders, key=lambda idx: costs[idx])
        return population[winner]

    def solve(
        self,
        graph: RoadGraph,
        fleet: Optional[Union[str, Any]] = None,
        job_queue: Optional[Union[str, Any]] = None,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **kwargs,
    ) -> GAResult:
        """Execute Genetic Algorithm evolutionary optimization over delivery waypoints."""
        self.set_state(SolverState.RUNNING)
        self.notify_progress(0.0, "Starting Genetic Algorithm evolution")
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
            self.notify_progress(100.0, "Completed GA via Dijkstra fallback")
            self.set_state(SolverState.COMPLETED)
            return GAResult(
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
        generations = max(1, int(kwargs.get("generations", kwargs.get("iterations_per_temp", self.generations))))
        cx_rate = float(kwargs.get("crossover_rate", self.crossover_rate))
        mut_rate = float(kwargs.get("mutation_rate", self.mutation_rate))
        mutation_op = kwargs.get("mutation_operator", self.mutation_operator)
        elite_k = max(1, min(pop_size - 1, self.elite_size))

        # Initialize chromosome population
        population: List[List[str]] = [list(initial_sequence)]
        nn_seq = nearest_neighbor_sequence(start_node, valid_waypoints, end_node, dist_matrix)
        for _ in range(pop_size - 1):
            shuffled = list(valid_waypoints)
            random.shuffle(shuffled)
            population.append([start_node] + shuffled + [end_node])

        costs: List[float] = [self._route_cost(ind, dist_matrix) for ind in population]
        best_idx = min(range(pop_size), key=lambda idx: costs[idx])
        best_solution = list(population[best_idx])
        best_cost = costs[best_idx]

        convergence_history: List[Tuple[int, float, float]] = [
            (0, sum(c for c in costs if math.isfinite(c)) / max(1, len(costs)), best_cost)
        ]
        evaluations = pop_size
        stagnant_gens = 0

        for gen in range(1, generations + 1):
            if self.state == SolverState.STOPPED:
                break

            improved_in_gen = False
            # Sort indices by fitness (lowest cost first) for Elitism
            ranked_indices = sorted(range(pop_size), key=lambda idx: costs[idx])
            new_population: List[List[str]] = [list(population[idx]) for idx in ranked_indices[:elite_k]]

            if gen == 1 and len(new_population) < pop_size:
                nn_mut = apply_mutation(nn_seq, MutationOperatorType.TWO_OPT)
                new_population.append(list(nn_seq))
                if len(new_population) < pop_size:
                    new_population.append(nn_mut)

            while len(new_population) < pop_size:
                parent1 = self._tournament_select(population, costs)
                parent2 = self._tournament_select(population, costs)

                if random.random() < cx_rate:
                    child = order_crossover_waypoints(parent1, parent2)
                else:
                    child = list(parent1)

                if random.random() < mut_rate:
                    child = apply_mutation(child, mutation_op)

                new_population.append(child)

            population = new_population
            costs = [self._route_cost(ind, dist_matrix) for ind in population]
            evaluations += pop_size

            gen_best_idx = min(range(pop_size), key=lambda idx: costs[idx])
            if costs[gen_best_idx] < best_cost - 1e-9:
                best_cost = costs[gen_best_idx]
                best_solution = list(population[gen_best_idx])
                improved_in_gen = True

            finite_costs = [c for c in costs if math.isfinite(c)]
            avg_cost = sum(finite_costs) / len(finite_costs) if finite_costs else best_cost
            convergence_history.append((gen, avg_cost, best_cost))

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
        telemetry_data["crossover_rate"] = cx_rate
        telemetry_data["mutation_rate"] = mut_rate

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if self.state != SolverState.STOPPED:
            self.notify_progress(100.0, "Completed Genetic Algorithm evolution")
            self.set_state(SolverState.COMPLETED)

        return GAResult(
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
