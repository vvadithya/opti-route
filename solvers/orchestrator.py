import time
import tracemalloc
from typing import Any, Dict, Optional, Type
from ..models.graph import RoadGraph
from .base_solver import AlgorithmStrategy, OptimizationResult, RouteResult
from .dijkstra_solver import DijkstraSolver
from .simulated_annealing import SimulatedAnnealingSolver, SAResult


class SolverOrchestrator:
    """
    Execution Orchestrator responsible for managing strategy instantiation,
    executing pathfinding/VRP solvers, tracking runtime memory and latency metrics,
    and returning formatted solutions for the MVC Controller and Benchmarking Engine.
    """

    def __init__(self):
        self._solvers: Dict[str, Type[AlgorithmStrategy]] = {
            "dijkstra": DijkstraSolver,
            "simulated_annealing": SimulatedAnnealingSolver,
            "sa": SimulatedAnnealingSolver,
        }

    def register_solver(self, name: str, solver_cls: Type[AlgorithmStrategy]) -> None:
        """Register a new solver strategy dynamically."""
        self._solvers[name.lower()] = solver_cls

    def create_solver(self, algorithm_name: str, **kwargs) -> AlgorithmStrategy:
        """Factory method to instantiate a solver strategy by name."""
        key = algorithm_name.lower()
        if key not in self._solvers:
            raise KeyError(
                f"Unknown solver '{algorithm_name}'. Available solvers: {list(self._solvers.keys())}"
            )

        solver_cls = self._solvers[key]

        # Extract keyword arguments relevant to solver __init__
        if issubclass(solver_cls, SimulatedAnnealingSolver):
            init_args = {
                k: v
                for k, v in kwargs.items()
                if k in ("initial_temp", "cooling_rate", "min_temp", "iterations_per_temp", "mutation_operator", "seed")
            }
            return solver_cls(**init_args)
        elif issubclass(solver_cls, DijkstraSolver):
            init_args = {k: v for k, v in kwargs.items() if k in ("steps_per_edge",)}
            return solver_cls(**init_args)

        return solver_cls()

    def execute(
        self,
        algorithm_name: str,
        graph: RoadGraph,
        fleet: Optional[Any] = None,
        job_queue: Optional[Any] = None,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Executes the specified solver, measuring latency and memory usage via tracemalloc.

        :param algorithm_name: Registered algorithm identifier ("dijkstra", "sa", "simulated_annealing").
        :param graph: RoadGraph instance.
        :param fleet: Optional vehicle fleet model or start_node ID string.
        :param job_queue: Optional job queue model or end_node ID string.
        :param start_node: Optional start node ID.
        :param end_node: Optional target node ID.
        :param kwargs: Additional hyper-parameters, waypoints, etc.
        :return: Enriched OptimizationResult containing cost, trajectory, and memory benchmarks.
        """
        solver = self.create_solver(algorithm_name, **kwargs)

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            result = solver.solve(
                graph,
                fleet=fleet,
                job_queue=job_queue,
                start_node=start_node,
                end_node=end_node,
                **kwargs,
            )
        finally:
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        peak_mb = peak_mem / (1024.0 * 1024.0)

        # Enrich result metadata
        result.execution_time_ms = elapsed_ms
        setattr(result, "peak_memory_mb", peak_mb)

        return result

    def format_benchmark_payload(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        Formats execution output into a clean dictionary payload for the MVC Controller
        and Meghana's Benchmarking Engine / Matplotlib Analytics Widget.
        """
        payload = {
            "algorithm": result.algorithm_name,
            "success": result.found,
            "total_cost": round(result.total_cost, 4) if result.found else float("inf"),
            "total_distance": round(getattr(result, "total_distance", result.total_cost), 4) if result.found else float("inf"),
            "path": result.path,
            "path_length": len(result.path),
            "trajectory_points": len(result.trajectory),
            "execution_time_ms": round(result.execution_time_ms, 3),
            "nodes_visited": result.nodes_visited,
            "peak_memory_mb": round(getattr(result, "peak_memory_mb", 0.0), 4),
        }

        if isinstance(result, SAResult):
            payload["convergence_history"] = result.convergence_history
            payload["temperature_steps"] = result.temperature_steps

        return payload
