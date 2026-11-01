"""
OptiRouteController: MVC Controller integrating models, abstract solvers,
file I/O (GraphFactory), benchmarking (Phase 4), and execution pipeline hooks
for Phase 2 algorithms (Dijkstra and Simulated Annealing).
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ..models.graph import RoadGraph
from ..models.graph_factory import GraphFactory, ProjectData
from ..models.job import DynamicJobQueue
from ..models.vehicle import VehicleFleet
from ..solvers.base_solver import (
    AlgorithmStrategy,
    BaseSolver,
    ExecutionParams,
    OptimizationResult,
    SolverState,
)
from ..solvers.orchestrator import SolverOrchestrator
from ..benchmarking.metrics_collector import BenchmarkResult, MetricsCollector
from ..benchmarking.algorithm_comparison import ComparisonRow
from ..benchmarking.convergence_plot import create_convergence_plot
from ..benchmarking.report_generator import export_csv, export_pdf


class OptiRouteController:
    """
    MVC Controller managing references to road graph models, vehicle fleet state,
    job queues, file parsers, benchmarking collectors, and optimization solvers.
    Provides pipeline hooks to configure and run algorithms, manage execution state,
    and deliver results to UI and Analytics.
    """

    def __init__(
        self,
        graph: Optional[RoadGraph] = None,
        fleet: Optional[Any] = None,
        job_queue: Optional[Any] = None,
    ):
        self.graph: RoadGraph = graph if graph is not None else RoadGraph()
        self.fleet: Any = fleet if fleet is not None else VehicleFleet()
        self.job_queue: Any = job_queue if job_queue is not None else DynamicJobQueue()
        self.orchestrator: SolverOrchestrator = SolverOrchestrator()
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self._run_counter: int = 0
        self.current_params: ExecutionParams = ExecutionParams()
        self.last_result: Optional[OptimizationResult] = None
        self.last_benchmark: Optional[BenchmarkResult] = None
        self.active_solver: Optional[AlgorithmStrategy] = None
        self._listeners: Dict[str, List[Callable[..., None]]] = {}

    def set_graph(self, graph: RoadGraph) -> None:
        """Set or update the RoadGraph model instance."""
        self.graph = graph
        self.notify_listeners("graph_updated", self.graph)

    def set_fleet(self, fleet: Any) -> None:
        """Set or update the VehicleFleet model instance."""
        self.fleet = fleet
        self.notify_listeners("fleet_updated", self.fleet)

    def set_job_queue(self, job_queue: Any) -> None:
        """Set or update the JobQueue model instance."""
        self.job_queue = job_queue
        self.notify_listeners("job_queue_updated", self.job_queue)

    def register_solver(self, name: str, solver_cls: Type[AlgorithmStrategy]) -> None:
        """Register a new solver strategy with the underlying orchestrator."""
        self.orchestrator.register_solver(name, solver_cls)

    def configure_execution(self, params: ExecutionParams) -> None:
        """Update solver execution hyper-parameters."""
        self.current_params = params
        self.notify_listeners("params_updated", self.current_params)

    def register_listener(self, event_name: str, callback: Callable[..., None]) -> None:
        """Register an observer callback for controller events."""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)

    def notify_listeners(self, event_name: str, *args, **kwargs) -> None:
        """Notify registered observer callbacks of an event."""
        for listener in self._listeners.get(event_name, []):
            listener(*args, **kwargs)

    def run_solver(
        self,
        algorithm_name: str,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Execute specified optimization algorithm pipeline using current model references and parameters.

        :param algorithm_name: Identifier for solver ('dijkstra', 'simulated_annealing', etc.).
        :param start_node: Optional start node ID override.
        :param end_node: Optional end node ID override.
        :param kwargs: Hyper-parameters or additional solver arguments.
        :return: OptimizationResult containing route solution, cost, trajectory, and benchmarks.
        """
        self.notify_listeners("execution_started", algorithm_name)

        # Build merged keyword parameters
        merged_kwargs = {
            "initial_temp": self.current_params.initial_temp,
            "cooling_rate": self.current_params.cooling_rate,
            "min_temp": self.current_params.min_temp,
            "iterations_per_temp": self.current_params.iterations_per_temp,
            "mutation_operator": self.current_params.mutation_operator,
            "steps_per_edge": self.current_params.steps_per_edge,
            "seed": self.current_params.seed,
            "waypoints": list(self.current_params.waypoints),
        }
        if self.current_params.custom_params:
            merged_kwargs.update(self.current_params.custom_params)
        merged_kwargs.update(kwargs)

        # Resolve start and end nodes
        s_node = start_node or self.current_params.start_node or "Depot"
        e_node = end_node or self.current_params.end_node or "Depot"

        try:
            solver_inst = self.orchestrator.create_solver(algorithm_name, **merged_kwargs)
            self.active_solver = solver_inst

            # Register progress signal hook on active solver if it supports signal hooks
            if isinstance(solver_inst, BaseSolver):
                solver_inst.register_signal_hook(
                    "progress", lambda pct, msg: self.notify_listeners("progress", pct, msg)
                )

            result = self.orchestrator.execute(
                algorithm_name,
                self.graph,
                fleet=self.fleet,
                job_queue=self.job_queue,
                start_node=s_node,
                end_node=e_node,
                **merged_kwargs,
            )
            self.last_result = result
            self._record_benchmark(result, s_node, e_node, **merged_kwargs)
            self.notify_listeners("execution_completed", result)
            return result
        except Exception as e:
            self.notify_listeners("execution_failed", str(e))
            raise e

    def _record_benchmark(
        self,
        result: OptimizationResult,
        start_node: str,
        end_node: str,
        **kwargs,
    ) -> Optional[BenchmarkResult]:
        """Convert solver result into a Phase 4 BenchmarkResult and store in MetricsCollector."""
        try:
            self._run_counter += 1
            alg_name = result.algorithm_name or "Solver"
            run_id = kwargs.get("run_id") or f"{alg_name.lower()}-run-{self._run_counter}"
            comp_group = kwargs.get("comparison_group") or f"{start_node}->{end_node}"
            dist_unit = getattr(self.graph, "distance_unit", "km") or "km"

            found_val = bool(result.found)
            raw_cost = getattr(result, "total_cost", None)
            raw_dist = getattr(result, "total_distance", raw_cost)
            finite_cost = float(raw_cost) if (found_val and raw_cost is not None and math.isfinite(raw_cost)) else None
            finite_dist = float(raw_dist) if (found_val and raw_dist is not None and math.isfinite(raw_dist)) else None
            dist_km = finite_dist if dist_unit.lower() in ("km", "kilometer", "kilometre", "kilometers", "kilometres") else None

            raw_history = getattr(result, "convergence_history", ()) or ()
            norm_history: List[Tuple[int, float]] = []
            for pt in raw_history:
                if len(pt) == 3:
                    it, _, c_val = pt
                elif len(pt) == 2:
                    it, c_val = pt
                else:
                    continue
                if isinstance(c_val, (int, float)) and math.isfinite(c_val):
                    norm_history.append((int(it), float(c_val)))

            iter_count: Optional[int] = None
            if hasattr(result, "convergence_history"):
                iter_count = max((p[0] for p in norm_history), default=0)

            peak_mem = getattr(result, "peak_memory_mb", None)
            if peak_mem is not None and (not math.isfinite(peak_mem) or peak_mem < 0):
                peak_mem = None

            bench_res = BenchmarkResult(
                algorithm_name=alg_name,
                run_id=str(run_id),
                execution_time_ms=max(0.0, float(result.execution_time_ms)),
                peak_traced_memory_mb=peak_mem,
                route_distance=finite_dist,
                route_distance_unit=dist_unit,
                route_distance_km=dist_km,
                total_solution_cost=finite_cost,
                iteration_count=iter_count,
                nodes_visited=max(0, int(result.nodes_visited)) if result.nodes_visited is not None else None,
                temperature_steps=getattr(result, "temperature_steps", None),
                found=found_val,
                comparison_group=str(comp_group),
                convergence_history=tuple(norm_history),
            )
            self.metrics_collector.add(bench_res)
            self.last_benchmark = bench_res
            self.notify_listeners("benchmark_recorded", bench_res)
            return bench_res
        except Exception:
            return None

    def load_project(self, path: Union[str, Path]) -> ProjectData:
        """
        Load graph, vehicle fleet, and job queue from a JSON or CSV file via GraphFactory (SRS F1, U2).
        """
        project = GraphFactory.load(path)
        self.set_graph(project.graph)
        self.set_fleet(project.fleet)
        self.set_job_queue(project.jobs)
        self.notify_listeners("project_loaded", project)
        return project

    def save_project(self, path: Union[str, Path]) -> Path:
        """
        Save active graph, vehicle fleet, and job queue to a JSON or CSV file via GraphFactory (SRS F1, U2).
        """
        fleet_obj = self.fleet if isinstance(self.fleet, VehicleFleet) else None
        jobs_obj = self.job_queue if isinstance(self.job_queue, DynamicJobQueue) else None
        saved_path = GraphFactory.save(path, self.graph, fleet_obj, jobs_obj)
        self.notify_listeners("project_saved", saved_path)
        return saved_path

    def get_benchmark_results(self) -> Tuple[BenchmarkResult, ...]:
        """Return all collected Phase 4 BenchmarkResult records."""
        return self.metrics_collector.results

    def get_comparison_rows(self) -> Tuple[ComparisonRow, ...]:
        """Return side-by-side AlgorithmComparison rows across recorded runs (SRS F14)."""
        return self.metrics_collector.compare()

    def get_convergence_figure(self, title: str = "Simulated Annealing Convergence"):
        """Create a Matplotlib Figure for the most recent convergence history (SRS F13)."""
        history = getattr(self.last_result, "convergence_history", ()) if self.last_result else ()
        return create_convergence_plot(history, title=title)

    def export_benchmark_csv(self, path: Union[str, Path]) -> Tuple[Path, Optional[Path]]:
        """Export recorded benchmark metrics and convergence history to CSV (SRS F15)."""
        return export_csv(path, self.metrics_collector.results)

    def export_benchmark_pdf(
        self,
        path: Union[str, Path],
        configuration: Optional[Dict[str, Any]] = None,
        explanation: Optional[str] = None,
    ) -> Path:
        """Export recorded benchmark metrics, comparison table, and plots to PDF (SRS F15)."""
        config = configuration or {
            "nodes": self.graph.node_count(),
            "edges": self.graph.edge_count(),
            "distance_unit": getattr(self.graph, "distance_unit", "km"),
            "initial_temp": self.current_params.initial_temp,
            "cooling_rate": self.current_params.cooling_rate,
            "iterations_per_temp": self.current_params.iterations_per_temp,
        }
        return export_pdf(path, self.metrics_collector.results, configuration=config, explanation=explanation)

    def run_dijkstra(
        self, start_node: str, end_node: str, **kwargs
    ) -> OptimizationResult:
        """Pipeline hook for executing Dijkstra's shortest path algorithm."""
        return self.run_solver("dijkstra", start_node=start_node, end_node=end_node, **kwargs)

    def run_simulated_annealing(
        self,
        start_node: str,
        end_node: str,
        waypoints: Optional[List[str]] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Pipeline hook for executing Simulated Annealing VRP solver."""
        if waypoints is not None:
            kwargs["waypoints"] = waypoints
        return self.run_solver(
            "simulated_annealing", start_node=start_node, end_node=end_node, **kwargs
        )

    def pause_execution(self) -> None:
        """Pause active solver execution if supported."""
        if isinstance(self.active_solver, BaseSolver):
            self.active_solver.pause()
            self.notify_listeners("execution_paused")

    def resume_execution(self) -> None:
        """Resume active solver execution if supported."""
        if isinstance(self.active_solver, BaseSolver):
            self.active_solver.resume()
            self.notify_listeners("execution_resumed")

    def stop_execution(self) -> None:
        """Stop active solver execution if supported."""
        if isinstance(self.active_solver, BaseSolver):
            self.active_solver.stop()
            self.notify_listeners("execution_stopped")

    def reset_execution(self) -> None:
        """Reset solver state and clears previous results."""
        if isinstance(self.active_solver, BaseSolver):
            self.active_solver.reset()
        self.last_result = None
        self.notify_listeners("execution_reset")

    def get_benchmark_payload(self) -> Dict[str, Any]:
        """Format last execution result into payload for Benchmarking engine & Matplotlib analytics."""
        if self.last_result is None:
            return {}
        return self.orchestrator.format_benchmark_payload(self.last_result)
