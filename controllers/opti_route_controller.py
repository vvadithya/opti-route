"""
OptiRouteController: MVC Controller integrating models, abstract solvers,
and execution pipeline hooks for Phase 2 algorithms (Dijkstra and Simulated Annealing).
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
from ..models.graph import RoadGraph
from ..solvers.base_solver import (
    AlgorithmStrategy,
    BaseSolver,
    ExecutionParams,
    OptimizationResult,
    SolverState,
)
from ..solvers.orchestrator import SolverOrchestrator


class OptiRouteController:
    """
    MVC Controller managing references to road graph models, vehicle fleet state,
    job queues, and optimization solvers. Provides pipeline hooks to configure
    and run algorithms, manage execution state, and deliver results to UI and Analytics.
    """

    def __init__(
        self,
        graph: Optional[RoadGraph] = None,
        fleet: Optional[Any] = None,
        job_queue: Optional[Any] = None,
    ):
        self.graph: RoadGraph = graph if graph is not None else RoadGraph()
        self.fleet: Any = fleet
        self.job_queue: Any = job_queue
        self.orchestrator: SolverOrchestrator = SolverOrchestrator()
        self.current_params: ExecutionParams = ExecutionParams()
        self.last_result: Optional[OptimizationResult] = None
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
            self.notify_listeners("execution_completed", result)
            return result
        except Exception as e:
            self.notify_listeners("execution_failed", str(e))
            raise e

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
