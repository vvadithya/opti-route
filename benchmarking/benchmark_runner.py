"""Run the supplied solvers externally and convert their outputs into metrics."""

import math
from typing import Any, Callable, Dict, Optional

from ..solvers.dijkstra_solver import DijkstraSolver
from ..solvers.simulated_annealing import SimulatedAnnealingSolver
from .metrics_collector import BenchmarkResult, MetricsCollector
from .performance_monitor import PerformanceMonitor


_SOLVERS = {
    "dijkstra": (DijkstraSolver, "Dijkstra"),
    "simulated_annealing": (SimulatedAnnealingSolver, "SimulatedAnnealing"),
    "sa": (SimulatedAnnealingSolver, "SimulatedAnnealing"),
}


class BenchmarkRunner:
    """Execute the actual project solvers and collect their returned measurements.

    The caller supplies the graph and solver arguments. Calls run synchronously;
    a GUI should invoke ``run`` from a worker thread to keep its event loop responsive.
    """

    def __init__(self, collector: Optional[MetricsCollector] = None) -> None:
        self.collector = collector

    def run(
        self,
        algorithm_name: str,
        graph: Any,
        run_id: str,
        solver_options: Optional[Dict[str, Any]] = None,
        solve_kwargs: Optional[Dict[str, Any]] = None,
        comparison_group: Optional[str] = None,
        distance_unit: str = "unspecified graph units",
        distance_to_km: Optional[Callable[[float], float]] = None,
    ) -> BenchmarkResult:
        """Run Dijkstra or Simulated Annealing with caller-supplied graph inputs."""
        if not isinstance(algorithm_name, str):
            raise TypeError("algorithm_name must be a string.")
        algorithm_key = algorithm_name.casefold()
        if algorithm_key not in _SOLVERS:
            raise ValueError(
                "Unknown algorithm {!r}; expected one of {}.".format(
                    algorithm_name, ", ".join(sorted(_SOLVERS))
                )
            )
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must not be empty.")
        if not isinstance(distance_unit, str) or not distance_unit.strip():
            raise ValueError("distance_unit must not be empty.")
        if comparison_group is not None and (
            not isinstance(comparison_group, str) or not comparison_group.strip()
        ):
            raise ValueError("comparison_group must be a non-empty string or None.")
        if distance_to_km is not None and not callable(distance_to_km):
            raise TypeError("distance_to_km must be callable or None.")

        solver_class, expected_name = _SOLVERS[algorithm_key]
        solver = solver_class(**(solver_options or {}))
        arguments = dict(solve_kwargs or {})
        monitor = PerformanceMonitor()

        monitor.start()
        try:
            solver_result = solver.solve(graph, **arguments)
        finally:
            measurement = monitor.stop()

        found = getattr(solver_result, "found", None)
        found_solution = found is not False
        raw_cost = getattr(solver_result, "total_cost", None)
        raw_distance = getattr(solver_result, "total_distance", None)
        total_cost = self._finite_value(raw_cost) if found_solution else None
        route_distance = self._finite_value(raw_distance) if found_solution else None

        route_distance_km = None
        if route_distance is not None and distance_to_km is not None:
            route_distance_km = distance_to_km(route_distance)
            if (
                isinstance(route_distance_km, bool)
                or not isinstance(route_distance_km, (int, float))
                or not math.isfinite(route_distance_km)
                or route_distance_km < 0
            ):
                raise ValueError("distance_to_km must return a finite, non-negative number.")

        raw_history = getattr(solver_result, "convergence_history", ())
        history = []
        for point in raw_history:
            if len(point) == 3:
                iteration, _, best_cost = point
            elif len(point) == 2:
                iteration, best_cost = point
            else:
                continue
            finite_cost = self._finite_value(best_cost)
            if finite_cost is not None:
                history.append((int(iteration), finite_cost))

        iteration_count = None
        if hasattr(solver_result, "convergence_history"):
            iteration_count = max((point[0] for point in history), default=0)

        notes = []
        if measurement.tracing_was_already_active:
            notes.append(
                "Peak traced memory unavailable because tracemalloc was already active."
            )
        if route_distance_km is None and route_distance is not None:
            notes.append(
                "Distance is in {!r}; no kilometre conversion was supplied.".format(distance_unit)
            )

        result = BenchmarkResult(
            algorithm_name=getattr(solver_result, "algorithm_name", expected_name),
            run_id=run_id,
            execution_time_ms=measurement.execution_time_ms,
            peak_traced_memory_mb=measurement.peak_traced_memory_mb,
            route_distance=route_distance,
            route_distance_unit=distance_unit,
            route_distance_km=route_distance_km,
            total_solution_cost=total_cost,
            iteration_count=iteration_count,
            nodes_visited=self._non_negative_integer(
                getattr(solver_result, "nodes_visited", None)
            ),
            temperature_steps=self._non_negative_integer(
                getattr(solver_result, "temperature_steps", None)
            ),
            found=found if isinstance(found, bool) else None,
            comparison_group=comparison_group,
            convergence_history=tuple(history),
            measurement_notes=tuple(notes),
        )
        if self.collector is not None:
            self.collector.add(result)
        return result

    @staticmethod
    def _finite_value(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _non_negative_integer(value: Any) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
