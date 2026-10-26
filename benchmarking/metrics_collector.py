"""Store and validate normalized benchmark results."""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BenchmarkResult:
    """One algorithm run with explicit units and unavailable values."""

    algorithm_name: str
    run_id: str
    execution_time_ms: float
    peak_traced_memory_mb: Optional[float] = None
    route_distance: Optional[float] = None
    route_distance_unit: str = "unspecified graph units"
    route_distance_km: Optional[float] = None
    total_solution_cost: Optional[float] = None
    iteration_count: Optional[int] = None
    nodes_visited: Optional[int] = None
    temperature_steps: Optional[int] = None
    found: Optional[bool] = None
    comparison_group: Optional[str] = None
    convergence_history: Tuple[Tuple[int, float], ...] = field(default_factory=tuple)
    measurement_notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        try:
            normalized_history = tuple(tuple(point) for point in self.convergence_history)
            normalized_notes = tuple(self.measurement_notes)
        except TypeError as error:
            raise ValueError("History and measurement notes must be iterable.") from error
        object.__setattr__(self, "convergence_history", normalized_history)
        object.__setattr__(self, "measurement_notes", normalized_notes)

        if not isinstance(self.algorithm_name, str) or not self.algorithm_name.strip():
            raise ValueError("algorithm_name must not be empty.")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must not be empty.")
        if (
            not isinstance(self.route_distance_unit, str)
            or not self.route_distance_unit.strip()
        ):
            raise ValueError("route_distance_unit must not be empty.")

        self._validate_number("execution_time_ms", self.execution_time_ms, allow_none=False)
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must not be negative.")

        for field_name in (
            "peak_traced_memory_mb",
            "route_distance",
            "route_distance_km",
            "total_solution_cost",
        ):
            value = getattr(self, field_name)
            self._validate_number(field_name, value)
            if value is not None and field_name != "total_solution_cost" and value < 0:
                raise ValueError("{} must not be negative.".format(field_name))

        for field_name in ("iteration_count", "nodes_visited", "temperature_steps"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("{} must be a non-negative integer or None.".format(field_name))

        if self.found is not None and not isinstance(self.found, bool):
            raise ValueError("found must be a boolean or None.")
        if self.comparison_group is not None and (
            not isinstance(self.comparison_group, str) or not self.comparison_group.strip()
        ):
            raise ValueError("comparison_group must be a non-empty string or None.")
        if any(not isinstance(note, str) for note in self.measurement_notes):
            raise ValueError("measurement_notes must contain only strings.")

        for point in self.convergence_history:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError("Each convergence point must be (iteration, best_cost).")
            iteration, cost = point
            if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
                raise ValueError("Convergence iterations must be non-negative integers.")
            self._validate_number("convergence cost", cost, allow_none=False)

    @staticmethod
    def _validate_number(
        name: str, value: Optional[float], allow_none: bool = True
    ) -> None:
        if value is None and allow_none:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{} must be a finite number or None.".format(name))
        if not math.isfinite(value):
            raise ValueError("{} must be finite.".format(name))


class MetricsCollector:
    """Keep benchmark runs distinct and provide filtered result access."""

    def __init__(self) -> None:
        self._results: List[BenchmarkResult] = []
        self._run_keys = set()

    @property
    def results(self) -> Tuple[BenchmarkResult, ...]:
        return tuple(self._results)

    def add(self, result: BenchmarkResult) -> None:
        if not isinstance(result, BenchmarkResult):
            raise TypeError("result must be a BenchmarkResult.")
        key = (result.algorithm_name.casefold(), result.run_id)
        if key in self._run_keys:
            raise ValueError(
                "Run identifier {!r} already exists for algorithm {!r}.".format(
                    result.run_id, result.algorithm_name
                )
            )
        self._run_keys.add(key)
        self._results.append(result)

    def get_results(
        self,
        algorithm_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[BenchmarkResult, ...]:
        if algorithm_name is not None and not isinstance(algorithm_name, str):
            raise TypeError("algorithm_name must be a string or None.")
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError("run_id must be a string or None.")
        return tuple(
            result
            for result in self._results
            if (
                algorithm_name is None
                or result.algorithm_name.casefold() == algorithm_name.casefold()
            )
            and (run_id is None or result.run_id == run_id)
        )

    def compare(self):
        """Return structured per-run comparisons using the recorded comparability labels."""
        from .algorithm_comparison import AlgorithmComparison

        return AlgorithmComparison().compare(self._results)
