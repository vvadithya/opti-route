"""Build unit-labelled comparison rows without implying unlike objectives match."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .metrics_collector import BenchmarkResult


@dataclass(frozen=True)
class ComparisonRow:
    algorithm_name: str
    run_id: str
    comparison_group: Optional[str]
    execution_time_ms: float
    peak_traced_memory_mb: Optional[float]
    route_distance: Optional[float]
    route_distance_unit: Optional[str]
    route_distance_km: Optional[float]
    total_solution_cost: Optional[float]
    iteration_count: Optional[int]
    found: Optional[bool]
    problem_metrics_comparable: bool
    note: str


class AlgorithmComparison:
    """Compare all supplied runs while gating distance/cost on explicit workload IDs."""

    def compare(self, results: Iterable[BenchmarkResult]) -> Tuple[ComparisonRow, ...]:
        runs = tuple(results)
        if any(not isinstance(result, BenchmarkResult) for result in runs):
            raise TypeError("All comparison inputs must be BenchmarkResult instances.")

        grouped_runs: Dict[str, List[BenchmarkResult]] = {}
        for result in runs:
            if result.comparison_group is not None:
                grouped_runs.setdefault(result.comparison_group, []).append(result)

        rows: List[ComparisonRow] = []
        for result in runs:
            same_group_runs = grouped_runs.get(result.comparison_group or "", [])
            comparable_group = len(same_group_runs) > 1
            comparable_distance = (
                comparable_group
                and len({run.route_distance_unit for run in same_group_runs}) == 1
            )
            comparable_kilometres = comparable_group and all(
                run.route_distance_km is not None for run in same_group_runs
            )
            notes = []
            if not comparable_group:
                notes.append(
                    "Cost and distance omitted: runs do not share an explicit comparison group."
                )
            elif not comparable_distance:
                notes.append("Distance omitted: route distance units differ between runs.")
            if result.algorithm_name.casefold() == "dijkstra":
                notes.append(
                    "Dijkstra is point-to-point; compare its route metrics with a routing "
                    "algorithm only when both runs use the same defined workload."
                )
            rows.append(
                ComparisonRow(
                    algorithm_name=result.algorithm_name,
                    run_id=result.run_id,
                    comparison_group=result.comparison_group,
                    execution_time_ms=result.execution_time_ms,
                    peak_traced_memory_mb=result.peak_traced_memory_mb,
                    route_distance=result.route_distance if comparable_distance else None,
                    route_distance_unit=(
                        result.route_distance_unit if comparable_distance else None
                    ),
                    route_distance_km=(
                        result.route_distance_km if comparable_kilometres else None
                    ),
                    total_solution_cost=(
                        result.total_solution_cost if comparable_group else None
                    ),
                    iteration_count=result.iteration_count,
                    found=result.found,
                    problem_metrics_comparable=comparable_group,
                    note=" ".join(notes),
                )
            )
        return tuple(rows)
