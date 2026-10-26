"""Export benchmark results to non-overwriting CSV and PDF reports."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .algorithm_comparison import AlgorithmComparison
from .convergence_plot import create_convergence_plot
from .metrics_collector import BenchmarkResult


_RESULT_COLUMNS = (
    "algorithm_name",
    "run_id",
    "comparison_group",
    "execution_time_ms",
    "peak_traced_memory_mb",
    "route_distance",
    "route_distance_unit",
    "route_distance_km",
    "total_solution_cost",
    "iteration_count",
    "nodes_visited",
    "temperature_steps",
    "found",
    "measurement_notes",
)


def export_csv(
    path: Any, results: Iterable[BenchmarkResult]
) -> Tuple[Path, Optional[Path]]:
    """Write a results CSV and, when needed, a separate convergence CSV.

    Existing destinations are never overwritten. Missing metric values are
    represented by empty CSV fields, not by zero.
    """
    result_list = _validated_results(results)
    output_path = Path(path)
    history_path = (
        output_path.with_name(output_path.stem + "_convergence.csv")
        if any(result.convergence_history for result in result_list)
        else None
    )
    destinations = [output_path] + ([history_path] if history_path else [])
    for destination in destinations:
        if destination.exists():
            raise FileExistsError("Report already exists: {}".format(destination))

    created_paths = []
    try:
        with output_path.open("x", newline="", encoding="utf-8") as output_file:
            created_paths.append(output_path)
            writer = csv.DictWriter(output_file, fieldnames=_RESULT_COLUMNS)
            writer.writeheader()
            for result in result_list:
                writer.writerow(_result_row(result))

        if history_path is not None:
            with history_path.open("x", newline="", encoding="utf-8") as history_file:
                created_paths.append(history_path)
                writer = csv.writer(history_file)
                writer.writerow(
                    ("algorithm_name", "run_id", "iteration", "best_so_far_cost")
                )
                for result in result_list:
                    for iteration, cost in result.convergence_history:
                        writer.writerow(
                            (result.algorithm_name, result.run_id, iteration, cost)
                        )
    except OSError as error:
        cleanup_errors = []
        for created_path in created_paths:
            try:
                created_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as cleanup_error:
                cleanup_errors.append(
                    "{}: {}".format(created_path, cleanup_error)
                )
        message = "Could not write benchmark CSV report: {}".format(error)
        if cleanup_errors:
            message += "; could not remove partial reports: {}".format(
                "; ".join(cleanup_errors)
            )
        raise OSError(message) from error

    return output_path, history_path


def export_pdf(
    path: Any,
    results: Iterable[BenchmarkResult],
    configuration: Optional[Dict[str, Any]] = None,
    explanation: Optional[str] = None,
) -> Path:
    """Create a readable PDF report using the project's Matplotlib dependency."""
    result_list = _validated_results(results)
    if configuration is not None and not isinstance(configuration, dict):
        raise TypeError("configuration must be a dictionary or None.")
    if explanation is not None and not isinstance(explanation, str):
        raise TypeError("explanation must be a string or None.")
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError(
            "PDF reports require Matplotlib. Install it with 'python -m pip install matplotlib'."
        ) from error

    output_path = Path(path)
    try:
        output_file = output_path.open("xb")
    except FileExistsError:
        raise FileExistsError("Report already exists: {}".format(output_path))
    except OSError as error:
        raise OSError("Could not open PDF report path: {}".format(error)) from error

    completed = False
    try:
        with output_file:
            with PdfPages(output_file) as pdf:
                _write_results_pages(pdf, Figure, result_list, configuration or {}, explanation)
                if result_list:
                    _write_comparison_page(pdf, Figure, result_list)
                for result in result_list:
                    if result.convergence_history:
                        figure = create_convergence_plot(
                            result,
                            title="{} - run {}".format(
                                result.algorithm_name, result.run_id
                            ),
                        )
                        pdf.savefig(figure)
        completed = True
    finally:
        if not completed:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                raise OSError(
                    "Could not remove incomplete PDF report {}: {}".format(
                        output_path, error
                    )
                ) from error
    return output_path


def _validated_results(results: Iterable[BenchmarkResult]) -> List[BenchmarkResult]:
    result_list = list(results)
    if any(not isinstance(result, BenchmarkResult) for result in result_list):
        raise TypeError("All report inputs must be BenchmarkResult instances.")
    return result_list


def _result_row(result: BenchmarkResult) -> Dict[str, Any]:
    return {
        "algorithm_name": result.algorithm_name,
        "run_id": result.run_id,
        "comparison_group": result.comparison_group,
        "execution_time_ms": result.execution_time_ms,
        "peak_traced_memory_mb": result.peak_traced_memory_mb,
        "route_distance": result.route_distance,
        "route_distance_unit": result.route_distance_unit,
        "route_distance_km": result.route_distance_km,
        "total_solution_cost": result.total_solution_cost,
        "iteration_count": result.iteration_count,
        "nodes_visited": result.nodes_visited,
        "temperature_steps": result.temperature_steps,
        "found": result.found,
        "measurement_notes": " | ".join(result.measurement_notes),
    }


def _write_results_pages(
    pdf: Any,
    figure_class: Any,
    results: List[BenchmarkResult],
    configuration: Dict[str, Any],
    explanation: Optional[str],
) -> None:
    cover = figure_class(figsize=(11.7, 8.3), constrained_layout=True)
    cover_axes = cover.subplots()
    cover_axes.axis("off")
    cover_axes.set_title("Opti-Route Benchmark Report", fontsize=18, pad=18)
    details = [
        "Report date: {}".format(datetime.now().astimezone().isoformat(timespec="seconds")),
        "Execution time: milliseconds (ms)",
        "Memory: peak traced Python allocations (MB), not process-wide memory",
        "Distance: the run's reported unit; kilometres are included only when converted",
    ]
    details.extend(
        "Configuration — {}: {}".format(key, value)
        for key, value in sorted(configuration.items())
    )
    details.append(
        "This report contains {} measured run(s); missing values were not replaced "
        "with zero.".format(len(results))
    )
    if explanation:
        details.append("Explanation: {}".format(explanation))
    details.extend(
        (
            "Dijkstra solves point-to-point shortest paths; Simulated Annealing "
            "searches waypoint routes.",
            "Cost is compared only within an explicitly shared workload group; "
            "distance also requires matching distance units.",
        )
    )
    for result in results:
        details.extend(
            "{} / {}: {}".format(result.algorithm_name, result.run_id, note)
            for note in result.measurement_notes
        )
    cover_axes.text(
        0.02,
        0.94,
        "\n\n".join(details),
        transform=cover_axes.transAxes,
        va="top",
        fontsize=10,
        wrap=True,
    )
    pdf.savefig(cover)

    rows_per_page = 14
    result_pages = [
        results[index : index + rows_per_page]
        for index in range(0, len(results), rows_per_page)
    ] or [[]]
    headings = [
        "Algorithm",
        "Run",
        "Time (ms)",
        "Peak traced memory (MB)",
        "Distance (reported units)",
        "Distance (km)",
        "Cost (solver units)",
        "Iterations",
        "Found",
    ]
    for page_number, page_results in enumerate(result_pages):
        figure = figure_class(figsize=(11.7, 8.3), constrained_layout=True)
        axes = figure.subplots()
        axes.axis("off")
        title = "Benchmark results"
        if page_number:
            title += " (continued)"
        axes.set_title(title, fontsize=16, pad=18)
        table_rows = [
            [
                result.algorithm_name,
                result.run_id,
                _format_value(result.execution_time_ms),
                _format_value(result.peak_traced_memory_mb),
                _format_distance(result),
                _format_value(result.route_distance_km),
                _format_value(result.total_solution_cost),
                _format_value(result.iteration_count),
                _format_value(result.found),
            ]
            for result in page_results
        ] or [["N/A"] * len(headings)]
        table = axes.table(
            cellText=table_rows,
            colLabels=headings,
            cellLoc="center",
            loc="upper center",
            bbox=[0.01, 0.15, 0.98, 0.76],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.15)
        pdf.savefig(figure)


def _write_comparison_page(pdf: Any, figure_class: Any, results: List[BenchmarkResult]) -> None:
    comparison_rows = AlgorithmComparison().compare(results)
    headings = [
        "Algorithm",
        "Run",
        "Group",
        "Time (ms)",
        "Peak memory (MB)",
        "Distance (reported units)",
        "Distance (km)",
        "Cost (solver units)",
        "Comparable",
    ]
    for page_start in range(0, len(comparison_rows), 14):
        page_rows = comparison_rows[page_start : page_start + 14]
        figure = figure_class(figsize=(11.7, 8.3), constrained_layout=True)
        axes = figure.subplots()
        axes.axis("off")
        title = "Algorithm comparison"
        if page_start:
            title += " (continued)"
        axes.set_title(title, fontsize=16, pad=18)
        rows = [
            [
                row.algorithm_name,
                row.run_id,
                row.comparison_group or "N/A",
                _format_value(row.execution_time_ms),
                _format_value(row.peak_traced_memory_mb),
                (
                    "{} {}".format(
                        _format_value(row.route_distance), row.route_distance_unit
                    )
                    if row.route_distance is not None
                    else "N/A"
                ),
                _format_value(row.route_distance_km),
                _format_value(row.total_solution_cost),
                _format_value(row.problem_metrics_comparable),
            ]
            for row in page_rows
        ]
        table = axes.table(
            cellText=rows,
            colLabels=headings,
            cellLoc="center",
            loc="upper center",
            bbox=[0.01, 0.25, 0.98, 0.66],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.15)
        axes.text(
            0.01,
            0.12,
            "Cost is compared only within a shared workload group; distance also "
            "requires matching units. Dijkstra's point-to-point objective may "
            "differ from Simulated Annealing's waypoint route objective.",
            transform=axes.transAxes,
            va="bottom",
            fontsize=9,
            wrap=True,
        )
        pdf.savefig(figure)


def _format_distance(result: BenchmarkResult) -> str:
    if result.route_distance is None:
        return "N/A"
    return "{} {}".format(
        _format_value(result.route_distance), result.route_distance_unit
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)
