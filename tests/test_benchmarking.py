"""Unit and integration tests for the independent Phase 4 components."""

import csv
import importlib
import importlib.util
import sys
import tempfile
import tracemalloc
import unittest
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
try:
    importlib.import_module("opti_route")
except ModuleNotFoundError as error:
    if error.name != "opti_route":
        raise
    package = ModuleType("opti_route")
    package.__path__ = [str(_REPOSITORY_ROOT)]
    sys.modules["opti_route"] = package

from opti_route.benchmarking.algorithm_comparison import AlgorithmComparison
from opti_route.benchmarking.benchmark_runner import BenchmarkRunner
from opti_route.benchmarking.convergence_plot import create_convergence_plot
from opti_route.benchmarking.metrics_collector import BenchmarkResult, MetricsCollector
from opti_route.benchmarking.performance_monitor import PerformanceMonitor
from opti_route.benchmarking.report_generator import export_csv, export_pdf
from opti_route.models.graph import Node, RoadGraph


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


def sample_result(run_id="run-1", **overrides):
    values = {
        "algorithm_name": "SimulatedAnnealing",
        "run_id": run_id,
        "execution_time_ms": 12.5,
        "peak_traced_memory_mb": 1.25,
        "route_distance": 8.0,
        "route_distance_unit": "test graph units",
        "total_solution_cost": 8.0,
        "iteration_count": 3,
        "nodes_visited": 3,
        "temperature_steps": 2,
        "found": True,
        "comparison_group": None,
        "convergence_history": ((1, 10.0), (2, 9.0), (3, 8.0)),
    }
    values.update(overrides)
    return BenchmarkResult(**values)


class PerformanceMonitorTests(unittest.TestCase):
    def test_measures_elapsed_time(self):
        monitor = PerformanceMonitor()
        with patch(
            "opti_route.benchmarking.performance_monitor.time.perf_counter",
            side_effect=(10.0, 10.125),
        ):
            monitor.start()
            measurement = monitor.stop()
        self.assertAlmostEqual(measurement.execution_time_ms, 125.0)
        if measurement.tracing_was_already_active:
            self.assertIsNone(measurement.peak_traced_memory_mb)
        else:
            self.assertIsNotNone(measurement.peak_traced_memory_mb)

    def test_measures_peak_traced_python_memory(self):
        tracing_was_active = tracemalloc.is_tracing()
        monitor = PerformanceMonitor()
        try:
            monitor.start()
            allocation = bytearray(1024 * 1024)
            measurement = monitor.stop()
            if tracing_was_active:
                self.assertIsNone(measurement.peak_traced_memory_mb)
            else:
                self.assertGreater(measurement.peak_traced_memory_mb, 0.5)
            del allocation
        finally:
            if not tracing_was_active and tracemalloc.is_tracing():
                tracemalloc.stop()

    def test_stops_tracing_when_operation_raises(self):
        tracing_was_active = tracemalloc.is_tracing()
        monitor = PerformanceMonitor()
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with monitor:
                    raise RuntimeError("synthetic failure")
            self.assertFalse(monitor.is_active)
            self.assertIsNotNone(monitor.measurement)
            self.assertEqual(tracemalloc.is_tracing(), tracing_was_active)
        finally:
            if not tracing_was_active and tracemalloc.is_tracing():
                tracemalloc.stop()

    def test_preserves_an_already_active_tracing_session(self):
        tracing_was_active = tracemalloc.is_tracing()
        if not tracing_was_active:
            tracemalloc.start()
        try:
            monitor = PerformanceMonitor()
            monitor.start()
            measurement = monitor.stop()
            self.assertTrue(measurement.tracing_was_already_active)
            self.assertIsNone(measurement.peak_traced_memory_mb)
            self.assertTrue(tracemalloc.is_tracing())
        finally:
            if not tracing_was_active:
                tracemalloc.stop()


class MetricsAndComparisonTests(unittest.TestCase):
    def test_collects_multiple_distinct_runs_and_filters_them(self):
        collector = MetricsCollector()
        first = sample_result()
        second = sample_result(run_id="run-2")
        collector.add(first)
        collector.add(second)
        self.assertEqual(len(collector.results), 2)
        self.assertEqual(collector.get_results(run_id="run-2"), (second,))
        with self.assertRaises(ValueError):
            collector.add(first)

    def test_missing_metrics_remain_none(self):
        result = BenchmarkResult(
            algorithm_name="Dijkstra",
            run_id="missing",
            execution_time_ms=1.0,
        )
        self.assertIsNone(result.peak_traced_memory_mb)
        self.assertIsNone(result.route_distance_km)
        self.assertIsNone(result.total_solution_cost)
        with self.assertRaises(ValueError):
            BenchmarkResult(
                algorithm_name="Dijkstra",
                run_id="invalid",
                execution_time_ms=-1.0,
            )

    def test_comparison_omits_unrelated_costs_but_keeps_runtime(self):
        first = sample_result()
        second = sample_result(
            run_id="dijkstra",
            algorithm_name="Dijkstra",
            comparison_group=None,
        )
        rows = AlgorithmComparison().compare((first, second))
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(rows[0].execution_time_ms, 0)
        self.assertIsNone(rows[0].total_solution_cost)
        self.assertIsNone(rows[1].route_distance)

    def test_comparison_includes_cost_for_same_explicit_workload(self):
        first = sample_result(comparison_group="same-test-workload")
        second = sample_result(
            run_id="dijkstra",
            algorithm_name="Dijkstra",
            comparison_group="same-test-workload",
        )
        rows = AlgorithmComparison().compare((first, second))
        self.assertTrue(all(row.problem_metrics_comparable for row in rows))
        self.assertEqual(rows[0].total_solution_cost, 8.0)

    def test_comparison_uses_kilometres_when_source_units_differ(self):
        first = sample_result(
            comparison_group="same-test-workload",
            route_distance_unit="metres",
            route_distance_km=0.008,
        )
        second = sample_result(
            run_id="kilometres",
            comparison_group="same-test-workload",
            route_distance_unit="miles",
            route_distance_km=0.008,
        )
        rows = AlgorithmComparison().compare((first, second))
        self.assertTrue(all(row.route_distance is None for row in rows))
        self.assertEqual(rows[0].route_distance_km, 0.008)


class BenchmarkRunnerTests(unittest.TestCase):
    @staticmethod
    def make_graph():
        graph = RoadGraph()
        for node_id, x in (("Depot", 0.0), ("A", 1.0), ("B", 2.0)):
            graph.add_node(Node(node_id, x, 0.0))
        graph.add_edge("Depot", "A", 1.0)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "Depot", 2.0)
        graph.add_edge("Depot", "B", 3.0)
        return graph

    def test_runs_actual_dijkstra_and_sa_interfaces(self):
        collector = MetricsCollector()
        runner = BenchmarkRunner(collector)
        graph = self.make_graph()

        dijkstra_result = runner.run(
            "dijkstra",
            graph,
            "dijkstra-test",
            solve_kwargs={"start_node": "Depot", "end_node": "B"},
            comparison_group="test-graph-route",
        )
        sa_result = runner.run(
            "simulated_annealing",
            graph,
            "sa-test",
            solver_options={
                "initial_temp": 2.0,
                "cooling_rate": 0.5,
                "min_temp": 0.5,
                "iterations_per_temp": 1,
                "seed": 42,
            },
            solve_kwargs={
                "start_node": "Depot",
                "end_node": "Depot",
                "waypoints": ["A", "B"],
            },
            comparison_group="test-graph-route",
        )
        self.assertTrue(dijkstra_result.found)
        self.assertTrue(sa_result.found)
        self.assertEqual(dijkstra_result.total_solution_cost, 2.0)
        self.assertGreater(len(sa_result.convergence_history), 0)
        self.assertEqual(len(collector.results), 2)


class CSVReportTests(unittest.TestCase):
    def test_csv_exports_results_and_separate_history_without_overwriting(self):
        result = sample_result(peak_traced_memory_mb=None)
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "benchmark.csv"
            result_path, history_path = export_csv(report, (result,))
            self.assertEqual(result_path, report)
            self.assertEqual(history_path.name, "benchmark_convergence.csv")
            with report.open(newline="", encoding="utf-8") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(rows[0]["peak_traced_memory_mb"], "")
            self.assertEqual(rows[0]["run_id"], "run-1")
            with self.assertRaises(FileExistsError):
                export_csv(report, (result,))


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib is required for plotting and PDF tests.")
class PlotAndReportTests(unittest.TestCase):
    def test_empty_and_non_empty_convergence_plots(self):
        empty_figure = create_convergence_plot(())
        populated_figure = create_convergence_plot(sample_result())
        self.assertEqual(len(empty_figure.axes), 1)
        self.assertEqual(len(populated_figure.axes[0].lines), 1)

    def test_pdf_generation_produces_a_non_overwriting_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "benchmark.pdf"
            written_path = export_pdf(
                report,
                (sample_result(),),
                configuration={"dataset": "synthetic unit-test data"},
            )
            self.assertEqual(written_path, report)
            self.assertTrue(report.read_bytes().startswith(b"%PDF"))
            with self.assertRaises(FileExistsError):
                export_pdf(report, ())
            empty_report = Path(temporary_directory) / "empty.pdf"
            export_pdf(empty_report, ())
            self.assertTrue(empty_report.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
