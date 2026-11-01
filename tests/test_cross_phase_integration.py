"""End-to-end integration tests verifying cross-module connections across Phases 1–4."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.controllers import OptiRouteController
from opti_route.models import (
    DynamicJobQueue,
    GraphFactory,
    Job,
    Node,
    NodeType,
    RoadGraph,
    Vehicle,
    VehicleFleet,
)


class TestCrossPhaseIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.sample_json = self.repo_root / "data" / "sample_network.json"
        self.sample_csv = self.repo_root / "data" / "sample_network.csv"

    def test_sample_datasets_load_and_roundtrip_via_controller(self) -> None:
        """Verify sample JSON and CSV datasets in data/ load cleanly into OptiRouteController."""
        self.assertTrue(self.sample_json.exists(), "data/sample_network.json must exist")
        self.assertTrue(self.sample_csv.exists(), "data/sample_network.csv must exist")

        controller = OptiRouteController()
        proj_json = controller.load_project(self.sample_json)
        self.assertEqual(proj_json.graph.node_count(), 9)
        self.assertEqual(proj_json.graph.edge_count(), 13)
        self.assertEqual(len(proj_json.fleet), 2)
        self.assertEqual(len(proj_json.jobs), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "saved.json"
            out_csv = Path(tmpdir) / "saved.csv"
            controller.save_project(out_json)
            controller.save_project(out_csv)

            reloaded_json = GraphFactory.load(out_json)
            reloaded_csv = GraphFactory.load(out_csv)
            self.assertEqual(
                GraphFactory.to_dict(reloaded_json.graph, reloaded_json.fleet, reloaded_json.jobs),
                GraphFactory.to_dict(reloaded_csv.graph, reloaded_csv.fleet, reloaded_csv.jobs),
            )

    def test_solvers_consume_vehicle_fleet_and_job_queue(self) -> None:
        """Verify Dijkstra and SA solvers consume VehicleFleet and DynamicJobQueue models."""
        controller = OptiRouteController()
        controller.load_project(self.sample_json)

        # 1. Dijkstra resolving start_depot from fleet and dropoff from job_queue
        dijkstra_res = controller.orchestrator.execute(
            "dijkstra",
            controller.graph,
            fleet=controller.fleet,
            job_queue=controller.job_queue,
        )
        self.assertTrue(dijkstra_res.found)
        self.assertEqual(dijkstra_res.path[0], "Depot")
        self.assertIn("V1", dijkstra_res.routes)

        # 2. Simulated Annealing extracting waypoints from DynamicJobQueue and assigning fleet routes
        sa_res = controller.run_simulated_annealing(
            start_node="Depot",
            end_node="Terminal",
            initial_temp=200.0,
            cooling_rate=0.85,
            iterations_per_temp=10,
            seed=42,
        )
        self.assertTrue(sa_res.found)
        self.assertIn("V1", sa_res.routes)
        self.assertIn("V2", sa_res.routes)
        self.assertEqual(sa_res.telemetry.get("jobs_routed"), 5)
        self.assertIn("vehicle_loads", sa_res.telemetry)
        self.assertIn("vehicle_fuel", sa_res.telemetry)

    def test_controller_records_benchmarks_and_exports_csv(self) -> None:
        """Verify controller automatically records Phase 4 benchmarks, compares runs, and exports CSV."""
        controller = OptiRouteController()
        controller.load_project(self.sample_json)

        controller.run_dijkstra("Depot", "Terminal")
        controller.run_simulated_annealing(
            "Depot",
            "Terminal",
            initial_temp=150.0,
            cooling_rate=0.80,
            iterations_per_temp=5,
            seed=7,
        )

        results = controller.get_benchmark_results()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].algorithm_name, "Dijkstra")
        self.assertEqual(results[1].algorithm_name, "SimulatedAnnealing")

        comp_rows = controller.get_comparison_rows()
        self.assertEqual(len(comp_rows), 2)
        self.assertTrue(all(r.problem_metrics_comparable for r in comp_rows))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "report.csv"
            out_csv, hist_csv = controller.export_benchmark_csv(csv_path)
            self.assertTrue(out_csv.exists())
            self.assertIsNotNone(hist_csv)
            self.assertTrue(hist_csv.exists())

    def test_gui_main_window_end_to_end(self) -> None:
        """Verify OptiRouteMainWindow launches, runs Dijkstra & SA, updates table, and toggles theme."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from opti_route.views import OptiRouteMainWindow, ThemeManager, ThemeMode, ensure_qt_bridge

        if not ensure_qt_bridge():
            self.skipTest("Neither PyQt6 nor PyQt5 is available.")

        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        controller = OptiRouteController()
        controller.load_project(self.sample_json)
        tm = ThemeManager()
        win = OptiRouteMainWindow(controller=controller, theme_manager=tm)

        # Run Dijkstra
        win.alg_combo.setCurrentIndex(0)
        win.on_run_solver()
        self.assertEqual(len(win.map_canvas._route_items), 1)
        self.assertEqual(win.comp_table.rowCount(), 1)

        # Run Simulated Annealing
        win.alg_combo.setCurrentIndex(1)
        win.iter_spin.setValue(5)
        win.on_run_solver()
        self.assertEqual(len(win.map_canvas._route_items), 1)
        self.assertEqual(win.comp_table.rowCount(), 2)

        # Toggle Theme
        win.on_toggle_theme()
        self.assertEqual(tm.current_mode, ThemeMode.LIGHT)
        win.close()


if __name__ == "__main__":
    unittest.main()
