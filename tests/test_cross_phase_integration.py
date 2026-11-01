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

        # Run Dijkstra (Unoptimized FIFO Baseline at index 3)
        win.alg_combo.setCurrentIndex(3)
        win.on_run_solver()
        self.assertEqual(len(win.map_canvas._route_items), 1)
        self.assertEqual(win.comp_table.rowCount(), 1)

        # Drag an intermediate node on the solved Dijkstra route and verify route overlay + animation sync
        dijkstra_path = list(win.map_canvas._active_route_path)
        self.assertGreaterEqual(len(dijkstra_path), 2)
        dragged_node_id = dijkstra_path[1]
        win.map_canvas._node_items[dragged_node_id].setPos(333.0, 444.0)
        route_poly = win.map_canvas._route_items[0].path()
        self.assertAlmostEqual(route_poly.elementAt(1).x, 333.0, places=3)
        self.assertAlmostEqual(route_poly.elementAt(1).y, 444.0, places=3)
        self.assertIn((333.0, 444.0), win.animation_controller._points)

        # Run Simulated Annealing (VRP Optimizer at index 0)
        win.alg_combo.setCurrentIndex(0)
        win.iter_spin.setValue(5)
        win.on_run_solver()
        self.assertEqual(len(win.map_canvas._route_items), 1)
        self.assertEqual(win.comp_table.rowCount(), 2)

        # Toggle Theme
        win.on_toggle_theme()
        self.assertEqual(tm.current_mode, ThemeMode.LIGHT)

        # Test Bellman-Ford (index 4), TLBO (index 1), and Genetic Algorithm (index 2) from GUI
        win.alg_combo.setCurrentIndex(4)  # Bellman-Ford (Unoptimized FIFO Baseline)
        win.on_run_solver()
        self.assertEqual(win.comp_table.rowCount(), 3)

        win.alg_combo.setCurrentIndex(1)  # TLBO (Teaching-Learning VRP)
        win.on_run_solver()
        self.assertEqual(win.comp_table.rowCount(), 4)

        win.alg_combo.setCurrentIndex(2)  # Genetic Algorithm (GA VRP)
        win.on_run_solver()
        self.assertEqual(win.comp_table.rowCount(), 5)

        # Test switching preset sample networks via toolbar selector
        self.assertGreaterEqual(win.sample_combo.count(), 9)
        win.sample_combo.setCurrentIndex(1)
        self.assertGreaterEqual(win.controller.graph.node_count(), 8)
        win.close()

    def test_all_nine_sample_networks_and_five_algorithms(self) -> None:
        """Verify all 9 sample networks in data/ load cleanly and all 5 algorithms solve them rapidly."""
        data_dir = self.repo_root / "data"
        json_files = sorted(data_dir.glob("*.json"))
        self.assertGreaterEqual(len(json_files), 9)

        controller = OptiRouteController()
        for jf in json_files:
            proj = controller.load_project(jf)
            self.assertGreaterEqual(proj.graph.node_count(), 8, f"{jf.name} should have >= 8 nodes")
            self.assertGreaterEqual(proj.graph.edge_count(), 12, f"{jf.name} should have >= 12 edges")
            self.assertGreaterEqual(len(proj.fleet), 2, f"{jf.name} should have >= 2 vehicles")
            self.assertGreaterEqual(len(proj.jobs), 4, f"{jf.name} should have >= 4 jobs")

        # Load default sample_network.json and compare all 5 algorithms
        controller.load_project(self.sample_json)
        res_dijk = controller.run_dijkstra("Depot", "Terminal")
        res_bf = controller.run_bellman_ford("Depot", "Terminal")
        self.assertTrue(res_dijk.found)
        self.assertTrue(res_bf.found)
        self.assertAlmostEqual(res_dijk.total_cost, res_bf.total_cost, places=5)
        self.assertEqual(res_dijk.path, res_bf.path)

        res_sa = controller.run_simulated_annealing("Depot", "Terminal", seed=42)
        res_tlbo = controller.run_tlbo("Depot", "Terminal", seed=42)
        res_ga = controller.run_genetic_algorithm("Depot", "Terminal", seed=42)

        for res in (res_sa, res_tlbo, res_ga):
            self.assertTrue(res.found)
            self.assertGreater(len(res.convergence_history), 1)
            self.assertIn("V1", res.routes)
            self.assertIn("V2", res.routes)
            self.assertLess(res.execution_time_ms, 50.0)


if __name__ == "__main__":
    unittest.main()
