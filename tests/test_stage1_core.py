import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Node, RoadGraph
from opti_route.core import (
    AlgorithmStrategy,
    BaseSolver,
    ExecutionParams,
    OptimizationResult,
    RouteResult,
    SolverState,
)
from opti_route.solvers import DijkstraSolver, SimulatedAnnealingSolver, SAResult
from opti_route.controllers import OptiRouteController


class DummyCustomSolver(BaseSolver):
    """Concrete solver implementation for testing abstract strategy contracts."""

    def __init__(self, name: str = "DummyCustom"):
        super().__init__(name=name)

    def solve(
        self,
        graph: RoadGraph,
        fleet=None,
        job_queue=None,
        **kwargs,
    ) -> OptimizationResult:
        self.set_state(SolverState.RUNNING)
        self.notify_progress(50.0, "Halfway done")
        res = OptimizationResult(
            path=["A", "B"],
            total_cost=10.0,
            total_distance=10.0,
            found=True,
            algorithm_name=self.name,
        )
        self.set_state(SolverState.COMPLETED)
        return res


class TestStage1CoreInfrastructure(unittest.TestCase):
    def setUp(self):
        """Set up road graph and controller for Stage 1 infrastructure testing."""
        self.graph = RoadGraph()
        nodes = [
            Node("Depot", 0.0, 0.0),
            Node("Hub1", 10.0, 0.0),
            Node("Hub2", 10.0, 10.0),
            Node("Customer1", 0.0, 10.0),
        ]
        for n in nodes:
            self.graph.add_node(n)

        self.graph.add_edge("Depot", "Hub1", 10.0)
        self.graph.add_edge("Hub1", "Hub2", 10.0)
        self.graph.add_edge("Hub2", "Customer1", 10.0)
        self.graph.add_edge("Customer1", "Depot", 10.0)

    def test_abstract_strategy_contract(self):
        """Verify AlgorithmStrategy cannot be instantiated directly and forces solve signature."""
        with self.assertRaises(TypeError):
            AlgorithmStrategy()  # Direct instantiation should fail

        dummy = DummyCustomSolver()
        self.assertIsInstance(dummy, AlgorithmStrategy)
        self.assertIsInstance(dummy, BaseSolver)

        res = dummy.solve(self.graph)
        self.assertIsInstance(res, OptimizationResult)
        self.assertTrue(res.found)
        self.assertEqual(res.algorithm_name, "DummyCustom")

    def test_solver_state_transitions_and_signals(self):
        """Test BaseSolver state transitions (Init, Running, Paused, Stopped, Completed) and signal hooks."""
        solver = DummyCustomSolver()
        self.assertEqual(solver.state, SolverState.INIT)

        state_changes = []
        progress_updates = []

        solver.register_signal_hook("state_change", lambda old, new: state_changes.append((old, new)))
        solver.set_progress_callback(lambda pct, msg: progress_updates.append((pct, msg)))

        solver.solve(self.graph)
        self.assertEqual(solver.state, SolverState.COMPLETED)
        self.assertEqual(len(state_changes), 2)  # INIT -> RUNNING, RUNNING -> COMPLETED
        self.assertEqual(progress_updates, [(50.0, "Halfway done")])

        # Test state transition helpers
        solver.pause()
        self.assertEqual(solver.state, SolverState.COMPLETED)  # pause only from RUNNING

        solver.set_state(SolverState.RUNNING)
        solver.pause()
        self.assertEqual(solver.state, SolverState.PAUSED)

        solver.resume()
        self.assertEqual(solver.state, SolverState.RUNNING)

        solver.stop()
        self.assertEqual(solver.state, SolverState.STOPPED)

        solver.reset()
        self.assertEqual(solver.state, SolverState.INIT)

    def test_execution_params_and_optimization_result_dtos(self):
        """Verify ExecutionParams and OptimizationResult DTO data containers."""
        params = ExecutionParams(
            initial_temp=500.0,
            cooling_rate=0.90,
            iterations_per_temp=20,
            mutation_operator="2-opt",
            waypoints=["Hub1", "Hub2"],
            start_node="Depot",
            end_node="Customer1",
        )
        self.assertEqual(params.initial_temp, 500.0)
        self.assertEqual(params.cooling_rate, 0.90)
        self.assertEqual(params.waypoints, ["Hub1", "Hub2"])

        res = OptimizationResult(
            path=["Depot", "Hub1"],
            total_cost=10.0,
            total_distance=10.0,
            nodes_visited=2,
            execution_time_ms=5.5,
            algorithm_name="Dijkstra",
            found=True,
            peak_memory_mb=0.12,
            telemetry={"key": "val"},
        )
        self.assertEqual(res.path, ["Depot", "Hub1"])
        self.assertEqual(res.total_cost, 10.0)
        self.assertAlmostEqual(res.peak_memory_mb, 0.12)
        self.assertEqual(res.telemetry, {"key": "val"})

        # Test backward compatibility of RouteResult
        route_res = RouteResult(path=["Depot"], total_cost=0.0, found=True)
        self.assertIsInstance(route_res, OptimizationResult)

    def test_opti_route_controller_model_management(self):
        """Test OptiRouteController managing graph, fleet, and job queue references."""
        controller = OptiRouteController(graph=self.graph)
        self.assertEqual(controller.graph.node_count(), 4)

        events_logged = []
        controller.register_listener("graph_updated", lambda g: events_logged.append("graph"))
        controller.register_listener("fleet_updated", lambda f: events_logged.append("fleet"))
        controller.register_listener("job_queue_updated", lambda jq: events_logged.append("job_queue"))

        new_graph = RoadGraph()
        controller.set_graph(new_graph)
        controller.set_fleet("MockFleet")
        controller.set_job_queue("MockJobQueue")

        self.assertEqual(events_logged, ["graph", "fleet", "job_queue"])
        self.assertEqual(controller.fleet, "MockFleet")
        self.assertEqual(controller.job_queue, "MockJobQueue")

    def test_opti_route_controller_pipeline_execution(self):
        """Test controller running Dijkstra and Simulated Annealing pipelines."""
        controller = OptiRouteController(graph=self.graph)

        execution_events = []
        controller.register_listener("execution_started", lambda alg: execution_events.append(f"start_{alg}"))
        controller.register_listener("execution_completed", lambda res: execution_events.append("completed"))

        # Configure execution params
        params = ExecutionParams(
            initial_temp=100.0,
            cooling_rate=0.8,
            iterations_per_temp=5,
            seed=42,
        )
        controller.configure_execution(params)

        # Run Dijkstra via controller
        dijkstra_res = controller.run_dijkstra(start_node="Depot", end_node="Customer1")
        self.assertTrue(dijkstra_res.found)
        self.assertEqual(dijkstra_res.algorithm_name, "Dijkstra")

        # Run Simulated Annealing via controller
        sa_res = controller.run_simulated_annealing(
            start_node="Depot",
            end_node="Depot",
            waypoints=["Hub1", "Hub2", "Customer1"],
        )
        self.assertTrue(sa_res.found)
        self.assertEqual(sa_res.algorithm_name, "SimulatedAnnealing")

        # Check listeners received execution start & completed events
        self.assertIn("start_dijkstra", execution_events)
        self.assertIn("start_simulated_annealing", execution_events)
        self.assertEqual(execution_events.count("completed"), 2)

        # Test benchmark payload formatting hook
        payload = controller.get_benchmark_payload()
        self.assertEqual(payload["algorithm"], "SimulatedAnnealing")
        self.assertTrue(payload["success"])
        self.assertIn("total_cost", payload)
        self.assertIn("convergence_history", payload)

    def test_controller_custom_solver_registration(self):
        """Test registering custom solver with controller."""
        controller = OptiRouteController(graph=self.graph)
        controller.register_solver("dummy", DummyCustomSolver)

        res = controller.run_solver("dummy")
        self.assertTrue(res.found)
        self.assertEqual(res.algorithm_name, "DummyCustom")


if __name__ == "__main__":
    unittest.main()
