import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Node, RoadGraph
from opti_route.solvers import (
    SimulatedAnnealingSolver,
    DijkstraSolver,
    SolverOrchestrator,
    apply_mutation,
    MutationOperatorType,
    two_opt_reversal,
    node_swap,
    node_insertion,
)


class TestStage3MutationAndOrchestration(unittest.TestCase):
    def setUp(self):
        """Set up a test road network graph."""
        self.graph = RoadGraph()
        nodes = [
            Node("Depot", 0.0, 0.0),
            Node("A", 5.0, 0.0),
            Node("B", 10.0, 5.0),
            Node("C", 5.0, 10.0),
            Node("D", 0.0, 5.0),
        ]
        for n in nodes:
            self.graph.add_node(n)

        # Fully connected graph
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                self.graph.add_edge(u.id, v.id, u.distance_to(v))

        self.route = ["Depot", "A", "B", "C", "D", "Depot"]

    def test_two_opt_reversal(self):
        """Verify 2-Opt reverses intermediate sub-segments while leaving endpoints intact."""
        mutated = two_opt_reversal(self.route)
        self.assertEqual(mutated[0], "Depot")
        self.assertEqual(mutated[-1], "Depot")
        self.assertEqual(sorted(mutated), sorted(self.route))

    def test_customer_swap(self):
        """Verify customer swap modifies internal sequence while preserving endpoints."""
        mutated = node_swap(self.route)
        self.assertEqual(mutated[0], "Depot")
        self.assertEqual(mutated[-1], "Depot")
        self.assertEqual(set(mutated), set(self.route))

    def test_customer_insertion(self):
        """Verify customer insertion moves node while maintaining endpoints and elements."""
        mutated = node_insertion(self.route)
        self.assertEqual(mutated[0], "Depot")
        self.assertEqual(mutated[-1], "Depot")
        self.assertEqual(set(mutated), set(self.route))

    def test_dynamic_operator_switching(self):
        """Test applying specific mutation operators by configuration."""
        route = ["Depot", "1", "2", "3", "4", "Depot"]
        m1 = apply_mutation(route, operator_type="2-opt")
        m2 = apply_mutation(route, operator_type="swap")
        m3 = apply_mutation(route, operator_type="insertion")
        m4 = apply_mutation(route, operator_type="random")

        for m in (m1, m2, m3, m4):
            self.assertEqual(m[0], "Depot")
            self.assertEqual(m[-1], "Depot")
            self.assertEqual(len(m), len(route))

    def test_solver_orchestrator_execution(self):
        """Verify SolverOrchestrator executes solvers, measures metrics, and formats payload."""
        orchestrator = SolverOrchestrator()

        # Run Dijkstra via Orchestrator
        dijkstra_res = orchestrator.execute(
            "dijkstra", self.graph, start_node="Depot", end_node="C"
        )
        self.assertTrue(dijkstra_res.found)
        self.assertGreater(dijkstra_res.execution_time_ms, 0.0)

        # Run Simulated Annealing via Orchestrator with waypoints & specific mutation operator
        sa_res = orchestrator.execute(
            "simulated_annealing",
            self.graph,
            start_node="Depot",
            end_node="Depot",
            waypoints=["A", "B", "C", "D"],
            mutation_operator="2-opt",
            iterations_per_temp=10,
        )
        self.assertTrue(sa_res.found)
        self.assertGreaterEqual(getattr(sa_res, "peak_memory_mb", 0.0), 0.0)

        # Test Benchmark Payload Generation
        payload = orchestrator.format_benchmark_payload(sa_res)
        self.assertEqual(payload["algorithm"], "SimulatedAnnealing")
        self.assertTrue(payload["success"])
        self.assertIn("total_cost", payload)
        self.assertIn("execution_time_ms", payload)
        self.assertIn("peak_memory_mb", payload)
        self.assertIn("convergence_history", payload)


if __name__ == "__main__":
    unittest.main()
