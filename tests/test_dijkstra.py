import sys
import os
import unittest

# Ensure opti-route directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Node, Edge, RoadGraph
from opti_route.solvers.base_solver import AlgorithmStrategy, BaseSolver, RouteResult
from opti_route.solvers.dijkstra_solver import DijkstraSolver



class TestStage1Dijkstra(unittest.TestCase):
    def setUp(self):
        """Set up a sample graph for testing."""
        self.graph = RoadGraph()
        # Add nodes (with x, y coordinates for trajectory interpolation)
        self.nodes = [
            Node(id="A", x=0.0, y=0.0, name="Depot"),
            Node(id="B", x=10.0, y=0.0, name="Hub 1"),
            Node(id="C", x=10.0, y=10.0, name="Hub 2"),
            Node(id="D", x=20.0, y=10.0, name="Customer D"),
            Node(id="E", x=30.0, y=0.0, name="Customer E"),
        ]
        for node in self.nodes:
            self.graph.add_node(node)

        # Add edges
        # A - B: 5
        # A - C: 12
        # B - C: 2
        # B - D: 9
        # C - D: 3
        # D - E: 4
        self.graph.add_edge("A", "B", 5.0)
        self.graph.add_edge("A", "C", 12.0)
        self.graph.add_edge("B", "C", 2.0)
        self.graph.add_edge("B", "D", 9.0)
        self.graph.add_edge("C", "D", 3.0)
        self.graph.add_edge("D", "E", 4.0)

    def test_strategy_interface(self):
        """Verify DijkstraSolver is an instance of AlgorithmStrategy."""
        solver = DijkstraSolver()
        self.assertIsInstance(solver, AlgorithmStrategy)
        self.assertIsInstance(solver, BaseSolver)

    def test_shortest_path_calculation(self):
        """Verify Dijkstra computes exact shortest path and cost from A to D."""
        solver = DijkstraSolver()
        res = solver.solve(self.graph, start_node="A", end_node="D")

        self.assertTrue(res.found)
        # Expected path: A -> B (5) -> C (2) -> D (3) = Total Cost 10.0
        self.assertEqual(res.path, ["A", "B", "C", "D"])
        self.assertAlmostEqual(res.total_cost, 10.0)
        self.assertGreater(res.nodes_visited, 0)
        self.assertGreaterEqual(res.execution_time_ms, 0.0)

    def test_same_start_and_end_node(self):
        """Verify handling when start and end node are identical."""
        solver = DijkstraSolver()
        res = solver.solve(self.graph, start_node="A", end_node="A")

        self.assertTrue(res.found)
        self.assertEqual(res.path, ["A"])
        self.assertEqual(res.total_cost, 0.0)
        self.assertEqual(len(res.trajectory), 1)
        self.assertEqual(res.trajectory[0], (0.0, 0.0))

    def test_trajectory_generation(self):
        """Verify that UI animation trajectory coordinates are generated."""
        solver = DijkstraSolver(steps_per_edge=5)
        res = solver.solve(self.graph, start_node="A", end_node="B")

        self.assertTrue(res.found)
        self.assertEqual(res.path, ["A", "B"])
        # With steps_per_edge=5, for 1 edge segment: 5 interpolated points + 1 end point = 6 points
        self.assertEqual(len(res.trajectory), 6)
        self.assertEqual(res.trajectory[0], (0.0, 0.0))
        self.assertEqual(res.trajectory[-1], (10.0, 0.0))

    def test_unreachable_node(self):
        """Verify solver behavior when target node is disconnected."""
        isolated_node = Node(id="Z", x=100.0, y=100.0, name="Isolated")
        self.graph.add_node(isolated_node)

        solver = DijkstraSolver()
        res = solver.solve(self.graph, start_node="A", end_node="Z")

        self.assertFalse(res.found)
        self.assertEqual(res.path, [])
        self.assertEqual(res.total_cost, float("inf"))
        self.assertEqual(res.trajectory, [])

    def test_invalid_node_error(self):
        """Verify ValueError is raised if non-existent node requested."""
        solver = DijkstraSolver()
        with self.assertRaises(ValueError):
            solver.solve(self.graph, start_node="NON_EXISTENT", end_node="A")


if __name__ == "__main__":
    unittest.main()
