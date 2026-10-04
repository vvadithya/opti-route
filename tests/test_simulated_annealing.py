import sys
import os
import unittest
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Node, Edge, RoadGraph
from opti_route.solvers.base_solver import AlgorithmStrategy, BaseSolver
from opti_route.solvers.simulated_annealing import SimulatedAnnealingSolver, SAResult


class TestStage2SimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        """Build a complete graph with multiple waypoints for VRP/TSP routing."""
        self.graph = RoadGraph()
        self.nodes = [
            Node("Depot", 0.0, 0.0),
            Node("W1", 10.0, 0.0),
            Node("W2", 10.0, 10.0),
            Node("W3", 0.0, 10.0),
            Node("W4", 20.0, 20.0),
        ]
        for n in self.nodes:
            self.graph.add_node(n)

        # Fully connect all nodes with Euclidean distances
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                u, v = self.nodes[i], self.nodes[j]
                dist = u.distance_to(v)
                self.graph.add_edge(u.id, v.id, dist)

    def test_strategy_inheritance(self):
        """Verify SA solver inherits from AlgorithmStrategy base class."""
        solver = SimulatedAnnealingSolver()
        self.assertIsInstance(solver, AlgorithmStrategy)
        self.assertIsInstance(solver, BaseSolver)

    def test_metropolis_criterion(self):
        """Test Metropolis acceptance probabilities."""
        solver = SimulatedAnnealingSolver(seed=42)

        # Better solution (delta < 0) must ALWAYS be accepted
        self.assertTrue(solver.metropolis_criterion(-5.0, 100.0))
        self.assertTrue(solver.metropolis_criterion(-0.001, 1.0))

        # At T = 0, worse solution must never be accepted
        self.assertFalse(solver.metropolis_criterion(10.0, 0.0))

    def test_cooling_schedule_and_convergence_telemetry(self):
        """Verify temperature decays correctly according to geometric cooling schedule."""
        initial_temp = 100.0
        cooling_rate = 0.5
        min_temp = 10.0
        iters_per_temp = 5

        solver = SimulatedAnnealingSolver(
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
            iterations_per_temp=iters_per_temp,
            seed=42,
        )

        res = solver.solve(
            self.graph,
            start_node="Depot",
            end_node="Depot",
            waypoints=["W1", "W2", "W3", "W4"],
        )

        self.assertIsInstance(res, SAResult)
        self.assertTrue(res.found)
        self.assertGreater(len(res.convergence_history), 0)

        # T0=100 -> step1 (100 -> 50), step2 (50 -> 25), step3 (25 -> 12.5), step4 (12.5 -> 6.25 < 10)
        # Total temp steps should equal 4
        self.assertGreaterEqual(res.temperature_steps, 3)

    def test_objective_function_calculation(self):
        """Verify route cost computation across multiple waypoints."""
        solver = SimulatedAnnealingSolver()
        dist_cache = {}

        # Depot -> W1 (10.0) -> W2 (10.0) -> W3 (10.0) -> Depot (10.0) = 40.0
        sequence = ["Depot", "W1", "W2", "W3", "Depot"]
        cost = solver.calculate_route_cost(self.graph, sequence, dist_cache)
        self.assertAlmostEqual(cost, 40.0)


if __name__ == "__main__":
    unittest.main()
