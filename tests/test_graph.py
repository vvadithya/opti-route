import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Edge, Node, NodeType, RoadGraph


def make_graph() -> RoadGraph:
    """Small test map:  A --1-- B --2--> C   (B->C is one-way), plus D alone."""
    graph = RoadGraph()
    graph.add_node(Node("A", 0.0, 0.0))
    graph.add_node(Node("B", 10.0, 0.0))
    graph.add_node(Node("C", 20.0, 0.0))
    graph.add_node(Node("D", 30.0, 0.0))
    graph.add_edge("A", "B", 1.0)
    graph.add_edge("B", "C", 2.0, directed=True)
    return graph


class TestNodes(unittest.TestCase):
    def test_add_node_and_lookup(self):
        graph = make_graph()
        self.assertEqual(graph.node_count(), 4)
        self.assertTrue(graph.has_node("A"))
        self.assertFalse(graph.has_node("Z"))
        self.assertEqual(graph.get_node("B").x, 10.0)
        self.assertIsNone(graph.get_node("Z"))

    def test_duplicate_node_id_rejected(self):
        graph = make_graph()
        with self.assertRaises(ValueError):
            graph.add_node(Node("A", 5.0, 5.0))
        self.assertEqual(graph.node_count(), 4)

    def test_bad_node_rejected(self):
        graph = RoadGraph()
        with self.assertRaises(ValueError):
            graph.add_node(Node("", 0.0, 0.0))
        with self.assertRaises(ValueError):
            graph.add_node(Node("X", float("nan"), 0.0))
        with self.assertRaises(ValueError):
            graph.add_node(Node("X", 0.0, float("inf")))
        self.assertEqual(graph.node_count(), 0)

    def test_node_type_defaults_and_depot_flag(self):
        graph = RoadGraph()
        graph.add_node(Node("J", 0.0, 0.0))
        graph.add_node(Node("Base", 1.0, 1.0, node_type=NodeType.DEPOT))
        self.assertEqual(graph.get_node("J").node_type, NodeType.JUNCTION)
        self.assertFalse(graph.get_node("J").is_depot)
        self.assertTrue(graph.get_node("Base").is_depot)

    def test_move_node(self):
        graph = make_graph()
        graph.move_node("A", 7.0, 8.0)
        self.assertEqual((graph.get_node("A").x, graph.get_node("A").y), (7.0, 8.0))
        # Road length does not change when a node is dragged.
        self.assertEqual(graph.get_neighbors("A"), [("B", 1.0)])

    def test_move_unknown_node_or_bad_coordinates(self):
        graph = make_graph()
        with self.assertRaises(KeyError):
            graph.move_node("Z", 1.0, 1.0)
        with self.assertRaises(ValueError):
            graph.move_node("A", float("nan"), 1.0)
        self.assertEqual(graph.get_node("A").x, 0.0)

    def test_remove_node_removes_its_edges(self):
        graph = make_graph()
        graph.remove_node("B")
        self.assertFalse(graph.has_node("B"))
        self.assertEqual(graph.edge_count(), 0)
        self.assertEqual(graph.get_neighbors("A"), [])
        self.assertEqual(graph.get_neighbors("C"), [])

    def test_remove_unknown_node(self):
        with self.assertRaises(KeyError):
            make_graph().remove_node("Z")


class TestEdges(unittest.TestCase):
    def test_undirected_edge_works_both_ways(self):
        graph = make_graph()
        self.assertIn(("B", 1.0), graph.get_neighbors("A"))
        self.assertIn(("A", 1.0), graph.get_neighbors("B"))

    def test_directed_edge_works_one_way(self):
        graph = make_graph()
        self.assertIn(("C", 2.0), graph.get_neighbors("B"))
        self.assertEqual(graph.get_neighbors("C"), [])
        self.assertTrue(graph.has_edge("B", "C"))
        self.assertFalse(graph.has_edge("C", "B"))

    def test_edge_count_counts_each_road_once(self):
        self.assertEqual(make_graph().edge_count(), 2)

    def test_edges_property_is_a_snapshot(self):
        graph = make_graph()
        snapshot = graph.edges
        self.assertEqual(len(snapshot), 2)
        self.assertIsInstance(snapshot[0], Edge)
        graph.remove_edge("A", "B")
        self.assertEqual(len(snapshot), 2)  # old snapshot unchanged
        self.assertEqual(len(graph.edges), 1)

    def test_invalid_edges_rejected(self):
        graph = make_graph()
        with self.assertRaises(KeyError):
            graph.add_edge("A", "Z", 1.0)
        with self.assertRaises(KeyError):
            graph.add_edge("Z", "A", 1.0)
        with self.assertRaises(ValueError):
            graph.add_edge("A", "D", 0.0)
        with self.assertRaises(ValueError):
            graph.add_edge("A", "D", -3.0)
        with self.assertRaises(ValueError):
            graph.add_edge("A", "D", float("nan"))
        with self.assertRaises(ValueError):
            graph.add_edge("A", "D", float("inf"))
        with self.assertRaises(ValueError):
            graph.add_edge("A", "D", "5")
        with self.assertRaises(ValueError):
            graph.add_edge("A", "A", 1.0)
        self.assertEqual(graph.edge_count(), 2)

    def test_remove_undirected_edge_either_way_round(self):
        graph = make_graph()
        self.assertEqual(graph.remove_edge("B", "A"), 1)
        self.assertFalse(graph.has_edge("A", "B"))
        self.assertEqual(graph.get_neighbors("A"), [])
        self.assertEqual(graph.get_neighbors("B"), [("C", 2.0)])

    def test_remove_directed_edge_only_in_its_direction(self):
        graph = make_graph()
        with self.assertRaises(KeyError):
            graph.remove_edge("C", "B")  # the road only goes B -> C
        graph.remove_edge("B", "C")
        self.assertFalse(graph.has_edge("B", "C"))

    def test_remove_missing_edge(self):
        with self.assertRaises(KeyError):
            make_graph().remove_edge("A", "D")

    def test_parallel_roads_are_both_kept_and_removed_together(self):
        graph = make_graph()
        graph.add_edge("A", "B", 5.0)
        self.assertEqual(graph.edge_count(), 3)
        self.assertEqual(graph.remove_edge("A", "B"), 2)
        self.assertEqual(graph.edge_count(), 1)


class TestAdjacencyMatrix(unittest.TestCase):
    def test_matrix_values(self):
        graph = make_graph()
        index = graph.node_index()
        matrix = graph.adjacency_matrix()
        self.assertEqual(index, {"A": 0, "B": 1, "C": 2, "D": 3})
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix[index["A"], index["A"]], 0.0)
        self.assertEqual(matrix[index["A"], index["B"]], 1.0)
        self.assertEqual(matrix[index["B"], index["A"]], 1.0)
        self.assertEqual(matrix[index["B"], index["C"]], 2.0)
        self.assertTrue(math.isinf(matrix[index["C"], index["B"]]))  # one-way
        self.assertTrue(math.isinf(matrix[index["A"], index["D"]]))  # no road

    def test_matrix_uses_cheapest_parallel_road(self):
        graph = make_graph()
        graph.add_edge("A", "B", 0.5)
        self.assertEqual(graph.adjacency_matrix()[0, 1], 0.5)

    def test_matrix_is_cached_and_rebuilt_after_changes(self):
        graph = make_graph()
        first = graph.adjacency_matrix()
        self.assertIs(first, graph.adjacency_matrix())  # same cached object
        graph.add_edge("C", "D", 4.0)
        second = graph.adjacency_matrix()
        self.assertIsNot(first, second)
        self.assertEqual(second[2, 3], 4.0)

    def test_matrix_shrinks_after_node_removal(self):
        graph = make_graph()
        graph.remove_node("A")
        self.assertEqual(graph.adjacency_matrix().shape, (3, 3))
        self.assertEqual(graph.node_index(), {"B": 0, "C": 1, "D": 2})

    def test_matrix_is_read_only(self):
        matrix = make_graph().adjacency_matrix()
        with self.assertRaises(ValueError):
            matrix[0, 1] = 99.0

    def test_empty_graph_matrix(self):
        self.assertEqual(RoadGraph().adjacency_matrix().shape, (0, 0))


class TestListenersAndSettings(unittest.TestCase):
    def test_listener_receives_each_kind_of_change(self):
        graph = RoadGraph()
        events = []
        graph.register_listener(lambda event, **details: events.append((event, details)))

        graph.add_node(Node("A", 0.0, 0.0))
        graph.add_node(Node("B", 1.0, 0.0))
        graph.add_edge("A", "B", 3.0)
        graph.move_node("A", 2.0, 2.0)
        graph.remove_edge("A", "B")
        graph.remove_node("B")

        self.assertEqual(
            [name for name, _ in events],
            ["node_added", "node_added", "edge_added", "node_moved", "edge_removed", "node_removed"],
        )
        self.assertEqual(events[0][1], {"node_id": "A"})
        self.assertEqual(events[3][1], {"node_id": "A", "x": 2.0, "y": 2.0})

    def test_failed_changes_do_not_notify(self):
        graph = make_graph()
        events = []
        graph.register_listener(lambda event, **details: events.append(event))
        with self.assertRaises(ValueError):
            graph.add_edge("A", "D", -1.0)
        with self.assertRaises(KeyError):
            graph.remove_node("Z")
        self.assertEqual(events, [])

    def test_unregister_listener(self):
        graph = RoadGraph()
        events = []

        def listener(event, **details):
            events.append(event)

        graph.register_listener(listener)
        graph.register_listener(listener)  # registering twice must not double-notify
        graph.add_node(Node("A", 0.0, 0.0))
        graph.unregister_listener(listener)
        graph.add_node(Node("B", 1.0, 0.0))
        self.assertEqual(events, ["node_added"])

    def test_distance_unit(self):
        self.assertEqual(RoadGraph().distance_unit, "km")
        self.assertEqual(RoadGraph(distance_unit="miles").distance_unit, "miles")
        with self.assertRaises(ValueError):
            RoadGraph(distance_unit="  ")


class TestBackwardCompatibility(unittest.TestCase):
    """The solvers, canvas and controller rely on these exact behaviours."""

    def test_nodes_dict_and_neighbors_format(self):
        graph = make_graph()
        self.assertIn("A", graph.nodes)
        self.assertEqual(graph.get_neighbors("A"), [("B", 1.0)])
        self.assertEqual(graph.get_neighbors("does-not-exist"), [])

    def test_node_distance_to(self):
        self.assertEqual(Node("A", 0.0, 0.0).distance_to(Node("B", 3.0, 4.0)), 5.0)

    def test_positional_node_construction_still_works(self):
        node = Node("A", 1.0, 2.0, "Depot A")
        self.assertEqual(node.name, "Depot A")


if __name__ == "__main__":
    unittest.main()