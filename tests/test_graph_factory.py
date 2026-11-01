import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models import graph_factory
from opti_route.models.graph import Node, NodeType, RoadGraph
from opti_route.models.graph_factory import GraphFactory, InvalidGraphFileError, ProjectData
from opti_route.models.job import DynamicJobQueue, Job
from opti_route.models.vehicle import Vehicle, VehicleFleet


def make_project():
    """A small project exercising every field: types, one-way road, fleet and jobs."""
    graph = RoadGraph(distance_unit="miles")
    graph.add_node(Node("Depot", 0.0, 0.0, "Main depot", NodeType.DEPOT))
    graph.add_node(Node("A", 3.0, 4.0))
    graph.add_node(Node("B", 6.0, 8.0, node_type=NodeType.HUB))
    graph.add_edge("Depot", "A", 5.0)
    graph.add_edge("A", "B", 5.0, directed=True)

    fleet = VehicleFleet()
    fleet.add(Vehicle("V1", 50.0, 40.0, 0.1, "Depot"))

    jobs = DynamicJobQueue()
    jobs.add(Job("J1", "A", "B", 20.0, 3.0))
    jobs.add(Job("J0", "Depot", "A", 5.0, 1.0))
    return graph, fleet, jobs


def valid_dict():
    return GraphFactory.to_dict(*make_project())


def changed(edit):
    """A deep copy of the valid dictionary with `edit(copy)` applied."""
    data = copy.deepcopy(valid_dict())
    edit(data)
    return data


class TestRoundTrip(unittest.TestCase):
    def test_save_then_load_gives_back_the_same_project(self):
        graph, fleet, jobs = make_project()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.json"
            returned = GraphFactory.save_json(path, graph, fleet, jobs)
            self.assertEqual(returned, path)
            project = GraphFactory.load_json(path)

        self.assertIsInstance(project, ProjectData)
        self.assertEqual(project.graph.distance_unit, "miles")
        self.assertEqual(list(project.graph.nodes), ["Depot", "A", "B"])
        self.assertEqual(project.graph.get_node("Depot").node_type, NodeType.DEPOT)
        self.assertEqual(project.graph.get_node("Depot").name, "Main depot")
        self.assertEqual(project.graph.get_node("B").node_type, NodeType.HUB)
        self.assertEqual(project.graph.edge_count(), 2)
        self.assertTrue(project.graph.has_edge("A", "B"))
        self.assertFalse(project.graph.has_edge("B", "A"))  # one-way road stayed one-way
        self.assertEqual(project.fleet.vehicles, fleet.vehicles)
        self.assertEqual(project.jobs.all_jobs(), jobs.all_jobs())
        self.assertEqual(project.warnings, [])

    def test_loaded_project_converts_back_to_the_same_dictionary(self):
        graph, fleet, jobs = make_project()
        project = GraphFactory.from_dict(GraphFactory.to_dict(graph, fleet, jobs))
        self.assertEqual(
            GraphFactory.to_dict(project.graph, project.fleet, project.jobs),
            GraphFactory.to_dict(graph, fleet, jobs),
        )

    def test_saved_file_is_readable_json_with_expected_layout(self):
        graph, fleet, jobs = make_project()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.json"
            GraphFactory.save_json(path, graph, fleet, jobs)
            document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["nodes"][0]["type"], "depot")
        self.assertEqual(document["edges"][1]["directed"], True)
        self.assertEqual([job["id"] for job in document["jobs"]], ["J0", "J1"])  # deadline order

    def test_graph_only_save_has_empty_fleet_and_jobs(self):
        graph, _, _ = make_project()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.json"
            GraphFactory.save_json(path, graph)
            project = GraphFactory.load_json(path)
        self.assertEqual(project.graph.node_count(), 3)
        self.assertEqual(len(project.fleet), 0)
        self.assertEqual(len(project.jobs), 0)

    def test_empty_graph_round_trips(self):
        project = GraphFactory.from_dict(GraphFactory.to_dict(RoadGraph()))
        self.assertEqual(project.graph.node_count(), 0)
        self.assertEqual(project.graph.edge_count(), 0)

    def test_loaded_map_works_with_the_solver(self):
        from opti_route.solvers.dijkstra_solver import DijkstraSolver

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.json"
            GraphFactory.save_json(path, *make_project())
            project = GraphFactory.load_json(path)
        result = DijkstraSolver().solve(project.graph, start_node="Depot", end_node="B")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["Depot", "A", "B"])
        self.assertEqual(result.total_cost, 10.0)


class TestOptionalFieldsAndDefaults(unittest.TestCase):
    def test_minimal_file_uses_defaults(self):
        project = GraphFactory.from_dict(
            {
                "nodes": [{"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 1, "y": 1}],
                "edges": [{"from": "A", "to": "B", "weight": 2}],
            }
        )
        self.assertEqual(project.graph.distance_unit, "km")
        self.assertEqual(project.graph.get_node("A").name, "")
        self.assertEqual(project.graph.get_node("A").node_type, NodeType.JUNCTION)
        self.assertTrue(project.graph.has_edge("B", "A"))  # undirected by default
        self.assertEqual(len(project.fleet), 0)
        self.assertEqual(len(project.jobs), 0)

    def test_unknown_extra_keys_are_ignored(self):
        data = changed(lambda d: d.update(comment="made by hand"))
        self.assertEqual(GraphFactory.from_dict(data).graph.node_count(), 3)

    def test_job_heavier_than_every_vehicle_is_a_warning_not_an_error(self):
        data = changed(lambda d: d["jobs"][0].update(weight=500.0))
        project = GraphFactory.from_dict(data)
        self.assertEqual(len(project.jobs), 2)
        self.assertEqual(len(project.warnings), 1)
        self.assertIn("500", project.warnings[0])


class TestInvalidStructure(unittest.TestCase):
    def assert_rejected(self, data, *fragments):
        with self.assertRaises(InvalidGraphFileError) as caught:
            GraphFactory.from_dict(data)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_top_level_must_be_an_object(self):
        for bad in ([], [1, 2], "text", 5, None):
            with self.subTest(value=bad):
                self.assert_rejected(bad, "top level")

    def test_required_sections(self):
        self.assert_rejected({"edges": []}, "Missing required section 'nodes'")
        self.assert_rejected({"nodes": []}, "Missing required section 'edges'")

    def test_sections_must_be_lists(self):
        for section in ("nodes", "edges", "vehicles", "jobs"):
            with self.subTest(section=section):
                data = changed(lambda d: d.update({section: {}}))
                self.assert_rejected(data, f"'{section}' must be a list")

    def test_unsupported_version(self):
        self.assert_rejected(changed(lambda d: d.update(version=2)), "version")

    def test_bad_distance_unit(self):
        self.assert_rejected(changed(lambda d: d.update(distance_unit="")), "distance_unit")
        self.assert_rejected(changed(lambda d: d.update(distance_unit=5)), "distance_unit")

    def test_entries_must_be_objects(self):
        for section in ("nodes", "edges", "vehicles", "jobs"):
            with self.subTest(section=section):
                data = changed(lambda d: d[section].append(5))
                self.assert_rejected(data, "must be an object")


class TestInvalidNodes(unittest.TestCase):
    def assert_rejected(self, edit, *fragments):
        with self.assertRaises(InvalidGraphFileError) as caught:
            GraphFactory.from_dict(changed(edit))
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_missing_fields_name_the_entry_and_field(self):
        for key in ("id", "x", "y"):
            with self.subTest(field=key):
                self.assert_rejected(lambda d: d["nodes"][1].pop(key), "nodes[1]", f"'{key}'")

    def test_duplicate_node_id(self):
        self.assert_rejected(lambda d: d["nodes"].append(dict(d["nodes"][0])), "nodes[3]", "already exists")

    def test_bad_coordinates(self):
        for bad in ("5", None, True, [1]):
            with self.subTest(value=bad):
                self.assert_rejected(lambda d: d["nodes"][1].update(x=bad), "nodes[1]", "coordinates")

    def test_bad_id(self):
        for bad in ("", "  ", 7, None):
            with self.subTest(value=bad):
                self.assert_rejected(lambda d: d["nodes"][1].update(id=bad), "nodes[1]")

    def test_bad_name_and_type(self):
        self.assert_rejected(lambda d: d["nodes"][1].update(name=5), "nodes[1]", "'name'")
        self.assert_rejected(lambda d: d["nodes"][1].update(type="castle"), "nodes[1]", "junction, depot, hub")
        self.assert_rejected(lambda d: d["nodes"][1].update(type=["depot"]), "nodes[1]", "'type'")


class TestInvalidEdges(unittest.TestCase):
    def assert_rejected(self, edit, *fragments):
        with self.assertRaises(InvalidGraphFileError) as caught:
            GraphFactory.from_dict(changed(edit))
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_missing_fields(self):
        for key in ("from", "to", "weight"):
            with self.subTest(field=key):
                self.assert_rejected(lambda d: d["edges"][0].pop(key), "edges[0]", f"'{key}'")

    def test_edge_to_unknown_node(self):
        self.assert_rejected(lambda d: d["edges"][0].update({"to": "Z"}), "edges[0]", "'Z'")
        self.assert_rejected(lambda d: d["edges"][0].update({"from": "Z"}), "edges[0]", "'Z'")

    def test_bad_weights(self):
        for bad in (0, -1, "5", None, True):
            with self.subTest(value=bad):
                self.assert_rejected(lambda d: d["edges"][0].update(weight=bad), "edges[0]", "weight")

    def test_self_loop(self):
        self.assert_rejected(lambda d: d["edges"][0].update({"to": "Depot"}), "edges[0]", "itself")

    def test_directed_must_be_a_boolean(self):
        for bad in ("yes", 1, None):
            with self.subTest(value=bad):
                self.assert_rejected(lambda d: d["edges"][0].update(directed=bad), "edges[0]", "'directed'")

    def test_duplicate_roads_rejected(self):
        two_way = {"from": "A", "to": "Depot", "weight": 1.0, "directed": False}
        self.assert_rejected(lambda d: d["edges"].append(two_way), "edges[2]", "already exists")
        one_way_over_two_way = {"from": "Depot", "to": "A", "weight": 1.0, "directed": True}
        self.assert_rejected(lambda d: d["edges"].append(one_way_over_two_way), "edges[2]", "already exists")

    def test_opposite_one_way_roads_are_allowed(self):
        data = changed(
            lambda d: d["edges"].extend(
                [
                    {"from": "B", "to": "A", "weight": 7.0, "directed": True},
                ]
            )
        )
        project = GraphFactory.from_dict(data)
        self.assertTrue(project.graph.has_edge("A", "B"))
        self.assertTrue(project.graph.has_edge("B", "A"))


class TestInvalidFleetAndJobs(unittest.TestCase):
    def assert_rejected(self, edit, *fragments):
        with self.assertRaises(InvalidGraphFileError) as caught:
            GraphFactory.from_dict(changed(edit))
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_vehicle_missing_field(self):
        for key in ("id", "capacity", "speed", "fuel_rate", "home_depot"):
            with self.subTest(field=key):
                self.assert_rejected(lambda d: d["vehicles"][0].pop(key), "vehicles[0]", f"'{key}'")

    def test_vehicle_bad_values(self):
        self.assert_rejected(lambda d: d["vehicles"][0].update(capacity=-5), "vehicles[0]", "capacity")
        self.assert_rejected(lambda d: d["vehicles"][0].update(speed=0), "vehicles[0]", "speed")
        self.assert_rejected(lambda d: d["vehicles"][0].update(fuel_rate="x"), "vehicles[0]", "fuel_rate")

    def test_duplicate_vehicle_id(self):
        self.assert_rejected(lambda d: d["vehicles"].append(dict(d["vehicles"][0])), "vehicles[1]", "already exists")

    def test_vehicle_depot_must_be_a_node(self):
        self.assert_rejected(lambda d: d["vehicles"][0].update(home_depot="Nope"), "vehicles", "'Nope'")

    def test_job_missing_field(self):
        for key in ("id", "pickup", "dropoff", "weight", "deadline"):
            with self.subTest(field=key):
                self.assert_rejected(lambda d: d["jobs"][0].pop(key), "jobs[0]", f"'{key}'")

    def test_job_bad_values(self):
        self.assert_rejected(lambda d: d["jobs"][0].update(weight=0), "jobs[0]", "weight")
        self.assert_rejected(lambda d: d["jobs"][0].update(deadline=-1), "jobs[0]", "deadline")
        self.assert_rejected(lambda d: d["jobs"][0].update(dropoff=d["jobs"][0]["pickup"]), "jobs[0]", "different")

    def test_duplicate_job_id(self):
        self.assert_rejected(lambda d: d["jobs"].append(dict(d["jobs"][0])), "jobs[2]", "already exists")

    def test_job_locations_must_be_nodes(self):
        self.assert_rejected(lambda d: d["jobs"][0].update(pickup="Nope"), "jobs", "'Nope'")
        self.assert_rejected(lambda d: d["jobs"][1].update(dropoff="Nope"), "jobs", "'Nope'")


class TestFileLevelProblems(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self):
        self._folder.cleanup()

    def write(self, name, content, mode="w", **kwargs):
        path = self.folder / name
        with path.open(mode, **kwargs) as handle:
            handle.write(content)
        return path

    def test_missing_file(self):
        with self.assertRaisesRegex(InvalidGraphFileError, "File not found"):
            GraphFactory.load_json(self.folder / "nope.json")

    def test_a_folder_is_not_a_file(self):
        with self.assertRaisesRegex(InvalidGraphFileError, "File not found"):
            GraphFactory.load_json(self.folder)

    def test_broken_json_reports_the_position(self):
        path = self.write("bad.json", '{"nodes": [')
        with self.assertRaisesRegex(InvalidGraphFileError, "line 1, column"):
            GraphFactory.load_json(path)

    def test_empty_file(self):
        path = self.write("empty.json", "")
        with self.assertRaises(InvalidGraphFileError):
            GraphFactory.load_json(path)

    def test_non_text_file(self):
        path = self.write("binary.json", b"\xff\xfe\x00\x80", mode="wb")
        with self.assertRaisesRegex(InvalidGraphFileError, "UTF-8"):
            GraphFactory.load_json(path)

    def test_nan_and_infinity_are_rejected(self):
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal):
                path = self.write(
                    "nan.json",
                    '{"nodes": [{"id": "A", "x": %s, "y": 0}], "edges": []}' % literal,
                )
                with self.assertRaisesRegex(InvalidGraphFileError, "finite"):
                    GraphFactory.load_json(path)

    def test_oversized_file_is_refused(self):
        path = self.write("big.json", json.dumps(valid_dict()))
        with patch.object(graph_factory, "MAX_FILE_BYTES", 10):
            with self.assertRaisesRegex(InvalidGraphFileError, "too large"):
                GraphFactory.load_json(path)

    def test_file_with_a_byte_order_mark_loads(self):
        path = self.write("bom.json", json.dumps(valid_dict()), encoding="utf-8-sig")
        self.assertEqual(GraphFactory.load_json(path).graph.node_count(), 3)

    def test_failed_load_raises_a_value_error_subclass(self):
        """The GUI can catch InvalidGraphFileError (or plain ValueError) to show a message."""
        with self.assertRaises(ValueError):
            GraphFactory.load_json(self.folder / "nope.json")


class TestSaving(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self):
        self._folder.cleanup()

    def test_saving_replaces_an_existing_file(self):
        path = self.folder / "map.json"
        path.write_text("old contents", encoding="utf-8")
        GraphFactory.save_json(path, *make_project())
        self.assertEqual(GraphFactory.load_json(path).graph.node_count(), 3)

    def test_no_temporary_file_is_left_behind(self):
        GraphFactory.save_json(self.folder / "map.json", *make_project())
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["map.json"])

    def test_failed_save_keeps_the_old_file_and_cleans_up(self):
        path = self.folder / "map.json"
        path.write_text("precious old data", encoding="utf-8")
        with patch("opti_route.models.graph_factory.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "Could not save"):
                GraphFactory.save_json(path, *make_project())
        self.assertEqual(path.read_text(encoding="utf-8"), "precious old data")
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["map.json"])

    def test_saving_into_a_missing_folder_raises_os_error(self):
        with self.assertRaisesRegex(OSError, "Could not save"):
            GraphFactory.save_json(self.folder / "no_such_folder" / "map.json", *make_project())


if __name__ == "__main__":
    unittest.main()