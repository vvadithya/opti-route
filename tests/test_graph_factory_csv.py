import csv
import io
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
from opti_route.models.graph_factory import CSV_COLUMNS, GraphFactory, InvalidGraphFileError
from opti_route.models.job import DynamicJobQueue, Job
from opti_route.models.vehicle import Vehicle, VehicleFleet

HEADER = ",".join(CSV_COLUMNS)


def make_project():
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


def as_dict(project):
    return GraphFactory.to_dict(project.graph, project.fleet, project.jobs)


class CsvTestCase(unittest.TestCase):
    """Gives each test a throwaway folder and a helper to write CSV text into it."""

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self):
        self._folder.cleanup()

    def write(self, text, name="map.csv", encoding="utf-8"):
        path = self.folder / name
        path.write_bytes(text.encode(encoding))
        return path

    def load_text(self, text):
        return GraphFactory.load_csv(self.write(text))

    def assert_rejected(self, text, *fragments):
        with self.assertRaises(InvalidGraphFileError) as caught:
            self.load_text(text)
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))


class TestCsvRoundTrip(CsvTestCase):
    def test_save_then_load_gives_back_the_same_project(self):
        graph, fleet, jobs = make_project()
        path = GraphFactory.save_csv(self.folder / "map.csv", graph, fleet, jobs)
        project = GraphFactory.load_csv(path)
        self.assertEqual(as_dict(project), GraphFactory.to_dict(graph, fleet, jobs))
        self.assertEqual(project.graph.distance_unit, "miles")
        self.assertFalse(project.graph.has_edge("B", "A"))  # one-way road stayed one-way

    def test_csv_and_json_describe_the_same_project(self):
        graph, fleet, jobs = make_project()
        from_csv = GraphFactory.load_csv(GraphFactory.save_csv(self.folder / "m.csv", graph, fleet, jobs))
        from_json = GraphFactory.load_json(GraphFactory.save_json(self.folder / "m.json", graph, fleet, jobs))
        self.assertEqual(as_dict(from_csv), as_dict(from_json))

    def test_graph_only_save(self):
        graph, _, _ = make_project()
        project = GraphFactory.load_csv(GraphFactory.save_csv(self.folder / "map.csv", graph))
        self.assertEqual(project.graph.node_count(), 3)
        self.assertEqual(len(project.fleet), 0)
        self.assertEqual(len(project.jobs), 0)

    def test_empty_graph_round_trips(self):
        project = GraphFactory.load_csv(GraphFactory.save_csv(self.folder / "map.csv", RoadGraph()))
        self.assertEqual(project.graph.node_count(), 0)
        self.assertEqual(project.graph.distance_unit, "km")

    def test_awkward_text_survives(self):
        """Commas, quotes, accents and line breaks in a name must come back unchanged."""
        graph = RoadGraph()
        for node_id, name in (
            ("N1", 'Depot, "north" gate'),
            ("N2", "Café Zürich — 北京"),
            ("N3", "two\nlines"),
        ):
            graph.add_node(Node(node_id, 1.0, 2.0, name))
        project = GraphFactory.load_csv(GraphFactory.save_csv(self.folder / "map.csv", graph))
        self.assertEqual(project.graph.get_node("N1").name, 'Depot, "north" gate')
        self.assertEqual(project.graph.get_node("N2").name, "Café Zürich — 北京")
        self.assertEqual(project.graph.get_node("N3").name, "two\nlines")


class TestCsvFileFormat(CsvTestCase):
    def test_file_starts_with_byte_order_mark_and_header(self):
        path = GraphFactory.save_csv(self.folder / "map.csv", *make_project())
        raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(raw[3:].decode("utf-8").splitlines()[0], HEADER)

    def test_rows_are_written_in_a_readable_order(self):
        path = GraphFactory.save_csv(self.folder / "map.csv", *make_project())
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
        self.assertEqual(
            [row["record"] for row in rows],
            ["unit", "node", "node", "node", "edge", "edge", "vehicle", "job", "job"],
        )
        self.assertEqual(rows[0]["value"], "miles")
        self.assertEqual(rows[4]["directed"], "false")
        self.assertEqual(rows[5]["directed"], "true")
        self.assertEqual([r["id"] for r in rows if r["record"] == "job"], ["J0", "J1"])

    def test_every_row_has_every_column(self):
        path = GraphFactory.save_csv(self.folder / "map.csv", *make_project())
        for row in csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))):
            self.assertEqual(len(row), len(CSV_COLUMNS))


class TestHandWrittenCsv(CsvTestCase):
    """Files typed or exported from a spreadsheet will not look like our own output."""

    def test_spreadsheet_style_file(self):
        text = (
            "Record, ID ,X,Y,Name,Type,From,To,Weight,Directed\r\n"
            "\r\n"
            'NODE,A,0,0,"Depot, north",depot,,,,\r\n'
            "node, B ,3,4,,,,,,\r\n"
            "Edge,,,,,,A, B ,5,\r\n"
        )
        project = self.load_text(text)
        self.assertEqual(list(project.graph.nodes), ["A", "B"])
        self.assertEqual(project.graph.get_node("A").name, "Depot, north")
        self.assertEqual(project.graph.get_node("A").node_type, NodeType.DEPOT)
        self.assertEqual(project.graph.get_node("B").node_type, NodeType.JUNCTION)
        self.assertEqual(project.graph.get_neighbors("A"), [("B", 5.0)])

    def test_node_type_must_be_lowercase(self):
        self.assert_rejected("record,id,x,y,type\nnode,A,0,0,Depot\n", "nodes[0]", "junction, depot, hub")

    def test_columns_may_be_in_any_order_and_unused_ones_left_out(self):
        project = self.load_text(
            "weight,to,from,record,y,x,id\n"
            ",,,node,0,0,A\n"
            ",,,node,1,1,B\n"
            "4,B,A,edge,,,\n"
        )
        self.assertEqual(project.graph.edge_count(), 1)
        self.assertEqual(project.graph.get_neighbors("A"), [("B", 4.0)])

    def test_flags_accept_several_spellings(self):
        header = "record,id,x,y,from,to,weight,directed\n"
        nodes = "".join(f"node,N{i},{i},0,,,,\n" for i in range(7))
        edges = "".join(
            f"edge,,,,N{i},N{i + 1},1,{flag}\n"
            for i, flag in enumerate(("TRUE", "yes", "1", "False", "NO", "0"))
        )
        project = self.load_text(header + nodes + edges)
        directions = [e.directed for e in project.graph.edges]
        self.assertEqual(directions, [True, True, True, False, False, False])

    def test_blank_lines_and_header_only_file_are_fine(self):
        self.assertEqual(self.load_text("record,id\n\n\n").graph.node_count(), 0)
        self.assertEqual(self.load_text("record\n").graph.node_count(), 0)

    def test_unit_row_is_optional_and_sets_the_distance_unit(self):
        self.assertEqual(self.load_text("record,id,x,y\nnode,A,0,0\n").graph.distance_unit, "km")
        self.assertEqual(
            self.load_text("record,id,x,y,value\nunit,,,,metres\nnode,A,0,0,\n").graph.distance_unit,
            "metres",
        )

    def test_file_with_byte_order_mark_loads(self):
        path = self.write("record,id,x,y\nnode,A,0,0\n", encoding="utf-8-sig")
        self.assertEqual(GraphFactory.load_csv(path).graph.node_count(), 1)

    def test_vehicle_and_job_rows(self):
        project = self.load_text(
            "record,id,x,y,capacity,speed,fuel_rate,home_depot,pickup,dropoff,weight,deadline\n"
            "node,Depot,0,0,,,,,,,,\n"
            "node,A,1,1,,,,,,,,\n"
            "vehicle,V1,,,80,45,0.2,Depot,,,,\n"
            "job,J1,,,,,,,Depot,A,12.5,6\n"
        )
        vehicle = project.fleet.get("V1")
        self.assertEqual((vehicle.capacity, vehicle.speed, vehicle.fuel_rate), (80.0, 45.0, 0.2))
        job = project.jobs.get("J1")
        self.assertEqual((job.pickup, job.dropoff, job.weight, job.deadline), ("Depot", "A", 12.5, 6.0))


class TestInvalidCsv(CsvTestCase):
    NODES = "record,id,x,y,from,to,weight,directed\nnode,A,0,0,,,,\nnode,B,1,1,,,,\n"

    def test_empty_file(self):
        self.assert_rejected("", "empty")

    def test_missing_record_column(self):
        self.assert_rejected("id,x,y\nA,0,0\n", "'record' column")

    def test_unknown_record_type_names_the_line(self):
        self.assert_rejected("record,id\nplanet,A\n", "line 2", "'planet'", "node, edge")

    def test_a_row_with_no_record_type(self):
        self.assert_rejected("record,id\n,A\n", "line 2", "unknown record type")

    def test_non_numeric_cell_names_line_and_column(self):
        self.assert_rejected("record,id,x,y\nnode,A,abc,0\n", "line 2", "column 'x'", "'abc'")

    def test_bad_flag(self):
        self.assert_rejected(self.NODES + "edge,,,,A,B,2,maybe\n", "line 4", "column 'directed'", "'maybe'")

    def test_row_with_more_cells_than_the_header(self):
        self.assert_rejected("record,id\nnode,A,extra\n", "line 2", "more cells")

    def test_unit_given_twice(self):
        self.assert_rejected("record,value\nunit,km\nunit,miles\n", "line 3", "twice")

    def test_missing_required_cell_is_reported_by_section_and_position(self):
        self.assert_rejected("record,id,x\nnode,A,1\n", "nodes[0]", "'y'")
        self.assert_rejected(self.NODES + "edge,,,,A,B,,\n", "edges[0]", "'weight'")

    def test_nan_and_infinity_are_rejected(self):
        for literal in ("nan", "inf", "-inf", "NaN", "Infinity"):
            with self.subTest(literal=literal):
                self.assert_rejected(f"record,id,x,y\nnode,A,{literal},0\n", "nodes[0]")

    def test_the_shared_rules_apply_to_csv_too(self):
        self.assert_rejected(self.NODES + "node,A,5,5,,,,\n", "nodes[2]", "already exists")
        self.assert_rejected(self.NODES + "edge,,,,A,Z,2,\n", "edges[0]", "'Z'")
        self.assert_rejected(self.NODES + "edge,,,,A,B,-2,\n", "edges[0]", "weight")
        self.assert_rejected(self.NODES + "edge,,,,A,B,2,\nedge,,,,B,A,2,\n", "edges[1]", "already exists")
        # A second header row looks like a record of an unknown type. The header is
        # line 1 and the two nodes are lines 2 and 3, so the stray row is line 4.
        self.assert_rejected(
            self.NODES + "record,id,capacity,speed,fuel_rate,home_depot\n", "line 4", "'record'"
        )

    def test_vehicle_and_job_references_are_checked(self):
        vehicle_header = "record,id,x,y,capacity,speed,fuel_rate,home_depot,pickup,dropoff,weight,deadline\n"
        base = vehicle_header + "node,A,0,0,,,,,,,,\nnode,B,1,1,,,,,,,,\n"
        self.assert_rejected(base + "vehicle,V1,,,10,10,0.1,Nowhere,,,,\n", "vehicles", "'Nowhere'")
        self.assert_rejected(base + "job,J1,,,,,,,A,Nowhere,5,2\n", "jobs", "'Nowhere'")
        self.assert_rejected(base + "job,J1,,,,,,,A,A,5,2\n", "jobs[0]", "different")

    def test_non_text_and_missing_files(self):
        path = self.folder / "binary.csv"
        path.write_bytes(b"\xff\xfe\x00\x80")
        with self.assertRaisesRegex(InvalidGraphFileError, "UTF-8"):
            GraphFactory.load_csv(path)
        with self.assertRaisesRegex(InvalidGraphFileError, "File not found"):
            GraphFactory.load_csv(self.folder / "nope.csv")

    def test_oversized_file_is_refused(self):
        path = self.write("record,id,x,y\nnode,A,0,0\n")
        with patch.object(graph_factory, "MAX_FILE_BYTES", 10):
            with self.assertRaisesRegex(InvalidGraphFileError, "too large"):
                GraphFactory.load_csv(path)

    def test_a_cell_over_the_csv_size_limit_is_reported_not_crashed(self):
        huge = "x" * (csv.field_size_limit() + 10)
        self.assert_rejected(f"record,id,x,y\nnode,{huge},0,0\n", "Invalid CSV")


class TestChoosingTheFormat(CsvTestCase):
    def test_load_and_save_choose_by_extension(self):
        project = make_project()
        for name in ("map.json", "map.csv", "MAP.JSON", "MAP.CSV"):
            with self.subTest(name=name):
                path = GraphFactory.save(self.folder / name, *project)
                self.assertEqual(as_dict(GraphFactory.load(path)), GraphFactory.to_dict(*project))

    def test_saved_files_really_are_in_the_matching_format(self):
        GraphFactory.save(self.folder / "map.json", *make_project())
        GraphFactory.save(self.folder / "map.csv", *make_project())
        json.loads((self.folder / "map.json").read_text(encoding="utf-8"))  # must not raise
        self.assertTrue((self.folder / "map.csv").read_text(encoding="utf-8-sig").startswith("record,"))

    def test_unsupported_extensions_are_refused(self):
        for name in ("map.txt", "map", "map.xlsx", "map.json.bak"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(InvalidGraphFileError, "Unsupported file type"):
                    GraphFactory.load(self.folder / name)
                with self.assertRaisesRegex(InvalidGraphFileError, "Unsupported file type"):
                    GraphFactory.save(self.folder / name, RoadGraph())
        self.assertEqual(list(self.folder.iterdir()), [])  # nothing was written


class TestCsvSaving(CsvTestCase):
    def test_saving_replaces_an_existing_file(self):
        path = self.write("old contents")
        GraphFactory.save_csv(path, *make_project())
        self.assertEqual(GraphFactory.load_csv(path).graph.node_count(), 3)

    def test_no_temporary_file_is_left_behind(self):
        GraphFactory.save_csv(self.folder / "map.csv", *make_project())
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["map.csv"])

    def test_failed_save_keeps_the_old_file_and_cleans_up(self):
        path = self.write("precious old data")
        with patch("opti_route.models.graph_factory.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "Could not save"):
                GraphFactory.save_csv(path, *make_project())
        self.assertEqual(path.read_text(encoding="utf-8"), "precious old data")
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["map.csv"])

    def test_saving_into_a_missing_folder_raises_os_error(self):
        with self.assertRaisesRegex(OSError, "Could not save"):
            GraphFactory.save_csv(self.folder / "missing" / "map.csv", *make_project())

    def test_loaded_csv_map_works_with_the_solver(self):
        from opti_route.solvers.dijkstra_solver import DijkstraSolver

        project = GraphFactory.load_csv(GraphFactory.save_csv(self.folder / "map.csv", *make_project()))
        result = DijkstraSolver().solve(project.graph, start_node="Depot", end_node="B")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["Depot", "A", "B"])
        self.assertEqual(result.total_cost, 10.0)


if __name__ == "__main__":
    unittest.main()