import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Node, RoadGraph
from opti_route.models.job import DynamicJobQueue, Job
from opti_route.models.vehicle import Vehicle, VehicleFleet


def make_job(job_id: str = "J1", **overrides) -> Job:
    values = {"pickup": "A", "dropoff": "B", "weight": 10.0, "deadline": 5.0}
    values.update(overrides)
    return Job(job_id, **values)


def ids(jobs) -> list:
    return [job.id for job in jobs]


class TestJob(unittest.TestCase):
    def test_valid_job_stores_its_values(self):
        job = make_job()
        self.assertEqual(job.id, "J1")
        self.assertEqual(job.pickup, "A")
        self.assertEqual(job.dropoff, "B")
        self.assertEqual(job.weight, 10.0)
        self.assertEqual(job.deadline, 5.0)

    def test_integers_become_floats_and_text_is_stripped(self):
        job = Job("  J9 ", " A ", " B ", 3, 8)
        self.assertEqual((job.id, job.pickup, job.dropoff), ("J9", "A", "B"))
        self.assertIsInstance(job.weight, float)
        self.assertIsInstance(job.deadline, float)

    def test_bad_text_fields_rejected(self):
        for field in ("pickup", "dropoff"):
            for bad in ("", "   ", None, 5):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ValueError):
                        make_job(**{field: bad})
        with self.assertRaises(ValueError):
            Job("", "A", "B", 1, 1)

    def test_bad_numbers_rejected(self):
        for field in ("weight", "deadline"):
            for bad in (0, -1, float("nan"), float("inf"), "10", None, True):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ValueError):
                        make_job(**{field: bad})

    def test_pickup_and_dropoff_must_differ(self):
        with self.assertRaises(ValueError):
            make_job(pickup="A", dropoff="A")

    def test_job_cannot_be_changed_after_creation(self):
        job = make_job()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            job.weight = 1.0

    def test_is_late(self):
        job = make_job(deadline=5.0)
        self.assertFalse(job.is_late(4.9))
        self.assertFalse(job.is_late(5.0))  # arriving exactly on time is fine
        self.assertTrue(job.is_late(5.1))

    def test_check_against_graph(self):
        graph = RoadGraph()
        graph.add_node(Node("A", 0.0, 0.0))
        graph.add_node(Node("B", 1.0, 1.0))
        self.assertEqual(make_job().check_against(graph), [])

        problems = make_job(pickup="X", dropoff="Y").check_against(graph)
        self.assertEqual(len(problems), 2)
        self.assertIn("pickup 'X'", problems[0])
        self.assertIn("dropoff 'Y'", problems[1])


class TestDynamicJobQueue(unittest.TestCase):
    def test_empty_queue(self):
        queue = DynamicJobQueue()
        self.assertEqual(len(queue), 0)
        self.assertIsNone(queue.peek())
        self.assertEqual(queue.all_jobs(), ())
        self.assertEqual(queue.total_weight, 0)
        self.assertEqual(queue.demand_by_node(), {})
        self.assertEqual(queue.waypoints(), ())
        with self.assertRaises(IndexError):
            queue.pop()

    def test_jobs_are_ordered_by_earliest_deadline(self):
        queue = DynamicJobQueue()
        queue.add(make_job("late", deadline=9.0))
        queue.add(make_job("soon", deadline=1.0))
        queue.add(make_job("middle", deadline=4.0))
        self.assertEqual(ids(queue.all_jobs()), ["soon", "middle", "late"])
        self.assertEqual(ids(queue), ["soon", "middle", "late"])  # iteration agrees
        self.assertEqual(queue.peek().id, "soon")

    def test_equal_deadlines_keep_the_order_added(self):
        queue = DynamicJobQueue()
        for job_id in ("first", "second", "third"):
            queue.add(make_job(job_id, deadline=3.0))
        self.assertEqual(ids(queue.all_jobs()), ["first", "second", "third"])

    def test_peek_does_not_remove(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1"))
        queue.peek()
        self.assertEqual(len(queue), 1)

    def test_pop_returns_jobs_in_deadline_order(self):
        queue = DynamicJobQueue()
        queue.add(make_job("b", deadline=2.0))
        queue.add(make_job("a", deadline=1.0))
        self.assertEqual(queue.pop().id, "a")
        self.assertEqual(queue.pop().id, "b")
        self.assertEqual(len(queue), 0)

    def test_jobs_can_be_added_after_others_were_removed(self):
        """'Dynamic': the queue keeps working while jobs come and go."""
        queue = DynamicJobQueue()
        queue.add(make_job("J1", deadline=5.0))
        queue.add(make_job("J2", deadline=6.0))
        queue.pop()
        queue.add(make_job("J3", deadline=1.0))
        self.assertEqual(ids(queue.all_jobs()), ["J3", "J2"])

    def test_duplicate_id_rejected(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1", weight=10.0))
        with self.assertRaises(ValueError):
            queue.add(make_job("J1", weight=99.0))
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.get("J1").weight, 10.0)

    def test_only_job_objects_accepted(self):
        with self.assertRaises(TypeError):
            DynamicJobQueue().add("J1")

    def test_remove_returns_the_job(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1"))
        removed = queue.remove("J1")
        self.assertEqual(removed.id, "J1")
        self.assertNotIn("J1", queue)
        with self.assertRaises(KeyError):
            queue.remove("J1")

    def test_get_and_contains(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1"))
        self.assertIn("J1", queue)
        self.assertNotIn("J2", queue)
        self.assertEqual(queue.get("J1").id, "J1")
        self.assertIsNone(queue.get("J2"))

    def test_clear(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1"))
        queue.add(make_job("J2"))
        queue.clear()
        self.assertEqual(len(queue), 0)
        queue.add(make_job("J1"))  # ids are free to reuse after clearing
        self.assertEqual(len(queue), 1)

    def test_all_jobs_is_a_snapshot(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1"))
        snapshot = queue.all_jobs()
        queue.add(make_job("J2"))
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(queue.all_jobs()), 2)

    def test_total_weight(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1", weight=10.0))
        queue.add(make_job("J2", weight=2.5))
        self.assertAlmostEqual(queue.total_weight, 12.5)

    def test_demand_by_node_adds_up_parcels_for_the_same_dropoff(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1", dropoff="B", weight=10.0))
        queue.add(make_job("J2", dropoff="B", weight=5.0))
        queue.add(make_job("J3", dropoff="C", weight=7.0))
        self.assertEqual(queue.demand_by_node(), {"B": 15.0, "C": 7.0})

    def test_waypoints_are_distinct_and_in_deadline_order(self):
        queue = DynamicJobQueue()
        queue.add(make_job("J1", pickup="A", dropoff="B", deadline=5.0))
        queue.add(make_job("J2", pickup="C", dropoff="B", deadline=2.0))
        queue.add(make_job("J3", pickup="A", dropoff="D", deadline=2.0))
        # J2 (deadline 2, added first), then J3 (deadline 2), then J1 (deadline 5).
        # Stops C,B then A,D then A,B -> duplicates dropped, first sighting kept.
        self.assertEqual(queue.waypoints(), ("C", "B", "A", "D"))

    def test_check_against_graph_collects_every_problem(self):
        graph = RoadGraph()
        graph.add_node(Node("A", 0.0, 0.0))
        graph.add_node(Node("B", 1.0, 1.0))
        queue = DynamicJobQueue()
        queue.add(make_job("ok", pickup="A", dropoff="B"))
        self.assertEqual(queue.check_against(graph), [])
        queue.add(make_job("bad", pickup="A", dropoff="Nowhere"))
        problems = queue.check_against(graph)
        self.assertEqual(len(problems), 1)
        self.assertIn("bad", problems[0])
        self.assertIn("Nowhere", problems[0])

    def test_check_against_fleet(self):
        queue = DynamicJobQueue()
        fleet = VehicleFleet()
        self.assertEqual(queue.check_against_fleet(fleet), [])  # nothing to carry

        queue.add(make_job("light", weight=10.0))
        self.assertEqual(len(queue.check_against_fleet(fleet)), 1)  # no vehicles at all

        fleet.add(Vehicle("V1", 50.0, 40.0, 0.1, "A"))
        fleet.add(Vehicle("V2", 80.0, 40.0, 0.1, "A"))
        self.assertEqual(queue.check_against_fleet(fleet), [])

        queue.add(make_job("heavy", weight=81.0))
        problems = queue.check_against_fleet(fleet)
        self.assertEqual(len(problems), 1)
        self.assertIn("heavy", problems[0])

    def test_listener_receives_each_kind_of_change(self):
        queue = DynamicJobQueue()
        events = []
        queue.register_listener(lambda event, **details: events.append((event, details)))
        queue.add(make_job("J1"))
        queue.add(make_job("J2"))
        queue.remove("J1")
        queue.pop()
        queue.clear()
        self.assertEqual(
            events,
            [
                ("job_added", {"job_id": "J1"}),
                ("job_added", {"job_id": "J2"}),
                ("job_removed", {"job_id": "J1"}),
                ("job_removed", {"job_id": "J2"}),
                ("queue_cleared", {}),
            ],
        )

    def test_failed_changes_do_not_notify_and_unregister_works(self):
        queue = DynamicJobQueue()
        events = []

        def listener(event, **details):
            events.append(event)

        queue.register_listener(listener)
        queue.register_listener(listener)  # twice must not double-notify
        queue.add(make_job("J1"))
        with self.assertRaises(ValueError):
            queue.add(make_job("J1"))
        with self.assertRaises(KeyError):
            queue.remove("missing")
        self.assertEqual(events, ["job_added"])
        queue.unregister_listener(listener)
        queue.add(make_job("J2"))
        self.assertEqual(events, ["job_added"])

    def test_queue_drives_the_simulated_annealing_solver(self):
        """waypoints() must be directly usable by Adithya's solver."""
        from opti_route.solvers.simulated_annealing import SimulatedAnnealingSolver

        graph = RoadGraph()
        for node_id, x in (("Depot", 0.0), ("A", 1.0), ("B", 2.0), ("C", 3.0)):
            graph.add_node(Node(node_id, x, 0.0))
        for first, second in (("Depot", "A"), ("A", "B"), ("B", "C"), ("C", "Depot")):
            graph.add_edge(first, second, 1.0)

        queue = DynamicJobQueue()
        queue.add(make_job("J1", pickup="A", dropoff="B", deadline=3.0))
        queue.add(make_job("J2", pickup="B", dropoff="C", deadline=4.0))
        self.assertEqual(queue.check_against(graph), [])

        solver = SimulatedAnnealingSolver(
            initial_temp=5.0, cooling_rate=0.5, min_temp=1.0, iterations_per_temp=3, seed=1
        )
        result = solver.solve(
            graph,
            start_node="Depot",
            end_node="Depot",
            waypoints=list(queue.waypoints()),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], "Depot")
        self.assertEqual(result.path[-1], "Depot")
        for stop in queue.waypoints():
            self.assertIn(stop, result.path)


if __name__ == "__main__":
    unittest.main()