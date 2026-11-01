import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opti_route.models.graph import Node, RoadGraph
from opti_route.models.vehicle import Vehicle, VehicleFleet


def make_vehicle(vehicle_id: str = "V1", **overrides) -> Vehicle:
    values = {"capacity": 100.0, "speed": 50.0, "fuel_rate": 0.1, "home_depot": "Depot"}
    values.update(overrides)
    return Vehicle(vehicle_id, **values)


class TestVehicle(unittest.TestCase):
    def test_valid_vehicle_stores_its_values(self):
        vehicle = make_vehicle()
        self.assertEqual(vehicle.id, "V1")
        self.assertEqual(vehicle.capacity, 100.0)
        self.assertEqual(vehicle.speed, 50.0)
        self.assertEqual(vehicle.fuel_rate, 0.1)
        self.assertEqual(vehicle.home_depot, "Depot")

    def test_integers_become_floats_and_text_is_stripped(self):
        vehicle = Vehicle("  V9  ", 80, 40, 1, "  Base ")
        self.assertEqual(vehicle.id, "V9")
        self.assertEqual(vehicle.home_depot, "Base")
        self.assertIsInstance(vehicle.capacity, float)
        self.assertIsInstance(vehicle.speed, float)

    def test_zero_fuel_rate_is_allowed(self):
        self.assertEqual(make_vehicle(fuel_rate=0).fuel_rate, 0.0)

    def test_bad_text_fields_rejected(self):
        with self.assertRaises(ValueError):
            Vehicle("", 1, 1, 1, "Depot")
        with self.assertRaises(ValueError):
            Vehicle("   ", 1, 1, 1, "Depot")
        with self.assertRaises(ValueError):
            Vehicle(5, 1, 1, 1, "Depot")
        with self.assertRaises(ValueError):
            Vehicle("V1", 1, 1, 1, "")
        with self.assertRaises(ValueError):
            Vehicle("V1", 1, 1, 1, None)

    def test_bad_numbers_rejected(self):
        for field in ("capacity", "speed"):
            for bad in (0, -1, float("nan"), float("inf"), "10", None, True):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ValueError):
                        make_vehicle(**{field: bad})
        for bad in (-0.5, float("nan"), float("inf"), "0.1", None, False):
            with self.subTest(field="fuel_rate", value=bad):
                with self.assertRaises(ValueError):
                    make_vehicle(fuel_rate=bad)

    def test_vehicle_cannot_be_changed_after_creation(self):
        vehicle = make_vehicle()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            vehicle.capacity = 5.0

    def test_can_carry(self):
        vehicle = make_vehicle(capacity=100.0)
        self.assertTrue(vehicle.can_carry(60.0))
        self.assertTrue(vehicle.can_carry(100.0))  # exactly full is allowed
        self.assertFalse(vehicle.can_carry(100.1))

    def test_travel_time_and_fuel(self):
        vehicle = make_vehicle(speed=50.0, fuel_rate=0.2)
        self.assertAlmostEqual(vehicle.travel_time_hours(25.0), 0.5)
        self.assertAlmostEqual(vehicle.fuel_used(30.0), 6.0)
        self.assertEqual(vehicle.fuel_used(0.0), 0.0)

    def test_equal_vehicles_compare_equal(self):
        self.assertEqual(make_vehicle(), make_vehicle())
        self.assertNotEqual(make_vehicle(), make_vehicle(capacity=5.0))


class TestVehicleFleet(unittest.TestCase):
    def test_empty_fleet(self):
        fleet = VehicleFleet()
        self.assertEqual(len(fleet), 0)
        self.assertEqual(fleet.vehicles, ())
        self.assertEqual(fleet.total_capacity, 0)
        self.assertIsNone(fleet.start_depot)
        self.assertEqual(fleet.depot_ids(), ())

    def test_add_get_contains_and_iterate_in_order(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V2"))
        fleet.add(make_vehicle("V1"))
        self.assertEqual(len(fleet), 2)
        self.assertIn("V1", fleet)
        self.assertNotIn("V3", fleet)
        self.assertEqual(fleet.get("V1").id, "V1")
        self.assertIsNone(fleet.get("V3"))
        self.assertEqual([v.id for v in fleet], ["V2", "V1"])  # order added

    def test_duplicate_id_rejected(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1"))
        with self.assertRaises(ValueError):
            fleet.add(make_vehicle("V1", capacity=5.0))
        self.assertEqual(len(fleet), 1)
        self.assertEqual(fleet.get("V1").capacity, 100.0)

    def test_only_vehicle_objects_accepted(self):
        fleet = VehicleFleet()
        with self.assertRaises(TypeError):
            fleet.add("V1")
        with self.assertRaises(TypeError):
            fleet.replace({"id": "V1"})

    def test_remove(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1"))
        fleet.remove("V1")
        self.assertEqual(len(fleet), 0)
        with self.assertRaises(KeyError):
            fleet.remove("V1")

    def test_replace_keeps_position(self):
        fleet = VehicleFleet()
        for vehicle_id in ("V1", "V2", "V3"):
            fleet.add(make_vehicle(vehicle_id))
        fleet.replace(make_vehicle("V2", capacity=250.0))
        self.assertEqual(fleet.get("V2").capacity, 250.0)
        self.assertEqual([v.id for v in fleet], ["V1", "V2", "V3"])
        with self.assertRaises(KeyError):
            fleet.replace(make_vehicle("V9"))

    def test_vehicles_property_is_a_snapshot(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1"))
        snapshot = fleet.vehicles
        fleet.add(make_vehicle("V2"))
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(fleet.vehicles), 2)

    def test_total_capacity(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1", capacity=100.0))
        fleet.add(make_vehicle("V2", capacity=50.5))
        self.assertAlmostEqual(fleet.total_capacity, 150.5)

    def test_start_depot_is_first_vehicles_depot(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1", home_depot="North"))
        fleet.add(make_vehicle("V2", home_depot="South"))
        self.assertEqual(fleet.start_depot, "North")
        fleet.remove("V1")
        self.assertEqual(fleet.start_depot, "South")

    def test_depot_ids_are_distinct_and_ordered(self):
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1", home_depot="North"))
        fleet.add(make_vehicle("V2", home_depot="South"))
        fleet.add(make_vehicle("V3", home_depot="North"))
        self.assertEqual(fleet.depot_ids(), ("North", "South"))

    def test_check_depots_reports_missing_nodes(self):
        graph = RoadGraph()
        graph.add_node(Node("Depot", 0.0, 0.0))
        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1", home_depot="Depot"))
        self.assertEqual(fleet.check_depots(graph), [])

        fleet.add(make_vehicle("V2", home_depot="Nowhere"))
        problems = fleet.check_depots(graph)
        self.assertEqual(len(problems), 1)
        self.assertIn("V2", problems[0])
        self.assertIn("Nowhere", problems[0])

    def test_fleet_works_with_the_existing_solver_interface(self):
        """The solvers read fleet.start_depot, so a real fleet must drive them."""
        from opti_route.solvers.dijkstra_solver import DijkstraSolver

        graph = RoadGraph()
        graph.add_node(Node("Depot", 0.0, 0.0))
        graph.add_node(Node("Shop", 3.0, 4.0))
        graph.add_edge("Depot", "Shop", 5.0)

        fleet = VehicleFleet()
        fleet.add(make_vehicle("V1", home_depot="Depot"))
        result = DijkstraSolver().solve(graph, fleet=fleet, end_node="Shop")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["Depot", "Shop"])
        self.assertEqual(result.total_cost, 5.0)


if __name__ == "__main__":
    unittest.main()