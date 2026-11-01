"""
Vehicle domain models for Opti-Route (Phase 1, requirement F3).

Contains:
- Vehicle     : one delivery vehicle (capacity, speed, fuel rate, home depot).
- VehicleFleet: the collection of vehicles available for routing.

Neither class has any GUI code, so both can be tested on their own.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple


def _require_text(name: str, value: Any) -> str:
    """Return value stripped of spaces, or raise ValueError if it is not real text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_number(name: str, value: Any, allow_zero: bool) -> float:
    """Return value as a float, or raise ValueError if it is not a valid number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0 or (value == 0 and not allow_zero):
        limit = "not be negative" if allow_zero else "be greater than zero"
        raise ValueError(f"{name} must {limit}.")
    return float(value)


@dataclass(frozen=True)
class Vehicle:
    """
    One delivery vehicle.

    capacity  : maximum total parcel weight it can carry (same unit as job weights).
    speed     : distance units travelled per hour (km/h when the graph is in km).
    fuel_rate : fuel used per distance unit (litres per km when the graph is in km).
    home_depot: id of the graph node where the vehicle starts and ends its route.

    A Vehicle cannot be changed after it is created. To edit one, build a new
    Vehicle and call VehicleFleet.replace().
    """

    id: str
    capacity: float
    speed: float
    fuel_rate: float
    home_depot: str

    def __post_init__(self) -> None:
        # The dataclass is frozen, so object.__setattr__ is used to store the
        # cleaned-up values (stripped text, numbers converted to float).
        object.__setattr__(self, "id", _require_text("Vehicle id", self.id))
        object.__setattr__(self, "home_depot", _require_text("home_depot", self.home_depot))
        object.__setattr__(self, "capacity", _require_number("capacity", self.capacity, False))
        object.__setattr__(self, "speed", _require_number("speed", self.speed, False))
        object.__setattr__(self, "fuel_rate", _require_number("fuel_rate", self.fuel_rate, True))

    def can_carry(self, weight: float) -> bool:
        """True if a load of this weight fits within the vehicle's capacity."""
        return weight <= self.capacity

    def travel_time_hours(self, distance: float) -> float:
        """Hours needed to drive this distance at the vehicle's speed."""
        return distance / self.speed

    def fuel_used(self, distance: float) -> float:
        """Fuel consumed driving this distance."""
        return distance * self.fuel_rate


class VehicleFleet:
    """The set of vehicles available for routing, kept in the order they were added."""

    def __init__(self) -> None:
        self._vehicles: Dict[str, Vehicle] = {}

    # --- editing -------------------------------------------------------

    def add(self, vehicle: Vehicle) -> None:
        """Add a vehicle. Raises ValueError if its id is already used."""
        if not isinstance(vehicle, Vehicle):
            raise TypeError("Only Vehicle objects can be added to a fleet.")
        if vehicle.id in self._vehicles:
            raise ValueError(f"Vehicle '{vehicle.id}' already exists in fleet.")
        self._vehicles[vehicle.id] = vehicle

    def replace(self, vehicle: Vehicle) -> None:
        """Swap in an updated Vehicle with the same id. Keeps its position in the fleet."""
        if not isinstance(vehicle, Vehicle):
            raise TypeError("Only Vehicle objects can be stored in a fleet.")
        if vehicle.id not in self._vehicles:
            raise KeyError(f"Vehicle '{vehicle.id}' does not exist in fleet.")
        self._vehicles[vehicle.id] = vehicle

    def remove(self, vehicle_id: str) -> None:
        """Remove a vehicle by id. Raises KeyError if it is not in the fleet."""
        if vehicle_id not in self._vehicles:
            raise KeyError(f"Vehicle '{vehicle_id}' does not exist in fleet.")
        del self._vehicles[vehicle_id]

    # --- lookup --------------------------------------------------------

    def get(self, vehicle_id: str) -> Optional[Vehicle]:
        """Return the vehicle with this id, or None."""
        return self._vehicles.get(vehicle_id)

    @property
    def vehicles(self) -> Tuple[Vehicle, ...]:
        """Read-only snapshot of all vehicles."""
        return tuple(self._vehicles.values())

    def __len__(self) -> int:
        return len(self._vehicles)

    def __iter__(self) -> Iterator[Vehicle]:
        return iter(self.vehicles)

    def __contains__(self, vehicle_id: object) -> bool:
        return vehicle_id in self._vehicles

    # --- fleet-wide information ---------------------------------------

    @property
    def total_capacity(self) -> float:
        """Combined capacity of every vehicle."""
        return sum(v.capacity for v in self._vehicles.values())

    @property
    def start_depot(self) -> Optional[str]:
        """
        Depot of the first vehicle, or None for an empty fleet. The solvers
        read this attribute to find where a route starts.
        """
        for vehicle in self._vehicles.values():
            return vehicle.home_depot
        return None

    def depot_ids(self) -> Tuple[str, ...]:
        """Distinct home depots used by the fleet, in the order first seen."""
        return tuple(dict.fromkeys(v.home_depot for v in self._vehicles.values()))

    def check_depots(self, graph: Any) -> List[str]:
        """
        Return a message for every vehicle whose home depot is not a node in
        the graph. An empty list means every depot is valid.
        """
        return [
            f"Vehicle '{v.id}' has home depot '{v.home_depot}', which is not in the graph."
            for v in self._vehicles.values()
            if not graph.has_node(v.home_depot)
        ]