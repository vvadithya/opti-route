"""
Delivery job models for Opti-Route (Phase 1, requirement F4).

Contains:
- Job            : one delivery order (pickup, drop-off, weight, deadline).
- DynamicJobQueue: the orders waiting to be routed. Jobs can be added and
                   removed at any time, and are always listed earliest
                   deadline first.

Neither class has any GUI code, so both can be tested on their own.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Reuse the validation helpers written for Vehicle.
from .vehicle import _require_number, _require_text


@dataclass(frozen=True)
class Job:
    """
    One delivery order.

    pickup  : id of the graph node where the parcel is collected.
    dropoff : id of the graph node where it must be delivered.
    weight  : parcel weight (same unit as vehicle capacity).
    deadline: latest delivery time, in hours from the start of the day.
              Hours match Vehicle.travel_time_hours().

    A Job cannot be changed after it is created. To edit one, remove it from
    the queue and add a new Job.
    """

    id: str
    pickup: str
    dropoff: str
    weight: float
    deadline: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("Job id", self.id))
        object.__setattr__(self, "pickup", _require_text("pickup", self.pickup))
        object.__setattr__(self, "dropoff", _require_text("dropoff", self.dropoff))
        object.__setattr__(self, "weight", _require_number("weight", self.weight, False))
        object.__setattr__(self, "deadline", _require_number("deadline", self.deadline, False))
        if self.pickup == self.dropoff:
            raise ValueError("pickup and dropoff must be different locations.")

    def is_late(self, arrival_time: float) -> bool:
        """True if arriving at this time (hours) misses the deadline."""
        return arrival_time > self.deadline

    def check_against(self, graph: Any) -> List[str]:
        """
        Return a message for each of this job's locations that is not a node
        in the graph. An empty list means the job fits the map.
        """
        problems = []
        for role, node_id in (("pickup", self.pickup), ("dropoff", self.dropoff)):
            if not graph.has_node(node_id):
                problems.append(
                    f"Job '{self.id}' has {role} '{node_id}', which is not in the graph."
                )
        return problems


# A listener receives the event name and keyword details about what changed.
QueueListener = Callable[..., None]


class DynamicJobQueue:
    """
    Orders waiting to be routed, earliest deadline first.

    Jobs with the same deadline are kept in the order they were added.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._order: Dict[str, int] = {}  # job id -> arrival number, breaks deadline ties
        self._counter = 0
        self._listeners: List[QueueListener] = []

    # --- change notifications -----------------------------------------

    def register_listener(self, callback: QueueListener) -> None:
        """Register callback(event_name, **details) to run after every change."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unregister_listener(self, callback: QueueListener) -> None:
        """Stop notifying a previously registered callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, event: str, **details: Any) -> None:
        for listener in list(self._listeners):
            listener(event, **details)

    # --- editing -------------------------------------------------------

    def add(self, job: Job) -> None:
        """Add a job. Raises ValueError if its id is already in the queue."""
        if not isinstance(job, Job):
            raise TypeError("Only Job objects can be added to the queue.")
        if job.id in self._jobs:
            raise ValueError(f"Job '{job.id}' already exists in queue.")
        self._jobs[job.id] = job
        self._order[job.id] = self._counter
        self._counter += 1
        self._notify("job_added", job_id=job.id)

    def remove(self, job_id: str) -> Job:
        """Remove a job by id and return it. Raises KeyError if it is not queued."""
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' does not exist in queue.")
        job = self._jobs.pop(job_id)
        del self._order[job_id]
        self._notify("job_removed", job_id=job_id)
        return job

    def pop(self) -> Job:
        """Remove and return the job with the earliest deadline. Raises IndexError if empty."""
        if not self._jobs:
            raise IndexError("pop from an empty job queue.")
        return self.remove(self.all_jobs()[0].id)

    def clear(self) -> None:
        """Remove every job."""
        self._jobs.clear()
        self._order.clear()
        self._notify("queue_cleared")

    # --- lookup --------------------------------------------------------

    def peek(self) -> Optional[Job]:
        """The job with the earliest deadline (left in the queue), or None if empty."""
        jobs = self.all_jobs()
        return jobs[0] if jobs else None

    def get(self, job_id: str) -> Optional[Job]:
        """Return the job with this id, or None."""
        return self._jobs.get(job_id)

    def all_jobs(self) -> Tuple[Job, ...]:
        """Snapshot of every job, earliest deadline first."""
        return tuple(
            sorted(self._jobs.values(), key=lambda j: (j.deadline, self._order[j.id]))
        )

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self) -> Iterator[Job]:
        return iter(self.all_jobs())

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._jobs

    # --- queue-wide information for the solvers -----------------------

    @property
    def total_weight(self) -> float:
        """Combined weight of every queued parcel."""
        return sum(job.weight for job in self._jobs.values())

    def demand_by_node(self) -> Dict[str, float]:
        """Total weight to be delivered at each drop-off node."""
        demand: Dict[str, float] = {}
        for job in self._jobs.values():
            demand[job.dropoff] = demand.get(job.dropoff, 0.0) + job.weight
        return demand

    def waypoints(self) -> Tuple[str, ...]:
        """
        Distinct node ids the vehicles must visit (each job's pickup, then its
        drop-off), in deadline order. Pass this to the solvers as 'waypoints'.
        """
        stops: List[str] = []
        for job in self.all_jobs():
            stops.extend((job.pickup, job.dropoff))
        return tuple(dict.fromkeys(stops))

    def check_against(self, graph: Any) -> List[str]:
        """Return a message for every job location that is not a node in the graph."""
        problems: List[str] = []
        for job in self.all_jobs():
            problems.extend(job.check_against(graph))
        return problems

    def check_against_fleet(self, fleet: Any) -> List[str]:
        """
        Return a message for every job too heavy for the largest vehicle,
        since no vehicle could ever carry it. An empty fleet is reported once.
        """
        vehicles = fleet.vehicles
        if not vehicles:
            return ["The fleet has no vehicles, so no job can be carried."] if self._jobs else []
        largest = max(v.capacity for v in vehicles)
        return [
            f"Job '{job.id}' weighs {job.weight}, more than the largest vehicle "
            f"capacity of {largest}."
            for job in self.all_jobs()
            if job.weight > largest
        ]