import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..models.graph import RoadGraph, Node


class SolverState(Enum):
    """Execution state enum for optimization solvers."""

    INIT = "INIT"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ExecutionParams:
    """Parameter wrapper for solver configurations."""

    initial_temp: float = 80.0
    cooling_rate: float = 0.75
    min_temp: float = 0.5
    iterations_per_temp: int = 6
    mutation_operator: str = "random"
    steps_per_edge: int = 10
    seed: Optional[int] = None
    waypoints: List[str] = field(default_factory=list)
    start_node: Optional[str] = None
    end_node: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Encapsulates the unified solution returned by pathfinding and routing solvers."""

    path: List[str] = field(default_factory=list)
    routes: Dict[str, List[str]] = field(default_factory=dict)
    total_cost: float = float("inf")
    total_distance: float = 0.0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    nodes_visited: int = 0
    execution_time_ms: float = 0.0
    algorithm_name: str = ""
    found: bool = False
    peak_memory_mb: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)


class RouteResult(OptimizationResult):
    """Subclass of OptimizationResult providing strict backward compatibility with Phase 2 solvers."""

    pass


class AlgorithmStrategy(ABC):
    """Abstract Strategy Base Class enforcing a uniform solver interface."""

    @abstractmethod
    def solve(
        self,
        graph: RoadGraph,
        fleet: Optional[Any] = None,
        job_queue: Optional[Any] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Execute pathfinding or optimization algorithm.

        :param graph: The RoadGraph instance to search.
        :param fleet: Optional VehicleFleet model or start_node ID string.
        :param job_queue: Optional JobQueue model or end_node ID string.
        :param kwargs: Additional keyword arguments (start_node, end_node, params, waypoints, etc.).
        :return: OptimizationResult object containing path, total cost, trajectory, and benchmarks.
        """
        pass


class BaseSolver(AlgorithmStrategy, ABC):
    """
    Base solver providing generic solver state handling (Init, Running, Paused, Stopped),
    progress callbacks, signal hooks, and trajectory generation for UI animation.
    """

    def __init__(self, name: str = "BaseSolver", params: Optional[ExecutionParams] = None):
        self.name = name
        self.state: SolverState = SolverState.INIT
        self.params: ExecutionParams = params if params is not None else ExecutionParams()
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._signal_hooks: Dict[str, List[Callable[..., None]]] = {}

    def set_state(self, state: SolverState) -> None:
        """Update solver execution state and emit state change signal hooks."""
        old_state = self.state
        self.state = state
        self.emit_signal("state_change", old_state, state)

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Register progress callback function receiving (progress_percentage, status_message)."""
        self._progress_callback = callback

    def notify_progress(self, percentage: float, status_msg: str = "") -> None:
        """Invoke progress callback and emit progress signal if registered."""
        if self._progress_callback:
            self._progress_callback(percentage, status_msg)
        self.emit_signal("progress", percentage, status_msg)

    def register_signal_hook(self, event_name: str, hook: Callable[..., None]) -> None:
        """Register a callback/listener hook for specific solver events."""
        if event_name not in self._signal_hooks:
            self._signal_hooks[event_name] = []
        self._signal_hooks[event_name].append(hook)

    def emit_signal(self, event_name: str, *args, **kwargs) -> None:
        """Trigger registered signal hooks for event_name."""
        for hook in self._signal_hooks.get(event_name, []):
            hook(*args, **kwargs)

    def pause(self) -> None:
        """Pause solver execution."""
        if self.state == SolverState.RUNNING:
            self.set_state(SolverState.PAUSED)

    def resume(self) -> None:
        """Resume solver execution."""
        if self.state == SolverState.PAUSED:
            self.set_state(SolverState.RUNNING)

    def stop(self) -> None:
        """Halt solver execution."""
        self.set_state(SolverState.STOPPED)

    def reset(self) -> None:
        """Reset solver state to INIT."""
        self.set_state(SolverState.INIT)

    def generate_trajectory(
        self, graph: RoadGraph, path: List[str], steps_per_edge: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Generates fine-grained (x, y) coordinates along path edges for smooth UI sprite animation.

        :param graph: RoadGraph containing Node geometry.
        :param path: Ordered list of node IDs.
        :param steps_per_edge: Number of interpolation points per edge segment.
        :return: List of (x, y) floating point coordinate tuples.
        """
        if not path or len(path) < 2:
            if path and len(path) == 1:
                node = graph.get_node(path[0])
                if node:
                    return [(node.x, node.y)]
            return []

        trajectory: List[Tuple[float, float]] = []

        for i in range(len(path) - 1):
            curr_node = graph.get_node(path[i])
            next_node = graph.get_node(path[i + 1])

            if not curr_node or not next_node:
                continue

            # Linear interpolation between curr_node and next_node
            for s in range(steps_per_edge):
                t = s / steps_per_edge
                x = curr_node.x + t * (next_node.x - curr_node.x)
                y = curr_node.y + t * (next_node.y - curr_node.y)
                trajectory.append((x, y))

        # Include exact end point
        end_node_obj = graph.get_node(path[-1])
        if end_node_obj:
            trajectory.append((end_node_obj.x, end_node_obj.y))

        return trajectory
