from __future__ import annotations

import math
from numbers import Real
from typing import List, Optional, Tuple

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QProgressBar

from ..models.graph import RoadGraph
from ..solvers.base_solver import OptimizationResult
from .map_canvas import MapCanvas


class RouteAnimationController(QObject):
    """Animate a route result without affecting solver execution."""

    _INTERVAL_MS_AT_1X = 100
    _INTERPOLATION_STEPS_PER_EDGE = 20

    def __init__(
        self,
        canvas: MapCanvas,
        progress_bar: QProgressBar,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._progress_bar = progress_bar
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._points: List[Tuple[float, float]] = []
        self._index = -1
        self._interval_ms = self._INTERVAL_MS_AT_1X

    def set_route(
        self,
        result: Optional[OptimizationResult],
        graph: Optional[RoadGraph],
    ) -> None:
        """Replace the playback route and reset the current animation."""
        self.reset()
        self._points = self._build_points(result, graph)

    def set_speed(self, slider_value: int) -> None:
        """Update playback speed from the 0.5x-5.0x UI slider."""
        value = min(50, max(5, slider_value))
        self._interval_ms = max(1, round(1000 / value))
        if self._timer.isActive():
            self._timer.setInterval(self._interval_ms)

    def play(self) -> None:
        """Start or resume route playback."""
        if not self._points:
            return
        if self._index >= len(self._points) - 1:
            self.reset()
        if self._index < 0:
            self._index = 0
            self._show_current_point()
        if len(self._points) == 1:
            self._progress_bar.setValue(100)
            return
        self._timer.start(self._interval_ms)

    def pause(self) -> None:
        """Pause route playback without pausing solver computation."""
        self._timer.stop()

    def step(self) -> None:
        """Advance playback by one coordinate sample."""
        self._timer.stop()
        if not self._points:
            return
        if self._index < 0:
            self._index = 0
        elif self._index < len(self._points) - 1:
            self._index += 1
        self._show_current_point()
        self._update_progress()

    def reset(self) -> None:
        """Stop playback, hide the vehicle, and reset the progress indicator."""
        self._timer.stop()
        self._index = -1
        self._canvas.hide_vehicle()
        self._progress_bar.setValue(0)

    def _advance(self) -> None:
        if self._index >= len(self._points) - 1:
            self._timer.stop()
            return
        self._index += 1
        self._show_current_point()
        self._update_progress()
        if self._index == len(self._points) - 1:
            self._timer.stop()

    def _show_current_point(self) -> None:
        x, y = self._points[self._index]
        self._canvas.set_vehicle_position(x, y)

    def _update_progress(self) -> None:
        if len(self._points) <= 1:
            self._progress_bar.setValue(100)
            return
        self._progress_bar.setValue(round(self._index * 100 / (len(self._points) - 1)))

    @classmethod
    def _build_points(
        cls,
        result: Optional[OptimizationResult],
        graph: Optional[RoadGraph],
    ) -> List[Tuple[float, float]]:
        if result is None or not result.found or not result.path or graph is None:
            return []

        nodes = []
        for node_id in result.path:
            point = cls._node_coordinates(graph.get_node(node_id))
            if point is None:
                return []
            nodes.append(point)

        trajectory = cls._validated_points(result.trajectory)
        if len(nodes) == 1:
            if trajectory and len(trajectory) == 1 and cls._same_point(trajectory[0], nodes[0]):
                return trajectory
            return nodes

        if trajectory and len(trajectory) >= 2:
            if cls._same_point(trajectory[0], nodes[0]) and cls._same_point(trajectory[-1], nodes[-1]):
                return trajectory

        points = [nodes[0]]
        for start, end in zip(nodes, nodes[1:]):
            for step in range(1, cls._INTERPOLATION_STEPS_PER_EDGE + 1):
                fraction = step / cls._INTERPOLATION_STEPS_PER_EDGE
                points.append((
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                ))
        return points

    @staticmethod
    def _same_point(first: Tuple[float, float], second: Tuple[float, float]) -> bool:
        return all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6) for a, b in zip(first, second))

    @staticmethod
    def _validated_points(points: object) -> Optional[List[Tuple[float, float]]]:
        if not isinstance(points, (list, tuple)) or not points:
            return None
        validated = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool):
                return None
            if not isinstance(x, Real) or not isinstance(y, Real):
                return None
            try:
                coordinates = float(x), float(y)
            except (OverflowError, ValueError):
                return None
            if not all(math.isfinite(value) for value in coordinates):
                return None
            validated.append(coordinates)
        return validated

    @staticmethod
    def _node_coordinates(node: object) -> Optional[Tuple[float, float]]:
        if node is None:
            return None
        x, y = getattr(node, "x", None), getattr(node, "y", None)
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        if not isinstance(x, Real) or not isinstance(y, Real):
            return None
        try:
            coordinates = float(x), float(y)
        except (OverflowError, ValueError):
            return None
        if not all(math.isfinite(value) for value in coordinates):
            return None
        return coordinates
