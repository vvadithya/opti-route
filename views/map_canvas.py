"""
Opti-Route Map Canvas
Phase 3 Subsystem: Interactive Graph Canvas (SRS F2, F9, F10, F11 & Use Case U3)

Implements interactive road-network visualization using PyQt6 QGraphicsView and QGraphicsScene:
- Node items with coordinate-mapped positions, labels, and hover tooltips.
- Weighted edge segments with distance/cost text annotations.
- Route overlays highlighting computed paths.
- Vehicle sprite marker for real-time trajectory animation.
- Pan, wheel-zoom, and automatic map fitting (fit_to_view).
- Dynamic theme switching between Dark Theme (#1E1E1E) and Light Theme (#FAFAFA).
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple

try:
    from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
    from PyQt6.QtGui import (
        QBrush,
        QColor,
        QFont,
        QPainter,
        QPainterPath,
        QPen,
        QWheelEvent,
    )
    from PyQt6.QtWidgets import (
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsLineItem,
        QGraphicsPathItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from .theme_manager import ThemeManager, ThemeMode


class NodeGraphicsItem(QGraphicsEllipseItem if HAS_PYQT else object):  # type: ignore
    """Interactive visual representation of a RoadGraph Node."""

    def __init__(
        self,
        node_id: str,
        name: str,
        x: float,
        y: float,
        radius: float = 14.0,
        is_depot: bool = False,
    ):
        if HAS_PYQT:
            super().__init__(-radius, -radius, radius * 2, radius * 2)
            self.setPos(x, y)
            self.setZValue(10)
            self.setAcceptHoverEvents(True)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        self.node_id = node_id
        self.node_name = name
        self.world_x = x
        self.world_y = y
        self.radius = radius
        self.is_depot = is_depot
        self.label_item: Optional[QGraphicsSimpleTextItem] = None

        if HAS_PYQT:
            disp_name = f" ({name})" if name else ""
            self.setToolTip(f"Node: {node_id}{disp_name}\nCoordinates: ({x:.1f}, {y:.1f})")


if HAS_PYQT:
    class MapCanvas(QGraphicsView):
        """
        Interactive Road Network Map Canvas built on PyQt6 QGraphicsView.
        Consumes real RoadGraph model and renders topology with dual-theme styling.
        """

        if HAS_PYQT:
            node_selected = pyqtSignal(str)
            canvas_clicked = pyqtSignal(float, float)

        def __init__(self, theme_manager: Optional[ThemeManager] = None, parent: Optional[QWidget] = None):
            if HAS_PYQT:
                super().__init__(parent)
                self._scene = QGraphicsScene(self)
                self.setScene(self._scene)

                # View configuration
                self.setRenderHints(
                    QPainter.RenderHint.Antialiasing
                    | QPainter.RenderHint.TextAntialiasing
                    | QPainter.RenderHint.SmoothPixmapTransform
                )
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
                self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            self.theme_manager: ThemeManager = theme_manager if theme_manager is not None else ThemeManager()
            self._graph_ref = None

            # Tracked visual items
            self._node_items: Dict[str, NodeGraphicsItem] = {}
            self._edge_items: List[QGraphicsLineItem] = []
            self._label_items: List[QGraphicsSimpleTextItem] = []
            self._route_items: List[QGraphicsPathItem] = []
            self._vehicle_item: Optional[QGraphicsEllipseItem] = None

            # Zoom limits
            self._current_zoom = 1.0
            self._min_zoom = 0.15
            self._max_zoom = 12.0

            if HAS_PYQT:
                self.apply_theme(self.theme_manager)

        # -------------------------------------------------------------------------
        # Theme Styling
        # -------------------------------------------------------------------------

        def apply_theme(self, theme_manager: ThemeManager) -> None:
            """Apply theme palette colors to scene background and all visual elements."""
            if not HAS_PYQT:
                return

            self.theme_manager = theme_manager
            p = self.theme_manager.get_palette()

            # Canvas background
            self._scene.setBackgroundBrush(QBrush(QColor(p["canvas_bg"])))

            # Update node items
            for n_id, item in self._node_items.items():
                if item.is_depot:
                    fill_color = QColor(p["depot_fill"])
                    stroke_color = QColor(p["depot_stroke"])
                else:
                    fill_color = QColor(p["node_fill"])
                    stroke_color = QColor(p["node_stroke"])

                item.setBrush(QBrush(fill_color))
                item.setPen(QPen(stroke_color, 2.0))

                if item.label_item:
                    item.label_item.setBrush(QBrush(QColor(p["text_primary"])))

            # Update edge lines
            edge_pen = QPen(QColor(p["edge_line"]), 1.8, Qt.PenStyle.SolidLine)
            for edge_line in self._edge_items:
                edge_line.setPen(edge_pen)

            # Update edge weight labels
            weight_brush = QBrush(QColor(p["edge_weight_text"]))
            for lbl in self._label_items:
                lbl.setBrush(weight_brush)

            # Update route overlay items
            route_pen = QPen(QColor(p["route_line"]), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            for r_item in self._route_items:
                r_item.setPen(route_pen)

            # Update vehicle marker
            if self._vehicle_item:
                self._vehicle_item.setBrush(QBrush(QColor(p["vehicle_fill"])))
                self._vehicle_item.setPen(QPen(QColor(p["vehicle_stroke"]), 2.0))

        # -------------------------------------------------------------------------
        # Graph Loading and Rendering
        # -------------------------------------------------------------------------

        def set_graph(self, graph) -> None:
            """
            Populate canvas from a RoadGraph model instance using public APIs only.
            Renders nodes, labels, and weighted edges.
            """
            if not HAS_PYQT:
                self._graph_ref = graph
                return

            self._graph_ref = graph
            self._scene.clear()

            self._node_items.clear()
            self._edge_items.clear()
            self._label_items.clear()
            self._route_items.clear()
            self._vehicle_item = None

            if graph is None or graph.node_count() == 0:
                return

            p = self.theme_manager.get_palette()
            font_node = QFont("Helvetica", 10, QFont.Weight.Bold)
            font_weight = QFont("Helvetica", 8)

            # 1. Render Edges (with weights at midpoints)
            # We track rendered pairs to avoid duplicate lines in undirected graphs
            rendered_edges: Set[Tuple[str, str]] = set()

            for u_id, u_node in graph.nodes.items():
                for v_id, weight in graph.get_neighbors(u_id):
                    v_node = graph.get_node(v_id)
                    if not v_node:
                        continue

                    pair_key = (min(u_id, v_id), max(u_id, v_id))
                    if pair_key in rendered_edges:
                        continue
                    rendered_edges.add(pair_key)

                    # Draw edge line
                    line_item = self._scene.addLine(
                        u_node.x, u_node.y, v_node.x, v_node.y,
                        QPen(QColor(p["edge_line"]), 1.8, Qt.PenStyle.SolidLine)
                    )
                    assert line_item is not None  # stub: addLine -> QGraphicsLineItem | None
                    line_item.setZValue(2)
                    self._edge_items.append(line_item)

                    # Edge weight label at midpoint
                    mid_x = (u_node.x + v_node.x) / 2.0
                    mid_y = (u_node.y + v_node.y) / 2.0
                    weight_text = self._scene.addSimpleText(f"{weight:.1f}", font_weight)
                    assert weight_text is not None  # stub: addSimpleText -> QGraphicsSimpleTextItem | None
                    weight_text.setBrush(QBrush(QColor(p["edge_weight_text"])))
                    weight_text.setPos(mid_x + 3, mid_y + 3)
                    weight_text.setZValue(3)
                    self._label_items.append(weight_text)

            # 2. Render Nodes
            for n_id, node in graph.nodes.items():
                is_depot = "depot" in n_id.lower() or "depot" in (node.name or "").lower()
                radius = 16.0 if is_depot else 13.0

                node_item = NodeGraphicsItem(
                    node_id=n_id,
                    name=node.name,
                    x=node.x,
                    y=node.y,
                    radius=radius,
                    is_depot=is_depot,
                )

                fill_color = QColor(p["depot_fill"] if is_depot else p["node_fill"])
                stroke_color = QColor(p["depot_stroke"] if is_depot else p["node_stroke"])
                node_item.setBrush(QBrush(fill_color))
                node_item.setPen(QPen(stroke_color, 2.0))

                self._scene.addItem(node_item)
                self._node_items[n_id] = node_item

                # Node Label text
                label_text = self._scene.addSimpleText(n_id, font_node)
                assert label_text is not None  # stub: addSimpleText -> QGraphicsSimpleTextItem | None
                label_text.setBrush(QBrush(QColor(p["text_primary"])))
                # Center label beneath node
                label_rect = label_text.boundingRect()
                label_text.setPos(node.x - label_rect.width() / 2.0, node.y + radius + 2)
                label_text.setZValue(12)
                node_item.label_item = label_text

            # Fit all elements within initial viewport
            self.fit_to_view()

        # -------------------------------------------------------------------------
        # Route Overlays (SRS F9)
        # -------------------------------------------------------------------------

        def clear_routes(self) -> None:
            """Remove all route overlay graphics items."""
            if not HAS_PYQT:
                return
            for item in self._route_items:
                self._scene.removeItem(item)
            self._route_items.clear()

        def draw_route(
            self,
            path: List[str],
            trajectory: Optional[List[Tuple[float, float]]] = None,
            color_override: Optional[str] = None,
        ) -> None:
            """
            Render a computed route path overlay (SRS F9).
            If fine-grained trajectory points are available, renders smooth interpolated curve.
            """
            if not HAS_PYQT or self._graph_ref is None:
                return

            self.clear_routes()

            if trajectory and len(trajectory) >= 2:
                # Draw using fine-grained trajectory points
                painter_path = QPainterPath()
                painter_path.moveTo(trajectory[0][0], trajectory[0][1])
                for pt in trajectory[1:]:
                    painter_path.lineTo(pt[0], pt[1])
            elif path and len(path) >= 2:
                # Fallback to straight node-to-node line segments
                painter_path = QPainterPath()
                first_node = self._graph_ref.get_node(path[0])
                if not first_node:
                    return
                painter_path.moveTo(first_node.x, first_node.y)

                for nid in path[1:]:
                    node = self._graph_ref.get_node(nid)
                    if node:
                        painter_path.lineTo(node.x, node.y)
            else:
                return

            route_color = color_override or self.theme_manager.get_color("route_line")
            pen = QPen(
                QColor(route_color),
                4.5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )

            path_item = self._scene.addPath(painter_path, pen)
            assert path_item is not None  # stub: addPath -> QGraphicsPathItem | None
            path_item.setZValue(5)
            self._route_items.append(path_item)

        # -------------------------------------------------------------------------
        # Vehicle Traversal Sprite (SRS F10)
        # -------------------------------------------------------------------------

        def set_vehicle_position(self, x: float, y: float) -> None:
            """Update vehicle sprite position during animated traversal."""
            if not HAS_PYQT:
                return

            p = self.theme_manager.get_palette()
            v_radius = 8.0

            if self._vehicle_item is None:
                self._vehicle_item = self._scene.addEllipse(
                    -v_radius, -v_radius, v_radius * 2, v_radius * 2,
                    QPen(QColor(p["vehicle_stroke"]), 2.0),
                    QBrush(QColor(p["vehicle_fill"]))
                )
                assert self._vehicle_item is not None  # stub: addEllipse -> QGraphicsEllipseItem | None
                self._vehicle_item.setZValue(20)

            self._vehicle_item.setPos(x, y)

        def hide_vehicle(self) -> None:
            """Hide vehicle marker when traversal is reset or completed."""
            if self._vehicle_item and HAS_PYQT:
                self._scene.removeItem(self._vehicle_item)
                self._vehicle_item = None

        # -------------------------------------------------------------------------
        # Navigation & Zoom Controls
        # -------------------------------------------------------------------------

        def wheelEvent(self, event: Optional[QWheelEvent]) -> None:
            """Handle mouse wheel zooming with cursor anchor."""
            if not HAS_PYQT:
                return
            if event is None:
                return

            zoom_in_factor = 1.15
            zoom_out_factor = 1.0 / zoom_in_factor

            if event.angleDelta().y() > 0:
                new_zoom = self._current_zoom * zoom_in_factor
                if new_zoom <= self._max_zoom:
                    self.scale(zoom_in_factor, zoom_in_factor)
                    self._current_zoom = new_zoom
            else:
                new_zoom = self._current_zoom * zoom_out_factor
                if new_zoom >= self._min_zoom:
                    self.scale(zoom_out_factor, zoom_out_factor)
                    self._current_zoom = new_zoom

            event.accept()

        def zoom_in(self) -> None:
            """Zoom in by 20%."""
            if not HAS_PYQT:
                return
            factor = 1.2
            if self._current_zoom * factor <= self._max_zoom:
                self.scale(factor, factor)
                self._current_zoom *= factor

        def zoom_out(self) -> None:
            """Zoom out by 20%."""
            if not HAS_PYQT:
                return
            factor = 1.0 / 1.2
            if self._current_zoom * factor >= self._min_zoom:
                self.scale(factor, factor)
                self._current_zoom *= factor

        def reset_zoom(self) -> None:
            """Reset view transform to 1:1 scale."""
            if not HAS_PYQT:
                return
            self.resetTransform()
            self._current_zoom = 1.0

        def fit_to_view(self) -> None:
            """Auto-fit all scene elements within visible canvas viewport with padding."""
            if not HAS_PYQT or self._scene is None:
                return

            items_rect = self._scene.itemsBoundingRect()
            if items_rect.isNull() or items_rect.isEmpty():
                return

            # Add 15% margin
            margin = max(items_rect.width(), items_rect.height()) * 0.15
            padded_rect = items_rect.adjusted(-margin, -margin, margin, margin)
            self._scene.setSceneRect(padded_rect)
            self.fitInView(padded_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._current_zoom = 1.0

else:
    class MapCanvas:
        """Fallback implementation when PyQt6 is not available."""

        def __init__(
            self,
            theme_manager: Optional[ThemeManager] = None,
            parent: Optional[object] = None,
            *args,
            **kwargs,
        ) -> None:
            raise RuntimeError(
                "PyQt6 is required to instantiate MapCanvas. "
                "Please install PyQt6 using 'pip install PyQt6'."
            )
