"""
Opti-Route Main Window
Phase 3 & Phase 4 Integrated Desktop UI (SRS F1-F15 & Use Cases U1-U4)

Constructs the primary PyQt6 desktop application window:
- Top Toolbar with Import/Export Map (JSON/CSV via GraphFactory), Theme Switcher (#1E1E1E / #FAFAFA),
  Auto-Fit, and Zoom controls.
- Central interactive MapCanvas for graph visualization, node dragging, and multi-vehicle route overlays.
- Left Control Sidebar for algorithm selection, SA hyper-parameters, mutation operators,
  interactive graph/fleet/job management, and background QThread solver execution.
- Bottom Playback Bar for vehicle traversal animation controls.
- Right Analytics & Benchmarking Panel integrating PerformanceMonitor metrics, live Convergence Plot
  (Matplotlib FigureCanvasQTAgg with native Qt Painter fallback), AlgorithmComparison table,
  and CSV/PDF report exporters (Use Case U4).
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    def pyqtSlot(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    HAS_MPL_QT = True
except ImportError:
    HAS_MPL_QT = False

from ..models.graph import Node, NodeType
from ..models.graph_factory import InvalidGraphFileError
from ..models.job import Job
from ..models.vehicle import Vehicle
from .animation_controller import RouteAnimationController
from .map_canvas import MapCanvas
from .theme_manager import ThemeManager, ThemeMode


if HAS_PYQT:
    class SolverWorkerThread(QThread):
        """Background QThread worker executing route optimization without blocking the GUI (SRS 2.3, U1)."""

        result_ready = pyqtSignal(object)
        error_occurred = pyqtSignal(str)

        def __init__(
            self,
            controller: Any,
            is_sa: bool,
            start_node: str,
            target_node: str,
            sa_kwargs: dict,
            parent: Optional[QWidget] = None,
        ):
            super().__init__(parent)
            self.controller = controller
            self.is_sa = is_sa
            self.start_node = start_node
            self.target_node = target_node
            self.sa_kwargs = sa_kwargs

        def run(self) -> None:
            try:
                if self.is_sa:
                    res = self.controller.run_simulated_annealing(
                        start_node=self.start_node,
                        end_node=self.target_node,
                        **self.sa_kwargs,
                    )
                else:
                    res = self.controller.run_dijkstra(
                        start_node=self.start_node,
                        end_node=self.target_node,
                    )
                self.result_ready.emit(res)
            except Exception as exc:
                self.error_occurred.emit(str(exc))


    class NativeConvergencePlotWidget(QWidget):
        """
        Lightweight QPainter-based convergence chart widget used either alongside or as a fallback
        when matplotlib's QtAgg backend is unavailable, ensuring SRS F13 always renders cleanly.
        """

        def __init__(self, theme_manager: ThemeManager, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.theme_manager = theme_manager
            self._points: List[Tuple[int, float]] = []
            self.setMinimumHeight(170)

        def set_history(self, history: Any) -> None:
            pts: List[Tuple[int, float]] = []
            for item in (history or ()):
                if len(item) == 3:
                    it, _, cost = item
                elif len(item) == 2:
                    it, cost = item
                else:
                    continue
                if isinstance(cost, (int, float)) and math.isfinite(cost):
                    pts.append((int(it), float(cost)))
            self._points = pts
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            p = self.theme_manager.get_palette()

            rect = self.rect().adjusted(4, 4, -4, -4)
            painter.fillRect(rect, QColor(p["input_bg"]))
            painter.setPen(QPen(QColor(p["panel_border"]), 1.0))
            painter.drawRoundedRect(rect, 6, 6)

            plot_rect = rect.adjusted(42, 24, -14, -26)
            painter.setPen(QColor(p["text_primary"]))
            painter.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(10, 4, -10, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, "Cost Convergence vs. Iteration")

            if not self._points:
                painter.setPen(QColor(p["text_muted"]))
                painter.setFont(QFont("Helvetica", 9))
                painter.drawText(
                    rect.adjusted(14, 24, -14, -10),
                    int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                    "Run SA, TLBO, or Genetic Algorithm to plot convergence",
                )
                return

            # Axes
            painter.setPen(QPen(QColor(p["panel_border"]), 1.0))
            painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
            painter.drawLine(plot_rect.bottomLeft(), plot_rect.topLeft())

            xs = [pt[0] for pt in self._points]
            ys = [pt[1] for pt in self._points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max(1, max_x - min_x)
            span_y = max(1e-6, max_y - min_y)

            path = QPainterPath()
            for idx, (it, cost) in enumerate(self._points):
                px = plot_rect.left() + ((it - min_x) / span_x) * plot_rect.width()
                py = plot_rect.bottom() - ((cost - min_y) / span_y) * plot_rect.height()
                if idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)

            painter.setPen(QPen(QColor(p["accent"]), 2.0))
            painter.drawPath(path)

            # Axis labels
            painter.setPen(QColor(p["text_secondary"]))
            painter.setFont(QFont("Helvetica", 7))
            painter.drawText(4, plot_rect.top() + 8, f"{max_y:.1f}")
            painter.drawText(4, plot_rect.bottom(), f"{min_y:.1f}")
            painter.drawText(plot_rect.left(), rect.bottom() - 4, f"Iter {min_x}")
            painter.drawText(plot_rect.right() - 45, rect.bottom() - 4, f"Iter {max_x}")


    class OptiRouteMainWindow(QMainWindow):
        """
        Main Application Window for Opti-Route.
        Coordinates MVC Controller, Map Canvas, Theme System, File I/O, and Benchmarking Analytics.
        """

        def __init__(
            self,
            controller=None,
            theme_manager: Optional[ThemeManager] = None,
            parent: Optional[QWidget] = None,
        ):
            if HAS_PYQT:
                super().__init__(parent)

            self.controller = controller
            self.theme_manager = theme_manager if theme_manager is not None else ThemeManager()
            self._worker_thread: Optional[SolverWorkerThread] = None
            self._mpl_canvas = None
            self._initial_fit_done: bool = False

            if HAS_PYQT:
                self.setWindowTitle("Opti-Route: Adaptive Vehicle Routing & Task Dispatch (Group-15)")
                self.resize(1360, 840)
                self.setMinimumSize(1024, 640)

                # Build UI Components
                self._init_canvas()
                self._init_toolbar()
                self._init_sidebar()
                self._init_bottom_playback_bar()
                self._init_analytics_panel()
                self._init_central_layout()
                self._init_playback_controller()
                self._init_statusbar()

                # Apply initial theme stylesheet and canvas colors
                self._apply_active_theme()

                # Populate graph into canvas if controller already has a graph
                if self.controller and getattr(self.controller, "graph", None):
                    self.load_graph(self.controller.graph)

        # -------------------------------------------------------------------------
        # Initialization Helpers
        # -------------------------------------------------------------------------

        def _init_canvas(self) -> None:
            """Initialize the interactive MapCanvas."""
            self.map_canvas = MapCanvas(theme_manager=self.theme_manager, parent=self)

        def _init_toolbar(self) -> None:
            """Construct top application toolbar with File I/O, Theme, and View controls."""
            toolbar = QToolBar("Main Controls", self)
            toolbar.setMovable(False)
            self.addToolBar(toolbar)

            # 1. File Import / Export Buttons & Preset Sample Network Selector (SRS F1 & Use Case U2)
            self.sample_combo = QComboBox(self)
            self.sample_combo.setMinimumHeight(26)
            self.sample_combo.setMinimumWidth(210)
            self._populate_sample_networks()
            self.sample_combo.currentIndexChanged.connect(self.on_select_sample_network)
            toolbar.addWidget(self.sample_combo)

            self.import_btn = QPushButton("📂 Import Map", self)
            self.import_btn.setObjectName("toolbar_button")
            self.import_btn.clicked.connect(self.on_import_graph)
            toolbar.addWidget(self.import_btn)

            self.export_btn = QPushButton("💾 Export Map", self)
            self.export_btn.setObjectName("toolbar_button")
            self.export_btn.clicked.connect(self.on_export_graph)
            toolbar.addWidget(self.export_btn)

            toolbar.addSeparator()

            # 2. Theme Toggle Button (SRS F11)
            self.theme_btn = QPushButton(self._theme_button_label(), self)
            self.theme_btn.setObjectName("toolbar_button")
            self.theme_btn.clicked.connect(self.on_toggle_theme)
            toolbar.addWidget(self.theme_btn)

            toolbar.addSeparator()

            # 3. Fit to View (SRS U3)
            self.fit_btn = QPushButton("Fit Map to View", self)
            self.fit_btn.setObjectName("toolbar_button")
            self.fit_btn.clicked.connect(self.map_canvas.fit_to_view)
            toolbar.addWidget(self.fit_btn)

            # 4. Zoom Controls
            self.zoom_in_btn = QPushButton("Zoom In (+)", self)
            self.zoom_in_btn.setObjectName("toolbar_button")
            self.zoom_in_btn.clicked.connect(self.map_canvas.zoom_in)
            toolbar.addWidget(self.zoom_in_btn)

            self.zoom_out_btn = QPushButton("Zoom Out (-)", self)
            self.zoom_out_btn.setObjectName("toolbar_button")
            self.zoom_out_btn.clicked.connect(self.map_canvas.zoom_out)
            toolbar.addWidget(self.zoom_out_btn)

            self.zoom_reset_btn = QPushButton("Reset Zoom", self)
            self.zoom_reset_btn.setObjectName("toolbar_button")
            self.zoom_reset_btn.clicked.connect(self.map_canvas.reset_zoom)
            toolbar.addWidget(self.zoom_reset_btn)

        def _init_sidebar(self) -> None:
            """Construct left sidebar control panel."""
            self.sidebar_frame = QFrame(self)
            self.sidebar_frame.setObjectName("control_panel")
            self.sidebar_frame.setFixedWidth(330)

            outer_layout = QVBoxLayout(self.sidebar_frame)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_layout.setSpacing(0)

            scroll_area = QScrollArea(self.sidebar_frame)
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

            scroll_content = QWidget(scroll_area)
            scroll_content.setObjectName("scroll_content")
            layout = QVBoxLayout(scroll_content)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(6)

            # Header
            title = QLabel("Routing Controls", scroll_content)
            title.setObjectName("heading_label")
            layout.addWidget(title)

            subtitle = QLabel("Strategy & Network Management", scroll_content)
            subtitle.setObjectName("muted_label")
            layout.addWidget(subtitle)

            # 1. Algorithm Selection Group
            alg_group = QGroupBox("Optimization Strategy", scroll_content)
            alg_layout = QVBoxLayout(alg_group)

            self.alg_combo = QComboBox(alg_group)
            self.alg_combo.setMinimumHeight(28)
            self.alg_combo.addItems([
                "Dijkstra Shortest Path",
                "Simulated Annealing (VRP/TSP)",
                "Bellman-Ford Shortest Path",
                "TLBO (Teaching-Learning VRP)",
                "Genetic Algorithm (GA VRP)",
            ])
            self.alg_combo.currentIndexChanged.connect(self._on_algorithm_changed)
            alg_layout.addWidget(self.alg_combo)
            layout.addWidget(alg_group)

            # 2. Location Selection Group
            loc_group = QGroupBox("Locations", scroll_content)
            loc_layout = QGridLayout(loc_group)
            loc_layout.setVerticalSpacing(8)

            loc_layout.addWidget(QLabel("Start Node:"), 0, 0)
            self.start_node_combo = QComboBox(loc_group)
            self.start_node_combo.setMinimumHeight(28)
            loc_layout.addWidget(self.start_node_combo, 0, 1)

            loc_layout.addWidget(QLabel("Target Node:"), 1, 0)
            self.target_node_combo = QComboBox(loc_group)
            self.target_node_combo.setMinimumHeight(28)
            loc_layout.addWidget(self.target_node_combo, 1, 1)

            layout.addWidget(loc_group)

            # 3. Parameters Group (Simulated Annealing & Meta-Heuristic Hyper-parameters - SRS F7, F8)
            self.params_group = QGroupBox("SA / Meta-Heuristic Params", scroll_content)
            params_layout = QGridLayout(self.params_group)
            params_layout.setVerticalSpacing(5)

            params_layout.addWidget(QLabel("Initial Temp:"), 0, 0)
            self.init_temp_spin = QDoubleSpinBox(self.params_group)
            self.init_temp_spin.setMinimumHeight(28)
            self.init_temp_spin.setRange(1.0, 10000.0)
            self.init_temp_spin.setValue(80.0)
            params_layout.addWidget(self.init_temp_spin, 0, 1)

            params_layout.addWidget(QLabel("Cooling Rate:"), 1, 0)
            self.cooling_spin = QDoubleSpinBox(self.params_group)
            self.cooling_spin.setMinimumHeight(28)
            self.cooling_spin.setRange(0.01, 0.99)
            self.cooling_spin.setSingleStep(0.01)
            self.cooling_spin.setValue(0.75)
            params_layout.addWidget(self.cooling_spin, 1, 1)

            params_layout.addWidget(QLabel("Min Temp:"), 2, 0)
            self.min_temp_spin = QDoubleSpinBox(self.params_group)
            self.min_temp_spin.setMinimumHeight(28)
            self.min_temp_spin.setDecimals(4)
            self.min_temp_spin.setRange(0.0001, 100.0)
            self.min_temp_spin.setValue(0.5)
            params_layout.addWidget(self.min_temp_spin, 2, 1)

            params_layout.addWidget(QLabel("Iterations:"), 3, 0)
            self.iter_spin = QSpinBox(self.params_group)
            self.iter_spin.setMinimumHeight(28)
            self.iter_spin.setRange(1, 1000)
            self.iter_spin.setValue(6)
            params_layout.addWidget(self.iter_spin, 3, 1)

            params_layout.addWidget(QLabel("Mutation Op:"), 4, 0)
            self.mutation_combo = QComboBox(self.params_group)
            self.mutation_combo.setMinimumHeight(28)
            self.mutation_combo.addItems(["random", "2-opt", "swap", "insertion"])
            params_layout.addWidget(self.mutation_combo, 4, 1)

            layout.addWidget(self.params_group)
            self.params_group.setVisible(False)  # Hidden when Dijkstra is selected

            # 4. Execution Action Buttons (SRS U1 Basic & Alternative Flow)
            exec_row = QHBoxLayout()
            self.run_btn = QPushButton("Calculate Route", scroll_content)
            self.run_btn.setObjectName("primary_button")
            self.run_btn.setMinimumHeight(32)
            self.run_btn.clicked.connect(self.on_run_solver)
            exec_row.addWidget(self.run_btn, stretch=3)

            self.stop_btn = QPushButton("⏹ Stop", scroll_content)
            self.stop_btn.setMinimumHeight(32)
            self.stop_btn.clicked.connect(self.on_stop_solver)
            exec_row.addWidget(self.stop_btn, stretch=1)
            layout.addLayout(exec_row)

            # 5. Interactive Map & Job Editor Group (SRS F2, F3, F4 & Use Case U2)
            edit_group = QGroupBox("Graph, Fleet && Job Editor", scroll_content)
            edit_grid = QGridLayout(edit_group)
            edit_grid.setSpacing(6)

            self.add_node_btn = QPushButton("+ Node", edit_group)
            self.add_node_btn.clicked.connect(self.on_add_node_dialog)
            edit_grid.addWidget(self.add_node_btn, 0, 0)

            self.add_edge_btn = QPushButton("+ Road", edit_group)
            self.add_edge_btn.clicked.connect(self.on_add_edge_dialog)
            edit_grid.addWidget(self.add_edge_btn, 0, 1)

            self.del_node_btn = QPushButton("- Node", edit_group)
            self.del_node_btn.clicked.connect(self.on_remove_node_dialog)
            edit_grid.addWidget(self.del_node_btn, 1, 0)

            self.add_job_btn = QPushButton("+ Delivery Job", edit_group)
            self.add_job_btn.clicked.connect(self.on_add_job_dialog)
            edit_grid.addWidget(self.add_job_btn, 1, 1)

            self.add_vehicle_btn = QPushButton("+ Add Vehicle to Fleet", edit_group)
            self.add_vehicle_btn.clicked.connect(self.on_add_vehicle_dialog)
            edit_grid.addWidget(self.add_vehicle_btn, 2, 0, 1, 2)

            layout.addWidget(edit_group)

            # 6. Network, Fleet & Job Summary Card
            summary_card = QFrame(scroll_content)
            summary_card.setObjectName("card")
            summary_layout = QVBoxLayout(summary_card)
            summary_layout.setContentsMargins(8, 8, 8, 8)
            summary_layout.setSpacing(4)

            self.summary_title = QLabel("Network & Fleet Topology", summary_card)
            self.summary_title.setObjectName("heading_label")
            summary_layout.addWidget(self.summary_title)

            self.node_count_lbl = QLabel("Nodes: 0", summary_card)
            summary_layout.addWidget(self.node_count_lbl)

            self.edge_count_lbl = QLabel("Road Segments: 0", summary_card)
            summary_layout.addWidget(self.edge_count_lbl)

            self.fleet_count_lbl = QLabel("Vehicles: 0", summary_card)
            summary_layout.addWidget(self.fleet_count_lbl)

            self.job_count_lbl = QLabel("Queued Jobs: 0", summary_card)
            summary_layout.addWidget(self.job_count_lbl)

            layout.addWidget(summary_card)
            layout.addStretch()

            scroll_area.setWidget(scroll_content)
            outer_layout.addWidget(scroll_area)

        def _init_bottom_playback_bar(self) -> None:
            """Construct bottom animation playback bar (SRS F10)."""
            self.playback_frame = QFrame(self)
            self.playback_frame.setObjectName("card")
            self.playback_frame.setFixedHeight(58)

            layout = QHBoxLayout(self.playback_frame)
            layout.setContentsMargins(12, 6, 12, 6)
            layout.setSpacing(12)

            # Playback control buttons
            self.play_btn = QPushButton("▶ Play", self.playback_frame)
            self.play_btn.setFixedWidth(88)
            self.play_btn.setMinimumHeight(30)
            layout.addWidget(self.play_btn)

            self.pause_btn = QPushButton("⏸ Pause", self.playback_frame)
            self.pause_btn.setFixedWidth(88)
            self.pause_btn.setMinimumHeight(30)
            layout.addWidget(self.pause_btn)

            self.step_btn = QPushButton("⏭ Step", self.playback_frame)
            self.step_btn.setFixedWidth(88)
            self.step_btn.setMinimumHeight(30)
            layout.addWidget(self.step_btn)

            self.reset_btn = QPushButton("↺ Reset", self.playback_frame)
            self.reset_btn.setFixedWidth(88)
            self.reset_btn.setMinimumHeight(30)
            layout.addWidget(self.reset_btn)

            # Traversal Progress
            layout.addWidget(QLabel("Progress:", self.playback_frame))
            self.progress_bar = QProgressBar(self.playback_frame)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFixedWidth(180)
            layout.addWidget(self.progress_bar)

            # Speed Multiplier Slider
            layout.addWidget(QLabel("Speed:", self.playback_frame))
            self.speed_slider = QSlider(Qt.Orientation.Horizontal, self.playback_frame)
            self.speed_slider.setRange(5, 50)  # 0.5x to 5.0x
            self.speed_slider.setValue(10)      # 1.0x default
            self.speed_slider.setFixedWidth(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            layout.addWidget(self.speed_slider)

            self.speed_lbl = QLabel("1.0x", self.playback_frame)
            self.speed_lbl.setFixedWidth(36)
            layout.addWidget(self.speed_lbl)

            layout.addStretch()

        def _init_playback_controller(self) -> None:
            """Connect playback controls to route animation."""
            self.animation_controller = RouteAnimationController(
                self.map_canvas, self.progress_bar, parent=self
            )
            self.play_btn.clicked.connect(self.animation_controller.play)
            self.pause_btn.clicked.connect(self.animation_controller.pause)
            self.step_btn.clicked.connect(self.animation_controller.step)
            self.reset_btn.clicked.connect(self.animation_controller.reset)
            self.animation_controller.set_speed(self.speed_slider.value())

        def _init_analytics_panel(self) -> None:
            """Construct Phase 4 Real-Time Benchmarking & Convergence Analytics Panel (SRS F12-F15, U4)."""
            self.analytics_frame = QFrame(self)
            self.analytics_frame.setObjectName("control_panel")
            self.analytics_frame.setFixedWidth(360)

            layout = QVBoxLayout(self.analytics_frame)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            heading = QLabel("Benchmarking & Analytics", self.analytics_frame)
            heading.setObjectName("heading_label")
            layout.addWidget(heading)

            subheading = QLabel("Live Telemetry, Convergence & Comparison", self.analytics_frame)
            subheading.setObjectName("muted_label")
            layout.addWidget(subheading)

            # 1. Latest Run Telemetry Card (SRS F12)
            metrics_card = QFrame(self.analytics_frame)
            metrics_card.setObjectName("card")
            metrics_grid = QGridLayout(metrics_card)
            metrics_grid.setContentsMargins(8, 8, 8, 8)
            metrics_grid.setSpacing(4)

            self.metric_alg_lbl = QLabel("Algorithm: —", metrics_card)
            self.metric_cost_lbl = QLabel("Total Cost: —", metrics_card)
            self.metric_dist_lbl = QLabel("Route Distance: —", metrics_card)
            self.metric_time_lbl = QLabel("Runtime: —", metrics_card)
            self.metric_mem_lbl = QLabel("Peak Memory: —", metrics_card)
            self.metric_iter_lbl = QLabel("Iterations / Visited: —", metrics_card)

            metrics_grid.addWidget(self.metric_alg_lbl, 0, 0, 1, 2)
            metrics_grid.addWidget(self.metric_cost_lbl, 1, 0)
            metrics_grid.addWidget(self.metric_dist_lbl, 1, 1)
            metrics_grid.addWidget(self.metric_time_lbl, 2, 0)
            metrics_grid.addWidget(self.metric_mem_lbl, 2, 1)
            metrics_grid.addWidget(self.metric_iter_lbl, 3, 0, 1, 2)
            layout.addWidget(metrics_card)

            # 2. Convergence Plot Widget (SRS F13)
            self.plot_container = QFrame(self.analytics_frame)
            self.plot_container.setObjectName("card")
            self.plot_layout = QVBoxLayout(self.plot_container)
            self.plot_layout.setContentsMargins(4, 4, 4, 4)

            self.native_plot_widget = NativeConvergencePlotWidget(self.theme_manager, self.plot_container)
            self.plot_layout.addWidget(self.native_plot_widget)
            layout.addWidget(self.plot_container)

            # 3. Comparative Evaluation Table (SRS F14)
            comp_group = QGroupBox("Algorithm Comparison (All Solvers)", self.analytics_frame)
            comp_layout = QVBoxLayout(comp_group)
            comp_layout.setContentsMargins(6, 10, 6, 6)

            self.comp_table = QTableWidget(0, 4, comp_group)
            self.comp_table.setHorizontalHeaderLabels(["Algo", "Cost", "Time (ms)", "Mem (MB)"])
            header = self.comp_table.horizontalHeader()
            if header is not None:
                header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            v_header = self.comp_table.verticalHeader()
            if v_header is not None:
                v_header.setVisible(False)
            self.comp_table.setMaximumHeight(150)
            comp_layout.addWidget(self.comp_table)
            layout.addWidget(comp_group)

            # 4. Report Export Controls (SRS F15)
            export_row = QHBoxLayout()
            self.export_csv_btn = QPushButton("📊 Export CSV", self.analytics_frame)
            self.export_csv_btn.clicked.connect(self.on_export_benchmark_csv)
            export_row.addWidget(self.export_csv_btn)

            self.export_pdf_btn = QPushButton("📄 Export PDF", self.analytics_frame)
            self.export_pdf_btn.clicked.connect(self.on_export_benchmark_pdf)
            export_row.addWidget(self.export_pdf_btn)
            layout.addLayout(export_row)

            layout.addStretch()

        def _init_central_layout(self) -> None:
            """Assemble main central layout with splitters."""
            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)

            main_vbox = QVBoxLayout(central_widget)
            main_vbox.setContentsMargins(0, 0, 0, 0)
            main_vbox.setSpacing(0)

            # Horizontal Splitter: Sidebar | MapCanvas | Analytics
            h_splitter = QSplitter(Qt.Orientation.Horizontal, central_widget)
            h_splitter.addWidget(self.sidebar_frame)
            h_splitter.addWidget(self.map_canvas)
            h_splitter.addWidget(self.analytics_frame)
            h_splitter.setStretchFactor(0, 0)
            h_splitter.setStretchFactor(1, 1)
            h_splitter.setStretchFactor(2, 0)

            main_vbox.addWidget(h_splitter, stretch=1)
            main_vbox.addWidget(self.playback_frame, stretch=0)

        def _init_statusbar(self) -> None:
            """Construct status bar."""
            self.status_bar = QStatusBar(self)
            self.setStatusBar(self.status_bar)
            self._update_status_bar()

        def showEvent(self, event) -> None:
            """Fit map to view once the window and canvas viewport have received their initial layout size."""
            super().showEvent(event)
            if not self._initial_fit_done:
                self._initial_fit_done = True
                self.map_canvas.fit_to_view()

        # -------------------------------------------------------------------------
        # Event Handlers & State Updates
        # -------------------------------------------------------------------------

        def _theme_button_label(self) -> str:
            """Format theme button label based on current mode."""
            return "☀️ Switch to Light Theme" if self.theme_manager.is_dark else "🌙 Switch to Dark Theme"

        def _apply_active_theme(self) -> None:
            """Apply theme QSS to entire window and restyle canvas."""
            if not HAS_PYQT:
                return
            qApp = QApplication.instance()
            if isinstance(qApp, QApplication):
                qApp.setStyleSheet(self.theme_manager.get_stylesheet())
            self.map_canvas.apply_theme(self.theme_manager)
            self.native_plot_widget.update()
            self.theme_btn.setText(self._theme_button_label())
            self._update_status_bar()

        @pyqtSlot()
        def on_toggle_theme(self) -> None:
            """One-click theme toggle between Dark (#1E1E1E) and Light (#FAFAFA) (SRS F11)."""
            self.theme_manager.toggle_theme()
            self._apply_active_theme()

        def _on_algorithm_changed(self, index: int) -> None:
            """Show or hide meta-heuristic hyper-parameter controls based on selected algorithm."""
            is_meta = index in (1, 3, 4)
            self.params_group.setVisible(is_meta)

        def _populate_sample_networks(self) -> None:
            """Discover all sample network JSON files in the data/ directory for 1-click loading."""
            self._sample_network_paths: List[Path] = []
            data_dir = Path(__file__).resolve().parents[1] / "data"
            friendly_names = {
                "sample_network.json": "🗺️ Sample 0: Metro Distribution (Default)",
                "sample_01_downtown_grid.json": "🗺️ Sample 1: Downtown Manhattan Grid",
                "sample_02_airport_cargo_hub.json": "🗺️ Sample 2: Airport Express Cargo Hub",
                "sample_03_port_container_dispatch.json": "🗺️ Sample 3: Seaport Container Dispatch",
                "sample_04_medical_cold_chain.json": "🗺️ Sample 4: Medical Cold-Chain Emergency",
                "sample_05_ecommerce_last_mile.json": "🗺️ Sample 5: E-Commerce Last-Mile Cluster",
                "sample_06_industrial_corridor.json": "🗺️ Sample 6: Regional Industrial Corridor",
                "sample_07_campus_micro_transit.json": "🗺️ Sample 7: Smart Campus Micro-Transit",
                "sample_08_multi_depot_metropolis.json": "🗺️ Sample 8: Multi-Depot Metropolis Ring",
            }
            if data_dir.exists():
                json_files = sorted(
                    data_dir.glob("*.json"),
                    key=lambda p: (0 if p.name == "sample_network.json" else 1, p.name),
                )
                for p in json_files:
                    self._sample_network_paths.append(p)
                    label = friendly_names.get(p.name, f"🗺️ {p.stem}")
                    self.sample_combo.addItem(label)

        def on_select_sample_network(self, index: int) -> None:
            """Load the selected preset sample network into the controller and canvas."""
            if not self.controller or index < 0 or index >= len(getattr(self, "_sample_network_paths", [])):
                return
            sample_path = self._sample_network_paths[index]
            try:
                project = self.controller.load_project(sample_path)
                self.load_graph(project.graph)
                self.map_canvas.fit_to_view()
                self.status_bar.showMessage(
                    f"Loaded '{sample_path.name}': {project.graph.node_count()} nodes, "
                    f"{project.graph.edge_count()} roads, {len(project.fleet)} vehicles, {len(project.jobs)} jobs."
                )
            except Exception as exc:
                QMessageBox.warning(self, "Sample Load Error", f"Failed to load sample network:\n{exc}")

        def _on_speed_changed(self, value: int) -> None:
            """Update speed multiplier label."""
            speed_factor = value / 10.0
            self.speed_lbl.setText(f"{speed_factor:.1f}x")
            self.animation_controller.set_speed(value)

        def _update_status_bar(self) -> None:
            """Update status bar information."""
            if not HAS_PYQT:
                return
            theme_str = "Dark Theme (#1E1E1E)" if self.theme_manager.is_dark else "Light Theme (#FAFAFA)"
            node_c = len(self.map_canvas._node_items)
            edge_c = len(self.map_canvas._edge_items)
            self.status_bar.showMessage(f"Opti-Route Ready | Theme: {theme_str} | Network: {node_c} nodes, {edge_c} edges")

        def _refresh_summary_labels(self) -> None:
            """Update network, fleet, and job counts in the sidebar card."""
            if not HAS_PYQT or not self.controller:
                return
            graph = getattr(self.controller, "graph", None)
            fleet = getattr(self.controller, "fleet", None)
            jobs = getattr(self.controller, "job_queue", None)

            if graph is not None:
                self.node_count_lbl.setText(f"Nodes: {graph.node_count()}")
                self.edge_count_lbl.setText(f"Road Segments: {graph.edge_count()}")
            if fleet is not None and hasattr(fleet, "__len__") and not isinstance(fleet, str):
                self.fleet_count_lbl.setText(f"Vehicles: {len(fleet)} (Cap: {getattr(fleet, 'total_capacity', 0.0):.0f})")
            if jobs is not None and hasattr(jobs, "__len__") and not isinstance(jobs, str):
                self.job_count_lbl.setText(f"Queued Jobs: {len(jobs)} (Load: {getattr(jobs, 'total_weight', 0.0):.0f})")

        def load_graph(self, graph) -> None:
            """Load and display a RoadGraph model on the canvas and in UI selectors."""
            if graph is None:
                return

            self.animation_controller.set_route(None, graph)
            self.map_canvas.set_graph(graph)

            if HAS_PYQT:
                # Update location selectors
                prev_start = self.start_node_combo.currentText()
                prev_target = self.target_node_combo.currentText()
                self.start_node_combo.clear()
                self.target_node_combo.clear()

                node_ids = sorted(list(graph.nodes.keys()))
                self.start_node_combo.addItems(node_ids)
                self.target_node_combo.addItems(node_ids)

                if prev_start in node_ids:
                    self.start_node_combo.setCurrentText(prev_start)
                elif "Depot" in node_ids:
                    self.start_node_combo.setCurrentText("Depot")

                if prev_target in node_ids:
                    self.target_node_combo.setCurrentText(prev_target)
                elif len(node_ids) > 1:
                    self.target_node_combo.setCurrentIndex(len(node_ids) - 1)

                self._refresh_summary_labels()
                self._update_status_bar()

        # -------------------------------------------------------------------------
        # File Import / Export Handlers (SRS F1 & Use Case U2)
        # -------------------------------------------------------------------------

        @pyqtSlot()
        def on_import_graph(self) -> None:
            """Import road network, fleet, and job queue from JSON or CSV file (SRS F1)."""
            if not self.controller:
                return
            data_dir = Path(__file__).resolve().parents[1] / "data"
            default_dir = str(data_dir if data_dir.exists() else Path.cwd())
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Road Network & Jobs",
                default_dir,
                "Map Files (*.json *.csv);;JSON Files (*.json);;CSV Files (*.csv)",
            )
            if not file_path:
                return
            try:
                project = self.controller.load_project(file_path)
                self.load_graph(project.graph)
                self.status_bar.showMessage(
                    f"Imported '{Path(file_path).name}': {project.graph.node_count()} nodes, "
                    f"{len(project.fleet)} vehicles, {len(project.jobs)} jobs."
                )
            except (InvalidGraphFileError, Exception) as exc:
                QMessageBox.warning(self, "Import Error", f"Failed to import map file:\n{exc}")

        @pyqtSlot()
        def on_export_graph(self) -> None:
            """Export current road network, fleet, and job queue to JSON or CSV file (SRS F1)."""
            if not self.controller:
                return
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Road Network & Jobs",
                str(Path.cwd() / "exported_network.json"),
                "JSON Files (*.json);;CSV Files (*.csv)",
            )
            if not file_path:
                return
            try:
                saved = self.controller.save_project(file_path)
                self.status_bar.showMessage(f"Exported network to '{saved.name}'.")
            except Exception as exc:
                QMessageBox.warning(self, "Export Error", f"Failed to export map file:\n{exc}")

        # -------------------------------------------------------------------------
        # Interactive Map, Fleet & Job Editing Dialogs (SRS F2, F3, F4)
        # -------------------------------------------------------------------------

        @pyqtSlot()
        def on_add_node_dialog(self) -> None:
            """Add a new node to the road graph interactively (SRS F2)."""
            if not self.controller or not getattr(self.controller, "graph", None):
                return
            node_id, ok = QInputDialog.getText(self, "Add Node", "Node ID (e.g. Cust-F):")
            if not ok or not node_id.strip():
                return
            x_val, ok_x = QInputDialog.getDouble(self, "Add Node", "X Coordinate:", 400.0, -5000.0, 5000.0, 1)
            if not ok_x:
                return
            y_val, ok_y = QInputDialog.getDouble(self, "Add Node", "Y Coordinate:", 300.0, -5000.0, 5000.0, 1)
            if not ok_y:
                return
            try:
                self.controller.graph.add_node(Node(id=node_id.strip(), x=x_val, y=y_val, name=node_id.strip()))
                self.load_graph(self.controller.graph)
            except Exception as exc:
                QMessageBox.warning(self, "Add Node Error", str(exc))

        @pyqtSlot()
        def on_add_edge_dialog(self) -> None:
            """Add a weighted road edge between two existing nodes (SRS F2)."""
            if not self.controller or not getattr(self.controller, "graph", None):
                return
            node_ids = sorted(list(self.controller.graph.nodes.keys()))
            if len(node_ids) < 2:
                QMessageBox.information(self, "Add Road", "At least 2 nodes are required to add a road.")
                return
            src, ok1 = QInputDialog.getItem(self, "Add Road", "Source Node:", node_ids, 0, False)
            if not ok1:
                return
            dst, ok2 = QInputDialog.getItem(self, "Add Road", "Target Node:", node_ids, min(1, len(node_ids) - 1), False)
            if not ok2:
                return
            weight, ok3 = QInputDialog.getDouble(self, "Add Road", "Road Weight / Distance (km):", 10.0, 0.1, 10000.0, 1)
            if not ok3:
                return
            try:
                self.controller.graph.add_edge(src, dst, weight, directed=False)
                self.load_graph(self.controller.graph)
            except Exception as exc:
                QMessageBox.warning(self, "Add Road Error", str(exc))

        @pyqtSlot()
        def on_remove_node_dialog(self) -> None:
            """Remove a node and its connected edges from the road graph (SRS F2)."""
            if not self.controller or not getattr(self.controller, "graph", None):
                return
            node_ids = sorted(list(self.controller.graph.nodes.keys()))
            if not node_ids:
                return
            nid, ok = QInputDialog.getItem(self, "Remove Node", "Select Node to Remove:", node_ids, 0, False)
            if not ok:
                return
            try:
                self.controller.graph.remove_node(nid)
                self.load_graph(self.controller.graph)
            except Exception as exc:
                QMessageBox.warning(self, "Remove Node Error", str(exc))

        @pyqtSlot()
        def on_add_vehicle_dialog(self) -> None:
            """Add a delivery vehicle to the controller's VehicleFleet (SRS F3)."""
            if not self.controller or not hasattr(self.controller.fleet, "add"):
                return
            vid, ok = QInputDialog.getText(self, "Add Vehicle", "Vehicle ID (e.g. V3):")
            if not ok or not vid.strip():
                return
            cap, ok_c = QInputDialog.getDouble(self, "Add Vehicle", "Capacity (kg):", 100.0, 1.0, 10000.0, 1)
            if not ok_c:
                return
            spd, ok_s = QInputDialog.getDouble(self, "Add Vehicle", "Speed (km/h):", 60.0, 1.0, 300.0, 1)
            if not ok_s:
                return
            depot = self.start_node_combo.currentText() or "Depot"
            try:
                self.controller.fleet.add(
                    Vehicle(id=vid.strip(), capacity=cap, speed=spd, fuel_rate=0.12, home_depot=depot)
                )
                self._refresh_summary_labels()
                self.status_bar.showMessage(f"Added Vehicle '{vid.strip()}' (Capacity: {cap} kg, Depot: {depot}).")
            except Exception as exc:
                QMessageBox.warning(self, "Add Vehicle Error", str(exc))

        @pyqtSlot()
        def on_add_job_dialog(self) -> None:
            """Add a delivery job to the controller's DynamicJobQueue (SRS F4)."""
            if not self.controller or not hasattr(self.controller.job_queue, "add"):
                return
            node_ids = sorted(list(self.controller.graph.nodes.keys()))
            if len(node_ids) < 2:
                return
            jid, ok = QInputDialog.getText(self, "Add Delivery Job", "Job ID (e.g. J6):")
            if not ok or not jid.strip():
                return
            pickup, ok1 = QInputDialog.getItem(self, "Add Delivery Job", "Pickup Node:", node_ids, 0, False)
            if not ok1:
                return
            dropoff, ok2 = QInputDialog.getItem(self, "Add Delivery Job", "Drop-off Node:", node_ids, len(node_ids) - 1, False)
            if not ok2:
                return
            weight, ok3 = QInputDialog.getDouble(self, "Add Delivery Job", "Parcel Weight (kg):", 20.0, 0.1, 5000.0, 1)
            if not ok3:
                return
            deadline, ok4 = QInputDialog.getDouble(self, "Add Delivery Job", "Deadline (hours):", 4.0, 0.1, 168.0, 1)
            if not ok4:
                return
            try:
                self.controller.job_queue.add(
                    Job(id=jid.strip(), pickup=pickup, dropoff=dropoff, weight=weight, deadline=deadline)
                )
                self._refresh_summary_labels()
                self.status_bar.showMessage(f"Queued Job '{jid.strip()}': {pickup} -> {dropoff} ({weight} kg, {deadline}h).")
            except Exception as exc:
                QMessageBox.warning(self, "Add Job Error", str(exc))

        # -------------------------------------------------------------------------
        # Solver Execution & Analytics Updates (SRS U1 & U4)
        # -------------------------------------------------------------------------

        @pyqtSlot()
        def on_stop_solver(self) -> None:
            """Stop active solver execution early and display best solution found so far (SRS U1)."""
            if self.controller:
                self.controller.stop_execution()
                self.status_bar.showMessage("Solver stopped by user.")

        @pyqtSlot()
        def on_run_solver(self) -> None:
            """Execute selected algorithm via controller and update canvas & analytics panel."""
            if not self.controller or not getattr(self.controller, "graph", None):
                return

            start_n = self.start_node_combo.currentText()
            target_n = self.target_node_combo.currentText()

            if not start_n or not target_n:
                return

            alg_idx = self.alg_combo.currentIndex()

            try:
                if alg_idx == 0:
                    result = self.controller.run_dijkstra(start_node=start_n, end_node=target_n)
                elif alg_idx == 2:
                    result = self.controller.run_bellman_ford(start_node=start_n, end_node=target_n)
                else:
                    # Extract waypoints from active job queue if available; otherwise visit all intermediate nodes
                    waypoints = None
                    jq = getattr(self.controller, "job_queue", None)
                    if jq is None or not hasattr(jq, "__len__") or len(jq) == 0:
                        waypoints = [nid for nid in self.controller.graph.nodes.keys() if nid not in (start_n, target_n)]

                    common_kwargs = dict(
                        start_node=start_n,
                        end_node=target_n,
                        waypoints=waypoints,
                        initial_temp=self.init_temp_spin.value(),
                        cooling_rate=self.cooling_spin.value(),
                        min_temp=self.min_temp_spin.value(),
                        iterations_per_temp=self.iter_spin.value(),
                        mutation_operator=self.mutation_combo.currentText(),
                    )
                    if alg_idx == 1:
                        result = self.controller.run_simulated_annealing(**common_kwargs)
                    elif alg_idx == 3:
                        result = self.controller.run_tlbo(**common_kwargs)
                    else:
                        result = self.controller.run_genetic_algorithm(**common_kwargs)

                self._handle_solver_result(result)
            except Exception as e:
                self.animation_controller.set_route(None, self.controller.graph)
                self.status_bar.showMessage(f"Solver Error: {str(e)}")

        def _handle_solver_result(self, result: Any) -> None:
            """Render route result on canvas and refresh Phase 4 benchmarking analytics."""
            if result and result.found:
                self.animation_controller.set_route(result, self.controller.graph)
                routes_dict = getattr(result, "routes", None)
                if routes_dict and len(routes_dict) > 1 and not getattr(result, "trajectory", None):
                    self.map_canvas.draw_multi_routes(routes_dict)
                else:
                    self.map_canvas.draw_route(result.path, result.trajectory)
                self._update_analytics_panel(result)
                self.status_bar.showMessage(
                    f"{result.algorithm_name} Complete | Cost: {result.total_cost:.2f} | Time: {result.execution_time_ms:.2f}ms"
                )
            else:
                self.animation_controller.set_route(None, self.controller.graph)
                self.map_canvas.clear_routes()
                self.status_bar.showMessage("No feasible route found.")

        def _update_analytics_panel(self, result: Any) -> None:
            """Populate telemetry metrics, convergence chart, and comparison table (SRS F12-F14)."""
            if not result:
                return
            unit = getattr(self.controller.graph, "distance_unit", "km") if self.controller else "km"
            dist = getattr(result, "total_distance", result.total_cost)
            mem = getattr(result, "peak_memory_mb", 0.0) or 0.0

            self.metric_alg_lbl.setText(f"Algorithm: {result.algorithm_name}")
            self.metric_cost_lbl.setText(f"Cost: {result.total_cost:.2f}")
            self.metric_dist_lbl.setText(f"Dist: {dist:.2f} {unit}")
            self.metric_time_lbl.setText(f"Time: {result.execution_time_ms:.2f} ms")
            self.metric_mem_lbl.setText(f"Mem: {mem:.4f} MB")
            self.metric_iter_lbl.setText(f"Iterations / Visited: {result.nodes_visited}")

            # Update Convergence Plot
            history = getattr(result, "convergence_history", ()) or ()
            self.native_plot_widget.set_history(history)

            # Update Comparison Table from MetricsCollector
            if self.controller and hasattr(self.controller, "get_benchmark_results"):
                runs = self.controller.get_benchmark_results()
                self.comp_table.setRowCount(len(runs))
                for r_idx, bench in enumerate(runs):
                    alg_item = QTableWidgetItem(bench.algorithm_name)
                    cost_str = f"{bench.total_solution_cost:.2f}" if bench.total_solution_cost is not None else "—"
                    time_str = f"{bench.execution_time_ms:.2f}"
                    mem_str = f"{bench.peak_traced_memory_mb:.4f}" if bench.peak_traced_memory_mb is not None else "—"
                    self.comp_table.setItem(r_idx, 0, alg_item)
                    self.comp_table.setItem(r_idx, 1, QTableWidgetItem(cost_str))
                    self.comp_table.setItem(r_idx, 2, QTableWidgetItem(time_str))
                    self.comp_table.setItem(r_idx, 3, QTableWidgetItem(mem_str))

        # -------------------------------------------------------------------------
        # Benchmark Report Exporters (SRS F15 & Use Case U4)
        # -------------------------------------------------------------------------

        @pyqtSlot()
        def on_export_benchmark_csv(self) -> None:
            """Export recorded benchmark metrics to CSV via Phase 4 report_generator."""
            if not self.controller or not self.controller.get_benchmark_results():
                QMessageBox.information(self, "Export CSV", "Run at least one optimization before exporting.")
                return
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Benchmark Report (CSV)",
                str(Path.cwd() / "benchmark_report.csv"),
                "CSV Files (*.csv)",
            )
            if not file_path:
                return
            try:
                out_csv, hist_csv = self.controller.export_benchmark_csv(file_path)
                msg = f"Saved benchmark CSV: {out_csv.name}"
                if hist_csv:
                    msg += f" (+ {hist_csv.name})"
                self.status_bar.showMessage(msg)
            except Exception as exc:
                QMessageBox.warning(self, "CSV Export Error", str(exc))

        @pyqtSlot()
        def on_export_benchmark_pdf(self) -> None:
            """Export recorded benchmark metrics and charts to PDF via Phase 4 report_generator."""
            if not self.controller or not self.controller.get_benchmark_results():
                QMessageBox.information(self, "Export PDF", "Run at least one optimization before exporting.")
                return
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Benchmark Report (PDF)",
                str(Path.cwd() / "benchmark_report.pdf"),
                "PDF Files (*.pdf)",
            )
            if not file_path:
                return
            try:
                out_pdf = self.controller.export_benchmark_pdf(file_path)
                self.status_bar.showMessage(f"Saved benchmark PDF report: {out_pdf.name}")
            except Exception as exc:
                QMessageBox.warning(self, "PDF Export Error", str(exc))

else:
    class OptiRouteMainWindow:
        """Fallback implementation when PyQt6 is not available."""

        def __init__(
            self,
            controller=None,
            theme_manager: Optional[ThemeManager] = None,
            parent: Optional[object] = None,
            *args,
            **kwargs,
        ) -> None:
            raise RuntimeError(
                "PyQt6 is required to instantiate OptiRouteMainWindow. "
                "Please install PyQt6 using 'pip install PyQt6'."
            )
